//! 2-D drivers of the AV1 DCTs, generic over [`I32Lanes`]: each 1-D pass
//! transforms `V::LANES` columns (or rows) at once, with a tile transpose
//! between the passes.
//!
//! Forward (libaom `fwd_txfm2d`): columns, then rows. With lanes running
//! along a row, the column pass is lane-parallel as loaded; it stores its
//! output transposed (in registers, `V::LANES`-square tiles at a time) so
//! the row pass is lane-parallel too, and the row pass then stores
//! coefficients straight into libaom's column-major layout.
//!
//! Inverse (libaom `inv_txfm2d_add_c`): rows, then columns, the mirror image:
//! column-major coefficients load lane-parallel for the row pass, which
//! stores its output transposed for the column pass, which writes whole
//! output rows.
//!
//! The one buffer between the passes is left uninitialised: zeroing it
//! (`memset` of up to 4 KiB per call) cost more than the 32x32 butterflies'
//! own stores. Every element is written by the first pass before the second
//! pass reads it.
//!
//! The functions are `unsafe` only because `V`'s instructions need its ISA
//! (see the `simd` module docs): call them from a `#[target_feature]`
//! function of that ISA, or with the scalar `i32` lane type.

#![allow(clippy::inline_always)]

use super::dct;
use super::{COSPI_12, cospi};
use core::mem::MaybeUninit;

use crate::codec_txfm::generic::as_uninit;
use crate::simd::I32Lanes;

/// Per-size forward configuration (libaom `fwd_shift_NxN`,
/// `av1_fwd_cos_bit_col/row` for square sizes).
struct ForwardConfig {
    input_shift: u32,
    mid_shift: u32,
    cos_bit_col: u32,
    cos_bit_row: u32,
}

const FORWARD_4: ForwardConfig = ForwardConfig {
    input_shift: 2,
    mid_shift: 0,
    cos_bit_col: 13,
    cos_bit_row: 13,
};
const FORWARD_8: ForwardConfig = ForwardConfig {
    input_shift: 2,
    mid_shift: 1,
    cos_bit_col: 13,
    cos_bit_row: 13,
};
const FORWARD_16: ForwardConfig = ForwardConfig {
    input_shift: 2,
    mid_shift: 2,
    cos_bit_col: 13,
    cos_bit_row: 12,
};
const FORWARD_32: ForwardConfig = ForwardConfig {
    input_shift: 2,
    mid_shift: 4,
    cos_bit_col: 12,
    cos_bit_row: 12,
};

/// Inverse transforms always use 12-bit cosines (libaom `INV_COS_BIT`).
pub(super) const INVERSE_COS_BIT: u32 = 12;
/// Rounding shift after the inverse column pass, every square size
/// (libaom `inv_shift_NxN[1]`).
pub(super) const INVERSE_COLUMN_SHIFT: u32 = 4;

/// Store one pass's `n` output vectors (`lanes[k]`: element `k` of the
/// `V::LANES` transforms at `first..first + LANES`) transposed into `buf`:
/// `buf[(first + i) * n + k] = lane i of lanes[k]`.
#[inline(always)]
pub(crate) fn store_pass<V: I32Lanes>(
    lanes: &[V],
    first: usize,
    n: usize,
    buf: &mut [MaybeUninit<i32>],
) {
    for k0 in (0..n).step_by(V::LANES) {
        V::store_transposed(&lanes[k0..k0 + V::LANES], &mut buf[first * n + k0..], n);
    }
}

/// [`store_pass`] of the first `lanes.len()` outputs only, into rows of
/// `stride` values: `buf[(first + i) * stride + k] = lane i of lanes[k]`.
#[cfg(any(
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
fn store_pass_prefix<V: I32Lanes>(
    lanes: &[V],
    first: usize,
    stride: usize,
    buf: &mut [MaybeUninit<i32>],
) {
    for k0 in (0..lanes.len()).step_by(V::LANES) {
        V::store_transposed(
            &lanes[k0..k0 + V::LANES],
            &mut buf[first * stride + k0..],
            stride,
        );
    }
}

/// Transpose the `N`x`N` block `src` into `dst` (`dst[c * N + r] = src[r *
/// N + c]`), converting between the column-major coefficient layout and
/// raster order. Each column strip is loaded once and stored as transposed
/// `V::LANES`-square tiles.
#[inline(always)]
pub(crate) fn transpose<V: I32Lanes, const N: usize>(src: &[i32], dst: &mut [i32]) {
    let src = &src[..N * N];
    let dst = core::ptr::from_mut(&mut dst[..N * N]) as *mut [MaybeUninit<i32>];
    // SAFETY: `MaybeUninit<i32>` has the layout of `i32`, and the view only
    // ever receives initialised values, so the `i32` slice stays valid.
    let dst = unsafe { &mut *dst };
    for c0 in (0..N).step_by(V::LANES) {
        let strip: [V; N] = core::array::from_fn(|r| V::load(&src[r * N + c0..]));
        for r0 in (0..N).step_by(V::LANES) {
            V::store_transposed(&strip[r0..r0 + V::LANES], &mut dst[c0 * N + r0..], N);
        }
    }
}

/// Defines the forward and inverse 2-D DCT of one size.
macro_rules! dct_2d {
    ($forward:ident, $forward_layout:ident, $columns:ident, $rows:ident, $inverse:ident,
     $n:literal, $fdct:ident, $idct:ident, $cfg:ident, $row_shift:literal) => {
        /// Forward 2-D DCT of an `n`x`n` residual (row-major at `stride`)
        /// into column-major coefficients `out`.
        ///
        /// # Safety
        ///
        /// `V`'s ISA must be available (module docs).
        #[inline(always)]
        pub(crate) unsafe fn $forward<V: I32Lanes>(
            residual: &[i16],
            stride: usize,
            out: &mut [i32],
        ) {
            // SAFETY: forwarded contract.
            unsafe { $forward_layout::<V, false>(residual, stride, out) }
        }

        /// The forward DCT storing column-major coefficients, or raster
        /// (row-major, VP9's) ones when `RASTER`: the row pass's outputs
        /// are then stored as transposed tiles.
        ///
        /// # Safety
        ///
        /// As the column-major form.
        #[inline(always)]
        pub(crate) unsafe fn $forward_layout<V: I32Lanes, const RASTER: bool>(
            residual: &[i16],
            stride: usize,
            out: &mut [i32],
        ) {
            const N: usize = $n;
            let mut buf = [MaybeUninit::<i32>::uninit(); N * N];
            // SAFETY: forwarded contract.
            let transposed = unsafe { $columns::<V>(residual, stride, &mut buf) };
            // SAFETY: forwarded contract.
            unsafe { $rows::<V, RASTER>(transposed, out) };
        }

        /// The column pass: column `c`'s outputs at `transposed[c * N..]`.
        ///
        /// # Safety
        ///
        /// As the column-major form.
        #[inline(always)]
        pub(crate) unsafe fn $columns<'buf, V: I32Lanes>(
            residual: &[i16],
            stride: usize,
            buf: &'buf mut [MaybeUninit<i32>; $n * $n],
        ) -> &'buf [i32] {
            const N: usize = $n;
            let cfg = &$cfg;
            let mut x = [V::splat(0); N];
            let mut y = [V::splat(0); N];
            for c0 in (0..N).step_by(V::LANES) {
                for (r, lane) in x.iter_mut().enumerate() {
                    *lane = V::load_i16(&residual[r * stride + c0..]).shl(cfg.input_shift);
                }
                dct::$fdct(&x, &mut y, cospi(cfg.cos_bit_col), cfg.cos_bit_col);
                for lane in &mut y {
                    *lane = lane.round_shift(cfg.mid_shift);
                }
                // Column c0 + i's outputs become row c0 + i of the buffer.
                store_pass::<V>(&y, c0, N, buf);
            }
            // SAFETY: the loop above wrote every element (for each c0, rows
            // c0..c0 + LANES of the buffer, all N columns), and
            // `MaybeUninit<i32>` has the layout of `i32`. A view, not a
            // `transmute` by value, which would copy the buffer.
            let elements = buf.as_ptr().cast::<i32>();
            // SAFETY: as stated above.
            unsafe { core::slice::from_raw_parts(elements, N * N) }
        }

        /// The row pass over the column pass's output, into column-major or
        /// (`RASTER`) raster `out`.
        ///
        /// # Safety
        ///
        /// As the column-major form.
        #[inline(always)]
        pub(crate) unsafe fn $rows<V: I32Lanes, const RASTER: bool>(
            transposed: &[i32],
            out: &mut [i32],
        ) {
            const N: usize = $n;
            let cfg = &$cfg;
            let mut x = [V::splat(0); N];
            let mut y = [V::splat(0); N];
            for r0 in (0..N).step_by(V::LANES) {
                for (c, lane) in x.iter_mut().enumerate() {
                    *lane = V::load(&transposed[c * N + r0..]);
                }
                dct::$fdct(&x, &mut y, cospi(cfg.cos_bit_row), cfg.cos_bit_row);
                if RASTER {
                    // Lane i of y[k] is coefficient (r0 + i, k).
                    store_pass::<V>(&y, r0, N, as_uninit(&mut out[..N * N]));
                } else {
                    for (k, lane) in y.iter().enumerate() {
                        lane.store(&mut out[k * N + r0..]);
                    }
                }
            }
        }

        /// Inverse 2-D DCT of column-major `coeffs`, added to the `n`x`n`
        /// prediction in `dst` (at `dst_stride`) with 8-bit clipping.
        ///
        /// # Safety
        ///
        /// `V`'s ISA must be available (module docs).
        #[inline(always)]
        pub(crate) unsafe fn $inverse<V: I32Lanes>(
            coeffs: &[i32],
            dst: &mut [u8],
            dst_stride: usize,
        ) {
            const N: usize = $n;
            let mut buf = [MaybeUninit::<i32>::uninit(); N * N];
            let mut x = [V::splat(0); N];
            let mut y = [V::splat(0); N];
            for r0 in (0..N).step_by(V::LANES) {
                for (c, lane) in x.iter_mut().enumerate() {
                    *lane = V::load(&coeffs[c * N + r0..]).clamp16();
                }
                dct::$idct(&x, &mut y, &COSPI_12, INVERSE_COS_BIT);
                for lane in &mut y {
                    *lane = lane.round_shift($row_shift);
                }
                // Row r0 + i's outputs become row r0 + i of the buffer.
                store_pass::<V>(&y, r0, N, &mut buf);
            }
            // SAFETY: the loop above wrote every element (for each r0, rows
            // r0..r0 + LANES of the buffer, all N columns), and
            // `MaybeUninit<i32>` has the layout of `i32`. A view, not a
            // `transmute` by value, which would copy the buffer.
            let transposed: &[i32] = {
                let elements = buf.as_ptr().cast::<i32>();
                // SAFETY: as stated above.
                unsafe { core::slice::from_raw_parts(elements, N * N) }
            };
            for c0 in (0..N).step_by(V::LANES) {
                for (r, lane) in x.iter_mut().enumerate() {
                    *lane = V::load(&transposed[r * N + c0..]).clamp16();
                }
                dct::$idct(&x, &mut y, &COSPI_12, INVERSE_COS_BIT);
                for (r, lane) in y.iter().enumerate() {
                    lane.round_shift(INVERSE_COLUMN_SHIFT)
                        .add_to_u8(&mut dst[r * dst_stride + c0..]);
                }
            }
        }
    };
}

// Row shifts: libaom `inv_shift_NxN[0]`, negated.
dct_2d!(
    forward_4,
    forward_4_layout,
    forward_4_column_pass,
    forward_4_row_pass,
    inverse_4,
    4,
    fdct4,
    idct4,
    FORWARD_4,
    0
);
dct_2d!(
    forward_8,
    forward_8_layout,
    forward_8_column_pass,
    forward_8_row_pass,
    inverse_8,
    8,
    fdct8,
    idct8,
    FORWARD_8,
    1
);
dct_2d!(
    forward_16,
    forward_16_layout,
    forward_16_column_pass,
    forward_16_row_pass,
    inverse_16,
    16,
    fdct16,
    idct16,
    FORWARD_16,
    2
);
dct_2d!(
    forward_32,
    forward_32_layout,
    forward_32_column_pass,
    forward_32_row_pass,
    inverse_32,
    32,
    fdct32,
    idct32,
    FORWARD_32,
    2
);

/// Defines `$name<V>`, the raster `$n`-point forward with a proven-zero
/// test between the passes: every row output is at most its input line's
/// L1 norm plus `FORWARD_ROW_ROUNDING` (coefficients at most 1; checked by
/// `range_probe`), so when every line's L1 stays within `zero_limit` minus
/// that rounding, every coefficient has magnitude at most `zero_limit` and
/// the row pass is skipped (returning false, `out` untouched). Otherwise it
/// returns true with the forward's exact coefficients.
macro_rules! dct_bounded {
    ($name:ident, $n:literal, $columns:ident, $rows:ident, $rounding:expr) => {
        /// The bounded raster forward (see `dct_bounded!`).
        ///
        /// # Safety
        ///
        /// `V`'s ISA must be available (module docs).
        #[inline(always)]
        pub(crate) unsafe fn $name<V: I32Lanes>(
            residual: &[i16],
            stride: usize,
            zero_limit: u32,
            out: &mut [i32],
        ) -> bool {
            const N: usize = $n;
            let mut buf = [MaybeUninit::<i32>::uninit(); N * N];
            // SAFETY: forwarded contract.
            let transposed = unsafe { $columns::<V>(residual, stride, &mut buf) };
            if line_l1_within::<V, N>(transposed, zero_limit.saturating_sub($rounding)) {
                return false;
            }
            // SAFETY: forwarded contract.
            unsafe { $rows::<V, true>(transposed, out) };
            true
        }
    };
}

/// Whether the L1 norm of every row-pass input line (line `r`: element `r`
/// of every column's outputs, `transposed[c * N + r]`) is at most `limit`.
#[inline(always)]
pub(crate) fn line_l1_within<V: I32Lanes, const N: usize>(transposed: &[i32], limit: u32) -> bool {
    let limit = V::splat(limit.min(i32::MAX as u32) as i32);
    let mut over = 0u32;
    for r0 in (0..N).step_by(V::LANES) {
        let mut l1 = V::splat(0);
        for c in 0..N {
            l1 = l1.add(V::load(&transposed[c * N + r0..]).abs());
        }
        over += l1.gt(limit).count_nonzero();
    }
    over == 0
}

dct_bounded!(
    forward_16_bounded,
    16,
    forward_16_column_pass,
    forward_16_row_pass,
    super::FORWARD_ROW_ROUNDING[0]
);
dct_bounded!(
    forward_32_bounded,
    32,
    forward_32_column_pass,
    forward_32_row_pass,
    super::FORWARD_ROW_ROUNDING[1]
);

/// Defines `$name<V, KEEP, RASTER>`, the low-frequency `$n`-point forward
/// (`txfm::low_keep`): the column pass keeps its `KEEP` lowest outputs, the
/// row pass runs only on those and keeps its `KEEP` lowest, column-major or
/// (`RASTER`) raster; the rest of `out` is zero. Equal to the full forward
/// with the other coefficients zeroed (each kept output is computed exactly
/// as the full forward computes it).
macro_rules! dct_low {
    ($name:ident, $n:literal, $fdct:ident, $cfg:ident) => {
        /// The low-frequency forward (see `dct_low!`).
        ///
        /// # Safety
        ///
        /// `V`'s ISA must be available (module docs); `V::LANES` divides
        /// `KEEP`.
        #[cfg(any(
            target_arch = "x86_64",
            all(target_arch = "wasm32", target_feature = "simd128")
        ))]
        #[inline(always)]
        pub(crate) unsafe fn $name<V: I32Lanes, const KEEP: usize, const RASTER: bool>(
            residual: &[i16],
            stride: usize,
            out: &mut [i32],
        ) {
            const N: usize = $n;
            let cfg = &$cfg;
            let out = &mut out[..N * N];
            // Only discarded coefficients need zeroing; the kept square
            // is completely overwritten below in either output layout.
            for row in out[..KEEP * N].chunks_exact_mut(N) {
                row[KEEP..].fill(0);
            }
            out[KEEP * N..].fill(0);
            // Column `c`'s `KEEP` lowest outputs at `buf[c * KEEP..]`.
            let mut buf = [MaybeUninit::<i32>::uninit(); N * N];
            let mut x = [V::splat(0); N];
            let mut y = [V::splat(0); N];
            for c0 in (0..N).step_by(V::LANES) {
                for (r, lane) in x.iter_mut().enumerate() {
                    *lane = V::load_i16(&residual[r * stride + c0..]).shl(cfg.input_shift);
                }
                dct::$fdct(&x, &mut y, cospi(cfg.cos_bit_col), cfg.cos_bit_col);
                for lane in &mut y[..KEEP] {
                    *lane = lane.round_shift(cfg.mid_shift);
                }
                store_pass::<V>(&y[..KEEP], c0, KEEP, &mut buf);
            }
            let transposed: &[i32] = {
                let elements = buf.as_ptr().cast::<i32>();
                // SAFETY: the loop above wrote buffer rows c0..c0 + LANES
                // (all `KEEP` values each) for every c0: `N * KEEP` values.
                unsafe { core::slice::from_raw_parts(elements, N * KEEP) }
            };
            for r0 in (0..KEEP).step_by(V::LANES) {
                for (c, lane) in x.iter_mut().enumerate() {
                    *lane = V::load(&transposed[c * KEEP + r0..]);
                }
                dct::$fdct(&x, &mut y, cospi(cfg.cos_bit_row), cfg.cos_bit_row);
                if RASTER {
                    // Lane i of y[k] is coefficient (r0 + i, k).
                    store_pass_prefix::<V>(&y[..KEEP], r0, N, as_uninit(out));
                } else {
                    for (k, lane) in y[..KEEP].iter().enumerate() {
                        lane.store(&mut out[k * N + r0..]);
                    }
                }
            }
        }
    };
}

dct_low!(forward_16_low, 16, fdct16, FORWARD_16);
dct_low!(forward_32_low, 32, fdct32_low16, FORWARD_32);

/// 64x64 forward configuration (libaom `fwd_shift_64x64` = {0, -2, -2},
/// `av1_fwd_cos_bit_col/row` = 13 / 10).
const FORWARD_64: ForwardConfig = ForwardConfig {
    input_shift: 0,
    mid_shift: 2,
    cos_bit_col: 13,
    cos_bit_row: 10,
};
/// Rounding shift after the 64x64 row pass (`fwd_shift_64x64[2]`).
const FORWARD_64_FINAL_SHIFT: u32 = 2;
/// Coefficients kept by a 64x64 transform: the top-left 32x32
/// (libaom `av1_get_adjusted_tx_size`).
pub(super) const CODED_64: usize = 32;
/// libaom `inv_shift_64x64[0]`, negated.
pub(super) const INVERSE_64_ROW_SHIFT: u32 = 2;

/// Forward 64x64 DCT of a residual (row-major at `stride`), keeping the
/// top-left 32x32 coefficients, column-major, in `out` (libaom
/// `fwd_txfm2d_c` with `TX_64X64`), on the full 64-point graph: the scalar
/// reference that [`forward_64_pruned`] is checked against.
///
/// # Safety
///
/// `V`'s ISA must be available (module docs).
#[inline(always)]
pub(crate) unsafe fn forward_64<V: I32Lanes>(residual: &[i16], stride: usize, out: &mut [i32]) {
    // SAFETY: forwarded caller contract.
    unsafe { forward_64_impl::<V, false>(residual, stride, out) }
}

/// [`forward_64`] with both passes pruned to the 32 lowest outputs
/// (`dct::fdct64_low32`): only those outputs of each column transform feed
/// the row transforms, and only those of each row transform are kept, so the
/// butterflies feeding outputs 32..63 are skipped. Bit-exact with
/// [`forward_64`] (checkasm).
///
/// # Safety
///
/// `V`'s ISA must be available (module docs).
// The AVX2 and SIMD128 tables use it; NEON has its own pruned driver.
#[cfg(any(
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
pub(crate) unsafe fn forward_64_pruned<V: I32Lanes>(
    residual: &[i16],
    stride: usize,
    out: &mut [i32],
) {
    // SAFETY: forwarded caller contract.
    unsafe { forward_64_impl::<V, true>(residual, stride, out) }
}

#[inline(always)]
unsafe fn forward_64_impl<V: I32Lanes, const LOW32: bool>(
    residual: &[i16],
    stride: usize,
    out: &mut [i32],
) {
    const N: usize = 64;
    let cfg = &FORWARD_64;
    // Rows 0..31 of the column pass's output, transposed: `buf[c * 32 + r]`.
    let mut buf = [MaybeUninit::<i32>::uninit(); N * CODED_64];
    let mut x = [V::splat(0); N];
    let mut y = [V::splat(0); N];
    for c0 in (0..N).step_by(V::LANES) {
        for (r, lane) in x.iter_mut().enumerate() {
            *lane = V::load_i16(&residual[r * stride + c0..]).shl(cfg.input_shift);
        }
        if LOW32 {
            dct::fdct64_low32(&x, &mut y, cospi(cfg.cos_bit_col), cfg.cos_bit_col);
        } else {
            dct::fdct64(&x, &mut y, cospi(cfg.cos_bit_col), cfg.cos_bit_col);
        }
        for lane in &mut y[..CODED_64] {
            *lane = lane.round_shift(cfg.mid_shift);
        }
        // Column c0 + i's low outputs become row c0 + i of the buffer.
        store_pass::<V>(&y[..CODED_64], c0, CODED_64, &mut buf);
    }
    // SAFETY: the loop above wrote every element (for each c0, buffer rows
    // c0..c0 + LANES, all 32 columns), and `MaybeUninit<i32>` has the layout
    // of `i32`.
    let transposed: &[i32] = {
        let elements = buf.as_ptr().cast::<i32>();
        // SAFETY: as stated above.
        unsafe { core::slice::from_raw_parts(elements, N * CODED_64) }
    };
    forward_64_rows_impl::<V, LOW32>(transposed, out);
}

/// The low-frequency 64x64 forward (`txfm::low_keep`): the pruned 64-point
/// graphs of [`forward_64_pruned`], with the column pass keeping its `KEEP`
/// lowest outputs and the row pass running only on those and keeping its
/// `KEEP` lowest, column-major in the coded 32x32 square; the rest of `out`
/// is zero. Equal to [`forward_64_pruned`] with the other coefficients
/// zeroed (each kept output is computed as the pruned forward computes it).
/// The x86 counterpart of NEON's `forward_64_low`: the full pruned forward
/// was 1.2-2.0 % of x86 AV1 Faster instructions (2026-11-11).
///
/// # Safety
///
/// `V`'s ISA must be available (module docs); `V::LANES` divides `KEEP`.
#[cfg(any(
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
pub(crate) unsafe fn forward_64_low<V: I32Lanes, const KEEP: usize>(
    residual: &[i16],
    stride: usize,
    out: &mut [i32],
) {
    const N: usize = 64;
    let cfg = &FORWARD_64;
    let out = &mut out[..CODED_64 * CODED_64];
    out.fill(0);
    // Column `c`'s `KEEP` lowest outputs at `buf[c * KEEP..]`.
    let mut buf = [MaybeUninit::<i32>::uninit(); N * CODED_64];
    let mut x = [V::splat(0); N];
    let mut y = [V::splat(0); N];
    for c0 in (0..N).step_by(V::LANES) {
        for (r, lane) in x.iter_mut().enumerate() {
            *lane = V::load_i16(&residual[r * stride + c0..]).shl(cfg.input_shift);
        }
        dct::fdct64_low32(&x, &mut y, cospi(cfg.cos_bit_col), cfg.cos_bit_col);
        for lane in &mut y[..KEEP] {
            *lane = lane.round_shift(cfg.mid_shift);
        }
        store_pass::<V>(&y[..KEEP], c0, KEEP, &mut buf);
    }
    let transposed: &[i32] = {
        let elements = buf.as_ptr().cast::<i32>();
        // SAFETY: the loop above wrote buffer rows c0..c0 + LANES (all
        // `KEEP` values each) for every c0: `N * KEEP` values.
        unsafe { core::slice::from_raw_parts(elements, N * KEEP) }
    };
    for r0 in (0..KEEP).step_by(V::LANES) {
        for (c, lane) in x.iter_mut().enumerate() {
            *lane = V::load(&transposed[c * KEEP + r0..]);
        }
        dct::fdct64_low32(&x, &mut y, cospi(cfg.cos_bit_row), cfg.cos_bit_row);
        for (k, lane) in y[..KEEP].iter().enumerate() {
            lane.round_shift(FORWARD_64_FINAL_SHIFT)
                .store(&mut out[k * CODED_64 + r0..]);
        }
    }
}

/// Wide32-point reference row pass for the native benchmark baseline.
#[cfg(all(target_arch = "aarch64", test))]
#[inline(always)]
pub(crate) fn forward_32_rows<V: I32Lanes>(transposed: &[i32], out: &mut [i32]) {
    const N: usize = 32;
    let mut x = [V::splat(0); N];
    let mut y = [V::splat(0); N];
    for r0 in (0..N).step_by(V::LANES) {
        for (c, lane) in x.iter_mut().enumerate() {
            *lane = V::load(&transposed[c * N + r0..]);
        }
        dct::fdct32(
            &x,
            &mut y,
            cospi(FORWARD_32.cos_bit_row),
            FORWARD_32.cos_bit_row,
        );
        for (k, lane) in y.iter().enumerate() {
            lane.store(&mut out[k * N + r0..]);
        }
    }
}

/// Pruned row pass for the NEON low-bit-depth driver; the scalar oracle keeps
/// the full graph in both passes.
#[cfg(all(target_arch = "aarch64", test))]
pub(crate) fn forward_64_rows_low32<V: I32Lanes>(transposed: &[i32], out: &mut [i32]) {
    forward_64_rows_impl::<V, true>(transposed, out);
}

#[inline(always)]
fn forward_64_rows_impl<V: I32Lanes, const LOW32: bool>(transposed: &[i32], out: &mut [i32]) {
    const N: usize = 64;
    let cfg = &FORWARD_64;
    let mut x = [V::splat(0); N];
    let mut y = [V::splat(0); N];
    for r0 in (0..CODED_64).step_by(V::LANES) {
        for (c, lane) in x.iter_mut().enumerate() {
            *lane = V::load(&transposed[c * CODED_64 + r0..]);
        }
        if LOW32 {
            dct::fdct64_low32(&x, &mut y, cospi(cfg.cos_bit_row), cfg.cos_bit_row);
        } else {
            dct::fdct64(&x, &mut y, cospi(cfg.cos_bit_row), cfg.cos_bit_row);
        }
        for (k, lane) in y[..CODED_64].iter().enumerate() {
            lane.round_shift(FORWARD_64_FINAL_SHIFT)
                .store(&mut out[k * CODED_64 + r0..]);
        }
    }
}

/// Inverse 64x64 DCT of the top-left 32x32 column-major `coeffs` (the rest
/// are zero), added to the 64x64 prediction in `dst` (at `dst_stride`) with
/// 8-bit clipping (libaom `inv_txfm2d_add_c` with `TX_64X64`). Rows 32..63
/// of the input are zero and transform to zero, so only 32 row transforms
/// run, and the column transforms see zeros in their upper half.
///
/// # Safety
///
/// `V`'s ISA must be available (module docs).
#[inline(always)]
pub(crate) unsafe fn inverse_64<V: I32Lanes>(coeffs: &[i32], dst: &mut [u8], dst_stride: usize) {
    const N: usize = 64;
    // The 32 row transforms' outputs, row-major: `buf[r * 64 + c]`.
    let mut buf = [MaybeUninit::<i32>::uninit(); CODED_64 * N];
    let mut x = [V::splat(0); N];
    let mut y = [V::splat(0); N];
    for r0 in (0..CODED_64).step_by(V::LANES) {
        for (c, lane) in x[..CODED_64].iter_mut().enumerate() {
            *lane = V::load(&coeffs[c * CODED_64 + r0..]).clamp16();
        }
        for lane in &mut x[CODED_64..] {
            *lane = V::splat(0);
        }
        dct::idct64(&x, &mut y, &COSPI_12, INVERSE_COS_BIT);
        for lane in &mut y {
            *lane = lane.round_shift(INVERSE_64_ROW_SHIFT);
        }
        // Row r0 + i's outputs become row r0 + i of the buffer.
        store_pass::<V>(&y, r0, N, &mut buf);
    }
    // SAFETY: the loop above wrote every element (for each r0, buffer rows
    // r0..r0 + LANES, all 64 columns), and `MaybeUninit<i32>` has the layout
    // of `i32`.
    let rows: &[i32] = {
        let elements = buf.as_ptr().cast::<i32>();
        // SAFETY: as stated above.
        unsafe { core::slice::from_raw_parts(elements, CODED_64 * N) }
    };
    for c0 in (0..N).step_by(V::LANES) {
        for (r, lane) in x[..CODED_64].iter_mut().enumerate() {
            *lane = V::load(&rows[r * N + c0..]).clamp16();
        }
        for lane in &mut x[CODED_64..] {
            *lane = V::splat(0);
        }
        dct::idct64(&x, &mut y, &COSPI_12, INVERSE_COS_BIT);
        for (r, lane) in y.iter().enumerate() {
            lane.round_shift(INVERSE_COLUMN_SHIFT)
                .add_to_u8(&mut dst[r * dst_stride + c0..]);
        }
    }
}

// EOB-derived zero extents select graphs with structural-zero arithmetic
// removed at generation time. Full inverse kernels remain independent references.
// Native five-pair300-frame AV1 CPU falls6.53% bodycam/2.98% crowd;
// streams remain exact. Same-path host GOP300 instructions fall0.91%.
macro_rules! sparse_inverse {
    ($name:ident, $n:literal, $coded:literal, $row_shift:literal,
     $low8:ident, $low16:ident, $full:ident) => {
        /// Inverse with coefficients confined to a low8/16/32 square.
        ///
        /// Coefficients outside `limit` must be zero for numerical equivalence;
        /// coefficient values do not affect memory safety.
        ///
        /// # Safety
        /// `V`'s ISA must be available and `limit` must be8,16 or32.
        #[inline(always)]
        pub(crate) unsafe fn $name<V: I32Lanes>(
            coeffs: &[i32],
            dst: &mut [u8],
            stride: usize,
            limit: usize,
        ) {
            const N: usize = $n;
            let mut buf = [MaybeUninit::<i32>::uninit(); $coded * N];
            let mut x = [V::splat(0); N];
            let mut y = [V::splat(0); N];
            for r0 in (0..limit).step_by(V::LANES) {
                for (c, lane) in x[..limit].iter_mut().enumerate() {
                    *lane = V::load(&coeffs[c * $coded + r0..]).clamp16();
                }
                match limit {
                    8 => dct::$low8(&x, &mut y, &COSPI_12, INVERSE_COS_BIT),
                    16 => dct::$low16(&x, &mut y, &COSPI_12, INVERSE_COS_BIT),
                    _ => dct::$full(&x, &mut y, &COSPI_12, INVERSE_COS_BIT),
                }
                for lane in &mut y {
                    *lane = lane.round_shift($row_shift);
                }
                store_pass::<V>(&y, r0, N, &mut buf);
            }
            // SAFETY: store_pass initialized exactly the first limit complete
            // rows. The remaining MaybeUninit rows are neither viewed nor read.
            let rows = {
                // SAFETY: exactly limit complete rows were initialized above.
                unsafe { core::slice::from_raw_parts(buf.as_ptr().cast::<i32>(), limit * N) }
            };
            for c0 in (0..N).step_by(V::LANES) {
                for (r, lane) in x[..limit].iter_mut().enumerate() {
                    *lane = V::load(&rows[r * N + c0..]).clamp16();
                }
                match limit {
                    8 => dct::$low8(&x, &mut y, &COSPI_12, INVERSE_COS_BIT),
                    16 => dct::$low16(&x, &mut y, &COSPI_12, INVERSE_COS_BIT),
                    _ => dct::$full(&x, &mut y, &COSPI_12, INVERSE_COS_BIT),
                }
                for (r, lane) in y.iter().enumerate() {
                    lane.round_shift(INVERSE_COLUMN_SHIFT)
                        .add_to_u8(&mut dst[r * stride + c0..]);
                }
            }
        }
    };
}
sparse_inverse!(
    inverse_32_sparse,
    32,
    32,
    2,
    idct32_low8,
    idct32_low16,
    idct32
);
sparse_inverse!(
    inverse_64_sparse,
    64,
    32,
    2,
    idct64_low8,
    idct64_low16,
    idct64_low32
);

/// AV1 dequantisation (`dequantize_one`, AV1 §7.12.3) of `levels` into
/// `out`, `V::LANES` AC levels at a time; coefficient 0 (the DC step) and a
/// ragged tail use the reference.
///
/// Exact for every `i32` level: only the low 24 bits of `|level| · step`
/// survive the mask, and those depend only on the low 32 bits of the
/// operands, so a wrapping 32-bit multiply suffices (`|i32::MIN|` wraps to
/// the same bits as its unsigned magnitude). The mask is a left shift by 8
/// and a logical right shift by `8 + log_scale`.
///
/// # Safety
///
/// `V`'s ISA must be available (module docs). `out.len() >= levels.len()`.
#[inline(always)]
pub(crate) unsafe fn dequantize<V: I32Lanes>(
    levels: &[i32],
    q: &super::Av1Quantizer,
    out: &mut [i32],
) {
    /// Bits above the 24-bit dequantisation mask in a 32-bit lane.
    const ABOVE_MASK: u32 = 32 - super::DEQUANT_MASK.count_ones();
    let Some((&dc, ac)) = levels.split_first() else {
        return;
    };
    out[0] = super::dequantize_one(dc, 0, q);
    let step = V::splat(q.ac_step);
    let (lo, hi) = (V::splat(i32::from(i16::MIN)), V::splat(i32::from(i16::MAX)));
    let vector_len = ac.len() / V::LANES * V::LANES;
    let out_ac = &mut out[1..=ac.len()];
    for (src, dst) in ac[..vector_len]
        .chunks_exact(V::LANES)
        .zip(out_ac.chunks_exact_mut(V::LANES))
    {
        let level = V::load(src);
        level
            .abs()
            .mul(step)
            .shl(ABOVE_MASK)
            .shr_logical(ABOVE_MASK + q.log_scale)
            .with_sign_of(level)
            .max(lo)
            .min(hi)
            .store(dst);
    }
    for (i, (&level, dq)) in ac
        .iter()
        .zip(out_ac.iter_mut())
        .enumerate()
        .skip(vector_len)
    {
        *dq = super::dequantize_one(level, i + 1, q);
    }
}
