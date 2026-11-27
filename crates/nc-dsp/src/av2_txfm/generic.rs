//! The AV2 DCT pass drivers generic over [`I32Lanes`]: each 1-D pass
//! transforms `V::LANES` lines at once, with register tile transposes
//! between passes (as `codec_txfm::generic`). The `i32` instantiation is
//! the scalar reference.
//!
//! The 1-D transforms are the partial butterflies shared with HEVC
//! (`codec_txfm::butterfly`) on AV2's basis [`DCT_BASIS`]: the integer
//! matrices nest (module docs of `av2_txfm`), so the result is exactly
//! AVM's matrix product.
//!
//! The functions are `unsafe` only because `V`'s instructions need its ISA
//! (see the `simd` module docs): call them from a `#[target_feature]`
//! function of that ISA, or with the scalar `i32` lane type.

#![allow(clippy::inline_always)]
#![allow(unsafe_code)]

use core::mem::MaybeUninit;

use super::{DC_WEIGHT, DCT_BASIS, MAX_CODED_SIDE, MAX_POINTS};
use crate::codec_txfm::butterfly::{DctBasis, forward_dct, inverse_dct, odd_matrix};
use crate::codec_txfm::generic::{as_uninit, written};
use crate::simd::I32Lanes;
use crate::txfm::generic::{store_pass, store_pass_sat16};

/// Coded coefficients of the largest transform.
const MAX_CODED_AREA: usize = MAX_CODED_SIDE * MAX_CODED_SIDE;

/// Range of the inverse column pass output: the 8-bit residual range
/// (AVM `inv_txfm_c`: `±(1 << bd)`, upper bound exclusive).
pub(super) const RESIDUAL_MIN: i32 = -256;
pub(super) const RESIDUAL_MAX: i32 = 255;

/// Inverse of a decoded 2:1 DCT rectangle. Decode stores its coefficients
/// column-major, so a group of neighbouring rows is already in the lane
/// order of the shared row butterfly.
///
/// # Safety
/// V's ISA is enabled, its lane count divides both dimensions, and
/// `rows` bounds the nonzero rows of the active coefficient scan.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
pub(crate) unsafe fn inverse_rect_from_columns<
    V: I32Lanes,
    const W: usize,
    const H: usize,
    const AREA: usize,
    const NORMALIZE: bool,
>(
    coefficients: &[i16; AREA],
    rows: usize,
    shifts: (u32, u32),
    residual: &mut [i16; AREA],
) {
    use crate::av2_dec::itx::{RECT2_SCALE, RECT2_SHIFT};
    const { assert!(AREA == W * H) };
    let mut intermediate = [0i16; AREA];
    let zero = V::splat(0);
    let scale = V::splat(RECT2_SCALE);
    let mut horizontal = [zero; W];
    let mut horizontal_out = [zero; W];
    for row in (0..rows).step_by(V::LANES) {
        for (x, lane) in horizontal.iter_mut().enumerate() {
            let coefficient = V::load_i16(&coefficients[x * H + row..]);
            *lane = if NORMALIZE {
                coefficient.mul(scale).round_shift(RECT2_SHIFT)
            } else {
                coefficient
            };
        }
        inverse_dct::<Av2Basis, V, W>(&horizontal, W, &mut horizontal_out);
        for lane in &mut horizontal_out {
            *lane = lane.round_shift(shifts.0);
        }
        store_pass_sat16::<V>(&horizontal_out, row, W, as_uninit(&mut intermediate));
    }

    let (low, high) = (V::splat(RESIDUAL_MIN), V::splat(RESIDUAL_MAX));
    let mut vertical = [zero; H];
    let mut vertical_out = [zero; H];
    for x in (0..W).step_by(V::LANES) {
        for (row, lane) in vertical.iter_mut().enumerate().take(rows) {
            *lane = V::load_i16(&intermediate[row * W + x..]);
        }
        inverse_dct::<Av2Basis, V, H>(&vertical, rows, &mut vertical_out);
        for (row, lane) in vertical_out.iter().enumerate() {
            let sample = lane.round_shift(shifts.1).max(low).min(high);
            sample.store_sat16(&mut residual[row * W + x..]);
        }
    }
}

/// AV2's basis ([`DCT_BASIS`], up to 64 points) for the shared butterflies.
pub(super) struct Av2Basis;

impl DctBasis for Av2Basis {
    const DC: i32 = DC_WEIGHT;
    const ODD2: [[i32; 1]; 1] = odd_matrix::<1, MAX_POINTS>(&DCT_BASIS);
    const ODD4: [[i32; 2]; 2] = odd_matrix::<2, MAX_POINTS>(&DCT_BASIS);
    const ODD8: [[i32; 4]; 4] = odd_matrix::<4, MAX_POINTS>(&DCT_BASIS);
    const ODD16: [[i32; 8]; 8] = odd_matrix::<8, MAX_POINTS>(&DCT_BASIS);
    const ODD32: [[i32; 16]; 16] = odd_matrix::<16, MAX_POINTS>(&DCT_BASIS);
    const ODD64: Option<[[i32; 32]; 32]> = Some(odd_matrix::<32, MAX_POINTS>(&DCT_BASIS));
}

/// Inverse 2-D DCT of side `C` (AVM `inv_txfm_c` without the pixel add):
/// row-major `coeffs` (`C * C`) to the row-major residual `out` (`C * C`).
/// Rows and columns from `nonzero` on must be all zero (`C` when nothing
/// is known): a row of zeros transforms to zeros, so those rows skip their
/// pass, and both passes skip the products of the zero inputs. Low-rate
/// blocks end early in the scan, so most of a large transform's
/// coefficients are zero.
///
/// The coefficients, the row pass's clamped output and the residual are
/// 16-bit values (AVM `clamp_buf` to `bd + 8` bits, then `±(1 << bd)`), so
/// they are stored as `i16`; only the products and their sums before each
/// rounding shift need the 32-bit lanes (they reach `2^27`, module docs).
///
/// # Safety
///
/// `V`'s ISA must be available (module docs); `V::LANES <= C`.
#[inline(always)]
pub(crate) unsafe fn inverse<V: I32Lanes, const C: usize>(
    coeffs: &[i16],
    nonzero: usize,
    shifts: (u32, u32),
    out: &mut [i16],
) {
    // Each arm inlines a copy of the passes with its input bound constant
    // (see `corner_bound`).
    // SAFETY: forwarded contract.
    unsafe {
        match corner_bound::<C>(nonzero) {
            CornerBound::Quarter => inverse_within::<V, C>(coeffs, nonzero, C / 4, shifts, out),
            CornerBound::Half => inverse_within::<V, C>(coeffs, nonzero, C / 2, shifts, out),
            CornerBound::Full => inverse_within::<V, C>(coeffs, nonzero, C, shifts, out),
        }
    }
}

/// The 1-D inverse passes are compiled for three input bounds: a corner
/// within a quarter or a half of the side, or any.
#[derive(Clone, Copy)]
pub(super) enum CornerBound {
    Quarter,
    Half,
    Full,
}

/// The smallest [`CornerBound`] of side `C` holding a `nonzero` corner.
#[inline(always)]
pub(super) const fn corner_bound<const C: usize>(nonzero: usize) -> CornerBound {
    if nonzero <= C / 4 {
        CornerBound::Quarter
    } else if nonzero <= C / 2 {
        CornerBound::Half
    } else {
        CornerBound::Full
    }
}

/// [`inverse`] with every 1-D pass's inputs from `inputs` (at least
/// `nonzero`) on known to be zero.
///
/// # Safety
///
/// As [`inverse`].
#[inline(always)]
unsafe fn inverse_within<V: I32Lanes, const C: usize>(
    coeffs: &[i16],
    nonzero: usize,
    inputs: usize,
    shifts: (u32, u32),
    out: &mut [i16],
) {
    debug_assert!(nonzero <= inputs);
    let lanes = V::LANES;
    let mut x = [V::splat(0); C];
    let mut y = [V::splat(0); C];
    // The input transposed, `t[k * C + r] = coeffs[r][k]`, so that the row
    // pass is lane-parallel. Tiles outside the nonzero corner are stored as
    // zeros without loading.
    let mut t = [MaybeUninit::<i16>::uninit(); MAX_CODED_AREA];
    for r0 in (0..C).step_by(lanes) {
        for k0 in (0..C).step_by(lanes) {
            for (i, lane) in x[..lanes].iter_mut().enumerate() {
                *lane = if r0 < nonzero && k0 < nonzero {
                    V::load_i16(&coeffs[(r0 + i) * C + k0..])
                } else {
                    V::splat(0)
                };
            }
            V::store_transposed_sat16(&x[..lanes], &mut t[k0 * C + r0..], C);
        }
    }
    // SAFETY: every (r0, k0) tile of the C x C buffer was written above.
    let t = unsafe { written(&t, C * C) };
    // The row pass's output, row-major, clamped to 16 bits by the
    // saturating store (AVM `clamp_buf`).
    let mut m = [MaybeUninit::<i16>::uninit(); MAX_CODED_AREA];
    for r0 in (0..C).step_by(lanes) {
        if r0 < nonzero {
            for (k, lane) in x.iter_mut().enumerate() {
                *lane = V::load_i16(&t[k * C + r0..]);
            }
            inverse_dct::<Av2Basis, V, C>(&x, inputs, &mut y);
            for lane in &mut y {
                *lane = lane.round_shift(shifts.0);
            }
        } else {
            y = [V::splat(0); C];
        }
        store_pass_sat16::<V>(&y, r0, C, &mut m);
    }
    // SAFETY: rows r0..r0 + LANES of `m` were written for every r0.
    let m = unsafe { written(&m, C * C) };
    let (low, high) = (V::splat(RESIDUAL_MIN), V::splat(RESIDUAL_MAX));
    for x0 in (0..C).step_by(lanes) {
        for (r, lane) in x.iter_mut().enumerate() {
            *lane = V::load_i16(&m[r * C + x0..]);
        }
        // Rows of `m` from `nonzero` on are zero (zero coefficient rows).
        inverse_dct::<Av2Basis, V, C>(&x, inputs, &mut y);
        for (r, lane) in y.iter().enumerate() {
            let sample = lane.round_shift(shifts.1).max(low).min(high);
            sample.store_sat16(&mut out[r * C + x0..]);
        }
    }
}

/// Inverse 64x64 DCT: the 32x32 inverse of the coded coefficients, each
/// residual sample repeated in a 2x2 square (AVM `inv_txfm_c` for sides
/// above `MAX_TX_SIZE >> 1`).
///
/// # Safety
///
/// As [`inverse`].
#[inline(always)]
pub(crate) unsafe fn inverse64<V: I32Lanes>(
    coeffs: &[i16],
    nonzero: usize,
    shifts: (u32, u32),
    out: &mut [i16],
) {
    let mut coded = [0; MAX_CODED_AREA];
    // SAFETY: forwarded contract.
    unsafe { inverse::<V, MAX_CODED_SIDE>(coeffs, nonzero, shifts, &mut coded) };
    repeat_inverse64(&coded, out);
}

/// Expand the coded inverse's samples to the 64x64 reconstruction grid.
#[inline(always)]
pub(super) fn repeat_inverse64(coded: &[i16; MAX_CODED_AREA], out: &mut [i16]) {
    const SIDE: usize = 2 * MAX_CODED_SIDE;
    let (pairs, _) = out[..SIDE * SIDE].as_chunks_mut::<{ 2 * SIDE }>();
    let (rows, _) = coded.as_chunks::<MAX_CODED_SIDE>();
    for (pair, row) in pairs.iter_mut().zip(rows) {
        let (upper, lower) = pair.split_at_mut(SIDE);
        let (samples, _) = upper.as_chunks_mut::<2>();
        for (sample, &value) in samples.iter_mut().zip(row) {
            *sample = [value; 2];
        }
        lower.copy_from_slice(upper);
    }
}

/// Forward 2-D DCT of side `N` (AVM `fwd_txfm_c`) of the residual at
/// `stride` into the row-major coded coefficients `out` (`C * C`, `C =
/// min(N, 32)`).
///
/// # Safety
///
/// `V`'s ISA must be available (module docs); `V::LANES <= C`.
#[inline(always)]
pub(crate) unsafe fn forward<V: I32Lanes, const N: usize, const C: usize>(
    residual: &[i16],
    stride: usize,
    shifts: (u32, u32),
    out: &mut [i32],
) {
    let lanes = V::LANES;
    let mut x = [V::splat(0); N];
    let mut y = [V::splat(0); N];
    // The column pass's coded outputs, column-major: `g[col * C + k]`.
    let mut g = [MaybeUninit::<i32>::uninit(); MAX_POINTS * MAX_CODED_SIDE];
    for x0 in (0..N).step_by(lanes) {
        for (r, lane) in x.iter_mut().enumerate() {
            *lane = V::load_i16(&residual[r * stride + x0..]);
        }
        forward_dct::<Av2Basis, V, N>(&x, &mut y, C);
        for lane in &mut y[..C] {
            *lane = lane.round_shift(shifts.0);
        }
        store_pass::<V>(&y[..C], x0, C, &mut g);
    }
    // SAFETY: rows x0..x0 + LANES of `g` were written for every x0.
    let g = unsafe { written(&g, N * C) };
    forward_rows::<V, N, C>(g, &mut x, &mut y, shifts.1, out);
}

/// Complete the wide row pass from the shared column-major layout.
#[inline(always)]
pub(super) fn forward_rows<V: I32Lanes, const N: usize, const C: usize>(
    columns: &[i32],
    x: &mut [V; N],
    y: &mut [V; N],
    shift: u32,
    out: &mut [i32],
) {
    let out = as_uninit(&mut out[..C * C]);
    for k0 in (0..C).step_by(V::LANES) {
        for (column, lane) in x.iter_mut().enumerate() {
            *lane = V::load(&columns[column * C + k0..]);
        }
        forward_dct::<Av2Basis, V, N>(x, y, C);
        for lane in &mut y[..C] {
            *lane = lane.round_shift(shift);
        }
        store_pass::<V>(&y[..C], k0, C, out);
    }
}

/// Instantiates the AV2 kernels for one ISA: `$v4` lanes for 4x4, `$v`
/// otherwise.
macro_rules! av2_txfm_kernels {
    ($(#[$attr:meta])*; $v4:ty, $v:ty) => {
        use $crate::av2_txfm::generic as g;
        use $crate::av2_txfm::{Av2TxSize, Av2TxfmFns};

        /// The AV2 transform table of this ISA.
        pub(crate) fn av2_txfm_fns() -> Av2TxfmFns {
            Av2TxfmFns {
                forward: [forward4, forward8, forward16, forward32, forward64],
                inverse: [inverse4, inverse8, inverse16, inverse32, inverse64],
            }
        }

        $crate::av2_txfm::generic::av2_txfm_kernels!(@size [$(#[$attr])*]
            forward4, inverse4, $v4, Tx4, 4);
        $crate::av2_txfm::generic::av2_txfm_kernels!(@size [$(#[$attr])*]
            forward8, inverse8, $v, Tx8, 8);
        $crate::av2_txfm::generic::av2_txfm_kernels!(@size [$(#[$attr])*]
            forward16, inverse16, $v, Tx16, 16);
        $crate::av2_txfm::generic::av2_txfm_kernels!(@size [$(#[$attr])*]
            forward32, inverse32, $v, Tx32, 32);

        $(#[$attr])*
        unsafe fn forward64(residual: &[i16], stride: usize, out: &mut [i32]) {
            // SAFETY: see the module docs.
            unsafe {
                g::forward::<$v, 64, 32>(residual, stride, Av2TxSize::Tx64.forward_shifts(), out)
            }
        }

        $(#[$attr])*
        unsafe fn inverse64(coeffs: &[i16], nonzero: usize, out: &mut [i16]) {
            // SAFETY: see the module docs.
            unsafe { g::inverse64::<$v>(coeffs, nonzero, Av2TxSize::Tx64.inverse_shifts(), out) }
        }
    };
    (@size [$(#[$attr:meta])*]
     $forward:ident, $inverse:ident, $lanes:ty, $size:ident, $n:literal) => {
        $(#[$attr])*
        unsafe fn $forward(residual: &[i16], stride: usize, out: &mut [i32]) {
            // SAFETY: see the module docs.
            unsafe {
                g::forward::<$lanes, $n, $n>(
                    residual, stride, Av2TxSize::$size.forward_shifts(), out,
                )
            }
        }

        $(#[$attr])*
        unsafe fn $inverse(coeffs: &[i16], nonzero: usize, out: &mut [i16]) {
            // SAFETY: see the module docs.
            unsafe {
                g::inverse::<$lanes, $n>(coeffs, nonzero, Av2TxSize::$size.inverse_shifts(), out)
            }
        }
    };
}

pub(crate) use av2_txfm_kernels;
