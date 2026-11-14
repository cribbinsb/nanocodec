//! Guarded AV1 inverse transforms on eight or sixteen 16-bit lanes, written
//! once over [`NarrowLanes`] (AVX2 in `avx2_inverse`, NEON in
//! `neon_inverse`, SIMD128 in `simd128_inverse`): every size whose sides are
//! at least 8 (square, 2:1 and 4:1) and every 1-D kernel AV1 defines there
//! (DCT 8–64 with the EOB-sparse 32/64 graphs, ADST and flipped ADST 8/16,
//! identity 8–32).
//!
//! At 8-bit depth libaom clamps every inverse stage to 16 bits (AV1
//! §7.13.3). When the L1 norm of one 1-D transform's inputs is at most that
//! kernel's [`l1_limit`], no node of its graph can leave `i16` (proved by
//! `range_probe`), so every clamp is the identity and the graph runs exactly
//! on 16-bit lanes: wrapping adds, widening multiplies with a rounding
//! narrow, and rounding doubling-high multiplies for single and equal
//! weights. Each pass checks the norm of its transforms while loading their
//! inputs; a batch above the limit runs the libaom-exact 32-bit graph
//! instead, so the result equals the scalar reference for every input. Real
//! dequantised blocks are almost always far below the limits.
//!
//! The idea of 16-bit intermediates for 8-bit inverse transforms follows
//! dav1d (BSD-2-Clause; see `THIRD_PARTY.md`); dav1d relies on the bitstream
//! conformance range instead of a guard, so nothing here is transliterated.
//!
//! Arithmetic per pass follows libaom's `inv_txfm2d_add_c` (see the
//! `av1_inverse` module docs): row inputs clamped to 16 bits, scaled by
//! `1/sqrt(2)` for 2:1 rectangles, the row transform, the size's row shift,
//! then the column transform and a shift by 4, added with 8-bit clipping.
//! Flipped ADSTs reverse their outputs. Inside the guard the row outputs fit
//! 16 bits, which is libaom's column input clamp; the wide fallback shifts
//! and clamps in 32 bits.
//!
//! Kernel contract: see the `txfm` module docs. The drivers have no
//! `#[target_feature]` of their own: NEON and SIMD128 are on for the whole
//! build, and the AVX2 entry points enable AVX2, into which the drivers
//! inline (`inline(always)`; the rare wide fallbacks stay out of line).

#![allow(clippy::inline_always)]

use core::mem::MaybeUninit;

use super::generic::{CODED_64, INVERSE_COLUMN_SHIFT, INVERSE_COS_BIT};
use super::inverse_lanes::InverseLanes;
#[cfg(test)]
use super::inverse_lanes::NARROW_INVERSE_L1_LIMIT;
use super::{COSPI_12, dct};
use crate::av1_inverse::one_d::{iadst8, iadst16, identity};
use crate::av1_inverse::{Av1TxSize, Kernel1d};
use crate::simd::I32Lanes;

/// Side of the transposed tiles: eight 16-bit lanes (one 128-bit register,
/// or each 128-bit half of a 256-bit one).
pub(super) const GROUP: usize = 8;
/// Most transforms per batch (sixteen lanes of a 256-bit register).
const MAX_LANES: usize = 16;
/// Largest coded side (64-sample sides code only their first 32).
const CODED_MAX: usize = CODED_64;
/// The row shifts AV1 uses for sizes with both sides at least 8 (§7.13.3
/// `Transform_Row_Shift`): 1 or 2.
pub(super) const ROW_SHIFT_1: u32 = 1;
pub(super) const ROW_SHIFT_2: u32 = 2;
/// Rounding shift after the column pass.
pub(super) const COLUMN_SHIFT: u32 = INVERSE_COLUMN_SHIFT;
/// Row-pass outputs of the largest block: 32 coded rows of 64.
const ROWS_CAPACITY: usize = CODED_MAX * 64;
/// `round(2^12 / sqrt(2))`: the 2:1 rectangle input scale (§7.13.3).
const INV_SQRT2: i32 = 2896;

/// [`Self::LANES`] 16-bit lanes (one transform each) of an inverse pass,
/// exact only inside the L1 limit (the [`InverseLanes`] arithmetic wraps),
/// plus the loads, stores and shuffles the pass drivers need. Lanes come in
/// groups of [`GROUP`] (128-bit halves), and shuffles stay within a group.
pub(super) trait NarrowLanes: InverseLanes {
    /// Transforms per batch: [`GROUP`] or `2 * GROUP`.
    const LANES: usize;
    /// Per-lane L1 norms, saturating.
    type Norm: Copy;
    /// The libaom-exact 32-bit lanes of the fallback graph.
    type Wide: I32Lanes;
    fn norm_zero() -> Self::Norm;
    /// Add `|self|` to each lane's norm (saturating, so an overflowing norm
    /// still exceeds the limit; `|i16::MIN|` reads correctly as unsigned).
    fn accumulate_l1(self, norm: Self::Norm) -> Self::Norm;
    /// Every lane's norm is at most `limit`.
    fn within_limit(norm: Self::Norm, limit: u16) -> bool;
    /// The first `LANES` `i32` values of `src` saturated to 16 bits
    /// (libaom's input clamp).
    fn load_i32_saturating(src: &[i32]) -> Self;
    /// The first `LANES` values of `src`.
    fn load_i16(src: &[i16]) -> Self;
    /// Store lane group `g` (lanes `8g..8g + 8`) at `dst + g * group_stride`.
    ///
    /// # Safety
    ///
    /// Each of those eight-value spans must be writable.
    unsafe fn store_groups(self, dst: *mut i16, group_stride: usize);
    /// `round_shift(self, SHIFT)` exactly (no intermediate overflow).
    fn round_shift<const SHIFT: u32>(self) -> Self;
    /// The 8x8 transpose of eight rows, in each lane group independently.
    fn transpose(rows: [Self; GROUP]) -> [Self; GROUP];
    /// Add the residual `self` to the first `LANES` pixels with 8-bit
    /// clipping.
    fn add_to_pixels(self, pixels: &mut [u8]);
}

/// The largest input L1 norm for which every node of `kind`'s `n`-point
/// graph (and every equal-weight pre-add) fits `i16`, rounded down from the
/// limits `range_probe` proves. Identity scales by up to 4 and has no stage
/// clamps, so its limit is an amplitude bound.
pub(super) const fn l1_limit(kind: Kernel1d, n: usize) -> u16 {
    match (kind, n) {
        (Kernel1d::Dct, _) => DCT_L1_LIMIT,
        (Kernel1d::Adst | Kernel1d::FlipAdst, _) => ADST_L1_LIMIT,
        (Kernel1d::Identity, 8) => IDENTITY8_L1_LIMIT,
        (Kernel1d::Identity, 16) => IDENTITY16_L1_LIMIT,
        (Kernel1d::Identity, _) => IDENTITY32_L1_LIMIT,
    }
}

/// DCT 8–64 (the shared inverse limit, `inverse_lanes`).
pub(super) const DCT_L1_LIMIT: u16 = super::inverse_lanes::NARROW_INVERSE_L1_LIMIT;
/// ADST 8 and 16: the DCT's limit also holds (proved: 23277 and 23180).
pub(super) const ADST_L1_LIMIT: u16 = DCT_L1_LIMIT;
/// Identity 8: `2x` (`i16::MAX / 2`).
pub(super) const IDENTITY8_L1_LIMIT: u16 = 16_383;
/// Identity 16: `2 sqrt(2) x`, rounded.
pub(super) const IDENTITY16_L1_LIMIT: u16 = 11_583;
/// Identity 32: `4x` (`i16::MAX / 4`).
pub(super) const IDENTITY32_L1_LIMIT: u16 = 8_191;

/// The 8-, 16-, 32- or 64-point inverse DCT with inputs `live..`
/// structurally zero.
#[inline(always)]
fn idct<V: InverseLanes, const N: usize>(x: &[V; N], y: &mut [V; N], live: usize) {
    let (cospi, bit) = (&COSPI_12, INVERSE_COS_BIT);
    match N {
        8 => dct::idct8(
            x.as_slice().try_into().unwrap(),
            y.as_mut_slice().try_into().unwrap(),
            cospi,
            bit,
        ),
        16 => dct::idct16(
            x.as_slice().try_into().unwrap(),
            y.as_mut_slice().try_into().unwrap(),
            cospi,
            bit,
        ),
        32 => {
            let (x, y) = (
                x.as_slice().try_into().unwrap(),
                y.as_mut_slice().try_into().unwrap(),
            );
            if live <= 8 {
                dct::idct32_low8(x, y, cospi, bit);
            } else if live <= 16 {
                dct::idct32_low16(x, y, cospi, bit);
            } else {
                V::idct32(x, y, cospi, bit);
            }
        }
        64 => {
            let (x, y) = (
                x.as_slice().try_into().unwrap(),
                y.as_mut_slice().try_into().unwrap(),
            );
            if live <= 8 {
                dct::idct64_low8(x, y, cospi, bit);
            } else if live <= 16 {
                dct::idct64_low16(x, y, cospi, bit);
            } else {
                dct::idct64_low32(x, y, cospi, bit);
            }
        }
        _ => unreachable!("inverse DCT sizes are 8, 16, 32 and 64"),
    }
}

/// The `N`-point 1-D inverse `kind` of every lane (inputs `live..` zero),
/// with a flipped ADST's outputs reversed.
#[inline(always)]
fn transform<V: InverseLanes, const N: usize>(
    kind: Kernel1d,
    x: &[V; N],
    y: &mut [V; N],
    live: usize,
) {
    match (kind, N) {
        (Kernel1d::Dct, _) => idct(x, y, live),
        (Kernel1d::Adst | Kernel1d::FlipAdst, 8) => iadst8(
            x.as_slice().try_into().unwrap(),
            y.as_mut_slice().try_into().unwrap(),
            &COSPI_12,
            INVERSE_COS_BIT,
        ),
        (Kernel1d::Adst | Kernel1d::FlipAdst, 16) => iadst16(
            x.as_slice().try_into().unwrap(),
            y.as_mut_slice().try_into().unwrap(),
            &COSPI_12,
            INVERSE_COS_BIT,
        ),
        (Kernel1d::Identity, 8 | 16 | 32) => identity(x, y),
        _ => unreachable!("no {N}-point {kind:?} (checked by Av1TxType::fits)"),
    }
    if kind == Kernel1d::FlipAdst {
        y.reverse();
    }
}

/// A 2-D inverse: the column (`vertical`) and row (`horizontal`) kernels and
/// the nonzero extent of the coefficients.
#[derive(Clone, Copy)]
pub(super) struct Inverse {
    pub vertical: Kernel1d,
    pub horizontal: Kernel1d,
    /// Coefficient rows from this one on are zero (at most the coded height).
    pub rows: usize,
    /// Coefficient columns from this one on are zero (at most the coded
    /// width).
    pub columns: usize,
}

/// Inverse `W`x`H` transform of the column-major coefficients
/// (`coeffs[c * coded_h + r]`), added to `dst` with 8-bit clipping.
///
/// # Panics
///
/// If a buffer is too short or the extent exceeds the coded area.
#[inline(always)]
pub(super) fn inverse<V: NarrowLanes, const W: usize, const H: usize>(
    inverse: Inverse,
    coeffs: &[i32],
    dst: &mut [u8],
    stride: usize,
) {
    let coded_h = H.min(CODED_MAX);
    assert!(V::LANES <= W && V::LANES <= H && V::LANES <= MAX_LANES);
    assert!(inverse.rows <= coded_h && inverse.columns <= W.min(CODED_MAX));
    // Row transforms' outputs, row-major: `rows[r * W + c]`. Coefficient rows
    // from `inverse.rows` on are zero and transform to zero: whole batches
    // of them are skipped, and the column pass treats them as zero inputs.
    let batches = inverse.rows.div_ceil(V::LANES);
    let live_rows = batches * V::LANES;
    let mut rows = [MaybeUninit::<i16>::uninit(); ROWS_CAPACITY];
    assert!(live_rows * W <= ROWS_CAPACITY && live_rows <= coded_h);
    for r0 in (0..live_rows).step_by(V::LANES) {
        let output = row_batch::<V, W, H>(inverse, coeffs, r0);
        for k0 in (0..W).step_by(GROUP) {
            let mut tile = [V::splat(0); GROUP];
            tile.copy_from_slice(&output[k0..k0 + GROUP]);
            for (i, row) in V::transpose(tile).into_iter().enumerate() {
                // SAFETY: lane group g of `row` is row r0 + 8g + i, columns
                // k0..k0 + 8: inside the `live_rows * W` prefix checked
                // above. `MaybeUninit<i16>` has `i16`'s layout.
                unsafe {
                    row.store_groups(rows.as_mut_ptr().add((r0 + i) * W + k0).cast(), GROUP * W);
                }
            }
        }
    }
    // SAFETY: the loop above wrote rows 0..live_rows completely (every batch
    // stores all W columns of its LANES rows); only those are viewed.
    let rows = unsafe { core::slice::from_raw_parts(rows.as_ptr().cast::<i16>(), live_rows * W) };
    for c0 in (0..W).step_by(V::LANES) {
        column_batch::<V, W, H>(inverse.vertical, rows, live_rows, c0, dst, stride);
    }
}

/// The size with these sides.
const fn size<const W: usize, const H: usize>() -> Av1TxSize {
    Av1TxSize {
        width_log2: W.trailing_zeros() as u8,
        height_log2: H.trailing_zeros() as u8,
    }
}

/// Row transforms of rows `r0..r0 + LANES` (one per lane), shifted by the
/// size's row shift. Coefficients are clamped to 16 bits on load (libaom's
/// input clamp) and prescaled for 2:1 rectangles.
#[inline(always)]
fn row_batch<V: NarrowLanes, const W: usize, const H: usize>(
    inverse: Inverse,
    coeffs: &[i32],
    r0: usize,
) -> [V; W] {
    let coded_h = H.min(CODED_MAX);
    let two_to_one = const { size::<W, H>().is_two_to_one() };
    let mut x = [V::splat(0); W];
    let mut norm = V::norm_zero();
    for (c, lane) in x[..inverse.columns].iter_mut().enumerate() {
        let v = V::load_i32_saturating(&coeffs[c * coded_h + r0..][..V::LANES]);
        norm = v.accumulate_l1(norm);
        // The 16-bit clamp after the scaling (libaom `clamp_buf`) is the
        // identity: the scale shrinks.
        *lane = if two_to_one {
            V::mul_round(INV_SQRT2, v, INVERSE_COS_BIT)
        } else {
            v
        };
    }
    if !V::within_limit(norm, l1_limit(inverse.horizontal, W)) {
        return wide_row_batch::<V, W, H>(inverse, coeffs, r0);
    }
    let mut y = [V::splat(0); W];
    transform(inverse.horizontal, &x, &mut y, inverse.columns);
    // Inside the guard the outputs fit 16 bits, so this rounding shift equals
    // libaom's 32-bit one.
    for lane in &mut y {
        *lane = match const { size::<W, H>().row_shift() } {
            ROW_SHIFT_1 => lane.round_shift::<ROW_SHIFT_1>(),
            ROW_SHIFT_2 => lane.round_shift::<ROW_SHIFT_2>(),
            _ => unreachable!("sizes with both sides at least 8 shift rows by 1 or 2"),
        };
    }
    y
}

/// [`row_batch`] above the L1 limit: libaom's clamped 32-bit graph,
/// `V::Wide::LANES` lanes at a time, then the row shift and libaom's 16-bit
/// column input clamp, which the saturating narrow applies.
#[inline(never)]
fn wide_row_batch<V: NarrowLanes, const W: usize, const H: usize>(
    inverse: Inverse,
    coeffs: &[i32],
    r0: usize,
) -> [V; W] {
    let coded_h = H.min(CODED_MAX);
    let size = size::<W, H>();
    let mut wide = [[0_i32; MAX_LANES]; W];
    let mut x = [<V::Wide as I32Lanes>::splat(0); W];
    let mut out = x;
    for part in (0..V::LANES).step_by(V::Wide::LANES) {
        for (c, lane) in x[..inverse.columns].iter_mut().enumerate() {
            let v = I32Lanes::clamp16(V::Wide::load(&coeffs[c * coded_h + r0 + part..]));
            *lane = if size.is_two_to_one() {
                v.mul_scalar(INV_SQRT2).round_shift(INVERSE_COS_BIT)
            } else {
                v
            };
        }
        transform(inverse.horizontal, &x, &mut out, inverse.columns);
        for (k, value) in out.iter().enumerate() {
            value
                .round_shift(size.row_shift())
                .store(&mut wide[k][part..]);
        }
    }
    let mut y = [V::splat(0); W];
    for (lane, values) in y.iter_mut().zip(&wide) {
        *lane = V::load_i32_saturating(values);
    }
    y
}

/// Column transforms of columns `c0..c0 + LANES` (one per lane), rounded and
/// added to the prediction with 8-bit clipping. Rows from `live_rows` on are
/// zero.
#[inline(always)]
fn column_batch<V: NarrowLanes, const W: usize, const H: usize>(
    vertical: Kernel1d,
    rows: &[i16],
    live_rows: usize,
    c0: usize,
    dst: &mut [u8],
    stride: usize,
) {
    let mut x = [V::splat(0); H];
    let mut norm = V::norm_zero();
    for (r, lane) in x[..live_rows].iter_mut().enumerate() {
        let v = V::load_i16(&rows[r * W + c0..][..V::LANES]);
        norm = v.accumulate_l1(norm);
        *lane = v;
    }
    if !V::within_limit(norm, l1_limit(vertical, H)) {
        wide_column_batch::<V, W, H>(vertical, rows, live_rows, c0, dst, stride);
        return;
    }
    let mut y = [V::splat(0); H];
    transform(vertical, &x, &mut y, live_rows);
    for (r, value) in y.iter().enumerate() {
        value
            .round_shift::<COLUMN_SHIFT>()
            .add_to_pixels(&mut dst[r * stride + c0..][..V::LANES]);
    }
}

/// [`column_batch`] above the L1 limit: libaom's clamped 32-bit graph. The
/// row outputs are already within 16 bits (libaom's column input clamp).
#[inline(never)]
fn wide_column_batch<V: NarrowLanes, const W: usize, const H: usize>(
    vertical: Kernel1d,
    rows: &[i16],
    live_rows: usize,
    c0: usize,
    dst: &mut [u8],
    stride: usize,
) {
    let mut x = [<V::Wide as I32Lanes>::splat(0); H];
    let mut out = x;
    for part in (0..V::LANES).step_by(V::Wide::LANES) {
        for (r, lane) in x[..live_rows].iter_mut().enumerate() {
            *lane = V::Wide::load_i16(&rows[r * W + c0 + part..]);
        }
        transform(vertical, &x, &mut out, live_rows);
        for (r, value) in out.iter().enumerate() {
            value
                .round_shift(INVERSE_COLUMN_SHIFT)
                .add_to_u8(&mut dst[r * stride + c0 + part..]);
        }
    }
}

impl Inverse {
    /// The extent of a `W`x`H` transform whose coefficient rows from
    /// `nonzero_rows` on are zero (every coded column may be nonzero).
    pub(super) const fn of<const W: usize, const H: usize>(
        (vertical, horizontal): (Kernel1d, Kernel1d),
        nonzero_rows: usize,
    ) -> Self {
        let coded_h = if H < CODED_MAX { H } else { CODED_MAX };
        Self {
            vertical,
            horizontal,
            rows: if nonzero_rows < coded_h {
                nonzero_rows
            } else {
                coded_h
            },
            columns: if W < CODED_MAX { W } else { CODED_MAX },
        }
    }
}

/// The square `DCT_DCT` of side `N` whose nonzero coefficients lie in the
/// top-left `limit` square (8, 16 or 32; at most the coded side).
#[inline(always)]
pub(super) fn inverse_dct<V: NarrowLanes, const N: usize>(
    coeffs: &[i32],
    dst: &mut [u8],
    stride: usize,
    limit: usize,
) {
    let extent = Inverse {
        vertical: Kernel1d::Dct,
        horizontal: Kernel1d::Dct,
        rows: limit,
        columns: limit,
    };
    inverse::<V, N, N>(extent, coeffs, dst, stride);
}

/// The guard-boundary check every ISA's tests run.
#[cfg(test)]
pub(super) mod tests {
    use super::{CODED_MAX as COEFF_STRIDE, NARROW_INVERSE_L1_LIMIT};
    use crate::txfm::{DctSize, scalar};

    /// A sparse inverse entry point: `(coeffs, dst, stride, limit)`.
    pub(in crate::txfm) type SparseInverse = unsafe fn(&[i32], &mut [u8], usize, usize);

    /// Coefficients whose first row (one transform input vector) has L1
    /// norm exactly `l1`, spread over `limit` inputs with a sign pattern;
    /// the other rows are small so the column pass stays narrow.
    fn boundary_block(limit: usize, l1: i32, pattern: usize) -> Vec<i32> {
        let mut coeffs = vec![0_i32; COEFF_STRIDE * COEFF_STRIDE];
        let per_input = l1 / limit as i32;
        for c in 0..limit {
            let magnitude = per_input + i32::from(c == 0) * (l1 - per_input * limit as i32);
            let negative = match pattern {
                0 => false,
                1 => c % 2 == 1,
                2 => (c / 2) % 2 == 1,
                _ => (c * 7 + 3) % 5 < 2,
            };
            coeffs[c * COEFF_STRIDE] = if negative { -magnitude } else { magnitude };
            for r in 1..limit {
                coeffs[c * COEFF_STRIDE + r] = ((c * 31 + r * 17) % 9) as i32 - 4;
            }
        }
        coeffs
    }

    /// The 32- and 64-point sparse inverses equal the scalar reference just
    /// inside and just outside the guard.
    ///
    /// # Safety
    ///
    /// The entry points' ISA must be available.
    pub(in crate::txfm) unsafe fn assert_guard_boundary(
        inverse_32_sparse: SparseInverse,
        inverse_64_sparse: SparseInverse,
    ) {
        const PREDICTION: u8 = 128;
        let reference = scalar::txfm_fns();
        let limit_norm = i32::from(NARROW_INVERSE_L1_LIMIT);
        for n in [32, 64] {
            for limit in [8, 16, 32] {
                for l1 in [limit_norm, limit_norm + 1] {
                    for pattern in 0..4 {
                        let coeffs = boundary_block(limit, l1, pattern);
                        let mut expected = vec![PREDICTION; n * n];
                        let mut actual = expected.clone();
                        // SAFETY: the caller's ISA; buffers cover the n x n
                        // block and 32x32 coefficients.
                        unsafe {
                            if n == 32 {
                                (reference.inverse_dct_add[DctSize::N32 as usize])(
                                    &coeffs,
                                    &mut expected,
                                    n,
                                );
                                inverse_32_sparse(&coeffs, &mut actual, n, limit);
                            } else {
                                (reference.inverse_dct_add_64)(&coeffs, &mut expected, n);
                                inverse_64_sparse(&coeffs, &mut actual, n, limit);
                            }
                        }
                        assert_eq!(
                            actual, expected,
                            "N{n} low{limit} l1 {l1} pattern {pattern}"
                        );
                    }
                }
            }
        }
    }
}
