//! AVX2 lanes of the guarded 16-bit AV1 inverse DCTs (`narrow_inverse`):
//! sixteen 16-bit lanes per 256-bit register for 16x16 and larger, eight per
//! 128-bit register (VEX-encoded) for 8x8; wrapping adds,
//! `pmaddwd` on interleaved pairs for two-weight rotations and the rounding
//! Q15 multiply (`pmulhrsw`) for single and equal weights, exact inside the
//! L1 guard. The 32-bit `I32x8` graph of the AVX2 table is the fallback above
//! it.
//!
//! The lane layout and instruction choices follow dav1d's x86 inverse
//! transforms (`src/x86/itx_avx2.asm`, BSD-2-Clause; see `THIRD_PARTY.md`);
//! the code is the shared generic graph, not a transliteration.
//!
//! Kernel contract: see the `txfm` module docs. Every entry point enables
//! AVX2, and the driver and graphs inline into it (`inline(always)`).

#![allow(clippy::inline_always)]

use core::arch::x86_64::{
    __m128i, __m256i, _mm_abs_epi16, _mm_add_epi16, _mm_add_epi32, _mm_adds_epu16,
    _mm_cvtepu8_epi16, _mm_loadl_epi64, _mm_loadu_si128, _mm_madd_epi16, _mm_mulhrs_epi16,
    _mm_packs_epi32, _mm_packus_epi16, _mm_set1_epi16, _mm_set1_epi32, _mm_setzero_si128,
    _mm_srai_epi32, _mm_storel_epi64, _mm_storeu_si128, _mm_sub_epi16, _mm_subs_epu16,
    _mm_testz_si128, _mm_unpackhi_epi16, _mm_unpackhi_epi32, _mm_unpackhi_epi64,
    _mm_unpacklo_epi16, _mm_unpacklo_epi32, _mm_unpacklo_epi64, _mm256_abs_epi16, _mm256_add_epi16,
    _mm256_add_epi32, _mm256_adds_epu16, _mm256_castsi256_si128, _mm256_cvtepu8_epi16,
    _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_mulhrs_epi16,
    _mm256_packs_epi32, _mm256_packus_epi16, _mm256_permute4x64_epi64, _mm256_set1_epi16,
    _mm256_set1_epi32, _mm256_setzero_si256, _mm256_srai_epi32, _mm256_sub_epi16,
    _mm256_subs_epu16, _mm256_testz_si256, _mm256_unpackhi_epi16, _mm256_unpackhi_epi32,
    _mm256_unpackhi_epi64, _mm256_unpacklo_epi16, _mm256_unpacklo_epi32, _mm256_unpacklo_epi64,
};

use super::generic::INVERSE_COS_BIT;
use super::inverse_lanes::InverseLanes;
use super::narrow_inverse::{self, GROUP, NarrowLanes};
use crate::av1_inverse::Kernel1d;
use crate::simd::avx2::I32x8;

/// `pmulhrsw` computes `(a * b + 2^14) >> 15`; a weight scaled by
/// `2^(15 - COS_BIT)` turns that into libaom's `round_shift(a * w, COS_BIT)`.
const Q15_WEIGHT_SHIFT: u32 = 15 - INVERSE_COS_BIT;
/// libaom's rounding offset of a `COS_BIT` shift.
const COS_ROUND: i32 = 1 << (INVERSE_COS_BIT - 1);

/// Eight 16-bit lanes of one inverse pass, valid only inside the L1 limit.
#[derive(Clone, Copy)]
pub(super) struct Narrow(__m128i);

/// `round_shift(x, COS_BIT)` of two `i32x4` halves, narrowed (the guard
/// keeps the results in 16 bits, so the saturating pack is exact).
#[inline(always)]
fn round_narrow(low: __m128i, high: __m128i) -> Narrow {
    // SAFETY: AVX2 entry points only (module docs); register arithmetic.
    unsafe {
        let round = _mm_set1_epi32(COS_ROUND);
        Narrow(_mm_packs_epi32(
            _mm_srai_epi32::<{ INVERSE_COS_BIT as i32 }>(_mm_add_epi32(low, round)),
            _mm_srai_epi32::<{ INVERSE_COS_BIT as i32 }>(_mm_add_epi32(high, round)),
        ))
    }
}

impl InverseLanes for Narrow {
    #[inline(always)]
    fn splat(value: i32) -> Self {
        // SAFETY: AVX2 entry points only; the graphs splat only zero.
        unsafe { Self(_mm_set1_epi16(value as i16)) }
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // SAFETY: AVX2 entry points only; the guard keeps the sum in 16 bits.
        unsafe { Self(_mm_add_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        // SAFETY: AVX2 entry points only; the guard keeps the difference in
        // 16 bits.
        unsafe { Self(_mm_sub_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn half_btf(w0: i32, a: Self, w1: i32, b: Self, bit: u32) -> Self {
        debug_assert_eq!(bit, INVERSE_COS_BIT);
        // Equal-magnitude weights (the cospi[32] rotations) add or subtract
        // first: the guard keeps that pre-add in 16 bits too.
        if w0.abs() < 1 << INVERSE_COS_BIT {
            if w0 == w1 {
                return Self::mul_round(w0, a.add(b), bit);
            }
            if w0 == -w1 {
                return Self::mul_round(w0, a.sub(b), bit);
            }
        }
        // SAFETY: AVX2 entry points only. Lanes (a_i, b_i) dotted with
        // (w0, w1): exact 32-bit sums of Q12 products (weights fit 16 bits).
        unsafe {
            let weights = _mm_set1_epi32((w1 << 16) | (w0 & 0xffff));
            round_narrow(
                _mm_madd_epi16(_mm_unpacklo_epi16(a.0, b.0), weights),
                _mm_madd_epi16(_mm_unpackhi_epi16(a.0, b.0), weights),
            )
        }
    }

    #[inline(always)]
    fn mul_round(w: i32, a: Self, bit: u32) -> Self {
        debug_assert_eq!(bit, INVERSE_COS_BIT);
        // SAFETY: AVX2 entry points only; register arithmetic.
        unsafe {
            if w.abs() < 1 << INVERSE_COS_BIT {
                // The scaled weight is not -2^15, so the Q15 multiply rounds
                // exactly as libaom's round_shift.
                Self(_mm_mulhrs_epi16(
                    a.0,
                    _mm_set1_epi16((w << Q15_WEIGHT_SHIFT) as i16),
                ))
            } else {
                // `cospi[0]` (4096) does not fit the Q15 form: (a, 0) pairs
                // dotted with (w, 0).
                let zero = _mm_setzero_si128();
                let weights = _mm_set1_epi32(w & 0xffff);
                round_narrow(
                    _mm_madd_epi16(_mm_unpacklo_epi16(a.0, zero), weights),
                    _mm_madd_epi16(_mm_unpackhi_epi16(a.0, zero), weights),
                )
            }
        }
    }

    /// Inside the guard every node already fits 16 bits.
    #[inline(always)]
    fn clamp16(self) -> Self {
        self
    }
}

impl NarrowLanes for Narrow {
    const LANES: usize = GROUP;
    type Norm = __m128i;
    type Wide = I32x8;

    #[inline(always)]
    fn norm_zero() -> __m128i {
        // SAFETY: AVX2 entry points only.
        unsafe { _mm_setzero_si128() }
    }

    #[inline(always)]
    fn accumulate_l1(self, norm: __m128i) -> __m128i {
        // SAFETY: AVX2 entry points only. `|i16::MIN|` wraps to 0x8000,
        // which reads correctly as unsigned.
        unsafe { _mm_adds_epu16(norm, _mm_abs_epi16(self.0)) }
    }

    #[inline(always)]
    fn within_limit(norm: __m128i, limit: u16) -> bool {
        // SAFETY: AVX2 entry points only. The saturating difference is zero
        // exactly where the norm is at most the limit.
        unsafe {
            let excess = _mm_subs_epu16(norm, _mm_set1_epi16(limit as i16));
            _mm_testz_si128(excess, excess) != 0
        }
    }

    #[inline(always)]
    fn load_i32_saturating(src: &[i32]) -> Self {
        let src = &src[..GROUP];
        // SAFETY: AVX2 entry points only; two 16-byte loads of an
        // eight-element slice. The signed saturating pack is libaom's 16-bit
        // input clamp.
        unsafe {
            Self(_mm_packs_epi32(
                _mm_loadu_si128(src.as_ptr().cast()),
                _mm_loadu_si128(src.as_ptr().add(4).cast()),
            ))
        }
    }

    #[inline(always)]
    fn load_i16(src: &[i16]) -> Self {
        let src = &src[..GROUP];
        // SAFETY: AVX2 entry points only; a 16-byte load of an eight-element
        // array.
        unsafe { Self(_mm_loadu_si128(src.as_ptr().cast())) }
    }

    #[inline(always)]
    unsafe fn store_groups(self, dst: *mut i16, _group_stride: usize) {
        // SAFETY: AVX2 entry points only; a 16-byte store of the
        // caller's eight writable values.
        unsafe { _mm_storeu_si128(dst.cast(), self.0) }
    }

    #[inline(always)]
    fn round_shift<const SHIFT: u32>(self) -> Self {
        // SAFETY: AVX2 entry points only. `pmulhrsw` by `2^(15 - SHIFT)` is
        // `(x * 2^(15 - SHIFT) + 2^14) >> 15 = (x + 2^(SHIFT - 1)) >> SHIFT`,
        // computed in 32 bits, so it cannot overflow.
        unsafe { Self(_mm_mulhrs_epi16(self.0, _mm_set1_epi16(1 << (15 - SHIFT)))) }
    }

    #[inline(always)]
    fn transpose(rows: [Self; GROUP]) -> [Self; GROUP] {
        transpose8(rows.map(|row| row.0)).map(Self)
    }

    #[inline(always)]
    fn add_to_pixels(self, pixels: &mut [u8]) {
        let pixels = &mut pixels[..GROUP];
        // SAFETY: AVX2 entry points only; an eight-byte load and store of an
        // eight-byte array. The residual and pixel sum is within i16; the
        // unsigned saturating pack is the 8-bit clip.
        unsafe {
            let prediction = _mm_cvtepu8_epi16(_mm_loadl_epi64(pixels.as_ptr().cast()));
            let sum = _mm_add_epi16(self.0, prediction);
            _mm_storel_epi64(pixels.as_mut_ptr().cast(), _mm_packus_epi16(sum, sum));
        }
    }
}

/// Transpose eight rows of eight 16-bit lanes: unpacks by 16-bit lane, then
/// by 32-bit pairs, then by 64-bit halves.
#[inline(always)]
fn transpose8(rows: [__m128i; GROUP]) -> [__m128i; GROUP] {
    // SAFETY: AVX2 entry points only; register shuffles.
    unsafe {
        let pairs: [__m128i; GROUP] = core::array::from_fn(|k| {
            let (x, y) = (rows[k & !1], rows[k | 1]);
            if k % 2 == 0 {
                _mm_unpacklo_epi16(x, y)
            } else {
                _mm_unpackhi_epi16(x, y)
            }
        });
        // `quads[4 * h + q]`: columns 2q and 2q + 1 of rows 4h..4h + 4.
        let quads: [__m128i; GROUP] = core::array::from_fn(|k| {
            let (block, pair) = (k / 4 * 4, k % 4);
            let (x, y) = (pairs[block + pair / 2], pairs[block + 2 + pair / 2]);
            if pair % 2 == 0 {
                _mm_unpacklo_epi32(x, y)
            } else {
                _mm_unpackhi_epi32(x, y)
            }
        });
        core::array::from_fn(|column| {
            let (x, y) = (quads[column / 2], quads[4 + column / 2]);
            if column % 2 == 0 {
                _mm_unpacklo_epi64(x, y)
            } else {
                _mm_unpackhi_epi64(x, y)
            }
        })
    }
}

/// Sixteen 16-bit lanes of one inverse pass (two lane groups of eight, one
/// per 128-bit half), valid only inside the L1 limit: the 16x16 and larger
/// drivers run sixteen transforms per batch, as dav1d's AVX2 inverses do.
#[derive(Clone, Copy)]
pub(super) struct Narrow16(__m256i);

/// [`round_narrow`] on sixteen lanes: the in-lane pack restores the order
/// that the in-lane unpacks of the products took apart.
#[inline(always)]
fn round_narrow16(low: __m256i, high: __m256i) -> Narrow16 {
    // SAFETY: AVX2 entry points only (module docs); register arithmetic.
    unsafe {
        let round = _mm256_set1_epi32(COS_ROUND);
        Narrow16(_mm256_packs_epi32(
            _mm256_srai_epi32::<{ INVERSE_COS_BIT as i32 }>(_mm256_add_epi32(low, round)),
            _mm256_srai_epi32::<{ INVERSE_COS_BIT as i32 }>(_mm256_add_epi32(high, round)),
        ))
    }
}

impl InverseLanes for Narrow16 {
    /// Complete the independent odd/even halves separately to shorten live
    /// ranges. Existing row/column guards and wide fallbacks are unchanged.
    #[inline(always)]
    fn idct32(input: &[Self; 32], output: &mut [Self; 32], cospi: &[i32; 64], bit: u32) {
        super::dct::idct32_half_scheduled(input, output, cospi, bit);
    }

    #[inline(always)]
    fn splat(value: i32) -> Self {
        // SAFETY: AVX2 entry points only; the graphs splat only zero.
        unsafe { Self(_mm256_set1_epi16(value as i16)) }
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // SAFETY: AVX2 entry points only; the guard keeps the sum in 16 bits.
        unsafe { Self(_mm256_add_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        // SAFETY: AVX2 entry points only; the guard keeps the difference in
        // 16 bits.
        unsafe { Self(_mm256_sub_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn half_btf(w0: i32, a: Self, w1: i32, b: Self, bit: u32) -> Self {
        debug_assert_eq!(bit, INVERSE_COS_BIT);
        // As `Narrow::half_btf`.
        if w0.abs() < 1 << INVERSE_COS_BIT {
            if w0 == w1 {
                return Self::mul_round(w0, a.add(b), bit);
            }
            if w0 == -w1 {
                return Self::mul_round(w0, a.sub(b), bit);
            }
        }
        // SAFETY: AVX2 entry points only; exact 32-bit sums of Q12 products.
        unsafe {
            let weights = _mm256_set1_epi32((w1 << 16) | (w0 & 0xffff));
            round_narrow16(
                _mm256_madd_epi16(_mm256_unpacklo_epi16(a.0, b.0), weights),
                _mm256_madd_epi16(_mm256_unpackhi_epi16(a.0, b.0), weights),
            )
        }
    }

    #[inline(always)]
    fn mul_round(w: i32, a: Self, bit: u32) -> Self {
        debug_assert_eq!(bit, INVERSE_COS_BIT);
        // SAFETY: AVX2 entry points only; register arithmetic.
        unsafe {
            if w.abs() < 1 << INVERSE_COS_BIT {
                // As `Narrow::mul_round`.
                Self(_mm256_mulhrs_epi16(
                    a.0,
                    _mm256_set1_epi16((w << Q15_WEIGHT_SHIFT) as i16),
                ))
            } else {
                let zero = _mm256_setzero_si256();
                let weights = _mm256_set1_epi32(w & 0xffff);
                round_narrow16(
                    _mm256_madd_epi16(_mm256_unpacklo_epi16(a.0, zero), weights),
                    _mm256_madd_epi16(_mm256_unpackhi_epi16(a.0, zero), weights),
                )
            }
        }
    }

    /// Inside the guard every node already fits 16 bits.
    #[inline(always)]
    fn clamp16(self) -> Self {
        self
    }
}

impl NarrowLanes for Narrow16 {
    const LANES: usize = 2 * GROUP;
    type Norm = __m256i;
    type Wide = I32x8;

    #[inline(always)]
    fn norm_zero() -> __m256i {
        // SAFETY: AVX2 entry points only.
        unsafe { _mm256_setzero_si256() }
    }

    #[inline(always)]
    fn accumulate_l1(self, norm: __m256i) -> __m256i {
        // SAFETY: AVX2 entry points only; as `Narrow::accumulate_l1`.
        unsafe { _mm256_adds_epu16(norm, _mm256_abs_epi16(self.0)) }
    }

    #[inline(always)]
    fn within_limit(norm: __m256i, limit: u16) -> bool {
        // SAFETY: AVX2 entry points only; as `Narrow::within_limit`.
        unsafe {
            let excess = _mm256_subs_epu16(norm, _mm256_set1_epi16(limit as i16));
            _mm256_testz_si256(excess, excess) != 0
        }
    }

    #[inline(always)]
    fn load_i32_saturating(src: &[i32]) -> Self {
        let src = &src[..2 * GROUP];
        // SAFETY: AVX2 entry points only; two 32-byte loads of a sixteen-
        // element slice. The in-lane saturating pack interleaves the halves
        // by 64 bits; the permute restores their order.
        unsafe {
            let packed = _mm256_packs_epi32(
                _mm256_loadu_si256(src.as_ptr().cast()),
                _mm256_loadu_si256(src.as_ptr().add(GROUP).cast()),
            );
            Self(_mm256_permute4x64_epi64::<0b11_01_10_00>(packed))
        }
    }

    #[inline(always)]
    fn load_i16(src: &[i16]) -> Self {
        let src = &src[..2 * GROUP];
        // SAFETY: AVX2 entry points only; a 32-byte load of a sixteen-element
        // slice.
        unsafe { Self(_mm256_loadu_si256(src.as_ptr().cast())) }
    }

    #[inline(always)]
    unsafe fn store_groups(self, dst: *mut i16, group_stride: usize) {
        // SAFETY: AVX2 entry points only; the caller's two eight-value spans.
        unsafe {
            _mm_storeu_si128(dst.cast(), _mm256_castsi256_si128(self.0));
            _mm_storeu_si128(
                dst.add(group_stride).cast(),
                _mm256_extracti128_si256::<1>(self.0),
            );
        }
    }

    #[inline(always)]
    fn round_shift<const SHIFT: u32>(self) -> Self {
        // SAFETY: AVX2 entry points only; as `Narrow::round_shift`.
        unsafe {
            Self(_mm256_mulhrs_epi16(
                self.0,
                _mm256_set1_epi16(1 << (15 - SHIFT)),
            ))
        }
    }

    #[inline(always)]
    fn transpose(rows: [Self; GROUP]) -> [Self; GROUP] {
        transpose8x2(rows.map(|row| row.0)).map(Self)
    }

    #[inline(always)]
    fn add_to_pixels(self, pixels: &mut [u8]) {
        let pixels = &mut pixels[..2 * GROUP];
        // SAFETY: AVX2 entry points only; a sixteen-byte load and store of a
        // sixteen-byte slice. The residual and pixel sum is within i16; the
        // unsigned saturating pack is the 8-bit clip, and the permute
        // gathers its two in-lane halves into the low 128 bits.
        unsafe {
            let prediction = _mm256_cvtepu8_epi16(_mm_loadu_si128(pixels.as_ptr().cast()));
            let sum = _mm256_add_epi16(self.0, prediction);
            let bytes = _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packus_epi16(sum, sum));
            _mm_storeu_si128(pixels.as_mut_ptr().cast(), _mm256_castsi256_si128(bytes));
        }
    }
}

/// [`transpose8`] in each 128-bit half of eight 256-bit rows.
#[inline(always)]
fn transpose8x2(rows: [__m256i; GROUP]) -> [__m256i; GROUP] {
    // SAFETY: AVX2 entry points only; register shuffles.
    unsafe {
        let pairs: [__m256i; GROUP] = core::array::from_fn(|k| {
            let (x, y) = (rows[k & !1], rows[k | 1]);
            if k % 2 == 0 {
                _mm256_unpacklo_epi16(x, y)
            } else {
                _mm256_unpackhi_epi16(x, y)
            }
        });
        let quads: [__m256i; GROUP] = core::array::from_fn(|k| {
            let (block, pair) = (k / 4 * 4, k % 4);
            let (x, y) = (pairs[block + pair / 2], pairs[block + 2 + pair / 2]);
            if pair % 2 == 0 {
                _mm256_unpacklo_epi32(x, y)
            } else {
                _mm256_unpackhi_epi32(x, y)
            }
        });
        core::array::from_fn(|column| {
            let (x, y) = (quads[column / 2], quads[4 + column / 2]);
            if column % 2 == 0 {
                _mm256_unpacklo_epi64(x, y)
            } else {
                _mm256_unpackhi_epi64(x, y)
            }
        })
    }
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn inverse_8(coeffs: &[i32], dst: &mut [u8], stride: usize) {
    narrow_inverse::inverse_dct::<Narrow, 8>(coeffs, dst, stride, 8);
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn inverse_16(coeffs: &[i32], dst: &mut [u8], stride: usize) {
    narrow_inverse::inverse_dct::<Narrow16, 16>(coeffs, dst, stride, 16);
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn inverse_32(coeffs: &[i32], dst: &mut [u8], stride: usize) {
    narrow_inverse::inverse_dct::<Narrow16, 32>(coeffs, dst, stride, 32);
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn inverse_64(coeffs: &[i32], dst: &mut [u8], stride: usize) {
    narrow_inverse::inverse_dct::<Narrow16, 64>(coeffs, dst, stride, 32);
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn inverse_32_sparse(
    coeffs: &[i32],
    dst: &mut [u8],
    stride: usize,
    limit: usize,
) {
    narrow_inverse::inverse_dct::<Narrow16, 32>(coeffs, dst, stride, limit);
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn inverse_64_sparse(
    coeffs: &[i32],
    dst: &mut [u8],
    stride: usize,
    limit: usize,
) {
    narrow_inverse::inverse_dct::<Narrow16, 64>(coeffs, dst, stride, limit);
}

/// The AV1 inverse of a `W`x`H` transform with both sides at least 8 and
/// the 1-D kernels `(vertical, horizontal)`, column-major `coeffs` whose rows
/// from `nonzero_rows` on are zero, added to `dst` (the `av1_inverse` table
/// entry for those sizes).
///
/// # Safety
///
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn av1_inverse_add<const W: usize, const H: usize>(
    kernels: (Kernel1d, Kernel1d),
    coeffs: &[i32],
    nonzero_rows: usize,
    dst: &mut [u8],
    stride: usize,
) {
    let extent = narrow_inverse::Inverse::of::<W, H>(kernels, nonzero_rows);
    // Sixteen lanes need sixteen rows and columns.
    if W >= 2 * GROUP && H >= 2 * GROUP {
        narrow_inverse::inverse::<Narrow16, W, H>(extent, coeffs, dst, stride);
    } else {
        narrow_inverse::inverse::<Narrow, W, H>(extent, coeffs, dst, stride);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn guard_boundary_matches_scalar_on_both_sides() {
        if !crate::Isa::Avx2.is_supported() {
            return;
        }
        // SAFETY: AVX2 is supported (checked above).
        unsafe {
            super::super::narrow_inverse::tests::assert_guard_boundary(
                super::inverse_32_sparse,
                super::inverse_64_sparse,
            );
        }
    }
}
