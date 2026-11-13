//! AVX2 VP9 inverse lanes: paired i16 rotations into eight i32 products.
//! Explicit low-word packing retains wrapping semantics for extreme inputs.
//! Rotations follow libvpx's SSE2/AVX2 inverse transforms (BSD-3-Clause;
//! `THIRD_PARTY.md`): interleave the two inputs, then one `pmaddwd`.

use super::{COS_BITS, Lanes, Products};
use crate::Vp9Pass;
use crate::txfm::inverse_lanes::InverseLanes;
use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi16, _mm_cvtepu8_epi16, _mm_cvtsi32_si128, _mm_loadl_epi64,
    _mm_loadu_si128, _mm_mulhrs_epi16, _mm_packus_epi16, _mm_packus_epi32, _mm_set1_epi16,
    _mm_storel_epi64, _mm_storeu_si128, _mm_sub_epi16, _mm_unpackhi_epi16, _mm_unpacklo_epi16,
    _mm256_add_epi32, _mm256_and_si256, _mm256_castsi128_si256, _mm256_castsi256_si128,
    _mm256_cvtepi16_epi32, _mm256_extracti128_si256, _mm256_inserti128_si256, _mm256_loadu_si256,
    _mm256_madd_epi16, _mm256_set1_epi32, _mm256_sra_epi32, _mm256_srai_epi32, _mm256_storeu_si256,
    _mm256_sub_epi32,
};

const COS_SHIFT: i32 = COS_BITS as i32;
const LOW_WORD: i32 = 0xffff;

#[derive(Clone, Copy)]
struct Narrow(__m128i);
#[derive(Clone, Copy)]
struct Wide(__m256i);

impl Products for Wide {
    type Narrow = Narrow;
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // SAFETY: these private lanes are constructed only by the AVX2 kernel.
        unsafe { Self(_mm256_add_epi32(self.0, other.0)) }
    }
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        // SAFETY: these private lanes are constructed only by the AVX2 kernel.
        unsafe { Self(_mm256_sub_epi32(self.0, other.0)) }
    }
    #[inline(always)]
    fn narrow(self) -> Narrow {
        // SAFETY: AVX2 is enabled. Masking before unsigned packing makes it
        // a truncation, not saturation, even for an arbitrary signed input.
        unsafe {
            let low = _mm256_and_si256(self.0, _mm256_set1_epi32(LOW_WORD));
            Narrow(_mm_packus_epi32(
                _mm256_castsi256_si128(low),
                _mm256_extracti128_si256::<1>(low),
            ))
        }
    }
    #[inline(always)]
    fn round14(self) -> Narrow {
        // SAFETY: AVX2 is enabled. Wrapping i32 products preserve the rounded
        // low16 result by the modular proof in the parent module.
        unsafe {
            Self(_mm256_srai_epi32::<COS_SHIFT>(_mm256_add_epi32(
                self.0,
                _mm256_set1_epi32(1 << (COS_BITS - 1)),
            )))
            .narrow()
        }
    }
}

impl Narrow {
    /// `round_shift(w * self, 14)` from `doubled = 2 * w`
    /// ([`super::doubled_weight`]).
    #[inline(always)]
    fn mul_doubled(self, doubled: i16) -> Self {
        // SAFETY: AVX2 is enabled; register-only arithmetic.
        unsafe { Self(_mm_mulhrs_epi16(self.0, _mm_set1_epi16(doubled))) }
    }
}

impl InverseLanes for Narrow {
    #[inline(always)]
    fn splat(value: i32) -> Self {
        // SAFETY: private lanes are used only by the AVX2 entry point.
        unsafe { Self(_mm_set1_epi16(value as i16)) }
    }
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // SAFETY: AVX2 is enabled. DCT additions are followed by WRAPLOW.
        unsafe { Self(_mm_add_epi16(self.0, other.0)) }
    }
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        // SAFETY: AVX2 is enabled. DCT subtractions are followed by WRAPLOW.
        unsafe { Self(_mm_sub_epi16(self.0, other.0)) }
    }
    #[inline(always)]
    fn half_btf(w0: i32, a: Self, w1: i32, b: Self, bit: u32) -> Self {
        debug_assert_eq!(bit, COS_BITS);
        Self::products(w0, a, w1, b).round14()
    }
    #[inline(always)]
    fn mul_round(w: i32, a: Self, bit: u32) -> Self {
        debug_assert_eq!(bit, COS_BITS);
        match super::doubled_weight(w) {
            Some(doubled) => a.mul_doubled(doubled),
            None => a.wide(w).round14(),
        }
    }
    #[inline(always)]
    fn clamp16(self) -> Self {
        self
    }
}

impl Lanes for Narrow {
    type Wide = Wide;
    #[inline(always)]
    fn wide(self, weight: i32) -> Wide {
        // One pmaddwd with a zero second weight: cheaper than widening and a
        // 32-bit multiply, and each product is exact in i32.
        Self::products(weight, self, 0, self)
    }
    #[inline(always)]
    fn products(w0: i32, a: Self, w1: i32, b: Self) -> Wide {
        // SAFETY: AVX2 is enabled. Signed cosine weights fit i16; pmaddwd
        // evaluates all eight paired rotations with full i32 products.
        unsafe {
            let lo = _mm_unpacklo_epi16(a.0, b.0);
            let hi = _mm_unpackhi_epi16(a.0, b.0);
            let pairs = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi);
            let packed = i32::from(w0 as u16) | (i32::from(w1 as u16) << 16);
            Wide(_mm256_madd_epi16(pairs, _mm256_set1_epi32(packed)))
        }
    }
    #[inline(always)]
    fn load_i32_wrapping(src: &[i32]) -> Self {
        let src = &src[..super::LANES];
        // SAFETY: the checked slice holds one unaligned eight-element load.
        unsafe { Wide(_mm256_loadu_si256(src.as_ptr().cast())).narrow() }
    }
    #[inline(always)]
    fn load_i16(src: &[i16]) -> Self {
        let src = &src[..super::LANES];
        // SAFETY: the checked slice holds all eight i16 samples.
        unsafe { Self(_mm_loadu_si128(src.as_ptr().cast())) }
    }
    #[inline(always)]
    fn store_i16(self, dst: &mut [i16]) {
        let dst = &mut dst[..super::LANES];
        // SAFETY: the checked destination holds the unaligned vector store.
        unsafe {
            _mm_storeu_si128(dst.as_mut_ptr().cast(), self.0);
        }
    }
    #[inline(always)]
    fn store_shifted(self, dst: &mut [i32], shift: u32) {
        let dst = &mut dst[..super::LANES];
        // SAFETY: AVX2 is enabled and the checked destination holds eight i32
        // outputs. Widen before rounding to avoid i16 rounding overflow.
        unsafe {
            let sum = _mm256_add_epi32(
                _mm256_cvtepi16_epi32(self.0),
                _mm256_set1_epi32(1 << (shift - 1)),
            );
            let out = _mm256_sra_epi32(sum, _mm_cvtsi32_si128(shift as i32));
            _mm256_storeu_si256(dst.as_mut_ptr().cast(), out);
        }
    }
    #[inline(always)]
    unsafe fn add_shifted(self, dst: *mut u8, shift: u32) {
        // SAFETY: AVX2 is enabled and the caller's destination holds the
        // eight samples. `mulhrs` by 2^(15 - shift) is exactly
        // `(x + 2^(shift - 1)) >> shift` (libvpx rounds the same way); the
        // residual plus a pixel fits i16 and `packus` clips to 0..=255.
        unsafe {
            let residual = _mm_mulhrs_epi16(self.0, _mm_set1_epi16(1 << (15 - shift)));
            let pixels = _mm_cvtepu8_epi16(_mm_loadl_epi64(dst.cast_const().cast()));
            let bytes = _mm_packus_epi16(_mm_add_epi16(residual, pixels), residual);
            _mm_storel_epi64(dst.cast(), bytes);
        }
    }
    #[inline(always)]
    fn zip_lo(self, other: Self) -> Self {
        // SAFETY: register-only permutation inside the AVX2 kernel.
        unsafe { Self(_mm_unpacklo_epi16(self.0, other.0)) }
    }
    #[inline(always)]
    fn zip_hi(self, other: Self) -> Self {
        // SAFETY: register-only permutation inside the AVX2 kernel.
        unsafe { Self(_mm_unpackhi_epi16(self.0, other.0)) }
    }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn inverse(
    log2: u8,
    columns: Vp9Pass,
    rows: Vp9Pass,
    input: &[i32],
    output: &mut [i32],
) {
    super::inverse::<Narrow>(log2, columns, rows, input, output);
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn inverse_add(
    log2: u8,
    columns: Vp9Pass,
    rows: Vp9Pass,
    input: &[i32],
    nonzero: (usize, usize),
    dst: &mut [u8],
    dst_stride: usize,
) {
    super::inverse_add::<Narrow>(log2, (columns, rows), input, nonzero, (dst, dst_stride));
}
