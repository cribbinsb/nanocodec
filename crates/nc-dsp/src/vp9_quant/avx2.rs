//! AVX2 VP9 quantization: eight coefficients per vector, the exact
//! reciprocal products by two `vpmuludq` (even and odd lanes).

use super::{RECIPROCAL_BITS, Vp9QuantStep, split};
use core::arch::x86_64::{
    __m256i, _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_max_epu32, _mm_shuffle_epi32,
    _mm256_abs_epi32, _mm256_add_epi32, _mm256_blend_epi32, _mm256_castsi256_si128,
    _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_max_epu32, _mm256_min_epu32,
    _mm256_mul_epu32, _mm256_or_si256, _mm256_set1_epi32, _mm256_setzero_si256, _mm256_sign_epi32,
    _mm256_sll_epi32, _mm256_srli_epi64, _mm256_storeu_si256, _mm256_testz_si256,
};

/// Blend mask taking the odd 32-bit lanes from the second operand.
const ODD_LANES: i32 = 0b1010_1010;

#[target_feature(enable = "avx2")]
pub(super) unsafe fn quantize(
    coefficients: &[i32],
    dc: &Vp9QuantStep,
    ac: &Vp9QuantStep,
    levels: &mut [i32],
) -> bool {
    split::<8>(coefficients, dc, ac, levels, |input, output| {
        // SAFETY: AVX2 is enabled; every load and store is one whole
        // eight-lane chunk of the checked slices.
        unsafe {
            let reciprocal = _mm256_set1_epi32(ac.reciprocal as i32);
            let offset = _mm256_set1_epi32(ac.offset as i32);
            let cap = _mm256_set1_epi32(ac.cap as i32);
            let shift = _mm_cvtsi32_si128(ac.shift as i32);
            let mut any: __m256i = _mm256_setzero_si256();
            for (c, level) in input.iter().zip(output) {
                let c = _mm256_loadu_si256(c.as_ptr().cast());
                let numerator =
                    _mm256_add_epi32(_mm256_sll_epi32(_mm256_abs_epi32(c), shift), offset);
                let even = _mm256_mul_epu32(numerator, reciprocal);
                let odd = _mm256_mul_epu32(
                    _mm256_srli_epi64::<{ RECIPROCAL_BITS as i32 }>(numerator),
                    reciprocal,
                );
                // Even lanes' quotients are their products' high halves; the
                // odd lanes' already sit in the high halves.
                let quotient = _mm256_blend_epi32::<ODD_LANES>(
                    _mm256_srli_epi64::<{ RECIPROCAL_BITS as i32 }>(even),
                    odd,
                );
                let magnitude = _mm256_min_epu32(quotient, cap);
                any = _mm256_or_si256(any, magnitude);
                // A zero coefficient has a zero level (offset < step).
                _mm256_storeu_si256(level.as_mut_ptr().cast(), _mm256_sign_epi32(magnitude, c));
            }
            _mm256_testz_si256(any, any) == 0
        }
    })
}

/// AVX2 [`super::MagnitudeFn`]: eight absolute values per vector, an
/// unsigned running maximum, then a horizontal reduction.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn largest_magnitude(coefficients: &[i32]) -> u32 {
    let (whole, tail) = coefficients.as_chunks::<8>();
    // SAFETY: AVX2 is enabled; every load is one whole eight-lane chunk.
    let vector = unsafe {
        let mut largest = _mm256_setzero_si256();
        for chunk in whole {
            let c = _mm256_loadu_si256(chunk.as_ptr().cast());
            largest = _mm256_max_epu32(largest, _mm256_abs_epi32(c));
        }
        let half = _mm_max_epu32(
            _mm256_castsi256_si128(largest),
            _mm256_extracti128_si256::<1>(largest),
        );
        let quarter = _mm_max_epu32(half, _mm_shuffle_epi32::<0b0100_1110>(half));
        let eighth = _mm_max_epu32(quarter, _mm_shuffle_epi32::<0b1011_0001>(quarter));
        _mm_cvtsi128_si32(eighth) as u32
    };
    // SAFETY: the scalar reference has no preconditions.
    vector.max(unsafe { super::largest_magnitude(tail) })
}
