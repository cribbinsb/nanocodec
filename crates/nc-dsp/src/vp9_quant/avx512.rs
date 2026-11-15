//! AVX-512 VP9 quantization: the AVX2 kernel (`avx2.rs`) on sixteen
//! coefficients per vector, the exact reciprocal products by two
//! `vpmuludq` (even and odd lanes), and the sign restored by a masked
//! negation (AVX-512 has no `vpsignd`). The largest-magnitude scan keeps
//! AVX2: most blocks are 4x4 or 8x8, below one 32-word vector.

use super::{RECIPROCAL_BITS, Vp9QuantStep, split};
use core::arch::x86_64::{
    __m512i, _mm_cvtsi32_si128, _mm256_loadu_si256, _mm512_abs_epi32, _mm512_add_epi32,
    _mm512_cmplt_epi32_mask, _mm512_cvtepi16_epi32, _mm512_mask_blend_epi32, _mm512_mask_sub_epi32,
    _mm512_min_epu32, _mm512_mul_epu32, _mm512_or_si512, _mm512_set1_epi32, _mm512_setzero_si512,
    _mm512_sll_epi32, _mm512_srli_epi64, _mm512_storeu_si512, _mm512_test_epi32_mask,
};

/// Blend mask taking the odd 32-bit lanes from the second operand.
const ODD_LANES: u16 = 0xAAAA;

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub(super) unsafe fn quantize(
    coefficients: &[i16],
    dc: &Vp9QuantStep,
    ac: &Vp9QuantStep,
    levels: &mut [i32],
) -> bool {
    split::<16>(coefficients, dc, ac, levels, |input, output| {
        let reciprocal = _mm512_set1_epi32(ac.reciprocal as i32);
        let offset = _mm512_set1_epi32(ac.offset as i32);
        let cap = _mm512_set1_epi32(ac.cap as i32);
        let shift = _mm_cvtsi32_si128(ac.shift as i32);
        let zero = _mm512_setzero_si512();
        let mut any: __m512i = zero;
        for (c, level) in input.iter().zip(output) {
            // SAFETY: AVX-512F/BW/VL are enabled; every load and store is one
            // whole sixteen-lane chunk of the checked slices.
            let c = _mm512_cvtepi16_epi32(unsafe { _mm256_loadu_si256(c.as_ptr().cast()) });
            let numerator = _mm512_add_epi32(_mm512_sll_epi32(_mm512_abs_epi32(c), shift), offset);
            let even = _mm512_mul_epu32(numerator, reciprocal);
            let odd = _mm512_mul_epu32(
                _mm512_srli_epi64::<{ RECIPROCAL_BITS }>(numerator),
                reciprocal,
            );
            // Even lanes' quotients are their products' high halves; the
            // odd lanes' already sit in the high halves.
            let quotient = _mm512_mask_blend_epi32(
                ODD_LANES,
                _mm512_srli_epi64::<{ RECIPROCAL_BITS }>(even),
                odd,
            );
            let magnitude = _mm512_min_epu32(quotient, cap);
            any = _mm512_or_si512(any, magnitude);
            let negative = _mm512_cmplt_epi32_mask(c, zero);
            let signed = _mm512_mask_sub_epi32(magnitude, negative, zero, magnitude);
            // SAFETY: as above.
            unsafe { _mm512_storeu_si512(level.as_mut_ptr().cast(), signed) };
        }
        _mm512_test_epi32_mask(any, any) != 0
    })
}
