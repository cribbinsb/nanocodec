//! Sixteen-coefficient AVX-512 quantisation through the existing exact
//! reciprocal proof and scalar DC rule. Masked final groups avoid reading
//! padding. Widened products preserve the same signed coefficient gains.
use super::{Av1Quantizer, Reciprocal, TxfmFns, quant_terms, quantize_one};
use core::arch::x86_64::{
    _mm_cvtsi32_si128, _mm512_abs_epi32, _mm512_add_epi32, _mm512_add_epi64,
    _mm512_cmplt_epi32_mask, _mm512_cmpneq_epi32_mask, _mm512_mask_add_epi32,
    _mm512_mask_blend_epi32, _mm512_mask_storeu_epi32, _mm512_mask_sub_epi32,
    _mm512_maskz_loadu_epi32, _mm512_min_epi32, _mm512_mul_epi32, _mm512_mul_epu32,
    _mm512_reduce_add_epi32, _mm512_reduce_add_epi64, _mm512_set1_epi32, _mm512_set1_epi64,
    _mm512_setzero_si512, _mm512_sll_epi32, _mm512_slli_epi32, _mm512_slli_epi64, _mm512_srl_epi64,
    _mm512_srli_epi64, _mm512_sub_epi32,
};

pub(super) fn install(table: &mut TxfmFns) {
    table.quantize = quantize;
    table.quantize_from = quantize_from;
    table.dequantize = dequantize;
    table.coefficient_gain = gain;
}

/// Source and destination hold len values and are equal or disjoint.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn levels(src: *const i32, dst: *mut i32, len: usize, q: &Av1Quantizer) -> u32 {
    if len == 0 {
        return 0;
    }
    // SAFETY: len is nonzero and both buffers cover it.
    let dc = unsafe { quantize_one(*src, 0, q) };
    // SAFETY: as above; the source value was loaded before this write.
    unsafe {
        *dst = dc;
    }
    let (offset, divisor) = quant_terms(q, q.ac_step);
    let Some(reciprocal) = Reciprocal::new(divisor) else {
        let mut count = u32::from(dc != 0);
        for i in 1..len {
            // SAFETY: i lies inside both buffers, loaded before stored.
            unsafe {
                let value = quantize_one(*src.add(i), i, q);
                *dst.add(i) = value;
                count += u32::from(value != 0);
            }
        }
        return count;
    };
    let zero = _mm512_setzero_si512();
    let mut counts = zero;
    let input_shift = _mm_cvtsi32_si128(7 + q.log_scale as i32);
    let result_shift = _mm_cvtsi32_si128(reciprocal.shift as i32);
    let multiplier = _mm512_set1_epi64(i64::from(reciprocal.multiplier));
    for i in (1..len).step_by(16) {
        let mask = (u32::MAX >> (32 - (len - i).min(16))) as u16;
        // SAFETY: only valid lanes are read; input and output may alias.
        let c = unsafe { _mm512_maskz_loadu_epi32(mask, src.add(i).cast()) };
        let numerator = _mm512_add_epi32(
            _mm512_sll_epi32(_mm512_abs_epi32(c), input_shift),
            _mm512_set1_epi32(offset as i32),
        );
        let even = _mm512_srl_epi64(_mm512_mul_epu32(numerator, multiplier), result_shift);
        let odd = _mm512_srl_epi64(
            _mm512_mul_epu32(_mm512_srli_epi64::<32>(numerator), multiplier),
            result_shift,
        );
        let quotient = _mm512_mask_blend_epi32(0xaaaa, even, _mm512_slli_epi64::<32>(odd));
        let magnitude = _mm512_min_epi32(quotient, _mm512_set1_epi32(q.max_level));
        let signed =
            _mm512_mask_sub_epi32(magnitude, _mm512_cmplt_epi32_mask(c, zero), zero, magnitude);
        // SAFETY: the same mask bounds the store to the valid output lanes.
        unsafe {
            _mm512_mask_storeu_epi32(dst.add(i).cast(), mask, signed);
        }
        counts = _mm512_mask_add_epi32(
            counts,
            _mm512_cmpneq_epi32_mask(signed, zero) & mask,
            counts,
            _mm512_set1_epi32(1),
        );
    }
    u32::from(dc != 0) + _mm512_reduce_add_epi32(counts) as u32
}
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn quantize(coeffs: &mut [i32], q: &Av1Quantizer) -> u32 {
    // Hardware timing favours AVX2 for these small coefficient blocks.
    if coeffs.len() <= 16 {
        // SAFETY: same validated slices; AVX2 is required by this tier.
        return unsafe { super::avx2::quantize(coeffs, q) };
    }
    // SAFETY: same complete slice for input/output, each load precedes store.
    unsafe { levels(coeffs.as_ptr(), coeffs.as_mut_ptr(), coeffs.len(), q) }
}
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn quantize_from(coeffs: &[i32], q: &Av1Quantizer, out: &mut [i32]) -> u32 {
    // Hardware timing favours AVX2 for these small coefficient blocks.
    if coeffs.len() <= 16 {
        // SAFETY: same validated slices; AVX2 is required by this tier.
        return unsafe { super::avx2::quantize_from(coeffs, q, out) };
    }
    let out = &mut out[..coeffs.len()];
    // SAFETY: equal-sized disjoint slices and this ISA is available.
    unsafe { levels(coeffs.as_ptr(), out.as_mut_ptr(), coeffs.len(), q) }
}
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn dequantize(levels: &[i32], q: &Av1Quantizer, out: &mut [i32]) {
    // Hardware timing favours AVX2 for these small coefficient blocks.
    if levels.len() <= 64 {
        // SAFETY: same validated slices; AVX2 is required by this tier.
        return unsafe { super::avx2::dequantize(levels, q, out) };
    }
    // SAFETY: forward the same validated slices and ISA to the wide body.
    unsafe { dequantize_wide(levels, q, out) }
}

// Avoid saving the wide body's registers before taking the small-block fallback.
#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn dequantize_wide(levels: &[i32], q: &Av1Quantizer, out: &mut [i32]) {
    // SAFETY: the dispatcher validates the slices and ISA.
    unsafe {
        super::generic::dequantize::<crate::simd::avx512::I32x16>(levels, q, out);
    }
}
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn gain(original: &[i32], dequantized: &[i32]) -> i64 {
    let len = original.len().min(dequantized.len());
    let mut sum = _mm512_setzero_si512();
    for i in (0..len).step_by(16) {
        let mask = (u32::MAX >> (32 - (len - i).min(16))) as u16;
        // SAFETY: masks cover only the common prefix of both slices.
        let (a, b) = unsafe {
            (
                _mm512_maskz_loadu_epi32(mask, original.as_ptr().add(i).cast()),
                _mm512_maskz_loadu_epi32(mask, dequantized.as_ptr().add(i).cast()),
            )
        };
        let term = _mm512_sub_epi32(_mm512_slli_epi32::<1>(a), b);
        let even = _mm512_mul_epi32(term, b);
        let odd = _mm512_mul_epi32(_mm512_srli_epi64::<32>(term), _mm512_srli_epi64::<32>(b));
        sum = _mm512_add_epi64(sum, _mm512_add_epi64(even, odd));
    }
    _mm512_reduce_add_epi64(sum)
}
