//! Sixteen-coefficient AVX-512 quantisation through the existing exact
//! reciprocal proof and scalar DC rule. Masked final groups avoid reading
//! padding. Widened products preserve the same signed coefficient gains.

#![allow(
    clippy::inline_always,
    reason = "the storage loads and stores must inline into the ISA kernels"
)]
use super::avx2::Lanes8;
use super::scalar::{fix_dc, quantize_tail};
use super::{Av1Quantizer, Reciprocal, TxfmFns, quant_terms};
use core::arch::x86_64::{
    __m512i, _mm_cvtsi32_si128, _mm256_maskz_loadu_epi16, _mm512_abs_epi32, _mm512_add_epi32,
    _mm512_add_epi64, _mm512_cmplt_epi32_mask, _mm512_cmpneq_epi32_mask, _mm512_cvtepi16_epi32,
    _mm512_mask_add_epi32, _mm512_mask_blend_epi32, _mm512_mask_cvtepi32_storeu_epi16,
    _mm512_mask_storeu_epi32, _mm512_mask_sub_epi32, _mm512_maskz_loadu_epi32, _mm512_min_epi32,
    _mm512_mul_epi32, _mm512_mul_epu32, _mm512_reduce_add_epi32, _mm512_reduce_add_epi64,
    _mm512_set1_epi32, _mm512_set1_epi64, _mm512_setzero_si512, _mm512_sll_epi32,
    _mm512_slli_epi32, _mm512_slli_epi64, _mm512_srl_epi64, _mm512_srli_epi64, _mm512_sub_epi32,
};

pub(super) fn install(table: &mut TxfmFns) {
    table.quantize = quantize;
    table.quantize_from = quantize_from::<i16>;
    table.quantize_from_wide = quantize_from::<i32>;
    table.dequantize = dequantize;
    table.coefficient_gain = gain;
}

/// Quantiser storage loaded and stored sixteen 32-bit lanes at a time,
/// masked.
trait Lanes16: Lanes8 {
    /// The `mask` lanes of the sixteen values at `src`, sign-extended (the
    /// others zero).
    ///
    /// # Safety
    ///
    /// AVX-512 is enabled; `src` holds the masked values.
    unsafe fn load16(mask: u16, src: *const Self) -> __m512i;
    /// Store the `mask` lanes (each fitting `Self`) at `dst`.
    ///
    /// # Safety
    ///
    /// AVX-512 is enabled; `dst` holds the masked values.
    unsafe fn store16(dst: *mut Self, mask: u16, lanes: __m512i);
}

impl Lanes16 for i32 {
    #[inline(always)]
    unsafe fn load16(mask: u16, src: *const Self) -> __m512i {
        // SAFETY: caller contract.
        unsafe { _mm512_maskz_loadu_epi32(mask, src.cast()) }
    }
    #[inline(always)]
    unsafe fn store16(dst: *mut Self, mask: u16, lanes: __m512i) {
        // SAFETY: caller contract.
        unsafe { _mm512_mask_storeu_epi32(dst.cast(), mask, lanes) }
    }
}

impl Lanes16 for i16 {
    #[inline(always)]
    unsafe fn load16(mask: u16, src: *const Self) -> __m512i {
        // SAFETY: caller contract.
        unsafe { _mm512_cvtepi16_epi32(_mm256_maskz_loadu_epi16(mask, src)) }
    }
    #[inline(always)]
    unsafe fn store16(dst: *mut Self, mask: u16, lanes: __m512i) {
        // Truncating narrow: every lane fits 16 bits.
        // SAFETY: caller contract.
        unsafe { _mm512_mask_cvtepi32_storeu_epi16(dst.cast(), mask, lanes) }
    }
}

/// Masked vectors of 16 from coefficient 0 at the AC step, then the DC
/// fix-up (`fix_dc`). Source and destination hold len values and are equal
/// or disjoint.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn levels<T: Lanes16>(src: *const T, dst: *mut T, len: usize, q: &Av1Quantizer) -> u32 {
    let (offset, divisor) = quant_terms(q, q.ac_step);
    let Some(reciprocal) = Reciprocal::new(divisor) else {
        // SAFETY: forwarded contract.
        return unsafe { quantize_tail(src, dst, 0, len, q) };
    };
    if len == 0 {
        return 0;
    }
    // SAFETY: len is nonzero; read before the vector pass overwrites it.
    let dc = unsafe { (*src).widen() };
    let zero = _mm512_setzero_si512();
    let mut counts = zero;
    let input_shift = _mm_cvtsi32_si128(7 + q.log_scale as i32);
    let result_shift = _mm_cvtsi32_si128(reciprocal.shift as i32);
    let multiplier = _mm512_set1_epi64(i64::from(reciprocal.multiplier));
    for i in (0..len).step_by(16) {
        let mask = (u32::MAX >> (32 - (len - i).min(16))) as u16;
        // SAFETY: only valid lanes are read; input and output may alias.
        let c = unsafe { T::load16(mask, src.add(i)) };
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
            T::store16(dst.add(i), mask, signed);
        }
        counts = _mm512_mask_add_epi32(
            counts,
            _mm512_cmpneq_epi32_mask(signed, zero) & mask,
            counts,
            _mm512_set1_epi32(1),
        );
    }
    // SAFETY: the vector pass wrote `dst[0]` (len >= 1).
    unsafe { fix_dc(dst, dc, _mm512_reduce_add_epi32(counts) as u32, q) }
}
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn quantize(coeffs: &mut [i16], q: &Av1Quantizer) -> u32 {
    // Hardware timing favours AVX2 for these small coefficient blocks.
    if coeffs.len() <= 16 {
        // SAFETY: same validated slices; AVX2 is required by this tier.
        return unsafe { super::avx2::quantize(coeffs, q) };
    }
    // SAFETY: same complete slice for input/output, each load precedes store.
    unsafe { levels(coeffs.as_ptr(), coeffs.as_mut_ptr(), coeffs.len(), q) }
}
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn quantize_from<T: Lanes16>(coeffs: &[T], q: &Av1Quantizer, out: &mut [T]) -> u32 {
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
unsafe fn dequantize(levels: &[i16], q: &Av1Quantizer, out: &mut [i16]) {
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
unsafe fn dequantize_wide(levels: &[i16], q: &Av1Quantizer, out: &mut [i16]) {
    // SAFETY: the dispatcher validates the slices and ISA.
    unsafe {
        super::generic::dequantize::<crate::simd::avx512::I32x16>(levels, q, out);
    }
}
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn gain(original: &[i16], dequantized: &[i16]) -> i64 {
    let len = original.len().min(dequantized.len());
    let mut sum = _mm512_setzero_si512();
    for i in (0..len).step_by(16) {
        let mask = (u32::MAX >> (32 - (len - i).min(16))) as u16;
        // SAFETY: masks cover only the common prefix of both slices.
        let (a, b) = unsafe {
            (
                _mm512_cvtepi16_epi32(_mm256_maskz_loadu_epi16(mask, original.as_ptr().add(i))),
                _mm512_cvtepi16_epi32(_mm256_maskz_loadu_epi16(mask, dequantized.as_ptr().add(i))),
            )
        };
        let term = _mm512_sub_epi32(_mm512_slli_epi32::<1>(a), b);
        let even = _mm512_mul_epi32(term, b);
        let odd = _mm512_mul_epi32(_mm512_srli_epi64::<32>(term), _mm512_srli_epi64::<32>(b));
        sum = _mm512_add_epi64(sum, _mm512_add_epi64(even, odd));
    }
    _mm512_reduce_add_epi64(sum)
}
