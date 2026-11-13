//! Plane SSE only; native AVX-512 does not replace browser input conversion.
//! A row is bounded by `MAX_SSE_ROW`, so even the total fits unsigned u32.
use core::arch::x86_64::{
    _mm256_maskz_loadu_epi8, _mm512_add_epi32, _mm512_cvtepu8_epi16, _mm512_madd_epi16,
    _mm512_reduce_add_epi32, _mm512_setzero_si512, _mm512_sub_epi16,
};
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub(crate) unsafe fn sse_row(a: &[u8], b: &[u8]) -> u32 {
    let mut sum = _mm512_setzero_si512();
    for x in (0..a.len()).step_by(32) {
        let mask = u32::MAX >> (32 - (a.len() - x).min(32));
        // SAFETY: equally sized validated rows; masked loads stop at their end.
        let (a, b) = unsafe {
            (
                _mm256_maskz_loadu_epi8(mask, a.as_ptr().add(x).cast()),
                _mm256_maskz_loadu_epi8(mask, b.as_ptr().add(x).cast()),
            )
        };
        let delta = _mm512_sub_epi16(_mm512_cvtepu8_epi16(a), _mm512_cvtepu8_epi16(b));
        sum = _mm512_add_epi32(sum, _mm512_madd_epi16(delta, delta));
    }
    _mm512_reduce_add_epi32(sum) as u32
}
