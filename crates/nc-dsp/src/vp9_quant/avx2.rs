//! AVX2 largest coefficient magnitude (VP9's all-zero test).

use core::arch::x86_64::{
    _mm_cvtsi128_si32, _mm_max_epu16, _mm_shuffle_epi32, _mm_srli_epi32, _mm256_abs_epi16,
    _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_max_epu16,
    _mm256_setzero_si256,
};

/// AVX2 [`super::MagnitudeFn`]: sixteen absolute values per vector, an
/// unsigned 16-bit running maximum (`|i16::MIN|` reads as 32768), then a
/// horizontal reduction.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn largest_magnitude(coefficients: &[i16]) -> u32 {
    let (whole, tail) = coefficients.as_chunks::<16>();
    // SAFETY: AVX2 is enabled; every load is one whole sixteen-lane chunk.
    let vector = unsafe {
        let mut largest = _mm256_setzero_si256();
        for chunk in whole {
            let c = _mm256_loadu_si256(chunk.as_ptr().cast());
            largest = _mm256_max_epu16(largest, _mm256_abs_epi16(c));
        }
        let half = _mm_max_epu16(
            _mm256_castsi256_si128(largest),
            _mm256_extracti128_si256::<1>(largest),
        );
        let quarter = _mm_max_epu16(half, _mm_shuffle_epi32::<0b0100_1110>(half));
        let eighth = _mm_max_epu16(quarter, _mm_shuffle_epi32::<0b1011_0001>(quarter));
        let sixteenth = _mm_max_epu16(eighth, _mm_srli_epi32::<16>(eighth));
        (_mm_cvtsi128_si32(sixteenth) as u32) & u32::from(u16::MAX)
    };
    // SAFETY: the scalar reference has no preconditions.
    vector.max(unsafe { super::largest_magnitude(tail) })
}
