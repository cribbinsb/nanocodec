//! AVX2 Gram rows: sixteen values per vector, `vpmaddwd` products summed
//! in eight 32-bit lanes per pair, reduced in 64 bits once per pair.

use super::GRAM_ROW_ALIGN;
use core::arch::x86_64::{
    __m256i, _mm256_add_epi32, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_setzero_si256,
    _mm256_storeu_si256,
};

/// One vector of a row.
#[target_feature(enable = "avx2")]
fn load(values: &[i16; GRAM_ROW_ALIGN]) -> __m256i {
    // SAFETY: AVX2 is enabled; the load reads exactly the 32 bytes of `values`.
    unsafe { _mm256_loadu_si256(values.as_ptr().cast()) }
}

/// The sum of the eight 32-bit lanes of `sum`, in 64 bits.
#[target_feature(enable = "avx2")]
fn reduce(sum: __m256i) -> i64 {
    let mut lanes = [0i32; 8];
    // SAFETY: AVX2 is enabled; the store writes exactly the 32 bytes of `lanes`.
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), sum) };
    lanes.iter().map(|&lane| i64::from(lane)).sum()
}

/// AVX2 [`super::GramRowsFn`]: each 32-bit lane takes two products of
/// every sixteenth column, so it sums at most 2^9 products.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn gram_rows(rows: &[&[i16]], out: &mut [i64]) {
    let mut entry = 0;
    for (i, first) in rows.iter().enumerate() {
        let (a, _) = first.as_chunks::<GRAM_ROW_ALIGN>();
        for second in &rows[i..] {
            let (b, _) = second.as_chunks::<GRAM_ROW_ALIGN>();
            let mut sum = _mm256_setzero_si256();
            for (a, b) in a.iter().zip(b) {
                sum = _mm256_add_epi32(sum, _mm256_madd_epi16(load(a), load(b)));
            }
            out[entry] += reduce(sum);
            entry += 1;
        }
    }
}
