//! AVX-512 end of block: 32 levels per step, a masked `vpmaxuw` under
//! `vptestmw` (`level != 0`); blocks under 64 levels run the AVX2 kernel.

use core::arch::x86_64::{
    _mm256_max_epu16, _mm512_castsi512_si256, _mm512_extracti64x4_epi64, _mm512_loadu_si512,
    _mm512_mask_max_epu16, _mm512_setzero_si512, _mm512_test_epi16_mask,
};

/// Levels per step.
const LANES: usize = 32;
/// Blocks shorter than this run the AVX2 kernel (16x16 and up use 512 bits).
const MIN_LEVELS: usize = 2 * LANES;

/// The AVX-512 [`super::EobFn`].
///
/// # Safety
///
/// The CPU must support the AVX-512 tier (`Isa::Avx512`).
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub(super) unsafe fn end_of_block(levels: &[i16], table: &[u16]) -> usize {
    let n = levels.len().min(table.len());
    if n < MIN_LEVELS {
        // SAFETY: the AVX-512 tier includes AVX2.
        return unsafe { super::avx2::end_of_block(levels, table) };
    }
    let (whole, _) = levels[..n].as_chunks::<LANES>();
    let (ends, _) = table[..n].as_chunks::<LANES>();
    let mut largest = _mm512_setzero_si512();
    for (chunk, end) in whole.iter().zip(ends) {
        // SAFETY: each load is one whole 32-element chunk.
        let (level, end) = unsafe {
            (
                _mm512_loadu_si512(chunk.as_ptr().cast()),
                _mm512_loadu_si512(end.as_ptr().cast()),
            )
        };
        let coded = _mm512_test_epi16_mask(level, level);
        largest = _mm512_mask_max_epu16(largest, coded, largest, end);
    }
    let half = _mm256_max_epu16(
        _mm512_castsi512_si256(largest),
        _mm512_extracti64x4_epi64::<1>(largest),
    );
    let done = whole.len() * LANES;
    usize::from(super::avx2::horizontal_max(half))
        .max(super::scalar(&levels[done..n], &table[done..n]))
}
