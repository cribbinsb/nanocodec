//! AVX2 end of block: 16 levels per step, the table masked by `level != 0`
//! (`vpcmpeqw` then `vpandn`), a running `vpmaxuw`, then a horizontal
//! maximum by `phminposuw` of the complement.

use core::arch::x86_64::{
    __m256i, _mm_cvtsi128_si32, _mm_max_epu16, _mm_minpos_epu16, _mm_or_si128, _mm_set1_epi16,
    _mm_shuffle_epi32, _mm_xor_si128, _mm256_abs_epi32, _mm256_andnot_si256,
    _mm256_castsi256_si128, _mm256_cmpeq_epi16, _mm256_cmpeq_epi32, _mm256_extracti128_si256,
    _mm256_loadu_si256, _mm256_max_epu16, _mm256_or_si256, _mm256_packs_epi32,
    _mm256_permute4x64_epi64, _mm256_setzero_si256,
};

/// Levels per step.
const LANES: usize = 16;

/// The maximum unsigned 16-bit lane of `lanes`.
///
/// # Safety
///
/// AVX2 must be enabled in the caller.
#[inline]
#[target_feature(enable = "avx2")]
pub(super) fn horizontal_max(lanes: __m256i) -> u16 {
    let half = _mm_max_epu16(
        _mm256_castsi256_si128(lanes),
        _mm256_extracti128_si256::<1>(lanes),
    );
    // `phminposuw` finds the minimum: of the complements, the maximum.
    let ones = _mm_set1_epi16(-1);
    let minimum = _mm_minpos_epu16(_mm_xor_si128(half, ones));
    !(_mm_cvtsi128_si32(minimum) as u16)
}

/// The AVX2 [`super::EobFn`].
///
/// # Safety
///
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn end_of_block(levels: &[i16], table: &[u16]) -> usize {
    let n = levels.len().min(table.len());
    let (whole, _) = levels[..n].as_chunks::<LANES>();
    let (ends, _) = table[..n].as_chunks::<LANES>();
    let zero = _mm256_setzero_si256();
    let mut largest = zero;
    for (chunk, end) in whole.iter().zip(ends) {
        // SAFETY: each load is one whole 16-element chunk.
        let (level, end) = unsafe {
            (
                _mm256_loadu_si256(chunk.as_ptr().cast()),
                _mm256_loadu_si256(end.as_ptr().cast()),
            )
        };
        let coded = _mm256_andnot_si256(_mm256_cmpeq_epi16(level, zero), end);
        largest = _mm256_max_epu16(largest, coded);
    }
    let done = whole.len() * LANES;
    usize::from(horizontal_max(largest)).max(super::scalar(&levels[done..n], &table[done..n]))
}

/// The OR of the eight 32-bit lanes.
#[inline]
#[target_feature(enable = "avx2")]
fn horizontal_or(lanes: __m256i) -> u32 {
    let half = _mm_or_si128(
        _mm256_castsi256_si128(lanes),
        _mm256_extracti128_si256::<1>(lanes),
    );
    let quarter = _mm_or_si128(half, _mm_shuffle_epi32::<0b0100_1110>(half));
    let eighth = _mm_or_si128(quarter, _mm_shuffle_epi32::<0b1011_0001>(quarter));
    _mm_cvtsi128_si32(eighth) as u32
}

/// The AVX2 [`super::WideEobFn`]: 16 levels per step as two 8-lane loads,
/// whose zero masks narrow (`vpackssdw`, then `vpermq` to restore the
/// order) onto the 16 table entries.
///
/// # Safety
///
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn end_of_block_wide(levels: &[i32], table: &[u16]) -> (usize, u32) {
    let n = levels.len().min(table.len());
    let (whole, _) = levels[..n].as_chunks::<LANES>();
    let (ends, _) = table[..n].as_chunks::<LANES>();
    let zero = _mm256_setzero_si256();
    let (mut largest, mut bits) = (zero, zero);
    for (chunk, end) in whole.iter().zip(ends) {
        // SAFETY: each load is one half of a whole 16-element chunk.
        let (low, high, end) = unsafe {
            (
                _mm256_loadu_si256(chunk.as_ptr().cast()),
                _mm256_loadu_si256(chunk[LANES / 2..].as_ptr().cast()),
                _mm256_loadu_si256(end.as_ptr().cast()),
            )
        };
        let zeros = _mm256_permute4x64_epi64::<0b1101_1000>(_mm256_packs_epi32(
            _mm256_cmpeq_epi32(low, zero),
            _mm256_cmpeq_epi32(high, zero),
        ));
        largest = _mm256_max_epu16(largest, _mm256_andnot_si256(zeros, end));
        bits = _mm256_or_si256(
            bits,
            _mm256_or_si256(_mm256_abs_epi32(low), _mm256_abs_epi32(high)),
        );
    }
    let done = whole.len() * LANES;
    let (end, tail_bits) = super::scalar_wide(&levels[done..n], &table[done..n]);
    (
        usize::from(horizontal_max(largest)).max(end),
        horizontal_or(bits) | tail_bits,
    )
}
