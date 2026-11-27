//! Independent 8x8-cell residual statistics. Adjacent cells share contiguous
//! byte loads. Each cell owns its word sums and squared-difference lanes.

use core::arch::x86_64::{
    _mm_loadu_si128, _mm_storeu_si128, _mm_unpacklo_epi64, _mm256_add_epi16, _mm256_add_epi32,
    _mm256_castsi256_si128, _mm256_cvtepu8_epi16, _mm256_extracti128_si256, _mm256_hadd_epi32,
    _mm256_loadu_si256, _mm256_madd_epi16, _mm256_set1_epi16, _mm256_setzero_si256,
    _mm256_storeu_si256, _mm256_sub_epi16, _mm512_add_epi16, _mm512_add_epi32,
    _mm512_castsi512_si256, _mm512_cvtepu8_epi16, _mm512_dpwssd_epi32, _mm512_madd_epi16,
    _mm512_permutex2var_epi32, _mm512_set1_epi16, _mm512_setr_epi32, _mm512_setzero_si512,
    _mm512_shuffle_epi32, _mm512_sub_epi16,
};

use super::Variance;

/// Adjacent pairs and exact-width tails follow the existing row SSE family.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn avx2(src: &[u8], ss: usize, pred: &[u8], ps: usize, out: &mut [Variance]) {
    let (pairs, tail) = out.as_chunks_mut::<2>();
    for (i, pair) in pairs.iter_mut().enumerate() {
        let x = 16 * i;
        let mut sums = _mm256_setzero_si256();
        let mut squares = sums;
        for y in 0..8 {
            // SAFETY: the validated row spans both complete 8x8 cells.
            let (a, b) = unsafe {
                (
                    _mm_loadu_si128(src.as_ptr().add(y * ss + x).cast()),
                    _mm_loadu_si128(pred.as_ptr().add(y * ps + x).cast()),
                )
            };
            let difference = _mm256_sub_epi16(_mm256_cvtepu8_epi16(a), _mm256_cvtepu8_epi16(b));
            sums = _mm256_add_epi16(sums, difference);
            squares = _mm256_add_epi32(squares, _mm256_madd_epi16(difference, difference));
        }
        // Each 128-bit lane owns one cell. Each word sum is at most 2040.
        let sums = _mm256_madd_epi16(sums, _mm256_set1_epi16(1));
        let totals = _mm256_hadd_epi32(squares, sums);
        let totals = _mm256_hadd_epi32(totals, totals);
        let low = _mm256_castsi256_si128(totals);
        let high = _mm256_extracti128_si256::<1>(totals);
        let packed = _mm_unpacklo_epi64(low, high);
        // SAFETY: Variance has a C layout of one u32 followed by one i32.
        unsafe { _mm_storeu_si128(pair.as_mut_ptr().cast(), packed) };
    }
    if let [last] = tail {
        let x = 16 * pairs.len();
        // SAFETY: the remaining cell covers exactly eight bytes per row.
        *last = unsafe { super::avx2::variance_w8(&src[x..], ss, &pred[x..], ps, 8) };
    }
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
pub(super) unsafe fn icl(src: &[u8], ss: usize, pred: &[u8], ps: usize, out: &mut [Variance]) {
    let (groups, tail) = out.as_chunks_mut::<4>();
    for (i, cells) in groups.iter_mut().enumerate() {
        let x = 32 * i;
        let mut sums = _mm512_setzero_si512();
        let mut squares = sums;
        for y in 0..8 {
            // SAFETY: the validated row spans all four 8x8 cells.
            let (a, b) = unsafe {
                (
                    _mm256_loadu_si256(src.as_ptr().add(y * ss + x).cast()),
                    _mm256_loadu_si256(pred.as_ptr().add(y * ps + x).cast()),
                )
            };
            let difference = _mm512_sub_epi16(_mm512_cvtepu8_epi16(a), _mm512_cvtepu8_epi16(b));
            sums = _mm512_add_epi16(sums, difference);
            squares = _mm512_dpwssd_epi32(squares, difference, difference);
        }
        let sum_pairs = _mm512_madd_epi16(sums, _mm512_set1_epi16(1));
        let total = |lanes| {
            let pairs = _mm512_add_epi32(lanes, _mm512_shuffle_epi32::<0x4e>(lanes));
            _mm512_add_epi32(pairs, _mm512_shuffle_epi32::<0xb1>(pairs))
        };
        let indices = _mm512_setr_epi32(0, 16, 4, 20, 8, 24, 12, 28, 0, 0, 0, 0, 0, 0, 0, 0);
        let packed = _mm512_permutex2var_epi32(total(squares), indices, total(sum_pairs));
        // SAFETY: four C-layout variance entries occupy 32 bytes.
        unsafe { _mm256_storeu_si256(cells.as_mut_ptr().cast(), _mm512_castsi512_si256(packed)) };
    }
    if !tail.is_empty() {
        let x = 32 * groups.len();
        // SAFETY: the caller covers the tail and ICL also supports AVX2.
        unsafe { avx2(&src[x..], ss, &pred[x..], ps, tail) };
    }
}
