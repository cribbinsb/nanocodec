//! AVX2 intra estimate statistics: up to 32 bytes of a row per vector, the
//! even samples as the low bytes of its 16-bit lanes (`vpand`), every
//! square by `vpmaddwd` of the 16-bit differences.

use core::arch::x86_64::{
    __m256i, _mm_add_epi32, _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_loadl_epi64, _mm_loadu_si128,
    _mm_shuffle_epi32, _mm256_add_epi32, _mm256_and_si256, _mm256_castsi256_si128,
    _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_set1_epi16,
    _mm256_setzero_si256, _mm256_sub_epi16, _mm256_zextsi128_si256,
};

use super::{EstimateStats, SAMPLE_STEP};

/// Bytes of a row per vector.
const CHUNK: usize = 32;

/// The even samples of `bytes` (`n` of them, 4 to 32; the rest of the
/// vector zero) as 16-bit lanes.
#[inline]
#[target_feature(enable = "avx2")]
fn even(bytes: &[u8], n: usize) -> __m256i {
    // SAFETY: every load reads exactly `n` checked bytes.
    let raw = unsafe {
        match n {
            4 => _mm256_zextsi128_si256(_mm_cvtsi32_si128(
                bytes[..4].as_ptr().cast::<i32>().read_unaligned(),
            )),
            8 => _mm256_zextsi128_si256(_mm_loadl_epi64(bytes[..8].as_ptr().cast())),
            16 => _mm256_zextsi128_si256(_mm_loadu_si128(bytes[..16].as_ptr().cast())),
            _ => _mm256_loadu_si256(bytes[..CHUNK].as_ptr().cast()),
        }
    };
    _mm256_and_si256(raw, _mm256_set1_epi16(0x00ff))
}

/// The sum of the eight 32-bit lanes.
#[inline]
#[target_feature(enable = "avx2")]
fn total(lanes: __m256i) -> u32 {
    let half = _mm_add_epi32(
        _mm256_castsi256_si128(lanes),
        _mm256_extracti128_si256::<1>(lanes),
    );
    let quarter = _mm_add_epi32(half, _mm_shuffle_epi32::<0b0100_1110>(half));
    let eighth = _mm_add_epi32(quarter, _mm_shuffle_epi32::<0b1011_0001>(quarter));
    _mm_cvtsi128_si32(eighth) as u32
}

/// The AVX2 [`super::StatsFn`].
///
/// # Safety
///
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn stats(src: &[u8], stride: usize, n: usize, edge: &[u8]) -> EstimateStats {
    assert!(super::ESTIMATE_SIDES.contains(&n) && edge.len() > 3 * n);
    let width = n.min(CHUNK);
    let ones = _mm256_set1_epi16(1);
    // The lanes that hold samples (a block narrower than a vector leaves
    // the rest zero in every load but the horizontal prediction's splat).
    let valid = even(&[u8::MAX; CHUNK], width);
    let zero = _mm256_setzero_si256();
    let (mut sum, mut squares) = (zero, zero);
    let mut sse = [zero; 4];
    let square = |a: __m256i, b: __m256i| {
        let d = _mm256_sub_epi16(a, b);
        _mm256_madd_epi16(d, d)
    };
    for r in (0..n).step_by(SAMPLE_STEP) {
        let left = _mm256_and_si256(_mm256_set1_epi16(i16::from(edge[n - 1 - r])), valid);
        for c0 in (0..n).step_by(CHUNK) {
            let v = even(&src[r * stride + c0..], width);
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(v, ones));
            squares = _mm256_add_epi32(squares, _mm256_madd_epi16(v, v));
            let predictions = [
                even(&edge[n + 1 + c0..], width),
                left,
                even(&edge[n + 2 + r + c0..], width),
                even(&edge[n - r + c0..], width),
            ];
            for (acc, p) in sse.iter_mut().zip(predictions) {
                *acc = _mm256_add_epi32(*acc, square(v, p));
            }
        }
    }
    EstimateStats {
        sum: total(sum),
        squares: total(squares),
        sse: sse.map(|lanes| total(lanes)),
    }
}
