//! The 4×4 extended-warp byte filter from dav2d's `ext_warp4x4_8bpc`
//! and `ext_warp4x4t_8bpc` (BSD-2-Clause, `THIRD_PARTY.md`). Shares
//! the AV1 narrow window load, paired row layout and output sinks.

use core::arch::x86_64::{
    __m128i, __m256i, _mm_cvtepi8_epi16, _mm_cvtsi32_si128, _mm_loadl_epi64, _mm_loadu_si128,
    _mm_setr_epi8, _mm_setzero_si128, _mm_shuffle_epi32, _mm_srli_si128, _mm_unpacklo_epi8,
    _mm_unpacklo_epi16, _mm256_add_epi16, _mm256_add_epi32, _mm256_broadcastd_epi32,
    _mm256_broadcastsi128_si256, _mm256_broadcastw_epi16, _mm256_castsi256_si128,
    _mm256_extracti128_si256, _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_mulhrs_epi16,
    _mm256_packs_epi32, _mm256_set_m128i, _mm256_set1_epi16, _mm256_set1_epi32,
    _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_srai_epi32, _mm256_sub_epi16,
};

use super::{PrepOut, PutOut, Sink};
use crate::mc::avx2_byte_dot::short_row;

const SIDE: usize = 4;
const WINDOW: usize = 11;
const TAPS: usize = 8;
const PAIRS: usize = TAPS / 2;

#[target_feature(enable = "avx2")]
#[inline]
fn byte_taps(taps: &[i8; TAPS]) -> [__m256i; PAIRS] {
    // SAFETY: the eight signed bytes form four adjacent pairs.
    let weights = unsafe { _mm_loadl_epi64(taps.as_ptr().cast()) };
    [
        _mm256_broadcastw_epi16(weights),
        _mm256_broadcastw_epi16(_mm_srli_si128::<2>(weights)),
        _mm256_broadcastw_epi16(_mm_srli_si128::<4>(weights)),
        _mm256_broadcastw_epi16(_mm_srli_si128::<6>(weights)),
    ]
}

#[target_feature(enable = "avx2")]
#[inline]
fn word_taps(taps: &[i8; TAPS]) -> [__m256i; PAIRS] {
    // SAFETY: the eight weights each fit a signed word.
    let weights = _mm_cvtepi8_epi16(unsafe { _mm_loadl_epi64(taps.as_ptr().cast()) });
    [
        _mm256_broadcastsi128_si256(_mm_shuffle_epi32::<0x00>(weights)),
        _mm256_broadcastsi128_si256(_mm_shuffle_epi32::<0x55>(weights)),
        _mm256_broadcastsi128_si256(_mm_shuffle_epi32::<0xaa>(weights)),
        _mm256_broadcastsi128_si256(_mm_shuffle_epi32::<0xff>(weights)),
    ]
}

#[target_feature(enable = "avx2")]
#[inline]
fn horizontal_taps(taps: &[i8; TAPS]) -> [__m256i; 2] {
    // SAFETY: the two groups each read four of the eight weights.
    let weights = unsafe { _mm_loadl_epi64(taps.as_ptr().cast()) };
    [
        _mm256_broadcastd_epi32(weights),
        _mm256_broadcastd_epi32(_mm_srli_si128::<4>(weights)),
    ]
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn horizontal_pair<const PADDED: bool>(
    earlier: *const u8,
    later: *const u8,
    [a, b]: [__m256i; 2],
) -> __m256i {
    let load = |src: *const u8| {
        // SAFETY: a tight row has 11 samples, a padded one has 16.
        unsafe {
            if PADDED {
                _mm_loadu_si128(src.cast())
            } else {
                short_row::<PAIRS, SIDE>(src)
            }
        }
    };
    let pixels = _mm256_set_m128i(load(later), load(earlier));
    let first = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6,
    ));
    let second = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10,
    ));
    let outer = _mm256_maddubs_epi16(_mm256_shuffle_epi8(pixels, first), a);
    let inner = _mm256_maddubs_epi16(_mm256_shuffle_epi8(pixels, second), b);
    // The two alternating four-tap groups each have positive weight <=127
    // and negative weight >=-14 over the 63 phases. Their signed word sums
    // add exactly in dwords, even when the full eight-tap sum exceeds i16.
    _mm256_madd_epi16(_mm256_add_epi16(outer, inner), _mm256_set1_epi16(1))
}

/// The eight-tap sums can exceed signed 16 bits at full precision. Each
/// two-tap dot fits, so subtracting 128 * 128 in the wrapping sum recovers
/// a signed centred value. Restore the centre after rounding to four
/// intermediate fraction bits.
#[target_feature(enable = "avx2")]
#[inline]
fn intermediate(sum: __m256i) -> __m256i {
    let centred = _mm256_sub_epi16(sum, _mm256_set1_epi16(16384));
    _mm256_add_epi16(
        _mm256_mulhrs_epi16(centred, _mm256_set1_epi16(1 << 12)),
        _mm256_set1_epi16(2048),
    )
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn horizontal<const PADDED: bool, S: Sink>(
    src: *const u8,
    stride: usize,
    taps: &[i8; TAPS],
    out: S,
) {
    let taps = horizontal_taps(taps);
    for row in [0, 2] {
        // SAFETY: two source rows have the declared width and two output
        // rows have four writable samples.
        unsafe {
            let sum =
                horizontal_pair::<PADDED>(src.add(row * stride), src.add((row + 1) * stride), taps);
            let words = if S::PREP {
                _mm256_srai_epi32::<3>(_mm256_add_epi32(sum, _mm256_set1_epi32(4)))
            } else {
                _mm256_srai_epi32::<7>(_mm256_add_epi32(sum, _mm256_set1_epi32(68)))
            };
            out.two::<SIDE>(row, _mm256_packs_epi32(words, words));
        }
    }
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn vertical<S: Sink>(src: *const u8, stride: usize, taps: &[i8; TAPS], out: S) {
    let taps = byte_taps(taps);
    // SAFETY: all eleven input rows cover four samples beginning at src.
    let rows: [__m128i; WINDOW] = unsafe {
        core::array::from_fn(|i| {
            _mm_cvtsi32_si128(src.add(i * stride).cast::<i32>().read_unaligned())
        })
    };
    for row in [0, 2] {
        let mut sum = _mm256_setzero_si256();
        for (j, &taps) in taps.iter().enumerate() {
            let y = row + j * 2;
            let early = _mm_unpacklo_epi8(rows[y], rows[y + 1]);
            let late = _mm_unpacklo_epi8(rows[y + 1], rows[y + 2]);
            let inputs = _mm256_set_m128i(late, early);
            sum = _mm256_add_epi16(sum, _mm256_maddubs_epi16(inputs, taps));
        }
        let value = if S::PREP {
            intermediate(sum)
        } else {
            let centred = _mm256_sub_epi16(sum, _mm256_set1_epi16(16384));
            _mm256_add_epi16(
                _mm256_mulhrs_epi16(centred, _mm256_set1_epi16(1 << 8)),
                _mm256_set1_epi16(128),
            )
        };
        // SAFETY: both output rows have four writable samples.
        unsafe { out.two::<SIDE>(row, value) };
    }
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn both<const PADDED: bool, S: Sink>(
    src: *const u8,
    stride: usize,
    h: &[i8; TAPS],
    v: &[i8; TAPS],
    out: S,
) {
    let horizontal_taps = horizontal_taps(h);
    let vertical_taps = word_taps(v);
    let mut middle = [_mm_setzero_si128(); WINDOW];
    for row in [0, 2, 4, 6, 8, 10] {
        let next = (row + 1).min(WINDOW - 1);
        // SAFETY: the source covers eleven rows of eleven samples.
        let sum = unsafe {
            horizontal_pair::<PADDED>(
                src.add(row * stride),
                src.add(next * stride),
                horizontal_taps,
            )
        };
        let words = _mm256_srai_epi32::<3>(_mm256_add_epi32(sum, _mm256_set1_epi32(4)));
        let mid = _mm256_packs_epi32(words, words);
        middle[row] = _mm256_castsi256_si128(mid);
        middle[next] = _mm256_extracti128_si256::<1>(mid);
    }
    for row in [0, 2] {
        let mut sum = _mm256_set1_epi32(if S::PREP { 64 } else { 1024 });
        for (j, &taps) in vertical_taps.iter().enumerate() {
            let y = row + j * 2;
            let early = _mm_unpacklo_epi16(middle[y], middle[y + 1]);
            let late = _mm_unpacklo_epi16(middle[y + 1], middle[y + 2]);
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(_mm256_set_m128i(late, early), taps));
        }
        let words = if S::PREP {
            _mm256_srai_epi32::<7>(sum)
        } else {
            _mm256_srai_epi32::<11>(sum)
        };
        // SAFETY: both output rows have four writable samples.
        unsafe { out.two::<SIDE>(row, _mm256_packs_epi32(words, words)) };
    }
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn filter<S: Sink>(
    src: *const u8,
    stride: usize,
    taps: (Option<&[i8; TAPS]>, Option<&[i8; TAPS]>),
    out: S,
    padded: bool,
) {
    // SAFETY: the wrappers select a checked source footprint and output.
    unsafe {
        match taps {
            (Some(x), Some(y)) if padded => both::<true, _>(src, stride, x, y, out),
            (Some(x), Some(y)) => both::<false, _>(src, stride, x, y, out),
            (Some(x), None) if padded => horizontal::<true, _>(src, stride, x, out),
            (Some(x), None) => horizontal::<false, _>(src, stride, x, out),
            (None, Some(y)) => vertical(src, stride, y, out),
            (None, None) => unreachable!("zero-phase extended warp is a copy"),
        }
    }
}

/// # Safety
/// The AVX2 tier is enabled. The source covers the needed tap footprint
/// and the destination has four editable rows of four samples.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn put(
    dst: *mut u8,
    dst_stride: usize,
    src: *const u8,
    src_stride: usize,
    h: Option<&[i8; TAPS]>,
    v: Option<&[i8; TAPS]>,
    padded: bool,
) {
    let out = PutOut {
        dst,
        stride: dst_stride,
    };
    // SAFETY: the caller validated the footprint.
    unsafe { filter(src, src_stride, (h, v), out, padded) };
}

/// # Safety
/// As `put`, with the destination at compound precision.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn prep(
    dst: *mut i16,
    dst_stride: usize,
    src: *const u8,
    src_stride: usize,
    h: Option<&[i8; TAPS]>,
    v: Option<&[i8; TAPS]>,
    padded: bool,
) {
    let out = PrepOut {
        dst,
        stride: dst_stride,
    };
    // SAFETY: the caller validated the footprint.
    unsafe { filter(src, src_stride, (h, v), out, padded) };
}
