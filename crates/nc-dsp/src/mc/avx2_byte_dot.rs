//! Shared AVX2 byte-tap convolution for AV1, HEVC, VVC and VP9.
//! Both halves of the vector gather byte pairs for eight outputs. For a
//! wide row they cover neighbouring outputs, and for a narrow pair they
//! cover neighbouring rows. The caller chooses the taps and rounding.
//!
//! The schedule follows dav1d's AVX2 MC (BSD-2-Clause, see `THIRD_PARTY.md`).

use core::arch::x86_64::{
    __m128i, __m256i, _mm_cvtsi32_si128, _mm_insert_epi8, _mm_loadl_epi64, _mm_loadu_si128,
    _mm_or_si128, _mm_slli_si128, _mm256_add_epi16, _mm256_broadcastsi128_si256,
    _mm256_castsi128_si256, _mm256_inserti128_si256, _mm256_loadu_si256, _mm256_maddubs_epi16,
    _mm256_set1_epi16, _mm256_shuffle_epi8,
};

const LANES: usize = 16;
const HALF: usize = 8;

/// Tap pairs as signed byte pairs in every 16-bit lane (`vpmaddubsw`).
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) fn byte_pairs<const P: usize>(pairs: [(i16, i16); P]) -> [__m256i; P] {
    pairs.map(|(lo, hi)| _mm256_set1_epi16(i16::from_le_bytes([lo as i8 as u8, hi as i8 as u8])))
}

/// `vpshufb` masks gathering the pixels of tap pair `p` for 8 outputs per
/// lane: output `o` of a lane takes bytes `o + 2p` and `o + 2p + 1`, from
/// lane byte `high_start` on in the high lane (8 or 16 outputs a row).
const fn masks<const P: usize>(high_offset: usize) -> [[u8; 32]; P] {
    let mut out = [[0; 32]; P];
    let mut p = 0;
    while p < P {
        let mut i = 0;
        while i < 32 {
            let output = i % 16 / 2;
            let shift = if i >= 16 { high_offset } else { 0 };
            out[p][i] = (output + 2 * p + i % 2 + shift) as u8;
            i += 1;
        }
        p += 1;
    }
    out
}

/// Source bytes `n` outputs of a row read with `P` tap pairs.
pub(super) const fn span(n: usize, pairs: usize) -> usize {
    n + 2 * pairs - 1
}

/// Masks of one 16-output row: the high lane is loaded from byte
/// `span(16, P) - 16`, outputs 8..16 start 8 bytes in, so its masks start
/// `8 - (span - 16)` bytes in.
const MASKS_ROW_4: [[u8; 32]; 2] = masks::<2>(HALF + 16 - span(LANES, 2));
const MASKS_ROW_6: [[u8; 32]; 3] = masks::<3>(HALF + 16 - span(LANES, 3));
const MASKS_ROW_8: [[u8; 32]; 4] = masks::<4>(HALF + 16 - span(LANES, 4));
/// Masks of two rows of up to 8 outputs, one per lane.
const MASKS_TWO_4: [[u8; 32]; 2] = masks::<2>(0);
const MASKS_TWO_6: [[u8; 32]; 3] = masks::<3>(0);
const MASKS_TWO_8: [[u8; 32]; 4] = masks::<4>(0);

/// The filter sum of the pixels in `pixels` gathered by `masks`.
#[target_feature(enable = "avx2")]
#[inline]
fn pair_sums<const P: usize>(pixels: __m256i, masks: &[[u8; 32]], taps: &[__m256i; P]) -> __m256i {
    let mut sum = _mm256_set1_epi16(0);
    for (mask, weight) in masks.iter().zip(taps) {
        // SAFETY: `mask` is 32 readable bytes.
        let mask = unsafe { _mm256_loadu_si256(mask.as_ptr().cast()) };
        let product = _mm256_maddubs_epi16(_mm256_shuffle_epi8(pixels, mask), *weight);
        sum = _mm256_add_epi16(sum, product);
    }
    sum
}

/// Unrounded horizontal sums of 16 outputs of the row at `src`, reading
/// exactly `span(16, P)` bytes.
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn row_16<const P: usize>(src: *const u8, taps: &[__m256i; P]) -> __m256i {
    // SAFETY: the caller provides span(16, P) readable bytes; both 16-byte
    // loads lie inside them.
    let pixels = unsafe {
        _mm256_inserti128_si256::<1>(
            _mm256_castsi128_si256(_mm_loadu_si128(src.cast())),
            _mm_loadu_si128(src.add(span(LANES, P) - 16).cast()),
        )
    };
    let masks = match P {
        2 => &MASKS_ROW_4[..],
        3 => &MASKS_ROW_6[..],
        _ => &MASKS_ROW_8[..],
    };
    pair_sums(pixels, masks, taps)
}

/// The `span(N, P)` bytes (7 to 15) of a row at `src`, from two
/// overlapping 8-byte loads (4-byte loads for 7). The nine-byte
/// H.264 window uses a byte insert to load the final sample.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn short_row<const P: usize, const N: usize>(src: *const u8) -> __m128i {
    let bytes = span(N, P);
    // SAFETY: the caller provides `bytes` readable bytes.
    unsafe {
        if bytes == 7 {
            let low = _mm_cvtsi32_si128(src.cast::<i32>().read_unaligned());
            let high = _mm_cvtsi32_si128(src.add(3).cast::<i32>().read_unaligned());
            return _mm_or_si128(low, _mm_slli_si128::<3>(high));
        }
        let low = _mm_loadl_epi64(src.cast());
        if bytes == 9 {
            return _mm_insert_epi8::<8>(low, i32::from(src.add(8).read()));
        }
        let high = _mm_loadl_epi64(src.add(bytes - 8).cast());
        let high = match bytes {
            11 => _mm_slli_si128::<3>(high),
            13 => _mm_slli_si128::<5>(high),
            _ => _mm_slli_si128::<7>(high),
        };
        _mm_or_si128(low, high)
    }
}

/// Unrounded horizontal sums of `N` (4 or 8) outputs of two rows, `a` in
/// the low lane and `b` in the high one.
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn two_rows<const P: usize, const N: usize>(
    a: *const u8,
    b: *const u8,
    taps: &[__m256i; P],
) -> __m256i {
    // SAFETY: forwarded from the caller.
    let pixels = unsafe {
        _mm256_inserti128_si256::<1>(
            _mm256_castsi128_si256(short_row::<P, N>(a)),
            short_row::<P, N>(b),
        )
    };
    let masks = match P {
        2 => &MASKS_TWO_4[..],
        3 => &MASKS_TWO_6[..],
        _ => &MASKS_TWO_8[..],
    };
    pair_sums(pixels, masks, taps)
}

/// The final narrow row of a VP9 strip can be unpaired.
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn one_narrow_row<const N: usize>(
    src: *const u8,
    taps: &[__m256i; 4],
) -> __m256i {
    // SAFETY: the caller provides span(N, 4) readable bytes.
    let pixels = unsafe { _mm256_broadcastsi128_si256(short_row::<4, N>(src)) };
    pair_sums(pixels, &MASKS_TWO_8, taps)
}
