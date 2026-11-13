//! AVX2 forward colour conversion (browser encoder input), 32 pixels of a
//! row pair per iteration, straight from packed RGBA without a planar
//! transpose.
//!
//! Luma: `vpmaddubsw` with per-pixel signed byte weights `[66, -127, 25,
//! 0]` gives the 16-bit pairs `[66 R - 127 G, 25 B]`; the missing `256 G`
//! is the (R, G) word with R's byte masked off. One `vphaddw` sums
//! each pixel's pair (the total, at most 56100 plus the 4224 rounding and
//! offset, fits `u16`: the wrapping adds are exact), and a logical shift
//! and an unsigned pack give the bytes; a 32-bit permute restores pixel
//! order after the in-lane horizontal add and pack.
//!
//! Chroma: one in-lane byte shuffle puts each channel of a horizontal pixel
//! pair side by side, `vpmaddubsw` by ones sums them, the two rows are
//! added (at most 1020) and `(sum + 2) >> 2` is the exact rounded mean.
//! `vpmaddwd` with `[w_r, w_g, w_b, 0]` gives two 32-bit partial sums per
//! group, `vphaddd` adds them, and the rounding plus offset and a shift give
//! values already inside 16..=240 (`convert` module docs), so the packs never
//! clamp. A final 64-bit permute and byte shuffle put Cb and Cr in order.
//!
//! The approach (byte-weight `pmaddubsw`, horizontal adds, no planar
//! transpose) follows libyuv's `ARGBToYRow_AVX2` / `ARGBToUVRow_AVX2`
//! (BSD 3-clause, Copyright 2011 The `LibYuv` Project Authors; see
//! `THIRD_PARTY.md`); the arithmetic is our exact 8.8 form with exact 2x2
//! means, not libyuv's 7-bit weights and averaged pairs. The scalar row
//! functions take the tail columns and a lone last row.
//!
//! Kernel contract: see the `convert` module docs.
use super::{bt601_forward as matrix, scalar};
use core::arch::x86_64::{
    __m256i, _mm_storeu_si128, _mm256_add_epi16, _mm256_add_epi32, _mm256_and_si256,
    _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_hadd_epi16, _mm256_hadd_epi32,
    _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_packs_epi32,
    _mm256_packus_epi16, _mm256_permute4x64_epi64, _mm256_permutevar8x32_epi32, _mm256_set1_epi8,
    _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setr_epi8, _mm256_setr_epi32, _mm256_shuffle_epi8,
    _mm256_srai_epi32, _mm256_srli_epi16, _mm256_storeu_si256,
};

/// Pixels of each row per iteration (four 8-pixel vectors).
const STEP: usize = 32;

/// `ROUND + (offset << SHIFT)`, the starting value of a weighted sum.
const fn bias(offset: i32) -> i32 {
    matrix::ROUND + (offset << matrix::SHIFT)
}

/// The part of G's weight (129) that a signed byte cannot hold, added
/// separately as G's byte in place in its 16-bit word.
const GREEN_HIGH: i32 = 256;

/// Signed byte weights `[66, -127, 25, 0]`: R, G's weight less
/// `GREEN_HIGH`, B and alpha.
const fn luma_byte_weights() -> i32 {
    let [r, g, b] = matrix::Y;
    (r & 0xff) | (((g - GREEN_HIGH) & 0xff) << 8) | ((b & 0xff) << 16)
}
const _: () = assert!(matrix::Y[0] < 128 && matrix::Y[2] < 128);
const _: () = assert!(matrix::Y[1] - GREEN_HIGH >= -128 && matrix::Y[1] < GREEN_HIGH);

/// Luma sums (before rounding) of eight packed pixels, as the 16-bit pairs
/// `[66 R + 129 G, 25 B]` per pixel.
#[inline]
#[target_feature(enable = "avx2")]
fn luma_pairs(pixels: __m256i) -> __m256i {
    let partial = _mm256_maddubs_epi16(pixels, _mm256_set1_epi32(luma_byte_weights()));
    // The (R, G) word without R is 256 G.
    let green_high = _mm256_and_si256(pixels, _mm256_set1_epi32(0xff00));
    _mm256_add_epi16(partial, green_high)
}

/// Luma bytes of 32 pixels (four vectors of 8, in order).
#[inline]
#[target_feature(enable = "avx2")]
fn luma32(pixels: [__m256i; 4]) -> __m256i {
    let rounding = _mm256_set1_epi16(bias(matrix::LUMA_OFFSET) as i16);
    let [a, b, c, d] = pixels.map(|p| luma_pairs(p));
    // In-lane horizontal adds: [px 0-3, 8-11 | 4-7, 12-15] and likewise
    // for pixels 16-31.
    let low = _mm256_srli_epi16::<8>(_mm256_add_epi16(_mm256_hadd_epi16(a, b), rounding));
    let high = _mm256_srli_epi16::<8>(_mm256_add_epi16(_mm256_hadd_epi16(c, d), rounding));
    // Dwords [0-3, 8-11, 16-19, 24-27 | 4-7, 12-15, 20-23, 28-31].
    let bytes = _mm256_packus_epi16(low, high);
    _mm256_permutevar8x32_epi32(bytes, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7))
}

/// Channel sums of the horizontal pixel pairs of eight packed pixels:
/// 16-bit `[R, G, B, A]` per pair (four pairs).
#[inline]
#[target_feature(enable = "avx2")]
fn pair_sums(pixels: __m256i) -> __m256i {
    let by_channel = _mm256_setr_epi8(
        0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15, 0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13,
        10, 14, 11, 15,
    );
    _mm256_maddubs_epi16(_mm256_shuffle_epi8(pixels, by_channel), _mm256_set1_epi8(1))
}

/// Cb or Cr of 16 groups as 32-bit lanes, from the means of four vectors
/// of four groups: `[g0, g1, g4, g5 | g2, g3, g6, g7]` for the first two
/// vectors and likewise (`+ 8`) for the last two.
#[inline]
#[target_feature(enable = "avx2")]
fn chroma16(means: [__m256i; 4], [wr, wg, wb]: [i32; 3]) -> [__m256i; 2] {
    let weights = _mm256_setr_epi32(
        (wr & 0xffff) | (wg << 16),
        wb & 0xffff,
        (wr & 0xffff) | (wg << 16),
        wb & 0xffff,
        (wr & 0xffff) | (wg << 16),
        wb & 0xffff,
        (wr & 0xffff) | (wg << 16),
        wb & 0xffff,
    );
    let rounding = _mm256_set1_epi32(bias(matrix::CHROMA_OFFSET));
    let [a, b, c, d] = means.map(|m| _mm256_madd_epi16(m, weights));
    [_mm256_hadd_epi32(a, b), _mm256_hadd_epi32(c, d)]
        .map(|sum| _mm256_srai_epi32::<8>(_mm256_add_epi32(sum, rounding)))
}

/// Cb (low 128 bits) and Cr (high) bytes of the 16 groups of 32 pixels of
/// a row pair.
#[inline]
#[target_feature(enable = "avx2")]
fn chroma32(upper: [__m256i; 4], lower: [__m256i; 4]) -> __m256i {
    let two = _mm256_set1_epi16(2);
    let mut means = [two; 4];
    for (mean, (top, bottom)) in means.iter_mut().zip(upper.into_iter().zip(lower)) {
        let sum = _mm256_add_epi16(pair_sums(top), pair_sums(bottom));
        *mean = _mm256_srli_epi16::<2>(_mm256_add_epi16(sum, two));
    }
    let [cb_low, cb_high] = chroma16(means, matrix::CB);
    let [cr_low, cr_high] = chroma16(means, matrix::CR);
    // Words [g0 g1 g4 g5 g8 g9 g12 g13 | g2 g3 g6 g7 g10 g11 g14 g15].
    let cb = _mm256_packs_epi32(cb_low, cb_high);
    let cr = _mm256_packs_epi32(cr_low, cr_high);
    // Quadwords [Cb lane 0, Cr lane 0 | Cb lane 1, Cr lane 1] -> [Cb lane 0,
    // Cb lane 1 | Cr lane 0, Cr lane 1], then interleave the group pairs.
    let bytes = _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packus_epi16(cb, cr));
    let order = _mm256_setr_epi8(
        0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15, 0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12,
        13, 6, 7, 14, 15,
    );
    _mm256_shuffle_epi8(bytes, order)
}

/// Four vectors of 8 packed pixels from `src`.
///
/// # Safety
///
/// `src` must be valid for reading 128 bytes; the CPU must support AVX2.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load32(src: *const u8) -> [__m256i; 4] {
    // SAFETY: 4 * 32 = 128 readable bytes (caller contract).
    [0, 1, 2, 3].map(|i| unsafe { _mm256_loadu_si256(src.add(32 * i).cast()) })
}

/// # Safety
///
/// Kernel contract (`convert` module docs); the CPU must support AVX2.
#[allow(clippy::too_many_arguments, reason = "matches RgbaToI420Fn")]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn rgba_to_i420(
    rgba: &[u8],
    luma: &mut [u8],
    luma_stride: usize,
    cb: &mut [u8],
    cr: &mut [u8],
    chroma_stride: usize,
    width: usize,
    height: usize,
) {
    let vector_width = width - width % STEP;
    for row in 0..height / 2 {
        let rows = &rgba[8 * width * row..][..8 * width];
        let (top, bottom) = rows.split_at(4 * width);
        let (luma_top, luma_bottom) = luma[2 * row * luma_stride..].split_at_mut(luma_stride);
        let (luma_top, luma_bottom) = (&mut luma_top[..width], &mut luma_bottom[..width]);
        let cb_row = &mut cb[chroma_stride * row..][..width / 2];
        let cr_row = &mut cr[chroma_stride * row..][..width / 2];
        for x in (0..vector_width).step_by(STEP) {
            // SAFETY: x + 32 <= width: 128 RGBA bytes and 32 luma samples
            // of each row and 16 chroma samples are in bounds.
            unsafe {
                let upper = load32(top.as_ptr().add(4 * x));
                let lower = load32(bottom.as_ptr().add(4 * x));
                _mm256_storeu_si256(luma_top.as_mut_ptr().add(x).cast(), luma32(upper));
                _mm256_storeu_si256(luma_bottom.as_mut_ptr().add(x).cast(), luma32(lower));
                let chroma = chroma32(upper, lower);
                _mm_storeu_si128(
                    cb_row.as_mut_ptr().add(x / 2).cast(),
                    _mm256_castsi256_si128(chroma),
                );
                _mm_storeu_si128(
                    cr_row.as_mut_ptr().add(x / 2).cast(),
                    _mm256_extracti128_si256::<1>(chroma),
                );
            }
        }
        scalar::rgba_to_luma_row(&top[4 * vector_width..], &mut luma_top[vector_width..]);
        scalar::rgba_to_luma_row(
            &bottom[4 * vector_width..],
            &mut luma_bottom[vector_width..],
        );
        scalar::rgba_to_chroma_row(
            rows,
            vector_width / 2,
            &mut cb_row[vector_width / 2..],
            &mut cr_row[vector_width / 2..],
        );
    }
    if height % 2 == 1 {
        scalar::rgba_to_luma_row(
            &rgba[4 * width * (height - 1)..],
            &mut luma[(height - 1) * luma_stride..(height - 1) * luma_stride + width],
        );
    }
}
