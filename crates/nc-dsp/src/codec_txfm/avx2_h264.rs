//! H.264 8x8 pixel subtraction and four 4x4 forward transforms.
//! The packed low-bit-depth schedule follows `OpenH264`
//! `WelsDctFourT4_avx2` (BSD-2-Clause, see `THIRD_PARTY.md`).

use super::generic::{H264Fdct8Paired, Line};
use super::{CodecTxfmFns, H264Quantizer, QUANT_MF_8X8_POS};
use crate::simd::I32Lanes;
use crate::simd::avx2::{I32x8, transpose_i32x8};
use core::arch::x86_64::{
    __m256i, _mm_cvtsi32_si128, _mm_loadl_epi64, _mm_storeu_si128, _mm256_abs_epi16,
    _mm256_add_epi16, _mm256_blend_epi32, _mm256_broadcastq_epi64, _mm256_castsi128_si256,
    _mm256_castsi256_si128, _mm256_cmpeq_epi16, _mm256_cmpgt_epi16, _mm256_extracti128_si256,
    _mm256_loadu_si256, _mm256_madd_epi16, _mm256_mulhi_epu16, _mm256_mullo_epi16,
    _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setr_epi16, _mm256_setzero_si256,
    _mm256_shuffle_epi8, _mm256_shuffle_epi32, _mm256_sign_epi16, _mm256_slli_epi16,
    _mm256_srl_epi16, _mm256_storeu_si256, _mm256_sub_epi16, _mm256_unpackhi_epi64,
    _mm256_unpacklo_epi64, _mm256_xor_si256,
};
use nc_common::PlaneView;

/// Transform the four 4x4 blocks of an 8x8 pixel difference.
///
/// # Safety
/// AVX2 is enabled and both views cover 8x8.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn h264_forward_chroma_pixels(
    src: &PlaneView<'_>,
    pred: &PlaneView<'_>,
    out: &mut [[i16; 16]; 4],
) {
    let [top, bottom] = <&mut [_; 2]>::try_from(out.as_chunks_mut::<2>().0).expect("four blocks");
    // SAFETY: both views cover 8x8 and AVX2 is enabled.
    unsafe {
        forward_four_pixels(
            (src.data().as_ptr(), src.stride()),
            (pred.data().as_ptr(), pred.stride()),
            [top, bottom],
        );
    }
}

/// Subtract and transform the luma macroblock's sixteen 4x4 blocks.
///
/// # Safety
/// Both views cover 16x16 and AVX2 is enabled.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn h264_forward_luma_pixels(
    src: &PlaneView<'_>,
    pred: &PlaneView<'_>,
    out: &mut [[i16; 16]; 16],
) {
    // SAFETY: forwarded ISA and view contracts.
    unsafe { super::generic::h264_forward_luma_pixels(src, pred, out, forward_four_pixels) };
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn forward_four_pixels(
    (s, ss): (*const u8, usize),
    (p, ps): (*const u8, usize),
    [top, bottom]: [&mut [[i16; 16]; 2]; 2],
) {
    // Expand bytes to words in butterfly order (0, 3, 1, 2) for each block.
    const EXTEND: [i8; 32] = [
        0, -128, 3, -128, 1, -128, 2, -128, 4, -128, 7, -128, 5, -128, 6, -128, 0, -128, 3, -128,
        1, -128, 2, -128, 4, -128, 7, -128, 5, -128, 6, -128,
    ];
    const SWAP_PAIRS: [i8; 32] = [
        2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9,
        14, 15, 12, 13,
    ];
    // SAFETY: each constant contains one full vector.
    let extend = unsafe { _mm256_loadu_si256(EXTEND.as_ptr().cast()) };
    // SAFETY: likewise for the swap indices.
    let swap = unsafe { _mm256_loadu_si256(SWAP_PAIRS.as_ptr().cast()) };
    let signs = _mm256_set1_epi32(-65_535); // Repeated words (1, -1).
    let factors = _mm256_setr_epi16(1, 2, -1, -2, 1, 2, -1, -2, 1, 2, -1, -2, 1, 2, -1, -2);
    // SAFETY: the two views contain all eight rows. Each load reads eight bytes.
    let rows = core::array::from_fn(|r| unsafe {
        _mm256_sub_epi16(
            load_paired_rows(s, ss, r, extend),
            load_paired_rows(p, ps, r, extend),
        )
    });
    // The vertical butterfly operates on rows r and r+4 in separate halves.
    // The horizontal butterfly pairs neighbours in each reordered group.
    let coeffs = vertical(rows).map(|row| {
        let a = _mm256_sign_epi16(row, signs);
        let b = _mm256_add_epi16(a, _mm256_shuffle_epi8(row, swap));
        _mm256_add_epi16(
            _mm256_shuffle_epi32::<0xb1>(b),
            _mm256_mullo_epi16(b, factors),
        )
    });
    for (i, [a, b]) in coeffs.as_chunks::<2>().0.iter().enumerate() {
        let left = _mm256_unpacklo_epi64(*a, *b);
        let right = _mm256_unpackhi_epi64(*a, *b);
        // SAFETY: each store fills eight coefficients of a complete block.
        unsafe {
            _mm_storeu_si128(
                top[0].as_mut_ptr().add(8 * i).cast(),
                _mm256_castsi256_si128(left),
            );
            _mm_storeu_si128(
                top[1].as_mut_ptr().add(8 * i).cast(),
                _mm256_castsi256_si128(right),
            );
            _mm_storeu_si128(
                bottom[0].as_mut_ptr().add(8 * i).cast(),
                _mm256_extracti128_si256::<1>(left),
            );
            _mm_storeu_si128(
                bottom[1].as_mut_ptr().add(8 * i).cast(),
                _mm256_extracti128_si256::<1>(right),
            );
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_paired_rows(base: *const u8, stride: usize, r: usize, mask: __m256i) -> __m256i {
    // SAFETY: the caller validated all eight rows. The loads are unaligned.
    let top = unsafe { _mm_loadl_epi64(base.add(r * stride).cast()) };
    // SAFETY: the same footprint in the lower four rows.
    let bottom =
        _mm256_broadcastq_epi64(unsafe { _mm_loadl_epi64(base.add((r + 4) * stride).cast()) });
    _mm256_shuffle_epi8(
        _mm256_blend_epi32::<0xf0>(_mm256_castsi128_si256(top), bottom),
        mask,
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn vertical([a, b, c, d]: [__m256i; 4]) -> [__m256i; 4] {
    let (s03, d03) = (_mm256_add_epi16(a, d), _mm256_sub_epi16(a, d));
    let (s12, d12) = (_mm256_add_epi16(b, c), _mm256_sub_epi16(b, c));
    [
        _mm256_add_epi16(s03, s12),
        _mm256_add_epi16(_mm256_slli_epi16::<1>(d03), d12),
        _mm256_sub_epi16(s03, s12),
        _mm256_sub_epi16(d03, _mm256_slli_epi16::<1>(d12)),
    ]
}

/// Transform and quantise the 8x8 pixel difference for chroma coding.
///
/// # Safety
/// As for [`h264_forward_chroma_pixels`].
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn h264_forward_quantize_chroma_pixels(
    src: &PlaneView<'_>,
    pred: &PlaneView<'_>,
    q: &H264Quantizer,
    ac: &mut [[i16; 16]; 4],
) -> ([i16; 4], u32) {
    // SAFETY: forwarded view and ISA contract.
    unsafe {
        h264_forward_chroma_pixels(src, pred, ac);
        super::generic::h264_quantize_chroma::<I32x8>(q, ac)
    }
}

/// Install the same paired-column transform for individual blocks and the
/// shared macroblock quantizer.
pub(super) fn install(table: &mut CodecTxfmFns) {
    table.h264_forward_8x8 = h264_forward_8x8;
    table.h264_forward_quantize_mb = h264_forward_quantize_mb;
}

const QUANT_WORDS_8X8: [[i16; 64]; 6] = {
    let mut out = [[0; 64]; 6];
    let mut q = 0;
    while q < 6 {
        let mut i = 0;
        while i < 64 {
            let factor = QUANT_MF_8X8_POS[q][i];
            assert!(factor <= i16::MAX as i32);
            out[q][i] = factor as i16;
            i += 1;
        }
        q += 1;
    }
    out
};

/// `OpenH264`'s `WelsQuantFour4x4_avx2` keeps 16 coefficients in words.
/// Preserve this encoder's post-multiply offset by carrying from the low
/// product explicitly before shifting the high product. At 8×8 precision
/// the result magnitude fits a signed word for every i16 coefficient.
#[target_feature(enable = "avx2")]
unsafe fn quantize8_words(block: &mut [i16; 64], q: &H264Quantizer) -> u32 {
    let (qbits, offset) = q.forward_terms(1);
    let shift = _mm_cvtsi32_si128(qbits as i32 - 16);
    let rounding = _mm256_set1_epi16((offset >> 16) as i16);
    let unsigned = _mm256_set1_epi16(i16::MIN);
    let threshold = _mm256_set1_epi16(i16::MAX.wrapping_sub(offset as i16));
    let zero = _mm256_setzero_si256();
    let mut zeros = zero;
    for (coefficients, factors) in block
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(QUANT_WORDS_8X8[q.rem].as_chunks::<16>().0)
    {
        // SAFETY: both chunks are one full vector.
        let (raw, scale) = unsafe {
            (
                _mm256_loadu_si256(coefficients.as_ptr().cast()),
                _mm256_loadu_si256(factors.as_ptr().cast()),
            )
        };
        let magnitude = _mm256_abs_epi16(raw);
        let low = _mm256_mullo_epi16(magnitude, scale);
        let high = _mm256_mulhi_epu16(magnitude, scale);
        let carry = _mm256_cmpgt_epi16(_mm256_xor_si256(low, unsigned), threshold);
        let rounded = _mm256_sub_epi16(_mm256_add_epi16(high, rounding), carry);
        let level = _mm256_sign_epi16(_mm256_srl_epi16(rounded, shift), raw);
        zeros = _mm256_sub_epi16(zeros, _mm256_cmpeq_epi16(level, zero));
        // SAFETY: the chunk has sixteen writable words.
        unsafe { _mm256_storeu_si256(coefficients.as_mut_ptr().cast(), level) };
    }
    let zeros = _mm256_madd_epi16(zeros, _mm256_set1_epi16(1));
    64 - crate::simd::avx2::sum_i32x8(zeros) as u32
}

/// Exact H.264 forward at the ordinary encoder scale. Each tile is
/// transposed in registers. The odd terms use the same signed-word
/// products as the AVX-512 macroblock path.
///
/// # Safety
/// AVX2 is enabled and `residual` covers 8x8 samples at `stride`.
/// Residuals are differences of two 8-bit samples.
#[inline(never)]
#[target_feature(enable = "avx2")]
unsafe fn h264_forward_8x8(residual: &[i16], stride: usize, out: &mut [i16; 64]) {
    let mut rows = [I32x8::splat(0); 8];
    for (r, row) in rows.iter_mut().enumerate() {
        // SAFETY: each of the eight rows holds eight checked samples.
        let input = unsafe { core::slice::from_raw_parts(residual.as_ptr().add(r * stride), 8) };
        *row = I32x8::load_i16(input);
    }
    let columns = transpose_i32x8(&rows);
    let mut coefficients = [I32x8::splat(0); 8];
    H264Fdct8Paired::apply(&columns, &mut coefficients);
    let rows = transpose_i32x8(&coefficients);
    H264Fdct8Paired::apply(&rows, &mut coefficients);
    for (row, dst) in coefficients.into_iter().zip(out.as_chunks_mut::<8>().0) {
        row.round_shift(6).store_sat16(dst);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn h264_forward_quantize_mb(
    residual: &[i16],
    stride: usize,
    zero_mask: u8,
    q: &H264Quantizer,
    out: &mut [[i16; 64]; 4],
) -> [u32; 4] {
    for (i, block) in out.iter_mut().enumerate() {
        if zero_mask & (1 << i) != 0 {
            block.fill(0);
        } else {
            let start = (i / 2) * 8 * stride + (i % 2) * 8;
            // SAFETY: the caller validates all four macroblock quadrants.
            unsafe { h264_forward_8x8(&residual[start..], stride, block) };
        }
    }
    let mut counts = [0; 4];
    // Keep quantiser constants live across blocks without any transform call.
    for (i, block) in out.iter_mut().enumerate() {
        if zero_mask & (1 << i) == 0 {
            // SAFETY: AVX2 is enabled here.
            counts[i] = unsafe { quantize8_words(block, q) };
        }
    }
    counts
}
