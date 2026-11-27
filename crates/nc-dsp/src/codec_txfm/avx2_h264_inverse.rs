//! Register-only H.264 4x4 inverse transform and predictor addition.
//! `OpenH264`'s fused blocks motivated the packing, with 32-bit butterflies
//! to accept the full scaled coefficient range.

#![allow(clippy::inline_always)]

use super::CodecTxfmFns;
use super::generic::{self as g, H264AddAc4, H264Idct4};
use crate::simd::avx2::{I32x4, I32x8};
use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi16, _mm_add_epi32, _mm_cvtepu8_epi16, _mm_cvtsi32_si128,
    _mm_cvtsi128_si32, _mm_extract_epi32, _mm_insert_epi32, _mm_loadl_epi64, _mm_loadu_si128,
    _mm_packs_epi32, _mm_packus_epi16, _mm_set1_epi32, _mm_srai_epi32, _mm_storel_epi64,
    _mm256_add_epi16, _mm256_add_epi32, _mm256_blend_epi32, _mm256_castsi256_si128,
    _mm256_cvtepi16_epi32, _mm256_cvtepu8_epi16, _mm256_extracti128_si256, _mm256_loadu_si256,
    _mm256_permute2x128_si256, _mm256_permute4x64_epi64, _mm256_set_m128i, _mm256_set1_epi32,
    _mm256_setr_epi32, _mm256_shuffle_epi32, _mm256_sign_epi32, _mm256_srai_epi32,
    _mm256_unpackhi_epi16, _mm256_unpackhi_epi32, _mm256_unpackhi_epi64, _mm256_unpacklo_epi16,
    _mm256_unpacklo_epi32, _mm256_unpacklo_epi64,
};

const KERNELS: H264AddAc4 = H264AddAc4 {
    single: add_single,
    pair: Some(add_pair),
};

pub(super) fn install(table: &mut CodecTxfmFns) {
    table.h264_add_inverse_4x4 = add_inverse;
    table.h264_add_scaled_4x4 = add_scaled;
    table.h264_add_scaled_mb = add_macroblock;
}

#[target_feature(enable = "avx2")]
unsafe fn add_inverse(
    levels: &[[i16; 16]],
    dc: &[i16],
    scale: (&[i32; 16], i32),
    blocks_wide: usize,
    dst: &mut [u8],
    stride: usize,
) {
    // SAFETY: dispatch enables AVX2 and validates the destination region.
    unsafe {
        g::h264_add_inverse_4x4::<I32x4, I32x8>(
            levels,
            dc,
            scale,
            blocks_wide,
            dst,
            stride,
            KERNELS,
        );
    }
}

#[target_feature(enable = "avx2")]
unsafe fn add_scaled(coeffs: &[[i16; 16]], blocks_wide: usize, dst: &mut [u8], stride: usize) {
    // SAFETY: dispatch enables AVX2 and validates the destination region.
    unsafe { g::h264_add_scaled_4x4::<I32x4>(coeffs, blocks_wide, dst, stride, KERNELS) };
}

#[target_feature(enable = "avx2")]
unsafe fn add_macroblock(
    coeffs: &[[i16; 16]; 16],
    masks: (u16, u16),
    dst: &mut [u8],
    stride: usize,
) {
    // SAFETY: dispatch enables AVX2 and validates the destination region.
    unsafe { g::h264_add_scaled_mb::<I32x4>(coeffs, masks, dst, stride, KERNELS) };
}

#[inline]
#[target_feature(enable = "avx2")]
fn round_shift(v: __m128i) -> __m128i {
    _mm_srai_epi32::<6>(_mm_add_epi32(v, _mm_set1_epi32(32)))
}

/// Apply the horizontal H.264 inverse butterfly to two rows in their
/// natural order. Each 128-bit lane holds one row throughout the pass.
#[inline]
#[target_feature(enable = "avx2")]
fn inverse_rows(input: __m256i) -> __m256i {
    let shifted = _mm256_blend_epi32::<0xaa>(input, _mm256_srai_epi32::<1>(input));
    let signs = _mm256_setr_epi32(1, 1, -1, -1, 1, 1, -1, -1);
    // In each lane the four terms are c0+c2, c1+(c3>>1),
    // c0-c2 and (c1>>1)-c3, in that order.
    let terms = _mm256_add_epi32(
        _mm256_sign_epi32(input, signs),
        _mm256_shuffle_epi32::<0x4e>(shifted),
    );
    let signs = _mm256_setr_epi32(1, -1, 1, -1, 1, -1, 1, -1);
    let outputs = _mm256_add_epi32(
        _mm256_sign_epi32(terms, signs),
        _mm256_shuffle_epi32::<0xb1>(terms),
    );
    _mm256_shuffle_epi32::<0x78>(outputs)
}

/// # Safety
/// AVX2 is enabled and the destination covers a 4x4 block.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn add_single(coeffs: &[i16; 16], dst: &mut [u8], stride: usize) {
    // SAFETY: both loads read exactly eight of the sixteen coefficients.
    let (top, bottom) = unsafe {
        (
            inverse_rows(_mm256_cvtepi16_epi32(_mm_loadu_si128(
                coeffs.as_ptr().cast(),
            ))),
            inverse_rows(_mm256_cvtepi16_epi32(_mm_loadu_si128(
                coeffs.as_ptr().add(8).cast(),
            ))),
        )
    };
    let [a, b, c, d] = H264Idct4::butterfly(
        [
            _mm256_castsi256_si128(top),
            _mm256_extracti128_si256::<1>(top),
            _mm256_castsi256_si128(bottom),
            _mm256_extracti128_si256::<1>(bottom),
        ]
        .map(I32x4),
    );
    // Each pass grows magnitudes by at most four, so the rounded
    // residual plus the predictor fits in signed words.
    let delta = _mm256_set_m128i(
        _mm_packs_epi32(round_shift(c.0), round_shift(d.0)),
        _mm_packs_epi32(round_shift(a.0), round_shift(b.0)),
    );
    let dst = dst.as_mut_ptr();
    // SAFETY: all four rows hold four readable and writable bytes.
    unsafe {
        let p0 = dst.cast::<[u8; 4]>();
        let p1 = dst.add(stride).cast::<[u8; 4]>();
        let p2 = dst.add(2 * stride).cast::<[u8; 4]>();
        let p3 = dst.add(3 * stride).cast::<[u8; 4]>();
        let pred = _mm_cvtsi32_si128(i32::from_le_bytes(p0.read()));
        let pred = _mm_insert_epi32::<1>(pred, i32::from_le_bytes(p1.read()));
        let pred = _mm_insert_epi32::<2>(pred, i32::from_le_bytes(p2.read()));
        let pred = _mm_insert_epi32::<3>(pred, i32::from_le_bytes(p3.read()));
        let sum = _mm256_add_epi16(_mm256_cvtepu8_epi16(pred), delta);
        let pixels = _mm_packus_epi16(
            _mm256_castsi256_si128(sum),
            _mm256_extracti128_si256::<1>(sum),
        );
        p0.write(_mm_cvtsi128_si32(pixels).to_le_bytes());
        p1.write(_mm_extract_epi32::<1>(pixels).to_le_bytes());
        p2.write(_mm_extract_epi32::<2>(pixels).to_le_bytes());
        p3.write(_mm_extract_epi32::<3>(pixels).to_le_bytes());
    }
}

/// Independent 4x4 dword transposes in the 128-bit lanes.
#[inline]
#[target_feature(enable = "avx2")]
fn transpose_pair([a, b, c, d]: [I32x8; 4]) -> [I32x8; 4] {
    let lo01 = _mm256_unpacklo_epi32(a.0, b.0);
    let hi01 = _mm256_unpackhi_epi32(a.0, b.0);
    let lo23 = _mm256_unpacklo_epi32(c.0, d.0);
    let hi23 = _mm256_unpackhi_epi32(c.0, d.0);
    [
        _mm256_unpacklo_epi64(lo01, lo23),
        _mm256_unpackhi_epi64(lo01, lo23),
        _mm256_unpacklo_epi64(hi01, hi23),
        _mm256_unpackhi_epi64(hi01, hi23),
    ]
    .map(I32x8)
}

/// # Safety
/// AVX2 is enabled and the destination covers the two adjacent blocks.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn add_pair([left, right]: [&[i16; 16]; 2], dst: &mut [u8], stride: usize) {
    // SAFETY: these loads each consume one complete block.
    let (l, r) = unsafe {
        (
            _mm256_loadu_si256(left.as_ptr().cast()),
            _mm256_loadu_si256(right.as_ptr().cast()),
        )
    };
    // Keep the left and right blocks in independent 128-bit lanes.
    let top = _mm256_permute2x128_si256::<0x20>(l, r);
    let bottom = _mm256_permute2x128_si256::<0x31>(l, r);
    let even = _mm256_unpacklo_epi16(top, bottom);
    let odd = _mm256_unpackhi_epi16(top, bottom);
    let lo = _mm256_unpacklo_epi16(even, odd);
    let hi = _mm256_unpackhi_epi16(even, odd);
    let widen = |columns: __m256i| {
        let grouped = _mm256_permute4x64_epi64::<0xd8>(columns);
        [
            I32x8(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(grouped))),
            I32x8(_mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(
                grouped,
            ))),
        ]
    };
    let [c0, c1] = widen(lo);
    let [c2, c3] = widen(hi);
    let columns = H264Idct4::butterfly([c0, c1, c2, c3]);
    let rows = H264Idct4::butterfly(transpose_pair(columns));
    let dst = dst.as_mut_ptr();
    for (r, row) in rows.into_iter().enumerate() {
        let rounded = _mm256_srai_epi32::<6>(_mm256_add_epi32(row.0, _mm256_set1_epi32(32)));
        let delta = _mm_packs_epi32(
            _mm256_castsi256_si128(rounded),
            _mm256_extracti128_si256::<1>(rounded),
        );
        // SAFETY: each checked row of the two blocks has eight bytes.
        unsafe {
            let p = dst.add(r * stride).cast();
            let pred = _mm_cvtepu8_epi16(_mm_loadl_epi64(p));
            let sum = _mm_add_epi16(pred, delta);
            _mm_storel_epi64(p, _mm_packus_epi16(sum, sum));
        }
    }
}
