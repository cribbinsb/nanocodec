//! Register-only H.264 4x4 inverse transform and predictor addition.
//! `OpenH264`'s fused blocks motivated the packing, with 32-bit butterflies
//! to accept the full scaled coefficient range.

#![allow(clippy::inline_always)]

use super::CodecTxfmFns;
use super::generic::{self as g, H264AddAc4, H264Idct4};
use crate::simd::avx2::{I32x4, I32x8, transpose_i32x4};
use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi16, _mm_add_epi32, _mm_cvtepi16_epi32, _mm_cvtepu8_epi16,
    _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_extract_epi32, _mm_insert_epi32, _mm_loadl_epi64,
    _mm_loadu_si128, _mm_packs_epi32, _mm_packus_epi16, _mm_set1_epi32, _mm_srai_epi32,
    _mm_srli_si128, _mm_storel_epi64, _mm_unpackhi_epi16, _mm_unpacklo_epi16, _mm256_add_epi32,
    _mm256_castsi256_si128, _mm256_cvtepi16_epi32, _mm256_extracti128_si256, _mm256_loadu_si256,
    _mm256_permute2x128_si256, _mm256_permute4x64_epi64, _mm256_set1_epi32, _mm256_srai_epi32,
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

/// # Safety
/// AVX2 is enabled and the destination covers a 4x4 block.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn add_single(coeffs: &[i16; 16], dst: &mut [u8], stride: usize) {
    // SAFETY: both loads read exactly eight of the sixteen coefficients.
    let (top, bottom) = unsafe {
        (
            _mm_loadu_si128(coeffs.as_ptr().cast()),
            _mm_loadu_si128(coeffs.as_ptr().add(8).cast()),
        )
    };
    let even = _mm_unpacklo_epi16(top, bottom);
    let odd = _mm_unpackhi_epi16(top, bottom);
    let lo = _mm_unpacklo_epi16(even, odd);
    let hi = _mm_unpackhi_epi16(even, odd);
    let columns = [
        _mm_cvtepi16_epi32(lo),
        _mm_cvtepi16_epi32(_mm_srli_si128::<8>(lo)),
        _mm_cvtepi16_epi32(hi),
        _mm_cvtepi16_epi32(_mm_srli_si128::<8>(hi)),
    ];
    let columns = H264Idct4::butterfly(columns.map(I32x4));
    let rows = H264Idct4::butterfly(transpose_i32x4(&columns));

    let dst = dst.as_mut_ptr();
    for (r, [upper, lower]) in rows.as_chunks::<2>().0.iter().enumerate() {
        // SAFETY: the destination holds four readable and writable bytes
        // in each row. Byte arrays have alignment one.
        unsafe {
            let p = dst.add(2 * r * stride).cast::<[u8; 4]>();
            let q = dst.add((2 * r + 1) * stride).cast::<[u8; 4]>();
            let packed = _mm_insert_epi32::<1>(
                _mm_cvtsi32_si128(i32::from_le_bytes(p.read())),
                i32::from_le_bytes(q.read()),
            );
            let pred = _mm_cvtepu8_epi16(packed);
            // Each pass grows magnitudes by at most four. After rounding
            // and shifting by six the predictor sum fits in i16.
            let delta = _mm_packs_epi32(round_shift(upper.0), round_shift(lower.0));
            let sum = _mm_add_epi16(pred, delta);
            let pixels = _mm_packus_epi16(sum, sum);
            p.write(_mm_cvtsi128_si32(pixels).to_le_bytes());
            q.write(_mm_extract_epi32::<1>(pixels).to_le_bytes());
        }
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
