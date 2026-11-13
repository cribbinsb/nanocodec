//! Full-pel puts, equal-weight averages, and width-32 HEVC fractional MC.
//! Fractional HEVC uses the shared filter graph on thirty-two word lanes;
//! all other fractional prediction retains AVX2. No reference rules change.
use super::{Filter, McFns};
use core::arch::x86_64::{
    _mm256_mask_storeu_epi8, _mm512_add_epi16, _mm512_cvtusepi16_epi8, _mm512_maskz_loadu_epi16,
    _mm512_max_epi16, _mm512_set1_epi16, _mm512_setzero_si512, _mm512_srai_epi16,
};
pub(crate) fn install(table: &mut McFns) {
    table.av1[4] = av164;
    table.av1_smooth[4] = av1_smooth64;
    table.av1_sharp[4] = av1_sharp64;
    table.hevc_luma[3] = hevc_luma32;
    table.hevc_luma[4] = hevc_luma64;
    table.hevc_chroma[4] = hevc_chroma64;
    table.h264_luma[4] = h264_luma64;
    table.h264_chroma[4] = h264_chroma64;
    table.vvc_luma[4] = vvc_luma64;
    table.vvc_affine_luma[4] = vvc_affine_luma64;
    table.vvc_chroma[4] = vvc_chroma64;
    table.compound.average[2..].copy_from_slice(&[average::<16>, average::<32>, average::<64>]);
}
/// One complete 32-pixel row per vector, including short-height blocks.
/// Whole-pel copies retain their measured AVX2 path. Other codecs/widths
/// stay unchanged until the same widening has its own performance evidence.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn hevc_luma32(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    h: usize,
    fx: usize,
    fy: usize,
) {
    if fx == 0 && fy == 0 {
        // SAFETY: same checked window; this tier includes AVX2.
        unsafe {
            super::avx2::separable_w32::<super::HevcLumaTaps, 8>(src, ss, dst, ds, h, fx, fy);
        };
        return;
    }
    // SAFETY: same checked filter window and enabled ISA as this wrapper.
    unsafe { hevc_luma32_fractional(src, ss, dst, ds, h, fx, fy) }
}

// Keep the fractional stack frame out of the whole-pel routing path.
#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn hevc_luma32_fractional(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    h: usize,
    fx: usize,
    fy: usize,
) {
    let taps = <super::HevcLumaTaps as super::SeparableTaps<8>>::TABLE;
    // SAFETY: Dsp checked the exact 32-wide filter window and output, and
    // the wrapper enables all features needed by the private lane type.
    unsafe {
        super::generic::hevc::<crate::simd::avx512_i16::I16x32, 8>(
            src,
            ss,
            dst,
            ds,
            32,
            h,
            &taps[fx],
            &taps[fy],
            (fx, fy),
        );
    }
}

macro_rules! put {
    ($name:ident, $width:literal, $filter:ident, $fallback:path) => {
        #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
        unsafe fn $name(
            src: &[u8],
            ss: usize,
            dst: &mut [u8],
            ds: usize,
            h: usize,
            fx: usize,
            fy: usize,
        ) {
            if fx == 0 && fy == 0 {
                let before = Filter::$filter.before();
                // SAFETY: the validated filter window contains this whole-pel block.
                unsafe {
                    crate::block_copy::avx512::copy(
                        &src[before * ss + before..],
                        ss,
                        dst,
                        ds,
                        $width,
                        h,
                    );
                }
            } else {
                // SAFETY: same window contract; AVX2 is included in this tier.
                unsafe {
                    $fallback(src, ss, dst, ds, h, fx, fy);
                }
            }
        }
    };
}
put!(av164, 64, Av1, super::av1_avx2::put::<super::Regular, 64>);
put!(
    av1_smooth64,
    64,
    Av1Smooth,
    super::av1_avx2::put::<super::Smooth, 64>
);
put!(
    av1_sharp64,
    64,
    Av1Sharp,
    super::av1_avx2::put::<super::Sharp, 64>
);
put!(
    hevc_luma64,
    64,
    HevcLuma,
    super::avx2::separable_w64::<super::HevcLumaTaps, 8>
);
put!(
    hevc_chroma64,
    64,
    HevcChroma,
    super::avx2::separable_w64::<super::HevcChromaTaps, 4>
);
put!(h264_luma64, 64, H264Luma, super::avx2::h264_luma_w64);
put!(h264_chroma64, 64, H264Chroma, super::avx2::h264_chroma_w64);
put!(
    vvc_luma64,
    64,
    VvcLuma,
    super::avx2::separable_w64::<super::VvcLumaTaps, 8>
);
put!(
    vvc_affine_luma64,
    64,
    VvcAffineLuma,
    super::avx2::separable_w64::<super::VvcAffineLumaTaps, 8>
);
put!(
    vvc_chroma64,
    64,
    VvcChroma,
    super::avx2::separable_w64::<super::VvcChromaTaps, 4>
);

/// Same compound precision and wrapping-word contract as the AVX2 driver:
/// round (p0 + p1) by five bits, then clip to pixels.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn average<const W: usize>(
    a: &[i16],
    b: &[i16],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    h: usize,
) {
    for y in 0..h {
        for x in (0..W).step_by(32) {
            let mask = u32::MAX >> (32 - (W - x).min(32));
            // SAFETY: the mask covers only validated words in both rows.
            let (a, b) = unsafe {
                (
                    _mm512_maskz_loadu_epi16(mask, a.as_ptr().add(y * ss + x).cast()),
                    _mm512_maskz_loadu_epi16(mask, b.as_ptr().add(y * ss + x).cast()),
                )
            };
            let sum = _mm512_add_epi16(_mm512_add_epi16(a, b), _mm512_set1_epi16(16));
            let rounded = _mm512_max_epi16(_mm512_srai_epi16::<5>(sum), _mm512_setzero_si512());
            // SAFETY: the same pixel mask bounds this destination row.
            unsafe {
                _mm256_mask_storeu_epi8(
                    dst.as_mut_ptr().add(y * ds + x).cast(),
                    mask,
                    _mm512_cvtusepi16_epi8(rounded),
                );
            }
        }
    }
}
