//! The AVX-512 prediction table: AV1, HEVC and VVC blocks 32 and 64 wide
//! on the shared passes' 32-output rows (`av1_avx512`), whole-pel puts of
//! 64-wide blocks, AV1 narrow two-axis filtering and equal-weight averages.
//! No reference rules change.
use super::av1_avx2;
use super::av1_avx512::{Avx512Narrow, Avx512Wide};
use super::{
    Av1Family, Filter, HevcChromaTaps, HevcLumaTaps, McFns, Regular, SeparableTaps, Sharp, Smooth,
    TapPair, VvcAffineLumaTaps, VvcChromaTaps, VvcLumaTaps,
};
use core::arch::x86_64::{
    _mm256_mask_storeu_epi8, _mm512_add_epi16, _mm512_cvtusepi16_epi8, _mm512_maskz_loadu_epi16,
    _mm512_max_epi16, _mm512_set1_epi16, _mm512_setzero_si512, _mm512_srai_epi16,
};

/// Width classes (`BlockSize::width_class`).
const W32: usize = 3;
const W64: usize = 4;

pub(crate) fn install(table: &mut McFns) {
    table.vp9 = crate::vp9_mc_simd::families(crate::Isa::Avx512);
    table.av1[..3].copy_from_slice(&[
        av1_put::<Regular, 4>,
        av1_put::<Regular, 8>,
        av1_put::<Regular, 16>,
    ]);
    table.av1_smooth[..3].copy_from_slice(&[
        av1_put::<Smooth, 4>,
        av1_put::<Smooth, 8>,
        av1_put::<Smooth, 16>,
    ]);
    table.av1_sharp[..3].copy_from_slice(&[
        av1_put::<Sharp, 4>,
        av1_put::<Sharp, 8>,
        av1_put::<Sharp, 16>,
    ]);
    for (prep, family) in table.compound.prep.iter_mut().zip([
        [
            av1_prep::<Regular, 4>,
            av1_prep::<Regular, 8>,
            av1_prep::<Regular, 16>,
        ],
        [
            av1_prep::<Smooth, 4>,
            av1_prep::<Smooth, 8>,
            av1_prep::<Smooth, 16>,
        ],
        [
            av1_prep::<Sharp, 4>,
            av1_prep::<Sharp, 8>,
            av1_prep::<Sharp, 16>,
        ],
    ]) {
        prep[..3].copy_from_slice(&family);
    }
    table.av1[W32] = av1_put::<Regular, 32>;
    table.av1_smooth[W32] = av1_put::<Smooth, 32>;
    table.av1_sharp[W32] = av1_put::<Sharp, 32>;
    table.av1[W64] = av164;
    table.av1_smooth[W64] = av1_smooth64;
    table.av1_sharp[W64] = av1_sharp64;
    for (prep, family) in table.compound.prep.iter_mut().zip([
        [av1_prep::<Regular, 32>, av1_prep::<Regular, 64>],
        [av1_prep::<Smooth, 32>, av1_prep::<Smooth, 64>],
        [av1_prep::<Sharp, 32>, av1_prep::<Sharp, 64>],
    ]) {
        prep[W32..].copy_from_slice(&family);
    }
    if let Some(mixed) = table.av1_mixed.as_mut() {
        mixed.put[..3].copy_from_slice(&[
            av1_put_mixed::<4>,
            av1_put_mixed::<8>,
            av1_put_mixed::<16>,
        ]);
        mixed.prep[..3].copy_from_slice(&[
            av1_prep_mixed::<4>,
            av1_prep_mixed::<8>,
            av1_prep_mixed::<16>,
        ]);
        mixed.put[W32..].copy_from_slice(&[av1_put_mixed::<32>, av1_put_mixed::<64>]);
        mixed.prep[W32..].copy_from_slice(&[av1_prep_mixed::<32>, av1_prep_mixed::<64>]);
    }
    table.hevc_luma[W32] = separable::<HevcLumaTaps, 8, 32>;
    table.hevc_chroma[W32] = separable::<HevcChromaTaps, 4, 32>;
    table.vvc_luma[W32] = separable::<VvcLumaTaps, 8, 32>;
    table.vvc_affine_luma[W32] = separable::<VvcAffineLumaTaps, 8, 32>;
    table.vvc_chroma[W32] = separable::<VvcChromaTaps, 4, 32>;
    table.hevc_luma[W64] = hevc_luma64;
    table.hevc_chroma[W64] = hevc_chroma64;
    table.h264_luma[W64] = h264_luma64;
    table.h264_chroma[W64] = h264_chroma64;
    table.vvc_luma[W64] = vvc_luma64;
    table.vvc_affine_luma[W64] = vvc_affine_luma64;
    table.vvc_chroma[W64] = vvc_chroma64;
    table.compound.average[2..].copy_from_slice(&[average::<16>, average::<32>, average::<64>]);
}

/// AV1 `put` of family `F` at width `W` on Ice Lake.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
unsafe fn av1_put<F: Av1Family, const W: usize>(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    h: usize,
    fx: usize,
    fy: usize,
) {
    // SAFETY: the `mc` contract is the caller's; this tier's ISA is enabled.
    unsafe {
        if W <= 16 {
            av1_avx2::put::<F, W, Avx512Narrow>(src, ss, dst, ds, h, fx, fy);
        } else {
            av1_avx2::put::<F, W, Avx512Wide>(src, ss, dst, ds, h, fx, fy);
        }
    }
}

/// AV1 `prep` of family `F` at width `W` on Ice Lake.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
unsafe fn av1_prep<F: Av1Family, const W: usize>(
    src: &[u8],
    ss: usize,
    dst: &mut [i16],
    ds: usize,
    h: usize,
    fx: usize,
    fy: usize,
) {
    // SAFETY: as for `av1_put`.
    unsafe {
        if W <= 16 {
            av1_avx2::prep::<F, W, Avx512Narrow>(src, ss, dst, ds, h, fx, fy);
        } else {
            av1_avx2::prep::<F, W, Avx512Wide>(src, ss, dst, ds, h, fx, fy);
        }
    }
}

/// AV1 `put` with mixed families (`mc::Av1MixedFn`) on Ice Lake.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
unsafe fn av1_put_mixed<const W: usize>(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    h: usize,
    fracs: (usize, usize),
    taps: TapPair<'_>,
) {
    // SAFETY: as for `av1_put`.
    unsafe {
        if W <= 16 {
            av1_avx2::put_mixed::<W, Avx512Narrow>(src, ss, dst, ds, h, fracs, taps);
        } else {
            av1_avx2::put_mixed::<W, Avx512Wide>(src, ss, dst, ds, h, fracs, taps);
        }
    }
}

/// AV1 `prep` with mixed families (`mc::Av1MixedFn`) on Ice Lake.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
unsafe fn av1_prep_mixed<const W: usize>(
    src: &[u8],
    ss: usize,
    dst: &mut [i16],
    ds: usize,
    h: usize,
    fracs: (usize, usize),
    taps: TapPair<'_>,
) {
    // SAFETY: as for `av1_put`.
    unsafe {
        if W <= 16 {
            av1_avx2::prep_mixed::<W, Avx512Narrow>(src, ss, dst, ds, h, fracs, taps);
        } else {
            av1_avx2::prep_mixed::<W, Avx512Wide>(src, ss, dst, ds, h, fracs, taps);
        }
    }
}

/// HEVC/VVC separable `put` (`T`'s `N`-tap table) at width `W` (32 or 64)
/// on 32-output rows.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn separable<T: SeparableTaps<N>, const N: usize, const W: usize>(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    h: usize,
    fx: usize,
    fy: usize,
) {
    // SAFETY: as for `av1_put`.
    unsafe { av1_avx2::separable::<T, N, W, Avx512Wide>(src, ss, dst, ds, h, fx, fy) }
}

/// A 64-wide `put` whose whole-pel blocks take the AVX-512 block copy and
/// fractional ones `$fallback`.
macro_rules! put {
    ($name:ident, $width:literal, $filter:ident, $fallback:path) => {
        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
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
                    crate::block_copy::avx512::copy::<false>(
                        &src[before * ss + before..],
                        ss,
                        dst,
                        ds,
                        $width,
                        h,
                    );
                }
            } else {
                // SAFETY: same window contract; the fractional kernel's ISA (AVX2
                // or AVX-512) is enabled in this tier.
                unsafe {
                    $fallback(src, ss, dst, ds, h, fx, fy);
                }
            }
        }
    };
}
put!(av164, 64, Av1, av1_put::<Regular, 64>);
put!(av1_smooth64, 64, Av1Smooth, av1_put::<Smooth, 64>);
put!(av1_sharp64, 64, Av1Sharp, av1_put::<Sharp, 64>);
put!(hevc_luma64, 64, HevcLuma, separable::<HevcLumaTaps, 8, 64>);
put!(
    hevc_chroma64,
    64,
    HevcChroma,
    separable::<HevcChromaTaps, 4, 64>
);
put!(h264_luma64, 64, H264Luma, super::avx2::h264_luma_w64);
put!(h264_chroma64, 64, H264Chroma, super::avx2::h264_chroma_w64);
put!(vvc_luma64, 64, VvcLuma, separable::<VvcLumaTaps, 8, 64>);
put!(
    vvc_affine_luma64,
    64,
    VvcAffineLuma,
    separable::<VvcAffineLumaTaps, 8, 64>
);
put!(
    vvc_chroma64,
    64,
    VvcChroma,
    separable::<VvcChromaTaps, 4, 64>
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
