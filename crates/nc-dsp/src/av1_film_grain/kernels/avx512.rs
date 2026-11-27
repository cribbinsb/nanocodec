//! Byte grain lookup and block application adapted from dav1d 1.5.4
//! `src/x86/filmgrain_avx512.asm` (BSD-2-Clause, see `THIRD_PARTY.md`).

use super::{Av1GrainScaling, Kernels, blocks, x86};
use crate::av1_film_grain::{Av1FilmGrainParams, GrainBlocks, Template};
use core::arch::x86_64::{
    __m512i, _mm_setr_epi8, _mm256_loadu_si256, _mm256_maskz_loadu_epi8, _mm256_storeu_si256,
    _mm512_add_epi8, _mm512_add_epi16, _mm512_avg_epu8, _mm512_broadcast_i32x4,
    _mm512_castsi256_si512, _mm512_castsi512_si256, _mm512_extracti64x4_epi64, _mm512_inserti64x4,
    _mm512_loadu_si512, _mm512_maddubs_epi16, _mm512_mask_blend_epi8, _mm512_mask_mov_epi8,
    _mm512_max_epu8, _mm512_min_epu8, _mm512_movepi8_mask, _mm512_mulhrs_epi16, _mm512_packs_epi16,
    _mm512_packus_epi16, _mm512_permutex2var_epi8, _mm512_set_epi32, _mm512_set1_epi8,
    _mm512_set1_epi16, _mm512_set1_epi32, _mm512_setzero_si512, _mm512_shuffle_epi8,
    _mm512_srai_epi16, _mm512_storeu_si512, _mm512_unpackhi_epi8, _mm512_unpacklo_epi8,
};
use nc_common::{PlaneView, PlaneViewMut};

pub(super) const KERNELS: Kernels = Kernels {
    luma,
    chroma,
    luma_plane: Some(luma_plane),
    ..x86::KERNELS
};

struct NoiseScale {
    lut: [__m512i; 4],
    rounding: __m512i,
    lower: __m512i,
    upper: __m512i,
    interleave: __m512i,
}

impl NoiseScale {
    #[target_feature(enable = "avx512f,avx512bw")]
    #[inline]
    unsafe fn new(scaling: &Av1GrainScaling, bounds: [u8; 2]) -> Self {
        let lut = scaling.lut.as_ptr();
        Self {
            // SAFETY: the four vectors partition the 256-byte table.
            lut: unsafe {
                [
                    _mm512_loadu_si512(lut.cast()),
                    _mm512_loadu_si512(lut.add(64).cast()),
                    _mm512_loadu_si512(lut.add(128).cast()),
                    _mm512_loadu_si512(lut.add(192).cast()),
                ]
            },
            rounding: _mm512_set1_epi16(1 << (15 - scaling.shift)),
            lower: _mm512_set1_epi8(bounds[0] as i8),
            upper: _mm512_set1_epi8(bounds[1] as i8),
            // Interleave each lane's two groups of eight scales for byte
            // dot products with one nonzero grain input per output word.
            interleave: _mm512_broadcast_i32x4(_mm_setr_epi8(
                0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15,
            )),
        }
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
    #[inline]
    fn apply(&self, pixels: __m512i, noise: __m512i) -> __m512i {
        self.apply_indexed(pixels, pixels, noise)
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
    #[inline]
    fn apply_indexed(&self, pixels: __m512i, indices: __m512i, noise: __m512i) -> __m512i {
        let [lut0, lut1, lut2, lut3] = self.lut;
        let low = _mm512_permutex2var_epi8(lut0, indices, lut1);
        let high = _mm512_permutex2var_epi8(lut2, indices, lut3);
        let scales = _mm512_mask_blend_epi8(_mm512_movepi8_mask(indices), low, high);
        let scales = _mm512_shuffle_epi8(scales, self.interleave);
        let zero = _mm512_setzero_si512();
        let product_lo = _mm512_maddubs_epi16(scales, _mm512_unpacklo_epi8(noise, zero));
        let product_hi = _mm512_maddubs_epi16(scales, _mm512_unpackhi_epi8(zero, noise));
        let sum_lo = _mm512_add_epi16(
            _mm512_unpacklo_epi8(pixels, zero),
            _mm512_mulhrs_epi16(product_lo, self.rounding),
        );
        let sum_hi = _mm512_add_epi16(
            _mm512_unpackhi_epi8(pixels, zero),
            _mm512_mulhrs_epi16(product_hi, self.rounding),
        );
        let clipped = _mm512_packus_epi16(sum_lo, sum_hi);
        _mm512_min_epu8(_mm512_max_epu8(clipped, self.lower), self.upper)
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
unsafe fn chroma(
    source: &[u8],
    luma: &[u8],
    grain: &[i8],
    scaling: &Av1GrainScaling,
    bounds: [u8; 2],
    coefficients: [i16; 3],
    out: &mut [u8],
) {
    const EVEN: [u8; 64] = {
        let mut indices = [0; 64];
        let mut i = 0;
        while i < indices.len() {
            indices[i] = (i * 2) as u8;
            i += 1;
        }
        indices
    };
    let [mult, luma_mult, offset] = coefficients;
    let use_simd = i8::try_from(mult).is_ok()
        && i8::try_from(luma_mult).is_ok()
        && (-256..256).contains(&offset);
    let end = if use_simd {
        source.len().min(luma.len() / 2) / 64 * 64
    } else {
        0
    };
    // SAFETY: checked complete vectors and pairs, and the row contract
    // gives enough grain and output bytes. The remainder uses the AVX2
    // row kernel, which also handles an unpaired final luma sample.
    unsafe {
        if end != 0 {
            let scale = NoiseScale::new(scaling, bounds);
            let even = _mm512_loadu_si512(EVEN.as_ptr().cast());
            let odd = _mm512_add_epi8(even, _mm512_set1_epi8(1));
            let weights = _mm512_set1_epi16((luma_mult << 8) | (mult & 255));
            let bias = _mm512_set1_epi16(offset);
            let from_luma = coefficients == [0, 64, 0];
            for x in (0..end).step_by(64) {
                let first = _mm512_loadu_si512(luma.as_ptr().add(2 * x).cast());
                let second = _mm512_loadu_si512(luma.as_ptr().add(2 * x + 64).cast());
                let average = _mm512_avg_epu8(
                    _mm512_permutex2var_epi8(first, even, second),
                    _mm512_permutex2var_epi8(first, odd, second),
                );
                let pixels = _mm512_loadu_si512(source.as_ptr().add(x).cast());
                let indices = if from_luma {
                    average
                } else {
                    // A saturated product sum still clips to the same byte after
                    // the six-bit shift and the signed nine-bit offset.
                    let index = |pairs| {
                        let sum = _mm512_maddubs_epi16(pairs, weights);
                        _mm512_add_epi16(_mm512_srai_epi16::<6>(sum), bias)
                    };
                    _mm512_packus_epi16(
                        index(_mm512_unpacklo_epi8(pixels, average)),
                        index(_mm512_unpackhi_epi8(pixels, average)),
                    )
                };
                let noise = _mm512_loadu_si512(grain.as_ptr().add(x).cast());
                let result = scale.apply_indexed(pixels, indices, noise);
                _mm512_storeu_si512(out.as_mut_ptr().add(x).cast(), result);
            }
        }
        (x86::KERNELS.chroma)(
            &source[end..],
            &luma[2 * end..],
            &grain[end..],
            scaling,
            bounds,
            coefficients,
            &mut out[end..],
        );
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
unsafe fn luma(
    source: &[u8],
    grain: &[i8],
    scaling: &Av1GrainScaling,
    bounds: [u8; 2],
    out: &mut [u8],
) {
    let vectors = source.len() / 64 * 64;
    // SAFETY: the grain and output rows cover the source row; each
    // iteration accesses a full 64-byte group.
    unsafe {
        if vectors != 0 {
            let scale = NoiseScale::new(scaling, bounds);
            for x in (0..vectors).step_by(64) {
                let pixels = _mm512_loadu_si512(source.as_ptr().add(x).cast());
                let noise = _mm512_loadu_si512(grain.as_ptr().add(x).cast());
                _mm512_storeu_si512(out.as_mut_ptr().add(x).cast(), scale.apply(pixels, noise));
            }
        }
        (x86::KERNELS.luma)(
            &source[vectors..],
            &grain[vectors..],
            scaling,
            bounds,
            &mut out[vectors..],
        );
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn luma_plane(
    (params, blocks): (&Av1FilmGrainParams, GrainBlocks),
    template: &Template<i8>,
    scaling: &Av1GrainScaling,
    bounds: [u8; 2],
    (source, output): (&PlaneView<'_>, &mut PlaneViewMut<'_>),
    (width, height): (usize, usize),
) -> bool {
    if !blocks::whole_blocks(blocks, (width, height)) {
        return false;
    }
    // SAFETY: synthesis validated the views, and whole_blocks checked the
    // extent. The offsets leave a complete block and overlap in the template.
    unsafe {
        let scale = NoiseScale::new(scaling, bounds);
        blocks::visit(
            (params, blocks),
            template,
            (source, output),
            (width, height),
            |block| {
                let (s, ss) = block.source;
                let (d, ds) = block.output;
                let (near, gs) = block.noise;
                let near = near.cast();
                let first = if let Some((top, top_left)) = block.above {
                    let current = load_pair(near, gs);
                    let old = load_pair(top.cast(), gs);
                    let current = if let Some(left) = block.left {
                        left_overlap(current, left.cast(), gs)
                    } else {
                        current
                    };
                    let old = if let Some(left) = top_left {
                        left_overlap(old, left.cast(), gs)
                    } else {
                        old
                    };
                    let weights = _mm512_set_epi32(
                        0x1b11_1b11,
                        0x1b11_1b11,
                        0x1b11_1b11,
                        0x1b11_1b11,
                        0x1b11_1b11,
                        0x1b11_1b11,
                        0x1b11_1b11,
                        0x1b11_1b11,
                        0x111b_111b,
                        0x111b_111b,
                        0x111b_111b,
                        0x111b_111b,
                        0x111b_111b,
                        0x111b_111b,
                        0x111b_111b,
                        0x111b_111b,
                    );
                    store_pair(
                        d,
                        ds,
                        scale.apply(load_pair(s, ss), mix(old, current, weights)),
                    );
                    2
                } else {
                    0
                };
                if let Some(left) = block.left {
                    for y in (first..block.rows).step_by(2) {
                        let samples = load_pair(near.add(y * gs), gs);
                        let noise = left_overlap(samples, left.add(y * gs).cast(), gs);
                        store_pair(
                            d.add(y * ds),
                            ds,
                            scale.apply(load_pair(s.add(y * ss), ss), noise),
                        );
                    }
                } else {
                    for y in (first..block.rows).step_by(2) {
                        let noise = load_pair(near.add(y * gs), gs);
                        store_pair(
                            d.add(y * ds),
                            ds,
                            scale.apply(load_pair(s.add(y * ss), ss), noise),
                        );
                    }
                }
            },
        );
    }
    true
}

#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn load_pair(src: *const u8, stride: usize) -> __m512i {
    // SAFETY: the caller guarantees 32 bytes in both rows.
    unsafe {
        _mm512_inserti64x4::<1>(
            _mm512_castsi256_si512(_mm256_loadu_si256(src.cast())),
            _mm256_loadu_si256(src.add(stride).cast()),
        )
    }
}

#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn store_pair(dst: *mut u8, stride: usize, values: __m512i) {
    // SAFETY: the caller guarantees 32 writable bytes in both rows.
    unsafe {
        _mm256_storeu_si256(dst.cast(), _mm512_castsi512_si256(values));
        _mm256_storeu_si256(
            dst.add(stride).cast(),
            _mm512_extracti64x4_epi64::<1>(values),
        );
    }
}

#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn mix(old: __m512i, new: __m512i, weights: __m512i) -> __m512i {
    let rounding = _mm512_set1_epi16(1024);
    let low = _mm512_maddubs_epi16(weights, _mm512_unpacklo_epi8(old, new));
    let high = _mm512_maddubs_epi16(weights, _mm512_unpackhi_epi8(old, new));
    _mm512_packs_epi16(
        _mm512_mulhrs_epi16(low, rounding),
        _mm512_mulhrs_epi16(high, rounding),
    )
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
unsafe fn left_overlap(new: __m512i, left: *const u8, stride: usize) -> __m512i {
    // SAFETY: each row has exactly two required overlap bytes. The mask
    // prevents reads past the overlap into the rest of the template.
    let old = unsafe {
        _mm512_inserti64x4::<1>(
            _mm512_castsi256_si512(_mm256_maskz_loadu_epi8(3, left.cast())),
            _mm256_maskz_loadu_epi8(3, left.add(stride).cast()),
        )
    };
    let pairs = _mm512_unpacklo_epi8(old, new);
    let sum = _mm512_maddubs_epi16(_mm512_set1_epi32(0x1b11_111b), pairs);
    let mixed = _mm512_mulhrs_epi16(sum, _mm512_set1_epi16(1024));
    let blended = _mm512_packs_epi16(mixed, mixed);
    _mm512_mask_mov_epi8(new, 0x0000_0003_0000_0003, blended)
}
