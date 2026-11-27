//! The 16-column GDF schedule from dav2d 237229e's
//! `src/x86/looprestoration_avx2.asm` (`gdf_prep_8bpc`, BSD-2-Clause).
//! Copyright © 2026, `VideoLAN` and dav2d authors and Two Orioles, LLC.
//! See `THIRD_PARTY.md` for attribution and the license.
//! Reuse pairwise gradient rows between two-output-row groups. Filter
//! symmetric pairs on signed words, accumulate with `vpmaddwd`, and
//! finish with three packed response digits and scalar byte lookups.
//! The caller supplies the already extended, 10-bit stripe window.

use core::arch::x86_64::{
    __m128i, __m256i, _mm_abs_epi16, _mm_add_epi16, _mm_cvtsi128_si32, _mm_loadu_si32,
    _mm_sub_epi16, _mm256_abs_epi16, _mm256_add_epi16, _mm256_add_epi32, _mm256_blendv_epi8,
    _mm256_cmpgt_epi16, _mm256_dpwssd_epi32, _mm256_loadu_si256, _mm256_madd_epi16,
    _mm256_max_epi16, _mm256_min_epi16, _mm256_mulhrs_epi16, _mm256_mullo_epi16,
    _mm256_packs_epi32, _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setzero_si256,
    _mm256_shuffle_epi8, _mm256_sign_epi16, _mm256_srli_epi16, _mm256_storeu_si256,
    _mm256_sub_epi16, _mm256_sub_epi32, _mm256_unpackhi_epi16, _mm256_unpacklo_epi16,
    _mm256_xor_si256,
};

use super::{
    GDF_BORDER, GDF_ERROR_STRIDE, GDF_MAX_SIDE, GRADIENT_DIRECTIONS, GdfParams, GdfWindow,
};
use crate::simd::packed_pair;

const LANES: usize = 16;
const GRADIENT_COLUMNS: usize = GDF_MAX_SIDE + 2;
type Gradients = [[i16; GRADIENT_COLUMNS]; 4];

/// Two horizontal rows' directional gradients over the centre column
/// and one extra column at each side.
#[target_feature(enable = "avx2")]
#[inline(never)]
unsafe fn gradient_pair(window: &GdfWindow<'_>, y: isize, out: &mut Gradients) {
    let stride = window.stride as isize;
    let start = (y + GDF_BORDER as isize) * stride + GDF_BORDER as isize - 1;
    // SAFETY: the border covers the two adjacent rows at columns
    // -1..=width and their ±tap neighbourhood.
    unsafe {
        let first = window.samples.as_ptr().offset(start);
        let second = first.offset(stride);
        let taps = GRADIENT_DIRECTIONS.map(|(dy, dx)| dy * stride + dx);
        for x in (0..window.width).step_by(LANES) {
            let a = first.add(x);
            let b = second.add(x);
            let ca = _mm256_loadu_si256(a.cast());
            let ca = _mm256_add_epi16(ca, ca);
            let cb = _mm256_loadu_si256(b.cast());
            let cb = _mm256_add_epi16(cb, cb);
            for (grad, tap) in out.iter_mut().zip(taps) {
                let gradient = |p: *const u16, twice: __m256i| {
                    let low = _mm256_loadu_si256(p.offset(-tap).cast());
                    let high = _mm256_loadu_si256(p.offset(tap).cast());
                    _mm256_abs_epi16(_mm256_sub_epi16(twice, _mm256_add_epi16(low, high)))
                };
                let sum = _mm256_add_epi16(gradient(a, ca), gradient(b, cb));
                _mm256_storeu_si256(grad.as_mut_ptr().add(x).cast(), sum);
            }
        }
        let a = first.add(window.width);
        let b = second.add(window.width);
        let ca = _mm_loadu_si32(a.cast());
        let ca = _mm_add_epi16(ca, ca);
        let cb = _mm_loadu_si32(b.cast());
        let cb = _mm_add_epi16(cb, cb);
        for (grad, tap) in out.iter_mut().zip(taps) {
            let gradient = |p: *const u16, twice: __m128i| {
                let low = _mm_loadu_si32(p.offset(-tap).cast());
                let high = _mm_loadu_si32(p.offset(tap).cast());
                _mm_abs_epi16(_mm_sub_epi16(twice, _mm_add_epi16(low, high)))
            };
            let sum = _mm_add_epi16(gradient(a, ca), gradient(b, cb));
            grad.as_mut_ptr()
                .add(window.width)
                .cast::<i32>()
                .write_unaligned(_mm_cvtsi128_si32(sum));
        }
    }
}

/// The two halves of the widened accumulator in the natural pair-dot
/// order. Packing the result interleaves them into pixel order again.
#[derive(Clone, Copy)]
struct Response {
    low: __m256i,
    high: __m256i,
}

impl Response {
    #[target_feature(enable = "avx2")]
    #[inline]
    fn splat(value: i16) -> Self {
        let lanes = _mm256_set1_epi32(i32::from(value));
        Self {
            low: lanes,
            high: lanes,
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    fn accumulate<const SUBTRACT: bool>(&mut self, a: __m256i, b: __m256i, [wa, wb]: [i16; 2]) {
        let weights = _mm256_set1_epi32(packed_pair(wa, wb));
        let low = _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), weights);
        let high = _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), weights);
        if SUBTRACT {
            self.low = _mm256_sub_epi32(self.low, low);
            self.high = _mm256_sub_epi32(self.high, high);
        } else {
            self.low = _mm256_add_epi32(self.low, low);
            self.high = _mm256_add_epi32(self.high, high);
        }
    }

    #[target_feature(enable = "avx2,avx512f,avx512vl,avx512vnni")]
    #[inline]
    unsafe fn add_dot(&mut self, a: __m256i, b: __m256i, [wa, wb]: [i16; 2]) {
        let weights = _mm256_set1_epi32(packed_pair(wa, wb));
        self.low = _mm256_dpwssd_epi32(self.low, _mm256_unpacklo_epi16(a, b), weights);
        self.high = _mm256_dpwssd_epi32(self.high, _mm256_unpackhi_epi16(a, b), weights);
    }

    #[inline(always)]
    #[allow(clippy::inline_always, reason = "fold the target variant at the row")]
    unsafe fn accumulate_taps<const VNNI: bool>(
        &mut self,
        a: __m256i,
        b: __m256i,
        weights: [i16; 2],
    ) {
        // SAFETY: the row enables AVX2 and enables AVX-512 VNNI for VNNI.
        unsafe {
            if VNNI {
                self.add_dot(a, b, weights);
            } else {
                self.accumulate::<true>(a, b, weights);
            }
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    fn negated(self) -> Self {
        let zero = _mm256_setzero_si256();
        Self {
            low: _mm256_sub_epi32(zero, self.low),
            high: _mm256_sub_epi32(zero, self.high),
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    fn pack(self) -> __m256i {
        _mm256_packs_epi32(self.low, self.high)
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn digit(words: __m256i, scale: __m256i, largest: __m256i) -> __m256i {
    // Q15 saturation is already outside the output digits' range.
    // Sign the multiplier first, so the rounding acts on a magnitude
    // even at the negative-word limit. Restore the response's sign.
    let magnitude = _mm256_mulhrs_epi16(words, _mm256_sign_epi16(scale, words));
    let signed = _mm256_sign_epi16(magnitude, words);
    _mm256_add_epi16(_mm256_min_epi16(signed, largest), scale)
}

#[derive(Clone, Copy)]
struct Classes {
    transpose: __m256i,
    mirror: __m256i,
    bias: Response,
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn classify(
    (top, bottom): (&Gradients, &Gradients),
    x: usize,
    params: &GdfParams,
) -> Classes {
    // Swap adjacent words within each 2x2 classifier group.
    const SWAP: [u8; 32] = [
        2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9,
        14, 15, 12, 13,
    ];
    // SAFETY: the fixed table contains one vector of byte indices.
    let shuffle = unsafe { _mm256_loadu_si256(SWAP.as_ptr().cast()) };
    let mut totals = [_mm256_setzero_si256(); 4];
    let mut capped = totals;
    for d in 0..4 {
        // SAFETY: the two gradient rows contain columns -1..=width.
        // x <= width-16 and the second load advances just two columns.
        let (a0, b0, a2, b2) = unsafe {
            let a = top[d].as_ptr().add(x);
            let b = bottom[d].as_ptr().add(x);
            (
                _mm256_loadu_si256(a.cast()),
                _mm256_loadu_si256(b.cast()),
                _mm256_loadu_si256(a.add(2).cast()),
                _mm256_loadu_si256(b.add(2).cast()),
            )
        };
        let two_columns = _mm256_add_epi16(_mm256_add_epi16(a0, b0), _mm256_add_epi16(a2, b2));
        // Both columns of each 2×2 group get the same class. The four-row
        // sum is bounded by 16*2046, within signed words.
        let sums = _mm256_add_epi16(two_columns, _mm256_shuffle_epi8(two_columns, shuffle));
        totals[d] = sums;
        capped[d] = _mm256_min_epi16(
            _mm256_srli_epi16::<4>(sums),
            _mm256_set1_epi16(params.gradient_alpha[d] as i16),
        );
    }
    let mut bias = Response::splat(params.bias[2]);
    bias.accumulate::<false>(
        capped[0],
        capped[1],
        [params.gradient_weight[0], params.gradient_weight[1]],
    );
    bias.accumulate::<false>(
        capped[2],
        capped[3],
        [params.gradient_weight[2], params.gradient_weight[3]],
    );
    let ones = _mm256_set1_epi16(-1);
    Classes {
        transpose: _mm256_xor_si256(_mm256_cmpgt_epi16(totals[0], totals[1]), ones),
        mirror: _mm256_xor_si256(_mm256_cmpgt_epi16(totals[2], totals[3]), ones),
        bias,
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn swap_if(mask: __m256i, a: __m256i, b: __m256i) -> (__m256i, __m256i) {
    (
        _mm256_blendv_epi8(a, b, mask),
        _mm256_blendv_epi8(b, a, mask),
    )
}

#[target_feature(enable = "avx2")]
#[inline]
fn clip_pair(a: __m256i, b: __m256i, alpha: u16) -> __m256i {
    let high = _mm256_set1_epi16(alpha as i16);
    let low = _mm256_set1_epi16(-(alpha as i16));
    let a = _mm256_max_epi16(low, _mm256_min_epi16(a, high));
    let b = _mm256_max_epi16(low, _mm256_min_epi16(b, high));
    _mm256_max_epi16(
        _mm256_set1_epi16(-511),
        _mm256_min_epi16(_mm256_set1_epi16(512), _mm256_add_epi16(a, b)),
    )
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn difference(centre: __m256i, row: *const u16, offset: isize) -> __m256i {
    // SAFETY: caller checked a vector of 10-bit samples beginning at this tap.
    _mm256_sub_epi16(centre, unsafe {
        _mm256_loadu_si256(row.offset(offset).cast())
    })
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn pair_differences(centre: __m256i, row: *const u16, offset: isize) -> [__m256i; 2] {
    // SAFETY: both vectors fall inside the caller's checked six-tap border.
    unsafe {
        [
            difference(centre, row, -offset),
            difference(centre, row, offset),
        ]
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn clip_pairs(
    [a0, a1]: [__m256i; 2],
    [b0, b1]: [__m256i; 2],
    mask: __m256i,
    [alpha0, alpha1]: [u16; 2],
) -> (__m256i, __m256i) {
    let (a0, b0) = swap_if(mask, a0, b0);
    let (a1, b1) = swap_if(mask, a1, b1);
    (clip_pair(a0, a1, alpha0), clip_pair(b0, b1, alpha1))
}

#[target_feature(enable = "avx2")]
unsafe fn row(
    pixels: *const u16,
    stride: isize,
    classes: Classes,
    params: &GdfParams,
    out: &mut [i8],
) {
    // SAFETY: the caller supplies the validated window and output.
    unsafe { row_core::<false>(pixels, stride, classes, params, out) };
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn row_icl(
    pixels: *const u16,
    stride: isize,
    classes: Classes,
    params: &GdfParams,
    out: &mut [i8],
) {
    // SAFETY: as for the AVX2 row. The wider register file gives the
    // same sixteen-column filter more space for its live tap pairs.
    unsafe { row_core::<true>(pixels, stride, classes, params, out) };
}

#[inline(always)]
#[allow(
    clippy::inline_always,
    reason = "generate one intrinsic schedule for both x86 tiers"
)]
unsafe fn row_core<const VNNI: bool>(
    pixels: *const u16,
    stride: isize,
    classes: Classes,
    params: &GdfParams,
    out: &mut [i8],
) {
    // SAFETY: pixel vectors and every tap up to six columns and rows are
    // inside the stripe window checked once at dispatch.
    unsafe {
        let centre = _mm256_loadu_si256(pixels.cast());
        let mut first = Response::splat(params.bias[0]);
        let mut second = Response::splat(params.bias[1]);
        let mut third = classes.bias;
        // VNNI adds the signed products. Accumulate the negated response
        // from its negated bias, then restore the sign once per output.
        if VNNI {
            first = first.negated();
            second = second.negated();
            third = third.negated();
        }
        let axis = |k: usize| {
            let distance = k as isize + 1;
            let vertical = pair_differences(centre, pixels, distance * stride);
            let horizontal = pair_differences(centre, pixels, distance);
            clip_pairs(vertical, horizontal, classes.transpose, params.alpha[k])
        };
        for k in 0..2 {
            let (v, h) = axis(k);
            first.accumulate_taps::<VNNI>(v, h, params.weight1[3 * k]);
            second.accumulate_taps::<VNNI>(v, h, params.weight1[3 * k + 1]);
            third.accumulate_taps::<VNNI>(v, h, params.weight1[3 * k + 2]);
        }
        // As in dav2d, combine distances 3/4 and 5/6 into the two
        // halfwords of each dot product. Pack the completed third response
        // before the remaining taps to shorten its register live range.
        let (v3, h3) = axis(2);
        third.accumulate_taps::<VNNI>(v3, h3, params.weight1[6]);
        let third = if VNNI { third.negated() } else { third }.pack();
        let (v4, h4) = axis(3);
        let [w3, w4, w5, w6] = params.weight2[0];
        let [u3, u4, u5, u6] = params.weight2[1];
        first.accumulate_taps::<VNNI>(v3, v4, [w3, w4]);
        second.accumulate_taps::<VNNI>(h3, h4, [u3, u4]);
        let (v5, h5) = axis(4);
        let (v6, h6) = axis(5);
        first.accumulate_taps::<VNNI>(v5, v6, [w5, w6]);
        second.accumulate_taps::<VNNI>(h5, h6, [u5, u6]);

        let d0 = pair_differences(centre, pixels, stride + 1);
        let d1 = pair_differences(centre, pixels, stride - 1);
        let (a, b) = clip_pairs(d0, d1, classes.mirror, params.alpha[6]);
        first.accumulate_taps::<VNNI>(a, b, params.weight3[0]);
        second.accumulate_taps::<VNNI>(a, b, params.weight3[1]);

        // Transpose first, then mirror, as in the trained table's directions.
        let [td0, td1] = pair_differences(centre, pixels, 2 * stride + 1);
        let [wd0, wd1] = pair_differences(centre, pixels, stride + 2);
        let [tu0, tu1] = pair_differences(centre, pixels, 2 * stride - 1);
        let [wu0, wu1] = pair_differences(centre, pixels, stride - 2);
        let mask = classes.transpose;
        let (td0, wd0) = swap_if(mask, td0, wd0);
        let (td1, wd1) = swap_if(mask, td1, wd1);
        let (tu0, wu0) = swap_if(mask, tu0, wu0);
        let (tu1, wu1) = swap_if(mask, tu1, wu1);
        let (a, b) = clip_pairs([td0, td1], [tu0, tu1], classes.mirror, params.alpha[7]);
        first.accumulate_taps::<VNNI>(a, b, params.weight4[0]);
        let (a, b) = clip_pairs([wd0, wd1], [wu0, wu1], classes.mirror, params.alpha[8]);
        second.accumulate_taps::<VNNI>(a, b, params.weight4[1]);

        let scale = _mm256_set1_epi16(params.scale as i16);
        let largest = _mm256_set1_epi16(params.scale as i16 - 1);
        let base = _mm256_add_epi16(scale, scale);
        let first = if VNNI { first.negated() } else { first };
        let a = digit(first.pack(), scale, largest);
        let second = if VNNI { second.negated() } else { second };
        let b = digit(second.pack(), scale, largest);
        let c = digit(third, scale, largest);
        let index = _mm256_add_epi16(
            _mm256_mullo_epi16(_mm256_add_epi16(_mm256_mullo_epi16(a, base), b), base),
            c,
        );
        let mut indices = [0u16; LANES];
        _mm256_storeu_si256(indices.as_mut_ptr().cast(), index);
        let lut = params.error_lut.as_ptr();
        for (dst, i) in out.iter_mut().zip(indices) {
            // The three digits are in 0..2*scale and the caller checked
            // that the complete cube of table entries is available.
            *dst = *lut.add(i as usize);
        }
    }
}

/// # Safety
/// AVX2 is available. Window and output extents have been checked, width
/// is a nonzero multiple of 16, and all samples are in 10-bit range.
/// Each parameter cap and the LUT cover the trained 5/8-digit scales.
/// For `ICL`, the Ice Lake instruction set must also be enabled.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn error<const ICL: bool>(
    window: &GdfWindow<'_>,
    params: &GdfParams,
    out: &mut [i8],
) {
    let mut previous = [[0; GRADIENT_COLUMNS]; 4];
    let mut next = previous;
    let stride = window.stride;
    let row = if ICL { row_icl } else { row };
    // SAFETY: the validated window has six context rows above and below.
    unsafe { gradient_pair(window, -1, &mut previous) };
    for y in (0..window.height).step_by(2) {
        // SAFETY: gradients for y+1 and y+2 use only one more context row.
        unsafe { gradient_pair(window, y as isize + 1, &mut next) };
        for x in (0..window.width).step_by(LANES) {
            let base = (y + GDF_BORDER) * stride + x + GDF_BORDER;
            // SAFETY: the block, all responses and both adjacent output rows
            // are inside the checked source and output footprints.
            unsafe {
                let classes = classify((&previous, &next), x, params);
                let p = window.samples.as_ptr().add(base);
                let offset = y * GDF_ERROR_STRIDE + x;
                row(
                    p,
                    stride as isize,
                    classes,
                    params,
                    &mut out[offset..offset + LANES],
                );
                let offset = offset + GDF_ERROR_STRIDE;
                row(
                    p.add(stride),
                    stride as isize,
                    classes,
                    params,
                    &mut out[offset..offset + LANES],
                );
            }
        }
        core::mem::swap(&mut previous, &mut next);
    }
}
