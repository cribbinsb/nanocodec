//! Narrow H.264 two-axis half pel. The horizontal byte dots and signed
//! vertical word dots follow libavc's `ih264_inter_pred_filters_ssse3.c`.

use super::{WEIGHTS, load, narrow};
use crate::mc::avx2_byte_dot;
use core::arch::x86_64::{
    __m128i, __m256i, _mm_avg_epu8, _mm_cvtsi128_si32, _mm_extract_epi32, _mm_mulhrs_epi16,
    _mm_packs_epi32, _mm_packus_epi16, _mm_set1_epi16, _mm_storel_epi64, _mm_unpackhi_epi16,
    _mm_unpacklo_epi16, _mm_unpacklo_epi32, _mm_unpacklo_epi64, _mm256_add_epi32,
    _mm256_castsi128_si256, _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_madd_epi16,
    _mm256_set_m128i, _mm256_set1_epi16, _mm256_set1_epi32, _mm256_srai_epi32,
};

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn horizontal<const W: usize>(
    a: *const u8,
    b: *const u8,
    taps: &[__m256i; 3],
) -> [__m128i; 2] {
    // SAFETY: each input points to W + 5 checked bytes.
    let h = unsafe { avx2_byte_dot::two_rows::<3, W>(a, b, taps) };
    [_mm256_castsi256_si128(h), _mm256_extracti128_si256::<1>(h)]
}

#[target_feature(enable = "avx2")]
#[inline]
fn adjacent<const W: usize>(a: __m128i, b: __m128i) -> __m256i {
    let low = _mm_unpacklo_epi16(a, b);
    if W == 4 {
        _mm256_castsi128_si256(low)
    } else {
        _mm256_set_m128i(_mm_unpackhi_epi16(a, b), low)
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn vertical(pairs: [__m256i; 3], coeffs: [__m256i; 3]) -> __m128i {
    let [a, b, c] = core::array::from_fn(|i| _mm256_madd_epi16(pairs[i], coeffs[i]));
    let sum = _mm256_add_epi32(_mm256_add_epi32(a, b), c);
    let rounded = _mm256_srai_epi32::<10>(_mm256_add_epi32(sum, _mm256_set1_epi32(512)));
    let words = _mm_packs_epi32(
        _mm256_castsi256_si128(rounded),
        _mm256_extracti128_si256::<1>(rounded),
    );
    _mm_packus_epi16(words, words)
}

#[target_feature(enable = "avx2")]
#[inline]
fn horizontal_half(row: __m128i) -> __m128i {
    let half = _mm_mulhrs_epi16(row, _mm_set1_epi16(1024));
    _mm_packus_epi16(half, half)
}

/// Center half-pel, optionally averaged with a horizontal or vertical half.
/// # Safety
/// AVX2 is available, W is 4 or 8, height is even and at least two, and
/// both offsets are in 1–3 and at least one equals 2.
#[target_feature(enable = "avx2")]
#[inline]
pub(in crate::mc::avx2) unsafe fn predict<const W: usize, const FX: usize, const FY: usize>(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    height: usize,
    phase: (usize, usize),
) {
    let (fx, fy) = if FX == 0 { phase } else { (FX, FY) };
    crate::mc::generic::check_h264_window(src, ss, dst, ds, (W, height, 5));
    let taps = WEIGHTS.map(|w| _mm256_set1_epi16(w));
    let coeffs = [0xfffb_0001u32 as i32, 0x0014_0014, 0x0001_fffb].map(|v| _mm256_set1_epi32(v));
    // SAFETY: the checked window contains all six taps for each output.
    // The final horizontal call repeats the last row rather than loading
    // the unused row beyond the bottom edge.
    unsafe {
        let base = src.as_ptr();
        let output = dst.as_mut_ptr();
        // libavc forms the vertical half at the selected column before
        // averaging it with the two-axis half pel.
        if fx != 2 {
            narrow::<W>((base.add(2 + fx / 2), ss), (output, ds), height, 2, None);
        }
        let [r0, r1] = horizontal::<W>(base, base.add(ss), &taps);
        let [r2, r3] = horizontal::<W>(base.add(2 * ss), base.add(3 * ss), &taps);
        let [r4, r5] = horizontal::<W>(base.add(4 * ss), base.add(5 * ss), &taps);
        let first = [r0, r1, r2, r3, r4, r5];
        let mut pairs: [__m256i; 5] =
            core::array::from_fn(|i| adjacent::<W>(first[i], first[i + 1]));
        let (mut r2, mut r3, mut r4, mut r5) = (r2, r3, r4, r5);
        for y in (0..height).step_by(2) {
            let next = base.add((y + 6) * ss);
            let second = if y + 2 == height { next } else { next.add(ss) };
            let [r6, r7] = horizontal::<W>(next, second, &taps);
            let [p0, p1, p2, p3, p4] = pairs;
            let p5 = adjacent::<W>(r5, r6);
            let (h0, h1) = if fy == 1 { (r2, r3) } else { (r3, r4) };
            let out = output.add(y * ds);
            if W == 4 {
                let join =
                    |a, b| _mm256_set_m128i(_mm256_castsi256_si128(b), _mm256_castsi256_si128(a));
                let mut pixels = vertical([join(p0, p1), join(p2, p3), join(p4, p5)], coeffs);
                if fy != 2 {
                    pixels = _mm_avg_epu8(pixels, horizontal_half(_mm_unpacklo_epi64(h0, h1)));
                } else if fx != 2 {
                    pixels = _mm_avg_epu8(
                        pixels,
                        _mm_unpacklo_epi32(load::<W>(out), load::<W>(out.add(ds))),
                    );
                }
                out.cast::<i32>().write_unaligned(_mm_cvtsi128_si32(pixels));
                out.add(ds)
                    .cast::<i32>()
                    .write_unaligned(_mm_extract_epi32::<1>(pixels));
            } else {
                let mut a = vertical([p0, p2, p4], coeffs);
                let mut b = vertical([p1, p3, p5], coeffs);
                if fy != 2 {
                    a = _mm_avg_epu8(a, horizontal_half(h0));
                    b = _mm_avg_epu8(b, horizontal_half(h1));
                } else if fx != 2 {
                    a = _mm_avg_epu8(a, load::<W>(out));
                    b = _mm_avg_epu8(b, load::<W>(out.add(ds)));
                }
                _mm_storel_epi64(out.cast(), a);
                _mm_storel_epi64(out.add(ds).cast(), b);
            }
            pairs = [p2, p3, p4, p5, adjacent::<W>(r6, r7)];
            (r2, r3, r4, r5) = (r4, r5, r6, r7);
        }
    }
}
