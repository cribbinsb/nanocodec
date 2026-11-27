//! H.264 vertical luma interpolation with paired byte taps.
//! The two-row window and multiply-add groups follow libavc's
//! `common/x86/ih264_inter_pred_filters_ssse3.c` (Apache-2.0).

use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi16, _mm_avg_epu8, _mm_cvtsi32_si128, _mm_cvtsi128_si32,
    _mm_extract_epi32, _mm_extract_epi64, _mm_loadl_epi64, _mm_loadu_si128, _mm_maddubs_epi16,
    _mm_mulhrs_epi16, _mm_packus_epi16, _mm_set1_epi16, _mm_storel_epi64, _mm_storeu_si128,
    _mm_unpackhi_epi8, _mm_unpacklo_epi8, _mm_unpacklo_epi32, _mm_unpacklo_epi64, _mm256_add_epi16,
    _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_maddubs_epi16, _mm256_mulhrs_epi16,
    _mm256_set_m128i, _mm256_set1_epi16,
};

const WEIGHTS: [i16; 3] = [0xfb01u16 as i16, 0x1414, 0x01fb];

#[target_feature(enable = "avx2")]
#[inline]
fn half128(pairs: [__m128i; 3], weights: [__m128i; 3]) -> __m128i {
    let [a, b, c] = core::array::from_fn(|i| _mm_maddubs_epi16(pairs[i], weights[i]));
    let sum = _mm_add_epi16(_mm_add_epi16(a, b), c);
    _mm_mulhrs_epi16(sum, _mm_set1_epi16(1024))
}

#[target_feature(enable = "avx2")]
#[inline]
fn half256(pairs: [__m256i; 3], weights: [__m256i; 3]) -> __m256i {
    let [a, b, c] = core::array::from_fn(|i| _mm256_maddubs_epi16(pairs[i], weights[i]));
    let sum = _mm256_add_epi16(_mm256_add_epi16(a, b), c);
    _mm256_mulhrs_epi16(sum, _mm256_set1_epi16(1024))
}

#[target_feature(enable = "avx2")]
#[inline]
fn pack128(words: __m256i) -> __m128i {
    _mm_packus_epi16(
        _mm256_castsi256_si128(words),
        _mm256_extracti128_si256::<1>(words),
    )
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load<const W: usize>(src: *const u8) -> __m128i {
    // SAFETY: the caller supplies W readable samples.
    unsafe {
        match W {
            4 => _mm_cvtsi32_si128(src.cast::<i32>().read_unaligned()),
            8 => _mm_loadl_epi64(src.cast()),
            _ => _mm_loadu_si128(src.cast()),
        }
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn interleave_wide(a: __m128i, b: __m128i) -> __m256i {
    _mm256_set_m128i(_mm_unpackhi_epi8(a, b), _mm_unpacklo_epi8(a, b))
}

/// # Safety
/// AVX2 is enabled, width is 4, 8 or 16, height is even and phase is 1–3.
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn predict<const W: usize>(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    height: usize,
    phase: usize,
) {
    crate::mc::generic::check_h264_window(src, ss, dst, ds, (W, height, 5));
    let output = dst.as_mut_ptr();
    // SAFETY: the checked window covers the rows and columns read by both
    // schedules. All output stores write exactly W samples.
    unsafe {
        let source = src.as_ptr().add(2);
        if W == 16 {
            wide((source, ss), (output, ds), height, phase);
        } else {
            narrow::<W>((source, ss), (output, ds), height, phase, None);
        }
    }
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn narrow<const W: usize>(
    (source, ss): (*const u8, usize),
    (output, ds): (*mut u8, usize),
    height: usize,
    phase: usize,
    horizontal: Option<*const u8>,
) {
    // SAFETY: the input rows include the five-tap halo after the last output.
    unsafe {
        let w128 = WEIGHTS.map(|w| _mm_set1_epi16(w));
        let w256 = WEIGHTS.map(|w| _mm256_set1_epi16(w));
        let first: [__m128i; 5] = core::array::from_fn(|i| load::<W>(source.add(i * ss)));
        let mut pairs =
            core::array::from_fn::<_, 4, _>(|i| _mm_unpacklo_epi8(first[i], first[i + 1]));
        let (mut r2, mut r3, mut r4) = (first[2], first[3], first[4]);
        for y in (0..height).step_by(2) {
            let r5 = load::<W>(source.add((y + 5) * ss));
            let r6 = load::<W>(source.add((y + 6) * ss));
            let [p0, p1, p2, p3] = pairs;
            let p4 = _mm_unpacklo_epi8(r4, r5);
            let p5 = _mm_unpacklo_epi8(r5, r6);
            let mut bytes = if W == 4 {
                let words = half128(
                    [
                        _mm_unpacklo_epi64(p0, p1),
                        _mm_unpacklo_epi64(p2, p3),
                        _mm_unpacklo_epi64(p4, p5),
                    ],
                    w128,
                );
                _mm_packus_epi16(words, words)
            } else {
                let words = half256(
                    [
                        _mm256_set_m128i(p1, p0),
                        _mm256_set_m128i(p3, p2),
                        _mm256_set_m128i(p5, p4),
                    ],
                    w256,
                );
                pack128(words)
            };
            if let Some(top) = horizontal {
                bytes = _mm_avg_epu8(bytes, horizontal_pair::<W>(top.add(y * ss), ss, &w256));
            } else if phase & 1 != 0 {
                let (a, b) = if phase == 1 { (r2, r3) } else { (r3, r4) };
                let integer = if W == 4 {
                    _mm_unpacklo_epi32(a, b)
                } else {
                    _mm_unpacklo_epi64(a, b)
                };
                bytes = _mm_avg_epu8(bytes, integer);
            }
            let out = output.add(y * ds);
            store_pair::<W>(out, ds, bytes);
            pairs = [p2, p3, p4, p5];
            (r2, r3, r4) = (r4, r5, r6);
        }
    }
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn wide(
    (source, ss): (*const u8, usize),
    (output, ds): (*mut u8, usize),
    height: usize,
    phase: usize,
) {
    // SAFETY: predict checked every full 16-byte row load and store.
    unsafe {
        let weights = WEIGHTS.map(|w| _mm256_set1_epi16(w));
        let first: [__m128i; 5] = core::array::from_fn(|i| load::<16>(source.add(i * ss)));
        let mut pairs =
            core::array::from_fn::<_, 4, _>(|i| interleave_wide(first[i], first[i + 1]));
        let (mut r2, mut r3, mut r4) = (first[2], first[3], first[4]);
        for y in (0..height).step_by(2) {
            let r5 = load::<16>(source.add((y + 5) * ss));
            let r6 = load::<16>(source.add((y + 6) * ss));
            let [p0, p1, p2, p3] = pairs;
            let p4 = interleave_wide(r4, r5);
            let p5 = interleave_wide(r5, r6);
            let mut a = pack128(half256([p0, p2, p4], weights));
            let mut b = pack128(half256([p1, p3, p5], weights));
            if phase & 1 != 0 {
                let (u, v) = if phase == 1 { (r2, r3) } else { (r3, r4) };
                a = _mm_avg_epu8(a, u);
                b = _mm_avg_epu8(b, v);
            }
            let out = output.add(y * ds);
            _mm_storeu_si128(out.cast(), a);
            _mm_storeu_si128(out.add(ds).cast(), b);
            pairs = [p2, p3, p4, p5];
            (r2, r3, r4) = (r4, r5, r6);
        }
    }
}

/// Two horizontal halves packed as two adjacent four- or eight-byte rows.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn horizontal_pair<const W: usize>(
    source: *const u8,
    stride: usize,
    weights: &[__m256i; 3],
) -> __m128i {
    // SAFETY: the caller provides W + 5 readable bytes on each row.
    let sum =
        unsafe { crate::mc::avx2_byte_dot::two_rows::<3, W>(source, source.add(stride), weights) };
    let words = _mm256_mulhrs_epi16(sum, _mm256_set1_epi16(1024));
    if W == 4 {
        let lo = _mm256_castsi256_si128(words);
        let hi = _mm256_extracti128_si256::<1>(words);
        let both = _mm_unpacklo_epi64(lo, hi);
        _mm_packus_epi16(both, both)
    } else {
        pack128(words)
    }
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn store_pair<const W: usize>(out: *mut u8, stride: usize, rows: __m128i) {
    // SAFETY: the caller provides two independently writable W-byte rows.
    unsafe {
        if W == 4 {
            out.cast::<i32>().write_unaligned(_mm_cvtsi128_si32(rows));
            out.add(stride)
                .cast::<i32>()
                .write_unaligned(_mm_extract_epi32::<1>(rows));
        } else {
            _mm_storel_epi64(out.cast(), rows);
            out.add(stride)
                .cast::<i64>()
                .write_unaligned(_mm_extract_epi64::<1>(rows));
        }
    }
}

/// Two-row horizontal interpolation following libavc's packed tap and
/// quarter-pel averaging schedule.
/// # Safety
/// AVX2 is available, W is 4 or 8, height is even and phase is 1–3.
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn horizontal<const W: usize>(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    height: usize,
    phase: usize,
) {
    crate::mc::generic::check_h264_window(src, ss, dst, ds, (W, height, 5));
    let weights = WEIGHTS.map(|w| _mm256_set1_epi16(w));
    // SAFETY: the checked window covers both rows and their six-tap windows.
    unsafe {
        let source = src.as_ptr().add(2 * ss);
        let output = dst.as_mut_ptr();
        for y in (0..height).step_by(2) {
            let from = source.add(y * ss);
            let mut bytes = horizontal_pair::<W>(from, ss, &weights);
            if phase != 2 {
                let integer = from.add(2 + phase / 2);
                let a = load::<W>(integer);
                let b = load::<W>(integer.add(ss));
                let pixels = if W == 4 {
                    _mm_unpacklo_epi32(a, b)
                } else {
                    _mm_unpacklo_epi64(a, b)
                };
                bytes = _mm_avg_epu8(bytes, pixels);
            }
            store_pair::<W>(output.add(y * ds), ds, bytes);
        }
    }
}

/// Diagonal quarter-pel outputs average a vertical and a horizontal half.
/// The tap groups follow libavc. Each pair of halves stays in registers
/// until the average is stored.
/// # Safety
/// AVX2 is available, width is 4 or 8, height is even, and both offsets
/// are 1 or 3.
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn diagonal<const W: usize>(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    height: usize,
    (fx, fy): (usize, usize),
) {
    crate::mc::generic::check_h264_window(src, ss, dst, ds, (W, height, 5));
    // SAFETY: the checked source covers all six filter taps of both
    // passes and the one-sample shifts.
    unsafe {
        let source = src.as_ptr();
        narrow::<W>(
            (source.add(2 + fx / 2), ss),
            (dst.as_mut_ptr(), ds),
            height,
            2,
            Some(source.add((2 + fy / 2) * ss)),
        );
    }
}

#[path = "avx2_h264_center.rs"]
mod center_filter;
pub(super) use center_filter::predict as center;
