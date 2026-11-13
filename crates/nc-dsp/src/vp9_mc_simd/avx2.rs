//! AVX2 VP9 interpolation: paired unsigned-pixel × signed-tap byte multiply-adds
//! (`vpmaddubsw`), as in libvpx's AVX2 convolution (BSD-3-Clause;
//! `THIRD_PARTY.md`), with a table-independent exact accumulation instead of
//! libvpx's order-dependent saturating one.
//!
//! Each tap pair's product sum fits i16 (`pairs_fit_i16`). The four pair sums
//! are added with wrapping i16 arithmetic, i.e. modulo 2^16. Every filter sum
//! lies in a window narrower than 2^16, so adding [`OFFSET`] maps it into
//! [0, 65535] (`offset_window_holds_every_sum`) where the u16 value is exact.
//! A logical shift then undoes the offset with a saturating subtract, which
//! also clips negative results to zero; the final narrow clips to 255.

use super::driver::{ROUND, RowKernel, SHIFT, layout};
use crate::Vp9Filter;
use core::arch::x86_64::{
    __m128i, __m256i, _mm_avg_epu8, _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_loadl_epi64,
    _mm_loadu_si128, _mm_or_si128, _mm_packus_epi16, _mm_slli_si128, _mm_srli_si128,
    _mm_storel_epi64, _mm_storeu_si128, _mm_unpackhi_epi8, _mm_unpacklo_epi8, _mm256_add_epi16,
    _mm256_broadcastsi128_si256, _mm256_castsi128_si256, _mm256_castsi256_si128,
    _mm256_extracti128_si256, _mm256_inserti128_si256, _mm256_loadu_si256, _mm256_maddubs_epi16,
    _mm256_packus_epi16, _mm256_set1_epi16, _mm256_shuffle_epi8, _mm256_srli_epi16,
    _mm256_subs_epu16, _mm256_unpackhi_epi8, _mm256_unpacklo_epi8,
};

/// Output units added to every sum before the shift and removed after it.
const OFFSET_UNITS: i16 = 128;
/// [`OFFSET_UNITS`] at filter precision: 128 << 7 = 16384.
const OFFSET: i16 = OFFSET_UNITS << SHIFT;

struct Avx2;

#[inline(always)]
unsafe fn load<const N: usize>(p: *const u8) -> __m128i {
    // SAFETY: the caller supplies exactly N readable bytes, N in 4/8/16.
    unsafe {
        match N {
            4 => _mm_cvtsi32_si128(p.cast::<i32>().read_unaligned()),
            8 => _mm_loadl_epi64(p.cast()),
            _ => _mm_loadu_si128(p.cast()),
        }
    }
}

/// Interleaves two rows of N pixels: output i's pair in bytes 2i and 2i+1,
/// outputs 0..8 in the low lane and 8..16 in the high lane.
#[inline(always)]
unsafe fn pair(a: __m128i, b: __m128i) -> __m256i {
    // SAFETY: the ISA entry enables AVX2.
    unsafe {
        _mm256_inserti128_si256::<1>(
            _mm256_castsi128_si256(_mm_unpacklo_epi8(a, b)),
            _mm_unpackhi_epi8(a, b),
        )
    }
}

/// The rounded, nonnegative 16-bit results of filter sums (see the module
/// comment for the offset); the final narrow clips them to 255.
#[inline(always)]
unsafe fn units(sum: __m256i) -> __m256i {
    // SAFETY: the ISA entry enables AVX2.
    unsafe {
        let biased = _mm256_add_epi16(sum, _mm256_set1_epi16(ROUND + OFFSET));
        _mm256_subs_epu16(
            _mm256_srli_epi16::<{ SHIFT as i32 }>(biased),
            _mm256_set1_epi16(OFFSET_UNITS),
        )
    }
}

/// The 8-bit results of N outputs from their filter sums.
#[inline(always)]
unsafe fn finish(sum: __m256i) -> __m128i {
    // SAFETY: the ISA entry enables AVX2.
    unsafe {
        let units = units(sum);
        _mm_packus_epi16(
            _mm256_castsi256_si128(units),
            _mm256_extracti128_si256::<1>(units),
        )
    }
}

/// The 8-tap sum of four sliding tap-pair vectors (see [`column_sliding`]).
#[inline(always)]
unsafe fn tap_sum(pairs: &[__m256i; 4], f: &[__m256i; 5]) -> __m256i {
    // SAFETY: the ISA entry enables AVX2.
    unsafe {
        add_pairs([
            _mm256_maddubs_epi16(pairs[0], f[0]),
            _mm256_maddubs_epi16(pairs[1], f[1]),
            _mm256_maddubs_epi16(pairs[2], f[2]),
            _mm256_maddubs_epi16(pairs[3], f[3]),
        ])
    }
}

/// The vertical 8-tap pass of one N-wide column, two output rows per step,
/// after libvpx's `vpx_filter_block1d{4,8,16}_v8_avx2` (BSD-3-Clause;
/// `THIRD_PARTY.md`). Vector `k` of the window holds the interleaved source
/// rows `2k, 2k + 1` of output row `y` in its low lane and rows `2k + 1,
/// 2k + 2` of row `y + 1` in its high lane, so each step loads and
/// interleaves only the two new rows and slides the window by one vector.
/// 16-wide columns keep two windows (bytes 0..8 and 8..16 of each row).
/// An odd last row falls back to the per-row schedule.
#[inline(always)]
unsafe fn column_sliding<const N: usize, const LAYOUT: u8, const AVG: bool>(
    src: *const u8,
    stride: usize,
    (dst, dst_stride): (*mut u8, usize),
    h: usize,
    f: &[__m256i; 5],
) {
    // SAFETY: the caller provides h + 7 readable rows of N bytes and h
    // writable (and, for AVG, readable) rows of N bytes.
    unsafe {
        let row = |k: usize| load::<N>(src.add(k * stride));
        let window = |a: __m128i, b: __m128i, c: __m128i| {
            let (upper, lower) = (lanes(a, b), lanes(b, c));
            (
                _mm256_unpacklo_epi8(upper, lower),
                _mm256_unpackhi_epi8(upper, lower),
            )
        };
        // Separate variables, not an array: a rotated array stayed in memory.
        let mut last = row(0);
        let (middle, next) = (row(1), row(2));
        let (mut low0, mut high0) = window(last, middle, next);
        last = next;
        let (middle, next) = (row(3), row(4));
        let (mut low1, mut high1) = window(last, middle, next);
        last = next;
        let (middle, next) = (row(5), row(6));
        let (mut low2, mut high2) = window(last, middle, next);
        last = next;
        let mut y = 0;
        while y + 1 < h {
            let (middle, next) = (row(y + 7), row(y + 8));
            let (low3, high3) = window(last, middle, next);
            last = next;
            let out = dst.add(y * dst_stride);
            if N == 16 {
                let both = _mm256_packus_epi16(
                    units(tap_sum(&[low0, low1, low2, low3], f)),
                    units(tap_sum(&[high0, high1, high2, high3], f)),
                );
                Avx2::store::<N, AVG>(out, _mm256_castsi256_si128(both));
                Avx2::store::<N, AVG>(out.add(dst_stride), _mm256_extracti128_si256::<1>(both));
            } else {
                let bytes = finish(tap_sum(&[low0, low1, low2, low3], f));
                Avx2::store::<N, AVG>(out, bytes);
                Avx2::store::<N, AVG>(out.add(dst_stride), _mm_srli_si128::<8>(bytes));
            }
            (low0, low1, low2) = (low1, low2, low3);
            (high0, high1, high2) = (high1, high2, high3);
            y += 2;
        }
        if y < h {
            super::driver::column_rows::<Avx2, N, LAYOUT, AVG>(
                src.add(y * stride),
                stride,
                (dst.add(y * dst_stride), dst_stride),
                h - y,
                f,
            );
        }
    }
}

/// Wrapping sum of the four tap-pair sums: exact modulo 2^16 (see the
/// module comment).
#[inline(always)]
unsafe fn add_pairs(pairs: [__m256i; 4]) -> __m256i {
    // SAFETY: the ISA entry enables AVX2.
    unsafe {
        _mm256_add_epi16(
            _mm256_add_epi16(pairs[0], pairs[1]),
            _mm256_add_epi16(pairs[2], pairs[3]),
        )
    }
}

/// Byte shuffle gathering the pixels of taps 2*`pair` and 2*`pair`+1 for each
/// output. For N=16 the high lane holds source bytes 7..23, hence the +1.
const fn masks<const N: usize>(pair: usize) -> [u8; 32] {
    let mut out = [0; 32];
    let mut i = 0;
    while i < 32 {
        let high_lane = if N == 16 && i >= 16 { 1 } else { 0 };
        out[i] = (i % 16 / 2 + i % 2 + 2 * pair + high_lane) as u8;
        i += 1;
    }
    out
}

/// Taps `k` and `k + 1` as signed bytes, repeated in every 16-bit lane.
#[inline(always)]
unsafe fn weights(taps: &[i16; 8], k: usize) -> __m256i {
    // Nonidentity phases have every tap in -127..=127 (`tables_match_their_layouts`).
    let byte = |tap: i16| tap as i8 as u8;
    // SAFETY: called only from the AVX2 entry.
    unsafe { _mm256_set1_epi16(i16::from_le_bytes([byte(taps[k]), byte(taps[k + 1])])) }
}

#[inline(always)]
unsafe fn horizontal<const N: usize, const PAIR: usize>(
    pixels: __m256i,
    weight: __m256i,
) -> __m256i {
    // SAFETY: AVX2 is enabled; this full mask is a compile-time constant.
    unsafe {
        _mm256_maddubs_epi16(
            _mm256_shuffle_epi8(pixels, _mm256_loadu_si256(masks::<N>(PAIR).as_ptr().cast())),
            weight,
        )
    }
}

/// Source bytes 0..N+7 of one row, arranged for [`masks`]: N=16 puts bytes
/// 0..16 and 7..23 in the two lanes; shorter rows OR two overlapping eight-byte
/// loads (equal bytes overlap). The last load ends exactly at the final tap.
#[inline(always)]
unsafe fn row<const N: usize>(src: *const u8) -> __m256i {
    // SAFETY: the caller validated N + 7 readable bytes at `src`.
    unsafe {
        if N == 16 {
            _mm256_inserti128_si256::<1>(
                _mm256_castsi128_si256(_mm_loadu_si128(src.cast())),
                _mm_loadu_si128(src.add(7).cast()),
            )
        } else {
            _mm256_broadcastsi128_si256(narrow_row::<N>(src))
        }
    }
}

/// Source bytes `0..N + 7` of one row of N = 4 or 8 outputs in one 128-bit
/// lane: two overlapping eight-byte loads combined by OR (equal bytes overlap), the
/// last ending exactly at the final tap.
#[inline(always)]
unsafe fn narrow_row<const N: usize>(src: *const u8) -> __m128i {
    // SAFETY: the caller validated N + 7 readable bytes at `src`.
    unsafe {
        let low = _mm_loadl_epi64(src.cast());
        let high = _mm_loadl_epi64(src.add(N - 1).cast());
        let tail = if N == 4 {
            _mm_slli_si128::<3>(high)
        } else {
            _mm_slli_si128::<7>(high)
        };
        _mm_or_si128(low, tail)
    }
}

/// Two 128-bit values as the low and high lanes of one vector.
#[inline(always)]
unsafe fn lanes(low: __m128i, high: __m128i) -> __m256i {
    // SAFETY: the ISA entry enables AVX2.
    unsafe { _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(low), high) }
}

impl RowKernel for Avx2 {
    /// Tap pairs (0,1), (2,3), (4,5), (6,7) and the bilinear centre (3,4).
    type Filters = [__m256i; 5];
    type Row = __m128i;
    const LANES: usize = 16;
    /// Rows of 4 or 8 outputs fill half a vector: two rows share one, as
    /// libvpx's AVX2 convolution pairs rows (BSD-3-Clause).
    const PAIR_ROWS: bool = true;
    #[inline(always)]
    unsafe fn prepare(taps: &[i16; 8]) -> Self::Filters {
        // SAFETY: called only with AVX2 enabled and a nonidentity phase.
        unsafe {
            [
                weights(taps, 0),
                weights(taps, 2),
                weights(taps, 4),
                weights(taps, 6),
                weights(taps, 3),
            ]
        }
    }
    #[inline(always)]
    unsafe fn load<const N: usize>(src: *const u8) -> __m128i {
        // SAFETY: the caller provides N readable bytes.
        unsafe { load::<N>(src) }
    }
    #[inline(always)]
    unsafe fn store<const N: usize, const AVG: bool>(dst: *mut u8, row: __m128i) {
        // SAFETY: the caller provides N writable bytes, also readable when AVG; each store writes N.
        unsafe {
            let row = if AVG {
                _mm_avg_epu8(row, Self::load::<N>(dst))
            } else {
                row
            };
            match N {
                4 => dst.cast::<i32>().write_unaligned(_mm_cvtsi128_si32(row)),
                8 => _mm_storel_epi64(dst.cast(), row),
                _ => _mm_storeu_si128(dst.cast(), row),
            }
        }
    }
    #[inline(always)]
    unsafe fn horizontal<const N: usize, const LAYOUT: u8>(
        src: *const u8,
        f: &Self::Filters,
    ) -> __m128i {
        // SAFETY: the caller provides N + 7 readable bytes; the loads end at
        // the last tap, with no overread.
        unsafe {
            let sum = if LAYOUT == layout::BILINEAR {
                _mm256_maddubs_epi16(pair(load::<N>(src.add(3)), load::<N>(src.add(4))), f[4])
            } else {
                let pixels = row::<N>(src);
                add_pairs([
                    horizontal::<N, 0>(pixels, f[0]),
                    horizontal::<N, 1>(pixels, f[1]),
                    horizontal::<N, 2>(pixels, f[2]),
                    horizontal::<N, 3>(pixels, f[3]),
                ])
            };
            finish(sum)
        }
    }
    #[inline(always)]
    unsafe fn vertical<const N: usize, const LAYOUT: u8>(
        rows: &[__m128i; 8],
        f: &Self::Filters,
    ) -> __m128i {
        // SAFETY: the ISA entry enables AVX2.
        unsafe {
            let sum = if LAYOUT == layout::BILINEAR {
                _mm256_maddubs_epi16(pair(rows[3], rows[4]), f[4])
            } else {
                add_pairs(
                    [0, 1, 2, 3]
                        .map(|j| _mm256_maddubs_epi16(pair(rows[2 * j], rows[2 * j + 1]), f[j])),
                )
            };
            finish(sum)
        }
    }
    #[inline(always)]
    unsafe fn horizontal_pair<const N: usize, const LAYOUT: u8>(
        src: *const u8,
        stride: usize,
        f: &Self::Filters,
    ) -> [__m128i; 2] {
        // SAFETY: the caller provides N + 7 readable bytes in both rows.
        unsafe {
            if N == 16 {
                return [
                    Self::horizontal::<N, LAYOUT>(src, f),
                    Self::horizontal::<N, LAYOUT>(src.add(stride), f),
                ];
            }
            let next = src.add(stride);
            let sum = if LAYOUT == layout::BILINEAR {
                let pairs = lanes(
                    _mm_unpacklo_epi8(load::<N>(src.add(3)), load::<N>(src.add(4))),
                    _mm_unpacklo_epi8(load::<N>(next.add(3)), load::<N>(next.add(4))),
                );
                _mm256_maddubs_epi16(pairs, f[4])
            } else {
                let pixels = lanes(narrow_row::<N>(src), narrow_row::<N>(next));
                add_pairs([
                    horizontal::<N, 0>(pixels, f[0]),
                    horizontal::<N, 1>(pixels, f[1]),
                    horizontal::<N, 2>(pixels, f[2]),
                    horizontal::<N, 3>(pixels, f[3]),
                ])
            };
            // Row 0's outputs in bytes 0..N, row 1's in bytes 8..8 + N.
            let bytes = finish(sum);
            [bytes, bytes]
        }
    }
    #[inline(always)]
    unsafe fn vertical_pair<const N: usize, const LAYOUT: u8>(
        rows: &[__m128i; 9],
        f: &Self::Filters,
    ) -> [__m128i; 2] {
        // SAFETY: the ISA entry enables AVX2.
        unsafe {
            let tap_pair = |k: usize| {
                lanes(
                    _mm_unpacklo_epi8(rows[k], rows[k + 1]),
                    _mm_unpacklo_epi8(rows[k + 1], rows[k + 2]),
                )
            };
            if N == 16 {
                return [
                    Self::vertical::<N, LAYOUT>(rows[..8].try_into().expect("eight rows"), f),
                    Self::vertical::<N, LAYOUT>(rows[1..].try_into().expect("eight rows"), f),
                ];
            }
            let sum = if LAYOUT == layout::BILINEAR {
                _mm256_maddubs_epi16(tap_pair(3), f[4])
            } else {
                add_pairs([0, 1, 2, 3].map(|j| _mm256_maddubs_epi16(tap_pair(2 * j), f[j])))
            };
            let bytes = finish(sum);
            [bytes, bytes]
        }
    }
    #[inline(always)]
    unsafe fn column<const N: usize, const LAYOUT: u8, const AVG: bool>(
        src: *const u8,
        stride: usize,
        dst: (*mut u8, usize),
        h: usize,
        f: &Self::Filters,
    ) {
        // SAFETY: as for `RowKernel::column`.
        unsafe {
            if LAYOUT == layout::BILINEAR {
                super::driver::column_rows::<Self, N, LAYOUT, AVG>(src, stride, dst, h, f);
            } else {
                column_sliding::<N, LAYOUT, AVG>(src, stride, dst, h, f);
            }
        }
    }
    #[inline(always)]
    unsafe fn store_second<const N: usize, const AVG: bool>(dst: *mut u8, pair: [__m128i; 2]) {
        // SAFETY: as for `store`; narrow pairs keep the second row in
        // bytes 8.., 16-wide pairs are two separate rows.
        unsafe {
            if N == 16 {
                Self::store::<N, AVG>(dst, pair[1]);
            } else {
                Self::store::<N, AVG>(dst, _mm_srli_si128::<8>(pair[1]));
            }
        }
    }
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn convolve(
    kind: Vp9Filter,
    src: &[u8],
    stride: usize,
    phase: (usize, usize),
    size: (usize, usize),
    dst: &mut [u8],
    dst_stride: usize,
    average: bool,
) {
    // SAFETY: dispatch validated the input and this entry enables AVX2.
    unsafe {
        if average {
            super::driver::entry::<Avx2, true>(kind, src, stride, phase, size, dst, dst_stride);
        } else {
            super::driver::entry::<Avx2, false>(kind, src, stride, phase, size, dst, dst_stride);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::driver::tests::{KINDS, magnitudes};
    use super::*;

    /// The tap pairs `vpmaddubsw` sums for a table: bilinear's centre pair, or
    /// the four aligned pairs of an 8-tap table.
    fn pairs(kind: Vp9Filter, taps: &[i16; 8]) -> Vec<&[i16]> {
        if kind == Vp9Filter::Bilinear {
            vec![&taps[3..5]]
        } else {
            taps.chunks(2).collect()
        }
    }

    #[test]
    fn pairs_fit_i16() {
        for kind in KINDS {
            for taps in &kind.kernel()[1..] {
                for pair in pairs(kind, taps) {
                    let (positive, negative) = magnitudes(pair);
                    assert!(positive * 255 <= i32::from(i16::MAX), "{kind:?} {taps:?}");
                    assert!(-negative * 255 >= i32::from(i16::MIN), "{kind:?} {taps:?}");
                }
            }
        }
    }

    #[test]
    fn offset_window_holds_every_sum() {
        let bias = i32::from(ROUND) + i32::from(OFFSET);
        for kind in KINDS {
            for taps in &kind.kernel()[1..] {
                let (positive, negative) = magnitudes(taps);
                assert!(bias - negative * 255 >= 0, "{kind:?} {taps:?}");
                assert!(
                    bias + positive * 255 <= i32::from(u16::MAX),
                    "{kind:?} {taps:?}"
                );
            }
        }
    }
}
