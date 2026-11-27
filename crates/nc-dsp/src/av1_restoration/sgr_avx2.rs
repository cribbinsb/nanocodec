//! The AVX2 arithmetic of the self-guided stripe kernel ([`super::sgr_stripe`],
//! which holds the schedule and the layout): a port of dav1d 1.5.4
//! `src/x86/looprestoration_avx2.asm` (`sgr_filter_5x5_8bpc`,
//! `sgr_filter_3x3_8bpc`, `sgr_filter_mix_8bpc`; BSD-2-Clause, see
//! `THIRD_PARTY.md`). The bytes of a step are widened to 16-bit taps by
//! `pshufb`, the squares summed by `pmaddwd` on tap pairs (a [`SumRow`]
//! keeps them in `punpck` order: columns 0–3 and 8–11, then 4–7 and
//! 12–15); the mixed projection blends both differences into one
//! `pmaddwd`, the single one is `src + pmulhrsw(d, 16 · w)`.
//!
//! `x` is `round(256 / (z + 1))` from `rcpps`, exact over `0 < z < 255` (see
//! the proof at [`super::avx2`]'s `sgr_factor`), capped at 255 and 0 for
//! `z ≥ 255`.

#![allow(
    clippy::many_single_char_names,
    reason = "the spec's and dav1d's sample, tap and statistic names"
)]
#![allow(clippy::inline_always)]
#![allow(unsafe_code)]

use core::arch::x86_64::{
    __m256i, _mm_loadl_epi64, _mm_loadu_si128, _mm_packs_epi32, _mm_packus_epi16,
    _mm_shuffle_epi32, _mm_storel_epi64, _mm_storeu_si128, _mm256_add_epi16, _mm256_add_epi32,
    _mm256_and_si256, _mm256_andnot_si256, _mm256_blend_epi16, _mm256_castps_si256,
    _mm256_castsi128_si256, _mm256_castsi256_ps, _mm256_castsi256_si128, _mm256_cmpgt_epi32,
    _mm256_cvtepi32_ps, _mm256_cvtepu8_epi32, _mm256_cvtps_epi32, _mm256_extracti128_si256,
    _mm256_inserti128_si256, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_min_epi16,
    _mm256_mul_ps, _mm256_mulhrs_epi16, _mm256_mullo_epi16, _mm256_mullo_epi32, _mm256_or_si256,
    _mm256_packs_epi32, _mm256_rcp_ps, _mm256_set1_epi16, _mm256_set1_epi32, _mm256_set1_ps,
    _mm256_setr_epi8, _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_shuffle_ps,
    _mm256_slli_epi32, _mm256_srai_epi32, _mm256_srli_epi32, _mm256_storeu_si256, _mm256_sub_epi32,
    _mm256_unpackhi_epi16, _mm256_unpacklo_epi16,
};

use nc_common::PlaneViewMut;

use super::self_guided::{MTABLE_BITS, SGR_BITS, SgrStripe, SgrStripeScratch, one_by_x};
use super::sgr_stripe::{
    B_ROUND, BLEND_SHIFT, Kept, ONE_ROW_SHIFT, STEP, Single, StripeLanes, SumRow, TWO_ROW_SHIFT,
    X_BITS,
};

/// Output columns per step of the neighbourhoods and the mixed projection
/// (eight 32-bit lanes).
const MIX_STEP: usize = 8;
/// `(1 << (MTABLE_BITS - 1)) + (1 << MTABLE_BITS)`: rounds `z = p · s >>
/// 20` and adds one, giving `z + 1`, the reciprocal's denominator.
const Z_ROUND_PLUS_ONE: i32 = 3 << (MTABLE_BITS - 1);

// --------------------------------------------------- horizontal sums ----

/// One step's box sums: 16 sums in 16 bits, their squares in 32.
#[derive(Clone, Copy)]
struct Sums {
    sum: __m256i,
    squares: [__m256i; 2],
}

impl Sums {
    #[inline(always)]
    unsafe fn add(self, other: Self) -> Self {
        // SAFETY: AVX2 is enabled in the calling kernel.
        unsafe {
            Self {
                sum: _mm256_add_epi16(self.sum, other.sum),
                squares: [
                    _mm256_add_epi32(self.squares[0], other.squares[0]),
                    _mm256_add_epi32(self.squares[1], other.squares[1]),
                ],
            }
        }
    }
}

/// `pshufb` patterns widening bytes `k ..= k + 7` of each 128-bit lane to
/// 16-bit lanes (dav1d's `sgr_shuf`).
#[derive(Clone, Copy)]
struct Taps {
    shuffle: [__m256i; 5],
}

impl Taps {
    #[inline(always)]
    unsafe fn new() -> Self {
        let pattern = |k: i8| {
            // SAFETY: AVX2 is enabled in the calling kernel.
            unsafe {
                _mm256_setr_epi8(
                    k,
                    -1,
                    k + 1,
                    -1,
                    k + 2,
                    -1,
                    k + 3,
                    -1,
                    k + 4,
                    -1,
                    k + 5,
                    -1,
                    k + 6,
                    -1,
                    k + 7,
                    -1,
                    k,
                    -1,
                    k + 1,
                    -1,
                    k + 2,
                    -1,
                    k + 3,
                    -1,
                    k + 4,
                    -1,
                    k + 5,
                    -1,
                    k + 6,
                    -1,
                    k + 7,
                    -1,
                )
            }
        };
        Self {
            shuffle: [pattern(0), pattern(1), pattern(2), pattern(3), pattern(4)],
        }
    }

    /// The window bytes of one step: box column `c` of the step reads
    /// bytes `c ..= c + 4` (radius 2) of `src`; the high lane starts 8 on.
    #[inline(always)]
    unsafe fn bytes(src: *const u8) -> __m256i {
        // SAFETY: the caller's window row holds 24 bytes from `src`.
        unsafe {
            _mm256_inserti128_si256::<1>(
                _mm256_castsi128_si256(_mm_loadu_si128(src.cast())),
                _mm_loadu_si128(src.add(8).cast()),
            )
        }
    }

    /// Tap `k` (0 to 4) of each of the step's 16 box columns.
    #[inline(always)]
    unsafe fn tap(self, bytes: __m256i, k: usize) -> __m256i {
        // SAFETY: AVX2 is enabled in the calling kernel.
        unsafe { _mm256_shuffle_epi8(bytes, self.shuffle[k]) }
    }
}

/// `Σ a² + b²` of 16-bit lane pairs, in `punpck` order.
#[inline(always)]
unsafe fn pair_squares(a: __m256i, b: __m256i) -> [__m256i; 2] {
    // SAFETY: AVX2 is enabled in the calling kernel.
    unsafe {
        let lo = _mm256_unpacklo_epi16(a, b);
        let hi = _mm256_unpackhi_epi16(a, b);
        [_mm256_madd_epi16(lo, lo), _mm256_madd_epi16(hi, hi)]
    }
}

/// Accumulate both halves of a word-pair sum of squares. Ice Lake's
/// VNNI fuses each pair of products and the running square sum.
#[inline(always)]
unsafe fn add_squares<const ICL: bool>(acc: [__m256i; 2], a: __m256i, b: __m256i) -> [__m256i; 2] {
    // SAFETY: called in the selected complete-stripe target feature.
    unsafe {
        if ICL {
            let lo = _mm256_unpacklo_epi16(a, b);
            let hi = _mm256_unpackhi_epi16(a, b);
            [
                core::arch::x86_64::_mm256_dpwssd_epi32(acc[0], lo, lo),
                core::arch::x86_64::_mm256_dpwssd_epi32(acc[1], hi, hi),
            ]
        } else {
            add_pairs(acc, pair_squares(a, b))
        }
    }
}

/// The centre tap's square (at most `255²`, exact in 16 bits), widened.
#[inline(always)]
unsafe fn centre_square(t: __m256i) -> [__m256i; 2] {
    // SAFETY: AVX2 is enabled in the calling kernel.
    unsafe {
        let square = _mm256_mullo_epi16(t, t);
        let zero = _mm256_setzero_si256();
        [
            _mm256_unpacklo_epi16(square, zero),
            _mm256_unpackhi_epi16(square, zero),
        ]
    }
}

#[inline(always)]
unsafe fn add_pairs(a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
    // SAFETY: AVX2 is enabled in the calling kernel.
    unsafe { [_mm256_add_epi32(a[0], b[0]), _mm256_add_epi32(a[1], b[1])] }
}

/// Tap 2 from taps 0 and 4 (`shufps`, dav1d's saving of one `pshufb`):
/// columns 0–3 are tap 0's columns 2–5, columns 4–7 tap 4's columns 2–5.
#[inline(always)]
unsafe fn middle_tap(t0: __m256i, t4: __m256i) -> __m256i {
    /// dav1d's `q2121`.
    const MIDDLE: i32 = 0b10_01_10_01;
    // SAFETY: AVX2 is enabled in the calling kernel.
    unsafe {
        _mm256_castps_si256(_mm256_shuffle_ps::<MIDDLE>(
            _mm256_castsi256_ps(t0),
            _mm256_castsi256_ps(t4),
        ))
    }
}

/// Radius-1 sums of one step (taps 1 to 3).
#[inline(always)]
unsafe fn sums3<const ICL: bool>(taps: Taps, bytes: __m256i) -> Sums {
    // SAFETY: AVX2 is enabled in the calling kernel.
    unsafe {
        let (t1, t2, t3) = (taps.tap(bytes, 1), taps.tap(bytes, 2), taps.tap(bytes, 3));
        Sums {
            sum: _mm256_add_epi16(_mm256_add_epi16(t1, t2), t3),
            squares: add_squares::<ICL>(centre_square(t2), t1, t3),
        }
    }
}

/// Radius-1 and radius-2 sums of one step from one set of taps.
#[inline(always)]
unsafe fn sums35<const ICL: bool>(taps: Taps, bytes: __m256i) -> (Sums, Sums) {
    // SAFETY: AVX2 is enabled in the calling kernel.
    unsafe {
        let (t0, t4) = (taps.tap(bytes, 0), taps.tap(bytes, 4));
        let (t1, t3) = (taps.tap(bytes, 1), taps.tap(bytes, 3));
        let t2 = middle_tap(t0, t4);
        let three = Sums {
            sum: _mm256_add_epi16(_mm256_add_epi16(t1, t2), t3),
            squares: add_squares::<ICL>(centre_square(t2), t1, t3),
        };
        let five = Sums {
            sum: _mm256_add_epi16(three.sum, _mm256_add_epi16(t0, t4)),
            squares: add_squares::<ICL>(three.squares, t0, t4),
        };
        (three, five)
    }
}

/// Radius-2 sums of one step.
#[inline(always)]
unsafe fn sums5<const ICL: bool>(taps: Taps, bytes: __m256i) -> Sums {
    // SAFETY: AVX2 is enabled in the calling kernel.
    unsafe {
        let (t0, t4) = (taps.tap(bytes, 0), taps.tap(bytes, 4));
        let (t1, t3) = (taps.tap(bytes, 1), taps.tap(bytes, 3));
        let t2 = middle_tap(t0, t4);
        let outer = _mm256_add_epi16(_mm256_add_epi16(t0, t4), _mm256_add_epi16(t1, t3));
        Sums {
            sum: _mm256_add_epi16(outer, t2),
            squares: add_squares::<ICL>(add_squares::<ICL>(centre_square(t2), t1, t3), t0, t4),
        }
    }
}

// ------------------------------------------------------- statistics ----

/// The constants of one radius's statistics.
#[derive(Clone, Copy)]
struct Statistics {
    /// Box size `n` (9 or 25).
    count: i32,
    /// `s`.
    scale: __m256i,
    /// `one_by_x(n)` in 16-bit lane 0 of each 32-bit lane.
    reciprocal: __m256i,
}

impl Statistics {
    #[inline(always)]
    fn new(count: i32, scale: u32) -> Self {
        // SAFETY: AVX2 is enabled in the calling kernel.
        unsafe {
            Self {
                count,
                // `s` is below 2^12 (`Sgr_Params`).
                scale: _mm256_set1_epi32(scale as i32),
                reciprocal: _mm256_set1_epi32(one_by_x(count as u32) as i32),
            }
        }
    }

    /// The packed `x | B << 12` words of one step's box sums, in `punpck`
    /// order (AV1 §7.17.3 `p`, `z`, `a2`, `b2`).
    #[inline(always)]
    unsafe fn packed(self, box_sums: Sums) -> [__m256i; 2] {
        // SAFETY: AVX2 is enabled in the calling kernel.
        unsafe {
            let zero = _mm256_setzero_si256();
            let sums = [
                _mm256_unpacklo_epi16(box_sums.sum, zero),
                _mm256_unpackhi_epi16(box_sums.sum, zero),
            ];
            [
                self.packed_half(sums[0], box_sums.squares[0]),
                self.packed_half(sums[1], box_sums.squares[1]),
            ]
        }
    }

    /// [`Self::packed`] of 8 columns: box sums and sums of squares in
    /// 32-bit lanes.
    #[inline(always)]
    unsafe fn packed_half(self, sum: __m256i, square: __m256i) -> __m256i {
        // SAFETY: AVX2 is enabled in the calling kernel.
        unsafe {
            let unit = _mm256_set1_ps(f32::from(1_u16 << SGR_BITS));
            let unit_bits = _mm256_castps_si256(unit);
            let b_mask = _mm256_set1_epi32(-1 << X_BITS);
            let rounding = _mm256_set1_epi32(B_ROUND);
            let z_round = _mm256_set1_epi32(Z_ROUND_PLUS_ONE);
            {
                // p = n · Σx² − (Σx)², at most about 2^23 (radius 2) and
                // 1.3 · 2^20 (radius 1): exact; `p · s` below 2^32.
                let scaled_squares = if self.count == 25 {
                    _mm256_mullo_epi32(square, _mm256_set1_epi32(25))
                } else {
                    _mm256_add_epi32(_mm256_slli_epi32::<3>(square), square)
                };
                let p = _mm256_sub_epi32(scaled_squares, _mm256_madd_epi16(sum, sum));
                let z1 = _mm256_srli_epi32::<{ MTABLE_BITS as i32 }>(_mm256_add_epi32(
                    _mm256_mullo_epi32(p, self.scale),
                    z_round,
                ));
                let z1_float = _mm256_cvtepi32_ps(z1);
                let x = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_rcp_ps(z1_float), unit));
                // Positive floats order as their bits: z + 1 < 256 caps x
                // at 255 (z = 0), else x = 0 (A = 256).
                let cap = _mm256_srli_epi32::<24>(_mm256_cmpgt_epi32(
                    unit_bits,
                    _mm256_castps_si256(z1_float),
                ));
                let x = _mm256_min_epi16(x, cap);
                let b = _mm256_and_si256(
                    _mm256_add_epi32(
                        _mm256_mullo_epi32(_mm256_madd_epi16(sum, self.reciprocal), x),
                        rounding,
                    ),
                    b_mask,
                );
                _mm256_or_si256(b, x)
            }
        }
    }
}

/// Store one step's packed words in column order (dav1d's four 128-bit
/// stores undo the `punpck` order).
#[inline(always)]
unsafe fn store_packed(dst: *mut i32, k: usize, [lo, hi]: [__m256i; 2]) {
    // SAFETY: the caller's step is inside the row (`Layout`).
    unsafe {
        let dst = dst.add(k * STEP);
        _mm_storeu_si128(dst.cast(), _mm256_castsi256_si128(lo));
        _mm_storeu_si128(dst.add(4).cast(), _mm256_castsi256_si128(hi));
        _mm_storeu_si128(dst.add(8).cast(), _mm256_extracti128_si256::<1>(lo));
        _mm_storeu_si128(dst.add(12).cast(), _mm256_extracti128_si256::<1>(hi));
    }
}

// ------------------------------------------------- neighbourhoods ----

/// Split packed words into `x` sums and `B` sums.
#[inline(always)]
unsafe fn unpack(v: __m256i) -> (__m256i, __m256i) {
    // SAFETY: AVX2 is enabled in the calling kernel.
    unsafe {
        let x_mask = _mm256_set1_epi32(-1 << X_BITS);
        (
            _mm256_andnot_si256(x_mask, v),
            _mm256_srli_epi32::<X_BITS>(v),
        )
    }
}

/// Packed words of box columns `j − 1`, `j`, `j + 1` for 8 output columns
/// from `j` (`packed` starts at box column −1).
#[inline(always)]
unsafe fn neighbours(packed: *const i32, j: usize) -> [__m256i; 3] {
    // SAFETY: the caller's columns are inside the row (`Layout`).
    unsafe {
        [
            _mm256_loadu_si256(packed.add(j).cast()),
            _mm256_loadu_si256(packed.add(j + 1).cast()),
            _mm256_loadu_si256(packed.add(j + 2).cast()),
        ]
    }
}

/// A radius-2 box row's weights 5, 6, 5 (one packed add: at most
/// `16 · 255` in the `x` field).
#[inline(always)]
unsafe fn weigh565([l, c, r]: [__m256i; 3]) -> __m256i {
    // SAFETY: AVX2 is enabled in the calling kernel.
    unsafe {
        let s = _mm256_add_epi32(_mm256_add_epi32(l, c), r);
        _mm256_add_epi32(_mm256_add_epi32(_mm256_slli_epi32::<2>(s), s), c)
    }
}

/// A radius-1 box row's weights 2, 2, 2 and 3, 4, 3 (together 16 a side:
/// the centre row's 4, 4, 4 is split between the two halves).
#[inline(always)]
unsafe fn weigh222_343([l, c, r]: [__m256i; 3]) -> (__m256i, __m256i) {
    // SAFETY: AVX2 is enabled in the calling kernel.
    unsafe {
        let outer = _mm256_add_epi32(l, r);
        let s = _mm256_add_epi32(outer, c);
        let w222 = _mm256_add_epi32(s, s);
        (w222, _mm256_sub_epi32(_mm256_add_epi32(w222, w222), outer))
    }
}

/// `B − x · src` of one radius, with its rounding (in `B`), shifted.
#[inline(always)]
unsafe fn difference<const SHIFT: i32>(x: __m256i, b: __m256i, src: __m256i) -> __m256i {
    // SAFETY: AVX2 is enabled in the calling kernel. `x` and `src` are
    // below 2^15, so `pmaddwd` is their product.
    unsafe { _mm256_srai_epi32::<SHIFT>(_mm256_sub_epi32(b, _mm256_madd_epi16(x, src))) }
}

/// The radius-1 `x` and `B` sums of output row `i` from the new box row
/// `i + 1`: the 222 row of box row `i` with the 343 rows of `i − 1` and
/// `i + 1`. Leaves box row `i + 1`'s 222 and 343 rows for row `i + 1`.
#[inline(always)]
unsafe fn radius1_sums(packed: *const i32, (ab222, ab343): Kept, j: usize) -> (__m256i, __m256i) {
    // SAFETY: AVX2 is enabled in the calling kernel; columns inside rows.
    unsafe {
        let (new222, new343) = weigh222_343(neighbours(packed, j));
        let old222 = _mm256_loadu_si256(ab222.add(j).cast());
        let above = _mm256_add_epi32(old222, _mm256_loadu_si256(ab343.add(j).cast()));
        let below = _mm256_add_epi32(old222, new343);
        _mm256_storeu_si256(ab222.add(j).cast(), new222);
        _mm256_storeu_si256(ab343.add(j).cast(), new343);
        let (above, below) = (unpack(above), unpack(below));
        (
            _mm256_add_epi32(above.0, below.0),
            _mm256_add_epi32(above.1, below.1),
        )
    }
}

/// Keep a radius-1 box row's 222 and 343 rows (the first two box rows,
/// which have no output row of their own).
#[inline(always)]
unsafe fn radius1_start(packed: *const i32, (ab222, ab343): Kept, w: usize) {
    // SAFETY: AVX2 is enabled in the calling kernel; columns inside rows.
    unsafe {
        for j in (0..w).step_by(MIX_STEP) {
            let (w222, w343) = weigh222_343(neighbours(packed, j));
            _mm256_storeu_si256(ab222.add(j).cast(), w222);
            _mm256_storeu_si256(ab343.add(j).cast(), w343);
        }
    }
}

/// Keep the first radius-2 box row's 565 sums.
#[inline(always)]
unsafe fn radius2_start(packed: *const i32, (x565, b565): Kept, w: usize) {
    // SAFETY: AVX2 is enabled in the calling kernel; columns inside rows.
    unsafe {
        for j in (0..w).step_by(MIX_STEP) {
            let (x, b) = unpack(weigh565(neighbours(packed, j)));
            _mm256_storeu_si256(x565.add(j).cast(), x);
            _mm256_storeu_si256(b565.add(j).cast(), b);
        }
    }
}

/// The radius-2 `x` and `B` sums of an output row: between two box rows
/// (the new one's 565 sums plus those kept, which it replaces), or on the
/// kept one.
#[inline(always)]
unsafe fn radius2_sums<const BETWEEN: bool>(
    packed: *const i32,
    (x565, b565): Kept,
    j: usize,
) -> (__m256i, __m256i) {
    // SAFETY: AVX2 is enabled in the calling kernel; columns inside rows.
    unsafe {
        let kept = (
            _mm256_loadu_si256(x565.add(j).cast()),
            _mm256_loadu_si256(b565.add(j).cast()),
        );
        if !BETWEEN {
            return kept;
        }
        let (x, b) = unpack(weigh565(neighbours(packed, j)));
        _mm256_storeu_si256(x565.add(j).cast(), x);
        _mm256_storeu_si256(b565.add(j).cast(), b);
        (_mm256_add_epi32(x, kept.0), _mm256_add_epi32(b, kept.1))
    }
}

// ----------------------------------------------------------- output ----

/// Write 8 output samples (32-bit lanes, saturated to bytes) from column
/// `j`, or those left at the row's end.
#[inline(always)]
unsafe fn store8(dst: &mut [u8], j: usize, v: __m256i) {
    // SAFETY: AVX2 is enabled in the calling kernel; the 8-byte store is
    // inside `dst` (checked), else the tail goes through a local.
    unsafe {
        let words = _mm_packs_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
        let bytes = _mm_packus_epi16(words, words);
        if j + MIX_STEP <= dst.len() {
            _mm_storel_epi64(dst.as_mut_ptr().add(j).cast(), bytes);
        } else {
            let mut tail = [0_u8; 16];
            _mm_storeu_si128(tail.as_mut_ptr().cast(), bytes);
            let n = dst.len() - j;
            dst[j..].copy_from_slice(&tail[..n]);
        }
    }
}

/// The two passes' projection weights, `w0` in 16-bit lane 0 and `w1` in
/// lane 1, scaled by 4 so the rounding is dav1d's `−4096`... `>> 13`.
#[inline(always)]
unsafe fn mix_weights([w0, w1]: [i32; 2]) -> __m256i {
    // SAFETY: AVX2 is enabled in the calling kernel. |4 · w| < 2^15.
    unsafe { _mm256_set1_epi32(((w1 << 2) << 16) | ((w0 << 2) & 0xffff)) }
}

/// One output row of a mixed set: `src + ((w0 · d5 + w1 · d3 + 2^10) >>
/// 11)` (dav1d's `pblendw` of the two differences into one `pmaddwd`).
#[inline(always)]
unsafe fn mix_row<const BETWEEN: bool>(
    src: &[u8],
    dst: &mut [u8],
    (packed5, five): (*const i32, Kept),
    (packed3, three): (*const i32, Kept),
    weights: [i32; 2],
) {
    let w = dst.len();
    assert!(src.len() >= w.next_multiple_of(MIX_STEP));
    // SAFETY: AVX2 is enabled in the calling kernel; `src` covers each
    // 8-byte load (checked), the rows each step's columns (`Layout`).
    unsafe {
        let weights = mix_weights(weights);
        let rounding = _mm256_set1_epi32(1 << (BLEND_SHIFT + 1));
        for j in (0..w).step_by(MIX_STEP) {
            let s = _mm256_cvtepu8_epi32(_mm_loadl_epi64(src.as_ptr().add(j).cast()));
            let (x5, b5) = radius2_sums::<BETWEEN>(packed5, five, j);
            let (x3, b3) = radius1_sums(packed3, three, j);
            // The differences' low 16 bits (they fit): radius 2's from a
            // logical shift, radius 1's in the high half after `<< 7`.
            let d5 = _mm256_sub_epi32(b5, _mm256_madd_epi16(x5, s));
            let d5 = if BETWEEN {
                _mm256_srli_epi32::<TWO_ROW_SHIFT>(d5)
            } else {
                _mm256_srli_epi32::<ONE_ROW_SHIFT>(d5)
            };
            let d3 = _mm256_slli_epi32::<{ 16 - TWO_ROW_SHIFT }>(_mm256_sub_epi32(
                b3,
                _mm256_madd_epi16(x3, s),
            ));
            let pair = _mm256_blend_epi16::<0b1010_1010>(d5, d3);
            let delta = _mm256_srai_epi32::<{ BLEND_SHIFT + 2 }>(_mm256_add_epi32(
                _mm256_madd_epi16(pair, weights),
                rounding,
            ));
            store8(dst, j, _mm256_add_epi32(delta, s));
        }
    }
}

/// The `x` and `B` sums of 8 columns from `j` of `kind`'s neighbourhood,
/// and the difference.
#[inline(always)]
unsafe fn single_difference(
    kind: Single,
    packed: *const i32,
    kept: Kept,
    j: usize,
    src: __m256i,
) -> __m256i {
    // SAFETY: as the sums' functions.
    unsafe {
        match kind {
            Single::FiveBetween => {
                let (x, b) = radius2_sums::<true>(packed, kept, j);
                difference::<TWO_ROW_SHIFT>(x, b, src)
            }
            Single::FiveOn => {
                let (x, b) = radius2_sums::<false>(packed, kept, j);
                difference::<ONE_ROW_SHIFT>(x, b, src)
            }
            Single::Three => {
                let (x, b) = radius1_sums(packed, kept, j);
                difference::<TWO_ROW_SHIFT>(x, b, src)
            }
        }
    }
}

/// One output row of a single-pass set, 16 columns a step: `src +
/// pmulhrsw(d, 16 · w)`, which is `src + ((w · d + 2^10) >> 11)`.
#[inline(always)]
unsafe fn single_row(
    src: &[u8],
    dst: &mut [u8],
    weight: i32,
    (kind, packed, kept): (Single, *const i32, Kept),
) {
    let w = dst.len();
    assert!(src.len() >= w.next_multiple_of(STEP));
    // SAFETY: AVX2 is enabled in the calling kernel; `src` covers each
    // 16-byte load (checked).
    unsafe {
        // |16 · w| < 2^15 for every legal weight.
        let weight = _mm256_set1_epi16((weight << (15 - BLEND_SHIFT)) as i16);
        for j in (0..w).step_by(STEP) {
            let bytes = _mm_loadu_si128(src.as_ptr().add(j).cast());
            let s = [
                _mm256_cvtepu8_epi32(bytes),
                _mm256_cvtepu8_epi32(_mm_shuffle_epi32::<0b11_10>(bytes)),
            ];
            let d = [
                single_difference(kind, packed, kept, j, s[0]),
                single_difference(kind, packed, kept, j + MIX_STEP, s[1]),
            ];
            // Both packs interleave the 128-bit halves alike.
            let v = _mm256_add_epi16(
                _mm256_mulhrs_epi16(_mm256_packs_epi32(d[0], d[1]), weight),
                _mm256_packs_epi32(s[0], s[1]),
            );
            let bytes =
                _mm_packus_epi16(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
            let bytes = _mm_shuffle_epi32::<0b11_01_10_00>(bytes);
            if j + STEP <= w {
                _mm_storeu_si128(dst.as_mut_ptr().add(j).cast(), bytes);
            } else {
                let mut tail = [0_u8; STEP];
                _mm_storeu_si128(tail.as_mut_ptr().cast(), bytes);
                dst[j..].copy_from_slice(&tail[..w - j]);
            }
        }
    }
}

mod icl;

/// The shared x86 lanes of [`super::sgr_stripe`].
struct X86<const ICL: bool>;

impl<const ICL: bool> StripeLanes for X86<ICL> {
    type Sums = Sums;
    type Stats = Statistics;

    #[inline(always)]
    unsafe fn statistics(count: i32, scale: u32) -> Statistics {
        Statistics::new(count, scale)
    }

    #[inline(always)]
    unsafe fn load(row: SumRow, k: usize) -> Sums {
        // SAFETY: the caller's step is inside the row; AVX2 is enabled.
        unsafe {
            Sums {
                sum: _mm256_loadu_si256(row.sum.add(k * STEP).cast()),
                squares: [
                    _mm256_loadu_si256(row.squares.add(k * STEP).cast()),
                    _mm256_loadu_si256(row.squares.add(k * STEP + 8).cast()),
                ],
            }
        }
    }

    #[inline(always)]
    unsafe fn store(row: SumRow, k: usize, v: Sums) {
        // SAFETY: as `load`.
        unsafe {
            _mm256_storeu_si256(row.sum.add(k * STEP).cast(), v.sum);
            _mm256_storeu_si256(row.squares.add(k * STEP).cast(), v.squares[0]);
            _mm256_storeu_si256(row.squares.add(k * STEP + 8).cast(), v.squares[1]);
        }
    }

    #[inline(always)]
    unsafe fn add(a: Sums, b: Sums) -> Sums {
        // SAFETY: AVX2 is enabled in the calling kernel.
        unsafe { a.add(b) }
    }

    #[inline(always)]
    unsafe fn sums3(src: *const u8) -> Sums {
        // SAFETY: AVX2 is enabled; the caller's row holds 24 bytes.
        unsafe { sums3::<ICL>(Taps::new(), Taps::bytes(src)) }
    }

    #[inline(always)]
    unsafe fn sums5(src: *const u8) -> Sums {
        // SAFETY: as `sums3`.
        unsafe { sums5::<ICL>(Taps::new(), Taps::bytes(src)) }
    }

    #[inline(always)]
    unsafe fn sums35(src: *const u8) -> (Sums, Sums) {
        // SAFETY: as `sums3`.
        unsafe { sums35::<ICL>(Taps::new(), Taps::bytes(src)) }
    }

    #[inline(always)]
    unsafe fn store_packed(stats: Statistics, sums: Sums, dst: *mut i32, k: usize) {
        // SAFETY: the step lies inside the row and the appropriate x86
        // features are enabled by the stripe entry point.
        unsafe {
            if ICL {
                icl::store_packed(stats, sums, dst.add(k * STEP));
            } else {
                store_packed(dst, k, stats.packed(sums));
            }
        }
    }

    #[inline(always)]
    unsafe fn radius1_start(packed: *const i32, kept: Kept, w: usize) {
        // SAFETY: as the function's contract.
        unsafe { radius1_start(packed, kept, w) }
    }

    #[inline(always)]
    unsafe fn radius2_start(packed: *const i32, kept: Kept, w: usize) {
        // SAFETY: as the function's contract.
        unsafe { radius2_start(packed, kept, w) }
    }

    #[inline(always)]
    unsafe fn mix_row<const BETWEEN: bool>(
        src: &[u8],
        dst: &mut [u8],
        five: (*const i32, Kept),
        three: (*const i32, Kept),
        weights: [i32; 2],
    ) {
        // SAFETY: as the function's contract.
        unsafe {
            if ICL {
                icl::mix_row::<BETWEEN>(src, dst, five, three, weights);
            } else {
                mix_row::<BETWEEN>(src, dst, five, three, weights);
            }
        }
    }

    #[inline(always)]
    unsafe fn single_row(
        src: &[u8],
        dst: &mut [u8],
        weight: i32,
        neighbourhood: (Single, *const i32, Kept),
    ) {
        // SAFETY: as the function's contract.
        unsafe {
            if ICL {
                icl::single_row(src, dst, weight, neighbourhood);
            } else {
                single_row(src, dst, weight, neighbourhood);
            }
        }
    }
}

/// Self-guided-filter a stripe region ([`super::sgr_stripe::run`]).
///
/// # Safety
///
/// AVX2 must be available. Extents are checked by `run`.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn sgr_stripe(
    params: SgrStripe,
    window: (&[u8], usize),
    size: (usize, usize),
    scratch: &mut SgrStripeScratch,
    dst: (&mut PlaneViewMut<'_>, usize, usize),
) {
    // SAFETY: AVX2 is enabled (this function's target feature).
    unsafe { super::sgr_stripe::run::<X86<false>>(params, window, size, scratch, dst) }
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
pub(super) unsafe fn sgr_stripe_icl(
    params: SgrStripe,
    window: (&[u8], usize),
    size: (usize, usize),
    scratch: &mut SgrStripeScratch,
    dst: (&mut PlaneViewMut<'_>, usize, usize),
) {
    // SAFETY: this entry enables the instructions of the complete stripe.
    unsafe { super::sgr_stripe::run::<X86<true>>(params, window, size, scratch, dst) }
}
