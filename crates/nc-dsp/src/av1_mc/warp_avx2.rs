//! The AVX2 8x8 warp kernel (AV1 §7.11.3.5), a port of dav1d 1.5.4
//! `warp_affine_8x8_8bpc_avx2` (`src/x86/mc_avx2.asm`, BSD-2-Clause, see
//! `THIRD_PARTY.md`).
//!
//! Horizontal pass, per window row: each output's 8 source bytes are
//! shuffled into the order of [`WARPED_FILTERS_PAIRED`] (taps 0, 2, 4, 6,
//! 1, 3, 5, 7) so that one `pmaddubsw` gives four 2-tap sums per output and
//! `phaddw` two 4-tap sums, all within 16 bits (`warp_kernels` tests);
//! `pmaddwd` by `2^13` adds the two halves in 32 bits, and adding `2^15`
//! leaves `Round2(sum, 3)` in the upper 16 bits. Two outputs per 64-bit
//! half, outputs 0–3 in the low 128-bit lane and 4–7 in the high one.
//!
//! Vertical pass, per output row: consecutive intermediate rows are
//! interleaved by column once, the eight columns' filters transposed into
//! tap pairs, and four `pmaddwd` give the 32-bit sums of all eight columns.

#![allow(unsafe_code, clippy::inline_always)]

use core::arch::x86_64::{
    __m128i, __m256i, _mm_cvtsi32_si128, _mm_loadl_epi64, _mm_loadu_si128, _mm256_add_epi32,
    _mm256_blend_epi32, _mm256_broadcastsi128_si256, _mm256_castsi128_si256,
    _mm256_castsi256_si128, _mm256_hadd_epi16, _mm256_inserti128_si256, _mm256_loadu_si256,
    _mm256_loadu2_m128i, _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_max_epi32,
    _mm256_min_epi32, _mm256_mullo_epi32, _mm256_packs_epi32, _mm256_permute4x64_epi64,
    _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setr_epi32, _mm256_setzero_si256,
    _mm256_shuffle_epi8, _mm256_sra_epi32, _mm256_srai_epi32, _mm256_storeu_si256,
    _mm256_unpackhi_epi16, _mm256_unpackhi_epi32, _mm256_unpackhi_epi64, _mm256_unpacklo_epi16,
    _mm256_unpacklo_epi32, _mm256_unpacklo_epi64,
};

use super::warp::{PHASE_ROUND, PHASE_SHIFT, WARPED_FILTERS_I16, WARPED_FILTERS_PAIRED};
use super::warp_kernels::{
    WARP_BLOCK, WINDOW_SIDE, WINDOW_STRIDE, WarpOut, WarpPhases, window_span,
};

/// Byte order of outputs 0, 1 (low lane) and 4, 5 (high lane): each
/// output's taps 0, 2, 4, 6, 1, 3, 5, 7 over its 8 source bytes.
const SHUFFLE_A: [u8; 32] = [
    0, 2, 4, 6, 1, 3, 5, 7, 1, 3, 5, 7, 2, 4, 6, 8, //
    4, 6, 8, 10, 5, 7, 9, 11, 5, 7, 9, 11, 6, 8, 10, 12,
];
/// Outputs 2, 3 (low lane) and 6, 7 (high lane).
const SHUFFLE_B: [u8; 32] = [
    2, 4, 6, 8, 3, 5, 7, 9, 3, 5, 7, 9, 4, 6, 8, 10, //
    6, 8, 10, 12, 7, 9, 11, 13, 7, 9, 11, 13, 8, 10, 12, 14,
];
/// `pmaddwd` weight that adds the two 4-tap halves shifted up by 13.
const HALF_SUM_SCALE: i16 = 1 << 13;
/// Rounding that leaves `Round2(sum, 3)` in the upper 16 bits.
const UPPER_ROUND: i32 = 1 << 15;
/// Highest filter row.
const LAST_FILTER: i32 = WARPED_FILTERS_I16.len() as i32 - 1;

/// The filter rows of eight phases `start + step · c`, clamped to the
/// table (valid models never leave it; the clamp is what makes the
/// unchecked filter lookups below sound whatever the input).
#[target_feature(enable = "avx2")]
#[inline]
fn filter_rows(start: i32, step: i32) -> [usize; WARP_BLOCK] {
    {
        let columns = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        let phases = _mm256_add_epi32(
            _mm256_set1_epi32(start + PHASE_ROUND),
            _mm256_mullo_epi32(columns, _mm256_set1_epi32(step)),
        );
        let rows = _mm256_srai_epi32::<{ PHASE_SHIFT as i32 }>(phases);
        let rows = _mm256_min_epi32(
            _mm256_max_epi32(rows, _mm256_set1_epi32(0)),
            _mm256_set1_epi32(LAST_FILTER),
        );
        let mut out = [0i32; WARP_BLOCK];
        // SAFETY: the store covers `out`.
        unsafe { _mm256_storeu_si256(out.as_mut_ptr().cast(), rows) };
        out.map(|row| row as usize)
    }
}

/// Filters `a` and `b` (rows of [`WARPED_FILTERS_PAIRED`]) for the low
/// lane and `a + 4`, `b + 4` for the high lane: `[f(a), f(b) | f(a + 4),
/// f(b + 4)]`.
#[target_feature(enable = "avx2")]
#[inline]
fn filter_pair(rows: &[usize; WARP_BLOCK], a: usize, b: usize) -> __m256i {
    let first = _mm256_inserti128_si256::<1>(
        _mm256_castsi128_si256(paired_filter(rows, a)),
        paired_filter(rows, a + 4),
    );
    let second = _mm256_inserti128_si256::<1>(
        _mm256_castsi128_si256(paired_filter(rows, b)),
        paired_filter(rows, b + 4),
    );
    _mm256_unpacklo_epi64(first, second)
}

/// Column `c`'s horizontal filter (row `rows[c]` of
/// [`WARPED_FILTERS_PAIRED`]) in the low 8 bytes.
#[target_feature(enable = "avx2")]
#[inline]
fn paired_filter(rows: &[usize; WARP_BLOCK], c: usize) -> __m128i {
    // SAFETY: `filter_rows` clamps every row into the table; the 8-byte
    // load reads that whole filter row.
    unsafe {
        let filter = WARPED_FILTERS_PAIRED.get_unchecked(rows[c]);
        _mm_loadl_epi64(filter.as_ptr().cast())
    }
}

/// One horizontal row: the 8 outputs of `source` (16 readable bytes) at
/// filter `rows`, as 16-bit lanes in order.
#[target_feature(enable = "avx2")]
#[inline]
fn horizontal_row(source: &[u8; WINDOW_STRIDE], rows: &[usize; WARP_BLOCK]) -> __m128i {
    // SAFETY: the 16-byte loads read the whole source row and patterns.
    let (src, shuffle_a, shuffle_b) = unsafe {
        (
            _mm256_broadcastsi128_si256(_mm_loadu_si128(source.as_ptr().cast())),
            _mm256_loadu_si256(SHUFFLE_A.as_ptr().cast()),
            _mm256_loadu_si256(SHUFFLE_B.as_ptr().cast()),
        )
    };
    let (taps_a, taps_b) = (filter_pair(rows, 0, 1), filter_pair(rows, 2, 3));
    let sums_a = _mm256_maddubs_epi16(_mm256_shuffle_epi8(src, shuffle_a), taps_a);
    let sums_b = _mm256_maddubs_epi16(_mm256_shuffle_epi8(src, shuffle_b), taps_b);
    let halves = _mm256_hadd_epi16(sums_a, sums_b);
    let scaled = _mm256_madd_epi16(halves, _mm256_set1_epi16(HALF_SUM_SCALE));
    let rounded = _mm256_srai_epi32::<16>(_mm256_add_epi32(scaled, _mm256_set1_epi32(UPPER_ROUND)));
    // Lanes: outputs 0-3 low, 4-7 high; pack within lanes and gather.
    let words = _mm256_packs_epi32(rounded, rounded);
    _mm256_castsi256_si128(_mm256_permute4x64_epi64::<0b1000>(words))
}

/// Columns 0–3 (low lane) and 4–7 (high lane) of two intermediate rows,
/// interleaved: dword `c` holds (`upper[c]`, `lower[c]`).
#[target_feature(enable = "avx2")]
#[inline]
fn interleave_rows(upper: __m128i, lower: __m128i) -> __m256i {
    let both_upper = _mm256_broadcastsi128_si256(upper);
    let both_lower = _mm256_broadcastsi128_si256(lower);
    let low = _mm256_unpacklo_epi16(both_upper, both_lower);
    let high = _mm256_unpackhi_epi16(both_upper, both_lower);
    // Columns 0-3 from `low`'s low lane, 4-7 from `high`'s high lane.
    _mm256_blend_epi32::<0b1111_0000>(low, high)
}

/// Column `c`'s vertical filter (row `rows[c]`) in the low lane and
/// column `c + 4`'s in the high lane.
#[target_feature(enable = "avx2")]
#[inline]
fn filter_lanes(rows: &[usize; WARP_BLOCK], c: usize) -> __m256i {
    let (low, high) = (
        &WARPED_FILTERS_I16[rows[c]],
        &WARPED_FILTERS_I16[rows[c + 4]],
    );
    // SAFETY: each 16-byte load reads a whole filter row.
    unsafe { _mm256_loadu2_m128i(high.as_ptr().cast(), low.as_ptr().cast()) }
}

/// The vertical taps of the eight columns' filter `rows`, transposed into
/// tap pairs: vector `p` holds, in dword `c` (columns 0–3 low lane, 4–7
/// high lane), taps `2p` and `2p + 1` of column `c`'s filter.
#[target_feature(enable = "avx2")]
#[inline]
fn tap_pairs(rows: &[usize; WARP_BLOCK]) -> [__m256i; 4] {
    let (f0, f1) = (filter_lanes(rows, 0), filter_lanes(rows, 1));
    let (f2, f3) = (filter_lanes(rows, 2), filter_lanes(rows, 3));
    let a01 = _mm256_unpacklo_epi32(f0, f1);
    let b01 = _mm256_unpackhi_epi32(f0, f1);
    let a23 = _mm256_unpacklo_epi32(f2, f3);
    let b23 = _mm256_unpackhi_epi32(f2, f3);
    [
        _mm256_unpacklo_epi64(a01, a23),
        _mm256_unpackhi_epi64(a01, a23),
        _mm256_unpacklo_epi64(b01, b23),
        _mm256_unpackhi_epi64(b01, b23),
    ]
}

/// The AVX2 kernel ([`super::warp_kernels::WarpFn`]).
#[target_feature(enable = "avx2")]
pub(super) unsafe fn warp8x8(
    (window, stride): (&[u8], usize),
    phases: &WarpPhases,
    round1: u32,
    out: &mut WarpOut,
) {
    assert!(window.len() >= window_span(stride), "warp window too short");
    let (h_start, alpha, beta) = phases.x;
    let mut intermediate = [_mm_cvtsi32_si128(0); WINDOW_SIDE];
    for (i, row) in intermediate.iter_mut().enumerate() {
        let source: &[u8; WINDOW_STRIDE] = window[i * stride..][..WINDOW_STRIDE]
            .try_into()
            .expect("a whole window row");
        let start = h_start + beta * (i as i32 - 3);
        *row = horizontal_row(source, &filter_rows(start, alpha));
    }
    let mut pairs = [_mm256_setzero_si256(); WINDOW_SIDE - 1];
    for (k, pair) in pairs.iter_mut().enumerate() {
        *pair = interleave_rows(intermediate[k], intermediate[k + 1]);
    }
    let (v_start, gamma, delta) = phases.y;
    let shift = _mm_cvtsi32_si128(round1 as i32);
    let round = _mm256_set1_epi32(1 << (round1 - 1));
    for (r, line) in out.as_chunks_mut::<WARP_BLOCK>().0.iter_mut().enumerate() {
        let taps = tap_pairs(&filter_rows(v_start + delta * r as i32, gamma));
        let mut sum = round;
        for (p, &tap) in taps.iter().enumerate() {
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(pairs[r + 2 * p], tap));
        }
        let rounded = _mm256_sra_epi32(sum, shift);
        // SAFETY: AVX2 is enabled; the store covers the 8 outputs of `line`.
        unsafe { _mm256_storeu_si256(line.as_mut_ptr().cast(), rounded) };
    }
}
