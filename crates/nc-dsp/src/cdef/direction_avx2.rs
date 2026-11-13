//! AVX2 CDEF direction search (AV1 §7.15.1): the same results as the scalar
//! reference (libaom `cdef_find_dir_c`), in about half the instructions of
//! the 128-bit generic search.
//!
//! The layout is dav1d's `cdef_dir_8bpc` (`src/x86/cdef_avx2.asm`,
//! BSD-2-Clause, Copyright © 2018 `VideoLAN` and dav1d authors, Copyright ©
//! 2018 Two Orioles, LLC; see `THIRD_PARTY.md`), reimplemented with
//! intrinsics: register `k` holds block row `k` in its low half and row
//! `7 - k` in its high half, so every sum built from the registers in order
//! gives one direction in the low half and its mirror image in the high
//! half at once:
//!
//! - lines along the columns and rows (directions 6 and 2);
//! - rows shifted by their index: diagonals 0 and 4 (whose lines come out
//!   reversed, which leaves the sum of squares unchanged);
//! - row pairs shifted by the pair index: directions 7 and 5;
//! - column pairs shifted by the row index: directions 1 and 3.
//!
//! No transpose is needed. Line `i` of a direction is paired with its mirror
//! line (same length) before squaring, so each `pmaddwd` gives two lines'
//! squares, weighted by `840 / length`.

use core::arch::x86_64::{
    __m256i, _mm_add_epi16, _mm_cvtsi128_si32, _mm_hadd_epi16, _mm_loadu_si128, _mm_max_epi32,
    _mm_minpos_epu16, _mm_packs_epi32, _mm_set_epi64x, _mm_shuffle_epi32, _mm_srli_epi32,
    _mm_sub_epi32, _mm256_add_epi16, _mm256_add_epi32, _mm256_broadcastsi128_si256,
    _mm256_castsi256_si128, _mm256_cvtepu8_epi16, _mm256_extracti128_si256, _mm256_hadd_epi16,
    _mm256_hadd_epi32, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_mullo_epi32,
    _mm256_permute2x128_si256, _mm256_permutevar8x32_epi32, _mm256_set_m128i, _mm256_set1_epi16,
    _mm256_set1_epi32, _mm256_setr_epi32, _mm256_shuffle_epi8, _mm256_shufflelo_epi16,
    _mm256_slli_si256, _mm256_srli_si256, _mm256_storeu_si256, _mm256_sub_epi16,
    _mm256_unpackhi_epi16, _mm256_unpacklo_epi16,
};

/// Block side.
const N: usize = 8;
/// Sample offset that centres 8-bit samples on zero.
const CENTRE: i16 = 128;
/// `840 / length` of the mirrored line pairs of the diagonal directions
/// (lengths 1..=7, then the full line 8 with nothing).
const DIAGONAL_WEIGHTS: [[i32; 4]; 2] = [[840, 420, 280, 210], [168, 140, 120, 105]];
/// `840 / length` of the short line pairs (2, 4, 6 samples) and line 3 of
/// directions 1, 3, 5 and 7.
const SKEW_SHORT_WEIGHTS: [i32; 4] = [420, 210, 140, 105];
/// `840 / 8`: lines of all eight samples.
const FULL_WEIGHT: i32 = 105;
/// The contrast is the cost margin over the orthogonal direction, shifted
/// down by this.
const VAR_SHIFT: i32 = 10;
/// Word shuffle of the diagonal overflow lines: lane `j` from lane `6 - j`
/// (line `14 - j`), lane 7 kept (zero), in both halves.
const REVERSE_7: [i8; 16] = [12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15];
/// Positions of directions 0..=7 in the final horizontal sums, whose halves
/// come out as `[0, 6, 1, 7 | 4, 2, 3, 5]`.
const COST_ORDER: [i32; 8] = [0, 2, 5, 6, 4, 7, 1, 3];

/// Every 16-bit lane of each 128-bit half moved up by `$k` lanes (zeros
/// in).
macro_rules! lanes_up {
    ($v:expr, $k:literal) => {
        _mm256_slli_si256::<{ 2 * $k }>($v)
    };
}

/// Every 16-bit lane of each 128-bit half moved down by `$k` lanes.
macro_rules! lanes_down {
    ($v:expr, $k:literal) => {
        _mm256_srli_si256::<{ 2 * $k }>($v)
    };
}

/// Cost terms of interleaved line pairs: each pair's two squares summed
/// (`pmaddwd`), times the pair's weight.
#[target_feature(enable = "avx2")]
#[inline]
fn weighted_squares(pairs: __m256i, weights: __m256i) -> __m256i {
    _mm256_mullo_epi32(_mm256_madd_epi16(pairs, pairs), weights)
}

/// The four lanes `weights` in both 128-bit halves.
#[target_feature(enable = "avx2")]
#[inline]
fn set_halves([a, b, c, d]: [i32; 4]) -> __m256i {
    _mm256_setr_epi32(a, b, c, d, a, b, c, d)
}

/// The 128-bit halves of `v` exchanged.
#[target_feature(enable = "avx2")]
#[inline]
fn swap_halves(v: __m256i) -> __m256i {
    _mm256_permute2x128_si256::<0x01>(v, v)
}

/// Cost of a skewed direction pair from four registers each shifted up by
/// its index: lines 0..=7 and the overflow lines 8..=10, then three short
/// mirrored pairs and five full lines.
#[target_feature(enable = "avx2")]
#[inline]
fn skew_cost(parts: [__m256i; 4]) -> __m256i {
    let lines = _mm256_add_epi16(
        _mm256_add_epi16(parts[0], lanes_up!(parts[1], 1)),
        _mm256_add_epi16(lanes_up!(parts[2], 2), lanes_up!(parts[3], 3)),
    );
    let overflow = _mm256_add_epi16(
        lanes_down!(parts[1], 7),
        _mm256_add_epi16(lanes_down!(parts[2], 6), lanes_down!(parts[3], 5)),
    );
    // Lines 10, 9, 8 against lines 0, 1, 2; lane 3 of the overflow is zero
    // and pairs line 3 with nothing.
    let mirrored = _mm256_shufflelo_epi16::<0b11_00_01_10>(overflow);
    let short = _mm256_unpacklo_epi16(lines, mirrored);
    let full = _mm256_unpackhi_epi16(overflow, lines);
    _mm256_add_epi32(
        weighted_squares(short, set_halves(SKEW_SHORT_WEIGHTS)),
        weighted_squares(full, _mm256_set1_epi32(FULL_WEIGHT)),
    )
}

/// libaom `cdef_find_dir` on the 8x8 block of 8-bit samples at the start of
/// `src` (row stride `stride`): the direction and the directional contrast.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn find_dir(src: &[u8], stride: usize) -> (u8, i32) {
    let block = &src[..(N - 1) * stride + N];
    let row = |i: usize| -> i64 {
        // SAFETY: row `i < 8` lies in the checked 8x8 footprint.
        unsafe {
            block
                .as_ptr()
                .add(i * stride)
                .cast::<i64>()
                .read_unaligned()
        }
    };
    let centre = _mm256_set1_epi16(CENTRE);
    // reg[k]: row k (low half) and row 7 - k (high half), as centred words;
    // the last four are the first four with their halves exchanged.
    let first: [__m256i; N / 2] = core::array::from_fn(|k| {
        let bytes = _mm_set_epi64x(row(N - 1 - k), row(k));
        _mm256_sub_epi16(_mm256_cvtepu8_epi16(bytes), centre)
    });
    let reg: [__m256i; N] = core::array::from_fn(|k| {
        if k < N / 2 {
            first[k]
        } else {
            swap_halves(first[N - 1 - k])
        }
    });

    // Directions 6 (low: column sums) and 2 (high: row sums, rows 0..3
    // then 7..4).
    let quarter = _mm256_add_epi16(
        _mm256_add_epi16(reg[0], reg[1]),
        _mm256_add_epi16(reg[2], reg[3]),
    );
    let columns = _mm_add_epi16(
        _mm256_castsi256_si128(quarter),
        _mm256_extracti128_si256::<1>(quarter),
    );
    let quads = _mm256_hadd_epi16(
        _mm256_hadd_epi16(reg[0], reg[1]),
        _mm256_hadd_epi16(reg[2], reg[3]),
    );
    let rows = _mm_hadd_epi16(
        _mm256_castsi256_si128(quads),
        _mm256_extracti128_si256::<1>(quads),
    );
    let axis = weighted_squares(
        _mm256_set_m128i(rows, columns),
        _mm256_set1_epi32(FULL_WEIGHT),
    );

    // Directions 0 (low) and 4 (high): rows shifted by their index.
    let lines = _mm256_add_epi16(
        _mm256_add_epi16(
            _mm256_add_epi16(reg[0], lanes_up!(reg[1], 1)),
            _mm256_add_epi16(lanes_up!(reg[2], 2), lanes_up!(reg[3], 3)),
        ),
        _mm256_add_epi16(
            _mm256_add_epi16(lanes_up!(reg[4], 4), lanes_up!(reg[5], 5)),
            _mm256_add_epi16(lanes_up!(reg[6], 6), lanes_up!(reg[7], 7)),
        ),
    );
    let overflow = _mm256_add_epi16(
        _mm256_add_epi16(
            _mm256_add_epi16(lanes_down!(reg[1], 7), lanes_down!(reg[2], 6)),
            _mm256_add_epi16(lanes_down!(reg[3], 5), lanes_down!(reg[4], 4)),
        ),
        _mm256_add_epi16(
            _mm256_add_epi16(lanes_down!(reg[5], 3), lanes_down!(reg[6], 2)),
            lanes_down!(reg[7], 1),
        ),
    );
    // SAFETY: `REVERSE_7` holds the sixteen bytes loaded.
    let reverse =
        _mm256_broadcastsi128_si256(unsafe { _mm_loadu_si128(REVERSE_7.as_ptr().cast()) });
    let mirrored = _mm256_shuffle_epi8(overflow, reverse);
    let diagonal = _mm256_add_epi32(
        weighted_squares(
            _mm256_unpacklo_epi16(lines, mirrored),
            set_halves(DIAGONAL_WEIGHTS[0]),
        ),
        weighted_squares(
            _mm256_unpackhi_epi16(lines, mirrored),
            set_halves(DIAGONAL_WEIGHTS[1]),
        ),
    );

    // Directions 7 (low) and 5 (high): row pairs shifted by the pair index.
    let pairs_down = skew_cost(core::array::from_fn(|k| {
        _mm256_add_epi16(reg[2 * k], reg[2 * k + 1])
    }));
    // Directions 1 (low) and 3 (high): column pairs shifted by the row;
    // register k holds rows k and k + 4 (7 - k and 3 - k above).
    let pairs_across = skew_cost(core::array::from_fn(|k| {
        _mm256_hadd_epi16(reg[k], reg[k + N / 2])
    }));

    let sums = _mm256_hadd_epi32(
        _mm256_hadd_epi32(diagonal, axis),
        _mm256_hadd_epi32(pairs_across, pairs_down),
    );
    // SAFETY: `COST_ORDER` holds the eight lanes loaded.
    let order = unsafe { _mm256_loadu_si256(COST_ORDER.as_ptr().cast()) };
    best_direction(_mm256_permutevar8x32_epi32(sums, order))
}

/// The first direction of highest cost in `costs` (directions 0..=7) and
/// its margin over the orthogonal direction, scaled to the contrast.
#[target_feature(enable = "avx2")]
#[inline]
fn best_direction(costs: __m256i) -> (u8, i32) {
    let low = _mm256_castsi256_si128(costs);
    let high = _mm256_extracti128_si256::<1>(costs);
    let mut best = _mm_max_epi32(low, high);
    best = _mm_max_epi32(best, _mm_shuffle_epi32::<0b01_00_11_10>(best));
    best = _mm_max_epi32(best, _mm_shuffle_epi32::<0b10_11_00_01>(best));
    // Every cost but the best becomes negative; saturated to 16 bits and
    // read as unsigned, the first zero is the smallest value, so ties go to
    // the lowest direction as in the reference.
    let margins = _mm_packs_epi32(_mm_sub_epi32(low, best), _mm_sub_epi32(high, best));
    let dir = (_mm_cvtsi128_si32(_mm_srli_epi32::<16>(_mm_minpos_epu16(margins))) as usize) & 7;
    let mut cost = [0i32; N];
    // SAFETY: `cost` holds the eight 32-bit lanes stored.
    unsafe { _mm256_storeu_si256(cost.as_mut_ptr().cast(), costs) };
    let best_cost = _mm_cvtsi128_si32(best);
    (dir as u8, (best_cost - cost[dir ^ 4]) >> VAR_SHIFT)
}
