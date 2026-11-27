//! AVX-512 rows of 32 outputs for the shared AV1/HEVC/VVC prediction
//! passes of `av1_avx2` (blocks 32 and 64 wide): the same byte and word
//! multiply-adds and [`Rounding`] schemes, on four 128-bit lanes of 8
//! outputs each instead of two, so the results are identical.
//!
//! - Horizontal: one masked load of the row's `span(32, P)` bytes, then one
//!   `vpermb` (VBMI) per tap pair gathers each lane's byte pairs (lane `k`
//!   holds outputs `8k..8k+8`), multiplied by `vpmaddubsw`. Wide AV1-family
//!   one-direction filters instead use VNNI with three shared gathers.
//! - Vertical, one direction: each 32-byte row is spread once (`vpermq`,
//!   lane `k` low half = bytes `8k..8k+8`), then a tap pair is one
//!   `vpunpcklbw` and one `vpmaddubsw`.
//! - Vertical after a horizontal pass: rows of 32 rounded words,
//!   interleaved once each into an even and an odd window of row pairs
//!   (`vpunpck{l,h}wd`, `vpmaddwd`), two output rows per step, as the AVX2
//!   strips do.
//!
//! Results come out in natural order (lane `k` = outputs `8k..8k+8`):
//! `prep` stores them as they are, `put` packs to bytes and gathers the
//! lanes' low halves with one `vpermq`.
//!
//! Kernel contract: see the `mc` module docs; the entry points
//! (`av1_avx2::{put, prep, separable}`) check the window and destination.

#![allow(
    clippy::inline_always,
    reason = "the pass selection must inline into the AVX-512 kernels"
)]

use super::av1_avx2::{Avx2Wide, Sink, WidePasses, Window, Words, tap_pairs};
use super::avx2_byte_dot::span;
use super::{PREP_LAST_ROUND, PREP_LAST_SHIFT, ROUND_2, ROUND_6, Rounding, TAPS};
use core::arch::x86_64::{
    __m256i, __m512i, _mm_cvtsi32_si128, _mm_loadl_epi64, _mm_maskz_loadu_epi8,
    _mm256_broadcastsi128_si256, _mm256_castsi128_si256, _mm256_dpbusd_epi32, _mm256_dpwssds_epi32,
    _mm256_inserti128_si256, _mm256_loadu_si256, _mm256_packs_epi32, _mm256_permutex2var_epi16,
    _mm256_set1_epi32, _mm256_shuffle_epi8, _mm256_sra_epi32, _mm256_srai_epi32,
    _mm256_storeu_si256, _mm512_add_epi16, _mm512_add_epi32, _mm512_castsi256_si512,
    _mm512_castsi512_si256, _mm512_cvtepi32_epi16, _mm512_dpbusd_epi32, _mm512_dpwssds_epi32,
    _mm512_loadu_si512, _mm512_madd_epi16, _mm512_maddubs_epi16, _mm512_maskz_loadu_epi8,
    _mm512_mulhrs_epi16, _mm512_packs_epi32, _mm512_packus_epi16, _mm512_packus_epi32,
    _mm512_permutex2var_epi16, _mm512_permutexvar_epi8, _mm512_permutexvar_epi64, _mm512_set_epi64,
    _mm512_set1_epi16, _mm512_set1_epi32, _mm512_setzero_si512, _mm512_shuffle_epi8,
    _mm512_shuffle_i32x4, _mm512_sra_epi32, _mm512_srai_epi16, _mm512_srai_epi32,
    _mm512_srli_epi16, _mm512_storeu_si512, _mm512_unpackhi_epi16, _mm512_unpacklo_epi8,
    _mm512_unpacklo_epi16,
};

/// Outputs of one vector (32 lanes of 16 bits).
const LANES: usize = 32;
/// Outputs of one 128-bit lane.
const HALF: usize = 8;
/// Bytes of one 128-bit lane.
const LANE_BYTES: usize = 16;

/// The AVX-512 passes of 32-output rows (`W` 32 or 64).
pub(super) struct Avx512Wide;

/// Ice Lake narrow AV1 uses the packed two-row filter from dav1d's
/// `mc_avx512.asm` (BSD-2-Clause). One-axis filters keep the shared passes.
pub(super) struct Avx512Narrow;

impl WidePasses for Avx512Narrow {
    #[inline(always)]
    unsafe fn horizontal<const W: usize, const P: usize, S: Sink, R: Rounding, const C: usize>(
        window: Window,
        lead: usize,
        out: S,
        h: usize,
        taps: &[i16],
    ) {
        // SAFETY: the same checked footprints and AVX2 support.
        unsafe { Avx2Wide::horizontal::<W, P, S, R, C>(window, lead, out, h, taps) };
    }

    #[inline(always)]
    unsafe fn vertical<const W: usize, const P: usize, S: Sink, R: Rounding, const C: usize>(
        window: Window,
        lead: usize,
        out: S,
        h: usize,
        taps: &[i16],
    ) {
        // SAFETY: the same checked footprints and AVX2 support.
        unsafe { Avx2Wide::vertical::<W, P, S, R, C>(window, lead, out, h, taps) };
    }

    #[inline(always)]
    unsafe fn both<const W: usize, const P: usize, S: Sink, R: Rounding, const C: usize>(
        window: Window,
        lead: usize,
        out: S,
        h: usize,
        taps: (&[i16], &[i16]),
    ) {
        debug_assert!(matches!(W, 4 | 8) && R::HALVED && lead == TAPS / 2 - P);
        // SAFETY: only the checked narrow AV1 paths install these passes.
        unsafe {
            if h == 4 {
                both_narrow::<W, P, 2, S, R>(window, out, h, taps);
            } else {
                both_narrow::<W, P, P, S, R>(window, out, h, taps);
            }
        }
    }
}

/// Dot four horizontally adjacent bytes, with two four-pixel rows in
/// separate lanes. Leave each rounded result in its dword for the vertical
/// word permutation.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
unsafe fn horizontal_four_pair(
    first: *const u8,
    second: *const u8,
    gather: __m256i,
    taps: __m256i,
) -> __m256i {
    // SAFETY: eight bytes fit in the checked row from column two. The
    // shuffle uses the first seven bytes.
    let rows = unsafe {
        _mm256_inserti128_si256::<1>(
            _mm256_castsi128_si256(_mm_loadl_epi64(first.cast())),
            _mm_loadl_epi64(second.cast()),
        )
    };
    let dots = _mm256_dpbusd_epi32(
        _mm256_set1_epi32(2),
        _mm256_shuffle_epi8(rows, gather),
        taps,
    );
    _mm256_srai_epi32::<2>(dots)
}

/// For an eight-wide row, each of its two lanes gathers four output
/// pixels. Each dot uses a different group of four taps.
const fn eight_gather(offset: u8) -> [u8; 64] {
    let mut gather = [0; 64];
    let mut byte = 0;
    while byte < gather.len() {
        let col = (byte / 16 % 2 * 4 + byte % 16 / 4) as u8;
        gather[byte] = offset + col + (byte % 4) as u8;
        byte += 1;
    }
    gather
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
unsafe fn horizontal_eight_pair<const P: usize>(
    first: *const u8,
    second: *const u8,
    taps: [__m512i; 2],
) -> __m512i {
    const GATHER: [[u8; 64]; 2] = [eight_gather(0), eight_gather(4)];
    // SAFETY: only bytes in the checked horizontal tap span are loaded.
    // GATHER contains the two full permutation vectors.
    unsafe {
        let mask = (1 << span(8, P)) - 1;
        let low = _mm_maskz_loadu_epi8(mask, first.cast());
        let high = _mm_maskz_loadu_epi8(mask, second.cast());
        let pair = _mm512_castsi256_si512(_mm256_inserti128_si256::<1>(
            _mm256_castsi128_si256(low),
            high,
        ));
        let rows = _mm512_shuffle_i32x4::<0x50>(pair, pair);
        let mut sum = _mm512_set1_epi32(2);
        for (gather, taps) in GATHER.iter().zip(taps) {
            let gather = _mm512_loadu_si512(gather.as_ptr().cast());
            sum = _mm512_dpbusd_epi32(sum, _mm512_shuffle_epi8(rows, gather), taps);
        }
        _mm512_srai_epi32::<2>(sum)
    }
}

/// First output: previous high row and next low row. Second output:
/// next low and high rows. The sums' low words are at even indices.
const fn narrow_pairs<const W: usize>() -> [i16; 32] {
    let mut indices = [0; 32];
    let mut col = 0;
    while col < W {
        let i = 2 * col;
        indices[i] = (2 * W + i) as i16;
        indices[i + 1] = (4 * W + i) as i16;
        indices[2 * W + i] = (4 * W + i) as i16;
        indices[2 * W + i + 1] = (6 * W + i) as i16;
        col += 1;
    }
    indices
}

/// Keep one interleaved vertical pair per tap pair for each of the two
/// independently strided output rows.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
unsafe fn both_narrow<const W: usize, const H: usize, const V: usize, S: Sink, R: Rounding>(
    window: Window,
    out: S,
    h: usize,
    (h_taps, v_taps): (&[i16], &[i16]),
) {
    // All AV1 four-wide families use only taps 2 through 5 horizontally.
    let h_taps = <&[i16; TAPS]>::try_from(h_taps).unwrap();
    let v_taps = <&[i16; TAPS]>::try_from(v_taps).unwrap();
    let first = if W == 4 { 2 } else { TAPS / 2 - H };
    let v_lead = TAPS / 2 - V;
    let horizontal = &h_taps[first..first + if W == 4 { 4 } else { 2 * H }];
    let first_four = <[i16; 4]>::try_from(&horizontal[..4]).unwrap();
    let pack = |taps: [i16; 4]| i32::from_le_bytes(taps.map(|tap| ((tap >> 1) as i8) as u8));
    let h0 = pack(first_four);
    let h1 = if W == 4 {
        0
    } else if H == 3 {
        pack([horizontal[4], horizontal[5], 0, 0])
    } else {
        pack(<[i16; 4]>::try_from(&horizontal[4..8]).unwrap())
    };
    let gather = _mm256_broadcastsi128_si256(core::arch::x86_64::_mm_setr_epi8(
        0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6,
    ));
    // SAFETY: the index table is a complete vector.
    let pair = unsafe { _mm512_loadu_si512(narrow_pairs::<W>().as_ptr().cast()) };
    let interleave = |prev, next| {
        if W == 4 {
            _mm512_castsi256_si512(_mm256_permutex2var_epi16(
                _mm512_castsi512_si256(prev),
                _mm512_castsi512_si256(pair),
                _mm512_castsi512_si256(next),
            ))
        } else {
            _mm512_permutex2var_epi16(prev, pair, next)
        }
    };
    let v_taps = &v_taps[v_lead..v_lead + 2 * V];
    let weights: [__m512i; V] = core::array::from_fn(|j| {
        let lo = v_taps[2 * j] >> 1;
        let hi = v_taps[2 * j + 1] >> 1;
        _mm512_set1_epi32(i32::from(lo as u16) | (i32::from(hi) << 16))
    });
    let rows = |a, b| {
        // SAFETY: each row covers the horizontal tap span. Calls below
        // use rows from v_lead through v_lead + h + 2 * V - 2.
        unsafe {
            let lhs = window.at(v_lead + a, first);
            let rhs = window.at(v_lead + b, first);
            if W == 4 {
                _mm512_castsi256_si512(horizontal_four_pair(
                    lhs,
                    rhs,
                    gather,
                    _mm256_set1_epi32(h0),
                ))
            } else {
                horizontal_eight_pair::<H>(lhs, rhs, [_mm512_set1_epi32(h0), _mm512_set1_epi32(h1)])
            }
        }
    };
    let mut previous = rows(0, 0);
    let mut pairs: [__m512i; V] = core::array::from_fn(|j| {
        if j + 1 == V {
            return _mm512_setzero_si512();
        }
        let next = rows(2 * j + 1, 2 * j + 2);
        let result = interleave(previous, next);
        previous = next;
        result
    });
    let (round, shift) = if S::PREP {
        (PREP_LAST_ROUND, PREP_LAST_SHIFT)
    } else {
        (R::LAST_ROUND, R::LAST_SHIFT)
    };
    let round = _mm512_set1_epi32(round);
    let shift = _mm_cvtsi32_si128(shift);
    for row in (0..h).step_by(2) {
        let next = rows(row + 2 * V - 1, row + 2 * V);
        pairs[V - 1] = interleave(previous, next);
        let mut acc = round;
        for j in 0..V {
            // The horizontal sum is bounded by 255 * 120 before its
            // divide by four, so the signed dword dot never saturates.
            // This form keeps LLVM from splitting a dot into madd+add.
            acc = if W == 4 {
                _mm512_castsi256_si512(_mm256_dpwssds_epi32(
                    _mm512_castsi512_si256(acc),
                    _mm512_castsi512_si256(pairs[j]),
                    _mm512_castsi512_si256(weights[j]),
                ))
            } else {
                _mm512_dpwssds_epi32(acc, pairs[j], weights[j])
            };
        }
        let words = if W == 4 {
            let lanes = _mm256_sra_epi32(_mm512_castsi512_si256(acc), shift);
            _mm256_packs_epi32(lanes, lanes)
        } else {
            // AV1's two-pass intermediate stays within i16, including
            // the filters' negative lobes.
            _mm512_cvtepi32_epi16(_mm512_sra_epi32(acc, shift))
        };
        // SAFETY: both rows are within the checked narrow output.
        unsafe { out.two::<W>(row, words) };
        pairs = core::array::from_fn(|j| pairs[(j + 1).min(V - 1)]);
        previous = next;
    }
}

impl WidePasses for Avx512Wide {
    #[inline(always)]
    unsafe fn horizontal<const W: usize, const P: usize, S: Sink, R: Rounding, const C: usize>(
        window: Window,
        lead: usize,
        out: S,
        h: usize,
        taps: &[i16],
    ) {
        // SAFETY: forwarded from the caller, whose ISA is this tier.
        unsafe { horizontal::<W, P, S, R, C>(window, lead, out, h, taps) }
    }
    #[inline(always)]
    unsafe fn vertical<const W: usize, const P: usize, S: Sink, R: Rounding, const C: usize>(
        window: Window,
        lead: usize,
        out: S,
        h: usize,
        taps: &[i16],
    ) {
        // SAFETY: forwarded from the caller, whose ISA is this tier.
        unsafe { vertical::<W, P, S, R, C>(window, lead, out, h, taps) }
    }
    #[inline(always)]
    unsafe fn both<const W: usize, const P: usize, S: Sink, R: Rounding, const C: usize>(
        window: Window,
        lead: usize,
        out: S,
        h: usize,
        taps: (&[i16], &[i16]),
    ) {
        // SAFETY: forwarded from the caller, whose ISA is this tier.
        unsafe { both::<W, P, S, R, C>(window, lead, out, h, taps) }
    }
}

impl Words for __m512i {
    #[inline]
    unsafe fn store_bytes(self, dst: *mut u8) {
        // SAFETY: the caller enables this tier and provides 32 bytes.
        unsafe {
            _mm256_storeu_si256(dst.cast(), pack_32(self));
        }
    }
    #[inline]
    unsafe fn store_words(self, dst: *mut i16) {
        // SAFETY: the caller enables this tier and provides 32 words.
        unsafe { _mm512_storeu_si512(dst.cast(), self) };
    }
}

/// Pack the eight outputs in each lane into one 32-byte row.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
pub(crate) fn pack_32(words: __m512i) -> __m256i {
    let lows = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
    _mm512_castsi512_si256(_mm512_permutexvar_epi64(
        lows,
        _mm512_packus_epi16(words, words),
    ))
}

/// Tap pairs as signed byte pairs in every 16-bit lane (`vpmaddubsw`).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
pub(crate) fn byte_pairs<const P: usize>(pairs: [(i16, i16); P]) -> [__m512i; P] {
    // A plain loop, not `array::map`: the closure would be compiled outside
    // this function's target features (the outlined-SIMD check).
    let mut out = [_mm512_setzero_si512(); P];
    for (slot, (lo, hi)) in out.iter_mut().zip(pairs) {
        *slot = _mm512_set1_epi16(i16::from_le_bytes([lo as i8 as u8, hi as i8 as u8]));
    }
    out
}

/// Tap pairs as 16-bit pairs in every 32-bit lane (`vpmaddwd`).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn word_pairs<const P: usize>(pairs: [(i16, i16); P]) -> [__m512i; P] {
    let mut out = [_mm512_setzero_si512(); P];
    for (slot, (lo, hi)) in out.iter_mut().zip(pairs) {
        *slot = _mm512_set1_epi32((u32::from(lo as u16) | u32::from(hi as u16) << 16) as i32);
    }
    out
}

/// `vpermb` indexes gathering tap pair `p` of 32 outputs from a row loaded
/// at byte 0: byte `i` of lane `k` (output `o = 8k + i / 2`) takes row byte
/// `o + 2p + i % 2`.
const fn gathers<const P: usize>() -> [[u8; 64]; P] {
    let mut out = [[0; 64]; P];
    let mut p = 0;
    while p < P {
        let mut i = 0;
        while i < 64 {
            let output = i / LANE_BYTES * HALF + i % LANE_BYTES / 2;
            out[p][i] = (output + 2 * p + i % 2) as u8;
            i += 1;
        }
        p += 1;
    }
    out
}

const GATHERS_2: [[u8; 64]; 1] = gathers::<1>();
const GATHERS_4: [[u8; 64]; 2] = gathers::<2>();
const GATHERS_6: [[u8; 64]; 3] = gathers::<3>();
const GATHERS_8: [[u8; 64]; 4] = gathers::<4>();

/// The `P` gather vectors of [`gathers`].
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
pub(crate) fn gather_vectors<const P: usize>() -> [__m512i; P] {
    let table = match P {
        1 => &GATHERS_2[..],
        2 => &GATHERS_4[..],
        3 => &GATHERS_6[..],
        _ => &GATHERS_8[..],
    };
    // SAFETY: each index row is 64 readable bytes.
    core::array::from_fn(|p| unsafe { _mm512_loadu_si512(table[p].as_ptr().cast()) })
}

/// Unrounded horizontal sums of 32 outputs of the row at `src`, reading
/// exactly `span(32, P)` bytes.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
pub(crate) unsafe fn row_32<const P: usize>(
    src: *const u8,
    gathers: &[__m512i; P],
    taps: &[__m512i; P],
) -> __m512i {
    let mask = u64::MAX >> (64 - span(LANES, P));
    // SAFETY: the caller provides span(32, P) readable bytes; the mask
    // suppresses the rest of the 64-byte load.
    let pixels = unsafe { _mm512_maskz_loadu_epi8(mask, src.cast()) };
    let mut sum = _mm512_setzero_si512();
    for (gather, weight) in gathers.iter().zip(taps) {
        let pairs = _mm512_permutexvar_epi8(*gather, pixels);
        sum = _mm512_add_epi16(sum, _mm512_maddubs_epi16(pairs, *weight));
    }
    sum
}

/// `vpmulhrsw` by `m`: `(x + m') >> k` for `m = 2^(15 - k)`.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn round(x: __m512i, m: i16) -> __m512i {
    _mm512_mulhrs_epi16(x, _mm512_set1_epi16(m))
}

/// [`Rounding::FIRST`] on 32 sums.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn first<R: Rounding>(sum: __m512i) -> __m512i {
    match R::FIRST {
        Some(m) => round(sum, m),
        None => sum,
    }
}

/// The vertical pass of a 2-D filter over `P` interleaved row pairs,
/// rounded for `put` or `prep` (as `av1_avx2::vertical_words`).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn vertical_words<R: Rounding, const P: usize>(
    pairs: &[(__m512i, __m512i); P],
    taps: &[__m512i; P],
    prep: bool,
) -> __m512i {
    let (round, shift) = if prep {
        (PREP_LAST_ROUND, PREP_LAST_SHIFT)
    } else {
        (R::LAST_ROUND, R::LAST_SHIFT)
    };
    let mut lo = _mm512_set1_epi32(round);
    let mut hi = lo;
    for ((low, high), weight) in pairs.iter().zip(taps) {
        lo = _mm512_add_epi32(lo, _mm512_madd_epi16(*low, *weight));
        hi = _mm512_add_epi32(hi, _mm512_madd_epi16(*high, *weight));
    }
    let shift = _mm_cvtsi32_si128(shift);
    _mm512_packs_epi32(_mm512_sra_epi32(lo, shift), _mm512_sra_epi32(hi, shift))
}

/// Gather four adjacent tap bytes per output for `vpdpbusd`. The two
/// output groups share the middle gather. Their lane layout packs back
/// into the same 32-word order as [`row_32`].
const fn vnni_gather_indices() -> [[u8; 64]; 3] {
    let mut indices = [[0; 64]; 3];
    let mut group = 0;
    while group < 3 {
        let mut byte = 0;
        while byte < 64 {
            let output = byte / 16 * 8 + byte % 16 / 4;
            indices[group][byte] = (output + group * 4 + byte % 4) as u8;
            byte += 1;
        }
        group += 1;
    }
    indices
}

const VNNI_GATHERS: [[u8; 64]; 3] = vnni_gather_indices();

/// Dav1d's AVX-512 ICL four-tap dot structure for the wide horizontal
/// AV1 pass (`mc_avx512.asm`, BSD-2-Clause). Uses the checked exact
/// footprint and the shared output sinks.
///
/// # Safety
/// The caller enables the ICL tier and checks the source and output.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
unsafe fn horizontal_vnni<const W: usize, const P: usize, S: Sink, const C: usize>(
    window: Window,
    lead: usize,
    out: S,
    height: usize,
    taps: [(i16, i16); P],
) {
    let mut groups = [0u32; 2];
    for (i, (a, b)) in taps.into_iter().enumerate() {
        let pair = u16::from_le_bytes([a as u8, b as u8]);
        groups[i / 2] |= u32::from(pair) << ((i % 2) * 16);
    }
    let weights = [
        _mm512_set1_epi32(groups[0] as i32),
        _mm512_set1_epi32(groups[1] as i32),
    ];
    // SAFETY: each constant holds the 64 gather bytes.
    let [g0, g1, g2] = unsafe {
        [
            _mm512_loadu_si512(VNNI_GATHERS[0].as_ptr().cast()),
            _mm512_loadu_si512(VNNI_GATHERS[1].as_ptr().cast()),
            _mm512_loadu_si512(VNNI_GATHERS[2].as_ptr().cast()),
        ]
    };
    let mask = u64::MAX >> (64 - span(LANES, P));
    // The two rounded shifts of put are exactly (sum + 34) >> 6.
    let bias = _mm512_set1_epi32(if S::PREP { 2 } else { 34 });
    for r in 0..height {
        for c0 in (0..W).step_by(LANES) {
            // SAFETY: the input window holds span(32, P) bytes on this row.
            let pixels =
                unsafe { _mm512_maskz_loadu_epi8(mask, window.at(r + C, c0 + lead).cast()) };
            let a = _mm512_permutexvar_epi8(g0, pixels);
            let b = _mm512_permutexvar_epi8(g1, pixels);
            let c = _mm512_permutexvar_epi8(g2, pixels);
            let lo = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(bias, a, weights[0]), b, weights[1]);
            let hi = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(bias, b, weights[0]), c, weights[1]);
            let words = if S::PREP {
                // Halved AV1 taps have sum bounds within signed words, with
                // two points of rounding headroom (their extrema are multiples of 255).
                _mm512_srai_epi16::<2>(_mm512_packs_epi32(lo, hi))
            } else {
                // The byte store clips any negative or oversized result.
                _mm512_srli_epi16::<6>(_mm512_packus_epi32(lo, hi))
            };
            // SAFETY: the caller checked this complete output row segment.
            unsafe { out.wide(r, c0, words) };
        }
    }
}

/// Horizontal only: a row of 32 outputs per vector.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
unsafe fn horizontal<const W: usize, const P: usize, S: Sink, R: Rounding, const C: usize>(
    window: Window,
    lead: usize,
    out: S,
    h: usize,
    taps: &[i16],
) {
    if R::HALVED && P >= 3 {
        // SAFETY: ICL dispatch enables VNNI and the same checked footprint
        // covers the zero-masked gathers for six or eight taps.
        unsafe {
            horizontal_vnni::<W, P, S, C>(window, lead, out, h, tap_pairs::<R, P>(taps, lead));
        }
        return;
    }
    let taps = byte_pairs::<P>(tap_pairs::<R, P>(taps, lead));
    let gathers = gather_vectors::<P>();
    for r in 0..h {
        for c0 in (0..W).step_by(LANES) {
            // SAFETY: row r + C, columns c0 + lead.. (see `av1_avx2::put`).
            unsafe {
                let t = first::<R>(row_32::<P>(window.at(r + C, c0 + lead), &gathers, &taps));
                let words = if S::PREP {
                    t
                } else {
                    round(t, R::HORIZONTAL_PUT)
                };
                out.wide(r, c0, words);
            }
        }
    }
}

/// Vertical only: a row of 32 outputs per vector from byte-interleaved row
/// pairs, each row spread once so lane `k` starts with its bytes `8k..`.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
unsafe fn vertical<const W: usize, const P: usize, S: Sink, R: Rounding, const C: usize>(
    window: Window,
    lead: usize,
    out: S,
    h: usize,
    taps: &[i16],
) {
    let taps = byte_pairs::<P>(tap_pairs::<R, P>(taps, lead));
    for c0 in (0..W).step_by(LANES) {
        // SAFETY: rows lead.. at columns c0 + C.. and this output strip
        // were validated by the entry point.
        unsafe {
            column_32(
                h,
                &taps,
                |k| _mm256_loadu_si256(window.at(lead + k, c0 + C).cast()),
                |r, sum| out.wide(r, c0, round(sum, if S::PREP { ROUND_2 } else { ROUND_6 })),
            );
        }
    }
}

/// Expand each group of eight bytes to the front of one 128-bit lane.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
pub(crate) fn spread_32(row: __m256i) -> __m512i {
    let spread = _mm512_set_epi64(3, 3, 2, 2, 1, 1, 0, 0);
    _mm512_permutexvar_epi64(spread, _mm512_castsi256_si512(row))
}

/// Sum the byte-tap pairs from rows in the layout of [`spread_32`].
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
pub(crate) fn vertical_32<const P: usize>(rows: &[__m512i; 8], taps: &[__m512i; P]) -> __m512i {
    let mut sum = _mm512_setzero_si512();
    for (j, weight) in taps.iter().enumerate() {
        let pairs = _mm512_unpacklo_epi8(rows[2 * j], rows[2 * j + 1]);
        sum = _mm512_add_epi16(sum, _mm512_maddubs_epi16(pairs, *weight));
    }
    sum
}

/// Byte-input sliding window for the wide column pass, before codec rounding.
/// `load` provides h + 2 × P − 1 rows of 32 source bytes.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
pub(crate) fn column_32<const P: usize>(
    h: usize,
    taps: &[__m512i; P],
    mut load: impl FnMut(usize) -> __m256i,
    mut store: impl FnMut(usize, __m512i),
) {
    super::byte_column_pairs::<P, _, _>(
        h,
        _mm512_setzero_si512(),
        |i| spread_32(load(i)),
        |a, b| _mm512_unpacklo_epi8(a, b),
        |r, pairs| {
            let mut sum = _mm512_setzero_si512();
            for (j, weight) in taps.iter().enumerate() {
                sum = _mm512_add_epi16(sum, _mm512_maddubs_epi16(pairs[2 * j], *weight));
            }
            store(r, sum);
        },
    );
}

/// Both directions: rounded horizontal rows of 32 outputs, interleaved in
/// pairs once each into an even window (rows r, r+1; r+2, r+3; ...) and an
/// odd one (r+1, r+2; ...), giving two output rows per step.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
unsafe fn both<const W: usize, const P: usize, S: Sink, R: Rounding, const C: usize>(
    window: Window,
    lead: usize,
    out: S,
    h: usize,
    (h_taps, v_taps): (&[i16], &[i16]),
) {
    let h_taps = byte_pairs::<P>(tap_pairs::<R, P>(h_taps, lead));
    let v_taps = word_pairs::<P>(tap_pairs::<R, P>(v_taps, lead));
    let gathers = gather_vectors::<P>();
    let interleave =
        |a: __m512i, b: __m512i| (_mm512_unpacklo_epi16(a, b), _mm512_unpackhi_epi16(a, b));
    for c0 in (0..W).step_by(LANES) {
        let row = |k: usize| {
            // SAFETY: row lead + k, columns c0 + lead.. (see `av1_avx2::put`).
            first::<R>(unsafe { row_32::<P>(window.at(lead + k, c0 + lead), &gathers, &h_taps) })
        };
        let zero = _mm512_setzero_si512();
        let mut even: [(__m512i, __m512i); P] = [(zero, zero); P];
        let mut odd = even;
        // Rows 0..2P-1: even pairs (0,1), (2,3), ...; odd pairs (1,2),
        // (3,4), ...; the last of each is made in the loop.
        let mut previous = row(0);
        for j in 0..P - 1 {
            let (a, b) = (row(2 * j + 1), row(2 * j + 2));
            even[j] = interleave(previous, a);
            odd[j] = interleave(a, b);
            previous = b;
        }
        for r in (0..h).step_by(2) {
            let (a, b) = (row(r + 2 * P - 1), row(r + 2 * P));
            even[P - 1] = interleave(previous, a);
            odd[P - 1] = interleave(a, b);
            previous = b;
            for (k, set) in [&even, &odd].into_iter().enumerate() {
                let words = vertical_words::<R, P>(set, &v_taps, S::PREP);
                // SAFETY: output row r + k, columns c0..c0 + 32 (see
                // `av1_avx2::put`).
                unsafe { out.wide(r + k, c0, words) };
            }
            even = core::array::from_fn(|j| if j + 1 < P { even[j + 1] } else { even[j] });
            odd = core::array::from_fn(|j| if j + 1 < P { odd[j + 1] } else { odd[j] });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{GATHERS_8, HALF, LANE_BYTES};

    /// Lane `k`'s byte pairs are those of outputs `8k..8k+8` and stay in
    /// the row's first `span(32, 4)` = 39 bytes.
    #[test]
    fn gathers_pair_each_lanes_outputs() {
        for (p, gather) in GATHERS_8.iter().enumerate() {
            for (i, &byte) in gather.iter().enumerate() {
                let output = i / LANE_BYTES * HALF + i % LANE_BYTES / 2;
                assert_eq!(usize::from(byte), output + 2 * p + i % 2);
                assert!(byte < 39);
            }
        }
    }
}
