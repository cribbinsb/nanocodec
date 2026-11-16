//! AVX-512 rows of 32 outputs for the shared AV1/HEVC/VVC prediction
//! passes of `av1_avx2` (blocks 32 and 64 wide): the same byte and word
//! multiply-adds and [`Rounding`] schemes, on four 128-bit lanes of 8
//! outputs each instead of two, so the results are identical.
//!
//! - Horizontal: one masked load of the row's `span(32, P)` bytes, then one
//!   `vpermb` (VBMI) per tap pair gathers each lane's byte pairs (lane `k`
//!   holds outputs `8k..8k+8`), multiplied by `vpmaddubsw`.
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

use super::av1_avx2::{Sink, WidePasses, Window, Words, span, tap_pairs};
use super::{PREP_LAST_ROUND, PREP_LAST_SHIFT, ROUND_2, ROUND_6, Rounding};
use core::arch::x86_64::{
    __m512i, _mm_cvtsi32_si128, _mm256_loadu_si256, _mm256_storeu_si256, _mm512_add_epi16,
    _mm512_add_epi32, _mm512_castsi256_si512, _mm512_castsi512_si256, _mm512_loadu_si512,
    _mm512_madd_epi16, _mm512_maddubs_epi16, _mm512_maskz_loadu_epi8, _mm512_mulhrs_epi16,
    _mm512_packs_epi32, _mm512_packus_epi16, _mm512_permutexvar_epi8, _mm512_permutexvar_epi64,
    _mm512_set_epi64, _mm512_set1_epi16, _mm512_set1_epi32, _mm512_setzero_si512, _mm512_sra_epi32,
    _mm512_storeu_si512, _mm512_unpackhi_epi16, _mm512_unpacklo_epi8, _mm512_unpacklo_epi16,
};

/// Outputs of one vector (32 lanes of 16 bits).
const LANES: usize = 32;
/// Outputs of one 128-bit lane.
const HALF: usize = 8;
/// Bytes of one 128-bit lane.
const LANE_BYTES: usize = 16;

/// The AVX-512 passes of 32-output rows (`W` 32 or 64).
pub(super) struct Avx512Wide;

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
            // Lane k's low 8 bytes are outputs 8k..8k+8: gather quadwords
            // 0, 2, 4, 6.
            let lows = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
            let bytes = _mm512_permutexvar_epi64(lows, _mm512_packus_epi16(self, self));
            _mm256_storeu_si256(dst.cast(), _mm512_castsi512_si256(bytes));
        }
    }
    #[inline]
    unsafe fn store_words(self, dst: *mut i16) {
        // SAFETY: the caller enables this tier and provides 32 words.
        unsafe { _mm512_storeu_si512(dst.cast(), self) };
    }
}

/// Tap pairs as signed byte pairs in every 16-bit lane (`vpmaddubsw`).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn byte_pairs<const P: usize>(pairs: [(i16, i16); P]) -> [__m512i; P] {
    pairs.map(|(lo, hi)| _mm512_set1_epi16(i16::from_le_bytes([lo as i8 as u8, hi as i8 as u8])))
}

/// Tap pairs as 16-bit pairs in every 32-bit lane (`vpmaddwd`).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn word_pairs<const P: usize>(pairs: [(i16, i16); P]) -> [__m512i; P] {
    pairs.map(|(lo, hi)| {
        _mm512_set1_epi32((u32::from(lo as u16) | u32::from(hi as u16) << 16) as i32)
    })
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

const GATHERS_4: [[u8; 64]; 2] = gathers::<2>();
const GATHERS_6: [[u8; 64]; 3] = gathers::<3>();
const GATHERS_8: [[u8; 64]; 4] = gathers::<4>();

/// The `P` gather vectors of [`gathers`].
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn gather_vectors<const P: usize>() -> [__m512i; P] {
    let table = match P {
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
unsafe fn row_32<const P: usize>(
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

/// Horizontal only: a row of 32 outputs per vector.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
unsafe fn horizontal<const W: usize, const P: usize, S: Sink, R: Rounding, const C: usize>(
    window: Window,
    lead: usize,
    out: S,
    h: usize,
    taps: &[i16],
) {
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
    let spread = _mm512_set_epi64(3, 3, 2, 2, 1, 1, 0, 0);
    for c0 in (0..W).step_by(LANES) {
        // SAFETY: rows lead.. at columns c0 + C.. (see `av1_avx2::put`).
        let load = |k: usize| unsafe {
            let row = _mm256_loadu_si256(window.at(lead + k, c0 + C).cast());
            _mm512_permutexvar_epi64(spread, _mm512_castsi256_si512(row))
        };
        let mut rows = [_mm512_setzero_si512(); 8];
        for (k, row) in rows.iter_mut().enumerate().take(2 * P - 1) {
            *row = load(k);
        }
        for r in 0..h {
            rows[2 * P - 1] = load(r + 2 * P - 1);
            let mut sum = _mm512_setzero_si512();
            for (j, weight) in taps.iter().enumerate() {
                let pairs = _mm512_unpacklo_epi8(rows[2 * j], rows[2 * j + 1]);
                sum = _mm512_add_epi16(sum, _mm512_maddubs_epi16(pairs, *weight));
            }
            let words = round(sum, if S::PREP { ROUND_2 } else { ROUND_6 });
            // SAFETY: output row r, columns c0..c0 + 32 (see `av1_avx2::put`).
            unsafe { out.wide(r, c0, words) };
            super::slide(&mut rows, 1, 2 * P - 1);
        }
    }
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
