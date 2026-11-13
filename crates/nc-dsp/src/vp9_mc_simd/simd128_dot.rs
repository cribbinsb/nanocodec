//! Relaxed-SIMD VP9 interpolation (the browser's relaxed module): tap pairs
//! through `i16x8.relaxed_dot_i8x16_i7x16_s`, a byte-pair multiply-add
//! (x86 `pmaddubsw`, Arm `smull` + `addp`), so one instruction forms 16
//! products where the plain SIMD128 kernel (`simd128.rs`) forms 8.
//!
//! The dot multiplies signed bytes by 7-bit ones. Pixels become signed by
//! subtracting 128 (`p ^ 0x80`); the 7-bit operand is a tap magnitude, at
//! most 127 in every nonidentity phase (`tables_match_their_layouts`), so its
//! top bit is clear and both platforms' readings of it agree. A pair sum is at
//! most `2 * 128 * 127 = 32512` in magnitude, so x86's saturation never acts:
//! the relaxed result is the exact sum on every platform
//! (`dot_is_exact_for_every_pair` checks every operand pair).
//!
//! Taps of one sign are paired (the [`layout`] fixes which taps are
//! negative), so the positive and negative sums stay separate as in the plain
//! kernel: `P = dot + dot + 128 Σ|t|` over the positive taps (the bias
//! restores the subtracted 128s; the true `P` fits u16, so wrapping 16-bit
//! adds give it exactly), likewise `N`, then the same saturating difference,
//! rounding shift and narrow. The rounding is added to `P` before the
//! difference: `max(P + 64 - N, 0) >> 7 == (max(P - N, 0) + 64) >> 7`, since a
//! negative `P - N` above -64 shifts to 0 either way.
//!
//! The horizontal pass loads a row's `N + 7` source bytes once (bytes 0-7 and
//! 7-14 in the two halves) and forms each pair by one in-range byte swizzle;
//! the vertical pass interleaves two source rows per pair.
//!
//! Measured against the plain kernel (`nc-checkasm --bench-vp9-mc`, node 22
//! on x86, regular taps, ns per block): horizontal 8x8 36.3 -> 36.0, 16x16
//! 86.9 -> 71.9, 64x64 1165 -> 800; vertical 36.6 -> 35.4, 89.3 -> 84.7,
//! 1205 -> 1092; 2-D 84.1 -> 73.6, 198.9 -> 170.9, 2461 -> 2041. Whole
//! WASM decode of our 720p streams -2 to -5 % (`docs/results/wasm-speed.md`).

use super::driver::{RowKernel, SHIFT, layout};
use crate::vp9_mc::VP9_TAPS;
use core::arch::wasm32::{
    i8x16_relaxed_swizzle, i8x16_shuffle, i16x8_add, i16x8_relaxed_dot_i8x16_i7x16, i16x8_splat,
    u8x16, u8x16_avgr, u8x16_narrow_i16x8, u8x16_splat, u16x8_shr, u16x8_splat, u16x8_sub_sat,
    v128, v128_load32_lane, v128_load32_zero, v128_load64_lane, v128_load64_zero,
    v128_store32_lane, v128_store64_lane, v128_xor,
};

/// Turns an unsigned byte into its signed offset from 128.
const SIGN_FLIP: u8 = 0x80;
/// The value `SIGN_FLIP` takes from each pixel.
const PIXEL_BIAS: u16 = 128;
/// Rounding of the final shift (`FILTER_BITS` = 7).
const ROUND: u16 = 1 << (SHIFT - 1);
/// Tap pairs of one sign per layout: up to two positive, then up to two
/// negative pairs (`None`: no pair).
type Pairs = [Option<(usize, usize)>; 4];

const fn pairs(layout: u8) -> Pairs {
    match layout {
        // Taps 0, 2, 5 and 7 negative.
        layout::ALTERNATING => [Some((1, 3)), Some((4, 6)), Some((0, 2)), Some((5, 7))],
        // Taps 0, 1, 6 and 7 negative.
        layout::OUTER => [Some((2, 3)), Some((4, 5)), Some((0, 1)), Some((6, 7))],
        // Only the centre taps, both positive.
        _ => [Some((3, 4)), None, None, None],
    }
}

/// Lane of source byte `j` (0..15) after the horizontal pass's two loads:
/// bytes 0-7 in lanes 0-7, bytes 7-14 in lanes 8-15.
const fn source_lane(j: usize) -> u8 {
    (if j < 8 { j } else { j + 1 }) as u8
}

/// Swizzle indices gathering, for each output `i`, bytes `i + k1` and
/// `i + k2` into lanes `2i` and `2i + 1`. Outputs past N read lanes the
/// loads left zero (or bytes of later outputs): their results are not stored.
const fn pair_indices(k1: usize, k2: usize) -> [u8; 16] {
    let mut out = [0u8; 16];
    let mut i = 0;
    while i < 8 {
        out[2 * i] = source_lane(i + k1);
        out[2 * i + 1] = source_lane(i + k2);
        i += 1;
    }
    out
}

#[inline(always)]
fn constant(b: [u8; 16]) -> v128 {
    u8x16(
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13],
        b[14], b[15],
    )
}

pub(super) struct Simd128Dot;

/// One phase prepared for the dot: each pair's two tap magnitudes as byte
/// pairs and its swizzle indices for the horizontal pass (made here, once per
/// block: built inside the row loop they cost ten times the filter), and the
/// bias of the positive (with the rounding) and negative sums.
pub(super) struct DotFilters {
    pairs: [v128; 4],
    indices: [v128; 4],
    positive_bias: v128,
    negative_bias: v128,
}

/// The rounded, shifted 8-bit results of eight outputs from the dot of each
/// pair (`dot(p)`: pair `p`'s signed pixel pairs against its taps).
#[inline(always)]
fn finish<const LAYOUT: u8>(f: &DotFilters, dot: impl Fn(usize) -> v128) -> v128 {
    let p = pairs(LAYOUT);
    let mut positive = f.positive_bias;
    let mut negative = f.negative_bias;
    for (index, pair) in p.iter().enumerate() {
        if pair.is_some() {
            if index < 2 {
                positive = i16x8_add(positive, dot(index));
            } else {
                negative = i16x8_add(negative, dot(index));
            }
        }
    }
    let sums = u16x8_shr(u16x8_sub_sat(positive, negative), SHIFT);
    u8x16_narrow_i16x8(sums, u16x8_splat(0))
}

impl RowKernel for Simd128Dot {
    type Filters = DotFilters;
    type Row = v128;
    const LANES: usize = 8;
    #[inline(always)]
    unsafe fn prepare<const LAYOUT: u8>(taps: &[i16; VP9_TAPS]) -> DotFilters {
        let magnitude = |k: usize| taps[k].unsigned_abs();
        let p = pairs(LAYOUT);
        let pair = |index: usize| match p[index] {
            Some((k1, k2)) => i16x8_splat((magnitude(k1) | magnitude(k2) << 8) as i16),
            None => u16x8_splat(0),
        };
        // Σ|t| over the taps of each sign (the pairs cover every nonzero tap).
        let sum = |range: core::ops::Range<usize>| -> u16 {
            range
                .filter_map(|index| p[index])
                .map(|(k1, k2)| magnitude(k1) + magnitude(k2))
                .sum()
        };
        let indices = |index: usize| match p[index] {
            Some((k1, k2)) => constant(pair_indices(k1, k2)),
            None => u16x8_splat(0),
        };
        DotFilters {
            pairs: [pair(0), pair(1), pair(2), pair(3)],
            // Opaque to the optimiser: folded into the row loop as constants,
            // V8 rebuilds each with four instructions per use instead of one load.
            indices: core::hint::black_box([indices(0), indices(1), indices(2), indices(3)]),
            positive_bias: u16x8_splat(PIXEL_BIAS * sum(0..2) + ROUND),
            negative_bias: u16x8_splat(PIXEL_BIAS * sum(2..4)),
        }
    }
    #[inline(always)]
    unsafe fn load<const N: usize>(src: *const u8) -> v128 {
        // SAFETY: the caller provides N readable bytes; each load reads N.
        unsafe {
            if N == 4 {
                v128_load32_zero(src.cast())
            } else {
                v128_load64_zero(src.cast())
            }
        }
    }
    #[inline(always)]
    unsafe fn store<const N: usize, const AVG: bool>(dst: *mut u8, row: v128) {
        // SAFETY: the caller provides N writable bytes, also readable when AVG.
        unsafe {
            let row = if AVG {
                u8x16_avgr(row, Self::load::<N>(dst))
            } else {
                row
            };
            if N == 4 {
                v128_store32_lane::<0>(row, dst.cast());
            } else {
                v128_store64_lane::<0>(row, dst.cast());
            }
        }
    }
    #[inline(always)]
    unsafe fn horizontal<const N: usize, const LAYOUT: u8>(src: *const u8, f: &DotFilters) -> v128 {
        // SAFETY: the caller provides N + 7 readable bytes: bytes 0..8 and,
        // from byte 7, the remaining N - 1 (4 or 8 bytes, within N + 7).
        let row = unsafe {
            let low = v128_load64_zero(src.cast());
            if N == 4 {
                v128_load32_lane::<2>(low, src.add(7).cast())
            } else {
                v128_load64_lane::<1>(low, src.add(7).cast())
            }
        };
        let signed = v128_xor(row, u8x16_splat(SIGN_FLIP));
        finish::<LAYOUT>(f, |index| {
            let gathered = i8x16_relaxed_swizzle(signed, f.indices[index]);
            i16x8_relaxed_dot_i8x16_i7x16(gathered, f.pairs[index])
        })
    }
    #[inline(always)]
    unsafe fn vertical<const N: usize, const LAYOUT: u8>(
        rows: &[v128; VP9_TAPS],
        f: &DotFilters,
    ) -> v128 {
        let p = pairs(LAYOUT);
        finish::<LAYOUT>(f, |index| {
            let (k1, k2) = p[index].expect("finish asks only for present pairs");
            let interleaved = i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(
                rows[k1], rows[k2],
            );
            let signed = v128_xor(interleaved, u8x16_splat(SIGN_FLIP));
            i16x8_relaxed_dot_i8x16_i7x16(signed, f.pairs[index])
        })
    }
}
