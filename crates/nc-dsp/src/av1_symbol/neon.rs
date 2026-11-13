//! NEON symbol kernels: the bounds of up to 16 symbols in two 8-lane
//! vectors (kernels only; contract in the parent module).
use super::{Av1SymbolInterval, LANES, MIN_PROBABILITY, PROB_TOP, PROBABILITY_SHIFT, padded};
use core::arch::aarch64::{
    uint16x8_t, vaddq_u16, vaddvq_u16, vandq_u16, vbslq_u16, vcgtq_u16, vcltq_u16, vcombine_u16,
    vdupq_n_s16, vdupq_n_u16, vget_low_u16, vld1q_u16, vmull_high_u16, vmull_u16, vshlq_n_u16,
    vshlq_u16, vshrn_n_u32, vshrq_n_u16, vst1q_u16, vsubq_u16,
};

/// Load 8 lanes.
#[target_feature(enable = "neon")]
#[inline]
fn load(lanes: &[u16]) -> uint16x8_t {
    let lanes: &[u16; LANES] = lanes[..LANES].try_into().expect("8 lanes");
    // SAFETY: `lanes` is 8 readable `u16`s.
    unsafe { vld1q_u16(lanes.as_ptr()) }
}

/// Store 8 lanes.
#[target_feature(enable = "neon")]
#[inline]
fn store(lanes: &mut [u16], v: uint16x8_t) {
    let lanes: &mut [u16; LANES] = (&mut lanes[..LANES]).try_into().expect("8 lanes");
    // SAFETY: `lanes` is 8 writable `u16`s.
    unsafe { vst1q_u16(lanes.as_mut_ptr(), v) }
}

/// Lane indices of the low and high vector.
#[target_feature(enable = "neon")]
#[inline]
fn lane_index(high: bool) -> uint16x8_t {
    let base = if high { LANES as u16 } else { 0 };
    let index: [u16; LANES] = core::array::from_fn(|i| base + i as u16);
    load(&index)
}

/// The bounds (`cur`) of eight symbols, 0 from the last symbol on.
#[target_feature(enable = "neon")]
#[inline]
fn bounds(icdf: uint16x8_t, index: uint16x8_t, scale: uint16x8_t, last: uint16x8_t) -> uint16x8_t {
    // `((range >> 8) * (icdf >> 6)) >> 1`: the high halves of the products
    // of `(icdf >> 6) << 7` (below 65536 because icdf < 32768) and
    // `(range >> 8) << 8`.
    let probability =
        vshlq_n_u16::<PRODUCT_SHIFT>(vshrq_n_u16::<{ PROBABILITY_SHIFT as i32 }>(icdf));
    let split = vcombine_u16(
        vshrn_n_u32::<16>(vmull_u16(vget_low_u16(probability), vget_low_u16(scale))),
        vshrn_n_u32::<16>(vmull_high_u16(probability, scale)),
    );
    // MIN_PROBABILITY * (last - i), wrapping past `last` (masked below).
    let offset = vshlq_n_u16::<{ MIN_PROBABILITY.ilog2() as i32 }>(vsubq_u16(last, index));
    vandq_u16(vaddq_u16(split, offset), vcltq_u16(index, last))
}

/// The left shift that makes the high half of a 16x16-bit product of the
/// shifted probability and `(range >> 8) << 8` the bound's split.
const PRODUCT_SHIFT: i32 = 16 - 8 - (7 - PROBABILITY_SHIFT as i32);

/// How many lanes have a bound above `value` (a prefix).
#[target_feature(enable = "neon")]
#[inline]
fn above(bounds: uint16x8_t, value: uint16x8_t) -> usize {
    usize::from(vaddvq_u16(vshrq_n_u16::<15>(vcgtq_u16(bounds, value))))
}

#[target_feature(enable = "neon")]
#[inline]
fn find_symbol_tf<const N: usize>(
    icdf: &[u16; N],
    symbols: usize,
    range: u32,
    value: u32,
) -> Av1SymbolInterval {
    let scale = vdupq_n_u16((range & 0xff00) as u16);
    let last = vdupq_n_u16((symbols - 1) as u16);
    let value_lanes = vdupq_n_u16(value as u16);
    // `range` then the bounds: the interval of symbol `s` is
    // `[edges[s + 1], edges[s])`.
    let mut edges = [0u16; 2 * LANES + 1];
    edges[0] = range as u16;
    let lanes: [u16; 2 * LANES] = padded(icdf);
    let low = bounds(load(&lanes[..LANES]), lane_index(false), scale, last);
    store(&mut edges[1..], low);
    let mut symbol = above(low, value_lanes);
    if N > LANES {
        let high = bounds(load(&lanes[LANES..]), lane_index(true), scale, last);
        store(&mut edges[1 + LANES..], high);
        symbol += above(high, value_lanes);
    }
    Av1SymbolInterval {
        symbol,
        lower: u32::from(edges[symbol + 1]),
        upper: u32::from(edges[symbol]),
    }
}

/// Adapt eight lanes: towards 32768 below `symbol`, towards 0 from it on.
#[target_feature(enable = "neon")]
#[inline]
fn adapt(
    icdf: uint16x8_t,
    index: uint16x8_t,
    symbol: uint16x8_t,
    down: core::arch::aarch64::int16x8_t,
) -> uint16x8_t {
    let towards_top = vaddq_u16(
        icdf,
        vshlq_u16(vsubq_u16(vdupq_n_u16(PROB_TOP), icdf), down),
    );
    let towards_zero = vsubq_u16(icdf, vshlq_u16(icdf, down));
    vbslq_u16(vcltq_u16(index, symbol), towards_top, towards_zero)
}

#[target_feature(enable = "neon")]
#[inline]
fn adapt_cdf_tf<const N: usize>(icdf: &mut [u16; N], symbol: usize, rate: u32) {
    // A negative shift count shifts right.
    let down = vdupq_n_s16(-(rate as i16));
    let symbol = vdupq_n_u16(symbol as u16);
    let mut lanes: [u16; 2 * LANES] = padded(icdf);
    let low = adapt(load(&lanes[..LANES]), lane_index(false), symbol, down);
    store(&mut lanes[..LANES], low);
    if N > LANES {
        let high = adapt(load(&lanes[LANES..]), lane_index(true), symbol, down);
        store(&mut lanes[LANES..], high);
    }
    icdf.copy_from_slice(&lanes[..N]);
}

/// [`super::av1_find_symbol`] with neon.
#[inline]
pub(super) fn find_symbol<const N: usize>(
    icdf: &[u16; N],
    symbols: usize,
    range: u32,
    value: u32,
) -> Av1SymbolInterval {
    // SAFETY: neon is the target's baseline, enabled in every build that
    // compiles this module, which is all the kernel requires.
    unsafe { find_symbol_tf(icdf, symbols, range, value) }
}

/// [`super::av1_adapt_cdf`] with neon.
#[inline]
pub(super) fn adapt_cdf<const N: usize>(icdf: &mut [u16; N], symbol: usize, rate: u32) {
    // SAFETY: as for `find_symbol`.
    unsafe { adapt_cdf_tf(icdf, symbol, rate) }
}
