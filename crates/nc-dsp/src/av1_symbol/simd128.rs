//! WebAssembly SIMD128 symbol kernels: the bounds of up to 16 symbols in
//! two 8-lane vectors (kernels only; contract in the parent module).
use super::{Av1SymbolInterval, LANES, MIN_PROBABILITY, PROB_TOP, PROBABILITY_SHIFT, padded};
use core::arch::wasm32::{
    i8x16_shuffle, u16x8_add, u16x8_bitmask, u16x8_gt, u16x8_lt, u16x8_shl, u16x8_shr, u16x8_splat,
    u16x8_sub, u32x4_extmul_high_u16x8, u32x4_extmul_low_u16x8, v128, v128_and, v128_bitselect,
    v128_load, v128_store,
};

/// Load 8 lanes.
#[target_feature(enable = "simd128")]
#[inline]
fn load(lanes: &[u16]) -> v128 {
    let lanes: &[u16; LANES] = lanes[..LANES].try_into().expect("8 lanes");
    // SAFETY: `lanes` is 16 readable bytes; wasm loads need no alignment.
    unsafe { v128_load(lanes.as_ptr().cast()) }
}

/// Store 8 lanes.
#[target_feature(enable = "simd128")]
#[inline]
fn store(lanes: &mut [u16], v: v128) {
    let lanes: &mut [u16; LANES] = (&mut lanes[..LANES]).try_into().expect("8 lanes");
    // SAFETY: `lanes` is 16 writable bytes; wasm stores need no alignment.
    unsafe { v128_store(lanes.as_mut_ptr().cast(), v) }
}

/// Lane indices of the low and high vector.
#[target_feature(enable = "simd128")]
#[inline]
fn lane_index(high: bool) -> v128 {
    let base = if high { LANES as u16 } else { 0 };
    let index: [u16; LANES] = core::array::from_fn(|i| base + i as u16);
    load(&index)
}

/// The left shift that makes the high half of a 16x16-bit product of the
/// shifted probability and `(range >> 8) << 8` the bound's split.
const PRODUCT_SHIFT: u32 = 16 - 8 - (7 - PROBABILITY_SHIFT);

/// The bounds (`cur`) of eight symbols, 0 from the last symbol on.
#[target_feature(enable = "simd128")]
#[inline]
fn bounds(icdf: v128, index: v128, scale: v128, last: v128) -> v128 {
    // `((range >> 8) * (icdf >> 6)) >> 1`: the high halves of the products
    // of `(icdf >> 6) << 7` (below 65536 because icdf < 32768) and
    // `(range >> 8) << 8`, picked out of the 32-bit products.
    let probability = u16x8_shl(u16x8_shr(icdf, PROBABILITY_SHIFT), PRODUCT_SHIFT);
    let low = u32x4_extmul_low_u16x8(probability, scale);
    let high = u32x4_extmul_high_u16x8(probability, scale);
    let split =
        i8x16_shuffle::<2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31>(low, high);
    // MIN_PROBABILITY * (last - i), wrapping past `last` (masked below).
    let offset = u16x8_shl(u16x8_sub(last, index), MIN_PROBABILITY.ilog2());
    v128_and(u16x8_add(split, offset), u16x8_lt(index, last))
}

/// How many lanes have a bound above `value` (a prefix).
#[target_feature(enable = "simd128")]
#[inline]
fn above(bounds: v128, value: v128) -> usize {
    u16x8_bitmask(u16x8_gt(bounds, value)).count_ones() as usize
}

#[target_feature(enable = "simd128")]
#[inline]
pub(super) fn find_symbol<const N: usize>(
    icdf: &[u16; N],
    symbols: usize,
    range: u32,
    value: u32,
) -> Av1SymbolInterval {
    let scale = u16x8_splat((range & 0xff00) as u16);
    let last = u16x8_splat((symbols - 1) as u16);
    let value_lanes = u16x8_splat(value as u16);
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
#[target_feature(enable = "simd128")]
#[inline]
fn adapt(icdf: v128, index: v128, symbol: v128, rate: u32) -> v128 {
    let towards_top = u16x8_add(
        icdf,
        u16x8_shr(u16x8_sub(u16x8_splat(PROB_TOP), icdf), rate),
    );
    let towards_zero = u16x8_sub(icdf, u16x8_shr(icdf, rate));
    v128_bitselect(towards_top, towards_zero, u16x8_lt(index, symbol))
}

#[target_feature(enable = "simd128")]
#[inline]
pub(super) fn adapt_cdf<const N: usize>(icdf: &mut [u16; N], symbol: usize, rate: u32) {
    let symbol = u16x8_splat(symbol as u16);
    let mut lanes: [u16; 2 * LANES] = padded(icdf);
    let low = adapt(load(&lanes[..LANES]), lane_index(false), symbol, rate);
    store(&mut lanes[..LANES], low);
    if N > LANES {
        let high = adapt(load(&lanes[LANES..]), lane_index(true), symbol, rate);
        store(&mut lanes[LANES..], high);
    }
    icdf.copy_from_slice(&lanes[..N]);
}
