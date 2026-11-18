//! Fused WebAssembly SIMD128 symbol step, guided by dav1d 1.5.4 `src/x86/msac.asm`
//! and `src/arm/64/msac.S` (BSD-2-Clause, Copyright 2019 VideoLAN/dav1d authors;
//! `THIRD_PARTY.md`). Interval search and CDF adaptation share the loaded
//! probabilities and comparison mask without linear-memory stack spills on the
//! hot 4-, 8- and 16-symbol capacities.
use super::{
    Av1SymbolStep, LANES, MIN_PROBABILITY, PROBABILITY_SHIFT, padded,
    step::{interval_4, interval_8, interval_16, normalize},
};
use core::arch::wasm32::{
    i8x16_shuffle, i16x8_shr, i16x8_splat, i16x8_sub, u16x8, u16x8_add, u16x8_bitmask, u16x8_gt,
    u16x8_lt, u16x8_shl, u16x8_shr, u16x8_splat, u16x8_sub, u32x4_extmul_high_u16x8,
    u32x4_extmul_low_u16x8, u64x2_extract_lane, v128, v128_and, v128_load, v128_load64_zero,
    v128_store, v128_store64_lane,
};

const PRODUCT_SHIFT: u32 = 16 - 8 - (7 - PROBABILITY_SHIFT);

#[target_feature(enable = "simd128")]
#[inline]
fn lane_index(high: bool) -> v128 {
    if high {
        u16x8(8, 9, 10, 11, 12, 13, 14, 15)
    } else {
        u16x8(0, 1, 2, 3, 4, 5, 6, 7)
    }
}

#[target_feature(enable = "simd128")]
#[inline]
fn bounds8(icdf: v128, index: v128, scale: v128, last: v128) -> v128 {
    let prob = u16x8_shl(u16x8_shr(icdf, PROBABILITY_SHIFT), PRODUCT_SHIFT);
    let low = u32x4_extmul_low_u16x8(prob, scale);
    let high = u32x4_extmul_high_u16x8(prob, scale);
    let split =
        i8x16_shuffle::<2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31>(low, high);
    let offset = u16x8_shl(u16x8_sub(last, index), MIN_PROBABILITY.ilog2());
    v128_and(u16x8_add(split, offset), u16x8_lt(index, last))
}

/// Signed biased adaptation: `p - ((p - (before & (32769 - 2^rate))) >>s rate)`.
#[target_feature(enable = "simd128")]
#[inline]
fn adapt8(p: v128, before: v128, rate: u32) -> v128 {
    let bias = i16x8_splat((32769 - (1u32 << rate)) as i16);
    let delta = i16x8_sub(p, v128_and(before, bias));
    i16x8_sub(p, i16x8_shr(delta, rate))
}

#[target_feature(enable = "simd128")]
#[inline]
fn four(
    icdf: &mut [u16; 4],
    symbols: usize,
    window: u64,
    range: u32,
    rate: Option<u32>,
) -> Av1SymbolStep {
    // SAFETY: `icdf` holds four initialized `u16`s (8 bytes).
    let p = unsafe { v128_load64_zero(icdf.as_ptr().cast()) };
    let scale = u16x8_splat((range & 0xff00) as u16);
    let last = u16x8_splat((symbols - 1) as u16);
    let val = u16x8_splat((window >> 48) as u16);
    let b = bounds8(p, lane_index(false), scale, last);
    let before = u16x8_gt(b, val);
    let symbol = u16x8_bitmask(before).count_ones() as usize;
    if let Some(rate) = rate {
        // SAFETY: `icdf` is 8 writable bytes.
        unsafe { v128_store64_lane::<0>(adapt8(p, before, rate), icdf.as_mut_ptr().cast()) };
    }
    let (lower, upper) = interval_4(range, u64x2_extract_lane::<0>(b), symbol);
    normalize(symbol, window, lower, upper - lower)
}

#[target_feature(enable = "simd128")]
#[inline]
fn eight<const N: usize>(
    icdf: &mut [u16; N],
    symbols: usize,
    window: u64,
    range: u32,
    rate: Option<u32>,
) -> Av1SymbolStep {
    let p = if N == LANES {
        // SAFETY: `N == 8`, so `icdf` is 16 readable bytes.
        unsafe { v128_load(icdf.as_ptr().cast()) }
    } else {
        let lanes: [u16; LANES] = padded(icdf);
        // SAFETY: `lanes` is 16 readable bytes.
        unsafe { v128_load(lanes.as_ptr().cast()) }
    };
    let scale = u16x8_splat((range & 0xff00) as u16);
    let last = u16x8_splat((symbols - 1) as u16);
    let val = u16x8_splat((window >> 48) as u16);
    let b = bounds8(p, lane_index(false), scale, last);
    let before = u16x8_gt(b, val);
    let symbol = u16x8_bitmask(before).count_ones() as usize;
    if let Some(rate) = rate {
        let adapted = adapt8(p, before, rate);
        if N == LANES {
            // SAFETY: `N == 8`, so `icdf` is 16 writable bytes.
            unsafe { v128_store(icdf.as_mut_ptr().cast(), adapted) };
        } else {
            let mut lanes = [0u16; LANES];
            // SAFETY: `lanes` is 16 writable bytes.
            unsafe { v128_store(lanes.as_mut_ptr().cast(), adapted) };
            icdf.copy_from_slice(&lanes[..N]);
        }
    }
    let (lo, hi) = (u64x2_extract_lane::<0>(b), u64x2_extract_lane::<1>(b));
    let (lower, upper) = interval_8(range, lo, hi, symbol);
    normalize(symbol, window, lower, upper - lower)
}

#[target_feature(enable = "simd128")]
#[inline]
fn sixteen<const N: usize>(
    icdf: &mut [u16; N],
    symbols: usize,
    window: u64,
    range: u32,
    rate: Option<u32>,
) -> Av1SymbolStep {
    let (p0, p1) = if N == 2 * LANES {
        // SAFETY: `N == 16`, so `icdf` is 32 readable bytes.
        unsafe {
            let ptr = icdf.as_ptr();
            (v128_load(ptr.cast()), v128_load(ptr.add(LANES).cast()))
        }
    } else {
        let lanes: [u16; 2 * LANES] = padded(icdf);
        // SAFETY: `lanes` is 32 readable bytes.
        unsafe {
            let ptr = lanes.as_ptr();
            (v128_load(ptr.cast()), v128_load(ptr.add(LANES).cast()))
        }
    };
    let scale = u16x8_splat((range & 0xff00) as u16);
    let last = u16x8_splat((symbols - 1) as u16);
    let val = u16x8_splat((window >> 48) as u16);
    let b0 = bounds8(p0, lane_index(false), scale, last);
    let b1 = bounds8(p1, lane_index(true), scale, last);
    let (bf0, bf1) = (u16x8_gt(b0, val), u16x8_gt(b1, val));
    let symbol = (u16x8_bitmask(bf0).count_ones() + u16x8_bitmask(bf1).count_ones()) as usize;
    if let Some(rate) = rate {
        let (a0, a1) = (adapt8(p0, bf0, rate), adapt8(p1, bf1, rate));
        if N == 2 * LANES {
            // SAFETY: `N == 16`, so `icdf` is 32 writable bytes.
            unsafe {
                let ptr = icdf.as_mut_ptr();
                v128_store(ptr.cast(), a0);
                v128_store(ptr.add(LANES).cast(), a1);
            }
        } else {
            let mut lanes = [0u16; 2 * LANES];
            // SAFETY: `lanes` is 32 writable bytes.
            unsafe {
                let ptr = lanes.as_mut_ptr();
                v128_store(ptr.cast(), a0);
                v128_store(ptr.add(LANES).cast(), a1);
            }
            icdf.copy_from_slice(&lanes[..N]);
        }
    }
    let bounds64 = (
        u64x2_extract_lane::<0>(b0),
        u64x2_extract_lane::<1>(b0),
        u64x2_extract_lane::<0>(b1),
        u64x2_extract_lane::<1>(b1),
    );
    let (lower, upper) = interval_16(range, bounds64, symbol);
    normalize(symbol, window, lower, upper - lower)
}

#[inline]
pub(super) fn decode<const N: usize>(
    icdf: &mut [u16; N],
    symbols: usize,
    window: u64,
    range: u32,
    rate: Option<u32>,
) -> Av1SymbolStep {
    if N == 4 {
        let icdf4: &mut [u16; 4] = (&mut icdf[..4]).try_into().expect("4 lanes");
        four(icdf4, symbols, window, range, rate)
    } else if N <= LANES {
        eight(icdf, symbols, window, range, rate)
    } else {
        sixteen(icdf, symbols, window, range, rate)
    }
}
