//! One symbol's interval, optional adaptation and normalization; input
//! refill and adaptation counters remain owned by the codec's reader.

use super::{AV1_MAX_SYMBOLS, MIN_PROBABILITY, PROB_TOP, PROBABILITY_SHIFT};

/// Normalized arithmetic-reader state after one decoded symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Av1SymbolStep {
    /// Decoded alphabet index.
    pub symbol: usize,
    /// Inverted bit window after subtracting the interval and shifting.
    pub window: u64,
    /// Normalized interval, in 32768..=65535.
    pub range: u32,
    /// Bits consumed by normalization; the caller refills if necessary.
    pub shift: u32,
}

/// Decode a symbol with storage capacity `3..=16`. The inverse CDF follows
/// the parent contract; `window >> 48 < range`, with normalized `range`.
/// `Some(rate)` adapts at rate `4..=7`; `None` leaves the CDF unchanged.
/// This operation does not access input bytes or update the CDF counter.
#[inline]
pub fn av1_decode_symbol_step<const N: usize>(
    icdf: &mut [u16; N],
    symbols: usize,
    window: u64,
    range: u32,
    adapt_rate: Option<u32>,
) -> Av1SymbolStep {
    assert!(
        (3..=AV1_MAX_SYMBOLS).contains(&N),
        "symbol step capacity is 3..=16"
    );
    debug_assert!(super::check_contract(icdf, symbols));
    debug_assert!((32768..=65535).contains(&range) && window >> 48 < u64::from(range));
    debug_assert!(adapt_rate.is_none_or(|r| (4..=7).contains(&r)));
    if N == 3 {
        let icdf3: &mut [u16; 3] = (&mut icdf[..3]).try_into().expect("3 lanes");
        return three(icdf3, symbols, window, range, adapt_rate);
    }
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    return super::sse2_step::decode(icdf, symbols, window, range, adapt_rate);
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    return super::neon_step::decode(icdf, symbols, window, range, adapt_rate);
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    return super::simd128_step::decode(icdf, symbols, window, range, adapt_rate);
    #[allow(unreachable_code)]
    super::reference::decode_symbol_step(icdf, symbols, window, range, adapt_rate)
}

/// Pure-GPR fused step for 3-symbol capacity (`coeff_base_eob`): at most two
/// nonzero bounds (`icdf[2] == 0` by contract), extracted and adapted without
/// vector round-trips or data-dependent branches.
#[inline]
fn three(
    icdf: &mut [u16; 3],
    symbols: usize,
    window: u64,
    range: u32,
    adapt_rate: Option<u32>,
) -> Av1SymbolStep {
    let value = (window >> 48) as u32;
    let scale = range >> 8;
    let b0 = (((scale * (u32::from(icdf[0]) >> PROBABILITY_SHIFT)) >> (7 - PROBABILITY_SHIFT))
        + MIN_PROBABILITY * (symbols - 1) as u32) as u16;
    let b1 = if symbols > 2 {
        (((scale * (u32::from(icdf[1]) >> PROBABILITY_SHIFT)) >> (7 - PROBABILITY_SHIFT))
            + MIN_PROBABILITY) as u16
    } else {
        0
    };
    let symbol = usize::from(value < u32::from(b0)) + usize::from(value < u32::from(b1));
    if let Some(rate) = adapt_rate {
        let adapt1 = |p: u16, towards_top: bool| -> u16 {
            let p = u32::from(p);
            (if towards_top {
                p + ((u32::from(PROB_TOP) - p) >> rate)
            } else {
                p - (p >> rate)
            }) as u16
        };
        icdf[0] = adapt1(icdf[0], symbol > 0);
        if symbols > 2 {
            icdf[1] = adapt1(icdf[1], symbol > 1);
        }
    }
    let lo = u64::from(b0) | (u64::from(b1) << 16);
    let (lower, upper) = interval_4(range, lo, symbol);
    normalize(symbol, window, lower, upper - lower)
}

/// `(lower, upper)` for `symbol` in `0..4` from packed `[b0, b1, b2, 0]`.
#[inline]
pub(super) fn interval_4(range: u32, lo: u64, symbol: usize) -> (u32, u32) {
    let uppers = (lo << 16) | u64::from(range as u16);
    let shift = (symbol as u32) << 4;
    (
        ((lo >> shift) & 0xffff) as u32,
        ((uppers >> shift) & 0xffff) as u32,
    )
}

/// `(lower, upper)` for `symbol` in `0..8` from packed `[b0..b3]` and `[b4..b6, 0]`.
#[allow(dead_code, reason = "used by SSE2 and SIMD128 steps")]
#[inline]
pub(super) fn interval_8(range: u32, lo: u64, hi: u64, symbol: usize) -> (u32, u32) {
    let up_lo = (lo << 16) | u64::from(range as u16);
    let up_hi = (hi << 16) | (lo >> 48);
    let shift = ((symbol & 3) as u32) << 4;
    let (l_word, u_word) = if symbol < 4 { (lo, up_lo) } else { (hi, up_hi) };
    (
        ((l_word >> shift) & 0xffff) as u32,
        ((u_word >> shift) & 0xffff) as u32,
    )
}

/// `(lower, upper)` for `symbol` in `0..16` from four packed `u64` words.
#[allow(dead_code, reason = "used by SIMD steps")]
#[inline]
pub(super) fn interval_16(
    range: u32,
    (lo0, hi0, lo1, hi1): (u64, u64, u64, u64),
    symbol: usize,
) -> (u32, u32) {
    if symbol < 8 {
        interval_8(range, lo0, hi0, symbol)
    } else {
        interval_8((hi0 >> 48) as u32, lo1, hi1, symbol - 8)
    }
}

#[allow(dead_code, reason = "used by SIMD steps and `three`")]
#[inline]
pub(super) fn normalize(symbol: usize, window: u64, lower: u32, range: u32) -> Av1SymbolStep {
    let shift = range.leading_zeros() - 16;
    Av1SymbolStep {
        symbol,
        window: (window - (u64::from(lower) << 48)) << shift,
        range: range << shift,
        shift,
    }
}
