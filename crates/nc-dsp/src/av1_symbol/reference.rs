//! Scalar references of the symbol kernels, written as the spec's loops
//! (AV1 §8.2.6 `read_symbol`).
use super::{Av1SymbolInterval, MIN_PROBABILITY, PROB_TOP, PROBABILITY_SHIFT};

/// `cur` of symbol `i` of an `symbols`-symbol alphabet: symbols up to `i`
/// lie at or above it.
fn bound(icdf: &[u16], i: usize, symbols: usize, range: u32) -> u32 {
    (((range >> 8) * (u32::from(icdf[i]) >> PROBABILITY_SHIFT)) >> (7 - PROBABILITY_SHIFT))
        + MIN_PROBABILITY * (symbols - 1 - i) as u32
}

/// [`super::av1_find_symbol`]: the first symbol whose bound is at or
/// below `value`.
#[must_use]
pub fn find_symbol<const N: usize>(
    icdf: &[u16; N],
    symbols: usize,
    range: u32,
    value: u32,
) -> Av1SymbolInterval {
    let mut upper = range;
    for symbol in 0..symbols {
        let lower = bound(icdf, symbol, symbols, range);
        if value >= lower {
            return Av1SymbolInterval {
                symbol,
                lower,
                upper,
            };
        }
        upper = lower;
    }
    unreachable!("the last symbol's bound is 0")
}

/// [`super::av1_adapt_cdf`].
pub fn adapt_cdf<const N: usize>(icdf: &mut [u16; N], symbol: usize, symbols: usize, rate: u32) {
    for (i, p) in icdf[..symbols - 1].iter_mut().enumerate() {
        if i < symbol {
            *p += (PROB_TOP - *p) >> rate;
        } else {
            *p -= *p >> rate;
        }
    }
}

/// Full symbol step, using the independent scalar interval and adaptation loops.
#[must_use]
pub fn decode_symbol_step<const N: usize>(
    icdf: &mut [u16; N],
    symbols: usize,
    window: u64,
    range: u32,
    rate: Option<u32>,
) -> super::Av1SymbolStep {
    let interval = find_symbol(icdf, symbols, range, (window >> 48) as u32);
    if let Some(rate) = rate {
        adapt_cdf(icdf, interval.symbol, symbols, rate);
    }
    let range = interval.upper - interval.lower;
    let shift = range.leading_zeros() - 16;
    super::Av1SymbolStep {
        symbol: interval.symbol,
        window: (window - (u64::from(interval.lower) << 48)) << shift,
        range: range << shift,
        shift,
    }
}
