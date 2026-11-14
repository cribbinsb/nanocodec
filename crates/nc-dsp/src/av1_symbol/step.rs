//! One symbol's interval, optional adaptation and normalization; input
//! refill and adaptation counters remain owned by the codec's reader.

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

/// Decode a symbol with storage capacity 5..=8. The inverse CDF follows
/// the parent contract; `window >> 48 < range`, with normalized `range`.
/// `Some(rate)` adapts at rate 4..=7; `None` leaves the CDF unchanged.
/// This operation does not access input bytes or update the CDF counter.
#[inline]
pub fn av1_decode_symbol_step<const N: usize>(
    icdf: &mut [u16; N],
    symbols: usize,
    window: u64,
    range: u32,
    adapt_rate: Option<u32>,
) -> Av1SymbolStep {
    assert!((5..=8).contains(&N), "symbol step capacity is 5..=8");
    debug_assert!(super::check_contract(icdf, symbols));
    debug_assert!((32768..=65535).contains(&range) && window >> 48 < u64::from(range));
    debug_assert!(adapt_rate.is_none_or(|r| (4..=7).contains(&r)));
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    return super::neon_step::decode(icdf, symbols, window, range, adapt_rate);
    #[allow(unreachable_code)]
    super::reference::decode_symbol_step(icdf, symbols, window, range, adapt_rate)
}

#[cfg(all(target_arch = "aarch64", not(miri)))]
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
