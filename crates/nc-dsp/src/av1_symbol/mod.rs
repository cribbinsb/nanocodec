//! AV1 (and AV2) arithmetic-decoder symbol kernels (AV1 §8.2.6
//! `read_symbol`): finding the decoded symbol's interval from an inverse
//! CDF, and the CDF adaptation that follows it.
//!
//! Both work on all of a CDF's entries at once, in 16-bit lanes, as dav1d's
//! `src/x86/msac.asm` does (BSD-2-Clause, Copyright (c) 2019, `VideoLAN` and
//! dav1d authors; see `THIRD_PARTY.md`); reimplemented, not transliterated.
//! A symbol is decoded per few coded bits, so these run tens of millions of
//! times per second of video and sit inline in the decoders' syntax code.
//!
//! # No runtime dispatch
//!
//! The SIMD versions use only each target's baseline vector unit: SSE2 on
//! x86-64, NEON (ASIMD) on aarch64, and SIMD128 on wasm32 builds compiled
//! with it (the browser module). They are chosen at compile time, not
//! through a [`crate::Dsp`] table: an indirect call per symbol would cost
//! more than the kernel saves, and no CPU check is needed for a baseline
//! feature. Other targets use [`reference`]. `nc-checkasm` compares the
//! compiled-in version with [`reference`].
//!
//! # Contract
//!
//! An inverse CDF `icdf` of an `n`-symbol alphabet (`2 <= n <= N <= 16`)
//! holds `32768 - CDF(i)` for each symbol `i`: decreasing, below 32768,
//! with `icdf[n - 1] = 0` and zero padding after it. Adaptation keeps all
//! of that true.

#[cfg(all(target_arch = "aarch64", not(miri)))]
#[allow(unsafe_code)]
mod neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
mod simd128;
#[cfg(all(target_arch = "x86_64", not(miri)))]
#[allow(unsafe_code)]
mod sse2;

pub mod reference;

/// Bits dropped from a probability before the interval split
/// (`EC_PROB_SHIFT`).
pub(crate) const PROBABILITY_SHIFT: u32 = 6;
/// The interval each symbol keeps at least (`EC_MIN_PROB`, per symbol
/// above the coded one).
pub(crate) const MIN_PROBABILITY: u32 = 4;
/// The largest alphabet.
pub const AV1_MAX_SYMBOLS: usize = 16;
/// Probability scale: CDFs are 15-bit fixed point.
pub(crate) const PROB_TOP: u16 = 1 << 15;
/// Lanes of one vector of the SIMD versions.
#[allow(dead_code, reason = "used by the SIMD versions only")]
pub(crate) const LANES: usize = 8;

/// The interval a decoded symbol takes in `SymbolRange` (§8.2.6): the
/// symbol, and the bounds `cur` and `prev` below and above it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Av1SymbolInterval {
    /// The decoded symbol.
    pub symbol: usize,
    /// The interval's lower bound, which `SymbolValue` loses.
    pub lower: u32,
    /// The interval's upper bound: the new range is `upper - lower`.
    pub upper: u32,
}

/// The symbol whose interval holds `value` (`SymbolValue`) in a range of
/// `range` (`SymbolRange`, 32768..=65535, above `value`), for inverse CDF
/// `icdf` of a `symbols`-symbol alphabet (module contract).
#[inline]
#[must_use]
pub fn av1_find_symbol<const N: usize>(
    icdf: &[u16; N],
    symbols: usize,
    range: u32,
    value: u32,
) -> Av1SymbolInterval {
    debug_assert!(check_contract(icdf, symbols) && value < range && range <= 0xffff);
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    return sse2::find_symbol(icdf, symbols, range, value);
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    return neon::find_symbol(icdf, symbols, range, value);
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    return simd128::find_symbol(icdf, symbols, range, value);
    #[allow(unreachable_code)]
    reference::find_symbol(icdf, symbols, range, value)
}

/// Move inverse CDF `icdf` of a `symbols`-symbol alphabet towards
/// `symbol` at adaptation `rate` (§8.2.6; the rate from the CDF's counter
/// and alphabet size is the caller's).
#[inline]
pub fn av1_adapt_cdf<const N: usize>(
    icdf: &mut [u16; N],
    symbol: usize,
    symbols: usize,
    rate: u32,
) {
    debug_assert!(check_contract(icdf, symbols) && symbol < symbols);
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    return sse2::adapt_cdf(icdf, symbol, rate);
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    return neon::adapt_cdf(icdf, symbol, rate);
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    return simd128::adapt_cdf(icdf, symbol, rate);
    #[allow(unreachable_code)]
    reference::adapt_cdf(icdf, symbol, symbols, rate);
}

/// Whether `icdf` meets the module contract for `symbols` symbols.
#[must_use]
pub fn check_contract<const N: usize>(icdf: &[u16; N], symbols: usize) -> bool {
    (2..=N).contains(&symbols)
        && N <= AV1_MAX_SYMBOLS
        && icdf[symbols - 1..].iter().all(|&p| p == 0)
        && icdf.windows(2).all(|w| w[0] >= w[1])
        && icdf[0] < PROB_TOP
}

/// `icdf` zero-padded to `M` lanes (the SIMD versions' vector widths).
#[allow(dead_code, reason = "used by the SIMD versions only")]
#[inline]
pub(crate) fn padded<const N: usize, const M: usize>(icdf: &[u16; N]) -> [u16; M] {
    let mut lanes = [0; M];
    lanes[..N].copy_from_slice(icdf);
    lanes
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small deterministic generator (no test dependency on `rand`).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self, bound: u32) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((self.0 >> 33) % u64::from(bound)) as u32
        }
    }

    /// A random inverse CDF meeting the module contract.
    fn random_icdf<const N: usize>(rng: &mut Lcg, symbols: usize) -> [u16; N] {
        let mut icdf = [0u16; N];
        let mut previous = u32::from(PROB_TOP) - 1;
        for p in &mut icdf[..symbols - 1] {
            previous = rng.next(previous + 1);
            *p = previous as u16;
        }
        icdf
    }

    fn check<const N: usize>(rng: &mut Lcg) {
        for symbols in 2..=N {
            for _ in 0..2000 {
                let icdf: [u16; N] = random_icdf(rng, symbols);
                let range = (1 << 15) + rng.next(1 << 15);
                let value = rng.next(range);
                assert_eq!(
                    av1_find_symbol(&icdf, symbols, range, value),
                    reference::find_symbol(&icdf, symbols, range, value),
                    "{icdf:?} symbols {symbols} range {range} value {value}"
                );
                let symbol = rng.next(symbols as u32) as usize;
                let rate = 4 + rng.next(4);
                let (mut fast, mut slow) = (icdf, icdf);
                av1_adapt_cdf(&mut fast, symbol, symbols, rate);
                reference::adapt_cdf(&mut slow, symbol, symbols, rate);
                assert_eq!(fast, slow, "{icdf:?} symbol {symbol} rate {rate}");
            }
        }
    }

    #[test]
    fn compiled_kernels_match_the_reference() {
        let mut rng = Lcg(7);
        check::<3>(&mut rng);
        check::<4>(&mut rng);
        check::<5>(&mut rng);
        check::<8>(&mut rng);
        check::<9>(&mut rng);
        check::<13>(&mut rng);
        check::<16>(&mut rng);
    }
}
