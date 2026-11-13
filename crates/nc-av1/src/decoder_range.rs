//! Independent AV1 arithmetic decoder (AV1 §8.2.2–8.2.6: `init_symbol`,
//! `read_symbol`, `read_bool`, `read_literal`, `exit_symbol`) and the
//! adaptive CDFs it reads with (§8.2.6).
//!
//! The reader keeps the spec's `SymbolValue` in the top 16 bits of a 64-bit
//! window whose lower bits hold up to 48 further (inverted) stream bits, so
//! renormalisation is one shift by `15 - FloorLog2(SymbolRange)` and a
//! refill once every few symbols, not a loop over single bits; symbols are
//! found by counting interval bounds above the value, without data-dependent
//! branches. This follows the design of dav1d's `src/msac.c`
//! (BSD-2-Clause, Copyright (c) 2018, `VideoLAN` and dav1d authors; see
//! `THIRD_PARTY.md`), reimplemented, not transliterated.
//!
//! Reading never fails: past the end of the data the window fills with the
//! spec's padding, and [`SymbolReader::overran`] reports a tile that read
//! far beyond it (a truncated or corrupt tile), checked once per tile
//! instead of per symbol.
use crate::cdf::{ResetCounters, adaptation_rate, advance_counter};
use nc_dsp::av1_symbol::{av1_adapt_cdf, av1_find_symbol};

/// Bits dropped from a probability before the interval split
/// (`EC_PROB_SHIFT`).
const PROBABILITY_SHIFT: u32 = 6;
/// The interval each symbol keeps at least (`EC_MIN_PROB`, per symbol
/// above the coded one).
const MIN_PROBABILITY: u32 = 4;
/// Bits of the window.
const WINDOW_BITS: u32 = u64::BITS;
/// Bits of `SymbolValue` (at the top of the window) and of `SymbolRange`.
const VALUE_BITS: u32 = 16;
/// Where `SymbolValue` starts in the window.
const VALUE_SHIFT: u32 = WINDOW_BITS - VALUE_BITS;
/// The renormalised range is at least `1 << RANGE_TOP_BIT` (§8.2.6 shifts
/// until bit 15 is set).
const RANGE_TOP_BIT: u32 = 15;
/// `init_symbol` reads 15 bits into `SymbolValue` (§8.2.2), so the window
/// starts that many bits short of full.
const INIT_BITS: i32 = 15;
/// Refills past the end of the data (each adds 48 bits of the spec's
/// padding) before the tile counts as overrun. `SymbolValue` of a
/// conformant tile reaches at most 15 bits into the padding (§8.2.4), which
/// can take a second refill when the first found only a few bits of
/// padding room; a third means `SymbolValue` reached at least 49 bits past
/// the data, so the tile is truncated or corrupt.
const MAX_PADDING_REFILLS: u32 = 2;
/// The fixed 1/2 probability of `read_literal` bits, as an inverse CDF.
const HALF: u16 = 1 << 14;

/// One adaptive CDF of up to `N` symbols as the decoder keeps it: the
/// inverse CDF (entries past the alphabet in use are zero padding) and its
/// adaptation counter. Unlike the encoder's [`crate::cdf::Cdf`] it carries
/// no symbol-cost cache, so a tile's CDF set is half the size to copy and
/// to keep in cache, and adapting stores nothing beyond the CDF (about 20M
/// instructions per stream).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Cdf<const N: usize> {
    pub icdf: [u16; N],
    count: u8,
}

impl<const N: usize> Cdf<N> {
    /// A fresh CDF (counter 0) with inverse CDF `icdf`.
    pub(crate) const fn new(icdf: [u16; N]) -> Self {
        Self { icdf, count: 0 }
    }
    /// Move the CDF towards `symbol` of an `nsyms`-symbol alphabet (the
    /// update at the end of `read_symbol`, §8.2.6), all entries at once.
    #[inline]
    fn adapt(&mut self, symbol: usize, nsyms: usize) {
        av1_adapt_cdf(
            &mut self.icdf,
            symbol,
            nsyms,
            adaptation_rate(self.count, nsyms),
        );
        advance_counter(&mut self.count);
    }
}

impl<const N: usize> ResetCounters for Cdf<N> {
    fn reset_counters(&mut self) {
        self.count = 0;
    }
}

/// Builds the decoder's adaptive CDFs from a (nested) default table.
pub(crate) trait DecoderDefaults {
    /// The same nesting with each inverse CDF made a [`Cdf`].
    type Cdfs;
    /// Fresh CDFs with these defaults.
    fn cdfs(&self) -> Self::Cdfs;
}

impl<const N: usize> DecoderDefaults for [u16; N] {
    type Cdfs = Cdf<N>;
    fn cdfs(&self) -> Cdf<N> {
        Cdf::new(*self)
    }
}

impl<T: DecoderDefaults, const M: usize> DecoderDefaults for [T; M] {
    type Cdfs = [T::Cdfs; M];
    fn cdfs(&self) -> Self::Cdfs {
        std::array::from_fn(|i| self[i].cdfs())
    }
}

/// The symbol decoder of one tile.
pub(crate) struct SymbolReader<'a> {
    /// The tile's data not yet in the window.
    data: &'a [u8],
    /// `SymbolValue` in the top [`VALUE_BITS`], then the next stream bits,
    /// inverted (the spec's `SymbolValue` is built from inverted bits).
    window: u64,
    /// `SymbolRange`.
    range: u32,
    /// Stream bits in the window below `SymbolValue`; negative when a
    /// renormalisation shifted in more than were there.
    bits: i32,
    /// Refills that found the data exhausted.
    padding_refills: u32,
    /// Whether decoded symbols adapt their CDFs (`disable_cdf_update = 0`).
    adapt: bool,
}

impl<'a> SymbolReader<'a> {
    /// `init_symbol()` (§8.2.2) over a tile's data.
    pub(crate) fn new(data: &'a [u8]) -> Self {
        let mut reader = Self {
            data,
            window: 0,
            range: 1 << RANGE_TOP_BIT,
            bits: -INIT_BITS,
            padding_refills: 0,
            adapt: true,
        };
        reader.refill();
        reader
    }
    /// This reader with symbols adapting their CDFs (`adapt`) or not (a
    /// frame with `disable_cdf_update = 1`, AV1 §5.9.2).
    pub(crate) fn adapting(self, adapt: bool) -> Self {
        Self { adapt, ..self }
    }
    /// Whether the tile read well past the end of its data (truncated or
    /// corrupt): symbols read since are padding, not stream.
    pub(crate) fn overran(&self) -> bool {
        self.padding_refills > MAX_PADDING_REFILLS
    }

    /// Fill the window below its valid bits with whole bytes, inverted;
    /// past the end of the data, with the padding's ones.
    #[inline(never)]
    fn refill(&mut self) {
        // The bit position of the next byte's least significant bit.
        let mut shift = VALUE_SHIFT as i32 - 8 - self.bits;
        loop {
            let Some((&byte, rest)) = self.data.split_first() else {
                // Every bit below the valid ones becomes padding.
                let top = shift + 8;
                self.window |= if top >= WINDOW_BITS as i32 {
                    u64::MAX
                } else {
                    (1u64 << top) - 1
                };
                self.padding_refills += 1;
                shift = -8;
                break;
            };
            self.data = rest;
            self.window |= u64::from(!byte) << shift;
            shift -= 8;
            if shift < 0 {
                break;
            }
        }
        self.bits = VALUE_SHIFT as i32 - 8 - shift;
    }

    /// The renormalisation of §8.2.6 after an interval of `range` was
    /// chosen and its lower bound already taken from the window.
    #[inline]
    fn renormalize(&mut self, window: u64, range: u32) {
        let shift = range.leading_zeros() - (u32::BITS - 1 - RANGE_TOP_BIT);
        self.window = window << shift;
        self.range = range << shift;
        self.bits -= shift as i32;
        if self.bits < 0 {
            self.refill();
        }
    }

    /// `SymbolValue`.
    #[inline]
    fn value(&self) -> u32 {
        (self.window >> VALUE_SHIFT) as u32
    }

    /// The bound of the interval above symbol `symbol` in an `n`-symbol
    /// alphabet: symbols up to `symbol` lie at or above it (`cur` of
    /// §8.2.6 `read_symbol`).
    #[inline]
    fn bound(&self, icdf: u16, symbol: usize, n: usize) -> u32 {
        (((self.range >> 8) * (u32::from(icdf) >> PROBABILITY_SHIFT)) >> (7 - PROBABILITY_SHIFT))
            + MIN_PROBABILITY * (n - 1 - symbol) as u32
    }

    /// `read_symbol` (§8.2.6) with inverse CDF `icdf` of a
    /// `symbols`-symbol alphabet, without adaptation: the interval search
    /// is nc-dsp's, over all bounds at once.
    #[inline]
    fn decode<const N: usize>(&mut self, icdf: &[u16; N], symbols: usize) -> usize {
        let interval = av1_find_symbol(icdf, symbols, self.range, self.value());
        self.renormalize(
            self.window - (u64::from(interval.lower) << VALUE_SHIFT),
            interval.upper - interval.lower,
        );
        interval.symbol
    }

    /// A two-symbol `read_symbol` whose symbol 0 has inverse CDF `icdf0`:
    /// true for symbol 1.
    #[inline]
    fn decode_bool(&mut self, icdf0: u16) -> bool {
        let bound = self.bound(icdf0, 0, 2);
        let below = u64::from(bound) << VALUE_SHIFT;
        // Symbol 0 lies above the bound, symbol 1 below it. The interval is
        // picked arithmetically (a mask), not by a branch the data decides.
        let one = self.window < below;
        let zero_mask = u64::from(one).wrapping_sub(1);
        let range = if one { bound } else { self.range - bound };
        self.renormalize(self.window - (below & zero_mask), range);
        one
    }

    /// `read_symbol` with an adaptive CDF of a `symbols`-symbol alphabet.
    #[inline]
    pub(crate) fn symbol<const N: usize>(&mut self, cdf: &mut Cdf<N>, symbols: usize) -> usize {
        let symbol = self.decode(&cdf.icdf, symbols);
        if self.adapt {
            cdf.adapt(symbol, symbols);
        }
        symbol
    }
    /// A binary symbol: [`Self::symbol`] with the two-symbol interval
    /// split computed directly (the most frequent symbol kind).
    #[inline]
    pub(crate) fn boolean(&mut self, cdf: &mut Cdf<2>) -> bool {
        let one = self.decode_bool(cdf.icdf[0]);
        if self.adapt {
            cdf.adapt(usize::from(one), 2);
        }
        one
    }
    /// A binary symbol with a fixed (non-adapting) symbol-0 inverse CDF,
    /// such as the partition flags gathered from a larger CDF (§8.3.2).
    #[inline]
    pub(crate) fn fixed_boolean(&mut self, icdf0: u16) -> bool {
        self.decode_bool(icdf0)
    }
    /// `read_literal(bits)` (§8.2.5): equiprobable bits, most significant
    /// first.
    pub(crate) fn literal(&mut self, bits: u32) -> u32 {
        let mut value = 0;
        for _ in 0..bits {
            value = (value << 1) | u32::from(self.decode_bool(HALF));
        }
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cdf::Cdf as EncoderCdf;
    use crate::range_coder::SymbolWriter;
    #[test]
    fn independent_adaptive_and_literal_roundtrip() {
        let mut w = SymbolWriter::new();
        let mut cdf = EncoderCdf::new([30000, 16000, 5000, 0]);
        let mut cdf16 = EncoderCdf::new([
            30000, 28000, 26000, 24000, 22000, 20000, 18000, 16000, 14000, 12000, 10000, 8000,
            6000, 4000, 2000, 0,
        ]);
        let mut state = 12345u32;
        let mut symbols = Vec::new();
        for _ in 0..10000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let s = (state >> 30) as usize;
            let t = (state >> 12) as usize % 16;
            symbols.push((s, state & 1, t, state & 0x3ff));
            w.write_symbol(s, &mut cdf, 4);
            w.write_literal_bit(state & 1 != 0);
            w.write_symbol(t, &mut cdf16, 16);
            for bit in (0..10).rev() {
                w.write_literal_bit(state >> bit & 1 != 0);
            }
        }
        let bytes = w.finish();
        let mut r = SymbolReader::new(&bytes);
        let mut decoded = Cdf::new([30000, 16000, 5000, 0]);
        let mut decoded16 = Cdf::new([
            30000, 28000, 26000, 24000, 22000, 20000, 18000, 16000, 14000, 12000, 10000, 8000,
            6000, 4000, 2000, 0,
        ]);
        for (s, b, t, bits) in symbols {
            assert_eq!(r.symbol(&mut decoded, 4), s);
            assert_eq!(r.literal(1), b);
            assert_eq!(r.symbol(&mut decoded16, 16), t);
            assert_eq!(r.literal(10), bits);
        }
        assert_eq!(cdf.icdf, decoded.icdf);
        assert!(!r.overran());
    }

    #[test]
    fn reading_far_past_the_data_is_reported() {
        let mut r = SymbolReader::new(&[0x55]);
        r.literal(16);
        assert!(!r.overran());
        r.literal(200);
        assert!(r.overran());
    }
}
