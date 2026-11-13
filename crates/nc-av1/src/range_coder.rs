//! The AV1 symbol (range) encoder: the inverse of the decoder in AV1 §8.2.
//!
//! This is the Daala/libaom `od_ec` encoder (libaom `aom_dsp/entenc.c`,
//! BSD-2-Clause, Alliance for Open Media), restructured: a 32-bit `low`
//! window, output bytes buffered with their carry bit ("precarry") and the
//! carries resolved once in [`SymbolWriter::finish`].
//!
//! CDFs are in libaom's inverse convention (`32768 - cdf`, last entry 0).
//! Symbols coded with a [`Cdf`] adapt it afterwards unless the frame sets
//! `disable_cdf_update` ([`SymbolWriter::without_adaptation`]); booleans and
//! literals (`L(n)`) use fixed probabilities (AV1 §8.2.3).

use crate::cdf::Cdf;
use nc_encoder::symbol_cost::BitCounter;

/// Probability scale: CDFs are 15-bit fixed point (AV1 §4.10.7).
const PROB_TOP: u32 = 1 << 15;
/// Low bits of each probability dropped before the range multiply
/// (`EC_PROB_SHIFT`).
const PROB_SHIFT: u32 = 6;
/// Minimum range reserved per symbol so none has zero width (`EC_MIN_PROB`).
const MIN_PROB: u32 = 4;
/// Probability of a literal bit being 1, as used by `L(1)` (AV1 §8.2.3).
const HALF: u32 = PROB_TOP / 2;
/// Reserved interval marker: real alphabets have at most sixteen symbols.
const OMITTED_INTERVAL: u16 = u16::MAX;

/// Something symbols can be coded into: the real [`SymbolWriter`], or a
/// [`BitCounter`] that only adds up their cost. Syntax writers are generic
/// over it, so rate estimates follow exactly the code that writes the bits.
pub trait SymbolSink {
    /// Code symbol `s` of `nsyms` with `cdf` (adapting it, if the sink codes).
    fn write_symbol<const N: usize>(&mut self, s: usize, cdf: &mut Cdf<N>, nsyms: usize);
    /// Code one equiprobable bit, `L(1)`.
    fn write_literal_bit(&mut self, bit: bool);

    /// Code a binary symbol with `cdf`.
    fn write_bool(&mut self, bit: bool, cdf: &mut Cdf<2>) {
        self.write_symbol(usize::from(bit), cdf, 2);
    }

    /// Code the low `n` bits of `value`, most significant first: AV1 `L(n)`.
    fn write_literal(&mut self, value: u32, n: u32) {
        for i in (0..n).rev() {
            self.write_literal_bit((value >> i) & 1 != 0);
        }
    }
}

/// Rate estimation: adds up the cost of what would be coded, leaving the
/// CDFs untouched (integer 1/512-bit costs from each CDF's cache,
/// [`Cdf::cost_q9`]).
impl SymbolSink for BitCounter {
    fn write_symbol<const N: usize>(&mut self, s: usize, cdf: &mut Cdf<N>, _nsyms: usize) {
        self.add_q9(cdf.cost_q9(s));
    }

    fn write_literal_bit(&mut self, _bit: bool) {
        self.add_bypass(1);
    }
}

impl SymbolSink for SymbolWriter {
    fn write_symbol<const N: usize>(&mut self, s: usize, cdf: &mut Cdf<N>, nsyms: usize) {
        SymbolWriter::write_symbol(self, s, cdf, nsyms);
    }

    fn write_literal_bit(&mut self, bit: bool) {
        SymbolWriter::write_literal_bit(self, bit);
    }
}

/// Codes symbols into a reserved place of a deferred [`SymbolWriter`]
/// ([`SymbolWriter::span`]).
#[derive(Debug)]
pub struct SpanWriter<'a> {
    adapt: bool,
    intervals: &'a mut Vec<Interval>,
}

impl SymbolSink for SpanWriter<'_> {
    fn write_symbol<const N: usize>(&mut self, s: usize, cdf: &mut Cdf<N>, nsyms: usize) {
        self.intervals.push(Interval::symbol(s, &cdf.icdf, nsyms));
        if self.adapt {
            cdf.adapt(s, nsyms);
        }
    }

    fn write_literal_bit(&mut self, bit: bool) {
        self.intervals.push(Interval::literal(bit));
    }
}

/// One frozen probability interval, before range normalization.
#[derive(Debug, Clone, Copy)]
struct Interval {
    fl: u16,
    fh: u16,
    remaining: u16,
}

impl Interval {
    /// Symbol `s` of an alphabet of `nsyms` with inverse CDF `icdf`.
    fn symbol(s: usize, icdf: &[u16], nsyms: usize) -> Self {
        debug_assert!(s < nsyms && nsyms <= icdf.len());
        debug_assert_eq!(icdf[nsyms - 1], 0, "CDF not terminated");
        let fl = if s > 0 {
            u32::from(icdf[s - 1])
        } else {
            PROB_TOP
        };
        Self {
            fl: fl as u16,
            fh: icdf[s],
            remaining: (nsyms - 1 - s) as u16,
        }
    }

    fn literal(bit: bool) -> Self {
        if bit {
            Self {
                fl: HALF as u16,
                fh: 0,
                remaining: 0,
            }
        } else {
            Self {
                fl: PROB_TOP as u16,
                fh: HALF as u16,
                remaining: 1,
            }
        }
    }
}

/// A deferred fixed-probability bit that can be chosen after reconstruction.
#[derive(Debug, Clone, Copy)]
pub struct LiteralSlot(usize);

/// A place in a deferred stream where symbols decided after reconstruction
/// are inserted (the loop restoration units each superblock codes before
/// its partition, AV1 §5.11.57).
#[derive(Debug, Clone, Copy)]
pub struct SpanSlot(usize);

/// The symbols of one [`SpanSlot`]: where they go (the number of
/// intervals recorded before them) and their intervals.
#[derive(Debug, Default)]
struct Span {
    position: usize,
    intervals: Vec<Interval>,
}

/// Writes symbols into an AV1 tile's entropy-coded data.
#[derive(Debug)]
pub struct SymbolWriter {
    /// Output bytes, each with a possible pending carry in bit 8.
    precarry: Vec<u16>,
    /// Deferred intervals preserve adaptive probabilities without coding twice.
    intervals: Option<Vec<Interval>>,
    /// Symbols inserted into the deferred stream after the fact, in stream
    /// order.
    spans: Vec<Span>,
    low: u32,
    rng: u32,
    /// Bits in `low` not yet ready for output, offset by −9 (see `new`).
    cnt: i32,
    /// Whether coded symbols adapt their CDFs (`disable_cdf_update = 0`).
    adapt: bool,
}

impl Default for SymbolWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl SymbolWriter {
    /// A fresh encoder. `cnt` starts at −9 so it crosses zero once a byte
    /// plus a carry bit have accumulated.
    #[must_use]
    pub fn new() -> Self {
        Self {
            precarry: Vec::with_capacity(1 << 12),
            intervals: None,
            spans: Vec::new(),
            low: 0,
            rng: 0x8000,
            cnt: -9,
            adapt: true,
        }
    }

    /// This writer for a frame with `disable_cdf_update = 1` (AV1 §5.9.2):
    /// symbols leave their CDFs as they were, so the frame codes and prices
    /// every symbol with the CDFs it started from.
    #[must_use]
    pub fn without_adaptation(self) -> Self {
        Self {
            adapt: false,
            ..self
        }
    }

    /// Record intervals for later coding, allowing fixed bits to be filled in.
    #[must_use]
    pub fn new_deferred() -> Self {
        Self {
            intervals: Some(Vec::with_capacity(1 << 12)),
            ..Self::new()
        }
    }

    /// Reserve a literal bit in a deferred stream (initially false).
    pub fn reserve_literal(&mut self) -> LiteralSlot {
        let intervals = self.intervals.as_mut().expect("writer must be deferred");
        let slot = LiteralSlot(intervals.len());
        intervals.push(Interval::literal(false));
        slot
    }

    /// Choose a reserved literal without changing any adaptive CDF state.
    pub fn set_literal(&mut self, slot: LiteralSlot, bit: bool) {
        let intervals = self.intervals.as_mut().expect("writer must be deferred");
        intervals[slot.0] = Interval::literal(bit);
    }

    /// Remove a reserved bit when the frame instead signals one global strength.
    pub fn omit_literal(&mut self, slot: LiteralSlot) {
        let intervals = self.intervals.as_mut().expect("writer must be deferred");
        intervals[slot.0].remaining = OMITTED_INTERVAL;
    }

    /// Reserve a place for symbols chosen later ([`Self::span`]).
    pub fn reserve_span(&mut self) -> SpanSlot {
        let position = self
            .intervals
            .as_ref()
            .expect("writer must be deferred")
            .len();
        self.spans.push(Span {
            position,
            intervals: Vec::new(),
        });
        SpanSlot(self.spans.len() - 1)
    }

    /// A sink coding into the reserved place `slot`. Its symbols adapt
    /// their CDFs as they are written, so spans sharing a CDF are filled in
    /// stream order.
    pub fn span(&mut self, slot: SpanSlot) -> SpanWriter<'_> {
        SpanWriter {
            adapt: self.adapt,
            intervals: &mut self.spans[slot.0].intervals,
        }
    }

    /// Code symbol `s` from an alphabet of `nsyms` symbols with `cdf`, then
    /// adapt the CDF (the inverse of AV1 §8.2.6 `read_symbol`).
    pub fn write_symbol<const N: usize>(&mut self, s: usize, cdf: &mut Cdf<N>, nsyms: usize) {
        self.write_fixed(s, &cdf.icdf, nsyms);
        if self.adapt {
            cdf.adapt(s, nsyms);
        }
    }

    /// Code a binary symbol with `cdf`, then adapt it.
    pub fn write_bool(&mut self, bit: bool, cdf: &mut Cdf<2>) {
        self.write_symbol(usize::from(bit), cdf, 2);
    }

    /// Code symbol `s` with a CDF that does not adapt: one derived for this
    /// symbol only (`split_or_horz` / `split_or_vert`, AV1 §8.3.2).
    pub fn write_fixed(&mut self, s: usize, icdf: &[u16], nsyms: usize) {
        let interval = Interval::symbol(s, icdf, nsyms);
        if let Some(intervals) = self.intervals.as_mut() {
            intervals.push(interval);
        } else {
            self.write_interval(interval);
        }
    }

    fn write_interval(&mut self, interval: Interval) {
        let fl = u32::from(interval.fl);
        let fh = u32::from(interval.fh);
        let remaining = u32::from(interval.remaining);
        let r = self.rng;
        let scaled = |f: u32| ((r >> 8) * (f >> PROB_SHIFT)) >> (7 - PROB_SHIFT);
        let (low, rng) = if fl < PROB_TOP {
            let u = scaled(fl) + MIN_PROB * (remaining + 1);
            let v = scaled(fh) + MIN_PROB * remaining;
            (self.low + (r - u), u - v)
        } else {
            (self.low, r - (scaled(fh) + MIN_PROB * remaining))
        };
        self.normalize(low, rng);
    }

    /// Code one equiprobable bit: AV1 `L(1)` (§4.10.2, `read_literal`).
    pub fn write_literal_bit(&mut self, bit: bool) {
        if let Some(intervals) = self.intervals.as_mut() {
            intervals.push(Interval::literal(bit));
            return;
        }
        let r = self.rng;
        let v = (((r >> 8) * (HALF >> PROB_SHIFT)) >> (7 - PROB_SHIFT)) + MIN_PROB;
        let (low, rng) = if bit {
            (self.low + (r - v), v)
        } else {
            (self.low, r - v)
        };
        self.normalize(low, rng);
    }

    /// Renormalise `rng` back into [32768, 65536), moving finished bytes of
    /// `low` into the output.
    fn normalize(&mut self, mut low: u32, rng: u32) {
        debug_assert!((1..=0xffff).contains(&rng));
        let d = 16 - (32 - rng.leading_zeros()) as i32; // leading zeros in 16 bits
        let mut c = self.cnt;
        let mut s = c + d;
        if s >= 0 {
            c += 16;
            let mut mask = (1u32 << c) - 1;
            if s >= 8 {
                self.precarry.push((low >> c) as u16);
                low &= mask;
                c -= 8;
                mask >>= 8;
            }
            self.precarry.push((low >> c) as u16);
            s = c + d - 24;
            low &= mask;
        }
        self.low = low << d;
        self.rng = rng << d;
        self.cnt = s;
    }

    /// Flush and return the coded bytes: the minimum number of bits that
    /// decode the symbols correctly whatever follows (libaom `od_ec_enc_done`).
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        if let Some(intervals) = self.intervals.take() {
            let spans = std::mem::take(&mut self.spans);
            let mut spans = spans.into_iter().peekable();
            for (position, interval) in intervals.into_iter().enumerate() {
                while let Some(span) = spans.next_if(|span| span.position == position) {
                    span.intervals
                        .into_iter()
                        .for_each(|i| self.write_interval(i));
                }
                if interval.remaining != OMITTED_INTERVAL {
                    self.write_interval(interval);
                }
            }
            for span in spans {
                span.intervals
                    .into_iter()
                    .for_each(|i| self.write_interval(i));
            }
        }
        // Round `low` up to the shortest bit pattern inside the final range,
        // then emit its remaining bytes.
        let mut pending = self.cnt;
        let mut remaining_bits = 10 + pending;
        let mask: u32 = 0x3fff;
        let mut end = ((self.low + mask) & !mask) | (mask + 1);
        if remaining_bits > 0 {
            let mut keep = (1u32 << (pending + 16)) - 1;
            loop {
                self.precarry.push((end >> (pending + 16)) as u16);
                end &= keep;
                remaining_bits -= 8;
                pending -= 8;
                keep >>= 8;
                if remaining_bits <= 0 {
                    break;
                }
            }
        }
        let mut out = vec![0u8; self.precarry.len()];
        let mut carry = 0u32;
        for (dst, &v) in out.iter_mut().zip(&self.precarry).rev() {
            carry += u32::from(v);
            *dst = carry as u8;
            carry >>= 8;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_stream_flushes_to_minimal_bytes() {
        // With no symbols the flush still emits the terminating bits.
        let out = SymbolWriter::new().finish();
        assert!(!out.is_empty() && out.len() <= 2);
    }

    #[test]
    fn deferred_adaptation_and_late_literals_match_immediate() {
        let mut immediate = SymbolWriter::new();
        let mut deferred = SymbolWriter::new_deferred();
        let mut direct_cdf = Cdf::new([30000, 16000, 5000, 0]);
        let mut deferred_cdf = direct_cdf;
        let fixed = [24576, 16384, 8192, 0];
        let mut slots = Vec::new();
        let mut state = 0x1234_5678_u32;
        for i in 0..100_000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let symbol = (state >> 30) as usize;
            immediate.write_symbol(symbol, &mut direct_cdf, 4);
            deferred.write_symbol(symbol, &mut deferred_cdf, 4);
            immediate.write_fixed(3 - symbol, &fixed, 4);
            deferred.write_fixed(3 - symbol, &fixed, 4);
            let bit = state & 1 != 0;
            immediate.write_literal_bit(bit);
            if i % 17 == 0 {
                slots.push((deferred.reserve_literal(), bit));
            } else {
                deferred.write_literal_bit(bit);
            }
        }
        for (slot, bit) in slots.into_iter().rev() {
            deferred.set_literal(slot, bit);
        }
        assert_eq!(direct_cdf.icdf, deferred_cdf.icdf);
        assert_eq!(immediate.finish(), deferred.finish());
        assert_eq!(
            SymbolWriter::new().finish(),
            SymbolWriter::new_deferred().finish()
        );
    }

    #[test]
    fn omitted_late_literal_matches_a_stream_without_that_bit() {
        let mut direct = SymbolWriter::new();
        let mut deferred = SymbolWriter::new_deferred();
        let fixed = [24576, 16384, 8192, 0];
        direct.write_fixed(2, &fixed, 4);
        deferred.write_fixed(2, &fixed, 4);
        let slot = deferred.reserve_literal();
        direct.write_fixed(1, &fixed, 4);
        deferred.write_fixed(1, &fixed, 4);
        deferred.omit_literal(slot);
        assert_eq!(direct.finish(), deferred.finish());
    }

    #[test]
    fn spans_filled_late_match_symbols_coded_in_place() {
        let fixed = [24576, 16384, 8192, 0];
        let mut direct = SymbolWriter::new();
        let mut deferred = SymbolWriter::new_deferred();
        let (mut direct_cdf, mut deferred_cdf) = (Cdf::new([20000, 0]), Cdf::new([20000, 0]));
        let mut slots = Vec::new();
        for i in 0..1000usize {
            // A span at the very start, between symbols, and at the end.
            if i % 7 == 0 {
                direct.write_bool(i % 3 == 0, &mut direct_cdf);
                direct.write_literal(i as u32 & 31, 5);
                slots.push(deferred.reserve_span());
            }
            direct.write_fixed(i % 4, &fixed, 4);
            deferred.write_fixed(i % 4, &fixed, 4);
        }
        direct.write_bool(true, &mut direct_cdf);
        let last = deferred.reserve_span();
        for (n, slot) in slots.into_iter().enumerate() {
            let i = n * 7;
            let mut span = deferred.span(slot);
            span.write_bool(i % 3 == 0, &mut deferred_cdf);
            span.write_literal(i as u32 & 31, 5);
        }
        deferred.span(last).write_bool(true, &mut deferred_cdf);
        assert_eq!(direct_cdf.icdf, deferred_cdf.icdf);
        assert_eq!(direct.finish(), deferred.finish());
    }

    #[test]
    fn many_symbols_do_not_overflow() {
        let icdf = [24576u16, 16384, 8192, 0];
        let mut w = SymbolWriter::new();
        for i in 0..100_000usize {
            w.write_fixed(i % 4, &icdf, 4);
            w.write_literal_bit(i % 3 == 0);
        }
        let out = w.finish();
        // ~2 bits per symbol + 1 per literal bit.
        assert!(out.len() > 30_000 && out.len() < 45_000, "{}", out.len());
    }
}
