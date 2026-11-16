//! The AV1 symbol (range) encoder: the inverse of the decoder in AV1 §8.2.
//!
//! This is the Daala/libaom `od_ec` encoder (libaom `aom_dsp/entenc.c`,
//! BSD-2-Clause, Alliance for Open Media) in the 64-bit-window form of
//! libaom and SVT-AV1 v4.3.0 (`Source/Lib/Codec/bitstream_unit.c`,
//! BSD-3-Clause-Clear, Alliance for Open Media): `low` holds up to 40
//! pending bits, so bytes leave it only every five or six symbol bytes,
//! from an out-of-line flush that writes them straight into the output and
//! adds any carry into the bytes already written. Binary symbols and
//! literal bits take the two-symbol split directly (SVT `aom_write_symbol`).
//!
//! Frames whose syntax has gaps filled after reconstruction (CDEF indices,
//! loop restoration units) record their symbols as frozen intervals in one
//! flat log and code them in [`SymbolWriter::finish_frame`], run by run
//! between the inserted spans. The log and the output keep their capacity
//! from frame to frame.
//!
//! CDFs are in libaom's inverse convention (`32768 - cdf`, last entry 0).
//! Symbols coded with a [`Cdf`] adapt it afterwards unless the frame sets
//! `disable_cdf_update` ([`SymbolWriter::restart`]); booleans and
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
/// `cnt` of a fresh encoder: it crosses zero once a byte plus a carry bit
/// have accumulated in `low`.
const CNT_START: i32 = -9;
/// Pending bits after which `low` is flushed: its 64 bits less a carry
/// byte and the 16 bits the next symbol may shift in (libaom `56 - 16`).
const FLUSH_BITS: i32 = 40;
/// Bits of the 64-bit window above [`FLUSH_BITS`].
const WINDOW_SPARE: i32 = 64 - FLUSH_BITS;
/// Bytes a flush may write: one whole window.
const WINDOW_BYTES: usize = 8;
/// Initial capacity of the output and of the interval log.
const INITIAL_CAPACITY: usize = 1 << 12;

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
    #[allow(
        clippy::inline_always,
        reason = "one call per coded symbol; out of line its call overhead was ~20 instructions a symbol (callgrind, AV1 Default)"
    )]
    #[inline(always)]
    fn write_symbol<const N: usize>(&mut self, s: usize, cdf: &mut Cdf<N>, nsyms: usize) {
        SymbolWriter::write_symbol(self, s, cdf, nsyms);
    }

    #[inline]
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
    span: &'a mut Span,
}

impl SpanWriter<'_> {
    fn push(&mut self, interval: Interval) {
        self.intervals.push(interval);
        self.span.end = self.intervals.len();
    }
}

impl SymbolSink for SpanWriter<'_> {
    fn write_symbol<const N: usize>(&mut self, s: usize, cdf: &mut Cdf<N>, nsyms: usize) {
        self.push(Interval::symbol(s, &cdf.icdf, nsyms));
        if self.adapt {
            cdf.adapt(s, nsyms);
        }
    }

    fn write_literal_bit(&mut self, bit: bool) {
        self.push(Interval::literal(bit));
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
    #[inline]
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

    #[inline]
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

/// The symbols of one [`SpanSlot`]: where they go (the number of intervals
/// logged before them) and their run `start..end` in the span log.
#[derive(Debug, Default)]
struct Span {
    position: usize,
    start: usize,
    end: usize,
}

/// The symbols of a deferred frame, in stream order but for the spans,
/// whose symbols are logged apart as they are filled in.
#[derive(Debug, Default)]
struct Recording {
    intervals: Vec<Interval>,
    spans: Vec<Span>,
    span_intervals: Vec<Interval>,
}

impl Recording {
    fn clear(&mut self) {
        self.intervals.clear();
        self.spans.clear();
        self.span_intervals.clear();
    }
}

/// `r_hi` (the range's top eight bits) times probability `f` at the coder's
/// resolution.
#[inline]
fn scale(r_hi: u32, f: u32) -> u32 {
    (r_hi * (f >> PROB_SHIFT)) >> (7 - PROB_SHIFT)
}

/// Add one to the number whose last byte ends `out` (a carry out of `low`).
fn propagate_carry(out: &mut [u8]) {
    for byte in out.iter_mut().rev() {
        let (sum, overflow) = byte.overflowing_add(1);
        *byte = sum;
        if !overflow {
            return;
        }
    }
}

/// The arithmetic core: interval state and the bytes it has finished.
#[derive(Debug)]
struct RangeEncoder {
    out: Vec<u8>,
    low: u64,
    rng: u32,
    /// Pending bits in `low`, offset by [`CNT_START`].
    cnt: i32,
}

impl RangeEncoder {
    fn new() -> Self {
        Self {
            out: Vec::with_capacity(INITIAL_CAPACITY),
            low: 0,
            rng: 0x8000,
            cnt: CNT_START,
        }
    }

    /// Code the frozen interval `iv` (libaom `od_ec_encode_q15`).
    #[inline]
    fn encode(&mut self, iv: Interval) {
        let r = self.rng;
        let r_hi = r >> 8;
        let reserve = MIN_PROB * u32::from(iv.remaining);
        let fl = u32::from(iv.fl);
        let (low, upper) = if fl < PROB_TOP {
            let u = scale(r_hi, fl) + reserve + MIN_PROB;
            (self.low + u64::from(r - u), u)
        } else {
            (self.low, r)
        };
        self.normalize(low, upper - (scale(r_hi, u32::from(iv.fh)) + reserve));
    }

    /// Code `bit` of a two-symbol alphabet whose inverse CDF starts at `f`
    /// (libaom `od_ec_encode_bool_q15`; equal to [`Self::encode`] of its
    /// interval).
    #[inline]
    fn encode_bool(&mut self, bit: bool, f: u32) {
        let r = self.rng;
        let v = scale(r >> 8, f) + MIN_PROB;
        let (low, rng) = if bit {
            (self.low + u64::from(r - v), v)
        } else {
            (self.low, r - v)
        };
        self.normalize(low, rng);
    }

    /// Renormalise `rng` back into [32768, 65536), flushing `low` when it
    /// is about to overflow its window.
    #[inline]
    fn normalize(&mut self, low: u64, rng: u32) {
        debug_assert!((1..=0xffff).contains(&rng));
        // Leading zeros of `rng` in 16 bits.
        let d = rng.leading_zeros() as i32 - 16;
        if self.cnt + d >= FLUSH_BITS {
            self.flush(low, rng, d);
        } else {
            self.low = low << d;
            self.rng = rng << d;
            self.cnt += d;
        }
    }

    /// Move the finished whole bytes of `low` to the output, carrying into
    /// the bytes already there (libaom `od_ec_enc_normalize`, flush branch).
    #[cold]
    #[inline(never)]
    fn flush(&mut self, low: u64, rng: u32, d: i32) {
        let s = self.cnt + d;
        // Whole bytes ready, plus the byte `cnt`'s offset holds back.
        let ready_bits = (s & !7) + 8;
        let kept_bits = (self.cnt + WINDOW_SPARE - ready_bits) as u32;
        let output = low >> kept_bits;
        if output & (1 << ready_bits) != 0 {
            propagate_carry(&mut self.out);
        }
        // The carry bit falls off the top of the shifted window.
        let window = (output << (64 - ready_bits)).to_be_bytes();
        let len = self.out.len();
        self.out.extend_from_slice(&window);
        self.out.truncate(len + (ready_bits >> 3) as usize);
        debug_assert!((ready_bits >> 3) as usize <= WINDOW_BYTES);
        self.low = (low & ((1 << kept_bits) - 1)) << d;
        self.rng = rng << d;
        self.cnt = (s & 7) - 8;
    }

    /// Flush and return the coded bytes: the minimum number of bits that
    /// decode the symbols correctly whatever follows (libaom
    /// `od_ec_enc_done`). The encoder starts afresh.
    fn finish(&mut self) -> Vec<u8> {
        // Round `low` up to the shortest bit pattern inside the final range,
        // then emit its remaining bytes.
        let mask: u64 = 0x3fff;
        let mut end = ((self.low + mask) & !mask) | (mask + 1);
        let mut c = self.cnt;
        // `cnt >= CNT_START`, so at least one byte is emitted.
        while 10 + c > 0 {
            let shift = (c + 16) as u32;
            let value = end >> shift;
            if value & 0x100 != 0 {
                propagate_carry(&mut self.out);
            }
            self.out.push(value as u8);
            end &= (1 << shift) - 1;
            c -= 8;
        }
        let capacity = self.out.capacity();
        let out = std::mem::replace(&mut self.out, Vec::with_capacity(capacity));
        self.reset();
        out
    }

    /// Drop everything coded and start afresh, keeping the output buffer.
    fn reset(&mut self) {
        self.out.clear();
        self.low = 0;
        self.rng = 0x8000;
        self.cnt = CNT_START;
    }
}

/// Writes symbols into an AV1 tile's entropy-coded data.
#[derive(Debug)]
pub struct SymbolWriter {
    coder: RangeEncoder,
    /// The deferred frame's symbols, coded at [`Self::finish_frame`].
    recording: Recording,
    /// Whether symbols are logged rather than coded at once.
    deferred: bool,
    /// Whether coded symbols adapt their CDFs (`disable_cdf_update = 0`).
    adapt: bool,
}

impl Default for SymbolWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl SymbolWriter {
    /// A fresh encoder that codes symbols as they are written.
    #[must_use]
    pub fn new() -> Self {
        Self {
            coder: RangeEncoder::new(),
            recording: Recording::default(),
            deferred: false,
            adapt: true,
        }
    }

    /// Record intervals for later coding, allowing fixed bits to be filled in.
    #[cfg(test)]
    #[must_use]
    pub fn new_deferred() -> Self {
        let mut writer = Self::new();
        writer.restart(true, true);
        writer
    }

    /// Start a new tile on this writer, keeping its buffers. `deferred`
    /// logs symbols for [`Self::finish_frame`] so that reserved literals and
    /// spans can be filled in later; `adapt` false is a frame with
    /// `disable_cdf_update = 1` (AV1 §5.9.2): symbols leave their CDFs as
    /// they were, so the frame codes and prices every symbol with the CDFs
    /// it started from.
    pub fn restart(&mut self, deferred: bool, adapt: bool) {
        self.coder.reset();
        self.recording.clear();
        if deferred && self.recording.intervals.capacity() == 0 {
            self.recording.intervals.reserve(INITIAL_CAPACITY);
        }
        self.deferred = deferred;
        self.adapt = adapt;
    }

    fn intervals(&mut self) -> &mut Vec<Interval> {
        assert!(self.deferred, "writer must be deferred");
        &mut self.recording.intervals
    }

    /// Reserve a literal bit in a deferred stream (initially false).
    pub fn reserve_literal(&mut self) -> LiteralSlot {
        let intervals = self.intervals();
        let slot = LiteralSlot(intervals.len());
        intervals.push(Interval::literal(false));
        slot
    }

    /// Choose a reserved literal without changing any adaptive CDF state.
    pub fn set_literal(&mut self, slot: LiteralSlot, bit: bool) {
        self.intervals()[slot.0] = Interval::literal(bit);
    }

    /// Remove a reserved bit when the frame instead signals one global strength.
    pub fn omit_literal(&mut self, slot: LiteralSlot) {
        self.intervals()[slot.0].remaining = OMITTED_INTERVAL;
    }

    /// Reserve a place for symbols chosen later ([`Self::span`]).
    pub fn reserve_span(&mut self) -> SpanSlot {
        let position = self.intervals().len();
        self.recording.spans.push(Span {
            position,
            ..Span::default()
        });
        SpanSlot(self.recording.spans.len() - 1)
    }

    /// A sink coding into the reserved place `slot`, replacing anything
    /// coded there before. Its symbols adapt their CDFs as they are
    /// written, so spans sharing a CDF are filled in stream order.
    pub fn span(&mut self, slot: SpanSlot) -> SpanWriter<'_> {
        let Recording {
            spans,
            span_intervals,
            ..
        } = &mut self.recording;
        let span = &mut spans[slot.0];
        span.start = span_intervals.len();
        span.end = span.start;
        SpanWriter {
            adapt: self.adapt,
            intervals: span_intervals,
            span,
        }
    }

    /// Code symbol `s` from an alphabet of `nsyms` symbols with `cdf`, then
    /// adapt the CDF (the inverse of AV1 §8.2.6 `read_symbol`).
    #[allow(
        clippy::inline_always,
        reason = "one call per coded symbol; out of line its call overhead was ~20 instructions a symbol (callgrind, AV1 Default)"
    )]
    #[inline(always)]
    pub fn write_symbol<const N: usize>(&mut self, s: usize, cdf: &mut Cdf<N>, nsyms: usize) {
        if self.deferred {
            self.recording
                .intervals
                .push(Interval::symbol(s, &cdf.icdf, nsyms));
        } else if N == 2 {
            debug_assert_eq!(nsyms, 2);
            self.coder.encode_bool(s != 0, u32::from(cdf.icdf[0]));
        } else {
            self.coder.encode(Interval::symbol(s, &cdf.icdf, nsyms));
        }
        if self.adapt {
            cdf.adapt(s, nsyms);
        }
    }

    /// Code a binary symbol with `cdf`, then adapt it.
    #[inline]
    pub fn write_bool(&mut self, bit: bool, cdf: &mut Cdf<2>) {
        self.write_symbol(usize::from(bit), cdf, 2);
    }

    /// Code symbol `s` with a CDF that does not adapt: one derived for this
    /// symbol only (`split_or_horz` / `split_or_vert`, AV1 §8.3.2).
    pub fn write_fixed(&mut self, s: usize, icdf: &[u16], nsyms: usize) {
        let interval = Interval::symbol(s, icdf, nsyms);
        if self.deferred {
            self.recording.intervals.push(interval);
        } else {
            self.coder.encode(interval);
        }
    }

    /// Code one equiprobable bit: AV1 `L(1)` (§4.10.2, `read_literal`).
    #[inline]
    pub fn write_literal_bit(&mut self, bit: bool) {
        if self.deferred {
            self.recording.intervals.push(Interval::literal(bit));
        } else {
            self.coder.encode_bool(bit, HALF);
        }
    }

    /// Code what a deferred frame logged, flush, and return the tile's
    /// bytes. The writer then starts a new tile like the one it finished,
    /// keeping its buffers.
    pub fn finish_frame(&mut self) -> Vec<u8> {
        if self.deferred {
            let Recording {
                intervals,
                spans,
                span_intervals,
            } = &self.recording;
            let mut start = 0;
            for span in spans {
                self.coder.encode_run(&intervals[start..span.position]);
                self.coder.encode_run(&span_intervals[span.start..span.end]);
                start = span.position;
            }
            self.coder.encode_run(&intervals[start..]);
            self.recording.clear();
        }
        self.coder.finish()
    }

    /// Flush and return the coded bytes ([`Self::finish_frame`]).
    #[cfg(test)]
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        self.finish_frame()
    }
}

impl RangeEncoder {
    /// Code a run of logged intervals, skipping omitted literals.
    fn encode_run(&mut self, intervals: &[Interval]) {
        for &interval in intervals {
            if interval.remaining != OMITTED_INTERVAL {
                self.encode(interval);
            }
        }
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
