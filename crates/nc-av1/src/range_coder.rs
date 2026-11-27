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
//! between the inserted spans. Every logged symbol, binary, literal or
//! omitted, is one general interval, so the replay is a single branch-free
//! step with the coder state in registers. The log and the output keep their
//! capacity from frame to frame.
//!
//! CDFs are in libaom's inverse convention (`32768 - cdf`, last entry 0).
//! Symbols coded with a [`Cdf`] adapt it afterwards unless the frame sets
//! `disable_cdf_update` ([`SymbolWriter::restart`]); booleans and
//! literals (`L(n)`) use fixed probabilities (AV1 §8.2.3).

use std::sync::OnceLock;

use nc_encoder::symbol_cost::BitCounter;

use crate::cdf::{Cdf, CdfContext, CdfJournal};

/// Probability scale: CDFs are 15-bit fixed point (AV1 §4.10.7).
const PROB_TOP: u32 = 1 << 15;
/// Low bits of each probability dropped before the range multiply
/// (`EC_PROB_SHIFT`).
const PROB_SHIFT: u32 = 6;
/// Minimum range reserved per symbol so none has zero width (`EC_MIN_PROB`).
const MIN_PROB: u32 = 4;
/// Probability of a literal bit being 1, as used by `L(1)` (AV1 §8.2.3).
const HALF: u32 = PROB_TOP / 2;
/// Maximum literal bits in one deferred log record.
pub(crate) const MAX_LITERAL_RUN: u32 = 24;
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

    fn write_literal(&mut self, _value: u32, n: u32) {
        self.add_bypass(n);
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

    #[inline]
    fn write_literal(&mut self, value: u32, n: u32) {
        if self.deferred && (2..=MAX_LITERAL_RUN).contains(&n) {
            self.recording.intervals.push(Interval::literals(value, n));
        } else if !self.deferred && (2..=MAX_LITERAL_RUN).contains(&n) {
            let (low, range, count) = (self.coder.low, self.coder.rng, self.coder.cnt);
            let (low, range, count) = self.coder.encode_literals(value, n, low, range, count);
            (self.coder.low, self.coder.rng, self.coder.cnt) = (low, range, count);
        } else {
            for i in (0..n).rev() {
                self.write_literal_bit((value >> i) & 1 != 0);
            }
        }
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

/// Symbols logged after a mark ([`SymbolWriter::save_after`]), to be put
/// back after other codings were tried.
#[derive(Debug, Default)]
pub struct SymbolTail {
    intervals: Vec<Interval>,
}

impl SymbolTail {
    /// Forget the noted symbols.
    pub fn clear(&mut self) {
        self.intervals.clear();
    }
}

/// One logged symbol, frozen before range normalization: the interval of
/// symbol `s` in an alphabet of `nsyms` (libaom `od_ec_encode_q15`'s
/// arguments). A two-symbol code or literal bit is the same interval of a
/// two-symbol alphabet, and an omitted literal the whole range of a
/// one-symbol alphabet, which leaves the coder exactly as it was.
#[derive(Debug, Clone, Copy)]
struct Interval(u32);

impl Interval {
    // Store the bits that range coding consumes in the low 25 bits. Pricing
    // also needs the difference of the discarded six-bit tails, in [-63, 63].
    const UPPER_BITS: u32 = 10; // includes PROB_TOP >> PROB_SHIFT
    const LOWER_BITS: u32 = 9; // a lower edge is strictly below PROB_TOP
    const REMAINING_BITS: u32 = 6; // AV1 alphabets have at most 16 symbols
    const REMAINING_SHIFT: u32 = Self::UPPER_BITS + Self::LOWER_BITS;
    const TAIL_SHIFT: u32 = Self::REMAINING_SHIFT + Self::REMAINING_BITS;

    // The two top bits of the upper edge are never both set for a
    // probability (0..=512). A literal run uses that pair as its tag,
    // and the top six bits as the run length.
    const LITERAL_TAG: u32 = 0b11 << 8;
    const LITERAL_LENGTH: u32 = 26;

    #[inline]
    fn literals(value: u32, n: u32) -> Self {
        let bits = value & ((1 << n) - 1);
        Self(Self::LITERAL_TAG | (bits & 0xff) | ((bits >> 8) << 10) | (n << Self::LITERAL_LENGTH))
    }

    #[inline]
    fn is_literal_run(self) -> bool {
        self.0 & Self::LITERAL_TAG == Self::LITERAL_TAG
    }

    #[inline]
    fn literal_count(self) -> u32 {
        self.0 >> Self::LITERAL_LENGTH
    }

    #[inline]
    fn literal_bits(self) -> u32 {
        (self.0 & 0xff) | (((self.0 >> 10) & 0xffff) << 8)
    }

    #[inline]
    const fn new(fl: u32, fh: u32, remaining: u32) -> Self {
        debug_assert!(fh <= fl && fl <= PROB_TOP && fh < PROB_TOP);
        debug_assert!(remaining < (1 << Self::REMAINING_BITS));
        let mask = (1 << PROB_SHIFT) - 1;
        let tail = (fl & mask).wrapping_sub(fh & mask);
        Self(
            (fl >> PROB_SHIFT)
                | ((fh >> PROB_SHIFT) << Self::UPPER_BITS)
                | (remaining << Self::REMAINING_SHIFT)
                | (tail << Self::TAIL_SHIFT),
        )
    }

    #[inline]
    fn upper(self) -> u32 {
        self.0 & ((1 << Self::UPPER_BITS) - 1)
    }

    #[inline]
    fn lower(self) -> u32 {
        (self.0 >> Self::UPPER_BITS) & ((1 << Self::LOWER_BITS) - 1)
    }

    #[inline]
    fn reserve(self) -> u32 {
        ((self.0 >> Self::REMAINING_SHIFT) & ((1 << Self::REMAINING_BITS) - 1)) * MIN_PROB
    }

    #[inline]
    fn width(self) -> usize {
        let tail = self.0 as i32 >> Self::TAIL_SHIFT;
        (((self.upper() - self.lower()) << PROB_SHIFT) as i32 + tail) as usize
    }

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
        Self::new(fl, u32::from(icdf[s]), (nsyms - 1 - s) as u32)
    }

    /// `bit` of a two-symbol alphabet whose inverse CDF starts at `f`
    /// ([`Self::symbol`] of `[f, 0]`).
    #[inline]
    fn binary(bit: bool, f: u16) -> Self {
        if bit {
            Self::new(u32::from(f), 0, 0)
        } else {
            Self::new(PROB_TOP, u32::from(f), 1)
        }
    }

    /// The only symbol of a one-symbol alphabet: the whole range, so coding
    /// it changes nothing (an omitted literal).
    const NOTHING: Self = Self::new(PROB_TOP, 0, 0);

    /// An equiprobable bit, `L(1)`.
    #[inline]
    fn literal(bit: bool) -> Self {
        if bit {
            Self::new(HALF, 0, 0)
        } else {
            Self::new(PROB_TOP, HALF, 1)
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
/// resolution. Probability `f` has already dropped its low bits.
#[inline]
fn scale(r_hi: u32, f: u32) -> u32 {
    (r_hi * f) >> (7 - PROB_SHIFT)
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
        let reserve = iv.reserve();
        let fl = iv.upper();
        let (low, upper) = if fl < PROB_TOP >> PROB_SHIFT {
            let u = scale(r_hi, fl) + reserve + MIN_PROB;
            (self.low + u64::from(r - u), u)
        } else {
            (self.low, r)
        };
        self.normalize(low, upper - (scale(r_hi, iv.lower()) + reserve));
    }

    /// Code `bit` of a two-symbol alphabet whose inverse CDF starts at `f`
    /// (libaom `od_ec_encode_bool_q15`; equal to [`Self::encode`] of its
    /// interval).
    #[inline]
    fn encode_bool(&mut self, bit: bool, f: u32) {
        let r = self.rng;
        let v = scale(r >> 8, f >> PROB_SHIFT) + MIN_PROB;
        let (low, rng) = if bit {
            (self.low + u64::from(r - v), v)
        } else {
            (self.low, r - v)
        };
        self.normalize(low, rng);
    }

    /// Code a run of L(1) from an arbitrary range. Once the low byte of
    /// the range is 8, the two subranges are equal and normalization
    /// restores the same range. The first 1 bit always gets us there.
    #[inline(never)]
    fn encode_literals(
        &mut self,
        value: u32,
        mut bits: u32,
        mut low: u64,
        mut range: u32,
        mut count: i32,
    ) -> (u64, u32, i32) {
        while bits != 0 {
            if range & 255 == 8 {
                let take = bits.min((FLUSH_BITS - count - 1) as u32);
                bits -= take;
                let prefix = (value >> bits) & ((1 << take) - 1);
                low = (low << take) + u64::from(range) * u64::from(prefix);
                count += take as i32;
                if bits == 0 {
                    break;
                }
            }

            bits -= 1;
            let half = ((range >> 8) << 7) + MIN_PROB;
            let lower = range - half;
            let next = if value >> bits & 1 == 0 {
                lower
            } else {
                low += u64::from(lower);
                half
            };
            let shift = next.leading_zeros() as i32 - 16;
            if count + shift >= FLUSH_BITS {
                self.cnt = count;
                self.flush(low, next, shift);
                (low, range, count) = (self.low, self.rng, self.cnt);
            } else {
                low <<= shift;
                range = next << shift;
                count += shift;
            }
        }
        (low, range, count)
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
            nc_common::propagate_carry(&mut self.out);
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
                nc_common::propagate_carry(&mut self.out);
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
    /// Whether and how coded symbols adapt their CDFs.
    adaptation: Adaptation,
    /// The node trials' record of the CDF context's adaptations
    /// (`backend/node_trial.rs`; inactive outside trials).
    journal: CdfJournal,
    /// [`Self::coded_bits`] of a deferred frame: logged symbols priced so
    /// far, and their bits.
    priced: (usize, f64),
}

/// What coding a symbol does to its CDF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Adaptation {
    /// Nothing (`disable_cdf_update = 1`, AV1 §5.9.2).
    Fixed,
    /// It adapts.
    Adapt,
    /// It adapts, its state before recorded in the journal (inside node
    /// trials): a state of its own, so coding outside trials tests no more
    /// than whether to adapt.
    Journaled,
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
            priced: (0, 0.),
            recording: Recording::default(),
            deferred: false,
            adaptation: Adaptation::Adapt,
            journal: CdfJournal::default(),
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
        self.priced = (0, 0.);
        if deferred && self.recording.intervals.capacity() == 0 {
            self.recording.intervals.reserve(INITIAL_CAPACITY);
        }
        self.deferred = deferred;
        self.adaptation = if adapt {
            Adaptation::Adapt
        } else {
            Adaptation::Fixed
        };
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
        self.intervals()[slot.0] = Interval::NOTHING;
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
            adapt: self.adaptation != Adaptation::Fixed,
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
            let interval = if N == 2 {
                Interval::binary(s != 0, cdf.icdf[0])
            } else {
                Interval::symbol(s, &cdf.icdf, nsyms)
            };
            self.recording.intervals.push(interval);
        } else if N == 2 {
            debug_assert_eq!(nsyms, 2);
            self.coder.encode_bool(s != 0, u32::from(cdf.icdf[0]));
        } else {
            self.coder.encode(Interval::symbol(s, &cdf.icdf, nsyms));
        }
        if self.adaptation != Adaptation::Fixed {
            if self.adaptation == Adaptation::Journaled {
                self.record(cdf);
            }
            cdf.adapt(s, nsyms);
        }
    }

    /// Journal `cdf`'s state before it adapts. Out of line: inlined into
    /// every symbol write, it made `Cdf::adapt` calls out of line in the
    /// coefficient writer, where no trial journals (AV1 Default, callgrind:
    /// +0.16 % instructions inlined, +0.06 % out of line).
    #[cold]
    #[inline(never)]
    fn record<const N: usize>(&mut self, cdf: &Cdf<N>) {
        self.journal.record(cdf, || cdf.state());
    }

    /// Record the adaptations of `cdfs`' CDFs from now on (the first node
    /// trial of a superblock opens).
    pub fn start_journal(&mut self, cdfs: &CdfContext) {
        self.journal.start(cdfs);
        if self.adaptation == Adaptation::Adapt {
            self.adaptation = Adaptation::Journaled;
        }
    }

    /// Stop recording and forget the journal (the outermost trial closed).
    pub fn stop_journal(&mut self) {
        self.journal.stop();
        if self.adaptation == Adaptation::Journaled {
            self.adaptation = Adaptation::Adapt;
        }
    }

    /// The journal of the CDF context's adaptations, which node trials
    /// mark, undo and redo.
    pub fn journal_mut(&mut self) -> &mut CdfJournal {
        &mut self.journal
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

    /// Log the symbols from here on instead of coding them at once (a
    /// frame's first node trial): the log is coded at
    /// [`Self::finish_frame`] from the coder's state at this point, so the
    /// bytes are those of coding at once. Spans and literal slots are only
    /// reserved in frames deferred from their start.
    pub fn start_logging(&mut self) {
        self.deferred = true;
    }

    /// Where the next symbol of a deferred frame is logged: a mark to
    /// price ([`Self::bits_since`]), note ([`Self::save_after`]) or
    /// replace ([`Self::restore_after`]) what is coded after it (the RD partition
    /// search's node trials, `backend/node_trial.rs`). Spans are reserved
    /// only at superblock starts, so a mark inside a superblock has none
    /// after it.
    #[must_use]
    pub fn mark(&self) -> usize {
        assert!(self.deferred, "writer must be deferred");
        self.recording.intervals.len()
    }

    /// Note in `tail` the symbols logged after `mark`.
    pub fn save_after(&self, mark: usize, tail: &mut SymbolTail) {
        tail.intervals.clear();
        tail.intervals
            .extend_from_slice(&self.recording.intervals[mark..]);
    }

    /// Forget the symbols logged after `mark` and log `tail`'s instead (a
    /// kept coding put back; literal slots reserved in it keep their
    /// places, as the tail was taken at the same mark).
    pub fn restore_after(&mut self, mark: usize, tail: &SymbolTail) {
        let intervals = self.intervals();
        intervals.truncate(mark);
        intervals.extend_from_slice(&tail.intervals);
    }

    /// The bits the symbols logged after `mark` cost: `-log2` of each
    /// interval's probability (the coder's own range scaling and its
    /// `EC_MIN_PROB` floor left out).
    #[must_use]
    pub fn bits_since(&self, mark: usize) -> f32 {
        // Keep the original f32 costs and summation order. Different trials
        // can price the same intervals, so compute each possible cost once.
        static COSTS: OnceLock<Box<[f32]>> = OnceLock::new();
        let costs = COSTS.get_or_init(|| {
            let top = (PROB_TOP as f32).log2();
            (0..=PROB_TOP)
                .map(|width| top - (width.max(1) as f32).log2())
                .collect()
        });
        let mut bits = 0.;
        for &iv in &self.recording.intervals[mark..] {
            if iv.is_literal_run() {
                // Keep rounding at every symbol, as before packing the run.
                for _ in 0..iv.literal_count() {
                    bits += 1.;
                }
            } else {
                bits += costs[iv.width()];
            }
        }
        bits
    }

    /// Bits the tile holds so far: the bytes the coder finished and the bits
    /// pending in it, plus any logged symbols priced ([`Self::bits_since`]):
    /// a deferred frame logs all of them, a node trial those after it began.
    /// Each logged symbol is priced once (in-frame rate control reads this
    /// after every superblock) unless the log was cut back past the last
    /// reading (a superblock undone), when it is priced afresh. Spans filled
    /// in later are not in it.
    pub fn coded_bits(&mut self) -> f64 {
        let pending = (self.coder.cnt - CNT_START).max(0);
        let finished = (self.coder.out.len() * 8) as f64 + f64::from(pending);
        let logged = self.recording.intervals.len();
        let (from, bits) = if self.priced.0 <= logged {
            self.priced
        } else {
            (0, 0.)
        };
        let bits = bits + f64::from(self.bits_since(from));
        self.priced = (logged, bits);
        finished + bits
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
    /// Code a run of logged symbols: [`Self::encode`] with `low`, `rng` and
    /// `cnt` held in locals, written back only around a flush, and the
    /// symbol's lower edge chosen without a branch (an upper edge at the
    /// top leaves `low` as it was): 35.3 instructions a symbol against 38.5
    /// for a marker dispatch to `encode` / `encode_bool` (callgrind, AV1
    /// Default, 3.56 M symbols). Out of line: called once per run, and kept one function so
    /// that profiles show the replay apart (the writer-path figures in
    /// `docs/analysis/zen5-fixed-qp-goal.md` count it).
    #[inline(never)]
    fn encode_run(&mut self, intervals: &[Interval]) {
        let (mut low, mut rng, mut cnt) = (self.low, self.rng, self.cnt);
        for iv in intervals {
            if iv.is_literal_run() {
                (low, rng, cnt) =
                    self.encode_literals(iv.literal_bits(), iv.literal_count(), low, rng, cnt);
                continue;
            }
            let r_hi = rng >> 8;
            let reserve = iv.reserve();
            let fl = iv.upper();
            let upper = if fl < PROB_TOP >> PROB_SHIFT {
                scale(r_hi, fl) + reserve + MIN_PROB
            } else {
                rng
            };
            low += u64::from(rng - upper);
            let next = upper - (scale(r_hi, iv.lower()) + reserve);
            debug_assert!((1..=0xffff).contains(&next));
            let d = next.leading_zeros() as i32 - 16;
            if cnt + d >= FLUSH_BITS {
                self.cnt = cnt;
                self.flush(low, next, d);
                (low, rng, cnt) = (self.low, self.rng, self.cnt);
            } else {
                low <<= d;
                rng = next << d;
                cnt += d;
            }
        }
        (self.low, self.rng, self.cnt) = (low, rng, cnt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_interval_preserves_the_range_edges_and_full_width() {
        for base in [0, 64, 4032, 32_704] {
            for fl in base..=base + 64 {
                for fh in base..=fl.min(PROB_TOP - 1) {
                    let interval = Interval::new(fl, fh, 15);
                    assert_eq!(interval.upper(), fl >> PROB_SHIFT);
                    assert_eq!(interval.lower(), fh >> PROB_SHIFT);
                    assert_eq!(interval.reserve(), 15 * MIN_PROB);
                    assert_eq!(interval.width(), (fl - fh) as usize);
                }
            }
        }
        assert_eq!(Interval::NOTHING.width(), PROB_TOP as usize);
    }

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
    fn literal_runs_preserve_bytes_and_pricing() {
        let mut immediate = SymbolWriter::new();
        let mut run_now = SymbolWriter::new();
        let mut loose = SymbolWriter::new_deferred();
        let mut packed = SymbolWriter::new_deferred();
        let fixed = [24321, 16769, 8837, 0];
        let mut state = 7u32;
        for i in 0..100_000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let len = 1 + (state >> 24) % 32;
            let value = match i % 4 {
                0 => 0,
                1 => 1,
                2 => u32::MAX,
                _ => state,
            };
            let symbol = (state >> 16) as usize & 3;
            for writer in [&mut immediate, &mut run_now, &mut loose, &mut packed] {
                writer.write_fixed(symbol, &fixed, 4);
            }
            for writer in [&mut immediate, &mut loose] {
                for j in (0..len).rev() {
                    writer.write_literal_bit(value >> j & 1 != 0);
                }
            }
            run_now.write_literal(value, len);
            packed.write_literal(value, len);
            if i % 30 == 0 {
                assert_eq!(loose.coded_bits().to_bits(), packed.coded_bits().to_bits());
            }
        }
        assert_eq!(
            loose.bits_since(0).to_bits(),
            packed.bits_since(0).to_bits()
        );
        let expected = immediate.finish();
        assert_eq!(run_now.finish(), expected);
        assert_eq!(loose.finish(), expected);
        assert_eq!(packed.finish(), expected);
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
