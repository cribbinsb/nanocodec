//! AV2 multi-symbol range coding and bypass bits, with bounded native decoding.
//!
//! AVM v1.0.0 `avm_dsp/{entenc,entdec,entcode,prob}` is the reference (Clear
//! BSD + separate AOM patent terms). AV2 differs from AV1 in probability scaling,
//! true bypass coding and trained CDF adaptation rates. No AV1 coder is reused.
//!
//! A writer can also record its committed coding operations and replay them
//! once the tile is coded, filling in symbols that were deferred because
//! their value is only chosen after reconstruction (the CDEF strength index,
//! [`crate::cdef`]). Recorded symbols keep the probabilities they were coded
//! with, so the replay codes exactly the tile the live writer measured, plus
//! the deferred symbols, without coding any block twice.
//!
//! Rate-distortion trials price candidates without coding them: between
//! [`Writer::start_estimate`] and [`Writer::finish_estimate`] every symbol
//! adds its table rate ([`nc_encoder::symbol_cost`]) for the interval the
//! coder would narrow to, and adapts a scratch copy of the models as coding
//! would; nothing is coded and the tile's models and record are untouched.
//! Only the chosen candidate is then coded. Unlike AV1, whose CDFs carry
//! over between frames, AV2 tiles start from the defaults and adapt fast,
//! so a non-adapting estimate over-priced residual candidates by 2-4% in
//! static P frames; the scratch copy removes that bias. Scratch models
//! are copied on first use in each estimate, not as a whole set.

use crate::cdf_models::TileCdfs;
use nc_common::decode::{DecodeError, Result};
use nc_encoder::symbol_cost::{BitCounter, probability_cost_q9};

const TOP: u32 = 32768;

fn validate_cdf(cdf: &[u16]) -> Result<()> {
    if !(2..=16).contains(&cdf.len())
        || cdf.last() != Some(&0)
        || u32::from(cdf[0]) >= TOP
        || cdf.windows(2).any(|p| p[0] < p[1])
    {
        return Err(DecodeError::Malformed("AV2 invalid inverse CDF"));
    }
    Ok(())
}

/// Low bits of a 15-bit CDF edge that the coder drops.
const EDGE_DROPPED_BITS: u32 = 7;
/// Bits of a coded edge: the kept 8 bits scaled up to 12.
const EDGE_BITS: u32 = 12;
/// The minimum-probability increments are in 1/16 of an edge step.
const INCREMENT_STEPS: usize = 16;

/// The 12-bit edge the coder uses for inverse CDF value `probability` at
/// `symbol` of `count`: its top 8 bits plus AVM's minimum-probability
/// increment (`av2_prob_inc_tbl[n-2][s]`, `floor(16*(n-1-s)/n)`).
fn coded_edge(probability: u16, symbol: usize, count: usize) -> u32 {
    let increment = (INCREMENT_STEPS * (count - 1 - symbol) / count) as u32;
    ((u32::from(probability) >> EDGE_DROPPED_BITS) << (EDGE_BITS - 8)) + increment
}

fn scaled(probability: u16, range: u32, symbol: usize, count: usize) -> u32 {
    (((range >> 8) * coded_edge(probability, symbol, count)) >> 7) << 3
}

/// Rate of `symbol` under inverse CDF `cdf` in 1/512 bit, from the interval
/// the coder actually narrows to ([`coded_edge`]) rather than the raw CDF:
/// the edge rounding and minimum increments make near-certain symbols cost
/// several times the raw estimate (skip flags in static content) and cap
/// rare ones at about 9 bits. Inlined, so that the alphabet size of a
/// `Cdf<N>` is a constant in its divisions.
#[inline]
fn coded_cost_q9(cdf: &[u16], symbol: usize) -> u32 {
    let count = cdf.len();
    let upper = if symbol == 0 {
        1 << EDGE_BITS
    } else {
        coded_edge(cdf[symbol - 1], symbol - 1, count)
    };
    let width = upper - coded_edge(cdf[symbol], symbol, count);
    probability_cost_q9(width << (15 - EDGE_BITS))
}

/// Rate of `symbol` under the fixed (non-adapting) inverse CDF `cdf`, in
/// 1/512 bit, as an estimate prices it.
pub(crate) fn fixed_cost_q9(cdf: &[u16], symbol: usize) -> u32 {
    coded_cost_q9(cdf, symbol)
}

/// An inherited model keeps 3/4 of its symbol count
/// (`reset_cdf_symbol_counter`: `(count * 3) >> 2`).
const INHERITED_COUNT_NUMERATOR: u8 = 3;
const INHERITED_COUNT_SHIFT: u8 = 2;

/// An inverse CDF and its three trained adaptation-rate offsets, plus the
/// rate estimate a scratch copy of it belongs to. Equality ignores the
/// estimate.
#[derive(Debug, Clone, Copy)]
pub struct Cdf<const N: usize> {
    inverse: [u16; N],
    count: u8,
    rate_offsets: [u8; 3],
    /// In a [`Writer`]'s scratch models: the estimate that last copied this
    /// model from the live one (copy on first use). Always 0 in live models.
    estimate: u32,
}

impl<const N: usize> PartialEq for Cdf<N> {
    fn eq(&self, other: &Self) -> bool {
        (self.inverse, self.count, self.rate_offsets)
            == (other.inverse, other.count, other.rate_offsets)
    }
}

impl<const N: usize> Eq for Cdf<N> {}

impl<const N: usize> Cdf<N> {
    /// Construct a 2..16-symbol CDF with AVM's stored rate offsets (not raw PARA).
    /// # Errors
    /// Rejects malformed probabilities or rates that cannot update 15-bit CDFs.
    pub fn new(inverse: [u16; N], rate_offsets: [u8; 3]) -> Result<Self> {
        validate_cdf(&inverse)?;
        if rate_offsets.iter().any(|&r| r > 13) {
            return Err(DecodeError::Malformed("AV2 CDF adaptation rate"));
        }
        Ok(Self {
            inverse,
            count: 0,
            rate_offsets,
            estimate: 0,
        })
    }

    /// Rate of `symbol` under this model as it stands, in 1/512 bit.
    #[inline]
    pub(crate) fn cost_q9(&self, symbol: usize) -> u32 {
        coded_cost_q9(&self.inverse, symbol)
    }

    /// Move the model back along its adaptation-rate schedule, keeping its
    /// probabilities: AVM `reset_cdf_symbol_counter` scales the symbol
    /// count by 3/4, applied to a frame's final models before later frames
    /// inherit them.
    pub fn restart_adaptation(&mut self) {
        self.count = (self.count * INHERITED_COUNT_NUMERATOR) >> INHERITED_COUNT_SHIFT;
    }

    /// Current inverse probabilities, including the final zero boundary.
    #[must_use]
    pub const fn inverse(&self) -> &[u16; N] {
        &self.inverse
    }

    fn adapt(&mut self, symbol: usize) {
        let interval = usize::from(self.count > 15) + usize::from(self.count > 31);
        let rate = 2 + self.rate_offsets[interval];
        for (i, value) in self.inverse[..N - 1].iter_mut().enumerate() {
            if i < symbol {
                *value += ((TOP - u32::from(*value)) >> rate) as u16;
            } else {
                *value -= *value >> rate;
            }
        }
        self.count = (self.count + 1).min(32);
    }
}

/// One committed coding operation of a recording [`Writer`].
#[derive(Debug, Clone, Copy)]
enum Operation {
    /// A symbol coded with the inverse CDF values at its two edges
    /// (`upper` is unused for symbol 0, whose upper edge is the range).
    Symbol {
        symbol: u8,
        count: u8,
        upper: u16,
        lower: u16,
    },
    /// Bypass bits.
    Literal { value: u32, bits: u8 },
    /// A symbol whose value is chosen after the tile is coded.
    Deferred,
}

/// Per-tile native AV2 entropy writer; no global probability state.
#[derive(Debug)]
pub struct Writer {
    models: Option<Box<TileCdfs>>,
    precarry: Vec<u16>,
    low: u64,
    range: u32,
    pending: i32,
    /// Committed operations, when recording for [`Writer::replay`].
    log: Option<Vec<Operation>>,
    /// Rate of everything priced since [`Writer::start_estimate`], while
    /// estimating instead of coding.
    estimate: Option<BitCounter>,
    /// The models an estimate prices and adapts. Each model is copied from
    /// `models` when an estimate first uses it (its `Cdf::estimate` is not
    /// `estimate_id`), so an estimate copies only the few models a block
    /// touches rather than the whole set.
    scratch: Option<Box<TileCdfs>>,
    /// Price estimates with the live models, frozen for the trial, instead
    /// of adapting a scratch copy ([`Self::freeze_estimates`]).
    frozen_estimates: bool,
    /// Number of the current (or last) estimate; scratch models carrying
    /// another number are stale.
    estimate_id: u32,
}

impl Default for Writer {
    fn default() -> Self {
        Self::new()
    }
}

impl Writer {
    /// Start a new independent entropy tile.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            models: None,
            precarry: Vec::new(),
            low: 0,
            range: TOP,
            pending: -9,
            log: None,
            estimate: None,
            scratch: None,
            estimate_id: 0,
            frozen_estimates: false,
        }
    }

    /// Price rate estimates with the tile's live models as they stand,
    /// without adapting them within a trial (`av2.estimate_adaptation` 0):
    /// cheaper, and close because the live models already carry every
    /// coded block's adaptation.
    pub(crate) fn freeze_estimates(&mut self) {
        self.frozen_estimates = true;
    }

    /// Price instead of code until [`Self::finish_estimate`]: each symbol
    /// and bypass bit adds its rate, and adaptive models adapt in a scratch
    /// copy as they would when coded, so the estimate follows the coded
    /// size within a candidate. The tile's own models, its output and its
    /// record are untouched, and deferred symbols are skipped (they are
    /// priced when they are filled in).
    pub(crate) fn start_estimate(&mut self) {
        debug_assert!(self.estimate.is_none(), "rate estimates do not nest");
        self.estimate = Some(BitCounter::default());
        // A tile prices far fewer than 2^32 candidates.
        self.estimate_id = self
            .estimate_id
            .checked_add(1)
            .expect("estimate numbers do not wrap");
        if let (None, Some(models)) = (&self.scratch, self.models.as_deref()) {
            // Live models carry estimate 0, so every copied model is stale.
            self.scratch = Some(Box::new(models.clone()));
        }
    }

    /// Whether a frozen estimate runs ([`Self::freeze_estimates`]): a
    /// caller may then price a whole syntax element from
    /// [`Self::live_models`] directly and add the total with
    /// [`Self::add_estimate_q9`].
    pub(crate) fn estimates_frozen_now(&self) -> bool {
        self.estimate.is_some() && self.frozen_estimates
    }

    /// The tile's live adaptive models, or `None` when it does not adapt.
    pub(crate) fn live_models(&mut self) -> Option<&mut TileCdfs> {
        self.models.as_deref_mut()
    }

    /// The running estimate so far, in 1/512 bit.
    pub(crate) fn estimate_q9(&self) -> u32 {
        self.estimate.as_ref().expect("an estimate is running").q9()
    }

    /// Add `q9` (1/512 bit) priced under the live probabilities to the
    /// running estimate.
    pub(crate) fn add_estimate_q9(&mut self, q9: u32) {
        self.estimate
            .as_mut()
            .expect("an estimate is running")
            .add_q9(q9);
    }

    /// Stop estimating and return the bits priced since
    /// [`Self::start_estimate`].
    pub(crate) fn finish_estimate(&mut self) -> f64 {
        self.estimate
            .take()
            .map_or(0.0, |counter| f64::from(counter.bits()))
    }

    /// Record every committed operation so that [`Self::replay`] can code
    /// the tile again with deferred symbols filled in; the recording writer
    /// itself only adapts its models and logs. Call before coding any
    /// symbols.
    pub(crate) fn record(&mut self) {
        self.log = Some(Vec::new());
    }

    /// Reserve a symbol whose value is chosen after the tile is coded: the
    /// live coder codes nothing here and [`Self::replay`] asks for it.
    /// Its probability models must not be used by any other syntax, so they
    /// are still untouched when the replay codes it.
    /// Returns whether it reserved one (estimates reserve nothing).
    pub(crate) fn defer(&mut self) -> bool {
        if self.estimate.is_some() {
            return false;
        }
        self.log
            .as_mut()
            .expect("deferred symbols need a recording writer")
            .push(Operation::Deferred);
        true
    }

    /// Enable normative within-tile adaptation from fresh frame defaults.
    /// Call before coding any symbols, once for each tile.
    pub fn enable_adaptation(&mut self, qindex: u8) {
        self.models = Some(Box::new(TileCdfs::new(qindex)));
    }

    /// Adapt within the tile starting from `models` (a reference frame's
    /// final models: cross-frame context) instead of the defaults.
    pub fn adapt_from(&mut self, models: TileCdfs) {
        self.models = Some(Box::new(models));
    }

    /// Code a syntax model, retaining fixed probabilities when adaptation is off.
    /// # Errors
    /// Rejects invalid CDFs or symbols outside the alphabet.
    pub fn model<const N: usize>(
        &mut self,
        symbol: usize,
        fixed: &[u16],
        select: impl Fn(&mut TileCdfs) -> &mut Cdf<N>,
    ) -> Result<()> {
        let Some(models) = self.models.as_deref_mut() else {
            return self.fixed(symbol, fixed);
        };
        if let Some(counter) = self.estimate.as_mut() {
            if self.frozen_estimates {
                // Price with the live model as it stands; no scratch copy
                // and no adaptation inside the trial.
                if symbol >= N {
                    return Err(DecodeError::Malformed("AV2 symbol outside CDF"));
                }
                counter.add_q9(coded_cost_q9(&select(models).inverse, symbol));
                return Ok(());
            }
            let scratch = self
                .scratch
                .as_deref_mut()
                .expect("an adaptive estimate has its scratch models");
            let cdf = select(scratch);
            if cdf.estimate != self.estimate_id {
                *cdf = *select(models);
                cdf.estimate = self.estimate_id;
            }
            // Adaptive CDFs are valid by construction (`Cdf::new`).
            if symbol >= N {
                return Err(DecodeError::Malformed("AV2 symbol outside CDF"));
            }
            counter.add_q9(coded_cost_q9(&cdf.inverse, symbol));
            cdf.adapt(symbol);
            return Ok(());
        }
        let Some(mut models) = self.models.take() else {
            return self.fixed(symbol, fixed);
        };
        let result = self.symbol(symbol, select(&mut models));
        self.models = Some(models);
        result
    }

    /// The inverse CDF the next [`Self::model`] call with `fixed` and
    /// `select` would code with outside an estimate: the live adaptive model,
    /// or `fixed` when the tile does not adapt.
    pub(crate) fn current<const N: usize>(
        &self,
        fixed: &[u16; N],
        select: impl Fn(&TileCdfs) -> &Cdf<N>,
    ) -> [u16; N] {
        self.models
            .as_deref()
            .map_or(*fixed, |models| *select(models).inverse())
    }

    /// Rate in 1/512 bit of coding `symbol` with the model the next
    /// [`Self::model`] call with `fixed` and `select` would use outside an
    /// estimate (the live adaptive model, or `fixed`), without coding or
    /// adapting anything: the trellis prices many candidate levels against
    /// the tile's probabilities as they stand.
    pub(crate) fn price_q9<const N: usize>(
        &mut self,
        symbol: usize,
        fixed: &[u16],
        select: impl Fn(&mut TileCdfs) -> &mut Cdf<N>,
    ) -> u32 {
        match self.models.as_deref_mut() {
            Some(models) => coded_cost_q9(&select(models).inverse, symbol),
            None => coded_cost_q9(fixed, symbol),
        }
    }

    /// Write a symbol without updating the CDF (`disable_cdf_update = 1`).
    /// # Errors
    /// Rejects an invalid CDF or a symbol outside its alphabet.
    #[inline]
    pub fn fixed(&mut self, symbol: usize, cdf: &[u16]) -> Result<()> {
        validate_cdf(cdf)?;
        self.code_valid(symbol, cdf)
    }

    /// [`Self::fixed`] for a CDF known to be valid: an adaptive [`Cdf`]
    /// is validated when built and adaptation keeps it valid, so its
    /// symbols skip the per-symbol check of every entry.
    #[inline]
    fn code_valid(&mut self, symbol: usize, cdf: &[u16]) -> Result<()> {
        debug_assert!(validate_cdf(cdf).is_ok());
        if symbol >= cdf.len() {
            return Err(DecodeError::Malformed("AV2 symbol outside CDF"));
        }
        if let Some(counter) = self.estimate.as_mut() {
            counter.add_q9(coded_cost_q9(cdf, symbol));
            return Ok(());
        }
        let upper = if symbol == 0 { 0 } else { cdf[symbol - 1] };
        if self.log.is_some() {
            // A recording writer's own interval is never output: the replay
            // codes the tile again from the log (coding it live as well was
            // ~1.2% of the encoder's Jetson cycles).
            // Alphabets have at most 16 symbols (`validate_cdf`).
            self.log_operation(Operation::Symbol {
                symbol: symbol as u8,
                count: cdf.len() as u8,
                upper,
                lower: cdf[symbol],
            });
        } else {
            self.code_interval(symbol, cdf.len(), upper, cdf[symbol]);
        }
        Ok(())
    }

    /// Append to the recording. Kept out of line so that symbol coding,
    /// inlined at every call site, stays small when not recording.
    #[inline(never)]
    fn log_operation(&mut self, operation: Operation) {
        if let Some(log) = self.log.as_mut() {
            log.push(operation);
        }
    }

    /// Narrow the interval to `symbol` of `count`, whose inverse CDF edges
    /// are `upper` (ignored for symbol 0) and `lower`.
    #[inline]
    fn code_interval(&mut self, symbol: usize, count: usize, upper: u16, lower: u16) {
        let upper = if symbol == 0 {
            self.range
        } else {
            scaled(upper, self.range, symbol - 1, count)
        };
        let lower = scaled(lower, self.range, symbol, count);
        self.low += u64::from(self.range - upper);
        self.range = upper - lower;
        self.normalize(0);
    }

    /// Write a symbol and apply final AV2's trained-rate adaptation.
    /// # Errors
    /// Rejects symbols outside the alphabet.
    pub fn symbol<const N: usize>(&mut self, symbol: usize, cdf: &mut Cdf<N>) -> Result<()> {
        if let Some(counter) = self.estimate.as_mut() {
            // A caller's own model is priced but left as it is.
            if symbol >= N {
                return Err(DecodeError::Malformed("AV2 symbol outside CDF"));
            }
            counter.add_q9(coded_cost_q9(&cdf.inverse, symbol));
            return Ok(());
        }
        self.code_valid(symbol, &cdf.inverse)?;
        cdf.adapt(symbol);
        Ok(())
    }

    /// Write true bypass bits, most significant first, without changing range.
    /// # Errors
    /// Rejects widths above 32 and values that exceed their declared width.
    pub fn literal(&mut self, value: u32, bits: u32) -> Result<()> {
        if bits > 32 || (bits < 32 && value >= (1 << bits)) {
            return Err(DecodeError::Malformed("AV2 bypass literal exceeds width"));
        }
        if let Some(counter) = self.estimate.as_mut() {
            counter.add_bypass(bits);
            return Ok(());
        }
        if self.log.is_some() {
            // Coded by the replay only (as symbols).
            self.log_operation(Operation::Literal {
                value,
                bits: bits as u8,
            });
            return Ok(());
        }
        // Range is constant across bypass bits. Compose up to 16 updates before
        // normalizing; the u64 accumulator and two-byte carry flush bound this.
        let mut remaining = bits;
        while remaining != 0 {
            let count = remaining.min(16);
            let chunk = (value >> (remaining - count)) & ((1 << count) - 1);
            self.low = (self.low << count) + u64::from(self.range) * u64::from(chunk);
            self.normalize(count as i32);
            remaining -= count;
        }
        Ok(())
    }

    fn normalize(&mut self, bypass: i32) {
        let shift = if bypass != 0 {
            0
        } else {
            self.range.leading_zeros() as i32 - 16
        };
        let mut pending = self.pending + bypass;
        let mut ready = pending + shift;
        if ready >= 0 {
            pending += 16;
            let mut mask = (1_u64 << pending) - 1;
            if ready >= 8 {
                self.precarry.push((self.low >> pending) as u16);
                self.low &= mask;
                pending -= 8;
                mask >>= 8;
            }
            self.precarry.push((self.low >> pending) as u16);
            self.low &= mask;
            ready = pending + shift - 24;
        }
        self.low <<= shift;
        self.range <<= shift;
        self.pending = ready;
    }

    /// Finish a recording writer's tile: code its operations again into a
    /// fresh writer, calling `fill` at each deferred symbol (in coding
    /// order) to code it there. `fill` sees this writer's final models, in
    /// which the deferred symbols' models are still at their initial
    /// values. A writer that did not record just finishes. Returns the
    /// tile's bytes and its final models (`None` when it did not adapt).
    /// # Errors
    /// Propagates errors from `fill`.
    pub(crate) fn replay(
        mut self,
        mut fill: impl FnMut(&mut Self) -> Result<()>,
    ) -> Result<(Vec<u8>, Option<Box<TileCdfs>>)> {
        let Some(log) = self.log.take() else {
            return Ok(self.finish_with_models());
        };
        let mut replay = Self {
            models: self.models.take(),
            ..Self::new()
        };
        for operation in log {
            match operation {
                Operation::Symbol {
                    symbol,
                    count,
                    upper,
                    lower,
                } => replay.code_interval(usize::from(symbol), usize::from(count), upper, lower),
                Operation::Literal { value, bits } => replay.literal(value, u32::from(bits))?,
                Operation::Deferred => fill(&mut replay)?,
            }
        }
        Ok(replay.finish_with_models())
    }

    /// [`Self::finish`], also returning the tile's final models (`None`
    /// when it did not adapt).
    #[must_use]
    pub fn finish_with_models(mut self) -> (Vec<u8>, Option<Box<TileCdfs>>) {
        let models = self.models.take();
        (self.finish(), models)
    }

    /// Finish the tile with AV2's minimal interval termination and carry handling.
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        debug_assert!(
            self.log.is_none(),
            "a recording writer codes nothing live: finish it by replay"
        );
        let mut remaining = self.pending + 10;
        let mut shift = self.pending + 16;
        let mut end = ((self.low + 0x3fff) & !0x3fff_u64) | 0x4000;
        while remaining > 0 {
            self.precarry.push((end >> shift) as u16);
            end &= (1_u64 << shift) - 1;
            remaining -= 8;
            shift -= 8;
        }
        let mut bytes = vec![0; self.precarry.len()];
        let mut carry = 0_u32;
        for (byte, pending) in bytes.iter_mut().zip(self.precarry).rev() {
            carry += u32::from(pending);
            *byte = carry as u8;
            carry >>= 8;
        }
        bytes
    }
}

/// Bits of the reader's window.
const WINDOW_BITS: i32 = u64::BITS as i32;
/// Bits of the arithmetic value at the top of the window: 16 for the
/// value below the range, one more for a bypass bit shifted in before it
/// is compared with the range.
const VALUE_BITS: i32 = 17;
/// Where the value starts in the window.
const VALUE_SHIFT: i32 = WINDOW_BITS - VALUE_BITS;
/// Bits `Reader::new` reads into the value (AVM `od_ec_dec_init`).
const INIT_BITS: i32 = 15;
/// Implicit zero bits that may follow a tile's final byte.
const PADDING_BITS: usize = 15;

/// Bounded AV2 entropy reader using a direct 15-bit arithmetic interval.
///
/// The value sits at the top of a 64-bit window whose lower bits hold the
/// next stream bits, so renormalisation is one shift and a refill of whole
/// bytes every few symbols, not a loop over single bits (the design of the
/// AV1 reader, after dav1d's `msac`). Reading the 16th implicit zero bit
/// past the data is an error, exactly as a bit-at-a-time reader finds it.
#[derive(Debug)]
pub struct Reader<'a> {
    models: Option<Box<TileCdfs>>,
    /// The tile's data not yet in the window.
    data: &'a [u8],
    /// The value in the top [`VALUE_BITS`], then the next stream bits.
    window: u64,
    /// Stream bits in the window below the value; negative when a shift
    /// took more than were there.
    bits: i32,
    /// Stream bits shifted into the value so far, and the most that may be.
    consumed: usize,
    limit: usize,
    range: u32,
}

impl<'a> Reader<'a> {
    /// Stream bits shifted into the value so far (analysis: differences
    /// give what a block cost).
    #[cfg(feature = "analysis")]
    #[must_use]
    pub(crate) fn bits_consumed(&self) -> usize {
        self.consumed
    }

    /// Start one tile. At most 15 implicit zero bits may follow its final byte.
    /// # Errors
    /// Rejects empty input.
    pub fn new(data: &'a [u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(DecodeError::Truncated);
        }
        let mut reader = Self {
            models: None,
            data,
            window: 0,
            bits: -INIT_BITS,
            consumed: INIT_BITS as usize,
            limit: data.len() * 8 + PADDING_BITS,
            range: TOP,
        };
        reader.refill();
        Ok(reader)
    }

    /// Enable normative within-tile adaptation before reading syntax symbols.
    pub fn enable_adaptation(&mut self, qindex: u8) {
        self.models = Some(Box::new(TileCdfs::new(qindex)));
    }

    /// Adapt within the tile starting from `models` (cross-frame context).
    pub fn adapt_from(&mut self, models: TileCdfs) {
        self.models = Some(Box::new(models));
    }

    /// The tile's models after its last symbol; `None` when it did not
    /// adapt.
    #[must_use]
    pub fn into_models(self) -> Option<Box<TileCdfs>> {
        self.models
    }

    /// Decode a syntax model with the tile's signalled adaptation policy.
    /// # Errors
    /// Rejects malformed CDFs, truncated streams and invalid intervals.
    pub fn model<const N: usize>(
        &mut self,
        fixed: &[u16],
        select: impl FnOnce(&mut TileCdfs) -> &mut Cdf<N>,
    ) -> Result<usize> {
        let Some(mut models) = self.models.take() else {
            return self.fixed(fixed);
        };
        let result = self.symbol(select(&mut models));
        self.models = Some(models);
        result
    }

    /// Fill the window below its valid bits with whole bytes. Past the
    /// data the window's zero bits are the implicit padding, so it only
    /// counts as full.
    #[inline(never)]
    fn refill(&mut self) {
        let mut shift = VALUE_SHIFT - 8 - self.bits;
        while shift >= 0 {
            let Some((&byte, rest)) = self.data.split_first() else {
                shift = -8;
                break;
            };
            self.data = rest;
            self.window |= u64::from(byte) << shift;
            shift -= 8;
        }
        self.bits = VALUE_SHIFT - 8 - shift;
    }

    /// Shift `count` stream bits into the value.
    #[inline]
    fn shift_in(&mut self, count: u32) -> Result<()> {
        self.window <<= count;
        self.bits -= count as i32;
        self.consumed += count as usize;
        if self.consumed > self.limit {
            return Err(DecodeError::Truncated);
        }
        if self.bits < 0 {
            self.refill();
        }
        Ok(())
    }

    /// The arithmetic value.
    #[inline]
    fn value(&self) -> u32 {
        (self.window >> VALUE_SHIFT) as u32
    }

    /// Renormalise until the range is at least [`TOP`] again.
    #[inline]
    fn normalize(&mut self) -> Result<()> {
        let shift = self.range.leading_zeros() - TOP.leading_zeros();
        self.range <<= shift;
        self.shift_in(shift)
    }

    /// Read one symbol with fixed probabilities.
    /// # Errors
    /// Rejects malformed CDFs, invalid intervals and exhausted input.
    pub fn fixed(&mut self, cdf: &[u16]) -> Result<usize> {
        validate_cdf(cdf)?;
        self.read_valid(cdf)
    }

    /// [`Self::fixed`] for a CDF known to be valid (see `Writer::code_valid`).
    #[inline]
    fn read_valid(&mut self, cdf: &[u16]) -> Result<usize> {
        debug_assert!(validate_cdf(cdf).is_ok());
        let value = self.value();
        let mut lower = 0;
        for (symbol, &probability) in cdf.iter().enumerate() {
            let upper = self.range - scaled(probability, self.range, symbol, cdf.len());
            if value < upper {
                self.window -= u64::from(lower) << VALUE_SHIFT;
                self.range = upper - lower;
                self.normalize()?;
                return Ok(symbol);
            }
            lower = upper;
        }
        Err(DecodeError::Malformed("AV2 entropy symbol"))
    }

    /// Read a symbol and apply final AV2's trained-rate adaptation.
    /// # Errors
    /// Rejects invalid intervals and exhausted input.
    pub fn symbol<const N: usize>(&mut self, cdf: &mut Cdf<N>) -> Result<usize> {
        let symbol = self.read_valid(&cdf.inverse)?;
        cdf.adapt(symbol);
        Ok(symbol)
    }

    /// Read true bypass bits with no probability scaling or adaptation.
    /// # Errors
    /// Rejects widths above 32 and exhausted input.
    pub fn literal(&mut self, bits: u32) -> Result<u32> {
        if bits > 32 {
            return Err(DecodeError::Malformed("AV2 bypass width exceeds 32"));
        }
        let mut result = 0;
        for _ in 0..bits {
            self.shift_in(1)?;
            let bit = self.value() >= self.range;
            if bit {
                self.window -= u64::from(self.range) << VALUE_SHIFT;
            }
            result = (result << 1) | u32::from(bit);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_encoder::symbol_cost::COST_UNITS_PER_BIT;

    #[test]
    fn grouped_literals_match_single_bits_at_carry_boundaries() {
        for bits in 0..=32 {
            let maximum = if bits == 32 {
                u32::MAX
            } else {
                (1 << bits) - 1
            };
            for value in [0, maximum, maximum / 2] {
                for prefix in 0..16 {
                    let mut grouped = Writer::new();
                    let mut scalar = Writer::new();
                    grouped.literal(0, prefix).unwrap();
                    for _ in 0..prefix {
                        scalar.literal(0, 1).unwrap();
                    }
                    assert!(grouped.literal(0, 33).is_err());
                    if bits < 32 {
                        assert!(grouped.literal(maximum + 1, bits).is_err());
                    }
                    grouped.literal(value, bits).unwrap();
                    for bit in (0..bits).rev() {
                        scalar.literal((value >> bit) & 1, 1).unwrap();
                    }
                    assert_eq!(grouped.finish(), scalar.finish());
                }
            }
        }
    }

    #[test]
    fn estimates_leave_models_and_bytes_unchanged() {
        let fixed = [22_725, 21_668, 0];
        let mut baseline = Writer::new();
        let mut trial = Writer::new();
        baseline.enable_adaptation(120);
        trial.enable_adaptation(120);
        for index in 0..80 {
            let symbol = index % 3;
            baseline
                .model(symbol, &fixed, |m| &mut m.inter_mode[0])
                .unwrap();
            trial
                .model(symbol, &fixed, |m| &mut m.inter_mode[0])
                .unwrap();
            let models = trial.models.as_deref().cloned();
            trial.start_estimate();
            for step in 0..40 {
                trial
                    .model(step % 3, &fixed, |m| &mut m.inter_mode[0])
                    .unwrap();
                trial.literal(step as u32, 6).unwrap();
            }
            assert!(trial.finish_estimate() > 40.0 * 6.0);
            assert_eq!(trial.models.as_deref().cloned(), models);
        }
        assert_eq!(trial.models, baseline.models);
        assert_eq!(trial.finish(), baseline.finish());
    }

    #[test]
    fn estimates_copy_models_from_the_live_state() {
        let fixed = [22_725, 21_668, 0];
        let mut writer = Writer::new();
        writer.enable_adaptation(120);
        for index in 0..20 {
            let live = *writer.models.as_deref().unwrap().inter_mode[0].inverse();
            writer.start_estimate();
            for _ in 0..3 {
                writer
                    .model(index % 3, &fixed, |m| &mut m.inter_mode[0])
                    .unwrap();
            }
            let estimate = writer.finish_estimate();
            // The first symbol is priced from the live model; the scratch
            // copy then adapts, so repeats cost less than the first.
            let first = f64::from(coded_cost_q9(&live, index % 3)) / f64::from(COST_UNITS_PER_BIT);
            assert!(estimate > first && estimate < 3.0 * first);
            writer
                .model((index + 1) % 3, &fixed, |m| &mut m.inter_mode[0])
                .unwrap();
        }
    }

    #[test]
    fn estimates_price_from_cost_tables() {
        let fixed = [29000, 11000, 0];
        let mut writer = Writer::new();
        writer.enable_adaptation(120);
        for index in 0..64 {
            writer
                .model(index % 3, &fixed, |m| &mut m.inter_mode[0])
                .unwrap();
        }
        let adapted = *writer.models.as_deref().unwrap().inter_mode[0].inverse();
        writer.start_estimate();
        writer.literal(0x1234_5678, 32).unwrap();
        assert!((writer.finish_estimate() - 32.0).abs() < 1e-10);
        for symbol in 0..3 {
            writer.start_estimate();
            writer
                .model(symbol, &fixed, |m| &mut m.inter_mode[0])
                .unwrap();
            writer.fixed(symbol, &fixed).unwrap();
            let expected = coded_cost_q9(&adapted, symbol) + coded_cost_q9(&fixed, symbol);
            let expected = f64::from(expected) / f64::from(COST_UNITS_PER_BIT);
            assert!((writer.finish_estimate() - expected).abs() < 1e-6);
        }
        writer.start_estimate();
        assert!(writer.model(3, &fixed, |m| &mut m.inter_mode[0]).is_err());
        writer.finish_estimate();
    }

    #[test]
    fn replay_fills_deferred_symbols_and_skips_estimates() {
        let cdf = [29000, 11000, 0];
        let mut direct = Writer::new();
        direct.enable_adaptation(120);
        let mut recorded = Writer::new();
        recorded.enable_adaptation(120);
        recorded.record();
        for index in 0..300 {
            let symbol = index % 3;
            direct.fixed(symbol, &cdf).unwrap();
            recorded.fixed(symbol, &cdf).unwrap();
            if index % 7 == 0 {
                recorded.start_estimate();
                recorded.defer();
                recorded.literal(5, 3).unwrap();
                recorded.finish_estimate();
                direct
                    .model(index % 2, &[16_384, 0], |m| &mut m.cdef_index0[1])
                    .unwrap();
                recorded.defer();
            }
            direct.literal(index as u32 & 0xff, 8).unwrap();
            recorded.literal(index as u32 & 0xff, 8).unwrap();
        }
        let mut filled = 0;
        let (packet, _) = recorded
            .replay(|writer| {
                let symbol = filled % 2;
                filled += 1;
                writer.model(symbol, &[16_384, 0], |m| &mut m.cdef_index0[1])
            })
            .unwrap();
        assert_eq!(filled, 300_usize.div_ceil(7));
        assert_eq!(packet, direct.finish());
    }

    #[test]
    fn mixed_symbols_and_bypass_roundtrip() {
        let mut writer = Writer::new();
        let mut cdf = Cdf::new([30000, 16000, 5000, 0], [2, 3, 4]).unwrap();
        let mut state = 12345_u32;
        let mut values = Vec::new();
        for _ in 0..20000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let symbol = (state >> 30) as usize;
            let bits = state % 17;
            let literal = (state >> 8) & ((1 << bits) - 1);
            writer.symbol(symbol, &mut cdf).unwrap();
            writer.literal(literal, bits).unwrap();
            values.push((symbol, literal, bits));
        }
        let packet = writer.finish();
        assert_eq!(
            packet,
            include_bytes!("../tests/data/avm-entropy.bin").as_slice()
        );
        let mut reader = Reader::new(&packet).unwrap();
        let mut decoded_cdf = Cdf::new([30000, 16000, 5000, 0], [2, 3, 4]).unwrap();
        for (symbol, literal, bits) in values {
            assert_eq!(reader.symbol(&mut decoded_cdf).unwrap(), symbol);
            assert_eq!(reader.literal(bits).unwrap(), literal);
        }
        assert_eq!(cdf, decoded_cdf);
    }

    #[test]
    fn all_alphabets_and_full_width_bypass() {
        for size in 2..=16 {
            let cdf: Vec<u16> = (0..size)
                .map(|i| (TOP as usize * (size - 1 - i) / size) as u16)
                .collect();
            let mut writer = Writer::new();
            for symbol in 0..size {
                writer.fixed(symbol, &cdf).unwrap();
                writer.literal(u32::MAX, 32).unwrap();
                writer.literal(0, 32).unwrap();
            }
            let packet = writer.finish();
            let mut reader = Reader::new(&packet).unwrap();
            for symbol in 0..size {
                assert_eq!(reader.fixed(&cdf).unwrap(), symbol);
                assert_eq!(reader.literal(32).unwrap(), u32::MAX);
                assert_eq!(reader.literal(32).unwrap(), 0);
            }
        }
    }

    #[test]
    fn rejects_bad_cdfs_and_exhausted_packets() {
        let mut writer = Writer::new();
        for cdf in [&[][..], &[0], &[32768, 0], &[1, 2, 0], &[100, 1]] {
            assert!(writer.fixed(0, cdf).is_err());
        }
        assert!(writer.fixed(2, &[16000, 0]).is_err());
        assert!(writer.literal(4, 2).is_err());
        assert!(Reader::new(&[]).is_err());
        let mut reader = Reader::new(&[0]).unwrap();
        assert!(reader.literal(32).is_err());
    }
}
