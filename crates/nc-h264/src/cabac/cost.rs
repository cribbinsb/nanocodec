//! CABAC bit-cost estimation: the syntax writers run against a
//! [`BitCounter`] instead of the arithmetic encoder to price a macroblock
//! before it is coded (rate-distortion decisions).
//!
//! A bin coded with context state `pStateIdx` costs `-log2 p` bits, where
//! `p` is the probability the engine gives the bin's value (H.264 §9.3.4.2:
//! `codIRangeLPS / codIRange`). The counter reads the current context
//! states and leaves them unchanged, so a macroblock is priced with the
//! probabilities it starts with (adaptation within one macroblock is
//! ignored). Bypass bins cost one bit; terminate bins with value 0 cost
//! nothing (`codIRange - 2` of `codIRange`).

use super::{BinRun, BinSink, Context, MbPart};
use crate::tables::CONTEXT_COUNT;
use crate::tables::RANGE_TAB_LPS;

/// Number of probability states (`pStateIdx` 0..=63, §9.3.1.1).
const PROBABILITY_STATES: usize = 64;

/// Fixed-point unit of [`BinCosts`]: 1/256 bit.
const COST_UNITS_PER_BIT: f64 = 256.0;
/// Midpoints of the four `codIRange` quarters that index `rangeTabLPS`
/// (`(codIRange >> 6) & 3` over 256..=511, §9.3.4.2).
const RANGE_QUARTER_MIDPOINTS: [f64; 4] = [288.0, 352.0, 416.0, 480.0];

/// The cost in 1/256 bit by packed context state and bin value, the mean
/// over the four range quarters.
#[derive(Debug, Clone)]
pub struct BinCosts {
    costs: [u16; PROBABILITY_STATES * 4],
}

impl BinCosts {
    /// The cost table of the standard's probability states.
    #[must_use]
    pub fn new() -> Self {
        let mut costs = [0u16; PROBABILITY_STATES * 4];
        for (cost, lps_ranges) in costs.as_chunks_mut::<4>().0.iter_mut().zip(&RANGE_TAB_LPS) {
            let mean_bits = |probability: &dyn Fn(f64) -> f64| {
                let total: f64 = lps_ranges
                    .iter()
                    .zip(RANGE_QUARTER_MIDPOINTS)
                    .map(|(&lps, range)| -probability(f64::from(lps) / range).log2())
                    .sum();
                (total / RANGE_QUARTER_MIDPOINTS.len() as f64 * COST_UNITS_PER_BIT).round() as u16
            };
            let mps = mean_bits(&|p_lps| 1.0 - p_lps);
            let lps = mean_bits(&|p_lps| p_lps);
            // The low bits of the index are valMPS and the coded bin.
            *cost = [mps, lps, lps, mps];
        }
        Self { costs }
    }
}

impl BinCosts {
    /// The cost of `bin` in a context in state `ctx`, in 1/256 bit.
    #[inline]
    fn units(&self, ctx: Context, bin: bool) -> u32 {
        let index = ctx.state.wrapping_shl(1) | u8::from(bin);
        u32::from(self.costs[usize::from(index)])
    }
}

impl Default for BinCosts {
    fn default() -> Self {
        Self::new()
    }
}

/// A [`BinSink`] that adds up the cost of the bins written to it.
#[derive(Debug)]
pub struct BitCounter<'a, const TRACK_PARTS: bool = true> {
    contexts: &'a [Context; CONTEXT_COUNT],
    costs: &'a BinCosts,
    /// Running totals per [`MbPart`] in 1/256 bit.
    units: [u32; MbPart::COUNT],
    /// Index of the part being written.
    part: usize,
}

impl<'a, const TRACK_PARTS: bool> BitCounter<'a, TRACK_PARTS> {
    pub(super) fn new(contexts: &'a [Context; CONTEXT_COUNT], costs: &'a BinCosts) -> Self {
        Self {
            contexts,
            costs,
            units: [0; MbPart::COUNT],
            part: MbPart::Header.index(),
        }
    }

    /// A counter on the same context states with nothing counted.
    #[must_use]
    pub fn fresh(&self) -> Self {
        Self::new(self.contexts, self.costs)
    }

    /// Bits counted so far.
    #[must_use]
    pub fn bits(&self) -> f32 {
        let units = if TRACK_PARTS {
            self.units.iter().sum()
        } else {
            self.units[0]
        };
        units as f32 / COST_UNITS_PER_BIT as f32
    }

    /// Bits of `bin` in context `ctx_idx` with the current states (not
    /// counted): what rate-distortion quantisation prices one bin with.
    #[must_use]
    pub fn bin_bits(&self, ctx_idx: usize, bin: bool) -> f32 {
        self.costs.units(self.contexts[ctx_idx], bin) as f32 / COST_UNITS_PER_BIT as f32
    }

    #[inline]
    fn add(&mut self, units: u32) {
        if TRACK_PARTS {
            self.units[self.part] += units;
        } else {
            self.units[0] += units;
        }
    }
}

impl BitCounter<'_> {
    /// Bits counted so far in `part`.
    #[must_use]
    pub fn part_bits(&self, part: MbPart) -> f32 {
        self.units[part.index()] as f32 / COST_UNITS_PER_BIT as f32
    }
}

impl<const TRACK_PARTS: bool> BinSink for BitCounter<'_, TRACK_PARTS> {
    #[inline]
    fn encode_decision(&mut self, ctx_idx: usize, bin: bool) {
        self.add(self.costs.units(self.contexts[ctx_idx], bin));
    }

    #[inline]
    fn encode_bypass(&mut self, _bin: bool) {
        self.add(COST_UNITS_PER_BIT as u32);
    }

    #[inline]
    fn encode_bypass_bits(&mut self, _bits: u32, count: u32) {
        self.add(count * COST_UNITS_PER_BIT as u32);
    }

    fn encode_terminate(&mut self, bin: bool) {
        // A 1 ends the slice (flush: about 7 bits); a 0 costs ~2/510 bit.
        const FLUSH_BITS: u32 = 7;
        if bin {
            self.add(FLUSH_BITS * COST_UNITS_PER_BIT as u32);
        }
    }

    #[inline]
    fn begin_part(&mut self, part: MbPart) {
        if TRACK_PARTS {
            self.part = part.index();
        }
    }

    /// The run's bins (one block's levels, all in the current part) are
    /// added up in one register-held total rather than through `units[part]`
    /// per bin. Exact: the same integer sum.
    #[inline]
    fn code_run<T: BinRun>(&mut self, run: T) -> T::Output {
        let mut local = RunCounter {
            contexts: self.contexts,
            costs: self.costs,
            units: 0,
        };
        let output = run.code(&mut local);
        self.add(local.units);
        output
    }
}

/// [`BitCounter`] inside one [`BinRun`]: the bins of one part.
struct RunCounter<'a> {
    contexts: &'a [Context; CONTEXT_COUNT],
    costs: &'a BinCosts,
    /// Running total in 1/256 bit.
    units: u32,
}

impl BinSink for RunCounter<'_> {
    #[inline]
    fn encode_decision(&mut self, ctx_idx: usize, bin: bool) {
        self.units += self.costs.units(self.contexts[ctx_idx], bin);
    }

    #[inline]
    fn encode_repeated_decision(&mut self, ctx_idx: usize, bin: bool, count: u32) {
        // A count uses the same context state for the whole macroblock.
        self.units += count * self.costs.units(self.contexts[ctx_idx], bin);
    }

    #[inline]
    fn encode_bypass(&mut self, _bin: bool) {
        self.units += COST_UNITS_PER_BIT as u32;
    }

    #[inline]
    fn encode_bypass_bits(&mut self, _bits: u32, count: u32) {
        self.units += count * COST_UNITS_PER_BIT as u32;
    }

    fn encode_terminate(&mut self, _bin: bool) {
        unreachable!("a bin run holds no terminate bin");
    }

    fn begin_part(&mut self, _part: MbPart) {
        unreachable!("a bin run stays in one part");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn costs_follow_the_probability_states() {
        let costs = BinCosts::new();
        // State 0 is equiprobable: both values cost about one bit.
        for &value in &costs.costs[..4] {
            assert!((230..=290).contains(&value), "state 0 cost {value}");
        }
        // Costs are monotonic: higher states make the MPS cheaper and
        // the LPS dearer.
        for pair in costs.costs.as_chunks::<4>().0.windows(2) {
            assert!(pair[1][0] <= pair[0][0]);
            assert!(pair[1][1] >= pair[0][1]);
        }
        assert!(costs.units(Context::new(62, false), false) < 10);
        assert!(costs.units(Context::new(62, false), true) > 5 * 256);
        for state in 0..64 {
            let a = Context::new(state, false);
            let b = Context::new(state, true);
            assert_eq!(costs.units(a, false), costs.units(b, true));
            assert_eq!(costs.units(a, true), costs.units(b, false));
        }
    }
}
