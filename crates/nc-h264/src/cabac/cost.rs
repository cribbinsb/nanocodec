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

use super::{BinSink, Context, MbPart};
use crate::tables::RANGE_TAB_LPS;

/// Number of probability states (`pStateIdx` 0..=63, §9.3.1.1).
const PROBABILITY_STATES: usize = 64;

/// Fixed-point unit of [`BinCosts`]: 1/256 bit.
const COST_UNITS_PER_BIT: f64 = 256.0;
/// Midpoints of the four `codIRange` quarters that index `rangeTabLPS`
/// (`(codIRange >> 6) & 3` over 256..=511, §9.3.4.2).
const RANGE_QUARTER_MIDPOINTS: [f64; 4] = [288.0, 352.0, 416.0, 480.0];

/// The cost of a bin by context state: `[MPS, LPS]` in 1/256 bit, the
/// mean over the four range quarters.
#[derive(Debug, Clone)]
pub struct BinCosts {
    costs: [[u16; 2]; PROBABILITY_STATES],
}

impl BinCosts {
    /// The cost table of the standard's probability states.
    #[must_use]
    pub fn new() -> Self {
        let mut costs = [[0u16; 2]; PROBABILITY_STATES];
        for (cost, lps_ranges) in costs.iter_mut().zip(RANGE_TAB_LPS.iter()) {
            let mean_bits = |probability: &dyn Fn(f64) -> f64| {
                let total: f64 = lps_ranges
                    .iter()
                    .zip(RANGE_QUARTER_MIDPOINTS)
                    .map(|(&lps, range)| -probability(f64::from(lps) / range).log2())
                    .sum();
                (total / RANGE_QUARTER_MIDPOINTS.len() as f64 * COST_UNITS_PER_BIT).round() as u16
            };
            *cost = [mean_bits(&|p_lps| 1.0 - p_lps), mean_bits(&|p_lps| p_lps)];
        }
        Self { costs }
    }
}

impl Default for BinCosts {
    fn default() -> Self {
        Self::new()
    }
}

/// A [`BinSink`] that adds up the cost of the bins written to it.
#[derive(Debug)]
pub struct BitCounter<'a> {
    contexts: &'a [Context],
    costs: &'a BinCosts,
    /// Running totals per [`MbPart`] in 1/256 bit.
    units: [u32; MbPart::COUNT],
    /// Index of the part being written.
    part: usize,
}

impl<'a> BitCounter<'a> {
    pub(super) fn new(contexts: &'a [Context], costs: &'a BinCosts) -> Self {
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
        self.units.iter().sum::<u32>() as f32 / COST_UNITS_PER_BIT as f32
    }

    /// Bits of `bin` in context `ctx_idx` with the current states (not
    /// counted): what rate-distortion quantisation prices one bin with.
    #[must_use]
    pub fn bin_bits(&self, ctx_idx: usize, bin: bool) -> f32 {
        let ctx = self.contexts[ctx_idx];
        let lps = usize::from(bin != ctx.val_mps());
        f32::from(self.costs.costs[usize::from(ctx.p_state_idx())][lps]) / COST_UNITS_PER_BIT as f32
    }

    /// Bits counted so far in `part`.
    #[must_use]
    pub fn part_bits(&self, part: MbPart) -> f32 {
        self.units[part.index()] as f32 / COST_UNITS_PER_BIT as f32
    }
}

impl BinSink for BitCounter<'_> {
    #[inline]
    fn encode_decision(&mut self, ctx_idx: usize, bin: bool) {
        let ctx = self.contexts[ctx_idx];
        let lps = usize::from(bin != ctx.val_mps());
        self.units[self.part] += u32::from(self.costs.costs[usize::from(ctx.p_state_idx())][lps]);
    }

    #[inline]
    fn encode_bypass(&mut self, _bin: bool) {
        self.units[self.part] += COST_UNITS_PER_BIT as u32;
    }

    fn encode_terminate(&mut self, bin: bool) {
        // A 1 ends the slice (flush: about 7 bits); a 0 costs ~2/510 bit.
        const FLUSH_BITS: u32 = 7;
        if bin {
            self.units[self.part] += FLUSH_BITS * COST_UNITS_PER_BIT as u32;
        }
    }

    fn begin_part(&mut self, part: MbPart) {
        self.part = part.index();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn costs_follow_the_probability_states() {
        let costs = BinCosts::new();
        // State 0 is equiprobable: both values cost about one bit.
        for value in costs.costs[0] {
            assert!((230..=290).contains(&value), "state 0 cost {value}");
        }
        // Costs are monotonic: higher states make the MPS cheaper and
        // the LPS dearer.
        for pair in costs.costs.windows(2) {
            assert!(pair[1][0] <= pair[0][0]);
            assert!(pair[1][1] >= pair[0][1]);
        }
        assert!(costs.costs[62][0] < 10);
        assert!(costs.costs[62][1] > 5 * 256);
    }
}
