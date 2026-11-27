//! CABAC rate estimation for the encoder's rate-distortion trials (intra
//! `PART_NxN`, transform splits).
//!
//! A trial runs the real syntax writers ([`crate::residual::write_residual`]
//! and friends) into a [`BitEstimator`] instead of the arithmetic coder.
//! Each context-coded bin costs `−log2 p` of its context state's
//! probability (the model behind HEVC Table 9-46) and adapts the trial's own
//! copy of the contexts as the coder would; a bypass bin costs one bit. The
//! per-state costs are a table built once per encoder ([`BinCosts`]), since
//! a trial prices hundreds of bins.

use crate::cabac::{BinSink, Context};

/// Context variables: `(pStateIdx << 1) | valMps` for 64 states.
const CONTEXT_VALUES: usize = 128;

/// Rate and next context of each bin value in each context state.
#[derive(Debug, Clone)]
pub struct BinCosts {
    /// Indexed by the packed context state and bin value.
    bins: [PricedBin; CONTEXT_VALUES * 2],
}

#[derive(Debug, Clone, Copy)]
struct PricedBin {
    bits: f32,
    next: Context,
}

impl BinCosts {
    /// The table of [`Context::bits`] for every state and bin.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bins: std::array::from_fn(|index| {
                let ctx = Context((index >> 1) as u8);
                let bin = index & 1 != 0;
                let mut next = ctx;
                next.update(bin);
                PricedBin {
                    bits: ctx.bits(bin),
                    next,
                }
            }),
        }
    }

    /// Bits of `bin` in `ctx`.
    #[must_use]
    pub fn bin(&self, ctx: Context, bin: bool) -> f32 {
        self.priced_bin(ctx, bin).bits
    }

    #[inline]
    fn priced_bin(&self, ctx: Context, bin: bool) -> PricedBin {
        let index = (ctx.0 << 1) | u8::from(bin);
        self.bins[usize::from(index)]
    }
}

impl Default for BinCosts {
    fn default() -> Self {
        Self::new()
    }
}

/// A [`BinSink`] that adds up estimated bits.
#[derive(Debug)]
pub struct BitEstimator<'a> {
    costs: &'a BinCosts,
    /// Bits so far.
    pub bits: f32,
}

impl<'a> BitEstimator<'a> {
    /// An empty estimate priced by `costs`.
    #[must_use]
    pub fn new(costs: &'a BinCosts) -> Self {
        Self { costs, bits: 0.0 }
    }
}

impl BinSink for BitEstimator<'_> {
    fn encode_bin(&mut self, ctx: &mut Context, bin: bool) {
        let priced = self.costs.priced_bin(*ctx, bin);
        self.bits += priced.bits;
        *ctx = priced.next;
    }

    fn encode_bypass(&mut self, _value: u32, count: u32) {
        self.bits += count as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::CabacWriter;
    use crate::contexts::{Contexts, SliceKind};
    use crate::residual::write_residual;
    use crate::scan::{ScanKind, Scans, coded_sub_blocks};

    #[test]
    fn priced_bins_match_the_cabac_state_machine() {
        let costs = BinCosts::new();
        for state in 0..CONTEXT_VALUES as u8 {
            for bin in [false, true] {
                let mut actual = Context(state);
                let bits = actual.bits(bin);
                let mut expected = actual;
                expected.update(bin);
                let mut estimator = BitEstimator::new(&costs);
                estimator.encode_bin(&mut actual, bin);
                assert_eq!(estimator.bits.to_bits(), bits.to_bits());
                assert_eq!(actual, expected);
            }
        }
    }

    /// The estimate of a long residual sequence tracks the bits the
    /// arithmetic coder actually spends on it.
    #[test]
    fn estimate_tracks_the_coder() {
        let scans = Scans::new();
        let costs = BinCosts::new();
        let mut levels = [0i16; 64];
        let (mut coded_bits, mut estimate) = (0.0, 0.0);
        let mut writer = CabacWriter::new();
        let mut ctx_w = Contexts::for_slice(SliceKind::Intra, 32);
        let mut ctx_e = ctx_w.clone();
        for block in 0..200 {
            for (i, level) in levels.iter_mut().enumerate() {
                let v = ((block * 7 + i * 13) % 11) as i16 - 5;
                *level = if i < 20 { v } else { 0 };
            }
            levels[0] = 3;
            let scan = scans.get(3, ScanKind::Diagonal);
            let coded = (&levels[..], coded_sub_blocks(&levels, scan));
            write_residual(&mut writer, &mut ctx_w, scan, coded, true, true);
            let mut e = BitEstimator::new(&costs);
            write_residual(&mut e, &mut ctx_e, scan, coded, true, true);
            estimate += e.bits;
        }
        writer.encode_terminate(true);
        let mut data = writer.finish();
        data.trailing_bits();
        coded_bits += data.into_bytes().len() as f32 * 8.0;
        let ratio = estimate / coded_bits;
        assert!((0.97..1.03).contains(&ratio), "estimate/coded = {ratio}");
    }
}
