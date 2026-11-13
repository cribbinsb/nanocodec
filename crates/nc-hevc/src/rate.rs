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

use crate::cabac::{BinSink, Context, NEXT_STATE_LPS, NEXT_STATE_MPS};

/// Context variables: `(pStateIdx << 1) | valMps` for 64 states.
const CONTEXT_VALUES: usize = 128;

/// Bits of each bin value in each context state.
#[derive(Debug, Clone)]
pub struct BinCosts {
    /// `[context value][bin]`.
    bits: [[f32; 2]; CONTEXT_VALUES],
}

impl BinCosts {
    /// The table of [`Context::bits`] for every state and bin.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bits: std::array::from_fn(|value| {
                let ctx = Context(value as u8);
                [ctx.bits(false), ctx.bits(true)]
            }),
        }
    }

    /// Bits of `bin` in `ctx`.
    #[must_use]
    pub fn bin(&self, ctx: Context, bin: bool) -> f32 {
        self.bits[usize::from(ctx.0)][usize::from(bin)]
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
        self.bits += self.costs.bin(*ctx, bin);
        ctx.0 = if bin == ctx.mps() {
            NEXT_STATE_MPS[usize::from(ctx.0)]
        } else {
            NEXT_STATE_LPS[usize::from(ctx.0)]
        };
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
    use crate::scan::{ScanKind, Scans};

    /// The estimate of a long residual sequence tracks the bits the
    /// arithmetic coder actually spends on it.
    #[test]
    fn estimate_tracks_the_coder() {
        let scans = Scans::new();
        let costs = BinCosts::new();
        let mut levels = [0i32; 64];
        let (mut coded_bits, mut estimate) = (0.0, 0.0);
        let mut writer = CabacWriter::new();
        let mut ctx_w = Contexts::for_slice(SliceKind::Intra, 32);
        let mut ctx_e = ctx_w.clone();
        for block in 0..200 {
            for (i, level) in levels.iter_mut().enumerate() {
                let v = ((block * 7 + i * 13) % 11) as i32 - 5;
                *level = if i < 20 { v } else { 0 };
            }
            levels[0] = 3;
            let scan = scans.get(3, ScanKind::Diagonal);
            write_residual(&mut writer, &mut ctx_w, scan, &levels, true, true);
            let mut e = BitEstimator::new(&costs);
            write_residual(&mut e, &mut ctx_e, scan, &levels, true, true);
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
