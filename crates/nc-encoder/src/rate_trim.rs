//! Rate-aware trimming of plain-quantised levels: the cheap substitute for
//! the trellis (`trellis.rs`) on rungs that cannot afford it, shared by every
//! codec (docs/handoff/cheap-rdoq.md).
//!
//! A dead-zone quantiser rounds each coefficient on its own. Most of what a
//! trellis changes on top of it is a few cheap decisions (Xu, Nguyen Canh and
//! Jeon, "Simplified Rate-Distortion Optimized Quantization for HEVC", BMSB
//! 2018; SVT-AV1's `shave_coeff`): ±1 levels whose bits are worth more than
//! the error they remove go to zero. Three rules, each on the nonzero levels
//! only (a block costs a walk over its nonzero levels, no entropy contexts):
//!
//! - **Trailing ±1** ([`RateTrim::trailing`]): from the end of block
//!   backwards, a ±1 is dropped while the distortion it adds costs less than
//!   λ times its bits, the zeros between it and the previous level and the
//!   shorter end of block (AV2's `coeff_trim`, which this generalises).
//! - **Interior ±1** ([`RateTrim::interior_bits`]): a ±1 before the end of
//!   block is dropped when its added distortion is below λ times a
//!   position-dependent bit estimate (higher at high frequencies, where a
//!   significant coefficient is the unlikely symbol): a rate-aware dead zone
//!   for the zero/one boundary.
//! - **Whole block** ([`RateTrim::block`]): a block whose remaining levels
//!   cost more (static estimate, end of block included, less the empty
//!   block's signalling) than the distortion they remove is emptied.
//!
//! The rate model is static (no entropy contexts): the codec's own RD
//! decisions after it, which price the real syntax, remain the final judge.
//! Each codec supplies its scan, λ and the distortion a level's removal adds
//! ([`trim_levels`]'s `added`), in sample-SSE units.

use crate::params::{ParamOverrides, QuantParams};

/// Estimated bits of a trailing level of one: its sign plus its
/// last-position or significance symbol (one is the likeliest level).
pub const LEVEL_ONE_BITS: f64 = 2.0;
/// Estimated bits of one zero level between a trailing level and the
/// previous nonzero one (high-frequency zeros are cheap).
pub const ZERO_BITS: f64 = 0.3;
/// Estimated bits an all-zero block saves beyond its last level: the end of
/// block or last-position class, against a cheaper all-zero flag.
pub const EMPTY_BITS: f64 = 2.0;
/// Estimated extra bits per doubling of a level above one (its greater-than
/// flags and remainder), for the whole-block rule.
const LEVEL_DOUBLING_BITS: f64 = 2.0;

/// The registered trimming controls (`quant.rd_trim*`) as every backend
/// applies them; `None` from [`Self::from_params`] while `quant.rd_trim` is 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RateTrim {
    /// Multiple of the mode-decision λ (`quant.rd_trim_lambda_scale`).
    pub lambda_scale: f64,
    /// Whether trailing ±1 levels are trimmed.
    pub trailing: bool,
    /// Bits charged to an interior ±1 at the first and the last scan
    /// position (linear between); `None` leaves interior levels alone.
    pub interior_bits: Option<(f64, f64)>,
    /// Whether a block that is not worth its bits is emptied.
    pub block: bool,
    /// Whether key frames are trimmed too.
    pub key_frames: bool,
}

impl RateTrim {
    /// The trimming `quant` switches on, if any.
    #[must_use]
    pub fn from_params(quant: &QuantParams) -> Option<Self> {
        (quant.rd_trim >= 0.5).then(|| Self {
            lambda_scale: f64::from(quant.rd_trim_lambda_scale),
            trailing: quant.rd_trim_trailing >= 0.5,
            interior_bits: (quant.rd_trim_interior_bits > 0.0
                || quant.rd_trim_interior_hf_bits > 0.0)
                .then_some((
                    f64::from(quant.rd_trim_interior_bits),
                    f64::from(quant.rd_trim_interior_hf_bits),
                )),
            block: quant.rd_trim_block >= 0.5,
            key_frames: quant.rd_trim_key >= 0.5,
        })
    }

    /// [`Self::from_params`] of an encode's parameter overrides.
    #[must_use]
    pub fn from_overrides(overrides: &ParamOverrides) -> Option<Self> {
        Self::from_params(&QuantParams::from_overrides(overrides))
    }

    /// Whether a frame that is a key frame (`key`) or not is trimmed.
    #[must_use]
    pub fn covers(&self, key: bool) -> bool {
        !key || self.key_frames
    }
}

/// The squared error that zeroing a level of `magnitude` adds to a
/// coefficient of `value` quantiser steps, at `step_sse` sample SSE per
/// squared step: `v² − (v − m)² = m (2v − m)`, for the multiply-shift
/// quantisers (H.264, HEVC, VVC) whose trellis models price candidates the
/// same way.
#[must_use]
pub fn level_removal_sse(value: f64, magnitude: u32, step_sse: f64) -> f64 {
    let m = f64::from(magnitude);
    m * (2.0 * value - m) * step_sse
}

/// The squared error that zeroing a level adds to a coefficient of
/// magnitude `coefficient` whose level reconstructs to `dequantized` (both in
/// the codec's coefficient units, `to_sse` sample SSE per squared unit):
/// `c² − (c − dq)² = dq (2c − dq)`, for the reciprocal quantisers (AV1, VP9,
/// AV2) whose trellis models price candidates in coefficient units.
#[must_use]
pub fn dequantized_removal_sse(coefficient: i64, dequantized: i64, to_sse: f64) -> f64 {
    (dequantized * (2 * coefficient - dequantized)) as f64 * to_sse
}

/// Approximate bits of an end of block at `eob` beyond the first positions:
/// the offset bits of its class, about `log2(eob)` (AV1/AV2 end-of-block
/// classes; HEVC/VVC last-position prefixes grow alike).
fn eob_offset_bits(eob: usize) -> f64 {
    (eob as f64).log2()
}

/// Drop trailing ±1 `levels` (raster, coded in `scan` order up to `eob`)
/// last first, while the distortion each adds (`added(raster)`, sample-SSE
/// units) is below `lambda` times its estimated bits (AV2's `coeff_trim`).
/// Returns the new end of block.
pub fn trim_trailing_ones<L, P>(
    levels: &mut [L],
    scan: &[P],
    eob: usize,
    lambda: f64,
    mut added: impl FnMut(usize) -> f64,
) -> usize
where
    L: Copy + Default + PartialEq + Into<i32>,
    P: Copy + Into<usize>,
{
    let Some(mut last) = eob.checked_sub(1) else {
        return 0;
    };
    loop {
        let pos: usize = scan[last].into();
        if levels[pos].into().abs() != 1 {
            return last + 1;
        }
        let previous = scan[..last]
            .iter()
            .rposition(|&p| levels[p.into()] != L::default());
        let zeros = last - previous.map_or(0, |p| p + 1);
        let bits = LEVEL_ONE_BITS
            + ZERO_BITS * zeros as f64
            + match previous {
                Some(p) => eob_offset_bits(last + 1) - eob_offset_bits(p + 1),
                None => EMPTY_BITS + eob_offset_bits(last + 1),
            };
        if added(pos) >= lambda * bits {
            return last + 1;
        }
        levels[pos] = L::default();
        match previous {
            Some(p) => last = p,
            None => return 0,
        }
    }
}

/// [`RateTrim`]'s three rules on one block: `levels` raster, coded in `scan`
/// order up to `eob`; `lambda` the mode-decision λ (sample SSE per bit,
/// before [`RateTrim::lambda_scale`]); `added(raster, magnitude)` the
/// distortion that zeroing the level of that magnitude adds. Returns the new
/// end of block (0: the block is empty).
pub fn trim_levels<L, P>(
    trim: &RateTrim,
    levels: &mut [L],
    scan: &[P],
    eob: usize,
    lambda: f64,
    mut added: impl FnMut(usize, u32) -> f64,
) -> usize
where
    L: Copy + Default + PartialEq + Into<i32>,
    P: Copy + Into<usize>,
{
    let lambda = lambda * trim.lambda_scale;
    let mut eob = eob;
    if trim.trailing {
        eob = trim_trailing_ones(levels, scan, eob, lambda, |pos| added(pos, 1));
    }
    if let Some((low, high)) = trim.interior_bits
        && eob > 1
    {
        // The last level stays: dropping it is the trailing rule's call.
        let span = (scan.len().max(2) - 1) as f64;
        for (i, &p) in scan[..eob - 1].iter().enumerate() {
            let pos: usize = p.into();
            if levels[pos].into().abs() != 1 {
                continue;
            }
            let bits = low + (high - low) * (i as f64 / span);
            if added(pos, 1) < lambda * bits {
                levels[pos] = L::default();
            }
        }
    }
    if trim.block && eob > 0 {
        let mut distortion = 0.0;
        let mut bits = EMPTY_BITS + eob_offset_bits(eob);
        let mut previous = None;
        for (i, &p) in scan[..eob].iter().enumerate() {
            let pos: usize = p.into();
            let magnitude = levels[pos].into().unsigned_abs();
            if magnitude == 0 {
                continue;
            }
            let zeros = i - previous.map_or(0, |q: usize| q + 1);
            bits += LEVEL_ONE_BITS
                + ZERO_BITS * zeros as f64
                + LEVEL_DOUBLING_BITS * f64::from(magnitude).log2();
            distortion += added(pos, magnitude);
            previous = Some(i);
            if distortion >= lambda * bits {
                // Early out: the levels already remove more than they could cost.
                break;
            }
        }
        if distortion < lambda * bits {
            levels.fill(L::default());
            return 0;
        }
    }
    // Interior drops never move the end of block, the trailing rule did.
    eob
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> RateTrim {
        RateTrim {
            lambda_scale: 1.0,
            trailing: true,
            interior_bits: None,
            block: false,
            key_frames: false,
        }
    }

    /// Distortion added by zeroing a level of `m` for a coefficient of
    /// `value` steps, at one sample-SSE unit per squared step.
    fn added_for(values: &[f64]) -> impl Fn(usize, u32) -> f64 + '_ {
        move |pos, m| {
            let v = values[pos];
            let m = f64::from(m);
            v * v - (v - m) * (v - m)
        }
    }

    #[test]
    fn trailing_rule_drops_weak_isolated_ones_and_keeps_strong_levels() {
        let scan: Vec<usize> = (0..16).collect();
        let mut values = [0.0; 16];
        let mut levels = [0i32; 16];
        values[0] = 3.0;
        levels[0] = 3;
        values[15] = 0.7;
        levels[15] = -1;
        let eob = trim_levels(&settings(), &mut levels, &scan, 16, 0.5, added_for(&values));
        assert_eq!((eob, levels[0], levels[15]), (1, 3, 0));
        // A strong trailing one right after the previous level stays.
        values[1] = 1.4;
        levels[1] = 1;
        let eob = trim_levels(&settings(), &mut levels, &scan, 2, 0.5, added_for(&values));
        assert_eq!((eob, levels[1]), (2, 1));
    }

    #[test]
    fn interior_rule_follows_the_frequency_ramp() {
        let scan: Vec<usize> = (0..16).collect();
        let mut values = [0.0; 16];
        let mut levels = [0i32; 16];
        for (pos, value) in [(0, 0.6), (12, 0.6), (15, 2.0)] {
            values[pos] = value;
            levels[pos] = if value > 1.0 { 2 } else { 1 };
        }
        let trim = RateTrim {
            trailing: false,
            interior_bits: Some((0.0, 1.0)),
            ..settings()
        };
        // Added 0.2 each; λ·bits is 0 at DC and 0.4 at position 12.
        let eob = trim_levels(&trim, &mut levels, &scan, 16, 0.5, added_for(&values));
        assert_eq!((eob, levels[0], levels[12], levels[15]), (16, 1, 0, 2));
    }

    #[test]
    fn block_rule_empties_blocks_not_worth_their_bits() {
        let scan: Vec<usize> = (0..16).collect();
        let mut values = [0.0; 16];
        let mut levels = [0i32; 16];
        values[3] = 0.9;
        levels[3] = 1;
        let trim = RateTrim {
            trailing: false,
            block: true,
            ..settings()
        };
        assert_eq!(
            trim_levels(&trim, &mut levels, &scan, 4, 0.5, added_for(&values)),
            0
        );
        assert_eq!(levels[3], 0);
        values[3] = 4.0;
        levels[3] = 4;
        assert_eq!(
            trim_levels(&trim, &mut levels, &scan, 4, 0.5, added_for(&values)),
            4
        );
        assert_eq!(levels[3], 4);
    }
}
