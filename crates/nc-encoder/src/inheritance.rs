//! Inheritance: how much of a high-quality frame's quality the frames after
//! it keep (`docs/handoff/unified-structure.md` §12).
//!
//! Both the QP cadence (`qp_cadence.rs`) and a layered mini-GOP
//! (`structure.rs`) code one frame in four finer and rely on the frames
//! between to inherit it: static background is copied, so the coarser frames
//! keep the finer frame's quality almost free. Where they do not inherit
//! (temporal noise, which no reference predicts), every bit a high-quality
//! frame spends above the level is lost within the mini-GOP, in a chain and
//! more so in layers.
//!
//! The measure is the luma PSNR the other frames of a mini-GOP lose against
//! its high-quality frame, per QP unit they are coded coarser: a running
//! average over mini-GOPs. It needs no extra analysis (the frame's squared
//! error after coding) and reads the same in both shapes: on the 31
//! long-window clips of draw 2026-11-14 (AV1 Default, 600 frames) the value
//! measured in a layered stream was 0.96 of the chain's (rank correlation
//! 0.96), so a shape switch does not move it.
//!
//! What it separates (same clips, forced L1T3 against flat, PSNR-YUV rate):
//! below 0.16 dB per QP unit layering won on 13 of 18 clips (mean -3.2%); at
//! or above it won on 3 of 13 (mean +6.8%, every low-light clip +5.6 to
//! +14.3%). The static share, which the cadence strength reads today, ranks
//! the same clips the wrong way (rank correlation -0.26 against +0.60).
//!
//! Two decisions read it, both off by default:
//!
//! - the shape (`layers.adaptive_max_inheritance_loss`): a mini-GOP is
//!   layered only while the loss is below the limit (`structure.rs`);
//! - the depth (`structure.inherited_depth`, rate control only): the cadence
//!   is at least as strong as the loss allows, full while it is low and less
//!   as it grows, whatever the static share says. The static share stays as
//!   the other floor: it says how much of the picture is copied, and the
//!   loss, a whole-frame figure, is dominated by the part that is not (a
//!   half-still picture beside noise recovered 2 dB short after a starved
//!   key when the loss alone set the strength,
//!   `crates/nc-cli/tests/static_recovery.rs`). It supersedes the motion
//!   gate, a proxy for the same thing. At a fixed quantizer the depth never
//!   follows content.

use crate::params::{LayerParams, StructureParams};
use crate::structure::INHERITANCE_LOSS_OFF;

/// Peak 8-bit sample value squared, for PSNR.
const PEAK_SQUARED: f64 = 255. * 255.;
/// Least QP-unit gap to the high-quality frame for a frame to count: below
/// it the PSNR difference is noise over a near-zero divisor.
const MIN_QP_GAP: f64 = 1.;
/// Weight of the newest mini-GOP in the running average: about ten
/// mini-GOPs, the time constant of the structure's motion trend
/// (`structure.rs`, `TREND_WEIGHT` per frame is faster; a mini-GOP's loss is
/// one noisy sample of four frames).
const AVERAGE_WEIGHT: f64 = 0.1;
/// Least depth the measure allows: with no quantizer gap between the tiers
/// nothing more could be measured, and a scene that starts to inherit
/// again (the noise stops) would never get its cadence back. Unmeasured
/// content gets it too, as a probe: starting at the full depth instead
/// spent the first start-up base frame of a starved pan at -7.2 QP units
/// where nothing inherited it, and the 12 frames after it ran 1.5 to
/// 1.7 dB below the Default (HEVC, `pool_jaad_0110`, 150 kbps: the worst
/// 4-frame window 22.06 against 22.92 dB).
const MIN_DEPTH: f64 = 0.25;

/// The tier of a coded frame, as the measure reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tier {
    /// A key frame: intra coded, so its quality says nothing about what
    /// inter frames inherit. It ends the open mini-GOP.
    Key,
    /// The frame a mini-GOP starts at: a cadence base frame, or the base
    /// layer frame of a layered mini-GOP.
    HighQuality,
    /// Every other inter frame.
    Other,
}

/// The cadence depth as a function of the loss (`structure.inherited_depth`).
#[derive(Debug, Clone, Copy, PartialEq)]
struct DepthRule {
    /// `structure.depth_full_loss`: full depth up to this loss.
    full_loss: f64,
    /// `structure.depth_zero_loss`: no depth from this loss on.
    zero_loss: f64,
}

impl DepthRule {
    fn depth(self, loss: f64) -> f64 {
        let span = (self.zero_loss - self.full_loss).max(f64::EPSILON);
        ((self.zero_loss - loss) / span).clamp(MIN_DEPTH, 1.)
    }
}

/// The running inheritance measure of one stream.
#[derive(Debug, Clone)]
pub(crate) struct Inheritance {
    depth_rule: Option<DepthRule>,
    /// QP units and luma PSNR of the open mini-GOP's high-quality frame.
    high_quality: Option<(f64, f64)>,
    /// Sum and count of the open mini-GOP's per-frame losses.
    open: (f64, u32),
    /// dB lost per QP unit, averaged over mini-GOPs; `None` until one has
    /// been measured.
    loss: Option<f64>,
}

impl Inheritance {
    /// The measure, if the shape or the depth reads it.
    pub(crate) fn new(layers: &LayerParams, structure: &StructureParams) -> Option<Self> {
        let depth_rule = (structure.inherited_depth >= 0.5).then(|| DepthRule {
            full_loss: f64::from(structure.depth_full_loss),
            zero_loss: f64::from(structure.depth_zero_loss),
        });
        let shape = layers.adaptive_max_inheritance_loss < INHERITANCE_LOSS_OFF;
        (shape || depth_rule.is_some()).then_some(Self {
            depth_rule,
            high_quality: None,
            open: (0., 0),
            loss: None,
        })
    }

    /// A frame of `tier` was coded at `qp_units` (6 log2 of its quantizer
    /// step) with luma mean squared error `mse`.
    pub(crate) fn observe(&mut self, tier: Tier, qp_units: f64, mse: f64) {
        let psnr = 10. * (PEAK_SQUARED / mse.max(1e-3)).log10();
        match tier {
            Tier::Key => {
                self.close();
                self.high_quality = None;
            }
            Tier::HighQuality => {
                self.close();
                self.high_quality = Some((qp_units, psnr));
            }
            Tier::Other => {
                if let Some((hq_qp, hq_psnr)) = self.high_quality
                    && qp_units - hq_qp >= MIN_QP_GAP
                {
                    self.open.0 += (hq_psnr - psnr) / (qp_units - hq_qp);
                    self.open.1 += 1;
                }
            }
        }
    }

    /// Fold the open mini-GOP into the average.
    fn close(&mut self) {
        let (sum, count) = std::mem::take(&mut self.open);
        if count > 0 {
            let sample = sum / f64::from(count);
            self.loss = Some(
                self.loss
                    .map_or(sample, |old| old + AVERAGE_WEIGHT * (sample - old)),
            );
        }
    }

    /// dB of luma PSNR lost per QP unit of offset from the high-quality
    /// frame; `None` until a mini-GOP has been measured.
    pub(crate) fn loss(&self) -> Option<f64> {
        self.loss
    }

    /// The least cadence strength (0..1) the loss asks for, or `None` when
    /// the depth does not follow it. Unmeasured content gets the probe
    /// depth ([`MIN_DEPTH`]): enough to set the tiers apart and measure it.
    pub(crate) fn depth(&self) -> Option<f64> {
        self.depth_rule
            .map(|rule| self.loss.map_or(MIN_DEPTH, |loss| rule.depth(loss)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measure(depth: bool) -> Inheritance {
        Inheritance {
            depth_rule: depth.then_some(DepthRule {
                full_loss: 0.25,
                zero_loss: 0.75,
            }),
            high_quality: None,
            open: (0., 0),
            loss: None,
        }
    }

    /// Mean squared error of a frame at `psnr` dB.
    fn mse(psnr: f64) -> f64 {
        PEAK_SQUARED / 10f64.powf(psnr / 10.)
    }

    /// Code `mini_gops` mini-GOPs whose three other frames are 4 QP units
    /// coarser than the high-quality frame and `drop` dB below it.
    fn run(m: &mut Inheritance, mini_gops: u32, drop: f64) {
        for _ in 0..mini_gops {
            m.observe(Tier::HighQuality, 30., mse(40.));
            for _ in 0..3 {
                m.observe(Tier::Other, 34., mse(40. - drop));
            }
        }
        m.observe(Tier::HighQuality, 30., mse(40.));
    }

    #[test]
    fn the_loss_is_the_psnr_drop_per_qp_unit() {
        let mut m = measure(false);
        assert_eq!(m.loss(), None);
        run(&mut m, 1, 0.4);
        assert!((m.loss().unwrap() - 0.1).abs() < 1e-9);
        // The average follows a change of content over about ten mini-GOPs.
        run(&mut m, 30, 1.6);
        assert!((m.loss().unwrap() - 0.4).abs() < 0.02);
    }

    #[test]
    fn frames_without_a_quantizer_gap_and_keys_are_not_measured() {
        let mut m = measure(false);
        m.observe(Tier::HighQuality, 30., mse(40.));
        m.observe(Tier::Other, 30.5, mse(38.));
        m.observe(Tier::Key, 20., mse(45.));
        // The frames after a key have no high-quality inter frame to compare with.
        m.observe(Tier::Other, 34., mse(39.));
        m.observe(Tier::HighQuality, 30., mse(40.));
        assert_eq!(m.loss(), None);
    }

    #[test]
    fn depth_probes_until_measured_then_falls_with_the_loss() {
        let mut m = measure(true);
        assert_eq!(m.depth(), Some(MIN_DEPTH));
        run(&mut m, 1, 0.4);
        assert_eq!(m.depth(), Some(1.));
        let mut noisy = measure(true);
        run(&mut noisy, 1, 2.);
        assert!((noisy.depth().unwrap() - 0.5).abs() < 1e-9);
        let mut lost = measure(true);
        run(&mut lost, 1, 4.);
        assert_eq!(lost.depth(), Some(MIN_DEPTH));
        assert_eq!(measure(false).depth(), None);
    }
}
