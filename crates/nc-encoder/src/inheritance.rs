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

/// Fewest closed mini-GOPs before the cadence depth may exceed 1.0: avoids
/// amplifying offsets on a single noisy initial measurement.
const MIN_DEEP_SAMPLES: u32 = 2;
/// Default `depth_max` when `structure.depth_model = 1` and `structure.depth_max`
/// is left at its 1.0 default.
const DEFAULT_MODEL_MAX_DEPTH: f64 = 1.5;
/// EMA weight per frame for the linear depth model's causal content features
/// (~30-frame window time constant).
const MODEL_FEATURE_ALPHA: f64 = 0.03;
/// The key period (frames) from which `kf_long` starts to rise, and its span to 1.
const MODEL_SHORT_KEY_PERIOD: f64 = 150.;
/// The model's score at which the depth reaches `depth_max` (scores at or above 0 keep 1.0).
const MODEL_SCORE_SPAN: f64 = 1.5;
/// Floor of the bits per pixel before its logarithm.
const MODEL_MIN_BPP: f64 = 1e-6;

/// The pooled 8-feature Ridge fit of `structure.depth_model` on stored structure-oracle
/// CBR walks (docs/analysis/structure-chooser-offline-2026-11-24.md): a negative score
/// favours a deeper cadence.
struct DepthModelWeights {
    bias: f64,
    log_bpp: f64,
    kf_long: f64,
    frames_since_key: f64,
    in_startup: f64,
    static_share: f64,
    camera_motion: f64,
    dark_share: f64,
    mean_luma: f64,
}

const DEPTH_MODEL: DepthModelWeights = DepthModelWeights {
    bias: 8.1846,
    log_bpp: 0.1307,
    kf_long: -1.1498,
    frames_since_key: -2.4037,
    in_startup: 1.6948,
    static_share: -5.7958,
    camera_motion: 0.1441,
    dark_share: 1.5214,
    mean_luma: -0.01256,
};

/// Per-frame causal context passed to [`Inheritance::depth_for`].
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DepthContext {
    pub(crate) key_period: u64,
    pub(crate) frames_since_key: u64,
    pub(crate) in_startup: bool,
    pub(crate) bpp: f64,
    pub(crate) camera_motion: Option<f64>,
}

/// Running causal averages of raw input features for `structure.depth_model`.
#[derive(Debug, Clone, Copy, Default)]
struct DepthModelState {
    initialized: bool,
    static_share: f64,
    camera_motion: f64,
    dark_frac: f64,
    mean_luma: f64,
}

impl DepthModelState {
    fn observe(&mut self, static_share: f64, camera_motion: f64, dark_frac: f64, mean_luma: f64) {
        if self.initialized {
            self.static_share += MODEL_FEATURE_ALPHA * (static_share - self.static_share);
            self.camera_motion += MODEL_FEATURE_ALPHA * (camera_motion - self.camera_motion);
            self.dark_frac += MODEL_FEATURE_ALPHA * (dark_frac - self.dark_frac);
            self.mean_luma += MODEL_FEATURE_ALPHA * (mean_luma - self.mean_luma);
        } else {
            self.initialized = true;
            self.static_share = static_share;
            self.camera_motion = camera_motion;
            self.dark_frac = dark_frac;
            self.mean_luma = mean_luma;
        }
    }
}

/// The depth model's score for the frame `context` describes, from the running content
/// features `model`.
fn model_score(context: DepthContext, model: DepthModelState) -> f64 {
    let w = &DEPTH_MODEL;
    let kf_long = ((context.key_period as f64 - MODEL_SHORT_KEY_PERIOD) / MODEL_SHORT_KEY_PERIOD)
        .clamp(0., 1.);
    let since_key = if context.key_period > 0 {
        (context.frames_since_key as f64 / context.key_period as f64).clamp(0., 1.)
    } else {
        0.
    };
    let in_startup = if context.in_startup { 1. } else { 0. };
    w.bias
        + w.log_bpp * context.bpp.max(MODEL_MIN_BPP).ln()
        + w.kf_long * kf_long
        + w.frames_since_key * since_key
        + w.in_startup * in_startup
        + w.static_share * model.static_share
        + w.camera_motion * model.camera_motion
        + w.dark_share * model.dark_frac
        + w.mean_luma * model.mean_luma
}

/// Linear backoff from `1.0` at `0.5 * limit` to `0.0` at `limit` (`1.0` when `limit <= 0`).
fn backoff(value: f64, limit: f64) -> f64 {
    if limit <= 0. {
        return 1.;
    }
    let half = 0.5 * limit;
    ((limit - value) / half).clamp(0., 1.)
}

/// The cadence depth as a function of the loss (`structure.inherited_depth`).
#[derive(Debug, Clone, Copy, PartialEq)]
struct DepthRule {
    /// `structure.depth_full_loss`: full depth up to this loss.
    full_loss: f64,
    /// `structure.depth_zero_loss`: no depth from this loss on.
    zero_loss: f64,
    /// `structure.depth_max`: maximum depth on fully inherited scenes (1.0 = off).
    max_depth: f64,
    /// `structure.depth_deep_loss`: loss at which depth reaches `max_depth`.
    deep_loss: f64,
    /// `structure.depth_deep_min_period`: shortest key period (frames) for depth > 1.
    deep_min_period: u64,
    /// `structure.depth_deep_max_motion`: camera motion backoff limit (samples/frame).
    deep_max_motion: f64,
    /// `structure.depth_deep_max_noise`: sensor noise backoff limit (sigma).
    deep_max_noise: f64,
    /// `structure.depth_model`: use the 8-feature linear chooser instead of hand thresholds.
    depth_model: bool,
}

impl DepthRule {
    fn depth(self, loss: f64) -> f64 {
        let span = (self.zero_loss - self.full_loss).max(f64::EPSILON);
        ((self.zero_loss - loss) / span).clamp(MIN_DEPTH, 1.)
    }

    fn depth_for(
        self,
        loss: f64,
        samples: u32,
        ctx: DepthContext,
        noise_sigma: f64,
        model: DepthModelState,
    ) -> f64 {
        let base = self.depth(loss);
        if self.depth_model {
            if samples < MIN_DEEP_SAMPLES || loss >= self.full_loss || !model.initialized {
                return base;
            }
            let max_d = if self.max_depth > 1. {
                self.max_depth
            } else {
                DEFAULT_MODEL_MAX_DEPTH
            };
            let score = model_score(ctx, model);
            let deepen = (-score / MODEL_SCORE_SPAN).clamp(0., 1.);
            return 1. + (max_d - 1.) * deepen;
        }
        if self.max_depth <= 1.
            || samples < MIN_DEEP_SAMPLES
            || loss >= self.full_loss
            || ctx.key_period < self.deep_min_period
        {
            return base;
        }
        let deep_span = (self.full_loss - self.deep_loss).max(f64::EPSILON);
        let t_loss = ((self.full_loss - loss) / deep_span).clamp(0., 1.);
        let t_motion = ctx
            .camera_motion
            .map_or(1., |m| backoff(m, self.deep_max_motion));
        let t_noise = backoff(noise_sigma, self.deep_max_noise);
        1. + (self.max_depth - 1.) * t_loss * t_motion * t_noise
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
    /// Number of closed mini-GOPs folded into `loss`.
    samples: u32,
    /// Latest frame's sensor noise estimate (luma sigma).
    noise_sigma: f64,
    /// Running causal features for `structure.depth_model`.
    model_state: DepthModelState,
}

impl Inheritance {
    /// The measure, if the shape or the depth reads it.
    pub(crate) fn new(layers: &LayerParams, structure: &StructureParams) -> Option<Self> {
        let depth_rule = (structure.inherited_depth >= 0.5).then(|| DepthRule {
            full_loss: f64::from(structure.depth_full_loss),
            zero_loss: f64::from(structure.depth_zero_loss),
            max_depth: f64::from(structure.depth_max),
            deep_loss: f64::from(structure.depth_deep_loss),
            deep_min_period: structure.depth_deep_min_period.round().max(0.) as u64,
            deep_max_motion: f64::from(structure.depth_deep_max_motion),
            deep_max_noise: f64::from(structure.depth_deep_max_noise),
            depth_model: structure.depth_model >= 0.5,
        });
        let shape = layers.adaptive_max_inheritance_loss < INHERITANCE_LOSS_OFF;
        (shape || depth_rule.is_some()).then_some(Self {
            depth_rule,
            high_quality: None,
            open: (0., 0),
            loss: None,
            samples: 0,
            noise_sigma: 0.,
            model_state: DepthModelState::default(),
        })
    }

    /// Whether `structure.depth_model` is active.
    pub(crate) fn uses_depth_model(&self) -> bool {
        self.depth_rule.is_some_and(|r| r.depth_model)
    }

    /// Update the running causal content features for `structure.depth_model`.
    pub(crate) fn observe_frame_features(
        &mut self,
        static_share: f64,
        camera_motion: f64,
        dark_frac: f64,
        mean_luma: f64,
    ) {
        self.model_state
            .observe(static_share, camera_motion, dark_frac, mean_luma);
    }

    /// Record the latest frame's estimated sensor noise sigma (for `structure.depth_deep_max_noise`).
    pub(crate) fn set_noise_sigma(&mut self, sigma: f64) {
        self.noise_sigma = sigma;
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
            self.samples = self.samples.saturating_add(1);
        }
    }

    /// dB of luma PSNR lost per QP unit of offset from the high-quality
    /// frame; `None` until a mini-GOP has been measured.
    pub(crate) fn loss(&self) -> Option<f64> {
        self.loss
    }

    /// The least cadence strength (0..1) without the deep-scene extension.
    #[cfg(test)]
    pub(crate) fn depth(&self) -> Option<f64> {
        self.depth_for(DepthContext::default())
    }

    /// The least cadence strength (`0..=depth_max`) the loss asks for, or `None`
    /// when the depth does not follow it. Unmeasured content gets the probe
    /// depth ([`MIN_DEPTH`]): enough to set the tiers apart and measure it.
    pub(crate) fn depth_for(&self, ctx: DepthContext) -> Option<f64> {
        self.depth_rule.map(|rule| {
            self.loss.map_or(MIN_DEPTH, |loss| {
                rule.depth_for(loss, self.samples, ctx, self.noise_sigma, self.model_state)
            })
        })
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
                max_depth: 1.0,
                deep_loss: 0.10,
                deep_min_period: 150,
                deep_max_motion: 1.0,
                deep_max_noise: 1.5,
                depth_model: false,
            }),
            high_quality: None,
            open: (0., 0),
            loss: None,
            samples: 0,
            noise_sigma: 0.,
            model_state: DepthModelState::default(),
        }
    }

    #[test]
    fn deep_depth_rises_on_inherited_still_scenes_and_backs_off_on_motion_or_noise() {
        let mut m = Inheritance {
            depth_rule: Some(DepthRule {
                full_loss: 0.25,
                zero_loss: 0.75,
                max_depth: 1.5,
                deep_loss: 0.10,
                deep_min_period: 150,
                deep_max_motion: 1.0,
                deep_max_noise: 1.5,
                depth_model: false,
            }),
            high_quality: None,
            open: (0., 0),
            loss: None,
            samples: 0,
            noise_sigma: 0.2,
            model_state: DepthModelState::default(),
        };
        let ctx = |kp, mot| DepthContext {
            key_period: kp,
            camera_motion: Some(mot),
            ..DepthContext::default()
        };
        // Fully inherited (0.05 dB/QP loss), 3 mini-GOPs.
        run(&mut m, 3, 0.2);
        // Long key period, still camera, low noise -> full 1.5 depth.
        assert!((m.depth_for(ctx(300, 0.1)).unwrap() - 1.5).abs() < 1e-6);
        // Short key period -> capped at 1.0.
        assert!((m.depth_for(ctx(60, 0.1)).unwrap() - 1.0).abs() < 1e-6);
        // Camera motion -> backs off to 1.0.
        assert!((m.depth_for(ctx(300, 1.5)).unwrap() - 1.0).abs() < 1e-6);
        // High sensor noise -> backs off to 1.0.
        m.set_noise_sigma(2.0);
        assert!((m.depth_for(ctx(300, 0.1)).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn linear_depth_model_boosts_bright_still_long_gop_and_guards_dark_or_moving() {
        let mut m = Inheritance {
            depth_rule: Some(DepthRule {
                full_loss: 0.25,
                zero_loss: 0.75,
                max_depth: 1.0,
                deep_loss: 0.10,
                deep_min_period: 150,
                deep_max_motion: 1.0,
                deep_max_noise: 1.5,
                depth_model: true,
            }),
            high_quality: None,
            open: (0., 0),
            loss: None,
            samples: 0,
            noise_sigma: 0.0,
            model_state: DepthModelState::default(),
        };
        run(&mut m, 3, 0.2);
        // Bright still scene, post-startup, 300-frame GOP -> boosts above 1.0 toward 1.5.
        m.observe_frame_features(0.95, 0.0, 0.02, 130.0);
        let good = DepthContext {
            key_period: 300,
            frames_since_key: 210,
            in_startup: false,
            bpp: 0.03,
            camera_motion: Some(0.0),
        };
        let d_good = m.depth_for(good).unwrap();
        assert!(
            d_good > 1.35 && d_good <= 1.5,
            "expected deep boost, got {d_good}"
        );

        // Startup stretch -> stays at 1.0.
        let startup = DepthContext {
            frames_since_key: 30,
            in_startup: true,
            ..good
        };
        assert!((m.depth_for(startup).unwrap() - 1.0).abs() < 1e-6);

        // Dark lowlight scene -> stays at 1.0 even when still.
        m.model_state = DepthModelState::default();
        m.observe_frame_features(0.75, 0.2, 0.85, 22.0);
        assert!((m.depth_for(good).unwrap() - 1.0).abs() < 1e-6);

        // Moving handheld/bodycam -> stays at 1.0.
        m.model_state = DepthModelState::default();
        m.observe_frame_features(0.15, 6.0, 0.10, 110.0);
        assert!((m.depth_for(good).unwrap() - 1.0).abs() < 1e-6);
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
