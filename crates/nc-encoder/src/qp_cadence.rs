//! Pseudo-layer quantizer cadence for flat low-delay P streams.
//!
//! SVT-AV1's real-time flat IPP gives every fourth frame "base" treatment and
//! its rate control spends more bits there (a virtual three-layer allocation,
//! `rc_rtc_cbr.c: calc_pframe_target_size`). Each P frame still references the
//! previous frame, so the stream stays plain IPP and standard-conformant: only
//! the frame (slice) quantizer changes. A finer base frame is inherited by the
//! following frames wherever they copy or skip (static background), which
//! costs fewer bits than refining every frame equally.
//!
//! Offsets are in *QP units*: one unit is a 2^(1/6) quantizer-step ratio, the
//! H.264/HEVC QP step. AV1 converts them through its qindex step table, so the
//! same parameter means the same step ratio in every codec.
//!
//! Positions count from the most recent key frame: position 0 is the key
//! itself, positions that are multiples of the period are base frames, odd
//! positions are leaf frames and the rest are middle frames.
//!
//! The offsets apply in proportion to a strength (0..1): by default the
//! frame's still-block share (`rate.cadence_static_scaling`). With
//! `rate.cadence_motion_gate` a frame with little camera motion and a low
//! compensable share also gets the full cadence: on clean talking-head
//! content (conference) the still-block share is low although later frames
//! inherit most of the picture, and the scaled cadence left those streams
//! almost flat (layers-conf, docs/handoff/hierarchical-p.md "Conference
//! content": period 8 unscaled -10..-20% PSNR-YUV vs flat CBR in HEVC, H.264
//! and AV1). Camera motion still turns it off.
//!
//! An optional start-up period (`rate.cadence_startup_period`) applies to the
//! first `rate.cadence_startup_frames` after a key, rounded up to a whole
//! start-up cycle so the period changes only at a base frame. A deeper
//! cadence pays while the frames still inherit a recent key, but over a long
//! static scene its rarer base frames must each re-refine more decay (VVC
//! `long_static_chirla_s024c2`, one key, 300 kbps: period 8 base frames took
//! ~70% of every 8-frame budget and the other frames ran ~3 QP coarser,
//! -0.13 dB Y vs period 4; with 5 s keys period 8 won +0.28 dB).

use crate::layer::LayerPosition;
use crate::params::RateParams;
use crate::structure::MAX_TEMPORAL_LAYERS;

/// QP units per octave of quantizer step (H.264 §8.5.12.1: step doubles every 6 QP).
pub(crate) const QP_UNITS_PER_OCTAVE: f64 = 6.;

/// Fraction of each motion limit below which the motion gate gives the full
/// cadence; between it and the limit the strength falls linearly to 0, so
/// a scene near a limit does not flip between full and no cadence.
const MOTION_FULL_FRACTION: f64 = 0.75;

/// The motion of one inter frame against its immediate predecessor, as
/// the motion gate reads it (`structure::MotionSignal`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct MotionHint {
    /// Camera motion in luma samples per frame.
    pub camera_motion: f64,
    /// Compensable share of the frame difference.
    pub compensable: f64,
}

/// Pseudo-layer role of an inter frame within the cadence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CadenceRole {
    /// Every `period`th frame after a key: coded finer.
    Base,
    /// Even positions between base frames (period 4: position 2).
    Middle,
    /// Odd positions: never the reference of a base-level frame.
    Leaf,
}

impl CadenceRole {
    /// Number of roles, for per-role rate models.
    pub const COUNT: usize = 3;

    /// The role of a real temporal layer: T0 is the base, the top
    /// (non-reference) layer is the leaf, and T1 of three layers the middle.
    pub const fn of_layer(position: LayerPosition) -> Self {
        if position.layer == 0 {
            Self::Base
        } else if position.is_top_layer() {
            Self::Leaf
        } else {
            Self::Middle
        }
    }

    /// Dense index for per-role tables.
    pub const fn index(self) -> usize {
        match self {
            Self::Base => 0,
            Self::Middle => 1,
            Self::Leaf => 2,
        }
    }
}

/// The configured cadence: period and per-role QP offsets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct QpCadence {
    period: u64,
    /// Period of the first `startup_frames` frames after a key (0 = none).
    startup_period: u64,
    /// QP units subtracted on the start-up base frames.
    startup_base_boost: f64,
    /// The last start-up frame: a base frame of the start-up cadence.
    startup_end: u64,
    /// QP units subtracted on base frames.
    base_boost: f64,
    /// QP units added on leaf frames.
    leaf_penalty: f64,
    /// 0: full offsets on every frame; 1: offsets scaled by the static share.
    static_scaling: f64,
    /// Motion limits of the motion gate, or `None` when it is off.
    motion_limits: Option<MotionHint>,
}

impl QpCadence {
    pub fn from_params(params: &RateParams) -> Self {
        let startup_period = params.cadence_startup_period.round() as u64;
        let startup_frames = params.cadence_startup_frames.round() as u64;
        Self {
            period: (params.cadence_period.round() as u64).max(1),
            startup_period,
            startup_end: if startup_period > 1 {
                startup_frames.div_ceil(startup_period) * startup_period
            } else {
                0
            },
            base_boost: f64::from(params.cadence_base_boost),
            startup_base_boost: if params.cadence_startup_base_boost > 0. {
                f64::from(params.cadence_startup_base_boost)
            } else {
                f64::from(params.cadence_base_boost)
            },
            leaf_penalty: f64::from(params.cadence_leaf_penalty),
            static_scaling: f64::from(params.cadence_static_scaling),
            motion_limits: (params.cadence_motion_gate >= 0.5).then(|| MotionHint {
                camera_motion: f64::from(params.cadence_max_camera_motion),
                compensable: f64::from(params.cadence_max_compensable),
            }),
        }
    }

    /// The cadence with every offset multiplied by `scale` (a backend's
    /// `RateModel::cadence_scale`).
    pub fn scaled(self, scale: f64) -> Self {
        Self {
            base_boost: self.base_boost * scale,
            startup_base_boost: self.startup_base_boost * scale,
            leaf_penalty: self.leaf_penalty * scale,
            ..self
        }
    }

    /// Whether every inter frame gets the same quantizer (the cadence is off).
    pub fn is_flat(&self) -> bool {
        (self.period <= 1 && self.startup_end == 0)
            || (self.base_boost == 0. && self.startup_base_boost == 0. && self.leaf_penalty == 0.)
    }

    /// The period in force at `distance` frames after the last key, and
    /// the distance its cycles count from.
    fn period_at(&self, distance: u64) -> (u64, u64) {
        if distance <= self.startup_end {
            (self.startup_period, 0)
        } else {
            (self.period, self.startup_end)
        }
    }

    /// Role of the inter frame `distance` frames after the last key (`distance >= 1`).
    pub fn role(&self, distance: u64) -> CadenceRole {
        let (period, origin) = self.period_at(distance);
        if period > 1 && (distance - origin).is_multiple_of(period) {
            CadenceRole::Base
        } else if distance % 2 == 1 {
            CadenceRole::Leaf
        } else {
            CadenceRole::Middle
        }
    }

    /// Offset strength (0..1) for a frame with this share of static blocks
    /// and, with the motion gate, this motion. A finer base frame pays off
    /// only where later frames inherit it; under camera motion every frame
    /// re-codes the picture.
    pub fn strength(&self, static_fraction: f64, motion: Option<MotionHint>) -> f64 {
        let still = static_strength(self.static_scaling, static_fraction);
        match (self.motion_limits, motion) {
            (Some(limits), Some(motion)) => still.max(low_motion_strength(motion, limits)),
            _ => still,
        }
    }

    /// Quantizer offset of the inter frame `distance` frames after the last
    /// key at `strength`, in QP units (negative is finer).
    pub fn offset(&self, distance: u64, strength: f64) -> f64 {
        if self.is_flat() {
            return 0.;
        }
        let base_boost = if distance <= self.startup_end {
            self.startup_base_boost
        } else {
            self.base_boost
        };
        strength
            * match self.role(distance) {
                CadenceRole::Base => -base_boost,
                CadenceRole::Middle => 0.,
                CadenceRole::Leaf => self.leaf_penalty,
            }
    }

    /// The distances from the last key of the cadence cycle holding the
    /// frame `distance` frames after it (`distance >= 1`), in coding order.
    pub fn cycle(&self, distance: u64) -> std::ops::RangeInclusive<u64> {
        let (period, origin) = self.period_at(distance);
        let period = period.max(1);
        let start = origin + (distance - origin - 1) / period * period;
        start + 1..=start + period
    }
}

/// Per-layer quantizer offsets of a hierarchical-P structure, in QP units,
/// from the `layers.*` step ratios. T0 keeps the level (configured)
/// quantizer and upper layers are coarser. Anchoring the ladder at T0
/// rather than at the top layer keeps the lowest rates reachable: with the
/// top layer at the maximum quantizer, T0 would be pinned eight QP finer
/// (H.264 L1T2 bodycam: 751 kbps at QP 51 vs 319 kbps flat).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LayerLadder {
    offsets: [f64; MAX_TEMPORAL_LAYERS as usize],
}

impl LayerLadder {
    /// The ladder of T0 and T1 step ratios `base` and `middle` (to the top
    /// layer's step) for a structure of `layers` layers.
    pub fn new((base, middle): (f32, f32), layers: u8) -> Self {
        let ratios = match layers {
            1 => [1.0, 1.0, 1.0],
            2 => [base, 1.0, 1.0],
            _ => [base, middle, 1.0],
        };
        Self {
            offsets: ratios
                .map(|r| QP_UNITS_PER_OCTAVE * (f64::from(r) / f64::from(ratios[0])).log2()),
        }
    }

    /// Offset of the frame at `position` (0 for T0, positive is coarser).
    pub fn offset(&self, position: LayerPosition) -> f64 {
        self.offsets[usize::from(position.layer)]
    }
}

/// Strength (0..1) the motion gate gives a frame: 1 below
/// `MOTION_FULL_FRACTION` of both limits, 0 at either limit.
fn low_motion_strength(motion: MotionHint, limits: MotionHint) -> f64 {
    let load = |value: f64, limit: f64| {
        if limit > 0. {
            value / limit
        } else {
            f64::INFINITY
        }
    };
    let worst = load(motion.camera_motion, limits.camera_motion)
        .max(load(motion.compensable, limits.compensable));
    ((1. - worst) / (1. - MOTION_FULL_FRACTION)).clamp(0., 1.)
}

/// Strength (0..1) of a static-content boost: `scaling` 0 applies it fully,
/// 1 in proportion to the share of static blocks.
pub(crate) fn static_strength(scaling: f64, static_fraction: f64) -> f64 {
    1. - scaling * (1. - static_fraction.clamp(0., 1.))
}

/// Quantizer-step multiplier for an offset in QP units.
pub(crate) fn step_ratio(offset: f64) -> f64 {
    (offset / QP_UNITS_PER_OCTAVE).exp2()
}

/// The quantizer in `bounds` whose step is nearest (in log step) to `step`,
/// the first one on a tie. `steps[q]` must be increasing over `bounds`.
///
/// Log distances fall, then rise along increasing steps, so only the step
/// just below `step` (the first of a run of equal steps) and the first at
/// or above it can be nearest: two binary searches find them, instead of
/// a logarithm per quantizer. Block AQ asks once per leaf, over up to 256
/// AV1 quantizers (about 1 % of WASM AV1 encode time); the answer is the
/// full scan's ([`nearest_by_scan`], which also takes non-positive or
/// non-finite steps).
pub(crate) fn nearest_quantizer(steps: &[f64], bounds: (u32, u32), step: f64) -> u32 {
    let (low, high) = bounds;
    if low > high || !(step.is_finite() && step > 0.) {
        return nearest_by_scan(steps, low..=high, step).unwrap_or(low);
    }
    let range = &steps[low as usize..=high as usize];
    let first_at_or_above = |value: f64| low + range.partition_point(|&s| s < value) as u32;
    let above = first_at_or_above(step);
    let below = if above > low {
        first_at_or_above(steps[above as usize - 1])
    } else {
        low
    };
    nearest_by_scan(steps, [below, above.min(high)], step).unwrap_or(low)
}

/// The first of `candidates` whose step is nearest (in log step) to `step`.
fn nearest_by_scan(
    steps: &[f64],
    candidates: impl IntoIterator<Item = u32>,
    step: f64,
) -> Option<u32> {
    let distance = |q: u32| (steps[q as usize] / step).ln().abs();
    candidates
        .into_iter()
        .min_by(|&a, &b| distance(a).total_cmp(&distance(b)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::ParamOverrides;

    fn cadence(period: f32, boost: f32, penalty: f32) -> QpCadence {
        let mut o = ParamOverrides::default();
        o.set("rate.cadence_period", period);
        o.set("rate.cadence_base_boost", boost);
        o.set("rate.cadence_leaf_penalty", penalty);
        o.set("rate.cadence_static_scaling", 0.);
        QpCadence::from_params(&RateParams::from_overrides(&o))
    }

    #[test]
    fn default_is_the_static_scaled_four_frame_cadence() {
        let c = QpCadence::from_params(&RateParams::default());
        assert_eq!(c.cycle(1).count(), 4);
        assert_eq!(c.offset(4, c.strength(1., None)), -3.);
        assert_eq!(c.offset(4, c.strength(0., None)), 0.);
        let flat = cadence(1., 3., 1.);
        assert!(flat.is_flat());
        assert!((1..9).all(|d| flat.offset(d, 1.) == 0.));
    }

    #[test]
    fn four_frame_roles_follow_svt_pseudo_layers() {
        use CadenceRole::{Base, Leaf, Middle};
        let c = cadence(4., 3., 1.);
        assert_eq!(
            c.cycle(3).map(|d| c.role(d)).collect::<Vec<_>>(),
            [Leaf, Middle, Leaf, Base]
        );
        assert_eq!(c.offset(4, 1.), -3.);
        assert_eq!(c.offset(5, 1.), 1.);
        assert_eq!(c.offset(6, 1.), 0.);
        assert_eq!(c.offset(8, 0.5), -1.5);
    }

    #[test]
    fn startup_period_switches_at_a_base_frame() {
        use CadenceRole::{Base, Leaf, Middle};
        let mut o = ParamOverrides::default();
        o.set("rate.cadence_period", 4.);
        o.set("rate.cadence_startup_period", 8.);
        o.set("rate.cadence_startup_frames", 12.);
        o.set("rate.cadence_startup_base_boost", 6.);
        o.set("rate.cadence_static_scaling", 0.);
        let c = QpCadence::from_params(&RateParams::from_overrides(&o));
        // 12 frames round up to the start-up base frame 16.
        assert_eq!(c.role(4), Middle);
        assert_eq!(c.role(8), Base);
        assert_eq!(c.role(12), Middle);
        assert_eq!(c.role(16), Base);
        assert_eq!(c.role(20), Base);
        assert_eq!(c.cycle(10), 9..=16);
        assert_eq!(
            c.cycle(17).map(|d| c.role(d)).collect::<Vec<_>>(),
            [Leaf, Middle, Leaf, Base]
        );
        assert_eq!(c.cycle(16), 9..=16);
        assert_eq!(c.offset(16, 1.), -6.);
        assert_eq!(c.offset(20, 1.), -3.);
        let plain = cadence(4., 3., 1.);
        assert_eq!(plain.role(8), Base);
        assert_eq!(plain.role(6), Middle);
    }

    #[test]
    fn motion_gate_gives_low_motion_frames_the_full_cadence() {
        let mut o = ParamOverrides::default();
        o.set("rate.cadence_motion_gate", 1.);
        o.set("rate.cadence_max_camera_motion", 2.);
        o.set("rate.cadence_max_compensable", 0.6);
        let c = QpCadence::from_params(&RateParams::from_overrides(&o));
        let motion = |camera_motion, compensable| {
            Some(MotionHint {
                camera_motion,
                compensable,
            })
        };
        // A talking head: few still blocks, little motion.
        assert_eq!(c.strength(0.1, motion(0.2, 0.3)), 1.);
        // Halfway between 75% of the camera limit and the limit.
        assert!((c.strength(0., motion(1.75, 0.1)) - 0.5).abs() < 1e-12);
        // A pan: only the still share counts.
        assert!((c.strength(0.3, motion(3., 0.1)) - 0.3).abs() < 1e-12);
        assert!((c.strength(0.3, motion(0.1, 0.9)) - 0.3).abs() < 1e-12);
        assert!((c.strength(0.3, None) - 0.3).abs() < 1e-12);
        let off = QpCadence::from_params(&RateParams::default());
        assert!((off.strength(0.1, motion(0.2, 0.3)) - 0.1).abs() < 1e-12);
    }

    #[test]
    fn static_scaling_fades_offsets_under_motion() {
        let mut c = cadence(4., 3., 1.);
        assert_eq!(c.strength(0.2, None), 1.);
        c.static_scaling = 1.;
        assert!((c.strength(0.2, None) - 0.2).abs() < 1e-12);
        assert_eq!(c.strength(1., None), 1.);
    }

    #[test]
    fn svt_layer_ladder_in_qp_units() {
        let three = LayerLadder::new((0.4, 0.7), 3);
        let at = |layer| LayerPosition {
            layer,
            layers: 3,
            position: 0,
            period: 4,
        };
        assert_eq!(three.offset(at(0)), 0.);
        assert!((three.offset(at(1)) - 6. * (0.7f64 / 0.4).log2()).abs() < 1e-6);
        assert!((three.offset(at(2)) - 6. * (1f64 / 0.4).log2()).abs() < 1e-6);
        assert_eq!(CadenceRole::of_layer(at(1)), CadenceRole::Middle);
        assert_eq!(CadenceRole::of_layer(at(2)), CadenceRole::Leaf);
    }

    #[test]
    fn integer_offsets_are_exact_on_qp_tables() {
        let steps: Vec<f64> = (0..52).map(|q| step_ratio(f64::from(q))).collect();
        for q in 10..40u32 {
            let step = steps[q as usize] * step_ratio(-3.);
            assert_eq!(nearest_quantizer(&steps, (0, 51), step), q - 3);
        }
        assert_eq!(nearest_quantizer(&steps, (20, 30), steps[5]), 20);
    }

    #[test]
    fn nearest_quantizer_matches_the_full_scan() {
        // Codec-like tables, one with runs of equal steps, against targets
        // between, on and beyond the steps, and degenerate ones.
        let smooth: Vec<f64> = (0..256).map(|q| 4. + f64::from(q).powf(1.3)).collect();
        let runs: Vec<f64> = (0..64).map(|q| f64::from(q / 3 + 1)).collect();
        for steps in [&smooth, &runs] {
            let last = steps.len() as u32 - 1;
            for bounds in [(0, last), (3, last / 2), (7, 7), (9, 8)] {
                let probes = (0..400)
                    .map(|i| f64::from(i) * steps[last as usize] / 300.)
                    .chain(steps.iter().copied())
                    .chain([0., -1., f64::INFINITY, f64::NAN]);
                for step in probes {
                    let scan =
                        nearest_by_scan(steps, bounds.0..=bounds.1, step).unwrap_or(bounds.0);
                    assert_eq!(
                        nearest_quantizer(steps, bounds, step),
                        scan,
                        "bounds {bounds:?} step {step}"
                    );
                }
            }
        }
    }
}
