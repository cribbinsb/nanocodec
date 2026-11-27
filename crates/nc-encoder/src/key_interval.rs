//! The key interval as a feature of key-frame settings: `ln(N / 300)`, with
//! `N` the frames the key serves until the next scheduled key, and the one
//! place that applies it to a key frame's search settings.
//!
//! Why: a key's value scales with the frames that predict from it, and the
//! CPU of its searches amortises over them, so a key-frame setting tuned for
//! one interval is wrong at another (owner, 2026-11-22: at least 150- and
//! 300-frame intervals are considered in every key decision). Settings that
//! read this feature take it as one term of a linear function whose weight is
//! 0 by default (today's constant setting), so default streams are unchanged:
//! - rate control: `rate.key_level_finer_ln_interval` (the key's quality
//!   requirement), `rate.key_boost_ln_interval` (its bit budget),
//!   `rate.fixed_qp_key_ln_interval` (the fixed-QP key quantizer);
//! - key searches ([`key_search_params`], [`key_effort_params`]):
//!   `planner.key_rd_margin_ln_interval`, `planner.key_rd_gate_ln_interval`,
//!   `decision.intra_rd_margin_ln_interval`,
//!   `decision.intra_rd_count_ln_interval`,
//!   `effort.key_regular_fraction_ln_interval`.
//!
//! `N` is the key period: every key restarts the schedule (`key_clock.rs`), so
//! each key serves the period until the next scheduled one (a later requested
//! or scene-cut key cannot be foreseen). A stream with no scheduled key
//! (intra refresh, an infinite period) counts as [`OPEN_GOP_SECONDS`].
//!
//! Sign convention: a positive weight puts more into keys of longer
//! intervals (finer, larger, more searched) and less into shorter ones. A
//! term never turns a gate or a search on or off; it only moves a setting
//! that is already on.

use crate::params::{DecisionParams, EffortParams, PlannerParams};

/// The interval the constant settings were tuned at (10 s at 30 fps, the
/// BD-rate sweeps' key period): the feature is 0 there.
const REFERENCE_FRAMES: f64 = 300.0;
/// An open-ended GOP (no scheduled key, or intra refresh) counts as this
/// many seconds of frames.
const OPEN_GOP_SECONDS: f64 = 10.0;
/// The fewest alternatives a key leaf search that is on keeps under
/// `decision.intra_rd_count_ln_interval` (it never turns a search off).
const MIN_LEAF_ALTERNATIVES: f32 = 1.0;

/// `ln(N / REFERENCE_FRAMES)` for a key period of `period` frames at `fps`
/// (0 or `u32::MAX`: open-ended, [`OPEN_GOP_SECONDS`] of frames).
pub(crate) fn ln_interval(period: u64, fps: f64) -> f64 {
    let frames = if period == 0 || period >= u64::from(u32::MAX) {
        OPEN_GOP_SECONDS * fps
    } else {
        period as f64
    };
    (frames.max(1.0) / REFERENCE_FRAMES).ln()
}

/// `value` moved by `weight` per unit of `ln`, where both are non-zero:
/// a gate at 0 (off) stays off, and a gate on stays on (at least
/// `f32::MIN_POSITIVE`).
fn shifted(value: f32, weight: f32, ln: f64) -> f32 {
    if value > 0.0 && weight != 0.0 {
        (value + weight * ln as f32).max(f32::MIN_POSITIVE)
    } else {
        value
    }
}

/// `value` scaled by `exp(weight * ln)` (a factor of `N / 300` to the power
/// `weight`), where both are non-zero: for thresholds whose effect is
/// multiplicative (a gain per sample, a cost ratio).
fn scaled(value: f32, weight: f32, ln: f64) -> f32 {
    if value > 0.0 && weight != 0.0 {
        value * (f64::from(weight) * ln).exp() as f32
    } else {
        value
    }
}

/// A key frame's search settings at key interval feature `ln`:
/// - the RD partition search's margins (`planner.key_rd_margin`,
///   `planner.key_rd_merge_margin`) shifted by
///   `planner.key_rd_margin_ln_interval`;
/// - its fitted trial gate (`planner.key_rd_gate`, a gain per sample a trial
///   must promise) divided by `(N / 300)^planner.key_rd_gate_ln_interval`:
///   a trial's gain is inherited by the frames up to the next key;
/// - the leaf search's margin (`decision.intra_rd_margin`) scaled by
///   `(N / 300)^decision.intra_rd_margin_ln_interval`, and its alternative
///   counts (`decision.intra_rd_modes`, `_transforms`, `_chroma`) shifted
///   by `decision.intra_rd_count_ln_interval` (at least one while on).
pub(crate) fn key_search_params(
    mut decision: DecisionParams,
    mut planner: PlannerParams,
    ln: f64,
) -> (DecisionParams, PlannerParams) {
    let margin_weight = planner.key_rd_margin_ln_interval;
    planner.key_rd_margin = shifted(planner.key_rd_margin, margin_weight, ln);
    planner.key_rd_merge_margin = shifted(planner.key_rd_merge_margin, margin_weight, ln);
    planner.key_rd_gate = scaled(planner.key_rd_gate, -planner.key_rd_gate_ln_interval, ln);
    decision.intra_rd_margin = scaled(
        decision.intra_rd_margin,
        decision.intra_rd_margin_ln_interval,
        ln,
    );
    let count_weight = decision.intra_rd_count_ln_interval;
    for count in [
        &mut decision.intra_rd_modes,
        &mut decision.intra_rd_transforms,
        &mut decision.intra_rd_chroma,
    ] {
        *count = shifted(*count, count_weight, ln).max(count.min(MIN_LEAF_ALTERNATIVES));
    }
    (decision, planner)
}

/// A key frame's effort classes at key interval feature `ln`: the share of
/// its superblocks coded with the regular settings
/// (`effort.key_regular_fraction`, below 1 only) shifted by
/// `effort.key_regular_fraction_ln_interval`, within 0..=1.
pub(crate) fn key_effort_params(mut effort: EffortParams, ln: f64) -> EffortParams {
    let weight = effort.key_regular_fraction_ln_interval;
    if effort.key_regular_fraction < 1.0 && weight != 0.0 {
        effort.key_regular_fraction =
            (effort.key_regular_fraction + weight * ln as f32).clamp(0.0, 1.0);
    }
    effort
}

/// QP units (6 per octave of quantizer step) a key's setting moves finer:
/// `constant` plus `per_ln_interval` per unit of `ln` (exactly `constant`
/// at weight 0).
pub(crate) fn finer_qp(constant: f64, per_ln_interval: f64, ln: f64) -> f64 {
    if per_ln_interval == 0.0 {
        constant
    } else {
        constant + per_ln_interval * ln
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::ParamOverrides;

    #[test]
    fn zero_at_the_reference_interval() {
        assert_eq!(ln_interval(300, 30.0), 0.0);
    }

    #[test]
    fn open_ended_gops_count_as_ten_seconds() {
        assert_eq!(ln_interval(0, 30.0), 0.0);
        assert_eq!(ln_interval(u64::MAX, 30.0), 0.0);
        assert!((ln_interval(150, 30.0) - 0.5f64.ln()).abs() < 1e-12);
    }

    fn params(settings: &[(&str, f32)]) -> (DecisionParams, PlannerParams, EffortParams) {
        let mut overrides = ParamOverrides::default();
        for &(name, value) in settings {
            overrides.set(name, value);
        }
        (
            DecisionParams::from_overrides(&overrides),
            PlannerParams::from_overrides(&overrides),
            EffortParams::from_overrides(&overrides),
        )
    }

    #[test]
    fn weight_zero_leaves_every_setting_as_it_is() {
        let (decision, planner, effort) = params(&[
            ("planner.key_rd_margin", 1.5),
            ("planner.key_rd_gate", 300.0),
            ("decision.intra_rd_margin", 1150.0),
            ("decision.intra_rd_modes", 3.0),
            ("effort.key_regular_fraction", 0.5),
        ]);
        let ln = ln_interval(150, 30.0);
        assert_eq!(
            key_search_params(decision, planner, ln),
            (decision, planner)
        );
        assert_eq!(key_effort_params(effort, ln), effort);
    }

    #[test]
    fn positive_weights_search_longer_intervals_more() {
        let (decision, planner, effort) = params(&[
            ("planner.key_rd_margin", 1.5),
            ("planner.key_rd_margin_ln_interval", 1.0),
            ("planner.key_rd_gate", 300.0),
            ("planner.key_rd_gate_ln_interval", 1.0),
            ("decision.intra_rd_margin", 1150.0),
            ("decision.intra_rd_margin_ln_interval", 1.0),
            ("decision.intra_rd_modes", 3.0),
            ("decision.intra_rd_count_ln_interval", 2.0),
            ("effort.key_regular_fraction", 0.5),
            ("effort.key_regular_fraction_ln_interval", 0.5),
        ]);
        let long = ln_interval(600, 30.0);
        let short = ln_interval(150, 30.0);
        let (d_long, p_long) = key_search_params(decision, planner, long);
        let (d_short, p_short) = key_search_params(decision, planner, short);
        assert!(p_long.key_rd_margin > p_short.key_rd_margin);
        assert!(p_long.key_rd_gate < 300.0 && p_short.key_rd_gate > 300.0);
        assert!((p_long.key_rd_gate * p_short.key_rd_gate - 300.0 * 300.0).abs() < 1.0);
        assert!(d_long.intra_rd_margin > d_short.intra_rd_margin);
        assert!(d_long.intra_rd_modes > d_short.intra_rd_modes);
        assert!(
            key_effort_params(effort, long).key_regular_fraction
                > key_effort_params(effort, short).key_regular_fraction
        );
    }

    #[test]
    fn a_term_never_turns_a_search_on_or_off() {
        let (decision, planner, _) = params(&[
            ("planner.key_rd_margin_ln_interval", 2.0),
            ("decision.intra_rd_modes", 1.0),
            ("decision.intra_rd_count_ln_interval", 4.0),
        ]);
        let (d, p) = key_search_params(decision, planner, ln_interval(30, 30.0));
        assert_eq!(p.key_rd_margin, 0.0);
        assert_eq!(d.intra_rd_transforms, 0.0);
        assert_eq!(d.intra_rd_modes, MIN_LEAF_ALTERNATIVES);
    }

    #[test]
    fn finer_qp_is_the_constant_at_weight_zero() {
        assert_eq!(finer_qp(2.0, 0.0, 5.0), 2.0);
        assert_eq!(finer_qp(2.0, 1.0, 0.5f64.ln()), 2.0 + 0.5f64.ln());
    }
}
