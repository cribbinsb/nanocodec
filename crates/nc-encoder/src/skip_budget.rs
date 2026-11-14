//! The skip-aware budget (`rate.skip_budget`): an inter frame's budget
//! shrinks with the share of the picture the noise skip
//! (`decision.noise_skip`, `noise_skip.rs`) took in the frame before, and
//! what it leaves unspent is not credited to later frames.
//!
//! Why: the noise skip stops coding sensor noise on still blocks, but rate
//! control then spends the same target on what is left, at an ever finer
//! quantizer (a still webcam at 1 Mbps: the AV1 inter quantizer falls from
//! qindex 102 to 45, the rate stays at the target). The target is a ceiling
//! (owner, 2026-11-08): a picture that is mostly noise-skipped needs only
//! the bits of its changing part.
//!
//! Why it is safe: the noise skip's test is against the current source, so
//! where content changes (an object, a light change) blocks stop passing
//! and the share falls in the next frame, restoring the budget; the
//! smallest scale ([`SkipBudget::scale`]'s floor) keeps a mostly still
//! frame able to refine its moving part, and starved streams scale less.
//! What a scaled frame leaves unspent, up to the share its scale took, is
//! forgiven rather than banked for a later burst ([`forgiven`]); bits it
//! spends above its scaled allowance are ordinary debt, never extra debt.

use crate::bpp_curve;
use crate::params::RateParams;

/// Noise-skipped share of the previous inter frame and the scale law.
#[derive(Debug, Clone)]
pub(crate) struct SkipBudget {
    /// `rate.skip_budget`: budget lost per unit of skipped share on a
    /// generous stream ([`bpp_curve::GENEROUS_BPP`]); 0 = off.
    strength: f64,
    /// `rate.skip_budget_starved`: the same on a starved stream
    /// ([`bpp_curve::STARVED_BPP`]), where the skipped area's bits still
    /// buy visible refinement of the rest (a still webcam at 300 kbps,
    /// 854x480: strength 1 cost H.264 0.9 dB against the clean scene, 0.5
    /// cost 0.2 dB).
    strength_starved: f64,
    /// `rate.skip_budget_floor`: the smallest budget scale.
    floor: f64,
    /// Share of the last inter frame's luma area the noise skip took.
    skipped: f64,
}

impl SkipBudget {
    pub(crate) fn new(params: &RateParams) -> Self {
        Self {
            strength: f64::from(params.skip_budget),
            strength_starved: f64::from(params.skip_budget_starved),
            floor: f64::from(params.skip_budget_floor),
            skipped: 0.,
        }
    }

    /// Record the share (0..=1) of an inter frame's area the noise skip
    /// took.
    pub(crate) fn observe(&mut self, skipped_share: f64) {
        self.skipped = skipped_share.clamp(0., 1.);
    }

    /// Forget the share (a key: its successor starts unskipped).
    pub(crate) fn reset(&mut self) {
        self.skipped = 0.;
    }

    /// The next inter frame's budget scale at `bpp` bits per pixel: 1 when
    /// off.
    pub(crate) fn scale(&self, bpp: f64) -> f64 {
        if self.strength > 0. {
            let strength = bpp_curve::interpolate(bpp, self.strength_starved, self.strength);
            (1. - strength * self.skipped).max(self.floor)
        } else {
            1.
        }
    }
}

/// The part of a frame's underspend that is forgiven rather than
/// credited, for a frame whose budget would have been `unscaled` and was
/// planned at skip-budget `scale`: what it left of `unscaled`, up to the
/// part the scale took away.
///
/// Measured against the frame's own unscaled plan, not the uniform budget:
/// the plan already holds the repayment of a key's loan and of any debt, so
/// forgiving underspend against the uniform budget forgave that repayment
/// and ran streams over target (a static clip at 600 frames, AV1, 2.5%
/// over; a noisy still at 150 frames, H.264, 11% over). Bits spent above
/// the scaled plan are ordinary debt.
pub(crate) fn forgiven(bits: f64, unscaled: f64, scale: f64) -> f64 {
    (unscaled - bits).clamp(0., unscaled * (1. - scale))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A generous stream: the floor is `rate.skip_budget_floor`.
    const GENEROUS: f64 = 1.;

    fn budget(strength: f32, floor: f32) -> SkipBudget {
        let mut params = crate::ParamOverrides::default();
        params.set("rate.skip_budget", strength);
        params.set("rate.skip_budget_floor", floor);
        params.set("rate.skip_budget_starved", 0.5);
        SkipBudget::new(&RateParams::from_overrides(&params))
    }

    #[test]
    fn the_budget_follows_the_unskipped_share_down_to_the_floor() {
        let mut b = budget(1., 0.1);
        assert_eq!(b.scale(GENEROUS), 1.);
        b.observe(0.6);
        assert!((b.scale(GENEROUS) - 0.4).abs() < 1e-6);
        b.observe(1.);
        assert!((b.scale(GENEROUS) - 0.1).abs() < 1e-6);
        assert!(
            (b.scale(0.001) - 0.5).abs() < 1e-6,
            "a starved stream keeps more"
        );
        b.reset();
        assert_eq!(b.scale(GENEROUS), 1.);
    }

    #[test]
    fn only_an_actual_underspend_is_forgiven() {
        // Scale 0.5 of a 100-bit allowance: up to 50 bits are forgiven.
        assert_eq!(forgiven(20., 100., 0.5), 50.);
        assert_eq!(forgiven(80., 100., 0.5), 20.);
        assert_eq!(
            forgiven(130., 100., 0.5),
            0.,
            "an overspend is ordinary debt"
        );
        assert_eq!(forgiven(0., 100., 1.), 0., "off forgives nothing");
    }

    #[test]
    fn off_keeps_the_whole_budget() {
        let mut b = budget(0., 0.1);
        b.observe(1.);
        assert_eq!(b.scale(GENEROUS), 1.);
    }
}
