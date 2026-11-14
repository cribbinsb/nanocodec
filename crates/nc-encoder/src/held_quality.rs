//! The quality a still picture holds, and the price of refining it
//! (`rate.held_quality`).
//!
//! On a still scene almost every block is a copy of the reference, so an
//! inter frame at the level costs almost nothing and rate control lowers
//! the level frame after frame. Once the level (or a cadence base frame
//! below it) is finer than the step the still picture was last coded at,
//! the next frame refines every still block at once: a burst of 10-40 frame
//! budgets that the per-role bits model, which knows only the frame's step,
//! mispredicts by an order of magnitude (AV1 Default, 600 f, still points:
//! bursts 13-22x their prediction; with the cadence flat the bursts moved to
//! ordinary frames, so the drift, not the cadence, causes them;
//! `docs/analysis/persistent-region-coding-2026-11-14.md` §4).
//!
//! This model keeps the held step `H` (the finest step any frame coded the
//! still picture at since the key: a frame finer than `H` refined it) and
//! prices a frame at step `s < H` with the refinement it triggers,
//!
//! `coefficient · still share · log2(H / s)^exponent`
//!
//! learned online from the frames that refined (their bits beyond the
//! per-role model's prediction). Rate control adds it to its plan, so the
//! level stops where the refinement fits the budget instead of drifting
//! into a burst it pays back later; and the per-role model learns only from
//! frames that did not refine, so it stays the bits of a copy frame.
//!
//! The same idea was probed for base frames only (branch
//! `feature/burst-predictor`, a two-sided base-frame model): it damped the
//! limit cycle and fixed the misprediction but its keys starved; the key
//! side is now handled by `rate.key_hold_still` and `rate.period_ceiling`.

/// Still share below which a frame holds nothing: a moving scene's frames
/// code their blocks anew, so no step is held.
const MIN_STILL_SHARE: f64 = 0.5;
/// Octaves finer than the held step below which a frame does not count as
/// refining: neighbouring quantizers differ by about this much.
const MIN_REFINE_OCTAVES: f64 = 0.05;
/// Least share of a refining frame's bits the per-role model is credited
/// with when its prediction is larger than the frame (no negative samples).
const MIN_REFINE_SAMPLE: f64 = 0.;

/// The held-quality model of one controller.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct HeldQuality {
    /// `rate.held_exponent`: growth of the refinement with its octaves.
    exponent: f64,
    /// Weight of each new observation in the coefficient.
    smoothing: f64,
    /// The step the still picture holds (`None`: not still, or no frame yet).
    held: Option<f64>,
    /// Bits per unit of `still · octaves^exponent`, once a frame refined.
    coefficient: Option<f64>,
}

impl HeldQuality {
    pub fn new(exponent: f64, smoothing: f64) -> Self {
        Self {
            exponent,
            smoothing,
            held: None,
            coefficient: None,
        }
    }

    /// Octaves a frame at `step` is finer than the held step (0 when not).
    fn octaves(&self, step: f64) -> f64 {
        self.held.map_or(0., |held| (held / step).log2()).max(0.)
    }

    /// Whether a frame at `step` with `still` share refines the still picture.
    pub fn refines(&self, step: f64, still: f64) -> bool {
        still >= MIN_STILL_SHARE && self.octaves(step) > MIN_REFINE_OCTAVES
    }

    /// The predicted bits a frame at `step` with `still` share spends
    /// refining the still picture (0 before a refinement was seen).
    pub fn refine_bits(&self, step: f64, still: f64) -> f64 {
        match self.coefficient {
            Some(coefficient) if self.refines(step, still) => {
                coefficient * still * self.octaves(step).powf(self.exponent)
            }
            _ => 0.,
        }
    }

    /// A key at `step` replaces the picture: it holds that step.
    pub fn seed_key(&mut self, step: f64) {
        self.held = Some(step);
    }

    /// An inter frame of `bits` was coded at `step` with `still` share; the
    /// per-role model predicted `ordinary` bits for it without refinement.
    /// Learns the coefficient from a refining frame, then moves the held
    /// step (finer when it refined; forgotten when the scene moves).
    pub fn observe(&mut self, step: f64, still: f64, bits: f64, ordinary: f64) {
        if self.refines(step, still) {
            let units = still * self.octaves(step).powf(self.exponent);
            let sample = ((bits - ordinary) / units).max(MIN_REFINE_SAMPLE);
            self.coefficient = Some(
                self.coefficient
                    .map_or(sample, |c| c + self.smoothing * (sample - c)),
            );
        }
        self.held = if still < MIN_STILL_SHARE {
            None
        } else {
            Some(self.held.map_or(step, |held| held.min(step)))
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_finer_than_the_held_step_is_priced() {
        let mut held = HeldQuality::new(1., 0.5);
        held.seed_key(40.);
        assert_eq!(held.refine_bits(20., 1.), 0.);
        // One octave finer, fully still: 9000 bits beyond the prediction.
        held.observe(20., 1., 10_000., 1000.);
        assert_eq!(held.refine_bits(20., 1.), 0.);
        held.seed_key(40.);
        assert_eq!(held.refine_bits(20., 1.), 9000.);
        assert_eq!(held.refine_bits(10., 0.5), 9000.);
        assert_eq!(held.refine_bits(80., 1.), 0.);
    }

    #[test]
    fn a_moving_scene_holds_nothing() {
        let mut held = HeldQuality::new(1., 0.5);
        held.seed_key(40.);
        held.observe(30., 0.2, 1000., 1000.);
        assert!(!held.refines(10., 1.));
    }
}
