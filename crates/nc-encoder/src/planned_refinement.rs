//! When to plan a refinement of persistent still blocks (`rate.refine`).
//!
//! On still scenes most of our quality used to come from QP-cadence base
//! frames that the rate controller's limit cycle happened to land deep: they
//! refined the whole still picture at once and every later frame inherited
//! it (AV1 Default, 600 f, key every 10 s, still points: base bursts took
//! 59-82% of all bits and gave 84-127% of the still-region gain; the best
//! quarter, nearly all within 60 frames after a key, gave 50-87% of it with
//! a third of the burst bits; `docs/analysis/persistent-region-coding-2026-11-14.md`).
//! A planned refinement does that on purpose:
//!
//! - **When** (this module): at most once per `rate.refine_interval`, never
//!   within `rate.refine_horizon` of the next scheduled key (the key codes
//!   every block again), on flat inter frames only.
//! - **Where** (`ReuseTerm::plan_refinement`): still cells in order of
//!   forward value per bit, each at its reuse price `λ / (1 + R)`, above its
//!   source's noise.
//! - **Paid for** (`RefinementLoan`): a loan on top of the frame's ordinary
//!   budget, bounded by `rate.refine_budgets` and the strict buffer's room,
//!   repaid over `rate.refine_payback`; the frame's other blocks are planned
//!   as without it.
//!
//! The steady-state QP cadence gives no base boost while this is on
//! (`rate_control::spatial_aware`), so refinement comes from here alone.

use crate::params::RateParams;
use crate::rate_control::RateController;
use crate::reuse::{RefinementLimits, ReuseFrame, ReuseTerm};

/// Fewest frames a cell's source must have been still to be refined or held:
/// a shorter run predicts too little reuse to price it finer than the frame
/// (its price, `λ / (1 + R)`, is within 6 QP of it).
pub(crate) const HOLD_MIN_STILL_RUN: u8 = 4;

/// What the encoder does to still blocks between and in refinements: the
/// schedule (`rate.refine`), the hold (`rate.hold_still`, implied by
/// `rate.refine`) and the hold before a scheduled key (`rate.prekey_quiet`).
#[derive(Debug, Clone)]
pub(crate) struct StillPlan {
    schedule: Option<PlannedRefinement>,
    hold: bool,
    /// `rate.hold_recover`: hold only cells already as good as the level.
    hold_recover: bool,
    prekey_quiet_frames: u64,
    /// Whether the frame being planned is a refinement or holds its still
    /// blocks: its level is planned, and its model learnt, without its
    /// block offsets (the refined blocks' bits are a loan, the held blocks
    /// are copies).
    pub without_offsets: bool,
}

impl StillPlan {
    /// The plan of `params` at `fps`; `available`: the stream is flat,
    /// single-layer and codes block quantizers with the reuse term on.
    pub fn new(params: &RateParams, fps: f64, available: bool) -> Self {
        Self {
            schedule: PlannedRefinement::new(params, fps).filter(|_| available),
            hold: available && (params.hold_still >= 0.5 || params.refine >= 0.5),
            hold_recover: params.hold_recover >= 0.5,
            prekey_quiet_frames: if available {
                (f64::from(params.prekey_quiet) * fps).round() as u64
            } else {
                0
            },
            without_offsets: false,
        }
    }

    /// Plan inter frame `index` of `frame` (after `ReuseTerm::plan`; `flat`:
    /// it predicts from the frame before in a flat structure): a refinement
    /// when one is due and the rate controller has room for its loan, then
    /// the hold of every other still block when on or before a key.
    pub fn plan(
        &mut self,
        term: &mut ReuseTerm,
        frame: &ReuseFrame<'_>,
        controller: Option<&mut RateController>,
        (index, flat): (u64, bool),
    ) {
        self.without_offsets = false;
        if frame.key || !flat {
            return;
        }
        if let (Some(schedule), Some(controller)) = (self.schedule.as_mut(), controller)
            && schedule.due(index, frame.horizon)
            && let Some(room) = controller.refinement_room()
        {
            let estimate = term.plan_refinement(frame, schedule.limits(room));
            if estimate > 0. {
                controller.plan_refinement(estimate);
                schedule.refined(index);
                self.without_offsets = true;
            }
        }
        // Still blocks keep what they hold, but for those a planned
        // refinement refines: always with the hold, and before a scheduled
        // key (the key codes every block again).
        let quiet = frame
            .horizon
            .is_some_and(|to_key| to_key <= self.prekey_quiet_frames);
        if self.hold || quiet {
            term.hold_still(frame, HOLD_MIN_STILL_RUN, self.hold_recover);
            self.without_offsets = true;
        }
    }
}

/// The refinement schedule of one stream.
#[derive(Debug, Clone)]
pub(crate) struct PlannedRefinement {
    /// Frames between refinements at least.
    interval: u64,
    /// Frames before the next scheduled key within which none is planned.
    horizon: u64,
    /// `rate.refine_max_offset`.
    max_offset: f64,
    /// The last refined frame.
    last: Option<u64>,
}

impl PlannedRefinement {
    /// The schedule of `params` at `fps`, or `None` when `rate.refine` is off.
    pub fn new(params: &RateParams, fps: f64) -> Option<Self> {
        (params.refine >= 0.5).then(|| Self {
            interval: (f64::from(params.refine_interval) * fps).round() as u64,
            horizon: (f64::from(params.refine_horizon) * fps).round() as u64,
            max_offset: f64::from(params.refine_max_offset),
            last: None,
        })
    }

    /// Whether inter frame `index` may be refined, `to_key` frames before
    /// the next scheduled key (`None`: none scheduled).
    pub fn due(&self, index: u64, to_key: Option<u64>) -> bool {
        let spaced = self
            .last
            .is_none_or(|last| index.saturating_sub(last) >= self.interval);
        spaced && to_key.is_none_or(|frames| frames > self.horizon)
    }

    /// The limits of a refinement whose estimate lies within `(least, most)` bits.
    pub fn limits(&self, (least, most): (f64, f64)) -> RefinementLimits {
        RefinementLimits {
            least,
            most,
            max_offset: self.max_offset,
            min_run: HOLD_MIN_STILL_RUN,
        }
    }

    /// Inter frame `index` was planned as a refinement.
    pub fn refined(&mut self, index: u64) {
        self.last = Some(index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::ParamOverrides;

    fn schedule() -> PlannedRefinement {
        let mut overrides = ParamOverrides::default();
        overrides.set("rate.refine", 1.0);
        PlannedRefinement::new(&RateParams::from_overrides(&overrides), 30.).unwrap()
    }

    #[test]
    fn refinements_are_spaced_and_kept_away_from_the_next_key() {
        let mut s = schedule();
        assert!(s.due(10, Some(290)));
        s.refined(10);
        assert!(!s.due(20, Some(280)));
        assert!(s.due(40, Some(260)));
        assert!(!s.due(250, Some(50)));
        assert!(s.due(250, None));
    }

    #[test]
    fn off_by_default() {
        let params = RateParams::from_overrides(&ParamOverrides::default());
        assert!(PlannedRefinement::new(&params, 30.).is_none());
    }
}
