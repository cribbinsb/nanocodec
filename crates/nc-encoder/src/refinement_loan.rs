//! The loan behind a planned refinement (`rate.refine`, `planned_refinement.rs`):
//! bits a refinement frame takes on top of its ordinary budget, scheduled
//! like a key's loan and repaid by slightly smaller inter budgets.
//!
//! A key borrows its planned budget and the frames of its repayment phase
//! pay it back (`RateController::scheduled_key_debt`); the debt correction
//! sees only the debt beyond that schedule. A refinement is the same kind of
//! deliberate, planned overspend, so it is accounted the same way: the loan
//! is scheduled debt (outside the correction's excess), and each later inter
//! frame's budget is cut by an equal share of it over `rate.refine_payback`.
//! The long-run rate therefore stays on the target line, and the frame's
//! other blocks are planned exactly as without the refinement.
//!
//! The refined blocks' cost is estimated per block by the reuse term
//! (`ReuseTerm::plan_refinement`) and scaled by [`RefinementLoan::cost_scale`],
//! learnt from what refinement frames really cost beyond their ordinary
//! prediction: so the plan is told the loan instead of mispredicting it.

/// Smallest and largest learnt ratio of a refinement's real to estimated
/// cost: the estimate is a high-rate approximation, never off by more than
/// this on real content, and the clamp keeps one odd frame from silencing or
/// unleashing every later refinement.
const COST_SCALE_RANGE: (f64, f64) = (0.05, 20.);
/// Weight of each refinement frame's measured cost ratio in the learnt scale
/// (geometric smoothing: refinements come about once a second).
const COST_SCALE_SMOOTHING: f64 = 0.5;
/// A refinement that spends more than its loan by this factor schedules only
/// this much; the rest is ordinary debt, repaid by the faster correction.
const LOAN_OVERRUN: f64 = 1.5;
/// Initial cost scale before any refinement has been measured: cautious, so
/// the first refinement's loan is not underestimated into a re-code.
const INITIAL_COST_SCALE: f64 = 2.;

use crate::params::RateParams;

/// Rate control's part of still-scene coding: the refinement loan
/// (`rate.refine`), the quiet run-up to a key (`rate.prekey_quiet`) and the
/// held key (`rate.key_hold_still`).
#[derive(Debug, Clone)]
pub(crate) struct StillRate {
    loan: Option<RefinementLoan>,
    /// `rate.refine_budgets` and `rate.refine_min_budgets`.
    budgets: f64,
    min_budgets: f64,
    /// Whether the current frame's budget was cut to repay a refinement.
    repaying: bool,
    /// The current frame's ordinary prediction and budget (without its loan).
    ordinary: (f64, f64),
    /// `rate.prekey_quiet` in frames (0: off).
    prekey_quiet_frames: f64,
    /// `rate.key_hold_still` (buffers; 0: off).
    pub key_hold_buffers: f64,
}

impl StillRate {
    pub fn new(params: &RateParams, fps: f64) -> Self {
        Self {
            loan: (params.refine >= 0.5)
                .then(|| RefinementLoan::new(f64::from(params.refine_payback) * fps)),
            budgets: f64::from(params.refine_budgets),
            min_budgets: f64::from(params.refine_min_budgets),
            repaying: false,
            ordinary: (0., 0.),
            prekey_quiet_frames: f64::from(params.prekey_quiet) * fps,
            key_hold_buffers: f64::from(params.key_hold_still),
        }
    }

    /// The budget cut of the next inter frame that repays a refinement.
    pub fn repayment(&self) -> f64 {
        self.loan.as_ref().map_or(0., RefinementLoan::repayment)
    }

    /// Refinement debt scheduled for repayment.
    pub fn outstanding(&self) -> f64 {
        self.loan.as_ref().map_or(0., RefinementLoan::outstanding)
    }

    /// A cadence `offset` with no base boost `to_key` frames before a
    /// scheduled key, within `rate.prekey_quiet`.
    pub fn quieted(&self, to_key: Option<u64>, offset: f64) -> f64 {
        match to_key {
            Some(frames) if offset < 0. && (frames as f64) <= self.prekey_quiet_frames => 0.,
            _ => offset,
        }
    }

    /// The (smallest worthwhile, largest) estimate of a refinement, in the
    /// reuse term's estimated bits, for a uniform budget `uniform` and
    /// `room` bits the strict buffer leaves beyond the frame's own budget;
    /// `None` when off or without room.
    pub fn room(&self, uniform: f64, room: f64) -> Option<(f64, f64)> {
        let loan = self.loan.as_ref()?;
        let most = (self.budgets * uniform).min(room) / loan.cost_scale();
        let least = self.min_budgets * uniform / loan.cost_scale();
        (most > least).then_some((least, most))
    }

    /// Plan the next inter frame as a refinement estimated at `estimate` bits.
    pub fn plan(&mut self, estimate: f64) {
        if let Some(loan) = self.loan.as_mut() {
            loan.plan(estimate);
        }
    }

    /// The frame being selected (`key` or not) was predicted at `predicted`
    /// bits on a `budget`: returns its loan, which comes on top of both.
    pub fn on_select(&mut self, key: bool, predicted: f64, budget: f64) -> f64 {
        self.ordinary = (predicted, budget);
        self.repaying = !key && self.repayment() > 0.;
        let Some(loan) = self.loan.as_mut() else {
            return 0.;
        };
        if key {
            loan.cancel();
        }
        loan.planned()
    }

    /// Whether the frame awaiting feedback is a refinement.
    pub fn refining(&self) -> bool {
        self.loan.as_ref().is_some_and(|loan| loan.planned() > 0.)
    }

    /// The frame just coded took `bits`.
    pub fn settle(&mut self, bits: f64) {
        if let Some(loan) = self.loan.as_mut() {
            loan.settle(bits, self.ordinary, self.repaying);
        }
    }
}

/// Outstanding refinement loan and the cost model of refinements.
#[derive(Debug, Clone)]
pub(crate) struct RefinementLoan {
    /// Frames over which a loan is repaid (`rate.refine_payback` × fps).
    payback_frames: f64,
    /// Scheduled debt not yet repaid, in bits.
    outstanding: f64,
    /// The cut of each inter frame's budget that repays it.
    per_frame: f64,
    /// Loan planned for the frame being coded (0: not a refinement frame),
    /// and the estimate it came from.
    planned: f64,
    estimate: f64,
    /// Real refinement cost per estimated bit.
    cost_scale: f64,
}

impl RefinementLoan {
    pub fn new(payback_frames: f64) -> Self {
        Self {
            payback_frames: payback_frames.max(1.),
            outstanding: 0.,
            per_frame: 0.,
            planned: 0.,
            estimate: 0.,
            cost_scale: INITIAL_COST_SCALE,
        }
    }

    /// Real bits per estimated bit of a refinement.
    pub fn cost_scale(&self) -> f64 {
        self.cost_scale
    }

    /// Debt scheduled for repayment, outside the correction's excess.
    pub fn outstanding(&self) -> f64 {
        self.outstanding
    }

    /// The loan of the frame being coded (0 on an ordinary frame).
    pub fn planned(&self) -> f64 {
        self.planned
    }

    /// Plan a refinement whose blocks were estimated at `estimate` bits:
    /// the loan, `estimate` at the learnt cost scale.
    pub fn plan(&mut self, estimate: f64) -> f64 {
        self.estimate = estimate.max(0.);
        self.planned = self.estimate * self.cost_scale;
        self.planned
    }

    /// The budget cut of the next inter frame that repays the loan.
    pub fn repayment(&self) -> f64 {
        self.per_frame.min(self.outstanding)
    }

    /// The frame just coded took `bits`; `ordinary` is what its ordinary
    /// plan predicted and `budget` its ordinary budget. A refinement frame
    /// learns its real cost and schedules its loan; any other inter frame
    /// repaid its share (`repaying`: its budget was cut).
    pub fn settle(&mut self, bits: f64, ordinary: (f64, f64), repaying: bool) {
        if repaying {
            self.outstanding = (self.outstanding - self.per_frame).max(0.);
        }
        if self.planned <= 0. {
            return;
        }
        let (predicted, budget) = ordinary;
        let spent = (bits - predicted).max(0.);
        if self.estimate > 0. {
            let ratio = (spent / self.estimate).clamp(COST_SCALE_RANGE.0, COST_SCALE_RANGE.1);
            self.cost_scale = (self.cost_scale.ln() * (1. - COST_SCALE_SMOOTHING)
                + ratio.ln() * COST_SCALE_SMOOTHING)
                .exp();
        }
        let borrowed = (bits - budget).clamp(0., self.planned * LOAN_OVERRUN);
        self.outstanding += borrowed;
        self.per_frame = self.outstanding / self.payback_frames;
        self.planned = 0.;
        self.estimate = 0.;
    }

    /// The frame was not coded after all (a re-code plans it again).
    pub fn cancel(&mut self) {
        self.planned = 0.;
        self.estimate = 0.;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_loan_is_scheduled_and_repaid_over_the_payback() {
        let mut loan = RefinementLoan::new(10.);
        assert_eq!(loan.plan(2000.), 2000. * INITIAL_COST_SCALE);
        // Ordinary part predicted 500, budget 600; the frame took 1600.
        loan.settle(1600., (500., 600.), false);
        assert_eq!(loan.outstanding(), 1000.);
        assert_eq!(loan.repayment(), 100.);
        for _ in 0..10 {
            loan.settle(400., (0., 0.), true);
        }
        assert_eq!(loan.outstanding(), 0.);
        assert_eq!(loan.repayment(), 0.);
    }

    #[test]
    fn the_cost_scale_follows_real_refinement_costs() {
        let mut loan = RefinementLoan::new(10.);
        loan.plan(1000.);
        // Refinement cost 1000 beyond the ordinary prediction: ratio 1.
        loan.settle(1500., (500., 500.), false);
        let learnt = loan.cost_scale();
        assert!(learnt < INITIAL_COST_SCALE && learnt > 1.);
    }

    #[test]
    fn an_overrun_beyond_the_loan_is_left_to_the_correction() {
        let mut loan = RefinementLoan::new(10.);
        let planned = loan.plan(1000.);
        loan.settle(10_000., (500., 500.), false);
        assert_eq!(loan.outstanding(), planned * LOAN_OVERRUN);
    }
}
