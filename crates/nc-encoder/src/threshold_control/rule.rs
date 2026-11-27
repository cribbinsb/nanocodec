//! The controller's update rule: which way a tool's evidence moves its
//! multiplier, and how far the multiplier may go.
//!
//! The plain rule compares a cycle's loss rate with the target's
//! hysteresis band, whatever the evidence's size above `adapt.min_runs`.
//! At 64 runs and a target of 0.15 one standard error of the rate is 0.045,
//! as wide as the band's lower half, so a gate at its right place still
//! steps about one cycle in four (the sub-pel gate's multiplier wandered
//! between its bounds in the 2026-11-17 traces). With `adapt.confidence`
//! the rule is a sequential test: it moves only when the rate is outside
//! the band by that many standard errors, holds as soon as it is inside it
//! by as many, and otherwise waits for more evidence, so clear cases still
//! move after one cycle and unclear ones stop moving on noise.

use super::evidence::Evidence;

/// Cycles' worth of `adapt.min_runs` an undecided test may gather before
/// it is restarted: evidence older than that describes other content.
pub(super) const PATIENCE: f32 = 8.0;

/// Which way a cycle's evidence moves a multiplier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Verdict {
    /// The cheap alternative loses less than the target: less work.
    Less,
    /// The evidence is within the hysteresis of the target.
    Hold,
    /// The cheap alternative loses more than the target: more work.
    More,
    /// Not decided yet: the evidence is kept and added to.
    Wait,
}

/// A tool's test of its evidence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Rule {
    /// The loss rate the preset accepts.
    pub target: f32,
    /// The factor around the target within which the multiplier holds.
    pub hysteresis: f32,
    /// Standard errors by which the rate must clear the hysteresis band
    /// (`adapt.confidence`; 0: the plain rule).
    pub confidence: f32,
    /// Observations after which an undecided test holds and restarts.
    pub patience: f32,
}

impl Rule {
    pub(super) fn judge(self, evidence: Evidence) -> Verdict {
        let rate = evidence.rate();
        let (low, high) = (self.target / self.hysteresis, self.target * self.hysteresis);
        if self.confidence <= 0.0 {
            return if rate < low {
                Verdict::Less
            } else if rate > high {
                Verdict::More
            } else {
                Verdict::Hold
            };
        }
        // The standard error of a share of `samples` observations at the
        // target rate. Weighted evidence is treated alike: an approximation
        // on the cautious side only while its weights are similar.
        let samples = evidence.samples.max(1) as f32;
        let margin = self.confidence * (self.target * (1.0 - self.target) / samples).sqrt();
        if rate + margin < low {
            Verdict::Less
        } else if rate - margin > high {
            Verdict::More
        } else if (rate - margin >= low && rate + margin <= high) || samples >= self.patience {
            Verdict::Hold
        } else {
            Verdict::Wait
        }
    }
}

/// How a multiplier moves: the factor of one step and its bounds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Travel {
    pub step: f32,
    pub min: f32,
    pub max: f32,
}

impl Travel {
    /// `scale` after one step by `verdict` (above 1: the tool skips more).
    pub(super) fn moved(self, scale: f32, verdict: Verdict) -> f32 {
        let stepped = match verdict {
            Verdict::Less => scale * self.step,
            Verdict::Hold | Verdict::Wait => scale,
            Verdict::More => scale / self.step,
        };
        stepped.clamp(self.min, self.max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RULE: Rule = Rule {
        target: 0.2,
        hysteresis: 1.5,
        confidence: 0.0,
        patience: 800.0,
    };

    fn evidence(wins: u64, runs: u64) -> Evidence {
        Evidence {
            runs,
            wins,
            samples: runs as u32,
        }
    }

    #[test]
    fn the_multiplier_holds_within_the_hysteresis_of_the_target() {
        assert_eq!(RULE.judge(evidence(10, 100)), Verdict::Less);
        assert_eq!(RULE.judge(evidence(14, 100)), Verdict::Hold);
        assert_eq!(RULE.judge(evidence(29, 100)), Verdict::Hold);
        assert_eq!(RULE.judge(evidence(31, 100)), Verdict::More);
    }

    #[test]
    fn a_confident_rule_waits_for_evidence_that_clears_the_band() {
        let rule = Rule {
            confidence: 2.0,
            ..RULE
        };
        // 100 samples: one standard error is 0.04, the band 0.133..0.3.
        assert_eq!(rule.judge(evidence(10, 100)), Verdict::Wait);
        assert_eq!(rule.judge(evidence(4, 100)), Verdict::Less);
        assert_eq!(rule.judge(evidence(31, 100)), Verdict::Wait);
        assert_eq!(rule.judge(evidence(40, 100)), Verdict::More);
        // More evidence decides: 400 samples halve the standard error.
        assert_eq!(rule.judge(evidence(32, 400)), Verdict::Less);
        assert_eq!(rule.judge(evidence(80, 400)), Verdict::Hold);
        // An undecided test ends at its patience.
        assert_eq!(rule.judge(evidence(104, 799)), Verdict::Wait);
        assert_eq!(rule.judge(evidence(104, 800)), Verdict::Hold);
    }

    #[test]
    fn a_step_stays_inside_the_bounds() {
        let travel = Travel {
            step: 1.25,
            min: 0.5,
            max: 4.0,
        };
        assert_eq!(travel.moved(1.0, Verdict::Less), 1.25);
        assert_eq!(travel.moved(1.0, Verdict::More), 0.8);
        assert_eq!(travel.moved(1.0, Verdict::Hold), 1.0);
        assert_eq!(travel.moved(1.0, Verdict::Wait), 1.0);
        assert_eq!(travel.moved(3.9, Verdict::Less), 4.0);
        assert_eq!(travel.moved(0.55, Verdict::More), 0.5);
    }
}
