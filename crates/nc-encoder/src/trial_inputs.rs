//! The inputs of the trial value network: what the planner knew about a
//! node when it weighed its leaf against its quad split, as bytes.
//!
//! Where it fits: the RD partition search (`rd_split.rs`) codes a node both
//! ways only where a trial is worth its time. The value network scores that
//! from the planner's own evidence at the node ([`SplitEvidence`], the
//! planner's margin and its choice), so the model needs no measurement of
//! its own (docs/analysis/learned-search-2026-11-22.md, section 4.1: nine
//! such features keep what twenty-six did).
//!
//! Why bytes, and why here: the network runs in integer arithmetic so that
//! every platform decides alike, and its inputs must then be integers that
//! every platform computes alike. Ratios are formed with IEEE division and
//! multiplication only (exact and identical everywhere); the logarithm is
//! an integer one (leading zeros and a sixteen-entry table), never the
//! platform's `ln`. This function is the only implementation of the
//! features: the label dump records its bytes and training reads them, so
//! a trained model sees in the encoder exactly what it saw in training.

pub(crate) use nc_analysis::net_inputs::{INPUT_MAX, input, log2_steps};

use crate::split_selector::SplitEvidence;

/// Inputs per node.
pub(crate) const TRIAL_INPUT_COUNT: usize = 11;
/// Smallest node log2 size the inputs distinguish (8x8).
pub(crate) const MIN_LOG2: i32 = 3;
/// Input steps per level of node size.
pub(crate) const SIZE_STEP: i32 = 32;
/// Input steps per sixteenth of an octave of the planner's margin: the
/// margin's useful range is 1 to about 4.
pub(crate) const MARGIN_SCALE: i32 = 4;
/// The quadrants' variance gain is at most 1 (separate means cannot lose);
/// one sixteenth of an octave below 1 takes two input steps off the top.
const QUADRANT_GAIN_SCALE: i32 = 2;
/// The motion search's split / whole SAD ratio sits between about 1/4 and
/// sqrt(2): this input value stands for a ratio of 1.
const SAD_RATIO_ZERO: i32 = 96;
/// Input steps per sixteenth of an octave of that ratio.
const SAD_RATIO_SCALE: i32 = 3;
/// Regularises variance ratios of near-flat residuals (per sample), as the
/// split selector's.
const VARIANCE_FLOOR: f32 = 1e-3;

/// The value network's inputs at one node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct TrialInputs(pub [u8; TRIAL_INPUT_COUNT]);

impl TrialInputs {
    /// From the planner's evidence at a node, its margin there (the
    /// rejected choice's cost over the chosen one's, at least 1) and
    /// whether it chose the split.
    pub fn new(evidence: &SplitEvidence, margin: f32, planned_split: bool) -> Self {
        let log2_area = 2 * u32::from(evidence.log2);
        let area = (1_u32 << log2_area) as f32;
        let lambda = evidence.lambda.max(f32::MIN_POSITIVE);
        let node_var = evidence.leaf.n_variance_pow2(log2_area) as f32;
        let mean_energy = (evidence.leaf.sse as f32 - node_var).max(0.0);
        let per_lambda = |energy: f32| log2_steps(1.0 + energy / area / lambda);
        let (quadrant_gain, imbalance) = evidence.quadrants.map_or((0, 0), |quadrants| {
            let log2_quarter = log2_area - 2;
            let quarter = (1_u32 << log2_quarter) as f32;
            let vars = quadrants.map(|q| q.n_variance_pow2(log2_quarter) as f32);
            let sum: f32 = vars.iter().sum();
            let (low, high) = vars
                .iter()
                .fold((f32::MAX, 0.0f32), |(lo, hi), &v| (lo.min(v), hi.max(v)));
            let per_sample = quarter;
            (
                INPUT_MAX
                    + QUADRANT_GAIN_SCALE
                        * log2_steps(
                            (sum / area + VARIANCE_FLOOR) / (node_var / area + VARIANCE_FLOOR),
                        ),
                log2_steps((high / per_sample + 1.0) / (low / per_sample + 1.0)),
            )
        });
        let sad_ratio = evidence
            .search_sads
            .map_or(SAD_RATIO_ZERO, |(whole, split)| {
                SAD_RATIO_ZERO
                    + SAD_RATIO_SCALE * log2_steps((split as f32 + 1.0) / (whole as f32 + 1.0))
            });
        let flag = |on: bool| if on { INPUT_MAX } else { 0 };
        Self([
            input(SIZE_STEP * (i32::from(evidence.log2) - MIN_LOG2)),
            input(flag(planned_split)),
            input(MARGIN_SCALE * log2_steps(margin.max(1.0))),
            input(per_lambda(node_var)),
            input(per_lambda(mean_energy)),
            input(quadrant_gain),
            input(imbalance),
            input(log2_steps(1.0 + evidence.vector_distance as f32)),
            input(sad_ratio),
            input(flag(evidence.quadrants.is_some())),
            input(flag(evidence.search_sads.is_some())),
        ])
    }
}

#[cfg(test)]
mod tests {
    use nc_dsp::Variance;

    use super::*;

    #[test]
    fn inputs_stay_in_range_and_follow_the_evidence() {
        let evidence = |sse, quadrants, search_sads| SplitEvidence {
            log2: 5,
            lambda: 100.0,
            leaf: Variance { sse, sum: 0 },
            quadrants,
            vector_distance: 6,
            search_sads,
        };
        let flat = TrialInputs::new(&evidence(0, None, None), 1.0, false);
        let busy_quadrants = Some([
            Variance {
                sse: 60_000,
                sum: 0,
            },
            Variance::default(),
            Variance::default(),
            Variance::default(),
        ]);
        let busy = TrialInputs::new(
            &evidence(4_000_000, busy_quadrants, Some((900, 300))),
            3.0,
            true,
        );
        for inputs in [flat, busy] {
            assert!(inputs.0.iter().all(|&v| i32::from(v) <= INPUT_MAX));
        }
        assert_eq!(flat.0[0], busy.0[0]);
        assert!(busy.0[1] > flat.0[1] && busy.0[2] > flat.0[2] && busy.0[3] > flat.0[3]);
        assert_eq!((flat.0[9], flat.0[10]), (0, 0));
        assert_eq!(
            (i32::from(busy.0[9]), i32::from(busy.0[10])),
            (INPUT_MAX, INPUT_MAX)
        );
        assert!(i32::from(busy.0[8]) < SAD_RATIO_ZERO);
    }
}
