//! The inputs of the stop network (`stop_value.rs`): what the RD partition
//! search knows at a planned inter node once its planned candidate is
//! coded, as bytes.
//!
//! Where it fits: the post-trial stop (`rd_split.rs` `FirstTrialStop`)
//! skips a node's other candidates where a score of the planned
//! candidate's coding is above 0. Its linear terms read ten features; the
//! network reads all the causal signal the search holds there: the planned
//! coding's cost, bits, distortion, skip flag, levels, coded blocks and
//! largest transform; the node's size, plan and planner margin; the depth
//! context; λ; and the planner's own evidence at the node (the eleven trial
//! bytes, `trial_inputs.rs`) where it weighed the node both ways.
//!
//! Why bytes: as the trial bytes, every value is formed with exact IEEE
//! operations and the integer logarithm, so the network decides alike on
//! every platform; and this function is the only implementation of the
//! inputs: the label dump records its bytes (`labels.rs`
//! `NodeLabel::stop_inputs`) and training reads them.

use crate::backend::{NodeTrialParts, NodeTrialSummary};
use crate::depth_context::NodeContext;
use crate::trial_inputs::{
    INPUT_MAX, MARGIN_SCALE, MIN_LOG2, SIZE_STEP, TRIAL_INPUT_COUNT, TrialInputs, input, log2_steps,
};

/// Bytes the stop features take before the trial bytes.
const OWN_INPUTS: usize = 14;
/// Inputs per node: the stop features, the trial bytes and their flag.
pub(crate) const STOP_INPUT_COUNT: usize = OWN_INPUTS + TRIAL_INPUT_COUNT + 1;

/// Samples of the 8x8 block the per-area counts are taken per.
const UNIT_AREA: f32 = 64.0;
/// Depth context bytes: this value stands for "same size as the node", one
/// level of leaf size is this step (as the trial value network's context).
const CONTEXT_ZERO: i32 = 64;
const CONTEXT_STEP: i32 = 16;
/// Input steps per level of the largest transform's log2 side.
const TX_STEP: i32 = 16;
/// λ in half steps (eight per octave): λ up to 2^15 stays in range.
const LAMBDA_DIVISOR: i32 = 2;

/// What the planned candidate's coding left at a node, and what the search
/// knew of the node before it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StopEvidence {
    /// The node's log2 size and area in samples.
    pub log2: u8,
    pub area: f32,
    pub lambda: f32,
    /// The planned candidate's coded cost.
    pub cost: f32,
    /// Its parts, where the backend splits them.
    pub parts: Option<NodeTrialParts>,
    /// It coded as a skip CU.
    pub skip: bool,
    /// What its coding holds, where the backend reports it.
    pub summary: Option<NodeTrialSummary>,
    pub planned_split: bool,
    /// The planner's margin at the node (at least 1; 1 unknown).
    pub margin: f32,
    pub context: NodeContext,
    /// The planner's evidence, where it weighed the node both ways.
    pub trial: Option<TrialInputs>,
}

/// The stop network's inputs at one node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StopInputs(pub [u8; STOP_INPUT_COUNT]);

impl StopInputs {
    pub fn new(e: &StopEvidence) -> Self {
        let lambda = e.lambda.max(f32::MIN_POSITIVE);
        let per_lambda = |v: f32| log2_steps(1.0 + v / e.area / lambda);
        let per_unit = |v: f32| log2_steps(1.0 + v * UNIT_AREA / e.area);
        let (bits, distortion) = e
            .parts
            .map_or((0, 0), |p| (per_unit(p.bits), per_lambda(p.distortion)));
        let summary = e.summary.unwrap_or_default();
        let flag = |on: bool| if on { INPUT_MAX } else { 0 };
        let context = |level: i8| CONTEXT_ZERO + CONTEXT_STEP * i32::from(level);
        let own: [i32; OWN_INPUTS] = [
            SIZE_STEP * (i32::from(e.log2) - MIN_LOG2),
            flag(e.planned_split),
            MARGIN_SCALE * log2_steps(e.margin.max(1.0)),
            per_lambda(e.cost),
            bits,
            distortion,
            flag(e.parts.is_some()),
            flag(e.skip),
            per_unit(f32::from(summary.nonzero_levels)),
            per_unit(f32::from(summary.coded_blocks)),
            TX_STEP * i32::from(summary.max_luma_tx_log2),
            context(e.context.neighbours),
            context(e.context.co_located),
            log2_steps(lambda) / LAMBDA_DIVISOR,
        ];
        let mut bytes = [0u8; STOP_INPUT_COUNT];
        for (byte, value) in bytes.iter_mut().zip(own) {
            *byte = input(value);
        }
        if let Some(trial) = e.trial {
            bytes[OWN_INPUTS..OWN_INPUTS + TRIAL_INPUT_COUNT].copy_from_slice(&trial.0);
            bytes[STOP_INPUT_COUNT - 1] = input(INPUT_MAX);
        }
        Self(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(cost: f32, trial: Option<TrialInputs>) -> StopEvidence {
        StopEvidence {
            log2: 5,
            area: 1024.0,
            lambda: 200.0,
            cost,
            parts: Some(NodeTrialParts {
                distortion: cost / 2.0,
                bits: cost / 400.0,
                lambda: 200.0,
            }),
            skip: false,
            summary: Some(NodeTrialSummary {
                nonzero_levels: 40,
                coded_blocks: 6,
                max_luma_tx_log2: 4,
            }),
            planned_split: true,
            margin: 1.5,
            context: NodeContext {
                neighbours: -1,
                co_located: 9,
            },
            trial,
        }
    }

    #[test]
    fn inputs_stay_in_range_and_follow_the_coding() {
        let cheap = StopInputs::new(&evidence(1_000.0, None));
        let dear = StopInputs::new(&evidence(1_000_000.0, Some(TrialInputs([127; 11]))));
        for inputs in [cheap, dear] {
            assert!(inputs.0.iter().all(|&v| i32::from(v) <= INPUT_MAX));
        }
        assert!(dear.0[3] > cheap.0[3] && dear.0[4] > cheap.0[4] && dear.0[5] > cheap.0[5]);
        assert_eq!(cheap.0[11], 48);
        assert_eq!(cheap.0[12], 127);
        assert_eq!(cheap.0[STOP_INPUT_COUNT - 1], 0);
        assert_eq!(dear.0[STOP_INPUT_COUNT - 1], 127);
        assert_eq!(dear.0[OWN_INPUTS], 127);
    }
}
