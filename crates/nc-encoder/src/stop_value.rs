//! The stop network: a learned term of the post-trial stop's score
//! (`rd_split.rs` `FirstTrialStop`, `planner.rd_split_stop_net_*`).
//!
//! Where it fits: once a planned inter node's planned candidate is coded,
//! the stop skips the node's other candidates where its score is above 0.
//! The score's linear terms (`planner.rd_split_stop_*`) read ten features of
//! that coding. This network reads all the search knows there
//! (`stop_inputs.rs`) and adds `Σ_j readout_j · h_j − weight · value`,
//! where `h_j` are its last hidden activations in [0, 1] and `value` its
//! predicted log value of the alternatives (the linear terms' bias is the
//! constant).
//!
//! Why this form (owner, 2026-11-22: "today's heuristic is a degenerate
//! case of the network"): the output terms are registered parameters, all 0
//! by default, so the stop is exactly the linear one (today's search where
//! the linear terms are 0 too) and the network is not even run. The hidden
//! layers are fitted offline to what the alternatives' coding gained and
//! cost (`eval/learned/stop_train.py`, on the label dump's own input bytes);
//! the output terms are tuned on real encodes (`nceval.tune`), so the
//! offline fit only supplies features, and a tuner warm-started at 0 moves
//! away from the linear stop only where encodes measure a gain.
//!
//! How it runs: two clipped-`ReLU` layers and a linear value output in
//! integer arithmetic (`nc_dsp` `nn_dense_*`, exact on every tier); the
//! generated table `stop_value_weights.rs` records its training.

use nc_dsp::Dsp;

use crate::params::PlannerParams;
use crate::stop_inputs::{STOP_INPUT_COUNT, StopInputs};
use crate::stop_value_weights as net;

const _: () = assert!(net::INPUT_COUNT == STOP_INPUT_COUNT);

/// Last hidden activations the score reads (`planner.rd_split_stop_net_h*`).
pub(crate) const READOUTS: usize = net::HIDDEN_2;
/// The activations' largest value: 1.0 of a readout's unit.
const ACTIVATION_ONE: f32 = nc_dsp::NN_ACTIVATION_MAX as f32;

/// The network's output terms in the stop score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct StopNetwork {
    weight: f32,
    readout: [f32; READOUTS],
}

impl StopNetwork {
    /// The terms from `params`, `None` while they are all 0.
    pub fn of(params: &PlannerParams) -> Option<Self> {
        let readout = [
            params.rd_split_stop_net_h0,
            params.rd_split_stop_net_h1,
            params.rd_split_stop_net_h2,
            params.rd_split_stop_net_h3,
            params.rd_split_stop_net_h4,
            params.rd_split_stop_net_h5,
            params.rd_split_stop_net_h6,
            params.rd_split_stop_net_h7,
        ];
        (params.rd_split_stop_net_weight != 0.0 || readout.iter().any(|&r| r != 0.0)).then_some(
            Self {
                weight: params.rd_split_stop_net_weight,
                readout,
            },
        )
    }

    /// The terms' contribution to the stop score at `inputs`.
    pub fn score(&self, dsp: &Dsp, inputs: &StopInputs) -> f32 {
        let (value, hidden) = forward(dsp, inputs);
        let value = value as f32 / net::OUTPUT_ONE as f32;
        self.readout
            .iter()
            .zip(hidden)
            .fold(-self.weight * value, |sum, (r, h)| {
                sum + r * (f32::from(h) / ACTIVATION_ONE)
            })
    }
}

/// The network's integer value output (`net::OUTPUT_ONE` per unit) and its
/// last hidden activations.
fn forward(dsp: &Dsp, inputs: &StopInputs) -> (i32, [u8; READOUTS]) {
    let mut x = [0u8; net::PADDED];
    x[..STOP_INPUT_COUNT].copy_from_slice(&inputs.0);
    let mut hidden_1 = [0u8; net::PADDED];
    dsp.nn_dense_clipped(
        &x,
        &net::WEIGHTS_1,
        &net::BIAS_1,
        net::SHIFT_1,
        &mut hidden_1[..net::HIDDEN_1],
    );
    let mut hidden_2 = [0u8; net::PADDED];
    dsp.nn_dense_clipped(
        &hidden_1,
        &net::WEIGHTS_2,
        &net::BIAS_2,
        net::SHIFT_2,
        &mut hidden_2[..net::HIDDEN_2],
    );
    let mut out = [0i32; 1];
    dsp.nn_dense_linear(&hidden_2, &net::WEIGHTS_OUT, &[net::BIAS_OUT], &mut out);
    let mut readouts = [0u8; READOUTS];
    readouts.copy_from_slice(&hidden_2[..READOUTS]);
    (out[0], readouts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_matches_the_training_record() {
        let dsp = Dsp::new();
        for (inputs, value) in net::TEST_VECTORS {
            assert_eq!(forward(&dsp, &StopInputs(inputs)).0, value);
        }
    }

    #[test]
    fn zero_terms_are_off() {
        assert_eq!(StopNetwork::of(&PlannerParams::default()), None);
    }
}
