//! The zero-gate network: a learned term of the motion search's zero-vector
//! gates (`analysis.zero_gate_net_*`), per superblock.
//!
//! Where it fits: each superblock's search path (static, the zero vector
//! alone, a window at zero, or the full hierarchical search) is chosen by
//! comparing its zero-vector SAD per pixel with three gates
//! (`motion.rs` `ZeroVectorGates`); the gates are scaled by
//! `max(0, 1 + shift)`, where the linear shift (`analysis.zero_gate_shift_*`)
//! reads the frame role, λ, the source noise and the moving neighbours. This
//! network reads all the search knows before it chooses (the inputs below)
//! and adds `Σ_j readout_j · h_j − weight · value` to that shift: `h_j` its
//! last hidden activations in [0, 1], `value` its predicted log gain of the
//! full search over the zero vector per pixel.
//!
//! Why this form (owner, 2026-11-22: "today's heuristic is a degenerate
//! case of the network"): the output terms are registered parameters, all 0
//! by default, so the gates are exactly the linear ones and the network is
//! not run. The hidden layers are fitted offline to what the full search
//! gained over the zero vector on superblocks searched with the gates off
//! (`eval/learned/gate_train.py`, on the label dump's own input bytes); the
//! output terms are tuned on real encodes (`nceval.tune`).
//!
//! Inputs ([`GateEvidence`], as bytes by `net_inputs.rs`): the zero-vector
//! SAD per pixel; the source noise; the share of moving causal neighbours
//! and their mean vector length; the last frame's co-located vector length
//! (and whether there is one); the superblock's static share and source
//! variance per pixel; its visible share; and the frame's λ, role and static
//! share. Cost: one 12 -> 32 -> 8 -> 1 integer forward per superblock.

use nc_dsp::Dsp;

use crate::gate_net_weights as net;
use crate::net_inputs::{INPUT_MAX, input, log2_steps};

/// Inputs per superblock.
pub const GATE_INPUT_COUNT: usize = 12;
const _: () = assert!(net::INPUT_COUNT == GATE_INPUT_COUNT);
/// Last hidden activations the shift reads (`analysis.zero_gate_net_h*`).
pub const GATE_READOUTS: usize = net::HIDDEN_2;
/// The activations' largest value: 1.0 of a readout's unit.
const ACTIVATION_ONE: f32 = nc_dsp::NN_ACTIVATION_MAX as f32;
/// SAD and variance per pixel in double steps (32 per octave).
const PER_PIXEL_SCALE: i32 = 2;
/// λ in half steps (eight per octave): λ up to 2^15 stays in range.
const LAMBDA_DIVISOR: i32 = 2;

/// A frame's constant inputs, set by the encoder with the frame's search
/// settings.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GateFrame {
    /// The frame's λ (mode decision).
    pub lambda: f32,
    /// It is a non-base (droppable layer) frame.
    pub nonbase: bool,
}

/// The search's zero-gate network settings of a frame.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GateNetConfig {
    /// The output terms (`None`: the linear shift alone, the network off).
    pub network: Option<ZeroGateNetwork>,
    /// Record each SB's inputs in `SbStats::gate_inputs` (the label dump),
    /// whether or not the network runs.
    pub record_inputs: bool,
    /// The frame's constant inputs.
    pub frame: GateFrame,
}

/// What the search knows of a superblock before it chooses its path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateEvidence {
    /// The superblock's zero-vector SAD per pixel.
    pub zero_sad_per_pixel: f32,
    /// The running source noise estimate (luma sigma).
    pub noise_sigma: f32,
    /// Share of the searched causal neighbours whose vector is not zero.
    pub moving_share: f32,
    /// Their mean vector length (|x| + |y|, the search's vector units).
    pub neighbour_length: f32,
    /// The last frame's co-located mean vector length, if known.
    pub temporal_length: Option<f32>,
    /// Share of the superblock's 8x8 blocks pre-analysis finds static.
    pub static_share: f32,
    /// Source luma variance per pixel.
    pub variance_per_pixel: f32,
    /// Share of the superblock inside the picture.
    pub visible_share: f32,
    /// The frame's constant inputs.
    pub frame: GateFrame,
    /// The frame's static share (pre-analysis).
    pub frame_static_share: f32,
}

/// The network's inputs at one superblock.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GateInputs(pub [u8; GATE_INPUT_COUNT]);

impl GateInputs {
    /// The bytes of `e`.
    #[must_use]
    pub fn new(e: &GateEvidence) -> Self {
        let share = |v: f32| (v * INPUT_MAX as f32).round() as i32;
        let flag = |on: bool| if on { INPUT_MAX } else { 0 };
        let per_pixel = |v: f32| PER_PIXEL_SCALE * log2_steps(1.0 + v);
        let values: [i32; GATE_INPUT_COUNT] = [
            per_pixel(e.zero_sad_per_pixel),
            per_pixel(e.noise_sigma),
            share(e.moving_share),
            log2_steps(1.0 + e.neighbour_length),
            log2_steps(1.0 + e.temporal_length.unwrap_or(0.0)),
            flag(e.temporal_length.is_some()),
            share(e.static_share),
            log2_steps(1.0 + e.variance_per_pixel),
            share(e.visible_share),
            log2_steps(e.frame.lambda.max(f32::MIN_POSITIVE)) / LAMBDA_DIVISOR,
            flag(e.frame.nonbase),
            share(e.frame_static_share),
        ];
        Self(values.map(input))
    }
}

/// The network's output terms in the gates' scale shift.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZeroGateNetwork {
    /// Per unit of minus the predicted value.
    pub weight: f32,
    /// Per unit of each last hidden activation.
    pub readout: [f32; GATE_READOUTS],
}

impl ZeroGateNetwork {
    /// The terms, `None` while they are all 0.
    #[must_use]
    pub fn new(weight: f32, readout: [f32; GATE_READOUTS]) -> Option<Self> {
        (weight != 0.0 || readout.iter().any(|&r| r != 0.0)).then_some(Self { weight, readout })
    }

    /// The terms' contribution to the scale shift at `inputs`.
    #[must_use]
    pub fn shift(&self, dsp: &Dsp, inputs: &GateInputs) -> f32 {
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

/// The integer value output (`net::OUTPUT_ONE` per unit) and the last
/// hidden activations.
fn forward(dsp: &Dsp, inputs: &GateInputs) -> (i32, [u8; GATE_READOUTS]) {
    let mut x = [0u8; net::PADDED];
    x[..GATE_INPUT_COUNT].copy_from_slice(&inputs.0);
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
    let mut readouts = [0u8; GATE_READOUTS];
    readouts.copy_from_slice(&hidden_2[..GATE_READOUTS]);
    (out[0], readouts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_matches_the_training_record() {
        let dsp = Dsp::new();
        for (inputs, value) in net::TEST_VECTORS {
            assert_eq!(forward(&dsp, &GateInputs(inputs)).0, value);
        }
    }

    #[test]
    fn inputs_stay_in_range() {
        let e = GateEvidence {
            zero_sad_per_pixel: 1e6,
            noise_sigma: 3.0,
            moving_share: 1.0,
            neighbour_length: 40.0,
            temporal_length: None,
            static_share: 0.5,
            variance_per_pixel: 1e9,
            visible_share: 1.0,
            frame: GateFrame {
                lambda: 1e9,
                nonbase: true,
            },
            frame_static_share: 0.25,
        };
        let inputs = GateInputs::new(&e);
        assert!(inputs.0.iter().all(|&v| i32::from(v) <= INPUT_MAX));
        assert_eq!(inputs.0[2], 127);
        assert_eq!(inputs.0[5], 0);
        assert_eq!(inputs.0[10], 127);
    }

    #[test]
    fn zero_terms_are_off() {
        assert_eq!(ZeroGateNetwork::new(0.0, [0.0; GATE_READOUTS]), None);
    }
}
