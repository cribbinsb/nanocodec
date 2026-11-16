//! Quantizer and energy gates for the encoder's cheap paths: one rule, read
//! from what a block or frame shows, that decides where an approximation
//! (a partial forward transform, no transform-split trial, ...) is taken.
//!
//! An approximation that drops detail is harmless where the quantizer would
//! drop that detail anyway, or where the block has little residual to lose,
//! and costly elsewhere. AV1 Fast's partial forward transform, for example,
//! moves PSNR-Y by 0.04 dB at qindex 235 and by 3.1 dB at qindex 110
//! (docs/analysis/content-adaptive-cost-2026-11-16.md §2.2). SVT-AV1 gates
//! its N4 partial transform the same way, on the block's distortion against
//! its quantizer (`100 · dist < th · area · qindex`, `product_coding_loop.c`).
//!
//! The rule reads only the quantizer and the residual, which fixed-QP and
//! rate-controlled encodes both have, so the same values work in both modes:
//!
//! - **quantizer**: the codec's quantizer as a fraction of its range (0
//!   finest, 1 coarsest; [`crate::CodecBackend::relative_quantizer`]);
//! - **energy**: the block's prediction (or residual) error per sample, in
//!   squared quantizer steps, so the threshold does not depend on the
//!   resolution, the block size or the codec's quantizer scale. It is not
//!   normalised for contrast: dark, flat content has small errors and so takes
//!   the cheap path more often, as it should when its residual quantises to
//!   little (SVT's absolute thresholds have the same property).
//!
//! Each gate is a registered pair of parameters ([`GateParams`]); the codec
//! supplies the inputs ([`GateInput`]) and keeps its own syntax decisions.

use crate::param_group;

param_group! {
    /// Where the cheap paths apply ([`WorkGate`]). The defaults keep every
    /// codec as before the gates existed: the partial transform wherever
    /// its codec lever is on (quantizer 0), the transform-split trial at
    /// every quantizer (1.1). The energy halves of these gates are not
    /// registered yet: no codec measures the error at every point a path
    /// is decided (a trial and the coding of a block must decide alike).
    pub struct GateParams ("gate") {
        partial_transform_min_quantizer: 0.0, [0.0, 1.1],
            "relative quantizer (0 finest, 1 coarsest; above 1 never) from which a codec's partial forward transform (AV1 `av1.partial_forward`) applies: higher keeps the exact transform at more quantizers, at more transform work",
        tx_split_skip_min_quantizer: 1.1, [0.0, 1.1],
            "relative quantizer from which the transform-split trial is skipped (above 1: tried at every quantizer): lower skips it at more quantizers, faster",
    }
}

/// What a gate reads about a block or a frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateInput {
    /// The quantizer as a fraction of the codec's range: 0 finest, 1 coarsest.
    pub quantizer: f32,
    /// The quantizer step in sample units: the energy's scale.
    pub step: f32,
    /// The prediction (or residual) error's sum of squares and the samples it
    /// covers, where measured.
    pub energy: Option<(u64, u32)>,
}

/// One cheap path's gate: taken at quantizers from `min_quantizer` up, and
/// below that where the error per sample is under `max_energy` squared
/// steps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorkGate {
    /// Relative quantizer from which the cheap path is always taken.
    pub min_quantizer: f32,
    /// Error per sample, in squared steps, under which it is taken at any
    /// quantizer (0: never by energy).
    pub max_energy: f32,
}

impl WorkGate {
    /// The partial forward transform's gate.
    #[must_use]
    pub fn partial_transform(params: &GateParams) -> Self {
        Self {
            min_quantizer: params.partial_transform_min_quantizer,
            max_energy: 0.0,
        }
    }

    /// The gate on skipping the transform-split trial.
    #[must_use]
    pub fn tx_split_skip(params: &GateParams) -> Self {
        Self {
            min_quantizer: params.tx_split_skip_min_quantizer,
            max_energy: 0.0,
        }
    }

    /// Whether the quantizer alone admits the cheap path (a frame-level
    /// decision, before any block is measured).
    #[must_use]
    pub fn admits_quantizer(self, quantizer: f32) -> bool {
        quantizer >= self.min_quantizer
    }

    /// Whether `input` takes the cheap path.
    #[must_use]
    pub fn takes_cheap_path(self, input: GateInput) -> bool {
        self.admits_quantizer(input.quantizer)
            || input.energy.is_some_and(|(sse, samples)| {
                (sse as f32) < self.max_energy * input.step * input.step * samples as f32
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(quantizer: f32, sse: Option<u64>) -> GateInput {
        GateInput {
            quantizer,
            step: 10.0,
            energy: sse.map(|sse| (sse, 64)),
        }
    }

    #[test]
    fn defaults_take_the_cheap_path_everywhere() {
        let gate = WorkGate::partial_transform(&GateParams::default());
        assert!(gate.takes_cheap_path(input(0.0, None)));
        assert!(gate.takes_cheap_path(input(0.0, Some(u64::MAX))));
    }

    #[test]
    fn quantizer_and_energy_gates() {
        let gate = WorkGate {
            min_quantizer: 0.7,
            max_energy: 0.5,
        };
        assert!(gate.takes_cheap_path(input(0.7, None)));
        assert!(!gate.takes_cheap_path(input(0.5, None)));
        // 0.5 squared steps of 10 over 64 samples: 3200.
        assert!(gate.takes_cheap_path(input(0.5, Some(3199))));
        assert!(!gate.takes_cheap_path(input(0.5, Some(3200))));
        let never = WorkGate {
            min_quantizer: 1.1,
            max_energy: 0.0,
        };
        assert!(!never.takes_cheap_path(input(1.0, Some(0))));
    }
}
