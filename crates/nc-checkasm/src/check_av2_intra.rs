//! Bit-exactness of the AV2 intra kernels (directional runs and smooth,
//! through `Dsp::av2_intra_predict`) against scalar, for every predictor,
//! side and angle on random edges.

use nc_dsp::Dsp;
use nc_dsp::intra_prediction::av2::{AV2_INTRA_MAX_SIDE, Av2IntraEdges, Av2Predictor};

use crate::check::CheckReport;
use crate::check_codec::record;
use crate::rng::Rng;

/// Block sides of AV2 intra transforms.
const SIDES: [usize; 5] = [4, 8, 16, 32, 64];
/// Directional angles: every nominal direction with its deltas spans
/// 36..=212 degrees in steps of three; the axis-aligned ones are copies.
const ANGLES: std::ops::RangeInclusive<u16> = 36..=212;
const ANGLE_STEP: usize = 3;
/// Samples written on each edge: the block side, its extension and the
/// padding the 4-tap filter reads beyond it.
const EDGE_SAMPLES: usize = 2 * AV2_INTRA_MAX_SIDE + 2;

pub(crate) fn check_av2_intra(
    candidate: &Dsp,
    reference: &Dsp,
    iterations: usize,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let mut predictors = vec![
        Av2Predictor::Smooth,
        Av2Predictor::SmoothVertical,
        Av2Predictor::SmoothHorizontal,
    ];
    for angle in ANGLES.step_by(ANGLE_STEP).filter(|a| a % 90 != 0) {
        for luma in [false, true] {
            predictors.push(Av2Predictor::Directional { angle, luma });
        }
    }
    for _ in 0..iterations {
        let mut edges = Av2IntraEdges::new();
        // Extreme edges half the time: the 4-tap filter's clipping and the
        // smooth ramps' 16-bit range.
        let extreme = rng.below(2) == 0;
        let sample = |rng: &mut Rng| match (extreme, rng.below(2)) {
            (true, 0) => 0,
            (true, _) => u8::MAX,
            (false, _) => rng.byte(),
        };
        for value in &mut edges.above_mut()[..EDGE_SAMPLES] {
            *value = sample(rng);
        }
        for value in &mut edges.left_mut()[..EDGE_SAMPLES] {
            *value = sample(rng);
        }
        edges.set_corner(sample(rng));
        for side in SIDES {
            for &predictor in &predictors {
                let predict = |dsp: &Dsp| {
                    let mut out = vec![0; side * side];
                    dsp.av2_intra_predict(&mut edges.clone(), predictor, side, &mut out);
                    out
                };
                let same = predict(candidate) == predict(reference);
                record(report, candidate.isa(), same, || {
                    format!("av2_intra {predictor:?} side={side} extreme={extreme}")
                });
            }
        }
    }
}
