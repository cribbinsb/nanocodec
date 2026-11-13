//! Bit-exactness of AV1 filter intra through `Dsp::av1_intra_predict_filter`
//! and `Dsp::av1_intra_predict_filter_block` (the ISA's patch kernel)
//! against scalar, for every square side, every rectangular shape and mode
//! on random and extreme edges; and of directional prediction through
//! `Dsp::av1_intra_predict_angle` (the ISA's runs and tile transposes) at
//! every reachable angle and square side, with and without the edge
//! extensions; and of every square mode through `Dsp::av1_intra_predict`
//! (the ISA's smooth and Paeth kernels).

use nc_common::Plane;
use nc_dsp::Dsp;
use nc_dsp::intra_prediction::av1_block::{BlockEdges, EdgeAvail};
use nc_dsp::intra_prediction::filter::{FILTER_INTRA_MAX_SIZE, FilterIntraMode};
use nc_dsp::intra_prediction::{EdgeExtension, IntraEdges, IntraMode};

use crate::check::CheckReport;
use crate::check_codec::record;
use crate::rng::Rng;

/// Block sides filter intra allows.
const SIDES: [usize; 4] = [4, 8, 16, 32];
/// Rectangular transform shapes filter intra allows (both sides <= 32).
const RECTANGLES: [(usize, usize); 10] = [
    (4, 8),
    (8, 4),
    (8, 16),
    (16, 8),
    (16, 32),
    (32, 16),
    (4, 16),
    (16, 4),
    (8, 32),
    (32, 8),
];
/// Square sides of directional prediction.
const DIRECTIONAL_SIDES: [usize; 5] = [4, 8, 16, 32, 64];
/// Largest directional side, and the most angle deltas either way.
const DIRECTIONAL_MAX: usize = 64;
const MAX_ANGLE_DELTA: i16 = 3;
/// Degrees per angle delta step.
const ANGLE_STEP: i16 = 3;
/// Picture side: room for the largest block and its edges.
const PICTURE: usize = 2 * FILTER_INTRA_MAX_SIZE;
/// Top-left of the predicted block, so every edge sample is inside.
const ORIGIN: usize = FILTER_INTRA_MAX_SIZE;

/// See the module docs.
pub(crate) fn check_av1_intra(
    candidate: &Dsp,
    reference: &Dsp,
    iterations: usize,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    for _ in 0..iterations {
        // Extreme pictures half the time: the filter's full range and clipping.
        let extreme = rng.below(2) == 0;
        let mut plane = Plane::new(PICTURE, PICTURE, 0).unwrap();
        let mut view = plane.view_mut();
        for y in 0..PICTURE {
            for sample in &mut view.row_mut(y)[..PICTURE] {
                *sample = match (extreme, rng.below(2)) {
                    (true, 0) => 0,
                    (true, _) => u8::MAX,
                    (false, _) => rng.byte(),
                };
            }
        }
        for side in SIDES {
            let edges = IntraEdges::new(&plane, ORIGIN, ORIGIN, side);
            for mode in FilterIntraMode::ALL {
                let predict = |dsp: &Dsp| {
                    let mut out = Plane::new(side, side, 0).unwrap();
                    dsp.av1_intra_predict_filter(&edges, mode, &mut out.view_mut());
                    (0..side)
                        .flat_map(|y| out.view().row(y)[..side].to_vec())
                        .collect::<Vec<u8>>()
                };
                let same = predict(candidate) == predict(reference);
                record(report, candidate.isa(), same, || {
                    format!("av1_filter_intra {mode:?} side={side} extreme={extreme}")
                });
            }
        }
        let avail = EdgeAvail {
            have_above: true,
            have_left: true,
            ..EdgeAvail::default()
        };
        for (w, h) in RECTANGLES {
            let edges = BlockEdges::read(&plane, ORIGIN, ORIGIN, w, h, avail);
            for mode in FilterIntraMode::ALL {
                let predict = |dsp: &Dsp| {
                    let mut out = Plane::new(w, h, 0).unwrap();
                    dsp.av1_intra_predict_filter_block(&edges, mode, &mut out.view_mut());
                    (0..h)
                        .flat_map(|y| out.view().row(y)[..w].to_vec())
                        .collect::<Vec<u8>>()
                };
                let same = predict(candidate) == predict(reference);
                record(report, candidate.isa(), same, || {
                    format!("av1_filter_intra {mode:?} {w}x{h} extreme={extreme}")
                });
            }
        }
    }
}

/// See the module docs: one random or extreme picture per iteration, every
/// reachable angle (nominal ± 3 · delta) at every side.
pub(crate) fn check_av1_directional(
    candidate: &Dsp,
    reference: &Dsp,
    iterations: usize,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    // Room above-left of the block and two block sides beyond it.
    let picture = 3 * DIRECTIONAL_MAX;
    for _ in 0..iterations {
        let extreme = rng.below(2) == 0;
        let mut plane = Plane::new(picture, picture, 0).unwrap();
        let mut view = plane.view_mut();
        for y in 0..picture {
            for sample in &mut view.row_mut(y)[..picture] {
                *sample = match (extreme, rng.below(2)) {
                    (true, 0) => 0,
                    (true, _) => u8::MAX,
                    (false, _) => rng.byte(),
                };
            }
        }
        let extension = EdgeExtension {
            above_right: rng.below(2) == 0,
            below_left: rng.below(2) == 0,
        };
        for side in DIRECTIONAL_SIDES {
            let edges =
                IntraEdges::extended(&plane, DIRECTIONAL_MAX, DIRECTIONAL_MAX, side, extension);
            for mode in IntraMode::ALL {
                let predict = |dsp: &Dsp| {
                    let mut out = Plane::new(side, side, 0).unwrap();
                    dsp.av1_intra_predict(&edges, mode, &mut out.view_mut());
                    (0..side)
                        .flat_map(|y| out.view().row(y)[..side].to_vec())
                        .collect::<Vec<u8>>()
                };
                let same = predict(candidate) == predict(reference);
                record(report, candidate.isa(), same, || {
                    format!("av1_intra {mode:?} side={side} {extension:?} extreme={extreme}")
                });
            }
            for mode in IntraMode::ALL {
                let Some(nominal) = mode.nominal_angle() else {
                    continue;
                };
                for delta in -MAX_ANGLE_DELTA..=MAX_ANGLE_DELTA {
                    let angle = (nominal as i16 + ANGLE_STEP * delta) as u16;
                    let predict = |dsp: &Dsp| {
                        let mut out = Plane::new(side, side, 0).unwrap();
                        dsp.av1_intra_predict_angle(&edges, angle, &mut out.view_mut());
                        (0..side)
                            .flat_map(|y| out.view().row(y)[..side].to_vec())
                            .collect::<Vec<u8>>()
                    };
                    let same = predict(candidate) == predict(reference);
                    record(report, candidate.isa(), same, || {
                        format!(
                            "av1_directional angle={angle} side={side} {extension:?} extreme={extreme}"
                        )
                    });
                }
            }
        }
    }
}
