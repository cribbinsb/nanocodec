//! Bit-exactness of AV1 filter intra through `Dsp::av1_intra_predict_filter`
//! and `Dsp::av1_intra_predict_filter_block` (the ISA's patch kernel)
//! against scalar, for every square side, every rectangular shape and mode
//! on random and extreme edges; and of directional prediction through
//! `Dsp::av1_intra_predict_angle` (the ISA's runs and tile transposes) at
//! every reachable angle and square side, with and without the edge
//! extensions and the edge filter; and of every square mode through `Dsp::av1_intra_predict`
//! (the ISA's smooth and Paeth kernels); and that edges formed only where a
//! prediction reads them (`BlockEdges::read_for`) predict as the full edges.

use nc_common::{BitDepth, Plane};
use nc_dsp::Dsp;
use nc_dsp::intra_prediction::av1_block::{
    BlockEdges, EdgeAvail, EdgeFilterCtx, EdgeNeeds, IntraKind,
};
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
            let edges = BlockEdges::read(&plane, (ORIGIN, ORIGIN), (w, h), avail, BitDepth::EIGHT);
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
                    // With the edge filter: the prepared square edges go to
                    // the same kernels (or the scalar zones when upsampled).
                    let avail = EdgeAvail {
                        have_above: true,
                        have_left: true,
                        have_above_right: extension.above_right,
                        have_below_left: extension.below_left,
                    };
                    let block = BlockEdges::read(
                        &plane,
                        (DIRECTIONAL_MAX, DIRECTIONAL_MAX),
                        (side, side),
                        avail,
                        BitDepth::EIGHT,
                    );
                    for smooth in [false, true] {
                        let kind = IntraKind::Directional {
                            angle,
                            edge_filter: Some(EdgeFilterCtx {
                                smooth_above: smooth,
                                smooth_left: false,
                            }),
                        };
                        let predict = |dsp: &Dsp| {
                            let mut out = Plane::new(side, side, 0).unwrap();
                            dsp.av1_intra_predict_block(&block, kind, &mut out.view_mut());
                            (0..side)
                                .flat_map(|y| out.view().row(y)[..side].to_vec())
                                .collect::<Vec<u8>>()
                        };
                        let same = predict(candidate) == predict(reference);
                        record(report, candidate.isa(), same, || {
                            format!(
                                "av1_directional filtered angle={angle} side={side} smooth={smooth} \
                                 {extension:?} extreme={extreme}"
                            )
                        });
                    }
                }
            }
        }
    }
}

/// Block sides of AV1 intra transform blocks; shapes up to 4:1.
const BLOCK_SIDES: [usize; 5] = [4, 8, 16, 32, 64];
/// The non-directional kinds.
const PLAIN_KINDS: [IntraKind; 5] = [
    IntraKind::Dc,
    IntraKind::Smooth,
    IntraKind::SmoothVertical,
    IntraKind::SmoothHorizontal,
    IntraKind::Paeth,
];
/// Lowest and highest reachable directional angle (45 - 9, 203 + 9).
const ANGLE_RANGE: (u16, u16) = (36, 212);

/// `BlockEdges::read_into` with `EdgeNeeds::of` (only the edges a kind
/// reads, in place over another picture's edges, as the decoder forms them) predicts exactly what the full
/// `BlockEdges::read` does through the reference: every shape, kind and
/// reachable angle, with and without the edge filter, random
/// availability, the block near the picture's right and bottom edges
/// (reads clamped short of the block's side) or inside it; and filter
/// intra with `EdgeNeeds::sides`. The directional cases cover the
/// whole-block kernels (`zones_avx2`: every shape up to 16x16, both
/// resolutions, all three zones) against the scalar zones and runs.
pub(crate) fn check_av1_edge_needs(
    candidate: &Dsp,
    reference: &Dsp,
    iterations: usize,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    for _ in 0..iterations {
        for w in BLOCK_SIDES {
            for h in BLOCK_SIDES {
                if w > 4 * h || h > 4 * w {
                    continue;
                }
                check_edge_needs_shape(candidate, reference, (w, h), rng, report);
            }
        }
    }
}

fn check_edge_needs_shape(
    candidate: &Dsp,
    reference: &Dsp,
    (w, h): (usize, usize),
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    // The block's origin, and the picture ending inside or beyond its
    // extensions (as far as 1 sample past the origin).
    let origin = (DIRECTIONAL_MAX, DIRECTIONAL_MAX);
    let picture_w = origin.0 + 1 + rng.below(2 * w);
    let picture_h = origin.1 + 1 + rng.below(2 * h);
    let mut random_plane = || {
        let mut plane = Plane::new(picture_w, picture_h, 0).unwrap();
        let mut view = plane.view_mut();
        for y in 0..picture_h {
            for sample in &mut view.row_mut(y)[..picture_w] {
                *sample = rng.byte();
            }
        }
        plane
    };
    let plane = random_plane();
    // Edges of another picture, which `read_into` overwrites only where
    // the prediction reads: what it leaves must not matter.
    let stale = random_plane();
    let every_edge = EdgeAvail {
        have_above: true,
        have_left: true,
        have_above_right: true,
        have_below_left: true,
    };
    let avail = EdgeAvail {
        have_above: rng.below(4) != 0,
        have_left: rng.below(4) != 0,
        have_above_right: rng.below(2) == 0,
        have_below_left: rng.below(2) == 0,
    };
    let reuse = |need: EdgeNeeds| {
        let mut edges = BlockEdges::read(&stale, origin, (w, h), every_edge, BitDepth::EIGHT);
        edges.read_into(&plane, origin, (w, h), avail, BitDepth::EIGHT, need);
        edges
    };
    let full = BlockEdges::read(&plane, origin, (w, h), avail, BitDepth::EIGHT);
    let predict = |dsp: &Dsp, edges: &BlockEdges, kind: IntraKind| {
        let mut out = Plane::new(w, h, 0).unwrap();
        dsp.av1_intra_predict_block(edges, kind, &mut out.view_mut());
        out
    };
    let edge_filter = Some(EdgeFilterCtx {
        smooth_above: rng.below(2) == 0,
        smooth_left: rng.below(2) == 0,
    });
    let directional = (ANGLE_RANGE.0..=ANGLE_RANGE.1).flat_map(|angle| {
        [None, edge_filter].map(|edge_filter| IntraKind::Directional { angle, edge_filter })
    });
    for kind in PLAIN_KINDS.into_iter().chain(directional) {
        let needed = reuse(EdgeNeeds::of(kind, (w, h)));
        let same = same_samples(
            &predict(candidate, &needed, kind),
            &predict(reference, &full, kind),
        );
        record(report, candidate.isa(), same, || {
            format!("av1 edge needs {kind:?} {w}x{h} {avail:?} picture {picture_w}x{picture_h}")
        });
    }
    if w <= FILTER_INTRA_MAX_SIZE && h <= FILTER_INTRA_MAX_SIZE {
        let needed = reuse(EdgeNeeds::sides((w, h)));
        for mode in FilterIntraMode::ALL {
            let filter = |dsp: &Dsp, edges: &BlockEdges| {
                let mut out = Plane::new(w, h, 0).unwrap();
                dsp.av1_intra_predict_filter_block(edges, mode, &mut out.view_mut());
                out
            };
            let same = same_samples(&filter(candidate, &needed), &filter(reference, &full));
            record(report, candidate.isa(), same, || {
                format!("av1 edge needs filter {mode:?} {w}x{h} {avail:?}")
            });
        }
    }
}

/// The two planes' visible samples are equal.
fn same_samples(a: &Plane, b: &Plane) -> bool {
    let (a, b) = (a.view(), b.view());
    (0..a.height()).all(|y| a.row(y)[..a.width()] == b.row(y)[..b.width()])
}
