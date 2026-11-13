//! Every square VVC angular mode: scalar/SIMD equality, orientation symmetry,
//! constant preservation, axis aliases, the oriented (possibly transposed)
//! form and exact-sized/unaligned edge buffers.
use crate::{check::CheckReport, rng::Rng};
use nc_dsp::{Dsp, VvcIntraMode};

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    check_pdpc_rounding(candidate, reference, report);
    for side in [4, 8, 16, 32] {
        for pattern in 0..12 {
            let mut sample = |i| edge_sample(rng, pattern, i);
            let top: Vec<_> = (0..=2 * side).map(&mut sample).collect();
            let left: Vec<_> = (0..=2 * side).map(&mut sample).collect();
            let corner = sample(3);
            let (top, left) = (&top[1..], &left[1..]);
            for direction in 2..=66 {
                for luma in [false, true] {
                    let mode = VvcIntraMode::Angular { direction, luma };
                    let mut actual = vec![91; side * side + 2];
                    let mut expected = actual.clone();
                    candidate.vvc_intra_predict(side, mode, top, left, corner, &mut actual[1..]);
                    reference.vvc_intra_predict(side, mode, top, left, corner, &mut expected[1..]);
                    record(
                        report,
                        actual == expected,
                        candidate,
                        side,
                        direction,
                        luma,
                        pattern,
                        "scalar",
                    );
                    let guard = actual[0] == 91 && actual[side * side + 1] == 91;
                    record(
                        report, guard, candidate, side, direction, luma, pattern, "guard",
                    );
                    let oriented = oriented_matches(
                        candidate,
                        reference,
                        Edges { top, left, corner },
                        side,
                        mode,
                        &actual[1..=side * side],
                    );
                    record(
                        report, oriented, candidate, side, direction, luma, pattern, "oriented",
                    );
                    if pattern < 3 {
                        record(
                            report,
                            actual[1..=side * side]
                                .iter()
                                .all(|&v| i32::from(v) == corner),
                            candidate,
                            side,
                            direction,
                            luma,
                            pattern,
                            "constant",
                        );
                    }
                    let symmetric = transpose_symmetric(
                        candidate,
                        Edges { top, left, corner },
                        side,
                        direction,
                        luma,
                        &actual[1..=side * side],
                    );
                    record(
                        report,
                        symmetric,
                        candidate,
                        side,
                        direction,
                        luma,
                        pattern,
                        "transpose",
                    );
                    if direction == 18 || direction == 50 {
                        let mut axis = vec![0; side * side];
                        let mode = if direction == 18 {
                            VvcIntraMode::Horizontal
                        } else {
                            VvcIntraMode::Vertical
                        };
                        candidate.vvc_intra_predict(side, mode, top, left, corner, &mut axis);
                        record(
                            report,
                            axis == actual[1..=side * side],
                            candidate,
                            side,
                            direction,
                            luma,
                            pattern,
                            "axis",
                        );
                    }
                }
            }
        }
    }
}
/// Constant but different main/side edges exercise every signed PDPC
/// difference and half-way rounding case. Exact diagonals avoid interpolation
/// hiding a difference; all four sides cover every nonzero PDPC weight.
fn check_pdpc_rounding(candidate: &Dsp, reference: &Dsp, report: &mut CheckReport) {
    for side in [4, 8, 16, 32] {
        for difference in -255i32..=255 {
            let main = (-difference).max(0);
            let side_value = difference.max(0);
            let top = vec![main; 2 * side];
            let left = vec![side_value; 2 * side];
            for direction in [2, 66] {
                let mode = VvcIntraMode::Angular {
                    direction,
                    luma: true,
                };
                let mut actual = vec![91; side * side + 2];
                let mut expected = actual.clone();
                candidate.vvc_intra_predict(side, mode, &top, &left, main, &mut actual[1..]);
                reference.vvc_intra_predict(side, mode, &top, &left, main, &mut expected[1..]);
                record(
                    report,
                    actual == expected,
                    candidate,
                    side,
                    direction,
                    true,
                    (difference + 255) as usize,
                    "pdpc rounding",
                );
            }
        }
    }
}

/// One block's prepared references.
#[derive(Clone, Copy)]
struct Edges<'a> {
    top: &'a [i32],
    left: &'a [i32],
    corner: i32,
}

/// Negative diagonal 34 is self-transposed. All other angles exchange
/// horizontal/vertical orientation at 68-direction, with the edges swapped.
fn transpose_symmetric(
    candidate: &Dsp,
    edges: Edges<'_>,
    side: usize,
    direction: u8,
    luma: bool,
    natural: &[u8],
) -> bool {
    let mut transposed = vec![0; side * side];
    candidate.vvc_intra_predict(
        side,
        VvcIntraMode::Angular {
            direction: 68 - direction,
            luma,
        },
        edges.left,
        edges.top,
        edges.corner,
        &mut transposed,
    );
    (0..side).all(|y| (0..side).all(|x| natural[y * side + x] == transposed[x * side + y]))
}

/// The oriented form is the natural block `natural`, transposed exactly for
/// the horizontal angular directions, and the reference agrees.
fn oriented_matches(
    candidate: &Dsp,
    reference: &Dsp,
    edges: Edges<'_>,
    side: usize,
    mode: VvcIntraMode,
    natural: &[u8],
) -> bool {
    let Edges { top, left, corner } = edges;
    let mut oriented = vec![0; side * side];
    let mut oriented_reference = vec![0; side * side];
    let transposed =
        candidate.vvc_intra_predict_oriented(side, mode, top, left, corner, &mut oriented);
    let reference_transposed = reference.vvc_intra_predict_oriented(
        side,
        mode,
        top,
        left,
        corner,
        &mut oriented_reference,
    );
    let horizontal = matches!(mode, VvcIntraMode::Angular { direction, .. } if direction < 34 && direction != 18);
    let same_block = (0..side).all(|y| {
        (0..side).all(|x| {
            let at = if transposed {
                x * side + y
            } else {
                y * side + x
            };
            oriented[at] == natural[y * side + x]
        })
    });
    same_block
        && transposed == horizontal
        && transposed == reference_transposed
        && oriented == oriented_reference
}

fn record(
    report: &mut CheckReport,
    equal: bool,
    dsp: &Dsp,
    side: usize,
    direction: u8,
    luma: bool,
    pattern: usize,
    property: &str,
) {
    report.cases += 1;
    if !equal {
        report.failures.push(format!("{} vvc_angular side={side} direction={direction} luma={luma} pattern={pattern} {property}",dsp.isa()));
    }
}

fn edge_sample(rng: &mut Rng, pattern: usize, index: usize) -> i32 {
    match pattern {
        0 => 0,
        1 => 255,
        2 => 128,
        3 => {
            if index.is_multiple_of(2) {
                0
            } else {
                255
            }
        }
        4 => (index * 7 % 256) as i32,
        _ => i32::from(rng.byte()),
    }
}
