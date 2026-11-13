//! In-place VVC chroma against independently gathered original segments.
//! Both orientations, partial batches and guard bytes catch lane, transpose,
//! per-segment decision and write-footprint errors in the shared plane access.
use crate::{check::CheckReport, rng::Rng};
use nc_common::PlaneViewMut;
use nc_dsp::Dsp;

const WIDTH: usize = 19;
const HEIGHT: usize = 17;
const STRIDE: usize = 23;
const REACH: usize = 4;
const POSITIONS: usize = 2 * REACH;
const SHAPES: usize = 24;

fn sample(rng: &mut Rng, shape: usize, pos: usize, line: usize) -> u8 {
    let q = usize::from(pos >= REACH);
    match shape {
        0 => 0,
        1 => 255,
        2..=13 => (64 + line + q * (shape - 2)) as u8,
        14..=19 => (128 + (pos as i32 - 4) * (shape as i32 - 13)).clamp(0, 255) as u8,
        20 => {
            if pos.is_multiple_of(2) {
                0
            } else {
                255
            }
        }
        _ => rng.byte(),
    }
}

/// Original two-line API is the oracle even for the new scalar edge API.
fn reference_edge(
    reference: &Dsp,
    plane: &mut [u8],
    (x, y): (usize, usize),
    vertical: bool,
    lines: usize,
    beta: i32,
    tc: i32,
    large: bool,
    boundary: bool,
) {
    let at = |line: usize, pos: usize| {
        if vertical {
            (y + line) * STRIDE + x + pos - REACH
        } else {
            (y + pos - REACH) * STRIDE + x + line
        }
    };
    for start in (0..lines).step_by(2) {
        let mut edge: [[u8; POSITIONS]; 2] =
            std::array::from_fn(|line| std::array::from_fn(|pos| plane[at(start + line, pos)]));
        reference.vvc_deblock_chroma(&mut edge, beta, tc, large, boundary);
        for (line, row) in edge.into_iter().enumerate() {
            for (pos, value) in row.into_iter().enumerate() {
                plane[at(start + line, pos)] = value;
            }
        }
    }
}

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    check_shared_hevc_access(candidate, reference, rng, report);
    for beta in [0, 1, 7, 16, 32, 64, 88] {
        for tc in [0, 1, 2, 4, 16, 50, 100] {
            for shape in 0..SHAPES {
                for vertical in [false, true] {
                    for lines in [2, 4, 6, 8] {
                        // End exactly at a visible edge, with non-aligned rows.
                        let at = if vertical {
                            (REACH, HEIGHT - lines)
                        } else {
                            (WIDTH - lines, HEIGHT - REACH)
                        };
                        let mut plane: Vec<u8> = (0..STRIDE * HEIGHT).map(|_| rng.byte()).collect();
                        for line in 0..lines {
                            // Adjacent segments deliberately choose different filters.
                            let shape = (shape + (line / 2) * 7) % SHAPES;
                            for pos in 0..POSITIONS {
                                let (x, y) = if vertical {
                                    (at.0 + pos - REACH, at.1 + line)
                                } else {
                                    (at.0 + line, at.1 + pos - REACH)
                                };
                                plane[y * STRIDE + x] = sample(rng, shape, pos, line % 2);
                            }
                        }
                        for large in [false, true] {
                            for boundary in [false, true] {
                                let mut expected = plane.clone();
                                reference_edge(
                                    reference,
                                    &mut expected,
                                    at,
                                    vertical,
                                    lines,
                                    beta,
                                    tc,
                                    large,
                                    boundary,
                                );
                                for dsp in [reference, candidate] {
                                    let mut actual = plane.clone();
                                    let mut view =
                                        PlaneViewMut::new(&mut actual, STRIDE, WIDTH, HEIGHT)
                                            .unwrap();
                                    dsp.vvc_deblock_chroma_edge(
                                        &mut view, at, vertical, lines, beta, tc, large, boundary,
                                    );
                                    report.cases += 1;
                                    if actual != expected {
                                        report.failures.push(format!(
                                            "{} vvc_deblock_chroma_edge beta={beta} tc={tc} shape={shape} vertical={vertical} lines={lines} large={large} boundary={boundary}", dsp.isa()
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// HEVC uses the factored access too: include odd strides and batches ending
/// at the visible frame boundary, beyond its ordinary interior-edge tests.
fn check_shared_hevc_access(
    candidate: &Dsp,
    reference: &Dsp,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    for vertical in [false, true] {
        for lines in [4, 8] {
            for (beta, tc) in [(0, 0), (16, 1), (64, 24)] {
                let plane: Vec<u8> = (0..STRIDE * HEIGHT).map(|_| 96 + rng.byte() % 8).collect();
                let at = if vertical {
                    (REACH, HEIGHT - lines)
                } else {
                    (WIDTH - lines, HEIGHT - REACH)
                };
                for luma in [false, true] {
                    if luma && lines != 8 {
                        continue;
                    }
                    let mut outputs = [plane.clone(), plane.clone()];
                    for (dsp, out) in [reference, candidate].into_iter().zip(&mut outputs) {
                        let mut view = PlaneViewMut::new(out, STRIDE, WIDTH, HEIGHT).unwrap();
                        if luma {
                            dsp.hevc_deblock_luma_edge(&mut view, at, vertical, [tc, 0], beta);
                        } else {
                            dsp.hevc_deblock_chroma_edge(
                                &mut view,
                                at,
                                vertical,
                                lines,
                                [tc, 0, tc, 1],
                            );
                        }
                    }
                    report.cases += 1;
                    if outputs[0] != outputs[1] {
                        report.failures.push(format!("{} shared HEVC sample access vertical={vertical} lines={lines} luma={luma} beta={beta} tc={tc}",candidate.isa()));
                    }
                }
            }
        }
    }
}
