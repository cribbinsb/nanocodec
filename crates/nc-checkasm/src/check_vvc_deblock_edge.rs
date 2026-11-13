//! The in-place VVC luma edge (`Dsp::vvc_deblock_luma_edge`, 8 lines, two
//! segments) against the scalar reference: both orientations, every reach
//! pair, and segments of different shapes so they take different filters.
use crate::{check::CheckReport, rng::Rng};
use nc_common::PlaneViewMut;
use nc_dsp::Dsp;

/// Plane size and stride (the stride exceeds the width, as in pictures).
const WIDTH: usize = 24;
const HEIGHT: usize = 24;
const STRIDE: usize = 29;
/// The edge's first q0 sample, on both axes.
const EDGE: usize = 8;
const LINES: usize = 8;
const SHAPES: usize = 40;

/// Sample `pos` (0..16 across, edge between 7 and 8) of line `line` of a
/// segment of shape `shape`: flat, steps, ramps, smooth decision lines
/// with unrelated middle lines (per-line masks), or noise.
fn sample(rng: &mut Rng, shape: usize, pos: usize, line: usize) -> u8 {
    let q_side = usize::from(pos >= 8);
    match shape {
        0 => 0,
        1 => 255,
        2..=15 => (32 + shape + line + q_side * (shape - 2)) as u8,
        16..=31 => (128 + (pos as i32 - 8) * (shape as i32 - 16) + line as i32).clamp(0, 255) as u8,
        32..=37 if line.is_multiple_of(4) || line % 4 == 3 => (100 + q_side) as u8,
        _ => rng.byte(),
    }
}

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    short_thresholds(candidate, reference, report);
    for beta in [0, 16, 32, 64, 88] {
        for tc in [0, 1, 4, 16, 50, 100] {
            for shape in 0..SHAPES {
                let shapes = [shape, (shape * 7 + 3) % SHAPES];
                for vertical in [false, true] {
                    let mut plane: Vec<u8> = (0..STRIDE * HEIGHT).map(|_| rng.byte()).collect();
                    for line in 0..LINES {
                        for pos in 0..16 {
                            let value = sample(rng, shapes[line / 4], pos, line);
                            let (x, y) = if vertical {
                                (EDGE - 8 + pos, 4 + line)
                            } else {
                                (4 + line, EDGE - 8 + pos)
                            };
                            plane[y * STRIDE + x] = value;
                        }
                    }
                    let at = if vertical { (EDGE, 4) } else { (4, EDGE) };
                    for lengths in [
                        [1, 1],
                        [2, 2],
                        [3, 3],
                        [3, 5],
                        [3, 7],
                        [5, 3],
                        [5, 5],
                        [5, 7],
                        [7, 3],
                        [7, 5],
                        [7, 7],
                    ] {
                        for lines in [4, 8] {
                            let run = |dsp: &Dsp| {
                                let mut out = plane.clone();
                                let mut view = PlaneViewMut::new(&mut out, STRIDE, WIDTH, HEIGHT)
                                    .expect("check plane");
                                if lines == 8 {
                                    dsp.vvc_deblock_luma_edge(
                                        &mut view, at, vertical, beta, tc, lengths,
                                    );
                                } else {
                                    dsp.vvc_deblock_luma_segment(
                                        &mut view, at, vertical, beta, tc, lengths,
                                    );
                                }
                                out
                            };
                            report.cases += 1;
                            if run(candidate) != run(reference) {
                                report.failures.push(format!(
                                "{} vvc_deblock_luma_edge beta={beta} tc={tc} lengths={lengths:?} shapes={shapes:?} vertical={vertical} lines={lines}",
                                candidate.isa()
                            ));
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Short-only dispatch: strict equality at activity/spread/step thresholds,
/// both segment orders, and rough middle lines under smooth endpoint decisions.
fn short_thresholds(candidate: &Dsp, reference: &Dsp, report: &mut CheckReport) {
    for beta in [0, 4, 5, 16, 32, 88] {
        for tc in [0, 1, 3, 15, 100] {
            let step = ((5 * tc + 1) >> 1).min(100) as u8;
            let patterns = [
                [100; 8],
                [
                    100,
                    100,
                    100,
                    100,
                    100 + step,
                    100 + step,
                    100 + step,
                    100 + step,
                ],
                [100, 100, 101, 100, 100, 100, 100, 100],
                [104, 100, 100, 100, 100, 100, 100, 100],
                [0, 255, 0, 255, 0, 255, 0, 255],
            ];
            for (first, a) in patterns.iter().enumerate() {
                for (second, b) in patterns.iter().enumerate() {
                    for vertical in [false, true] {
                        let mut plane = [71u8; STRIDE * HEIGHT];
                        for line in 0..LINES {
                            let smooth = if line < 4 { a } else { b };
                            // Endpoints decide per segment; the middle lines must
                            // still obey their own normal-filter delta rejection.
                            let rough = [0, 0, 255, 0, 255, 0, 255, 255];
                            let row = if line % 4 == 1 { &rough } else { smooth };
                            for (pos, &value) in row.iter().enumerate() {
                                let (x, y) = if vertical {
                                    (EDGE - 4 + pos, 4 + line)
                                } else {
                                    (4 + line, EDGE - 4 + pos)
                                };
                                plane[y * STRIDE + x] = value;
                            }
                        }
                        let at = if vertical { (EDGE, 4) } else { (4, EDGE) };
                        let run = |dsp: &Dsp| {
                            let mut out = plane;
                            let mut view = PlaneViewMut::new(&mut out, STRIDE, WIDTH, HEIGHT)
                                .expect("check plane");
                            dsp.vvc_deblock_luma_edge(&mut view, at, vertical, beta, tc, [3; 2]);
                            out
                        };
                        report.cases += 1;
                        if run(candidate) != run(reference) {
                            report.failures.push(format!(
                                "{} vvc short thresholds beta={beta} tc={tc} patterns={first}/{second} vertical={vertical}",
                                candidate.isa()
                            ));
                        }
                    }
                }
            }
        }
    }
}
