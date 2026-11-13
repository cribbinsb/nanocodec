//! VP9 masked edge lines against scalar batching and individual SIMD edges,
//! including all final-segment lengths and exact picture footprints.

use crate::{check::CheckReport, check_codec::record, rng::Rng};
use nc_common::PlaneViewMut;
use nc_dsp::{Dsp, VP9_EDGE_LINES, Vp9EdgeFilter, Vp9FilterLimits, Vp9FilterWidth};

const SIDE: usize = 80;
const STRIDE: usize = SIDE + 3;
const START: usize = 3;
const WIDTHS: [Vp9FilterWidth; 3] = [
    Vp9FilterWidth::Four,
    Vp9FilterWidth::Eight,
    Vp9FilterWidth::Sixteen,
];

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for vertical in [false, true] {
        for count in [0, 1, 2, 3, 8] {
            for tail in 1..=VP9_EDGE_LINES {
                let lines = if count == 0 {
                    0
                } else {
                    (count - 1) * VP9_EDGE_LINES + tail
                };
                let (w, h) = if vertical {
                    (SIDE, START + lines)
                } else {
                    (START + lines, SIDE)
                };
                for mode in 0..4 {
                    let equal = Vp9EdgeFilter {
                        width: WIDTHS[rng.below(3)],
                        limits: Vp9FilterLimits::new(32),
                    };
                    let segments: Vec<_> = (0..count)
                        .map(|i| match mode {
                            0 => None,
                            2 if i % 3 == 0 => None,
                            1 | 2 => Some(equal),
                            _ => Some(Vp9EdgeFilter {
                                width: WIDTHS[rng.below(3)],
                                limits: Vp9FilterLimits::with_sharpness(
                                    [0, 1, 63, 84, 255][rng.below(5)],
                                    rng.below(8) as u8,
                                ),
                            }),
                        })
                        .collect();
                    let reach = segments
                        .iter()
                        .flatten()
                        .map(|f| {
                            if f.width == Vp9FilterWidth::Sixteen {
                                8
                            } else {
                                4
                            }
                        })
                        .max()
                        .unwrap_or(1);
                    for offset in [reach, SIDE - reach] {
                        let origin = if vertical {
                            (offset, START)
                        } else {
                            (START, offset)
                        };
                        for pattern in 0..4 {
                            let input: Vec<u8> = (0..(h - 1) * STRIDE + w + 2)
                                .map(|i| match pattern {
                                    0 => rng.byte(),
                                    1 => 120 + (i % 3) as u8,
                                    2 => 128,
                                    _ => {
                                        if i % 2 == 0 {
                                            0
                                        } else {
                                            255
                                        }
                                    }
                                })
                                .collect();
                            let run = |dsp: &Dsp, batch| {
                                run_case(
                                    dsp,
                                    &input,
                                    (w, h, lines),
                                    origin,
                                    vertical,
                                    &segments,
                                    batch,
                                )
                            };
                            let actual = run(candidate, true);
                            let scalar = run(reference, true);
                            let singles = run(candidate, false);
                            record(
                                report,
                                candidate.isa(),
                                actual == scalar && actual == singles,
                                || {
                                    format!(
                                        "VP9 line vertical={vertical} count={count} tail={tail} mode={mode} offset={offset} pattern={pattern}"
                                    )
                                },
                            );
                        }
                    }
                }
            }
        }
    }
}

fn run_case(
    dsp: &Dsp,
    input: &[u8],
    (w, h, lines): (usize, usize, usize),
    origin: (usize, usize),
    vertical: bool,
    segments: &[Option<Vp9EdgeFilter>],
    batch: bool,
) -> Vec<u8> {
    let mut out = input.to_vec();
    let len = out.len();
    let mut dst = PlaneViewMut::new(&mut out[1..len - 1], STRIDE, w, h).expect("plane");
    if batch {
        dsp.vp9_loop_filter_line(&mut dst, origin, vertical, segments);
    } else {
        let (across, along) = if vertical { (1, STRIDE) } else { (STRIDE, 1) };
        for (i, filter) in segments.iter().enumerate() {
            if let Some(filter) = filter {
                let start = i * VP9_EDGE_LINES;
                dsp.vp9_filter_edge(
                    dst.data_mut(),
                    origin.1 * STRIDE + origin.0 + start * along,
                    across,
                    along,
                    VP9_EDGE_LINES.min(lines - start),
                    filter.width,
                    filter.limits,
                );
            }
        }
    }
    out
}
