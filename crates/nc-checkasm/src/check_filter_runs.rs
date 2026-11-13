//! Sparse ordered loop-filter runs: mixed filters, adjacency, gaps, overlapping
//! edges, short tails and exact final-row storage against scalar traversal.

use crate::{check::CheckReport, check_codec::record, rng::Rng};
use nc_common::PlaneViewMut;
use nc_dsp::{Av1EdgeFilter, Av1FilterLength, Dsp, EdgeRun, HevcChromaFilter, HevcLumaFilter};

#[allow(clippy::too_many_lines)]
pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    const SIDE: usize = 64;
    check_av1_groups(candidate, reference, rng, report);
    for vertical in [false, true] {
        for padding in [0, 5] {
            let stride = SIDE + padding;
            for pattern in 0..32 {
                let mut input = vec![0; stride * (SIDE - 1) + SIDE];
                for sample in &mut input {
                    *sample = match pattern % 4 {
                        0 => rng.byte(),
                        1 => 120 + rng.byte() % 8,
                        2 => 128,
                        _ => {
                            if rng.below(2) == 0 {
                                0
                            } else {
                                255
                            }
                        }
                    };
                }
                let mut runs = Vec::new();
                for index in 0..24 {
                    let length = [
                        Av1FilterLength::Four,
                        Av1FilterLength::Six,
                        Av1FilterLength::Eight,
                        Av1FilterLength::Fourteen,
                    ][rng.below(4)];
                    let filter = Av1EdgeFilter {
                        length,
                        limit: 32,
                        blimit: 64,
                        thresh: (index % 4) as u8,
                    };
                    let across = 8 + rng.below(48);
                    let along = rng.below(48);
                    let (x, y) = if vertical {
                        (across, along)
                    } else {
                        (along, across)
                    };
                    runs.push(EdgeRun {
                        x,
                        y,
                        lines: 1 + rng.below(16),
                        filter,
                    });
                }
                // Equal contiguous entries must merge without changing order.
                let filter = runs[0].filter;
                for along in [0, 3, 8, 16] {
                    let (x, y) = if vertical { (8, along) } else { (along, 8) };
                    runs.push(EdgeRun {
                        x,
                        y,
                        lines: if along == 0 {
                            3
                        } else if along == 3 {
                            5
                        } else {
                            8
                        },
                        filter,
                    });
                }
                runs.push(EdgeRun {
                    x: usize::MAX,
                    y: usize::MAX,
                    lines: 0,
                    filter,
                });
                let mut output = [input.clone(), input.clone()];
                for (dsp, samples) in [candidate, reference].into_iter().zip(&mut output) {
                    let mut view = PlaneViewMut::new(samples, stride, SIDE, SIDE).expect("plane");
                    dsp.av1_loop_filter_runs(&mut view, vertical, &runs);
                }
                record(report, candidate.isa(), output[0] == output[1], || {
                    format!(
                        "av1 sparse runs vertical {vertical} padding {padding} pattern {pattern}"
                    )
                });

                for chroma in [false, true] {
                    let mut luma = Vec::new();
                    let mut chroma_runs = Vec::new();
                    for index in 0..32 {
                        let across = 4 + rng.below(57);
                        let along = rng.below(48);
                        let (x, y) = if vertical {
                            (across, along)
                        } else {
                            (along, across)
                        };
                        let tc = if index % 7 == 0 { 0 } else { rng.byte() };
                        luma.push(EdgeRun {
                            x,
                            y,
                            lines: 4 * (1 + rng.below(4)),
                            filter: HevcLumaFilter {
                                beta: rng.byte(),
                                tc,
                            },
                        });
                        chroma_runs.push(EdgeRun {
                            x,
                            y,
                            lines: 2 * (1 + rng.below(8)),
                            filter: HevcChromaFilter { tc },
                        });
                    }
                    for along in [0, 4, 8, 12] {
                        let (x, y) = if vertical { (4, along) } else { (along, 4) };
                        luma.push(EdgeRun {
                            x,
                            y,
                            lines: 4,
                            filter: HevcLumaFilter { beta: 64, tc: 8 },
                        });
                        chroma_runs.push(EdgeRun {
                            x,
                            y,
                            lines: 4,
                            filter: HevcChromaFilter { tc: 8 },
                        });
                    }
                    let mut output = [input.clone(), input.clone()];
                    for (dsp, samples) in [candidate, reference].into_iter().zip(&mut output) {
                        let mut view =
                            PlaneViewMut::new(samples, stride, SIDE, SIDE).expect("plane");
                        if chroma {
                            dsp.hevc_deblock_chroma_runs(&mut view, vertical, &chroma_runs);
                        } else {
                            dsp.hevc_deblock_luma_runs(&mut view, vertical, &luma);
                        }
                    }
                    record(report, candidate.isa(), output[0] == output[1], || {
                        format!(
                            "hevc sparse runs chroma {chroma} vertical {vertical} padding {padding} pattern {pattern}"
                        )
                    });
                }
            }
        }
    }
}

/// Exercise independent decisions in both halves of wide groups, all short
/// tails, flat2 rejection and exact final-row allocations.
fn check_av1_groups(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for vertical in [false, true] {
        for lines in [4, 8, 12, 16, 20, 28, 32, 48, 64] {
            let (width, height) = if vertical { (16, lines) } else { (lines, 16) };
            let (x, y) = if vertical { (8, 0) } else { (0, 8) };
            for padding in [0, 5] {
                let stride = width + padding;
                for length in [
                    Av1FilterLength::Four,
                    Av1FilterLength::Six,
                    Av1FilterLength::Eight,
                    Av1FilterLength::Fourteen,
                ] {
                    for pattern in 0..8 {
                        let mut input = vec![0; stride * (height - 1) + width];
                        for (i, sample) in input.iter_mut().enumerate() {
                            let (row, column) = (i / stride, i % stride);
                            let (along, across) = if vertical {
                                (row, column)
                            } else {
                                (column, row)
                            };
                            *sample = group_sample(pattern, along, across, rng);
                        }
                        let filter = Av1EdgeFilter {
                            length,
                            limit: 32,
                            blimit: 64,
                            thresh: 1,
                        };
                        let mut outputs = [input.clone(), input];
                        {
                            let mut dst = PlaneViewMut::new(&mut outputs[0], stride, width, height)
                                .expect("plane");
                            reference.av1_loop_filter_edge(
                                &mut dst,
                                (x, y),
                                vertical,
                                lines,
                                filter,
                            );
                        }
                        {
                            let mut dst = PlaneViewMut::new(&mut outputs[1], stride, width, height)
                                .expect("plane");
                            candidate.av1_loop_filter_runs(
                                &mut dst,
                                vertical,
                                &[EdgeRun {
                                    x,
                                    y,
                                    lines,
                                    filter,
                                }],
                            );
                        }
                        record(report, candidate.isa(), outputs[0] == outputs[1], || {
                            format!(
                                "av1 groups vertical {vertical} lines {lines} padding {padding} length {length:?} pattern {pattern}"
                            )
                        });
                    }
                }
            }
        }
    }
}

fn group_sample(pattern: usize, along: usize, across: usize, rng: &mut Rng) -> u8 {
    match pattern {
        0 => 128,
        1 => rng.byte(),
        2 => {
            if along % 16 < 8 {
                128
            } else {
                rng.byte()
            }
        }
        3 => {
            if along % 16 >= 8 {
                128
            } else {
                rng.byte()
            }
        }
        4 => 127 + (across % 3) as u8,
        5 => {
            if (2..14).contains(&across) {
                128
            } else {
                132
            }
        }
        6 => {
            if along % 16 < 8 && (!(2..14).contains(&across)) {
                132
            } else {
                128
            }
        }
        _ => {
            if across < 8 {
                120
            } else {
                128
            }
        }
    }
}
