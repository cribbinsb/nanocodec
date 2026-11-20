//! Sparse ordered loop-filter runs: mixed filters, adjacency, gaps, overlapping
//! edges, short tails and exact final-row storage against scalar traversal.

use crate::{check::CheckReport, check_codec::record, rng::Rng};
use nc_common::PlaneViewMut;
use nc_dsp::{Av1EdgeFilter, Av1FilterLength, Dsp, EdgeRun, HevcChromaFilter, HevcLumaFilter};

#[allow(clippy::too_many_lines)]
pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    const SIDE: usize = 64;
    check_h264_groups(candidate, reference, rng, report);
    check_av1_groups(candidate, reference, rng, report);
    check_hevc_groups(candidate, reference, rng, report);
    check_vvc_groups(candidate, reference, rng, report);
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
                    dsp.av1_loop_filter_runs(
                        &mut view,
                        vertical,
                        &runs,
                        &mut nc_dsp::LoopFilterLanes::default(),
                    );
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
        for lines in [4, 8, 12, 16, 20, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64] {
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
                    for pattern in 0..12 {
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
                                &mut nc_dsp::LoopFilterLanes::default(),
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
        8..=11 => {
            if along / 8 % 4 == pattern - 8 {
                128
            } else {
                rng.byte()
            }
        }
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

/// Different endpoint decisions in each 128-bit tile, with exact final-row
/// extents and untouched leading/trailing/padding bytes, including 32-line batches.
fn check_hevc_groups(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for vertical in [false, true] {
        for lines in [4, 8, 12, 16, 20, 28, 32, 36, 48, 60, 64, 68] {
            let (width, height) = if vertical { (8, lines) } else { (lines, 8) };
            let (x, y) = if vertical { (4, 0) } else { (0, 4) };
            for padding in [0, 5] {
                let stride = width + padding;
                let len = stride * (height - 1) + width;
                for pattern in 0..12 {
                    let mut input = vec![0xa7; 3 + len + 11];
                    for row in 0..height {
                        for col in 0..width {
                            let (along, across) = if vertical { (row, col) } else { (col, row) };
                            input[3 + row * stride + col] =
                                group_sample(pattern, along, across, rng);
                        }
                    }
                    for (beta, tc) in [
                        (0u8, 0u8),
                        (64, 8),
                        (1, 255),
                        (255, 255),
                        (255, 1),
                        (255, 0),
                    ] {
                        let mut outputs = [input.clone(), input.clone()];
                        for (dsp, data) in [candidate, reference].into_iter().zip(&mut outputs) {
                            let mut view =
                                PlaneViewMut::new(&mut data[3..3 + len], stride, width, height)
                                    .expect("plane");
                            let step = if pattern % 3 == 0 { 4 } else { 8 };
                            let runs: Vec<_> = (0..lines)
                                .step_by(step)
                                .map(|start| EdgeRun {
                                    x: x + if vertical { 0 } else { start },
                                    y: y + if vertical { start } else { 0 },
                                    lines: (lines - start).min(step),
                                    filter: HevcLumaFilter {
                                        beta: beta.saturating_sub((start % 13) as u8),
                                        tc: tc.saturating_sub((start % 7) as u8),
                                    },
                                })
                                .collect();
                            dsp.hevc_deblock_luma_runs(&mut view, vertical, &runs);
                        }
                        record(report, candidate.isa(), outputs[0] == outputs[1], || {
                            format!(
                                "hevc batch vertical={vertical} lines={lines} padding={padding} pattern={pattern} beta={beta} tc={tc}"
                            )
                        });
                    }
                }
            }
        }
    }
}

/// The oracle walks the original single-edge/segment APIs, independently of
/// the new run/tile iterator. Overlaps and long/narrow visits are fences.
#[allow(
    clippy::too_many_lines,
    reason = "keep the adversarial fixture and independent oracle together"
)]
fn check_vvc_groups(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    use nc_dsp::VvcLumaFilter;
    const SIDE: usize = 96;
    for vertical in [false, true] {
        for padding in [0, 5] {
            let stride = SIDE + padding;
            let len = stride * (SIDE - 1) + SIDE;
            for pattern in 0..12 {
                let mut input = vec![0xa7; 3 + len + 11];
                for y in 0..SIDE {
                    for x in 0..SIDE {
                        let (along, across) = if vertical { (y, x) } else { (x, y) };
                        input[3 + y * stride + x] = match pattern % 6 {
                            0 => 128,
                            1 => 120 + (across / 4 % 2 * 4) as u8,
                            2 => 120 + (across % 4 * 3) as u8,
                            3 => rng.byte(),
                            4 => {
                                if along / 8 % 4 == pattern / 6 {
                                    rng.byte()
                                } else {
                                    128
                                }
                            }
                            _ => {
                                if rng.below(2) == 0 {
                                    0
                                } else {
                                    255
                                }
                            }
                        };
                    }
                }
                for (beta, tc) in [(88, 15), (88, 100), (0, 0)] {
                    let filter = VvcLumaFilter {
                        beta,
                        tc,
                        lengths: [3; 2],
                    };
                    let make = |across, along, lines, lengths| {
                        let (x, y) = if vertical {
                            (across, along)
                        } else {
                            (along, across)
                        };
                        EdgeRun {
                            x,
                            y,
                            lines,
                            filter: VvcLumaFilter { lengths, ..filter },
                        }
                    };
                    let mut runs = vec![make(16, 0, 64, [3; 2])];
                    for across in [16, 24, 32, 40] {
                        let mut run = make(across, 64, 8, [3; 2]);
                        run.filter.beta = beta.saturating_sub((across / 8) as u8);
                        run.filter.tc = tc.saturating_sub((across / 16) as u8);
                        runs.push(run);
                    }
                    // An overlapping edge must see the preceding results.
                    runs.push(make(36, 64, 8, [3; 2]));
                    runs.push(make(36, 64, 8, [3; 2]));
                    runs.push(make(48, 64, 4, [7, 3]));
                    runs.push(make(48, 68, 8, [5, 5]));
                    runs.push(make(48, 76, 4, [1, 1]));
                    runs.push(make(56, 64, 8, [2, 2]));
                    runs.push(EdgeRun {
                        x: usize::MAX,
                        y: usize::MAX,
                        lines: 0,
                        filter,
                    });
                    let mut outputs = [input.clone(), input.clone()];
                    {
                        let mut view =
                            PlaneViewMut::new(&mut outputs[0][3..3 + len], stride, SIDE, SIDE)
                                .expect("plane");
                        candidate.vvc_deblock_luma_runs(&mut view, vertical, &runs);
                    }
                    {
                        let mut view =
                            PlaneViewMut::new(&mut outputs[1][3..3 + len], stride, SIDE, SIDE)
                                .expect("plane");
                        for run in &runs {
                            for start in (0..run.lines).step_by(8) {
                                let at = if vertical {
                                    (run.x, run.y + start)
                                } else {
                                    (run.x + start, run.y)
                                };
                                let lengths = run.filter.lengths.map(usize::from);
                                if run.lines - start == 4 {
                                    reference.vvc_deblock_luma_segment(
                                        &mut view,
                                        at,
                                        vertical,
                                        i32::from(run.filter.beta),
                                        i32::from(run.filter.tc),
                                        lengths,
                                    );
                                } else {
                                    reference.vvc_deblock_luma_edge(
                                        &mut view,
                                        at,
                                        vertical,
                                        i32::from(run.filter.beta),
                                        i32::from(run.filter.tc),
                                        lengths,
                                    );
                                }
                            }
                        }
                    }
                    record(report, candidate.isa(), outputs[0] == outputs[1], || {
                        format!(
                            "vvc batch vertical={vertical} padding={padding} pattern={pattern} beta={beta} tc={tc}"
                        )
                    });
                }
            }
        }
    }
}

/// Original edge calls are the oracle: pairs with disjoint footprints may
/// batch, but overlapping/dependent edges must read the previous result.
fn check_h264_groups(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    use nc_dsp::H264EdgeParams;
    const SIDE: usize = 80;
    for vertical in [false, true] {
        for padding in [0, 5] {
            let stride = SIDE + padding;
            let len = stride * (SIDE - 1) + SIDE;
            for pattern in 0..12 {
                let mut input = vec![0xa7; 3 + len + 11];
                for y in 0..SIDE {
                    for x in 0..SIDE {
                        let (along, across) = if vertical { (y, x) } else { (x, y) };
                        input[3 + y * stride + x] = group_sample(pattern, along, across, rng);
                    }
                }
                for strengths in [[2; 4], [4; 4], [0, 1, 3, 4], [0; 4]] {
                    let mut runs = Vec::new();
                    for (i, (across, along)) in [
                        (8, 0),
                        (16, 0),
                        (20, 0),
                        (20, 0),
                        (32, 16),
                        (32, 32),
                        (32, 48),
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        let (x, y) = if vertical {
                            (across, along)
                        } else {
                            (along, across)
                        };
                        runs.push(EdgeRun {
                            x,
                            y,
                            lines: 16,
                            filter: H264EdgeParams {
                                bs: strengths,
                                tc0: [0, 1 + i as u8, 13, 25],
                                alpha: 255 - 7 * i as u8,
                                beta: 18 - i as u8,
                            },
                        });
                    }
                    runs.push(EdgeRun {
                        x: usize::MAX,
                        y: usize::MAX,
                        lines: 0,
                        filter: H264EdgeParams::default(),
                    });
                    let mut outputs = [input.clone(), input.clone()];
                    {
                        let mut dst =
                            PlaneViewMut::new(&mut outputs[0][3..3 + len], stride, SIDE, SIDE)
                                .expect("plane");
                        candidate.h264_deblock_luma_runs(&mut dst, vertical, &runs);
                    }
                    {
                        let mut dst =
                            PlaneViewMut::new(&mut outputs[1][3..3 + len], stride, SIDE, SIDE)
                                .expect("plane");
                        for run in &runs {
                            if run.lines == 0 {
                                continue;
                            }
                            let p = run.filter;
                            reference.h264_deblock_luma_edge(
                                &mut dst,
                                (run.x, run.y),
                                vertical,
                                p.bs,
                                p.tc0,
                                (p.alpha, p.beta),
                            );
                        }
                    }
                    record(report, candidate.isa(), outputs[0] == outputs[1], || {
                        format!(
                            "h264 batch vertical={vertical} padding={padding} pattern={pattern} bs={strengths:?}"
                        )
                    });
                }
            }
        }
    }
}
