//! Batched deblocking against the existing single-edge API, with identical
//! fresh samples and the full suite's timer. One call includes the same lines.
use crate::bench::{BenchResult, time};
use nc_common::PlaneViewMut;
use nc_dsp::{Dsp, EdgeRun, HevcLumaFilter};

pub(crate) fn bench(dsp: &Dsp, out: &mut Vec<BenchResult>) {
    bench_h264_vvc(dsp, out);
    bench_h264_mb(dsp, out);
    for vertical in [false, true] {
        for strong in [false, true] {
            let names = match (vertical, strong) {
                (false, false) => ["hevc_batch_weak_h", "hevc_single_weak_h"],
                (false, true) => ["hevc_batch_strong_h", "hevc_single_strong_h"],
                (true, false) => ["hevc_batch_weak_v", "hevc_single_weak_v"],
                (true, true) => ["hevc_batch_strong_v", "hevc_single_strong_v"],
            };
            for lines in [8, 16, 32, 64] {
                let (width, height) = if vertical { (8, lines) } else { (lines, 8) };
                let stride = width + 5;
                let mut input = vec![0u8; stride * (height - 1) + width];
                for row in 0..height {
                    for col in 0..width {
                        let across = if vertical { col } else { row };
                        input[row * stride + col] = 120
                            + u8::from(across >= 4) * 4
                            + if strong { 0 } else { (across % 4 * 3) as u8 };
                    }
                }
                let mut output = input.clone();
                let at = if vertical { (4, 0) } else { (0, 4) };
                let run = EdgeRun {
                    x: at.0,
                    y: at.1,
                    lines,
                    filter: HevcLumaFilter { beta: 64, tc: 8 },
                };
                for (single, kernel) in names.into_iter().enumerate() {
                    let ns_per_call = time(|| {
                        output.copy_from_slice(&input);
                        let mut view =
                            PlaneViewMut::new(&mut output, stride, width, height).expect("plane");
                        if single == 0 {
                            dsp.hevc_deblock_luma_runs(&mut view, vertical, &[run]);
                        } else {
                            for start in (0..lines).step_by(8) {
                                let pos = if vertical {
                                    (at.0, start)
                                } else {
                                    (start, at.1)
                                };
                                dsp.hevc_deblock_luma_edge(&mut view, pos, vertical, [8; 2], 64);
                            }
                        }
                        u32::from(output[at.1 * stride + at.0])
                    });
                    out.push(BenchResult {
                        kernel,
                        size: (8, lines),
                        isa: dsp.isa(),
                        ns_per_call,
                    });
                }
            }
        }
    }
}

/// Independent parallel edges in each direction; all methods include the
/// same fresh-input copy and validated public API cost.
#[allow(
    clippy::too_many_lines,
    reason = "keep the paired single/batch benchmark fixture together"
)]
fn bench_h264_vvc(dsp: &Dsp, out: &mut Vec<BenchResult>) {
    use nc_dsp::{H264EdgeParams, VvcLumaFilter};
    for vertical in [false, true] {
        for strong in [false, true] {
            for count in [1, 2, 4] {
                const SIDE: usize = 48;
                let stride = SIDE + 5;
                let mut input = vec![0; stride * SIDE];
                for y in 0..SIDE {
                    for x in 0..SIDE {
                        let across = if vertical { x } else { y };
                        input[y * stride + x] = 120
                            + (across / 8 % 2 * 4) as u8
                            + if strong { 0 } else { (across % 4 * 2) as u8 };
                    }
                }
                let h264: Vec<_> = (0..count)
                    .map(|i| EdgeRun {
                        x: if vertical { 8 + i * 8 } else { 0 },
                        y: if vertical { 0 } else { 8 + i * 8 },
                        lines: 16,
                        filter: H264EdgeParams {
                            bs: [if strong { 4 } else { 2 }; 4],
                            tc0: [8; 4],
                            alpha: 255,
                            beta: 18,
                        },
                    })
                    .collect();
                let vvc: Vec<_> = h264
                    .iter()
                    .map(|r| EdgeRun {
                        x: r.x,
                        y: r.y,
                        lines: 8,
                        filter: VvcLumaFilter {
                            beta: 88,
                            tc: 15,
                            lengths: [3; 2],
                        },
                    })
                    .collect();
                let mut output = input.clone();
                for codec in ["h264", "vvc"] {
                    let names = match (codec, vertical, strong) {
                        ("h264", false, false) => ["h264_batch_weak_h", "h264_single_weak_h"],
                        ("h264", false, true) => ["h264_batch_strong_h", "h264_single_strong_h"],
                        ("h264", true, false) => ["h264_batch_weak_v", "h264_single_weak_v"],
                        ("h264", true, true) => ["h264_batch_strong_v", "h264_single_strong_v"],
                        (_, false, false) => ["vvc_batch_weak_h", "vvc_single_weak_h"],
                        (_, false, true) => ["vvc_batch_strong_h", "vvc_single_strong_h"],
                        (_, true, false) => ["vvc_batch_weak_v", "vvc_single_weak_v"],
                        (_, true, true) => ["vvc_batch_strong_v", "vvc_single_strong_v"],
                    };
                    for (single, kernel) in names.into_iter().enumerate() {
                        let ns_per_call = time(|| {
                            output.copy_from_slice(&input);
                            let mut dst =
                                PlaneViewMut::new(&mut output, stride, SIDE, SIDE).expect("plane");
                            if codec == "h264" {
                                if single == 0 {
                                    dsp.h264_deblock_luma_runs(&mut dst, vertical, &h264);
                                } else {
                                    for r in &h264 {
                                        let p = r.filter;
                                        dsp.h264_deblock_luma_edge(
                                            &mut dst,
                                            (r.x, r.y),
                                            vertical,
                                            p.bs,
                                            p.tc0,
                                            (p.alpha, p.beta),
                                        );
                                    }
                                }
                            } else if single == 0 {
                                dsp.vvc_deblock_luma_runs(&mut dst, vertical, &vvc);
                            } else {
                                for r in &vvc {
                                    dsp.vvc_deblock_luma_edge(
                                        &mut dst,
                                        (r.x, r.y),
                                        vertical,
                                        88,
                                        15,
                                        [3; 2],
                                    );
                                }
                            }
                            u32::from(output[8])
                        });
                        out.push(BenchResult {
                            kernel,
                            size: (if codec == "h264" { 16 } else { 8 }, count),
                            isa: dsp.isa(),
                            ns_per_call,
                        });
                    }
                }
            }
        }
    }
}

/// The encoder/decoder macroblock entry, including ordered dependent edges.
fn bench_h264_mb(dsp: &Dsp, out: &mut Vec<BenchResult>) {
    use nc_dsp::{H264EdgeParams, H264MbEdges};
    const SIDE: usize = 48;
    for strong in [false, true] {
        for paired in [false, true] {
            let mask = if paired { 0b0101_0101u8 } else { 0xff };
            let p = H264EdgeParams {
                bs: [if strong { 4 } else { 2 }; 4],
                tc0: [8; 4],
                alpha: 255,
                beta: 18,
            };
            let edges = H264MbEdges {
                mask,
                edges: [[p; 4]; 2],
            };
            let input: Vec<_> = (0..SIDE * SIDE)
                .map(|i| 120 + if strong { 0 } else { (i % SIDE % 4 * 2) as u8 })
                .collect();
            let mut output = input.clone();
            let names = match (paired, strong) {
                (false, false) => ["h264_mb_dense_weak", "h264_edges_dense_weak"],
                (false, true) => ["h264_mb_dense_strong", "h264_edges_dense_strong"],
                (true, false) => ["h264_mb_pair_weak", "h264_edges_pair_weak"],
                (true, true) => ["h264_mb_pair_strong", "h264_edges_pair_strong"],
            };
            for (single, kernel) in names.into_iter().enumerate() {
                let ns_per_call = time(|| {
                    output.copy_from_slice(&input);
                    let mut dst = PlaneViewMut::new(&mut output, SIDE, SIDE, SIDE).expect("plane");
                    if single == 0 {
                        dsp.h264_deblock_mb_luma(&mut dst, (8, 8), &edges);
                    } else {
                        for direction in 0..2 {
                            for edge in 0..4 {
                                if mask & (1 << (4 * direction + edge)) != 0 {
                                    let vertical = direction == 0;
                                    let at = if vertical {
                                        (8 + 4 * edge, 8)
                                    } else {
                                        (8, 8 + 4 * edge)
                                    };
                                    dsp.h264_deblock_luma_edge(
                                        &mut dst,
                                        at,
                                        vertical,
                                        p.bs,
                                        p.tc0,
                                        (p.alpha, p.beta),
                                    );
                                }
                            }
                        }
                    }
                    u32::from(output[8 * SIDE + 8])
                });
                out.push(BenchResult {
                    kernel,
                    size: (16, mask.count_ones() as usize),
                    isa: dsp.isa(),
                    ns_per_call,
                });
            }
        }
    }
}
