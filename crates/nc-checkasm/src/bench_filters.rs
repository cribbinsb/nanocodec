//! Filter benchmarks with fresh input each call, so repeated in-place filtering
//! cannot turn an active edge into a converged no-op.

use nc_common::{PlaneView, PlaneViewMut};
use nc_dsp::{
    AV1_EDGE_LINES, Av1EdgeFilter, Av1FilterLength, Av2EdgeFilter, BlockSize, CdefSource, Dsp,
};

use crate::bench::time;

const SIDE: usize = 32;
const AREA: usize = SIDE * SIDE;
const EDGE: (usize, usize) = (8, 8);
const RECT: (usize, usize, usize, usize) = (1, 1, SIDE - 1, SIDE - 1);

pub(crate) fn bench_filters(dsp: &Dsp, push: &mut impl FnMut(&'static str, BlockSize, f64)) {
    bench_vvc_alf(dsp, push);
    bench_cdef_picture_stride(dsp, push);
    bench_av1_lines(dsp, push);
    crate::bench_av1_grid::bench(dsp, push);
    crate::bench_vp9_line::bench(dsp, push);
    // A shallow edge plus low noise keeps both narrow and strong filters active.
    let input: [u8; AREA] =
        core::array::from_fn(|i| 120 + u8::from(i % SIDE >= EDGE.0) * 4 + (i % 3) as u8);
    let reference: [u8; AREA] = core::array::from_fn(|i| input[i].saturating_add((i % 5) as u8));
    let src = PlaneView::new(&input, SIDE, SIDE, SIDE).expect("plane");
    let rf = PlaneView::new(&reference, SIDE, SIDE, SIDE).expect("plane");
    let block = BlockSize::new(8, 8).expect("block");
    push(
        "cdef_find_dir",
        block,
        time(|| u32::from(dsp.cdef_find_dir(&src, EDGE).0)),
    );
    let mut padded = CdefSource::new();
    padded.fill_from(&src);
    for side in [4, 8] {
        let size = BlockSize::new(side, side).expect("block");
        let mut output = input;
        push(
            "cdef_filter",
            size,
            time(|| {
                let mut dst = PlaneViewMut::new(&mut output, SIDE, SIDE, SIDE).expect("plane");
                dsp.cdef_filter(side, &padded, EDGE, (8, 2), 3, 4, &mut dst);
                u32::from(output[EDGE.1 * SIDE + EDGE.0])
            }),
        );
    }
    for side in [4, 8] {
        let size = BlockSize::new(side, side).expect("block");
        for (name, strengths) in [
            ("cdef_filter_u8", (8, 2)),
            ("cdef_filter_u8_pri", (15, 0)),
            ("cdef_filter_u8_sec", (0, 4)),
        ] {
            let mut output = input;
            push(
                name,
                size,
                time(|| {
                    let mut dst = PlaneViewMut::new(&mut output, SIDE, SIDE, SIDE).expect("plane");
                    dsp.cdef_filter_interior(side, &src, EDGE, strengths, 3, 4, &mut dst);
                    u32::from(output[EDGE.1 * SIDE + EDGE.0])
                }),
            );
        }
    }
    let frame = BlockSize::new(SIDE, SIDE).expect("block");
    push(
        "sao_edge_stats",
        frame,
        time(|| dsp.sao_edge_stats(2, &rf, &src, RECT).count[0] as u32),
    );
    let mut output = input;
    push(
        "sao_edge_apply",
        frame,
        time(|| {
            let mut dst = PlaneViewMut::new(&mut output, SIDE, SIDE, SIDE).expect("plane");
            dsp.sao_edge_apply(2, [2, 1, -1, -2], &src, &mut dst, RECT);
            u32::from(output[SIDE + 1])
        }),
    );
    for masked in [false, true] {
        let exempt: Vec<bool> = (0..(RECT.2 - RECT.0) * (RECT.3 - RECT.1))
            .map(|i| i % 7 == 0)
            .collect();
        push(
            if masked {
                "sao_band_masked"
            } else {
                "sao_band_apply"
            },
            frame,
            time(|| {
                output.copy_from_slice(&input);
                let mut dst = PlaneViewMut::new(&mut output, SIDE, SIDE, SIDE).expect("plane");
                dsp.sao_band_apply_masked(
                    13,
                    [3, -3, 7, -7],
                    &mut dst,
                    RECT,
                    if masked { &exempt } else { &[] },
                );
                u32::from(output[SIDE + 1])
            }),
        );
    }
    bench_edges(dsp, &input, push);
    bench_av2_lines(dsp, push);
}

/// A whole H.264 luma macroblock, every edge filtered (bS 2; the
/// macroblock edges bS 4 in the second run, as for intra).
fn bench_h264_mb(
    dsp: &Dsp,
    input: &[u8; AREA],
    push: &mut impl FnMut(&'static str, BlockSize, f64),
) {
    let mut output = *input;
    let macroblock = BlockSize::new(16, 16).expect("block");
    for (name, outer) in [
        ("h264_deblock_mb_luma", 2),
        ("h264_deblock_mb_luma_intra", 4),
    ] {
        let edge = |e: usize| nc_dsp::H264EdgeParams {
            bs: [if e == 0 { outer } else { 2 }; 4],
            tc0: [3; 4],
            alpha: 40,
            beta: 20,
        };
        let edges = nc_dsp::H264MbEdges {
            mask: u8::MAX,
            edges: [core::array::from_fn(edge), core::array::from_fn(edge)],
        };
        push(
            name,
            macroblock,
            time(|| {
                output.copy_from_slice(input);
                let mut dst = PlaneViewMut::new(&mut output, SIDE, SIDE, SIDE).expect("plane");
                dsp.h264_deblock_mb_luma(&mut dst, EDGE, &edges);
                u32::from(output[EDGE.1 * SIDE + EDGE.0])
            }),
        );
    }
}

fn bench_hevc_chroma4(
    dsp: &Dsp,
    input: &[u8; AREA],
    vertical: bool,
    push: &mut impl FnMut(&'static str, BlockSize, f64),
) {
    let mut output = *input;
    push(
        if vertical {
            "hevc_deblock_chroma4_v"
        } else {
            "hevc_deblock_chroma4_h"
        },
        BlockSize::new(4, 4).expect("block"),
        time(|| {
            output.copy_from_slice(input);
            let mut dst = PlaneViewMut::new(&mut output, SIDE, SIDE, SIDE).expect("plane");
            dsp.hevc_deblock_chroma_edge(&mut dst, EDGE, vertical, 4, [3; 4]);
            u32::from(output[EDGE.1 * SIDE + EDGE.0])
        }),
    );
}

fn bench_edges(dsp: &Dsp, input: &[u8; AREA], push: &mut impl FnMut(&'static str, BlockSize, f64)) {
    let block = BlockSize::new(8, 8).expect("block");
    let mut output = *input;
    bench_h264_mb(dsp, input, push);
    for vertical in [false, true] {
        let names = if vertical {
            [
                "h264_deblock_luma_v",
                "h264_deblock_chroma_v",
                "hevc_deblock_luma_v",
                "hevc_deblock_chroma_v",
            ]
        } else {
            [
                "h264_deblock_luma_h",
                "h264_deblock_chroma_h",
                "hevc_deblock_luma_h",
                "hevc_deblock_chroma_h",
            ]
        };
        for (operation, name) in names.into_iter().enumerate() {
            push(
                name,
                block,
                time(|| {
                    output.copy_from_slice(input);
                    let mut dst = PlaneViewMut::new(&mut output, SIDE, SIDE, SIDE).expect("plane");
                    match operation {
                        0 => dsp.h264_deblock_luma_edge(
                            &mut dst,
                            EDGE,
                            vertical,
                            [2; 4],
                            [3; 4],
                            (40, 20),
                        ),
                        1 => dsp.h264_deblock_chroma_edge(
                            &mut dst,
                            EDGE,
                            vertical,
                            [2; 4],
                            [3; 4],
                            (40, 20),
                        ),
                        2 => dsp.hevc_deblock_luma_edge(&mut dst, EDGE, vertical, [3; 2], 40),
                        _ => dsp.hevc_deblock_chroma_edge(&mut dst, EDGE, vertical, 8, [3; 4]),
                    }
                    u32::from(output[EDGE.1 * SIDE + EDGE.0])
                }),
            );
        }
        bench_hevc_chroma4(dsp, input, vertical, push);
        let names = if vertical {
            [
                "av1_deblock4_v",
                "av1_deblock6_v",
                "av1_deblock8_v",
                "av1_deblock14_v",
            ]
        } else {
            [
                "av1_deblock4_h",
                "av1_deblock6_h",
                "av1_deblock8_h",
                "av1_deblock14_h",
            ]
        };
        for (name, length) in names.into_iter().zip([
            Av1FilterLength::Four,
            Av1FilterLength::Six,
            Av1FilterLength::Eight,
            Av1FilterLength::Fourteen,
        ]) {
            let filter = Av1EdgeFilter {
                length,
                limit: 20,
                blimit: 40,
                thresh: 2,
            };
            for lines in [4, 8, 16] {
                push(
                    name,
                    BlockSize::new(8, lines).expect("filter footprint"),
                    time(|| {
                        output.copy_from_slice(input);
                        let mut dst =
                            PlaneViewMut::new(&mut output, SIDE, SIDE, SIDE).expect("plane");
                        dsp.av1_loop_filter_edge(&mut dst, EDGE, vertical, lines, filter);
                        u32::from(output[EDGE.1 * SIDE + EDGE.0])
                    }),
                );
            }
        }
    }
}

/// Edge position of the AV2 edge-line benchmark: eight segments of four
/// lines span the plane, with the decision's reach of 8 on each side.
const AV2_COORDINATE: usize = SIDE / 2;
/// Step across the AV2 benchmark edge and the amplitude of its noise:
/// smooth sides take the widest ramps, rough ones (curvature above the
/// side threshold) exit the decision at its first test.
const AV2_CONTENTS: [(&str, &str, u8, u8); 2] = [
    ("av2_deblock_line_h", "av2_deblock_line_v", 6, 3),
    ("av2_deblock_rough_h", "av2_deblock_rough_v", 12, 48),
];

/// One whole AV2 edge line (8 segments, one batched decision and chunk of
/// ramps) per call, both orientations.
fn bench_av2_lines(dsp: &Dsp, push: &mut impl FnMut(&'static str, BlockSize, f64)) {
    let line = BlockSize::new(SIDE, SIDE).expect("block");
    let filter = Av2EdgeFilter {
        length_neg: 18,
        length_pos: 18,
        q_threshold: 40,
        side_threshold: 60,
    };
    let filters = [Some(filter); SIDE / 4];
    for (name_h, name_v, step, noise) in AV2_CONTENTS {
        for vertical in [false, true] {
            let input: [u8; AREA] = core::array::from_fn(|i| {
                let across = if vertical { i % SIDE } else { i / SIDE };
                100 + u8::from(across >= AV2_COORDINATE) * step + (i * 37 % 61) as u8 % noise
            });
            let mut output = input;
            push(
                if vertical { name_v } else { name_h },
                line,
                time(|| {
                    output.copy_from_slice(&input);
                    let mut dst = PlaneViewMut::new(&mut output, SIDE, SIDE, SIDE).expect("plane");
                    dsp.av2_deblock_line(&mut dst, AV2_COORDINATE, vertical, &filters);
                    u32::from(output[AV2_COORDINATE * SIDE + AV2_COORDINATE])
                }),
            );
        }
    }
}

/// VVC ALF on a 64x4 block row of a CTB (four filter rows; two Laplacian
/// grid rows, reported as 64x4 too), and the statistics of 512 samples
/// (a 64x64 CTB sampled every 2 rows and 4 columns, reported as 64x8).
fn bench_vvc_alf(dsp: &Dsp, push: &mut impl FnMut(&'static str, BlockSize, f64)) {
    const WIDTH: usize = 64;
    const SAMPLES: usize = 512;
    let data: Vec<Vec<u8>> = (0..7)
        .map(|r| {
            (0..WIDTH + 8)
                .map(|x| (120 + (x * 7 + r * 13) % 23) as u8)
                .collect()
        })
        .collect();
    let rows: [&[u8]; 7] = std::array::from_fn(|r| data[r].as_slice());
    let coefficients: Vec<[i16; nc_dsp::VVC_ALF_TAPS]> = (0..WIDTH / 4)
        .map(|b| std::array::from_fn(|k| ((b + 3 * k) % 17) as i16 - 8))
        .collect();
    let block_row = BlockSize::new(WIDTH, 4).expect("block");
    let mut out = [[0; WIDTH]; nc_dsp::VVC_ALF_BLOCK];
    push(
        "vvc_alf_filter_block_row",
        block_row,
        time(|| {
            dsp.vvc_alf_filter_block_row(
                &[rows; nc_dsp::VVC_ALF_BLOCK],
                &coefficients,
                &[7; nc_dsp::VVC_ALF_BLOCK],
                &mut out.each_mut().map(<[u8; WIDTH]>::as_mut_slice),
            );
            u32::from(out[1][5])
        }),
    );
    let grid: [&[u8]; 4] = std::array::from_fn(|r| rows[r]);
    let mut groups = [[0; 4]; WIDTH / 4 + 1];
    push(
        "vvc_alf_laplacians",
        block_row,
        time(|| {
            for _ in 0..2 {
                dsp.vvc_alf_laplacians(&grid, &mut groups);
            }
            u32::from(groups[3][1])
        }),
    );
    let mut sums = [[0; nc_dsp::VVC_ALF_SAMPLE_LANES]; nc_dsp::VVC_ALF_SAMPLES];
    let transposes = std::array::from_fn(|i| (i % 4) as u8);
    push(
        "vvc_alf_samples",
        BlockSize::new(WIDTH, 4).expect("block"),
        time(|| {
            // Two sampled rows per 64x4 block row.
            for _ in 0..2 {
                dsp.vvc_alf_samples(
                    &rows,
                    rows[0],
                    &transposes,
                    nc_dsp::VVC_ALF_SAMPLES,
                    &mut sums,
                );
            }
            sums[3][7] as u32
        }),
    );
    let samples: Vec<[i16; nc_dsp::VVC_ALF_SAMPLE_LANES]> = (0..SAMPLES)
        .map(|n| std::array::from_fn(|c| ((n * 31 + c * 7) % 401) as i16 - 200))
        .collect();
    let mut gram = [0; nc_dsp::VVC_ALF_DOT_LANES];
    push(
        "vvc_alf_gram",
        BlockSize::new(WIDTH, SAMPLES / WIDTH).expect("block"),
        time(|| {
            dsp.vvc_alf_gram(&samples, &mut gram);
            gram[5] as u32
        }),
    );
    // Nine rows of 64: one row of a 64-wide luma unit's 5-tap Wiener
    // statistics (eight folded features and the error).
    let features: Vec<Vec<i16>> = (0..9)
        .map(|r| {
            (0..WIDTH)
                .map(|x| ((x * 7 + r * 13) % 41) as i16 - 20)
                .collect()
        })
        .collect();
    let feature_rows: Vec<&[i16]> = features.iter().map(Vec::as_slice).collect();
    let mut sums = [0i64; 45];
    push(
        "gram_rows",
        BlockSize::new(WIDTH, 1).expect("block"),
        time(|| {
            dsp.gram_rows(&feature_rows, &mut sums);
            sums[5] as u32
        }),
    );
}

/// Compare batching with separate four-line calls over the same 64-line edge.
/// Repeated, masked and differing descriptors expose merging and driver costs.
fn bench_av1_lines(dsp: &Dsp, push: &mut impl FnMut(&'static str, BlockSize, f64)) {
    const LINE_SIDE: usize = 80;
    const SEGMENTS: usize = 16;
    let input: Vec<u8> = (0..LINE_SIDE * LINE_SIDE)
        .map(|i| 120 + (i % 3) as u8)
        .collect();
    let filter = Av1EdgeFilter {
        length: Av1FilterLength::Fourteen,
        limit: 32,
        blimit: 64,
        thresh: 8,
    };
    for vertical in [false, true] {
        for (mode, names) in if vertical {
            [
                (0, ["av1_line_equal_v", "av1_single_equal_v"]),
                (1, ["av1_line_masked_v", "av1_single_masked_v"]),
                (2, ["av1_line_mixed_v", "av1_single_mixed_v"]),
            ]
        } else {
            [
                (0, ["av1_line_equal_h", "av1_single_equal_h"]),
                (1, ["av1_line_masked_h", "av1_single_masked_h"]),
                (2, ["av1_line_mixed_h", "av1_single_mixed_h"]),
            ]
        } {
            let segments: [Option<Av1EdgeFilter>; SEGMENTS] = core::array::from_fn(|i| {
                if mode == 1 && i % 3 == 0 {
                    None
                } else {
                    Some(Av1EdgeFilter {
                        length: if mode == 2 && i % 2 == 0 {
                            Av1FilterLength::Four
                        } else {
                            filter.length
                        },
                        ..filter
                    })
                }
            });
            let mut output = input.clone();
            for (single, name) in names.into_iter().enumerate() {
                push(
                    name,
                    BlockSize::new(4, 64).expect("line"),
                    time(|| {
                        output.copy_from_slice(&input);
                        let mut dst =
                            PlaneViewMut::new(&mut output, LINE_SIDE, LINE_SIDE, LINE_SIDE)
                                .expect("plane");
                        if single == 0 {
                            dsp.av1_loop_filter_line(&mut dst, EDGE, vertical, &segments);
                        } else {
                            for (i, filter) in segments.iter().enumerate() {
                                if let Some(filter) = filter {
                                    let pos = if vertical {
                                        (EDGE.0, EDGE.1 + AV1_EDGE_LINES * i)
                                    } else {
                                        (EDGE.0 + AV1_EDGE_LINES * i, EDGE.1)
                                    };
                                    dsp.av1_loop_filter_edge(
                                        &mut dst,
                                        pos,
                                        vertical,
                                        AV1_EDGE_LINES,
                                        *filter,
                                    );
                                }
                            }
                        }
                        u32::from(output[EDGE.1 * LINE_SIDE + EDGE.0])
                    }),
                );
            }
        }
    }
}

/// View construction stays outside timing; these independently-strided rows
/// expose the actual gather/store cost of a full 8x8 interior luma block.
fn bench_cdef_picture_stride(dsp: &Dsp, push: &mut impl FnMut(&'static str, BlockSize, f64)) {
    const EXTENT: usize = 12;
    const SOURCE_STRIDE: usize = 1280;
    const DEST_STRIDE: usize = 1293;
    let input: Vec<u8> = (0..11 * SOURCE_STRIDE + EXTENT)
        .map(|i| 112 + ((i % SOURCE_STRIDE * 7 + i / SOURCE_STRIDE * 3) % 24) as u8)
        .collect();
    let src = PlaneView::new(&input, SOURCE_STRIDE, EXTENT, EXTENT).expect("source");
    for (name, strengths) in [
        ("cdef8_strided_both", (15, 4)),
        ("cdef8_strided_pri", (15, 0)),
        ("cdef8_strided_sec", (0, 4)),
        ("cdef8_strided_off", (0, 0)),
    ] {
        let mut output = vec![0; 11 * DEST_STRIDE + EXTENT];
        let mut dst = PlaneViewMut::new(&mut output, DEST_STRIDE, EXTENT, EXTENT).expect("output");
        push(
            name,
            BlockSize::new(8, 8).expect("block"),
            time(|| {
                dsp.cdef_filter_interior(8, &src, (2, 2), strengths, 3, 4, &mut dst);
                std::hint::black_box(&dst);
                0
            }),
        );
    }
}
