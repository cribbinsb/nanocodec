//! Batched VP9 edge-line timings against individual eight-line calls.

use crate::bench::time;
use nc_common::PlaneViewMut;
use nc_dsp::{BlockSize, Dsp, VP9_EDGE_LINES, Vp9EdgeFilter, Vp9FilterLimits, Vp9FilterWidth};

const EDGE: (usize, usize) = (8, 8);

/// Compare batching with separate eight-line calls over the same 64-line edge.
/// Repeated, masked and differing descriptors expose merging and driver costs.
/// Both non-flat narrow filtering and flat 16-tap smoothing keep real work active.
pub(crate) fn bench(dsp: &Dsp, push: &mut impl FnMut(&'static str, BlockSize, f64)) {
    for vertical in [false, true] {
        for flat in [false, true] {
            bench_content(dsp, push, vertical, flat);
        }
    }
}

fn bench_content(
    dsp: &Dsp,
    push: &mut impl FnMut(&'static str, BlockSize, f64),
    vertical: bool,
    flat: bool,
) {
    const LINE_SIDE: usize = 80;
    const SEGMENTS: usize = 8;
    let input: Vec<u8> = (0..LINE_SIDE * LINE_SIDE)
        .map(|i| {
            if flat {
                let across = if vertical {
                    i % LINE_SIDE
                } else {
                    i / LINE_SIDE
                };
                120 + 4 * u8::from(across >= EDGE.0)
            } else {
                120 + (i % 3) as u8
            }
        })
        .collect();
    let filter = Vp9EdgeFilter {
        width: Vp9FilterWidth::Sixteen,
        limits: Vp9FilterLimits::new(32),
    };
    for (mode, names) in names(vertical, flat).into_iter().enumerate() {
        let segments: [Option<Vp9EdgeFilter>; SEGMENTS] = core::array::from_fn(|i| {
            if mode == 1 && i % 3 == 0 {
                None
            } else {
                Some(Vp9EdgeFilter {
                    width: if mode == 2 && i % 2 == 0 {
                        Vp9FilterWidth::Four
                    } else {
                        filter.width
                    },
                    ..filter
                })
            }
        });
        let mut output = input.clone();
        for (single, name) in names.into_iter().enumerate() {
            push(
                name,
                BlockSize::new(VP9_EDGE_LINES, 64).expect("line"),
                time(|| {
                    output.copy_from_slice(&input);
                    let mut dst = PlaneViewMut::new(&mut output, LINE_SIDE, LINE_SIDE, LINE_SIDE)
                        .expect("plane");
                    if single == 0 {
                        dsp.vp9_loop_filter_line(&mut dst, EDGE, vertical, &segments);
                    } else {
                        for (i, filter) in segments.iter().enumerate() {
                            if let Some(filter) = filter {
                                let pos = if vertical {
                                    (EDGE.0, EDGE.1 + VP9_EDGE_LINES * i)
                                } else {
                                    (EDGE.0 + VP9_EDGE_LINES * i, EDGE.1)
                                };
                                dsp.vp9_filter_edge(
                                    dst.data_mut(),
                                    pos.1 * LINE_SIDE + pos.0,
                                    if vertical { 1 } else { LINE_SIDE },
                                    if vertical { LINE_SIDE } else { 1 },
                                    VP9_EDGE_LINES,
                                    filter.width,
                                    filter.limits,
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

fn names(vertical: bool, flat: bool) -> [[&'static str; 2]; 3] {
    match (vertical, flat) {
        (false, false) => [
            ["vp9_line_equal_h", "vp9_single_equal_h"],
            ["vp9_line_masked_h", "vp9_single_masked_h"],
            ["vp9_line_mixed_h", "vp9_single_mixed_h"],
        ],
        (false, true) => [
            ["vp9_line_flat_equal_h", "vp9_single_flat_equal_h"],
            ["vp9_line_flat_masked_h", "vp9_single_flat_masked_h"],
            ["vp9_line_flat_mixed_h", "vp9_single_flat_mixed_h"],
        ],
        (true, false) => [
            ["vp9_line_equal_v", "vp9_single_equal_v"],
            ["vp9_line_masked_v", "vp9_single_masked_v"],
            ["vp9_line_mixed_v", "vp9_single_mixed_v"],
        ],
        (true, true) => [
            ["vp9_line_flat_equal_v", "vp9_single_flat_equal_v"],
            ["vp9_line_flat_masked_v", "vp9_single_flat_masked_v"],
            ["vp9_line_flat_mixed_v", "vp9_single_flat_mixed_v"],
        ],
    }
}
