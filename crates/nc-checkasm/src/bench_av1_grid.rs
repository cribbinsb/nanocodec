//! AV1 whole-grid batching versus row-major individual edge calls.

use crate::bench::time;
use nc_common::PlaneViewMut;
use nc_dsp::{AV1_EDGE_LINES, Av1EdgeFilter, Av1FilterLength, BlockSize, Dsp};

const SIDE: usize = 64;
const UNITS: usize = SIDE / AV1_EDGE_LINES;
const STRIDE: usize = UNITS + 2;

pub(crate) fn bench(dsp: &Dsp, push: &mut impl FnMut(&'static str, BlockSize, f64)) {
    let input: [u8; SIDE * SIDE] = core::array::from_fn(|i| 120 + (i % 3) as u8);
    for vertical in [false, true] {
        let names = if vertical {
            [
                ["av1_grid_equal_v", "av1_edges_equal_v"],
                ["av1_grid_masked_v", "av1_edges_masked_v"],
                ["av1_grid_mixed_v", "av1_edges_mixed_v"],
            ]
        } else {
            [
                ["av1_grid_equal_h", "av1_edges_equal_h"],
                ["av1_grid_masked_h", "av1_edges_masked_h"],
                ["av1_grid_mixed_h", "av1_edges_mixed_h"],
            ]
        };
        for (mode, pair) in names.into_iter().enumerate() {
            let segments: [Option<Av1EdgeFilter>; STRIDE * UNITS] = core::array::from_fn(|i| {
                let (x, y) = (i % STRIDE, i / STRIDE);
                let across = if vertical { x } else { y };
                let length = if mode == 2 && (x + y) % 2 == 0 {
                    Av1FilterLength::Four
                } else {
                    Av1FilterLength::Fourteen
                };
                let position = across * AV1_EDGE_LINES;
                if x >= UNITS
                    || position < length.reach()
                    || position + length.reach() > SIDE
                    || (mode == 1 && (x + y) % 3 == 0)
                {
                    None
                } else {
                    Some(Av1EdgeFilter {
                        length,
                        limit: 32,
                        blimit: 64,
                        thresh: 8,
                    })
                }
            });
            let mut output = input;
            for (single, name) in pair.into_iter().enumerate() {
                push(
                    name,
                    BlockSize::new(SIDE, SIDE).expect("grid"),
                    time(|| {
                        output.copy_from_slice(&input);
                        let mut dst =
                            PlaneViewMut::new(&mut output, SIDE, SIDE, SIDE).expect("plane");
                        if single == 0 {
                            dsp.av1_loop_filter_grid(
                                &mut dst,
                                vertical,
                                0..UNITS,
                                &segments,
                                STRIDE,
                            );
                        } else {
                            for y in 0..UNITS {
                                for x in 0..UNITS {
                                    if let Some(filter) = segments[y * STRIDE + x] {
                                        dsp.av1_loop_filter_edge(
                                            &mut dst,
                                            (x * AV1_EDGE_LINES, y * AV1_EDGE_LINES),
                                            vertical,
                                            AV1_EDGE_LINES,
                                            filter,
                                        );
                                    }
                                }
                            }
                        }
                        u32::from(output[SIDE / 2 * SIDE + SIDE / 2])
                    }),
                );
            }
        }
    }
}
