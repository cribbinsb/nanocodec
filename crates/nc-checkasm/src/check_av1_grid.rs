//! AV1 edge grids against the row-major scalar oracle: partial units,
//! strided descriptors, mixed and overlapping edges, and selected row ranges.

use crate::{check::CheckReport, check_codec::record, rng::Rng};
use core::ops::Range;
use nc_common::PlaneViewMut;
use nc_dsp::{AV1_EDGE_LINES, Av1EdgeFilter, Av1FilterLength, Dsp};

const LENGTHS: [Av1FilterLength; 4] = [
    Av1FilterLength::Four,
    Av1FilterLength::Six,
    Av1FilterLength::Eight,
    Av1FilterLength::Fourteen,
];

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for (width, height) in [
        (8_usize, 8_usize),
        (16, 16),
        (17, 19),
        (18, 17),
        (19, 18),
        (32, 67),
        (67, 32),
    ] {
        let (columns, row_count) = (
            width.div_ceil(AV1_EDGE_LINES),
            height.div_ceil(AV1_EDGE_LINES),
        );
        let stride = width + 3;
        let grid_stride = columns + 2;
        for vertical in [false, true] {
            for rows in [0..row_count, 1..row_count, 1..1, row_count - 1..row_count] {
                for mode in 0..4 {
                    let segments = descriptors(
                        (width, height),
                        vertical,
                        rows.clone(),
                        grid_stride,
                        mode,
                        rng,
                    );
                    for pattern in 0..3 {
                        let input: Vec<u8> = (0..(height - 1) * stride + width + 2)
                            .map(|i| match pattern {
                                0 => rng.byte(),
                                1 => 120 + (i % 3) as u8,
                                _ => 100 + ((i / stride + i % stride) / 8) as u8,
                            })
                            .collect();
                        let run = |dsp: &Dsp, split: bool| {
                            let mut out = input.clone();
                            let len = out.len();
                            let mut dst =
                                PlaneViewMut::new(&mut out[1..len - 1], stride, width, height)
                                    .expect("plane");
                            if split {
                                for row in rows.clone() {
                                    dsp.av1_loop_filter_grid(
                                        &mut dst,
                                        vertical,
                                        row..row + 1,
                                        &segments,
                                        grid_stride,
                                    );
                                }
                            } else {
                                dsp.av1_loop_filter_grid(
                                    &mut dst,
                                    vertical,
                                    rows.clone(),
                                    &segments,
                                    grid_stride,
                                );
                            }
                            out
                        };
                        let actual = run(candidate, false);
                        let expected = run(reference, false);
                        let split = run(candidate, true);
                        record(
                            report,
                            candidate.isa(),
                            actual == expected && actual == split,
                            || {
                                format!(
                                    "AV1 grid {width}x{height} vertical={vertical} rows={rows:?} mode={mode} pattern={pattern}"
                                )
                            },
                        );
                    }
                }
            }
        }
    }
}

fn descriptors(
    (width, height): (usize, usize),
    vertical: bool,
    rows: Range<usize>,
    stride: usize,
    mode: usize,
    rng: &mut Rng,
) -> Vec<Option<Av1EdgeFilter>> {
    let poison = Av1EdgeFilter {
        length: Av1FilterLength::Fourteen,
        limit: 255,
        blimit: 254,
        thresh: 255,
    };
    // Nonselected rows and stride padding deliberately contain invalid edges.
    let mut grid = vec![Some(poison); stride * height.div_ceil(AV1_EDGE_LINES)];
    for y in rows {
        for x in 0..width.div_ceil(AV1_EDGE_LINES) {
            let filter = match mode {
                0 => None,
                1 => Some(poison),
                2 if (x + y) % 3 == 0 => None,
                2 => Some(Av1EdgeFilter {
                    length: Av1FilterLength::Four,
                    ..poison
                }),
                _ => Some(Av1EdgeFilter {
                    length: LENGTHS[rng.below(4)],
                    limit: rng.byte(),
                    blimit: rng.byte(),
                    thresh: rng.byte(),
                }),
            };
            let (pos, limit) = if vertical {
                (x * AV1_EDGE_LINES, width)
            } else {
                (y * AV1_EDGE_LINES, height)
            };
            grid[y * stride + x] =
                filter.filter(|f| pos >= f.length.reach() && pos + f.length.reach() <= limit);
        }
    }
    // The last grid row has no trailing stride padding in storage.
    grid.truncate((height.div_ceil(AV1_EDGE_LINES) - 1) * stride + width.div_ceil(AV1_EDGE_LINES));
    grid
}
