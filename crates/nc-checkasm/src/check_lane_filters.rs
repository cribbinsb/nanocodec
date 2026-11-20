//! The lane-layout loop filters (`nc_dsp` `deblock::lanes`) against the
//! scalar traversal, on decoder-shaped passes: a random tiling of square
//! transforms gives every 4x4 unit its edge (filter length from the smaller
//! transform across it, as AV1 §7.14.3 / VP9 §8.8.2) and its own level, so
//! one vector mixes widths and thresholds; samples are smooth blocks with
//! small noise (the smoothing filters' flat cases) or noise. Planes whose
//! last 32-line group leaves the allocation exercise the scalar fallback,
//! and one entry table serves every pass (the kernel clears it).

use crate::{check::CheckReport, check_codec::record, rng::Rng};
use nc_common::PlaneViewMut;
use nc_dsp::{Av1EdgeFilter, Av1FilterLength, Dsp, EdgeRun, LoopFilterLanes};

/// Transform sides of the tiling (log2 of 4..=32).
const MIN_SIDE_LOG2: usize = 2;
const MAX_SIDE_LOG2: usize = 5;
/// Samples per unit.
const UNIT: usize = 4;

/// Unit transform sides (log2) of a random tiling of `columns` x `rows` units.
fn tiling(columns: usize, rows: usize, rng: &mut Rng) -> Vec<usize> {
    let mut sides = vec![0; columns * rows];
    let block = 1 << (MAX_SIDE_LOG2 - MIN_SIDE_LOG2);
    for by in (0..rows).step_by(block) {
        for bx in (0..columns).step_by(block) {
            split(&mut sides, columns, rows, (bx, by), MAX_SIDE_LOG2, rng);
        }
    }
    sides
}

fn split(
    sides: &mut [usize],
    columns: usize,
    rows: usize,
    (x, y): (usize, usize),
    log2: usize,
    rng: &mut Rng,
) {
    if log2 > MIN_SIDE_LOG2 && rng.below(3) != 0 {
        let half = 1 << (log2 - 1 - MIN_SIDE_LOG2);
        for (dx, dy) in [(0, 0), (half, 0), (0, half), (half, half)] {
            split(sides, columns, rows, (x + dx, y + dy), log2 - 1, rng);
        }
        return;
    }
    let n = 1 << (log2 - MIN_SIDE_LOG2);
    for row in y..(y + n).min(rows) {
        for column in x..(x + n).min(columns) {
            sides[row * columns + column] = log2;
        }
    }
}

/// AV1 thresholds of `level` at `sharpness` (§7.14.4).
fn thresholds(level: u8, sharpness: u8, length: Av1FilterLength) -> Av1EdgeFilter {
    let shift = u32::from(sharpness > 0) + u32::from(sharpness > 4);
    let mut limit = level >> shift;
    if sharpness > 0 {
        limit = limit.min(9 - sharpness);
    }
    let limit = limit.max(1);
    Av1EdgeFilter {
        length,
        limit,
        blimit: 2 * (level + 2) + limit,
        thresh: level >> 4,
    }
}

/// The 4-line runs of one pass over a tiling (`vertical`: edges between
/// columns of units), with a random level per unit.
fn runs(
    sides: &[usize],
    columns: usize,
    rows: usize,
    vertical: bool,
    chroma: bool,
    sharpness: u8,
    rng: &mut Rng,
) -> Vec<EdgeRun<Av1EdgeFilter>> {
    let mut runs = Vec::new();
    for row in 0..rows {
        for column in 0..columns {
            let here = sides[row * columns + column];
            let (across, before) = if vertical {
                (
                    column,
                    column.checked_sub(1).map(|c| sides[row * columns + c]),
                )
            } else {
                (row, row.checked_sub(1).map(|r| sides[r * columns + column]))
            };
            let Some(before) = before else { continue };
            let side = here.min(before);
            // Only transform edges: a block's own interior has none.
            if !(across << MIN_SIDE_LOG2).is_multiple_of(1 << here) {
                continue;
            }
            let length = match (chroma, side) {
                (_, 2) => Av1FilterLength::Four,
                (true, _) => Av1FilterLength::Six,
                (false, 3) => Av1FilterLength::Eight,
                (false, _) => Av1FilterLength::Fourteen,
            };
            let level = rng.below(64) as u8;
            if level == 0 {
                continue;
            }
            let (x, y) = (column * UNIT, row * UNIT);
            runs.push(EdgeRun {
                x,
                y,
                lines: UNIT,
                filter: thresholds(level, sharpness, length),
            });
        }
    }
    runs
}

/// Smooth blocks with small steps and noise, or plain noise.
fn samples(len: usize, stride: usize, noisy: bool, rng: &mut Rng) -> Vec<u8> {
    let base = rng.byte();
    (0..len)
        .map(|i| {
            if noisy {
                return rng.byte();
            }
            let (x, y) = (i % stride, i / stride);
            let step = ((x / 8 + y / 8) % 3) as u8 * (rng.below(3) as u8);
            base.wrapping_add(step)
                .wrapping_add(u8::from(rng.below(4) == 0))
        })
        .collect()
}

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    let mut lanes = LoopFilterLanes::default();
    let mut reference_lanes = LoopFilterLanes::default();
    for (width, height, extra_rows) in [
        (128_usize, 96_usize, 0_usize),
        (96, 104, 0),
        (72, 40, 40),
        (136, 72, 0),
    ] {
        for chroma in [false, true] {
            for case in 0..24 {
                let stride = width + 8 * (case % 2);
                let (columns, rows) = (width / UNIT, height.div_ceil(UNIT));
                let sides = tiling(columns, rows, rng);
                let sharpness = (case % 8) as u8;
                let input = samples(stride * (height + extra_rows), stride, case % 3 == 2, rng);
                let mut output = [input.clone(), input];
                for vertical in [true, false] {
                    let runs = runs(&sides, columns, rows, vertical, chroma, sharpness, rng);
                    for (dsp, (samples, lanes)) in [candidate, reference]
                        .into_iter()
                        .zip(output.iter_mut().zip([&mut lanes, &mut reference_lanes]))
                    {
                        let mut view =
                            PlaneViewMut::new(samples, stride, width, height).expect("plane");
                        dsp.av1_loop_filter_runs(&mut view, vertical, &runs, lanes);
                    }
                }
                record(report, candidate.isa(), output[0] == output[1], || {
                    format!(
                        "av1 lane passes {width}x{height}+{extra_rows} chroma {chroma} case {case}"
                    )
                });
            }
        }
    }
}
