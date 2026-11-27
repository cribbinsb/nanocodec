//! Data-driven intra prediction (DIP): eleven features of the edges (the
//! corner and five averages along each of the top and left edges) feed one
//! of six trained linear predictors for an 8x8 grid of the block (fewer for
//! small blocks); the samples between grid points are interpolated linearly
//! from the grid and the edges.
//!
//! Ported from dav2d `src/ipred_tmpl.c` (`ipred_dip_c`; commit 237229e,
//! BSD-2-Clause, see `docs/licenses/dav2d`), which is bit-exact with AVM
//! v1.0.0's intra matrix prediction.

use nc_common::{BitDepth, Sample};

use super::tables::DIP_WEIGHTS;
use super::{Block, SampleRef, clip_pixel};

/// Features per side.
const SIDE_FEATURES: usize = 5;
/// All features: the corner and both sides.
const FEATURES: usize = 1 + 2 * SIDE_FEATURES;
/// Bits of the trained weights (4096 is 1.0).
const WEIGHT_BITS: u32 = 12;
/// Bits of the mode in DIP's mode argument (values 0..=5).
const MODE_MASK: i32 = 7;
/// The mode argument's transpose bit: swap the roles of the two edges.
const TRANSPOSE_FLAG: i32 = 16;
/// Side of the coarse prediction grid.
const GRID: usize = 8;
/// log2 of the edge samples one feature averages per 4 samples of side.
const FEATURE_LOG2: u32 = 2;
/// Bits of the linear interpolation between grid points.
const LERP_BITS: u32 = 3;

/// One side's features: the rounded means of the first five runs of
/// `side / 4` edge samples (the side and a quarter beyond it).
fn side_features<T: Sample>(samples: impl Iterator<Item = T>, side: usize, out: &mut [T]) {
    let run = side >> FEATURE_LOG2;
    let log2 = side.ilog2() - FEATURE_LOG2;
    let round = (side >> 3) as u32;
    let mut samples = samples;
    for feature in out.iter_mut() {
        let sum: u32 = samples.by_ref().take(run).map(Into::<u32>::into).sum();
        *feature = T::from_i32(((sum + round) >> log2) as i32);
    }
}

/// `DIP_PRED`: `mode` is the trained predictor (low bits) with
/// [`TRANSPOSE_FLAG`].
pub(super) fn dip<T: Sample>(
    dst: &mut Block<'_, T>,
    edge: SampleRef<'_, T>,
    w: usize,
    h: usize,
    mode: i32,
    depth: BitDepth,
) {
    debug_assert!(w.is_power_of_two() && (4..=64).contains(&w));
    debug_assert!(h.is_power_of_two() && (4..=64).contains(&h));
    let weights = &DIP_WEIGHTS[(mode & MODE_MASK) as usize];
    let transpose = mode & TRANSPOSE_FLAG != 0;

    let mut input = [T::default(); FEATURES];
    input[0] = edge.at(0);
    let (top_at, left_at) = if transpose { (6, 1) } else { (1, 6) };
    let top = (1..).map(|i| edge.at(i));
    side_features(top, w, &mut input[top_at..top_at + SIDE_FEATURES]);
    let left = (1..).map(|i| edge.at(-i));
    side_features(left, h, &mut input[left_at..left_at + SIDE_FEATURES]);
    let input_sum: i32 = input.iter().map(|&v| v.to_i32()).sum();

    // The grid points sit at the last sample of each step; small sides take
    // every other trained output.
    let step_x = (w / GRID).max(1);
    let step_y = (h / GRID).max(1);
    let (dw, dh) = if transpose { (GRID, 1) } else { (1, GRID) };
    let dw = dw << usize::from(w < GRID);
    let dh = dh << usize::from(h < GRID);
    for (iy, y) in (step_y - 1..h).step_by(step_y).enumerate() {
        let row = dst.row(y, w);
        for (ix, x) in (step_x - 1..w).step_by(step_x).enumerate() {
            let point = &weights[iy * dh + ix * dw];
            let sum: i32 = point
                .iter()
                .zip(input)
                .map(|(&k, v)| i32::from(k) * v.to_i32())
                .sum();
            row[x] = clip_pixel(
                ((sum + (1 << (WEIGHT_BITS - 1))) >> WEIGHT_BITS) - input_sum,
                depth,
            );
        }
    }

    if w > GRID {
        interpolate_rows(dst, edge, w, h, step_x, step_y);
    }
    if h > GRID {
        interpolate_columns(dst, edge, w, h, step_y);
    }
}

/// The point `z` eighths of the way from `p1` back to `p0`.
fn lerp<T: Sample>(p0: i32, p1: i32, z: usize) -> T {
    let z = z as i32;
    T::from_i32((p0 * z + p1 * (GRID as i32 - z)) >> LERP_BITS)
}

/// Fills each grid row between grid points, from the left edge sample.
fn interpolate_rows<T: Sample>(
    dst: &mut Block<'_, T>,
    edge: SampleRef<'_, T>,
    w: usize,
    h: usize,
    step_x: usize,
    step_y: usize,
) {
    let step_z = 16 >> (w.ilog2() - FEATURE_LOG2);
    for y in (step_y - 1..h).step_by(step_y) {
        let mut p0 = edge.wide(-(y as isize + 1));
        let row = dst.row(y, w);
        for x in (step_x - 1..w).step_by(step_x) {
            let p1 = row[x].to_i32();
            for (k, z) in (step_z..GRID).step_by(step_z).enumerate() {
                row[x - 1 - k] = lerp(p0, p1, z);
            }
            p0 = p1;
        }
    }
}

/// Fills each column between grid rows, from the top edge sample.
fn interpolate_columns<T: Sample>(
    dst: &mut Block<'_, T>,
    edge: SampleRef<'_, T>,
    w: usize,
    h: usize,
    step_y: usize,
) {
    let step_z = 16 >> (h.ilog2() - FEATURE_LOG2);
    for x in 0..w {
        let mut p0 = edge.wide(x as isize + 1);
        for y in (step_y - 1..h).step_by(step_y) {
            let p1 = dst.row(y, w)[x].to_i32();
            for (k, z) in (step_z..GRID).step_by(step_z).enumerate() {
                dst.row(y - 1 - k, w)[x] = lerp(p0, p1, z);
            }
            p0 = p1;
        }
    }
}
