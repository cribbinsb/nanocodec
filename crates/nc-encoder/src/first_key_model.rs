//! First-key frame size prediction from frame-0 source luma statistics.
//!
//! Before any key frame has been coded, [`RateController::bootstrap_step`] has
//! no observed key model. A single-scalar spatial variance hint mispredicts
//! first-key bits by 2-3.6x on textured or noisy scenes (causing whole-frame
//! buffer-overflow re-codes on 10-17% of pool first keys across AV1, H.264,
//! VP9, VVC, and AV2).
//!
//! This module computes a 32-bin log-spaced histogram of 8x8 block gradient,
//! 8x8 high-frequency Laplacian, and 2x/4x pyramid gradients on frame 0 once
//! per stream, then evaluates a deadzone survival curve
//! `mean_blocks log(1 + 2 * energy / step)` at any candidate quantizer step.
//! Calibrated per codec over the 184-family unsealed clip pool (5-fold
//! family-CV log-RMSE 0.14-0.16 vs 0.40-0.53 for `bootstrap_step`, strictly
//! monotonic in step on 184/184 clips).

use nc_common::PlaneView;

/// Floor of a histogram energy term before its logarithm (an empty histogram).
const MIN_ENERGY: f64 = 1e-4;
/// Bisection halvings of the log-step interval in `step_for_bits` (2^-20 of
/// the span: far finer than one quantizer step).
const BISECTION_STEPS: usize = 20;
/// Number of log-spaced energy bins in each per-block histogram.
const NUM_BINS: usize = 32;
/// Lowest bin edge (luma units).
const BIN_MIN: f64 = 0.25;
/// Log2 span from `BIN_MIN` (0.25) to 256.0: 10 octaves across 31 intervals.
const BINS_PER_OCTAVE: f64 = (NUM_BINS - 1) as f64 / 10.0;

/// Precomputed bin edges `0.25 * 2^(i * 10 / 31)` matching `np.geomspace(0.25, 256.0, 32)`.
fn bin_edges() -> [f64; NUM_BINS] {
    let mut edges = [0.0; NUM_BINS];
    for (i, edge) in edges.iter_mut().enumerate() {
        *edge = BIN_MIN * (i as f64 / BINS_PER_OCTAVE).exp2();
    }
    edges
}

/// Per-codec fitted weights for the first-key deadzone survival model.
///
/// `weights` order:
/// `[bias, w_log_px, w_log_step, w_lap1, w_lap2, w_g1, w_g2, w_g2sq, w_g4]`.
/// `fast_bias` is added to `ln(bits)` when key-frame trellis quantization is
/// disabled (`quant.trellis < 0.5 || quant.trellis_key < 0.5`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FirstKeyCalibration {
    /// Linear weights `[bias, w_px, w_step, w_lap1, w_lap2, w_g1, w_g2, w_g2sq, w_g4]`.
    pub weights: [f64; 9],
    /// Log-bits offset when key-frame trellis quantization is disabled.
    pub fast_bias: f64,
}

/// Compact frame-0 luma energy histograms (512 bytes) extracted once per stream.
#[derive(Debug, Clone)]
pub(crate) struct FirstKeyFeatures {
    pixels: f64,
    h_lap: [f32; NUM_BINS],
    h_g: [f32; NUM_BINS],
    h_g2: [f32; NUM_BINS],
    h_g4: [f32; NUM_BINS],
}

#[inline]
fn bin_index(val: f64) -> usize {
    if val <= BIN_MIN {
        0
    } else {
        let idx = ((val / BIN_MIN).log2() * BINS_PER_OCTAVE).ceil() as isize;
        idx.clamp(0, (NUM_BINS - 1) as isize) as usize
    }
}

impl FirstKeyFeatures {
    /// Extract 32-bin histograms of 8x8 block gradient, 8x8 Laplacian, and
    /// 2x/4x pyramid gradients from the frame-0 luma plane.
    ///
    /// Returns `None` for pictures smaller than 8x8 luma samples.
    pub(crate) fn extract(luma: PlaneView<'_, u8>) -> Option<Self> {
        let grid = BlockGrid {
            columns: luma.width() / 8,
            rows: luma.height() / 8,
        };
        if grid.columns == 0 || grid.rows == 0 {
            return None;
        }
        let (g_counts, lap_counts, half) = full_resolution_counts(luma, grid);
        let (half_width, half_height) = (grid.columns * 4, grid.rows * 4);
        let g2_counts = block_gradient_counts(&half, half_width, half_height, 4, grid, 16.0);
        let quarter = halve(&half, half_width, half_height);
        let g4_counts =
            block_gradient_counts(&quarter, half_width / 2, half_height / 2, 2, grid, 4.0);
        let share = 1.0 / (grid.columns * grid.rows) as f32;
        let normalised = |counts: [u32; NUM_BINS]| counts.map(|count| count as f32 * share);
        Some(Self {
            pixels: (luma.width() * luma.height()) as f64,
            h_lap: normalised(lap_counts),
            h_g: normalised(g_counts),
            h_g2: normalised(g2_counts),
            h_g4: normalised(g4_counts),
        })
    }

    /// Predict first-key coded bits at `sample_step` (orthonormal sample-domain
    /// quantizer step).
    pub(crate) fn predict_bits(
        &self,
        cal: &FirstKeyCalibration,
        sample_step: f64,
        fast_key: bool,
    ) -> f64 {
        let st = sample_step.max(BIN_MIN);
        let inv_st = 2.0 / st;
        let edges = bin_edges();

        let mut s_lap1 = 0.0;
        let mut s_lap2 = 0.0;
        let mut s_g1 = 0.0;
        let mut s_g2 = 0.0;
        let mut s_g2sq = 0.0;
        let mut s_g4 = 0.0;

        let bins = edges
            .iter()
            .zip(&self.h_lap)
            .zip(&self.h_g)
            .zip(&self.h_g2)
            .zip(&self.h_g4);
        for ((((&edge, &hl), &hg), &hg2), &hg4) in bins {
            let ratio = edge * inv_st;
            let l1 = ratio.ln_1p();
            let l2 = (ratio * ratio).ln_1p();
            let (hl, hg, hg2, hg4) = (f64::from(hl), f64::from(hg), f64::from(hg2), f64::from(hg4));
            s_lap1 += hl * l1;
            s_lap2 += hl * l2;
            s_g1 += hg * l1;
            s_g2 += hg2 * l1;
            s_g2sq += hg2 * l2;
            s_g4 += hg4 * l1;
        }

        let w = &cal.weights;
        let log_bits = w[0]
            + w[1] * self.pixels.ln()
            + w[2] * st.ln()
            + w[3] * s_lap1.max(MIN_ENERGY).ln()
            + w[4] * s_lap2.max(MIN_ENERGY).ln()
            + w[5] * s_g1.max(MIN_ENERGY).ln()
            + w[6] * s_g2.max(MIN_ENERGY).ln()
            + w[7] * s_g2sq.max(MIN_ENERGY).ln()
            + w[8] * s_g4.max(MIN_ENERGY).ln()
            + if fast_key { cal.fast_bias } else { 0.0 };
        log_bits.exp().max(1.0)
    }

    /// Solve for the native rate-control quantizer step whose predicted bits
    /// equal `target_bits`, via bisection over `[min_step, max_step]`.
    pub(crate) fn step_for_bits(
        &self,
        cal: &FirstKeyCalibration,
        sample_step_scale: f64,
        steps: &[f64],
        bounds: (u32, u32),
        target_bits: f64,
        fast_key: bool,
    ) -> f64 {
        let target = target_bits.max(1.0);
        let lo_step = steps[bounds.0 as usize] * sample_step_scale;
        let hi_step = steps[bounds.1 as usize] * sample_step_scale;
        let mut lo = lo_step.max(BIN_MIN).ln();
        let mut hi = hi_step.max(lo_step.max(BIN_MIN)).ln();

        if self.predict_bits(cal, lo.exp(), fast_key) <= target {
            return steps[bounds.0 as usize];
        }
        if self.predict_bits(cal, hi.exp(), fast_key) >= target {
            return steps[bounds.1 as usize];
        }
        for _ in 0..BISECTION_STEPS {
            let mid = f64::midpoint(lo, hi);
            if self.predict_bits(cal, mid.exp(), fast_key) > target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        f64::midpoint(lo, hi).exp() / sample_step_scale
    }
}

/// The picture's whole 8x8 blocks (columns x rows); each pyramid level keeps
/// one histogram entry per block of the full-resolution grid.
#[derive(Clone, Copy)]
struct BlockGrid {
    columns: usize,
    rows: usize,
}

/// Full-resolution histograms (8x8 block gradient, 8x8 Laplacian) and the 2x
/// box-downsampled luma of the grid's area.
fn full_resolution_counts(
    luma: PlaneView<'_, u8>,
    grid: BlockGrid,
) -> ([u32; NUM_BINS], [u32; NUM_BINS], Vec<f32>) {
    let (width, height) = (grid.columns * 8, grid.rows * 8);
    let half_width = width / 2;
    let mut half = vec![0.0f32; half_width * (height / 2)];
    let mut g_counts = [0u32; NUM_BINS];
    let mut lap_counts = [0u32; NUM_BINS];
    for block_row in 0..grid.rows {
        let mut gradient = vec![0u32; grid.columns];
        let mut laplacian = vec![0u32; grid.columns];
        for row_index in block_row * 8..block_row * 8 + 8 {
            let row = luma.row(row_index);
            let below = (row_index + 1 < height).then(|| luma.row(row_index + 1));
            let above = (row_index > 0).then(|| luma.row(row_index - 1));
            for (column, (g_sum, lap_sum)) in gradient.iter_mut().zip(&mut laplacian).enumerate() {
                for col in column * 8..column * 8 + 8 {
                    let pixel = i32::from(row[col]);
                    if col + 1 < width {
                        *g_sum += pixel.abs_diff(i32::from(row[col + 1]));
                    }
                    if let Some(below) = below {
                        *g_sum += pixel.abs_diff(i32::from(below[col]));
                    }
                    if let (Some(above), Some(below)) = (above, below)
                        && col > 0
                        && col + 1 < width
                    {
                        let lap = 4 * pixel
                            - i32::from(above[col])
                            - i32::from(below[col])
                            - i32::from(row[col - 1])
                            - i32::from(row[col + 1]);
                        *lap_sum += lap.unsigned_abs();
                    }
                }
            }
            if row_index % 2 == 0 {
                let below = luma.row(row_index + 1);
                let out = &mut half[(row_index / 2) * half_width..(row_index / 2 + 1) * half_width];
                for (half_col, sample) in out.iter_mut().enumerate() {
                    let col = half_col * 2;
                    let sum = u32::from(row[col])
                        + u32::from(row[col + 1])
                        + u32::from(below[col])
                        + u32::from(below[col + 1]);
                    *sample = sum as f32 * 0.25;
                }
            }
        }
        for (g_sum, lap_sum) in gradient.iter().zip(&laplacian) {
            g_counts[bin_index(f64::from(*g_sum) / 64.0)] += 1;
            lap_counts[bin_index(f64::from(*lap_sum) / 256.0)] += 1;
        }
    }
    (g_counts, lap_counts, half)
}

/// A pyramid level's histogram of block gradients: `side` x `side` blocks (one
/// per block of the full-resolution grid), each sum divided by `per_block`.
fn block_gradient_counts(
    plane: &[f32],
    width: usize,
    height: usize,
    side: usize,
    grid: BlockGrid,
    per_block: f64,
) -> [u32; NUM_BINS] {
    let mut counts = [0u32; NUM_BINS];
    for block_row in 0..grid.rows {
        let mut gradient = vec![0.0f32; grid.columns];
        for row_index in block_row * side..block_row * side + side {
            let row = &plane[row_index * width..(row_index + 1) * width];
            let below = (row_index + 1 < height)
                .then(|| &plane[(row_index + 1) * width..(row_index + 2) * width]);
            for (column, sum) in gradient.iter_mut().enumerate() {
                for col in column * side..column * side + side {
                    let sample = row[col];
                    if col + 1 < width {
                        *sum += (sample - row[col + 1]).abs();
                    }
                    if let Some(below) = below {
                        *sum += (sample - below[col]).abs();
                    }
                }
            }
        }
        for sum in gradient {
            counts[bin_index(f64::from(sum) / per_block)] += 1;
        }
    }
    counts
}

/// A 2x box-downsample of an even-sized plane.
fn halve(plane: &[f32], width: usize, height: usize) -> Vec<f32> {
    let half_width = width / 2;
    let mut out = vec![0.0f32; half_width * (height / 2)];
    for (half_row, out_row) in out.chunks_exact_mut(half_width).enumerate() {
        let row = &plane[half_row * 2 * width..(half_row * 2 + 1) * width];
        let below = &plane[(half_row * 2 + 1) * width..(half_row * 2 + 2) * width];
        for (half_col, sample) in out_row.iter_mut().enumerate() {
            let col = half_col * 2;
            *sample = 0.25 * (row[col] + row[col + 1] + below[col] + below[col + 1]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_common::Plane;

    #[test]
    fn monotonic_and_responsive_to_texture() {
        let cal = FirstKeyCalibration {
            weights: [
                0.635_606, 0.940_353, 0.14836, 0.841_426, 0.142_821, -1.421_371, 0.475_305,
                0.402_741, 0.694_098,
            ],
            fast_bias: 0.135_586,
        };
        let mut flat = Plane::<u8>::new(64, 64, 0).unwrap();
        flat.view_mut().data_mut().fill(128);
        let mut busy = Plane::<u8>::new(64, 64, 0).unwrap();
        {
            let mut view = busy.view_mut();
            for y in 0..64 {
                let row = view.row_mut(y);
                for (x, p) in row.iter_mut().enumerate() {
                    *p = (((x * 37 + y * 73) ^ (x * y)) & 0xff) as u8;
                }
            }
        }
        let f_flat = FirstKeyFeatures::extract(flat.view()).unwrap();
        let f_busy = FirstKeyFeatures::extract(busy.view()).unwrap();

        let b_flat = f_flat.predict_bits(&cal, 16.0, false);
        let b_busy = f_busy.predict_bits(&cal, 16.0, false);
        assert!(b_busy > b_flat * 3.0, "busy {b_busy} vs flat {b_flat}");

        let b_fine = f_busy.predict_bits(&cal, 8.0, false);
        let b_coarse = f_busy.predict_bits(&cal, 32.0, false);
        assert!(b_fine > b_busy && b_busy > b_coarse);
    }
}
