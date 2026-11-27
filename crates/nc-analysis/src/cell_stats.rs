//! Per-8x8-cell statistics of a block of source luma and its motion-
//! compensated residual: the cell grid a learned encoder model reads per
//! superblock (the label dump records it, `nc_encoder::labels`; a model in
//! deployment computes the same values here). Integer only, narrow types.
//!
//! Per cell, [`CELL_FEATURES`] values (`u16`, saturating):
//! - [`MEAN`]: mean sample value;
//! - [`VARIANCE`]: sample variance (sum of squared deviations / 64);
//! - [`GRADIENT`]: mean squared central-difference gradient magnitude,
//!   `(dx² + dy²) / 4` per sample (the differences span two samples);
//! - [`HOG`]..: a gradient-orientation histogram of [`HOG_BINS`] unsigned
//!   orientations (0° horizontal edge-normal ... 180°), each bin the summed
//!   `|dx| + |dy|` of the samples in it, divided by [`HOG_SCALE`];
//! - [`ZERO_SAD`]: SAD against the reference at the zero vector;
//! - [`VECTOR_SAD`]: SAD against the reference at the given vector;
//! - [`RESIDUAL_ENERGY`]: the residual's mean square at the given vector;
//! - [`NOISE_FLOOR`]: the smallest variance of the cell's four 4x4 quarters,
//!   a local floor of the sensor noise (texture raises it, flat noisy areas
//!   show the noise alone).
//!
//! Reference values are 0 without a reference (key frames).

use nc_common::PlaneView;

/// Cell side, luma samples.
pub const CELL: usize = 8;
/// Orientation bins of the histogram.
pub const HOG_BINS: usize = 8;
/// Index of the mean.
pub const MEAN: usize = 0;
/// Index of the variance.
pub const VARIANCE: usize = 1;
/// Index of the gradient energy.
pub const GRADIENT: usize = 2;
/// Index of the first orientation bin.
pub const HOG: usize = 3;
/// Index of the zero-vector SAD.
pub const ZERO_SAD: usize = HOG + HOG_BINS;
/// Index of the SAD at the vector.
pub const VECTOR_SAD: usize = ZERO_SAD + 1;
/// Index of the residual energy.
pub const RESIDUAL_ENERGY: usize = VECTOR_SAD + 1;
/// Index of the noise floor.
pub const NOISE_FLOOR: usize = RESIDUAL_ENERGY + 1;
/// Values per cell.
pub const CELL_FEATURES: usize = NOISE_FLOOR + 1;
/// The orientation bins hold `|dx| + |dy|` sums divided by this (a cell of
/// 64 samples of full-scale gradient would otherwise overflow u16).
pub const HOG_SCALE: u32 = 4;
/// Samples per cell.
const SAMPLES: u32 = (CELL * CELL) as u32;
/// Quarter side.
const QUARTER: usize = CELL / 2;
/// `tan` of the orientation bins' edges between horizontal and vertical
/// (−67.5°, −45°, −22.5°, 0°, 22.5°, 45°, 67.5°) in 1/256 units: a gradient
/// `(dx > 0, dy)` falls in the bin counted by the edges `dy · 256 > t · dx`
/// passes.
const BIN_EDGES: [i32; HOG_BINS - 1] = [-618, -256, -106, 0, 106, 256, 618];
/// The 1/256 units of [`BIN_EDGES`].
const EDGE_UNIT: i32 = 256;

/// The statistics of one cell.
pub type CellStats = [u16; CELL_FEATURES];

fn saturate(v: u32) -> u16 {
    u16::try_from(v).unwrap_or(u16::MAX)
}

/// The orientation bin of gradient (`dx`, `dy`), unsigned (a gradient and
/// its negation share a bin); `None` for a zero gradient.
fn orientation_bin(dx: i32, dy: i32) -> Option<usize> {
    let (dx, dy) = if dx < 0 { (-dx, -dy) } else { (dx, dy) };
    if dx == 0 {
        return (dy != 0).then_some(HOG_BINS - 1);
    }
    Some(
        BIN_EDGES
            .iter()
            .filter(|&&t| dy * EDGE_UNIT > t * dx)
            .count(),
    )
}

/// The statistics of the `CELL`-square cell at (`x`, `y`) of `source` (a
/// view with one sample of margin on every side, for the gradients), and
/// of its residual against `reference` at the zero vector and at `vector`
/// (views of the reference block at those positions).
#[must_use]
pub fn cell_stats(
    source: &PlaneView<'_>,
    (x, y): (usize, usize),
    references: Option<(&PlaneView<'_>, &PlaneView<'_>)>,
) -> CellStats {
    let mut out = [0u16; CELL_FEATURES];
    let (mut sum, mut sum_sq, mut gradient) = (0u32, 0u64, 0u32);
    let mut hog = [0u32; HOG_BINS];
    let mut quarters = [(0u32, 0u32); 4];
    for dy in 0..CELL {
        let (up, row, down) = (
            source.row(y + dy),
            source.row(y + dy + 1),
            source.row(y + dy + 2),
        );
        for dx in 0..CELL {
            let value = u32::from(row[x + dx + 1]);
            sum += value;
            sum_sq += u64::from(value * value);
            let quarter = &mut quarters[(dy / QUARTER) * 2 + dx / QUARTER];
            quarter.0 += value;
            quarter.1 += value * value;
            let gx = i32::from(row[x + dx + 2]) - i32::from(row[x + dx]);
            let gy = i32::from(down[x + dx + 1]) - i32::from(up[x + dx + 1]);
            gradient += (gx * gx + gy * gy).unsigned_abs();
            if let Some(bin) = orientation_bin(gx, gy) {
                hog[bin] += gx.unsigned_abs() + gy.unsigned_abs();
            }
        }
    }
    out[MEAN] = saturate(sum / SAMPLES);
    let deviation = sum_sq - u64::from(sum) * u64::from(sum) / u64::from(SAMPLES);
    out[VARIANCE] = saturate((deviation / u64::from(SAMPLES)) as u32);
    out[GRADIENT] = saturate(gradient / (4 * SAMPLES));
    for (bin, total) in hog.iter().enumerate() {
        out[HOG + bin] = saturate(total / HOG_SCALE);
    }
    let quarter_samples = (QUARTER * QUARTER) as u32;
    out[NOISE_FLOOR] = saturate(
        quarters
            .iter()
            .map(|&(total, squares)| (squares - total * total / quarter_samples) / quarter_samples)
            .min()
            .unwrap_or(0),
    );
    if let Some((zero, moved)) = references {
        let (mut zero_sad, mut sad, mut energy) = (0u32, 0u32, 0u32);
        for dy in 0..CELL {
            let row = &source.row(y + dy + 1)[x + 1..x + 1 + CELL];
            let (at_zero, at_vector) = (&zero.row(dy)[..CELL], &moved.row(dy)[..CELL]);
            for ((&sample, &zeroed), &moved_sample) in row.iter().zip(at_zero).zip(at_vector) {
                zero_sad += u32::from(sample.abs_diff(zeroed));
                let diff = u32::from(sample.abs_diff(moved_sample));
                sad += diff;
                energy += diff * diff;
            }
        }
        out[ZERO_SAD] = saturate(zero_sad);
        out[VECTOR_SAD] = saturate(sad);
        out[RESIDUAL_ENERGY] = saturate(energy / SAMPLES);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orientations_fold_and_split_the_half_circle() {
        assert_eq!(orientation_bin(0, 0), None);
        assert_eq!(orientation_bin(5, 0), orientation_bin(-5, 0));
        assert_eq!(orientation_bin(5, 0), Some(3));
        assert_eq!(orientation_bin(0, 5), Some(HOG_BINS - 1));
        assert_eq!(orientation_bin(5, 5), Some(5));
        assert_eq!(orientation_bin(5, -5), Some(1));
    }

    #[test]
    fn a_flat_cell_has_no_gradient_and_its_mean() {
        let data = vec![77u8; (CELL + 2) * (CELL + 2)];
        let view = PlaneView::new(&data, CELL + 2, CELL + 2, CELL + 2).unwrap();
        let stats = cell_stats(&view, (0, 0), None);
        assert_eq!(stats[MEAN], 77);
        assert_eq!(stats[VARIANCE], 0);
        assert_eq!(stats[GRADIENT], 0);
        assert!(stats[HOG..HOG + HOG_BINS].iter().all(|&b| b == 0));
    }
}
