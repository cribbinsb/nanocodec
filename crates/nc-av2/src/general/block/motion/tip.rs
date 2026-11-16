//! The TIP motion field: the projected field rescaled to the distance
//! between the two TIP references, its holes filled from neighbours and
//! smoothed (dav2d `refmvs.c` `tip_projection_c`, `tip_fill_holes_c`,
//! `tip_smoothen_c`; dav2d commit 237229e, BSD-2-Clause, see
//! `docs/licenses/dav2d`). Each works on one superblock row of `h8` rows
//! (the slices start at its first row) in blocks of 8 samples.
use super::vector::{TemporalMv, project_in_field};

/// Blocks are this many samples on a side.
const BLOCK_SAMPLES: usize = 8;
/// `65536 / n` for the 1..5 vectors smoothing averages.
const RECIPROCALS: [u32; 5] = [65536, 32768, 21845, 16384, 13107];

/// The neighbour offsets of a sample inside its block: left, right, up,
/// down, each only within the block (rows: from `y_start` to `y_end`).
fn neighbours(
    (x, y): (usize, usize),
    step: usize,
    x_end: usize,
    (y_start, y_end): (usize, usize),
) -> impl Iterator<Item = (isize, isize)> {
    let s = step as isize;
    [
        (x >= step).then_some((-s, 0)),
        (x + step < x_end).then_some((s, 0)),
        (y >= y_start + step).then_some((0, -s)),
        (y + step < y_end).then_some((0, s)),
    ]
    .into_iter()
    .flatten()
}

/// The blocks of a row: their first column and width.
fn blocks((start, end): (usize, usize), step: usize) -> impl Iterator<Item = (usize, usize)> {
    let block = BLOCK_SAMPLES * step;
    (start..end)
        .step_by(block)
        .map(move |sx| (sx, (end - sx).min(block)))
}

/// Rescale each sample to the TIP references' distance `delta` (dav2d
/// `tip_projection_c`).
pub fn project(
    vectors: &mut [TemporalMv],
    distances: &mut [u8],
    stride: usize,
    columns: (usize, usize),
    h8: usize,
    step: usize,
    delta: u8,
) {
    for (sx, x_end) in blocks(columns, step) {
        for y in (0..h8).step_by(step) {
            for x in (0..x_end).step_by(step) {
                let at = y * stride + sx + x;
                let distance = distances[at];
                distances[at] = delta;
                if vectors[at].is_valid() {
                    vectors[at] = project_in_field(vectors[at], i32::from(delta), distance);
                }
            }
        }
    }
}

/// Copy each sample to its empty neighbours in the block, in raster
/// order (dav2d `tip_fill_holes_c`).
pub fn fill_holes(
    vectors: &mut [TemporalMv],
    stride: usize,
    columns: (usize, usize),
    h8: usize,
    step: usize,
) {
    let block = BLOCK_SAMPLES * step;
    for (sx, x_end) in blocks(columns, step) {
        for y in (0..h8).step_by(step) {
            let y_start = y & !(block - 1);
            let y_end = (y_start + block).min(h8);
            for x in (0..x_end).step_by(step) {
                let at = y * stride + sx + x;
                let mv = vectors[at];
                if !mv.is_valid() {
                    continue;
                }
                for (dx, dy) in neighbours((x, y), step, x_end, (y_start, y_end)) {
                    let to = (at as isize + dy * stride as isize + dx) as usize;
                    if !vectors[to].is_valid() {
                        vectors[to] = mv;
                    }
                }
            }
        }
    }
}

/// Replace each sample by the mean of itself and its valid neighbours in
/// the block (dav2d `tip_smoothen_c`: each row is written back once the
/// row below has read it).
pub fn smoothen(
    vectors: &mut [TemporalMv],
    stride: usize,
    columns: (usize, usize),
    h8: usize,
    step: usize,
) {
    let block = BLOCK_SAMPLES * step;
    let mut line = [TemporalMv::INVALID; 2 * BLOCK_SAMPLES];
    for (sx, x_end) in blocks(columns, step) {
        let mut previous = None;
        for y in (0..h8).step_by(step) {
            let y_start = y & !(block - 1);
            let y_end = (y_start + block).min(h8);
            for x in (0..x_end).step_by(step) {
                let at = y * stride + sx + x;
                let mut sum = (0_i32, 0_i32);
                let mut count = 0;
                let around = std::iter::once((0, 0)).chain(neighbours(
                    (x, y),
                    step,
                    x_end,
                    (y_start, y_end),
                ));
                for (dx, dy) in around {
                    let mv = vectors[(at as isize + dy * stride as isize + dx) as usize];
                    if mv.is_valid() {
                        sum = (sum.0 + i32::from(mv.y), sum.1 + i32::from(mv.x));
                        count += 1;
                    }
                }
                if let Some(row) = previous {
                    vectors[row + sx + x] = line[x];
                }
                line[x] = if count == 0 {
                    TemporalMv::INVALID
                } else {
                    // dav2d multiplies in unsigned 32-bit arithmetic.
                    let m = |s: i32| {
                        ((s as u32)
                            .wrapping_mul(RECIPROCALS[count - 1])
                            .wrapping_add(0x8000)
                            .wrapping_sub(u32::from(s < 0))
                            >> 16) as i16
                    };
                    TemporalMv {
                        y: m(sum.0),
                        x: m(sum.1),
                    }
                };
            }
            previous = Some(y * stride);
        }
        if let Some(row) = previous {
            for x in (0..x_end).step_by(step) {
                vectors[row + sx + x] = line[x];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One vector in an empty block spreads right and down in raster
    /// order (up and left one step a sample), then smoothing keeps a
    /// uniform area.
    #[test]
    fn fill_and_smooth() {
        let stride = 32;
        let mut v = vec![TemporalMv::INVALID; stride * 8];
        let mv = TemporalMv { y: 6, x: -3 };
        v[3 * stride + 3] = mv;
        fill_holes(&mut v, stride, (0, 8), 8, 1);
        assert_eq!(v[3 * stride + 4], mv);
        assert_eq!(v[7 * stride + 7], mv);
        assert_eq!(v[2 * stride + 3], mv);
        assert!(!v[0].is_valid());
        smoothen(&mut v, stride, (0, 8), 8, 1);
        assert_eq!(v[7 * stride + 7], mv);
    }
}
