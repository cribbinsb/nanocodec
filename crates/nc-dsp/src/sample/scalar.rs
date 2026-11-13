//! Scalar reference implementations of the sample kernels.

use super::{Moments, SampleFns, box_mean, pair_mean};

/// The scalar kernel table.
pub(crate) fn sample_fns() -> SampleFns {
    SampleFns {
        moments_8x8_row: moments_row::<8>,
        moments_8x4_row: moments_row::<4>,
        downscale_2x,
        pair_mean_2x: pairs_2x::<false>,
        pair_pick_2x: pairs_2x::<true>,
    }
}

/// Moments of each 8-wide, `ROWS`-tall block along a row.
pub(crate) fn moments_row<const ROWS: usize>(src: &[u8], src_stride: usize, out: &mut [Moments]) {
    for (i, m) in out.iter_mut().enumerate() {
        *m = moments_block::<ROWS>(&src[8 * i..], src_stride);
    }
}

fn moments_block<const ROWS: usize>(src: &[u8], src_stride: usize) -> Moments {
    let mut m = Moments::default();
    for y in 0..ROWS {
        for &v in &src[y * src_stride..y * src_stride + 8] {
            m.sum += u32::from(v);
            m.sum_sq += u32::from(v) * u32::from(v);
        }
    }
    m
}

pub(crate) fn downscale_2x(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        downscale_2x_row(
            &src[2 * y * src_stride..],
            &src[(2 * y + 1) * src_stride..],
            &mut dst[y * dst_stride..y * dst_stride + w],
        );
    }
}

/// Each output from a horizontal pair of the same source row: the left
/// sample (`PICK`) or the rounded mean.
pub(crate) fn pairs_2x<const PICK: bool>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        pairs_2x_row::<PICK>(
            &src[y * src_stride..],
            &mut dst[y * dst_stride..y * dst_stride + w],
        );
    }
}

/// One output row of [`pairs_2x`]; SIMD versions use it for the tail.
pub(crate) fn pairs_2x_row<const PICK: bool>(row: &[u8], out: &mut [u8]) {
    for (x, o) in out.iter_mut().enumerate() {
        *o = if PICK {
            row[2 * x]
        } else {
            pair_mean(row[2 * x], row[2 * x + 1])
        };
    }
}

/// One output row from two input rows. SIMD versions use it for the tail
/// columns that do not fill a register.
pub(crate) fn downscale_2x_row(top: &[u8], bottom: &[u8], out: &mut [u8]) {
    for (x, o) in out.iter_mut().enumerate() {
        *o = box_mean(top[2 * x], top[2 * x + 1], bottom[2 * x], bottom[2 * x + 1]);
    }
}

#[cfg(test)]
mod tests {
    use super::{downscale_2x, moments_block, pairs_2x};

    #[test]
    fn moments_of_a_ramp() {
        let src: Vec<u8> = (0..64u8).collect();
        let m = moments_block::<8>(&src, 8);
        assert_eq!(m.sum, (0..64).sum::<u32>());
        assert_eq!(m.sum_sq, (0..64u32).map(|v| v * v).sum::<u32>());
        assert_eq!(m.mean_8x8(), 32); // 31.5 rounds up
        // Sum of squared deviations of 0..63 from 31.5 = 64 * (64^2 - 1) / 12.
        assert_eq!(m.sum_sq_dev_8x8(), 21840);
    }

    #[test]
    fn downscale_rounds_box_means() {
        let src = [0u8, 1, 10, 20, 2, 3, 30, 41];
        let mut dst = [0u8; 2];
        downscale_2x(&src, 4, &mut dst, 2, 2, 1);
        assert_eq!(dst, [2, 25]); // (6 + 2) >> 2, (101 + 2) >> 2
    }

    #[test]
    fn pairs_round_means_or_pick_the_left() {
        let src = [0u8, 1, 10, 21];
        let mut dst = [0u8; 2];
        pairs_2x::<false>(&src, 4, &mut dst, 2, 2, 1);
        assert_eq!(dst, [1, 16]);
        pairs_2x::<true>(&src, 4, &mut dst, 2, 2, 1);
        assert_eq!(dst, [0, 10]);
    }
}
