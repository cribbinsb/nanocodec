//! Scalar reference implementations of the pixel kernels.
//!
//! These define the exact results every SIMD version must reproduce, so they
//! favour obviousness over speed. They are safe functions; they coerce to the
//! `unsafe fn` kernel pointer types.

use super::{GridBest, PixelFns, normalise_satd_4x4, normalise_satd_8x8};

/// The scalar kernel table.
pub(crate) fn pixel_fns() -> PixelFns {
    PixelFns {
        sad: [sad::<4>, sad::<8>, sad::<16>, sad::<32>, sad::<64>],
        sad_x4: [
            sad_x4::<4>,
            sad_x4::<8>,
            sad_x4::<16>,
            sad_x4::<32>,
            sad_x4::<64>,
        ],
        sad_quads_8x8_row,
        sad_4x4_macroblock,
        sad_scan_16x16_even_rows,
        sad_grid_best_64x64: [sad_grid_best_64x64::<1>, sad_grid_best_64x64::<2>],
        sad_grid_best_x4: None,
        sad_quads_16x16: sad_quads::<8>,
        satd_4x4: satd_tiled::<4>,
        satd_8x8: satd_tiled::<8>,
        satd_8x8_quads: None,
    }
}

pub(crate) fn sad<const W: usize>(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
    h: usize,
) -> u32 {
    (0..h)
        .map(|y| {
            let s = &src[y * src_stride..y * src_stride + W];
            let r = &rf[y * ref_stride..y * ref_stride + W];
            s.iter()
                .zip(r)
                .map(|(&a, &b)| u32::from(a.abs_diff(b)))
                .sum::<u32>()
        })
        .sum()
}

fn sad_x4<const W: usize>(
    src: &[u8],
    src_stride: usize,
    refs: [&[u8]; 4],
    ref_stride: usize,
    h: usize,
) -> [u32; 4] {
    refs.map(|rf| sad::<W>(src, src_stride, rf, ref_stride, h))
}

fn sad_quads_8x8_row(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
    out: &mut [[u32; 4]],
) {
    for (i, quads) in out.iter_mut().enumerate() {
        *quads = sad_quads::<4>(&src[8 * i..], src_stride, &rf[8 * i..], ref_stride);
    }
}

fn sad_scan_16x16_even_rows(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
    out: &mut [u32],
) {
    for (dx, result) in out.iter_mut().enumerate() {
        *result = sad::<16>(src, 2 * src_stride, &rf[dx..], 2 * ref_stride, 8);
    }
}

fn sad_grid_best_64x64<const STEP: usize>(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
    visible: &[u32; 64],
    best: &mut GridBest,
    tag: u32,
) {
    let grid = sad_grid_8x8_in_64x64::<STEP>(src, src_stride, rf, ref_stride);
    let cost = super::grid_costs(&grid, visible, STEP as u32);
    keep_min(&mut best.sad, &mut best.tag, &cost, tag);
}

/// Keep the minimum per block: where `cost[i] < best.sad[i]`, take the cost
/// and `tag`. SIMD versions use it for the elements past their last full
/// register.
pub(crate) fn keep_min(best_sad: &mut [u32], best_tag: &mut [u32], cost: &[u32], tag: u32) {
    for ((b, t), &c) in best_sad.iter_mut().zip(best_tag.iter_mut()).zip(cost) {
        if c < *b {
            *b = c;
            *t = tag;
        }
    }
}

/// The 8x8 SADs of a 64x64 block over every `STEP`-th row of each block.
fn sad_grid_8x8_in_64x64<const STEP: usize>(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
) -> [u32; 64] {
    let mut grid = [0u32; 64];
    for (i, result) in grid.iter_mut().enumerate() {
        let (x, y) = (8 * (i % 8), 8 * (i / 8));
        *result = sad::<8>(
            &src[y * src_stride + x..],
            STEP * src_stride,
            &rf[y * ref_stride + x..],
            STEP * ref_stride,
            8 / STEP,
        );
    }
    grid
}

/// Quadrant SADs of a `2 * HALF` square block.
fn sad_quads<const HALF: usize>(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
) -> [u32; 4] {
    [(0, 0), (HALF, 0), (0, HALF), (HALF, HALF)].map(|(x, y)| {
        sad::<HALF>(
            &src[y * src_stride + x..],
            src_stride,
            &rf[y * ref_stride + x..],
            ref_stride,
            HALF,
        )
    })
}

/// SATD of a `w` x `h` block as a sum of `T` x `T` tiles (`T` is 4 or 8).
fn satd_tiled<const T: usize>(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
    w: usize,
    h: usize,
) -> u32 {
    let mut total = 0;
    for ty in (0..h).step_by(T) {
        for tx in (0..w).step_by(T) {
            let s = &src[ty * src_stride + tx..];
            let r = &rf[ty * ref_stride + tx..];
            total += if T == 4 {
                normalise_satd_4x4(hadamard_abs_sum::<4, 16>(s, src_stride, r, ref_stride))
            } else {
                normalise_satd_8x8(hadamard_abs_sum::<8, 64>(s, src_stride, r, ref_stride))
            };
        }
    }
    total
}

/// Sum of absolute values of the 2-D Walsh-Hadamard transform of the
/// `N` x `N` difference block (`NN` = `N * N`).
fn hadamard_abs_sum<const N: usize, const NN: usize>(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
) -> u32 {
    let mut d = [0i32; NN];
    for y in 0..N {
        for x in 0..N {
            d[y * N + x] = i32::from(src[y * src_stride + x]) - i32::from(rf[y * ref_stride + x]);
        }
    }
    for row in d.as_chunks_mut::<N>().0 {
        walsh_hadamard_in_place(row, N, 1);
    }
    for col in 0..N {
        walsh_hadamard_in_place(&mut d[col..], N, N);
    }
    d.iter().map(|v| v.unsigned_abs()).sum()
}

/// In-place fast Walsh-Hadamard transform of the `n` elements
/// `v[0], v[step], ..., v[(n - 1) * step]` (`n` a power of two).
fn walsh_hadamard_in_place(v: &mut [i32], n: usize, step: usize) {
    let mut half = 1;
    while half < n {
        for start in (0..n).step_by(2 * half) {
            for i in start..start + half {
                let (a, b) = (v[i * step], v[(i + half) * step]);
                v[i * step] = a + b;
                v[(i + half) * step] = a - b;
            }
        }
        half *= 2;
    }
}

/// The 4x4 SADs of a macroblock: two rows of two luma 8x8 blocks, then each
/// chroma block.
fn sad_4x4_macroblock(
    (src, src_stride, rf, ref_stride): super::BlockPair<'_>,
    chroma: [super::BlockPair<'_>; 2],
) -> ([[u32; 4]; 4], [[u32; 4]; 2]) {
    let mut luma = [[0u32; 4]; 4];
    let (top, bottom) = luma.split_at_mut(2);
    sad_quads_8x8_row(src, src_stride, rf, ref_stride, top);
    sad_quads_8x8_row(
        &src[8 * src_stride..],
        src_stride,
        &rf[8 * ref_stride..],
        ref_stride,
        bottom,
    );
    let planes = chroma.map(|(s, ss, r, rs)| {
        let mut out = [[0u32; 4]; 1];
        sad_quads_8x8_row(s, ss, r, rs, &mut out);
        out[0]
    });
    (luma, planes)
}

#[cfg(test)]
mod tests {
    use super::{hadamard_abs_sum, sad};

    #[test]
    fn sad_of_known_blocks() {
        let src = [10u8; 16];
        let rf = [7u8; 16];
        assert_eq!(sad::<4>(&src, 4, &rf, 4, 4), 3 * 16);
    }

    #[test]
    fn hadamard_of_a_constant_difference_is_its_dc() {
        // A constant difference d has a single non-zero coefficient N*N*d.
        let src = [9u8; 64];
        let rf = [4u8; 64];
        assert_eq!(hadamard_abs_sum::<8, 64>(&src, 8, &rf, 8), 64 * 5);
        assert_eq!(hadamard_abs_sum::<4, 16>(&src, 8, &rf, 8), 16 * 5);
    }

    #[test]
    fn hadamard_preserves_energy() {
        // Parseval for the unnormalised WHT: sum(c^2) = N*N * sum(d^2).
        let src: Vec<u8> = (0..16u8).map(|i| i.wrapping_mul(37)).collect();
        let rf = [100u8; 16];
        let d: Vec<i64> = src.iter().map(|&s| i64::from(s) - 100).collect();
        let energy: i64 = d.iter().map(|v| v * v).sum();
        // Recompute the coefficients directly via the 4x4 definition.
        let h = [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]];
        let mut coeff_energy = 0i64;
        let mut abs_sum = 0i64;
        for u in 0..4 {
            for v in 0..4 {
                let c: i64 = (0..4)
                    .flat_map(|y| (0..4).map(move |x| (y, x)))
                    .map(|(y, x)| h[u][y] * h[v][x] * d[y * 4 + x])
                    .sum();
                coeff_energy += c * c;
                abs_sum += c.abs();
            }
        }
        assert_eq!(coeff_energy, 16 * energy);
        assert_eq!(
            i64::from(hadamard_abs_sum::<4, 16>(&src, 4, &rf, 4)),
            abs_sum
        );
    }
}
