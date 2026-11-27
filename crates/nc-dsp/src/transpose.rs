//! The transpose of a square block of 8-bit samples, for intra searches
//! that rank horizontal directions against the transposed source (SATD is
//! transpose invariant; HEVC `choose_intra_mode`, VVC `DirectionProxy`).
//!
//! Portable SWAR, no SIMD tiers: each 8x8 tile is eight `u64` rows whose
//! off-diagonal 4x4, 2x2 and 1x1 sub-blocks are swapped in turn (the
//! recursive block transpose), about 40 operations per 64 samples where the
//! per-sample scatter it replaces took about five instructions per sample
//! (HEVC Fast x86, callgrind).

use nc_common::PlaneView;

/// Side of the tiles transposed as eight `u64` rows.
const TILE: usize = 8;

/// The masks and shifts of the three swap rounds: the sub-block's side in
/// samples (its shift in bits is eight times that) and the mask of the
/// lower-left sub-blocks' bytes in a row.
const ROUNDS: [(usize, u64); 3] = [
    (4, 0x0000_0000_FFFF_FFFF),
    (2, 0x0000_FFFF_0000_FFFF),
    (1, 0x00FF_00FF_00FF_00FF),
];

/// Write the transpose of the `n`x`n` block `source` (`n` its width and
/// height) to `out`, row-major with stride `n`: `out[c * n + r]` is the
/// sample at row `r`, column `c`.
///
/// # Panics
///
/// When `source` is not square or `out` holds fewer than `n * n` samples.
pub fn transpose_square(source: &PlaneView<'_>, out: &mut [u8]) {
    let n = source.width();
    assert_eq!(source.height(), n, "a square block");
    let out = &mut out[..n * n];
    if !n.is_multiple_of(TILE) {
        for (row, line) in out.chunks_exact_mut(n).enumerate() {
            for (column, sample) in line.iter_mut().enumerate() {
                *sample = source.row(column)[row];
            }
        }
        return;
    }
    for tile_row in (0..n).step_by(TILE) {
        for tile_column in (0..n).step_by(TILE) {
            let mut rows: [u64; TILE] = std::array::from_fn(|k| {
                let samples = &source.row(tile_row + k)[tile_column..tile_column + TILE];
                u64::from_le_bytes(samples.try_into().expect("a tile row"))
            });
            transpose_tile(&mut rows);
            for (k, row) in rows.iter().enumerate() {
                let start = (tile_column + k) * n + tile_row;
                out[start..start + TILE].copy_from_slice(&row.to_le_bytes());
            }
        }
    }
}

/// Transpose an 8x8 tile held as eight little-endian rows (byte `c` of
/// `rows[r]` is column `c` of row `r`).
fn transpose_tile(rows: &mut [u64; TILE]) {
    for (side, mask) in ROUNDS {
        let shift = 8 * side;
        for upper in (0..TILE).filter(|&r| r & side == 0) {
            let lower = upper + side;
            // Swap the upper row's right sub-block with the lower row's left.
            let swapped = ((rows[upper] >> shift) ^ rows[lower]) & mask;
            rows[upper] ^= swapped << shift;
            rows[lower] ^= swapped;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_sample_by_sample_transpose() {
        for n in [4, 8, 16, 32, 64] {
            let stride = n + 3;
            let samples: Vec<u8> = (0..stride * n).map(|i| (i * 37 % 251) as u8).collect();
            let view = PlaneView::new(&samples, stride, n, n).unwrap();
            let mut out = vec![0; n * n];
            transpose_square(&view, &mut out);
            for r in 0..n {
                for c in 0..n {
                    assert_eq!(out[c * n + r], samples[r * stride + c], "n {n} ({r}, {c})");
                }
            }
        }
    }
}
