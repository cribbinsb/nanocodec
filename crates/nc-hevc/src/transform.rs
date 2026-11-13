//! HEVC core transforms (HEVC §8.6.4).
//!
//! The encoder runs nc-dsp's SIMD kernels (`Dsp::hevc_forward`,
//! `Dsp::hevc_inverse`). The scalar code here is their reference: it is
//! compiled for tests only, where it checks the kernels block for block.
//!
//! Every HEVC DCT is a sub-matrix of the 32-point `transMatrix` (HEVC
//! §8.6.4.2, equations 8-315..8-318): the N-point basis vector `k` is row
//! `k * 32 / N` of the 32-point matrix, truncated to N entries. Both
//! directions are computed with the even/odd ("partial butterfly")
//! decomposition in exact integer arithmetic, which equals the matrix
//! product (unit-tested), so the inverse is the normative one. The forward
//! transform is the encoder's choice; it uses the scaling of the HM
//! reference encoder, which keeps its output in the range the quantiser
//! expects.
//!
//! The matrix values are the specification's; they were taken from kvazaar's
//! `kvz_g_dct_32` (BSD-3-Clause, see `THIRD_PARTY.md`).

/// The transform of a `2^log2` block (§8.6.4.2, `trType`): the DST for
/// 4x4 intra luma, otherwise the DCT of that size; `None` outside 4..32.
#[must_use]
pub fn kind(log2: u8, intra_luma: bool) -> Option<nc_dsp::HevcTransform> {
    if log2 == MIN_LOG2 && intra_luma {
        Some(nc_dsp::HevcTransform::Dst4)
    } else {
        nc_dsp::HevcTransform::dct(log2)
    }
}

/// Constant reconstructed residual when the DCT has only a DC coefficient.
/// Preserve both normative rounding stages and the intermediate clamp.
pub fn inverse_dc(dc: i32) -> i32 {
    const DC_BASIS: i32 = 64;
    let first =
        ((DC_BASIS * dc + (1 << (INV_SHIFT_1 - 1))) >> INV_SHIFT_1).clamp(COEFF_MIN, COEFF_MAX);
    (DC_BASIS * first + (1 << (INV_SHIFT_2 - 1))) >> INV_SHIFT_2
}

/// Smallest and largest transform: 4x4 to 32x32.
pub const MIN_LOG2: u8 = 2;
/// Largest transform size (log2).
pub const MAX_LOG2: u8 = 5;

/// Range of transform coefficients and intermediate values (HEVC §8.6.2,
/// `CoeffMinY` / `CoeffMaxY` without extended precision).
pub const COEFF_MIN: i32 = -32768;
/// See [`COEFF_MIN`].
pub const COEFF_MAX: i32 = 32767;

/// Bit depth of every sample (Main profile).
const BIT_DEPTH: u32 = 8;
/// Shift after the first (vertical) inverse stage (HEVC §8.6.4.2).
const INV_SHIFT_1: u32 = 7;
/// Shift after the second inverse stage: `20 − BitDepth`.
const INV_SHIFT_2: u32 = 20 - BIT_DEPTH;

#[cfg(test)]
/// The 32-point DCT matrix, `transMatrix` of HEVC §8.6.4.2.
#[rustfmt::skip]
const DCT32: [[i8; 32]; 32] = [
    [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
    [90, 90, 88, 85, 82, 78, 73, 67, 61, 54, 46, 38, 31, 22, 13, 4, -4, -13, -22, -31, -38, -46, -54, -61, -67, -73, -78, -82, -85, -88, -90, -90],
    [90, 87, 80, 70, 57, 43, 25, 9, -9, -25, -43, -57, -70, -80, -87, -90, -90, -87, -80, -70, -57, -43, -25, -9, 9, 25, 43, 57, 70, 80, 87, 90],
    [90, 82, 67, 46, 22, -4, -31, -54, -73, -85, -90, -88, -78, -61, -38, -13, 13, 38, 61, 78, 88, 90, 85, 73, 54, 31, 4, -22, -46, -67, -82, -90],
    [89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89, 89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89],
    [88, 67, 31, -13, -54, -82, -90, -78, -46, -4, 38, 73, 90, 85, 61, 22, -22, -61, -85, -90, -73, -38, 4, 46, 78, 90, 82, 54, 13, -31, -67, -88],
    [87, 57, 9, -43, -80, -90, -70, -25, 25, 70, 90, 80, 43, -9, -57, -87, -87, -57, -9, 43, 80, 90, 70, 25, -25, -70, -90, -80, -43, 9, 57, 87],
    [85, 46, -13, -67, -90, -73, -22, 38, 82, 88, 54, -4, -61, -90, -78, -31, 31, 78, 90, 61, 4, -54, -88, -82, -38, 22, 73, 90, 67, 13, -46, -85],
    [83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83],
    [82, 22, -54, -90, -61, 13, 78, 85, 31, -46, -90, -67, 4, 73, 88, 38, -38, -88, -73, -4, 67, 90, 46, -31, -85, -78, -13, 61, 90, 54, -22, -82],
    [80, 9, -70, -87, -25, 57, 90, 43, -43, -90, -57, 25, 87, 70, -9, -80, -80, -9, 70, 87, 25, -57, -90, -43, 43, 90, 57, -25, -87, -70, 9, 80],
    [78, -4, -82, -73, 13, 85, 67, -22, -88, -61, 31, 90, 54, -38, -90, -46, 46, 90, 38, -54, -90, -31, 61, 88, 22, -67, -85, -13, 73, 82, 4, -78],
    [75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75, 75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75],
    [73, -31, -90, -22, 78, 67, -38, -90, -13, 82, 61, -46, -88, -4, 85, 54, -54, -85, 4, 88, 46, -61, -82, 13, 90, 38, -67, -78, 22, 90, 31, -73],
    [70, -43, -87, 9, 90, 25, -80, -57, 57, 80, -25, -90, -9, 87, 43, -70, -70, 43, 87, -9, -90, -25, 80, 57, -57, -80, 25, 90, 9, -87, -43, 70],
    [67, -54, -78, 38, 85, -22, -90, 4, 90, 13, -88, -31, 82, 46, -73, -61, 61, 73, -46, -82, 31, 88, -13, -90, -4, 90, 22, -85, -38, 78, 54, -67],
    [64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64],
    [61, -73, -46, 82, 31, -88, -13, 90, -4, -90, 22, 85, -38, -78, 54, 67, -67, -54, 78, 38, -85, -22, 90, 4, -90, 13, 88, -31, -82, 46, 73, -61],
    [57, -80, -25, 90, -9, -87, 43, 70, -70, -43, 87, 9, -90, 25, 80, -57, -57, 80, 25, -90, 9, 87, -43, -70, 70, 43, -87, -9, 90, -25, -80, 57],
    [54, -85, -4, 88, -46, -61, 82, 13, -90, 38, 67, -78, -22, 90, -31, -73, 73, 31, -90, 22, 78, -67, -38, 90, -13, -82, 61, 46, -88, 4, 85, -54],
    [50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50, 50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50],
    [46, -90, 38, 54, -90, 31, 61, -88, 22, 67, -85, 13, 73, -82, 4, 78, -78, -4, 82, -73, -13, 85, -67, -22, 88, -61, -31, 90, -54, -38, 90, -46],
    [43, -90, 57, 25, -87, 70, 9, -80, 80, -9, -70, 87, -25, -57, 90, -43, -43, 90, -57, -25, 87, -70, -9, 80, -80, 9, 70, -87, 25, 57, -90, 43],
    [38, -88, 73, -4, -67, 90, -46, -31, 85, -78, 13, 61, -90, 54, 22, -82, 82, -22, -54, 90, -61, -13, 78, -85, 31, 46, -90, 67, 4, -73, 88, -38],
    [36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36],
    [31, -78, 90, -61, 4, 54, -88, 82, -38, -22, 73, -90, 67, -13, -46, 85, -85, 46, 13, -67, 90, -73, 22, 38, -82, 88, -54, -4, 61, -90, 78, -31],
    [25, -70, 90, -80, 43, 9, -57, 87, -87, 57, -9, -43, 80, -90, 70, -25, -25, 70, -90, 80, -43, -9, 57, -87, 87, -57, 9, 43, -80, 90, -70, 25],
    [22, -61, 85, -90, 73, -38, -4, 46, -78, 90, -82, 54, -13, -31, 67, -88, 88, -67, 31, 13, -54, 82, -90, 78, -46, 4, 38, -73, 90, -85, 61, -22],
    [18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18, 18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18],
    [13, -38, 61, -78, 88, -90, 85, -73, 54, -31, 4, 22, -46, 67, -82, 90, -90, 82, -67, 46, -22, -4, 31, -54, 73, -85, 90, -88, 78, -61, 38, -13],
    [9, -25, 43, -57, 70, -80, 87, -90, 90, -87, 80, -70, 57, -43, 25, -9, -9, 25, -43, 57, -70, 80, -87, 90, -90, 87, -80, 70, -57, 43, -25, 9],
    [4, -13, 22, -31, 38, -46, 54, -61, 67, -73, 78, -82, 85, -88, 90, -90, 90, -90, 88, -85, 82, -78, 73, -67, 61, -54, 46, -38, 31, -22, 13, -4],
];

#[cfg(test)]
/// Coefficient `n` of basis vector `k` of the `2^log2`-point DCT.
fn basis(log2: u8, k: usize, n: usize) -> i32 {
    i32::from(DCT32[k << (MAX_LOG2 - log2)][n])
}

/// Side of the largest transform.
pub const MAX_SIZE: usize = 1 << MAX_LOG2;
/// Coefficients of the largest transform.
pub const MAX_COEFFS: usize = MAX_SIZE * MAX_SIZE;
#[cfg(test)]
/// Half the largest transform: the size of the butterflies' scratch arrays.
const HALF_MAX: usize = 1 << (MAX_LOG2 - 1);

#[cfg(test)]
/// One-dimensional forward DCT of `x` (length `n`, a power of two from 4
/// to 32) into `out`, by even/odd decomposition (the "partial butterfly"
/// of the HM reference encoder). The `N`-point matrix has even rows equal
/// to the `N/2`-point matrix applied to `x[i] + x[N−1−i]`, and odd rows
/// that only need `x[i] − x[N−1−i]`, so each level halves the work.
/// Products are exact, so the result equals the plain matrix product.
fn forward_1d(x: &[i64], out: &mut [i64]) {
    let n = x.len();
    if n == 1 {
        out[0] = x[0] * i64::from(DCT32[0][0]);
        return;
    }
    let half = n / 2;
    let log2 = n.trailing_zeros() as u8;
    // Stack buffers: this runs once per row and column of every block.
    let (mut even, mut odd, mut even_out) = ([0i64; HALF_MAX], [0i64; HALF_MAX], [0i64; HALF_MAX]);
    for i in 0..half {
        even[i] = x[i] + x[n - 1 - i];
        odd[i] = x[i] - x[n - 1 - i];
    }
    forward_1d(&even[..half], &mut even_out[..half]);
    for (j, &e) in even_out[..half].iter().enumerate() {
        out[2 * j] = e;
    }
    for k in (1..n).step_by(2) {
        out[k] = (0..half)
            .map(|i| i64::from(basis(log2, k, i)) * odd[i])
            .sum();
    }
}

#[cfg(test)]
/// One-dimensional inverse DCT of coefficients `c` into `out`: the even
/// coefficients through the half-size inverse give `E`, the odd ones `O`,
/// and `out[i] = E[i] + O[i]`, `out[N−1−i] = E[i] − O[i]`.
fn inverse_1d(c: &[i64], out: &mut [i64]) {
    let n = c.len();
    if n == 1 {
        out[0] = c[0] * i64::from(DCT32[0][0]);
        return;
    }
    let half = n / 2;
    let log2 = n.trailing_zeros() as u8;
    let (mut even_in, mut even) = ([0i64; HALF_MAX], [0i64; HALF_MAX]);
    for j in 0..half {
        even_in[j] = c[2 * j];
    }
    inverse_1d(&even_in[..half], &mut even[..half]);
    for i in 0..half {
        let odd: i64 = (1..n)
            .step_by(2)
            .filter(|&k| c[k] != 0)
            .map(|k| i64::from(basis(log2, k, i)) * c[k])
            .sum();
        out[i] = even[i] + odd;
        out[n - 1 - i] = even[i] - odd;
    }
}

/// Largest basis coefficient magnitude (`transMatrix` entries are ≤ 90).
const MAX_BASIS: u64 = 90;

/// An upper bound on every coefficient magnitude [`forward`] can produce
/// for a residual whose absolute values sum to `abs_sum`. Each pass sums
/// at most `MAX_BASIS` times its input magnitudes and rounds (adding at
/// most one per output); the second pass's share of the first pass's
/// rounding, `MAX_BASIS · n / 2^(log2 + 6)`, is below 2.
#[must_use]
pub fn forward_bound(log2: u8, abs_sum: u64) -> u64 {
    let shift = 2 * u32::from(log2) + BIT_DEPTH - 3;
    ((MAX_BASIS * MAX_BASIS * abs_sum) >> shift) + 4
}

#[cfg(test)]
/// Forward 2-D DCT of a `2^log2`-square residual block (row-major), as in
/// the HM encoder: rows first with shift `log2 + BitDepth − 9`, then
/// columns with shift `log2 + 6`.
pub fn forward(log2: u8, residual: &[i16], coeffs: &mut [i32]) {
    let n = 1usize << log2;
    let shift1 = u32::from(log2) + BIT_DEPTH - 9;
    let shift2 = u32::from(log2) + 6;
    // Stack scratch: no allocation per transform.
    let mut tmp = [0i64; MAX_COEFFS];
    let (mut line, mut out) = ([0i64; MAX_SIZE], [0i64; MAX_SIZE]);
    let (line, out) = (&mut line[..n], &mut out[..n]);
    for y in 0..n {
        for (l, &r) in line.iter_mut().zip(&residual[y * n..(y + 1) * n]) {
            *l = i64::from(r);
        }
        forward_1d(line, out);
        for (k, &v) in out.iter().enumerate() {
            tmp[y * n + k] = i64::from(round_shift_i64(v, shift1));
        }
    }
    for x in 0..n {
        for (y, l) in line.iter_mut().enumerate() {
            *l = tmp[y * n + x];
        }
        forward_1d(line, out);
        for (k, &v) in out.iter().enumerate() {
            coeffs[k * n + x] = round_shift_i64(v, shift2).clamp(COEFF_MIN, COEFF_MAX);
        }
    }
}

#[cfg(test)]
/// Normative inverse 2-D DCT (HEVC §8.6.4.2) of scaled coefficients `d`
/// (row-major, `d[y * n + x]`), producing the residual.
pub fn inverse(log2: u8, d: &[i32], residual: &mut [i32]) {
    let n = 1usize << log2;
    let (mut line, mut out) = ([0i64; MAX_SIZE], [0i64; MAX_SIZE]);
    let (line, out) = (&mut line[..n], &mut out[..n]);
    // Stage 1: each column (fixed x) is inverse transformed, then clipped.
    let mut g = [0i32; MAX_COEFFS];
    for x in 0..n {
        for (k, l) in line.iter_mut().enumerate() {
            *l = i64::from(d[k * n + x]);
        }
        inverse_1d(line, out);
        for (y, &v) in out.iter().enumerate() {
            g[y * n + x] = round_shift_i64(v, INV_SHIFT_1).clamp(COEFF_MIN, COEFF_MAX);
        }
    }
    // Stage 2: each row (fixed y).
    for y in 0..n {
        for (l, &v) in line.iter_mut().zip(&g[y * n..(y + 1) * n]) {
            *l = i64::from(v);
        }
        inverse_1d(line, out);
        for (r, &v) in residual[y * n..(y + 1) * n].iter_mut().zip(out.iter()) {
            *r = round_shift_i64(v, INV_SHIFT_2);
        }
    }
}

#[cfg(test)]
fn round_shift_i64(value: i64, shift: u32) -> i32 {
    ((value + (1 << (shift - 1))) >> shift) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eight_point_basis_matches_the_specification() {
        // HEVC §8.6.4.2: second row of the 8-point matrix.
        let row: Vec<i32> = (0..8).map(|n| basis(3, 1, n)).collect();
        assert_eq!(row, [89, 75, 50, 18, -18, -50, -75, -89]);
    }

    /// The plain matrix products the butterflies must reproduce exactly.
    fn matrix_1d(log2: u8, x: &[i64], inverse: bool) -> Vec<i64> {
        let n = x.len();
        (0..n)
            .map(|o| {
                (0..n)
                    .map(|i| {
                        let b = if inverse {
                            basis(log2, i, o)
                        } else {
                            basis(log2, o, i)
                        };
                        i64::from(b) * x[i]
                    })
                    .sum()
            })
            .collect()
    }

    #[test]
    fn butterflies_equal_the_matrix_product() {
        let mut seed = 0x1234_5678_9abc_def0u64;
        for log2 in MIN_LOG2..=MAX_LOG2 {
            let n = 1usize << log2;
            for _ in 0..50 {
                let x: Vec<i64> = (0..n)
                    .map(|_| {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        (seed % 65536) as i64 - 32768
                    })
                    .collect();
                let mut out = vec![0; n];
                forward_1d(&x, &mut out);
                assert_eq!(out, matrix_1d(log2, &x, false));
                inverse_1d(&x, &mut out);
                assert_eq!(out, matrix_1d(log2, &x, true));
            }
        }
    }

    #[test]
    fn forward_bound_holds() {
        let mut seed = 0xdead_beef_cafe_f00du64;
        for log2 in MIN_LOG2..=MAX_LOG2 {
            let n = 1usize << log2;
            for amplitude in [1i16, 3, 20, 255] {
                let residual: Vec<i16> = (0..n * n)
                    .map(|_| {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        (seed % (2 * amplitude as u64 + 1)) as i16 - amplitude
                    })
                    .collect();
                let abs_sum = residual.iter().map(|r| u64::from(r.unsigned_abs())).sum();
                let mut c = vec![0; n * n];
                forward(log2, &residual, &mut c);
                let max = c.iter().map(|v| u64::from(v.unsigned_abs())).max().unwrap();
                assert!(
                    max <= forward_bound(log2, abs_sum),
                    "log2 {log2} amp {amplitude}"
                );
            }
        }
    }

    #[test]
    fn dc_shortcut_matches_every_scaled_coefficient() {
        // All dequantized coefficients are clamped to this signed range.
        // The 4x4 reference exercises every rounding boundary exhaustively;
        // larger SIMD transforms additionally cover both extrema and signs.
        let mut coeffs = [0; 16];
        let mut out = [0; 16];
        for dc in -32768..=32767 {
            coeffs[0] = dc;
            inverse(2, &coeffs, &mut out);
            assert!(out.iter().all(|&v| v == inverse_dc(dc)), "dc {dc}");
        }
        let dsp = nc_dsp::Dsp::new();
        for log2 in MIN_LOG2..=MAX_LOG2 {
            let n = 1usize << log2;
            let mut coeffs = vec![0; n * n];
            let mut out = vec![0; n * n];
            for dc in (-32768..=32767).step_by(127).chain([32767, -1, 0, 1]) {
                coeffs[0] = dc;
                dsp.hevc_inverse(nc_dsp::HevcTransform::dct(log2).unwrap(), &coeffs, &mut out);
                assert!(
                    out.iter().all(|&v| v == inverse_dc(dc)),
                    "log2 {log2} dc {dc}"
                );
            }
        }
    }

    #[test]
    fn dc_coefficient_round_trips_to_a_flat_residual() {
        for log2 in MIN_LOG2..=MAX_LOG2 {
            let n = 1usize << log2;
            let residual = vec![20i16; n * n];
            let mut coeffs = vec![0; n * n];
            forward(log2, &residual, &mut coeffs);
            assert!(coeffs[1..].iter().all(|&c| c == 0), "log2 {log2}");
            // The forward and inverse transforms are scaled to cancel.
            let mut out = vec![0; n * n];
            inverse(log2, &coeffs, &mut out);
            assert!(
                out.iter().all(|&r| (r - 20).abs() <= 1),
                "log2 {log2}: {:?}",
                &out[..4]
            );
        }
    }

    #[test]
    fn nc_dsp_kernels_match_the_reference() {
        use crate::quant::{dequantize, quantize};
        use nc_dsp::{Dsp, HevcQuantizer, HevcTransform};
        let dsp = Dsp::new();
        let mut seed = 0x5151_7777_abcd_0123u64;
        for log2 in MIN_LOG2..=MAX_LOG2 {
            let n = 1usize << log2;
            let t = HevcTransform::dct(log2).unwrap();
            for (qp, intra) in [(0, true), (22, false), (37, true), (51, false)] {
                let residual: Vec<i16> = (0..n * n)
                    .map(|_| {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        (seed % 511) as i16 - 255
                    })
                    .collect();
                let (mut want, mut got) = (vec![0; n * n], vec![0; n * n]);
                forward(log2, &residual, &mut want);
                dsp.hevc_forward(t, &residual, n, &mut got);
                assert_eq!(got, want, "forward {n}");
                let rounding = if intra {
                    crate::quant::INTRA_ROUNDING_512THS
                } else {
                    crate::quant::INTER_ROUNDING_512THS
                };
                let q = HevcQuantizer::with_rounding(qp as u8, log2, rounding);
                quantize(&mut want, qp, log2, intra);
                dsp.hevc_quantize(&mut got, &q);
                assert_eq!(got, want, "quantise {n} qp {qp}");
                let (mut sw, mut sg) = (vec![0; n * n], vec![0; n * n]);
                dequantize(&want, qp, log2, &mut sw);
                dsp.hevc_dequantize(&got, &q, &mut sg);
                assert_eq!(sg, sw, "dequantise {n} qp {qp}");
                let (mut rw, mut rg) = (vec![0; n * n], vec![0; n * n]);
                inverse(log2, &sw, &mut rw);
                dsp.hevc_inverse(t, &sg, &mut rg);
                assert_eq!(rg, rw, "inverse {n}");
            }
        }
    }
}
