//! Shared reversible 4×4 lifting transform used by VP9 and AV2.
//!
//! The lifting stages and pass order are identical in libvpx v1.16.0 and
//! final AVM v1.0.0. Codec wrappers retain their distinct coefficient scaling
//! and dequantization. See `THIRD_PARTY.md` for reference licensing.

fn forward_1d([mut a, mut b, mut c, mut d]: [i32; 4]) -> [i32; 4] {
    a += b;
    d -= c;
    let half = (a - d) >> 1;
    b = half - b;
    c = half - c;
    a -= c;
    d += b;
    [a, c, d, b]
}

fn inverse_1d([mut a, mut c, mut d, mut b]: [i32; 4]) -> [i32; 4] {
    a += c;
    d -= b;
    let half = (a - d) >> 1;
    b = half - b;
    c = half - c;
    a -= b;
    d += c;
    [a, b, c, d]
}

/// Transform a 4×4 raster residual into unscaled reversible coefficients.
/// Inputs must be in the codec residual range (8/10/12-bit sample differences).
#[must_use]
pub fn forward(input: &[i32; 16]) -> [i32; 16] {
    let mut columns = [0; 16];
    for x in 0..4 {
        let column = forward_1d(std::array::from_fn(|y| input[y * 4 + x]));
        for y in 0..4 {
            columns[y * 4 + x] = column[y];
        }
    }
    let mut output = [0; 16];
    for y in 0..4 {
        let row = forward_1d(std::array::from_fn(|x| columns[y * 4 + x]));
        output[y * 4..y * 4 + 4].copy_from_slice(&row);
    }
    output
}

/// Invert unscaled 4×4 reversible coefficients into a raster residual.
/// Coefficient magnitudes must be below 2²⁴, as ensured by codec dequantization.
#[must_use]
pub fn inverse(input: &[i32; 16]) -> [i32; 16] {
    let mut rows = [0; 16];
    for y in 0..4 {
        let row = inverse_1d(std::array::from_fn(|x| input[y * 4 + x]));
        rows[y * 4..y * 4 + 4].copy_from_slice(&row);
    }
    let mut output = [0; 16];
    for x in 0..4 {
        let column = inverse_1d(std::array::from_fn(|y| rows[y * 4 + x]));
        for y in 0..4 {
            output[y * 4 + x] = column[y];
        }
    }
    output
}
