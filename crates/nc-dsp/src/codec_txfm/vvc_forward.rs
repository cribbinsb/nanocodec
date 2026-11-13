//! H.266 rectangular forward DCT-II for the encoder: the `width` x `height`
//! separable transform with HEVC's integer basis, rows first.
//!
//! Pass shifts follow `VVenC` `TrQuant::xT` (BSD-3-Clause-Clear,
//! `THIRD_PARTY.md`): `log2(width) - 1` after the rows and
//! `log2(height) + 6` after the columns at 8-bit, so a square block gives
//! exactly HEVC's forward transform. The DCT-II bases nest: row `k` of the
//! `N`-point basis is row `k * 32 / N` of the 32-point one, restricted to
//! its first `N` samples (the 2-point basis included). A one-sample side
//! has no transform (§8.7.4.1), so a 1-wide block is one column pass at
//! `log2(height) - 1`, which the row pass's gain of 128 reproduces. A
//! plain matrix product (no butterflies): only rectangular blocks use it;
//! squares keep the shared HEVC kernels.
use super::DCT32;

/// Largest transform side.
const MAX_SIDE: usize = 32;

/// Forward DCT-II of a `width` x `height` residual (row-major, `width` per
/// row) into `out` (row-major coefficients).
///
/// # Panics
///
/// If a side is not a power of two up to 32, the block has fewer than 16
/// samples, or a slice is shorter than the block.
pub(crate) fn forward(width: usize, height: usize, residual: &[i16], out: &mut [i32]) {
    check(width, height, residual.len(), out.len());
    let area = width * height;
    // Row-pass scratch sized to the block: zeroing a 32x32 buffer for
    // every small block was 2.5 % of VVC Default's x86 instructions.
    match area {
        ..=16 => forward_sized::<16>((width, height), residual, out),
        17..=64 => forward_sized::<64>((width, height), residual, out),
        65..=256 => forward_sized::<256>((width, height), residual, out),
        _ => forward_sized::<{ MAX_SIDE * MAX_SIDE }>((width, height), residual, out),
    }
}

/// Shared public entry validation before selecting an ISA kernel.
pub(crate) fn check(width: usize, height: usize, input_len: usize, output_len: usize) {
    let side = |n: usize| n.is_power_of_two() && n <= MAX_SIDE;
    assert!(side(width) && side(height) && width * height >= MIN_SAMPLES);
    assert!(input_len >= width * height && output_len >= width * height);
}

/// [`forward`] of a block of at most `AREA` samples.
fn forward_sized<const AREA: usize>(
    (width, height): (usize, usize),
    residual: &[i16],
    out: &mut [i32],
) {
    let area = width * height;
    let mut rows = [0i32; AREA];
    for (y, samples) in residual[..area].chunks_exact(width).enumerate() {
        for k in 0..width {
            rows[k * height + y] = if width == 1 {
                i32::from(samples[0]) * UNTRANSFORMED_GAIN
            } else {
                let basis = &DCT32[k * MAX_SIDE / width][..width];
                let sum: i32 = basis
                    .iter()
                    .zip(samples)
                    .map(|(&b, &r)| i32::from(b) * i32::from(r))
                    .sum();
                round_shift(sum, width.ilog2() - 1)
            };
        }
    }
    let column_shift = height.ilog2() + 6;
    for k in 0..height {
        let basis = &DCT32[k * MAX_SIDE / height][..height];
        for x in 0..width {
            let column = &rows[x * height..(x + 1) * height];
            let sum: i32 = basis
                .iter()
                .zip(column)
                .map(|(&b, &c)| i32::from(b) * c)
                .sum();
            out[k * width + x] = round_shift(sum, column_shift);
        }
    }
}

/// The smallest block (a 4x4, or an intra sub-partition's 1x16 or 2x8).
const MIN_SAMPLES: usize = 16;
/// The row pass's gain on a one-sample row (no transform there): twice the
/// basis' 64, standing in for the missing `log2(width) - 1` = -1 shift.
const UNTRANSFORMED_GAIN: i32 = 128;

fn round_shift(value: i32, shift: u32) -> i32 {
    if shift == 0 {
        return value;
    }
    (value + (1 << (shift - 1))) >> shift
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Dsp, HevcTransform};

    fn residual(width: usize, height: usize) -> Vec<i16> {
        (0..width * height)
            .map(|i| ((i * 37 + i / width * 11) % 511) as i16 - 255)
            .collect()
    }

    #[test]
    fn square_blocks_match_the_hevc_forward() {
        let dsp = Dsp::new();
        for side in [4, 8, 16, 32] {
            let input = residual(side, side);
            let mut expected = vec![0; side * side];
            let tx = HevcTransform::dct(side.ilog2() as u8).unwrap();
            dsp.hevc_forward(tx, &input, side, &mut expected);
            let mut got = vec![0; side * side];
            forward(side, side, &input, &mut got);
            assert_eq!(got, expected, "{side}");
        }
    }

    #[test]
    fn rectangular_round_trip_is_close() {
        // Forward then the normative inverse at QP 4 (unit step) gives the
        // residual back to within the rounding a square block of the same
        // span shows (32x32: 4, 8x8: 1).
        let dsp = Dsp::new();
        for (width, height) in [
            (32, 16),
            (16, 32),
            (32, 8),
            (8, 32),
            (16, 4),
            (4, 16),
            (8, 2),
            (2, 8),
            (32, 2),
            (1, 16),
            (32, 1),
        ] {
            let input = residual(width, height);
            let mut coefficients = vec![0; width * height];
            forward(width, height, &input, &mut coefficients);
            let quantizer = crate::VvcQuantizer::rect(4, width.ilog2() as u8, height.ilog2() as u8);
            let mut levels = vec![0; width * height];
            dsp.vvc_quantize_from(&coefficients, &quantizer, &mut levels);
            let mut back = vec![0; width * height];
            let kinds = [crate::VvcTransformKind::Dct2; 2];
            dsp.vvc_inverse_primary((width, height), kinds, &levels, 4, false, &mut back);
            for (a, b) in input.iter().zip(&back) {
                assert!(
                    (i32::from(*a) - b).abs() <= 3,
                    "{width}x{height}: {a} vs {b}"
                );
            }
        }
    }
}
