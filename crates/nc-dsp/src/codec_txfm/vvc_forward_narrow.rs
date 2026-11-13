//! Exact VVC one-/two-sample strips using the shared HEVC line butterflies.
//! The dense matrix implementation remains the independent scalar reference.
//! Only the long dimension uses a butterfly; pass order and rounding match
//! `vvc_forward` (`VVenC` `TrQuant::xT`, BSD-3-Clause-Clear; `THIRD_PARTY.md`).
//! Partial butterflies reduce products but retain odd-frequency dot products.

use super::generic::{HevcFdct8, HevcFdct16, HevcFdct32, Line};
use crate::simd::I32Lanes;

const MAX_SIDE: usize = 32;
const MAX_AREA: usize = 2 * MAX_SIDE;
const BASIS_GAIN: i32 = 64;
const SINGLE_ROW_GAIN: i32 = 2 * BASIS_GAIN;

/// The public DSP entry has checked power-of-two sides and slice lengths.
/// Legal narrow shapes have one side 1/2, area 16..64 and long side 8/16/32.
/// With 8-bit residuals, narrow row results are at most 32640 in magnitude;
/// even the conservative long-pass bound 32*90*32640 fits i32. No clipping
/// or change in rounding is needed, including for negative half-way values.
#[inline(never)]
pub(super) fn forward(width: usize, height: usize, input: &[i16], out: &mut [i32]) {
    super::vvc_forward::check(width, height, input.len(), out.len());
    assert!(width.min(height) <= 2);
    let area = width * height;
    let input = &input[..area];
    let out = &mut out[..area];
    let mut intermediate = [0i32; MAX_AREA];
    let mut x = [0i32; MAX_SIDE];
    let mut y = [0i32; MAX_SIDE];
    if width <= 2 {
        // Keep columns contiguous for the long second pass. Width 1's
        // gain 128 substitutes for the absent log2(width)-1 = -1 shift.
        for (r, row) in input.chunks_exact(width).enumerate() {
            let a = i32::from(row[0]);
            if width == 1 {
                intermediate[r] = SINGLE_ROW_GAIN * a;
            } else {
                let b = i32::from(row[1]);
                intermediate[r] = BASIS_GAIN * (a + b);
                intermediate[height + r] = BASIS_GAIN * (a - b);
            }
        }
        for c in 0..width {
            long_line(
                &intermediate[c * height..(c + 1) * height],
                &mut y[..height],
            );
            for (r, &value) in y[..height].iter().enumerate() {
                out[r * width + c] = value.round_shift(height.ilog2() + 6);
            }
        }
    } else {
        // One or two long rows, rounded before the short column transform.
        for (r, row) in input.chunks_exact(width).enumerate() {
            for (value, &sample) in x[..width].iter_mut().zip(row) {
                *value = i32::from(sample);
            }
            long_line(&x[..width], &mut y[..width]);
            for (c, &value) in y[..width].iter().enumerate() {
                intermediate[r * width + c] = value.round_shift(width.ilog2() - 1);
            }
        }
        if height == 1 {
            // The sole column basis value 64, followed by shift 6, is identity.
            out.copy_from_slice(&intermediate[..width]);
        } else {
            let (upper, lower) = out.split_at_mut(width);
            for c in 0..width {
                let (a, b) = (intermediate[c], intermediate[width + c]);
                upper[c] = (BASIS_GAIN * (a + b)).round_shift(7);
                lower[c] = (BASIS_GAIN * (a - b)).round_shift(7);
            }
        }
    }
}

#[inline]
fn long_line(input: &[i32], output: &mut [i32]) {
    match input.len() {
        8 => HevcFdct8::apply(input, output),
        16 => HevcFdct16::apply(input, output),
        32 => HevcFdct32::apply(input, output),
        _ => unreachable!("validated narrow transform long side"),
    }
}
