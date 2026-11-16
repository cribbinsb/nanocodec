//! VVC motion compensation from a reference picture of another size
//! (reference picture resampling, H.266 §8.5.6.3.2 and §8.5.6.3.4 with
//! `RprConstraintsActiveFlag`): every predicted sample has its own
//! reference position (`xIntL`, `xFracL` from the scaled position) and the
//! ratio picks the filter set. Separable like the unscaled kernels: each
//! column is filtered horizontally over the rows the block needs, then each
//! row vertically. Scalar and generic over the sample type: only pictures
//! predicted across a resolution switch take this path.
//!
//! Filter coefficients (the RPR luma, affine-luma and chroma sets) are those
//! of the BSD-licensed VTM `InterpolationFilter`; see `THIRD_PARTY.md`.

use nc_common::{BitDepth, PlaneView, Sample};

use super::{VVC_AFFINE_LUMA, VVC_CHROMA, VVC_LUMA};

/// Fractional bits of the scaled sample positions (`ScalingRatio::BITS −
/// 4`: positions in 1/1024 samples).
pub const VVC_RPR_POSITION_BITS: u32 = 10;
/// Fractional bits of a luma and a chroma filter phase (1/16, 1/32).
const LUMA_PHASE_BITS: u32 = 4;
const CHROMA_PHASE_BITS: u32 = 5;
/// The second (vertical) stage's shift, `shift2` (§8.5.6.3.2).
const SECOND_SHIFT: u32 = 6;
/// The luma half-sample position, where the alternative half-sample filter
/// (`hpelIfIdx` 1) applies.
const HALF_SAMPLE_PHASE: usize = 8;
/// The alternative luma half-sample filter (§8.5.6.3.2, Table 27 row
/// `hpelIfIdx` 1).
const ALTERNATIVE_HALF: [i16; 8] = [0, 3, 9, 20, 20, 9, 3, 0];

/// The filter set one axis of a scaled prediction uses (§8.5.6.3.2: chosen
/// from the axis's scaling ratio by the caller).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VvcRprFilter {
    /// The regular interpolation filter (ratio up to 1.25).
    Regular,
    /// The regular filter with the alternative half-sample filter (an
    /// unscaled axis of a block whose motion uses `hpelIfIdx` 1).
    AlternativeHalf,
    /// The filter for ratios above 1.25.
    Ratio1p5,
    /// The filter for ratios above 1.75.
    Ratio2,
}

/// One block predicted from a scaled reference.
#[derive(Debug, Clone, Copy)]
pub struct VvcScaledBlock {
    /// The predicted samples (width, height).
    pub size: (usize, usize),
    /// The reference position of the first sample, in 1/1024 samples of the
    /// reference plane (`xInt0`, `yInt0` before rounding, §8.5.6.3.1).
    pub origin: (i32, i32),
    /// The position step between samples, 1/1024 (`(scalingRatio + 8) >> 4`).
    pub step: (i32, i32),
    /// The first and last readable column, then row, of the reference
    /// plane (the picture's, or a subpicture's treated as a picture).
    pub bounds: [i32; 4],
}

/// The coefficients of filter `filter` (luma, affine luma or chroma) by
/// phase.
fn table(filter: VvcRprFilter, luma: bool, affine: bool) -> &'static [[i16; 8]] {
    match (filter, affine) {
        (VvcRprFilter::Ratio1p5, false) => &LUMA_RPR1,
        (VvcRprFilter::Ratio2, false) => &LUMA_RPR2,
        (VvcRprFilter::Ratio1p5, true) => &AFFINE_LUMA_RPR1,
        (VvcRprFilter::Ratio2, true) => &AFFINE_LUMA_RPR2,
        (_, true) => &VVC_AFFINE_LUMA,
        _ => {
            debug_assert!(luma);
            &VVC_LUMA
        }
    }
}

/// The chroma coefficients of `filter` by phase.
fn chroma_table(filter: VvcRprFilter) -> &'static [[i16; 4]] {
    match filter {
        VvcRprFilter::Ratio1p5 => &CHROMA_RPR1,
        VvcRprFilter::Ratio2 => &CHROMA_RPR2,
        _ => &VVC_CHROMA,
    }
}

/// The taps of one phase, as eight with the first at offset −3 (chroma's
/// four sit at −1..=2).
fn taps(filter: VvcRprFilter, (luma, affine): (bool, bool), phase: usize) -> [i32; 8] {
    if !luma {
        let c = chroma_table(filter)[phase];
        return [
            0,
            0,
            i32::from(c[0]),
            i32::from(c[1]),
            i32::from(c[2]),
            i32::from(c[3]),
            0,
            0,
        ];
    }
    if filter == VvcRprFilter::AlternativeHalf && phase == HALF_SAMPLE_PHASE && !affine {
        return ALTERNATIVE_HALF.map(i32::from);
    }
    table(filter, luma, affine)[phase].map(i32::from)
}

/// The integer sample and filter phase of scaled position `position`.
fn split(position: i32, phase_bits: u32) -> (i32, usize) {
    let rounded = position + (1 << (VVC_RPR_POSITION_BITS - phase_bits - 1));
    (
        rounded >> VVC_RPR_POSITION_BITS,
        ((rounded >> (VVC_RPR_POSITION_BITS - phase_bits)) & ((1 << phase_bits) - 1)) as usize,
    )
}

/// Predict `block` of a luma (`luma`, with `affine` its 6-tap filter) or
/// chroma plane from `reference` with `filters` (horizontal, vertical),
/// writing the 14-bit high-precision samples (as the unscaled
/// `predict_high_precision`) to `out`, row-major.
///
/// # Panics
///
/// If `out` is shorter than the block or the bounds lie outside the plane.
pub fn vvc_predict_scaled<T: Sample>(
    reference: &PlaneView<'_, T>,
    block: &VvcScaledBlock,
    filters: [VvcRprFilter; 2],
    (luma, affine): (bool, bool),
    depth: BitDepth,
    out: &mut [i16],
) {
    // Folds to the constant 8 in the 8-bit instance.
    let depth = depth.of::<T>();
    let (width, height) = block.size;
    let [left, top, right, bottom] = block.bounds;
    assert!(out.len() >= width * height);
    assert!(left >= 0 && top >= 0 && right >= left && bottom >= top);
    assert!((right as usize) < reference.width() && (bottom as usize) < reference.height());
    let phase_bits = if luma {
        LUMA_PHASE_BITS
    } else {
        CHROMA_PHASE_BITS
    };
    // shift1 = Min(4, BitDepth − 8) (§8.5.6.3.2).
    let first_shift = (depth.bits() - 8).min(4);
    let rows: Vec<(i32, usize)> = (0..height as i32)
        .map(|r| split(block.origin.1 + r * block.step.1, phase_bits))
        .collect();
    let first_row = rows[0].0 - 3;
    let span = (rows[rows.len() - 1].0 + 4 - first_row + 1) as usize;
    // Horizontally filtered rows first_row.., one column per predicted column.
    let mut filtered = vec![0i32; span * width];
    for column in 0..width {
        let (x, phase) = split(block.origin.0 + column as i32 * block.step.0, phase_bits);
        let coefficients = taps(filters[0], (luma, affine), phase);
        for row in 0..span {
            let y = (first_row + row as i32).clamp(top, bottom) as usize;
            let line = reference.row(y);
            let sum: i32 = coefficients
                .iter()
                .enumerate()
                .map(|(k, &c)| c * line[(x + k as i32 - 3).clamp(left, right) as usize].to_i32())
                .sum();
            filtered[row * width + column] = sum >> first_shift;
        }
    }
    for (r, &(y, phase)) in rows.iter().enumerate() {
        let coefficients = taps(filters[1], (luma, affine), phase);
        let base = (y - 3 - first_row) as usize;
        for column in 0..width {
            let sum: i32 = coefficients
                .iter()
                .enumerate()
                .map(|(k, &c)| c * filtered[(base + k) * width + column])
                .sum();
            out[r * width + column] = (sum >> SECOND_SHIFT) as i16;
        }
    }
}

/// Luma filter for scaling ratios above 1.25 (1.5x downsampling).
#[rustfmt::skip]
pub(crate) const LUMA_RPR1: [[i16; 8]; 16] = [
    [-1, -5, 17, 42, 17, -5, -1, 0],
    [0, -5, 15, 41, 19, -5, -1, 0],
    [0, -5, 13, 40, 21, -4, -1, 0],
    [0, -5, 11, 39, 24, -4, -2, 1],
    [0, -5, 9, 38, 26, -3, -2, 1],
    [0, -5, 7, 38, 28, -2, -3, 1],
    [1, -5, 5, 36, 30, -1, -3, 1],
    [1, -4, 3, 35, 32, 0, -4, 1],
    [1, -4, 2, 33, 33, 2, -4, 1],
    [1, -4, 0, 32, 35, 3, -4, 1],
    [1, -3, -1, 30, 36, 5, -5, 1],
    [1, -3, -2, 28, 38, 7, -5, 0],
    [1, -2, -3, 26, 38, 9, -5, 0],
    [1, -2, -4, 24, 39, 11, -5, 0],
    [0, -1, -4, 21, 40, 13, -5, 0],
    [0, -1, -5, 19, 41, 15, -5, 0],
];

/// Luma filter for scaling ratios above 1.75 (2x downsampling).
#[rustfmt::skip]
pub(crate) const LUMA_RPR2: [[i16; 8]; 16] = [
    [-4, 2, 20, 28, 20, 2, -4, 0],
    [-4, 0, 19, 29, 21, 5, -4, -2],
    [-4, -1, 18, 29, 22, 6, -4, -2],
    [-4, -1, 16, 29, 23, 7, -4, -2],
    [-4, -1, 16, 28, 24, 7, -4, -2],
    [-4, -1, 14, 28, 25, 8, -4, -2],
    [-3, -3, 14, 27, 26, 9, -3, -3],
    [-3, -1, 12, 28, 25, 10, -4, -3],
    [-3, -3, 11, 27, 27, 11, -3, -3],
    [-3, -4, 10, 25, 28, 12, -1, -3],
    [-3, -3, 9, 26, 27, 14, -3, -3],
    [-2, -4, 8, 25, 28, 14, -1, -4],
    [-2, -4, 7, 24, 28, 16, -1, -4],
    [-2, -4, 7, 23, 29, 16, -1, -4],
    [-2, -4, 6, 22, 29, 18, -1, -4],
    [-2, -4, 5, 21, 29, 19, 0, -4],
];

/// The affine (6-tap, in the 8-tap layout) filter for ratios above 1.25.
#[rustfmt::skip]
pub(crate) const AFFINE_LUMA_RPR1: [[i16; 8]; 16] = [
    [0, -6, 17, 42, 17, -5, -1, 0],
    [0, -5, 15, 41, 19, -5, -1, 0],
    [0, -5, 13, 40, 21, -4, -1, 0],
    [0, -5, 11, 39, 24, -4, -1, 0],
    [0, -5, 9, 38, 26, -3, -1, 0],
    [0, -5, 7, 38, 28, -2, -2, 0],
    [0, -4, 5, 36, 30, -1, -2, 0],
    [0, -3, 3, 35, 32, 0, -3, 0],
    [0, -3, 2, 33, 33, 2, -3, 0],
    [0, -3, 0, 32, 35, 3, -3, 0],
    [0, -2, -1, 30, 36, 5, -4, 0],
    [0, -2, -2, 28, 38, 7, -5, 0],
    [0, -1, -3, 26, 38, 9, -5, 0],
    [0, -1, -4, 24, 39, 11, -5, 0],
    [0, -1, -4, 21, 40, 13, -5, 0],
    [0, -1, -5, 19, 41, 15, -5, 0],
];

/// The affine filter for ratios above 1.75.
#[rustfmt::skip]
pub(crate) const AFFINE_LUMA_RPR2: [[i16; 8]; 16] = [
    [0, -2, 20, 28, 20, 2, -4, 0],
    [0, -4, 19, 29, 21, 5, -6, 0],
    [0, -5, 18, 29, 22, 6, -6, 0],
    [0, -5, 16, 29, 23, 7, -6, 0],
    [0, -5, 16, 28, 24, 7, -6, 0],
    [0, -5, 14, 28, 25, 8, -6, 0],
    [0, -6, 14, 27, 26, 9, -6, 0],
    [0, -4, 12, 28, 25, 10, -7, 0],
    [0, -6, 11, 27, 27, 11, -6, 0],
    [0, -7, 10, 25, 28, 12, -4, 0],
    [0, -6, 9, 26, 27, 14, -6, 0],
    [0, -6, 8, 25, 28, 14, -5, 0],
    [0, -6, 7, 24, 28, 16, -5, 0],
    [0, -6, 7, 23, 29, 16, -5, 0],
    [0, -6, 6, 22, 29, 18, -5, 0],
    [0, -6, 5, 21, 29, 19, -4, 0],
];

/// Chroma filter for ratios above 1.25.
#[rustfmt::skip]
pub(crate) const CHROMA_RPR1: [[i16; 4]; 32] = [
    [12, 40, 12, 0],
    [11, 40, 13, 0],
    [10, 40, 15, -1],
    [9, 40, 16, -1],
    [8, 40, 17, -1],
    [8, 39, 18, -1],
    [7, 39, 19, -1],
    [6, 38, 21, -1],
    [5, 38, 22, -1],
    [4, 38, 23, -1],
    [4, 37, 24, -1],
    [3, 36, 25, 0],
    [3, 35, 26, 0],
    [2, 34, 28, 0],
    [2, 33, 29, 0],
    [1, 33, 30, 0],
    [1, 31, 31, 1],
    [0, 30, 33, 1],
    [0, 29, 33, 2],
    [0, 28, 34, 2],
    [0, 26, 35, 3],
    [0, 25, 36, 3],
    [-1, 24, 37, 4],
    [-1, 23, 38, 4],
    [-1, 22, 38, 5],
    [-1, 21, 38, 6],
    [-1, 19, 39, 7],
    [-1, 18, 39, 8],
    [-1, 17, 40, 8],
    [-1, 16, 40, 9],
    [-1, 15, 40, 10],
    [0, 13, 40, 11],
];

/// Chroma filter for ratios above 1.75.
#[rustfmt::skip]
pub(crate) const CHROMA_RPR2: [[i16; 4]; 32] = [
    [17, 30, 17, 0],
    [17, 30, 18, -1],
    [16, 30, 18, 0],
    [16, 30, 18, 0],
    [15, 30, 18, 1],
    [14, 30, 18, 2],
    [13, 29, 19, 3],
    [13, 29, 19, 3],
    [12, 29, 20, 3],
    [11, 28, 21, 4],
    [10, 28, 22, 4],
    [10, 27, 22, 5],
    [9, 27, 23, 5],
    [9, 26, 24, 5],
    [8, 26, 24, 6],
    [7, 26, 25, 6],
    [7, 25, 25, 7],
    [6, 25, 26, 7],
    [6, 24, 26, 8],
    [5, 24, 26, 9],
    [5, 23, 27, 9],
    [5, 22, 27, 10],
    [4, 22, 28, 10],
    [4, 21, 28, 11],
    [3, 20, 29, 12],
    [3, 19, 29, 13],
    [3, 19, 29, 13],
    [2, 18, 30, 14],
    [1, 18, 30, 15],
    [0, 18, 30, 16],
    [0, 18, 30, 16],
    [-1, 18, 30, 17],
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Dsp, Filter};

    /// At ratio 1 (a step of one sample) the scaled path is the unscaled
    /// high-precision prediction.
    #[test]
    fn unit_ratio_matches_the_unscaled_kernel() {
        let samples: Vec<u8> = (0..40 * 40)
            .map(|i| ((i * 29 + i / 40 * 7) % 256) as u8)
            .collect();
        let plane = PlaneView::new(&samples, 40, 40, 40).unwrap();
        let unit = 1 << VVC_RPR_POSITION_BITS;
        for (luma, phase_bits) in [(true, 4), (false, 5)] {
            for phase in [(0, 0), (3, 9), (8, 8), ((1 << phase_bits) - 1, 1)] {
                let (x, y): (usize, usize) = (12, 10);
                let origin = (
                    x * unit + (phase.0 << (VVC_RPR_POSITION_BITS - phase_bits)),
                    y * unit + (phase.1 << (VVC_RPR_POSITION_BITS - phase_bits)),
                );
                let block = VvcScaledBlock {
                    size: (8, 4),
                    origin: (origin.0 as i32, origin.1 as i32),
                    step: (unit as i32, unit as i32),
                    bounds: [0, 0, 39, 39],
                };
                let mut scaled = [0i16; 32];
                let filters = [VvcRprFilter::Regular; 2];
                vvc_predict_scaled(
                    &plane,
                    &block,
                    filters,
                    (luma, false),
                    BitDepth::EIGHT,
                    &mut scaled,
                );
                let filter = if luma {
                    Filter::VvcLuma
                } else {
                    Filter::VvcChroma
                };
                let (before, taps) = (filter.before(), filter.taps());
                let start = (y - before) * 40 + (x - before);
                let window =
                    PlaneView::new(&samples[start..], 40, 8 + taps - 1, 4 + taps - 1).unwrap();
                let mut expected = [0i16; 32];
                Dsp::new().predict_high_precision(
                    filter,
                    (8, 4),
                    &window,
                    phase,
                    false,
                    &mut expected,
                );
                assert_eq!(scaled, expected, "luma {luma} phase {phase:?}");
            }
        }
    }
}
