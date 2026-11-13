//! Combining predictions into the final block (AV1 §7.11.3.1, §7.11.3.10,
//! §7.11.3.14, §7.11.3.15): compound average and distance-weighted
//! average, mask blends for compound and inter-intra prediction, and the
//! OBMC blend with a neighbour's prediction.
//!
//! Compound inputs are at compound precision ([`Dsp::av1_predict_compound`](crate::Dsp::av1_predict_compound),
//! `InterPostRound` bits above pixels); inter-intra and OBMC inputs are
//! 8-bit samples.

use nc_common::{PlaneView, PlaneViewMut};

use super::{INTER_POST_ROUND, clip_pixel, round2};
use crate::Dsp;

/// Bits of a blend mask weight (`AOM_BLEND_A64_ROUND_BITS`): weights run
/// from 0 to 64.
pub const AV1_MASK_BITS: u32 = 6;
/// The weight that takes all of the first input.
pub const AV1_MASK_MAX: u8 = 1 << AV1_MASK_BITS;
/// Widest block (luma samples).
const MAX_BLOCK: usize = 128;
/// Bits of the distance weights, which sum to 16 (`DIST_PRECISION_BITS`).
const DISTANCE_BITS: u32 = 4;
/// `MAX_FRAME_DISTANCE`: order-hint distances are clamped to this.
const MAX_FRAME_DISTANCE: u32 = 31;
/// `Quant_Dist_Weight` (AV1 §7.11.3.15): the distance ratio thresholds.
const QUANT_DIST_WEIGHT: [[u32; 2]; 4] = [[2, 3], [2, 5], [2, 7], [1, MAX_FRAME_DISTANCE]];
/// `Quant_Dist_Lookup` (AV1 §7.11.3.15): the weight pairs.
const QUANT_DIST_LOOKUP: [[u8; 2]; 4] = [[9, 7], [11, 5], [12, 4], [13, 3]];

/// `obmc_mask_N` (AV1 §7.11.3.10, `get_obmc_mask`): the weight of the
/// block's own prediction by distance from the shared edge, for overlaps of
/// 2 to 32 samples (an overlap of 1 keeps the block's own prediction).
const OBMC_MASK_2: [u8; 2] = [45, 64];
const OBMC_MASK_4: [u8; 4] = [39, 50, 59, 64];
const OBMC_MASK_8: [u8; 8] = [36, 42, 48, 53, 57, 61, 64, 64];
const OBMC_MASK_16: [u8; 16] = [
    34, 37, 40, 43, 46, 49, 52, 54, 56, 58, 60, 61, 64, 64, 64, 64,
];
const OBMC_MASK_32: [u8; 32] = [
    33, 35, 36, 38, 40, 41, 43, 44, 45, 47, 48, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60, 60, 61, 62,
    64, 64, 64, 64, 64, 64, 64, 64,
];

/// A compound prediction: `h` rows of `w` samples at compound precision,
/// `stride` apart.
#[derive(Debug, Clone, Copy)]
pub struct Av1CompoundPred<'a> {
    /// The samples.
    pub data: &'a [i16],
    /// Distance between rows.
    pub stride: usize,
}

impl<'a> Av1CompoundPred<'a> {
    /// The first `w` samples of row `row`.
    fn row(&self, row: usize, w: usize) -> &'a [i16] {
        &self.data[row * self.stride..][..w]
    }

    fn at(&self, row: usize, column: usize) -> i32 {
        i32::from(self.data[row * self.stride + column])
    }
}

/// A blend mask at the resolution of the block's luma: weights 0 to 64 for
/// the first input, `stride` apart. A plane subsampled by `(sub_x, sub_y)`
/// averages the 2 or 4 luma weights each of its samples covers (AV1
/// §7.11.3.14).
#[derive(Debug, Clone, Copy)]
pub struct Av1Mask<'a> {
    /// The weights.
    pub data: &'a [u8],
    /// Distance between rows.
    pub stride: usize,
    /// Horizontal subsampling of the blended plane relative to the mask.
    pub sub_x: bool,
    /// Vertical subsampling of the blended plane relative to the mask.
    pub sub_y: bool,
}

impl Av1Mask<'_> {
    /// The weights of plane row `row`, columns `0..out.len()` (as
    /// [`Self::weight`], a row at a time).
    fn weights_row(&self, row: usize, out: &mut [i16]) {
        let line = |r: usize| &self.data[r * self.stride..];
        let n = out.len();
        match (self.sub_x, self.sub_y) {
            (false, false) => {
                for (o, &m) in out.iter_mut().zip(&line(row)[..n]) {
                    *o = i16::from(m);
                }
            }
            (true, false) => {
                for (o, pair) in out
                    .iter_mut()
                    .zip(line(row)[..2 * n].as_chunks::<2>().0.iter())
                {
                    *o = ((u16::from(pair[0]) + u16::from(pair[1]) + 1) >> 1) as i16;
                }
            }
            (false, true) => {
                let (top, bottom) = (line(2 * row), line(2 * row + 1));
                for ((o, &t), &b) in out.iter_mut().zip(&top[..n]).zip(&bottom[..n]) {
                    *o = ((u16::from(t) + u16::from(b) + 1) >> 1) as i16;
                }
            }
            (true, true) => {
                let (top, bottom) = (line(2 * row), line(2 * row + 1));
                let pairs = top[..2 * n]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .zip(bottom[..2 * n].as_chunks::<2>().0.iter());
                for (o, (t, b)) in out.iter_mut().zip(pairs) {
                    let sum = u16::from(t[0]) + u16::from(t[1]) + u16::from(b[0]) + u16::from(b[1]);
                    *o = ((sum + 2) >> 2) as i16;
                }
            }
        }
    }

    /// The weight of plane sample `(row, column)`.
    fn weight(&self, row: usize, column: usize) -> i32 {
        let at = |r: usize, c: usize| i32::from(self.data[r * self.stride + c]);
        match (self.sub_x, self.sub_y) {
            (false, false) => at(row, column),
            (true, false) => round2(at(row, 2 * column) + at(row, 2 * column + 1), 1),
            (false, true) => round2(at(2 * row, column) + at(2 * row + 1, column), 1),
            (true, true) => round2(
                at(2 * row, 2 * column)
                    + at(2 * row, 2 * column + 1)
                    + at(2 * row + 1, 2 * column)
                    + at(2 * row + 1, 2 * column + 1),
                2,
            ),
        }
    }
}

/// Write `value(row, column)` to every sample of the `w`x`h` block.
fn fill(
    dst: &mut PlaneViewMut<'_>,
    (w, h): (usize, usize),
    mut value: impl FnMut(usize, usize) -> u8,
) {
    assert!(
        dst.width() >= w && dst.height() >= h,
        "AV1 blend destination {}x{} is smaller than {w}x{h}",
        dst.width(),
        dst.height()
    );
    for r in 0..h {
        for (c, out) in dst.row_mut(r)[..w].iter_mut().enumerate() {
            *out = value(r, c);
        }
    }
}

/// Compound average (AV1 §7.11.3.1, `compound_idx` set):
/// `Clip1(Round2(p0 + p1, 1 + InterPostRound))`.
pub fn av1_blend_average(
    p0: Av1CompoundPred<'_>,
    p1: Av1CompoundPred<'_>,
    size: (usize, usize),
    dst: &mut PlaneViewMut<'_>,
) {
    fill(dst, size, |r, c| {
        clip_pixel(round2(p0.at(r, c) + p1.at(r, c), 1 + INTER_POST_ROUND))
    });
}

/// The distance weights of a compound block (AV1 §7.11.3.15) from the
/// order-hint distances of its two references to the current frame
/// (`get_relative_dist`, any sign): the weights of the first and second
/// prediction, summing to 16.
#[must_use]
pub fn av1_distance_weights(distance_ref0: i32, distance_ref1: i32) -> [u8; 2] {
    let clamp = |d: i32| d.unsigned_abs().min(MAX_FRAME_DISTANCE);
    let (d0, d1) = (clamp(distance_ref1), clamp(distance_ref0));
    let order = usize::from(d0 <= d1);
    let row = if d0 == 0 || d1 == 0 {
        QUANT_DIST_LOOKUP.len() - 1
    } else {
        (0..QUANT_DIST_LOOKUP.len() - 1)
            .find(|&i| {
                let (c0, c1) = (QUANT_DIST_WEIGHT[i][order], QUANT_DIST_WEIGHT[i][1 - order]);
                if order == 1 {
                    d0 * c0 > d1 * c1
                } else {
                    d0 * c0 < d1 * c1
                }
            })
            .unwrap_or(QUANT_DIST_LOOKUP.len() - 1)
    };
    [
        QUANT_DIST_LOOKUP[row][order],
        QUANT_DIST_LOOKUP[row][1 - order],
    ]
}

/// Distance-weighted compound average (AV1 §7.11.3.1, `compound_idx`
/// clear): `Clip1(Round2(w0 * p0 + w1 * p1, 4 + InterPostRound))` with the
/// [`av1_distance_weights`].
pub fn av1_blend_distance(
    p0: Av1CompoundPred<'_>,
    p1: Av1CompoundPred<'_>,
    [w0, w1]: [u8; 2],
    size: (usize, usize),
    dst: &mut PlaneViewMut<'_>,
) {
    let (w0, w1) = (i32::from(w0), i32::from(w1));
    fill(dst, size, |r, c| {
        clip_pixel(round2(
            w0 * p0.at(r, c) + w1 * p1.at(r, c),
            DISTANCE_BITS + INTER_POST_ROUND,
        ))
    });
}

/// Masked compound blend (AV1 §7.11.3.14, wedge and difference-weighted
/// compound): `Clip1(Round2(m * p0 + (64 - m) * p1, 6 + InterPostRound))`,
/// a row at a time through `dsp`'s blend kernel.
///
/// # Panics
///
/// If `dst` does not cover the block, the block is wider than 128
/// samples, or a prediction or the mask is too short.
pub fn av1_blend_mask(
    dsp: &Dsp,
    p0: Av1CompoundPred<'_>,
    p1: Av1CompoundPred<'_>,
    mask: Av1Mask<'_>,
    (w, h): (usize, usize),
    dst: &mut PlaneViewMut<'_>,
) {
    assert!(
        dst.width() >= w && dst.height() >= h,
        "AV1 blend destination {}x{} is smaller than {w}x{h}",
        dst.width(),
        dst.height()
    );
    let mut weights = [0i16; MAX_BLOCK];
    let weights = &mut weights[..w];
    for r in 0..h {
        mask.weights_row(r, weights);
        dsp.av1_mask_blend_row(
            p0.row(r, w),
            p1.row(r, w),
            weights,
            &mut dst.row_mut(r)[..w],
        );
    }
}

/// Inter-intra blend (AV1 §7.11.3.14 with `IsInterIntra`), in place: `dst`
/// holds the 8-bit inter prediction and becomes
/// `Round2(m * intra + (64 - m) * inter, 6)`.
pub fn av1_blend_interintra(
    intra: &PlaneView<'_>,
    mask: Av1Mask<'_>,
    (w, h): (usize, usize),
    dst: &mut PlaneViewMut<'_>,
) {
    blend_in_place(dst, intra, (w, h), |r, c| mask.weight(r, c));
}

/// `dst = Round2(m * a + (64 - m) * dst, 6)` over the block, with `m` from
/// `weight(row, column)`.
fn blend_in_place(
    dst: &mut PlaneViewMut<'_>,
    a: &PlaneView<'_>,
    (w, h): (usize, usize),
    weight: impl Fn(usize, usize) -> i32,
) {
    assert!(
        dst.width() >= w && dst.height() >= h && a.width() >= w && a.height() >= h,
        "AV1 blend of {w}x{h}: inputs too small"
    );
    let max = i32::from(AV1_MASK_MAX);
    for r in 0..h {
        let source = &a.row(r)[..w];
        for (c, (out, &s)) in dst.row_mut(r)[..w].iter_mut().zip(source).enumerate() {
            let m = weight(r, c);
            *out = round2(
                m * i32::from(s) + (max - m) * i32::from(*out),
                AV1_MASK_BITS,
            ) as u8;
        }
    }
}

/// Which neighbour's prediction an OBMC blend overlaps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Av1ObmcEdge {
    /// The block above: the weights vary down the rows.
    Above,
    /// The block to the left: the weights vary along the columns.
    Left,
}

/// The OBMC weights of the block's own prediction for an overlap of
/// `length` samples (`get_obmc_mask`).
fn obmc_mask(length: usize) -> &'static [u8] {
    match length {
        1 => &[AV1_MASK_MAX],
        2 => &OBMC_MASK_2,
        4 => &OBMC_MASK_4,
        8 => &OBMC_MASK_8,
        16 => &OBMC_MASK_16,
        32 => &OBMC_MASK_32,
        _ => panic!("OBMC overlap of {length} samples"),
    }
}

/// OBMC blend (AV1 §7.11.3.10), in place: `dst` holds the block's own
/// prediction over the `w`x`h` overlap region and becomes
/// `Round2(m * dst + (64 - m) * obmc, 6)`, where `obmc` is the neighbour's
/// prediction and `m` the weight by distance from the shared `edge` (the
/// overlap is `h` rows for [`Av1ObmcEdge::Above`], `w` columns for
/// [`Av1ObmcEdge::Left`]; 1, 2, 4, 8, 16 or 32).
pub fn av1_obmc_blend(
    obmc: &PlaneView<'_>,
    edge: Av1ObmcEdge,
    (w, h): (usize, usize),
    dst: &mut PlaneViewMut<'_>,
) {
    let max = i32::from(AV1_MASK_MAX);
    let mask = obmc_mask(if edge == Av1ObmcEdge::Above { h } else { w });
    // Blending the neighbour's prediction with weight 64 - m is blending the
    // block's own with weight m.
    blend_in_place(dst, obmc, (w, h), |r, c| {
        max - i32::from(if edge == Av1ObmcEdge::Above {
            mask[r]
        } else {
            mask[c]
        })
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Equal distances weigh both references equally after the lookup's
    /// tie rule, and swapping the distances swaps the weights.
    #[test]
    fn distance_weights_sum_to_16_and_mirror() {
        for d0 in -40..40 {
            for d1 in -40..40 {
                let [a, b] = av1_distance_weights(d0, d1);
                assert_eq!(a + b, 16);
                if d0.unsigned_abs().min(31) != d1.unsigned_abs().min(31) {
                    assert_eq!(av1_distance_weights(d1, d0), [b, a]);
                }
            }
        }
    }
}
