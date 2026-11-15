//! DC prediction: the mean of the top and/or left edge, or mid-grey; with
//! intra bi-prediction the quarter of the block nearest each edge ramps
//! from the edge to the mean (`DC_IBP_WEIGHTS`).
//!
//! Ported from dav2d `src/ipred_tmpl.c` (`ipred_dc*_c`, `dc_gen*`,
//! `fast_div32_dc`; commit 237229e, BSD-2-Clause, see `docs/licenses/dav2d`).

use super::tables::{DC_IBP_WEIGHTS, DIV_RECIP};
use super::{ANGLE_IBP_FLAG, Block, SampleRef, clip_pixel, fill};

/// Mid-grey, the prediction without edges.
const DC_128: u8 = 128;
/// Bits of the bi-prediction weights (they blend out of 128).
const IBP_WEIGHT_BITS: u32 = 7;
/// Bits of the reciprocal in [`DIV_RECIP`] (512 is 1.0).
const RECIP_BITS: u32 = 9;
/// Fraction bits of a divisor's index into [`DIV_RECIP`].
const RECIP_INDEX_BITS: u32 = 7;
/// The bi-predicted part of a side is a quarter of it.
const IBP_PART_LOG2: u32 = 2;

/// `num / den` for `den` in 1..=255 by the reciprocal table, as AVM divides
/// sums of a non-power-of-two count of samples (`fast_div32_dc`).
pub(super) fn fast_div32_dc(num: u32, den: u32) -> u32 {
    debug_assert!(den > 0 && u8::try_from(den).is_ok());
    let shift = den.ilog2();
    let index = (den - (1 << shift)) << (RECIP_INDEX_BITS - shift);
    let shift = shift + RECIP_BITS;
    (num * u32::from(DIV_RECIP[index as usize]) + ((1 << shift) >> 1)) >> shift
}

/// The rounded mean of the `width` top samples (`dc_gen_top`).
fn mean_top(edge: SampleRef<'_>, width: usize) -> u32 {
    let sum: u32 = edge.slice(1, width).iter().map(|&p| u32::from(p)).sum();
    (sum + (width as u32 >> 1)) >> width.trailing_zeros()
}

/// The rounded mean of the `height` left samples (`dc_gen_left`).
fn mean_left(edge: SampleRef<'_>, height: usize) -> u32 {
    let sum: u32 = edge
        .slice(-(height as isize), height)
        .iter()
        .map(|&p| u32::from(p))
        .sum();
    (sum + (height as u32 >> 1)) >> height.trailing_zeros()
}

/// The mean of both edges (`dc_gen`): a shift when the count is a power of
/// two, else the reciprocal division.
fn mean_both(edge: SampleRef<'_>, width: usize, height: usize) -> u32 {
    let count = (width + height) as u32;
    let top: u32 = edge.slice(1, width).iter().map(|&p| u32::from(p)).sum();
    let left: u32 = edge
        .slice(-(height as isize), height)
        .iter()
        .map(|&p| u32::from(p))
        .sum();
    let sum = top + left;
    if count.is_power_of_two() {
        return (sum + (width as u32)) >> count.trailing_zeros();
    }
    u32::from(clip_pixel(fast_div32_dc(sum, count) as i32))
}

/// The `n` ramp weights of an `n`-sample bi-predicted part.
fn ramp(n: usize) -> &'static [u8] {
    &DC_IBP_WEIGHTS[n..2 * n]
}

/// Blends `edge_sample` towards `dc` by `weight` out of 128.
fn blend(edge_sample: u8, dc: u32, weight: u8) -> u8 {
    let weight = u32::from(weight);
    let full = 1 << IBP_WEIGHT_BITS;
    ((u32::from(edge_sample) * (full - weight) + dc * weight + (full >> 1)) >> IBP_WEIGHT_BITS)
        as u8
}

/// Rows `0..rows` of columns `from..width` ramp down from the top edge.
fn ramp_top(
    dst: &mut Block<'_>,
    edge: SampleRef<'_>,
    from: usize,
    width: usize,
    rows: usize,
    dc: u32,
) {
    let top = edge.slice(1, width);
    for (y, &weight) in ramp(rows).iter().enumerate() {
        let row = dst.row(y, width);
        for x in from..width {
            row[x] = blend(top[x], dc, weight);
        }
    }
}

/// Columns `0..cols` of rows `from..height` ramp across from the left edge.
fn ramp_left(
    dst: &mut Block<'_>,
    edge: SampleRef<'_>,
    from: usize,
    height: usize,
    cols: usize,
    dc: u32,
) {
    let weights = ramp(cols);
    for y in from..height {
        let left = edge.at(-(y as isize + 1));
        for (out, &weight) in dst.row(y, cols).iter_mut().zip(weights) {
            *out = blend(left, dc, weight);
        }
    }
}

/// `DC_PRED`: the mean of the top and left edges.
pub(super) fn dc(
    dst: &mut Block<'_>,
    edge: SampleRef<'_>,
    width: usize,
    height: usize,
    angle: i32,
) {
    let dc = mean_both(edge, width, height);
    if angle & ANGLE_IBP_FLAG == 0 {
        fill(dst, width, height, dc as u8);
        return;
    }
    let rows = height >> IBP_PART_LOG2;
    let cols = width >> IBP_PART_LOG2;
    // The corner quarter belongs to the longer side's ramp (the top's when
    // square), as AVM orders the two passes.
    let top_from = if width < height { cols } else { 0 };
    ramp_top(dst, edge, top_from, width, rows, dc);
    let left_from = if width >= height { rows } else { 0 };
    ramp_left(dst, edge, left_from, height, cols, dc);
    fill(
        &mut dst.sub(rows, cols),
        width - cols,
        height - rows,
        dc as u8,
    );
}

/// `TOP_DC_PRED`: the mean of the top edge.
pub(super) fn dc_top(
    dst: &mut Block<'_>,
    edge: SampleRef<'_>,
    width: usize,
    height: usize,
    angle: i32,
) {
    let dc = mean_top(edge, width);
    let rows = if angle & ANGLE_IBP_FLAG != 0 {
        height >> IBP_PART_LOG2
    } else {
        0
    };
    ramp_top(dst, edge, 0, width, rows, dc);
    fill(&mut dst.sub(rows, 0), width, height - rows, dc as u8);
}

/// `LEFT_DC_PRED`: the mean of the left edge.
pub(super) fn dc_left(
    dst: &mut Block<'_>,
    edge: SampleRef<'_>,
    width: usize,
    height: usize,
    angle: i32,
) {
    let dc = mean_left(edge, height);
    let cols = if angle & ANGLE_IBP_FLAG != 0 {
        width >> IBP_PART_LOG2
    } else {
        0
    };
    ramp_left(dst, edge, 0, height, cols, dc);
    fill(&mut dst.sub(0, cols), width - cols, height, dc as u8);
}

/// `DC_128_PRED`: mid-grey.
pub(super) fn dc_128(dst: &mut Block<'_>, width: usize, height: usize) {
    fill(dst, width, height, DC_128);
}
