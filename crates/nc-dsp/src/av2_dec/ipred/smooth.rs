//! AV2 smooth prediction: each sample blends a linear ramp towards the
//! bottom-left (vertical) or top-right (horizontal) sample with the nearest
//! edge sample, by a weight that falls off with distance faster in smaller
//! blocks (`SMOOTH_WEIGHTS`). The two-way predictor averages both.
//!
//! Ported from dav2d `src/ipred_tmpl.c` (`ipred_smooth*_c`; commit 237229e,
//! BSD-2-Clause, see `docs/licenses/dav2d`). Not shared with
//! [`crate::intra_prediction::av2`]'s smooth, which serves our encoder's
//! square blocks through its SIMD kernel table.

use nc_common::Sample;

use super::tables::SMOOTH_WEIGHTS;
use super::{Block, SampleRef};

/// Bits of the blend weights (they blend out of 64).
const WEIGHT_BITS: u32 = 6;
/// Blocks of this many samples or more use the slower fall-off.
const MEDIUM_BLOCK: usize = 64;
/// Blocks of more than this many samples use the slowest fall-off.
const LARGE_BLOCK: usize = 512;

/// The weights of a `width` x `height` block, indexed by distance.
fn weights(width: usize, height: usize) -> &'static [u8; 64] {
    let samples = width * height;
    let scale = usize::from(samples >= MEDIUM_BLOCK) + usize::from(samples > LARGE_BLOCK);
    &SMOOTH_WEIGHTS[scale]
}

/// One direction of the smooth prediction: the ramp from `far` (the sample
/// past the block's far side) to `near`, `offset` of `len` (a power of two)
/// steps from the far side, pulled towards `near` by `weight`.
fn smooth_1d(near: i32, far: i32, offset: usize, len: usize, weight: u8) -> i32 {
    let ramp = far + (((near - far) * offset as i32 + (len >> 1) as i32) >> len.trailing_zeros());
    ramp + (((near - ramp) * i32::from(weight) + (1 << (WEIGHT_BITS - 1))) >> WEIGHT_BITS)
}

/// `SMOOTH_PRED`: the mean of the vertical and horizontal predictions.
pub(super) fn smooth<T: Sample>(
    dst: &mut Block<'_, T>,
    edge: SampleRef<'_, T>,
    width: usize,
    height: usize,
) {
    let weights = weights(width, height);
    let right = edge.wide(width as isize + 1);
    let bottom = edge.wide(-(height as isize + 1));
    let top = edge.slice(1, width);
    for y in 0..height {
        let left = edge.wide(-(y as isize + 1));
        for (x, out) in dst.row(y, width).iter_mut().enumerate() {
            let above = top[x].to_i32();
            let ver = smooth_1d(above, bottom, height - 1 - y, height, weights[y]);
            let hor = smooth_1d(left, right, width - 1 - x, width, weights[x]);
            *out = T::from_i32((ver + hor + 1) >> 1);
        }
    }
}

/// `SMOOTH_V_PRED`: the ramp from the top edge to the bottom-left sample.
pub(super) fn smooth_v<T: Sample>(
    dst: &mut Block<'_, T>,
    edge: SampleRef<'_, T>,
    width: usize,
    height: usize,
) {
    let weights = weights(width, height);
    let bottom = edge.wide(-(height as isize + 1));
    let top = edge.slice(1, width);
    for (y, &weight) in weights.iter().enumerate().take(height) {
        for (out, &above) in dst.row(y, width).iter_mut().zip(top) {
            *out = T::from_i32(smooth_1d(
                above.to_i32(),
                bottom,
                height - 1 - y,
                height,
                weight,
            ));
        }
    }
}

/// `SMOOTH_H_PRED`: the ramp from the left edge to the top-right sample.
pub(super) fn smooth_h<T: Sample>(
    dst: &mut Block<'_, T>,
    edge: SampleRef<'_, T>,
    width: usize,
    height: usize,
) {
    let weights = weights(width, height);
    let right = edge.wide(width as isize + 1);
    for y in 0..height {
        let left = edge.wide(-(y as isize + 1));
        for (x, out) in dst.row(y, width).iter_mut().enumerate() {
            *out = T::from_i32(smooth_1d(left, right, width - 1 - x, width, weights[x]));
        }
    }
}
