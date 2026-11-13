//! The scalar weighted prediction, the oracle of the vector kernels.

use super::HevcWeights;

/// One output sample.
#[inline]
pub(crate) fn sample(weights: &HevcWeights, a: i16, b: i16) -> u8 {
    let v = i32::from(a) * i32::from(weights.w0) + i32::from(b) * i32::from(weights.w1);
    ((v + weights.round) >> weights.shift).clamp(0, 255) as u8
}

/// Weighted prediction of one row segment.
#[inline]
pub(crate) fn row(weights: &HevcWeights, a: &[i16], b: &[i16], dst: &mut [u8]) {
    for ((d, &x), &y) in dst.iter_mut().zip(a).zip(b) {
        *d = sample(weights, x, y);
    }
}

/// The scalar kernel ([`super::WeightFn`]).
pub(crate) fn weight(
    weights: &HevcWeights,
    a: &[i16],
    b: &[i16],
    w: usize,
    h: usize,
    dst: &mut [u8],
    dst_stride: usize,
) {
    for y in 0..h {
        row(
            weights,
            &a[y * w..(y + 1) * w],
            &b[y * w..(y + 1) * w],
            &mut dst[y * dst_stride..y * dst_stride + w],
        );
    }
}
