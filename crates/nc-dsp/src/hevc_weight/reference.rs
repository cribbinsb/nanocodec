//! The scalar weighted prediction, the oracle of the vector kernels.
//! Generic over the sample type and depth (one source for the 8-bit table
//! and the high-bit-depth one); only the final clip depends on the depth,
//! the 14-bit inputs and the weights are the same at every depth (the codec
//! folds `shift1 = 14 − BitDepth` into `HevcWeights::shift`).

use nc_common::{BitDepth, Sample};

use super::HevcWeights;

/// One output sample.
#[allow(clippy::inline_always, reason = "inlined into the ISA instances")]
#[inline(always)]
pub(crate) fn sample<T: Sample>(weights: &HevcWeights, a: i16, b: i16, depth: BitDepth) -> T {
    // Folds to the constant 8 in the 8-bit instance.
    let depth = depth.of::<T>();
    let v = i32::from(a) * i32::from(weights.w0) + i32::from(b) * i32::from(weights.w1);
    depth.clip_to((v + weights.round) >> weights.shift)
}

/// Weighted prediction of one row segment.
#[allow(clippy::inline_always, reason = "inlined into the ISA instances")]
#[inline(always)]
pub(crate) fn row<T: Sample>(
    weights: &HevcWeights,
    a: &[i16],
    b: &[i16],
    dst: &mut [T],
    depth: BitDepth,
) {
    // Folds to the constant 8 in the 8-bit instance.
    let depth = depth.of::<T>();
    for ((d, &x), &y) in dst.iter_mut().zip(a).zip(b) {
        *d = sample(weights, x, y, depth);
    }
}

/// The scalar kernel at any depth ([`super::WeightFn`] with the depth);
/// always inlined, so a vector ISA's instance (`crate::high_depth`)
/// compiles it with its target features.
#[allow(clippy::too_many_arguments, clippy::inline_always)]
#[inline(always)]
pub(crate) fn weight_at<T: Sample>(
    weights: &HevcWeights,
    a: &[i16],
    b: &[i16],
    w: usize,
    h: usize,
    dst: &mut [T],
    dst_stride: usize,
    depth: BitDepth,
) {
    // Folds to the constant 8 in the 8-bit instance.
    let depth = depth.of::<T>();
    for y in 0..h {
        row(
            weights,
            &a[y * w..(y + 1) * w],
            &b[y * w..(y + 1) * w],
            &mut dst[y * dst_stride..y * dst_stride + w],
            depth,
        );
    }
}

/// The scalar 8-bit kernel ([`super::WeightFn`]).
pub(crate) fn weight(
    weights: &HevcWeights,
    a: &[i16],
    b: &[i16],
    w: usize,
    h: usize,
    dst: &mut [u8],
    dst_stride: usize,
) {
    weight_at(weights, a, b, w, h, dst, dst_stride, BitDepth::EIGHT);
}
