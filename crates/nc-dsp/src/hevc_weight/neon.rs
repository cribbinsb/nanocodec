//! The [`super::lanes`] weighted prediction kernel compiled for NEON.

use super::{HevcWeights, WeightFn, lanes};
use crate::simd::neon::{I16x4, I16x8};

/// The NEON kernel.
pub(super) const WEIGHT: WeightFn = weight as WeightFn;

#[target_feature(enable = "neon")]
unsafe fn weight(
    weights: &HevcWeights,
    a: &[i16],
    b: &[i16],
    w: usize,
    h: usize,
    dst: &mut [u8],
    dst_stride: usize,
) {
    // SAFETY: the ISA is enabled here; the dispatcher checked the slices.
    unsafe { lanes::weight::<I16x8, I16x8, I16x4>(weights, a, b, w, h, dst, dst_stride) }
}
