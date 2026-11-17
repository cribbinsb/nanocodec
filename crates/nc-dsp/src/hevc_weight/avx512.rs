//! The [`super::lanes`] weighted prediction kernel compiled for AVX-512 (`u8` and `u16`).

use nc_common::BitDepth;

use super::{HevcWeights, WeightFn, lanes};
use crate::high_depth::Weight16Fn;
use crate::simd::avx2::{I16x4, I16x8};
use crate::simd::avx512_i16::I16x32;

/// The AVX-512 8-bit kernel.
pub(super) const WEIGHT: WeightFn = weight as WeightFn;
/// The AVX-512 `u16` kernel.
pub(super) const WEIGHT_U16: Weight16Fn = weight_u16 as Weight16Fn;

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
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
    unsafe { lanes::weight::<I16x32, I16x8, I16x4>(weights, a, b, w, h, dst, dst_stride) }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn weight_u16(
    weights: &HevcWeights,
    a: &[i16],
    b: &[i16],
    w: usize,
    h: usize,
    dst: &mut [u16],
    dst_stride: usize,
    depth: BitDepth,
) {
    // SAFETY: the ISA is enabled here; the dispatcher checked the slices.
    unsafe {
        lanes::weight_u16::<I16x32, I16x8, I16x4>(weights, a, b, w, h, dst, dst_stride, depth);
    }
}
