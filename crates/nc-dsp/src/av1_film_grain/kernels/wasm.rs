//! simd128 instantiations of shared AV1 grain row arithmetic.
#![allow(unsafe_code)]
use super::{Av1GrainScaling, Kernels, generic};
use crate::simd::simd128::I16x8;
pub(super) const KERNELS: Kernels = Kernels {
    luma,
    chroma,
    blend,
};
#[target_feature(enable = "simd128")]
unsafe fn luma(
    source: &[u8],
    grain: &[i16],
    scaling: &Av1GrainScaling,
    bounds: [u8; 2],
    out: &mut [u8],
) {
    // SAFETY: this wrapper enables the lane ISA.
    unsafe { generic::luma::<I16x8>(source, grain, scaling, bounds, out) }
}
#[target_feature(enable = "simd128")]
unsafe fn chroma(
    source: &[u8],
    luma: &[u8],
    grain: &[i16],
    scaling: &Av1GrainScaling,
    bounds: [u8; 2],
    coefficients: [i16; 3],
    out: &mut [u8],
) {
    // SAFETY: this wrapper enables the lane ISA.
    unsafe {
        generic::chroma::<I16x8>(source, luma, grain, scaling, bounds, coefficients, out);
    }
}
#[target_feature(enable = "simd128")]
unsafe fn blend(old: &[i16], new: &mut [i16], weights: [i16; 2]) {
    // SAFETY: this wrapper enables the lane ISA.
    unsafe { generic::blend::<I16x8>(old, new, weights) }
}
