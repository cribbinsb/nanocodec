//! neon instantiations of shared AV1 grain row arithmetic.
#![allow(unsafe_code)]
use super::{Kernels, generic};
use crate::simd::neon::{I16x8, I32x4};
pub(super) const KERNELS: Kernels = Kernels {
    luma,
    chroma,
    blend,
};
#[target_feature(enable = "neon")]
unsafe fn luma(
    source: &[u8],
    grain: &[i16],
    scaling: &[i32; 256],
    shift: u8,
    bounds: [u8; 2],
    out: &mut [u8],
) {
    // SAFETY: this wrapper enables the lane ISA.
    unsafe { generic::luma::<I16x8>(source, grain, scaling, shift, bounds, out) }
}
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn chroma(
    source: &[u8],
    luma: &[u8],
    grain: &[i16],
    scaling: &[i32; 256],
    shift: u8,
    bounds: [u8; 2],
    coefficients: [i16; 3],
    out: &mut [u8],
) {
    // SAFETY: this wrapper enables the lane ISA.
    unsafe {
        generic::chroma::<I16x8, I32x4>(
            source,
            luma,
            grain,
            scaling,
            shift,
            bounds,
            coefficients,
            out,
        );
    }
}
#[target_feature(enable = "neon")]
unsafe fn blend(old: &[i16], new: &mut [i16], weights: [i16; 2]) {
    // SAFETY: this wrapper enables the lane ISA.
    unsafe { generic::blend::<I16x8>(old, new, weights) }
}
