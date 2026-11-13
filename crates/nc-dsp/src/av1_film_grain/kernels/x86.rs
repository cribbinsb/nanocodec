//! avx2 instantiations of shared AV1 grain row arithmetic.
#![allow(unsafe_code)]
use super::{Kernels, generic};
use crate::simd::avx2::{I16x16, I32x8};
pub(super) const KERNELS: Kernels = Kernels {
    luma,
    chroma,
    blend,
};
#[target_feature(enable = "avx2")]
unsafe fn luma(
    source: &[u8],
    grain: &[i16],
    scaling: &[i32; 256],
    shift: u8,
    bounds: [u8; 2],
    out: &mut [u8],
) {
    // SAFETY: this wrapper enables the lane ISA.
    unsafe { generic::luma::<I16x16>(source, grain, scaling, shift, bounds, out) }
}
#[target_feature(enable = "avx2")]
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
        generic::chroma::<I16x16, I32x8>(
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
#[target_feature(enable = "avx2")]
unsafe fn blend(old: &[i16], new: &mut [i16], weights: [i16; 2]) {
    // SAFETY: this wrapper enables the lane ISA.
    unsafe { generic::blend::<I16x16>(old, new, weights) }
}
