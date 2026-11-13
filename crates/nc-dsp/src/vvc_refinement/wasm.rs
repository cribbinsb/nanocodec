//! simd128 wrappers for shared VVC refinement lane arithmetic.
#![allow(unsafe_code)]
use super::{Kernels, generic};
use crate::simd::simd128::{I16x8, I32x4};
pub(super) const KERNELS: Kernels = Kernels {
    bilinear,
    dmvr,
    bdof,
    prof,
};
#[target_feature(enable = "simd128")]
unsafe fn bilinear(
    size: (usize, usize),
    input: &nc_common::PlaneView<'_>,
    phases: (usize, usize),
    out: &mut [i16],
) {
    // SAFETY: this wrapper enables the lane ISA; public API checks buffers.
    unsafe {
        generic::bilinear::<I16x8>(size, input, phases, out);
    }
}
#[target_feature(enable = "simd128")]
unsafe fn dmvr(size: (usize, usize), inputs: [&[i16]; 2]) -> ([i32; 2], bool) {
    // SAFETY: this wrapper enables the lane ISA; public API checks buffers.
    unsafe { generic::dmvr::<I32x4>(size, inputs) }
}
#[target_feature(enable = "simd128")]
unsafe fn bdof(size: (usize, usize), inputs: [&[i16]; 2], out: &mut [u8]) {
    // SAFETY: this wrapper enables the four-lane ISA; public API checks buffers.
    unsafe {
        super::bdof::bdof::<I32x4>(size, inputs, out);
    }
}
#[target_feature(enable = "simd128")]
unsafe fn prof(samples: &[i16; 36], corrections: &[[i32; 2]; 16], out: &mut [i16; 16]) {
    // SAFETY: this wrapper enables the four-lane ISA; fixed arrays bound loads.
    unsafe {
        generic::prof::<I32x4>(samples, corrections, out);
    }
}
