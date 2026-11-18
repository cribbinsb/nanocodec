//! SIMD128 wrappers for shared VVC refinement lane arithmetic (`u8` and `u16`).
#![allow(unsafe_code)]
use nc_common::{BitDepth, PlaneView};

use super::{Kernels, bdof as bdof_lanes, generic};
use crate::simd::simd128::{I16x4, I16x8, I32x4};

pub(super) const KERNELS: Kernels = Kernels {
    bilinear,
    dmvr,
    bdof,
    prof,
};

#[target_feature(enable = "simd128")]
unsafe fn bilinear(
    size: (usize, usize),
    input: &PlaneView<'_>,
    phases: (usize, usize),
    out: &mut [i16],
) {
    // SAFETY: this wrapper enables the lane ISA; public API checks buffers.
    unsafe {
        generic::bilinear::<I16x8, I16x4>(size, input, phases, out);
    }
}

#[target_feature(enable = "simd128")]
pub(crate) unsafe fn bilinear_u16(
    size: (usize, usize),
    input: &PlaneView<'_, u16>,
    phases: (usize, usize),
    out: &mut [i16],
    depth: BitDepth,
) {
    // SAFETY: this wrapper enables the lane ISA; public API checks buffers.
    unsafe {
        generic::bilinear_u16::<I16x8, I16x4>(size, input, phases, out, depth);
    }
}

#[target_feature(enable = "simd128")]
unsafe fn dmvr(size: (usize, usize), inputs: [&[i16]; 2]) -> ([i32; 2], bool) {
    // SAFETY: this wrapper enables the lane ISA; public API checks buffers.
    unsafe { generic::dmvr::<I16x8>(size, inputs) }
}

#[target_feature(enable = "simd128")]
unsafe fn bdof(size: (usize, usize), inputs: [&[i16]; 2], out: &mut [u8]) {
    // SAFETY: this wrapper enables SIMD128; public API checks buffers.
    unsafe {
        bdof_lanes::bdof::<I32x4, I16x8, I16x4>(size, inputs, out);
    }
}

#[target_feature(enable = "simd128")]
pub(crate) unsafe fn bdof_u16(
    size: (usize, usize),
    inputs: [&[i16]; 2],
    out: &mut [u16],
    depth: BitDepth,
) {
    // SAFETY: this wrapper enables SIMD128; public API checks buffers.
    unsafe {
        bdof_lanes::bdof_at::<u16, I32x4, I16x8, I16x4>(size, inputs, out, depth);
    }
}

#[target_feature(enable = "simd128")]
unsafe fn prof(samples: &[i16; 36], corrections: &[[i8; 2]; 16], out: &mut [i16; 16]) {
    // SAFETY: this wrapper enables the four-lane ISA; fixed arrays bound loads.
    unsafe {
        generic::prof::<I32x4>(samples, corrections, out);
    }
}
