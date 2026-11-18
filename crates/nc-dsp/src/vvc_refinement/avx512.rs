//! AVX-512 (Ice Lake / Zen 4) wrappers for shared VVC refinement lane arithmetic.
#![allow(unsafe_code)]
use nc_common::{BitDepth, PlaneView};

use super::{Kernels, bdof as bdof_lanes, generic};
use crate::simd::avx2::{I16x4, I16x8, I16x16, I32x4};
use crate::simd::avx512_i16::I16x32;

pub(super) const KERNELS: Kernels = Kernels {
    bilinear,
    dmvr,
    bdof,
    bdof_pad,
    prof,
};

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn bilinear(
    size: (usize, usize),
    input: &PlaneView<'_>,
    phases: (usize, usize),
    out: &mut [i16],
) {
    // SAFETY: this wrapper enables AVX-512F/BW/VL; public API checks buffers.
    unsafe {
        if size.0 >= 32 {
            generic::bilinear::<I16x32, I16x4>(size, input, phases, out);
        } else {
            generic::bilinear::<I16x16, I16x4>(size, input, phases, out);
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub(crate) unsafe fn bilinear_u16(
    size: (usize, usize),
    input: &PlaneView<'_, u16>,
    phases: (usize, usize),
    out: &mut [i16],
    depth: BitDepth,
) {
    // SAFETY: this wrapper enables AVX-512F/BW/VL; public API checks buffers.
    unsafe {
        if size.0 >= 32 {
            generic::bilinear_u16::<I16x32, I16x4>(size, input, phases, out, depth);
        } else {
            generic::bilinear_u16::<I16x16, I16x4>(size, input, phases, out, depth);
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn dmvr(size: (usize, usize), inputs: [&[i16]; 2]) -> ([i32; 2], bool) {
    // SAFETY: this wrapper enables AVX-512F/BW/VL; public API checks buffers.
    unsafe {
        if size.0 >= 16 {
            generic::dmvr::<I16x16>(size, inputs)
        } else {
            generic::dmvr::<I16x8>(size, inputs)
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn bdof(size: (usize, usize), inputs: [&[i16]; 2], out: &mut [u8]) {
    // SAFETY: this wrapper enables AVX-512F/BW/VL; public API checks buffers.
    unsafe {
        if size.0 >= 16 {
            bdof_lanes::bdof::<I32x4, I16x16, I16x4>(size, inputs, out);
        } else {
            bdof_lanes::bdof::<I32x4, I16x8, I16x4>(size, inputs, out);
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub(crate) unsafe fn bdof_u16(
    size: (usize, usize),
    inputs: [&[i16]; 2],
    out: &mut [u16],
    depth: BitDepth,
) {
    // SAFETY: this wrapper enables AVX-512F/BW/VL; public API checks buffers.
    unsafe {
        if size.0 >= 16 {
            bdof_lanes::bdof_at::<u16, I32x4, I16x16, I16x4>(size, inputs, out, depth);
        } else {
            bdof_lanes::bdof_at::<u16, I32x4, I16x8, I16x4>(size, inputs, out, depth);
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn bdof_pad(
    size: (usize, usize),
    window: &PlaneView<'_>,
    interior: &[i16],
    out: &mut [i16],
) {
    // SAFETY: this wrapper enables AVX-512F/BW/VL; public API checks buffers.
    unsafe {
        generic::bdof_pad::<I16x16, I16x8, I16x4>(size, window, interior, out);
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub(crate) unsafe fn bdof_pad_u16(
    size: (usize, usize),
    window: &PlaneView<'_, u16>,
    interior: &[i16],
    out: &mut [i16],
    depth: BitDepth,
) {
    // SAFETY: this wrapper enables AVX-512F/BW/VL; public API checks buffers.
    unsafe {
        generic::bdof_pad_u16::<I16x16, I16x8, I16x4>(size, window, interior, out, depth);
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn prof(samples: &[i16; 36], corrections: &[[i8; 2]; 16], out: &mut [i16; 16]) {
    // SAFETY: this wrapper enables AVX-512F/BW/VL; fixed arrays bound loads.
    unsafe {
        generic::prof::<I32x4>(samples, corrections, out);
    }
}
