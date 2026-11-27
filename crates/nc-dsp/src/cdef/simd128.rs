//! WebAssembly SIMD128 CDEF kernels: the generic filter on 8 lanes, one row of an 8x8
//! luma block or two rows of a 4x4 chroma block per vector.

use super::{CdefArgs, CdefFns, HighCdefFns, direction, generic};
use crate::simd::simd128::I16x8;

/// The SIMD128 table (the build enables `simd128`).
pub(crate) fn cdef_fns() -> CdefFns {
    CdefFns {
        find_dir,
        block8,
        block4,
        block8_interior: super::simd128_interior::block::<8>,
        block4_interior: super::simd128_interior::block::<4>,
        border: None,
        block4_interior_pair: Some(super::simd128_interior::block4_pair),
    }
}

/// The SIMD128 high-bit-depth (`u16`) table.
pub(crate) const HIGH_FNS: HighCdefFns = HighCdefFns {
    find_dir: find_dir_u16,
    edge: [block4_edge_u16, block8_edge_u16],
    border4: None,
    interior: [block4_interior_u16, block8_interior_u16],
    interior_pair4: Some(block4_pair_u16),
};

#[target_feature(enable = "simd128")]
unsafe fn find_dir(src: &[u8], stride: usize) -> (u8, i32) {
    // SAFETY: SIMD128 is enabled for this function (dispatch contract).
    unsafe { direction::find_dir::<I16x8>(src, stride) }
}

#[target_feature(enable = "simd128")]
unsafe fn find_dir_u16(src: &[u16], stride: usize, bits_above: u32) -> (u8, i32) {
    // SAFETY: SIMD128 is enabled for this function (dispatch contract).
    unsafe { direction::find_dir_u16::<I16x8>(src, stride, bits_above) }
}

#[target_feature(enable = "simd128")]
unsafe fn block8(
    src: &[i16],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u8],
    dst_stride: usize,
) {
    // SAFETY: SIMD128 is enabled for this function (dispatch contract).
    unsafe { generic::filter_any_dir::<I16x8, 1, 12>(src, stride, origin, args, dst, dst_stride) }
}

#[target_feature(enable = "simd128")]
unsafe fn block4(
    src: &[i16],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u8],
    dst_stride: usize,
) {
    // SAFETY: SIMD128 is enabled for this function (dispatch contract).
    unsafe { generic::filter_any_dir::<I16x8, 2, 8>(src, stride, origin, args, dst, dst_stride) }
}

#[target_feature(enable = "simd128")]
unsafe fn block8_edge_u16(
    src: &[i16],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u16],
    dst_stride: usize,
) {
    // SAFETY: SIMD128 is enabled for this function (dispatch contract).
    unsafe {
        generic::filter_any_dir_u16::<I16x8, 1, 12>(src, stride, origin, args, dst, dst_stride);
    }
}

#[target_feature(enable = "simd128")]
unsafe fn block4_edge_u16(
    src: &[i16],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u16],
    dst_stride: usize,
) {
    // SAFETY: SIMD128 is enabled for this function (dispatch contract).
    unsafe {
        generic::filter_any_dir_u16::<I16x8, 2, 8>(src, stride, origin, args, dst, dst_stride);
    }
}

#[target_feature(enable = "simd128")]
unsafe fn block8_interior_u16(
    src: &[u16],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u16],
    dst_stride: usize,
) {
    // SAFETY: SIMD128 is enabled; valid 10/12-bit `u16` samples (`0..=4095`) fit
    // positive `i16`.
    unsafe {
        let src_i16 = core::slice::from_raw_parts(src.as_ptr().cast::<i16>(), src.len());
        generic::filter_any_dir_u16::<I16x8, 1, 12>(src_i16, stride, origin, args, dst, dst_stride);
    }
}

#[target_feature(enable = "simd128")]
unsafe fn block4_interior_u16(
    src: &[u16],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u16],
    dst_stride: usize,
) {
    // SAFETY: SIMD128 is enabled; valid 10/12-bit `u16` samples (`0..=4095`) fit
    // positive `i16`.
    unsafe {
        let src_i16 = core::slice::from_raw_parts(src.as_ptr().cast::<i16>(), src.len());
        generic::filter_any_dir_u16::<I16x8, 2, 8>(src_i16, stride, origin, args, dst, dst_stride);
    }
}

#[target_feature(enable = "simd128")]
unsafe fn block4_pair_u16(
    src: [&[u16]; 2],
    stride: usize,
    origins: [usize; 2],
    args: &CdefArgs,
    dst: [*mut u16; 2],
    dst_stride: usize,
) {
    // SAFETY: SIMD128 is enabled; `filter_pair4_any_dir_u16` validates both windows.
    unsafe {
        generic::filter_pair4_any_dir_u16::<I16x8, 2>(src, stride, origins, args, dst, dst_stride);
    }
}
