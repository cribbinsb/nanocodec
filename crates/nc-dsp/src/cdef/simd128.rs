//! WebAssembly SIMD128 CDEF kernels: the generic filter on 8 lanes, one row of an 8x8
//! luma block or two rows of a 4x4 chroma block per vector.

use super::{CdefArgs, CdefFns, direction, generic};
use crate::simd::simd128::I16x8;

/// The SIMD128 table (the build enables `simd128`).
pub(crate) fn cdef_fns() -> CdefFns {
    CdefFns {
        find_dir,
        block8,
        block4,
        block8_interior: super::simd128_interior::block::<8>,
        block4_interior: super::simd128_interior::block::<4>,
        block4_interior_pair: Some(super::simd128_interior::block4_pair),
    }
}

#[target_feature(enable = "simd128")]
unsafe fn find_dir(src: &[u8], stride: usize) -> (u8, i32) {
    // SAFETY: SIMD128 is enabled for this function (dispatch contract).
    unsafe { direction::find_dir::<I16x8>(src, stride) }
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
