//! NEON CDEF kernels: the generic 16-bit filter on 8 lanes (one row of an 8x8
//! luma block or two rows of a 4x4 chroma block per vector.

use super::{CdefArgs, CdefFns, direction, generic, narrow};
use crate::simd::neon::I16x8;
use crate::simd::neon_u8::U8x16;

/// The NEON table (NEON is always present on `AArch64`).
pub(crate) fn cdef_fns() -> CdefFns {
    CdefFns {
        find_dir,
        block8,
        block4,
        block8_interior,
        block4_interior,
        block4_interior_pair: None,
    }
}

#[target_feature(enable = "neon")]
unsafe fn find_dir(src: &[u8], stride: usize) -> (u8, i32) {
    // SAFETY: NEON is enabled for this function (dispatch contract).
    unsafe { direction::find_dir::<I16x8>(src, stride) }
}

#[target_feature(enable = "neon")]
unsafe fn block8(
    src: &[i16],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u8],
    dst_stride: usize,
) {
    // SAFETY: NEON is enabled for this function (dispatch contract).
    unsafe { generic::filter_any_dir::<I16x8, 1, 12>(src, stride, origin, args, dst, dst_stride) }
}

#[target_feature(enable = "neon")]
unsafe fn block4(
    src: &[i16],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u8],
    dst_stride: usize,
) {
    // SAFETY: NEON is enabled for this function (dispatch contract).
    unsafe { generic::filter_any_dir::<I16x8, 2, 8>(src, stride, origin, args, dst, dst_stride) }
}

#[target_feature(enable = "neon")]
unsafe fn block8_interior(
    src: &[u8],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u8],
    dst_stride: usize,
) {
    // SAFETY: NEON is enabled for this function (dispatch contract).
    unsafe { narrow::filter_any::<U8x16, 8, 12>(src, stride, origin, args, dst, dst_stride) }
}

#[target_feature(enable = "neon")]
unsafe fn block4_interior(
    src: &[u8],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u8],
    dst_stride: usize,
) {
    // SAFETY: NEON is enabled for this function (dispatch contract).
    unsafe { narrow::filter_any::<U8x16, 4, 8>(src, stride, origin, args, dst, dst_stride) }
}
