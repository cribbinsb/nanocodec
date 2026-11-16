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
        block4_interior_pair: Some(block4_pair),
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

/// Two interior 4x4 blocks ([`super::CdefPairFn8`] contract), one after the
/// other in one call, as SIMD128's `block4_pair`: a vector holds one 4x4
/// block, so the lanes share nothing, but the call and the strength set-up
/// are paid once per unit's U and V instead of twice.
#[target_feature(enable = "neon")]
unsafe fn block4_pair(
    src: [&[u8]; 2],
    stride: usize,
    origins: [usize; 2],
    args: &CdefArgs,
    dirs: [usize; 2],
    dst: [*mut u8; 2],
    dst_stride: usize,
) {
    const N: usize = 4;
    for i in 0..2 {
        // SAFETY: the caller guarantees four writable rows of four samples
        // at `dst[i]` that nothing else references (`CdefPairFn8`).
        let out = unsafe { core::slice::from_raw_parts_mut(dst[i], (N - 1) * dst_stride + N) };
        let args = args.with_dir(dirs[i] as u8);
        // SAFETY: NEON is enabled for this function; the kernel checks the
        // block's source window against `src[i]`.
        unsafe {
            narrow::filter_any::<U8x16, N, 8>(src[i], stride, origins[i], &args, out, dst_stride);
        }
    }
}
