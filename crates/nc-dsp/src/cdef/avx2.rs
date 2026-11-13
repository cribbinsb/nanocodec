//! AVX2 CDEF kernels: the generic filter with two block rows per vector
//! (16 lanes for 8x8 luma blocks, 8 for 4x4 chroma); interior blocks use
//! the packed-byte kernels in `avx2_interior`.

use super::{CdefArgs, CdefFns, generic};
use crate::simd::avx2::{I16x8, I16x16};

/// The AVX2 table.
///
/// # Safety
///
/// The CPU must support AVX2 (`Isa::Avx2.is_supported()`).
pub(crate) unsafe fn cdef_fns() -> CdefFns {
    CdefFns {
        find_dir: super::direction_avx2::find_dir,
        block8,
        block4,
        block8_interior: super::avx2_interior::block::<8>,
        block4_interior: super::avx2_interior::block::<4>,
        block4_interior_pair: Some(super::avx2_interior::block4_pair),
    }
}

#[target_feature(enable = "avx2")]
unsafe fn block8(
    src: &[i16],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u8],
    dst_stride: usize,
) {
    // SAFETY: AVX2 is enabled for this function (dispatch contract).
    unsafe { generic::filter_any_dir::<I16x16, 2, 12>(src, stride, origin, args, dst, dst_stride) }
}

#[target_feature(enable = "avx2")]
unsafe fn block4(
    src: &[i16],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u8],
    dst_stride: usize,
) {
    // SAFETY: AVX2 is enabled for this function (dispatch contract).
    unsafe { generic::filter_any_dir::<I16x8, 2, 8>(src, stride, origin, args, dst, dst_stride) }
}
