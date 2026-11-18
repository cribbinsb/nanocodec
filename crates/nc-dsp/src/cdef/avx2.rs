//! AVX2 CDEF kernels: the generic filter with two block rows per vector
//! (16 lanes for 8x8 luma blocks, 8 for 4x4 chroma at 8 bpc; 16 lanes with
//! four 4-sample rows for 4x4 chroma at 10/12 bpc); 8-bit interior blocks use
//! the packed-byte kernels in `avx2_interior`.

use super::{CdefArgs, CdefFns, HighCdefFns, generic};
use crate::simd::avx2::{I16x8, I16x16};

/// The AVX2 8-bit table.
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

/// The AVX2 high-bit-depth (`u16`) table.
pub(crate) const HIGH_FNS: HighCdefFns = HighCdefFns {
    find_dir: super::direction_avx2::find_dir_u16,
    edge: [block4_edge_u16, block8_edge_u16],
    interior: [block4_interior_u16, block8_interior_u16],
    interior_pair4: None,
};

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

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn block8_edge_u16(
    src: &[i16],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u16],
    dst_stride: usize,
) {
    // SAFETY: AVX2 is enabled for this function (dispatch contract).
    unsafe {
        generic::filter_any_dir_u16::<I16x16, 2, 12>(src, stride, origin, args, dst, dst_stride);
    }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn block4_edge_u16(
    src: &[i16],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u16],
    dst_stride: usize,
) {
    // SAFETY: AVX2 is enabled for this function; 4 rows of 4 words fit `I16x16`
    // in a single pass.
    unsafe {
        generic::filter_any_dir_u16::<I16x16, 4, 8>(src, stride, origin, args, dst, dst_stride);
    }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn block8_interior_u16(
    src: &[u16],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u16],
    dst_stride: usize,
) {
    // SAFETY: AVX2 is enabled; valid 10/12-bit `u16` samples (`0..=4095`) have
    // identical bit patterns as positive `i16` values.
    unsafe {
        let src_i16 = core::slice::from_raw_parts(src.as_ptr().cast::<i16>(), src.len());
        block8_edge_u16(src_i16, stride, origin, args, dst, dst_stride);
    }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn block4_interior_u16(
    src: &[u16],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u16],
    dst_stride: usize,
) {
    // SAFETY: AVX2 is enabled; valid 10/12-bit `u16` samples (`0..=4095`) fit
    // positive `i16`.
    unsafe {
        let src_i16 = core::slice::from_raw_parts(src.as_ptr().cast::<i16>(), src.len());
        block4_edge_u16(src_i16, stride, origin, args, dst, dst_stride);
    }
}
