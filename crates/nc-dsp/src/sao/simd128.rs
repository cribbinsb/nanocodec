//! SIMD128 SAO edge-offset kernels: the generic kernels on I16x8.

use super::{Rect, SaoFns, SaoStats, generic};
use crate::simd::simd128::I16x8;

/// The SIMD128 table.
///
/// The build enables `simd128`.
pub(crate) fn sao_fns() -> SaoFns {
    SaoFns { stats, apply, band }
}

#[target_feature(enable = "simd128")]
unsafe fn stats(
    src: &[u8],
    src_stride: usize,
    rec: &[u8],
    rec_stride: usize,
    offs: (isize, isize),
    rect: Rect,
) -> SaoStats {
    // SAFETY: SIMD128 is enabled for this function (dispatch contract).
    unsafe { generic::stats::<I16x8>(src, src_stride, rec, rec_stride, offs, rect) }
}

#[target_feature(enable = "simd128")]
unsafe fn apply(
    input: &[u8],
    in_stride: usize,
    out: &mut [u8],
    out_stride: usize,
    offs: (isize, isize),
    offsets: [i8; 4],
    rect: Rect,
) {
    // SAFETY: SIMD128 is enabled for this function (dispatch contract).
    unsafe {
        super::bytes::edge::<super::simd128_bytes::Lanes, I16x8>(
            input, in_stride, out, out_stride, offs, offsets, rect,
        );
    }
}

#[target_feature(enable = "simd128")]
unsafe fn band(
    plane: &mut [u8],
    stride: usize,
    rect: Rect,
    position: u8,
    offsets: [i8; 4],
    exempt: &[bool],
) {
    // SAFETY: dispatch checked the rectangle and exemption mask; this enables the ISA.
    unsafe {
        super::bytes::band::<super::simd128_bytes::Lanes>(
            plane, stride, rect, position, offsets, exempt,
        );
    }
}

/// Shared HEVC/VVC u16 edge-offset application at the stream's depth.
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn apply_high(
    input: &[u16],
    in_stride: usize,
    out: &mut [u16],
    out_stride: usize,
    offs: (isize, isize),
    offsets: [i16; 4],
    rect: Rect,
    depth: nc_common::BitDepth,
) {
    // SAFETY: the depth table checked views/halo and enabled SIMD128.
    unsafe {
        generic::apply_at::<I16x8, u16>(
            input, in_stride, out, out_stride, offs, offsets, rect, depth,
        );
    }
}
