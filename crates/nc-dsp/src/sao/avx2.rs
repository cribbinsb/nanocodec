//! AVX2 SAO edge-offset kernels: the generic kernels on I16x16.

use super::{Rect, SaoFns, SaoStats, generic};
use crate::simd::avx2::I16x16;

/// The AVX2 table.
///
/// # Safety
///
/// The CPU must support AVX2 (`Isa::Avx2.is_supported()`).
pub(crate) unsafe fn sao_fns() -> SaoFns {
    SaoFns { stats, apply, band }
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn stats(
    src: &[u8],
    src_stride: usize,
    rec: &[u8],
    rec_stride: usize,
    offs: (isize, isize),
    rect: Rect,
) -> SaoStats {
    // SAFETY: AVX2 is enabled for this function (dispatch contract).
    unsafe { generic::stats::<I16x16>(src, src_stride, rec, rec_stride, offs, rect) }
}

#[target_feature(enable = "avx2")]
unsafe fn apply(
    input: &[u8],
    in_stride: usize,
    out: &mut [u8],
    out_stride: usize,
    offs: (isize, isize),
    offsets: [i8; 4],
    rect: Rect,
) {
    // SAFETY: AVX2 is enabled for this function (dispatch contract).
    unsafe {
        super::bytes::edge::<super::avx2_bytes::Lanes, I16x16>(
            input, in_stride, out, out_stride, offs, offsets, rect,
        );
    }
}

#[target_feature(enable = "avx2")]
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
        super::bytes::band::<super::avx2_bytes::Lanes>(
            plane, stride, rect, position, offsets, exempt,
        );
    }
}
