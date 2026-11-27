//! AVX2 SAO edge-offset kernels: the generic kernels on I16x16.

use super::{Rect, SaoFns, SaoStats, generic};
use crate::simd::avx2::{I16x8, I16x16};

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
    use super::avx2_bytes::Lanes;
    let width = rect.x1 - rect.x0;
    let rest = width % 32;
    // SAFETY: the dispatcher checked both source neighbours and all stores.
    // Splitting into disjoint column ranges retains the checked extents.
    unsafe {
        if rest >= 16 || width < 32 {
            let middle = rect.x1 - rest;
            if middle != rect.x0 {
                super::bytes::edge::<Lanes<32>, I16x16>(
                    input,
                    in_stride,
                    out,
                    out_stride,
                    offs,
                    offsets,
                    Rect { x1: middle, ..rect },
                );
            }
            super::bytes::edge::<Lanes<16>, I16x8>(
                input,
                in_stride,
                out,
                out_stride,
                offs,
                offsets,
                Rect { x0: middle, ..rect },
            );
        } else {
            super::bytes::edge::<Lanes<32>, I16x8>(
                input, in_stride, out, out_stride, offs, offsets, rect,
            );
        }
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
    let width = rect.x1 - rect.x0;
    // SAFETY: dispatch checked the rectangle and exemption mask.
    unsafe {
        if width < 32 || width % 32 >= 16 {
            super::bytes::band::<super::avx2_bytes::Lanes<16>>(
                plane, stride, rect, position, offsets, exempt,
            );
        } else {
            super::bytes::band::<super::avx2_bytes::Lanes<32>>(
                plane, stride, rect, position, offsets, exempt,
            );
        }
    }
}
