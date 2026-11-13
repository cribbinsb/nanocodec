//! NEON deblocking kernels: H.264 on sixteen byte lanes (`h264_bytes`; single-plane
//! chroma on I16x8); HEVC on I16x8.

use super::{DeblockFns, Edge, HevcEdge, generic, hevc};
use crate::simd::neon::I16x8;

/// The NEON table.
///
/// NEON is always present on `AArch64`.
pub(crate) fn deblock_fns() -> DeblockFns {
    DeblockFns {
        luma,
        chroma,
        chroma_pair,
        hevc_luma,
        hevc_chroma,
        av2_ramps,
        av1_edge,
        av1_line: super::av1_line::neon,
        av1_grid: super::av1_grid::neon,
        av1_runs: super::runs::neon::av1,
        hevc_luma_runs: super::runs::neon::luma,
        hevc_chroma_runs: super::runs::neon::chroma,
        av2_chunk,
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn av1_edge(
    plane: &mut [u8],
    q0: usize,
    across: usize,
    along: usize,
    lines: usize,
    filter: super::av1::Av1EdgeFilter,
) {
    // SAFETY: the ISA is enabled for this function (dispatch contract).
    unsafe {
        super::av1_packed::filter_edge::<crate::lpf_bytes::neon::B>(
            plane, q0, across, along, lines, filter,
        );
    }
}

#[target_feature(enable = "neon")]
unsafe fn luma(plane: &mut [u8], edge: &Edge) {
    // SAFETY: NEON is enabled for this function (dispatch contract).
    unsafe { super::h264_bytes::luma::<super::h264_bytes::neon::B>(plane, edge) }
}

#[target_feature(enable = "neon")]
unsafe fn chroma(plane: &mut [u8], edge: &Edge) {
    // SAFETY: NEON is enabled for this function (dispatch contract).
    unsafe { generic::chroma::<I16x8, I16x8>(plane, edge) }
}

#[target_feature(enable = "neon")]
unsafe fn chroma_pair(u: &mut [u8], v: &mut [u8], edge: &Edge) {
    // SAFETY: NEON is enabled for this function (dispatch contract).
    unsafe { super::h264_bytes::chroma_pair::<super::h264_bytes::neon::B>(u, v, edge) }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn hevc_luma(plane: &mut [u8], edge: &HevcEdge) {
    // SAFETY: NEON is enabled for this function (dispatch contract).
    unsafe { hevc::luma::<I16x8>(plane, edge) }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn hevc_chroma(plane: &mut [u8], edge: &HevcEdge) {
    // SAFETY: NEON is enabled for this function (dispatch contract).
    unsafe { super::hevc_chroma::filter::<crate::lpf_bytes::neon::B>(plane, edge) }
}

#[target_feature(enable = "neon")]
unsafe fn av2_ramps(plane: &mut [u8], ramps: &super::av2::Av2Ramps) {
    use crate::simd::neon::I32x4;
    // SAFETY: NEON is enabled for this function (dispatch contract).
    unsafe {
        if ramps.along == 1 {
            super::av2_ramp::horizontal::<I32x4>(plane, ramps);
        } else {
            super::av2_ramp::vertical::<I32x4>(plane, ramps);
        }
    }
}

#[target_feature(enable = "neon")]
unsafe fn av2_chunk(plane: &mut [u8], segments: &super::av2::Av2Segments) {
    use crate::lpf_bytes::neon::B;
    // SAFETY: the ISA is enabled for this function (dispatch contract).
    unsafe {
        if segments.along == 1 {
            super::av2_decide::horizontal::<B>(plane, segments);
        } else {
            super::av2_decide::vertical::<B>(plane, segments);
        }
    }
}
