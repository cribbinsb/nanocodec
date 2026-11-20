//! NEON deblocking kernels: H.264 on sixteen byte lanes (`h264_bytes`; single-plane
//! chroma on I16x8); HEVC on I16x8.

use super::{DeblockFns, Edge, HevcEdge, generic, hevc};
use crate::simd::neon::I16x8;

/// The NEON table.
///
/// NEON is always present on `AArch64`.
pub(crate) fn deblock_fns() -> DeblockFns {
    DeblockFns {
        luma_pair: super::h264_runs::pair_neon,
        h264_luma_runs: super::h264_runs::neon,
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
        av1_lanes: None,
        hevc_luma_runs: super::runs::neon::luma,
        hevc_chroma_runs: super::runs::neon::chroma,
        vvc_luma_runs: super::runs::neon::vvc,
        av2_chunk,
    }
}

/// Sixteen independent lines share byte decisions; short tails keep the
/// established eight-line graph. The grouped lane adapter is also VP9's.
#[target_feature(enable = "neon")]
pub(super) unsafe fn av1_edge(
    plane: &mut [u8],
    q0: usize,
    across: usize,
    along: usize,
    lines: usize,
    filter: super::av1::Av1EdgeFilter,
) {
    // SAFETY: dispatch validates the footprint and NEON is enabled here.
    unsafe {
        if lines < 16 {
            av1_edge_narrow(plane, q0, across, along, lines, filter);
        } else {
            super::av1_packed::filter_wide::<
                crate::lpf_bytes::neon_wide::B,
                super::av1_tile::GroupedTile<crate::lpf_bytes::neon_wide::B>,
                16,
            >(plane, q0, across, along, lines, filter, av1_edge_narrow);
        }
    }
}

#[target_feature(enable = "neon")]
unsafe fn av1_edge_narrow(
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
