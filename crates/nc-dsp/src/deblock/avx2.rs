//! AVX2 deblocking kernels: H.264 on sixteen byte lanes (`h264_bytes`; single-plane
//! chroma on I16x8); HEVC on I16x8.

use super::{DeblockFns, Edge, HevcEdge, generic, hevc};
use crate::simd::avx2::I16x8;

/// The AVX2 table.
///
/// # Safety
///
/// The CPU must support AVX2 (`Isa::Avx2.is_supported()`).
pub(crate) unsafe fn deblock_fns() -> DeblockFns {
    DeblockFns {
        luma_pair: super::h264_runs::pair_avx2,
        h264_luma_runs: super::h264_runs::avx2,
        luma,
        chroma,
        chroma_pair,
        hevc_luma,
        hevc_chroma,
        av2_ramps,
        av1_edge,
        av1_line: super::av1_line::avx2,
        av1_grid: super::av1_grid::avx2,
        av1_runs: super::runs::avx2::av1,
        av1_lanes: Some(av1_lanes),
        hevc_luma_runs: super::runs::avx2::luma,
        hevc_chroma_runs: super::runs::avx2::chroma,
        vvc_luma_runs: super::runs::avx2::vvc,
        av2_chunk,
    }
}

#[target_feature(enable = "avx2")]
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
        super::av1_packed::filter_edge::<crate::lpf_bytes::avx2::B>(
            plane, q0, across, along, lines, filter,
        );
    }
}

/// The AV1 passes in the lane layout, 32 lines per vector.
#[target_feature(enable = "avx2")]
unsafe fn av1_lanes(
    plane: &mut [u8],
    stride: usize,
    vertical: bool,
    chroma: bool,
    lanes: &mut super::lanes::LoopFilterLanes,
) {
    use super::lanes::kernel::{Av1Chroma, Av1Luma};
    use crate::lpf_bytes::avx2_full::B;
    // SAFETY: AVX2 is enabled for this function (dispatch contract).
    unsafe {
        match (vertical, chroma) {
            (true, false) => lanes.filter::<B, Av1Luma, true>(plane, stride),
            (false, false) => lanes.filter::<B, Av1Luma, false>(plane, stride),
            (true, true) => lanes.filter::<B, Av1Chroma, true>(plane, stride),
            (false, true) => lanes.filter::<B, Av1Chroma, false>(plane, stride),
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn luma(plane: &mut [u8], edge: &Edge) {
    // SAFETY: AVX2 (and so SSE4.1) is enabled for this function (dispatch
    // contract).
    unsafe { super::h264_bytes::luma::<super::h264_bytes::x86::B>(plane, edge) }
}

#[target_feature(enable = "avx2")]
unsafe fn chroma(plane: &mut [u8], edge: &Edge) {
    // SAFETY: AVX2 is enabled for this function (dispatch contract).
    unsafe { generic::chroma::<I16x8, I16x8>(plane, edge) }
}

#[target_feature(enable = "avx2")]
unsafe fn chroma_pair(u: &mut [u8], v: &mut [u8], edge: &Edge) {
    // SAFETY: AVX2 (and so SSE4.1) is enabled for this function (dispatch
    // contract).
    unsafe { super::h264_bytes::chroma_pair::<super::h264_bytes::x86::B>(u, v, edge) }
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn hevc_luma(plane: &mut [u8], edge: &HevcEdge) {
    // SAFETY: AVX2 is enabled for this function (dispatch contract).
    unsafe { hevc::luma::<I16x8>(plane, edge) }
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn hevc_chroma(plane: &mut [u8], edge: &HevcEdge) {
    // SAFETY: AVX2 is enabled for this function (dispatch contract).
    unsafe { super::hevc_chroma::filter::<crate::lpf_bytes::avx2::B>(plane, edge) }
}

#[target_feature(enable = "avx2")]
unsafe fn av2_ramps(plane: &mut [u8], ramps: &super::av2::Av2Ramps) {
    use crate::simd::avx2::{I32x4, I32x8};
    // SAFETY: AVX2 is enabled for this function (dispatch contract).
    unsafe {
        if ramps.along == 1 {
            super::av2_ramp::horizontal::<I32x4>(plane, ramps);
        } else {
            super::av2_ramp::vertical::<I32x8>(plane, ramps);
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn av2_chunk(plane: &mut [u8], segments: &super::av2::Av2Segments) {
    use crate::lpf_bytes::avx2::B;
    // SAFETY: the ISA is enabled for this function (dispatch contract).
    unsafe {
        if segments.along == 1 {
            super::av2_decide::horizontal::<B>(plane, segments);
        } else {
            super::av2_decide::vertical::<B>(plane, segments);
        }
    }
}
