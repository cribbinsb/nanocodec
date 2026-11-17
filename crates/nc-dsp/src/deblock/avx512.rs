//! AVX-512 entries of the shared byte deblocking family. AV1 batches four
//! eight-line tiles; AV2 retains its eight-segment decision axis and batches
//! the four vertical ramp rows using the very same grouped adapter.
//! GFNI byte shifts apply to short edges too, using the AVX2 layout. CPUs
//! without GFNI retain the AVX2 table; neither codec's arithmetic is duplicated.
use super::av1::Av1EdgeFilter;
use super::av1_tile::GroupedTile;
use crate::lpf_bytes::{avx512, gfni};

pub(crate) fn install(table: &mut super::DeblockFns) {
    table.av1_edge = av1_edge;
    table.av1_line = super::av1_line::avx512;
    table.av1_grid = super::av1_grid::avx512;
    table.av1_runs = super::runs::avx512;
    table.luma_pair = super::h264_runs::pair_avx512;
    table.h264_luma_runs = super::h264_runs::avx512;
    table.hevc_luma_runs = super::runs::hevc_luma_avx512;
    table.vvc_luma_runs = super::runs::vvc_luma_avx512;
    table.av2_chunk = av2_chunk;
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,gfni")]
pub(super) unsafe fn av1_edge(
    plane: &mut [u8],
    q0: usize,
    across: usize,
    along: usize,
    lines: usize,
    filter: Av1EdgeFilter,
) {
    // SAFETY: dispatch validates the footprint; this entry enables the ISA.
    unsafe {
        super::av1_packed::filter_wide::<avx512::B, GroupedTile<avx512::B>, 32>(
            plane, q0, across, along, lines, filter, tail,
        );
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,gfni")]
unsafe fn tail(
    plane: &mut [u8],
    q0: usize,
    across: usize,
    along: usize,
    lines: usize,
    filter: Av1EdgeFilter,
) {
    // SAFETY: the parent validated these remaining lines; AVX2 is supported.
    unsafe {
        super::av1_packed::filter_wide::<gfni::Pair, GroupedTile<gfni::Pair>, 16>(
            plane, q0, across, along, lines, filter, short,
        );
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,gfni")]
unsafe fn av2_chunk(plane: &mut [u8], segments: &super::av2::Av2Segments) {
    // SAFETY: the footprint is validated by dispatch and the chunk entry;
    // this function enables the grouped ramp backend's ISA.
    unsafe {
        if segments.along == 1 {
            super::av2_decide::horizontal::<gfni::Narrow>(plane, segments);
        } else {
            super::av2_decide::vertical_with::<gfni::Narrow, avx512::B>(plane, segments);
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,gfni")]
unsafe fn short(
    plane: &mut [u8],
    q0: usize,
    across: usize,
    along: usize,
    lines: usize,
    filter: Av1EdgeFilter,
) {
    // SAFETY: parent checked all samples; this entry enables AVX2 and GFNI.
    unsafe {
        super::av1_packed::filter_edge::<gfni::Narrow>(plane, q0, across, along, lines, filter);
    }
}
