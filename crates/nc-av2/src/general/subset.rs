//! The block layer the general decoder uses until its own exists: the
//! subset tile decoder ([`crate::tile_decoder`], our encoder's syntax),
//! fed from the general headers. Each frame-level tool the subset lacks is
//! rejected here by name; sequence-level tools are rejected by the subset
//! sequence parse ([`crate::sequence::parse`]) of the same OBU.
//!
//! This module goes with the subset path (docs/handoff/av2.md "Two decode
//! paths, and when one goes").
use super::frame_header::{FrameHeader, FrameType, filter_mode};
use crate::ccso::CcsoParams;
use crate::cdef::CdefParams;
use crate::coeff::LevelCoding;
use crate::deblock::DeblockParams;
use crate::interp_filter::FrameFilter;
use crate::prediction::InterpFilter;
use crate::sequence::SequenceHeader;
use crate::tile_decoder::TileParams;
use crate::transform::QuantDeltas;
use crate::tx_type::TxSetMode;
use nc_common::decode::{DecodeError, Result};

/// Quarter-sample motion vectors (`mv_precision` 2), the subset's only.
const QUARTER_SAMPLE: u8 = 2;

/// The subset tile decoder's parameters for a frame, or the first tool of
/// the frame it lacks.
/// # Errors
/// Names the unsupported tool.
pub(crate) fn tile_params(sequence: &SequenceHeader, header: &FrameHeader) -> Result<TileParams> {
    check_frame_tools(sequence, header)?;
    let q = header.quantizer;
    Ok(TileParams {
        qindex: q.base as u8,
        quant_deltas: QuantDeltas {
            luma_dc: q.y_dc,
            chroma_ac: q.u_ac,
            chroma_dc: q.u_dc,
        },
        adaptive_cdf: !header.disable_cdf_update,
        deblock: deblock(header),
        cdef: cdef(header, sequence.single_picture)?,
        ccso: ccso(header)?,
        tx_set: TxSetMode::from_header(u32::from(header.reduced_tx_set))?,
        screen_content: header.screen_content,
        tx_select: header.tx_mode_select,
        reference_lists: sequence.reference_lists,
        levels: LevelCoding::of(header.parity_hiding, header.tcq),
        delta_q: header.delta_q,
    })
}

/// The frame's interpolation filter.
#[must_use]
pub(crate) fn frame_filter(header: &FrameHeader) -> FrameFilter {
    if header.filter == filter_mode::SWITCHABLE {
        FrameFilter::Switchable
    } else {
        FrameFilter::Fixed(
            InterpFilter::from_code(u32::from(header.filter)).expect("two-bit filter codes"),
        )
    }
}

/// Reject the frame-level tools outside the subset.
fn check_frame_tools(sequence: &SequenceHeader, header: &FrameHeader) -> Result<()> {
    let unsupported = |tool| Err(DecodeError::Unsupported(tool));
    match header.frame_type {
        FrameType::Intra => return unsupported("AV2 intra-only frames"),
        FrameType::Switch => return unsupported("AV2 switch frames"),
        FrameType::Key | FrameType::Inter => {}
    }
    if (header.width, header.height) != (sequence.max_width, sequence.max_height) {
        return unsupported("AV2 frame size other than the sequence maximum");
    }
    if header.tip.mode != 0 {
        return unsupported("AV2 temporal interpolated prediction");
    }
    if header.bridge_slot.is_some() {
        return unsupported("AV2 bridge frames");
    }
    if header.intra_block_copy.is_some() {
        return unsupported("AV2 intra block copy");
    }
    if header.tiles.columns() * header.tiles.rows() != 1 {
        return unsupported("AV2 multiple tiles");
    }
    if header.segmentation.is_some() {
        return unsupported("AV2 segmentation");
    }
    if header.quantizer.matrices.is_some() {
        return unsupported("AV2 quantizer matrices");
    }
    let q = header.quantizer;
    if q.v_ac != q.u_ac || q.v_dc != q.u_dc {
        return unsupported("AV2 separate U and V quantizer deltas");
    }
    if (q.base == 0) != header.lossless.all {
        return unsupported("AV2 lossless frames with quantizer deltas");
    }
    if header.deblock.sub_pu {
        return unsupported("AV2 sub-PU deblocking");
    }
    if header.gdf.mode != 0 {
        return unsupported("AV2 guided detail filter");
    }
    if header
        .restoration
        .planes
        .iter()
        .any(|plane| plane.kind != 0)
    {
        return unsupported("AV2 loop restoration");
    }
    if !header.is_intra() {
        if header.force_integer_mv || header.mv_precision != QUARTER_SAMPLE {
            return unsupported("AV2 motion vector precision other than quarter-sample");
        }
        if header.reference_select {
            return unsupported("AV2 compound reference selection");
        }
        if header.bawp {
            return unsupported("AV2 block adaptive weighted prediction");
        }
        if header.warp_delta {
            return unsupported("AV2 warp delta motion");
        }
        if header.use_motion_fields {
            return unsupported("AV2 temporal motion fields");
        }
    }
    Ok(())
}

fn deblock(header: &FrameHeader) -> DeblockParams {
    let d = header.deblock;
    let on = |level: bool, delta: i8| level.then_some(delta);
    DeblockParams::decoded([
        on(d.level_y[0], d.delta_q_y[0]),
        on(d.level_y[1], d.delta_q_y[1]),
        on(d.level_u, d.delta_q_u),
        on(d.level_v, d.delta_q_v),
    ])
}

fn cdef(header: &FrameHeader, single_picture: bool) -> Result<CdefParams> {
    super::filters::cdef_params(header, single_picture)
}

/// The frame's CCSO; the subset has no filter reuse.
fn ccso(header: &FrameHeader) -> Result<CcsoParams> {
    if header
        .ccso
        .planes
        .iter()
        .any(|plane| plane.enabled && (plane.reuse || plane.block_reuse))
    {
        return Err(DecodeError::Unsupported("AV2 CCSO filter reuse"));
    }
    super::filters::ccso_params(header)
}
