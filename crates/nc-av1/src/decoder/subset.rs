//! The frame-level gate in front of the block layer: maps a fully parsed
//! frame header onto the parameters [`super::tile`] decodes, or names the
//! first tool it does not decode yet. The names are what
//! `tools/ci/av1_conformance.sh` reports per vector, so they double as the
//! conformance gap list.
use super::frame_header::quant::{MAX_SEGMENTS, SEG_LVL_ALT_Q, SEG_LVL_MAX};
use super::frame_header::{
    CdefParams, FrameHeader, FrameSize, FrameType, GlobalMotion, LoopFilter, Restoration,
    TOTAL_REFS, TxMode,
};
use crate::obu::FrameKind;
use nc_common::decode::DecodeError;

fn unsupported(tool: &'static str) -> Result<(), DecodeError> {
    Err(DecodeError::Unsupported(tool))
}

/// What the block layer decodes a frame with.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)] // the specification's flags
pub(super) struct FrameParameters {
    pub kind: FrameKind,
    /// Luma size: the frame's, or the coded (8-aligned) size the block
    /// layer works in.
    pub width: usize,
    pub height: usize,
    pub base_q_idx: u32,
    /// (DC, AC) quantiser offsets of Y, U and V (§5.9.12).
    pub quant_deltas: [(i32, i32); 3],
    /// `reduced_tx_set`: the smaller transform-type sets.
    pub reduced_tx_set: bool,
    /// `TX_MODE_SELECT`: blocks code their transform sizes.
    pub tx_mode_select: bool,
    /// `CodedLossless` (segments are never lossless in the subset, so
    /// every block's `Lossless`): 4x4 Walsh-Hadamard transforms only.
    pub lossless: bool,
    /// Segmentation with spatially predicted ids and quantizer offsets
    /// only (`None`: off).
    pub segments: Option<SegmentQuantizers>,
    pub loop_filter: LoopFilter,
    pub cdef: CdefParams,
    /// The frame's size, including its superres upscaling.
    pub size: FrameSize,
    /// `lr_params()`: each plane's restoration type and unit size.
    pub restoration: Restoration,
    pub allow_high_precision_mv: bool,
    pub force_integer_mv: bool,
    /// `gm_params` per reference name.
    pub global_motion: [GlobalMotion; TOTAL_REFS],
    /// `RefFrameSignBias` per reference name.
    pub sign_bias: [bool; TOTAL_REFS],
    /// `reference_select`: blocks may be compound.
    pub reference_select: bool,
    /// `SkipModeFrame[]` when `skip_mode_present`.
    pub skip_mode_frames: Option<[i8; 2]>,
    /// `delta_q_res` (a shift) when superblocks code `delta_q_abs`
    /// (`delta_q_present`).
    pub delta_q_res: Option<u8>,
    /// `allow_screen_content_tools`: blocks may use palettes.
    pub screen_content: bool,
    /// `allow_intrabc`: blocks of this intra frame may copy from it.
    pub allow_intrabc: bool,
}

/// The segmentation the block layer decodes: ids coded with spatial
/// prediction (§5.11.9), kept from or temporally predicted by the primary
/// reference's map (§5.11.19), and segments that differ only in their
/// quantizer (`SEG_LVL_ALT_Q`, §7.12.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SegmentQuantizers {
    /// `FeatureData[segment][SEG_LVL_ALT_Q]` (0 where the feature is off).
    pub alt_q: [i16; MAX_SEGMENTS],
    /// `LastActiveSegId`.
    pub last_active: u8,
    /// `segmentation_update_map` and `segmentation_temporal_update`.
    pub update_map: bool,
    pub temporal_update: bool,
}

/// The block layer's frame parameters for `h`.
///
/// # Errors
/// `Unsupported` naming the first frame-level tool outside the block
/// layer's subset.
pub(super) fn frame_parameters(h: &FrameHeader) -> Result<FrameParameters, DecodeError> {
    let kind = match h.frame_type {
        FrameType::Inter => FrameKind::Inter,
        // The block layer decodes intra-only frames as key frames
        // (`FrameIsIntra`); their references and refresh are the frame
        // header's.
        FrameType::Key | FrameType::IntraOnly => FrameKind::Key,
        FrameType::Switch => return Err(DecodeError::Unsupported("AV1 switch frames")),
    };
    check_coding_tools(h)?;
    let segments = segment_quantizers(h)?;
    let q = &h.quant;
    if q.qm.is_some() {
        unsupported("AV1 quantizer matrices")?;
    }
    Ok(FrameParameters {
        kind,
        width: h.size.width,
        height: h.size.height,
        base_q_idx: q.base_q_idx,
        quant_deltas: [
            (q.delta_y_dc, 0),
            (q.delta_u_dc, q.delta_u_ac),
            (q.delta_v_dc, q.delta_v_ac),
        ],
        reduced_tx_set: h.reduced_tx_set,
        tx_mode_select: h.tx_mode == TxMode::Select,
        lossless: h.coded_lossless,
        segments,
        loop_filter: h.loop_filter,
        cdef: h.cdef,
        size: h.size,
        restoration: h.restoration,
        allow_high_precision_mv: h.allow_high_precision_mv,
        force_integer_mv: h.force_integer_mv,
        global_motion: h.global_motion,
        sign_bias: h.sign_bias,
        reference_select: h.reference_select,
        skip_mode_frames: h
            .skip_mode_frames
            .map(|frames| frames.map(|name| name as i8)),
        delta_q_res: h.delta.q_present.then_some(h.delta.q_res),
        screen_content: h.allow_screen_content_tools,
        allow_intrabc: h.allow_intrabc,
    })
}

/// The frame's segmentation, or `Unsupported` for the parts outside the
/// subset: features other than the quantizer, and lossless segments in a
/// frame that is not wholly lossless.
fn segment_quantizers(h: &FrameHeader) -> Result<Option<SegmentQuantizers>, DecodeError> {
    let s = &h.segmentation;
    if !s.enabled {
        return Ok(None);
    }
    let mut alt_q = [0; MAX_SEGMENTS];
    for (segment, (enabled, data)) in s.features.enabled.iter().zip(&s.features.data).enumerate() {
        if (0..SEG_LVL_MAX).any(|feature| feature != SEG_LVL_ALT_Q && enabled[feature]) {
            unsupported("AV1 segment features other than the quantizer")?;
        }
        if enabled[SEG_LVL_ALT_Q] {
            alt_q[segment] = data[SEG_LVL_ALT_Q];
        }
        if s.qindex(h.quant.base_q_idx, segment) == 0 && !h.coded_lossless {
            unsupported("AV1 lossless segments")?;
        }
    }
    Ok(Some(SegmentQuantizers {
        alt_q,
        last_active: s.last_active,
        update_map: s.update_map,
        temporal_update: s.temporal_update,
    }))
}

/// Prediction and coefficient tools.
fn check_coding_tools(h: &FrameHeader) -> Result<(), DecodeError> {
    if !h.size.width.is_multiple_of(2) || !h.size.height.is_multiple_of(2) {
        unsupported("AV1 odd frame size")?;
    }
    if h.delta.lf_present {
        unsupported("AV1 block loop filter deltas")?;
    }
    Ok(())
}
