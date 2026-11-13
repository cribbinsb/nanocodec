//! The frame-level gate in front of the block layer: maps a fully parsed
//! frame header onto the parameters [`super::tile`] decodes, or names the
//! first tool it does not decode yet. The names are what
//! `tools/ci/av1_conformance.sh` reports per vector, so they double as the
//! conformance gap list.
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
    /// `CodedLossless` (segmentation is outside the subset, so every
    /// block's `Lossless`): 4x4 Walsh-Hadamard transforms only.
    pub lossless: bool,
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

/// Prediction and coefficient tools.
fn check_coding_tools(h: &FrameHeader) -> Result<(), DecodeError> {
    if !h.size.width.is_multiple_of(2) || !h.size.height.is_multiple_of(2) {
        unsupported("AV1 odd frame size")?;
    }
    if h.segmentation.enabled {
        unsupported("AV1 segmentation")?;
    }
    if h.delta.lf_present {
        unsupported("AV1 block loop filter deltas")?;
    }
    Ok(())
}
