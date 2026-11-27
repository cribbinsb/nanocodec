//! Picture parameter sets (§7.3.2.2), with their slice groups
//! (`slice_groups.rs`) and the High extension (8x8 transform, picture
//! scaling matrices, a separate Cr QP offset) when present.

use super::reader::Reader;
use super::scaling::{self, ParsedMatrix};
use super::slice_groups::{self, SliceGroups};
use super::sps::{LISTS_8X8_Y, LISTS_8X8_YUV, MAX_SPS_ID};
use super::{DecodeError, Result};

/// `pic_parameter_set_id` range (§7.4.2.2).
pub(super) const MAX_PPS_ID: u32 = 255;
/// `num_ref_idx_l0_default_active_minus1` range for frames (§7.4.2.2).
pub(super) const MAX_REF_IDX_ACTIVE_MINUS1: u32 = 31;
/// `chroma_qp_index_offset` range (§7.4.2.2).
pub(super) const MAX_CHROMA_QP_OFFSET: i32 = 12;
/// `pic_init_qp_minus26` range (§7.4.2.2: from −(26 + `QpBdOffsetY`), here
/// at the deepest samples, 14 bits; the slice QP is checked against the
/// SPS's depth).
const MIN_INIT_QP_MINUS26: i32 = -(26 + 36);
const MAX_INIT_QP_MINUS26: i32 = 25;
/// `pic_init_qs_minus26` lower bound (§7.4.2.2: QS is 8-bit only).
const MIN_INIT_QS_MINUS26: i32 = -26;

/// `weighted_bipred_idc` (§7.4.2.2): the weighting of B slices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BipredWeights {
    /// 0: none (averaging).
    Default,
    /// 1: `pred_weight_table()` in the slice header.
    Explicit,
    /// 2: derived from picture order count distances (§8.4.2.3.1).
    Implicit,
}

/// The fields of a PPS that decoding uses.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "the flags mirror independent PPS syntax elements"
)]
pub(super) struct Pps {
    pub id: u32,
    pub sps_id: u32,
    /// `entropy_coding_mode_flag`: CABAC.
    pub cabac: bool,
    /// `bottom_field_pic_order_in_frame_present_flag`.
    pub bottom_field_pic_order_in_frame_present: bool,
    /// More than one slice group (`num_slice_groups_minus1` > 0).
    pub slice_groups: Option<SliceGroups>,
    /// `num_ref_idx_l0_default_active_minus1 + 1` and the list 1 one.
    pub num_ref_idx_default_active: [u32; 2],
    pub weighted_pred: bool,
    pub weighted_bipred: BipredWeights,
    pub pic_init_qp: i32,
    /// `26 + pic_init_qs_minus26`: the initial `QSY` of SP and SI slices.
    pub pic_init_qs: i32,
    /// `chroma_qp_index_offset` (Cb) and `second_chroma_qp_index_offset` (Cr).
    pub chroma_qp_offset: [i8; 2],
    pub deblocking_filter_control_present: bool,
    pub constrained_intra_pred: bool,
    /// `redundant_pic_cnt_present_flag`.
    pub redundant_pic_cnt_present: bool,
    pub transform_8x8_mode: bool,
    /// Picture scaling lists (fall-back applied at activation, Table 7-2).
    pub scaling: Option<ParsedMatrix>,
}

/// Parse a PPS RBSP; `chroma_444` tells whether the SPS of an id is 4:4:4,
/// whose picture scaling matrix has 8x8 lists for Cb and Cr as well
/// (§7.3.2.2: `6 + ( ( chroma_format_idc != 3 ) ? 2 : 6 ) *
/// transform_8x8_mode_flag` lists).
pub(super) fn parse(data: &[u8], chroma_444: impl FnOnce(u32) -> bool) -> Result<Pps> {
    let mut r = Reader::new(data);
    let id = r.ue_max(MAX_PPS_ID, "pic_parameter_set_id")?;
    let sps_id = r.ue_max(MAX_SPS_ID, "seq_parameter_set_id")?;
    let cabac = r.flag()?;
    let bottom_field_pic_order_in_frame_present = r.flag()?;
    let slice_groups = slice_groups::parse(&mut r)?;
    let mut num_ref_idx_default_active = [0; 2];
    for count in &mut num_ref_idx_default_active {
        *count = r.ue_max(
            MAX_REF_IDX_ACTIVE_MINUS1,
            "num_ref_idx_default_active_minus1",
        )? + 1;
    }
    let weighted_pred = r.flag()?;
    let weighted_bipred = match r.bits(2)? {
        0 => BipredWeights::Default,
        1 => BipredWeights::Explicit,
        2 => BipredWeights::Implicit,
        _ => return Err(DecodeError::Malformed("weighted_bipred_idc")),
    };
    let pic_init_qp = 26
        + r.se_range(
            MIN_INIT_QP_MINUS26,
            MAX_INIT_QP_MINUS26,
            "pic_init_qp_minus26",
        )?;
    let pic_init_qs = 26
        + r.se_range(
            MIN_INIT_QS_MINUS26,
            MAX_INIT_QP_MINUS26,
            "pic_init_qs_minus26",
        )?;
    let chroma_offset = |r: &mut Reader<'_>| {
        r.se_range(
            -MAX_CHROMA_QP_OFFSET,
            MAX_CHROMA_QP_OFFSET,
            "chroma_qp_index_offset",
        )
        .map(|v| v as i8)
    };
    let cb_offset = chroma_offset(&mut r)?;
    let deblocking_filter_control_present = r.flag()?;
    let constrained_intra_pred = r.flag()?;
    let redundant_pic_cnt_present = r.flag()?;
    let mut pps = Pps {
        id,
        sps_id,
        cabac,
        bottom_field_pic_order_in_frame_present,
        slice_groups,
        num_ref_idx_default_active,
        weighted_pred,
        weighted_bipred,
        pic_init_qp,
        pic_init_qs,
        chroma_qp_offset: [cb_offset; 2],
        deblocking_filter_control_present,
        constrained_intra_pred,
        redundant_pic_cnt_present,
        transform_8x8_mode: false,
        scaling: None,
    };
    if r.more_rbsp_data() {
        pps.transform_8x8_mode = r.flag()?;
        if r.flag()? {
            let count_8x8 = match (pps.transform_8x8_mode, chroma_444(sps_id)) {
                (false, _) => 0,
                (true, false) => LISTS_8X8_Y,
                (true, true) => LISTS_8X8_YUV,
            };
            pps.scaling = Some(scaling::parse_matrix(&mut r, count_8x8)?);
        }
        pps.chroma_qp_offset[1] = chroma_offset(&mut r)?;
    }
    Ok(pps)
}
