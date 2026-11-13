//! Picture parameter sets (§7.3.2.2) within Constrained High: one slice
//! group (no FMO) and no redundant pictures; the High extension (8x8
//! transform, picture scaling matrices, a separate Cr QP offset) is parsed
//! when present.

use super::reader::Reader;
use super::scaling::{self, ParsedMatrix};
use super::sps::MAX_SPS_ID;
use super::{DecodeError, Result};

/// `pic_parameter_set_id` range (§7.4.2.2).
pub(super) const MAX_PPS_ID: u32 = 255;
/// `num_ref_idx_l0_default_active_minus1` range for frames (§7.4.2.2).
pub(super) const MAX_REF_IDX_ACTIVE_MINUS1: u32 = 31;
/// `chroma_qp_index_offset` range (§7.4.2.2).
pub(super) const MAX_CHROMA_QP_OFFSET: i32 = 12;
/// `pic_init_qp_minus26` range for 8-bit video (§7.4.2.2).
const MIN_INIT_QP_MINUS26: i32 = -26;
const MAX_INIT_QP_MINUS26: i32 = 25;

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
    pub num_ref_idx_l0_default_active: u32,
    pub weighted_pred: bool,
    pub pic_init_qp: i32,
    /// `chroma_qp_index_offset` (Cb) and `second_chroma_qp_index_offset` (Cr).
    pub chroma_qp_offset: [i8; 2],
    pub deblocking_filter_control_present: bool,
    pub constrained_intra_pred: bool,
    pub transform_8x8_mode: bool,
    /// Picture scaling lists (fall-back applied at activation, Table 7-2).
    pub scaling: Option<ParsedMatrix>,
}

pub(super) fn parse(data: &[u8]) -> Result<Pps> {
    let mut r = Reader::new(data);
    let id = r.ue_max(MAX_PPS_ID, "pic_parameter_set_id")?;
    let sps_id = r.ue_max(MAX_SPS_ID, "seq_parameter_set_id")?;
    let cabac = r.flag()?;
    let bottom_field_pic_order_in_frame_present = r.flag()?;
    if r.ue()? != 0 {
        return Err(DecodeError::Unsupported("H264 slice groups (FMO)"));
    }
    let num_ref_idx_l0_default_active = r.ue_max(
        MAX_REF_IDX_ACTIVE_MINUS1,
        "num_ref_idx_l0_default_active_minus1",
    )? + 1;
    r.ue_max(
        MAX_REF_IDX_ACTIVE_MINUS1,
        "num_ref_idx_l1_default_active_minus1",
    )?;
    let weighted_pred = r.flag()?;
    r.bits(2)?; // weighted_bipred_idc: B slices only
    let pic_init_qp = 26
        + r.se_range(
            MIN_INIT_QP_MINUS26,
            MAX_INIT_QP_MINUS26,
            "pic_init_qp_minus26",
        )?;
    r.se()?; // pic_init_qs_minus26: SP/SI slices only
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
    if r.flag()? {
        return Err(DecodeError::Unsupported("H264 redundant pictures"));
    }
    let mut pps = Pps {
        id,
        sps_id,
        cabac,
        bottom_field_pic_order_in_frame_present,
        num_ref_idx_l0_default_active,
        weighted_pred,
        pic_init_qp,
        chroma_qp_offset: [cb_offset; 2],
        deblocking_filter_control_present,
        constrained_intra_pred,
        transform_8x8_mode: false,
        scaling: None,
    };
    if r.more_rbsp_data() {
        pps.transform_8x8_mode = r.flag()?;
        if r.flag()? {
            let count_8x8 = if pps.transform_8x8_mode {
                scaling::LISTS_8X8
            } else {
                0
            };
            pps.scaling = Some(scaling::parse_matrix(&mut r, count_8x8)?);
        }
        pps.chroma_qp_offset[1] = chroma_offset(&mut r)?;
    }
    Ok(pps)
}
