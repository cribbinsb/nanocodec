//! The picture-level facts the HEVC decoder records into [`StreamStats`]
//! when analysis is enabled (parameter sets §7.3.2, slice headers §7.3.6),
//! and the value conventions of its block records. `ctu` pushes the block
//! records at its parse points (§7.3.8); nothing here affects decoding.

use nc_common::stream_stats::{StreamStats, facts};

use super::Current;
use super::params::{Pps, Sps};
use super::slice::{SliceHeader, SliceSegment, SliceType};

/// Quarter-sample motion vector units per sample (§8.5.3.2.1).
const MV_UNITS_PER_SAMPLE: i64 = 4;

/// Per picture, weight 0: `log2(CtbSizeY)`, `MinCbLog2SizeY`, transform
/// sizes and depths, tool enables from the SPS and PPS.
const SPS_CTB_LOG2: &str = "sps_ctb_log2";
const SPS_MIN_CB_LOG2: &str = "sps_min_cb_log2";
const SPS_MIN_TB_LOG2: &str = "sps_min_tb_log2";
const SPS_MAX_TB_LOG2: &str = "sps_max_tb_log2";
const SPS_MAX_DEPTH_INTRA: &str = "sps_max_tu_depth_intra";
const SPS_MAX_DEPTH_INTER: &str = "sps_max_tu_depth_inter";
const SPS_AMP: &str = "sps_amp";
const SPS_SAO: &str = "sps_sao";
const SPS_TMVP: &str = "sps_temporal_mvp";
const SPS_STRONG_INTRA_SMOOTHING: &str = "sps_strong_intra_smoothing";
const PPS_TRANSFORM_SKIP: &str = "pps_transform_skip";
const PPS_SIGN_HIDING: &str = "pps_sign_hiding";
/// `diff_cu_qp_delta_depth`, or −1 when `cu_qp_delta_enabled_flag` is 0.
const PPS_CU_QP_DELTA_DEPTH: &str = "pps_cu_qp_delta_depth";
const PPS_WEIGHTED_PRED: &str = "pps_weighted_pred";
const PPS_SCALING_LISTS: &str = "pps_scaling_lists";
/// Per slice, weight 0: slice type (0 B, 1 P, 2 I as `slice_type`),
/// `SliceQpY`, reference counts, merge candidates and filter controls.
const SLICE_TYPE: &str = "slice_type";
const SLICE_QP: &str = "slice_qp";
const SLICE_NUM_REF_L0: &str = "slice_num_ref_l0";
const SLICE_MAX_MERGE: &str = "slice_max_merge";
const SLICE_DEBLOCK_DISABLED: &str = "slice_deblock_disabled";
const SLICE_BETA_OFFSET_DIV2: &str = "slice_beta_offset_div2";
const SLICE_TC_OFFSET_DIV2: &str = "slice_tc_offset_div2";
const SLICE_SAO_LUMA: &str = "slice_sao_luma";
const SLICE_SAO_CHROMA: &str = "slice_sao_chroma";
const SLICE_CB_QP_OFFSET: &str = "slice_cb_qp_offset";
/// Weight: the slice segment's RBSP bytes.
const SLICE_BYTES: &str = "slice_bytes";

/// Record a slice segment: the parameter sets' facts when it starts a
/// picture (`picture_start`), its slice header (not for a dependent segment) and its size.
pub(super) fn segment(
    s: &mut StreamStats,
    picture_start: Option<&Current>,
    segment: &SliceSegment,
    bytes: usize,
) {
    if let Some(current) = picture_start {
        picture(s, &current.sps, &current.pps);
    }
    if let Some(h) = &segment.header {
        slice(s, h);
    }
    s.fact(SLICE_BYTES, 0, bytes as u64);
}

/// Record the picture-level header facts at the first slice.
fn picture(s: &mut StreamStats, sps: &Sps, pps: &Pps) {
    s.begin_frame();
    s.fact(facts::MV_UNITS_PER_SAMPLE, MV_UNITS_PER_SAMPLE, 0);
    let header = [
        (SPS_CTB_LOG2, i64::from(sps.log2_ctb)),
        (SPS_MIN_CB_LOG2, i64::from(sps.log2_min_cb)),
        (SPS_MIN_TB_LOG2, i64::from(sps.log2_min_tb)),
        (SPS_MAX_TB_LOG2, i64::from(sps.log2_max_tb)),
        (SPS_MAX_DEPTH_INTRA, i64::from(sps.max_depth_intra)),
        (SPS_MAX_DEPTH_INTER, i64::from(sps.max_depth_inter)),
        (SPS_AMP, i64::from(sps.amp)),
        (SPS_SAO, i64::from(sps.sao)),
        (SPS_TMVP, i64::from(sps.temporal_mvp)),
        (
            SPS_STRONG_INTRA_SMOOTHING,
            i64::from(sps.strong_intra_smoothing),
        ),
        (PPS_TRANSFORM_SKIP, i64::from(pps.transform_skip)),
        (PPS_SIGN_HIDING, i64::from(pps.sign_hiding)),
        (
            PPS_CU_QP_DELTA_DEPTH,
            pps.cu_qp_delta_depth.map_or(-1, i64::from),
        ),
        (PPS_WEIGHTED_PRED, i64::from(pps.weighted_pred)),
        (
            PPS_SCALING_LISTS,
            i64::from(pps.scaling.is_some() || sps.scaling.is_some()),
        ),
    ];
    for (what, value) in header {
        s.fact(what, value, 0);
    }
}

/// Record a slice header (§7.3.6.1).
fn slice(s: &mut StreamStats, h: &SliceHeader) {
    let slice_type = match h.slice_type {
        SliceType::B => 0,
        SliceType::P => 1,
        SliceType::I => 2,
    };
    let fields = [
        (SLICE_TYPE, slice_type),
        (SLICE_QP, i64::from(h.qp)),
        (SLICE_NUM_REF_L0, h.num_ref_idx[0] as i64),
        (SLICE_MAX_MERGE, h.max_merge_candidates as i64),
        (SLICE_DEBLOCK_DISABLED, i64::from(h.deblock_disabled)),
        (SLICE_BETA_OFFSET_DIV2, i64::from(h.beta_offset_div2)),
        (SLICE_TC_OFFSET_DIV2, i64::from(h.tc_offset_div2)),
        (SLICE_SAO_LUMA, i64::from(h.sao[0])),
        (SLICE_SAO_CHROMA, i64::from(h.sao[1])),
        (SLICE_CB_QP_OFFSET, i64::from(h.cb_qp_offset)),
    ];
    for (what, value) in fields {
        s.fact(what, value, 0);
    }
}
