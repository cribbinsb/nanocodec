//! NAL unit headers and parameter sets (HEVC §7.3.1–7.3.3, §7.3.6.1).
//!
//! The stream uses one VPS, SPS and PPS, all with id 0, and resent before
//! every IDR picture with identical content, so a decoder (or a CMAF
//! segmenter) can start at any IDR. Between IDRs every picture is a P
//! picture predicted from one earlier picture. A flat chain's SPS carries a
//! single short-term reference picture set, "the picture with POC − 1".
//! Hierarchical-P structures (`nc_encoder` `structure.rs`) are temporal
//! sub-layers: one sub-layer per temporal layer (`TemporalId`), one SPS
//! reference picture set per mini-GOP position, and `TRAIL_N` top-layer
//! pictures, so sub-bitstream extraction by `TemporalId` (§10) leaves a
//! conforming stream.

use nc_common::colour::write_video_signal_type;
use nc_common::decode::{DecodeError, annex_b_units};
use nc_common::{BitWriter, ColourInfo};
use nc_encoder::{FrameRate, StreamSignal, TimingSignal};

/// NAL unit types used (HEVC Table 7-1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NalType {
    /// Coded slice segment of a trailing, sub-layer non-reference picture.
    TrailN = 0,
    /// Coded slice segment of a trailing, reference picture.
    TrailR = 1,
    /// Coded slice segment of an IDR picture without leading pictures.
    IdrWRadl = 19,
    /// Coded slice segment of a clean random access picture: an IRAP
    /// picture that keeps the POC count going (see `irap_slice_header`).
    Cra = 21,
    /// Video parameter set.
    Vps = 32,
    /// Sequence parameter set.
    Sps = 33,
    /// Picture parameter set.
    Pps = 34,
}

/// `nal_unit_header()` (HEVC §7.3.1.2): layer 0, `temporal_id`.
#[must_use]
pub fn nal_header(nal_type: NalType, temporal_id: u8) -> [u8; 2] {
    // forbidden_zero_bit (0) | nal_unit_type (6) | nuh_layer_id (6, = 0)
    // | nuh_temporal_id_plus1 (3).
    [(nal_type as u8) << 1, temporal_id + 1]
}

/// The temporal layer of an emitted access unit: its slice's `TemporalId`
/// (`nuh_temporal_id_plus1 − 1`), what sub-bitstream extraction (§10)
/// compares against the highest sub-layer it keeps.
///
/// # Errors
///
/// Malformed Annex B framing, or an access unit without a slice.
pub fn access_unit_layer(access_unit: &[u8]) -> Result<u8, DecodeError> {
    annex_b_units(access_unit)?
        .into_iter()
        .filter(|unit| unit.len() >= 2 && (unit[0] >> 1) < FIRST_NON_VCL_TYPE)
        .map(|unit| unit[1].saturating_sub(1))
        .next()
        .ok_or(DecodeError::Malformed("access unit without a slice"))
}

/// The first non-VCL NAL unit type (Table 7-1): smaller types are slices.
const FIRST_NON_VCL_TYPE: u8 = 32;

/// One short-term reference picture set (§7.3.7): the POC distances back to
/// each picture kept, and whether the current picture uses it.
pub type ReferenceSet = &'static [(u32, bool)];

/// How a P slice names its reference picture set.
#[derive(Debug, Clone, Copy)]
pub enum SliceReferenceSet<'a> {
    /// SPS set `index` of `count`.
    Sps {
        /// `short_term_ref_pic_set_idx`.
        index: usize,
        /// `num_short_term_ref_pic_sets`.
        count: usize,
    },
    /// Coded in the slice header (`short_term_ref_pic_set_sps_flag` 0):
    /// POC distances back and whether the picture is used, nearest first;
    /// the SPS has `sps_count` sets.
    Explicit {
        /// (distance, used) pairs.
        entries: &'a [(u32, bool)],
        /// `num_short_term_ref_pic_sets`.
        sps_count: usize,
    },
}

/// The SPS reference picture sets of each structure (indexed by
/// `temporal_layers - 1`), one per mini-GOP position: the reference
/// (used), plus any picture a later position still needs (not used). L1T3
/// position 3 keeps its T0 three pictures back for the next T0.
pub const REFERENCE_SETS: [&[ReferenceSet]; 3] = [
    &[&[(1, true)]],
    &[&[(2, true)], &[(1, true)]],
    &[
        &[(4, true)],
        &[(1, true)],
        &[(2, true)],
        &[(1, true), (3, false)],
    ],
];

/// `SubWidthC` = `SubHeightC` of 4:2:0 (Table 6-1): conformance window
/// offsets count chroma samples.
const CHROMA_SUBSAMPLING: usize = 2;
/// Coding geometry signalled in the SPS.
pub const LOG2_MIN_CB: u32 = 3;
/// CTU size 64.
pub const LOG2_CTB: u32 = 6;
/// `diff_cu_qp_delta_depth` (HEVC §7.4.3.3) with block QP: quantization
/// groups are whole CTBs. The engine plans offsets per 64x64 square
/// (`BLOCK_QP_LOG2`, as AV1's superblock `delta_q`), so a finer group would
/// only add deltas; a CTB group also makes `qPY_PRED` simply the previous
/// group's last `QpY` (§8.6.1: left/above count only inside the CTB).
pub const QUANT_GROUP_DEPTH: u32 = 0;
/// Smallest transform block, 4x4.
pub const LOG2_MIN_TB: u32 = 2;
/// Largest transform block, 32x32.
pub const LOG2_MAX_TB: u32 = 5;
/// `log2_max_pic_order_cnt_lsb_minus4`: 8-bit POC LSBs.
const LOG2_MAX_POC_LSB_MINUS4: u32 = 4;
/// `sign_data_hiding_enabled_flag`: one sign per 4x4 sub-block is carried
/// by the parity of its levels instead of a bit (about 1–2% fewer bits).
pub const SIGN_DATA_HIDING: bool = true;
/// Bits of `slice_pic_order_cnt_lsb`.
pub const POC_LSB_BITS: u32 = LOG2_MAX_POC_LSB_MINUS4 + 4;
/// Main profile (`general_profile_idc`, HEVC §A.3.2).
const PROFILE_MAIN: u32 = 1;
/// Main 10 (Main streams also conform to it, so its compatibility flag is set).
const PROFILE_MAIN10: u32 = 2;

/// Stream-level properties needed by the parameter sets.
#[derive(Debug, Clone, Copy)]
pub struct StreamInfo {
    /// Coded luma width (a multiple of the minimum CU size).
    pub width: usize,
    /// Coded luma height.
    pub height: usize,
    /// Luma samples cropped from the right and bottom of the coded picture
    /// for display (the conformance window; even for 4:2:0).
    pub crop: (usize, usize),
    /// `general_level_idc` (30 × level number).
    pub level_idc: u32,
    /// Initial QP signalled in the PPS (`init_qp_minus26 + 26`).
    pub init_qp: i32,
    /// Largest luma transform side, log2 (16 or32 samples).
    pub max_transform_log2: u8,
    /// Optional transform-tree depth below the CU in inter prediction.
    pub inter_transform_depth: u8,
    /// Optional transform-tree depth below the CU in intra prediction
    /// (`max_transform_hierarchy_depth_intra`; 0 except in the syntax
    /// stress test).
    pub intra_transform_depth: u8,
    /// `pps_cb_qp_offset` and `pps_cr_qp_offset`.
    pub chroma_qp_offset: i32,
    /// The engine chroma policy (`nc_encoder::chroma_policy`), if on:
    /// `pps_slice_chroma_qp_offsets_present_flag`, and each slice carries
    /// its picture's offset ([`SliceQp::chroma_offset`]).
    pub chroma_policy: Option<nc_encoder::chroma_policy::ChromaPolicy>,
    /// `pps_beta_offset_div2` and `pps_tc_offset_div2`.
    pub deblock_offsets_div2: (i32, i32),
    /// `deblocking_filter_override_enabled_flag`: slices may override the
    /// PPS deblocking control (`hevc.deblock_inter` 0, on every frame or on
    /// non-base frames, and the syntax stress test).
    pub deblock_override: bool,
    /// Temporal sub-layers (1 for a flat chain).
    pub temporal_layers: u8,
    /// Colour and timing signalling; when either is set, the SPS carries a
    /// VUI for it.
    pub signal: StreamSignal,
    /// Nominal frame rate (the VUI timing clock).
    pub frame_rate: FrameRate,
    /// A long-term reference is kept (`ltr.interval`): slices carry their
    /// own reference picture sets ([`SliceReferenceSet::Explicit`]).
    pub long_term: bool,
    /// CUs carry `cu_qp_delta` (`cu_qp_delta_enabled_flag`), one per
    /// CTB-sized quantization group (`diff_cu_qp_delta_depth` 0).
    pub block_qp: bool,
    /// `MaxNumMergeCand` of every P slice (1..=5,
    /// `mvpred::MAX_MERGE_CANDIDATES`).
    pub merge_candidates: usize,
}

impl StreamInfo {
    /// The reference picture sets of this stream's structure.
    #[must_use]
    pub fn reference_sets(&self) -> &'static [ReferenceSet] {
        REFERENCE_SETS[usize::from(self.temporal_layers - 1)]
    }

    /// Pictures the DPB must hold: the largest reference set plus the
    /// current picture.
    fn dpb_pictures(&self) -> u32 {
        let largest = self.reference_sets().iter().map(|set| set.len()).max();
        let largest = largest.unwrap_or(0) as u32;
        // A long-term reference is kept beside the structure's pictures.
        let kept = largest + u32::from(self.long_term);
        kept + 1
    }
}

/// `profile_tier_level(1, max_sub_layers_minus1)` for Main profile, Main
/// tier (HEVC §7.3.3); sub-layers signal no profile or level of their own.
fn profile_tier_level(w: &mut BitWriter, level_idc: u32, max_sub_layers_minus1: u32) {
    w.put_bits(0, 2); // general_profile_space
    w.put_bit(false); // general_tier_flag: Main tier
    w.put_bits(PROFILE_MAIN, 5);
    for j in 0..32 {
        w.put_bit(j == PROFILE_MAIN || j == PROFILE_MAIN10);
    }
    w.put_bit(true); // general_progressive_source_flag
    w.put_bit(false); // general_interlaced_source_flag
    w.put_bit(false); // general_non_packed_constraint_flag
    w.put_bit(true); // general_frame_only_constraint_flag
    w.put_bits(0, 32); // general_reserved_zero_43bits (first 32)
    w.put_bits(0, 11); //   ... (last 11)
    w.put_bit(false); // general_inbld_flag / reserved
    w.put_bits(level_idc, 8);
    for _ in 0..max_sub_layers_minus1 {
        w.put_bit(false); // sub_layer_profile_present_flag
        w.put_bit(false); // sub_layer_level_present_flag
    }
    if max_sub_layers_minus1 > 0 {
        for _ in max_sub_layers_minus1..8 {
            w.put_bits(0, 2); // reserved_zero_2bits
        }
    }
}

/// `video_parameter_set_rbsp()` (HEVC §7.3.2.1).
#[must_use]
pub fn vps(info: &StreamInfo) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put_bits(0, 4); // vps_video_parameter_set_id
    w.put_bits(0b11, 2); // vps_base_layer_internal_flag, vps_base_layer_available_flag
    w.put_bits(0, 6); // vps_max_layers_minus1
    let sub_layers_minus1 = u32::from(info.temporal_layers - 1);
    w.put_bits(sub_layers_minus1, 3); // vps_max_sub_layers_minus1
    // Every picture references only lower or equal sub-layers that no
    // lower-sub-layer picture has superseded, so nesting holds (§7.4.3.1).
    w.put_bit(true); // vps_temporal_id_nesting_flag
    w.put_bits(0xffff, 16); // vps_reserved_0xffff_16bits
    profile_tier_level(&mut w, info.level_idc, sub_layers_minus1);
    w.put_bit(true); // vps_sub_layer_ordering_info_present_flag
    write_dpb_sizes(&mut w, info);
    w.put_bits(0, 6); // vps_max_layer_id
    w.put_ue(0); // vps_num_layer_sets_minus1
    w.put_bit(false); // vps_timing_info_present_flag
    w.put_bit(false); // vps_extension_flag
    w.trailing_bits();
    w.into_bytes()
}

/// `seq_parameter_set_rbsp()` (HEVC §7.3.2.2): 8-bit 4:2:0, CTU 64, CU
/// 8..64, TU 4..32 with no transform hierarchy beyond what sizes imply,
/// and every optional tool off.
#[must_use]
pub fn sps(info: &StreamInfo) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put_bits(0, 4); // sps_video_parameter_set_id
    let sub_layers_minus1 = u32::from(info.temporal_layers - 1);
    w.put_bits(sub_layers_minus1, 3); // sps_max_sub_layers_minus1
    w.put_bit(true); // sps_temporal_id_nesting_flag
    profile_tier_level(&mut w, info.level_idc, sub_layers_minus1);
    w.put_ue(0); // sps_seq_parameter_set_id
    w.put_ue(1); // chroma_format_idc: 4:2:0
    w.put_ue(info.width as u32); // pic_width_in_luma_samples
    w.put_ue(info.height as u32); // pic_height_in_luma_samples
    // Pictures are coded in whole minimum CUs; the conformance window
    // (in chroma sample units, SubWidthC = SubHeightC = 2) crops the
    // display size back out (§7.4.3.2.1).
    let cropped = info.crop != (0, 0);
    w.put_bit(cropped); // conformance_window_flag
    if cropped {
        w.put_ue(0); // conf_win_left_offset
        w.put_ue((info.crop.0 / CHROMA_SUBSAMPLING) as u32); // conf_win_right_offset
        w.put_ue(0); // conf_win_top_offset
        w.put_ue((info.crop.1 / CHROMA_SUBSAMPLING) as u32); // conf_win_bottom_offset
    }
    w.put_ue(0); // bit_depth_luma_minus8
    w.put_ue(0); // bit_depth_chroma_minus8
    w.put_ue(LOG2_MAX_POC_LSB_MINUS4);
    w.put_bit(true); // sps_sub_layer_ordering_info_present_flag
    write_dpb_sizes(&mut w, info);
    w.put_ue(LOG2_MIN_CB - 3); // log2_min_luma_coding_block_size_minus3
    w.put_ue(LOG2_CTB - LOG2_MIN_CB); // log2_diff_max_min_luma_coding_block_size
    w.put_ue(LOG2_MIN_TB - 2); // log2_min_luma_transform_block_size_minus2
    w.put_ue(u32::from(info.max_transform_log2) - LOG2_MIN_TB); // log2_diff_max_min_luma_transform_block_size
    w.put_ue(u32::from(info.inter_transform_depth)); // max_transform_hierarchy_depth_inter
    w.put_ue(u32::from(info.intra_transform_depth)); // max_transform_hierarchy_depth_intra
    w.put_bit(false); // scaling_list_enabled_flag
    w.put_bit(false); // amp_enabled_flag
    w.put_bit(true); // sample_adaptive_offset_enabled_flag
    w.put_bit(false); // pcm_enabled_flag
    let sets = info.reference_sets();
    w.put_ue(sets.len() as u32); // num_short_term_ref_pic_sets
    for (index, set) in sets.iter().enumerate() {
        write_reference_set(&mut w, index, set);
    }
    w.put_bit(false); // long_term_ref_pics_present_flag
    w.put_bit(true); // sps_temporal_mvp_enabled_flag
    w.put_bit(false); // strong_intra_smoothing_enabled_flag
    // Without colour signalling there is no VUI at all, which keeps
    // default streams unchanged.
    let vui = info.signal.colour.is_signalled() || info.signal.timing != TimingSignal::Nominal;
    w.put_bit(vui); // vui_parameters_present_flag
    if vui {
        write_vui(&mut w, info.signal, info.frame_rate);
    }
    w.put_bit(false); // sps_extension_present_flag
    w.trailing_bits();
    w.into_bytes()
}

/// `vui_parameters()` (HEVC §E.2.1) carrying the video signal type and
/// timing, each when configured.
fn write_vui(w: &mut BitWriter, signal: StreamSignal, frame_rate: FrameRate) {
    w.put_bit(false); // aspect_ratio_info_present_flag
    w.put_bit(false); // overscan_info_present_flag
    write_signal_type(w, signal.colour);
    w.put_bit(false); // chroma_loc_info_present_flag
    w.put_bit(false); // neutral_chroma_indication_flag
    w.put_bit(false); // field_seq_flag
    w.put_bit(false); // frame_field_info_present_flag
    w.put_bit(false); // default_display_window_flag
    let timing = signal.timing != TimingSignal::Nominal;
    w.put_bit(timing); // vui_timing_info_present_flag
    if timing {
        // One clock tick per nominal picture.
        w.put_bits(frame_rate.den, 32); // vui_num_units_in_tick
        w.put_bits(frame_rate.num, 32); // vui_time_scale
        // POC counts frames (no reordering), so at a constant rate it is
        // proportional to output time, one tick per POC step.
        let proportional = signal.timing == TimingSignal::Constant;
        w.put_bit(proportional); // vui_poc_proportional_to_timing_flag
        if proportional {
            w.put_ue(0); // vui_num_ticks_poc_diff_one_minus1
        }
        w.put_bit(false); // vui_hrd_parameters_present_flag
    }
    w.put_bit(false); // bitstream_restriction_flag
}

/// `video_signal_type_present_flag` and, when set, the video signal type.
fn write_signal_type(w: &mut BitWriter, colour: ColourInfo) {
    w.put_bit(colour.is_signalled()); // video_signal_type_present_flag
    if colour.is_signalled() {
        write_video_signal_type(w, colour);
    }
}

/// `pic_parameter_set_rbsp()` (HEVC §7.3.2.3). The deblocking filter is on
/// with the stream's offsets.
#[must_use]
pub fn pps(info: &StreamInfo) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put_ue(0); // pps_pic_parameter_set_id
    w.put_ue(0); // pps_seq_parameter_set_id
    w.put_bit(false); // dependent_slice_segments_enabled_flag
    w.put_bit(false); // output_flag_present_flag
    w.put_bits(0, 3); // num_extra_slice_header_bits
    w.put_bit(SIGN_DATA_HIDING); // sign_data_hiding_enabled_flag
    w.put_bit(false); // cabac_init_present_flag
    w.put_ue(0); // num_ref_idx_l0_default_active_minus1
    w.put_ue(0); // num_ref_idx_l1_default_active_minus1
    w.put_se(info.init_qp - 26); // init_qp_minus26
    w.put_bit(false); // constrained_intra_pred_flag
    w.put_bit(false); // transform_skip_enabled_flag
    w.put_bit(info.block_qp); // cu_qp_delta_enabled_flag
    if info.block_qp {
        w.put_ue(QUANT_GROUP_DEPTH); // diff_cu_qp_delta_depth
    }
    w.put_se(info.chroma_qp_offset); // pps_cb_qp_offset
    w.put_se(info.chroma_qp_offset); // pps_cr_qp_offset
    w.put_bit(info.chroma_policy.is_some()); // pps_slice_chroma_qp_offsets_present_flag
    w.put_bit(false); // weighted_pred_flag
    w.put_bit(false); // weighted_bipred_flag
    w.put_bit(false); // transquant_bypass_enabled_flag
    w.put_bit(false); // tiles_enabled_flag
    w.put_bit(false); // entropy_coding_sync_enabled_flag
    w.put_bit(false); // pps_loop_filter_across_slices_enabled_flag
    w.put_bit(true); // deblocking_filter_control_present_flag
    w.put_bit(info.deblock_override); //   deblocking_filter_override_enabled_flag
    w.put_bit(false); //   pps_deblocking_filter_disabled_flag
    w.put_se(info.deblock_offsets_div2.0); //   pps_beta_offset_div2
    w.put_se(info.deblock_offsets_div2.1); //   pps_tc_offset_div2
    w.put_bit(false); // pps_scaling_list_data_present_flag
    w.put_bit(false); // lists_modification_present_flag
    w.put_ue(0); // log2_parallel_merge_level_minus2
    w.put_bit(false); // slice_segment_header_extension_present_flag
    w.put_bit(false); // pps_extension_present_flag
    w.trailing_bits();
    w.into_bytes()
}

/// A slice's quantiser syntax (HEVC §7.3.6.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceQp {
    /// `slice_qp_delta`: the slice QP minus the PPS initial QP.
    pub delta: i32,
    /// `slice_cb_qp_offset` = `slice_cr_qp_offset`, present when the PPS
    /// sets `pps_slice_chroma_qp_offsets_present_flag`.
    pub chroma_offset: Option<i32>,
}

impl SliceQp {
    fn write(self, w: &mut BitWriter) {
        w.put_se(self.delta); // slice_qp_delta
        if let Some(offset) = self.chroma_offset {
            w.put_se(offset); // slice_cb_qp_offset
            w.put_se(offset); // slice_cr_qp_offset
        }
    }
}

/// The POC syntax of a CRA picture's slice (absent for IDR pictures).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CraPoc {
    /// `slice_pic_order_cnt_lsb`.
    pub lsb: u32,
    /// `num_short_term_ref_pic_sets` of the SPS: the explicit (empty)
    /// reference set follows them.
    pub sps_sets: usize,
}

/// `slice_segment_header()` (HEVC §7.3.6.1) of the single I slice of an
/// IRAP picture, ending in `byte_alignment()`: an IDR picture (`cra`
/// `None`), or a CRA picture with its POC and an empty reference picture
/// set, which releases every earlier picture as an IDR does (§8.3.2).
///
/// A key frame after the first is a CRA picture so POC keeps counting
/// across it: an IDR restarts POC at 0, and a receiver that lost the IDR
/// would take the next pictures' references (POC deltas from 0) for
/// pictures it still holds from before. With CRA the lost key's POC is
/// missing from the receiver's DPB, and the pictures after it are refused
/// ("no reference picture", §8.3.2). Without leading pictures a CRA starts
/// decoding as an IDR does.
pub fn irap_slice_header(
    w: &mut BitWriter,
    cra: Option<CraPoc>,
    qp: SliceQp,
    sao: (bool, bool),
    deblocking: SliceDeblocking,
) {
    w.put_bit(true); // first_slice_segment_in_pic_flag
    w.put_bit(false); // no_output_of_prior_pics_flag (IRAP picture)
    w.put_ue(0); // slice_pic_parameter_set_id
    w.put_ue(SLICE_TYPE_I); // slice_type
    if let Some(cra) = cra {
        w.put_bits(cra.lsb, POC_LSB_BITS); // slice_pic_order_cnt_lsb
        w.put_bit(false); // short_term_ref_pic_set_sps_flag
        write_reference_set(w, cra.sps_sets, &[]); // st_ref_pic_set(): empty
        // No long-term syntax (long_term_ref_pics_present_flag 0); an I
        // slice uses no temporal vectors.
        w.put_bit(false); // slice_temporal_mvp_enabled_flag
    }
    w.put_bit(sao.0); // slice_sao_luma_flag
    w.put_bit(sao.1); // slice_sao_chroma_flag
    qp.write(w);
    deblocking.write(w);
    w.trailing_bits(); // byte_alignment(): a one bit, then zero bits
}

/// A slice's deblocking control (§7.3.6.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceDeblocking {
    /// The PPS does not enable overrides: nothing is coded and the PPS
    /// control applies.
    Fixed,
    /// `deblocking_filter_override_flag` 0: the PPS control applies.
    Inherited,
    /// `deblocking_filter_override_flag` 1 with the slice's own control.
    Override {
        /// `slice_deblocking_filter_disabled_flag`.
        disabled: bool,
        /// `slice_beta_offset_div2` (coded unless disabled).
        beta_offset_div2: i32,
        /// `slice_tc_offset_div2` (coded unless disabled).
        tc_offset_div2: i32,
    },
}

impl SliceDeblocking {
    /// The override syntax. `slice_loop_filter_across_slices_enabled_flag`
    /// that may follow is absent: the PPS disables filtering across slices.
    fn write(self, w: &mut BitWriter) {
        match self {
            Self::Fixed => {}
            Self::Inherited => w.put_bit(false), // deblocking_filter_override_flag
            Self::Override {
                disabled,
                beta_offset_div2,
                tc_offset_div2,
            } => {
                w.put_bit(true); // deblocking_filter_override_flag
                w.put_bit(disabled); // slice_deblocking_filter_disabled_flag
                if !disabled {
                    w.put_se(beta_offset_div2);
                    w.put_se(tc_offset_div2);
                }
            }
        }
    }

    /// The `(beta, tc)` offsets the filter uses, given the PPS ones, or
    /// `None` when the filter is off for the slice.
    #[must_use]
    pub fn offsets_div2(self, pps: (i32, i32)) -> Option<(i32, i32)> {
        match self {
            Self::Fixed | Self::Inherited => Some(pps),
            Self::Override { disabled: true, .. } => None,
            Self::Override {
                beta_offset_div2,
                tc_offset_div2,
                ..
            } => Some((beta_offset_div2, tc_offset_div2)),
        }
    }
}

/// `st_ref_pic_set(index)` (§7.3.7): negative pictures only, explicit.
fn write_reference_set(w: &mut BitWriter, index: usize, set: &[(u32, bool)]) {
    if index != 0 {
        w.put_bit(false); // inter_ref_pic_set_prediction_flag
    }
    w.put_ue(set.len() as u32); // num_negative_pics
    w.put_ue(0); // num_positive_pics
    let mut previous = 0;
    for &(distance, used) in set {
        w.put_ue(distance - previous - 1); // delta_poc_s0_minus1
        w.put_bit(used); // used_by_curr_pic_s0_flag
        previous = distance;
    }
}

/// Bits of `short_term_ref_pic_set_idx`: `Ceil(Log2(num_short_term_ref_pic_sets))`.
#[must_use]
pub fn reference_set_index_bits(sets: usize) -> u32 {
    sets.next_power_of_two().trailing_zeros()
}

/// `slice_segment_header()` of the single P slice of a trailing picture
/// with POC LSBs `poc_lsb`, predicted through `references`.
pub fn p_slice_header(
    w: &mut BitWriter,
    poc_lsb: u32,
    references: SliceReferenceSet<'_>,
    qp: SliceQp,
    sao: (bool, bool),
    temporal_mvp: bool,
    merge_candidates: usize,
    deblocking: SliceDeblocking,
) {
    w.put_bit(true); // first_slice_segment_in_pic_flag
    w.put_ue(0); // slice_pic_parameter_set_id
    w.put_ue(SLICE_TYPE_P); // slice_type
    w.put_bits(poc_lsb, POC_LSB_BITS); // slice_pic_order_cnt_lsb
    match references {
        SliceReferenceSet::Sps { index, count } => {
            w.put_bit(true); // short_term_ref_pic_set_sps_flag
            // short_term_ref_pic_set_idx (no bits with a single set).
            w.put_bits(index as u32, reference_set_index_bits(count));
        }
        SliceReferenceSet::Explicit { entries, sps_count } => {
            w.put_bit(false); // short_term_ref_pic_set_sps_flag
            // st_ref_pic_set(num_short_term_ref_pic_sets) (§7.3.7).
            write_reference_set(w, sps_count, entries);
        }
    }
    // slice_temporal_mvp_enabled_flag. The collocated picture is the one
    // reference (collocated_ref_idx is inferred 0 and not coded).
    w.put_bit(temporal_mvp);
    w.put_bit(sao.0); // slice_sao_luma_flag
    w.put_bit(sao.1); // slice_sao_chroma_flag
    w.put_bit(false); // num_ref_idx_active_override_flag (one reference)
    // five_minus_max_num_merge_cand (§7.4.7.1)
    w.put_ue((crate::mvpred::MAX_MERGE_CANDIDATES - merge_candidates) as u32);
    qp.write(w);
    deblocking.write(w);
    w.trailing_bits(); // byte_alignment()
}

/// `slice_type` value of a P slice (HEVC Table 7-7).
const SLICE_TYPE_P: u32 = 1;
/// `slice_type` value of an I slice (HEVC Table 7-7).
const SLICE_TYPE_I: u32 = 2;

/// DPB sizing of every sub-layer: the current picture and the largest
/// reference set (all sub-layers alike: an upper bound for each), no
/// reordering.
fn write_dpb_sizes(w: &mut BitWriter, info: &StreamInfo) {
    for _ in 0..info.temporal_layers {
        w.put_ue(info.dpb_pictures() - 1); // max_dec_pic_buffering_minus1
        w.put_ue(0); // max_num_reorder_pics
        w.put_ue(0); // max_latency_increase_plus1
    }
}

/// The lowest level whose picture-size and sample-rate limits hold
/// (HEVC Table A.8), as `general_level_idc`.
#[must_use]
pub fn level_idc(width: usize, height: usize, fps: f64) -> u32 {
    // (level_idc, MaxLumaPs, MaxLumaSr)
    const LEVELS: [(u32, u64, u64); 13] = [
        (30, 36_864, 552_960),
        (60, 122_880, 3_686_400),
        (63, 245_760, 7_372_800),
        (90, 552_960, 16_588_800),
        (93, 983_040, 33_177_600),
        (120, 2_228_224, 66_846_720),
        (123, 2_228_224, 133_693_440),
        (150, 8_912_896, 267_386_880),
        (153, 8_912_896, 534_773_760),
        (156, 8_912_896, 1_069_547_520),
        (180, 35_651_584, 1_069_547_520),
        (183, 35_651_584, 2_139_095_040),
        (186, 35_651_584, 4_278_190_080),
    ];
    let ps = (width * height) as u64;
    let sr = (ps as f64 * fps).ceil() as u64;
    let max_dim = width.max(height) as u64;
    LEVELS
        .iter()
        .find(|&&(_, max_ps, max_sr)| {
            // §A.4.1: each dimension at most sqrt(8 * MaxLumaPs).
            ps <= max_ps && sr <= max_sr && max_dim * max_dim <= 8 * max_ps
        })
        .map_or(LEVELS[LEVELS.len() - 1].0, |l| l.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_for_common_sizes() {
        assert_eq!(level_idc(1280, 720, 30.0), 93); // 3.1
        assert_eq!(level_idc(1920, 1080, 30.0), 120); // 4
        assert_eq!(level_idc(1920, 1080, 60.0), 123); // 4.1
        assert_eq!(level_idc(640, 360, 30.0), 63); // 2.1
    }

    #[test]
    fn nal_header_bytes() {
        assert_eq!(nal_header(NalType::Vps, 0), [0x40, 0x01]);
        assert_eq!(nal_header(NalType::IdrWRadl, 0), [0x26, 0x01]);
        assert_eq!(nal_header(NalType::TrailN, 2), [0x00, 0x03]);
    }

    #[test]
    fn reference_set_index_widths() {
        assert_eq!(reference_set_index_bits(1), 0);
        assert_eq!(reference_set_index_bits(2), 1);
        assert_eq!(reference_set_index_bits(4), 2);
    }
}
