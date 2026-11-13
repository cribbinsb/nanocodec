//! Parameter sets and slice headers (H.264 §7.3.2.1, §7.3.2.2, §7.3.3).
//!
//! One SPS and one PPS, both with the stream's parameter set id (0), are
//! re-sent before every IDR picture with identical content (constant ids keep
//! CMAF segmenters happy, see `eval/README.md` "Methodology: Anchors"). A resolution change
//! switches to the other id (0 ↔ 1): the new SPS changes content, and
//! decoders that re-activate a parameter set only when its id changes (JM)
//! would otherwise keep the old picture size.

use nc_common::BitWriter;
use nc_common::colour::write_video_signal_type;
use nc_encoder::{FrameRate, StreamSignal, TimingSignal};

use crate::deblock::DeblockControl;
use crate::references::{
    LONG_TERM_EXTRA_FRAMES, LONG_TERM_FRAME_INDICES, ListModification, Marking, PictureReferences,
};
use crate::tables::LEVELS;

/// High profile (H.264 §A.2.4): CABAC plus 8x8 transforms.
const PROFILE_IDC_HIGH: u32 = 100;
/// `chroma_format_idc` for 4:2:0.
const CHROMA_FORMAT_IDC_420: u32 = 1;
/// `log2_max_frame_num_minus4`: `frame_num` is 8 bits. A receiver tells a
/// loss by the gap in `frame_num` (§8.2.5.2); with 8 bits a gap goes
/// unnoticed only when exactly a multiple of 256 reference frames are lost,
/// or when an IDR picture is lost right after a picture with `frame_num` 0
/// (1 in 256), for 4 bits more per slice than the smallest field.
const LOG2_MAX_FRAME_NUM_MINUS4: u32 = 4;
const FRAME_NUM_BITS: u32 = LOG2_MAX_FRAME_NUM_MINUS4 + 4;
/// `MaxFrameNum` (eq. 7-10): `frame_num` counts modulo this.
pub const MAX_FRAME_NUM: u32 = 1 << FRAME_NUM_BITS;
/// `pic_order_cnt_type` 2: output order = decoding order, nothing in the
/// slice header (valid because frames are never reordered).
const PIC_ORDER_CNT_TYPE: u32 = 2;
/// `pic_init_qp_minus26`: slice QPs are sent as deltas from 26.
const PIC_INIT_QP: i32 = 26;
/// `slice_type` 7: I, and all slices of the picture are I (Table 7-6).
const SLICE_TYPE_I_ALL: u32 = 7;
/// `slice_type` 5: P, and all slices of the picture are P (Table 7-6).
const SLICE_TYPE_P_ALL: u32 = 5;
/// `modification_of_pic_nums_idc` 0: subtract from the predicted picture
/// number (Table 7-7).
const MODIFICATION_SUBTRACT: u32 = 0;
/// `modification_of_pic_nums_idc` 2: a long-term picture by
/// `long_term_pic_num`.
const MODIFICATION_LONG_TERM: u32 = 2;
/// `modification_of_pic_nums_idc` 3: end of the modification loop.
const MODIFICATION_END: u32 = 3;
/// `memory_management_control_operation` 0: end of the loop (Table 7-9).
const MMCO_END: u32 = 0;
/// `memory_management_control_operation` 1: a short-term picture becomes
/// unused for reference.
const MMCO_RELEASE_SHORT_TERM: u32 = 1;
/// `memory_management_control_operation` 2: a long-term picture becomes
/// unused for reference.
const MMCO_RELEASE_LONG_TERM: u32 = 2;
/// `memory_management_control_operation` 4: `max_long_term_frame_idx_plus1`.
const MMCO_MAX_LONG_TERM_FRAME_IDX: u32 = 4;
/// `memory_management_control_operation` 6: the current picture becomes a
/// long-term picture.
const MMCO_CURRENT_TO_LONG_TERM: u32 = 6;
/// Chroma cropping unit for 4:2:0 progressive video (§7.4.2.1.1, eq. 7-19/7-20).
const CROP_UNIT: usize = 2;

/// Stream-level parameters shared by the SPS, PPS and slice headers.
#[derive(Debug, Clone, Copy)]
pub struct StreamParams {
    /// Visible picture size.
    pub width: usize,
    /// Visible picture height.
    pub height: usize,
    /// Picture size in macroblocks.
    pub mb_cols: usize,
    /// Picture height in macroblocks.
    pub mb_rows: usize,
    /// Signalled in the VUI timing information.
    pub frame_rate: FrameRate,
    /// `level_idc` (Table A-1).
    pub level_idc: u8,
    /// `max_num_ref_frames`: the engine's reference slot count (§7.4.2.1.1).
    pub max_num_ref_frames: u32,
    /// `gaps_in_frame_num_value_allowed_flag`: set for three-layer streams
    /// so that their T0 layer alone is a conforming stream, and with a
    /// long-term reference so that a stream missing lost frames is one
    /// (`references.rs`).
    pub frame_num_gaps: bool,
    /// Colour and timing signalling written in the VUI.
    pub signal: StreamSignal,
    /// `fixed_frame_rate_flag`: not with variable-rate input, nor when the
    /// encoder may hold frames (`nc_encoder::holds_frames`).
    pub fixed_frame_rate: bool,
    /// `seq_parameter_set_id` and `pic_parameter_set_id` (0, or 1 after an
    /// odd number of resolution changes; see the module notes).
    pub parameter_set_id: u32,
}

impl StreamParams {
    /// Parameters for a `width` x `height` stream at `frame_rate`.
    pub fn new(width: usize, height: usize, frame_rate: FrameRate) -> Self {
        let mb_cols = width.div_ceil(16);
        let mb_rows = height.div_ceil(16);
        Self {
            width,
            height,
            mb_cols,
            mb_rows,
            frame_rate,
            level_idc: level_for(mb_cols, mb_rows, frame_rate),
            max_num_ref_frames: 1,
            frame_num_gaps: false,
            signal: StreamSignal::default(),
            fixed_frame_rate: true,
            parameter_set_id: 0,
        }
    }

    /// These parameters for a structure of `temporal_layers` layers keeping
    /// `reference_slots` references.
    #[must_use]
    pub fn with_structure(self, temporal_layers: u8, reference_slots: usize) -> Self {
        Self {
            max_num_ref_frames: reference_slots as u32,
            frame_num_gaps: temporal_layers > 2,
            ..self
        }
    }

    /// These parameters with a long-term reference beside the structure's
    /// short-term ones (`references.rs`).
    #[must_use]
    pub fn with_long_term_reference(self) -> Self {
        Self {
            max_num_ref_frames: self.max_num_ref_frames + LONG_TERM_EXTRA_FRAMES as u32,
            frame_num_gaps: true,
            ..self
        }
    }

    /// These parameters with `signal` written in the VUI, for an encoder
    /// that leaves out held frames if `holds_frames`.
    #[must_use]
    pub fn with_signal(self, signal: StreamSignal, holds_frames: bool) -> Self {
        Self {
            signal,
            fixed_frame_rate: signal.timing != TimingSignal::Variable && !holds_frames,
            ..self
        }
    }

    /// Number of macroblocks in a picture.
    pub fn mb_count(&self) -> usize {
        self.mb_cols * self.mb_rows
    }
}

/// The lowest level whose frame size and macroblock rate limits hold
/// (Table A-1); level 6 if none does (bring-up: never refuse a size).
fn level_for(mb_cols: usize, mb_rows: usize, frame_rate: FrameRate) -> u8 {
    let frame_mbs = (mb_cols * mb_rows) as u64;
    let mbs_per_second =
        (frame_mbs * u64::from(frame_rate.num)).div_ceil(u64::from(frame_rate.den));
    let longest_side = mb_cols.max(mb_rows) as u64;
    LEVELS
        .iter()
        .find(|l| {
            let max_fs = u64::from(l.max_frame_mbs);
            // §A.3.1 h/i: neither dimension may exceed sqrt(8 * MaxFS).
            frame_mbs <= max_fs
                && longest_side * longest_side <= 8 * max_fs
                && mbs_per_second <= u64::from(l.max_mbs_per_second)
        })
        .map_or(LEVELS[LEVELS.len() - 1].level_idc, |l| l.level_idc)
}

/// `seq_parameter_set_rbsp()` (§7.3.2.1.1), with VUI timing and bitstream
/// restrictions saying no frame is ever reordered.
pub fn sequence_parameter_set(p: &StreamParams) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put_bits(PROFILE_IDC_HIGH, 8);
    w.put_bits(0, 8); // constraint_set0..5_flag, reserved_zero_2bits
    w.put_bits(u32::from(p.level_idc), 8);
    w.put_ue(p.parameter_set_id); // seq_parameter_set_id
    w.put_ue(CHROMA_FORMAT_IDC_420);
    w.put_ue(0); // bit_depth_luma_minus8
    w.put_ue(0); // bit_depth_chroma_minus8
    w.put_bit(false); // qpprime_y_zero_transform_bypass_flag
    w.put_bit(false); // seq_scaling_matrix_present_flag
    w.put_ue(LOG2_MAX_FRAME_NUM_MINUS4);
    w.put_ue(PIC_ORDER_CNT_TYPE);
    w.put_ue(p.max_num_ref_frames);
    w.put_bit(p.frame_num_gaps); // gaps_in_frame_num_value_allowed_flag
    w.put_ue(p.mb_cols as u32 - 1); // pic_width_in_mbs_minus1
    w.put_ue(p.mb_rows as u32 - 1); // pic_height_in_map_units_minus1
    w.put_bit(true); // frame_mbs_only_flag
    w.put_bit(true); // direct_8x8_inference_flag
    let crop_right = (p.mb_cols * 16 - p.width) / CROP_UNIT;
    let crop_bottom = (p.mb_rows * 16 - p.height) / CROP_UNIT;
    let cropping = crop_right != 0 || crop_bottom != 0;
    w.put_bit(cropping); // frame_cropping_flag
    if cropping {
        w.put_ue(0); // frame_crop_left_offset
        w.put_ue(crop_right as u32);
        w.put_ue(0); // frame_crop_top_offset
        w.put_ue(crop_bottom as u32);
    }
    w.put_bit(true); // vui_parameters_present_flag
    write_vui(&mut w, p);
    w.trailing_bits();
    w.into_bytes()
}

/// `vui_parameters()` (§E.1.1): colour signalling when configured, timing
/// information and bitstream restrictions.
fn write_vui(w: &mut BitWriter, p: &StreamParams) {
    let (frame_rate, max_num_ref_frames) = (p.frame_rate, p.max_num_ref_frames);
    w.put_bit(false); // aspect_ratio_info_present_flag
    w.put_bit(false); // overscan_info_present_flag
    let colour = p.signal.colour;
    w.put_bit(colour.is_signalled()); // video_signal_type_present_flag
    if colour.is_signalled() {
        write_video_signal_type(w, colour);
    }
    w.put_bit(false); // chroma_loc_info_present_flag
    w.put_bit(true); // timing_info_present_flag
    // A frame lasts two ticks (§E.2.1: frame rate = time_scale / (2 * num_units_in_tick)).
    w.put_bits(frame_rate.den, 32); // num_units_in_tick
    w.put_bits(frame_rate.num * 2, 32); // time_scale
    // Variable-rate camera input, or a stream missing held frames, has no
    // fixed picture interval; the nominal rate then only defines the tick.
    w.put_bit(p.fixed_frame_rate); // fixed_frame_rate_flag
    w.put_bit(false); // nal_hrd_parameters_present_flag
    w.put_bit(false); // vcl_hrd_parameters_present_flag
    w.put_bit(false); // pic_struct_present_flag
    w.put_bit(true); // bitstream_restriction_flag
    w.put_bit(true); // motion_vectors_over_pic_boundaries_flag
    w.put_ue(0); // max_bytes_per_pic_denom: no limit
    w.put_ue(0); // max_bits_per_mb_denom: no limit
    w.put_ue(16); // log2_max_mv_length_horizontal
    w.put_ue(16); // log2_max_mv_length_vertical
    w.put_ue(0); // max_num_reorder_frames: zero-latency output
    w.put_ue(max_num_ref_frames); // max_dec_frame_buffering
}

/// `pic_parameter_set_rbsp()` (§7.3.2.2): CABAC, no 8x8 transform yet, no
/// scaling matrices, deblocking control in the slice header. Both chroma
/// components use `chroma_qp_offset` (§7.4.2.2).
pub fn picture_parameter_set(
    pic_parameter_set_id: u32,
    seq_parameter_set_id: u32,
    chroma_qp_offset: i8,
    entropy: Entropy,
) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put_ue(pic_parameter_set_id); // pic_parameter_set_id
    w.put_ue(seq_parameter_set_id); // seq_parameter_set_id
    w.put_bit(entropy == Entropy::Cabac); // entropy_coding_mode_flag
    w.put_bit(false); // bottom_field_pic_order_in_frame_present_flag
    w.put_ue(0); // num_slice_groups_minus1
    w.put_ue(0); // num_ref_idx_l0_default_active_minus1
    w.put_ue(0); // num_ref_idx_l1_default_active_minus1
    w.put_bit(false); // weighted_pred_flag
    w.put_bits(0, 2); // weighted_bipred_idc
    w.put_se(PIC_INIT_QP - 26); // pic_init_qp_minus26
    w.put_se(0); // pic_init_qs_minus26
    w.put_se(i32::from(chroma_qp_offset)); // chroma_qp_index_offset
    w.put_bit(true); // deblocking_filter_control_present_flag
    w.put_bit(false); // constrained_intra_pred_flag
    w.put_bit(false); // redundant_pic_cnt_present_flag
    w.put_bit(true); // transform_8x8_mode_flag: inter macroblocks may use 8x8
    w.put_bit(false); // pic_scaling_matrix_present_flag
    w.put_se(i32::from(chroma_qp_offset)); // second_chroma_qp_index_offset
    w.trailing_bits();
    w.into_bytes()
}

/// The entropy coder of the slice data (`entropy_coding_mode_flag`,
/// §7.4.2.2); one per stream, as the encoder writes one PPS per chroma
/// offset and never switches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Entropy {
    /// Exp-Golomb and CAVLC codes (§9.1, §9.2).
    Cavlc,
    /// CABAC (§9.3).
    Cabac,
}

/// `slice_header()` (§7.3.3) of the single I slice of an IDR picture. For
/// CABAC it is followed by `cabac_alignment_one_bit`s (§7.3.4), leaving the
/// writer byte aligned; CAVLC slice data follows the header directly. A
/// [`Marking::LongTerm`] IDR picture becomes the long-term reference.
pub fn idr_slice_header(
    parameter_set_id: u32,
    idr_pic_id: u32,
    marking: &Marking,
    slice_qp: u8,
    deblock: DeblockControl,
    entropy: Entropy,
) -> BitWriter {
    let mut w = BitWriter::new();
    w.put_ue(0); // first_mb_in_slice
    w.put_ue(SLICE_TYPE_I_ALL);
    w.put_ue(parameter_set_id); // pic_parameter_set_id
    w.put_bits(0, FRAME_NUM_BITS); // frame_num: 0 for IDR pictures
    w.put_ue(idr_pic_id);
    // dec_ref_pic_marking() for IDR (§7.3.3.3).
    w.put_bit(false); // no_output_of_prior_pics_flag
    w.put_bit(matches!(marking, Marking::LongTerm { .. })); // long_term_reference_flag
    w.put_se(i32::from(slice_qp) - PIC_INIT_QP); // slice_qp_delta
    write_deblocking_control(&mut w, deblock);
    if entropy == Entropy::Cabac {
        align_for_cabac(&mut w);
    }
    w
}

/// `slice_header()` of the single P slice of a picture with one active
/// reference (the PPS default), moved to index 0 by a list modification if
/// needed (see `references.rs`); sliding-window marking for reference
/// pictures unless they become long-term; the deblocking control, and with
/// CABAC (`cabac_init_idc` given) the init index and the alignment bits.
pub fn p_slice_header(
    parameter_set_id: u32,
    refs: &PictureReferences,
    slice_qp: u8,
    cabac_init_idc: Option<u8>,
    deblock: DeblockControl,
) -> BitWriter {
    debug_assert!(refs.frame_num < MAX_FRAME_NUM);
    let mut w = BitWriter::new();
    w.put_ue(0); // first_mb_in_slice
    w.put_ue(SLICE_TYPE_P_ALL);
    w.put_ue(parameter_set_id); // pic_parameter_set_id
    w.put_bits(refs.frame_num, FRAME_NUM_BITS);
    // num_ref_idx_active_override_flag: the PPS default is one reference.
    w.put_bit(refs.active_references > 1);
    if refs.active_references > 1 {
        w.put_ue(u32::from(refs.active_references) - 1); // num_ref_idx_l0_active_minus1
    }
    // ref_pic_list_modification() (§7.3.3.1).
    w.put_bit(refs.list_modification.is_some()); // ref_pic_list_modification_flag_l0
    if let Some(first) = refs.list_modification {
        for modification in core::iter::once(first).chain(refs.second_modification) {
            match modification {
                ListModification::Subtract(abs_diff_pic_num_minus1) => {
                    w.put_ue(MODIFICATION_SUBTRACT);
                    w.put_ue(abs_diff_pic_num_minus1);
                }
                ListModification::LongTerm(long_term_pic_num) => {
                    w.put_ue(MODIFICATION_LONG_TERM);
                    w.put_ue(long_term_pic_num);
                }
            }
        }
        w.put_ue(MODIFICATION_END);
    }
    // dec_ref_pic_marking() (§7.3.3.3) is present only for reference pictures.
    if refs.is_reference() {
        write_adaptive_marking(&mut w, &refs.marking);
    }
    if let Some(init_idc) = cabac_init_idc {
        w.put_ue(u32::from(init_idc));
    }
    w.put_se(i32::from(slice_qp) - PIC_INIT_QP); // slice_qp_delta
    write_deblocking_control(&mut w, deblock);
    if cabac_init_idc.is_some() {
        align_for_cabac(&mut w);
    }
    w
}

/// `dec_ref_pic_marking()` of a non-IDR reference picture (§7.3.3.3): the
/// sliding window, or memory management operations making the picture
/// long-term (§8.2.5.4.1, §8.2.5.4.6).
fn write_adaptive_marking(w: &mut BitWriter, marking: &Marking) {
    match marking {
        Marking::ShortTerm => w.put_bit(false), // adaptive_ref_pic_marking_mode_flag
        Marking::LongTerm {
            release,
            replaced,
            idx,
        } => {
            w.put_bit(true); // adaptive_ref_pic_marking_mode_flag
            for &difference_of_pic_nums_minus1 in release {
                w.put_ue(MMCO_RELEASE_SHORT_TERM);
                w.put_ue(difference_of_pic_nums_minus1);
            }
            if let Some(long_term_pic_num) = replaced {
                w.put_ue(MMCO_RELEASE_LONG_TERM);
                w.put_ue(*long_term_pic_num);
            }
            w.put_ue(MMCO_MAX_LONG_TERM_FRAME_IDX);
            w.put_ue(LONG_TERM_FRAME_INDICES); // max_long_term_frame_idx_plus1
            w.put_ue(MMCO_CURRENT_TO_LONG_TERM);
            w.put_ue(*idx); // long_term_frame_idx
            w.put_ue(MMCO_END);
        }
    }
}

/// Deblocking control of the slice header (present because the PPS sets
/// `deblocking_filter_control_present_flag`).
fn write_deblocking_control(w: &mut BitWriter, deblock: DeblockControl) {
    w.put_ue(u32::from(deblock.disable_idc));
    if deblock.offsets_coded() {
        w.put_se(i32::from(deblock.alpha_offset_div2));
        w.put_se(i32::from(deblock.beta_offset_div2));
    }
}

fn align_for_cabac(w: &mut BitWriter) {
    while !w.is_byte_aligned() {
        w.put_bit(true); // cabac_alignment_one_bit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FPS30: FrameRate = FrameRate { num: 30, den: 1 };

    #[test]
    fn levels_for_common_sizes() {
        assert_eq!(StreamParams::new(640, 360, FPS30).level_idc, 30);
        assert_eq!(StreamParams::new(1280, 720, FPS30).level_idc, 31);
        assert_eq!(StreamParams::new(1920, 1080, FPS30).level_idc, 40);
    }

    #[test]
    fn cropping_only_when_needed() {
        let exact = StreamParams::new(1280, 720, FPS30);
        let cropped = StreamParams::new(1920, 1080, FPS30);
        assert_eq!((cropped.mb_cols, cropped.mb_rows), (120, 68));
        assert_ne!(
            sequence_parameter_set(&exact),
            sequence_parameter_set(&cropped)
        );
    }
}
