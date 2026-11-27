//! `vui_parameters()` (H.265 §E.2.1) with its `hrd_parameters()` (§E.2.2)
//! and `sub_layer_hrd_parameters()` (§E.2.3). Decoding keeps only the
//! colour description; the rest is read so that the SPS extensions after
//! the VUI (`sps_range_extension()`, §7.3.2.2.2) can be reached.

use nc_common::ColourInfo;
use nc_common::colour::read_video_signal_type;
use nc_common::decode::{BitReader, DecodeError};

/// `aspect_ratio_idc` that carries an explicit sample aspect ratio.
const EXTENDED_SAR: u32 = 255;
/// `cpb_cnt_minus1` limit (§E.3.2).
const MAX_CPB_COUNT_MINUS1: u32 = 31;

/// `vui_parameters()` of an SPS with `max_sub_layers` sub-layers: its
/// colour description.
pub(super) fn parse(
    r: &mut BitReader<'_>,
    max_sub_layers: usize,
) -> Result<ColourInfo, DecodeError> {
    if r.read_bit()? && r.read_bits(8)? == EXTENDED_SAR {
        r.skip_bits(32)?; // sar_width, sar_height
    }
    if r.read_bit()? {
        r.read_bit()?; // overscan_appropriate_flag
    }
    let colour = if r.read_bit()? {
        read_video_signal_type(r)?
    } else {
        ColourInfo::default()
    };
    if r.read_bit()? {
        // chroma_sample_loc_type_top_field, _bottom_field
        r.read_ue()?;
        r.read_ue()?;
    }
    // neutral_chroma_indication_flag, field_seq_flag,
    // frame_field_info_present_flag
    r.skip_bits(3)?;
    if r.read_bit()? {
        // default display window offsets
        for _ in 0..4 {
            r.read_ue()?;
        }
    }
    if r.read_bit()? {
        // vui_num_units_in_tick, vui_time_scale
        r.skip_bits(64)?;
        if r.read_bit()? {
            r.read_ue()?; // vui_num_ticks_poc_diff_one_minus1
        }
        if r.read_bit()? {
            hrd_parameters(r, true, max_sub_layers)?;
        }
    }
    if r.read_bit()? {
        // tiles_fixed_structure_flag, motion_vectors_over_pic_boundaries_flag,
        // restricted_ref_pic_lists_flag
        r.skip_bits(3)?;
        // min_spatial_segmentation_idc, max_bytes_per_pic_denom,
        // max_bits_per_min_cu_denom, log2_max_mv_length_horizontal / vertical
        for _ in 0..5 {
            r.read_ue()?;
        }
    }
    Ok(colour)
}

/// `hrd_parameters(commonInfPresentFlag, maxNumSubLayersMinus1)` (§E.2.2),
/// read past.
fn hrd_parameters(
    r: &mut BitReader<'_>,
    common: bool,
    max_sub_layers: usize,
) -> Result<(), DecodeError> {
    let (mut nal, mut vcl, mut sub_pic) = (false, false, false);
    if common {
        nal = r.read_bit()?;
        vcl = r.read_bit()?;
        if nal || vcl {
            sub_pic = r.read_bit()?;
            if sub_pic {
                // tick_divisor_minus2 (8), du_cpb_removal_delay_increment_length_minus1
                // (5), sub_pic_cpb_params_in_pic_timing_sei_flag (1),
                // dpb_output_delay_du_length_minus1 (5)
                r.skip_bits(19)?;
            }
            r.skip_bits(8)?; // bit_rate_scale, cpb_size_scale
            if sub_pic {
                r.skip_bits(4)?; // cpb_size_du_scale
            }
            // initial_cpb_removal_delay_length_minus1,
            // au_cpb_removal_delay_length_minus1, dpb_output_delay_length_minus1
            r.skip_bits(15)?;
        }
    }
    for _ in 0..max_sub_layers {
        let fixed_general = r.read_bit()?;
        let fixed_within_cvs = fixed_general || r.read_bit()?;
        let mut low_delay = false;
        if fixed_within_cvs {
            r.read_ue()?; // elemental_duration_in_tc_minus1
        } else {
            low_delay = r.read_bit()?;
        }
        let cpb_count = if low_delay {
            1
        } else {
            r.read_ue_max(MAX_CPB_COUNT_MINUS1, "cpb_cnt_minus1")? + 1
        };
        for _ in 0..u32::from(nal) + u32::from(vcl) {
            sub_layer_hrd_parameters(r, cpb_count, sub_pic)?;
        }
    }
    Ok(())
}

/// `sub_layer_hrd_parameters()` (§E.2.3), read past.
fn sub_layer_hrd_parameters(
    r: &mut BitReader<'_>,
    cpb_count: u32,
    sub_pic: bool,
) -> Result<(), DecodeError> {
    for _ in 0..cpb_count {
        // bit_rate_value_minus1, cpb_size_value_minus1, and with sub-picture
        // parameters cpb_size_du_value_minus1, bit_rate_du_value_minus1
        for _ in 0..if sub_pic { 4 } else { 2 } {
            r.read_ue()?;
        }
        r.read_bit()?; // cbr_flag
    }
    Ok(())
}
