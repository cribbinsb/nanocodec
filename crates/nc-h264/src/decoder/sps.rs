//! Sequence parameter sets (§7.3.2.1.1) with their VUI (§E.1.1), limited to
//! what a Constrained High decoder must accept: 8-bit 4:2:0 or 4:0:0,
//! progressive (`frame_mbs_only_flag`), no lossless transform bypass.
//! Anything else is rejected as unsupported rather than misdecoded.

use nc_common::{ColourDescription, ColourInfo};

use super::reader::Reader;
use super::scaling::{self, ScalingMatrix};
use super::{DecodeError, Result};

/// `seq_parameter_set_id` range (§7.4.2.1.1).
pub(super) const MAX_SPS_ID: u32 = 31;
/// `max_num_ref_frames` and DPB frames are at most 16 (§A.3.1).
pub(super) const MAX_DPB_FRAMES: usize = 16;
/// Largest picture accepted, in macroblocks: level 5.2's `MaxFS` (Table A-1),
/// which bounds memory for hostile input as well.
const MAX_FRAME_MBS: usize = 36_864;
/// Luma samples per macroblock side.
const MB_SIZE: usize = 16;
/// Largest picture side in macroblocks (`sqrt(8 * MaxFS)`, §A.3.1 item f).
const MAX_SIDE_MBS: usize = 543;
/// `log2_max_frame_num_minus4` and `log2_max_pic_order_cnt_lsb_minus4` are 0..=12.
const MAX_LOG2_MINUS4: u32 = 12;
/// `num_ref_frames_in_pic_order_cnt_cycle` is 0..=255.
const MAX_POC_CYCLE: u32 = 255;
/// `cpb_cnt_minus1` is 0..=31 (§E.2.2).
const MAX_CPB_COUNT_MINUS1: u32 = 31;
/// `aspect_ratio_idc` of an explicit `sar_width` / `sar_height` (Table E-1).
const EXTENDED_SAR: u32 = 255;
/// Profiles whose SPS carries `chroma_format_idc` and the bit depths (§7.3.2.1.1).
const HIGH_PROFILES: [u32; 13] = [100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135];
/// `chroma_format_idc` values (Table 6-1).
const CHROMA_400: u32 = 0;
const CHROMA_420: u32 = 1;
const CHROMA_444: u32 = 3;

/// Picture order count derivation of the SPS (§8.2.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum PocType {
    /// Type 0: `pic_order_cnt_lsb` is coded with this many bits.
    Lsb { log2_max_lsb: u32 },
    /// Type 1: expected increments per `frame_num` cycle.
    Cycle {
        delta_pic_order_always_zero: bool,
        offset_for_non_ref_pic: i32,
        offset_for_top_to_bottom_field: i32,
        offsets_for_ref_frame: Vec<i32>,
    },
    /// Type 2: output order is decoding order.
    FrameNum,
}

/// Frame cropping in luma samples (§7.4.2.1.1: offsets times `CropUnitX`/Y).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) struct Crop {
    pub left: usize,
    pub right: usize,
    pub top: usize,
    pub bottom: usize,
}

/// The fields of an SPS that decoding uses.
#[derive(Debug, Clone, PartialEq)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "the flags mirror independent SPS syntax elements"
)]
pub(super) struct Sps {
    pub id: u32,
    pub profile_idc: u32,
    pub level_idc: u32,
    /// `constraint_set3_flag` (level 1b with Baseline/Main, §A.3.1).
    pub constraint_set3: bool,
    /// `chroma_format_idc` 0.
    pub monochrome: bool,
    /// `seq_scaling_matrix_present_flag`: PPS lists fall back by rule B.
    pub scaling_present: bool,
    /// The resolved sequence scaling matrix (flat when absent).
    pub scaling: ScalingMatrix,
    pub log2_max_frame_num: u32,
    pub poc: PocType,
    pub max_num_ref_frames: usize,
    pub gaps_in_frame_num_allowed: bool,
    pub width_mbs: usize,
    pub height_mbs: usize,
    pub crop: Crop,
    pub colour: ColourInfo,
    /// `max_num_reorder_frames` of the VUI bitstream restriction.
    pub max_num_reorder_frames: Option<u32>,
    /// `max_dec_frame_buffering` of the VUI bitstream restriction.
    pub max_dec_frame_buffering: Option<u32>,
}

impl Sps {
    /// `MaxFrameNum` (eq. 7-10).
    pub fn max_frame_num(&self) -> u32 {
        1 << self.log2_max_frame_num
    }

    pub fn mbs(&self) -> usize {
        self.width_mbs * self.height_mbs
    }

    /// The visible (cropped) size.
    pub fn visible_size(&self) -> (usize, usize) {
        (
            self.width_mbs * MB_SIZE - self.crop.left - self.crop.right,
            self.height_mbs * MB_SIZE - self.crop.top - self.crop.bottom,
        )
    }

    /// The DPB size in frames (§A.3.1 item h: `MaxDpbMbs` divided by the
    /// frame size in macroblocks, at most 16), or the VUI's
    /// `max_dec_frame_buffering`.
    pub fn dpb_frames(&self) -> usize {
        let level_frames = (max_dpb_mbs(self.level_idc, self.constraint_set3) / self.mbs())
            .clamp(1, MAX_DPB_FRAMES);
        self.max_dec_frame_buffering
            .map_or(level_frames, |frames| {
                (frames as usize).clamp(1, MAX_DPB_FRAMES)
            })
            .max(self.max_num_ref_frames)
    }
}

/// `MaxDpbMbs` of Table A-1 by `level_idc` (level 1b is `level_idc` 11 with
/// `constraint_set3_flag`, or 9).
fn max_dpb_mbs(level_idc: u32, constraint_set3: bool) -> usize {
    match level_idc {
        9 | 10 => 396,
        11 if constraint_set3 => 396,
        11 => 900,
        12 | 13 | 20 => 2_376,
        21 => 4_752,
        22 | 30 => 8_100,
        31 => 18_000,
        32 => 20_480,
        40 | 41 => 32_768,
        42 => 34_816,
        50 => 110_400,
        51 | 52 => 184_320,
        _ => 696_320,
    }
}

/// Parse an SPS RBSP (after the NAL header).
pub(super) fn parse(data: &[u8]) -> Result<Sps> {
    let mut r = Reader::new(data);
    let profile_idc = r.bits(8)?;
    let constraint_flags = r.bits(8)?;
    let level_idc = r.bits(8)?;
    let id = r.ue_max(MAX_SPS_ID, "seq_parameter_set_id")?;
    let mut monochrome = false;
    let mut scaling_present = false;
    let mut scaling = ScalingMatrix::FLAT;
    if HIGH_PROFILES.contains(&profile_idc) {
        let chroma_format_idc = r.ue_max(CHROMA_444, "chroma_format_idc")?;
        if chroma_format_idc == CHROMA_444 && r.flag()? {
            return Err(DecodeError::Unsupported("H264 separate colour planes"));
        }
        if chroma_format_idc != CHROMA_420 && chroma_format_idc != CHROMA_400 {
            return Err(DecodeError::Unsupported("H264 chroma format above 4:2:0"));
        }
        monochrome = chroma_format_idc == CHROMA_400;
        if r.ue()? != 0 || r.ue()? != 0 {
            return Err(DecodeError::Unsupported("H264 bit depth above 8"));
        }
        if r.flag()? {
            return Err(DecodeError::Unsupported("H264 lossless transform bypass"));
        }
        scaling_present = r.flag()?;
        if scaling_present {
            let parsed = scaling::parse_matrix(&mut r, scaling::LISTS_8X8)?;
            scaling = scaling::resolve(&parsed, None);
        }
    }
    let log2_max_frame_num = r.ue_max(MAX_LOG2_MINUS4, "log2_max_frame_num_minus4")? + 4;
    let poc = parse_poc(&mut r)?;
    let max_num_ref_frames = r.ue_max(MAX_DPB_FRAMES as u32, "max_num_ref_frames")? as usize;
    let gaps_in_frame_num_allowed = r.flag()?;
    let width_mbs = r.ue()? as usize + 1;
    let height_mbs = r.ue()? as usize + 1;
    if width_mbs > MAX_SIDE_MBS
        || height_mbs > MAX_SIDE_MBS
        || width_mbs * height_mbs > MAX_FRAME_MBS
    {
        return Err(DecodeError::Unsupported("H264 picture size"));
    }
    if !r.flag()? {
        return Err(DecodeError::Unsupported("interlaced H264"));
    }
    r.flag()?; // direct_8x8_inference_flag: B slices only
    let crop = parse_crop(&mut r, monochrome, width_mbs, height_mbs)?;
    let mut sps = Sps {
        id,
        profile_idc,
        level_idc,
        constraint_set3: constraint_flags & 0x10 != 0,
        monochrome,
        scaling_present,
        scaling,
        log2_max_frame_num,
        poc,
        max_num_ref_frames,
        gaps_in_frame_num_allowed,
        width_mbs,
        height_mbs,
        crop,
        colour: ColourInfo::default(),
        max_num_reorder_frames: None,
        max_dec_frame_buffering: None,
    };
    if r.flag()? {
        parse_vui(&mut r, &mut sps)?;
    }
    Ok(sps)
}

fn parse_poc(r: &mut Reader<'_>) -> Result<PocType> {
    Ok(match r.ue()? {
        0 => PocType::Lsb {
            log2_max_lsb: r.ue_max(MAX_LOG2_MINUS4, "log2_max_pic_order_cnt_lsb_minus4")? + 4,
        },
        1 => {
            let delta_pic_order_always_zero = r.flag()?;
            let offset_for_non_ref_pic = r.se()?;
            let offset_for_top_to_bottom_field = r.se()?;
            let cycle = r.ue_max(MAX_POC_CYCLE, "num_ref_frames_in_pic_order_cnt_cycle")?;
            let offsets_for_ref_frame = (0..cycle).map(|_| r.se()).collect::<Result<_>>()?;
            PocType::Cycle {
                delta_pic_order_always_zero,
                offset_for_non_ref_pic,
                offset_for_top_to_bottom_field,
                offsets_for_ref_frame,
            }
        }
        2 => PocType::FrameNum,
        _ => return Err(DecodeError::Malformed("pic_order_cnt_type")),
    })
}

/// `frame_cropping_flag` and offsets, in luma samples (eqs. 7-19..7-22 with
/// `frame_mbs_only_flag` 1: `CropUnitX` = `CropUnitY` = 2 for 4:2:0, 1 for 4:0:0).
fn parse_crop(
    r: &mut Reader<'_>,
    monochrome: bool,
    width_mbs: usize,
    height_mbs: usize,
) -> Result<Crop> {
    if !r.flag()? {
        return Ok(Crop::default());
    }
    let unit = if monochrome { 1 } else { 2 };
    // An offset beyond the coded size is invalid anyway (eqs. 7-19..7-22);
    // bounding it first keeps the products and sums below from overflowing
    // a 32-bit `usize` (WASM).
    let largest = (width_mbs.max(height_mbs) * MB_SIZE) as u32;
    let mut offset =
        || -> Result<usize> { Ok(r.ue_max(largest, "frame_crop_offset")? as usize * unit) };
    let crop = Crop {
        left: offset()?,
        right: offset()?,
        top: offset()?,
        bottom: offset()?,
    };
    if crop.left + crop.right >= width_mbs * MB_SIZE
        || crop.top + crop.bottom >= height_mbs * MB_SIZE
    {
        return Err(DecodeError::Malformed("frame cropping"));
    }
    // Pictures are handed out as 4:2:0, which needs an even visible size.
    if !(width_mbs * MB_SIZE - crop.left - crop.right).is_multiple_of(2)
        || !(height_mbs * MB_SIZE - crop.top - crop.bottom).is_multiple_of(2)
    {
        return Err(DecodeError::Unsupported("odd cropped 4:0:0 size"));
    }
    Ok(crop)
}

/// `vui_parameters()` (§E.1.1): colour signalling and the bitstream
/// restriction are kept, the rest is parsed and dropped.
fn parse_vui(r: &mut Reader<'_>, sps: &mut Sps) -> Result<()> {
    if r.flag()? && r.bits(8)? == EXTENDED_SAR {
        r.bits(16)?; // sar_width
        r.bits(16)?; // sar_height
    }
    if r.flag()? {
        r.flag()?; // overscan_appropriate_flag
    }
    if r.flag()? {
        r.bits(3)?; // video_format
        sps.colour.full_range = r.flag()?;
        if r.flag()? {
            sps.colour.description = Some(ColourDescription {
                primaries: r.bits(8)? as u8,
                transfer: r.bits(8)? as u8,
                matrix: r.bits(8)? as u8,
            });
        }
    }
    if r.flag()? {
        r.ue()?; // chroma_sample_loc_type_top_field
        r.ue()?; // chroma_sample_loc_type_bottom_field
    }
    if r.flag()? {
        r.bits(32)?; // num_units_in_tick
        r.bits(32)?; // time_scale
        r.flag()?; // fixed_frame_rate_flag
    }
    let nal_hrd = r.flag()?;
    if nal_hrd {
        parse_hrd(r)?;
    }
    let vcl_hrd = r.flag()?;
    if vcl_hrd {
        parse_hrd(r)?;
    }
    if nal_hrd || vcl_hrd {
        r.flag()?; // low_delay_hrd_flag
    }
    r.flag()?; // pic_struct_present_flag
    if r.flag()? {
        r.flag()?; // motion_vectors_over_pic_boundaries_flag
        r.ue()?; // max_bytes_per_pic_denom
        r.ue()?; // max_bits_per_mb_denom
        r.ue()?; // log2_max_mv_length_horizontal
        r.ue()?; // log2_max_mv_length_vertical
        sps.max_num_reorder_frames = Some(r.ue()?);
        sps.max_dec_frame_buffering = Some(r.ue()?);
    }
    Ok(())
}

/// `hrd_parameters()` (§E.1.2), parsed and dropped.
fn parse_hrd(r: &mut Reader<'_>) -> Result<()> {
    let count = r.ue_max(MAX_CPB_COUNT_MINUS1, "cpb_cnt_minus1")? + 1;
    r.bits(4)?; // bit_rate_scale
    r.bits(4)?; // cpb_size_scale
    for _ in 0..count {
        r.ue()?; // bit_rate_value_minus1
        r.ue()?; // cpb_size_value_minus1
        r.flag()?; // cbr_flag
    }
    // initial_cpb_removal_delay_length_minus1, cpb_removal_delay_length_minus1,
    // dpb_output_delay_length_minus1, time_offset_length.
    r.bits(20)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_common::BitWriter;

    /// A Baseline SPS of one macroblock whose crop offsets are `offset`.
    fn sps_with_crop(offset: u32) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.put_bits(66, 8); // profile_idc: Baseline
        w.put_bits(0, 8); // constraint flags
        w.put_bits(30, 8); // level_idc
        w.put_ue(0); // seq_parameter_set_id
        w.put_ue(0); // log2_max_frame_num_minus4
        w.put_ue(2); // pic_order_cnt_type
        w.put_ue(1); // max_num_ref_frames
        w.put_bit(false); // gaps_in_frame_num_value_allowed_flag
        w.put_ue(0); // pic_width_in_mbs_minus1
        w.put_ue(0); // pic_height_in_map_units_minus1
        w.put_bit(true); // frame_mbs_only_flag
        w.put_bit(true); // direct_8x8_inference_flag
        w.put_bit(true); // frame_cropping_flag
        for _ in 0..4 {
            w.put_ue(offset);
        }
        while !w.is_byte_aligned() {
            w.put_bit(false);
        }
        w.into_bytes()
    }

    /// Offsets near 2^31 used to be multiplied and summed before the size
    /// check, which overflows a 32-bit `usize` (WASM); they are now
    /// rejected as they are read.
    #[test]
    fn crop_offsets_are_bounded_before_arithmetic() {
        assert_eq!(
            parse(&sps_with_crop(0x7fff_ffff)).err(),
            Some(DecodeError::Malformed("frame_crop_offset"))
        );
    }
}
