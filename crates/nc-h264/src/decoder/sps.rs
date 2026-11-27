//! Sequence parameter sets (§7.3.2.1.1) with their VUI (§E.1.1): what the
//! High, High 10, High 4:2:2 and High 4:4:4 Predictive profiles (and their
//! Intra profiles, CAVLC 4:4:4 Intra) accept: 8- to 14-bit 4:0:0, 4:2:0,
//! 4:2:2 or 4:4:4, separately coded colour planes and lossless transform
//! bypass. Sequences that allow fields are accepted; their field and MBAFF
//! pictures are refused by the slice header. Anything else is rejected as
//! unsupported rather than misdecoded.

use nc_common::{BitDepth, BitDepths, ChromaFormat, ColourDescription, ColourInfo};

use super::chroma::{self, ChromaArray};
use super::reader::Reader;
use super::scaling::{self, ScalingMatrix};
use super::structure::Structure;
use super::{DecodeError, Result};

/// Range of `bit_depth_luma_minus8` / `bit_depth_chroma_minus8`
/// (§7.4.2.1.1: 14-bit samples at most, High 4:4:4 Predictive).
const MAX_BIT_DEPTH_MINUS8: u32 = 6;
/// `seq_parameter_set_id` range (§7.4.2.1.1).
pub(super) const MAX_SPS_ID: u32 = 31;
/// `max_num_ref_frames` and DPB frames are at most 16 (§A.3.1).
pub(super) const MAX_DPB_FRAMES: usize = 16;
/// Largest picture accepted, in macroblocks: the `MaxFS` of levels 6 to 6.2
/// (Table A-1, 8192x4320), the largest any level allows. Picture buffers
/// are sized from the SPS, so this only bounds hostile input.
pub(super) const MAX_FRAME_MBS: usize = 139_264;
/// Luma samples per macroblock side.
const MB_SIZE: usize = 16;
/// Largest picture side in macroblocks (`sqrt(8 * MaxFS)`, §A.3.1 item f).
const MAX_SIDE_MBS: usize = 1_055;
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
const CHROMA_422: u32 = 2;
const CHROMA_444: u32 = 3;
/// The 8x8 scaling lists of an SPS: Y only, or Y, Cb and Cr in 4:4:4
/// (§7.3.2.1.1: `( chroma_format_idc != 3 ) ? 8 : 12` lists in all).
pub(super) const LISTS_8X8_Y: usize = 2;
pub(super) const LISTS_8X8_YUV: usize = 6;

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
    /// `ChromaArrayType`: the chroma syntax of the macroblock layer.
    pub chroma: ChromaArray,
    /// `separate_colour_plane_flag`: three colour planes, each coded as a
    /// monochrome picture (`colour_plane_id`).
    pub separate_planes: bool,
    /// The decoded pictures' format.
    pub format: ChromaFormat,
    /// `qpprime_y_zero_transform_bypass_flag`: macroblocks with `QP′Y` 0
    /// are lossless (§8.5.15).
    pub transform_bypass: bool,
    /// `bit_depth_luma_minus8 + 8` and the chroma one (the luma one for
    /// every plane when they are coded separately: `ChromaArrayType` 0
    /// leaves the chroma depth unused, §7.4.2.1.1).
    pub depths: BitDepths,
    /// `seq_scaling_matrix_present_flag`: PPS lists fall back by rule B.
    pub scaling_present: bool,
    /// The resolved sequence scaling matrix (flat when absent).
    pub scaling: ScalingMatrix,
    pub log2_max_frame_num: u32,
    pub poc: PocType,
    pub max_num_ref_frames: usize,
    pub gaps_in_frame_num_allowed: bool,
    pub width_mbs: usize,
    /// Frame height in macroblocks.
    pub height_mbs: usize,
    /// `frame_mbs_only_flag`.
    pub frame_mbs_only: bool,
    /// `mb_adaptive_frame_field_flag`.
    pub mbaff: bool,
    /// `direct_8x8_inference_flag`: direct prediction reads the co-located
    /// corner 4x4 blocks (§8.4.1.2.1).
    pub direct_8x8_inference: bool,
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

    /// Macroblocks of a frame (`FrameHeightInMbs` rows).
    pub fn mbs(&self) -> usize {
        self.width_mbs * self.height_mbs
    }

    /// `PicHeightInMbs` of a picture of `structure`: a field has half the
    /// frame's rows (eq. 7-25).
    pub fn picture_height_mbs(&self, structure: Structure) -> usize {
        if structure.is_field() {
            self.height_mbs / 2
        } else {
            self.height_mbs
        }
    }

    /// `PicSizeInMbs` of a picture of `structure` (eq. 7-26).
    pub fn picture_mbs(&self, structure: Structure) -> usize {
        self.width_mbs * self.picture_height_mbs(structure)
    }

    /// Colour planes coded separately: 3 with `separate_colour_plane_flag`,
    /// else 1 (all components in each macroblock).
    pub fn coded_planes(&self) -> usize {
        if self.separate_planes { 3 } else { 1 }
    }

    /// The visible (cropped) size.
    pub fn visible_size(&self) -> (usize, usize) {
        (
            self.width_mbs * MB_SIZE - self.crop.left - self.crop.right,
            self.height_mbs * MB_SIZE - self.crop.top - self.crop.bottom,
        )
    }

    /// Frames that may precede a frame in decoding order but follow it in
    /// output order: the VUI's `max_num_reorder_frames`; else none for
    /// `pic_order_cnt_type` 2, whose output order is the decoding order
    /// (§8.2.1.3), so its pictures leave at once (openh264 signals no VUI
    /// restriction); else the DPB size (§C.4.5.3).
    pub fn reorder_frames(&self) -> usize {
        match (self.max_num_reorder_frames, &self.poc) {
            (Some(frames), _) => frames as usize,
            (None, PocType::FrameNum) => 0,
            (None, _) => self.dpb_frames(),
        }
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

/// The fields only the High profiles' SPSs carry (§7.3.2.1.1).
struct HighFields {
    chroma_format_idc: u32,
    separate_planes: bool,
    depths: BitDepths,
    transform_bypass: bool,
    scaling_present: bool,
    scaling: ScalingMatrix,
}

impl HighFields {
    /// What they are without them: 8-bit 4:2:0 (§7.4.2.1.1).
    const BASELINE: Self = Self {
        chroma_format_idc: CHROMA_420,
        separate_planes: false,
        depths: BitDepths::EIGHT,
        transform_bypass: false,
        scaling_present: false,
        scaling: ScalingMatrix::FLAT,
    };
}

/// `chroma_format_idc` to `seq_scaling_matrix_present_flag` and its lists.
fn parse_high_fields(r: &mut Reader<'_>) -> Result<HighFields> {
    let chroma_format_idc = r.ue_max(CHROMA_444, "chroma_format_idc")?;
    let separate_planes = chroma_format_idc == CHROMA_444 && r.flag()?;
    let mut depth = || -> Result<BitDepth> {
        let bits = r.ue_max(MAX_BIT_DEPTH_MINUS8, "bit_depth_minus8")? + 8;
        BitDepth::new(bits).ok_or(DecodeError::Malformed("bit depth"))
    };
    let luma = depth()?;
    let chroma = depth()?;
    let transform_bypass = r.flag()?;
    let scaling_present = r.flag()?;
    let mut scaling = ScalingMatrix::FLAT;
    if scaling_present {
        let lists_8x8 = if chroma_format_idc == CHROMA_444 {
            LISTS_8X8_YUV
        } else {
            LISTS_8X8_Y
        };
        let parsed = scaling::parse_matrix(r, lists_8x8)?;
        scaling = scaling::resolve(&parsed, None);
    }
    Ok(HighFields {
        chroma_format_idc,
        separate_planes,
        depths: BitDepths {
            luma,
            chroma: if separate_planes { luma } else { chroma },
        },
        transform_bypass,
        scaling_present,
        scaling,
    })
}

/// Parse an SPS RBSP (after the NAL header).
pub(super) fn parse(data: &[u8]) -> Result<Sps> {
    let mut r = Reader::new(data);
    let profile_idc = r.bits(8)?;
    let constraint_flags = r.bits(8)?;
    let level_idc = r.bits(8)?;
    let id = r.ue_max(MAX_SPS_ID, "seq_parameter_set_id")?;
    let high = if HIGH_PROFILES.contains(&profile_idc) {
        parse_high_fields(&mut r)?
    } else {
        HighFields::BASELINE
    };
    let HighFields {
        chroma_format_idc,
        separate_planes,
        depths,
        transform_bypass,
        scaling_present,
        scaling,
    } = high;
    let monochrome = chroma_format_idc == CHROMA_400;
    let chroma = match chroma_format_idc {
        CHROMA_420 => ChromaArray::Yuv420,
        CHROMA_422 => ChromaArray::Yuv422,
        CHROMA_444 if !separate_planes => ChromaArray::Yuv444,
        _ => ChromaArray::Mono,
    };
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
    // Field-capable sequences code frame heights in pairs of macroblock
    // rows (eq. 7-18); their field and MBAFF pictures are refused per slice.
    let frame_mbs_only = r.flag()?;
    let mbaff = !frame_mbs_only && r.flag()?;
    let height_mbs = if frame_mbs_only {
        height_mbs
    } else {
        2 * height_mbs
    };
    if height_mbs > MAX_SIDE_MBS || width_mbs * height_mbs > MAX_FRAME_MBS {
        return Err(DecodeError::Unsupported("H264 picture size"));
    }
    let direct_8x8_inference = r.flag()?;
    let crop = parse_crop(&mut r, chroma, frame_mbs_only, width_mbs, height_mbs)?;
    let mut sps = Sps {
        id,
        profile_idc,
        level_idc,
        constraint_set3: constraint_flags & 0x10 != 0,
        monochrome,
        chroma,
        separate_planes,
        format: chroma::picture_format(chroma_format_idc),
        transform_bypass,
        depths,
        scaling_present,
        scaling,
        log2_max_frame_num,
        poc,
        max_num_ref_frames,
        gaps_in_frame_num_allowed,
        width_mbs,
        height_mbs,
        frame_mbs_only,
        mbaff,
        direct_8x8_inference,
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

/// `frame_cropping_flag` and offsets, in luma samples (eqs. 7-19..7-22:
/// `CropUnitX` `SubWidthC`, 1 for `ChromaArrayType` 0; `CropUnitY`
/// `SubHeightC` (or 1) times `2 − frame_mbs_only_flag`).
fn parse_crop(
    r: &mut Reader<'_>,
    chroma: ChromaArray,
    frame_mbs_only: bool,
    width_mbs: usize,
    height_mbs: usize,
) -> Result<Crop> {
    if !r.flag()? {
        return Ok(Crop::default());
    }
    let (sub_x, sub_y) = if chroma.is_mono() {
        (0, 0)
    } else {
        chroma.subsampling()
    };
    let unit_x = 1 << sub_x;
    let unit_y = if frame_mbs_only {
        1 << sub_y
    } else {
        2 << sub_y
    };
    // An offset beyond the coded size is invalid anyway (eqs. 7-19..7-22);
    // bounding it first keeps the products and sums below from overflowing
    // a 32-bit `usize` (WASM).
    let largest = (width_mbs.max(height_mbs) * MB_SIZE) as u32;
    let mut offset = |unit: usize| -> Result<usize> {
        Ok(r.ue_max(largest, "frame_crop_offset")? as usize * unit)
    };
    let crop = Crop {
        left: offset(unit_x)?,
        right: offset(unit_x)?,
        top: offset(unit_y)?,
        bottom: offset(unit_y)?,
    };
    if crop.left + crop.right >= width_mbs * MB_SIZE
        || crop.top + crop.bottom >= height_mbs * MB_SIZE
    {
        return Err(DecodeError::Malformed("frame cropping"));
    }
    // A 4:0:0 window may be odd in size and origin (CropUnitX 1): it is
    // handed out as I420 with flat chroma of half its size, rounded up
    // (`ShownPicture::window`).
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
        baseline_sps((1, 1), Some(offset))
    }

    /// A Baseline SPS of `size` macroblocks, cropped by `crop` on each side.
    fn baseline_sps((width_mbs, height_mbs): (u32, u32), crop: Option<u32>) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.put_bits(66, 8); // profile_idc: Baseline
        w.put_bits(0, 8); // constraint flags
        w.put_bits(30, 8); // level_idc
        w.put_ue(0); // seq_parameter_set_id
        w.put_ue(0); // log2_max_frame_num_minus4
        w.put_ue(2); // pic_order_cnt_type
        w.put_ue(1); // max_num_ref_frames
        w.put_bit(false); // gaps_in_frame_num_value_allowed_flag
        w.put_ue(width_mbs - 1); // pic_width_in_mbs_minus1
        w.put_ue(height_mbs - 1); // pic_height_in_map_units_minus1
        w.put_bit(true); // frame_mbs_only_flag
        w.put_bit(true); // direct_8x8_inference_flag
        w.put_bit(crop.is_some()); // frame_cropping_flag
        for _ in 0..4 * usize::from(crop.is_some()) {
            w.put_ue(crop.unwrap_or_default());
        }
        w.put_bit(false); // vui_parameters_present_flag
        w.put_bit(true); // rbsp_stop_one_bit
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

    /// Levels 6 to 6.2 allow 8192x4320 (Table A-1: `MaxFS` 139 264, a side
    /// at most `sqrt(8 * MaxFS)` = 1055 macroblocks); larger is refused.
    #[test]
    fn level_6_picture_sizes_are_accepted() {
        let sps = parse(&baseline_sps((512, 270), None)).unwrap();
        assert_eq!(sps.mbs(), 138_240);
        // Above `MaxFS`, or one side above the bound.
        for too_big in [(512, 273), (1_056, 1)] {
            assert_eq!(
                parse(&baseline_sps(too_big, None)).err(),
                Some(DecodeError::Unsupported("H264 picture size"))
            );
        }
    }
}
