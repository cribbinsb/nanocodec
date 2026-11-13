//! Slice headers (§7.3.3) of I and P slices, with their
//! `ref_pic_list_modification()` (§7.3.3.1), `pred_weight_table()`
//! (§7.3.3.2) and `dec_ref_pic_marking()` (§7.3.3.3).

use super::dpb::{Marking, Mmco, Modification};
use super::pps::{MAX_PPS_ID, MAX_REF_IDX_ACTIVE_MINUS1, Pps};
use super::reader::Reader;
use super::sps::{PocType, Sps};
use super::{DecodeError, Result};
use crate::deblock::{DEBLOCK_IDC_NO_SLICE_EDGES, DEBLOCK_OFFSET_DIV2_MAX, DeblockControl};

/// `slice_type % 5` (Table 7-6).
const SLICE_TYPE_P: u32 = 0;
const SLICE_TYPE_B: u32 = 1;
const SLICE_TYPE_I: u32 = 2;
/// Largest `slice_type` (types 5..=9 repeat 0..=4 for the whole picture).
const MAX_SLICE_TYPE: u32 = 9;
/// Number of distinct slice types.
const SLICE_TYPES: u32 = 5;
/// `idr_pic_id` range (§7.4.3).
const MAX_IDR_PIC_ID: u32 = 65_535;
/// `cabac_init_idc` range.
const MAX_CABAC_INIT_IDC: u32 = 2;
/// `luma_log2_weight_denom` / `chroma_log2_weight_denom` range.
const MAX_LOG2_WEIGHT_DENOM: u32 = 7;
/// Range of explicit weights and offsets (8-bit video, §7.4.3.2).
const MIN_WEIGHT: i32 = -128;
const MAX_WEIGHT: i32 = 127;
/// Most reference list modification commands: one per list entry, plus
/// slack for repeated pictures (the list has at most 32 entries).
const MAX_MODIFICATIONS: usize = 33;
/// Most memory management operations per picture: each one releases or
/// marks one of at most 16 frames, plus the bookkeeping operations.
const MAX_MMCOS: usize = 66;

/// The decodable slice types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SliceType {
    I,
    P,
}

/// One reference's explicit weighted prediction (§7.4.3.2); the defaults
/// (`2^denom`, 0) when its flag is 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Weight {
    pub weight: i32,
    pub offset: i32,
}

/// `pred_weight_table()` of list 0.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct WeightTable {
    pub luma_log2_denom: u32,
    pub chroma_log2_denom: u32,
    /// Per reference index: luma, Cb, Cr.
    pub weights: Vec<[Weight; 3]>,
}

impl WeightTable {
    /// Whether every weight is the default, so weighting changes nothing
    /// (`((x * 2^d + 2^(d-1)) >> d) + 0 == x`).
    pub fn is_identity(&self, ref_idx: usize) -> bool {
        let denoms = [
            self.luma_log2_denom,
            self.chroma_log2_denom,
            self.chroma_log2_denom,
        ];
        self.weights[ref_idx]
            .iter()
            .zip(denoms)
            .all(|(w, denom)| w.weight == 1 << denom && w.offset == 0)
    }
}

/// The slice header fields decoding uses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SliceHeader {
    pub first_mb: usize,
    pub slice_type: SliceType,
    pub pps_id: u32,
    pub frame_num: u32,
    pub idr: bool,
    /// `nal_ref_idc != 0`.
    pub reference: bool,
    pub idr_pic_id: u32,
    pub pic_order_cnt_lsb: u32,
    pub delta_pic_order_cnt_bottom: i32,
    pub delta_pic_order_cnt: [i32; 2],
    pub num_ref_idx_active: usize,
    pub modifications: Vec<Modification>,
    /// Explicit weighted prediction (`weighted_pred_flag` and a P slice).
    pub weights: Option<WeightTable>,
    pub marking: Marking,
    pub cabac_init_idc: u8,
    pub qp: i32,
    pub deblock: DeblockControl,
}

/// Parse the first fields of a slice header: `first_mb_in_slice` and the
/// `pic_parameter_set_id` that selects the parameter sets for the rest.
pub(super) fn parse_pps_id(data: &[u8]) -> Result<u32> {
    let mut r = Reader::new(data);
    r.ue()?;
    r.ue_max(MAX_SLICE_TYPE, "slice_type")?;
    r.ue_max(MAX_PPS_ID, "pic_parameter_set_id")
}

/// Parse a whole slice header; leaves `r` at the first bit of
/// `slice_data()` (after `cabac_alignment_one_bit`s).
pub(super) fn parse(
    r: &mut Reader<'_>,
    sps: &Sps,
    pps: &Pps,
    idr: bool,
    nal_ref_idc: u32,
) -> Result<SliceHeader> {
    let first_mb = r.ue()? as usize;
    if first_mb >= sps.mbs() {
        return Err(DecodeError::Malformed("first_mb_in_slice"));
    }
    let slice_type = match r.ue_max(MAX_SLICE_TYPE, "slice_type")? % SLICE_TYPES {
        SLICE_TYPE_P if !idr => SliceType::P,
        SLICE_TYPE_I => SliceType::I,
        SLICE_TYPE_B => return Err(DecodeError::Unsupported("H264 B slices")),
        SLICE_TYPE_P => return Err(DecodeError::Malformed("P slice in IDR picture")),
        _ => return Err(DecodeError::Unsupported("H264 SP/SI slices")),
    };
    let pps_id = r.ue_max(MAX_PPS_ID, "pic_parameter_set_id")?;
    let frame_num = r.bits(sps.log2_max_frame_num)?;
    if idr && frame_num != 0 {
        return Err(DecodeError::Malformed("IDR frame_num"));
    }
    let idr_pic_id = if idr {
        r.ue_max(MAX_IDR_PIC_ID, "idr_pic_id")?
    } else {
        0
    };
    let (pic_order_cnt_lsb, delta_pic_order_cnt_bottom, delta_pic_order_cnt) =
        parse_poc_fields(r, sps, pps)?;
    let mut num_ref_idx_active = pps.num_ref_idx_l0_default_active as usize;
    let mut modifications = Vec::new();
    if slice_type == SliceType::P {
        if r.flag()? {
            num_ref_idx_active =
                r.ue_max(MAX_REF_IDX_ACTIVE_MINUS1, "num_ref_idx_l0_active_minus1")? as usize + 1;
        }
        if r.flag()? {
            modifications = parse_modifications(r)?;
        }
    }
    let weights = (slice_type == SliceType::P && pps.weighted_pred)
        .then(|| parse_weights(r, num_ref_idx_active, sps.monochrome))
        .transpose()?;
    let marking = if nal_ref_idc == 0 {
        Marking::NonReference
    } else if idr {
        r.flag()?; // no_output_of_prior_pics_flag: every picture is output
        Marking::Idr {
            long_term: r.flag()?,
        }
    } else if r.flag()? {
        Marking::Adaptive(parse_mmcos(r)?)
    } else {
        Marking::SlidingWindow
    };
    let cabac_init_idc = if pps.cabac && slice_type == SliceType::P {
        r.ue_max(MAX_CABAC_INIT_IDC, "cabac_init_idc")? as u8
    } else {
        0
    };
    let qp = pps.pic_init_qp + r.se()?;
    if !(0..=51).contains(&qp) {
        return Err(DecodeError::Malformed("slice QP"));
    }
    let deblock = if pps.deblocking_filter_control_present {
        parse_deblock(r)?
    } else {
        DeblockControl::DEFAULT
    };
    if pps.cabac {
        r.align(true)?;
    }
    Ok(SliceHeader {
        first_mb,
        slice_type,
        pps_id,
        frame_num,
        idr,
        reference: nal_ref_idc != 0,
        idr_pic_id,
        pic_order_cnt_lsb,
        delta_pic_order_cnt_bottom,
        delta_pic_order_cnt,
        num_ref_idx_active,
        modifications,
        weights,
        marking,
        cabac_init_idc,
        qp,
        deblock,
    })
}

/// The picture order count fields: `pic_order_cnt_lsb`,
/// `delta_pic_order_cnt_bottom` and `delta_pic_order_cnt[0..2]`.
fn parse_poc_fields(r: &mut Reader<'_>, sps: &Sps, pps: &Pps) -> Result<(u32, i32, [i32; 2])> {
    let (mut pic_order_cnt_lsb, mut delta_pic_order_cnt_bottom, mut delta_pic_order_cnt) =
        (0, 0, [0; 2]);
    match &sps.poc {
        PocType::Lsb { log2_max_lsb } => {
            pic_order_cnt_lsb = r.bits(*log2_max_lsb)?;
            if pps.bottom_field_pic_order_in_frame_present {
                delta_pic_order_cnt_bottom = r.se()?;
            }
        }
        PocType::Cycle {
            delta_pic_order_always_zero: false,
            ..
        } => {
            delta_pic_order_cnt[0] = r.se()?;
            if pps.bottom_field_pic_order_in_frame_present {
                delta_pic_order_cnt[1] = r.se()?;
            }
        }
        PocType::Cycle { .. } | PocType::FrameNum => {}
    }
    Ok((
        pic_order_cnt_lsb,
        delta_pic_order_cnt_bottom,
        delta_pic_order_cnt,
    ))
}

/// The list 0 loop of `ref_pic_list_modification()`, after its flag.
fn parse_modifications(r: &mut Reader<'_>) -> Result<Vec<Modification>> {
    let mut commands = Vec::new();
    loop {
        let command = match r.ue()? {
            0 => Modification::Subtract(r.ue()?),
            1 => Modification::Add(r.ue()?),
            2 => Modification::LongTerm(r.ue()?),
            3 => return Ok(commands),
            _ => return Err(DecodeError::Malformed("modification_of_pic_nums_idc")),
        };
        if commands.len() == MAX_MODIFICATIONS {
            return Err(DecodeError::Malformed("too many list modifications"));
        }
        commands.push(command);
    }
}

/// `pred_weight_table()` of a P slice (§7.3.3.2).
fn parse_weights(r: &mut Reader<'_>, count: usize, monochrome: bool) -> Result<WeightTable> {
    let luma_log2_denom = r.ue_max(MAX_LOG2_WEIGHT_DENOM, "luma_log2_weight_denom")?;
    let chroma_log2_denom = if monochrome {
        0
    } else {
        r.ue_max(MAX_LOG2_WEIGHT_DENOM, "chroma_log2_weight_denom")?
    };
    let default = |denom: u32| Weight {
        weight: 1 << denom,
        offset: 0,
    };
    let explicit = |r: &mut Reader<'_>| -> Result<Weight> {
        Ok(Weight {
            weight: r.se_range(MIN_WEIGHT, MAX_WEIGHT, "prediction weight")?,
            offset: r.se_range(MIN_WEIGHT, MAX_WEIGHT, "prediction offset")?,
        })
    };
    let mut weights = Vec::with_capacity(count);
    for _ in 0..count {
        let mut entry = [
            default(luma_log2_denom),
            default(chroma_log2_denom),
            default(chroma_log2_denom),
        ];
        if r.flag()? {
            entry[0] = explicit(r)?;
        }
        if !monochrome && r.flag()? {
            entry[1] = explicit(r)?;
            entry[2] = explicit(r)?;
        }
        weights.push(entry);
    }
    Ok(WeightTable {
        luma_log2_denom,
        chroma_log2_denom,
        weights,
    })
}

/// The `memory_management_control_operation` loop, after its flag.
fn parse_mmcos(r: &mut Reader<'_>) -> Result<Vec<Mmco>> {
    let mut operations = Vec::new();
    loop {
        let operation = match r.ue()? {
            0 => return Ok(operations),
            1 => Mmco::ReleaseShortTerm(r.ue()?),
            2 => Mmco::ReleaseLongTerm(r.ue()?),
            3 => Mmco::ShortTermToLongTerm {
                difference_of_pic_nums_minus1: r.ue()?,
                long_term_frame_idx: r.ue()?,
            },
            4 => Mmco::MaxLongTermFrameIdx(r.ue()?),
            5 => Mmco::ReleaseAll,
            6 => Mmco::CurrentToLongTerm(r.ue()?),
            _ => {
                return Err(DecodeError::Malformed(
                    "memory_management_control_operation",
                ));
            }
        };
        if operations.len() == MAX_MMCOS {
            return Err(DecodeError::Malformed("too many MMCOs"));
        }
        operations.push(operation);
    }
}

/// `disable_deblocking_filter_idc` and, unless it switches the filter off,
/// the two filter offsets (ranges §7.4.3).
fn parse_deblock(r: &mut Reader<'_>) -> Result<DeblockControl> {
    let disable_idc = r.ue_max(
        u32::from(DEBLOCK_IDC_NO_SLICE_EDGES),
        "disable_deblocking_filter_idc",
    )? as u8;
    let mut control = DeblockControl {
        disable_idc,
        ..DeblockControl::DEFAULT
    };
    if control.offsets_coded() {
        let max = i32::from(DEBLOCK_OFFSET_DIV2_MAX);
        control.alpha_offset_div2 = r.se_range(-max, max, "slice_alpha_c0_offset_div2")? as i8;
        control.beta_offset_div2 = r.se_range(-max, max, "slice_beta_offset_div2")? as i8;
    }
    Ok(control)
}
