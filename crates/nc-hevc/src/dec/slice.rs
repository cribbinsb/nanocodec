//! Slice segment headers (H.265 §7.3.6) with the reference picture list
//! modification (§7.3.6.2) and prediction weight table (§7.3.6.3).

use nc_common::decode::{BitReader, DecodeError};

use super::params::{MAX_DPB, MAX_PPS, Pps, ShortTermSet, Sps, deblock_offsets, short_term_set};

/// NAL unit types (Table 7-1).
pub(super) mod nal {
    pub const RADL_N: u8 = 6;
    pub const RASL_N: u8 = 8;
    pub const RASL_R: u8 = 9;
    /// Sub-layer non-reference pictures are the even types up to 14.
    pub const RSV_VCL_N14: u8 = 14;
    pub const BLA_W_LP: u8 = 16;
    pub const IDR_W_RADL: u8 = 19;
    pub const IDR_N_LP: u8 = 20;
    pub const CRA: u8 = 21;
    /// Last reserved IRAP type.
    pub const RSV_IRAP_23: u8 = 23;
    /// First non-VCL type.
    pub const SPS: u8 = 33;
    pub const PPS: u8 = 34;
    pub const EOS: u8 = 36;
    pub const EOB: u8 = 37;
}

/// `slice_type` (Table 7-7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SliceType {
    B,
    P,
    I,
}

/// Explicit weighted prediction of one reference (§7.4.7.3): luma and
/// Cb/Cr weight and offset.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct Weight {
    pub weight: [i32; 3],
    pub offset: [i32; 3],
}

/// `pred_weight_table()` results.
#[derive(Debug, Clone, Default)]
pub(super) struct Weights {
    /// `luma_log2_weight_denom`, `ChromaLog2WeightDenom`.
    pub log2_denom: [u32; 2],
    pub lists: [Vec<Weight>; 2],
}

/// Long-term entries of a slice (§7.4.7.1): `PocLsbLt` or full POC,
/// whether the MSB is present, and `UsedByCurrPicLt`.
#[derive(Debug, Clone, Copy)]
pub(super) struct LongTerm {
    /// `PocLsbLt`, or the full POC when `msb_present`.
    pub poc: i32,
    pub msb_present: bool,
    pub used: bool,
}

/// The parts of a slice segment header that dependent slice segments
/// inherit from their independent one.
#[allow(
    clippy::struct_excessive_bools,
    reason = "one field per slice header flag"
)]
#[derive(Debug, Clone)]
pub(super) struct SliceHeader {
    pub slice_type: SliceType,
    pub pic_output: bool,
    pub poc_lsb: u32,
    pub short_term: ShortTermSet,
    pub long_term: Vec<LongTerm>,
    pub temporal_mvp: bool,
    pub sao: [bool; 2],
    pub num_ref_idx: [usize; 2],
    /// `list_entry_lX`, when modified.
    pub list_entries: [Option<Vec<usize>>; 2],
    pub mvd_l1_zero: bool,
    pub cabac_init: bool,
    pub collocated_from_l0: bool,
    pub collocated_ref_idx: usize,
    pub weights: Option<Weights>,
    pub max_merge_candidates: usize,
    pub qp: i32,
    pub cb_qp_offset: i32,
    pub cr_qp_offset: i32,
    pub deblock_disabled: bool,
    pub beta_offset_div2: i32,
    pub tc_offset_div2: i32,
    pub loop_filter_across_slices: bool,
    /// `no_output_of_prior_pics_flag` (IRAP only).
    pub no_output_of_prior_pics: bool,
}

/// One parsed slice segment header.
#[derive(Debug, Clone)]
pub(super) struct SliceSegment {
    pub dependent: bool,
    /// `slice_segment_address` in raster CTBs.
    pub address: usize,
    /// `None` for a dependent segment (inherits the previous header).
    pub header: Option<SliceHeader>,
    /// Byte offset of `slice_segment_data()` in the RBSP.
    pub data_offset: usize,
}

/// `Ceil(Log2(n))`, the width of fixed-length indices into `n` entries.
fn ceil_log2(n: usize) -> u32 {
    usize::BITS - n.saturating_sub(1).leading_zeros()
}

/// `NumPicTotalCurr` (equation 7-55).
pub(super) fn num_pic_total_curr(short: &ShortTermSet, long: &[LongTerm]) -> usize {
    short
        .negative
        .iter()
        .chain(&short.positive)
        .filter(|e| e.1)
        .count()
        + long.iter().filter(|l| l.used).count()
}

/// Peek at `first_slice_segment_in_pic_flag` and `slice_pic_parameter_set_id`.
pub(super) fn first_and_pps(rbsp: &[u8], nal_type: u8) -> Result<(bool, usize), DecodeError> {
    let mut r = BitReader::new(rbsp);
    let first = r.read_bit()?;
    if (nal::BLA_W_LP..=nal::RSV_IRAP_23).contains(&nal_type) {
        r.read_bit()?;
    }
    let pps_id = r.read_ue()? as usize;
    // §7.4.7.1: slice_pic_parameter_set_id is in 0..=63.
    if pps_id >= MAX_PPS {
        return Err(DecodeError::Malformed("slice PPS id out of range"));
    }
    Ok((first, pps_id))
}

/// Parse a slice segment header (§7.3.6.1).
pub(super) fn parse(
    rbsp: &[u8],
    nal_type: u8,
    sps: &Sps,
    pps: &Pps,
) -> Result<SliceSegment, DecodeError> {
    let mut r = BitReader::new(rbsp);
    let first_in_picture = r.read_bit()?;
    let irap = (nal::BLA_W_LP..=nal::RSV_IRAP_23).contains(&nal_type);
    let no_output_of_prior_pics = irap && r.read_bit()?;
    r.read_ue()?; // slice_pic_parameter_set_id, read by `first_and_pps`
    let (ctb_w, ctb_h) = sps.ctbs();
    let mut dependent = false;
    let mut address = 0;
    if !first_in_picture {
        if pps.dependent_slices {
            dependent = r.read_bit()?;
        }
        address = r.read_bits(ceil_log2(ctb_w * ctb_h))? as usize;
        if address >= ctb_w * ctb_h {
            return Err(DecodeError::Malformed("slice_segment_address"));
        }
    }
    let header = if dependent {
        None
    } else {
        Some(independent(
            &mut r,
            nal_type,
            sps,
            pps,
            no_output_of_prior_pics,
        )?)
    };
    if pps.tiles.is_some() || pps.entropy_sync {
        let entry_points = r.read_ue()? as usize;
        if entry_points > ctb_w * ctb_h {
            return Err(DecodeError::Malformed("num_entry_point_offsets"));
        }
        if entry_points > 0 {
            let bits = r.read_ue()? + 1;
            if bits > 32 {
                return Err(DecodeError::Malformed("offset_len_minus1"));
            }
            // Substreams are contiguous; the decoder finds each one where
            // the previous ends, so the offsets are only skipped.
            r.skip_bits(entry_points * bits as usize)?;
        }
    }
    if pps.slice_header_extension {
        let length = r.read_ue()? as usize;
        r.skip_bits(8 * length)?;
    }
    // byte_alignment(): a one bit then zeros.
    if !r.read_bit()? {
        return Err(DecodeError::Malformed("slice header alignment"));
    }
    r.align_zero()?;
    Ok(SliceSegment {
        dependent,
        address,
        header,
        data_offset: r.position() / 8,
    })
}

/// The fields of an independent slice segment header.
#[allow(
    clippy::too_many_lines,
    reason = "follows the syntax table of §7.3.6.1"
)]
fn independent(
    r: &mut BitReader<'_>,
    nal_type: u8,
    sps: &Sps,
    pps: &Pps,
    no_output_of_prior_pics: bool,
) -> Result<SliceHeader, DecodeError> {
    r.skip_bits(pps.extra_slice_header_bits as usize)?;
    let slice_type = match r.read_ue()? {
        0 => SliceType::B,
        1 => SliceType::P,
        2 => SliceType::I,
        _ => return Err(DecodeError::Malformed("slice_type")),
    };
    let pic_output = !pps.output_flag_present || r.read_bit()?;
    let idr = matches!(nal_type, nal::IDR_W_RADL | nal::IDR_N_LP);
    let mut poc_lsb = 0;
    let mut short_term = ShortTermSet::default();
    let mut long_term = Vec::new();
    let mut temporal_mvp = false;
    if !idr {
        poc_lsb = r.read_bits(sps.log2_max_poc_lsb)?;
        short_term = if r.read_bit()? {
            let n = sps.short_term_sets.len();
            if n == 0 {
                return Err(DecodeError::Malformed("short_term_ref_pic_set_sps_flag"));
            }
            let index = r.read_bits(ceil_log2(n))? as usize;
            sps.short_term_sets
                .get(index)
                .ok_or(DecodeError::Malformed("short_term_ref_pic_set_idx"))?
                .clone()
        } else {
            short_term_set(r, &sps.short_term_sets, sps.short_term_sets.len(), true)?
        };
        if let Some(candidates) = &sps.long_term {
            long_term = long_term_entries(r, sps, candidates)?;
        }
        if sps.temporal_mvp {
            temporal_mvp = r.read_bit()?;
        }
    }
    let mut sao = [false; 2];
    if sps.sao {
        sao = [r.read_bit()?, r.read_bit()?];
    }
    let mut num_ref_idx = [0; 2];
    let mut list_entries = [None, None];
    let mut mvd_l1_zero = false;
    let mut cabac_init = false;
    let mut collocated_from_l0 = true;
    let mut collocated_ref_idx = 0;
    let mut weights = None;
    let mut max_merge_candidates = 0;
    if slice_type != SliceType::I {
        let lists = if slice_type == SliceType::B { 2 } else { 1 };
        num_ref_idx[..lists].copy_from_slice(&pps.num_ref_idx_default[..lists]);
        if r.read_bit()? {
            for n in &mut num_ref_idx[..lists] {
                *n = r.read_ue()? as usize + 1;
                if *n > MAX_DPB - 1 {
                    return Err(DecodeError::Malformed("num_ref_idx_active_minus1"));
                }
            }
        }
        let total = num_pic_total_curr(&short_term, &long_term);
        if total == 0 {
            return Err(DecodeError::Malformed("inter slice without references"));
        }
        if pps.lists_modification && total > 1 {
            for (list, entries) in list_entries.iter_mut().enumerate().take(lists) {
                if r.read_bit()? {
                    let mut v = Vec::with_capacity(num_ref_idx[list]);
                    for _ in 0..num_ref_idx[list] {
                        let entry = r.read_bits(ceil_log2(total))? as usize;
                        if entry >= total {
                            return Err(DecodeError::Malformed("list_entry"));
                        }
                        v.push(entry);
                    }
                    *entries = Some(v);
                }
            }
        }
        if slice_type == SliceType::B {
            mvd_l1_zero = r.read_bit()?;
        }
        if pps.cabac_init_present {
            cabac_init = r.read_bit()?;
        }
        if temporal_mvp {
            if slice_type == SliceType::B {
                collocated_from_l0 = r.read_bit()?;
            }
            let list = usize::from(!collocated_from_l0);
            if num_ref_idx[list] > 1 {
                collocated_ref_idx = r.read_ue()? as usize;
                if collocated_ref_idx >= num_ref_idx[list] {
                    return Err(DecodeError::Malformed("collocated_ref_idx"));
                }
            }
        }
        if (pps.weighted_pred && slice_type == SliceType::P)
            || (pps.weighted_bipred && slice_type == SliceType::B)
        {
            weights = Some(pred_weight_table(r, &num_ref_idx[..lists])?);
        }
        max_merge_candidates = 5usize
            .checked_sub(r.read_ue()? as usize)
            .filter(|&n| n >= 1)
            .ok_or(DecodeError::Malformed("five_minus_max_num_merge_cand"))?;
    }
    let qp = pps.init_qp + r.read_se()?;
    if !(0..=51).contains(&qp) {
        return Err(DecodeError::Malformed("slice_qp_delta"));
    }
    let (mut cb_qp_offset, mut cr_qp_offset) = (0, 0);
    if pps.slice_chroma_qp_offsets {
        cb_qp_offset = r.read_se()?;
        cr_qp_offset = r.read_se()?;
        if !(-12..=12).contains(&cb_qp_offset)
            || !(-12..=12).contains(&cr_qp_offset)
            || !(-12..=12).contains(&(cb_qp_offset + pps.cb_qp_offset))
            || !(-12..=12).contains(&(cr_qp_offset + pps.cr_qp_offset))
        {
            return Err(DecodeError::Malformed("slice chroma QP offset"));
        }
    }
    let control = pps.deblock;
    let mut deblock_disabled = control.disabled;
    let (mut beta_offset_div2, mut tc_offset_div2) =
        (control.beta_offset_div2, control.tc_offset_div2);
    if control.override_enabled && r.read_bit()? {
        deblock_disabled = r.read_bit()?;
        if !deblock_disabled {
            (beta_offset_div2, tc_offset_div2) = deblock_offsets(r)?;
        }
    }
    let loop_filter_across_slices =
        if pps.loop_filter_across_slices && (sao[0] || sao[1] || !deblock_disabled) {
            r.read_bit()?
        } else {
            pps.loop_filter_across_slices
        };
    Ok(SliceHeader {
        slice_type,
        pic_output,
        poc_lsb,
        short_term,
        long_term,
        temporal_mvp,
        sao,
        num_ref_idx,
        list_entries,
        mvd_l1_zero,
        cabac_init,
        collocated_from_l0,
        collocated_ref_idx,
        weights,
        max_merge_candidates,
        qp,
        cb_qp_offset,
        cr_qp_offset,
        deblock_disabled,
        beta_offset_div2,
        tc_offset_div2,
        loop_filter_across_slices,
        no_output_of_prior_pics,
    })
}

/// The long-term part of the header (§7.3.6.1) with `PocLsbLt` and
/// `DeltaPocMsbCycleLt` (equation 7-52) folded into one value per entry:
/// the LSBs, or `-(DeltaPocMsbCycleLt * MaxPicOrderCntLsb) + lsb` marked
/// `msb_present`, made absolute once the current POC is known.
fn long_term_entries(
    r: &mut BitReader<'_>,
    sps: &Sps,
    candidates: &[(u32, bool)],
) -> Result<Vec<LongTerm>, DecodeError> {
    let from_sps = if candidates.is_empty() {
        0
    } else {
        r.read_ue()? as usize
    };
    let from_slice = r.read_ue()? as usize;
    if from_sps > candidates.len() || from_sps + from_slice >= MAX_DPB {
        return Err(DecodeError::Malformed("num_long_term_pics"));
    }
    let mut entries = Vec::with_capacity(from_sps + from_slice);
    let mut cycle = 0i64;
    for i in 0..from_sps + from_slice {
        let (lsb, used) = if i < from_sps {
            let index = if candidates.len() > 1 {
                r.read_bits(ceil_log2(candidates.len()))? as usize
            } else {
                0
            };
            *candidates
                .get(index)
                .ok_or(DecodeError::Malformed("lt_idx_sps"))?
        } else {
            (r.read_bits(sps.log2_max_poc_lsb)?, r.read_bit()?)
        };
        let msb_present = r.read_bit()?;
        if i == 0 || i == from_sps {
            cycle = 0;
        }
        if msb_present {
            cycle += i64::from(r.read_ue()?);
        }
        let poc = if msb_present {
            i64::from(lsb) - cycle * (1i64 << sps.log2_max_poc_lsb)
        } else {
            i64::from(lsb)
        };
        entries.push(LongTerm {
            poc: i32::try_from(poc)
                .map_err(|_| DecodeError::Malformed("delta_poc_msb_cycle_lt"))?,
            msb_present,
            used,
        });
    }
    Ok(entries)
}

/// `pred_weight_table()` (§7.3.6.3, semantics §7.4.7.3), 8-bit samples.
fn pred_weight_table(r: &mut BitReader<'_>, num_ref_idx: &[usize]) -> Result<Weights, DecodeError> {
    let luma_denom = r.read_ue()?;
    if luma_denom > 7 {
        return Err(DecodeError::Malformed("luma_log2_weight_denom"));
    }
    let chroma_denom = i64::from(luma_denom) + i64::from(r.read_se()?);
    if !(0..=7).contains(&chroma_denom) {
        return Err(DecodeError::Malformed("delta_chroma_log2_weight_denom"));
    }
    let chroma_denom = chroma_denom as u32;
    let mut weights = Weights {
        log2_denom: [luma_denom, chroma_denom],
        lists: [Vec::new(), Vec::new()],
    };
    for (list, &count) in num_ref_idx.iter().enumerate() {
        let mut luma_flags = [false; MAX_DPB];
        let mut chroma_flags = [false; MAX_DPB];
        for f in &mut luma_flags[..count] {
            *f = r.read_bit()?;
        }
        for f in &mut chroma_flags[..count] {
            *f = r.read_bit()?;
        }
        for i in 0..count {
            let mut w = Weight {
                weight: [1 << luma_denom, 1 << chroma_denom, 1 << chroma_denom],
                offset: [0; 3],
            };
            if luma_flags[i] {
                let delta = r.read_se()?;
                let offset = r.read_se()?;
                if !(-128..=127).contains(&delta) || !(-128..=127).contains(&offset) {
                    return Err(DecodeError::Malformed("luma weight"));
                }
                w.weight[0] += delta;
                w.offset[0] = offset;
            }
            if chroma_flags[i] {
                for c in 1..3 {
                    let delta = r.read_se()?;
                    let delta_offset = r.read_se()?;
                    if !(-128..=127).contains(&delta) || !(-512..=511).contains(&delta_offset) {
                        return Err(DecodeError::Malformed("chroma weight"));
                    }
                    w.weight[c] += delta;
                    // Equation 7-56 with wpOffsetHalfRangeC = 128.
                    w.offset[c] = (128 + delta_offset - ((128 * w.weight[c]) >> chroma_denom))
                        .clamp(-128, 127);
                }
            }
            weights.lists[list].push(w);
        }
    }
    Ok(weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_length_index_widths() {
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(510), 9);
    }
}
