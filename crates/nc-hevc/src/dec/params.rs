//! Sequence and picture parameter sets of Main-profile streams (H.265
//! §7.3.2.2, §7.3.2.3), their short-term reference picture sets (§7.3.7,
//! §7.4.8) and scaling lists (§7.3.4, §7.4.5).
//!
//! The decoder supports what the Main and Main Still Picture profiles allow
//! (A.3.2, A.3.5): 8-bit 4:2:0 and no range, multilayer, 3D or SCC
//! extension tools; anything else is rejected as unsupported. The VPS
//! carries nothing single-layer decoding needs and is not parsed.

use nc_common::ColourInfo;
use nc_common::colour::read_video_signal_type;
use nc_common::decode::{BitReader, DecodeError};

use crate::scan::up_right_diagonal;

/// `sps_seq_parameter_set_id` range (§7.4.3.2.1).
pub(super) const MAX_SPS: usize = 16;
/// `pps_pic_parameter_set_id` range (§7.4.3.3.1).
pub(super) const MAX_PPS: usize = 64;
/// Most sub-layers (§7.4.3.2.1).
const MAX_SUB_LAYERS: usize = 7;
/// `num_short_term_ref_pic_sets` limit (§7.4.3.2.1).
const MAX_SHORT_TERM_SETS: u32 = 64;
/// `num_long_term_ref_pics_sps` limit.
const MAX_LONG_TERM_SPS: u32 = 32;
/// Entries of one reference picture set: `sps_max_dec_pic_buffering_minus1`
/// is at most 15, so is each list (§7.4.8).
pub(super) const MAX_DPB: usize = 16;
/// The largest picture side accepted, beyond any level's limit (Table A.8:
/// level 6.2 allows 35 651 584 luma samples, `sqrt(8 * MaxLumaPs)`).
const MAX_SIDE: u32 = 16888;
/// The most luma samples accepted (level 6.2, Table A.8).
const MAX_LUMA_SAMPLES: u64 = 35_651_584;
/// Tile columns / rows limit (level 6.2, Table A.8).
const MAX_TILE_COLUMNS: u32 = 20;
const MAX_TILE_ROWS: u32 = 22;
/// Largest log2 size difference in the SPS block-size syntax: the 64x64 CTB
/// over the 8x8 minimum coding block (§7.4.3.2.1).
const MAX_LOG2_DIFF: u32 = 3;
/// Largest `delta_poc_s*_minus1` / `abs_delta_rps_minus1` (§7.4.8).
const MAX_DELTA_POC_MINUS1: u32 = (1 << 15) - 1;
/// `aspect_ratio_idc` that carries an explicit sample aspect ratio.
const EXTENDED_SAR: u32 = 255;

/// One short-term reference picture set (§7.4.8): POC deltas relative to
/// the current picture and whether the current picture uses each.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct ShortTermSet {
    /// `DeltaPocS0` / `UsedByCurrPicS0`, closest first (negative deltas).
    pub negative: Vec<(i32, bool)>,
    /// `DeltaPocS1` / `UsedByCurrPicS1`, closest first.
    pub positive: Vec<(i32, bool)>,
}

impl ShortTermSet {
    fn len(&self) -> usize {
        self.negative.len() + self.positive.len()
    }
}

/// `ScalingFactor` of every size and matrix (§7.4.5), row-major.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ScalingLists {
    /// Index `[sizeId][matrixId]`: 4x4, 8x8, 16x16, 32x32 (only matrices 0
    /// and 3 of size 3 exist).
    factors: [[Vec<u8>; 6]; 4],
}

/// Table 7-6, intra matrices of sizeId 1..3, in up-right diagonal order.
#[rustfmt::skip]
const DEFAULT_INTRA: [u8; 64] = [
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 16, 17, 16, 17, 18, 17, 18, 18, 17, 18, 21, 19, 20,
    21, 20, 19, 21, 24, 22, 22, 24, 24, 22, 22, 24, 25, 25, 27, 30, 27, 25, 25, 29, 31, 35, 35, 31,
    29, 36, 41, 44, 41, 36, 47, 54, 54, 47, 65, 70, 65, 88, 88, 115,
];
/// Table 7-6, inter matrices of sizeId 1..3.
#[rustfmt::skip]
const DEFAULT_INTER: [u8; 64] = [
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 20, 20, 20,
    20, 20, 20, 20, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 28, 28, 28, 28, 28,
    28, 33, 33, 33, 33, 33, 41, 41, 41, 41, 54, 54, 54, 71, 71, 91,
];
/// Flat scaling factor (`m[x][y]` without scaling lists, §8.6.4.2).
pub(super) const FLAT_SCALE: u8 = 16;
/// Matrices per size; sizeId 3 has only intra and inter luma (§7.3.4).
const MATRICES: usize = 6;
const SIZES: usize = 4;

impl ScalingLists {
    /// `ScalingFactor[sizeId][matrixId]` for a `2^log2` transform
    /// (`matrixId`: 0..2 intra Y/Cb/Cr, 3..5 inter).
    pub fn factors(&self, log2: u8, matrix: usize) -> &[u8] {
        let size = usize::from(log2 - 2);
        // 32x32 chroma never occurs in 4:2:0; its matrix is the luma one.
        let matrix = if size == 3 { matrix / 3 * 3 } else { matrix };
        &self.factors[size][matrix]
    }

    /// The default lists (Table 7-5, 7-6), used when scaling lists are
    /// enabled without explicit data.
    pub fn default_lists() -> Self {
        Self::expand(&Self::default_lists_raw())
    }

    /// `scaling_list_data()` (§7.3.4).
    fn parse(r: &mut BitReader<'_>) -> Result<Self, DecodeError> {
        let defaults = Self::default_lists_raw();
        let mut lists: [[(Vec<u8>, u8); MATRICES]; SIZES] = defaults.clone();
        for size in 0..SIZES {
            let step = if size == 3 { 3 } else { 1 };
            for matrix in (0..MATRICES).step_by(step) {
                if r.read_bit()? {
                    lists[size][matrix] = explicit_list(r, size)?;
                } else {
                    // scaling_list_pred_mode_flag 0: copy a previous matrix
                    // of this size, or the default one for delta 0.
                    // §7.4.5: scaling_list_pred_matrix_id_delta <= matrixId / step.
                    let delta = r.read_ue_max(matrix as u32, "scaling_list_pred_matrix_id_delta")?
                        as usize
                        * step;
                    lists[size][matrix] = if delta == 0 {
                        defaults[size][matrix].clone()
                    } else {
                        let reference = matrix
                            .checked_sub(delta)
                            .ok_or(DecodeError::Malformed("scaling list reference"))?;
                        lists[size][reference].clone()
                    };
                }
            }
        }
        Ok(Self::expand(&lists))
    }

    fn default_lists_raw() -> [[(Vec<u8>, u8); MATRICES]; SIZES] {
        std::array::from_fn(|size| {
            std::array::from_fn(|matrix| {
                let coefficients: &[u8] = if size == 0 {
                    &[FLAT_SCALE; 16]
                } else if matrix < 3 {
                    &DEFAULT_INTRA
                } else {
                    &DEFAULT_INTER
                };
                (coefficients.to_vec(), FLAT_SCALE)
            })
        })
    }

    /// `ScalingFactor` from the coded lists (§7.4.5, equations 7-40 to
    /// 7-44): 4x4 and 8x8 lists map through the diagonal scan; 16x16 and
    /// 32x32 upsample the 8x8 list and replace the DC.
    fn expand(lists: &[[(Vec<u8>, u8); MATRICES]; SIZES]) -> Self {
        let scan4 = up_right_diagonal(4);
        let scan8 = up_right_diagonal(8);
        let factors = std::array::from_fn(|size| {
            std::array::from_fn(|matrix| {
                let (coefficients, dc) = &lists[size][matrix];
                let side = 4usize << size;
                let mut m = vec![FLAT_SCALE; side * side];
                let (scan, ratio) = if size == 0 {
                    (&scan4, 1)
                } else {
                    (&scan8, side / 8)
                };
                for (i, &(x, y)) in scan.iter().enumerate() {
                    for dy in 0..ratio {
                        for dx in 0..ratio {
                            m[(y * ratio + dy) * side + x * ratio + dx] = coefficients[i];
                        }
                    }
                }
                if size >= 2 {
                    m[0] = *dc;
                }
                m
            })
        });
        Self { factors }
    }
}

/// One explicitly coded scaling list and its DC (`scaling_list_pred_mode_flag` 1).
fn explicit_list(r: &mut BitReader<'_>, size: usize) -> Result<(Vec<u8>, u8), DecodeError> {
    let count = 16usize << (2 * size.min(1));
    let mut next = 8i32;
    let mut dc = FLAT_SCALE;
    if size > 1 {
        let coef = r.read_se()?;
        if !(-7..=247).contains(&coef) {
            return Err(DecodeError::Malformed("scaling_list_dc_coef_minus8"));
        }
        next = coef + 8;
        dc = next as u8;
    }
    let mut coefficients = Vec::with_capacity(count);
    for _ in 0..count {
        let delta = r.read_se()?;
        if !(-128..=127).contains(&delta) {
            return Err(DecodeError::Malformed("scaling_list_delta_coef"));
        }
        next = (next + delta + 256).rem_euclid(256);
        if next == 0 {
            return Err(DecodeError::Malformed("zero scaling list entry"));
        }
        coefficients.push(next as u8);
    }
    Ok((coefficients, dc))
}

/// `sps_range_extension()` and friends are outside Main: their presence
/// flags must be 0.
fn no_extensions(r: &mut BitReader<'_>) -> Result<(), DecodeError> {
    if r.read_bit()? {
        let flags = r.read_bits(8)?;
        // sps/pps_range_extension_flag is the first of the eight.
        if flags & 0x80 != 0 {
            return Err(DecodeError::Unsupported("range extension"));
        }
        // Other extensions do not affect single-layer Main decoding; their
        // data is ignored (§7.4.3.2.1: decoders shall ignore it).
    }
    Ok(())
}

/// A sequence parameter set (§7.4.3.2.1).
#[allow(clippy::struct_excessive_bools, reason = "one field per SPS flag")]
#[derive(Debug, Clone)]
pub(super) struct Sps {
    pub max_sub_layers: usize,
    pub width: usize,
    pub height: usize,
    /// Conformance window in luma samples: left, right, top, bottom.
    pub crop: [usize; 4],
    pub log2_max_poc_lsb: u32,
    /// Per sub-layer: `sps_max_dec_pic_buffering_minus1 + 1`,
    /// `sps_max_num_reorder_pics`, `SpsMaxLatencyPictures` (`None`: no limit).
    pub dpb: [(usize, usize, Option<u32>); MAX_SUB_LAYERS],
    pub log2_min_cb: u8,
    pub log2_ctb: u8,
    pub log2_min_tb: u8,
    pub log2_max_tb: u8,
    pub max_depth_inter: u8,
    pub max_depth_intra: u8,
    /// Scaling lists: `None` when `scaling_list_enabled_flag` is 0.
    pub scaling: Option<ScalingLists>,
    pub amp: bool,
    pub sao: bool,
    pub pcm: Option<Pcm>,
    pub short_term_sets: Vec<ShortTermSet>,
    /// `lt_ref_pic_poc_lsb_sps`, `used_by_curr_pic_lt_sps_flag`; `None`
    /// when `long_term_ref_pics_present_flag` is 0.
    pub long_term: Option<Vec<(u32, bool)>>,
    pub temporal_mvp: bool,
    pub strong_intra_smoothing: bool,
    pub colour: ColourInfo,
}

/// PCM parameters (§7.4.3.2.1).
#[derive(Debug, Clone, Copy)]
pub(super) struct Pcm {
    pub bit_depth_luma: u32,
    pub bit_depth_chroma: u32,
    pub log2_min: u8,
    pub log2_max: u8,
    pub loop_filter_disabled: bool,
}

impl Sps {
    /// `sps_seq_parameter_set_id` and the parsed set.
    #[allow(
        clippy::too_many_lines,
        reason = "follows the syntax table of §7.3.2.2"
    )]
    pub fn parse(rbsp: &[u8]) -> Result<(usize, Self), DecodeError> {
        let mut r = BitReader::new(rbsp);
        r.read_bits(4)?; // sps_video_parameter_set_id
        let max_sub_layers = r.read_bits(3)? as usize + 1;
        if max_sub_layers > MAX_SUB_LAYERS {
            return Err(DecodeError::Malformed("sps_max_sub_layers_minus1"));
        }
        r.read_bit()?; // sps_temporal_id_nesting_flag
        profile_tier_level(&mut r, max_sub_layers - 1)?;
        let id = r.read_ue()? as usize;
        if id >= MAX_SPS {
            return Err(DecodeError::Malformed("sps_seq_parameter_set_id"));
        }
        if r.read_ue()? != 1 {
            return Err(DecodeError::Unsupported("chroma format other than 4:2:0"));
        }
        let width = r.read_ue()?;
        let height = r.read_ue()?;
        if width == 0
            || height == 0
            || width > MAX_SIDE
            || height > MAX_SIDE
            || u64::from(width) * u64::from(height) > MAX_LUMA_SAMPLES
        {
            return Err(DecodeError::Unsupported("picture size"));
        }
        let mut crop = [0usize; 4];
        if r.read_bit()? {
            for c in &mut crop {
                // Offsets count chroma samples (SubWidthC = SubHeightC = 2).
                // Bounded before doubling, so the sums below cannot wrap a
                // 32-bit `usize` (WASM) into a valid-looking window.
                *c = 2 * r.read_ue_max(width.max(height), "conf_win_offset")? as usize;
            }
        }
        if crop[0] + crop[1] >= width as usize || crop[2] + crop[3] >= height as usize {
            return Err(DecodeError::Malformed("conformance window"));
        }
        if r.read_ue()? != 0 || r.read_ue()? != 0 {
            return Err(DecodeError::Unsupported("bit depth other than 8"));
        }
        // §7.4.3.2.1: log2_max_pic_order_cnt_lsb_minus4 is in 0..=12.
        let log2_max_poc_lsb = r.read_ue_max(12, "log2_max_pic_order_cnt_lsb_minus4")? + 4;
        let all_layers = r.read_bit()?;
        let mut dpb = [(1, 0, None); MAX_SUB_LAYERS];
        let first = if all_layers { 0 } else { max_sub_layers - 1 };
        for entry in &mut dpb[first..max_sub_layers] {
            let buffering =
                r.read_ue_max(MAX_DPB as u32, "sps_max_dec_pic_buffering_minus1")? as usize + 1;
            let reorder = r.read_ue_max(MAX_DPB as u32, "sps_max_num_reorder_pics")? as usize;
            let latency = r.read_ue()?;
            if buffering > MAX_DPB || reorder >= buffering {
                return Err(DecodeError::Malformed("sub-layer ordering info"));
            }
            // SpsMaxLatencyPictures = reorder + latency_increase_plus1 - 1.
            // Saturating: a limit beyond u32 never binds.
            let limit = (latency != 0).then(|| (latency - 1).saturating_add(reorder as u32));
            *entry = (buffering, reorder, limit);
        }
        for layer in 0..first {
            dpb[layer] = dpb[first];
        }
        // §7.4.3.2.1 bounds each size by the 64x64 CTB, which keeps the
        // sums and shifts below in range; the checks after them are exact.
        let log2_min_cb =
            r.read_ue_max(MAX_LOG2_DIFF, "log2_min_luma_coding_block_size_minus3")? + 3;
        let log2_ctb = log2_min_cb
            + r.read_ue_max(MAX_LOG2_DIFF, "log2_diff_max_min_luma_coding_block_size")?;
        let log2_min_tb =
            r.read_ue_max(MAX_LOG2_DIFF, "log2_min_luma_transform_block_size_minus2")? + 2;
        let log2_max_tb = log2_min_tb
            + r.read_ue_max(MAX_LOG2_DIFF, "log2_diff_max_min_luma_transform_block_size")?;
        if !(4..=6).contains(&log2_ctb)
            || log2_min_cb > log2_ctb
            || log2_min_tb >= log2_min_cb
            || log2_max_tb > 5.min(log2_ctb)
            || width % (1 << log2_min_cb) != 0
            || height % (1 << log2_min_cb) != 0
        {
            return Err(DecodeError::Malformed("block size limits"));
        }
        let max_depth_inter = r.read_ue()?;
        let max_depth_intra = r.read_ue()?;
        if max_depth_inter > log2_ctb - log2_min_tb || max_depth_intra > log2_ctb - log2_min_tb {
            return Err(DecodeError::Malformed("max_transform_hierarchy_depth"));
        }
        let scaling = if r.read_bit()? {
            Some(if r.read_bit()? {
                ScalingLists::parse(&mut r)?
            } else {
                ScalingLists::default_lists()
            })
        } else {
            None
        };
        let amp = r.read_bit()?;
        let sao = r.read_bit()?;
        let pcm = if r.read_bit()? {
            let bit_depth_luma = r.read_bits(4)? + 1;
            let bit_depth_chroma = r.read_bits(4)? + 1;
            let log2_min =
                r.read_ue_max(MAX_LOG2_DIFF, "log2_min_pcm_luma_coding_block_size_minus3")? + 3;
            let log2_max = log2_min
                + r.read_ue_max(
                    MAX_LOG2_DIFF,
                    "log2_diff_max_min_pcm_luma_coding_block_size",
                )?;
            if bit_depth_luma > 8 || bit_depth_chroma > 8 || log2_max > 5.min(log2_ctb) {
                return Err(DecodeError::Malformed("PCM parameters"));
            }
            Some(Pcm {
                bit_depth_luma,
                bit_depth_chroma,
                log2_min: log2_min as u8,
                log2_max: log2_max as u8,
                loop_filter_disabled: r.read_bit()?,
            })
        } else {
            None
        };
        let count = r.read_ue()?;
        if count > MAX_SHORT_TERM_SETS {
            return Err(DecodeError::Malformed("num_short_term_ref_pic_sets"));
        }
        let mut short_term_sets = Vec::with_capacity(count as usize);
        for index in 0..count as usize {
            let set = short_term_set(&mut r, &short_term_sets, index, false)?;
            short_term_sets.push(set);
        }
        let long_term = if r.read_bit()? {
            let count = r.read_ue()?;
            if count > MAX_LONG_TERM_SPS {
                return Err(DecodeError::Malformed("num_long_term_ref_pics_sps"));
            }
            let mut list = Vec::with_capacity(count as usize);
            for _ in 0..count {
                list.push((r.read_bits(log2_max_poc_lsb)?, r.read_bit()?));
            }
            Some(list)
        } else {
            None
        };
        let temporal_mvp = r.read_bit()?;
        let strong_intra_smoothing = r.read_bit()?;
        let colour = if r.read_bit()? {
            vui_colour(&mut r)?
        } else {
            ColourInfo::default()
        };
        // Everything after the colour description (timing, HRD, bitstream
        // restriction, extensions) is informative for Main decoding.
        Ok((
            id,
            Self {
                max_sub_layers,
                width: width as usize,
                height: height as usize,
                crop,
                log2_max_poc_lsb,
                dpb,
                log2_min_cb: log2_min_cb as u8,
                log2_ctb: log2_ctb as u8,
                log2_min_tb: log2_min_tb as u8,
                log2_max_tb: log2_max_tb as u8,
                max_depth_inter: max_depth_inter as u8,
                max_depth_intra: max_depth_intra as u8,
                scaling,
                amp,
                sao,
                pcm,
                short_term_sets,
                long_term,
                temporal_mvp,
                strong_intra_smoothing,
                colour,
            },
        ))
    }

    pub fn ctb_size(&self) -> usize {
        1 << self.log2_ctb
    }

    /// `PicWidthInCtbsY`, `PicHeightInCtbsY`.
    pub fn ctbs(&self) -> (usize, usize) {
        (
            self.width.div_ceil(self.ctb_size()),
            self.height.div_ceil(self.ctb_size()),
        )
    }
}

/// `profile_tier_level(1, maxNumSubLayersMinus1)` (§7.3.3). Nothing in it
/// changes decoding: unsupported tools are detected from the syntax itself.
fn profile_tier_level(r: &mut BitReader<'_>, sub_layers_minus1: usize) -> Result<(), DecodeError> {
    // general profile (88 bits) and level (8 bits).
    r.skip_bits(96)?;
    let mut present = [(false, false); MAX_SUB_LAYERS];
    for p in present.iter_mut().take(sub_layers_minus1) {
        *p = (r.read_bit()?, r.read_bit()?);
    }
    if sub_layers_minus1 > 0 {
        r.skip_bits(2 * (8 - sub_layers_minus1))?;
    }
    for &(profile, level) in present.iter().take(sub_layers_minus1) {
        if profile {
            r.skip_bits(88)?;
        }
        if level {
            r.skip_bits(8)?;
        }
    }
    Ok(())
}

/// The colour description of `vui_parameters()` (§E.2.1); the fields
/// before it are skipped.
fn vui_colour(r: &mut BitReader<'_>) -> Result<ColourInfo, DecodeError> {
    if r.read_bit()? && r.read_bits(8)? == EXTENDED_SAR {
        r.skip_bits(32)?; // sar_width, sar_height
    }
    if r.read_bit()? {
        r.read_bit()?; // overscan_appropriate_flag
    }
    if r.read_bit()? {
        read_video_signal_type(r)
    } else {
        Ok(ColourInfo::default())
    }
}

/// `st_ref_pic_set(stRpsIdx)` (§7.3.7) with the derivation of §7.4.8.
/// `in_slice`: parsed in a slice header (index `sets.len()`).
pub(super) fn short_term_set(
    r: &mut BitReader<'_>,
    sets: &[ShortTermSet],
    index: usize,
    in_slice: bool,
) -> Result<ShortTermSet, DecodeError> {
    if index != 0 && r.read_bit()? {
        // inter_ref_pic_set_prediction_flag: predict from a previous set.
        let delta_idx = if in_slice {
            // §7.4.8: delta_idx_minus1 is in 0..=idx - 1.
            r.read_ue_max(index as u32, "delta_idx_minus1")? as usize + 1
        } else {
            1
        };
        let reference = index
            .checked_sub(delta_idx)
            .and_then(|i| sets.get(i))
            .ok_or(DecodeError::Malformed("delta_idx_minus1"))?;
        let sign = r.read_bit()?;
        // §7.4.8: abs_delta_rps_minus1 is in 0..=2^15 - 1.
        let magnitude = r.read_ue_max(MAX_DELTA_POC_MINUS1, "abs_delta_rps_minus1")? + 1;
        let delta_rps = if sign {
            -(magnitude as i32)
        } else {
            magnitude as i32
        };
        // used_by_curr_pic_flag / use_delta_flag per reference entry, then
        // one for the reference picture itself (index NumDeltaPocs).
        let n = reference.len();
        let mut used = vec![false; n + 1];
        let mut use_delta = vec![true; n + 1];
        for j in 0..=n {
            used[j] = r.read_bit()?;
            if !used[j] {
                use_delta[j] = r.read_bit()?;
            }
        }
        let neg = reference.negative.len();
        let mut set = ShortTermSet::default();
        // Equation 7-61.
        for (j, &(d, _)) in reference.positive.iter().enumerate().rev() {
            let poc = d + delta_rps;
            if poc < 0 && use_delta[neg + j] {
                set.negative.push((poc, used[neg + j]));
            }
        }
        if delta_rps < 0 && use_delta[n] {
            set.negative.push((delta_rps, used[n]));
        }
        for (j, &(d, _)) in reference.negative.iter().enumerate() {
            let poc = d + delta_rps;
            if poc < 0 && use_delta[j] {
                set.negative.push((poc, used[j]));
            }
        }
        // Equation 7-62.
        for (j, &(d, _)) in reference.negative.iter().enumerate().rev() {
            let poc = d + delta_rps;
            if poc > 0 && use_delta[j] {
                set.positive.push((poc, used[j]));
            }
        }
        if delta_rps > 0 && use_delta[n] {
            set.positive.push((delta_rps, used[n]));
        }
        for (j, &(d, _)) in reference.positive.iter().enumerate() {
            let poc = d + delta_rps;
            if poc > 0 && use_delta[neg + j] {
                set.positive.push((poc, used[neg + j]));
            }
        }
        if set.len() >= MAX_DPB {
            return Err(DecodeError::Malformed("reference picture set size"));
        }
        return Ok(set);
    }
    let negative = r.read_ue_max(MAX_DPB as u32, "num_negative_pics")? as usize;
    let positive = r.read_ue_max(MAX_DPB as u32, "num_positive_pics")? as usize;
    if negative + positive >= MAX_DPB {
        return Err(DecodeError::Malformed("reference picture set size"));
    }
    let mut set = ShortTermSet::default();
    let mut poc = 0i32;
    for _ in 0..negative {
        // §7.4.8: delta_poc_s0_minus1 is in 0..=2^15 - 1.
        poc -= r.read_ue_max(MAX_DELTA_POC_MINUS1, "delta_poc_s0_minus1")? as i32 + 1;
        set.negative.push((poc, r.read_bit()?));
    }
    poc = 0;
    for _ in 0..positive {
        poc += r.read_ue_max(MAX_DELTA_POC_MINUS1, "delta_poc_s1_minus1")? as i32 + 1;
        set.positive.push((poc, r.read_bit()?));
    }
    if poc > 1 << 16 || set.negative.last().is_some_and(|&(d, _)| d < -(1 << 16)) {
        return Err(DecodeError::Malformed("delta_poc"));
    }
    Ok(set)
}

/// Tile column or row boundaries in CTBs (§6.5.1): `n + 1` ascending
/// entries from 0 to the picture size.
fn tile_boundaries(
    r: &mut BitReader<'_>,
    count: usize,
    uniform: bool,
) -> Result<Option<Vec<u32>>, DecodeError> {
    if uniform {
        return Ok(None);
    }
    let mut sizes = Vec::with_capacity(count);
    for _ in 0..count - 1 {
        sizes.push(r.read_ue_max(MAX_SIDE, "tile size")? + 1);
    }
    Ok(Some(sizes))
}

/// Tile layout as coded: explicit sizes of all but the last column / row,
/// or uniform spacing.
#[derive(Debug, Clone)]
pub(super) struct Tiles {
    pub columns: usize,
    pub rows: usize,
    /// Explicit `column_width_minus1 + 1` (all but the last), `None` if uniform.
    pub widths: Option<Vec<u32>>,
    pub heights: Option<Vec<u32>>,
    pub loop_filter_across: bool,
}

impl Tiles {
    /// Column (or row) boundaries in CTBs for `total` CTBs (equations 6-3, 6-4).
    pub fn boundaries(
        count: usize,
        explicit: Option<&[u32]>,
        total: usize,
    ) -> Result<Vec<usize>, DecodeError> {
        let mut bounds = Vec::with_capacity(count + 1);
        bounds.push(0);
        match explicit {
            None => {
                for i in 1..=count {
                    bounds.push(i * total / count);
                }
            }
            Some(sizes) => {
                let mut at = 0usize;
                for &s in sizes {
                    at += s as usize;
                    bounds.push(at);
                }
                if at >= total {
                    return Err(DecodeError::Malformed("tile sizes exceed the picture"));
                }
                bounds.push(total);
            }
        }
        if bounds.windows(2).any(|w| w[0] >= w[1]) {
            return Err(DecodeError::Malformed("empty tile"));
        }
        Ok(bounds)
    }
}

/// Deblocking control of a PPS (§7.4.3.3.1).
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct DeblockControl {
    pub override_enabled: bool,
    pub disabled: bool,
    pub beta_offset_div2: i32,
    pub tc_offset_div2: i32,
}

/// A picture parameter set (§7.4.3.3.1).
#[allow(clippy::struct_excessive_bools, reason = "one field per PPS flag")]
#[derive(Debug, Clone)]
pub(super) struct Pps {
    pub sps_id: usize,
    pub dependent_slices: bool,
    pub output_flag_present: bool,
    pub extra_slice_header_bits: u32,
    pub sign_hiding: bool,
    pub cabac_init_present: bool,
    pub num_ref_idx_default: [usize; 2],
    pub init_qp: i32,
    pub constrained_intra_pred: bool,
    pub transform_skip: bool,
    /// `diff_cu_qp_delta_depth` when `cu_qp_delta_enabled_flag`.
    pub cu_qp_delta_depth: Option<u8>,
    pub cb_qp_offset: i32,
    pub cr_qp_offset: i32,
    pub slice_chroma_qp_offsets: bool,
    pub weighted_pred: bool,
    pub weighted_bipred: bool,
    pub transquant_bypass: bool,
    pub tiles: Option<Tiles>,
    pub entropy_sync: bool,
    pub loop_filter_across_slices: bool,
    pub deblock: DeblockControl,
    pub scaling: Option<ScalingLists>,
    pub lists_modification: bool,
    pub log2_parallel_merge_level: u8,
    pub slice_header_extension: bool,
}

impl Pps {
    /// `pps_pic_parameter_set_id` and the parsed set.
    #[allow(
        clippy::too_many_lines,
        reason = "follows the syntax table of §7.3.2.3"
    )]
    pub fn parse(rbsp: &[u8]) -> Result<(usize, Self), DecodeError> {
        let mut r = BitReader::new(rbsp);
        let id = r.read_ue()? as usize;
        let sps_id = r.read_ue()? as usize;
        if id >= MAX_PPS || sps_id >= MAX_SPS {
            return Err(DecodeError::Malformed("parameter set id"));
        }
        let dependent_slices = r.read_bit()?;
        let output_flag_present = r.read_bit()?;
        let extra_slice_header_bits = r.read_bits(3)?;
        let sign_hiding = r.read_bit()?;
        let cabac_init_present = r.read_bit()?;
        let l0 =
            r.read_ue_max(MAX_DPB as u32, "num_ref_idx_l0_default_active_minus1")? as usize + 1;
        let l1 =
            r.read_ue_max(MAX_DPB as u32, "num_ref_idx_l1_default_active_minus1")? as usize + 1;
        if l0 > MAX_DPB - 1 || l1 > MAX_DPB - 1 {
            return Err(DecodeError::Malformed("num_ref_idx_default_active"));
        }
        let init_qp = 26 + r.read_se()?;
        if !(0..=51).contains(&init_qp) {
            return Err(DecodeError::Malformed("init_qp_minus26"));
        }
        let constrained_intra_pred = r.read_bit()?;
        let transform_skip = r.read_bit()?;
        let cu_qp_delta_depth = if r.read_bit()? {
            let depth = r.read_ue()?;
            if depth > 3 {
                return Err(DecodeError::Malformed("diff_cu_qp_delta_depth"));
            }
            Some(depth as u8)
        } else {
            None
        };
        let cb_qp_offset = r.read_se()?;
        let cr_qp_offset = r.read_se()?;
        if !(-12..=12).contains(&cb_qp_offset) || !(-12..=12).contains(&cr_qp_offset) {
            return Err(DecodeError::Malformed("chroma QP offset"));
        }
        let slice_chroma_qp_offsets = r.read_bit()?;
        let weighted_pred = r.read_bit()?;
        let weighted_bipred = r.read_bit()?;
        let transquant_bypass = r.read_bit()?;
        let tiles_enabled = r.read_bit()?;
        let entropy_sync = r.read_bit()?;
        let tiles = if tiles_enabled {
            let columns = r.read_ue_max(MAX_TILE_COLUMNS, "num_tile_columns_minus1")? + 1;
            let rows = r.read_ue_max(MAX_TILE_ROWS, "num_tile_rows_minus1")? + 1;
            if columns > MAX_TILE_COLUMNS || rows > MAX_TILE_ROWS || columns * rows < 2 {
                return Err(DecodeError::Malformed("tile count"));
            }
            let uniform = r.read_bit()?;
            let widths = tile_boundaries(&mut r, columns as usize, uniform)?;
            let heights = tile_boundaries(&mut r, rows as usize, uniform)?;
            Some(Tiles {
                columns: columns as usize,
                rows: rows as usize,
                widths,
                heights,
                loop_filter_across: r.read_bit()?,
            })
        } else {
            None
        };
        let loop_filter_across_slices = r.read_bit()?;
        let mut deblock = DeblockControl::default();
        if r.read_bit()? {
            deblock.override_enabled = r.read_bit()?;
            deblock.disabled = r.read_bit()?;
            if !deblock.disabled {
                (deblock.beta_offset_div2, deblock.tc_offset_div2) = deblock_offsets(&mut r)?;
            }
        }
        let scaling = if r.read_bit()? {
            Some(ScalingLists::parse(&mut r)?)
        } else {
            None
        };
        let lists_modification = r.read_bit()?;
        // §7.4.3.3: Log2ParMrgLevel is at most CtbLog2SizeY (6).
        let log2_parallel_merge_level = r.read_ue_max(4, "log2_parallel_merge_level_minus2")? + 2;
        let slice_header_extension = r.read_bit()?;
        no_extensions(&mut r)?;
        Ok((
            id,
            Self {
                sps_id,
                dependent_slices,
                output_flag_present,
                extra_slice_header_bits,
                sign_hiding,
                cabac_init_present,
                num_ref_idx_default: [l0, l1],
                init_qp,
                constrained_intra_pred,
                transform_skip,
                cu_qp_delta_depth,
                cb_qp_offset,
                cr_qp_offset,
                slice_chroma_qp_offsets,
                weighted_pred,
                weighted_bipred,
                transquant_bypass,
                tiles,
                entropy_sync,
                loop_filter_across_slices,
                deblock,
                scaling,
                lists_modification,
                log2_parallel_merge_level: log2_parallel_merge_level as u8,
                slice_header_extension,
            },
        ))
    }
}

/// `beta_offset_div2`, `tc_offset_div2` (both in −6..=6).
pub(super) fn deblock_offsets(r: &mut BitReader<'_>) -> Result<(i32, i32), DecodeError> {
    let beta = r.read_se()?;
    let tc = r.read_se()?;
    if !(-6..=6).contains(&beta) || !(-6..=6).contains(&tc) {
        return Err(DecodeError::Malformed("deblocking offsets"));
    }
    Ok((beta, tc))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_scaling_factors_follow_table_7_6() {
        let lists = ScalingLists::default_lists();
        assert!(lists.factors(2, 0).iter().all(|&m| m == FLAT_SCALE));
        // The last diagonal entry of the 8x8 intra list is its corner.
        assert_eq!(lists.factors(3, 0)[63], 115);
        assert_eq!(lists.factors(3, 3)[63], 91);
        // 16x16 replicates each 8x8 entry 2x2; the DC is 16.
        let m = lists.factors(4, 0);
        assert_eq!((m[0], m[15 * 16 + 15], m[14 * 16 + 14]), (16, 115, 115));
    }

    #[test]
    fn uniform_tiles_split_as_equation_6_3() {
        assert_eq!(Tiles::boundaries(3, None, 10).unwrap(), vec![0, 3, 6, 10]);
        assert_eq!(
            Tiles::boundaries(3, Some(&[2, 5]), 10).unwrap(),
            vec![0, 2, 7, 10]
        );
        assert!(Tiles::boundaries(2, Some(&[10]), 10).is_err());
    }

    /// An SPS (§7.3.2.2) up to the coded size, 64x64, then `rest`.
    fn sps_prefix(rest: impl FnOnce(&mut nc_common::BitWriter)) -> Vec<u8> {
        let mut w = nc_common::BitWriter::new();
        w.put_bits(0, 4); // sps_video_parameter_set_id
        w.put_bits(0, 3); // sps_max_sub_layers_minus1
        w.put_bit(true); // sps_temporal_id_nesting_flag
        for _ in 0..3 {
            w.put_bits(0, 32); // profile_tier_level: 96 bits
        }
        w.put_ue(0); // sps_seq_parameter_set_id
        w.put_ue(1); // chroma_format_idc: 4:2:0
        w.put_ue(64);
        w.put_ue(64);
        rest(&mut w);
        w.put_bits(0, 32); // never reached
        while !w.is_byte_aligned() {
            w.put_bit(false);
        }
        w.into_bytes()
    }

    /// Values near `u32::MAX` used to wrap in `+ 4` and in the window sums
    /// (on a 32-bit `usize`, WASM) before any range check.
    #[test]
    fn sps_values_are_bounded_before_arithmetic() {
        let window = sps_prefix(|w| {
            w.put_bit(true); // conformance_window_flag
            for _ in 0..4 {
                w.put_ue(0x8000_0000);
            }
        });
        assert_eq!(
            Sps::parse(&window).err(),
            Some(DecodeError::Malformed("conf_win_offset"))
        );
        let poc_lsb = sps_prefix(|w| {
            w.put_bit(false); // conformance_window_flag
            w.put_ue(0); // bit_depth_luma_minus8
            w.put_ue(0); // bit_depth_chroma_minus8
            w.put_ue(u32::MAX - 1); // log2_max_pic_order_cnt_lsb_minus4
        });
        assert_eq!(
            Sps::parse(&poc_lsb).err(),
            Some(DecodeError::Malformed("log2_max_pic_order_cnt_lsb_minus4"))
        );
    }
}
