//! H.266 §7.3.2 parameter sets and §7.3.7/8 picture/slice headers.
//!
//! The field order was checked against `VVenC`'s `VLCWriter` (BSD-3-Clause-Clear;
//! see `THIRD_PARTY.md`). Optional coding tools are disabled explicitly.
use crate::chroma_qp::ChromaQpTable;
pub(crate) use crate::coding_tree::Partitioning;
use nc_common::{
    BitWriter,
    decode::{BitReader, DecodeError, Result},
    nal,
};
use nc_encoder::{FrameRate, TimingSignal};

/// §7.4.3.4: `CtbLog2SizeY` is `sps_log2_ctu_size_minus5` + 5; a larger
/// CTU also signals `sps_max_luma_transform_size_64_flag` (§7.3.2.4).
pub(crate) const MIN_CTU_LOG2: u8 = 5;
/// log2 of `MinQtSizeY` (8x8): with multi-type trees off, the smallest
/// node, which carries no `split_cu_flag` (§7.3.11.4).
pub(crate) const MIN_QT_LOG2: u8 = 3;
/// log2 of the CTU sizes we code: 32x32 and 64x64.
pub(crate) const CTU_LOG2_RANGE: std::ops::RangeInclusive<u8> = MIN_CTU_LOG2..=6;
/// §7.4.3.4: `pps_pic_width/height_in_luma_samples` are multiples of
/// Max(8, `MinCbSizeY`); our `MinCbSizeY` is 4.
pub(crate) const CODED_SIZE_MULTIPLE: usize = 8;
pub(crate) const MAX_QP: u32 = 63;
/// Bounded native subset geometry, within the signalled level.
pub(crate) const MAX_DIMENSION: usize = 4096;
pub(crate) const SPS_NUT: u8 = 15;
pub(crate) const PPS_NUT: u8 = 16;
/// Prefix adaptation parameter set (ALF filters, `alf_aps.rs`).
pub(crate) const PREFIX_APS_NUT: u8 = 17;
pub(crate) const IDR_N_LP: u8 = 8;
pub(crate) const TRAIL_NUT: u8 = 0;
const MAIN_10_PROFILE: u32 = 1;
const LEVEL_6: u32 = 96;
const POC_BITS: u32 = 8;

/// Stream timing in the SPS without HRD buffer models
/// (`general_timing_hrd_parameters()` and `ols_timing_hrd_parameters()`,
/// H.266 §7.3.5.1-2): the clock tick and whether pictures are one tick
/// apart. Informative: it never changes decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Timing {
    /// `num_units_in_tick`: one tick per nominal picture.
    pub num_units_in_tick: u32,
    /// `time_scale`: ticks per `num_units_in_tick` seconds.
    pub time_scale: u32,
    /// `fixed_pic_rate_general_flag`: pictures are exactly one tick apart.
    /// Off for variable-rate captures (timestamps, skipped frames).
    pub fixed_rate: bool,
}

impl Timing {
    /// The timing the engine's signalling asks for; none for
    /// [`TimingSignal::Nominal`], which keeps the historical headers.
    pub(crate) fn from_signal(signal: TimingSignal, rate: FrameRate) -> Option<Self> {
        (signal != TimingSignal::Nominal).then_some(Self {
            num_units_in_tick: rate.den,
            time_scale: rate.num,
            fixed_rate: signal == TimingSignal::Constant,
        })
    }
}

/// The picture geometry a stream codes: the visible size, the coded size
/// (whole minimum coding units; CTUs crossing the right or bottom edge are
/// split implicitly, §7.3.11.4), the CTU size and the inter slices'
/// coding-tree constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Geometry {
    pub width: usize,
    pub height: usize,
    pub coded_width: usize,
    pub coded_height: usize,
    pub ctu_log2: u8,
    pub partitioning: Partitioning,
}

impl Geometry {
    /// The geometry the encoder codes for a `width` x `height` input.
    pub(crate) fn new(width: usize, height: usize, ctu_log2: u8) -> Self {
        debug_assert!(CTU_LOG2_RANGE.contains(&ctu_log2));
        Self {
            width,
            height,
            coded_width: width.next_multiple_of(CODED_SIZE_MULTIPLE),
            coded_height: height.next_multiple_of(CODED_SIZE_MULTIPLE),
            ctu_log2,
            partitioning: Partitioning::QUADTREE,
        }
    }

    /// This geometry with inter slices partitioned by `partitioning`.
    pub(crate) fn with_partitioning(self, partitioning: Partitioning) -> Self {
        Self {
            partitioning,
            ..self
        }
    }

    pub(crate) fn ctu_size(&self) -> usize {
        1 << self.ctu_log2
    }
}

pub(crate) fn nal(kind: u8, rbsp: &[u8], out: &mut Vec<u8>) {
    temporal_nal(kind, 0, rbsp, out);
}

/// H.266 §7.3.1.2: all NAL units of a picture carry its temporal id.
pub(crate) fn temporal_nal(kind: u8, layer: u8, rbsp: &[u8], out: &mut Vec<u8>) {
    nal::write_nal_unit(&[0, (kind << 3) | (layer + 1)], rbsp, out);
}

/// The optional coding tools a stream's SPS enables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools, reason = "the SPS's flags")]
pub(crate) struct Tools {
    /// `sps_alf_enabled_flag` (luma: fixed sets or APS filters, `alf.rs`).
    pub alf: bool,
    /// `sps_isp_enabled_flag` (intra sub-partitions: `isp.rs`).
    pub isp: bool,
    /// `sps_dep_quant_enabled_flag`; every slice then uses dependent
    /// quantisation (`dep_quant.rs`) instead of sign data hiding.
    pub dep_quant: bool,
    /// `sps_affine_enabled_flag` and its model (`affine.rs`).
    pub affine: AffineTool,
    /// `sps_temporal_mvp_enabled_flag` (collocated candidates: `temporal.rs`;
    /// no sub-block temporal candidates).
    pub temporal_mvp: bool,
    /// `MaxNumMergeCand` (`motion::MergeList`), 1 to 6.
    pub merge_candidates: usize,
    /// `sps_mts_enabled_flag` and which prediction types signal `mts_idx`
    /// (`mts.rs`); `None`: every transform is a DCT-II.
    pub mts: Option<crate::mts::Mts>,
    /// `sps_max_luma_transform_size_64_flag` (64x64 CTUs only): a 64x64 CU
    /// codes one 64-point transform unit (`transform_tree`).
    pub large_transforms: bool,
}

impl Tools {
    /// log2 of `MaxTbSizeY` (§7.4.3.4).
    pub(crate) fn max_tb_log2(&self) -> u8 {
        crate::transform_tree::MAX_TB_LOG2 + u8::from(self.large_transforms)
    }
}

/// Two merge candidates, as HEVC's backend: the second costs one context bin.
pub(crate) const DEFAULT_MERGE_CANDIDATES: usize = 2;

impl Default for Tools {
    fn default() -> Self {
        Self {
            alf: false,
            isp: false,
            dep_quant: false,
            affine: AffineTool::Off,
            temporal_mvp: false,
            merge_candidates: DEFAULT_MERGE_CANDIDATES,
            mts: None,
            large_transforms: false,
        }
    }
}

/// The affine motion an SPS enables (§7.4.3.4 `sps_affine_enabled_flag`,
/// `sps_6param_affine_enabled_flag`); sub-block merge lists of
/// `affine_candidates::MAX_SUBBLOCK_MERGE`, no affine AMVR, no PROF.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum AffineTool {
    #[default]
    Off,
    /// 4-parameter models only.
    FourParameter,
    /// 4- and 6-parameter models.
    SixParameter,
}

impl AffineTool {
    pub(crate) fn enabled(self) -> bool {
        self != Self::Off
    }

    /// `sps_6param_affine_enabled_flag`.
    pub(crate) fn six(self) -> bool {
        self == Self::SixParameter
    }
}

/// The SPS up to the VUI flag, around the chroma QP table: [`sps_head`],
/// the table, [`sps_tools`], then [`sps_timing`].
pub(crate) fn sps_prefix(
    geometry: &Geometry,
    chroma_qp: &ChromaQpTable,
    tools: Tools,
    timing: Option<Timing>,
) -> BitWriter {
    let mut w = sps_head(geometry, tools);
    chroma_qp.write(&mut w);
    sps_tools(&mut w, tools);
    sps_timing(&mut w, timing, 1);
    w
}

/// The SPS up to `sps_same_qp_table_for_chroma_flag` (§7.3.2.4), with the
/// transform flags of `tools` (the largest transform, MTS).
pub(crate) fn sps_head(geometry: &Geometry, tools: Tools) -> BitWriter {
    sps_layered_head(geometry, tools, 1)
}

fn sps_layered_head(geometry: &Geometry, tools: Tools, layers: u8) -> BitWriter {
    let &Geometry {
        width,
        height,
        coded_width: coded_w,
        coded_height: coded_h,
        ctu_log2,
        partitioning,
    } = geometry;
    let mut w = BitWriter::new();
    w.put_bits(0, 4); // sps_seq_parameter_set_id
    w.put_bits(0, 4); // sps_video_parameter_set_id: no VPS
    w.put_bits(u32::from(layers - 1), 3); // max sublayers minus one
    w.put_bits(1, 2); // 4:2:0
    w.put_bits(u32::from(ctu_log2 - MIN_CTU_LOG2), 2); // sps_log2_ctu_size_minus5
    w.put_bit(true); // PTL/DPB/HRD present
    w.put_bits(MAIN_10_PROFILE, 7);
    w.put_bit(false); // main tier
    w.put_bits(LEVEL_6, 8);
    w.put_bit(true); // frame only
    w.put_bit(false); // multilayer
    w.put_bit(false); // general constraints absent
    w.byte_align_zero();
    for _ in 1..layers {
        w.put_bit(false); // sublayer level inherited from highest layer
    }
    w.byte_align_zero();
    w.put_bits(0, 8); // subprofiles
    w.put_bit(false); // GDR
    w.put_bit(false); // reference picture resampling
    w.put_ue(coded_w as u32);
    w.put_ue(coded_h as u32);
    let cropped = (coded_w, coded_h) != (width, height);
    w.put_bit(cropped);
    if cropped {
        w.put_ue(0);
        w.put_ue(((coded_w - width) / 2) as u32);
        w.put_ue(0);
        w.put_ue(((coded_h - height) / 2) as u32);
    }
    w.put_bit(false); // subpicture info
    w.put_ue(0); // bitdepth minus eight
    w.put_bit(false); // entropy sync
    w.put_bit(false); // entry points
    w.put_bits(POC_BITS - 4, 4);
    w.put_bit(false); // POC MSB
    w.put_bits(0, 4); // extra PH/SH bytes
    if layers > 1 {
        w.put_bit(false); // sublayer DPB parameters inherited
    }
    w.put_ue(nc_encoder::reference_slots(layers) as u32); // references plus current picture
    w.put_ue(0); // max_num_reorder_pics
    w.put_ue(0); // max_latency_increase_plus1
    let min_cb_log2 = partitioning.min_cb_log2;
    w.put_ue(u32::from(min_cb_log2 - 2)); // sps_log2_min_luma_coding_block_size_minus2
    w.put_bit(false); // partition override
    w.put_ue(u32::from(MIN_QT_LOG2 - min_cb_log2)); // min QT 8x8
    w.put_ue(0); // intra MTT depth
    w.put_bit(false); // dual tree
    w.put_ue(u32::from(MIN_QT_LOG2 - min_cb_log2)); // inter min QT 8x8
    write_inter_partitioning(&mut w, partitioning);
    write_transform_sizes(&mut w, ctu_log2, tools.large_transforms);
    write_mts(&mut w, tools.mts);
    write_after_mts(&mut w);
    w
}

/// [`sps_head`] after the inter partitioning fields, before MTS.
fn write_transform_sizes(w: &mut BitWriter, ctu_log2: u8, large_transforms: bool) {
    debug_assert!(!large_transforms || ctu_log2 > MIN_CTU_LOG2);
    if ctu_log2 > MIN_CTU_LOG2 {
        // sps_max_luma_transform_size_64_flag (MaxTbLog2SizeY is 5 plus
        // it): without it 64x64 CUs split into 32x32 transform units
        // (`transform_tree`).
        w.put_bit(large_transforms);
    }
    w.put_bit(false); // transform skip
}

/// `sps_max_luma_transform_size_64_flag` of an SPS whose fields before it
/// are those of `geometry` (absent, so 0, with 32x32 CTUs).
pub(crate) fn read_large_transforms(bytes: &[u8], geometry: &Geometry) -> Result<bool> {
    // The flag and the transform skip flag end the fields before MTS.
    const FLAG_FROM_MTS: usize = 2;
    if geometry.ctu_log2 == MIN_CTU_LOG2 {
        return Ok(false);
    }
    let mut r = BitReader::new(bytes);
    r.skip_bits(sps_bits_before_mts(geometry) - FLAG_FROM_MTS)?;
    r.read_bit()
}

/// `sps_mts_enabled_flag` and, when set, the explicit intra and inter flags.
fn write_mts(w: &mut BitWriter, mts: Option<crate::mts::Mts>) {
    w.put_bit(mts.is_some());
    if let Some(mts) = mts {
        w.put_bit(mts.intra); // sps_explicit_mts_intra_enabled_flag
        w.put_bit(mts.inter); // sps_explicit_mts_inter_enabled_flag
    }
}

/// The MTS flags [`write_mts`] wrote at `r`.
pub(crate) fn read_mts(r: &mut BitReader<'_>) -> Result<Option<crate::mts::Mts>> {
    if !r.read_bit()? {
        return Ok(None);
    }
    Ok(Some(crate::mts::Mts {
        intra: r.read_bit()?,
        inter: r.read_bit()?,
    }))
}

/// [`sps_head`] after the MTS flags.
fn write_after_mts(w: &mut BitWriter) {
    w.put_bit(false); // LFNST
    w.put_bit(false); // joint CbCr
    w.put_bit(true); // same chroma QP table
}

/// The inter slices' multi-type tree fields of the SPS (§7.3.2.4):
/// `sps_max_mtt_hierarchy_depth_inter_slice`, then (when nonzero) the
/// largest binary and ternary split sizes over `MinQtSizeY`.
fn write_inter_partitioning(w: &mut BitWriter, partitioning: Partitioning) {
    w.put_ue(u32::from(partitioning.mtt_depth));
    if partitioning.mtt_depth > 0 {
        w.put_ue(u32::from(partitioning.bt_diff));
        w.put_ue(u32::from(partitioning.tt_diff));
    }
}

/// Bits of [`sps_head`] of `geometry` before the inter slices'
/// multi-type tree fields.
pub(crate) fn sps_bits_before_partitioning(geometry: &Geometry) -> usize {
    sps_bits_before_mts(geometry) - partitioning_bits(geometry)
}

/// Bits of [`sps_head`] of `geometry` before `sps_mts_enabled_flag`.
pub(crate) fn sps_bits_before_mts(geometry: &Geometry) -> usize {
    // Everything after the MTS flags is the same for every geometry.
    let mut after = BitWriter::new();
    write_mts(&mut after, None);
    write_after_mts(&mut after);
    sps_head(geometry, Tools::default()).bit_len() - after.bit_len()
}

/// Bits of the inter partitioning fields and the transform sizes after them.
fn partitioning_bits(geometry: &Geometry) -> usize {
    let mut fields = BitWriter::new();
    write_inter_partitioning(&mut fields, geometry.partitioning);
    write_transform_sizes(&mut fields, geometry.ctu_log2, false);
    fields.bit_len()
}

/// The SPS's fixed tool flags after the chroma QP table, up to
/// `sps_timing_hrd_params_present_flag`.
pub(crate) fn sps_tools(w: &mut BitWriter, tools: Tools) {
    w.put_bit(true); // SAO
    w.put_bit(tools.alf); // sps_alf_enabled_flag
    if tools.alf {
        w.put_bit(false); // sps_ccalf_enabled_flag
    }
    w.put_bit(false); // LMCS
    w.put_bit(false); // weighted prediction
    w.put_bit(false); // weighted biprediction
    w.put_bit(false); // long term references
    w.put_bit(false); // IDR RPL
    w.put_bit(true); // RPL1 copies RPL0
    w.put_ue(0); // no SPS RPLs
    w.put_bit(false); // wraparound
    w.put_bit(tools.temporal_mvp); // sps_temporal_mvp_enabled_flag
    if tools.temporal_mvp {
        w.put_bit(false); // sps_sbtmvp_enabled_flag
    }
    w.put_bit(true); // AMVR (sps_amvr_enabled_flag)
    w.put_bit(false); // BDOF
    w.put_bit(false); // SMVD
    w.put_bit(false); // DMVR
    w.put_bit(false); // MMVD
    // sps_six_minus_max_num_merge_cand
    w.put_ue((crate::motion::MAX_MERGE_CANDIDATES - tools.merge_candidates) as u32);
    w.put_bit(false); // SBT
    w.put_bit(tools.affine.enabled()); // sps_affine_enabled_flag
    if tools.affine.enabled() {
        // sps_five_minus_max_num_subblock_merge_cand
        w.put_ue((5 - crate::affine_candidates::MAX_SUBBLOCK_MERGE) as u32);
        w.put_bit(tools.affine.six()); // sps_6param_affine_enabled_flag
        w.put_bit(false); // sps_affine_amvr_enabled_flag (AMVR is on)
        w.put_bit(false); // sps_affine_prof_enabled_flag
    }
    w.put_bit(false); // BCW
    w.put_bit(false); // CIIP
    if tools.merge_candidates >= 2 {
        w.put_bit(false); // sps_gpm_enabled_flag (present from two merge candidates)
    }
    w.put_ue(0); // parallel merge level minus two
    w.put_bit(tools.isp); // sps_isp_enabled_flag
    w.put_bit(false); // MRL
    w.put_bit(false); // MIP
    w.put_bit(false); // CCLM
    w.put_bit(false); // horizontal chroma collocation
    w.put_bit(true); // vertical chroma collocation
    w.put_bit(false); // palette
    w.put_bit(false); // IBC
    w.put_bit(false); // LADF
    w.put_bit(false); // scaling lists
    w.put_bit(tools.dep_quant); // sps_dep_quant_enabled_flag
    w.put_bit(true); // sign hiding (sps_sign_data_hiding_enabled_flag)
    w.put_bit(false); // virtual boundaries
}

/// The SPS from `sps_timing_hrd_params_present_flag` (present: the PTL/DPB/HRD
/// flag is set) up to the VUI flag, for `layers` temporal sublayers.
fn sps_timing(w: &mut BitWriter, timing: Option<Timing>, layers: u8) {
    w.put_bit(timing.is_some()); // sps_timing_hrd_params_present_flag
    if let Some(timing) = timing {
        // general_timing_hrd_parameters() (§7.3.5.1), no NAL or VCL HRD.
        w.put_bits(timing.num_units_in_tick, 32);
        w.put_bits(timing.time_scale, 32);
        w.put_bit(false); // general_nal_hrd_params_present_flag
        w.put_bit(false); // general_vcl_hrd_params_present_flag
        if layers > 1 {
            // One set for the highest sublayer, inferred for the lower ones.
            w.put_bit(false); // sps_sublayer_cpb_params_present_flag
        }
        // ols_timing_hrd_parameters() (§7.3.5.2) for that sublayer.
        w.put_bit(timing.fixed_rate); // fixed_pic_rate_general_flag
        if timing.fixed_rate {
            // fixed_pic_rate_within_cvs_flag is inferred 1: one tick per picture.
            w.put_ue(0); // elemental_duration_in_tc_minus1
        } else {
            w.put_bit(false); // fixed_pic_rate_within_cvs_flag
        }
    }
    w.put_bit(false); // field sequence
}

/// The timing [`sps_timing`] wrote for one sublayer at `r`, which it leaves at
/// the VUI flag. Like [`read_sps_tools`] it reads, not checks: the decoder
/// compares the whole SPS with the one it would write.
pub(crate) fn read_sps_timing(r: &mut BitReader<'_>) -> Result<Option<Timing>> {
    let timing = if r.read_bit()? {
        let num_units_in_tick = r.read_bits(32)?;
        let time_scale = r.read_bits(32)?;
        r.skip_bits(2)?; // NAL and VCL HRD flags
        let fixed_rate = r.read_bit()?;
        if fixed_rate || r.read_bit()? {
            r.read_ue()?; // elemental_duration_in_tc_minus1
        }
        Some(Timing {
            num_units_in_tick,
            time_scale,
            fixed_rate,
        })
    } else {
        None
    };
    r.skip_bits(1)?; // field sequence
    Ok(timing)
}

/// The tools [`sps_tools`] wrote at `r`, which it leaves at
/// `sps_timing_hrd_params_present_flag`.
/// Fields this subset fixes are skipped, not checked: the decoder compares
/// the whole SPS with the one these tools write.
pub(crate) fn read_sps_tools(r: &mut BitReader<'_>) -> Result<Tools> {
    let mut tools = Tools::default();
    r.skip_bits(1)?; // SAO
    tools.alf = r.read_bit()?;
    if tools.alf {
        r.skip_bits(1)?; // CC-ALF
    }
    r.skip_bits(6)?; // LMCS, weighted (bi)prediction, LTR, IDR RPL, RPL1 copies RPL0
    r.read_ue()?; // SPS RPLs
    r.skip_bits(1)?; // wraparound
    tools.temporal_mvp = r.read_bit()?;
    if tools.temporal_mvp {
        r.skip_bits(1)?; // sub-block TMVP
    }
    r.skip_bits(5)?; // AMVR, BDOF, SMVD, DMVR, MMVD
    let fewer = r.read_ue()? as usize;
    if fewer >= crate::motion::MAX_MERGE_CANDIDATES {
        return Err(DecodeError::Malformed("VVC MaxNumMergeCand"));
    }
    tools.merge_candidates = crate::motion::MAX_MERGE_CANDIDATES - fewer;
    r.skip_bits(1)?; // SBT
    if r.read_bit()? {
        r.read_ue()?; // sub-block merge candidates
        tools.affine = if r.read_bit()? {
            AffineTool::SixParameter
        } else {
            AffineTool::FourParameter
        };
        r.skip_bits(2)?; // affine AMVR, PROF
    }
    r.skip_bits(2)?; // BCW, CIIP
    if tools.merge_candidates >= 2 {
        r.skip_bits(1)?; // GPM
    }
    r.read_ue()?; // parallel merge level
    tools.isp = r.read_bit()?;
    // MRL, MIP, CCLM, chroma collocation (2), palette, IBC, LADF, scaling lists.
    r.skip_bits(9)?;
    tools.dep_quant = r.read_bit()?;
    r.skip_bits(2)?; // sign hiding, virtual boundaries
    Ok(tools)
}

pub(crate) fn sps(
    geometry: &Geometry,
    colour: nc_common::ColourInfo,
    chroma_qp: &ChromaQpTable,
    tools: Tools,
    timing: Option<Timing>,
) -> Vec<u8> {
    sps_layered(geometry, colour, chroma_qp, tools, timing, 1)
}

/// SPS for the shared engine's low-delay temporal structure.
pub(crate) fn sps_layered(
    geometry: &Geometry,
    colour: nc_common::ColourInfo,
    chroma_qp: &ChromaQpTable,
    tools: Tools,
    timing: Option<Timing>,
    layers: u8,
) -> Vec<u8> {
    let mut w = if layers == 1 {
        sps_prefix(geometry, chroma_qp, tools, timing)
    } else {
        let mut w = sps_layered_head(geometry, tools, layers);
        chroma_qp.write(&mut w);
        sps_tools(&mut w, tools);
        sps_timing(&mut w, timing, layers);
        w
    };
    w.put_bit(colour.is_signalled());
    if colour.is_signalled() {
        let mut vui = BitWriter::new();
        vui.put_bit(true); // progressive source
        vui.put_bit(false); // interlaced source
        vui.put_bit(false); // no packing constraint
        vui.put_bit(false); // no projection constraint
        vui.put_bit(false); // aspect ratio absent
        vui.put_bit(false); // overscan absent
        vui.put_bit(true); // colour description, also needed to signal full range
        nc_common::colour::write_code_points(
            &mut vui,
            colour.description.unwrap_or(nc_common::ColourDescription {
                primaries: 2,
                transfer: 2,
                matrix: 2,
            }),
        );
        vui.put_bit(colour.full_range);
        vui.put_bit(false); // chroma location absent
        if !vui.is_byte_aligned() {
            vui.trailing_bits();
        }
        let bytes = vui.into_bytes();
        w.put_ue(bytes.len() as u32 - 1);
        w.byte_align_zero();
        w.put_bytes(&bytes);
    }
    w.put_bit(false); // extensions
    w.trailing_bits();
    w.into_bytes()
}

/// The PPS. `block_qp`: `pps_cu_qp_delta_enabled_flag`, block quantizers
/// coded as `cu_qp_delta` in CTB quantization groups (`quant_group.rs`).
pub(crate) fn pps(geometry: &Geometry, block_qp: bool) -> Vec<u8> {
    let (width, height) = (geometry.coded_width, geometry.coded_height);
    let mut w = BitWriter::new();
    w.put_bits(0, 6); // PPS id
    w.put_bits(0, 4); // SPS id
    w.put_bit(false); // mixed NAL types
    w.put_ue(width as u32);
    w.put_ue(height as u32);
    w.put_bit(false); // crop inherited from SPS
    w.put_bit(false); // scaling window
    w.put_bit(false); // output flag present
    w.put_bit(true); // no picture partition
    w.put_bit(false); // subpicture ID mapping
    w.put_bit(false); // CABAC init present
    w.put_ue(0); // default active L0 minus one
    w.put_ue(0); // default active L1 minus one
    w.put_bit(false); // RPL1 index present
    w.put_bit(false); // weighted prediction
    w.put_bit(false); // weighted biprediction
    w.put_bit(false); // wraparound
    w.put_se(0); // initial QP minus 26
    w.put_bit(block_qp); // pps_cu_qp_delta_enabled_flag
    w.put_bit(false); // chroma tool offsets
    w.put_bit(true); // deblocking control present
    w.put_bit(false); // deblocking override
    w.put_bit(false); // deblocking enabled
    w.put_se(0); // beta offset
    w.put_se(0); // tc offset
    w.put_bit(false); // PH extension
    w.put_bit(false); // SH extension
    w.put_bit(false); // PPS extension
    w.trailing_bits();
    w.into_bytes()
}

/// A slice's luma ALF (`sh_alf_enabled_flag` and the APS it names; Cb and
/// Cr always off).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SliceAlf {
    Off,
    /// CTBs may choose a fixed set or, with `aps`, the filters of that
    /// luma ALF APS (`sh_alf_aps_id_luma[0]`, `alf_aps.rs`).
    On {
        aps: Option<u8>,
    },
}

/// The slice header with its embedded picture header. `alf`: `None` when
/// the SPS has no ALF; `block_qp`: the PPS enables `cu_qp_delta`;
/// `temporal_mvp`: `ph_temporal_mvp_enabled_flag` of an inter picture,
/// `None` when the SPS has no TMVP (the collocated picture is the only
/// reference, so no `sh_collocated_*` syntax follows).
pub(crate) fn slice(
    qp: u32,
    inter: bool,
    poc: u8,
    alf: Option<SliceAlf>,
    block_qp: bool,
    temporal_mvp: Option<bool>,
) -> Vec<u8> {
    slice_layered(
        qp,
        inter,
        poc,
        alf,
        block_qp,
        temporal_mvp,
        &crate::references::Syntax::FLAT,
    )
}

/// Picture reference syntax follows the slots selected by the shared engine.
pub(crate) fn slice_layered(
    qp: u32,
    inter: bool,
    poc: u8,
    alf: Option<SliceAlf>,
    block_qp: bool,
    temporal_mvp: Option<bool>,
    references: &crate::references::Syntax,
) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put_bit(true); // PH embedded in slice header
    w.put_bit(!inter); // IRAP
    w.put_bit(inter && references.non_reference); // ph_non_ref_pic_flag
    if !inter {
        w.put_bit(false);
    } // not GDR
    w.put_bit(inter); // inter slices allowed
    if inter {
        w.put_bit(true);
    } // intra slices allowed
    w.put_ue(0); // PPS id
    w.put_bits(u32::from(poc), POC_BITS);
    if block_qp {
        // ph_cu_qp_delta_subdiv_intra_slice, then (P pictures)
        // ph_cu_qp_delta_subdiv_inter_slice: quantization groups are CTBs
        // (§7.3.2.8, §7.4.3.8).
        w.put_ue(QUANT_GROUP_SUBDIV);
        if inter {
            w.put_ue(QUANT_GROUP_SUBDIV);
        }
    }
    if inter {
        if let Some(temporal) = temporal_mvp {
            w.put_bit(temporal); // ph_temporal_mvp_enabled_flag
        }
        w.put_bit(false); // ph_mvd_l1_zero_flag; no RPL in PH
        w.put_ue(1); // P slice
        slice_alf(&mut w, alf);
        w.put_ue(references.count as u32);
        let mut previous = 0;
        for &distance in &references.distances[..references.count] {
            w.put_ue(distance - previous - 1); // cumulative short-term POC delta
            w.put_bit(true);
            previous = distance;
        }
        w.put_ue(0); // empty local RPL1
        if references.count > 1 {
            w.put_bit(false); // default one active reference; retain the other in DPB
        }
    } else {
        w.put_bit(false); // no_output_of_prior_pics_flag
        slice_alf(&mut w, alf);
    }
    w.put_se(qp as i32 - 26);
    w.put_bit(true); // luma SAO enabled
    w.put_bit(true); // chroma SAO enabled
    // sh_dep_quant_used_flag 1 when the SPS enables dependent quantisation
    // (then no sign hiding flag follows), else sh_sign_data_hiding_used_flag
    // 1: one set bit either way.
    w.put_bit(true);
    w.trailing_bits(); // byte_alignment before CABAC
    w.into_bytes()
}

/// `sh_alf_enabled_flag` and, when set, the luma APS ids (none or one),
/// then `sh_alf_cb_enabled_flag` and `sh_alf_cr_enabled_flag` 0 (§7.3.7).
fn slice_alf(w: &mut BitWriter, alf: Option<SliceAlf>) {
    match alf {
        None => {}
        Some(SliceAlf::Off) => w.put_bit(false),
        Some(SliceAlf::On { aps }) => {
            w.put_bit(true);
            w.put_bits(u32::from(aps.is_some()), APS_ID_BITS); // sh_num_alf_aps_ids_luma
            if let Some(id) = aps {
                w.put_bits(u32::from(id), APS_ID_BITS); // sh_alf_aps_id_luma[0]
            }
            w.put_bit(false); // sh_alf_cb_enabled_flag
            w.put_bit(false); // sh_alf_cr_enabled_flag
        }
    }
}

/// `ph_cu_qp_delta_subdiv_*_slice`: 0, one quantization group per CTB.
pub(crate) const QUANT_GROUP_SUBDIV: u32 = 0;

/// Bits of `sh_num_alf_aps_ids_luma` and of each `sh_alf_aps_id_luma`.
pub(crate) const APS_ID_BITS: u32 = 3;

/// Read the `TemporalId` of the first VCL NAL unit in an Annex-B access unit.
///
/// # Errors
/// Returns an error for invalid NAL headers or an access unit without a slice.
pub fn access_unit_layer(access_unit: &[u8]) -> nc_common::decode::Result<u8> {
    use nc_common::decode::{DecodeError, annex_b_units};
    for unit in annex_b_units(access_unit)? {
        if unit.len() < 2 || unit[0] != 0 || unit[1].trailing_zeros() >= 3 {
            return Err(DecodeError::Malformed("VVC NAL header"));
        }
        if unit[1] >> 3 < 12 {
            return Ok((unit[1] & 7) - 1);
        }
    }
    Err(DecodeError::Malformed("VVC access unit without a slice"))
}
