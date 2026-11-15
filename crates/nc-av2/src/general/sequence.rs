//! The complete AV2 sequence header (`sequence_header_obu`), every field of
//! AVM v1.0.0, for 8-bit 4:2:0 streams.
//!
//! Field order follows dav2d `obu.c` `parse_seq_hdr` and AVM
//! `av2_read_sequence_header_obu` with its `read_sequence_*_group_tool_flags`
//! groups. [`crate::sequence`] is the encoder's subset of the same syntax.
use super::bits::{flag, signed, uniform};
use super::tiles::{self, TileGeometry, TileLayout};
use nc_common::decode::{BitReader, DecodeError, Result};

/// Most segments of a frame with `enable_ext_seg` (AVM `MAX_SEGMENTS`).
pub const MAX_SEGMENTS: usize = 16;
/// Segments without the extension.
const BASE_SEGMENTS: usize = 8;
/// Clamp of a segment's quantizer delta (AVM `av2_read_seg_info`: ±`MAXQ`
/// extended range for 12-bit).
const SEGMENT_DELTA_Q_LIMIT: i32 = 351;
/// Bits of a segment's quantizer delta, sign included.
const SEGMENT_DELTA_Q_BITS: u32 = 10;
/// Offset of the 5-bit base quantizer deltas (AVM `DELTA_DCQUANT_MIN`).
const BASE_DELTA_Q_MIN: i8 = -23;
/// Default reference slots (AVM `REF_FRAMES`).
const DEFAULT_REFERENCE_SLOTS: u8 = 8;
/// Smallest `df_par_bits` (the field codes the excess).
const MIN_DEBLOCK_PARAM_BITS: u8 = 2;
/// Number of `def_max_drl_bits` and `def_max_bvp_drl_bits` values.
const DRL_BITS_CHOICES: u32 = 5;
const BVP_DRL_BITS_CHOICES: u32 = 3;

/// A tool the sequence turns off, on, or leaves to each frame (AVM's
/// `force_*` three-way fields).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Tri {
    /// Never used.
    #[default]
    Off,
    /// Always used.
    On,
    /// Each frame signals it.
    Adaptive,
}

/// Per-segment features (`seg_info`): quantizer deltas and the skip and
/// global-motion flags, as bit masks over segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SegmentFeatures {
    /// Each segment's quantizer delta.
    pub delta_q: [i16; MAX_SEGMENTS],
    /// Segments with a quantizer delta.
    pub delta_q_mask: u16,
    /// Segments whose blocks are skipped.
    pub skip_mask: u16,
    /// Segments whose blocks use global motion.
    pub global_mv_mask: u16,
}

impl SegmentFeatures {
    /// Parse `seg_info` for `segments` segments (dav2d `parse_seg_info`).
    /// # Errors
    /// Truncation.
    pub fn parse(bits: &mut BitReader<'_>, segments: usize) -> Result<Self> {
        let mut features = Self::default();
        for segment in 0..segments {
            let mask = 1 << segment;
            if bits.read_bit()? {
                features.delta_q_mask |= mask;
                features.delta_q[segment] = signed(bits, SEGMENT_DELTA_Q_BITS)?
                    .clamp(-SEGMENT_DELTA_Q_LIMIT, SEGMENT_DELTA_Q_LIMIT)
                    as i16;
            }
            if bits.read_bit()? {
                features.skip_mask |= mask;
            }
            if bits.read_bit()? {
                features.global_mv_mask |= mask;
            }
        }
        Ok(features)
    }
}

/// Sequence-level segmentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Segmentation {
    /// Sixteen segments instead of eight (`enable_ext_seg`).
    pub extended: bool,
    /// The sequence carries segment features frames may reuse
    /// (`seg_info_present_flag`), adaptively chosen when `adaptive`.
    pub features: Option<SegmentFeatures>,
    /// `seg_info_adaptive_flag` (meaningful with `features`).
    pub adaptive: bool,
}

impl Segmentation {
    /// Segments a frame may use.
    #[must_use]
    pub fn count(&self) -> usize {
        if self.extended {
            MAX_SEGMENTS
        } else {
            BASE_SEGMENTS
        }
    }
}

/// The conformance window (`conf_win_*_offset`), in luma samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Crop {
    /// `conf_win_left_offset`.
    pub left: u32,
    /// `conf_win_right_offset`.
    pub right: u32,
    /// `conf_win_top_offset`.
    pub top: u32,
    /// `conf_win_bottom_offset`.
    pub bottom: u32,
}

/// The partition tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(clippy::struct_excessive_bools, reason = "the header's flags")]
pub struct PartitionTools {
    /// log2 of the superblock size: 6, 7 or 8.
    pub superblock_log2: u8,
    /// Intra frames code separate luma and chroma trees (`enable_sdp`), and
    /// inter frames' intra regions too (`enable_extended_sdp`).
    pub separate_trees: bool,
    /// `enable_extended_sdp`.
    pub extended_separate_trees: bool,
    /// Three-way partitions (`enable_ext_partitions`) and the uneven
    /// four-way ones (`enable_uneven_4way_partitions`).
    pub extended: bool,
    /// `enable_uneven_4way_partitions`.
    pub uneven_four_way: bool,
    /// log2 of the largest prediction-block aspect ratio: 1, 2 or 3.
    pub max_aspect_ratio_log2: u8,
}

/// The intra tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(clippy::struct_excessive_bools, reason = "the header's flags")]
pub struct IntraTools {
    /// Data-driven intra prediction (`enable_intra_dip`).
    pub dip: bool,
    /// `enable_intra_edge_filter`.
    pub edge_filter: bool,
    /// Multiple reference lines (`enable_mrls`).
    pub multiple_reference_lines: bool,
    /// Chroma from luma, with its downsampling filter (`cfl_ds_filter_index`).
    pub cfl: bool,
    /// `cfl_ds_filter_index`, 0..=2.
    pub cfl_downsample_filter: u8,
    /// Multi-hypothesis cross-component prediction (`enable_mhccp`).
    pub mhccp: bool,
    /// Intra bi-prediction (`enable_ibp`).
    pub ibp: bool,
}

/// Motion modes beyond translation a sequence allows (`seq_enabled_motion_modes`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(clippy::struct_excessive_bools, reason = "the header's flags")]
pub struct MotionModes {
    /// Inter-intra prediction.
    pub inter_intra: bool,
    /// Warps fitted to neighbouring motion.
    pub warp_causal: bool,
    /// Warps coded as parameter deltas.
    pub warp_delta: bool,
    /// Warps extended from a neighbour's.
    pub warp_extend: bool,
    /// Frames signal their own subset (`seq_frame_motion_modes_present_flag`).
    pub frame_signalled: bool,
    /// Warp deltas code six parameters (`enable_six_param_warp_delta`).
    pub six_parameter_warp_delta: bool,
}

/// Reference lists and buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(clippy::struct_excessive_bools, reason = "the header's flags")]
pub struct ReferenceTools {
    /// The reference MV bank (`enable_refmvbank`).
    pub mv_bank: bool,
    /// DRL reordering: 0 off, 1 or 2 (`enable_drl_reorder`).
    pub drl_reorder: u8,
    /// Frames name their references (`enable_explicit_ref_frame_map`).
    pub explicit_map: bool,
    /// Reference slots, 1..=16 (`ref_frames`).
    pub slots: u8,
    /// Bits of a long-term frame identifier
    /// (`number_of_bits_for_lt_frame_id`).
    pub long_term_id_bits: u8,
    /// `def_max_drl_bits` (1..=5) and whether frames override it.
    pub max_drl_bits: u8,
    /// `allow_frame_max_drl_bits`.
    pub frame_max_drl_bits: bool,
    /// `def_max_bvp_drl_bits` (1..=3) and whether frames override it.
    pub max_bvp_drl_bits: u8,
    /// `allow_frame_max_bvp_drl_bits`.
    pub frame_max_bvp_drl_bits: bool,
    /// Compound pairs of one reference a frame may list
    /// (`num_same_ref_compound`, 0..=3).
    pub same_reference_compounds: u8,
    /// Frames may refresh by short flags (`enable_short_refresh_frame_flags`).
    pub short_refresh_flags: bool,
}

impl ReferenceTools {
    /// Bits that index a reference slot (dav2d `ref_frames_log2`).
    #[must_use]
    pub fn slot_bits(&self) -> u32 {
        match self.slots {
            0..=2 => u32::from(self.slots) - 1,
            slots => 1 + u32::from(slots - 1).ilog2(),
        }
    }
}

/// The inter tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(clippy::struct_excessive_bools, reason = "the header's flags")]
pub struct InterTools {
    /// Motion modes beyond translation.
    pub motion_modes: MotionModes,
    /// `enable_masked_compound`.
    pub masked_compound: bool,
    /// Temporal motion fields (`enable_ref_frame_mvs`), in reduced mode
    /// when `reduced_motion_fields`.
    pub motion_fields: bool,
    /// `reduced_ref_frame_mvs_mode`.
    pub reduced_motion_fields: bool,
    /// Bits of a frame's order hint, 1..=16; 0 in single-picture sequences.
    pub order_hint_bits: u8,
    /// Temporal interpolated prediction: 0 off, 1 or 2 (`enable_tip`), with
    /// hole filling, an explicit quantizer and MV refinement.
    pub tip: u8,
    /// `enable_tip_hole_fill`.
    pub tip_hole_fill: bool,
    /// `enable_tip_explicit_qp`.
    pub tip_explicit_qp: bool,
    /// `enable_tip_refinemv`.
    pub tip_refine_mv: bool,
    /// Motion trajectories (`enable_mv_traj`).
    pub mv_trajectories: bool,
    /// Block-adaptive weighted prediction (`enable_bawp`).
    pub bawp: bool,
    /// Compound weighted prediction (`enable_cwp`).
    pub cwp: bool,
    /// Implicit masked blending (`enable_imp_msk_bld`).
    pub implicit_masked_blend: bool,
    /// Deblocking of sub-prediction-unit edges (`enable_lf_sub_pu`).
    pub deblock_sub_pu: bool,
    /// Optical-flow refinement: 0 off, 1 per block, 2 always, 3 per frame
    /// (`enable_opfl_refine`).
    pub optical_flow: u8,
    /// Decoder-side MV refinement (`enable_refinemv`).
    pub refine_mv: bool,
    /// Backward reference updating (`enable_bru`).
    pub bru: bool,
    /// `enable_adaptive_mvd`.
    pub adaptive_mvd: bool,
    /// `enable_mvd_sign_derive`.
    pub mvd_sign_derivation: bool,
    /// `enable_flex_mvres`.
    pub flexible_mv_resolution: bool,
    /// `enable_global_motion`.
    pub global_motion: bool,
}

/// The transform and coefficient tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(clippy::struct_excessive_bools, reason = "the header's flags")]
pub struct TransformTools {
    /// Forward skip coding (`enable_fsc`).
    pub fsc: bool,
    /// Identity transforms in intra blocks (`enable_idtx_intra`; implied by
    /// forward skip coding).
    pub intra_identity: bool,
    /// Secondary transforms in intra and inter blocks (`enable_ist`,
    /// `enable_inter_ist`).
    pub intra_secondary: bool,
    /// `enable_inter_ist`.
    pub inter_secondary: bool,
    /// Chroma codes DCT only (`enable_chroma_dctonly`).
    pub chroma_dct_only: bool,
    /// Data-driven transforms in inter blocks (`enable_inter_ddt`).
    pub inter_ddt: bool,
    /// `reduced_tx_part_set`.
    pub reduced_partition_set: bool,
    /// Cross-chroma component transforms (`enable_cctx`).
    pub cctx: bool,
    /// Trellis-coded quantisation: 0 off, 1 every frame, 2 per frame
    /// (`enable_tcq`).
    pub tcq: u8,
    /// `enable_parity_hiding`.
    pub parity_hiding: bool,
    /// Frames may start from averaged CDFs (`enable_avg_cdf`), of the
    /// second type when `average_cdf_type`.
    pub average_cdf: bool,
    /// `avg_cdf_type`.
    pub average_cdf_type: bool,
}

/// Sequence quantizer offsets and which deltas frames code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(clippy::struct_excessive_bools, reason = "the header's flags")]
pub struct QuantTools {
    /// `separate_uv_delta_q`: U and V code their own deltas.
    pub separate_uv_delta_q: bool,
    /// `equal_ac_dc_q`: DC steps equal AC steps, no DC deltas.
    pub equal_ac_dc_q: bool,
    /// `base_y_dc_delta_q`.
    pub base_y_dc_delta: i8,
    /// `y_dc_delta_q_enabled`: frames code a luma DC delta.
    pub y_dc_delta_enabled: bool,
    /// `base_uv_dc_delta_q`.
    pub base_uv_dc_delta: i8,
    /// `uv_dc_delta_q_enabled`.
    pub uv_dc_delta_enabled: bool,
    /// `base_uv_ac_delta_q`.
    pub base_uv_ac_delta: i8,
    /// `uv_ac_delta_q_enabled`.
    pub uv_ac_delta_enabled: bool,
}

/// The in-loop filters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(clippy::struct_excessive_bools, reason = "the header's flags")]
pub struct FilterTools {
    /// `disable_loopfilters_across_tiles`.
    pub disable_across_tiles: bool,
    /// `enable_cdef`.
    pub cdef: bool,
    /// The guided detail filter (`enable_gdf`), in superblock-sized units
    /// when `gdf_unit_is_superblock`.
    pub gdf: bool,
    /// `gdf_unit_matches_sb_size` (64x64 superblocks only).
    pub gdf_unit_is_superblock: bool,
    /// Loop restoration and, per plane class (luma, chroma), the filters
    /// it leaves out: bit 0 PC-Wiener, bit 1 non-separable Wiener (dav2d
    /// `rst_disable_mask`).
    pub restoration: bool,
    /// See `restoration`.
    pub restoration_disable_mask: [u8; 2],
    /// `enable_ccso`.
    pub ccso: bool,
    /// `ccso_unit_matches_sb_size`.
    pub ccso_unit_is_superblock: bool,
    /// CDEF on blocks whose transforms are skipped (`cdef_on_skip_txfm`).
    pub cdef_on_skipped: Tri,
    /// Bits of the deblocking offsets, 2..=5 (`df_par_bits_minus2 + 2`).
    pub deblock_param_bits: u8,
}

/// A parsed sequence header.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[allow(clippy::struct_excessive_bools, reason = "the header's flags")]
pub struct SequenceHeader {
    /// `seq_header_id`, 0..=15.
    pub id: u8,
    /// `seq_profile_idc`.
    pub profile: u8,
    /// `single_picture_header_flag`: one still picture, inter syntax left out.
    pub single_picture: bool,
    /// `seq_level_idx`.
    pub level: u8,
    /// `seq_tier`.
    pub high_tier: bool,
    /// `still_picture`.
    pub still_picture: bool,
    /// The layer configuration record the sequence follows (`seq_lcr_id`).
    pub layer_config_id: u8,
    /// Highest temporal and embedded layers, with each layer's dependency
    /// mask over itself and the lower ones (bit `n`: layer `n`).
    pub max_temporal_layer: u8,
    /// `max_mlayer_id`.
    pub max_embedded_layer: u8,
    /// Temporal layers' masks.
    pub temporal_dependencies: [u8; 4],
    /// Embedded layers' masks.
    pub embedded_dependencies: [u8; 8],
    /// Frames are output in coded order (`monotonic_output_order_flag`).
    pub monotonic_output: bool,
    /// `max_frame_width`, 1..=65536.
    pub max_width: u32,
    /// `max_frame_height`, 1..=65536.
    pub max_height: u32,
    /// Bits of a frame-size override.
    pub width_bits: u8,
    /// See `width_bits`.
    pub height_bits: u8,
    /// The conformance window, when enabled.
    pub crop: Option<Crop>,
    /// `max_initial_display_delay_minus_1`, when signalled.
    pub max_initial_display_delay: Option<u8>,
    /// Decoder model information, when present.
    pub decoder_model: Option<DecoderModel>,
    /// Partition tools.
    pub partition: PartitionTools,
    /// Segmentation.
    pub segmentation: Segmentation,
    /// Intra tools.
    pub intra: IntraTools,
    /// Inter tools.
    pub inter: InterTools,
    /// Reference lists and buffers.
    pub references: ReferenceTools,
    /// `force_screen_content_tools`.
    pub screen_content_tools: Tri,
    /// `force_integer_mv`.
    pub force_integer_mv: Tri,
    /// Transform and coefficient tools.
    pub transform: TransformTools,
    /// Quantizer offsets.
    pub quant: QuantTools,
    /// In-loop filters.
    pub filters: FilterTools,
    /// A tile layout frames may reuse; `tile_layout_fixed` when every
    /// frame does (`seq_tile_info_present_flag` and its `allow_tile_info_change`
    /// complement).
    pub tile_layout: Option<TileLayout>,
    /// See `tile_layout`.
    pub tile_layout_fixed: bool,
    /// `film_grain_params_present`.
    pub film_grain: bool,
}

/// `decoder_model_info`: timing of the hypothetical decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecoderModel {
    /// `num_units_in_decoding_tick`.
    pub units_in_decoding_tick: u32,
    /// Decoder and encoder buffer delays, when signalled.
    pub buffer_delays: Option<(u32, u32)>,
}

/// Highest profile index of AVM v1.0.0.
const MAX_PROFILE: u32 = 4;
/// `chroma_format_idc` of 4:2:0 and `bit_depth_idc` of eight bits.
const CHROMA_FORMAT_420: u32 = 0;
const BIT_DEPTH_EIGHT: u32 = 1;
/// Levels from which the sequence codes a tier.
const FIRST_TIERED_LEVEL: u8 = 4;

/// Parse a sequence-header payload.
/// # Errors
/// Truncation, field values outside AVM v1.0.0, and formats other than
/// 8-bit 4:2:0 ([`DecodeError::Unsupported`]).
pub fn parse(payload: &[u8]) -> Result<SequenceHeader> {
    let bits = &mut BitReader::new(payload);
    let id = bits.read_uvlc()?;
    if id > 15 {
        return Err(DecodeError::Malformed("AV2 sequence id exceeds 15"));
    }
    let profile = bits.read_bits(5)?;
    if profile > MAX_PROFILE {
        return Err(DecodeError::Malformed("AV2 profile"));
    }
    let single_picture = bits.read_bit()?;
    let level = bits.read_bits(5)? as u8;
    let high_tier = level >= FIRST_TIERED_LEVEL && !single_picture && bits.read_bit()?;
    if bits.read_uvlc()? != CHROMA_FORMAT_420 {
        return Err(DecodeError::Unsupported("AV2 chroma format other than 420"));
    }
    if bits.read_uvlc()? != BIT_DEPTH_EIGHT {
        return Err(DecodeError::Unsupported("AV2 bit depth other than eight"));
    }
    let mut header = SequenceHeader {
        id: id as u8,
        profile: profile as u8,
        single_picture,
        level,
        high_tier,
        still_picture: single_picture,
        monotonic_output: single_picture,
        // Single pictures choose the screen-content tools per frame.
        screen_content_tools: Tri::Adaptive,
        force_integer_mv: Tri::Adaptive,
        ..SequenceHeader::default()
    };
    if !single_picture {
        header.layer_config_id = bits.read_bits(3)? as u8;
        header.still_picture = bits.read_bit()?;
        header.max_temporal_layer = bits.read_bits(2)? as u8;
        header.max_embedded_layer = bits.read_bits(3)? as u8;
        header.monotonic_output = bits.read_bit()?;
    }
    header.width_bits = bits.read_bits(4)? as u8 + 1;
    header.height_bits = bits.read_bits(4)? as u8 + 1;
    header.max_width = bits.read_bits(u32::from(header.width_bits))? + 1;
    header.max_height = bits.read_bits(u32::from(header.height_bits))? + 1;
    if bits.read_bit()? {
        header.crop = Some(Crop {
            left: bits.read_uvlc()?,
            right: bits.read_uvlc()?,
            top: bits.read_uvlc()?,
            bottom: bits.read_uvlc()?,
        });
    }
    if !single_picture {
        timing(bits, &mut header)?;
    }
    layer_dependencies(bits, &mut header)?;
    header.partition = partition(bits, single_picture)?;
    header.segmentation = segmentation(bits)?;
    header.intra = intra(bits)?;
    if !single_picture {
        inter_prediction(bits, &mut header.inter)?;
    }
    header.references = references(bits, single_picture)?;
    header.references.short_refresh_flags = inter_tools(bits, single_picture, &mut header.inter)?;
    if !single_picture {
        header.screen_content_tools = tri(bits)?;
        if header.screen_content_tools != Tri::Off {
            header.force_integer_mv = tri(bits)?;
        }
    }
    header.transform = transform(bits, single_picture)?;
    header.quant = quant(bits)?;
    header.filters = filters(bits, single_picture, header.partition.superblock_log2)?;
    if bits.read_bit()? {
        header.tile_layout_fixed = !bits.read_bit()?;
        let geometry = TileGeometry {
            size: (header.max_width, header.max_height),
            superblock_log2: header.partition.superblock_log2,
            sequence_superblock_log2: header.partition.superblock_log2,
            level,
            high_tier,
        };
        header.tile_layout = Some(tiles::parse(bits, geometry)?);
    }
    header.film_grain = bits.read_bit()?;
    extension_and_trailing_bits(bits, payload)?;
    Ok(header)
}

/// `Tri` of AVM's "adaptive, else on/off" two-bit pattern.
fn tri(bits: &mut BitReader<'_>) -> Result<Tri> {
    let adaptive = bits.read_bit()?;
    Ok(match (adaptive, adaptive || bits.read_bit()?) {
        (true, _) => Tri::Adaptive,
        (false, true) => Tri::On,
        (false, false) => Tri::Off,
    })
}

/// Display and decoder model information.
fn timing(bits: &mut BitReader<'_>, header: &mut SequenceHeader) -> Result<()> {
    if bits.read_bit()? {
        header.max_initial_display_delay = Some(bits.read_bits(4)? as u8);
    }
    if bits.read_bit()? {
        let units_in_decoding_tick = bits.read_bits(32)?;
        let buffer_delays = if bits.read_bit()? {
            Some((bits.read_uvlc()?, bits.read_uvlc()?))
        } else {
            None
        };
        header.decoder_model = Some(DecoderModel {
            units_in_decoding_tick,
            buffer_delays,
        });
    }
    Ok(())
}

/// Each layer's dependency mask over itself and the lower layers: every
/// one of them unless signalled (AVM `av2_read_mlayer_dependency_info`,
/// `av2_read_tlayer_dependency_info`). A temporal map signalled per
/// embedded layer is read; the base embedded layer's is kept.
fn layer_dependencies(bits: &mut BitReader<'_>, header: &mut SequenceHeader) -> Result<()> {
    for (layer, mask) in header.embedded_dependencies.iter_mut().enumerate() {
        *mask = u8::MAX >> (7 - layer);
    }
    for (layer, mask) in header.temporal_dependencies.iter_mut().enumerate() {
        *mask = u8::MAX >> (7 - layer);
    }
    let (max_embedded, max_temporal) = (header.max_embedded_layer, header.max_temporal_layer);
    if max_embedded > 0 && bits.read_bit()? {
        for layer in 1..=max_embedded {
            header.embedded_dependencies[usize::from(layer)] =
                bits.read_bits(u32::from(layer) + 1)? as u8;
        }
    }
    if max_temporal > 0 && bits.read_bit()? {
        let per_embedded_layer = max_embedded > 0 && bits.read_bit()?;
        let maps = if per_embedded_layer {
            max_embedded + 1
        } else {
            1
        };
        for map in 0..maps {
            for layer in 1..=max_temporal {
                let mask = bits.read_bits(u32::from(layer) + 1)? as u8;
                if map == 0 {
                    header.temporal_dependencies[usize::from(layer)] = mask;
                }
            }
        }
    }
    Ok(())
}

fn partition(bits: &mut BitReader<'_>, single_picture: bool) -> Result<PartitionTools> {
    let superblock_log2 = if bits.read_bit()? { 8 } else { 6 + flag(bits)? };
    let separate_trees = bits.read_bit()?;
    let extended_separate_trees = separate_trees && !single_picture && bits.read_bit()?;
    let extended = bits.read_bit()?;
    let uneven_four_way = extended && bits.read_bit()?;
    let max_aspect_ratio_log2 = if bits.read_bit()? { 1 + flag(bits)? } else { 3 };
    Ok(PartitionTools {
        superblock_log2,
        separate_trees,
        extended_separate_trees,
        extended,
        uneven_four_way,
        max_aspect_ratio_log2,
    })
}

fn segmentation(bits: &mut BitReader<'_>) -> Result<Segmentation> {
    let mut segmentation = Segmentation {
        extended: bits.read_bit()?,
        ..Segmentation::default()
    };
    if bits.read_bit()? {
        segmentation.adaptive = bits.read_bit()?;
        segmentation.features = Some(SegmentFeatures::parse(bits, segmentation.count())?);
    }
    Ok(segmentation)
}

/// `cfl_ds_filter_index` value that is reserved and read as zero.
const CFL_FILTER_RESERVED: u8 = 3;

fn intra(bits: &mut BitReader<'_>) -> Result<IntraTools> {
    let dip = bits.read_bit()?;
    let edge_filter = bits.read_bit()?;
    let multiple_reference_lines = bits.read_bit()?;
    let cfl = bits.read_bit()?;
    let mut cfl_downsample_filter = bits.read_bits(2)? as u8;
    if cfl_downsample_filter == CFL_FILTER_RESERVED {
        cfl_downsample_filter = 0;
    }
    Ok(IntraTools {
        dip,
        edge_filter,
        multiple_reference_lines,
        cfl,
        cfl_downsample_filter,
        mhccp: bits.read_bit()?,
        ibp: bits.read_bit()?,
    })
}

/// The first inter group: motion modes, masked compound, motion fields and
/// the order-hint width (sequences of more than one picture).
fn inter_prediction(bits: &mut BitReader<'_>, inter: &mut InterTools) -> Result<()> {
    let modes = &mut inter.motion_modes;
    modes.inter_intra = bits.read_bit()?;
    modes.warp_causal = bits.read_bit()?;
    modes.warp_delta = bits.read_bit()?;
    modes.warp_extend = bits.read_bit()?;
    if modes.inter_intra || modes.warp_causal || modes.warp_delta || modes.warp_extend {
        modes.frame_signalled = bits.read_bit()?;
    }
    if modes.warp_delta {
        modes.six_parameter_warp_delta = bits.read_bit()?;
    }
    inter.masked_compound = bits.read_bit()?;
    inter.motion_fields = bits.read_bit()?;
    if inter.motion_fields {
        inter.reduced_motion_fields = bits.read_bit()?;
    }
    inter.order_hint_bits = bits.read_bits(4)? as u8 + 1;
    Ok(())
}

fn references(bits: &mut BitReader<'_>, single_picture: bool) -> Result<ReferenceTools> {
    let mut references = ReferenceTools {
        mv_bank: bits.read_bit()?,
        drl_reorder: if bits.read_bit()? { 0 } else { 2 - flag(bits)? },
        // Single pictures: two slots, the smallest DRL.
        slots: 2,
        max_drl_bits: 1,
        ..ReferenceTools::default()
    };
    if !single_picture {
        references.explicit_map = bits.read_bit()?;
        references.slots = if bits.read_bit()? {
            bits.read_bits(4)? as u8 + 1
        } else {
            DEFAULT_REFERENCE_SLOTS
        };
        references.long_term_id_bits = bits.read_bits(3)? as u8;
        references.max_drl_bits = uniform(bits, DRL_BITS_CHOICES)? as u8 + 1;
        references.frame_max_drl_bits = bits.read_bit()?;
    }
    references.max_bvp_drl_bits = uniform(bits, BVP_DRL_BITS_CHOICES)? as u8 + 1;
    references.frame_max_bvp_drl_bits = bits.read_bit()?;
    if !single_picture {
        references.same_reference_compounds = bits.read_bits(2)? as u8;
    }
    Ok(references)
}

/// The second and third inter groups (TIP to global motion); returns
/// `enable_short_refresh_frame_flags`, which closes them.
fn inter_tools(
    bits: &mut BitReader<'_>,
    single_picture: bool,
    inter: &mut InterTools,
) -> Result<bool> {
    if !single_picture {
        inter.tip = if bits.read_bit()? { 1 + flag(bits)? } else { 0 };
        if inter.tip != 0 {
            inter.tip_hole_fill = bits.read_bit()?;
        }
        inter.mv_trajectories = bits.read_bit()?;
    }
    inter.bawp = bits.read_bit()?;
    if single_picture {
        return Ok(false);
    }
    inter.cwp = bits.read_bit()?;
    inter.implicit_masked_blend = bits.read_bit()?;
    inter.deblock_sub_pu = bits.read_bit()?;
    if inter.tip == 1 && inter.deblock_sub_pu {
        inter.tip_explicit_qp = bits.read_bit()?;
    }
    inter.optical_flow = bits.read_bits(2)? as u8;
    inter.refine_mv = bits.read_bit()?;
    if inter.tip != 0 && (inter.optical_flow != 0 || inter.refine_mv) {
        inter.tip_refine_mv = bits.read_bit()?;
    }
    inter.bru = bits.read_bit()?;
    inter.adaptive_mvd = bits.read_bit()?;
    inter.mvd_sign_derivation = bits.read_bit()?;
    inter.flexible_mv_resolution = bits.read_bit()?;
    inter.global_motion = bits.read_bit()?;
    bits.read_bit()
}

fn transform(bits: &mut BitReader<'_>, single_picture: bool) -> Result<TransformTools> {
    let fsc = bits.read_bit()?;
    let mut transform = TransformTools {
        fsc,
        intra_identity: fsc || bits.read_bit()?,
        intra_secondary: bits.read_bit()?,
        inter_secondary: bits.read_bit()?,
        chroma_dct_only: bits.read_bit()?,
        ..TransformTools::default()
    };
    if !single_picture {
        transform.inter_ddt = bits.read_bit()?;
    }
    transform.reduced_partition_set = bits.read_bit()?;
    transform.cctx = bits.read_bit()?;
    transform.tcq = flag(bits)?;
    if transform.tcq != 0 && !single_picture {
        transform.tcq += flag(bits)?;
    }
    if transform.tcq != 1 {
        transform.parity_hiding = bits.read_bit()?;
    }
    // Single pictures always start from the averaged CDFs of type one.
    transform.average_cdf = single_picture || bits.read_bit()?;
    if transform.average_cdf {
        transform.average_cdf_type = single_picture || bits.read_bit()?;
    }
    Ok(transform)
}

/// A 5-bit base quantizer delta.
fn base_delta(bits: &mut BitReader<'_>) -> Result<i8> {
    Ok(bits.read_bits(5)? as i8 + BASE_DELTA_Q_MIN)
}

fn quant(bits: &mut BitReader<'_>) -> Result<QuantTools> {
    let mut quant = QuantTools {
        separate_uv_delta_q: bits.read_bit()?,
        equal_ac_dc_q: bits.read_bit()?,
        ..QuantTools::default()
    };
    if !quant.equal_ac_dc_q {
        quant.base_y_dc_delta = base_delta(bits)?;
        quant.y_dc_delta_enabled = bits.read_bit()?;
        quant.base_uv_dc_delta = base_delta(bits)?;
        quant.uv_dc_delta_enabled = bits.read_bit()?;
    }
    quant.base_uv_ac_delta = base_delta(bits)?;
    quant.uv_ac_delta_enabled = bits.read_bit()?;
    if quant.equal_ac_dc_q {
        quant.base_uv_dc_delta = quant.base_uv_ac_delta;
    }
    Ok(quant)
}

/// Restoration filter bits of [`FilterTools::restoration_disable_mask`].
const NO_PC_WIENER: u8 = 1;
const NO_NONSEP_WIENER: u8 = 2;
/// Superblocks of 64 samples: the only size at which the guided detail
/// filter's unit is signalled.
const SMALLEST_SUPERBLOCK_LOG2: u8 = 6;

fn filters(
    bits: &mut BitReader<'_>,
    single_picture: bool,
    superblock_log2: u8,
) -> Result<FilterTools> {
    let mut filters = FilterTools {
        disable_across_tiles: bits.read_bit()?,
        cdef: bits.read_bit()?,
        gdf: bits.read_bit()?,
        ..FilterTools::default()
    };
    if filters.gdf && superblock_log2 == SMALLEST_SUPERBLOCK_LOG2 {
        filters.gdf_unit_is_superblock = bits.read_bit()?;
    }
    filters.restoration = bits.read_bit()?;
    if filters.restoration {
        let luma = (flag(bits)? * NO_PC_WIENER) | (flag(bits)? * NO_NONSEP_WIENER);
        // Chroma never uses PC-Wiener; its non-separable flag is its own
        // when signalled, else luma's.
        let chroma = if bits.read_bit()? {
            (flag(bits)? * NO_NONSEP_WIENER) | NO_PC_WIENER
        } else {
            luma | NO_PC_WIENER
        };
        filters.restoration_disable_mask = [luma, chroma];
    }
    filters.ccso = bits.read_bit()?;
    if filters.ccso {
        filters.ccso_unit_is_superblock = bits.read_bit()?;
    }
    filters.cdef_on_skipped = if single_picture {
        Tri::Adaptive
    } else {
        let always = bits.read_bit()?;
        match (always, always || bits.read_bit()?) {
            (true, _) => Tri::On,
            (false, true) => Tri::Off,
            (false, false) => Tri::Adaptive,
        }
    };
    filters.deblock_param_bits = MIN_DEBLOCK_PARAM_BITS + bits.read_bits(2)? as u8;
    Ok(filters)
}

/// The extension flag, any extension payload (every bit up to the final
/// trailing one), the trailing one bit and zero padding (dav2d
/// `read_extension_bits` and `check_trailing_bits`).
fn extension_and_trailing_bits(bits: &mut BitReader<'_>, payload: &[u8]) -> Result<()> {
    let malformed = DecodeError::Malformed("AV2 sequence trailing bits");
    let has_extension = bits.read_bit()?;
    // The last set bit of the payload is the trailing one.
    let last = payload
        .iter()
        .rposition(|&byte| byte != 0)
        .ok_or(malformed.clone())?;
    let trailing_one = last * 8 + 7 - payload[last].trailing_zeros() as usize;
    let position = bits.position();
    if position > trailing_one || (!has_extension && position != trailing_one) {
        return Err(malformed);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(hex: &str) -> Vec<u8> {
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect()
    }

    /// Sequence headers of other encoders' streams (avmenc v1.0.0 and the
    /// dav2d test vectors); the expected fields are dav2d's
    /// (`DEBUG_SEQ_HDR`).
    const SEGMENTED_128: &str =
        "800a00222bf1f0f9fde3f31fc0ff27fd20010148167bfb9a026afceefd69b9c56840";
    const LOW_DELAY: &str = "801a002627fb383cc000001cddc741357e377ea4dccad080";
    const SINGLE_PICTURE: &str = "821a989fece08004061b9088";
    const FEWEST_TOOLS: &str = "801a006627fb3809800000000068210000c283720684";

    #[test]
    fn default_tools_with_segments_and_128_superblocks() {
        let header = parse(&bytes(SEGMENTED_128)).unwrap();
        assert_eq!((header.max_width, header.max_height), (352, 288));
        assert!(!header.monotonic_output);
        assert_eq!(header.partition.superblock_log2, 7);
        assert!(header.partition.uneven_four_way);
        assert_eq!(header.partition.max_aspect_ratio_log2, 3);
        let features = header.segmentation.features.unwrap();
        assert_eq!(features.delta_q[..8], [-34, -26, -16, -7, -3, 0, 5, 11]);
        assert_eq!(features.delta_q_mask, 0xff);
        assert_eq!(header.intra.cfl_downsample_filter, 1);
        assert_eq!(header.inter.order_hint_bits, 7);
        assert_eq!(header.inter.optical_flow, 1);
        assert_eq!(header.references.slots, 8);
        assert_eq!(header.references.slot_bits(), 3);
        assert_eq!(header.references.max_drl_bits, 3);
        assert_eq!(header.references.same_reference_compounds, 2);
        assert_eq!(header.transform.tcq, 1);
        assert!(!header.transform.parity_hiding);
        assert_eq!(header.filters.restoration_disable_mask, [0, 1]);
        assert_eq!(header.filters.cdef_on_skipped, Tri::On);
        assert_eq!(header.filters.deblock_param_bits, 3);
        let layout = header.tile_layout.unwrap();
        assert_eq!((layout.columns(), layout.rows()), (1, 1));
        assert_eq!(layout.column_starts, [0, 3]);
        assert!(header.tile_layout_fixed);
    }

    #[test]
    fn low_delay_single_picture_and_fewest_tools() {
        let header = parse(&bytes(LOW_DELAY)).unwrap();
        assert_eq!((header.max_width, header.max_height), (640, 360));
        assert!(!header.inter.motion_modes.inter_intra);
        assert!(header.inter.motion_modes.warp_extend);
        assert_eq!(header.references.drl_reorder, 1);
        assert_eq!(header.tile_layout.unwrap().row_starts, [0, 6]);

        let header = parse(&bytes(SINGLE_PICTURE)).unwrap();
        assert!(header.single_picture && header.still_picture && header.monotonic_output);
        assert_eq!(header.references.slots, 2);
        assert_eq!(header.screen_content_tools, Tri::Adaptive);
        assert_eq!(header.filters.cdef_on_skipped, Tri::Adaptive);
        assert!(header.transform.average_cdf && header.transform.average_cdf_type);
        assert!(header.transform.chroma_dct_only);
        assert_eq!(header.tile_layout, None);

        let header = parse(&bytes(FEWEST_TOOLS)).unwrap();
        assert!(header.monotonic_output);
        assert_eq!(header.inter.tip, 0);
        assert_eq!(header.references.same_reference_compounds, 2);
        assert_eq!(header.screen_content_tools, Tri::Adaptive);
    }

    #[test]
    fn truncation_and_trailing_bits_are_rejected() {
        for hex in [SEGMENTED_128, LOW_DELAY, SINGLE_PICTURE, FEWEST_TOOLS] {
            let payload = bytes(hex);
            for length in 0..payload.len() {
                assert!(parse(&payload[..length]).is_err(), "{hex} cut to {length}");
            }
            let mut no_stop = payload.clone();
            *no_stop.last_mut().unwrap() = 0;
            assert!(parse(&no_stop).is_err());
        }
        // Formats other than 4:2:0 are outside the decoder.
        assert_eq!(
            parse(&bytes("8004")).unwrap_err(),
            DecodeError::Unsupported("AV2 chroma format other than 420")
        );
    }
}
