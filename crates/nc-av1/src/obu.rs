//! AV1 OBU framing, sequence header and frame header (AV1 §5.3–§5.9).
//!
//! The headers describe exactly the subset this encoder produces: profile 0
//! (8-bit 4:2:0), 64x64 superblocks, one tile, CDEF on, and every other
//! optional tool off (filter intra, intra edge filter, screen content,
//! order hints; film grain, superres and Wiener loop restoration only when
//! enabled). Every field is written
//! explicitly with its spec name so the header can be checked line by line
//! against §5.5 and §5.9.
//!
//! Hierarchical-P structures (`nc_encoder` `structure.rs`) are temporal
//! layers: frame OBUs carry `obu_extension_header()` with their
//! `temporal_id`, and the sequence header lists one operating point per
//! layer count kept (§7.18 drops the other layers' OBUs). Each frame
//! predicts from the reference slot the engine names (`ref_frame_idx`) and
//! refreshes the slots it names (`refresh_frame_flags`; none for the
//! droppable top layer).
//!
//! Frame ids (`frame_id_numbers_present_flag`, §5.5.1), when loss
//! resilience asks for them (`backend.rs`, `av1.frame_ids`): every frame
//! codes `current_frame_id` and, per reference, how many frames back the
//! frame it predicts from is (`delta_frame_id_minus_1`). A decoder that lost
//! a frame then finds the slot holding another frame than the one expected
//! and refuses the frame (§7.20, §7.21; ours returns `MissingReference`,
//! dav1d and libaom fail it) instead of predicting from the wrong picture,
//! so a receiver without a transport dependency signal still notices a
//! loss. The delta field is as short as the stream's farthest reference
//! allows ([`FrameIdLengths`]): `idLen` (9) + 7 x the delta length bits per
//! inter frame (a flat or layered stream: 9 + 7 x 2 = 23; with a long-term
//! reference and keys every 60 frames: 9 + 7 x 6 = 51; at most
//! 9 + 7 x 8 = 65), `idLen` per key frame.

use nc_common::colour::write_code_points;
use nc_common::{BitWriter, leb128};
use nc_encoder::{
    FrameRate, MAX_REFERENCE_SLOTS, SlotMask, SpatialConfig, StreamSignal, TimingSignal,
};

use crate::cdef::CdefParams;
use crate::grain::FrameGrain;
use crate::loopfilter::LoopFilterParams;
use crate::restoration::LrParams;
use crate::superres::{SUPERRES_DENOM_BITS, SUPERRES_DENOM_MIN, SUPERRES_NUM};

/// OBU types used (AV1 §6.2.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ObuType {
    /// `OBU_SEQUENCE_HEADER`.
    SequenceHeader = 1,
    /// `OBU_TEMPORAL_DELIMITER`.
    TemporalDelimiter = 2,
    /// `OBU_FRAME`: frame header plus tile group.
    Frame = 6,
}

/// `obu_extension_flag` in the OBU header byte.
pub const OBU_EXTENSION_FLAG: u8 = 0b100;
/// `obu_has_size_field` in the OBU header byte.
const OBU_HAS_SIZE_FIELD: u8 = 0b10;
/// Position of `temporal_id` in the extension byte (AV1 §5.3.3).
pub const TEMPORAL_ID_SHIFT: u8 = 5;
/// Position of `spatial_id` in the extension byte (AV1 §5.3.3).
pub const SPATIAL_ID_SHIFT: u8 = 3;

/// Append one OBU with `obu_has_size_field = 1` (AV1 §5.3.1), with an
/// `obu_extension_header()` carrying `(temporal_id, spatial_id)` if given
/// (AV1 §5.3.3).
pub fn write_obu_layer(
    out: &mut Vec<u8>,
    obu_type: ObuType,
    layer: Option<(u8, u8)>,
    payload: &[u8],
) {
    // obu_forbidden_bit(0) obu_type(4) obu_extension_flag obu_has_size_field(1) reserved(0)
    let extension = if layer.is_some() {
        OBU_EXTENSION_FLAG
    } else {
        0
    };
    out.push(((obu_type as u8) << 3) | extension | OBU_HAS_SIZE_FIELD);
    if let Some((temporal_id, spatial_id)) = layer {
        debug_assert!(temporal_id < 8 && spatial_id < 4);
        // temporal_id(3) spatial_id(2) extension_header_reserved_3bits(3)
        out.push((temporal_id << TEMPORAL_ID_SHIFT) | (spatial_id << SPATIAL_ID_SHIFT));
    }
    out.extend(leb128(payload.len() as u64));
    out.extend_from_slice(payload);
}

/// Append one OBU with `obu_has_size_field = 1` (AV1 §5.3.1), with an
/// `obu_extension_header()` carrying `temporal_id` (spatial id 0) if given.
pub fn write_obu(out: &mut Vec<u8>, obu_type: ObuType, temporal_id: Option<u8>, payload: &[u8]) {
    write_obu_layer(out, obu_type, temporal_id.map(|t| (t, 0)), payload);
}

/// `seq_level_idx` 31: no level constraints signalled (AV1 Annex A.3).
const SEQ_LEVEL_MAX_PARAMETERS: u32 = 31;
/// `seq_level_idx` above which `seq_tier` is coded (AV1 §5.5.1).
const LEVEL_WITH_TIER: u32 = 7;
/// Smallest base area using constrained Annex A.3 levels rather than `31`
/// (tiny synthetic test clips can exceed `MinCompRatio` on intra headers).
const MIN_CONSTRAINED_LEVEL_AREA: usize = 128 * 128;
/// Lowest fixed `base_q_idx` using constrained Annex A.3 levels rather than `31`.
const MIN_CONSTRAINED_LEVEL_QINDEX: u32 = 32;
/// `CSP_UNKNOWN` chroma sample position.
const CHROMA_SAMPLE_POSITION_UNKNOWN: u32 = 0;

/// Shortest `delta_frame_id_length_minus_2 + 2` the syntax allows.
const MIN_FRAME_ID_DELTA_BITS: u32 = 2;
/// Longest delta length that keeps `idLen` within the 16 bits §6.8.2
/// allows (and `additional_frame_id_length_minus_1` within its 3 bits).
const MAX_FRAME_ID_DELTA_BITS: u32 = 15;
/// Least `idLen`. A receiver that misses frames (lost, or refused while
/// it waits for a repair) compares the slot's id with the expected one
/// modulo 2^`idLen`, so a gap of a multiple of 2^`idLen` frames goes
/// unnoticed: 9 bits leave that to gaps of 512 frames (17 s at 30 fps),
/// long after a loss report has brought a key. With `idLen` one more than
/// the delta length (3 bits for a flat stream) gaps of 8 frames slipped
/// through: 4 and 12 mismatching frames in the losssim `detect` study's
/// AV1 rows without a long-term reference at 100 ms one way. The delta
/// length, coded seven times per frame, stays as short as the references
/// allow.
const MIN_FRAME_ID_BITS: u32 = 9;

/// Uncompressed bits per luma sample for 8-bit 4:2:0 (`BitDepth * YuvFactor = 8 * 1.5`).
const BITS_PER_LUMA_SAMPLE_420_8BIT: u64 = 12;

/// One defined AV1 Annex A.3 Main-tier level (`seq_profile == 0`, `ProfileFactor = 1.0`).
#[derive(Debug, Clone, Copy)]
struct AnnexA3Level {
    idx: u32,
    max_pic_size: u64,
    max_h_size: u64,
    max_v_size: u64,
    max_display_rate: u64,
    max_decode_rate: u64,
    max_header_rate: u64,
    max_bitrate_bps: u64,
    min_comp_basis: u64,
}

/// Defined AV1 Annex A.3 levels (2.0..=6.3, Main tier, Profile 0).
const ANNEX_A3_LEVELS: &[AnnexA3Level] = &[
    AnnexA3Level {
        idx: 0,
        max_pic_size: 147_456,
        max_h_size: 2_048,
        max_v_size: 1_152,
        max_display_rate: 4_423_680,
        max_decode_rate: 5_529_600,
        max_header_rate: 150,
        max_bitrate_bps: 1_500_000,
        min_comp_basis: 2,
    },
    AnnexA3Level {
        idx: 1,
        max_pic_size: 278_784,
        max_h_size: 2_816,
        max_v_size: 1_584,
        max_display_rate: 8_363_520,
        max_decode_rate: 10_454_400,
        max_header_rate: 150,
        max_bitrate_bps: 3_000_000,
        min_comp_basis: 2,
    },
    AnnexA3Level {
        idx: 4,
        max_pic_size: 665_856,
        max_h_size: 4_352,
        max_v_size: 2_448,
        max_display_rate: 19_975_680,
        max_decode_rate: 24_969_600,
        max_header_rate: 150,
        max_bitrate_bps: 6_000_000,
        min_comp_basis: 2,
    },
    AnnexA3Level {
        idx: 5,
        max_pic_size: 1_065_024,
        max_h_size: 5_504,
        max_v_size: 3_096,
        max_display_rate: 31_950_720,
        max_decode_rate: 39_938_400,
        max_header_rate: 150,
        max_bitrate_bps: 10_000_000,
        min_comp_basis: 2,
    },
    AnnexA3Level {
        idx: 8,
        max_pic_size: 2_359_296,
        max_h_size: 6_144,
        max_v_size: 3_456,
        max_display_rate: 70_778_880,
        max_decode_rate: 77_856_768,
        max_header_rate: 300,
        max_bitrate_bps: 12_000_000,
        min_comp_basis: 4,
    },
    AnnexA3Level {
        idx: 9,
        max_pic_size: 2_359_296,
        max_h_size: 6_144,
        max_v_size: 3_456,
        max_display_rate: 141_557_760,
        max_decode_rate: 155_713_536,
        max_header_rate: 300,
        max_bitrate_bps: 20_000_000,
        min_comp_basis: 4,
    },
    AnnexA3Level {
        idx: 12,
        max_pic_size: 8_912_896,
        max_h_size: 8_192,
        max_v_size: 4_352,
        max_display_rate: 267_386_880,
        max_decode_rate: 273_715_200,
        max_header_rate: 300,
        max_bitrate_bps: 30_000_000,
        min_comp_basis: 6,
    },
    AnnexA3Level {
        idx: 13,
        max_pic_size: 8_912_896,
        max_h_size: 8_192,
        max_v_size: 4_352,
        max_display_rate: 534_773_760,
        max_decode_rate: 547_430_400,
        max_header_rate: 300,
        max_bitrate_bps: 40_000_000,
        min_comp_basis: 8,
    },
    AnnexA3Level {
        idx: 14,
        max_pic_size: 8_912_896,
        max_h_size: 8_192,
        max_v_size: 4_352,
        max_display_rate: 1_069_547_520,
        max_decode_rate: 1_094_860_800,
        max_header_rate: 300,
        max_bitrate_bps: 60_000_000,
        min_comp_basis: 8,
    },
    AnnexA3Level {
        idx: 15,
        max_pic_size: 8_912_896,
        max_h_size: 8_192,
        max_v_size: 4_352,
        max_display_rate: 1_069_547_520,
        max_decode_rate: 1_176_502_272,
        max_header_rate: 300,
        max_bitrate_bps: 60_000_000,
        min_comp_basis: 8,
    },
    AnnexA3Level {
        idx: 16,
        max_pic_size: 35_651_584,
        max_h_size: 16_384,
        max_v_size: 8_704,
        max_display_rate: 1_069_547_520,
        max_decode_rate: 1_176_502_272,
        max_header_rate: 300,
        max_bitrate_bps: 60_000_000,
        min_comp_basis: 8,
    },
    AnnexA3Level {
        idx: 17,
        max_pic_size: 35_651_584,
        max_h_size: 16_384,
        max_v_size: 8_704,
        max_display_rate: 2_139_095_040,
        max_decode_rate: 2_189_721_600,
        max_header_rate: 300,
        max_bitrate_bps: 100_000_000,
        min_comp_basis: 8,
    },
    AnnexA3Level {
        idx: 18,
        max_pic_size: 35_651_584,
        max_h_size: 16_384,
        max_v_size: 8_704,
        max_display_rate: 4_278_190_080,
        max_decode_rate: 4_379_443_200,
        max_header_rate: 300,
        max_bitrate_bps: 160_000_000,
        min_comp_basis: 8,
    },
    AnnexA3Level {
        idx: 19,
        max_pic_size: 35_651_584,
        max_h_size: 16_384,
        max_v_size: 8_704,
        max_display_rate: 4_278_190_080,
        max_decode_rate: 4_706_009_088,
        max_header_rate: 300,
        max_bitrate_bps: 160_000_000,
        min_comp_basis: 8,
    },
];

/// Smallest Annex A.3 `seq_level_idx` covering an operating point's picture
/// size, display rate, total multi-layer decode rate, header rate, Main-tier
/// `MaxBitrate` and `MinCompRatio` (AV1 §5.5.1, §6.4.1, Annex A.3), falling
/// back to `31` (`Max parameters`).
#[must_use]
pub fn select_seq_level_idx(
    (width, height): (usize, usize),
    display_fps: FrameRate,
    decode_samples_per_sec: u64,
    headers_per_sec: u64,
    max_bitrate_bps: Option<u64>,
    unconstrained: bool,
) -> u32 {
    if unconstrained {
        return SEQ_LEVEL_MAX_PARAMETERS;
    }
    let w = width.next_multiple_of(8) as u64;
    let h = height.next_multiple_of(8) as u64;
    let pic_size = w * h;
    let display_rate = pic_size
        .saturating_mul(u64::from(display_fps.num))
        .div_ceil(u64::from(display_fps.den.max(1)));
    let uncompressed_bps = decode_samples_per_sec.saturating_mul(BITS_PER_LUMA_SAMPLE_420_8BIT);
    for level in ANNEX_A3_LEVELS {
        let bitrate_ok = max_bitrate_bps.is_none_or(|bps| {
            // Annex A.3: MinCompRatio = max(0.8, MinCompBasis * decode_rate / MaxDisplayRate).
            // Requiring uncompressed_bps / bps >= MinCompRatio simplifies to:
            //   4 * bps <= 5 * uncompressed_bps
            //   bps * MinCompBasis <= 12 * MaxDisplayRate
            bps <= level.max_bitrate_bps
                && bps.saturating_mul(4) <= uncompressed_bps.saturating_mul(5)
                && bps.saturating_mul(level.min_comp_basis)
                    <= level
                        .max_display_rate
                        .saturating_mul(BITS_PER_LUMA_SAMPLE_420_8BIT)
        });
        if pic_size <= level.max_pic_size
            && w <= level.max_h_size
            && h <= level.max_v_size
            && display_rate <= level.max_display_rate
            && decode_samples_per_sec <= level.max_decode_rate
            && headers_per_sec <= level.max_header_rate
            && bitrate_ok
        {
            return level.idx;
        }
    }
    SEQ_LEVEL_MAX_PARAMETERS
}

/// `operating_point_idc` bit of spatial layer 0 (AV1 §6.4.1).
const SPATIAL_LAYER_0: u32 = 1 << 8;

/// `operating_point_idc[op]` of a stream with `temporal_layers` layers:
/// 0 (every OBU decoded) for a flat stream; otherwise operating point `op`
/// keeps the lowest `temporal_layers − op` temporal layers.
#[must_use]
pub fn operating_point_idc(temporal_layers: u8, op: u8) -> u32 {
    if temporal_layers <= 1 {
        return 0;
    }
    SPATIAL_LAYER_0 | ((1 << (temporal_layers - op)) - 1)
}

/// The optional coding tools a sequence header enables.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SequenceTools {
    /// `enable_filter_intra`.
    pub filter_intra: bool,
    /// `film_grain_params_present`.
    pub film_grain: bool,
    /// `enable_superres`: frames may code `use_superres` (§5.9.8).
    pub superres: bool,
    /// `frame_id_numbers_present_flag` and the id lengths, when set.
    pub frame_ids: Option<FrameIdLengths>,
    /// `enable_restoration`: whether frames code `lr_params` (§5.9.20).
    pub restoration: RestorationTool,
}

/// `enable_restoration` of a sequence.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RestorationTool {
    /// Frames code no `lr_params`; no restoration.
    #[default]
    Disabled,
    /// Frames code `lr_params` and may restore.
    Enabled,
}

/// Sequence header OBU payload (AV1 §5.5). `width` is the upscaled
/// (stream) width.
#[must_use]
pub fn sequence_header(
    width: usize,
    height: usize,
    temporal_layers: u8,
    signal: StreamSignal,
    frame_rate: FrameRate,
    tools: SequenceTools,
) -> Vec<u8> {
    let mut w = BitWriter::new();
    write_sequence_header_prefix(&mut w, signal, frame_rate);
    let operating_points = temporal_layers.max(1);
    w.put_bits(u32::from(operating_points - 1), 5); // operating_points_cnt_minus_1
    for op in 0..operating_points {
        w.put_bits(operating_point_idc(temporal_layers, op), 12); // operating_point_idc[op]
        w.put_bits(SEQ_LEVEL_MAX_PARAMETERS, 5); // seq_level_idx[op]
        w.put_bit(false); // seq_tier[op] (present because seq_level_idx > 7)
    }
    write_sequence_header_tail(w, width, height, signal, tools)
}

/// Sequence header OBU payload for a two-layer spatial SVC stream
/// (AV1 §5.5, §6.4.1, Annex A.3).
///
/// `max_frame_width_minus_1` and `max_frame_height_minus_1` carry the full
/// enhancement dimensions (`spatial.full_size`); operating points are emitted
/// highest quality first with valid Annex A.3 `seq_level_idx` values.
///
/// # Panics
///
/// Panics if `spatial.operating_points` is empty.
#[must_use]
pub fn spatial_sequence_header(
    spatial: &SpatialConfig,
    signal: StreamSignal,
    frame_rate: FrameRate,
    tools: SequenceTools,
    fixed_base_q_idx: Option<u32>,
    max_bitrate_bps: Option<u64>,
    random_decisions: bool,
) -> Vec<u8> {
    assert!(
        !spatial.operating_points.is_empty(),
        "spatial sequence header requires at least one operating point"
    );
    let (width, height) = spatial.full_size;
    let mut w = BitWriter::new();
    write_sequence_header_prefix(&mut w, signal, frame_rate);

    let unconstrained = random_decisions
        || spatial.base_size.0 * spatial.base_size.1 < MIN_CONSTRAINED_LEVEL_AREA
        || fixed_base_q_idx.is_some_and(|q| q < MIN_CONSTRAINED_LEVEL_QINDEX);
    let fps_num = u64::from(frame_rate.num);
    let fps_den = u64::from(frame_rate.den.max(1));

    let op_count = spatial.operating_points.len();
    w.put_bits((op_count - 1) as u32, 5); // operating_points_cnt_minus_1
    for op in &spatial.operating_points {
        let drop_shift = spatial
            .temporal_layers
            .saturating_sub(1)
            .saturating_sub(op.max_temporal_id);
        let op_den = fps_den.saturating_mul(1u64 << drop_shift);
        // Annex A.3: the operating point decodes every layer up to
        // `max_spatial_id`, each at its own picture rate (a decimated `S1`
        // every `enhancement_interval`-th unit).
        let max_s = usize::from(op.max_spatial_id).min(spatial.layer_sizes.len() - 1);
        let (total_decode_rate, total_hdr_rate) = spatial.layer_sizes[..=max_s]
            .iter()
            .enumerate()
            .fold((0u64, 0u64), |(decode, hdr), (s, &(lw, lh))| {
                let den = if s == 1 && spatial.enhancement_interval > 1 {
                    fps_den.saturating_mul(u64::from(spatial.enhancement_interval))
                } else {
                    op_den
                };
                let pic = (lw.next_multiple_of(8) * lh.next_multiple_of(8)) as u64;
                (
                    decode.saturating_add(pic.saturating_mul(fps_num).div_ceil(den)),
                    hdr.saturating_add(fps_num.div_ceil(den)),
                )
            });
        // A lower operating point carries at most its layers' rate shares.
        let op_bps = max_bitrate_bps.map(|b| {
            if max_s + 1 >= spatial.layer_sizes.len() {
                b
            } else {
                let share: f64 = spatial.layer_rate_shares[..=max_s].iter().sum();
                (b as f64 * share).ceil() as u64
            }
        });
        let level = select_seq_level_idx(
            (op.width, op.height),
            op.frame_rate,
            total_decode_rate,
            total_hdr_rate,
            op_bps,
            unconstrained,
        );
        w.put_bits(u32::from(op.idc), 12); // operating_point_idc[op]
        w.put_bits(level, 5); // seq_level_idx[op]
        if level > LEVEL_WITH_TIER {
            w.put_bit(false); // seq_tier[op] (present only when seq_level_idx > 7, §5.5.1)
        }
    }
    write_sequence_header_tail(w, width, height, signal, tools)
}

fn write_sequence_header_prefix(w: &mut BitWriter, signal: StreamSignal, frame_rate: FrameRate) {
    w.put_bits(0, 3); // seq_profile: Main (8/10-bit 4:2:0)
    w.put_bit(false); // still_picture
    w.put_bit(false); // reduced_still_picture_header
    let timing = signal.timing != TimingSignal::Nominal;
    w.put_bit(timing); // timing_info_present_flag
    if timing {
        write_timing_info(w, signal.timing, frame_rate);
        w.put_bit(false); // decoder_model_info_present_flag
    }
    w.put_bit(false); // initial_display_delay_present_flag
}

fn write_sequence_header_tail(
    mut w: BitWriter,
    width: usize,
    height: usize,
    signal: StreamSignal,
    tools: SequenceTools,
) -> Vec<u8> {
    let colour = signal.colour;
    let width_bits = bits_for(width - 1);
    let height_bits = bits_for(height - 1);
    w.put_bits(width_bits - 1, 4); // frame_width_bits_minus_1
    w.put_bits(height_bits - 1, 4); // frame_height_bits_minus_1
    w.put_bits((width - 1) as u32, width_bits); // max_frame_width_minus_1
    w.put_bits((height - 1) as u32, height_bits); // max_frame_height_minus_1
    w.put_bit(tools.frame_ids.is_some()); // frame_id_numbers_present_flag
    if let Some(lengths) = tools.frame_ids {
        w.put_bits(lengths.delta_bits - MIN_FRAME_ID_DELTA_BITS, 4); // delta_frame_id_length_minus_2
        // additional_frame_id_length_minus_1: idLen beyond the delta length.
        w.put_bits(lengths.id_bits() - lengths.delta_bits - 1, 3);
    }
    w.put_bit(false); // use_128x128_superblock
    w.put_bit(tools.filter_intra); // enable_filter_intra
    w.put_bit(false); // enable_intra_edge_filter
    w.put_bit(false); // enable_interintra_compound
    w.put_bit(false); // enable_masked_compound
    w.put_bit(false); // enable_warped_motion
    w.put_bit(false); // enable_dual_filter
    w.put_bit(false); // enable_order_hint
    w.put_bit(false); // seq_choose_screen_content_tools
    w.put_bit(false); // seq_force_screen_content_tools = 0
    // seq_force_integer_mv is not coded: screen content tools are off.
    w.put_bit(tools.superres); // enable_superres
    w.put_bit(true); // enable_cdef
    w.put_bit(tools.restoration == RestorationTool::Enabled); // enable_restoration
    // color_config() (§5.5.2)
    w.put_bit(false); // high_bitdepth
    w.put_bit(false); // mono_chrome
    w.put_bit(colour.description.is_some()); // color_description_present_flag
    if let Some(description) = colour.description {
        // The identity matrix (the sRGB special case, which has no
        // color_range bit) is rejected for 4:2:0 at configuration.
        write_code_points(&mut w, description);
    }
    w.put_bit(colour.full_range); // color_range
    w.put_bits(CHROMA_SAMPLE_POSITION_UNKNOWN, 2); // chroma_sample_position
    w.put_bit(false); // separate_uv_delta_q
    w.put_bit(tools.film_grain); // film_grain_params_present
    w.trailing_bits();
    w.into_bytes()
}

/// `timing_info()` (AV1 §5.5.3): one display tick per nominal frame; a
/// constant-rate stream also promises one tick per picture.
fn write_timing_info(w: &mut BitWriter, timing: TimingSignal, frame_rate: FrameRate) {
    w.put_bits(frame_rate.den, 32); // num_units_in_display_tick
    w.put_bits(frame_rate.num, 32); // time_scale
    let equal = timing == TimingSignal::Constant;
    w.put_bit(equal); // equal_picture_interval
    if equal {
        w.put_uvlc(0); // num_ticks_per_picture_minus_1
    }
}

/// Number of bits needed to represent `value` (at least 1).
fn bits_for(value: usize) -> u32 {
    (usize::BITS - value.leading_zeros()).max(1)
}

/// Superblock size (log2, in luma samples) signalled by the sequence header.
const SUPERBLOCK_LOG2: u32 = 6;
/// Tile limits from AV1 §A.3 / §5.9.15.
const MAX_TILE_WIDTH: usize = 4096;
const MAX_TILE_AREA: usize = 4096 * 2304;
const MAX_TILE_COLS: usize = 64;
const MAX_TILE_ROWS: usize = 64;

/// `frame_type` syntax values (AV1 §5.9.2, §6.8.2).
const FRAME_TYPE_KEY: u32 = 0;
const FRAME_TYPE_INTER: u32 = 1;
const FRAME_TYPE_INTRA_ONLY: u32 = 2;

/// The two kinds of frame this encoder produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameKind {
    /// `KEY_FRAME` (or `INTRA_ONLY_FRAME` when `FrameHeader::intra_only` is set).
    Key,
    /// `INTER_FRAME` predicted from `LAST_FRAME` (and optionally `LAST2_FRAME`).
    Inter,
}

/// `primary_ref_frame` of inter frames: `LAST_FRAME − LAST_FRAME`, so
/// they load the CDFs saved with the reference's slot (AV1 §5.9.2
/// `load_cdfs`).
const PRIMARY_REF_LAST: u32 = 0;
/// `REFS_PER_FRAME`: `LAST_FRAME` .. `ALTREF_FRAME`.
const REFS_PER_FRAME: usize = 7;
/// Bits of `render_width_minus_1` / `render_height_minus_1` (§5.9.6).
const RENDER_SIZE_BITS: u32 = 16;
/// `EIGHTTAP` (regular) interpolation filter.
const INTERP_FILTER_EIGHTTAP: u32 = 0;

/// Slot-refresh scope of an intra (`KEY_FRAME` vs `INTRA_ONLY_FRAME`) header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntraRefresh {
    /// `KEY_FRAME` (`frame_type = 0`): implicitly refreshes all 8 slots.
    AllSlots,
    /// `INTRA_ONLY_FRAME` (`frame_type = 2`, AV1 §5.9.2 / §6.8.2): refreshes
    /// only `references.refresh`, preserving lower spatial layers' slots.
    SelectedSlots,
}

/// The frame-level choices a frame header carries.
#[allow(
    clippy::struct_excessive_bools,
    reason = "AV1 §5.9.2 frame header flags map 1:1 to bitstream syntax"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeader {
    /// Key or inter frame.
    pub kind: FrameKind,
    /// Slot-refresh mode when `kind == FrameKind::Key`.
    pub intra_refresh: IntraRefresh,
    /// Luma width: `FrameWidth`, the coded (downscaled) width under
    /// superres.
    pub width: usize,
    /// Luma height.
    pub height: usize,
    /// `base_q_idx` (1..=255; 0 would select lossless coding).
    pub base_q_idx: u32,
    /// `DeltaQUDc` = `DeltaQUAc` (−64..=63), shared by U and V
    /// (`separate_uv_delta_q = 0`): chroma quantiser offset in qindex units.
    pub chroma_delta_q: i32,
    /// `delta_q_present`: superblocks code `delta_qindex` (§5.11.12).
    pub delta_q_present: bool,
    /// Deblocking levels and sharpness of the frame.
    pub loop_filter: LoopFilterParams,
    /// `TX_MODE_SELECT` (blocks signal transform sizes) rather than
    /// `TX_MODE_LARGEST`.
    pub tx_mode_select: bool,
    /// `reference_select` (`AV1 §5.9.23`): blocks on inter frames signal
    /// `comp_mode` to choose between single and compound reference prediction.
    pub reference_select: bool,
    /// `disable_cdf_update`: symbols do not adapt their CDFs within the
    /// frame, and the frame saves the CDFs it loaded (frame-end update off).
    pub disable_cdf_update: bool,
    /// CDEF strengths of the frame.
    pub cdef: CdefParams,
    /// Optional second strength set, selected by one literal per filter block.
    pub cdef_alternate: Option<CdefParams>,
    /// `lr_params()` of a sequence with `enable_restoration`; `None` when
    /// the sequence header disables restoration.
    pub restoration: Option<LrParams>,
    /// Inter (and intra-only) frames: `ref_frame_idx` slots (`LAST_FRAME` and
    /// optional `LAST2_FRAME`) and `refresh_frame_flags`. Unused for key frames,
    /// which refresh every slot.
    pub references: SlotUse,
    /// `film_grain_params()`, when the sequence header enables film grain.
    pub film_grain: Option<FrameGrain>,
    /// `superres_params()` (§5.9.8) of a sequence with `enable_superres`:
    /// `Some(SuperresDenom)`, [`SUPERRES_NUM`] meaning `use_superres` = 0.
    /// `None` when the sequence header disables superres.
    pub superres_denom: Option<usize>,
    /// How the frame signals its size when it differs from the stream's.
    pub size: FrameSizeSyntax,
    /// `current_frame_id` and, for inter frames, the reference's distance,
    /// when the sequence header sets `frame_id_numbers_present_flag`.
    pub frame_id: Option<FrameId>,
}

/// The frame-id field lengths of a stream (§5.5.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameIdLengths {
    /// `delta_frame_id_length_minus_2 + 2`: a reference may be up to
    /// 2^`delta_bits` frames back.
    delta_bits: u32,
}

impl FrameIdLengths {
    /// The shortest lengths for a stream whose references reach at most
    /// `reach` frames back (`nc_encoder::reference_reach`).
    ///
    /// # Panics
    ///
    /// If `reach` needs a delta longer than `idLen`'s 16 bits leave room
    /// for (the engine bounds it to `nc_encoder::MAX_REFERENCE_AGE`).
    #[must_use]
    pub fn for_reach(reach: u64) -> Self {
        let delta_bits = reach
            .next_power_of_two()
            .trailing_zeros()
            .max(MIN_FRAME_ID_DELTA_BITS);
        assert!(
            delta_bits <= MAX_FRAME_ID_DELTA_BITS,
            "reference reach {reach}"
        );
        Self { delta_bits }
    }

    /// `idLen`: bits of `current_frame_id`.
    #[must_use]
    pub fn id_bits(self) -> u32 {
        // At least one more than the delta length: successive ids stay
        // within half the id range (§6.8.2), and a slot older than the delta
        // reach turns invalid (`mark_ref_frames`, §7.20) before its id could
        // come round again.
        (self.delta_bits + 1).max(MIN_FRAME_ID_BITS)
    }

    /// Bits of `delta_frame_id_minus_1`.
    #[must_use]
    pub fn delta_bits(self) -> u32 {
        self.delta_bits
    }
}

/// A frame's id syntax (§5.9.2, see the module notes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameId {
    /// The stream's field lengths.
    pub lengths: FrameIdLengths,
    /// `current_frame_id` (below 2^`idLen`).
    pub current: u32,
    /// Inter frames: `DeltaFrameId`, the current id minus the reference
    /// slot's (1..=2^ the delta length).
    pub reference_delta: u32,
}

/// The encoder's frame ids: the next frame's, and each slot's (`RefFrameId`).
#[derive(Debug, Clone)]
pub struct FrameIds {
    lengths: FrameIdLengths,
    next: u32,
    slots: [u32; MAX_REFERENCE_SLOTS],
}

impl FrameIds {
    /// Ids of a stream with `lengths`, starting at 0.
    #[must_use]
    pub fn new(lengths: FrameIdLengths) -> Self {
        Self {
            lengths,
            next: 0,
            slots: [0; MAX_REFERENCE_SLOTS],
        }
    }

    /// The stream's field lengths.
    #[must_use]
    pub fn lengths(&self) -> FrameIdLengths {
        self.lengths
    }

    /// The id syntax of the next frame, predicting from `reference_slot`
    /// unless it is a key frame.
    #[must_use]
    pub fn syntax(&self, kind: FrameKind, reference_slot: usize) -> FrameId {
        let range = 1 << self.lengths.id_bits();
        let reference_delta = match kind {
            FrameKind::Key => 0,
            FrameKind::Inter => (self.next + range - self.slots[reference_slot]) % range,
        };
        FrameId {
            lengths: self.lengths,
            current: self.next,
            reference_delta,
        }
    }

    /// The next frame was committed: the slots it refreshes (every slot
    /// for a key frame) now carry its id.
    pub fn commit(&mut self, key: bool, refresh: SlotMask) {
        for (slot, id) in self.slots.iter_mut().enumerate() {
            if key || refresh.contains(slot) {
                *id = self.next;
            }
        }
        self.next = (self.next + 1) % (1 << self.lengths.id_bits());
    }
}

/// The frame-size syntax of a frame (§5.9.5 – §5.9.7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameSizeSyntax {
    /// The sequence header's maximum frame size.
    pub stream: (usize, usize),
    /// Signalled display (`RenderWidth`, `RenderHeight`) dimensions (§5.9.6):
    /// `stream` on single-layer streams, or the layer's own display size on
    /// spatial SVC streams (§2.4).
    pub render: (usize, usize),
    /// `frame_size_override_flag`: the frame is coded at its own size
    /// (`width` x `height`) instead of the stream's.
    pub override_size: bool,
    /// Inter frames with `override_size`: `LAST_FRAME` (`ref_frame_idx[0]`)
    /// already holds this frame's coded and render sizes, so `found_ref`
    /// copies them (§5.9.7) instead of coding them.
    pub found_ref: bool,
}

/// The reference slots an inter (or intra-only) frame reads and refreshes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotUse {
    /// `ref_frame_idx[0]` (`LAST_FRAME`: primary reference).
    pub reference: u8,
    /// Optional `ref_frame_idx[1..7]` (`LAST2_FRAME`: secondary / inter-layer
    /// reference); defaults to `reference` when `None`.
    pub secondary: Option<u8>,
    /// `refresh_frame_flags`.
    pub refresh: u8,
}

impl FrameHeader {
    /// Write `frame_header_obu()` for a shown frame (AV1 §5.9.2), followed by
    /// `byte_alignment()` as required inside `OBU_FRAME`.
    pub fn write(&self, w: &mut BitWriter) {
        let inter = self.kind == FrameKind::Inter;
        let intra_only = !inter && self.intra_refresh == IntraRefresh::SelectedSlots;
        let frame_type = if inter {
            FRAME_TYPE_INTER
        } else if intra_only {
            FRAME_TYPE_INTRA_ONLY
        } else {
            FRAME_TYPE_KEY
        };
        w.put_bit(false); // show_existing_frame
        w.put_bits(frame_type, 2); // frame_type: KEY (0) / INTER (1) / INTRA_ONLY (2)
        w.put_bit(true); // show_frame
        // showable_frame: not coded for shown frames.
        if inter || intra_only {
            // Implied (1) for shown key frames; inter and intra-only frames
            // code error_resilient_mode explicitly (AV1 §5.9.2).
            w.put_bit(false); // error_resilient_mode
        }
        w.put_bit(self.disable_cdf_update); // disable_cdf_update
        // allow_screen_content_tools = seq_force_screen_content_tools (0): not
        // coded, so force_integer_mv = 0 is not coded either.
        if let Some(id) = self.frame_id {
            w.put_bits(id.current, id.lengths.id_bits()); // current_frame_id
        }
        w.put_bit(self.size.override_size); // frame_size_override_flag
        // order_hint: OrderHintBits = 0.
        if inter {
            w.put_bits(PRIMARY_REF_LAST, 3); // primary_ref_frame
            w.put_bits(u32::from(self.references.refresh), 8); // refresh_frame_flags
            // frame_refs_short_signaling: not coded without order hints.
            if let Some(id) = self.frame_id {
                // The engine bounds every reference to the reach the
                // lengths were sized for (`nc_encoder::reference_reach`).
                debug_assert!(
                    (1..=(1 << id.lengths.delta_bits())).contains(&id.reference_delta),
                    "reference {} frames back",
                    id.reference_delta
                );
            }
            let secondary = self
                .references
                .secondary
                .unwrap_or(self.references.reference);
            for i in 0..REFS_PER_FRAME {
                let slot = if i == 0 {
                    self.references.reference
                } else {
                    secondary
                };
                w.put_bits(u32::from(slot), 3); // ref_frame_idx[i]
                if let Some(id) = self.frame_id {
                    let delta_bits = id.lengths.delta_bits();
                    w.put_bits(id.reference_delta - 1, delta_bits); // delta_frame_id_minus_1
                }
            }
            if self.size.override_size {
                // error_resilient_mode is 0.
                self.write_frame_size_with_refs(w);
            } else {
                self.write_frame_size(w);
                self.write_render_size(w);
            }
            w.put_bit(false); // allow_high_precision_mv: quarter-sample vectors
            w.put_bit(false); // is_filter_switchable
            w.put_bits(INTERP_FILTER_EIGHTTAP, 2); // interpolation_filter
            w.put_bit(false); // is_motion_mode_switchable: SIMPLE only
        // use_ref_frame_mvs: not coded (enable_ref_frame_mvs = 0).
        } else {
            // primary_ref_frame: PRIMARY_REF_NONE implied for intra frames.
            if intra_only {
                // AV1 §5.9.2 / §6.8.2: INTRA_ONLY_FRAME codes refresh_frame_flags != 0xFF.
                w.put_bits(u32::from(self.references.refresh), 8);
            }
            self.write_frame_size(w);
            self.write_render_size(w);
            // allow_intrabc: not coded without screen content tools.
        }
        // The tile's final CDFs are saved with the frame for the next one;
        // with disable_cdf_update the flag is implied 1 and not coded.
        if !self.disable_cdf_update {
            w.put_bit(false); // disable_frame_end_update_cdf
        }
        self.write_tile_info(w);
        // quantization_params()
        w.put_bits(self.base_q_idx, 8); // base_q_idx
        w.put_bit(false); // DeltaQYDc: delta_coded
        // DeltaQUDc, DeltaQUAc (diff_uv_delta not coded): delta_coded,
        // then delta_q su(1+6).
        for _ in 0..2 {
            w.put_bit(self.chroma_delta_q != 0);
            if self.chroma_delta_q != 0 {
                w.put_bits((self.chroma_delta_q as u32) & 0x7f, 7);
            }
        }
        w.put_bit(false); // using_qmatrix
        w.put_bit(false); // segmentation_enabled
        // delta_q_params() (coded because base_q_idx > 0), §5.9.17.
        w.put_bit(self.delta_q_present);
        if self.delta_q_present {
            w.put_bits(0, 2); // delta_q_res: deltas in qindex units
            // delta_lf_params(), §5.9.18: allow_intrabc is 0, so
            // delta_lf_present is coded; the loop filter stays frame-level.
            w.put_bit(false);
        }
        self.write_in_loop_filters(w);
        w.put_bit(self.tx_mode_select); // tx_mode_select (else TX_MODE_LARGEST)
        if inter {
            w.put_bit(self.reference_select); // reference_select (§5.9.23)
            // skip_mode_present: not coded without order hints.
            // allow_warped_motion: not coded (enable_warped_motion = 0).
        }
        w.put_bit(false); // reduced_tx_set
        if inter {
            // global_motion_params(): identity model for every reference.
            for _ in 0..REFS_PER_FRAME {
                w.put_bit(false); // is_global
            }
        }
        if let Some(grain) = &self.film_grain {
            grain.write(w, inter); // film_grain_params()
        }
        w.byte_align_zero();
    }

    /// `frame_size()` (§5.9.5): the size itself only under
    /// `frame_size_override_flag` (otherwise the sequence header's), then
    /// `superres_params()`.
    fn write_frame_size(&self, w: &mut BitWriter) {
        if self.size.override_size {
            let (stream_width, stream_height) = self.size.stream;
            w.put_bits((self.width - 1) as u32, bits_for(stream_width - 1)); // frame_width_minus_1
            w.put_bits((self.height - 1) as u32, bits_for(stream_height - 1)); // frame_height_minus_1
        }
        self.write_superres_params(w);
    }

    /// `render_size()` (§5.9.6): renders at `self.size.render`.
    fn write_render_size(&self, w: &mut BitWriter) {
        let upscaled_width = if self.size.override_size {
            self.width
        } else {
            self.size.stream.0
        };
        let (render_width, render_height) = self.size.render;
        let different = (upscaled_width, self.height) != (render_width, render_height);
        w.put_bit(different); // render_and_frame_size_different
        if different {
            w.put_bits((render_width - 1) as u32, RENDER_SIZE_BITS); // render_width_minus_1
            w.put_bits((render_height - 1) as u32, RENDER_SIZE_BITS); // render_height_minus_1
        }
    }

    /// `frame_size_with_refs()` (§5.9.7): `found_ref` on the first
    /// reference (every `ref_frame_idx` names the same slot) when it has
    /// this frame's size, else all seven unset and the size coded.
    fn write_frame_size_with_refs(&self, w: &mut BitWriter) {
        if self.size.found_ref {
            w.put_bit(true); // found_ref
            self.write_superres_params(w);
            return;
        }
        for _ in 0..REFS_PER_FRAME {
            w.put_bit(false); // found_ref
        }
        self.write_frame_size(w);
        self.write_render_size(w);
    }

    /// `tile_info()` for a single uniformly spaced tile (AV1 §5.9.15).
    /// `superres_params()` (§5.9.8), coded only when the sequence enables
    /// superres.
    fn write_superres_params(&self, w: &mut BitWriter) {
        let Some(denom) = self.superres_denom else {
            return;
        };
        let use_superres = denom != SUPERRES_NUM;
        w.put_bit(use_superres); // use_superres
        if use_superres {
            w.put_bits((denom - SUPERRES_DENOM_MIN) as u32, SUPERRES_DENOM_BITS); // coded_denom
        }
    }

    /// `loop_filter_params()`, `cdef_params()` and `lr_params()`
    /// (§5.9.11, §5.9.19, §5.9.20).
    fn write_in_loop_filters(&self, w: &mut BitWriter) {
        self.loop_filter.write(w);
        self.cdef.write(w, self.cdef_alternate);
        if let Some(restoration) = self.restoration {
            restoration.write(w);
        }
    }

    fn write_tile_info(&self, w: &mut BitWriter) {
        let mi_cols = 2 * self.width.div_ceil(8);
        let mi_rows = 2 * self.height.div_ceil(8);
        let sb_shift = SUPERBLOCK_LOG2 - 2;
        let sb_cols = (mi_cols + (1 << sb_shift) - 1) >> sb_shift;
        let sb_rows = (mi_rows + (1 << sb_shift) - 1) >> sb_shift;
        let max_tile_width_sb = MAX_TILE_WIDTH >> SUPERBLOCK_LOG2;
        let max_tile_area_sb = MAX_TILE_AREA >> (2 * SUPERBLOCK_LOG2);
        let min_log2_tile_cols = tile_log2(max_tile_width_sb, sb_cols);
        let max_log2_tile_cols = tile_log2(1, sb_cols.min(MAX_TILE_COLS));
        let max_log2_tile_rows = tile_log2(1, sb_rows.min(MAX_TILE_ROWS));
        let min_log2_tiles = min_log2_tile_cols.max(tile_log2(max_tile_area_sb, sb_rows * sb_cols));
        assert!(min_log2_tiles == 0, "frame too large for a single tile");

        w.put_bit(true); // uniform_tile_spacing_flag
        if min_log2_tile_cols < max_log2_tile_cols {
            w.put_bit(false); // increment_tile_cols_log2: stay at 1 column
        }
        if max_log2_tile_rows > 0 {
            w.put_bit(false); // increment_tile_rows_log2: stay at 1 row
        }
        // One tile: context_update_tile_id and tile_size_bytes are not coded.
    }
}

/// Smallest `k` with `block << k >= target` (AV1 §5.9.15 `tile_log2`).
fn tile_log2(block: usize, target: usize) -> u32 {
    let mut k = 0;
    while (block << k) < target {
        k += 1;
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obu_header_byte_and_size() {
        let mut out = Vec::new();
        write_obu(&mut out, ObuType::TemporalDelimiter, None, &[]);
        assert_eq!(out, vec![0x12, 0x00]);
        out.clear();
        write_obu(&mut out, ObuType::Frame, Some(2), &[7]);
        assert_eq!(out, vec![0x36, 0x40, 0x01, 0x07]);
    }

    #[test]
    fn operating_points_drop_top_layers() {
        assert_eq!(operating_point_idc(1, 0), 0);
        assert_eq!(operating_point_idc(3, 0), 0x107);
        assert_eq!(operating_point_idc(3, 1), 0x103);
        assert_eq!(operating_point_idc(3, 2), 0x101);
    }

    #[test]
    fn frame_ids_count_frames_and_name_each_slot_by_distance() {
        let lengths = FrameIdLengths::for_reach(4);
        let mut ids = FrameIds::new(lengths);
        ids.commit(true, SlotMask::ALL);
        for _ in 0..3 {
            ids.commit(false, SlotMask::only(0));
        }
        // Slot 0 holds frame 3, slot 1 the key (frame 0).
        assert_eq!(ids.syntax(FrameKind::Inter, 0).reference_delta, 1);
        assert_eq!(ids.syntax(FrameKind::Inter, 1).reference_delta, 4);
        assert_eq!(ids.syntax(FrameKind::Key, 1).current, 4);
        for _ in 0..(1 << lengths.id_bits()) {
            ids.commit(false, SlotMask::only(0));
        }
        // The id wraps; the distance does not.
        assert_eq!(ids.syntax(FrameKind::Inter, 0).reference_delta, 1);
    }

    #[test]
    fn frame_id_lengths_are_the_shortest_the_reach_allows() {
        let delta_bits = |reach| FrameIdLengths::for_reach(reach).delta_bits();
        assert_eq!(delta_bits(1), MIN_FRAME_ID_DELTA_BITS);
        assert_eq!(delta_bits(4), 2);
        assert_eq!(delta_bits(5), 3);
        assert_eq!(delta_bits(60), 6);
        assert_eq!(delta_bits(nc_encoder::MAX_REFERENCE_AGE - 1), 8);
        assert_eq!(FrameIdLengths::for_reach(4).id_bits(), MIN_FRAME_ID_BITS);
        assert_eq!(FrameIdLengths::for_reach(255).id_bits(), MIN_FRAME_ID_BITS);
    }

    #[test]
    fn tile_log2_examples() {
        assert_eq!(tile_log2(1, 20), 5);
        assert_eq!(tile_log2(64, 20), 0);
        assert_eq!(tile_log2(1, 1), 0);
    }

    #[test]
    fn select_seq_level_idx_respects_rate_and_compression_limits() {
        let rate_30 = FrameRate { num: 30, den: 1 };
        let pic_720p = 1280 * 720;
        let decode_rate_720p30 = pic_720p * 30;
        // At 1280x720@30 without bitrate constraint, level 3.1 (idx 5) fits.
        assert_eq!(
            select_seq_level_idx((1280, 720), rate_30, decode_rate_720p30, 30, None, false),
            5
        );
        // Level 3.1 MaxBitrate is 10 Mbps; at 10 Mbps it still fits level 3.1 (idx 5),
        // provided MinCompRatio allows it: for level 3.1 (MinCompBasis = 4, MaxDisplayRate = 27_648_000),
        // max bps allowed by MinCompRatio is 12 * 27_648_000 / 4 = 82_944_000 bps.
        assert_eq!(
            select_seq_level_idx(
                (1280, 720),
                rate_30,
                decode_rate_720p30,
                30,
                Some(10_000_000),
                false,
            ),
            5
        );
        // At 10_000_001 bps, level 3.1 MaxBitrate (10 Mbps) is exceeded; bumps to level 4.0 (idx 8, 12 Mbps).
        assert_eq!(
            select_seq_level_idx(
                (1280, 720),
                rate_30,
                decode_rate_720p30,
                30,
                Some(10_000_001),
                false,
            ),
            8
        );
        // For tiny QCIF (176x144@30), uncompressed_bps = 12 * 176 * 144 * 30 = 9_123_840 bps.
        // MinCompRatio >= 0.8 requires 4 * bps <= 5 * uncompressed_bps => bps <= 11_404_800 bps.
        // Level 2.0 (idx 0) has MaxBitrate = 1_500_000 bps; at 1_500_000 bps it fits level 2.0.
        let pic_qcif = 176 * 144;
        assert_eq!(
            select_seq_level_idx(
                (176, 144),
                rate_30,
                pic_qcif * 30,
                30,
                Some(1_500_000),
                false
            ),
            0
        );
        // At 12_000_000 bps on QCIF, MinCompRatio (0.8 floor) is violated on every defined level,
        // so select_seq_level_idx falls back to SEQ_LEVEL_MAX_PARAMETERS (31).
        assert_eq!(
            select_seq_level_idx(
                (176, 144),
                rate_30,
                pic_qcif * 30,
                30,
                Some(12_000_000),
                false,
            ),
            SEQ_LEVEL_MAX_PARAMETERS
        );
        // High frame rate 4K (3840x2160@120) exceeds level 5.1 (idx 13) MaxDisplayRate,
        // fits level 5.2 (idx 14), and 8K@120 fits level 6.2 (idx 18) or 6.3 (idx 19).
        let rate_120 = FrameRate { num: 120, den: 1 };
        let pic_4k = 3840 * 2160;
        assert_eq!(
            select_seq_level_idx((3840, 2160), rate_120, pic_4k * 120, 120, None, false),
            14
        );
        // Unconstrained mode always selects SEQ_LEVEL_MAX_PARAMETERS (31).
        assert_eq!(
            select_seq_level_idx((1280, 720), rate_30, decode_rate_720p30, 30, None, true),
            SEQ_LEVEL_MAX_PARAMETERS
        );
    }
}
