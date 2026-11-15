//! The complete AV2 frame header (`uncompressed_header` of every frame OBU:
//! key frames, regular and leading tile groups, switch, random-access,
//! bridge, TIP and show-existing OBUs), every field of AVM v1.0.0 for 8-bit
//! 4:2:0 streams.
//!
//! Field order and derivations follow dav2d `obu.c` `parse_frame_hdr`
//! (commit 237229e) and AVM `read_uncompressed_header`. The in-loop filter
//! parameters are in [`filters`], loop restoration's in [`restoration`],
//! global motion's in [`global_motion`].
//!
//! Parsing reports each of dav2d's `DEBUG_FRAME_HDR` checkpoints (`post-ids`,
//! `post-quant`, ...) with its bit offset through a [`Trace`] hook, so a
//! stream's headers can be held against dav2d's trace field by field
//! (`examples/dump_headers.rs`, docs/handoff/av2.md).
pub mod filters;
pub mod global_motion;
pub mod restoration;

use std::rc::Rc;

use super::bits::{signed, uniform};
use super::references::{self, Current, MAX_REFERENCES, SlotInfo};
use super::sequence::{SegmentFeatures, SequenceHeader, Tri};
use super::tiles::{self, TileGeometry, TileLayout};
use filters::{Ccso, Cdef, Deblock, Gdf};
use global_motion::GlobalMotion;
use nc_common::decode::{BitReader, DecodeError, Result};
use restoration::Restoration;

/// Most reference slots of a sequence (AVM `REF_FRAMES` extended: dav2d
/// `DAV2D_NUM_REF_FRAMES`).
pub const MAX_SLOTS: usize = 16;

/// The frame OBU types (AV2 §6.2, dav2d `Dav2dObuType`).
pub mod obu_type {
    /// `OBU_CLOSED_LOOP_KEY`.
    pub const CLOSED_LOOP_KEY: u8 = 4;
    /// `OBU_OPEN_LOOP_KEY`.
    pub const OPEN_LOOP_KEY: u8 = 5;
    /// `OBU_LEADING_TILE_GROUP`.
    pub const LEADING_TILE_GROUP: u8 = 6;
    /// `OBU_REGULAR_TILE_GROUP`.
    pub const TILE_GROUP: u8 = 7;
    /// `OBU_SWITCH`.
    pub const SWITCH: u8 = 10;
    /// `OBU_LEADING_SEF` (show-existing frame).
    pub const LEADING_SHOW_EXISTING: u8 = 11;
    /// `OBU_REGULAR_SEF`.
    pub const SHOW_EXISTING: u8 = 12;
    /// `OBU_LEADING_TIP`.
    pub const LEADING_TIP: u8 = 13;
    /// `OBU_REGULAR_TIP`.
    pub const TIP: u8 = 14;
    /// `OBU_BRIDGE_FRAME`.
    pub const BRIDGE: u8 = 19;
    /// `OBU_RAS_FRAME` (random-access switch).
    pub const RANDOM_ACCESS_SWITCH: u8 = 21;

    /// Whether OBUs of `kind` carry a frame header.
    #[must_use]
    pub fn is_frame(kind: u8) -> bool {
        matches!(
            kind,
            CLOSED_LOOP_KEY
                | OPEN_LOOP_KEY
                | LEADING_TILE_GROUP
                | TILE_GROUP
                | SWITCH
                | LEADING_SHOW_EXISTING
                | SHOW_EXISTING
                | LEADING_TIP
                | TIP
                | BRIDGE
                | RANDOM_ACCESS_SWITCH
        )
    }

    /// Whether OBUs of `kind` hold a frame header only, no tile data
    /// (dav2d `hdr_only`).
    #[must_use]
    pub fn is_header_only(kind: u8) -> bool {
        matches!(
            kind,
            LEADING_SHOW_EXISTING | SHOW_EXISTING | LEADING_TIP | TIP | BRIDGE
        )
    }
}

/// Bits of a TIP frame's global weight index.
const TIP_WEIGHT_BITS: u32 = 3;
/// Bits of each TIP frame motion component's magnitude.
const TIP_MV_BITS: u32 = 4;
/// Bits of `primary_ref_frame`, of a signalled reference count, and of a
/// film grain parameter set id.
const SMALL_INDEX_BITS: u32 = 3;
/// Bits of the film grain random seed.
const GRAIN_SEED_BITS: u32 = 16;
/// Bits of the interpolation filter of a non-switchable frame.
const FILTER_BITS: u32 = 2;
/// Bits of a tile-size field width (`tile_size_bytes_minus_1`).
const TILE_SIZE_BYTES_BITS: u32 = 2;
/// Bits of a frame quantizer delta (sign included).
const QUANT_DELTA_BITS: u32 = 7;
/// Bits of a quantizer-matrix level, and of the matrix-set count.
const QM_LEVEL_BITS: u32 = 4;
const QM_COUNT_BITS: u32 = 2;
/// Bits of `log2(delta_q_res)`.
const DELTA_Q_RES_BITS: u32 = 2;
/// Bits of `reduced_tx_set`.
const REDUCED_TX_SET_BITS: u32 = 2;
/// Largest display order hint (AVM limit, dav2d `1 << 30`).
const MAX_ORDER: u32 = 1 << 30;
/// Choices of `max_drl_bits` and of `max_bvp_drl_bits` overrides.
const DRL_BITS_CHOICES: u32 = 5;
const BVP_DRL_BITS_CHOICES: u32 = 3;
/// Largest 8-bit quantizer index.
const MAX_QINDEX: i32 = 255;

/// A frame's kind (dav2d `Dav2dFrameType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FrameType {
    /// A key frame (closed or open loop).
    #[default]
    Key,
    /// An inter frame.
    Inter,
    /// An intra-only frame.
    Intra,
    /// A switch frame (switch and random-access OBUs).
    Switch,
}

impl FrameType {
    /// Key or intra-only: no references.
    #[must_use]
    pub fn is_intra(self) -> bool {
        matches!(self, Self::Key | Self::Intra)
    }
}

/// Motion modes a frame allows, as bits of [`MotionModes::mask`] (dav2d
/// `enum MotionMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MotionModes {
    /// Bit 0 translation (always), 1 inter-intra, 2 causal warp, 3 warp
    /// delta, 4 warp extension.
    pub mask: u8,
}

impl MotionModes {
    /// Translation only.
    pub const TRANSLATION: Self = Self { mask: 1 };
    /// Bit of warp-delta motion.
    pub const WARP_DELTA: u8 = 1 << 3;
    /// Bits of the modes beyond translation, lowest first.
    const OPTIONAL: [u8; 4] = [1 << 1, 1 << 2, 1 << 3, 1 << 4];

    /// The modes a sequence allows.
    #[must_use]
    pub fn of_sequence(sequence: &SequenceHeader) -> Self {
        let modes = sequence.inter.motion_modes;
        let flags = [
            modes.inter_intra,
            modes.warp_causal,
            modes.warp_delta,
            modes.warp_extend,
        ];
        let mut mask = Self::TRANSLATION.mask;
        for (bit, on) in Self::OPTIONAL.iter().zip(flags) {
            if on {
                mask |= bit;
            }
        }
        Self { mask }
    }
}

/// Temporal interpolated prediction of the frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Tip {
    /// 0 off, 1 a TIP reference for blocks, 2 the whole frame is the TIP
    /// prediction (TIP OBUs).
    pub mode: u8,
    /// `tip_frame_hole_fill`.
    pub hole_fill: bool,
    /// `tip_global_wtd_index`.
    pub global_weight: u8,
    /// The frame-level TIP picture is deblocked.
    pub apply_filter: bool,
    /// The whole-frame motion offset (row, column).
    pub motion: (i8, i8),
    /// Its interpolation filter (dav2d `Dav2dFilterMode`).
    pub filter: u8,
    /// The two reference ranks it interpolates between.
    pub references: [u8; 2],
}

/// dav2d's interpolation filter numbers (`Dav2dFilterMode`).
pub mod filter_mode {
    /// `DAV2D_FILTER_8TAP_REGULAR`.
    pub const REGULAR: u8 = 0;
    /// `DAV2D_FILTER_8TAP_SMOOTH`.
    pub const SMOOTH: u8 = 1;
    /// `DAV2D_FILTER_8TAP_SHARP`.
    pub const SHARP: u8 = 2;
    /// `DAV2D_FILTER_SWITCHABLE`.
    pub const SWITCHABLE: u8 = 4;
}

/// Intra block copy as a frame allows it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IntraBlockCopy {
    /// `allow_global_intrabc`: from anywhere already decoded.
    pub global: bool,
    /// `allow_local_intrabc`: from the current superblock's area.
    pub local: bool,
    /// `max_bvp_drl_bits`.
    pub max_drl_bits: u8,
}

/// The frame's quantizers (`quantization_params`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Quantizer {
    /// `base_q_idx`.
    pub base: u16,
    /// Deltas to `base`: luma DC, U DC and AC, V DC and AC.
    pub y_dc: i8,
    /// U DC delta.
    pub u_dc: i8,
    /// U AC delta.
    pub u_ac: i8,
    /// V DC delta.
    pub v_dc: i8,
    /// V AC delta.
    pub v_ac: i8,
    /// Quantizer matrices, when used.
    pub matrices: Option<QuantMatrices>,
}

/// The quantizer-matrix levels of each matrix set (`qm_y`, `qm_u`, `qm_v`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct QuantMatrices {
    /// Sets used: one, or up to four with segmentation.
    pub count: u8,
    /// Luma matrix level of each set.
    pub y: [u8; 4],
    /// U matrix level of each set.
    pub u: [u8; 4],
    /// V matrix level of each set.
    pub v: [u8; 4],
}

/// The frame's segmentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Segmentation {
    /// The segment features in force.
    pub features: SegmentFeatures,
    /// `segmentation_update_map`.
    pub update_map: bool,
    /// `segmentation_temporal_update`.
    pub temporal_update: bool,
    /// Segment ids are read before the skip flag (a segment skips or uses
    /// global motion).
    pub preskip: bool,
    /// The highest segment with any feature, if any.
    pub last_active: Option<u8>,
}

/// Segment-derived quantizers and lossless flags (dav2d
/// `segmentation.qidx`, `lossless`, `all_lossless`, `any_lossless`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Lossless {
    /// Each segment's quantizer index.
    pub qindex: [u16; super::sequence::MAX_SEGMENTS],
    /// Segments coded losslessly, as a bit mask.
    pub segments: u16,
    /// Every segment is lossless.
    pub all: bool,
}

/// Film grain applied to the frame's output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FilmGrain {
    /// The film grain model (FGM OBU parameter set) it uses.
    pub id: u8,
    /// `grain_seed`.
    pub seed: u16,
}

/// A show-existing-frame OBU: output the picture of a slot again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShowExisting {
    /// `frame_to_show_map_idx`.
    pub slot: u8,
    /// Film grain for this output.
    pub film_grain: Option<FilmGrain>,
}

/// One frame header: every field AVM v1.0.0 codes, with what dav2d derives
/// from them and from the reference slots.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[allow(clippy::struct_excessive_bools, reason = "the header's flags")]
pub struct FrameHeader {
    /// The OBU type the header came in ([`obu_type`]).
    pub obu_type: u8,
    /// Key, inter, intra-only or switch.
    pub frame_type: FrameType,
    /// Temporal and embedded layer (OBU extension).
    pub temporal_layer: u8,
    /// Embedded (spatial/quality) layer.
    pub embedded_layer: u8,
    /// Long-term frame id of a key frame (`ltr_id`), if coded.
    pub long_term_id: Option<u8>,
    /// Long-term ids of the references of open-loop key and random-access
    /// frames.
    pub long_term_references: Vec<u8>,
    /// `restricted_prediction_switch` of switch frames.
    pub restricted_prediction_switch: bool,
    /// Output at once (`show_frame`).
    pub show_immediate: bool,
    /// Output later in display order (`implicit_output_frame`).
    pub show_implicit: bool,
    /// `frame_size_override_flag`.
    pub frame_size_override: bool,
    /// The coded order hint.
    pub order_hint: u16,
    /// Display order: the order hint unwrapped (dav2d `frame_offset`).
    pub order: u32,
    /// `signal_primary_ref_frame`.
    pub primary_signalled: bool,
    /// `disable_cross_frame_cdf_init` (no CDFs from a reference).
    pub no_cross_frame_context: bool,
    /// The reference rank whose CDFs the frame starts from, and the second
    /// one averaging uses.
    pub primary_reference: Option<u8>,
    /// The second CDF source of averaging.
    pub secondary_reference: Option<u8>,
    /// Slots the frame replaces (`refresh_frame_flags`).
    pub refresh: u16,
    /// The frame names its references (`explicit_ref_frame_map`).
    pub explicit_references: bool,
    /// The slot of each reference rank (`ref_frame_idx`): `reference_count`
    /// distinct ones, the rest repeat the first.
    pub references: [u8; MAX_REFERENCES],
    /// `num_total_refs`.
    pub reference_count: u8,
    /// Display order of each reference (dav2d `refpoc`), kept for later
    /// frames' global motion.
    pub reference_orders: [u32; MAX_REFERENCES],
    /// References after / before the frame in display order.
    pub has_future_references: bool,
    /// Some reference precedes the frame in display order.
    pub has_past_references: bool,
    /// Luma size.
    pub width: u32,
    /// Luma height.
    pub height: u32,
    /// `use_ref_frame_mvs`, and the motion field sample step.
    pub use_motion_fields: bool,
    /// Temporal motion field sample step (1 or 2, in 8x8 units).
    pub motion_field_step: u8,
    /// Optical-flow refinement: 0 off, 1 switchable, 2 every block.
    pub optical_flow: u8,
    /// Temporal interpolated prediction.
    pub tip: Tip,
    /// log2 of the frame's superblock size.
    pub superblock_log2: u8,
    /// The frame's tiles.
    pub tiles: TileLayout,
    /// `context_update_tile_id`.
    pub context_update_tile: u16,
    /// `tile_size_bytes`.
    pub tile_size_bytes: u8,
    /// `allow_screen_content_tools`, `cur_frame_force_integer_mv`.
    pub screen_content: bool,
    /// Motion vectors are whole samples.
    pub force_integer_mv: bool,
    /// Intra block copy, when allowed.
    pub intra_block_copy: Option<IntraBlockCopy>,
    /// `max_drl_bits`.
    pub max_drl_bits: u8,
    /// Motion vector precision: 0 integer, 1 half, 2 quarter, 3 eighth.
    pub mv_precision: u8,
    /// The frame's interpolation filter ([`filter_mode`]).
    pub filter: u8,
    /// The motion modes blocks may use.
    pub motion_modes: MotionModes,
    /// `disable_cdf_update`.
    pub disable_cdf_update: bool,
    /// The quantizers.
    pub quantizer: Quantizer,
    /// Segmentation, when enabled.
    pub segmentation: Option<Segmentation>,
    /// Each segment's quantizer and lossless flag.
    pub lossless: Lossless,
    /// `log2(delta_q_res)` when superblocks code quantizer deltas.
    pub delta_q: Option<u8>,
    /// `tcq_mode`, `allow_parity_hiding`.
    pub tcq: bool,
    /// Luma DC parity hiding.
    pub parity_hiding: bool,
    /// Deblocking.
    pub deblock: Deblock,
    /// The guided detail filter.
    pub gdf: Gdf,
    /// CDEF, when enabled.
    pub cdef: Option<Cdef>,
    /// Loop restoration.
    pub restoration: Restoration,
    /// Cross-component sample offsets.
    pub ccso: Ccso,
    /// `TX_MODE_SELECT` (else the largest transforms).
    pub tx_mode_select: bool,
    /// `reference_select` (compound references allowed).
    pub reference_select: bool,
    /// `skip_mode_present`.
    pub skip_mode: bool,
    /// Block-adaptive weighted prediction.
    pub bawp: bool,
    /// `allow_warpmv_mode`.
    pub warp_delta: bool,
    /// `reduced_tx_set`.
    pub reduced_tx_set: u8,
    /// Global motion of each reference.
    pub global_motion: GlobalMotion,
    /// Film grain for the output, when applied.
    pub film_grain: Option<FilmGrain>,
}

impl FrameHeader {
    /// Key, intra-only: no references.
    #[must_use]
    pub fn is_intra(&self) -> bool {
        self.frame_type.is_intra()
    }

    /// Output now or later (dav2d `showable`).
    #[must_use]
    pub fn is_shown(&self) -> bool {
        self.show_immediate || self.show_implicit
    }

    /// What reference ranking needs of this frame once it is in a slot;
    /// `frame` identifies the decoded frame (slots it fills share it).
    #[must_use]
    pub fn slot_info(&self, frame: u32) -> SlotInfo {
        SlotInfo {
            frame,
            order: self.order,
            quantizer: self.quantizer.base.min(MAX_QINDEX as u16) as u8,
            inter: self.frame_type == FrameType::Inter,
            temporal_layer: self.temporal_layer,
            size: (self.width, self.height),
        }
    }
}

/// A parsed frame OBU header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Parsed {
    /// Output a slot's picture again.
    ShowExisting(ShowExisting),
    /// A frame to decode (or, for TIP OBUs, to predict whole).
    Frame(Box<FrameHeader>),
}

/// A frame header's reference slots: the header of the frame each holds
/// and its identity (slots one frame filled share it).
#[derive(Debug, Clone, Default)]
pub struct SlotHeader {
    /// The frame's header.
    pub header: Rc<FrameHeader>,
    /// Identity of the decoded frame.
    pub frame: u32,
}

/// Receives each of dav2d's `DEBUG_FRAME_HDR` checkpoints with the bit
/// offset from the start of the OBU payload.
pub type Trace<'t> = &'t mut dyn FnMut(&'static str, usize);

/// What a frame header depends on beyond its own bits.
#[derive(Debug)]
pub struct Context<'a> {
    /// The active sequence header.
    pub sequence: &'a SequenceHeader,
    /// The reference slots (`ref_frame_map`), `sequence.references.slots`
    /// of them.
    pub slots: &'a [Option<SlotHeader>],
    /// The OBU type and its layer ids.
    pub obu_type: u8,
    /// The OBU's temporal layer.
    pub temporal_layer: u8,
    /// The OBU's embedded layer.
    pub embedded_layer: u8,
}

/// Reads one header with its trace.
struct Parser<'a, 'r, 'd, 't> {
    context: &'r Context<'a>,
    bits: &'r mut BitReader<'d>,
    trace: Trace<'t>,
}

impl<'a> Parser<'a, '_, '_, '_> {
    fn checkpoint(&mut self, name: &'static str) {
        let position = self.bits.position();
        (self.trace)(name, position);
    }

    fn sequence(&self) -> &'a SequenceHeader {
        self.context.sequence
    }

    /// The header of the frame in `slot`.
    fn slot(&self, slot: u8) -> Result<&'a FrameHeader> {
        self.context
            .slots
            .get(usize::from(slot))
            .and_then(Option::as_ref)
            .map(|held| &*held.header)
            .ok_or(DecodeError::MissingKey)
    }

    /// The header of reference rank `rank`.
    fn reference(&self, header: &FrameHeader, rank: u8) -> Result<&'a FrameHeader> {
        self.slot(header.references[usize::from(rank)])
    }

    /// `a - b` in display order as the sequence counts it (dav2d
    /// `get_poc_diff`: zero without order hints).
    fn order_difference(&self, a: u32, b: u32) -> i32 {
        if self.sequence().inter.order_hint_bits == 0 {
            0
        } else {
            references::order_difference(a, b)
        }
    }
}

/// Parse a frame OBU's header from the start of its payload (after the
/// first-tile flag for OBUs with tile data, which the caller reads): every
/// field up to and including film grain. The tile-group fields that follow
/// are the caller's.
/// # Errors
/// Truncation, references to empty slots, out-of-range fields, and the
/// syntax dav2d leaves unimplemented (multi-frame headers, TIP frames with
/// an explicit quantizer).
pub fn parse(context: &Context<'_>, bits: &mut BitReader<'_>, trace: Trace<'_>) -> Result<Parsed> {
    let mut parser = Parser {
        context,
        bits,
        trace,
    };
    parser.parse()
}

impl Parser<'_, '_, '_, '_> {
    fn parse(&mut self) -> Result<Parsed> {
        let kind = self.context.obu_type;
        let multi_frame_header = if kind == obu_type::BRIDGE {
            0
        } else {
            self.bits.read_uvlc()?
        };
        if multi_frame_header != 0 {
            return Err(DecodeError::Unsupported("AV2 multi-frame headers"));
        }
        if self.bits.read_uvlc()? != u32::from(self.sequence().id) {
            return Err(DecodeError::Malformed("AV2 frame sequence id"));
        }
        self.checkpoint("post-ids");
        if matches!(
            kind,
            obu_type::SHOW_EXISTING | obu_type::LEADING_SHOW_EXISTING
        ) {
            return self.show_existing().map(Parsed::ShowExisting);
        }
        let mut header = FrameHeader {
            obu_type: kind,
            temporal_layer: self.context.temporal_layer,
            embedded_layer: self.context.embedded_layer,
            ..FrameHeader::default()
        };
        self.frame_type(&mut header)?;
        self.order_and_primary(&mut header)?;
        self.refresh(&mut header)?;
        if !header.is_intra() {
            self.reference_map(&mut header)?;
        }
        self.frame_size(&mut header)?;
        if !header.is_intra() {
            for rank in 0..MAX_REFERENCES {
                header.reference_orders[rank] = self.reference(&header, rank as u8)?.order;
            }
            self.motion_fields_and_tip(&mut header)?;
            if header.tip.mode == 2 {
                self.tip_frame(&mut header)?;
                self.film_grain(&mut header)?;
                return Ok(Parsed::Frame(Box::new(header)));
            }
        }
        self.coding_tools(&mut header)?;
        self.tiles(&mut header)?;
        self.quantizer(&mut header)?;
        self.segmentation(&mut header)?;
        self.quant_matrices(&mut header)?;
        self.delta_q_and_lossless(&mut header)?;
        filters::parse(self, &mut header)?;
        self.frame_modes(&mut header)?;
        if !header.is_intra() {
            global_motion::parse(self, &mut header)?;
        }
        self.film_grain(&mut header)?;
        Ok(Parsed::Frame(Box::new(header)))
    }

    /// A show-existing OBU: the slot, an optional order-hint check, film
    /// grain.
    fn show_existing(&mut self) -> Result<ShowExisting> {
        let sequence = self.sequence();
        let slot = self.bits.read_bits(sequence.references.slot_bits())? as u8;
        if slot >= sequence.references.slots {
            return Err(DecodeError::Malformed("AV2 show-existing slot"));
        }
        let order = self.slot(slot)?.order;
        let order_hint_bits = u32::from(sequence.inter.order_hint_bits);
        if !self.bits.read_bit()? {
            let coded = self.bits.read_bits(order_hint_bits)?;
            if coded != order {
                return Err(DecodeError::Malformed("AV2 show-existing order hint"));
            }
        }
        self.checkpoint("post-existing_frame_idx");
        let mut film_grain = None;
        if self.sequence().film_grain {
            film_grain = self.film_grain_fields()?;
            self.checkpoint("post-filmgrain");
        }
        Ok(ShowExisting { slot, film_grain })
    }

    /// The frame type, long-term ids and output flags.
    fn frame_type(&mut self, header: &mut FrameHeader) -> Result<()> {
        let sequence = self.sequence();
        if sequence.single_picture {
            header.frame_type = FrameType::Key;
            header.show_immediate = true;
            return Ok(());
        }
        let kind = header.obu_type;
        header.frame_type = match kind {
            obu_type::CLOSED_LOOP_KEY | obu_type::OPEN_LOOP_KEY => FrameType::Key,
            obu_type::RANDOM_ACCESS_SWITCH | obu_type::SWITCH => {
                header.restricted_prediction_switch = self.bits.read_bit()?;
                FrameType::Switch
            }
            obu_type::LEADING_TIP | obu_type::TIP | obu_type::BRIDGE => FrameType::Inter,
            _ if self.bits.read_bit()? => FrameType::Inter,
            _ => FrameType::Intra,
        };
        let id_bits = u32::from(sequence.references.long_term_id_bits);
        if header.frame_type == FrameType::Key {
            if id_bits != 0 {
                // Coded plus one; zero is "none" (dav2d `ltr_id` -1).
                header.long_term_id = (self.bits.read_bits(id_bits)? as u8).checked_sub(1);
            }
        } else if matches!(
            kind,
            obu_type::RANDOM_ACCESS_SWITCH | obu_type::OPEN_LOOP_KEY
        ) && id_bits != 0
        {
            let count = self.bits.read_bits(SMALL_INDEX_BITS)?;
            for _ in 0..count {
                header
                    .long_term_references
                    .push(self.bits.read_bits(id_bits)? as u8);
            }
        }
        if kind != obu_type::BRIDGE {
            if kind != obu_type::OPEN_LOOP_KEY {
                header.show_immediate = self.bits.read_bit()?;
            }
            if !header.show_immediate && !sequence.monotonic_output {
                header.show_implicit = self.bits.read_bit()?;
            }
        }
        self.checkpoint("post-frametype_bits");
        Ok(())
    }

    /// `frame_size_override_flag`, the order hint (unwrapped into display
    /// order past the latest shown reference) and the primary reference.
    fn order_and_primary(&mut self, header: &mut FrameHeader) -> Result<()> {
        if self.sequence().single_picture {
            return Ok(());
        }
        header.frame_size_override =
            header.frame_type == FrameType::Switch || self.bits.read_bit()?;
        let order_hint_bits = u32::from(self.sequence().inter.order_hint_bits);
        header.order_hint = self.bits.read_bits(order_hint_bits)? as u16;
        let latest = self
            .context
            .slots
            .iter()
            .flatten()
            .filter(|held| held.header.is_shown())
            .map(|held| held.header.order)
            .max()
            .unwrap_or(0);
        let range = 1_u32 << order_hint_bits;
        let half = range >> 1;
        let mut order = u32::from(header.order_hint);
        if latest >= order && latest - order >= half {
            order += (latest - order + half) & !(range - 1);
        }
        if order >= MAX_ORDER {
            return Err(DecodeError::Malformed("AV2 display order"));
        }
        header.order = order;
        if header.frame_type == FrameType::Inter {
            header.primary_signalled = self.bits.read_bit()?;
            if !matches!(header.obu_type, obu_type::LEADING_TIP | obu_type::TIP) {
                header.no_cross_frame_context = self.bits.read_bit()?;
            }
            if header.primary_signalled {
                header.primary_reference =
                    primary_rank(self.bits.read_bits(SMALL_INDEX_BITS)? as u8);
            }
        }
        self.checkpoint("post-frame_size_override_flag");
        Ok(())
    }

    /// `refresh_frame_flags`, in its short or full form.
    fn refresh(&mut self, header: &mut FrameHeader) -> Result<()> {
        let sequence = self.sequence();
        let slots = sequence.references.slots;
        let all = u16::MAX >> (16 - u32::from(slots));
        let slot_bits = sequence.references.slot_bits();
        let short = sequence.references.short_refresh_flags;
        let kind = header.obu_type;
        let layered = sequence.max_embedded_layer != 0;
        header.refresh = if kind == obu_type::CLOSED_LOOP_KEY && !layered {
            all
        } else if kind == obu_type::OPEN_LOOP_KEY || layered {
            if short {
                1 << self.bits.read_bits(slot_bits)?
            } else {
                self.bits.read_bits(u32::from(slots))? as u16
            }
        } else if header.frame_type != FrameType::Switch && short {
            if self.bits.read_bit()? {
                let slot = self.bits.read_bits(slot_bits)?;
                if slot >= u32::from(slots) {
                    return Err(DecodeError::Malformed("AV2 refreshed slot"));
                }
                1 << slot
            } else {
                0
            }
        } else {
            self.bits.read_bits(u32::from(slots))? as u16
        };
        self.checkpoint("post-refresh_frame_flags");
        Ok(())
    }

    /// The references of an inter or switch frame: named, or ranked from
    /// the slots (again with the frame size once it is known).
    fn reference_map(&mut self, header: &mut FrameHeader) -> Result<()> {
        let sequence = self.sequence();
        header.explicit_references = header.obu_type != obu_type::BRIDGE
            && (header.frame_type == FrameType::Switch
                || (sequence.references.explicit_map && self.bits.read_bit()?));
        if header.explicit_references {
            let count = self.bits.read_bits(SMALL_INDEX_BITS)? as u8;
            if usize::from(count) > MAX_REFERENCES.min(usize::from(sequence.references.slots)) {
                return Err(DecodeError::Malformed("AV2 reference count"));
            }
            for rank in 0..usize::from(count) {
                let slot = self.bits.read_bits(sequence.references.slot_bits())? as u8;
                if slot >= sequence.references.slots {
                    return Err(DecodeError::Malformed("AV2 reference slot"));
                }
                header.references[rank] = slot;
            }
            header.reference_count = count;
        } else {
            self.rank_references(header, None);
        }
        if header.reference_count == 0 {
            return Err(DecodeError::Malformed("AV2 inter frame without references"));
        }
        for rank in 0..header.reference_count {
            let order = self.reference(header, rank)?.order;
            let difference = self.order_difference(header.order, order);
            header.has_future_references |= difference < 0;
            header.has_past_references |= difference > 0;
        }
        self.checkpoint("post-refs");
        Ok(())
    }

    /// Rank the slots for an implicitly referenced frame (dav2d
    /// `get_ref_frames`), with the frame size when known.
    fn rank_references(&self, header: &mut FrameHeader, size: Option<(u32, u32)>) {
        let sequence = self.sequence();
        let dependencies = sequence.temporal_dependencies[usize::from(header.temporal_layer)];
        let infos: Vec<Option<SlotInfo>> = self
            .context
            .slots
            .iter()
            .map(|held| {
                held.as_ref()
                    .filter(|held| {
                        dependencies & (1 << held.header.temporal_layer) != 0
                            && held.header.embedded_layer <= header.embedded_layer
                    })
                    .map(|held| SlotInfo {
                        // Every layer the dependency map allows ranks.
                        temporal_layer: 0,
                        ..held.header.slot_info(held.frame)
                    })
            })
            .collect();
        let (ranked, count) = references::rank(
            &infos,
            Current {
                order: header.order,
                temporal_layer: 0,
                size,
            },
        );
        header.references = ranked;
        header.reference_count = count as u8;
    }

    /// The frame size: a reference's, coded, or the sequence maximum.
    fn frame_size(&mut self, header: &mut FrameHeader) -> Result<()> {
        let sequence = self.sequence();
        let mut size = None;
        if header.frame_size_override && header.frame_type == FrameType::Inter {
            for rank in 0..header.reference_count {
                if self.bits.read_bit()? {
                    let reference = self.reference(header, rank)?;
                    size = Some((reference.width, reference.height));
                    break;
                }
            }
        }
        let (width, height) = match size {
            Some(size) => size,
            None if header.frame_size_override => (
                self.bits.read_bits(u32::from(sequence.width_bits))? + 1,
                self.bits.read_bits(u32::from(sequence.height_bits))? + 1,
            ),
            None => (sequence.max_width, sequence.max_height),
        };
        header.width = width;
        header.height = height;
        self.checkpoint("post-framesize");
        if header.frame_type == FrameType::Inter && !sequence.references.explicit_map {
            self.rank_references(header, Some((width, height)));
            if header.reference_count == 0 {
                return Err(DecodeError::Malformed("AV2 no reference of a usable size"));
            }
            self.checkpoint("post-refs2");
        }
        Ok(())
    }

    /// Temporal motion fields, optical flow and TIP.
    fn motion_fields_and_tip(&mut self, header: &mut FrameHeader) -> Result<()> {
        let sequence = self.sequence();
        let inter = sequence.inter;
        if inter.motion_fields && header.frame_type == FrameType::Inter {
            header.use_motion_fields = self.bits.read_bit()?;
        }
        header.motion_field_step = 1 + u8::from(
            header.use_motion_fields
                && header.reference_count > 1
                && sequence.partition.superblock_log2 > SMALLEST_SUPERBLOCK_LOG2
                && self.bits.read_bit()?,
        );
        self.checkpoint("post-refmvbits");
        header.tip.filter = filter_mode::SHARP;
        let tip_frame = matches!(header.obu_type, obu_type::TIP | obu_type::LEADING_TIP);
        if inter.tip != 0 && header.reference_count > 1 && header.use_motion_fields {
            if tip_frame {
                header.tip.mode = 2;
                header.optical_flow = 2 * u8::from(inter.optical_flow != 0 && inter.tip_refine_mv);
            } else {
                header.tip.mode = u8::from(self.bits.read_bit()?);
                header.optical_flow = self.optical_flow()?;
            }
            if header.tip.mode != 0 {
                self.tip_parameters(header)?;
            }
            header.tip.references = self.tip_references(header)?;
        } else {
            header.optical_flow = self.optical_flow()?;
        }
        self.checkpoint("post-refinemv-tip");
        Ok(())
    }

    /// The frame's optical-flow refinement: the sequence's, or coded when
    /// the sequence leaves it to frames.
    fn optical_flow(&mut self) -> Result<u8> {
        let sequence_mode = self.sequence().inter.optical_flow;
        if sequence_mode < OPTICAL_FLOW_PER_FRAME {
            return Ok(sequence_mode);
        }
        Ok(if self.bits.read_bit()? {
            1
        } else {
            2 * u8::from(self.bits.read_bit()?)
        })
    }

    /// Hole filling, the global weight and, for TIP frames, the motion
    /// offset and filter.
    fn tip_parameters(&mut self, header: &mut FrameHeader) -> Result<()> {
        let inter = self.sequence().inter;
        if inter.tip_hole_fill {
            header.tip.hole_fill = self.bits.read_bit()?;
        }
        let both_sides = header.has_future_references && header.has_past_references;
        let refined = inter.tip_refine_mv && (inter.optical_flow != 0 || inter.refine_mv);
        if !both_sides || !refined {
            header.tip.global_weight = self.bits.read_bits(TIP_WEIGHT_BITS)? as u8;
        }
        if header.tip.mode == 2 {
            if !self.bits.read_bit()? {
                let mut row = self.bits.read_bits(TIP_MV_BITS)? as i8;
                let mut column = self.bits.read_bits(TIP_MV_BITS)? as i8;
                if row != 0 && self.bits.read_bit()? {
                    row = -row;
                }
                if column != 0 && self.bits.read_bit()? {
                    column = -column;
                }
                header.tip.motion = (row, column);
            }
            // Sharp, else regular or smooth.
            let sharp = self.bits.read_bit()?;
            header.tip.filter = match (sharp, sharp || self.bits.read_bit()?) {
                (true, _) => filter_mode::SHARP,
                (false, true) => filter_mode::REGULAR,
                (false, false) => filter_mode::SMOOTH,
            };
        }
        Ok(())
    }

    /// The two references TIP interpolates between: the nearest on each
    /// side, or the two nearest on the only side (dav2d
    /// `find_tip_ref_frames`).
    fn tip_references(&self, header: &FrameHeader) -> Result<[u8; 2]> {
        let count = usize::from(header.reference_count);
        if count == 1 {
            return Ok([0, 0]);
        }
        let mut distances = [0_i32; MAX_REFERENCES];
        for (rank, distance) in distances.iter_mut().enumerate().take(count) {
            let order = self.reference(header, rank as u8)?.order;
            *distance = self.order_difference(order, header.order);
        }
        // Ranks by distance (stable: equal distances keep rank order).
        let mut order: Vec<u8> = (0..count as u8).collect();
        order.sort_by_key(|&rank| distances[usize::from(rank)]);
        let past = distances[..count].iter().filter(|&&d| d < 0).count();
        Ok(if past == count {
            [order[count - 1], order[count - 2]]
        } else if past == 0 {
            [order[0], order[1]]
        } else {
            [order[past - 1], order[past]]
        })
    }

    /// The rest of a TIP frame: sub-PU deblocking, its quantizer, tiles
    /// when its prediction is filtered, the context references.
    fn tip_frame(&mut self, header: &mut FrameHeader) -> Result<()> {
        let sequence = self.sequence();
        if sequence.inter.deblock_sub_pu {
            header.deblock.sub_pu = self.bits.read_bit()?;
            if header.deblock.sub_pu {
                header.tip.apply_filter = self.bits.read_bit()?;
                if header.tip.apply_filter {
                    header.deblock.level_y = [true; 2];
                    header.deblock.level_u = true;
                    header.deblock.level_v = true;
                }
            }
        }
        self.checkpoint("post-tip_deblock");
        if sequence.inter.tip_explicit_qp {
            // dav2d 237229e leaves this syntax unimplemented.
            return Err(DecodeError::Unsupported(
                "AV2 TIP frames with an explicit quantizer",
            ));
        }
        let [first, second] = header.tip.references;
        header.quantizer.base = (self.reference(header, first)?.quantizer.base
            + self.reference(header, second)?.quantizer.base
            + 1)
            >> 1;
        if header.tip.apply_filter {
            self.tile_layout(header)?;
            self.checkpoint("post-tiling");
        } else {
            header.superblock_log2 = sequence.partition.superblock_log2;
            let superblocks = |size: u32| {
                let log2 = u32::from(header.superblock_log2);
                ((size + (1 << log2) - 1) >> log2) as u16
            };
            header.tiles = TileLayout {
                uniform: true,
                column_starts: vec![0, superblocks(header.width)],
                row_starts: vec![0, superblocks(header.height)],
            };
        }
        header.disable_cdf_update = true;
        let (primary, secondary) = self.context_sources(header)?;
        header.primary_reference = primary;
        header.secondary_reference = secondary;
        Ok(())
    }

    /// The inter frames nearest in quantizer, then in order, among the
    /// references (dav2d `derive_pri_sec_ref`).
    fn context_sources(&self, header: &FrameHeader) -> Result<(Option<u8>, Option<u8>)> {
        let mut infos = [None; MAX_REFERENCES];
        for rank in 0..header.reference_count {
            infos[usize::from(rank)] = Some(self.reference(header, rank)?.slot_info(0));
        }
        let quantizer = header.quantizer.base.min(MAX_QINDEX as u16) as u8;
        Ok(references::context_sources(
            &infos[..usize::from(header.reference_count)],
            quantizer,
            header.order,
        ))
    }

    /// Screen content, intra block copy and, for inter frames, the DRL
    /// size, MV precision, filter and motion modes; then
    /// `disable_cdf_update`.
    fn coding_tools(&mut self, header: &mut FrameHeader) -> Result<()> {
        let sequence = self.sequence();
        header.screen_content = self.tri(sequence.screen_content_tools)?;
        if header.screen_content {
            header.force_integer_mv = self.tri(sequence.force_integer_mv)?;
        }
        self.checkpoint("post-screencontent");
        if self.bits.read_bit()? {
            let global = header.is_intra() && self.bits.read_bit()?;
            let local = !global || self.bits.read_bit()?;
            let references = sequence.references;
            let max_drl_bits = if references.frame_max_bvp_drl_bits {
                self.default_or_uniform(BVP_DRL_BITS_CHOICES, references.max_bvp_drl_bits)?
            } else {
                references.max_bvp_drl_bits
            };
            header.intra_block_copy = Some(IntraBlockCopy {
                global,
                local,
                max_drl_bits,
            });
        }
        self.checkpoint("post-ibc");
        if !header.is_intra() {
            let references = sequence.references;
            header.max_drl_bits = if references.frame_max_drl_bits {
                self.default_or_uniform(DRL_BITS_CHOICES, references.max_drl_bits)?
            } else {
                references.max_drl_bits
            };
            if !header.force_integer_mv {
                header.mv_precision = if self.bits.read_bit()? {
                    2
                } else {
                    1 + 2 * u8::from(self.bits.read_bit()?)
                };
            }
            header.filter = if self.bits.read_bit()? {
                filter_mode::SWITCHABLE
            } else {
                self.bits.read_bits(FILTER_BITS)? as u8
            };
            let allowed = MotionModes::of_sequence(sequence);
            header.motion_modes = if sequence.inter.motion_modes.frame_signalled {
                let mut mask = MotionModes::TRANSLATION.mask;
                for bit in MotionModes::OPTIONAL {
                    if allowed.mask & bit != 0 && self.bits.read_bit()? {
                        mask |= bit;
                    }
                }
                MotionModes { mask }
            } else {
                allowed
            };
            self.checkpoint("post-frametype-specific-bits");
        }
        header.disable_cdf_update = self.bits.read_bit()?;
        self.checkpoint("post-disable_cdf_update");
        Ok(())
    }

    /// A value `default` (1-based) unless a flag says otherwise, then one
    /// of the other `count - 1` (dav2d `dav2d_get_ref_uniform`).
    fn default_or_uniform(&mut self, count: u32, default: u8) -> Result<u8> {
        let default = u32::from(default) - 1;
        if !self.bits.read_bit()? {
            return Ok(default as u8 + 1);
        }
        let value = uniform(self.bits, count - 1)?;
        Ok((value + u32::from(value >= default)) as u8 + 1)
    }

    /// A sequence three-way choice, read per frame when adaptive.
    fn tri(&mut self, tri: Tri) -> Result<bool> {
        Ok(match tri {
            Tri::Off => false,
            Tri::On => true,
            Tri::Adaptive => self.bits.read_bit()?,
        })
    }

    /// The frame's tiles, then the context-update tile and the tile-size
    /// width.
    fn tiles(&mut self, header: &mut FrameHeader) -> Result<()> {
        self.tile_layout(header)?;
        let sequence = self.sequence();
        let log2_tiles = tiles::log2(header.tiles.columns()) + tiles::log2(header.tiles.rows());
        if log2_tiles != 0 {
            if !sequence.transform.average_cdf_type {
                header.context_update_tile = self.bits.read_bits(log2_tiles)? as u16;
            }
            if usize::from(header.context_update_tile)
                >= header.tiles.columns() * header.tiles.rows()
            {
                return Err(DecodeError::Malformed("AV2 context update tile"));
            }
            header.tile_size_bytes = self.bits.read_bits(TILE_SIZE_BYTES_BITS)? as u8 + 1;
        }
        self.checkpoint("post-tiling");
        Ok(())
    }

    /// The frame's superblock size and tile layout: the sequence's reused,
    /// or coded (dav2d `parse_tile_info_frmhdr`).
    fn tile_layout(&mut self, header: &mut FrameHeader) -> Result<()> {
        let sequence = self.sequence();
        let sequence_log2 = sequence.partition.superblock_log2;
        // Intra frames of 256x256 sequences code 128x128 superblocks.
        header.superblock_log2 = if header.is_intra() {
            sequence_log2.min(LARGE_SUPERBLOCK_LOG2)
        } else {
            sequence_log2
        };
        let geometry = TileGeometry {
            size: (header.width, header.height),
            superblock_log2: header.superblock_log2,
            sequence_superblock_log2: sequence_log2,
            level: sequence.level,
            high_tier: sequence.high_tier,
        };
        let reusable = sequence
            .tile_layout
            .as_ref()
            .filter(|layout| tiles::reusable(layout, sequence, geometry));
        header.tiles = match reusable {
            Some(layout) if sequence.tile_layout_fixed || self.bits.read_bit()? => {
                tiles::reuse(layout, geometry)
            }
            _ => tiles::parse(self.bits, geometry)?,
        };
        Ok(())
    }

    /// `base_q_idx` and the deltas the sequence lets frames code; then the
    /// context references of inter frames, which depend on it.
    fn quantizer(&mut self, header: &mut FrameHeader) -> Result<()> {
        let quant = self.sequence().quant;
        let delta = |bits: &mut BitReader<'_>, enabled: bool| -> Result<i8> {
            Ok(if enabled && bits.read_bit()? {
                signed(bits, QUANT_DELTA_BITS)? as i8
            } else {
                0
            })
        };
        let q = &mut header.quantizer;
        q.base = self.bits.read_bits(8)? as u16;
        q.y_dc = delta(self.bits, quant.y_dc_delta_enabled)?;
        if quant.uv_dc_delta_enabled || quant.uv_ac_delta_enabled {
            let separate = quant.separate_uv_delta_q && self.bits.read_bit()?;
            q.u_dc = delta(self.bits, quant.uv_dc_delta_enabled)?;
            q.u_ac = delta(self.bits, quant.uv_ac_delta_enabled)?;
            if separate {
                q.v_dc = delta(self.bits, quant.uv_dc_delta_enabled)?;
                q.v_ac = delta(self.bits, quant.uv_ac_delta_enabled)?;
            } else {
                q.v_dc = q.u_dc;
                q.v_ac = q.u_ac;
            }
            if quant.equal_ac_dc_q {
                q.u_dc = q.u_ac;
                q.v_dc = q.v_ac;
            }
        }
        if header.frame_type == FrameType::Inter {
            let (primary, secondary) = self.context_sources(header)?;
            if !header.primary_signalled || primary.is_none() {
                header.primary_reference = primary;
            } else if header
                .primary_reference
                .is_some_and(|rank| rank >= header.reference_count)
            {
                return Err(DecodeError::Malformed("AV2 primary reference"));
            }
            header.secondary_reference = header.primary_reference.and_then(|chosen| {
                if secondary == Some(chosen) {
                    primary
                } else {
                    secondary
                }
            });
        }
        self.checkpoint("post-quant");
        Ok(())
    }

    /// `segmentation_enabled` and the features in force (the sequence's or
    /// coded), the map update flags.
    fn segmentation(&mut self, header: &mut FrameHeader) -> Result<()> {
        if self.bits.read_bit()? {
            let sequence = self.sequence().segmentation;
            let features = match sequence.features {
                Some(features) if !sequence.adaptive || self.bits.read_bit()? => features,
                _ => SegmentFeatures::parse(self.bits, sequence.count())?,
            };
            let (update_map, temporal_update) = if header.primary_reference.is_none() {
                (true, false)
            } else {
                let update = self.bits.read_bit()?;
                let temporal =
                    update && header.frame_type != FrameType::Switch && self.bits.read_bit()?;
                (update, temporal)
            };
            let preskip_mask = features.skip_mask | features.global_mv_mask;
            let active = preskip_mask | features.delta_q_mask;
            header.segmentation = Some(Segmentation {
                features,
                update_map,
                temporal_update,
                preskip: preskip_mask != 0,
                last_active: (active != 0).then(|| active.ilog2() as u8),
            });
        }
        self.checkpoint("post-segmentation");
        Ok(())
    }

    /// Quantizer-matrix levels.
    fn quant_matrices(&mut self, header: &mut FrameHeader) -> Result<()> {
        if self.bits.read_bit()? {
            let count = if header.segmentation.is_some() {
                self.bits.read_bits(QM_COUNT_BITS)? as u8 + 1
            } else {
                1
            };
            let separate = self.sequence().quant.separate_uv_delta_q;
            let mut matrices = QuantMatrices {
                count,
                ..QuantMatrices::default()
            };
            for set in 0..usize::from(count) {
                let y = self.bits.read_bits(QM_LEVEL_BITS)? as u8;
                matrices.y[set] = y;
                if self.bits.read_bit()? {
                    matrices.u[set] = y;
                    matrices.v[set] = y;
                } else {
                    let u = self.bits.read_bits(QM_LEVEL_BITS)? as u8;
                    matrices.u[set] = u;
                    matrices.v[set] = if separate {
                        self.bits.read_bits(QM_LEVEL_BITS)? as u8
                    } else {
                        u
                    };
                }
            }
            header.quantizer.matrices = Some(matrices);
        }
        self.checkpoint("post-qm");
        Ok(())
    }

    /// `delta_q_present`, the lossless segments, then the level coding
    /// (TCQ, parity hiding) of lossy frames.
    fn delta_q_and_lossless(&mut self, header: &mut FrameHeader) -> Result<()> {
        if header.quantizer.base != 0 && self.bits.read_bit()? {
            header.delta_q = Some(self.bits.read_bits(DELTA_Q_RES_BITS)? as u8);
        }
        self.checkpoint("post-delta_q");
        let quant = self.sequence().quant;
        let q = header.quantizer;
        let deltas_lossless = header.delta_q.is_none()
            && i32::from(q.y_dc) + i32::from(quant.base_y_dc_delta) <= 0
            && i32::from(q.u_dc) + i32::from(quant.base_uv_dc_delta) <= 0
            && i32::from(q.u_ac) + i32::from(quant.base_uv_ac_delta) <= 0
            && i32::from(q.v_dc) + i32::from(quant.base_uv_dc_delta) <= 0
            && i32::from(q.v_ac) + i32::from(quant.base_uv_ac_delta) <= 0;
        let mut lossless = Lossless {
            all: true,
            ..Lossless::default()
        };
        for segment in 0..super::sequence::MAX_SEGMENTS {
            let qindex = match &header.segmentation {
                Some(segmentation) => (i32::from(q.base)
                    + i32::from(segmentation.features.delta_q[segment]))
                .clamp(0, MAX_QINDEX) as u16,
                None => q.base,
            };
            lossless.qindex[segment] = qindex;
            if qindex == 0 && deltas_lossless {
                lossless.segments |= 1 << segment;
            } else {
                lossless.all = false;
            }
        }
        header.lossless = lossless;
        if !lossless.all {
            header.tcq = match self.sequence().transform.tcq {
                0 => false,
                TCQ_PER_FRAME => self.bits.read_bit()?,
                _ => true,
            };
            if !header.tcq && self.sequence().transform.parity_hiding {
                header.parity_hiding = self.bits.read_bit()?;
            }
        }
        self.checkpoint("post-tcq_parity");
        Ok(())
    }

    /// The transform mode, the inter frame-level modes and the reduced
    /// transform set.
    fn frame_modes(&mut self, header: &mut FrameHeader) -> Result<()> {
        if !header.lossless.all {
            header.tx_mode_select = self.bits.read_bit()?;
        }
        if !header.is_intra() {
            header.reference_select = self.bits.read_bit()?;
            if header.frame_type == FrameType::Inter {
                header.skip_mode = self.bits.read_bit()?;
            }
            // Coded but not allowed in switch frames.
            if self.sequence().inter.bawp {
                header.bawp = self.bits.read_bit()? && header.frame_type == FrameType::Inter;
            }
            if header.motion_modes.mask & MotionModes::WARP_DELTA != 0 {
                header.warp_delta = self.bits.read_bit()?;
            }
        }
        header.reduced_tx_set = self.bits.read_bits(REDUCED_TX_SET_BITS)? as u8;
        self.checkpoint("post-modebits");
        Ok(())
    }

    /// Film grain of a shown frame.
    fn film_grain(&mut self, header: &mut FrameHeader) -> Result<()> {
        if self.sequence().film_grain && header.is_shown() {
            header.film_grain = self.film_grain_fields()?;
            self.checkpoint("post-filmgrain");
        }
        Ok(())
    }

    /// `apply_grain` (implied in single pictures), the model id and seed.
    fn film_grain_fields(&mut self) -> Result<Option<FilmGrain>> {
        if !self.sequence().single_picture && !self.bits.read_bit()? {
            return Ok(None);
        }
        Ok(Some(FilmGrain {
            id: self.bits.read_bits(SMALL_INDEX_BITS)? as u8,
            seed: self.bits.read_bits(GRAIN_SEED_BITS)? as u16,
        }))
    }
}

/// `primary_ref_frame` as a rank: 7 is none.
fn primary_rank(coded: u8) -> Option<u8> {
    (coded != PRIMARY_REF_NONE).then_some(coded)
}

/// `primary_ref_frame` value meaning none (AVM `PRIMARY_REF_NONE`).
const PRIMARY_REF_NONE: u8 = 7;
/// log2 of 64x64 and 128x128 superblocks.
const SMALLEST_SUPERBLOCK_LOG2: u8 = 6;
const LARGE_SUPERBLOCK_LOG2: u8 = 7;
const LARGEST_SUPERBLOCK_LOG2: u8 = 8;
/// `enable_opfl_refine` value leaving the choice to frames.
const OPTICAL_FLOW_PER_FRAME: u8 = 3;
/// `enable_tcq` value leaving the choice to frames.
const TCQ_PER_FRAME: u8 = 2;
