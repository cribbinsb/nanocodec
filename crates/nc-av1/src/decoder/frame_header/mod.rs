//! `uncompressed_header()` (AV1 §5.9): every field of a frame header, given
//! the sequence header and what the reference slots keep (§7.20). The
//! quantiser/segmentation, loop-filter and motion parts live in submodules.
use super::sequence::{SELECT_PER_FRAME, Sequence};
use nc_common::decode::{BitReader, DecodeError};

mod filters;
mod motion;
mod quant;
mod tiles;

pub(crate) use filters::{CdefParams, LoopFilter, LrType, Restoration};
pub(crate) use motion::{GlobalMotion, WARPEDMODEL_PREC_BITS, WarpType, relative_distance};
pub(crate) use quant::{DeltaParams, Quantization, Segmentation};
pub(crate) use tiles::TileInfo;

/// `NUM_REF_FRAMES` (AV1 §3): reference slots.
pub(crate) const NUM_REF_FRAMES: usize = 8;
/// `REFS_PER_FRAME`: references an inter frame names (LAST..=ALTREF).
pub(crate) const REFS_PER_FRAME: usize = 7;
/// `PRIMARY_REF_NONE`: no frame to inherit CDFs and parameters from.
pub(crate) const PRIMARY_REF_NONE: u8 = 7;
/// Reference frame names (AV1 §6.10.24): index into per-reference arrays.
pub(crate) const INTRA_FRAME: usize = 0;
pub(crate) const LAST_FRAME: usize = 1;
pub(crate) const GOLDEN_FRAME: usize = 4;
pub(crate) const ALTREF_FRAME: usize = 7;
/// `TOTAL_REFS_PER_FRAME`: `INTRA_FRAME` plus the seven references.
pub(crate) const TOTAL_REFS: usize = 8;
/// `SUPERRES_NUM`, `SUPERRES_DENOM_MIN`, `SUPERRES_DENOM_BITS` (AV1 §3).
pub(crate) const SUPERRES_NUM: usize = 8;
const SUPERRES_DENOM_MIN: usize = 9;
const SUPERRES_DENOM_BITS: u32 = 3;
/// `interpolation_filter` value meaning per-block choice (AV1 §6.8.9).
pub(crate) const INTERP_SWITCHABLE: u8 = 4;
/// Bits of `render_width_minus_1` / `render_height_minus_1`.
const RENDER_SIZE_BITS: u32 = 16;

/// `frame_type` (AV1 §6.8.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum FrameType {
    #[default]
    Key,
    Inter,
    IntraOnly,
    Switch,
}
impl FrameType {
    fn from_bits(bits: u32) -> Self {
        match bits {
            0 => Self::Key,
            1 => Self::Inter,
            2 => Self::IntraOnly,
            _ => Self::Switch,
        }
    }
    /// `FrameIsIntra`.
    pub(crate) fn is_intra(self) -> bool {
        matches!(self, Self::Key | Self::IntraOnly)
    }
}

/// `TxMode` (AV1 §6.8.21).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TxMode {
    Only4x4,
    Largest,
    Select,
}

/// Frame dimensions (AV1 §5.9.5–8, §7.21 names).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct FrameSize {
    /// `FrameWidth`: coded width (after superres downscaling).
    pub width: usize,
    pub height: usize,
    /// `UpscaledWidth`: output width.
    pub upscaled_width: usize,
    pub render_width: usize,
    pub render_height: usize,
    /// `SuperresDenom` ([`SUPERRES_NUM`] without superres).
    pub superres_denom: usize,
}
impl FrameSize {
    /// `MiCols` (§5.9.9): 4x4 columns, a whole number of 8x8 blocks.
    pub(crate) fn mi_cols(&self) -> usize {
        2 * self.width.div_ceil(8)
    }
    pub(crate) fn mi_rows(&self) -> usize {
        2 * self.height.div_ceil(8)
    }
}

/// What a reference slot keeps of its frame's header for later headers
/// (AV1 §7.20 reference frame update process).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SlotHeader {
    pub frame_id: u32,
    pub frame_type: FrameType,
    pub size: FrameSize,
    pub order_hint: u32,
    /// `SavedOrderHints`: the order hints of the frame's own references.
    pub ref_order_hints: [u32; TOTAL_REFS],
    pub showable: bool,
    pub global_motion: [GlobalMotion; TOTAL_REFS],
    pub loop_filter_deltas: filters::LoopFilterDeltas,
    pub segmentation: quant::SegmentationFeatures,
    pub film_grain: FilmGrainParams,
}

/// The parsed frame header. Names follow the specification.
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::struct_excessive_bools)] // the specification's flags
pub(crate) struct FrameHeader {
    /// `show_existing_frame`: `Some(slot)` shows that slot and codes nothing.
    pub show_existing: Option<usize>,
    pub frame_type: FrameType,
    pub show_frame: bool,
    pub showable_frame: bool,
    pub error_resilient_mode: bool,
    pub disable_cdf_update: bool,
    pub allow_screen_content_tools: bool,
    pub force_integer_mv: bool,
    pub current_frame_id: u32,
    pub order_hint: u32,
    pub primary_ref_frame: u8,
    pub refresh_frame_flags: u8,
    /// `ref_frame_idx[]`: the slot of LAST..=ALTREF.
    pub ref_frame_idx: [u8; REFS_PER_FRAME],
    /// `OrderHints[]` per reference name (index 0 unused).
    pub order_hints: [u32; TOTAL_REFS],
    /// `RefFrameSignBias[]` per reference name.
    pub sign_bias: [bool; TOTAL_REFS],
    pub size: FrameSize,
    pub allow_intrabc: bool,
    pub allow_high_precision_mv: bool,
    /// `interpolation_filter`; [`INTERP_SWITCHABLE`] for per-block filters.
    pub interpolation_filter: u8,
    pub is_motion_mode_switchable: bool,
    pub use_ref_frame_mvs: bool,
    pub disable_frame_end_update_cdf: bool,
    pub tiles: TileInfo,
    pub quant: Quantization,
    pub segmentation: Segmentation,
    pub delta: DeltaParams,
    /// `CodedLossless`, `AllLossless`.
    pub coded_lossless: bool,
    pub all_lossless: bool,
    pub loop_filter: LoopFilter,
    pub cdef: CdefParams,
    pub restoration: Restoration,
    pub tx_mode: TxMode,
    pub reference_select: bool,
    /// `SkipModeFrame[]` when `skip_mode_present`.
    pub skip_mode_frames: Option<[u8; 2]>,
    pub allow_warped_motion: bool,
    pub reduced_tx_set: bool,
    /// `gm_params` per reference name (index 0 unused).
    pub global_motion: [GlobalMotion; TOTAL_REFS],
    pub film_grain: FilmGrainParams,
}

/// `film_grain_params()` (§5.9.30), kept raw for synthesis (§7.18.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(clippy::struct_excessive_bools)] // the specification's flags
pub(crate) struct FilmGrainParams {
    pub apply_grain: bool,
    pub grain_seed: u16,
    pub num_y_points: usize,
    pub point_y_value: [u8; 14],
    pub point_y_scaling: [u8; 14],
    pub chroma_scaling_from_luma: bool,
    pub num_cb_points: usize,
    pub point_cb_value: [u8; 10],
    pub point_cb_scaling: [u8; 10],
    pub num_cr_points: usize,
    pub point_cr_value: [u8; 10],
    pub point_cr_scaling: [u8; 10],
    pub grain_scaling_minus_8: u8,
    pub ar_coeff_lag: u8,
    pub ar_coeffs_y_plus_128: [u8; 24],
    pub ar_coeffs_cb_plus_128: [u8; 25],
    pub ar_coeffs_cr_plus_128: [u8; 25],
    pub ar_coeff_shift_minus_6: u8,
    pub grain_scale_shift: u8,
    pub cb_mult: u8,
    pub cb_luma_mult: u8,
    pub cb_offset: u16,
    pub cr_mult: u8,
    pub cr_luma_mult: u8,
    pub cr_offset: u16,
    pub overlap_flag: bool,
    pub clip_to_restricted_range: bool,
}

/// Where the frame header sits: the OBU's layer ids (buffer removal times
/// are coded per operating point containing them).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct LayerIds {
    pub temporal: u8,
    pub spatial: u8,
}

/// Parse `uncompressed_header()`; `slots` are the reference slots' saved
/// headers (`None`: empty). Leaves `r` after the header (before
/// `byte_alignment()`).
///
/// # Errors
/// Truncated or malformed syntax, references to empty slots, or tools
/// outside the 8-bit 4:2:0 Main subset.
pub(crate) fn parse(
    r: &mut BitReader<'_>,
    seq: &Sequence,
    slots: &[Option<&SlotHeader>; NUM_REF_FRAMES],
    layer: LayerIds,
) -> Result<FrameHeader, DecodeError> {
    let mut h = Parser { r, seq, slots };
    h.header(layer)
}

struct Parser<'r, 'a> {
    r: &'r mut BitReader<'a>,
    seq: &'r Sequence,
    slots: &'r [Option<&'r SlotHeader>; NUM_REF_FRAMES],
}

impl<'r> Parser<'r, '_> {
    fn slot(&self, index: usize) -> Result<&'r SlotHeader, DecodeError> {
        // An empty (or invalid, `mark_ref_frames`) slot: the frame that
        // filled it never arrived.
        self.slots[index].ok_or(DecodeError::MissingReference)
    }
    fn frame_id_bits(&self) -> u32 {
        self.seq.frame_ids.map_or(0, |ids| ids.id)
    }
    /// `temporal_point_info()` (§5.9.31): informative.
    fn temporal_point_info(&mut self) -> Result<(), DecodeError> {
        if let Some(model) = self.seq.decoder_model
            && !self.seq.equal_picture_interval
        {
            self.r
                .skip_bits(model.frame_presentation_time_bits as usize)?;
        }
        Ok(())
    }

    fn header(&mut self, layer: LayerIds) -> Result<FrameHeader, DecodeError> {
        let seq = self.seq;
        let mut h = empty_header();
        if seq.reduced_still_picture_header {
            h.show_frame = true;
        } else {
            if self.r.read_bit()? {
                return self.show_existing();
            }
            h.frame_type = FrameType::from_bits(self.r.read_bits(2)?);
            h.show_frame = self.r.read_bit()?;
            if h.show_frame {
                self.temporal_point_info()?;
            }
            h.showable_frame = if h.show_frame {
                h.frame_type != FrameType::Key
            } else {
                self.r.read_bit()?
            };
            h.error_resilient_mode = h.frame_type == FrameType::Switch
                || (h.frame_type == FrameType::Key && h.show_frame)
                || self.r.read_bit()?;
        }
        // A shown key frame resets every slot's order hint (§5.9.2).
        let reset_slots = h.frame_type == FrameType::Key && h.show_frame;
        let mut slot_hints: [u32; NUM_REF_FRAMES] = std::array::from_fn(|i| {
            self.slots[i]
                .filter(|_| !reset_slots)
                .map_or(0, |s| s.order_hint)
        });
        h.disable_cdf_update = self.r.read_bit()?;
        h.allow_screen_content_tools = if seq.force_screen_content_tools == SELECT_PER_FRAME {
            self.r.read_bit()?
        } else {
            seq.force_screen_content_tools != 0
        };
        h.force_integer_mv = if h.allow_screen_content_tools {
            if seq.force_integer_mv == SELECT_PER_FRAME {
                self.r.read_bit()?
            } else {
                seq.force_integer_mv != 0
            }
        } else {
            false
        } || h.frame_type.is_intra();
        if seq.frame_ids.is_some() {
            h.current_frame_id = self.r.read_bits(self.frame_id_bits())?;
        }
        let frame_size_override = h.frame_type == FrameType::Switch
            || (!seq.reduced_still_picture_header && self.r.read_bit()?);
        h.order_hint = self.r.read_bits(seq.order_hint_bits)?;
        h.primary_ref_frame = if h.frame_type.is_intra() || h.error_resilient_mode {
            PRIMARY_REF_NONE
        } else {
            self.r.read_bits(3)? as u8
        };
        self.buffer_removal_times(layer)?;
        let all_frames = u8::MAX;
        h.refresh_frame_flags = if h.frame_type == FrameType::Switch || reset_slots {
            all_frames
        } else {
            self.r.read_bits(8)? as u8
        };
        if (!h.frame_type.is_intra() || h.refresh_frame_flags != all_frames)
            && h.error_resilient_mode
            && seq.enable_order_hint
        {
            // ref_order_hint[]: a slot whose hint disagrees is invalid; its
            // hint is still what later order-hint arithmetic uses.
            for hint in &mut slot_hints {
                *hint = self.r.read_bits(seq.order_hint_bits)?;
            }
        }
        if h.frame_type.is_intra() {
            h.size = self.frame_size(frame_size_override)?;
            h.allow_intrabc = h.allow_screen_content_tools
                && h.size.upscaled_width == h.size.width
                && self.r.read_bit()?;
        } else {
            self.inter_references(&mut h, &slot_hints, frame_size_override)?;
        }
        h.disable_frame_end_update_cdf =
            seq.reduced_still_picture_header || h.disable_cdf_update || self.r.read_bit()?;
        self.coding_parameters(&mut h, &slot_hints)?;
        Ok(h)
    }

    /// The frame header from `tile_info()` on (§5.9.2), with what
    /// `load_previous()` (§7.20) inherits from the primary reference.
    fn coding_parameters(
        &mut self,
        h: &mut FrameHeader,
        slot_hints: &[u32; NUM_REF_FRAMES],
    ) -> Result<(), DecodeError> {
        let seq = self.seq;
        // load_previous() (§7.20): parameters inherited from the primary
        // reference, or the setup_past_independence() defaults.
        let previous = if h.primary_ref_frame == PRIMARY_REF_NONE {
            None
        } else {
            let slot = h.ref_frame_idx[usize::from(h.primary_ref_frame)];
            Some(self.slot(usize::from(slot))?)
        };
        h.tiles = tiles::parse(self.r, seq, &h.size)?;
        h.quant = quant::quantization(self.r, seq)?;
        h.segmentation = quant::segmentation(
            self.r,
            h.primary_ref_frame,
            previous.as_ref().map(|p| &p.segmentation),
        )?;
        h.delta = quant::delta_params(self.r, h.quant.base_q_idx, h.allow_intrabc)?;
        h.coded_lossless = (0..quant::MAX_SEGMENTS).all(|segment| {
            h.segmentation.qindex(h.quant.base_q_idx, segment) == 0 && h.quant.deltas_zero()
        });
        h.all_lossless = h.coded_lossless && h.size.width == h.size.upscaled_width;
        let no_filters = h.coded_lossless || h.allow_intrabc;
        h.loop_filter = filters::loop_filter(
            self.r,
            seq,
            no_filters,
            previous.as_ref().map(|p| p.loop_filter_deltas),
        )?;
        h.cdef = filters::cdef(self.r, seq, no_filters || !seq.enable_cdef)?;
        h.restoration = filters::restoration(
            self.r,
            seq,
            h.all_lossless || h.allow_intrabc || !seq.enable_restoration,
        )?;
        h.tx_mode = if h.coded_lossless {
            TxMode::Only4x4
        } else if self.r.read_bit()? {
            TxMode::Select
        } else {
            TxMode::Largest
        };
        h.reference_select = !h.frame_type.is_intra() && self.r.read_bit()?;
        // skip_mode_present is coded only where skip mode is allowed.
        h.skip_mode_frames = match motion::skip_mode_frames(seq, h, slot_hints) {
            Some(frames) if self.r.read_bit()? => Some(frames),
            _ => None,
        };
        h.allow_warped_motion = !h.frame_type.is_intra()
            && !h.error_resilient_mode
            && seq.enable_warped_motion
            && self.r.read_bit()?;
        h.reduced_tx_set = self.r.read_bit()?;
        if !h.frame_type.is_intra() {
            let previous_motion = previous
                .as_ref()
                .map_or([GlobalMotion::IDENTITY; TOTAL_REFS], |p| p.global_motion);
            h.global_motion =
                motion::global_motion(self.r, &previous_motion, h.allow_high_precision_mv)?;
        }
        h.film_grain = self.film_grain(h)?;
        Ok(())
    }

    /// `show_existing_frame = 1` (§5.9.2).
    fn show_existing(&mut self) -> Result<FrameHeader, DecodeError> {
        let index = self.r.read_bits(3)? as usize;
        self.temporal_point_info()?;
        let slot = self.slot(index)?;
        // display_frame_id must be the slot's RefFrameId (§6.8.2): another
        // id means the frame meant for the slot was lost.
        if self.seq.frame_ids.is_some() && self.r.read_bits(self.frame_id_bits())? != slot.frame_id
        {
            return Err(DecodeError::MissingReference);
        }
        let mut h = empty_header();
        h.show_existing = Some(index);
        h.frame_type = slot.frame_type;
        h.show_frame = true;
        h.size = slot.size;
        h.order_hint = slot.order_hint;
        h.film_grain = slot.film_grain;
        if slot.frame_type == FrameType::Key {
            h.refresh_frame_flags = u8::MAX;
        }
        Ok(h)
    }

    /// `buffer_removal_time[]` (§5.9.2): informative, skipped.
    fn buffer_removal_times(&mut self, layer: LayerIds) -> Result<(), DecodeError> {
        let Some(model) = self.seq.decoder_model else {
            return Ok(());
        };
        if !self.r.read_bit()? {
            return Ok(());
        }
        for op in &self.seq.operating_points[..self.seq.operating_point_count] {
            let idc = op.idc;
            let in_layer = (idc >> layer.temporal) & 1 != 0
                && (idc >> (8 + u32::from(layer.spatial))) & 1 != 0;
            if op.decoder_model_present && (idc == 0 || in_layer) {
                self.r.skip_bits(model.buffer_removal_time_bits as usize)?;
            }
        }
        Ok(())
    }

    /// The reference part of an inter frame's header (§5.9.2, from
    /// `frame_refs_short_signaling` to `RefFrameSignBias`).
    fn inter_references(
        &mut self,
        h: &mut FrameHeader,
        slot_hints: &[u32; NUM_REF_FRAMES],
        frame_size_override: bool,
    ) -> Result<(), DecodeError> {
        let seq = self.seq;
        let short_signaling = seq.enable_order_hint && self.r.read_bit()?;
        if short_signaling {
            let last = self.r.read_bits(3)? as u8;
            let gold = self.r.read_bits(3)? as u8;
            h.ref_frame_idx = motion::set_frame_refs(seq, h.order_hint, slot_hints, last, gold);
        }
        for i in 0..REFS_PER_FRAME {
            if !short_signaling {
                h.ref_frame_idx[i] = self.r.read_bits(3)? as u8;
            }
            if let Some(ids) = seq.frame_ids {
                let delta = self.r.read_bits(ids.delta)? + 1; // delta_frame_id_minus_1
                // expectedFrameId (§5.9.2) must be the slot's RefFrameId
                // (§6.8.2); another id means the slot holds an older frame
                // because the one it should hold was lost.
                let range = 1 << ids.id;
                let expected = (h.current_frame_id + range - delta) % range;
                if self.slot(usize::from(h.ref_frame_idx[i]))?.frame_id != expected {
                    return Err(DecodeError::MissingReference);
                }
            }
        }
        for &slot in &h.ref_frame_idx {
            self.slot(usize::from(slot))?;
        }
        h.size = if frame_size_override && !h.error_resilient_mode {
            self.frame_size_with_refs(h.ref_frame_idx)?
        } else {
            self.frame_size(frame_size_override)?
        };
        h.allow_high_precision_mv = !h.force_integer_mv && self.r.read_bit()?;
        h.interpolation_filter = if self.r.read_bit()? {
            INTERP_SWITCHABLE
        } else {
            self.r.read_bits(2)? as u8
        };
        h.is_motion_mode_switchable = self.r.read_bit()?;
        h.use_ref_frame_mvs =
            !h.error_resilient_mode && seq.enable_ref_frame_mvs && self.r.read_bit()?;
        for i in 0..REFS_PER_FRAME {
            let name = LAST_FRAME + i;
            let hint = slot_hints[usize::from(h.ref_frame_idx[i])];
            h.order_hints[name] = hint;
            h.sign_bias[name] =
                seq.enable_order_hint && motion::relative_distance(seq, hint, h.order_hint) > 0;
        }
        Ok(())
    }

    /// `frame_size()` and `render_size()` (§5.9.5, §5.9.6).
    fn frame_size(&mut self, override_flag: bool) -> Result<FrameSize, DecodeError> {
        let seq = self.seq;
        let (width, height) = if override_flag {
            (
                self.r.read_bits(seq.frame_width_bits)? as usize + 1,
                self.r.read_bits(seq.frame_height_bits)? as usize + 1,
            )
        } else {
            (seq.max_frame_width, seq.max_frame_height)
        };
        if width > seq.max_frame_width || height > seq.max_frame_height {
            return Err(DecodeError::Malformed("AV1 frame larger than the sequence"));
        }
        let mut size = self.superres(width, height)?;
        if self.r.read_bit()? {
            size.render_width = self.r.read_bits(RENDER_SIZE_BITS)? as usize + 1;
            size.render_height = self.r.read_bits(RENDER_SIZE_BITS)? as usize + 1;
        }
        Ok(size)
    }

    /// `frame_size_with_refs()` (§5.9.7).
    fn frame_size_with_refs(
        &mut self,
        ref_frame_idx: [u8; REFS_PER_FRAME],
    ) -> Result<FrameSize, DecodeError> {
        for slot in ref_frame_idx {
            if self.r.read_bit()? {
                let found = self.slot(usize::from(slot))?.size;
                let mut size = self.superres(found.upscaled_width, found.height)?;
                size.render_width = found.render_width;
                size.render_height = found.render_height;
                return Ok(size);
            }
        }
        self.frame_size(true)
    }

    /// `superres_params()` (§5.9.8): the coded width from the upscaled one.
    fn superres(&mut self, upscaled_width: usize, height: usize) -> Result<FrameSize, DecodeError> {
        let denom = if self.seq.enable_superres && self.r.read_bit()? {
            self.r.read_bits(SUPERRES_DENOM_BITS)? as usize + SUPERRES_DENOM_MIN
        } else {
            SUPERRES_NUM
        };
        let width = (upscaled_width * SUPERRES_NUM + denom / 2) / denom;
        Ok(FrameSize {
            width,
            height,
            upscaled_width,
            render_width: upscaled_width,
            render_height: height,
            superres_denom: denom,
        })
    }

    /// `film_grain_params()` (§5.9.30).
    fn film_grain(&mut self, h: &FrameHeader) -> Result<FilmGrainParams, DecodeError> {
        let seq = self.seq;
        if !seq.film_grain_params_present || (!h.show_frame && !h.showable_frame) {
            return Ok(FilmGrainParams::default());
        }
        if !self.r.read_bit()? {
            return Ok(FilmGrainParams::default());
        }
        let grain_seed = self.r.read_bits(16)? as u16;
        let update = h.frame_type != FrameType::Inter || self.r.read_bit()?;
        if !update {
            let slot = self.r.read_bits(3)? as u8;
            if !h.ref_frame_idx.contains(&slot) {
                return Err(DecodeError::Malformed(
                    "AV1 film grain from a non-reference",
                ));
            }
            let mut params = self.slot(usize::from(slot))?.film_grain;
            params.grain_seed = grain_seed;
            return Ok(params);
        }
        let mut g = FilmGrainParams {
            apply_grain: true,
            grain_seed,
            ..FilmGrainParams::default()
        };
        let r = &mut *self.r;
        g.num_y_points = read_points(r, 14, &mut g.point_y_value, &mut g.point_y_scaling)?;
        g.chroma_scaling_from_luma = !seq.mono_chrome && r.read_bit()?;
        if !seq.mono_chrome && !g.chroma_scaling_from_luma && g.num_y_points > 0 {
            g.num_cb_points = read_points(r, 10, &mut g.point_cb_value, &mut g.point_cb_scaling)?;
            g.num_cr_points = read_points(r, 10, &mut g.point_cr_value, &mut g.point_cr_scaling)?;
        } else if !seq.mono_chrome && !g.chroma_scaling_from_luma {
            // 4:2:0 without luma points codes no chroma points either.
        }
        g.grain_scaling_minus_8 = r.read_bits(2)? as u8;
        g.ar_coeff_lag = r.read_bits(2)? as u8;
        let luma_positions = 2 * usize::from(g.ar_coeff_lag) * (usize::from(g.ar_coeff_lag) + 1);
        let chroma_positions = luma_positions + usize::from(g.num_y_points > 0);
        if g.num_y_points > 0 {
            read_bytes(r, &mut g.ar_coeffs_y_plus_128[..luma_positions])?;
        }
        if g.chroma_scaling_from_luma || g.num_cb_points > 0 {
            read_bytes(r, &mut g.ar_coeffs_cb_plus_128[..chroma_positions])?;
        }
        if g.chroma_scaling_from_luma || g.num_cr_points > 0 {
            read_bytes(r, &mut g.ar_coeffs_cr_plus_128[..chroma_positions])?;
        }
        g.ar_coeff_shift_minus_6 = r.read_bits(2)? as u8;
        g.grain_scale_shift = r.read_bits(2)? as u8;
        if g.num_cb_points > 0 {
            g.cb_mult = r.read_bits(8)? as u8;
            g.cb_luma_mult = r.read_bits(8)? as u8;
            g.cb_offset = r.read_bits(9)? as u16;
        }
        if g.num_cr_points > 0 {
            g.cr_mult = r.read_bits(8)? as u8;
            g.cr_luma_mult = r.read_bits(8)? as u8;
            g.cr_offset = r.read_bits(9)? as u16;
        }
        g.overlap_flag = r.read_bit()?;
        g.clip_to_restricted_range = r.read_bit()?;
        Ok(g)
    }
}

/// A film grain scaling function: `num_*_points` (at most `max`) then
/// value/scaling pairs.
fn read_points(
    r: &mut BitReader<'_>,
    max: usize,
    values: &mut [u8],
    scaling: &mut [u8],
) -> Result<usize, DecodeError> {
    let count = r.read_bits(4)? as usize;
    if count > max {
        return Err(DecodeError::Malformed("AV1 film grain point count"));
    }
    for i in 0..count {
        values[i] = r.read_bits(8)? as u8;
        scaling[i] = r.read_bits(8)? as u8;
        // §6.8.20: each point's value is greater than the previous one;
        // the scaling lookup divides by their difference.
        if i > 0 && values[i] <= values[i - 1] {
            return Err(DecodeError::Malformed(
                "AV1 film grain points not increasing",
            ));
        }
    }
    Ok(count)
}

fn read_bytes(r: &mut BitReader<'_>, out: &mut [u8]) -> Result<(), DecodeError> {
    for value in out {
        *value = r.read_bits(8)? as u8;
    }
    Ok(())
}

/// A header with every field at its "not coded" value.
fn empty_header() -> FrameHeader {
    FrameHeader {
        show_existing: None,
        frame_type: FrameType::Key,
        show_frame: false,
        showable_frame: false,
        error_resilient_mode: false,
        disable_cdf_update: false,
        allow_screen_content_tools: false,
        force_integer_mv: false,
        current_frame_id: 0,
        order_hint: 0,
        primary_ref_frame: PRIMARY_REF_NONE,
        refresh_frame_flags: 0,
        ref_frame_idx: [0; REFS_PER_FRAME],
        order_hints: [0; TOTAL_REFS],
        sign_bias: [false; TOTAL_REFS],
        size: FrameSize::default(),
        allow_intrabc: false,
        allow_high_precision_mv: false,
        interpolation_filter: 0,
        is_motion_mode_switchable: false,
        use_ref_frame_mvs: false,
        disable_frame_end_update_cdf: true,
        tiles: TileInfo::default(),
        quant: Quantization::default(),
        segmentation: Segmentation::default(),
        delta: DeltaParams::default(),
        coded_lossless: false,
        all_lossless: false,
        loop_filter: LoopFilter::default(),
        cdef: CdefParams::default(),
        restoration: Restoration::default(),
        tx_mode: TxMode::Largest,
        reference_select: false,
        skip_mode_frames: None,
        allow_warped_motion: false,
        reduced_tx_set: false,
        global_motion: [GlobalMotion::IDENTITY; TOTAL_REFS],
        film_grain: FilmGrainParams::default(),
    }
}
