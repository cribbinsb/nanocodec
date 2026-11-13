//! AV1 decoder. Compressed syntax is parsed independently of the encoder;
//! only normative prediction, inverse transform and filtering kernels are
//! shared.
//!
//! Module map:
//! - `obu`: OBU framing and tile group headers (AV1 §5.3, §5.11.1);
//! - `sequence`, `frame_header`: complete sequence and frame header syntax
//!   (§5.5, §5.9) for 8-bit 4:2:0 Main streams;
//! - `subset`: the frame-level gate naming tools the block layer does not
//!   decode yet (the conformance gap list);
//! - `block_size`: the 22 block sizes and `Partition_Subsize`;
//! - `block_info`: the decoder's per-4x4 block state and its contexts;
//! - `motion`, `mv_stack`: the motion store and the reference MV stack
//!   (§7.10.2) of inter blocks;
//! - `availability`: which neighbours intra prediction may read;
//! - `cdf`: the decoder's CDF set;
//! - `transform`, `coefficients`: transform sizes, types and the
//!   coefficient syntax of every transform size;
//! - `tile`: partitions, block syntax and reconstruction;
//! - `deblock`, `cdef`: the loop filters from the decoder's block state;
//! - `restoration`: superres upscaling and loop restoration (§7.16–17);
//! - `film_grain`: film grain synthesis of shown frames (§7.18.3).
//!
//! Reference slots (§7.20) carry each frame's picture, CDFs and the header
//! state later headers inherit, so layered (hierarchical-P and spatial SVC)
//! streams and their lower operating points decode.
//!
//! Packet loss and reference drift: AV1 frame headers reference slot indices
//! (`0..7`), not frame numbers. If a spatial enhancement (`S1`) frame is lost
//! in transport without being stripped or repaired, a subsequent inter `S1`
//! frame that references the same slot will silently decode against the stale
//! picture still occupying that slot (matching `dav1d` and `libaom`). Avoiding
//! enhancement drift is the transport/encoder's responsibility: receivers strip
//! `S1` until an `S1`-only intra repair or key unit arrives, or report per-layer
//! loss (`Encoder::report_layer_loss`) so the encoder emits an `S1` repair.
use super::{Dsp, Picture, PlaneIndex, coeff_cdf_q_ctx};
use block_size::BlockSize;
use cdf::Cdfs;
use frame_header::{FrameHeader, NUM_REF_FRAMES, PRIMARY_REF_NONE, SlotHeader};
use motion_field::{MotionField, ProjectionSource, SavedMotion, TemporalField};
use nc_common::decode::{BitReader, DecodeError};
use nc_common::{PictureRecycler, ShownPicture};
use nc_dsp::av1_restoration::Av1LrScratch;
use sequence::Sequence;
use std::rc::Rc;

mod availability;
mod block_info;
mod block_size;
mod cdef;
mod cdf;
mod coefficients;
mod deblock;
mod film_grain;
mod frame_header;
mod motion;
mod motion_field;
mod mv_stack;
mod obu;
mod restoration;
mod sequence;
mod subset;
#[cfg(test)]
mod tests;
mod tile;
mod transform;

/// Luma border of reconstructed (reference) pictures: motion vectors may
/// point this far outside the frame.
const RECON_BORDER: usize = 64;
/// The chroma samples of monochrome frames (8-bit mid-grey).
const MONOCHROME_CHROMA: u8 = 128;
/// Largest temporal unit accepted: memory for untrusted input.
const MAX_TEMPORAL_UNIT_BYTES: usize = 256 * 1024 * 1024;
/// Highest operating point index an AV1 sequence header can signal
/// (`operating_points_cnt_minus_1` is a 5-bit field, AV1 §5.5.1).
pub const MAX_OPERATING_POINT: u8 = (sequence::MAX_OPERATING_POINTS - 1) as u8;

/// A decoded frame as kept in reference slots.
#[derive(Debug)]
struct SavedFrame {
    picture: Picture,
    cdfs: Cdfs,
    header: SlotHeader,
    spatial_id: u8,
    /// The vectors kept for temporal prediction (empty unless the
    /// sequence uses it).
    motion: SavedMotion,
}

/// Stateful AV1 decoder: one temporal unit (low-overhead OBU format, as in
/// IVF) in, the frame it shows out.
#[derive(Debug)]
pub struct Decoder {
    sequence: Option<Sequence>,
    slots: [Option<Rc<SavedFrame>>; NUM_REF_FRAMES],
    /// `RefValid` (§7.20): with frame ids, a slot whose frame is further
    /// back than a delta can name is invalid until refreshed, so a stale
    /// slot cannot match a later id after the ids wrap.
    ref_valid: [bool; NUM_REF_FRAMES],
    dsp: Dsp,
    /// Released pictures: replaced references and outputs the caller
    /// returns ([`Decoder::recycle`]) become later reconstructions and
    /// outputs. Decoding writes every sample of the coded area before
    /// reading it, so their stale samples never reach the output.
    pictures: PictureRecycler,
    /// The block layer's per-frame state, reused by every frame.
    maps: tile::FrameMaps,
    /// The buffer of the current frame's projected motion field.
    field: MotionField,
    /// Loop restoration's window and row buffers, reused by every frame.
    restoration: Av1LrScratch,
    /// The frame the last successful [`Decoder::decode_shown`] showed,
    /// held until the next decode.
    shown: Option<Rc<SavedFrame>>,
    /// That frame with film grain (§7.18.3), when its parameters apply
    /// grain: what is shown in its place (references stay grain-free).
    grain: Option<Picture>,
    /// Whether shown frames get the film grain their parameters signal.
    /// Off shows the grain-free picture, as `aomdec --skip-film-grain` and
    /// `dav1d --filmgrain 0` do: what quality is scored on, since the grain
    /// is noise by design.
    apply_film_grain: bool,
    /// Selected operating point index into the sequence header's
    /// `operating_point_idc[]` table (`0` = highest quality, AV1 §7.18).
    operating_point: u8,
    /// `spatial_id` of the latest frame shown across `decode_shown`,
    /// `decode_with_layer` or `decode`.
    last_spatial_id: Option<u8>,
}
impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

/// A frame whose tile groups are still arriving: its header and the
/// coded tiles so far.
struct PendingFrame<'a> {
    header: FrameHeader,
    spatial_id: u8,
    tiles: Vec<&'a [u8]>,
}

impl<'a> PendingFrame<'a> {
    /// Add the tiles of the tile group in `data`, which must continue
    /// from the last tile added (§7.5: tile groups in order).
    fn add_group(&mut self, data: &'a [u8]) -> Result<(), DecodeError> {
        let info = &self.header.tiles;
        let group = obu::tile_group(data, info)?;
        if group.start != self.tiles.len() {
            return Err(DecodeError::Malformed("AV1 tile group out of order"));
        }
        self.tiles.extend(group.tiles(info.tile_size_bytes)?);
        Ok(())
    }
    /// Every tile of the frame has arrived.
    fn complete(&self) -> bool {
        self.tiles.len() == self.header.tiles.count()
    }
}

/// The motion field (§7.9) of the frame of `header`, in the buffer
/// `field`: it projects from the references' kept vectors one superblock
/// row at a time, as the tiles decode (none when the frame does not use
/// them).
fn temporal_field<'a>(
    field: &'a mut MotionField,
    header: &FrameHeader,
    seq: &Sequence,
    references: &'a [Option<Rc<SavedFrame>>; frame_header::TOTAL_REFS],
) -> TemporalField<'a> {
    let sources = std::array::from_fn(|name| {
        references[name]
            .as_ref()
            .filter(|_| header.use_ref_frame_mvs)
            .map(|saved| ProjectionSource {
                header: &saved.header,
                motion: &saved.motion,
            })
    });
    let mi = (header.size.mi_cols(), header.size.mi_rows());
    TemporalField::new(field, seq, mi, header.order_hint, &sources)
}

impl Decoder {
    /// Start without sequence information or reference pictures.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sequence: None,
            slots: std::array::from_fn(|_| None),
            ref_valid: [true; NUM_REF_FRAMES],
            dsp: Dsp::new(),
            pictures: PictureRecycler::default(),
            maps: tile::FrameMaps::new(),
            field: MotionField::default(),
            restoration: Av1LrScratch::default(),
            shown: None,
            grain: None,
            apply_film_grain: true,
            operating_point: 0,
            last_spatial_id: None,
        }
    }
    /// Show frames with (`true`, the default) or without the film grain
    /// their parameters signal; references are grain-free either way.
    pub fn set_film_grain(&mut self, apply: bool) {
        self.apply_film_grain = apply;
    }
    /// Select the operating point (`0`..=`operating_points_cnt_minus_1`, at
    /// most [`MAX_OPERATING_POINT`], default `0` = highest quality, AV1 §7.18,
    /// §6.4.1). OBUs whose `(temporal_id, spatial_id)` falls outside the
    /// operating point's `operating_point_idc` mask are discarded without
    /// decoding.
    ///
    /// # Errors
    /// Returns [`DecodeError::Unsupported`] if `operating_point > MAX_OPERATING_POINT`
    /// or, once a sequence header has been decoded, if `operating_point` is
    /// outside `0..operating_points_cnt`.
    pub fn set_operating_point(&mut self, operating_point: u8) -> Result<(), DecodeError> {
        if operating_point > MAX_OPERATING_POINT {
            return Err(DecodeError::Unsupported("AV1 operating point out of range"));
        }
        if let Some(seq) = &mut self.sequence {
            let op = usize::from(operating_point);
            if op >= seq.operating_point_count {
                return Err(DecodeError::Unsupported("AV1 operating point out of range"));
            }
            seq.operating_point_idc = seq.operating_points[op].idc;
        }
        self.operating_point = operating_point;
        Ok(())
    }
    /// Active operating point index (`0` by default).
    #[must_use]
    pub fn operating_point(&self) -> u8 {
        self.operating_point
    }
    /// Return a picture [`Decoder::decode`] produced once the caller is done
    /// with it, so a later frame reuses its buffer instead of allocating.
    pub fn recycle(&mut self, picture: Picture) {
        self.pictures.give(picture);
    }
    /// Colour signalling of the active sequence header (`None` before the
    /// first one).
    #[must_use]
    pub fn colour(&self) -> Option<nc_common::ColourInfo> {
        self.sequence.as_ref().map(|s| s.colour)
    }
    /// Maximum frame dimensions (`max_frame_width`, `max_frame_height`) of the
    /// active sequence header (`None` before the first one).
    #[must_use]
    pub fn max_frame_size(&self) -> Option<(usize, usize)> {
        self.sequence
            .as_ref()
            .map(|s| (s.max_frame_width, s.max_frame_height))
    }
    /// `spatial_id` (`0` on single-layer or base-layer TUs, `1` on enhancement
    /// TUs) of the latest shown frame.
    #[must_use]
    pub fn shown_spatial_id(&self) -> Option<u8> {
        self.last_spatial_id
    }
    /// Decode one temporal unit and return `(picture, spatial_id)` for the
    /// highest spatial layer shown in the selected operating point, or `None`
    /// when every frame OBU in the unit belonged to a dropped temporal/spatial
    /// layer (`AV1 §7.18`). When `Ok(None)` or `Err` is returned, the
    /// previously shown picture ([`Decoder::shown`]) is retained.
    ///
    /// # Errors
    /// As [`Decoder::decode`].
    pub fn decode_with_layer(
        &mut self,
        temporal_unit: &[u8],
    ) -> Result<Option<(Picture, u8)>, DecodeError> {
        let Some(saved) = self.decode_unit(temporal_unit)? else {
            return Ok(None);
        };
        let grain = self.add_grain(&saved)?;
        let view = match &grain {
            Some(g) => frame_window(g, &saved.header.size),
            None => visible(&saved),
        };
        let mut output = self.pictures.take(view.width(), view.height(), 0)?;
        view.copy_to(&mut output);
        let spatial_id = saved.spatial_id;
        self.release_shown();
        self.last_spatial_id = Some(spatial_id);
        self.grain = grain;
        self.shown = Some(saved);
        Ok(Some((output, spatial_id)))
    }
    /// Decode one temporal unit and return the picture it shows, or `None`
    /// when every frame OBU in the unit belonged to a temporal/spatial layer
    /// dropped by the selected operating point (`AV1 §7.18`). When `Ok(None)`
    /// or `Err` is returned, [`Decoder::shown`] is retained.
    ///
    /// # Errors
    /// As [`Decoder::decode`].
    pub fn decode_optional(
        &mut self,
        temporal_unit: &[u8],
    ) -> Result<Option<Picture>, DecodeError> {
        self.decode_with_layer(temporal_unit)
            .map(|opt| opt.map(|(picture, _)| picture))
    }
    /// Decode one temporal unit and return `(picture, spatial_id)` for every
    /// shown spatial layer in the selected operating point in OBU order (matching
    /// `dav1d --alllayers 1` and `aomdec --all-layers`). When the operating
    /// point drops every OBU in the unit (`Ok(vec![])`) or decoding fails,
    /// [`Decoder::shown`] is retained.
    ///
    /// # Errors
    /// As [`Decoder::decode`].
    pub fn decode_all_layers(
        &mut self,
        temporal_unit: &[u8],
    ) -> Result<Vec<(Picture, u8)>, DecodeError> {
        let frames = self.decode_unit_frames(temporal_unit, true)?;
        let mut outputs = Vec::with_capacity(frames.len());
        let mut last_state = None;
        for saved in frames {
            let grain = self.add_grain(&saved)?;
            let view = match &grain {
                Some(g) => frame_window(g, &saved.header.size),
                None => visible(&saved),
            };
            let mut output = self.pictures.take(view.width(), view.height(), 0)?;
            view.copy_to(&mut output);
            outputs.push((output, saved.spatial_id));
            if let Some((_, Some(prev_grain))) = last_state.replace((saved, grain)) {
                self.pictures.give(prev_grain);
            }
        }
        if let Some((saved, grain)) = last_state {
            self.release_shown();
            self.last_spatial_id = Some(saved.spatial_id);
            self.grain = grain;
            self.shown = Some(saved);
        }
        Ok(outputs)
    }
    /// Decode one temporal unit that must produce a shown frame and return it
    /// (the highest spatial layer in the unit). For operating points where a
    /// temporal unit may be legally dropped in full, use
    /// [`Decoder::decode_optional`], [`Decoder::decode_with_layer`] or
    /// [`Decoder::decode_shown`]. A failed unit leaves the reference slots and
    /// [`Decoder::shown`] as the frames before the failing one left them.
    ///
    /// # Errors
    /// Returns [`DecodeError`] for malformed, truncated or unsupported syntax,
    /// a temporal unit that shows no frame, missing references, or dimensions
    /// beyond the supported allocation limit.
    pub fn decode(&mut self, temporal_unit: &[u8]) -> Result<Picture, DecodeError> {
        self.decode_with_layer(temporal_unit)?
            .map(|(picture, _)| picture)
            .ok_or(DecodeError::Malformed("AV1 temporal unit shows no frame"))
    }

    /// [`Decoder::decode`] without the output copy: the frame is shown by
    /// [`Decoder::shown`] until the next call. When the selected operating
    /// point drops every OBU in `temporal_unit`, or when decoding fails, the
    /// previous shown frame is retained.
    ///
    /// # Errors
    /// As [`Decoder::decode`].
    pub fn decode_shown(&mut self, temporal_unit: &[u8]) -> Result<(), DecodeError> {
        self.decode_and_show(temporal_unit).map(|_| ())
    }

    /// [`Decoder::decode_shown`], returning whether the unit showed a new
    /// frame (`false` when the selected operating point dropped all of it).
    ///
    /// # Errors
    /// As [`Decoder::decode`].
    pub fn decode_and_show(&mut self, temporal_unit: &[u8]) -> Result<bool, DecodeError> {
        let Some(saved) = self.decode_unit(temporal_unit)? else {
            return Ok(false);
        };
        let grain = self.add_grain(&saved)?;
        self.release_shown();
        self.last_spatial_id = Some(saved.spatial_id);
        self.grain = grain;
        self.shown = Some(saved);
        Ok(true)
    }

    /// The visible samples (the upscaled frame size) of the frame the last
    /// successful [`Decoder::decode_shown`] or [`Decoder::decode_with_layer`]
    /// showed.
    #[must_use]
    pub fn shown(&self) -> Option<ShownPicture<'_>> {
        let saved = self.shown.as_deref()?;
        Some(match &self.grain {
            Some(grain) => frame_window(grain, &saved.header.size),
            None => visible(saved),
        })
    }

    /// `saved` with film grain added (§7.18.3), or `None` when its
    /// parameters apply none or grain is switched off.
    fn add_grain(&mut self, saved: &SavedFrame) -> Result<Option<Picture>, DecodeError> {
        let params = &saved.header.film_grain;
        if !self.apply_film_grain || !params.apply_grain {
            return Ok(None);
        }
        let size = (saved.header.size.upscaled_width, saved.header.size.height);
        let mut output = self.pictures.take(size.0, size.1, 0)?;
        film_grain::apply(&self.dsp, params, &saved.picture, size, &mut output)?;
        Ok(Some(output))
    }

    /// Drop the hold on the shown frame; its picture is reused once no
    /// slot holds it either.
    fn release_shown(&mut self) {
        if let Some(grain) = self.grain.take() {
            self.pictures.give(grain);
        }
        if let Some(frame) = self.shown.take().and_then(|f| Rc::try_unwrap(f).ok()) {
            self.pictures.give(frame.picture);
        }
    }

    /// The frame one temporal unit shows (`None` only when every frame OBU in
    /// the unit was filtered out by `operating_point_idc`, `AV1 §7.18`, or had
    /// `show_frame == 0`).
    fn decode_unit(&mut self, temporal_unit: &[u8]) -> Result<Option<Rc<SavedFrame>>, DecodeError> {
        Ok(self.decode_unit_frames(temporal_unit, false)?.pop())
    }

    /// All shown frames in `temporal_unit` (when `all_layers` is `true`) or
    /// only the final shown frame (when `all_layers` is `false`). A failed
    /// unit keeps the slots' `RefValid` as they were: its frame header may
    /// have marked them (§7.20) from a `current_frame_id` that garbage or a
    /// damaged frame carried, and the next good frame would then be refused.
    fn decode_unit_frames(
        &mut self,
        temporal_unit: &[u8],
        all_layers: bool,
    ) -> Result<Vec<Rc<SavedFrame>>, DecodeError> {
        let ref_valid = self.ref_valid;
        let frames = self.decode_unit_frames_inner(temporal_unit, all_layers);
        if frames.is_err() {
            self.ref_valid = ref_valid;
        }
        frames
    }

    /// [`Decoder::decode_unit_frames`] without the rollback.
    fn decode_unit_frames_inner(
        &mut self,
        temporal_unit: &[u8],
        all_layers: bool,
    ) -> Result<Vec<Rc<SavedFrame>>, DecodeError> {
        if temporal_unit.len() > MAX_TEMPORAL_UNIT_BYTES {
            return Err(DecodeError::Unsupported("AV1 temporal unit size limit"));
        }
        let mut shown: Vec<Rc<SavedFrame>> = Vec::new();
        let mut dropped_or_unshown = false;
        let mut pending: Option<PendingFrame> = None;
        for unit in obu::split(temporal_unit)? {
            if unit.has_extension
                && !matches!(
                    unit.kind,
                    obu::OBU_SEQUENCE_HEADER | obu::OBU_TEMPORAL_DELIMITER
                )
                && let Some(seq) = &self.sequence
                && !seq.in_operating_point(unit.layer.temporal, unit.layer.spatial)
            {
                dropped_or_unshown = true;
                continue;
            }
            match unit.kind {
                obu::OBU_SEQUENCE_HEADER => self.sequence_header(unit.payload)?,
                obu::OBU_FRAME_HEADER | obu::OBU_REDUNDANT_FRAME_HEADER | obu::OBU_FRAME => {
                    if pending.is_some() {
                        // A redundant copy of the header being decoded.
                        continue;
                    }
                    // No sequence header yet: the key frame carrying it was lost.
                    let seq = self.sequence.ok_or(DecodeError::MissingKey)?;
                    let mut r = BitReader::new(unit.payload);
                    let header =
                        frame_header::parse(&mut r, &seq, &self.slot_headers(), unit.layer)?;
                    if let Some(index) = header.show_existing {
                        let existing = self.show_existing(index, &header)?;
                        if !all_layers
                            && let Some(prev) = shown.pop()
                            && let Ok(unused) = Rc::try_unwrap(prev)
                        {
                            self.pictures.give(unused.picture);
                        }
                        shown.push(existing);
                        continue;
                    }
                    if let Some(ids) = seq.frame_ids {
                        self.mark_ref_frames(header.current_frame_id, ids);
                    }
                    let mut frame = PendingFrame {
                        header,
                        spatial_id: unit.layer.spatial,
                        tiles: Vec::new(),
                    };
                    if unit.kind == obu::OBU_FRAME {
                        r.align_zero()?;
                        frame.add_group(r.remaining_bytes()?)?;
                    }
                    pending = Some(frame);
                }
                obu::OBU_TILE_GROUP => {
                    pending
                        .as_mut()
                        .ok_or(DecodeError::Malformed(
                            "AV1 tile group without frame header",
                        ))?
                        .add_group(unit.payload)?;
                }
                obu::OBU_TILE_LIST => return Err(DecodeError::Unsupported("AV1 tile list OBUs")),
                // Temporal delimiters, metadata, padding and reserved types
                // (ignored, §6.2.2) carry nothing to decode.
                _ => {}
            }
            if let Some(frame) = pending.take_if(|p| p.complete()) {
                if let Some(new_shown) =
                    self.decode_frame(&frame.header, frame.spatial_id, &frame.tiles)?
                {
                    if !all_layers
                        && let Some(prev) = shown.pop()
                        && let Ok(unused) = Rc::try_unwrap(prev)
                    {
                        self.pictures.give(unused.picture);
                    }
                    shown.push(new_shown);
                } else {
                    dropped_or_unshown = true;
                }
            }
        }
        if pending.is_some() {
            return Err(DecodeError::Malformed("AV1 frame missing tile group"));
        }
        if !shown.is_empty() || dropped_or_unshown {
            Ok(shown)
        } else {
            Err(DecodeError::Malformed("AV1 temporal unit shows no frame"))
        }
    }

    /// A sequence header: a changed one starts a new coded video sequence,
    /// whose references are no longer valid.
    fn sequence_header(&mut self, payload: &[u8]) -> Result<(), DecodeError> {
        let mut seq = sequence::parse(payload)?;
        let op = usize::from(self.operating_point);
        if op >= seq.operating_point_count {
            return Err(DecodeError::Unsupported("AV1 operating point out of range"));
        }
        seq.operating_point_idc = seq.operating_points[op].idc;
        if self.sequence != Some(seq) {
            self.slots.fill(None);
            self.sequence = Some(seq);
        }
        Ok(())
    }

    fn slot_headers(&self) -> [Option<&SlotHeader>; NUM_REF_FRAMES] {
        std::array::from_fn(|i| {
            self.slots[i]
                .as_deref()
                .filter(|_| self.ref_valid[i])
                .map(|s| &s.header)
        })
    }

    /// `mark_ref_frames()` (§7.20) for a frame with `current_frame_id`
    /// `current`: slots more than `2^delta` frames back become invalid.
    fn mark_ref_frames(&mut self, current: u32, ids: sequence::FrameIdLengths) {
        let range = 1 << ids.id;
        for (valid, slot) in self.ref_valid.iter_mut().zip(&self.slots) {
            if let Some(slot) = slot
                && (current + range - slot.header.frame_id) % range > (1 << ids.delta)
            {
                *valid = false;
            }
        }
    }

    /// The inter tools of the frame of `header`.
    ///
    /// # Errors
    /// A reference more than twice the frame's size or under a sixteenth
    /// (§7.11.3.3).
    fn inter_tools(
        header: &FrameHeader,
        seq: &Sequence,
        references: &[Option<Rc<SavedFrame>>; frame_header::TOTAL_REFS],
    ) -> Result<tile::InterTools, DecodeError> {
        let mut scales = [None; frame_header::TOTAL_REFS];
        for (scale, saved) in scales.iter_mut().zip(references) {
            if let Some(saved) = saved {
                let size = &saved.header.size;
                *scale = tile::scale_sizes(
                    (size.upscaled_width, size.height),
                    (header.size.width, header.size.height),
                )?;
            }
        }
        Ok(tile::InterTools {
            interpolation_filter: header.interpolation_filter,
            switchable_motion_mode: header.is_motion_mode_switchable,
            allow_warped_motion: header.allow_warped_motion,
            interintra: seq.enable_interintra_compound,
            dual_filter: seq.enable_dual_filter,
            masked_compound: seq.enable_masked_compound,
            jnt_comp: seq.enable_jnt_comp,
            use_ref_frame_mvs: header.use_ref_frame_mvs,
            ref_offsets: std::array::from_fn(|name| {
                frame_header::relative_distance(seq, header.order_hint, header.order_hints[name])
            }),
            saved_sides: seq.enable_ref_frame_mvs.then(|| {
                motion_field::reference_sides(seq, header.order_hint, &header.order_hints)
            }),
            scales,
        })
    }

    /// The reference frames of an inter frame by reference name (index 0,
    /// `INTRA_FRAME`, empty); none for intra frames.
    ///
    /// # Errors
    /// An empty slot.
    fn references(
        &self,
        header: &FrameHeader,
    ) -> Result<[Option<Rc<SavedFrame>>; frame_header::TOTAL_REFS], DecodeError> {
        let mut references: [Option<Rc<SavedFrame>>; frame_header::TOTAL_REFS] = Default::default();
        if header.frame_type.is_intra() {
            return Ok(references);
        }
        for (name, &index) in header.ref_frame_idx.iter().enumerate() {
            references[name + 1] = Some(Rc::clone(self.slot(index)?));
        }
        Ok(references)
    }

    fn slot(&self, index: u8) -> Result<&Rc<SavedFrame>, DecodeError> {
        self.slots[usize::from(index)]
            .as_ref()
            .ok_or(DecodeError::MissingReference)
    }

    /// `show_existing_frame` (§7.21): output a slot; a key frame shown this
    /// way also refreshes every slot.
    fn show_existing(
        &mut self,
        index: usize,
        header: &FrameHeader,
    ) -> Result<Rc<SavedFrame>, DecodeError> {
        let saved = Rc::clone(self.slot(index as u8)?);
        if header.refresh_frame_flags != 0 {
            self.refresh(&saved, header.refresh_frame_flags);
        }
        Ok(saved)
    }

    /// Decode one frame from its coded tiles; returns it if shown.
    fn decode_frame(
        &mut self,
        header: &FrameHeader,
        spatial_id: u8,
        tiles: &[&[u8]],
    ) -> Result<Option<Rc<SavedFrame>>, DecodeError> {
        let params = subset::frame_parameters(header)?;
        let seq = self
            .sequence
            .ok_or(DecodeError::Malformed("AV1 frame before sequence header"))?;
        let superblock = if seq.use_128x128_superblock {
            BlockSize::BLOCK_128X128
        } else {
            BlockSize::BLOCK_64X64
        };
        let references = self.references(header)?;
        let inter_tools = Self::inter_tools(header, &seq, &references)?;
        let cdfs = Box::new(if header.primary_ref_frame == PRIMARY_REF_NONE {
            Cdfs::new(coeff_cdf_q_ctx(params.base_q_idx))
        } else {
            let primary = header.ref_frame_idx[usize::from(header.primary_ref_frame)];
            self.slot(primary)?.cdfs.clone()
        });
        // Without the frame-end update (§7.4 `disable_frame_end_update_cdf`)
        // the slots keep the CDFs the frame started from.
        let start_cdfs = header.disable_frame_end_update_cdf.then(|| cdfs.clone());
        // Blocks cover whole 8x8 units beyond the frame size (§7.21).
        let (coded_width, coded_height) = super::coded_picture_size(params.width, params.height);
        let coded = subset::FrameParameters {
            width: coded_width,
            height: coded_height,
            ..params.clone()
        };
        let recon = self
            .pictures
            .take(coded_width, coded_height, RECON_BORDER)?;
        let temporal = temporal_field(&mut self.field, header, &seq, &references);
        let mut state = tile::FrameDecoder::new(
            coded,
            cdfs,
            &self.dsp,
            std::array::from_fn(|name| references[name].as_ref().map(|saved| &saved.picture)),
            recon,
            &mut self.maps,
            tile::SequenceTools {
                superblock,
                filter_intra: seq.enable_filter_intra,
                intra_edge_filter: seq.enable_intra_edge_filter,
                cdef: seq.enable_cdef,
                mono_chrome: seq.mono_chrome,
            },
            inter_tools,
            temporal,
        );
        state.tiles(&header.tiles, tiles, !header.disable_cdf_update)?;
        // The post-decode filters (§7.14–7.17): deblocking, CDEF,
        // superres, loop restoration, which also reads deblocked rows.
        state.deblock((params.width, params.height));
        let boundaries = state
            .restoration
            .as_ref()
            .map(|_| restoration::StripeBoundaries::save(&state.recon, &header.size))
            .transpose()?;
        state.cdef();
        let tile::FrameDecoder {
            recon,
            cdfs,
            restoration: units,
            saved_motion,
            ..
        } = state;
        let mut recon = self.upscale(recon, &header.size)?;
        if let (Some(boundaries), Some(units)) = (&boundaries, &units) {
            restoration::apply(
                (&self.dsp, &mut self.restoration),
                &mut recon,
                boundaries,
                units.units(),
                &header.size,
            )?;
        }
        if seq.mono_chrome {
            // Monochrome output shows mid-grey chroma (§7.18 output process:
            // `1 << (BitDepth - 1)`).
            for plane in [PlaneIndex::U, PlaneIndex::V] {
                recon.plane_mut(plane).view_mut().fill(MONOCHROME_CHROMA);
            }
        }
        // References clamp to the frame size (§7.11.3.3).
        let (width, height) = (header.size.upscaled_width, header.size.height);
        if (recon.width(), recon.height()) != (width, height) {
            recon.replicate_beyond(width, height);
        }
        recon.extend_borders();
        let saved = Rc::new(SavedFrame {
            picture: recon,
            cdfs: start_cdfs.map_or_else(|| cdfs.saved(), |start| *start),
            header: slot_header(header),
            spatial_id,
            motion: saved_motion.unwrap_or_default(),
        });
        // Our hold on the references ends here, so a slot the refresh
        // replaces can hand its picture back for reuse instead of freeing
        // it (with the hold, every frame allocated and zeroed a new one).
        drop(references);
        self.refresh(&saved, header.refresh_frame_flags);
        if header.show_frame {
            return Ok(Some(saved));
        }
        if let Ok(unused) = Rc::try_unwrap(saved) {
            self.pictures.give(unused.picture);
        }
        Ok(None)
    }

    /// The frame `coded` (after CDEF) at its upscaled width (§7.16): itself
    /// without superres, else a new picture, `coded` going back for reuse.
    fn upscale(
        &mut self,
        coded: Picture,
        size: &frame_header::FrameSize,
    ) -> Result<Picture, DecodeError> {
        if size.width == size.upscaled_width {
            return Ok(coded);
        }
        let (width, height) = super::coded_picture_size(size.upscaled_width, size.height);
        let mut upscaled = self.pictures.take(width, height, RECON_BORDER)?;
        restoration::superres_upscale(&coded, size, &mut upscaled)?;
        self.pictures.give(coded);
        Ok(upscaled)
    }

    /// Keep `saved` in the slots `refresh` names (§7.20); pictures no slot
    /// holds any more are released for reuse.
    fn refresh(&mut self, saved: &Rc<SavedFrame>, refresh: u8) {
        for (slot, held) in self.slots.iter_mut().enumerate() {
            if refresh & (1 << slot) != 0 {
                self.ref_valid[slot] = true;
            }
            if refresh & (1 << slot) != 0
                && let Some(replaced) = held.replace(Rc::clone(saved))
                && let Ok(frame) = Rc::try_unwrap(replaced)
            {
                self.pictures.give(frame.picture);
            }
        }
    }

    /// Test hook: overwrite every reference picture with `value`, as a
    /// decoder that lost a frame would hold wrong references.
    #[cfg(test)]
    pub(crate) fn corrupt_references(&mut self, value: u8) {
        for slot in self.slots.iter_mut().flatten() {
            let mut picture = slot.picture.clone();
            for plane in PlaneIndex::ALL {
                picture.plane_mut(plane).view_mut().fill(value);
            }
            picture.extend_borders();
            *slot = Rc::new(SavedFrame {
                picture,
                cdfs: slot.cdfs.clone(),
                header: slot.header.clone(),
                spatial_id: slot.spatial_id,
                motion: slot.motion.clone(),
            });
        }
    }

    /// Test hook: the block-layer parameters of the frame in `temporal_unit`,
    /// parsed against the current state (and its own sequence header).
    #[cfg(test)]
    fn peek_header(&self, temporal_unit: &[u8]) -> subset::FrameParameters {
        let mut seq = self.sequence;
        for unit in obu::split(temporal_unit).unwrap() {
            match unit.kind {
                obu::OBU_SEQUENCE_HEADER => seq = Some(sequence::parse(unit.payload).unwrap()),
                obu::OBU_FRAME => {
                    let mut r = BitReader::new(unit.payload);
                    let header = frame_header::parse(
                        &mut r,
                        &seq.unwrap(),
                        &self.slot_headers(),
                        unit.layer,
                    )
                    .unwrap();
                    return subset::frame_parameters(&header).unwrap();
                }
                _ => {}
            }
        }
        panic!("temporal unit without a frame");
    }
}

/// What a slot keeps of `h` for later frame headers (§7.20).
fn slot_header(h: &FrameHeader) -> SlotHeader {
    SlotHeader {
        frame_id: h.current_frame_id,
        frame_type: h.frame_type,
        size: h.size,
        order_hint: h.order_hint,
        ref_order_hints: h.order_hints,
        showable: h.showable_frame,
        global_motion: h.global_motion,
        loop_filter_deltas: h.loop_filter.deltas,
        segmentation: h.segmentation.features,
        film_grain: h.film_grain,
    }
}

/// The visible samples of a decoded frame: its upscaled size, from the
/// top left of the (8x8-padded) picture.
fn visible(saved: &SavedFrame) -> ShownPicture<'_> {
    frame_window(&saved.picture, &saved.header.size)
}

/// The visible samples (the upscaled frame size) of `picture`.
fn frame_window<'a>(picture: &'a Picture, size: &frame_header::FrameSize) -> ShownPicture<'a> {
    ShownPicture::window(picture, (0, 0), (size.upscaled_width, size.height))
        .expect("the picture covers the frame")
}

/// The temporal layer of an emitted temporal unit: its frame OBU's
/// `temporal_id` (0 without an extension header), what operating-point
/// selection (§7.18) compares against the layers it keeps.
///
/// # Errors
///
/// Malformed OBU framing, or a temporal unit without a frame.
pub fn temporal_unit_layer(temporal_unit: &[u8]) -> Result<u8, DecodeError> {
    obu::split(temporal_unit)?
        .into_iter()
        .find(|unit| matches!(unit.kind, obu::OBU_FRAME | obu::OBU_FRAME_HEADER))
        .map(|unit| unit.layer.temporal)
        .ok_or(DecodeError::Malformed("temporal unit without a frame"))
}

/// Every coded frame's `(temporal_id, spatial_id)` in `temporal_unit`, in OBU
/// order (AV1 §5.3.3).
///
/// # Errors
///
/// Malformed OBU framing, or a temporal unit without a frame.
pub fn temporal_unit_layers(temporal_unit: &[u8]) -> Result<Vec<(u8, u8)>, DecodeError> {
    let layers: Vec<(u8, u8)> = obu::split(temporal_unit)?
        .into_iter()
        .filter(|unit| matches!(unit.kind, obu::OBU_FRAME | obu::OBU_FRAME_HEADER))
        .map(|unit| (unit.layer.temporal, unit.layer.spatial))
        .collect();
    if layers.is_empty() {
        return Err(DecodeError::Malformed("temporal unit without a frame"));
    }
    Ok(layers)
}

/// Rebuild `temporal_unit` keeping every layer-shared OBU (temporal delimiter,
/// sequence header) and only the extension OBUs where `keep(temporal_id,
/// spatial_id)` returns `true` (AV1 §7.18 / RTP selective forwarding).
///
/// # Errors
///
/// Malformed OBU framing.
pub fn filter_temporal_unit_layers(
    temporal_unit: &[u8],
    mut keep: impl FnMut(u8, u8) -> bool,
) -> Result<Vec<u8>, DecodeError> {
    let mut out = Vec::with_capacity(temporal_unit.len());
    for unit in obu::split(temporal_unit)? {
        if !unit.has_extension || keep(unit.layer.temporal, unit.layer.spatial) {
            out.extend_from_slice(unit.raw);
        }
    }
    Ok(out)
}
