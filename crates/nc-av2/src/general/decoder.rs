//! The AV2 decoder's top level: OBU dispatch, the reference slots and the
//! output order (hidden frames, show-existing frames, frames shown later
//! in display order). Film grain is laid over a picture as it is output
//! ([`super::film_grain`]), never over the reference it stays in.
//!
//! Each temporal unit is decoded frame OBU by frame OBU; every picture to
//! show goes to an output queue in display order, read with
//! [`Decoder::next_output`] (and, at the end of the stream, after
//! [`Decoder::flush`]), in the order of [`super::output`].
//!
//! Blocks are decoded by the subset block layer ([`super::subset`]) until
//! the general one exists.
use std::collections::VecDeque;
use std::rc::Rc;

use super::block::motion::frame::{ReferenceMotion, TemporalField};
use super::block::segmentation::SegmentMap;
use super::block::{self, BlockTrace};
use super::cdfs::Cdfs;
use super::film_grain::{self, OutputGrain};
use super::filters;
use super::frame_header::restoration::kind;
use super::frame_header::{
    self, Context, FrameHeader, FrameType, MAX_SLOTS, Parsed, ShowExisting, SlotHeader,
    TIP_FRAME_MODE, obu_type,
};
use super::output::{OutputOrder, Release};
use super::references::{self, MAX_REFERENCES};
use super::sequence::{self, SequenceHeader};
use super::subset;
use crate::cdf_models::TileCdfs;
use crate::obu;
use crate::tile_decoder::{self, InterReference};
use nc_common::decode::{BitReader, DecodeError, Result};
use nc_common::{Picture, ShownPicture};
use nc_dsp::av2_dec::wedge::InterMasks;

/// OBU types other than frames that the decoder reads or skips (AV2 §6.2).
mod other_obu {
    pub const SEQUENCE_HEADER: u8 = 1;
    pub const TEMPORAL_DELIMITER: u8 = 2;
    pub const METADATA: u8 = 8;
    pub const METADATA_GROUP: u8 = 9;
    pub const FILM_GRAIN_MODEL: u8 = 23;
    pub const CONTENT_INTERPRETATION: u8 = 24;
    pub const PADDING: u8 = 25;
}

/// A frame's CCSO flags per 64x64 filter block and plane.
type CcsoFlags = Rc<Vec<[bool; 3]>>;
/// A frame's segment ids per luma 4x4 unit.
type Segments = Rc<SegmentMap>;

/// What a general-path frame leaves in its slots beside its picture and
/// models.
#[derive(Debug, Default)]
struct SlotExtras {
    ccso_flags: Option<CcsoFlags>,
    motion_field: Option<Rc<TemporalField>>,
    segments: Option<Segments>,
}

/// What a reference slot holds beyond the frame's header.
#[derive(Debug, Clone)]
struct SlotData {
    picture: Rc<Picture>,
    /// The frame's final CDFs, adaptation restarted (AVM `frame_context`),
    /// for frames that start from them; `None` where it did not adapt.
    context: Option<FrameContext>,
    /// The frame's CCSO flags per 64x64 block and plane, for frames that
    /// reuse them (dav2d `ccsomap`); `None` without CCSO.
    ccso_flags: Option<CcsoFlags>,
    /// The frame's temporal motion field (general-path inter frames of
    /// sequences with motion fields), projected by later frames.
    motion_field: Option<Rc<TemporalField>>,
    /// The segment map later frames predict from (dav2d `refs[].segmap`):
    /// `None` without segmentation.
    segments: Option<Segments>,
    /// The film grain of the frame's own header, for its output when it is
    /// released later (`show_implicit`).
    grain: Option<OutputGrain>,
}

/// A frame's final models, of the block layer that decoded it.
#[derive(Debug, Clone)]
enum FrameContext {
    /// The subset tile decoder's.
    Subset(Rc<TileCdfs>),
    /// The general block layer's.
    General(Rc<Cdfs>),
}

/// One picture in the output queue.
#[derive(Debug)]
struct Output {
    picture: Rc<Picture>,
    /// The grain laid over it when it is shown.
    grain: Option<OutputGrain>,
}

/// What a temporal unit changes, restored when it fails: the sequence,
/// the reference slots, the film grain models and the output order.
#[derive(Debug, Clone, Default)]
struct State {
    /// Shared with the saved state of a unit (cloning it allocates).
    sequence: Option<Rc<SequenceHeader>>,
    /// The same sequence header as the subset block layer reads it, or why
    /// that layer cannot decode its frames.
    subset_sequence: Option<std::result::Result<crate::sequence::SequenceHeader, DecodeError>>,
    /// Headers of the frames in the slots (`ref_frame_map`).
    headers: [Option<SlotHeader>; MAX_SLOTS],
    /// Their pictures and contexts.
    data: [Option<SlotData>; MAX_SLOTS],
    /// The film grain models FGM OBUs set, by id (dav2d `c->fgm`).
    grain_models: film_grain::Models,
    /// Frames decoded so far: the identity of the next.
    frames: u32,
    /// Which pictures go out when.
    order: OutputOrder,
}

/// An AV2 decoder: sequence state, reference slots and the output queue.
/// A temporal unit that fails leaves it as it was.
#[derive(Debug, Default)]
pub struct Decoder {
    state: State,
    /// Pictures to show, in display order.
    output: VecDeque<Output>,
    /// The picture [`Decoder::shown`] shows.
    current: Option<Rc<Picture>>,
    /// [`Decoder::decode_ref`]'s copy of the shown picture's window when it
    /// crops anything.
    cropped: Option<Picture>,
    /// Pictures the slots let go of during the current temporal unit,
    /// given to the workspace for reuse once the unit's saved state is gone
    /// (until then they are shared, and would be dropped instead).
    retired: Vec<Rc<Picture>>,
    /// Buffers reused from picture to picture.
    workspace: tile_decoder::Workspace,
    /// Where the general block layer's trace goes, if anywhere.
    block_trace: TraceSink,
    /// Diagnosis: general-path frames skip their loop filters (as dav2d's
    /// `--inloopfilters none`), to hold the reconstruction against it.
    unfiltered: bool,
    /// Diagnosis: frames not shown when decoded are output then too (as
    /// dav2d's `--outputinvisible 1`).
    invisible: bool,
    /// The wedge and inter-intra masks, built at the first general frame.
    masks: std::cell::OnceCell<InterMasks>,
}

/// A block-trace sink ([`Decoder::trace_blocks`]).
#[derive(Default)]
struct TraceSink(Option<BlockSink>);

/// A boxed block-trace receiver.
pub type BlockSink = Box<dyn FnMut(std::fmt::Arguments<'_>)>;

impl std::fmt::Debug for TraceSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.0.is_some() {
            "TraceSink(on)"
        } else {
            "TraceSink(off)"
        })
    }
}

impl Decoder {
    /// Start collecting coding-tool analysis into `stats` (block records,
    /// and per-frame tallies unless `records_only`) from the next frame on.
    /// Decoding is unchanged.
    #[cfg(feature = "analysis")]
    pub fn collect_stats(&mut self, stats: nc_common::stream_stats::StreamStats) {
        self.workspace.stats = Some(stats);
    }

    /// The analysis collected so far, if enabled: the block records of the
    /// last decoded frame and the tallies of every completed one.
    #[cfg(feature = "analysis")]
    #[must_use]
    pub fn stats(&self) -> Option<&nc_common::stream_stats::StreamStats> {
        self.workspace.stats.as_ref()
    }

    /// Complete the analysis and hand it over; collection stops.
    #[cfg(feature = "analysis")]
    pub fn take_stats(&mut self) -> Option<nc_common::stream_stats::StreamStats> {
        let mut stats = self.workspace.stats.take()?;
        stats.finish();
        Some(stats)
    }

    /// Decode one temporal unit: its pictures to show join the output
    /// queue.
    /// # Errors
    /// Rejects missing references, unsupported or malformed syntax; the
    /// decoder is then as before the unit.
    pub fn decode_unit(&mut self, unit: &[u8]) -> Result<()> {
        self.transaction(unit, false)
    }

    /// Decode `unit`, requiring it to show a picture when `shows`, and
    /// restore the state when it fails.
    fn transaction(&mut self, unit: &[u8], shows: bool) -> Result<()> {
        let saved = self.state.clone();
        let queued = self.output.len();
        let mut result = self.decode_obus(unit);
        if result.is_ok() && shows && self.output.len() == queued {
            result = Err(DecodeError::Unsupported(
                "AV2 temporal unit without a shown picture",
            ));
        }
        if result.is_err() {
            self.state = saved;
            self.output.truncate(queued);
        } else {
            drop(saved);
        }
        for picture in self.retired.drain(..) {
            self.workspace.release(picture);
        }
        result
    }

    fn decode_obus(&mut self, unit: &[u8]) -> Result<()> {
        let mut remaining = unit;
        while !remaining.is_empty() {
            let (obu, rest) = obu::parse(remaining)?;
            remaining = rest;
            self.decode_obu(&obu)?;
        }
        Ok(())
    }

    /// Show the next queued picture ([`Decoder::shown`]); `false` when none
    /// is queued.
    pub fn next_output(&mut self) -> bool {
        let Some(next) = self.output.pop_front() else {
            return false;
        };
        let picture = match &next.grain {
            Some(grain) => Rc::new(self.grained(&next.picture, grain)),
            None => next.picture,
        };
        if let Some(previous) = self.current.replace(picture) {
            self.workspace.release(previous);
        }
        true
    }

    /// `picture` with `grain` laid over it, in a picture of the workspace.
    fn grained(&mut self, picture: &Picture, grain: &OutputGrain) -> Picture {
        let mut output = self
            .workspace
            .pictures
            .take(picture.width(), picture.height(), 0)
            .expect("the geometry of a picture already made");
        let (dsp, ..) = self.workspace.filter_parts();
        grain.apply(dsp, picture, &mut output);
        output
    }

    /// End of stream: queue the hidden frames still due for output.
    pub fn flush(&mut self) {
        for slot in self.state.order.flush(&self.state.headers) {
            self.queue_slot(slot);
        }
    }

    /// The visible samples (the sequence's conformance window) of the
    /// picture shown.
    #[must_use]
    pub fn shown(&self) -> Option<ShownPicture<'_>> {
        let picture = self.current.as_deref()?;
        let sequence = self.state.sequence.as_ref()?;
        let crop = sequence.crop.unwrap_or_default();
        let width = picture
            .width()
            .checked_sub((crop.left + crop.right) as usize)?;
        let height = picture
            .height()
            .checked_sub((crop.top + crop.bottom) as usize)?;
        ShownPicture::window(
            picture,
            (crop.left as usize, crop.top as usize),
            (width, height),
        )
    }

    /// Decode a temporal unit that shows one picture (as our encoder's
    /// do) and show it.
    /// # Errors
    /// As [`Decoder::decode_unit`], and units that show nothing.
    pub fn decode_shown(&mut self, packet: &[u8]) -> Result<()> {
        self.transaction(packet, true)?;
        self.next_output();
        Ok(())
    }

    /// Decode a packet, returning the borrowed visible picture (the
    /// reconstruction inside the sequence's conformance window) to avoid a
    /// frame copy when nothing is cropped.
    /// The returned picture remains valid until the next mutable decoder operation.
    /// # Errors
    /// As [`Decoder::decode_shown`].
    pub fn decode_ref(&mut self, packet: &[u8]) -> Result<&Picture> {
        self.decode_shown(packet)?;
        let shown = self
            .shown()
            .ok_or(DecodeError::Malformed("AV2 conformance window"))?;
        let current = self.current.as_deref().expect("a shown picture");
        self.cropped = if (shown.width(), shown.height()) == (current.width(), current.height()) {
            None
        } else {
            let mut cropped = Picture::new(shown.width(), shown.height(), 0)?;
            shown.copy_to(&mut cropped);
            Some(cropped)
        };
        let current = self.current.as_deref().expect("a shown picture");
        Ok(self.cropped.as_ref().unwrap_or(current))
    }

    /// Decode a packet and return an owned copy; use `decode_ref` to avoid the copy.
    /// # Errors
    /// As [`Decoder::decode_shown`].
    pub fn decode(&mut self, packet: &[u8]) -> Result<Picture> {
        self.decode_ref(packet).cloned()
    }

    fn decode_obu(&mut self, unit: &obu::Obu<'_>) -> Result<()> {
        let kind = unit.header.obu_type;
        if unit.header.embedded_id != 0 {
            return Err(DecodeError::Unsupported("AV2 embedded layers"));
        }
        if !matches!(unit.header.extended_id, 0 | obu::GLOBAL_XLAYER_ID) {
            return Err(DecodeError::Unsupported("AV2 extended layers"));
        }
        if let Some(sequence) = &self.state.sequence
            && unit.header.temporal_id > sequence.max_temporal_layer
        {
            return Err(DecodeError::Malformed(
                "AV2 temporal layer above the sequence's",
            ));
        }
        match kind {
            other_obu::SEQUENCE_HEADER => self.sequence_header(unit.payload),
            other_obu::FILM_GRAIN_MODEL => {
                if self.state.sequence.is_none() {
                    return Err(DecodeError::MissingKey);
                }
                film_grain::parse(unit.payload, &mut self.state.grain_models)
            }
            other_obu::TEMPORAL_DELIMITER
            | other_obu::METADATA
            | other_obu::METADATA_GROUP
            | other_obu::CONTENT_INTERPRETATION
            | other_obu::PADDING => Ok(()),
            _ if obu_type::is_frame(kind) => self.frame(unit),
            // Multi-frame headers, quantizer matrices, layer and operating
            // point OBUs.
            _ => Err(DecodeError::Unsupported("AV2 OBU type outside the decoder")),
        }
    }

    /// A sequence header: a different one starts a new sequence (every
    /// slot emptied).
    fn sequence_header(&mut self, payload: &[u8]) -> Result<()> {
        let parsed = sequence::parse(payload)?;
        if self.state.sequence.as_deref() == Some(&parsed) {
            return Ok(());
        }
        let crop = parsed.crop.unwrap_or_default();
        let even = [crop.left, crop.right, crop.top, crop.bottom]
            .iter()
            .all(|offset| offset % 2 == 0);
        if !even
            || u64::from(crop.left) + u64::from(crop.right) >= u64::from(parsed.max_width)
            || u64::from(crop.top) + u64::from(crop.bottom) >= u64::from(parsed.max_height)
        {
            return Err(DecodeError::Unsupported(
                "AV2 conformance window of odd or empty size",
            ));
        }
        self.state.subset_sequence = Some(crate::sequence::parse(payload));
        self.state.headers = Default::default();
        self.state.grain_models = Default::default();
        for held in std::mem::take(&mut self.state.data).into_iter().flatten() {
            self.retired.push(held.picture);
        }
        self.state.order = OutputOrder::new(parsed.inter.order_hint_bits);
        self.state.sequence = Some(Rc::new(parsed));
        Ok(())
    }

    /// A frame OBU: a show-existing frame, or a frame to decode.
    fn frame(&mut self, unit: &obu::Obu<'_>) -> Result<()> {
        // No sequence header yet: the key frame was lost.
        let sequence = self
            .state
            .sequence
            .as_ref()
            .ok_or(DecodeError::MissingKey)?;
        let kind = unit.header.obu_type;
        let mut bits = BitReader::new(unit.payload);
        if !obu_type::is_header_only(kind) && !bits.read_bit()? {
            return Err(DecodeError::Unsupported(
                "AV2 several tile groups in a frame",
            ));
        }
        let context = Context {
            sequence,
            slots: &self.state.headers,
            obu_type: kind,
            temporal_layer: unit.header.temporal_id,
            embedded_layer: unit.header.embedded_id,
        };
        let parsed = frame_header::parse(&context, &mut bits, &mut |_, _| {})?;
        match parsed {
            Parsed::ShowExisting(shown) => self.show_existing(shown),
            Parsed::Frame(header) => {
                if obu_type::is_header_only(kind) {
                    bits.trailing_bits()?;
                    // A TIP frame is predicted whole from its TIP
                    // references; it has no tile data.
                    if header.tip.mode != TIP_FRAME_MODE {
                        return Err(DecodeError::Unsupported("AV2 bridge frames"));
                    }
                    return self.decode_frame(*header, &[]);
                }
                bits.align_zero()?;
                let tiles = bits.remaining_bytes()?;
                self.decode_frame(*header, tiles)
            }
        }
    }

    /// Output a slot's picture again; a key frame shown so also fills
    /// every slot.
    fn show_existing(&mut self, shown: ShowExisting) -> Result<()> {
        let slot = usize::from(shown.slot);
        let (Some(header), Some(data)) = (
            self.state.headers[slot].clone(),
            self.state.data[slot].clone(),
        ) else {
            return Err(DecodeError::MissingKey);
        };
        // The show-existing header's grain, not the frame's own (dav2d
        // `dav2d_parse_obus`).
        let grain = self.output_grain(shown.film_grain)?;
        self.queue_output(&data.picture, grain.as_ref(), header.header.order)?;
        if header.header.frame_type == FrameType::Key {
            for index in 0..MAX_SLOTS {
                self.state.headers[index] = Some(header.clone());
                if let Some(old) = self.state.data[index].replace(data.clone()) {
                    self.retired.push(old.picture);
                }
            }
        }
        Ok(())
    }

    /// Decode a frame's tiles, queue it if shown now, and refresh its slots.
    fn decode_frame(&mut self, header: FrameHeader, tiles: &[u8]) -> Result<()> {
        let subset_sequence = self
            .state
            .subset_sequence
            .clone()
            .ok_or(DecodeError::MissingKey)?;
        // Our encoder's frames take the subset tile decoder; every other
        // frame the general block layer.
        let Ok(params) =
            subset_sequence.and_then(|sequence| subset::tile_params(&sequence, &header))
        else {
            let (picture, mut models, extras) = self.decode_general(&header, tiles)?;
            // The tile's final models, counts scaled back, are what later
            // frames inherit (dav2d `dav2d_decode_frame_update_cdf`; one
            // tile: no averaging). A TIP frame reads no symbols and passes
            // on the models it started from as they are (dav2d `in_cdf`).
            if header.tip.mode != TIP_FRAME_MODE {
                models.restart_adaptation();
            }
            let context = FrameContext::General(Rc::from(models));
            return self.finish_frame(header, picture, Some(context), extras);
        };
        let size = (header.width as usize, header.height as usize);
        let (picture, models) = if header.is_intra() {
            tile_decoder::decode(tiles, size, params, None, None, &mut self.workspace)?
        } else {
            let count = usize::from(header.reference_count);
            let mut pictures = [None; MAX_REFERENCES];
            let mut distances = [1; MAX_REFERENCES];
            for rank in 0..count {
                let slot = usize::from(header.references[rank]);
                let (Some(held), Some(data)) = (&self.state.headers[slot], &self.state.data[slot])
                else {
                    return Err(DecodeError::MissingKey);
                };
                if (data.picture.width(), data.picture.height()) != size {
                    return Err(DecodeError::Unsupported("AV2 references of another size"));
                }
                pictures[rank] = Some(&*data.picture);
                distances[rank] = references::order_difference(header.order, held.header.order);
            }
            let inherited = match header.primary_reference {
                Some(rank) if !header.no_cross_frame_context => {
                    let slot = usize::from(header.references[usize::from(rank)]);
                    let data = self.state.data[slot]
                        .as_ref()
                        .ok_or(DecodeError::MissingKey)?;
                    match &data.context {
                        Some(FrameContext::Subset(models)) => Some(&**models),
                        Some(FrameContext::General(_)) => {
                            return Err(DecodeError::Unsupported(
                                "AV2 subset frames inheriting a general frame's CDFs",
                            ));
                        }
                        None => {
                            return Err(DecodeError::Malformed(
                                "AV2 inherited context did not adapt",
                            ));
                        }
                    }
                }
                _ => None,
            };
            tile_decoder::decode(
                tiles,
                size,
                params,
                Some(InterReference {
                    pictures: &pictures[..count],
                    distances: &distances[..count],
                    filter: subset::frame_filter(&header),
                    skip_mode: header.skip_mode,
                }),
                inherited,
                &mut self.workspace,
            )?
        };
        #[cfg(feature = "analysis")]
        if let Some(stats) = &mut self.workspace.stats {
            stats.finish();
        }
        let models = models.map(|mut models| {
            models.restart_adaptation();
            FrameContext::Subset(Rc::from(models))
        });
        self.finish_frame(header, picture, models, SlotExtras::default())
    }

    /// Queue a decoded frame for output when shown and keep it in the slots
    /// it refreshes, with its final models (`context`).
    fn finish_frame(
        &mut self,
        header: FrameHeader,
        picture: Picture,
        context: Option<FrameContext>,
        extras: SlotExtras,
    ) -> Result<()> {
        let picture = Rc::new(picture);
        let grain = self.output_grain(header.film_grain)?;
        if header.show_immediate {
            self.queue_output(&picture, grain.as_ref(), header.order)?;
        } else if self.invisible {
            self.output.push_back(Output {
                picture: Rc::clone(&picture),
                grain: grain.clone(),
            });
        }
        let refresh = header.refresh;
        let held = SlotHeader {
            header: Rc::new(header),
            frame: self.state.frames,
        };
        self.state.frames = self.state.frames.wrapping_add(1);
        let data = SlotData {
            picture,
            context,
            ccso_flags: extras.ccso_flags,
            motion_field: extras.motion_field,
            segments: extras.segments,
            grain,
        };
        for slot in 0..MAX_SLOTS {
            if refresh & (1 << slot) != 0 {
                self.state.headers[slot] = Some(held.clone());
                if let Some(old) = self.state.data[slot].replace(data.clone()) {
                    self.retired.push(old.picture);
                }
            }
        }
        Ok(())
    }

    /// Decode a frame through the general block layer: the filtered
    /// picture and the tile's final models, or the first tool not built
    /// yet.
    fn decode_general(
        &mut self,
        header: &FrameHeader,
        tiles: &[u8],
    ) -> Result<(Picture, Box<Cdfs>, SlotExtras)> {
        let sequence = self.state.sequence.clone().ok_or(DecodeError::MissingKey)?;
        let cdfs = self.initial_models(header, &sequence)?;
        let reused = self.reused_ccso_flags(header)?;
        let motion = self.reference_motion(header);
        let previous_segments = self.previous_segments(header);
        let trace = self.block_trace.0.as_mut().map(|sink| {
            let sink: BlockTrace<'_> = &mut **sink;
            sink
        });
        let reused_ccso = reused
            .each_ref()
            .map(|flags| flags.as_deref().map(Vec::as_slice));
        let references = std::array::from_fn(|rank| {
            (!header.is_intra() && rank < usize::from(header.reference_count))
                .then(|| {
                    let slot = usize::from(header.references[rank]);
                    self.state.data[slot].as_ref().map(|data| &*data.picture)
                })
                .flatten()
        });
        let masks = self.masks.get_or_init(InterMasks::new);
        let inputs = block::FrameInputs {
            reused_ccso,
            references,
            masks,
            motion: &motion,
            previous_segments: previous_segments.as_deref(),
        };
        let mut decoded = block::decode_tile(&sequence, header, cdfs, tiles, inputs, trace)?;
        let ccso_flags = header
            .ccso
            .enabled
            .then(|| Rc::new(decoded.filters.ccso_flags.clone()));
        // A frame that does not update its map keeps its reference's.
        let segments = header.segmentation.and_then(|segmentation| {
            let maps = decoded.filters.segments.as_ref()?;
            if segmentation.update_map {
                Some(Rc::new(maps.luma.clone()))
            } else {
                previous_segments.filter(|previous| previous.same_size(&maps.luma))
            }
        });
        if !self.unfiltered {
            let (dsp, ..) = self.workspace.filter_parts();
            filters::deblock::filter(
                &mut decoded.planes,
                &mut decoded.filters,
                header,
                &sequence,
                dsp,
            );
        }
        let (width, height) = (header.width as usize, header.height as usize);
        let mut picture = self.workspace.pictures.take(width, height, 0)?;
        for (index, plane) in picture.planes_mut().iter_mut().enumerate() {
            let (w, h) = (plane.width(), plane.height());
            let stride = decoded.planes.stride[index];
            let mut view = plane.view_mut();
            for (y, row) in decoded.planes.data[index]
                .chunks(stride)
                .take(h)
                .enumerate()
            {
                view.row_mut(y)[..w].copy_from_slice(&row[..w]);
            }
        }
        if !self.unfiltered {
            // Restoration and GDF read deblocked rows next to their
            // stripes.
            let restored = header.gdf.mode != 0
                || header
                    .restoration
                    .planes
                    .iter()
                    .any(|plane| plane.kind != kind::NONE);
            let deblocked = restored.then(|| picture.clone());
            let (dsp, cdef_scratch, ccso_source) = self.workspace.filter_parts();
            filters::cdef_and_ccso(
                &mut picture,
                &decoded.filters,
                (header, sequence.single_picture),
                dsp,
                (cdef_scratch, ccso_source),
            )?;
            if let Some(deblocked) = &deblocked {
                let downsample = sequence.intra.cfl_downsample_filter;
                self.restore(
                    &mut picture,
                    deblocked,
                    &decoded.filters,
                    (header, downsample),
                );
            }
        }
        let extras = SlotExtras {
            ccso_flags,
            motion_field: decoded.motion_field.map(Rc::new),
            segments,
        };
        Ok((picture, decoded.cdfs, extras))
    }

    /// Loop restoration, then GDF, of the picture after CDEF and CCSO
    /// (both read a copy of it, and `deblocked` rows next to their
    /// stripes); lossless units are put back from that copy.
    fn restore(
        &self,
        picture: &mut Picture,
        deblocked: &Picture,
        maps: &filters::FilterMaps,
        (header, downsample): (&FrameHeader, u8),
    ) {
        let source = picture.clone();
        filters::restoration::filter(picture, (&source, deblocked), maps, (header, downsample));
        if header.gdf.mode != 0 {
            let distances = self.reference_distances(header);
            let params = filters::gdf::params(header, &distances);
            filters::gdf::filter(picture, (&source, deblocked), maps, (header, params));
        }
        if header.lossless.segments != 0 {
            filters::keep_lossless(picture, &source, maps, header, filters::LOSSLESS_UNIT);
        }
    }

    /// The primary reference's segment map, where the frame predicts
    /// segment ids from it (dav2d `prev_segmap`).
    fn previous_segments(&self, header: &FrameHeader) -> Option<Segments> {
        let segmentation = header.segmentation?;
        if segmentation.update_map && !segmentation.temporal_update {
            return None;
        }
        let rank = header.primary_reference?;
        let slot = usize::from(header.references[usize::from(rank)]);
        self.state.data[slot].as_ref()?.segments.clone()
    }

    /// The models a general frame starts from (dav2d
    /// `dav2d_decode_frame_init`'s entropy setup): the defaults at its
    /// quantizer, or its primary reference's final models.
    fn initial_models(&self, header: &FrameHeader, sequence: &SequenceHeader) -> Result<Box<Cdfs>> {
        let Some(rank) = header
            .primary_reference
            .filter(|_| !header.no_cross_frame_context)
        else {
            return Ok(Box::new(Cdfs::new(header.quantizer.base)));
        };
        // dav2d `use_pri_sec_cdf`: the primary's models with the
        // secondary's weighed in at one eighth.
        let averaged = header.secondary_reference.is_some()
            && header.frame_type == FrameType::Inter
            && sequence.transform.average_cdf
            && !sequence.transform.average_cdf_type
            && header.tip.mode != TIP_FRAME_MODE;
        let mut models = self.general_models(header, rank)?;
        if averaged {
            let secondary = header
                .secondary_reference
                .ok_or(DecodeError::Malformed("AV2 secondary reference"))?;
            models.average_with(&*self.general_models(header, secondary)?);
        }
        Ok(models)
    }

    /// The final models of the frame at reference rank `rank`, which the
    /// general block layer decoded.
    fn general_models(&self, header: &FrameHeader, rank: u8) -> Result<Box<Cdfs>> {
        let slot = usize::from(header.references[usize::from(rank)]);
        let data = self.state.data[slot]
            .as_ref()
            .ok_or(DecodeError::MissingKey)?;
        match &data.context {
            Some(FrameContext::General(models)) => Ok(Box::new((**models).clone())),
            Some(FrameContext::Subset(_)) => Err(DecodeError::Unsupported(
                "AV2 general frames inheriting a subset frame's CDFs",
            )),
            None => Err(DecodeError::Malformed(
                "AV2 inherited context did not adapt",
            )),
        }
    }

    /// The CCSO flags each plane that reuses them takes from its
    /// reference (dav2d `prev_ccsomap`).
    fn reused_ccso_flags(&self, header: &FrameHeader) -> Result<[Option<CcsoFlags>; 3]> {
        let mut reused = [None, None, None];
        if !header.ccso.enabled {
            return Ok(reused);
        }
        for (plane, coded) in header.ccso.planes.iter().enumerate() {
            if !coded.block_reuse {
                continue;
            }
            let slot = usize::from(header.references[usize::from(coded.reference)]);
            let flags = self.state.data[slot]
                .as_ref()
                .and_then(|data| data.ccso_flags.clone())
                .ok_or(DecodeError::Malformed(
                    "AV2 CCSO flags reused from a frame without CCSO",
                ))?;
            reused[plane] = Some(flags);
        }
        Ok(reused)
    }

    /// Each reference rank's display-order distance from the frame (dav2d
    /// `absrefdist`).
    fn reference_distances(&self, header: &FrameHeader) -> Vec<u32> {
        (0..usize::from(header.reference_count))
            .filter_map(|rank| {
                let slot = usize::from(header.references[rank]);
                let held = self.state.headers[slot].as_ref()?;
                Some(references::order_difference(header.order, held.header.order).unsigned_abs())
            })
            .collect()
    }

    /// What each reference rank of an inter frame brings to its motion
    /// prediction: display orders and the temporal field, when it kept
    /// one of the frame's size (dav2d `dav2d_submit_frame`'s `ref_mvs`,
    /// `refrefpoc`, `refcnt`).
    fn reference_motion(&self, header: &FrameHeader) -> [ReferenceMotion; MAX_REFERENCES] {
        let mut motion: [ReferenceMotion; MAX_REFERENCES] = Default::default();
        if header.is_intra() {
            return motion;
        }
        let units = |size: u32| size.div_ceil(8);
        for (rank, reference) in motion.iter_mut().enumerate() {
            let slot = usize::from(header.references[rank]);
            let (Some(held), Some(data)) = (&self.state.headers[slot], &self.state.data[slot])
            else {
                continue;
            };
            let held = &held.header;
            reference.order = held.order;
            reference.reference_orders = held.reference_orders;
            reference.reference_count = held.reference_count;
            let same_size = (units(held.width), units(held.height))
                == (units(header.width), units(header.height));
            if header.use_motion_fields && same_size {
                reference.field.clone_from(&data.motion_field);
            }
        }
        motion
    }

    /// Diagnosis: output general-path frames before their loop filters
    /// (dav2d `--inloopfilters none`), which are then not applied.
    pub fn skip_loop_filters(&mut self) {
        self.unfiltered = true;
    }

    /// Diagnosis: also output each frame not shown when decoded, when it
    /// is decoded (dav2d `--outputinvisible 1`), to hold hidden frames
    /// against dav2d.
    pub fn output_invisible_frames(&mut self) {
        self.invisible = true;
    }

    /// Send the general block layer's trace (dav2d's `DEBUG_BLOCK_INFO`
    /// lines, `tools/dev/av2_foreign/block_trace.py`) to `sink`.
    pub fn trace_blocks(&mut self, sink: BlockSink) {
        self.block_trace = TraceSink(Some(sink));
    }

    /// The grain a header's film grain fields lay over its output.
    fn output_grain(&self, fields: Option<frame_header::FilmGrain>) -> Result<Option<OutputGrain>> {
        match fields {
            Some(fields) => OutputGrain::new(&self.state.grain_models, fields.id, fields.seed),
            None => Ok(None),
        }
    }

    /// Queue a picture of display order `order` for output with its
    /// `grain`, and the hidden frames it releases.
    /// # Errors
    /// Rejects a repeat of the last output's display order.
    fn queue_output(
        &mut self,
        picture: &Rc<Picture>,
        grain: Option<&OutputGrain>,
        order: u32,
    ) -> Result<()> {
        let released = self
            .state
            .order
            .show(&self.state.headers, order)
            .ok_or(DecodeError::Malformed("AV2 output order repeats a frame"))?;
        for release in released {
            match release {
                Release::Current => self.output.push_back(Output {
                    picture: Rc::clone(picture),
                    grain: grain.cloned(),
                }),
                Release::Slot(slot) => self.queue_slot(slot),
            }
        }
        Ok(())
    }

    /// Queue the picture of `slot`, with its frame's grain.
    fn queue_slot(&mut self, slot: usize) {
        if let Some(data) = &self.state.data[slot] {
            self.output.push_back(Output {
                picture: Rc::clone(&data.picture),
                grain: data.grain.clone(),
            });
        }
    }
}

/// The temporal layer of one temporal unit: its frame OBU's `temporal_id`
/// (key frames are layer 0). Dropping the units above a layer keeps a
/// conforming stream: frames reference only their own and lower layers.
/// # Errors
/// Rejects malformed OBUs and units without a frame.
pub fn temporal_unit_layer(temporal_unit: &[u8]) -> Result<u8> {
    let mut remaining = temporal_unit;
    while !remaining.is_empty() {
        let (unit, rest) = obu::parse(remaining)?;
        if obu_type::is_frame(unit.header.obu_type) {
            return Ok(unit.header.temporal_id);
        }
        remaining = rest;
    }
    Err(DecodeError::Malformed("AV2 temporal unit without a frame"))
}
