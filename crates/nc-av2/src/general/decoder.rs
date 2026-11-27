//! The AV2 decoder's top level: OBU dispatch, the reference slots and the
//! output order (hidden frames, show-existing frames, frames shown later
//! in display order). Film grain is laid over a picture as it is output
//! ([`super::film_grain`]), never over the reference it stays in.
//!
//! Each temporal unit is decoded frame OBU by frame OBU; every picture to
//! show goes to an output queue in display order, read with
//! [`Core::next_output`] (and, at the end of the stream, after
//! [`Core::flush`]), in the order of [`super::output`].
//!
//! The decoding process ([`Core`]) is generic over the sample type
//! ([`Av2Sample`]); the public decoder ([`super::instance::Decoder`]) holds
//! the instance of the current sequence's depth. Our own encoder's 8-bit
//! frames take the subset tile decoder ([`super::subset`]); every other
//! frame the general block layer.
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
    self, Context, FrameHeader, FrameType, MAX_SLOTS, Parsed, ShowExisting, SlotHeader, obu_type,
};
use super::multi_frame_header::{self, MultiFrameHeaders};
use super::output::{OutputOrder, Position, Release};
use super::references::{self, MAX_REFERENCES};
use super::sample::Av2Sample;
use super::sequence::{self, SequenceHeader};
use super::subset;
use super::tiles;
use crate::cdf_models::TileCdfs;
use crate::obu;
use crate::tile_decoder::{self, InterReference};
use nc_common::decode::{BitReader, DecodeError, Result};
use nc_common::{BitDepth, Picture, ShownPicture};
use nc_dsp::av2_dec::wedge::InterMasks;

/// OBU types other than frames that the decoder reads or skips (AV2 §6.2).
pub(super) mod other_obu {
    pub const SEQUENCE_HEADER: u8 = 1;
    pub const TEMPORAL_DELIMITER: u8 = 2;
    pub const MULTI_FRAME_HEADER: u8 = 3;
    pub const BUFFER_REMOVAL_TIMING: u8 = 15;
    pub const LAYER_CONFIGURATION_RECORD: u8 = 16;
    pub const ATLAS_SEGMENT: u8 = 17;
    pub const OPERATING_POINT_SET: u8 = 18;
    pub const METADATA: u8 = 8;
    pub const METADATA_GROUP: u8 = 9;
    pub const FILM_GRAIN_MODEL: u8 = 23;
    pub const CONTENT_INTERPRETATION: u8 = 24;
    pub const PADDING: u8 = 25;
}

/// Frames are decoded in whole blocks of this many samples a side.
const CODED_BLOCK: usize = 8;

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
struct SlotData<T: Av2Sample> {
    /// The decoded picture at its coded size (dav2d's 8-aligned `bw * 4` x
    /// `bh * 4`): later frames' prediction reads the samples past the
    /// frame's edge up to it, as dav2d and AVM do.
    picture: Rc<Picture<T>>,
    /// The frame's size (`frame_width`, `frame_height`): what is output.
    size: (usize, usize),
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

/// A frame whose tile groups have not all arrived: its header and the
/// data of its tiles so far.
#[derive(Debug)]
struct PendingFrame {
    header: Box<FrameHeader>,
    /// Bits of the frame header, which a later tile group may repeat.
    header_bits: usize,
    tiles: Vec<Vec<u8>>,
}

/// One picture in the output queue.
#[derive(Debug)]
struct Output<T: Av2Sample> {
    picture: Rc<Picture<T>>,
    /// The frame's size inside the picture's coded size.
    size: (usize, usize),
    /// The grain laid over it when it is shown.
    grain: Option<OutputGrain>,
}

/// What a temporal unit changes, restored when it fails: the sequence,
/// the reference slots, the film grain models and the output order.
#[derive(Debug, Clone, Default)]
struct State<T: Av2Sample> {
    /// Shared with the saved state of a unit (cloning it allocates).
    sequence: Option<Rc<SequenceHeader>>,
    /// The same sequence header as the subset block layer reads it, or why
    /// that layer cannot decode its frames.
    subset_sequence: Option<std::result::Result<crate::sequence::SequenceHeader, DecodeError>>,
    /// Headers of the frames in the slots (`ref_frame_map`).
    headers: [Option<SlotHeader>; MAX_SLOTS],
    /// Their pictures and contexts.
    data: [Option<SlotData<T>>; MAX_SLOTS],
    /// The film grain models FGM OBUs set, by id (dav2d `c->fgm`).
    grain_models: film_grain::Models,
    /// The multi-frame headers, by id (AVM `mfh_params`).
    multi_frame_headers: MultiFrameHeaders,
    /// Frames decoded so far: the identity of the next.
    frames: u32,
    /// Which pictures go out when.
    order: OutputOrder,
}

/// The AV2 decoding process for samples of type `T`: sequence state,
/// reference slots and the output queue. A temporal unit that fails
/// leaves it as it was.
#[derive(Debug, Default)]
pub(crate) struct Core<T: Av2Sample> {
    state: State<T>,
    /// Pictures to show, in display order.
    output: VecDeque<Output<T>>,
    /// The picture [`Core::shown`] shows, and the frame's size in it.
    current: Option<Rc<Picture<T>>>,
    current_size: (usize, usize),
    /// [`Core::decode_ref`]'s copy of the shown picture's window when it
    /// crops anything.
    cropped: Option<Picture<T>>,
    /// Pictures the slots let go of during the current temporal unit,
    /// given to the workspace for reuse once the unit's saved state is gone
    /// (until then they are shared, and would be dropped instead).
    retired: Vec<Rc<Picture<T>>>,
    /// Buffers reused from picture to picture.
    workspace: tile_decoder::Workspace<T>,
    /// Where the general block layer's trace goes, if anywhere.
    block_trace: TraceSink,
    /// Diagnosis: general-path frames skip their loop filters (as dav2d's
    /// `--inloopfilters none`), to hold the reconstruction against it.
    unfiltered: bool,
    /// Diagnosis: frames not shown when decoded are output then too (as
    /// dav2d's `--outputinvisible 1`).
    invisible: bool,
    /// Cached default CDF tables per AVM quantizer category (`0..4`).
    default_cdfs: [std::cell::OnceCell<Rc<Cdfs>>; 4],
    /// The frame whose later tile groups are still to come in this
    /// temporal unit.
    pending: Option<PendingFrame>,
    /// A frame header came in the current temporal unit (dav2d
    /// `seen_layer_in_tu`).
    frame_in_unit: bool,
}

/// A block-trace sink ([`Core::trace_blocks`]).
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

impl<T: Av2Sample> Core<T> {
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

    /// A core of samples `U` for the next sequence, with this one's
    /// settings: loop filter and output diagnosis, the block trace and the
    /// analysis (moved).
    pub fn successor<U: Av2Sample>(&mut self) -> Core<U> {
        Core::<U> {
            workspace: self.workspace.successor(),
            block_trace: std::mem::take(&mut self.block_trace),
            unfiltered: self.unfiltered,
            invisible: self.invisible,
            ..Core::default()
        }
    }

    /// Whether pictures are queued for output.
    pub fn has_output(&self) -> bool {
        !self.output.is_empty()
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
        // A frame's tile groups all belong to its temporal unit.
        if self.pending.take().is_some() && result.is_ok() {
            result = Err(DecodeError::Malformed("AV2 frame without all its tiles"));
        }
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
        self.frame_in_unit = false;
        let mut remaining = unit;
        while !remaining.is_empty() {
            let (obu, rest) = obu::parse(remaining)?;
            remaining = rest;
            self.decode_obu(&obu)?;
        }
        Ok(())
    }

    /// Show the next queued picture ([`Core::shown`]); `false` when none
    /// is queued.
    pub fn next_output(&mut self) -> bool {
        let Some(next) = self.output.pop_front() else {
            return false;
        };
        let picture = match &next.grain {
            Some(grain) => Rc::new(self.grained(&next.picture, next.size, grain)),
            None => next.picture,
        };
        self.current_size = next.size;
        if let Some(previous) = self.current.replace(picture) {
            self.workspace.release(previous);
        }
        true
    }

    /// The frame of `size` in `picture` with `grain` laid over it, in a
    /// picture of the workspace (of the frame's size rounded up to even:
    /// grain blocks start at the frame's origin, so a sample's grain does
    /// not depend on the frame's extent).
    fn grained(
        &mut self,
        picture: &Picture<T>,
        size: (usize, usize),
        grain: &OutputGrain,
    ) -> Picture<T> {
        let even = (size.0.next_multiple_of(2), size.1.next_multiple_of(2));
        let take = |pictures: &mut nc_common::PictureRecycler<T>| {
            pictures
                .take(even.0, even.1, 0)
                .expect("the geometry of a frame already decoded")
        };
        let mut frame = take(&mut self.workspace.pictures);
        ShownPicture::window(picture, (0, 0), even)
            .expect("the coded size covers the frame")
            .copy_to(&mut frame);
        let mut output = take(&mut self.workspace.pictures);
        let depth = self.depth();
        let (dsp, ..) = self.workspace.filter_parts();
        grain.apply(dsp, &frame, &mut output, depth);
        self.workspace.pictures.give(frame);
        output
    }

    /// The current sequence's sample depth ([`BitDepth::of`] this instance:
    /// the constant 8 bits in the `u8` one).
    pub fn depth(&self) -> BitDepth {
        self.state
            .sequence
            .as_ref()
            .map_or(BitDepth::EIGHT, |sequence| sequence.depth())
            .of::<T>()
    }

    /// A closed-loop key frame that starts a temporal unit: output every
    /// hidden frame still due, restart the output order, and empty the
    /// slots it does not refresh (with several embedded layers it refreshes
    /// only some; AVM `read_uncompressed_header` and `flush_remaining_frames`,
    /// dav2d after 237229e `dav2d_parse_obus`).
    fn restart_references(&mut self, refresh: u16) {
        self.flush();
        self.state.order.restart();
        for slot in 0..MAX_SLOTS {
            if refresh & (1 << slot) == 0 {
                self.state.headers[slot] = None;
                if let Some(held) = self.state.data[slot].take() {
                    self.retired.push(held.picture);
                }
            }
        }
    }

    /// Whether the active sequence is monochrome (pictures carry mid-grey
    /// chroma; avmdec writes their luma alone).
    pub fn is_monochrome(&self) -> bool {
        self.state
            .sequence
            .as_ref()
            .is_some_and(|sequence| sequence.monochrome)
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
    pub fn shown(&self) -> Option<ShownPicture<'_, T>> {
        let picture = self.current.as_deref()?;
        let sequence = self.state.sequence.as_ref()?;
        let crop = sequence.crop.unwrap_or_default();
        let (frame_width, frame_height) = self.current_size;
        let width = frame_width.checked_sub((crop.left + crop.right) as usize)?;
        let height = frame_height.checked_sub((crop.top + crop.bottom) as usize)?;
        ShownPicture::window(
            picture,
            (crop.left as usize, crop.top as usize),
            (width, height),
        )
    }

    /// Decode a temporal unit that shows one picture (as our encoder's
    /// do) and show it.
    /// # Errors
    /// As [`Core::decode_unit`], and units that show nothing.
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
    /// As [`Core::decode_shown`].
    pub fn decode_ref(&mut self, packet: &[u8]) -> Result<&Picture<T>> {
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

    fn decode_obu(&mut self, unit: &obu::Obu<'_>) -> Result<()> {
        let kind = unit.header.obu_type;
        // Extended layers are separate streams multiplexed together (AVM
        // `xlayer_id` contexts); embedded layers decode like any frame.
        if !matches!(unit.header.extended_id, 0 | obu::GLOBAL_XLAYER_ID) {
            return Err(DecodeError::Unsupported("AV2 extended layers"));
        }
        if let Some(sequence) = &self.state.sequence
            && (unit.header.temporal_id > sequence.max_temporal_layer
                || unit.header.embedded_id > sequence.max_embedded_layer)
        {
            return Err(DecodeError::Malformed("AV2 layer above the sequence's"));
        }
        match kind {
            other_obu::SEQUENCE_HEADER => self.sequence_header(unit.payload),
            other_obu::FILM_GRAIN_MODEL => {
                let sequence = self
                    .state
                    .sequence
                    .as_ref()
                    .ok_or(DecodeError::MissingKey)?;
                film_grain::parse(
                    unit.payload,
                    sequence.monochrome,
                    &mut self.state.grain_models,
                )
            }
            other_obu::MULTI_FRAME_HEADER => {
                multi_frame_header::parse(unit.payload, &mut self.state.multi_frame_headers)
            }
            other_obu::TEMPORAL_DELIMITER
            | other_obu::METADATA
            | other_obu::METADATA_GROUP
            | other_obu::CONTENT_INTERPRETATION
            | other_obu::PADDING
            // Layer and operating point descriptions and timing: what a
            // decoder would select or schedule; decoding every layer, as
            // avmdec and dav2d do by default, needs none of them.
            | other_obu::BUFFER_REMOVAL_TIMING
            | other_obu::LAYER_CONFIGURATION_RECORD
            | other_obu::ATLAS_SEGMENT
            | other_obu::OPERATING_POINT_SET => Ok(()),
            _ if obu_type::is_frame(kind) => self.frame(unit),
            // Quantizer matrices and multi-stream decoder operation.
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
        self.state.order =
            OutputOrder::new(parsed.inter.order_hint_bits, parsed.max_embedded_layer);
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
        // `is_first_tile_group`: later tile groups continue the pending
        // frame (dav2d `dav2d_parse_obus`).
        if !obu_type::is_header_only(kind) && !bits.read_bit()? {
            return self.later_tile_group(&mut bits);
        }
        // A new frame header drops a frame still short of tiles, as dav2d.
        self.pending = None;
        let first_in_unit = !self.frame_in_unit;
        self.frame_in_unit = true;
        let header_start = bits.remaining_bits();
        let context = Context {
            sequence,
            slots: &self.state.headers,
            obu_type: kind,
            temporal_layer: unit.header.temporal_id,
            embedded_layer: unit.header.embedded_id,
            multi_frame_headers: &self.state.multi_frame_headers,
        };
        let parsed = frame_header::parse(&context, &mut bits, &mut |_, _| {})?;
        let header_bits = header_start - bits.remaining_bits();
        match parsed {
            Parsed::ShowExisting(shown) => self.show_existing(shown),
            Parsed::Frame(header) => {
                if kind == obu_type::CLOSED_LOOP_KEY && first_in_unit {
                    self.restart_references(header.refresh);
                }
                if obu_type::is_header_only(kind) {
                    bits.trailing_bits()?;
                    // TIP and bridge frames are predicted whole from their
                    // references; they have no tile data.
                    return self.decode_frame(*header, &[]);
                }
                let (first, last) = tile_group_range(&header, &mut bits)?;
                if first != 0 {
                    return Err(DecodeError::Malformed("AV2 tile groups out of order"));
                }
                bits.align_zero()?;
                let tiles = split_tiles(&header, (first, last), bits.remaining_bytes()?)?;
                if last + 1 == tile_count(&header) {
                    return self.decode_frame(*header, &tiles);
                }
                self.pending = Some(PendingFrame {
                    header,
                    header_bits,
                    tiles: tiles.into_iter().map(<[u8]>::to_vec).collect(),
                });
                Ok(())
            }
        }
    }

    /// A tile group after a frame's first: its tiles join the pending
    /// frame, which is decoded once it has them all.
    fn later_tile_group(&mut self, bits: &mut BitReader<'_>) -> Result<()> {
        let pending = self.pending.as_mut().ok_or(DecodeError::Malformed(
            "AV2 tile group without a frame header",
        ))?;
        // A copy of the frame header may come again.
        if bits.read_bit()? {
            bits.skip_bits(pending.header_bits)?;
        }
        let (first, last) = tile_group_range(&pending.header, bits)?;
        if first != pending.tiles.len() {
            return Err(DecodeError::Malformed("AV2 tile groups out of order"));
        }
        bits.align_zero()?;
        let tiles = split_tiles(&pending.header, (first, last), bits.remaining_bytes()?)?;
        pending.tiles.extend(tiles.into_iter().map(<[u8]>::to_vec));
        if pending.tiles.len() < tile_count(&pending.header) {
            return Ok(());
        }
        let PendingFrame { header, tiles, .. } = self.pending.take().expect("a pending frame");
        let tiles: Vec<&[u8]> = tiles.iter().map(Vec::as_slice).collect();
        self.decode_frame(*header, &tiles)
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
        self.queue_output(
            (&data.picture, data.size),
            grain.as_ref(),
            Position::of(&header.header),
        )?;
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

    /// Decode a frame's tiles (each one's data, in raster order), queue it
    /// if shown now, and refresh its slots.
    fn decode_frame(&mut self, header: FrameHeader, tiles: &[&[u8]]) -> Result<()> {
        let sequence = self
            .state
            .sequence
            .as_ref()
            .ok_or(DecodeError::MissingKey)?;
        let crop = sequence.crop.unwrap_or_default();
        if header.width > sequence.max_width
            || header.height > sequence.max_height
            || u64::from(crop.left) + u64::from(crop.right) >= u64::from(header.width)
            || u64::from(crop.top) + u64::from(crop.bottom) >= u64::from(header.height)
        {
            return Err(DecodeError::Malformed(
                "AV2 frame size outside sequence or conformance window",
            ));
        }
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
            // The frame's final models (`block::decode_tiles`), counts
            // scaled back, are what later frames inherit (dav2d
            // `dav2d_decode_frame_update_cdf`). TIP and bridge frames read no
            // symbols and pass on the models they started from as they are
            // (dav2d `in_cdf`, AVM `set_primary_ref_frame_and_ctx`).
            if header.codes_symbols() {
                Rc::make_mut(&mut models).restart_adaptation();
            }
            let context = FrameContext::General(models);
            return self.finish_frame(header, picture, Some(context), extras);
        };
        let (picture, models) = self.decode_subset(&header, tiles, params)?;
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

    /// Decode one of our encoder's frames through the subset tile decoder
    /// (8-bit only: deeper sequences never pass its parameter check).
    fn decode_subset(
        &mut self,
        header: &FrameHeader,
        tiles: &[&[u8]],
        params: tile_decoder::TileParams,
    ) -> Result<(Picture<T>, Option<Box<TileCdfs>>)> {
        const DEEP: DecodeError = DecodeError::Unsupported("AV2 subset frame of more than 8 bits");
        let size = (header.width as usize, header.height as usize);
        // The subset decoder takes frames of one tile only.
        let tiles = tiles.first().copied().unwrap_or_default();
        let workspace = self.workspace.as_eight().ok_or(DEEP)?;
        let (picture, models) = if header.is_intra() {
            tile_decoder::decode(tiles, size, params, None, None, workspace)?
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
                pictures[rank] = Some(T::as_eight(&data.picture).ok_or(DEEP)?);
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
                    filter: subset::frame_filter(header),
                    skip_mode: header.skip_mode,
                }),
                inherited,
                workspace,
            )?
        };
        Ok((T::from_eight(picture).ok_or(DEEP)?, models))
    }

    /// Queue a decoded frame for output when shown and keep it in the slots
    /// it refreshes, with its final models (`context`).
    fn finish_frame(
        &mut self,
        header: FrameHeader,
        picture: Picture<T>,
        context: Option<FrameContext>,
        extras: SlotExtras,
    ) -> Result<()> {
        let picture = Rc::new(picture);
        let size = (header.width as usize, header.height as usize);
        let grain = self.output_grain(header.film_grain)?;
        if header.show_immediate {
            self.queue_output((&picture, size), grain.as_ref(), Position::of(&header))?;
        } else if self.invisible {
            self.output.push_back(Output {
                picture: Rc::clone(&picture),
                size,
                grain: grain.clone(),
            });
        }
        let refresh = header.refresh;
        // A key or switch frame refreshing several slots is held in the
        // first only and empties the others (AVM `update_frame_buffers`,
        // `av2_frame_clears_multiple_inserted_in_one`): a later frame that
        // overwrites that slot leaves it nowhere.
        let filled = if matches!(header.frame_type, FrameType::Key | FrameType::Switch)
            && refresh.count_ones() > 1
        {
            refresh & refresh.wrapping_neg()
        } else {
            refresh
        };
        let held = SlotHeader {
            header: Rc::new(header),
            frame: self.state.frames,
        };
        self.state.frames = self.state.frames.wrapping_add(1);
        let data = SlotData {
            picture,
            size,
            context,
            ccso_flags: extras.ccso_flags,
            motion_field: extras.motion_field,
            segments: extras.segments,
            grain,
        };
        for slot in 0..MAX_SLOTS {
            if refresh & (1 << slot) == 0 {
                continue;
            }
            let (header, data) = if filled & (1 << slot) != 0 {
                (Some(held.clone()), Some(data.clone()))
            } else {
                (None, None)
            };
            self.state.headers[slot] = header;
            if let Some(old) = std::mem::replace(&mut self.state.data[slot], data) {
                self.retired.push(old.picture);
            }
        }
        Ok(())
    }

    /// The scale of each reference rank of another size than the frame
    /// (dav2d `dav2d_submit_frame`'s `svc`), from the sizes of the frames
    /// in its slots.
    fn reference_scales(
        &self,
        header: &FrameHeader,
    ) -> Result<[Option<block::frame_refs::ReferenceScale>; MAX_REFERENCES]> {
        let mut scales = [None; MAX_REFERENCES];
        if header.is_intra() {
            return Ok(scales);
        }
        for (rank, scale) in scales
            .iter_mut()
            .enumerate()
            .take(usize::from(header.reference_count))
        {
            let slot = usize::from(header.references[rank]);
            let held = self.state.headers[slot]
                .as_ref()
                .ok_or(DecodeError::MissingKey)?;
            *scale = block::frame_refs::ReferenceScale::new(
                (held.header.width, held.header.height),
                (header.width, header.height),
            )?;
        }
        Ok(scales)
    }

    /// Decode a frame through the general block layer: the filtered
    /// picture and the tile's final models, or the first tool not built
    /// yet.
    fn decode_general(
        &mut self,
        header: &FrameHeader,
        tiles: &[&[u8]],
    ) -> Result<(Picture<T>, Rc<Cdfs>, SlotExtras)> {
        let sequence = self.state.sequence.clone().ok_or(DecodeError::MissingKey)?;
        // The coded size: the frame rounded up to whole 8x8 blocks (dav2d
        // `bw * 4`, `bh * 4`). It is decoded, filtered and kept whole.
        let (coded_w, coded_h) = coded_size(header);
        let scales = self.reference_scales(header)?;
        let filtered = self.workspace.pictures.take(coded_w, coded_h, 0)?;
        let cdfs = self.initial_models(header, &sequence)?;
        let reused = self.reused_ccso_flags(header)?;
        let motion = self.reference_motion(header);
        let previous_segments = self.previous_segments(header);
        let dsp = self.workspace.filter_parts().0;
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
        let masks = InterMasks::shared();
        let inputs = block::FrameInputs {
            reused_ccso,
            references,
            masks,
            dsp,
            motion: &motion,
            previous_segments: previous_segments.as_deref(),
            scales,
        };
        let mut decoded = block::decode_tiles(&sequence, header, cdfs, tiles, inputs, trace)?;
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
        let picture = self.filter_decoded_planes(&mut decoded, filtered, (header, &sequence))?;
        let extras = SlotExtras {
            ccso_flags,
            motion_field: decoded.motion_field.map(Rc::new),
            segments,
        };
        Ok((picture, decoded.cdfs, extras))
    }

    /// Run general-path loop filters on the 8x8-aligned coded surface
    /// (`f->bw * 4` x `f->bh * 4` in dav2d) and return it whole: later
    /// frames predict from it, the output crops it to the frame.
    fn filter_decoded_planes(
        &mut self,
        decoded: &mut block::DecodedTile<T>,
        mut filtered: Picture<T>,
        (header, sequence): (&FrameHeader, &SequenceHeader),
    ) -> Result<Picture<T>> {
        if !self.unfiltered {
            let (dsp, ..) = self.workspace.filter_parts();
            filters::deblock::filter(
                &mut decoded.planes,
                &mut decoded.filters,
                header,
                sequence,
                dsp,
            );
        }
        copy_tile_planes(&decoded.planes, &mut filtered, sequence);
        if self.unfiltered {
            return Ok(filtered);
        }
        // Restoration and GDF read deblocked rows next to their stripes.
        let restored = header.gdf.mode != 0
            || header
                .restoration
                .planes
                .iter()
                .any(|plane| plane.kind != kind::NONE);
        let deblocked = restored.then(|| filtered.clone());
        let depth = sequence.depth().of::<T>();
        let (dsp, cdef_scratch, ccso_source) = self.workspace.filter_parts();
        filters::cdef_and_ccso(
            &mut filtered,
            &decoded.filters,
            (
                header,
                sequence.single_picture,
                depth,
                sequence.filters.disable_across_tiles,
            ),
            dsp,
            (cdef_scratch, ccso_source),
        )?;
        if let Some(deblocked) = &deblocked {
            self.restore(
                &mut filtered,
                deblocked,
                &decoded.filters,
                (header, sequence),
            );
        }
        Ok(filtered)
    }

    /// Loop restoration, then GDF, of the picture after CDEF and CCSO
    /// (both read a copy of it, and `deblocked` rows next to their
    /// stripes); lossless units are put back from that copy.
    fn restore(
        &self,
        picture: &mut Picture<T>,
        deblocked: &Picture<T>,
        maps: &filters::FilterMaps,
        (header, sequence): (&FrameHeader, &SequenceHeader),
    ) {
        let depth = sequence.depth().of::<T>();
        let source = picture.clone();
        let frame = (header, sequence, depth);
        filters::restoration::filter(picture, (&source, deblocked), maps, frame);
        if header.gdf.mode != 0 {
            let distances = self.reference_distances(header);
            let params = filters::gdf::params(header, &distances);
            filters::gdf::filter(
                self.workspace.dsp(),
                picture,
                (&source, deblocked),
                maps,
                (header, params, depth, sequence.filters.disable_across_tiles),
            );
        }
        if header.lossless.segments != 0 {
            filters::keep_lossless(
                picture,
                &source,
                (maps, (0, 0)),
                header,
                filters::LOSSLESS_UNIT,
            );
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
    fn initial_models(&self, header: &FrameHeader, sequence: &SequenceHeader) -> Result<Rc<Cdfs>> {
        let Some(rank) = header
            .primary_reference
            .filter(|_| !header.no_cross_frame_context)
        else {
            let qindex = header.quantizer.base;
            let cat = match qindex {
                0..=90 => 0,
                91..=140 => 1,
                141..=190 => 2,
                _ => 3,
            };
            let defaults = self.default_cdfs[cat].get_or_init(|| Rc::new(Cdfs::new(qindex)));
            return Ok(Rc::clone(defaults));
        };
        // dav2d `use_pri_sec_cdf`: the primary's models with the
        // secondary's weighed in at one eighth.
        let averaged = header.secondary_reference.is_some()
            && header.frame_type == FrameType::Inter
            && sequence.transform.average_cdf
            && !sequence.transform.average_cdf_type
            && header.codes_symbols();
        let mut models = Rc::clone(self.general_models_rc(header, rank)?);
        if averaged {
            let secondary = header
                .secondary_reference
                .ok_or(DecodeError::Malformed("AV2 secondary reference"))?;
            let secondary_cdfs = &**self.general_models_rc(header, secondary)?;
            Rc::make_mut(&mut models).average_with(secondary_cdfs);
        }
        Ok(models)
    }

    /// The final models of the frame at reference rank `rank`, which the
    /// general block layer decoded.
    fn general_models_rc(&self, header: &FrameHeader, rank: u8) -> Result<&Rc<Cdfs>> {
        let slot = usize::from(header.references[usize::from(rank)]);
        let data = self.state.data[slot]
            .as_ref()
            .ok_or(DecodeError::MissingKey)?;
        match &data.context {
            Some(FrameContext::General(models)) => Ok(models),
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

    /// Queue a picture at output `position` with its `grain`, and the
    /// hidden frames it releases.
    /// # Errors
    /// Rejects a repeat of the last output's position.
    fn queue_output(
        &mut self,
        (picture, size): (&Rc<Picture<T>>, (usize, usize)),
        grain: Option<&OutputGrain>,
        position: Position,
    ) -> Result<()> {
        let released = self
            .state
            .order
            .show(&self.state.headers, position)
            .ok_or(DecodeError::Malformed("AV2 output order repeats a frame"))?;
        for release in released {
            match release {
                Release::Current => self.output.push_back(Output {
                    picture: Rc::clone(picture),
                    size,
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
                size: data.size,
                grain: data.grain.clone(),
            });
        }
    }
}

/// A frame's coded size: its size rounded up to whole 8x8 blocks (dav2d
/// `bw * 4`, `bh * 4`).
fn coded_size(header: &FrameHeader) -> (usize, usize) {
    (
        (header.width as usize).next_multiple_of(CODED_BLOCK),
        (header.height as usize).next_multiple_of(CODED_BLOCK),
    )
}

/// The tiles of a frame.
fn tile_count(header: &FrameHeader) -> usize {
    header.tiles.columns() * header.tiles.rows()
}

/// The first and last tile of a tile group (`tile_start_and_end_present_flag`
/// and its fields; dav2d `parse_tile_hdr`): every tile unless signalled.
fn tile_group_range(header: &FrameHeader, bits: &mut BitReader<'_>) -> Result<(usize, usize)> {
    let count = tile_count(header);
    if count == 1 || !bits.read_bit()? {
        return Ok((0, count - 1));
    }
    let index_bits = tiles::log2(header.tiles.columns()) + tiles::log2(header.tiles.rows());
    let first = bits.read_bits(index_bits)? as usize;
    let last = bits.read_bits(index_bits)? as usize;
    if first > last || last >= count {
        return Err(DecodeError::Malformed("AV2 tile group range"));
    }
    Ok((first, last))
}

/// Split a tile group's data into its tiles `first..=last`: each but the
/// last is prefixed by its size less one in `tile_size_bytes` bytes,
/// little-endian (dav2d `dav2d_decode_frame_init_cdf`).
fn split_tiles<'a>(
    header: &FrameHeader,
    (first, last): (usize, usize),
    mut data: &'a [u8],
) -> Result<Vec<&'a [u8]>> {
    let truncated = DecodeError::Malformed("AV2 tile size beyond its tile group");
    let size_bytes = usize::from(header.tile_size_bytes);
    let mut tiles = Vec::with_capacity(last - first + 1);
    for _ in first..last {
        let (size, rest) = data.split_at_checked(size_bytes).ok_or(truncated.clone())?;
        let size = size
            .iter()
            .rev()
            .fold(0_usize, |value, &byte| value << 8 | usize::from(byte))
            + 1;
        let (tile, rest) = rest.split_at_checked(size).ok_or(truncated.clone())?;
        tiles.push(tile);
        data = rest;
    }
    tiles.push(data);
    Ok(tiles)
}

/// Copy reconstructed superblock-padded tile planes into `picture`. A
/// monochrome frame reconstructs no chroma: its picture's chroma planes
/// are mid-grey, as AV1's output process makes them (AV1 §7.18), so every
/// consumer of the picture sees a grey image (avmdec writes luma alone).
fn copy_tile_planes<T: Av2Sample>(
    planes: &block::pixels::Planes<T>,
    picture: &mut Picture<T>,
    sequence: &SequenceHeader,
) {
    let mid = T::from_i32(sequence.depth().mid());
    for (index, plane) in picture.planes_mut().iter_mut().enumerate() {
        if index >= sequence.planes() {
            plane.view_mut().fill(mid);
            continue;
        }
        let (w, h) = (plane.width(), plane.height());
        let stride = planes.stride[index];
        let mut view = plane.view_mut();
        for (y, row) in planes.data[index].chunks(stride).take(h).enumerate() {
            view.row_mut(y)[..w].copy_from_slice(&row[..w]);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression for fuzz crash `db26671c5964ffcd`: a general-path AV2 frame
    /// whose even dimensions are not a multiple of 8 (`width % 8 != 0` or
    /// `height % 8 != 0`) must run CDEF/CCSO/restoration on the 8x8-aligned
    /// coded surface and crop back to `(width, height)` without panicking.
    #[test]
    fn non_multiple_of_eight_general_frame_filters_without_panicking() {
        let mut unit = vec![
            0x01, 0x08, // OBU_TEMPORAL_DELIMITER
            0x14, 0x04, 0x81, 0xf5, 0x6b, 0x00, 0xbf, 0xfe, 0x02, 0x00, 0x00, 0x76, 0x00, 0x00,
            0x00, 0x01, 0x48, 0xbe, 0xfb, 0xd0, 0x82, // OBU_SEQUENCE_HEADER
            0x40, // OBU_CLOSED_LOOP_KEY_FRAME (64-byte payload)
            0x10, 0xf0, 0x01, 0x01, 0xf0, 0xf0, 0x12, 0x28, 0xd6, 0xa0, 0xd1, 0xde, 0xa3, 0xa8,
            0x3d, 0x18, 0x9d, 0x23, 0x1a, 0x28, 0x65, 0xf6, 0xf9, 0x70, 0xfb, 0x12, 0x57, 0x13,
            0x9b, 0xdd, 0x16, 0xd8, 0x3e, 0x22, 0x91, 0xff, 0x32, 0xd8, 0xa4, 0x95,
        ];
        unit.resize(23 + 1 + 64, 0);
        let mut decoder = Core::<u8>::default();
        if decoder.decode_unit(&unit).is_ok() && decoder.next_output() {
            assert!(decoder.shown().is_some());
        }
    }
}
