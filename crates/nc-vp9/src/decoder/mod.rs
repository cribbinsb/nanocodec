//! VP9 Profile 0 decoder (8-bit 4:2:0): every conforming stream, not only
//! the subset nanocodec emits (VP9 bitstream specification v0.6; structure
//! after libvpx `vp9_decoder.c`, `vp9_dx_iface.c` and `vp9_decodeframe.c`,
//! BSD-3-Clause, see `THIRD_PARTY.md`).
//!
//! A packet holds one frame or a superframe (several frames and an index).
//! The decoder keeps what frames share: eight reference slots, four
//! probability contexts, loop-filter deltas, segmentation features and
//! map, and the previous frame's vectors. Each frame commits its changes
//! only after it decoded completely, so a failed packet changes nothing.
//!
//! Modules: `header` (uncompressed and compressed headers), `frame` (tiles
//! and partitions), `modes` and `mvref` (mode info and candidate vectors),
//! `recon` (prediction and residuals), `loopfilter`.

mod block;
mod frame;
mod header;
mod loopfilter;
mod modes;
mod mvref;
mod recon;

use std::sync::Arc;

use nc_common::decode::{DecodeError, Result};
use nc_common::{Picture, PictureRecycler, PlaneIndex, ShownPicture};

use self::block::{MvRef, REFS_PER_FRAME};
use self::frame::{BlockScratch, FrameDecoder, FrameInputs, FrameOutput};
use self::header::{
    FRAME_CONTEXTS, FrameHeader, HeaderState, LoopFilter, SLOTS, Segmentation, TxMode,
    read_compressed,
};
use self::recon::{Reference, extend_reference};
use crate::probs::{Adaptation, FrameContext};
use crate::superframe::{
    parse_uncompressed_header_info, split_superframe, strip_superframe_to_base,
};
use crate::types::MI_SIZE;

/// `reset_frame_context` values (VP9 §7.2): reset the frame's context, or all.
const RESET_ONE_CONTEXT: u8 = 2;
const RESET_ALL_CONTEXTS: u8 = 3;
/// Bytes a `show_existing_frame` header occupies in Profile 0.
const SHOW_EXISTING_BYTES: usize = 1;

/// A decoded picture, its visible size, and its spatial layer index within the
/// packet (`0` for single-layer or base layer `S0`, `1` for enhancement `S1`).
/// The picture's planes cover the visible size rounded up to even (4:2:0
/// chroma covers `(w + 1) / 2`).
#[derive(Debug, Clone)]
pub struct DecodedFrame {
    /// The picture, `width` and `height` rounded up to even.
    pub picture: Picture,
    /// Visible luma width.
    pub width: usize,
    /// Visible luma height.
    pub height: usize,
    /// Spatial layer index within the superframe (`0` = base, `1` = enhancement).
    pub spatial_id: u8,
}

/// A reference slot's picture and visible size.
#[derive(Debug)]
struct StoredFrame {
    picture: Picture,
    width: usize,
    height: usize,
    spatial_id: u8,
}

/// What the next frame needs to know of the last decoded one.
#[derive(Debug, Clone, Copy)]
struct LastFrame {
    width: usize,
    height: usize,
    show_frame: bool,
    key: bool,
}

/// Stateful VP9 decoder.
#[derive(Debug)]
pub struct Decoder {
    slots: [Option<Arc<StoredFrame>>; SLOTS],
    contexts: Box<[FrameContext; FRAME_CONTEXTS]>,
    loop_filter: LoopFilter,
    segmentation: Segmentation,
    /// `last_frame_seg_map` (empty before the first frame).
    last_segment_map: Vec<u8>,
    /// The last decoded frame's vectors (`prev_frame->mvs`).
    previous_mvs: Vec<MvRef>,
    last: Option<LastFrame>,
    /// `cm->intra_only`: kept through key frames, as libvpx does.
    intra_only: bool,
    /// Released pictures (references no slot holds any more,
    /// non-reference reconstructions, outputs the caller returns through
    /// [`Decoder::recycle`]), reused by later frames: a fresh zeroed picture
    /// per frame cost about 4% of decoding.
    pictures: PictureRecycler,
    /// The frame the last successful [`Decoder::decode_shown`] showed.
    shown: Option<Arc<StoredFrame>>,
    /// Block working memory between frames (taken by the frame being
    /// decoded; lost, and rebuilt, only when a frame fails).
    scratch: Option<Box<BlockScratch>>,
    /// When `true`, multi-frame Annex B superframes are stripped to their
    /// base layer (`S0`) via [`strip_superframe_to_base`] before
    /// decoding.
    base_only: bool,
    /// `spatial_id` of the latest frame shown.
    last_spatial_id: Option<u8>,
    /// Maximum `(width, height)` seen across frames in the stream (including
    /// enhancement frames stripped when [`Self::base_only`] is active).
    max_size: Option<(usize, usize)>,
    dsp: nc_dsp::Dsp,
}

impl Default for Decoder {
    fn default() -> Self {
        Self {
            slots: Default::default(),
            contexts: Box::new(std::array::from_fn(|_| FrameContext::default())),
            loop_filter: LoopFilter::default(),
            segmentation: Segmentation::default(),
            last_segment_map: Vec::new(),
            previous_mvs: Vec::new(),
            last: None,
            intra_only: false,
            pictures: PictureRecycler::default(),
            shown: None,
            scratch: None,
            base_only: false,
            last_spatial_id: None,
            max_size: None,
            dsp: nc_dsp::Dsp::new(),
        }
    }
}

impl Decoder {
    /// A decoder awaiting a key frame.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable or disable base-only mode (`S0`). When enabled, enhancement
    /// frames in Annex B superframes are stripped via
    /// [`strip_superframe_to_base`] and only the base layer is
    /// decoded and displayed.
    pub fn set_base_only(&mut self, base_only: bool) {
        self.base_only = base_only;
    }

    /// Whether base-only mode is active.
    #[must_use]
    pub fn base_only(&self) -> bool {
        self.base_only
    }

    /// `spatial_id` (`0` on single-layer or base-layer TUs, `1` on enhancement
    /// TUs) of the latest shown frame.
    #[must_use]
    pub fn shown_spatial_id(&self) -> Option<u8> {
        self.last_spatial_id
    }

    /// Maximum visible `(width, height)` seen in the stream so far (including
    /// enhancement frames inspected before stripping when [`Self::base_only`]
    /// is active).
    #[must_use]
    pub fn max_frame_size(&self) -> Option<(usize, usize)> {
        self.max_size
    }

    /// Decode one packet and return the picture it shows: the last frame
    /// of the packet when that frame is shown (as libvpx outputs).
    ///
    /// # Errors
    ///
    /// Truncated or malformed data, a profile other than 0, or a frame
    /// whose references are missing.
    pub fn decode_packet(&mut self, packet: &[u8]) -> Result<Option<DecodedFrame>> {
        self.release_shown();
        let Some(frame) = self.decode_frames(packet)? else {
            return Ok(None);
        };
        self.last_spatial_id = Some(frame.spatial_id);
        let decoded = visible(&frame, &mut self.pictures);
        self.release(frame);
        decoded.map(Some)
    }

    /// Decode one packet and return `(picture, spatial_id)` for the shown
    /// frame (even dimensions).
    ///
    /// # Errors
    ///
    /// As [`Self::decode`].
    pub fn decode_with_layer(&mut self, packet: &[u8]) -> Result<Option<(Picture, u8)>> {
        let Some(frame) = self.decode_packet(packet)? else {
            return Ok(None);
        };
        if (frame.picture.width(), frame.picture.height()) != (frame.width, frame.height) {
            return Err(DecodeError::Unsupported("VP9 odd frame dimensions"));
        }
        Ok(Some((frame.picture, frame.spatial_id)))
    }

    /// Decode all spatial layers present in `packet` in superframe order (`S0`
    /// then `S1`, or `S0` alone when [`Self::base_only`] is set) and return
    /// each layer's [`DecodedFrame`].
    ///
    /// # Errors
    ///
    /// As [`Self::decode_packet`].
    pub fn decode_all_layers(&mut self, packet: &[u8]) -> Result<Vec<DecodedFrame>> {
        self.release_shown();
        if self.base_only {
            let stripped = strip_superframe_to_base(packet)?;
            return Ok(self.decode_packet(&stripped)?.into_iter().collect());
        }
        if let Some(frames) = split_superframe(packet) {
            let mut outputs = Vec::with_capacity(frames.len());
            let mut last_stored = None;
            for (idx, raw) in frames.into_iter().enumerate() {
                let (stored, _) = self.decode_frame_layer(raw, idx as u8, true)?;
                // A unit's lone hidden base is shown again by the
                // `show_existing_frame` after it: one layer picture.
                let repeated = |s: &Arc<StoredFrame>| {
                    last_stored
                        .as_ref()
                        .is_some_and(|last| Arc::ptr_eq(last, s))
                };
                if let Some(stored) = stored.filter(|s| !repeated(s)) {
                    outputs.push(visible(&stored, &mut self.pictures)?);
                    if let Some(prev) = last_stored.replace(stored) {
                        self.release(prev);
                    }
                }
            }
            if let Some(last) = last_stored {
                self.last_spatial_id = Some(last.spatial_id);
                self.shown = Some(last);
            }
            Ok(outputs)
        } else {
            Ok(self.decode_packet(packet)?.into_iter().collect())
        }
    }

    /// Decode one packet that shows a frame of even size.
    ///
    /// # Errors
    ///
    /// As [`Self::decode_packet`], and a packet that shows nothing or a
    /// frame of odd size.
    pub fn decode(&mut self, packet: &[u8]) -> Result<Picture> {
        let frame = self
            .decode_packet(packet)?
            .ok_or(DecodeError::Unsupported("VP9 packet without a shown frame"))?;
        if (frame.picture.width(), frame.picture.height()) != (frame.width, frame.height) {
            return Err(DecodeError::Unsupported("VP9 odd frame dimensions"));
        }
        Ok(frame.picture)
    }

    /// [`Self::decode`] without the output copy: the frame is shown by
    /// [`Self::shown`] until the next call.
    ///
    /// # Errors
    ///
    /// As [`Self::decode`].
    pub fn decode_shown(&mut self, packet: &[u8]) -> Result<()> {
        if !self.decode_kept(packet)? {
            return Err(DecodeError::Unsupported("VP9 packet without a shown frame"));
        }
        if self
            .shown
            .as_ref()
            .is_some_and(|frame| !(frame.width.is_multiple_of(2) && frame.height.is_multiple_of(2)))
        {
            self.release_shown();
            return Err(DecodeError::Unsupported("VP9 odd frame dimensions"));
        }
        Ok(())
    }

    /// Decode one packet and keep the frame it shows, if any, until the
    /// next call, without copying it ([`Self::shown_frame`]); returns
    /// whether the packet shows a frame.
    ///
    /// # Errors
    ///
    /// As [`Self::decode_packet`].
    pub fn decode_kept(&mut self, packet: &[u8]) -> Result<bool> {
        // Release the previous frame first, so this one can reuse it.
        self.release_shown();
        self.shown = self.decode_frames(packet)?;
        self.last_spatial_id = self.shown.as_ref().map(|f| f.spatial_id);
        Ok(self.shown.is_some())
    }

    /// The decoder's own picture of the frame the last [`Self::decode_kept`]
    /// showed and its visible size (any parity; the picture is larger).
    #[must_use]
    pub fn shown_frame(&self) -> Option<(&Picture, (usize, usize))> {
        let frame = self.shown.as_ref()?;
        Some((&frame.picture, (frame.width, frame.height)))
    }

    /// The visible samples of the frame the last successful
    /// [`Self::decode_shown`] showed.
    #[must_use]
    pub fn shown(&self) -> Option<ShownPicture<'_>> {
        let frame = self.shown.as_ref()?;
        ShownPicture::window(&frame.picture, (0, 0), (frame.width, frame.height))
    }

    fn base_width_hint(&self) -> Option<usize> {
        self.slots
            .iter()
            .flatten()
            .map(|f| f.width)
            .min()
            .or_else(|| self.last.map(|l| l.width))
    }

    fn record_max_size(&mut self, width: usize, height: usize) {
        self.max_size = Some(
            self.max_size
                .map_or((width, height), |(mw, mh)| (mw.max(width), mh.max(height))),
        );
    }

    fn note_superframe_max_size(&mut self, packet: &[u8]) {
        let fallback_width = self.base_width_hint();
        if let Some(frames) = split_superframe(packet) {
            for raw in frames {
                if let Ok(info) = parse_uncompressed_header_info(raw, fallback_width)
                    && let Some((w, h)) = info.coded_size
                {
                    self.record_max_size(w, h);
                }
            }
        }
    }

    /// Decode the frames of one packet; returns the one it shows last.
    fn decode_frames(&mut self, packet: &[u8]) -> Result<Option<Arc<StoredFrame>>> {
        let stripped;
        let packet = if self.base_only {
            self.note_superframe_max_size(packet);
            stripped = strip_superframe_to_base(packet)?;
            &stripped[..]
        } else {
            packet
        };
        let mut shown = None;
        let mut show = |decoder: &mut Self, frame: Option<Arc<StoredFrame>>| {
            if let Some(previous) = std::mem::replace(&mut shown, frame) {
                decoder.release(previous);
            }
        };
        if let Some(frames) = split_superframe(packet) {
            for (idx, frame) in frames.into_iter().enumerate() {
                let (frame, _) = self.decode_frame_layer(frame, idx as u8, false)?;
                show(self, frame);
            }
        } else {
            let mut rest = packet;
            while !rest.is_empty() {
                let (frame, consumed) = self.decode_frame_layer(rest, 0, false)?;
                show(self, frame);
                // Encoders may pad a frame with zero bytes.
                rest = &rest[consumed..];
                let padding = rest.iter().take_while(|&&b| b == 0).count();
                rest = &rest[padding..];
            }
        }
        Ok(shown)
    }

    /// Drop the shown frame (see [`Self::release`]).
    fn release_shown(&mut self) {
        if let Some(frame) = self.shown.take() {
            self.release(frame);
        }
    }

    /// Drop a hold on `frame`, keeping its picture for reuse when nothing
    /// else (a reference slot) holds it.
    fn release(&mut self, frame: Arc<StoredFrame>) {
        if let Ok(frame) = Arc::try_unwrap(frame) {
            self.pictures.give(frame.picture);
        }
    }

    /// Return a picture [`Self::decode`] or [`Self::decode_packet`] produced
    /// once the caller is done with it, so a later frame reuses its buffer
    /// instead of allocating.
    pub fn recycle(&mut self, picture: Picture) {
        self.pictures.give(picture);
    }

    fn header_state(&self) -> HeaderState {
        HeaderState {
            slot_sizes: self
                .slots
                .each_ref()
                .map(|s| s.as_ref().map(|f| (f.width, f.height))),
            loop_filter: self.loop_filter,
            segmentation: self.segmentation,
        }
    }

    /// The probability contexts at the start of a frame
    /// (`vp9_setup_past_independence` resets them for intra and
    /// error-resilient frames).
    fn starting_contexts(&self, header: &FrameHeader) -> Box<[FrameContext; FRAME_CONTEXTS]> {
        let mut contexts = self.contexts.clone();
        if header.resets_past() {
            if header.key || header.error_resilient || header.reset_context == RESET_ALL_CONTEXTS {
                contexts
                    .iter_mut()
                    .for_each(|c| *c = FrameContext::default());
            } else if header.reset_context == RESET_ONE_CONTEXT {
                contexts[header.context_index] = FrameContext::default();
            }
        }
        contexts
    }

    /// Keep what a decoded frame leaves for later frames; returns the
    /// frame as stored in the slots it refreshes.
    fn commit(
        &mut self,
        header: &FrameHeader,
        contexts: Box<[FrameContext; FRAME_CONTEXTS]>,
        (recon, mvs, segment_map): (Picture, Vec<MvRef>, Vec<u8>),
        size_changed: bool,
        spatial_id: u8,
    ) -> Arc<StoredFrame> {
        let (width, height) = (header.width, header.height);
        self.record_max_size(width, height);
        self.contexts = contexts;
        self.loop_filter = header.loop_filter;
        self.segmentation = header.segmentation;
        if header.segmentation.enabled {
            self.last_segment_map = segment_map;
        } else if header.resets_past() || size_changed {
            self.last_segment_map = vec![0; mvs.len()];
        }
        let released = std::mem::replace(&mut self.previous_mvs, mvs);
        if let Some(scratch) = &mut self.scratch {
            scratch.mvs = released;
        }
        if !header.key {
            self.intra_only = header.intra_only;
        }
        self.last = Some(LastFrame {
            width,
            height,
            show_frame: header.show_frame,
            key: header.key,
        });
        let stored = Arc::new(StoredFrame {
            picture: recon,
            width,
            height,
            spatial_id,
        });
        for (slot, entry) in self.slots.iter_mut().enumerate() {
            if header.refresh & (1 << slot) != 0
                && let Some(released) = entry.replace(Arc::clone(&stored))
                && let Ok(frame) = Arc::try_unwrap(released)
            {
                self.pictures.give(frame.picture);
            }
        }
        stored
    }

    /// The block working memory of the last frame, or a new one.
    fn take_scratch(&mut self) -> Box<BlockScratch> {
        self.scratch
            .take()
            .unwrap_or_else(|| BlockScratch::new(&self.dsp))
    }

    /// Decode one frame; returns it if shown (or when `force_return` is set for
    /// [`Self::decode_all_layers`]), and the bytes used.
    fn decode_frame_layer(
        &mut self,
        data: &[u8],
        spatial_id: u8,
        force_return: bool,
    ) -> Result<(Option<Arc<StoredFrame>>, usize)> {
        let header = FrameHeader::parse(data, &self.header_state())?;
        if let Some(slot) = header.show_existing {
            let frame = self.slots[slot]
                .as_ref()
                .ok_or(DecodeError::MissingReference)?;
            return Ok((Some(Arc::clone(frame)), SHOW_EXISTING_BYTES));
        }
        if !header.is_intra() && self.last.is_none() {
            // The key frame was lost.
            return Err(DecodeError::MissingKey);
        }
        let (width, height) = (header.width, header.height);
        let mi_units = width.div_ceil(MI_SIZE) * height.div_ceil(MI_SIZE);
        let mut contexts = self.starting_contexts(&header);
        let context_index = header.frame_context();
        let mut fc = contexts[context_index].clone();
        let compressed_end = header.header_bytes + header.compressed_bytes;
        let compressed_data = data
            .get(header.header_bytes..compressed_end)
            .ok_or(DecodeError::Truncated)?;
        let compressed = read_compressed(compressed_data, &header, &mut fc)?;
        let size_changed = self
            .last
            .is_none_or(|l| (l.width, l.height) != (width, height));
        let scratch = self.take_scratch();
        // The last segment map clears with past independence and on a
        // size change (`vp9_init_context_buffers`).
        let cleared_map;
        let last_segment_map =
            if header.resets_past() || size_changed || self.last_segment_map.len() != mi_units {
                cleared_map = vec![0; mi_units];
                &cleared_map
            } else {
                &self.last_segment_map
            };
        let use_previous_mvs = !header.error_resilient
            && !size_changed
            && !self.intra_only
            && self.last.is_some_and(|l| l.show_frame && !l.key)
            && self.previous_mvs.len() == mi_units;
        let references = frame_references(&self.slots, &header)?;
        let inputs = FrameInputs {
            header: &header,
            compressed,
            context: &fc,
            references,
            previous_mvs: use_previous_mvs.then_some(self.previous_mvs.as_slice()),
            last_segment_map,
            dsp: &self.dsp,
        };
        let mut frame = FrameDecoder::new(inputs, &mut self.pictures, scratch)?;
        frame.decode_tiles(&data[compressed_end..])?;
        let (
            FrameOutput {
                mut recon,
                mvs,
                segment_map,
                counts,
            },
            scratch,
        ) = frame.finish();
        self.scratch = Some(scratch);
        if header.refresh != 0 {
            extend_reference(&mut recon, (width, height));
        }
        if !header.error_resilient && !header.parallel {
            let after_key = !header.is_intra() && self.last.is_some_and(|l| l.key);
            fc.adapt(
                &contexts[context_index],
                &counts,
                Adaptation {
                    after_key,
                    modes: !header.is_intra(),
                    tx_select: compressed.tx_mode == TxMode::Select,
                    switchable: header.filter.is_none(),
                    allow_hp: header.allow_hp,
                },
            );
        }
        if header.refresh_context {
            contexts[context_index] = fc;
        }
        let stored = self.commit(
            &header,
            contexts,
            (recon, mvs, segment_map),
            size_changed,
            spatial_id,
        );
        if header.show_frame || force_return {
            return Ok((Some(stored), data.len()));
        }
        // A hidden frame refreshing no slot is not kept.
        self.release(stored);
        Ok((None, data.len()))
    }
}

/// The visible part of a stored frame, its planes rounded up to even, in a
/// picture from `pictures` (every sample is copied, so stale ones are fine).
fn visible(frame: &StoredFrame, pictures: &mut PictureRecycler) -> Result<DecodedFrame> {
    let (width, height) = (
        frame.width.next_multiple_of(2),
        frame.height.next_multiple_of(2),
    );
    let mut picture = pictures
        .take(width, height, 0)
        .map_err(|_| DecodeError::Unsupported("VP9 frame size"))?;
    for plane in PlaneIndex::ALL {
        let source = frame.picture.plane(plane).view();
        let target = picture.plane_mut(plane);
        let (w, h) = (target.width(), target.height());
        let mut view = target.view_mut();
        for y in 0..h {
            view.row_mut(y)[..w].copy_from_slice(&source.row(y)[..w]);
        }
    }
    Ok(DecodedFrame {
        picture,
        width: frame.width,
        height: frame.height,
        spatial_id: frame.spatial_id,
    })
}

/// The references of an inter `header`'s frame (none for intra frames),
/// each scaled to the frame's size (`vp9_setup_scale_factors_for_frame`).
fn frame_references<'a>(
    slots: &'a [Option<Arc<StoredFrame>>; SLOTS],
    header: &FrameHeader,
) -> Result<[Option<Reference<'a>>; REFS_PER_FRAME]> {
    let mut references = [None; REFS_PER_FRAME];
    if !header.is_intra() {
        for (reference, &slot) in references.iter_mut().zip(&header.ref_slots) {
            let stored = slots[slot].as_ref().ok_or(DecodeError::MissingReference)?;
            *reference = Some(Reference::new(
                &stored.picture,
                (stored.width, stored.height),
                (header.width, header.height),
            ));
        }
    }
    Ok(references)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::superframe::SUPERFRAME_MARKER;

    #[test]
    fn superframe_index_splits_frames() {
        // Two frames of 3 and 2 bytes, one-byte sizes.
        let marker = SUPERFRAME_MARKER | 1;
        let packet = [1, 2, 3, 4, 5, marker, 3, 2, marker];
        let frames = split_superframe(&packet).unwrap();
        assert_eq!(frames, vec![&[1, 2, 3][..], &[4, 5][..]]);
        // Without a matching leading marker it is an ordinary frame.
        assert!(split_superframe(&[1, 2, 3, 2, marker]).is_none());
        assert!(split_superframe(&[0x82, 0x49]).is_none());
    }
}
