//! The AV2 decoder: stateful forward pictures with transactional reference
//! replacement.
use std::rc::Rc;

use crate::cdf_models::TileCdfs;
use crate::frame_header::REFERENCE_SLOTS;
use crate::tile_decoder::{self, TileParams};
use crate::{frame_header, obu, sequence::SequenceHeader};
use nc_common::{
    Picture, ShownPicture,
    decode::{DecodeError, Result},
};

/// A reference slot's entropy context (shared by the slots one picture
/// refreshes).
type SlotContext = Option<Rc<TileCdfs>>;

/// The decoder's reference slots as an inter frame reads them: pictures,
/// CDFs and display order hints.
type SlotView<'a> = (&'a [Option<Rc<Picture>>], &'a [SlotContext], &'a [u32]);

/// Decoder retaining sequence syntax and the eight reference slots.
#[derive(Debug, Default)]
pub struct Decoder {
    sequence: Option<SequenceHeader>,
    /// Reference slots (`ref_frame_map`); one picture may fill several.
    slots: [Option<Rc<Picture>>; REFERENCE_SLOTS as usize],
    /// Each slot's picture's final CDFs, adaptation restarted (AVM
    /// `frame_context`), for frames that inherit them; `None` where the
    /// picture did not adapt.
    contexts: [SlotContext; REFERENCE_SLOTS as usize],
    /// Each slot's picture's display order hint (AVM `display_order_hint`,
    /// the order hint unwrapped: 0 at the key frame), for reference
    /// distances.
    display_orders: [u32; REFERENCE_SLOTS as usize],
    /// The last decoded picture, shown by [`Decoder::shown`].
    current: Option<Rc<Picture>>,
    /// [`Decoder::decode_ref`]'s copy of the last picture's conformance
    /// window when it crops anything.
    cropped: Option<Picture>,
    /// Buffers reused from picture to picture.
    workspace: tile_decoder::Workspace,
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

    /// Decode a packet, returning the borrowed visible picture (the
    /// reconstruction inside the sequence's conformance window) to avoid a
    /// frame copy when nothing is cropped.
    /// The returned picture remains valid until the next mutable decoder operation.
    /// # Errors
    /// Rejects missing references, multiple pictures, unsupported or malformed syntax.
    pub fn decode_ref(&mut self, packet: &[u8]) -> Result<&Picture> {
        self.decode_shown(packet)?;
        let shown = self.shown().expect("a decoded picture is shown");
        let current = self.current.as_deref().expect("a decoded picture");
        self.cropped = if (shown.width(), shown.height()) == (current.width(), current.height()) {
            None
        } else {
            let mut cropped = Picture::new(shown.width(), shown.height(), 0)?;
            shown.copy_to(&mut cropped);
            Some(cropped)
        };
        let current = self.current.as_deref().expect("a decoded picture");
        Ok(self.cropped.as_ref().unwrap_or(current))
    }

    /// Decode a packet without copying its picture: [`Decoder::shown`]
    /// shows it until the next call.
    /// # Errors
    /// As [`Decoder::decode_ref`].
    pub fn decode_shown(&mut self, packet: &[u8]) -> Result<()> {
        let mut next_sequence = self.sequence;
        let mut remaining = packet;
        let mut picture = None;
        let mut refresh = 0;
        let mut context = None;
        let mut display_order = 0;
        let mut key = false;
        while !remaining.is_empty() {
            let (unit, rest) = obu::parse(remaining)?;
            remaining = rest;
            // Regular frames may sit in any temporal layer of the sequence;
            // everything else is base layer (`obu::parse` checks key frames).
            let layers = next_sequence.map_or(1, |s| s.layering.temporal_layers);
            let temporal_ok = unit.header.temporal_id == 0
                || (unit.header.obu_type == frame_header::REGULAR_TILE_GROUP
                    && unit.header.temporal_id < layers);
            if unit.header.obu_type != 2
                && (unit.header.extended_id != 0 || unit.header.embedded_id != 0 || !temporal_ok)
            {
                return Err(DecodeError::Unsupported(
                    "AV2 forward decoder requires base embedded layer",
                ));
            }
            match unit.header.obu_type {
                2 if unit.payload.is_empty() => {}
                1 => next_sequence = Some(crate::sequence::parse(unit.payload)?),
                4 | 7 => {
                    if picture.is_some() {
                        return Err(DecodeError::Malformed("AV2 packet has multiple pictures"));
                    }
                    let sequence =
                        // No sequence header yet: the key frame was lost.
                        next_sequence.ok_or(DecodeError::MissingKey)?;
                    let (decoded, models, refreshed) =
                        if unit.header.obu_type == frame_header::CLOSED_LOOP_KEY {
                            let (key_picture, models) =
                                Self::decode_key(&sequence, unit.payload, &mut self.workspace)?;
                            display_order = 0;
                            key = true;
                            (key_picture, models, KEY_SLOTS)
                        } else {
                            let (decoded, models, refreshed, order) = Self::decode_inter(
                                (&self.slots, &self.contexts, &self.display_orders),
                                &sequence,
                                unit.payload,
                                &mut self.workspace,
                            )?;
                            display_order = order;
                            (decoded, models, refreshed)
                        };
                    picture = Some(decoded);
                    refresh = refreshed;
                    #[cfg(feature = "analysis")]
                    if let Some(s) = &mut self.workspace.stats {
                        s.finish();
                    }
                    context = models.map(|mut models| {
                        models.restart_adaptation();
                        Rc::from(models)
                    });
                }
                _ => return Err(DecodeError::Unsupported("AV2 OBU outside forward subset")),
            }
        }
        let picture = picture.ok_or(DecodeError::Malformed("AV2 packet has no picture"))?;
        self.sequence = next_sequence;
        let picture = Rc::new(picture);
        for (i, (slot, slot_context)) in self.slots.iter_mut().zip(&mut self.contexts).enumerate() {
            let replaced = if refresh & (1 << i) != 0 {
                slot_context.clone_from(&context);
                self.display_orders[i] = display_order;
                slot.replace(Rc::clone(&picture))
            } else if key {
                // A key frame empties every slot it does not fill.
                *slot_context = None;
                slot.take()
            } else {
                None
            };
            if let Some(replaced) = replaced {
                self.workspace.release(replaced);
            }
        }
        if let Some(previous) = self.current.replace(picture) {
            self.workspace.release(previous);
        }
        Ok(())
    }

    /// The visible samples (the sequence's conformance window) of the last
    /// picture decoded.
    #[must_use]
    pub fn shown(&self) -> Option<ShownPicture<'_>> {
        let picture = self.current.as_deref()?;
        let size = self.sequence?.visible_size();
        ShownPicture::window(picture, (0, 0), size)
    }

    /// Decode the closed-loop key picture of `payload`, returning it with
    /// its final CDFs when it adapts.
    fn decode_key(
        sequence: &SequenceHeader,
        payload: &[u8],
        work: &mut tile_decoder::Workspace,
    ) -> Result<(Picture, Option<Box<TileCdfs>>)> {
        let (header, offset) = frame_header::parse_keyframe(sequence, payload)?;
        let params = TileParams {
            qindex: header.quantizer,
            quant_deltas: header.quant_deltas,
            adaptive_cdf: header.adaptive_cdf,
            deblock: header.deblock,
            cdef: header.cdef,
            ccso: header.ccso,
            tx_set: header.tx_set,
            tx_select: header.transform_select,
            screen_content: header.screen_content,
            levels: header.levels,
            delta_q: header.delta_q,
            reference_lists: sequence.reference_lists,
        };
        let (width, height) = (sequence.max_width as usize, sequence.max_height as usize);
        tile_decoder::decode(
            &payload[offset..],
            (width, height),
            params,
            None,
            None,
            work,
        )
    }

    /// Decode the inter picture of `payload` from its reference slots (and,
    /// with cross-frame context, the first one's CDFs), returning it with
    /// its final CDFs, the slots it refreshes and its display order hint.
    fn decode_inter(
        (slots, contexts, display_orders): SlotView<'_>,
        sequence: &SequenceHeader,
        payload: &[u8],
        work: &mut tile_decoder::Workspace,
    ) -> Result<(Picture, Option<Box<TileCdfs>>, u8, u32)> {
        let (header, offset) = frame_header::parse_inter(sequence, payload)?;
        let display_order = display_order_hint(
            header.order_hint,
            sequence.order_hint_bits,
            slots
                .iter()
                .zip(display_orders)
                .filter(|(slot, _)| slot.is_some())
                .map(|(_, &order)| order),
        );
        // Implicit references refresh every slot: any slot holds the reference.
        let slot = header.reference.unwrap_or(0);
        let picture_in = |slot: u8| {
            slots[usize::from(slot)]
                .as_deref()
                .ok_or(DecodeError::MissingKey)
        };
        let reference = picture_in(slot)?;
        let distance = |slot: u8| display_order.saturating_sub(display_orders[usize::from(slot)]);
        let second = match header.second_reference {
            Some(second) => {
                let distances = [distance(slot), distance(second)]
                    .map(|d| i32::try_from(d.min(MAX_RELATIVE_DISTANCE)).expect("small"));
                Some(tile_decoder::SecondReference {
                    picture: picture_in(second)?,
                    distances,
                })
            }
            None => None,
        };
        let (width, height) = (sequence.max_width as usize, sequence.max_height as usize);
        if reference.width() != width || reference.height() != height {
            return Err(DecodeError::Unsupported(
                "AV2 changed sequence requires keyframe",
            ));
        }
        let params = TileParams {
            qindex: header.quantizer,
            quant_deltas: header.quant_deltas,
            adaptive_cdf: header.adaptive_cdf,
            deblock: header.deblock,
            cdef: header.cdef,
            ccso: header.ccso,
            tx_set: header.tx_set,
            tx_select: header.transform_select,
            screen_content: false,
            levels: header.levels,
            delta_q: header.delta_q,
            reference_lists: sequence.reference_lists,
        };
        let inherited = if header.cross_frame_context {
            Some(
                contexts[usize::from(slot)]
                    .as_deref()
                    .ok_or(DecodeError::Malformed(
                        "AV2 inherited context did not adapt",
                    ))?,
            )
        } else {
            None
        };
        let (picture, models) = tile_decoder::decode(
            &payload[offset..],
            (width, height),
            params,
            Some(tile_decoder::InterReference {
                picture: reference,
                second,
                filter: header.filter,
                skip_mode: header.skip_mode,
            }),
            inherited,
            work,
        )?;
        Ok((picture, models, header.refresh_flags, display_order))
    }

    /// Decode a packet and return an owned copy; use `decode_ref` to avoid the copy.
    /// # Errors
    /// Propagates unsupported syntax, invalid reference state and truncation.
    pub fn decode(&mut self, packet: &[u8]) -> Result<Picture> {
        self.decode_ref(packet).cloned()
    }
}

/// The slots a closed-loop key frame fills: the first only (AVM
/// `av2_frame_clears_multiple_inserted_in_one` empties the others).
const KEY_SLOTS: u8 = 1;

/// AVM `RELATIVE_DIST_BITS` 8: reference distances are clamped to 127.
const MAX_RELATIVE_DISTANCE: u32 = 127;

/// The display order hint of a frame with `order_hint` (`bits` wide) whose
/// references' display order hints are `references` (AVM
/// `get_disp_order_hint`): the order hint unwrapped past the largest of
/// them.
fn display_order_hint(order_hint: u16, bits: u8, references: impl Iterator<Item = u32>) -> u32 {
    let largest = i64::from(references.max().unwrap_or(0));
    let factor = 1_i64 << bits;
    let mut order = i64::from(order_hint);
    while (largest - order).abs() >= factor >> 1 {
        if order > largest {
            break;
        }
        order += factor;
    }
    u32::try_from(order).expect("display order hints stay small")
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
        if matches!(
            unit.header.obu_type,
            frame_header::CLOSED_LOOP_KEY | frame_header::REGULAR_TILE_GROUP
        ) {
            return Ok(unit.header.temporal_id);
        }
        remaining = rest;
    }
    Err(DecodeError::Malformed("AV2 temporal unit without a frame"))
}
