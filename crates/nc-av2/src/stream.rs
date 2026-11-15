//! The AV2 decoder: stateful forward pictures with transactional reference
//! replacement.
use std::rc::Rc;

use crate::cdf_models::TileCdfs;
use crate::frame_header::REFERENCE_SLOTS;
use crate::general::references::{self, Current, MAX_REFERENCES, SlotInfo};
use crate::tile_decoder::{self, TileParams};
use crate::{frame_header, obu, sequence::SequenceHeader};
use nc_common::{
    Picture, ShownPicture,
    decode::{DecodeError, Result},
};

/// One reference slot (`ref_frame_map` entry): a picture, which may fill
/// several slots, with what later frames need of it.
#[derive(Debug, Clone)]
struct Slot {
    picture: Rc<Picture>,
    /// The picture's final CDFs, adaptation restarted (AVM
    /// `frame_context`), for frames that inherit them; `None` where the
    /// picture did not adapt.
    context: Option<Rc<TileCdfs>>,
    /// Display order (AVM `display_order_hint`, the order hint unwrapped:
    /// 0 at the key frame), quantizer, frame type and layer, for
    /// reference ranking and distances.
    info: SlotInfo,
}

/// Decoder retaining sequence syntax and the eight reference slots.
#[derive(Debug, Default)]
pub struct Decoder {
    sequence: Option<SequenceHeader>,
    /// Reference slots (`ref_frame_map`).
    slots: [Option<Slot>; REFERENCE_SLOTS as usize],
    /// Frames decoded so far: the identity of the next one in its slots.
    frames: u32,
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
        let mut decoded = None;
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
                    if decoded.is_some() {
                        return Err(DecodeError::Malformed("AV2 packet has multiple pictures"));
                    }
                    let sequence =
                        // No sequence header yet: the key frame was lost.
                        next_sequence.ok_or(DecodeError::MissingKey)?;
                    let frame = if unit.header.obu_type == frame_header::CLOSED_LOOP_KEY {
                        Self::decode_key(&sequence, unit.payload, &mut self.workspace)?
                    } else {
                        Self::decode_inter(
                            &self.slots,
                            &sequence,
                            (unit.payload, unit.header.temporal_id),
                            &mut self.workspace,
                        )?
                    };
                    decoded = Some(frame);
                    #[cfg(feature = "analysis")]
                    if let Some(s) = &mut self.workspace.stats {
                        s.finish();
                    }
                }
                _ => return Err(DecodeError::Unsupported("AV2 OBU outside forward subset")),
            }
        }
        let frame = decoded.ok_or(DecodeError::Malformed("AV2 packet has no picture"))?;
        self.sequence = next_sequence;
        let picture = Rc::new(frame.picture);
        let slot = Slot {
            picture: Rc::clone(&picture),
            context: frame.models.map(|mut models| {
                models.restart_adaptation();
                Rc::from(models)
            }),
            info: SlotInfo {
                frame: self.frames,
                size: (picture.width() as u32, picture.height() as u32),
                ..frame.info
            },
        };
        self.frames = self.frames.wrapping_add(1);
        for (i, held) in self.slots.iter_mut().enumerate() {
            let replaced = if frame.refresh & (1 << i) != 0 {
                held.replace(slot.clone())
            } else if !frame.info.inter {
                // A key frame empties every slot it does not fill.
                held.take()
            } else {
                None
            };
            if let Some(replaced) = replaced {
                self.workspace.release(replaced.picture);
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

    /// Decode the closed-loop key picture of `payload`.
    fn decode_key(
        sequence: &SequenceHeader,
        payload: &[u8],
        work: &mut tile_decoder::Workspace,
    ) -> Result<DecodedFrame> {
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
        let (picture, models) = tile_decoder::decode(
            &payload[offset..],
            (width, height),
            params,
            None,
            None,
            work,
        )?;
        Ok(DecodedFrame {
            picture,
            models,
            refresh: KEY_SLOTS,
            info: SlotInfo {
                frame: 0,
                order: 0,
                quantizer: header.quantizer,
                inter: false,
                temporal_layer: 0,
                size: (0, 0),
            },
        })
    }

    /// Decode the inter picture of `payload` (in temporal layer
    /// `temporal_layer`) from the reference slots: those it names, or with
    /// implicit references the distinct pictures ranked as AVM ranks them.
    fn decode_inter(
        slots: &[Option<Slot>],
        sequence: &SequenceHeader,
        (payload, temporal_layer): (&[u8], u8),
        work: &mut tile_decoder::Workspace,
    ) -> Result<DecodedFrame> {
        let (header, offset) = frame_header::parse_inter(sequence, payload)?;
        let infos: Vec<Option<SlotInfo>> = slots
            .iter()
            .map(|slot| slot.as_ref().map(|slot| slot.info))
            .collect();
        let display_order = display_order_hint(
            header.order_hint,
            sequence.order_hint_bits,
            infos.iter().flatten().map(|info| info.order),
        );
        let (width, height) = (sequence.max_width as usize, sequence.max_height as usize);
        let (ranked, count) = match (header.reference, header.second_reference) {
            (Some(first), Some(second)) => ([first, second, 0, 0, 0, 0, 0], 2),
            (Some(first), None) => ([first; MAX_REFERENCES], 1),
            (None, _) => references::rank(
                &infos,
                Current {
                    order: display_order,
                    temporal_layer,
                    size: Some((width as u32, height as u32)),
                },
            ),
        };
        let held = |slot: u8| {
            slots[usize::from(slot)]
                .as_ref()
                .ok_or(DecodeError::MissingKey)
        };
        let mut pictures = [None; MAX_REFERENCES];
        let mut distances = [1; MAX_REFERENCES];
        let mut ranked_infos = [None; MAX_REFERENCES];
        for rank in 0..count.max(1) {
            let slot = held(ranked[rank])?;
            if slot.picture.width() != width || slot.picture.height() != height {
                return Err(DecodeError::Unsupported(
                    "AV2 changed sequence requires keyframe",
                ));
            }
            pictures[rank] = Some(&*slot.picture);
            distances[rank] = references::order_difference(display_order, slot.info.order);
            ranked_infos[rank] = Some(slot.info);
        }
        let count = count.max(1);
        let primary = if !header.cross_frame_context {
            None
        } else if header.derived_primary {
            references::context_sources(&ranked_infos[..count], header.quantizer, display_order).0
        } else {
            Some(0)
        };
        let inherited = match primary {
            Some(rank) => Some(held(ranked[usize::from(rank)])?.context.as_deref().ok_or(
                DecodeError::Malformed("AV2 inherited context did not adapt"),
            )?),
            None => None,
        };
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
        let (picture, models) = tile_decoder::decode(
            &payload[offset..],
            (width, height),
            params,
            Some(tile_decoder::InterReference {
                pictures: &pictures[..count],
                distances: &distances[..count],
                filter: header.filter,
                skip_mode: header.skip_mode,
            }),
            inherited,
            work,
        )?;
        Ok(DecodedFrame {
            picture,
            models,
            refresh: header.refresh_flags,
            info: SlotInfo {
                frame: 0,
                order: display_order,
                quantizer: header.quantizer,
                inter: true,
                temporal_layer,
                size: (0, 0),
            },
        })
    }

    /// Decode a packet and return an owned copy; use `decode_ref` to avoid the copy.
    /// # Errors
    /// Propagates unsupported syntax, invalid reference state and truncation.
    pub fn decode(&mut self, packet: &[u8]) -> Result<Picture> {
        self.decode_ref(packet).cloned()
    }
}

/// A decoded frame with what its reference slots keep of it.
struct DecodedFrame {
    picture: Picture,
    /// Final CDFs, when the frame adapted.
    models: Option<Box<TileCdfs>>,
    /// The slots it fills (`refresh_frame_flags`).
    refresh: u8,
    /// Its ranking information (the decoder sets `frame` and `size`).
    info: SlotInfo,
}

/// The slots a closed-loop key frame fills: the first only (AVM
/// `av2_frame_clears_multiple_inserted_in_one` empties the others).
const KEY_SLOTS: u8 = 1;

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
