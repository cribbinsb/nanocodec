//! Stateful forward AV2 pictures with transactional reference replacement.
use std::rc::Rc;

use crate::cdf_models::TileCdfs;
use crate::frame_header::{REFERENCE_SLOTS, REFRESH_ALL};
use crate::tile_decoder::{self, TileParams};
use crate::{frame_header, inter, intra, obu, sequence::SequenceHeader};
use nc_common::{
    Picture, ShownPicture,
    decode::{DecodeError, Result},
};

/// Native encoder retaining the last reconstructed picture.
#[derive(Debug)]
pub struct Encoder {
    motion: bool,
    adaptive_cdf: bool,
    tx8: bool,
    reference: Option<Picture>,
    order_hint: u16,
}
impl Default for Encoder {
    fn default() -> Self {
        Self {
            motion: false,
            adaptive_cdf: true,
            tx8: true,
            reference: None,
            order_hint: 0,
        }
    }
}

impl Encoder {
    /// Enable the native motion-search path for forward pictures.
    #[must_use]
    pub fn with_motion() -> Self {
        Self {
            motion: true,
            ..Self::default()
        }
    }

    /// Select tile-local probability updates for subsequent pictures.
    pub fn set_cdf_adaptation(&mut self, enabled: bool) {
        self.adaptive_cdf = enabled;
    }

    /// Select lossy 8×8 luma transforms for subsequent pictures; qindex zero
    /// always retains the reversible 4×4 WHT. Default is true.
    pub fn set_tx8(&mut self, enabled: bool) {
        self.tx8 = enabled;
    }

    /// Encode a keyframe on startup, request or resolution change; otherwise forward.
    /// # Errors
    /// Rejects unsupported dimensions, syntax or picture allocation failures.
    pub fn encode(
        &mut self,
        source: &Picture,
        qindex: u8,
        force_keyframe: bool,
    ) -> Result<intra::EncodedFrame> {
        let reference = self.reference.as_ref().filter(|r| {
            !force_keyframe && (r.width(), r.height()) == (source.width(), source.height())
        });
        let encoded = if let Some(reference) = reference {
            inter::encode_inter_with_options(
                source,
                reference,
                qindex,
                self.order_hint,
                self.motion,
                crate::search::SearchParams {
                    cdf_adaptation: if self.adaptive_cdf { 1.0 } else { 0.0 },
                    tx8: if self.tx8 { 1.0 } else { 0.0 },
                    ..Default::default()
                },
            )?
        } else {
            intra::encode_keyframe_with_options(source, qindex, self.adaptive_cdf, self.tx8)?
        };
        self.order_hint = if reference.is_some() {
            (self.order_hint + 1) & 255
        } else {
            1
        };
        self.reference = Some(encoded.reconstruction.clone());
        Ok(encoded)
    }
}

/// A reference slot's entropy context (shared by the slots one picture
/// refreshes).
type SlotContext = Option<Rc<TileCdfs>>;

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
    /// The last decoded picture, shown by [`Decoder::shown`].
    current: Option<Rc<Picture>>,
    /// [`Decoder::decode_ref`]'s copy of the last picture's conformance
    /// window when it crops anything.
    cropped: Option<Picture>,
    /// Buffers reused from picture to picture.
    workspace: tile_decoder::Workspace,
}
impl Decoder {
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
                            let (key, models) =
                                Self::decode_key(&sequence, unit.payload, &mut self.workspace)?;
                            (key, models, REFRESH_ALL)
                        } else {
                            Self::decode_inter(
                                (&self.slots, &self.contexts),
                                &sequence,
                                unit.payload,
                                &mut self.workspace,
                            )?
                        };
                    picture = Some(decoded);
                    refresh = refreshed;
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
            if refresh & (1 << i) == 0 {
                continue;
            }
            slot_context.clone_from(&context);
            if let Some(replaced) = slot.replace(Rc::clone(&picture)) {
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
        if header.transform_select != (header.quantizer != 0) {
            return Err(DecodeError::Unsupported("AV2 key transform configuration"));
        }
        let params = TileParams {
            qindex: header.quantizer,
            quant_deltas: header.quant_deltas,
            adaptive_cdf: header.adaptive_cdf,
            deblock: header.deblock,
            cdef: header.cdef,
            ccso: header.ccso,
            tx_set: header.tx_set,
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

    /// Decode the inter picture of `payload` from its reference slot (and,
    /// with cross-frame context, that slot's CDFs), returning it with its
    /// final CDFs and the slots it refreshes.
    fn decode_inter(
        (slots, contexts): (&[Option<Rc<Picture>>], &[SlotContext]),
        sequence: &SequenceHeader,
        payload: &[u8],
        work: &mut tile_decoder::Workspace,
    ) -> Result<(Picture, Option<Box<TileCdfs>>, u8)> {
        let (header, offset) = frame_header::parse_inter(sequence, payload)?;
        // Implicit references refresh every slot: any slot holds the reference.
        let slot = header.reference.unwrap_or(0);
        let reference = slots[usize::from(slot)]
            .as_deref()
            .ok_or(DecodeError::MissingKey)?;
        let (width, height) = (sequence.max_width as usize, sequence.max_height as usize);
        if reference.width() != width || reference.height() != height {
            return Err(DecodeError::Unsupported(
                "AV2 changed sequence requires keyframe",
            ));
        }
        if header.transform_select != (header.quantizer != 0) {
            return Err(DecodeError::Unsupported(
                "AV2 forward transform configuration",
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
                filter: header.filter,
                skip_mode: header.skip_mode,
            }),
            inherited,
            work,
        )?;
        Ok((picture, models, header.refresh_flags))
    }

    /// Decode a packet and return an owned copy; use `decode_ref` to avoid the copy.
    /// # Errors
    /// Propagates unsupported syntax, invalid reference state and truncation.
    pub fn decode(&mut self, packet: &[u8]) -> Result<Picture> {
        self.decode_ref(packet).cloned()
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

#[cfg(test)]
mod tests {
    use super::*;
    use nc_common::PlaneIndex;
    #[test]
    fn adaptive_fixed_switching_and_rejected_trials_preserve_reference() {
        let mut source = Picture::new(16, 16, 0).unwrap();
        let mut encoder = Encoder::with_motion();
        let mut decoder = Decoder::default();
        for index in 0..12 {
            let qindex = [0, 100, 200][index % 3];
            for plane in PlaneIndex::ALL {
                let mut view = source.plane_mut(plane).view_mut();
                for y in 0..view.height() {
                    for (x, pixel) in view.row_mut(y).iter_mut().enumerate() {
                        *pixel = ((x * 19 + y * 31 + index * 17) % 256) as u8;
                    }
                }
            }
            encoder.set_cdf_adaptation(index % 2 == 0);
            encoder.set_tx8(index % 4 >= 2);
            let encoded = encoder.encode(&source, qindex, index == 6).unwrap();
            assert!(decoder.decode(&[]).is_err());
            let decoded = decoder.decode_ref(&encoded.packet).unwrap();
            for plane in PlaneIndex::ALL {
                let actual = decoded.plane(plane).view();
                let expected = encoded.reconstruction.plane(plane).view();
                let original = source.plane(plane).view();
                for y in 0..actual.height() {
                    assert_eq!(actual.row(y), expected.row(y));
                    if qindex == 0 {
                        assert_eq!(actual.row(y), original.row(y));
                    }
                }
            }
        }
    }

    #[test]
    fn forward_chain_and_failed_packet_preserve_reference() {
        for (qindex, use_motion, tx8) in [0, 1, 100, 200, 255].into_iter().flat_map(|q| {
            [
                (q, false, false),
                (q, true, false),
                (q, false, true),
                (q, true, true),
            ]
        }) {
            let mut encoder = if use_motion {
                Encoder::with_motion()
            } else {
                Encoder::default()
            };
            encoder.set_tx8(tx8);
            let mut decoder = Decoder::default();
            for index in 0..7 {
                let mut source = Picture::new(16, 16, 0).unwrap();
                for p in PlaneIndex::ALL {
                    let mut view = source.plane_mut(p).view_mut();
                    for row in 0..view.height() {
                        for (col, pixel) in view.row_mut(row).iter_mut().enumerate() {
                            *pixel = ((row * 11 + col * 37 + (index / 2) * 23) % 256) as u8;
                        }
                    }
                }
                let encoded = encoder.encode(&source, qindex, index == 4).unwrap();
                assert!(decoder.decode_ref(&[0]).is_err());
                let decoded = decoder.decode_ref(&encoded.packet).unwrap();
                for p in PlaneIndex::ALL {
                    let expected = encoded.reconstruction.plane(p).view();
                    let actual = decoded.plane(p).view();
                    for row in 0..expected.height() {
                        assert_eq!(expected.row(row), actual.row(row));
                    }
                }
            }
        }
    }
}
