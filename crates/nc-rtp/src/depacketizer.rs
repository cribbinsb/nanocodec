//! The receiver: RTP packets in (any order, some lost), decoder-ready units
//! out, plus the signals a sender needs back: a keyframe request when
//! decoding cannot continue, and the frames known lost.
//!
//! Each packet is parsed by its payload format (pieces of NAL units, OBUs
//! or VP9 frames) and, when negotiated, its Dependency Descriptor. The
//! assembler orders packets into units. A complete unit is delivered when
//! it is decodable; a unit with gaps is cut down to its complete layer
//! frames when the packets describe frames (Dependency Descriptor, VP9
//! descriptor). Decodability follows the frames' references when the
//! packets give them, otherwise the rule is "after a loss, wait for a key"
//! (docs/handoff/rtp.md "Receiver rules").

use std::collections::{BTreeSet, VecDeque};

use crate::assembler::{AssembledUnit, Assembler, Stored};
use crate::av1;
use crate::dependency_descriptor::{DependencyDescriptor, TemplateStructure};
use crate::error::Result;
use crate::nal::{self, h264::H264, hevc::Hevc, vvc::Vvc};
use crate::packet::RtpPacket;
use crate::packetizer::PayloadFormat;
use crate::payload::{FrameTag, ParsedPayload};
use crate::seq::Unwrapper;
use crate::vp9;

/// Default reorder window in packets: about a frame of 1200-byte packets
/// at 10 Mbit/s, 30 fps.
pub const DEFAULT_REORDER_WINDOW: usize = 64;
/// Frame identities remembered as decoded (for reference checks).
const DECODED_HISTORY: usize = 4096;

/// Depacketizer settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DepacketizerConfig {
    /// The payload format.
    pub format: PayloadFormat,
    /// Packets a gap may stay open before it counts as loss.
    pub reorder_window: usize,
    /// RFC 8285 id of the Dependency Descriptor, if negotiated.
    pub dependency_descriptor_id: Option<u8>,
    /// Keep delivering after a loss instead of waiting for a key unit: for
    /// streams that repair themselves (intra refresh, LTR recovery on loss
    /// feedback), where the decoder's own reference checks decide.
    pub deliver_after_loss: bool,
    /// Accept only this payload type (others are ignored).
    pub payload_type: Option<u8>,
}

impl DepacketizerConfig {
    /// Defaults for `format` (AV1: Dependency Descriptor id 1).
    #[must_use]
    pub fn new(format: PayloadFormat) -> Self {
        Self {
            format,
            reorder_window: DEFAULT_REORDER_WINDOW,
            dependency_descriptor_id: (format == PayloadFormat::Av1).then_some(1),
            deliver_after_loss: false,
            payload_type: None,
        }
    }
}

/// One delivered unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceivedUnit {
    /// The RTP timestamp.
    pub timestamp: u32,
    /// Decoder input: H.264/HEVC/VVC one Annex-B access unit, AV1 one
    /// temporal unit; VP9 one entry per layer frame (pack several with
    /// `nc_vp9::pack_superframe`).
    pub frames: Vec<Vec<u8>>,
    /// A key unit: decoding can start here.
    pub key: bool,
    /// Every packet of the unit arrived (false: only its complete,
    /// decodable layer frames are delivered).
    pub complete: bool,
    /// Spatial ids of the delivered layer frames, when the packets say.
    pub spatial_ids: Vec<u8>,
    /// Frame numbers of the delivered layer frames (Dependency Descriptor
    /// frame numbers or VP9 picture numbers, unwrapped), when the packets
    /// say: the numbers a receiver acknowledges in loss feedback.
    pub frame_ids: Vec<u64>,
}

/// Receiver counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReceiverStats {
    /// Packets accepted.
    pub packets: u64,
    /// Packets declared lost.
    pub lost_packets: u64,
    /// Packets that came after their place was consumed.
    pub late_packets: u64,
    /// Units delivered.
    pub units: u64,
    /// Units (or their layer frames) not delivered.
    pub dropped_units: u64,
}

/// A layer frame of a unit, as its packets describe it.
#[derive(Debug, Clone)]
struct FrameRun {
    tag: FrameTag,
    /// Indices of its packets in the unit.
    packets: Vec<usize>,
    complete: bool,
}

/// Turns RTP packets back into decoder input.
#[derive(Debug, Clone)]
pub struct Depacketizer {
    config: DepacketizerConfig,
    assembler: Assembler,
    structure: Option<TemplateStructure>,
    frame_numbers: Unwrapper<16>,
    picture_ids: Unwrapper<15>,
    decoded: BTreeSet<u64>,
    waiting_for_key: bool,
    keyframe_request: bool,
    lost_frames: Vec<u64>,
    ready: VecDeque<ReceivedUnit>,
    stats: ReceiverStats,
}

impl Depacketizer {
    /// A receiver for one RTP stream.
    #[must_use]
    pub fn new(config: DepacketizerConfig) -> Self {
        Self {
            assembler: Assembler::new(config.reorder_window),
            config,
            structure: None,
            frame_numbers: Unwrapper::new(),
            picture_ids: Unwrapper::new(),
            decoded: BTreeSet::new(),
            // Nothing decodes before the first key unit.
            waiting_for_key: true,
            keyframe_request: false,
            lost_frames: Vec::new(),
            ready: VecDeque::new(),
            stats: ReceiverStats::default(),
        }
    }

    /// Take one datagram.
    ///
    /// # Errors
    /// A packet that does not parse (it is dropped; the stream goes on).
    pub fn push(&mut self, datagram: &[u8]) -> Result<()> {
        let packet = RtpPacket::parse(datagram)?;
        if self
            .config
            .payload_type
            .is_some_and(|pt| pt != packet.header.payload_type)
        {
            return Ok(());
        }
        let mut parsed = self.parse_payload(&packet)?;
        parsed.info.dependency_descriptor = self
            .config
            .dependency_descriptor_id
            .and_then(|id| packet.extension(id))
            .map(<[u8]>::to_vec);
        self.stats.packets += 1;
        self.assembler.push(
            packet.header.sequence,
            packet.header.timestamp,
            packet.header.marker,
            parsed,
        );
        self.drain(false);
        Ok(())
    }

    /// End of stream: every gap is loss; release what is left.
    pub fn flush(&mut self) {
        self.drain(true);
    }

    /// Playout deadline (a time-based jitter buffer): packets of RTP
    /// timestamp `timestamp` or older are due, so a gap before one of
    /// them is loss now rather than when the reorder window overflows.
    pub fn set_playout_deadline(&mut self, timestamp: u32) {
        self.assembler.set_deadline(timestamp);
        self.drain(false);
    }

    /// The next delivered unit.
    pub fn pop(&mut self) -> Option<ReceivedUnit> {
        self.ready.pop_front()
    }

    /// True once after decoding stopped for want of a key unit: the caller
    /// sends a PLI or FIR (RFC 4585, RFC 5104).
    pub fn take_keyframe_request(&mut self) -> bool {
        std::mem::take(&mut self.keyframe_request)
    }

    /// Frames known lost or undecodable since the last call (Dependency
    /// Descriptor frame numbers or VP9 picture numbers, unwrapped), for
    /// loss feedback to the encoder.
    pub fn take_lost_frames(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.lost_frames)
    }

    /// Counters.
    #[must_use]
    pub fn stats(&self) -> ReceiverStats {
        ReceiverStats {
            lost_packets: self.assembler.lost_packets,
            late_packets: self.assembler.late_packets,
            ..self.stats
        }
    }

    fn parse_payload(&mut self, packet: &RtpPacket<'_>) -> Result<ParsedPayload> {
        match self.config.format {
            PayloadFormat::H264 => nal::depacketize::<H264>(packet.payload),
            PayloadFormat::Hevc => nal::depacketize::<Hevc>(packet.payload),
            PayloadFormat::Vvc => nal::depacketize::<Vvc>(packet.payload),
            PayloadFormat::Av1 => av1::depacketize(packet.payload),
            PayloadFormat::Vp9 => {
                let ids = &mut self.picture_ids;
                vp9::depacketize(packet.payload, |id| ids.unwrap(u64::from(id)))
            }
        }
    }

    /// The frame tag a Dependency Descriptor gives, updating the structure.
    fn parse_descriptor(&mut self, data: &[u8]) -> Option<FrameTag> {
        let descriptor = DependencyDescriptor::parse(data, self.structure.as_ref()).ok()?;
        if let Some(structure) = &descriptor.structure {
            self.structure = Some(structure.clone());
        }
        let frame = descriptor.resolve(self.structure.as_ref()?).ok()?;
        let id = self
            .frame_numbers
            .unwrap(u64::from(descriptor.frame_number));
        Some(FrameTag {
            id,
            start: descriptor.start_of_frame,
            end: descriptor.end_of_frame,
            references: Some(
                frame
                    .fdiffs
                    .iter()
                    .filter_map(|&d| id.checked_sub(u64::from(d)))
                    .collect(),
            ),
            spatial_id: frame.spatial_id,
            temporal_id: frame.temporal_id,
        })
    }

    fn drain(&mut self, flush: bool) {
        while let Some(mut unit) = self.assembler.pop(flush) {
            for (_, packet) in &mut unit.packets {
                if let Some(data) = packet.parsed.info.dependency_descriptor.take() {
                    let tag = self.parse_descriptor(&data);
                    packet.parsed.info.frame = tag.or(packet.parsed.info.frame.take());
                }
            }
            self.deliver(&unit);
        }
    }

    /// Decide what of one assembled unit reaches the decoder.
    fn deliver(&mut self, unit: &AssembledUnit) {
        let first = &unit.packets[0].1;
        let timestamp = first.raw_timestamp;
        let key = unit.packets.iter().any(|(_, p)| p.parsed.info.key);
        let starts_clean = first
            .parsed
            .pieces
            .first()
            .is_some_and(|piece| !piece.continues_previous);
        let runs = frame_runs(unit);
        if let Some(runs) = runs {
            self.deliver_frames(unit, &runs, timestamp, key);
            return;
        }
        let complete = !unit.gap_before && !unit.gap_inside && starts_clean;
        let usable = if key {
            complete || (!unit.gap_inside && starts_clean)
        } else {
            complete
        };
        if !usable {
            self.stats.dropped_units += 1;
            if !self.config.deliver_after_loss && !known_non_reference(unit) {
                self.start_waiting();
            }
            return;
        }
        if unit.gap_before && !key && !self.config.deliver_after_loss {
            // Whole units may have been lost before this one.
            self.start_waiting();
        }
        if self.waiting_for_key && !key && !self.config.deliver_after_loss {
            self.stats.dropped_units += 1;
            return;
        }
        if key {
            self.waiting_for_key = false;
        }
        let objects = objects(unit.packets.iter().map(|(_, p)| p));
        self.push_ready(
            timestamp,
            self.join(objects),
            key,
            true,
            Vec::new(),
            Vec::new(),
        );
    }

    /// Deliver the complete, decodable layer frames of a unit whose packets
    /// describe frames.
    fn deliver_frames(
        &mut self,
        unit: &AssembledUnit,
        runs: &[FrameRun],
        timestamp: u32,
        key: bool,
    ) {
        let all_complete = !unit.gap_inside && runs.iter().all(|run| run.complete);
        let undescribed = runs.iter().any(|run| run.tag.references.is_none());
        if unit.gap_before && undescribed && !key && !self.config.deliver_after_loss {
            // Frames that do not list their references (VP9 non-flexible
            // mode) after a loss: whole frames may be missing before them.
            self.start_waiting();
        }
        let mut objects_out = Vec::new();
        let mut spatial_ids = Vec::new();
        let mut frame_ids = Vec::new();
        for run in runs {
            let tag = &run.tag;
            let references_met = match &tag.references {
                Some(references) => references.iter().all(|r| self.decoded.contains(r)),
                // Not described: decodable unless a loss is pending.
                None => !self.waiting_for_key || key,
            };
            let key_frame = key && tag.spatial_id == 0;
            let decodable =
                run.complete && (references_met || key_frame || self.config.deliver_after_loss);
            if !decodable {
                self.lost_frames.push(tag.id);
                if tag.references.is_none() || tag.spatial_id == 0 {
                    self.start_waiting();
                }
                continue;
            }
            if key_frame {
                self.waiting_for_key = false;
            }
            self.decoded.insert(tag.id);
            while self.decoded.len() > DECODED_HISTORY {
                self.decoded.pop_first();
            }
            let packets = run.packets.iter().map(|&i| &unit.packets[i].1);
            objects_out.extend(objects(packets));
            spatial_ids.push(tag.spatial_id);
            frame_ids.push(tag.id);
        }
        if spatial_ids.is_empty() {
            self.stats.dropped_units += 1;
            return;
        }
        if !all_complete {
            self.stats.dropped_units += 1;
        }
        let frames = self.join(objects_out);
        self.push_ready(timestamp, frames, key, all_complete, spatial_ids, frame_ids);
    }

    fn start_waiting(&mut self) {
        if !self.waiting_for_key {
            self.waiting_for_key = true;
            self.keyframe_request = true;
        }
    }

    fn join(&self, objects: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
        match self.config.format {
            PayloadFormat::H264 | PayloadFormat::Hevc | PayloadFormat::Vvc => {
                vec![nal::join(&objects)]
            }
            PayloadFormat::Av1 => vec![av1::join(&objects)],
            PayloadFormat::Vp9 => objects,
        }
    }

    fn push_ready(
        &mut self,
        timestamp: u32,
        frames: Vec<Vec<u8>>,
        key: bool,
        complete: bool,
        spatial_ids: Vec<u8>,
        frame_ids: Vec<u64>,
    ) {
        self.stats.units += 1;
        self.ready.push_back(ReceivedUnit {
            timestamp,
            frames,
            key,
            complete,
            spatial_ids,
            frame_ids,
        });
    }
}

/// Concatenate packets' pieces into whole objects; a piece whose object
/// began or ends outside these packets is dropped with that object.
fn objects<'a>(packets: impl Iterator<Item = &'a Stored>) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut open: Option<Vec<u8>> = None;
    for packet in packets {
        for piece in &packet.parsed.pieces {
            let mut object = if piece.continues_previous {
                match open.take() {
                    Some(mut object) => {
                        object.extend_from_slice(&piece.data);
                        object
                    }
                    None => continue,
                }
            } else {
                piece.data.clone()
            };
            if piece.continues_next {
                open = Some(std::mem::take(&mut object));
            } else {
                out.push(object);
            }
        }
    }
    out
}

/// Split a unit into layer frames by the packets' frame tags; `None` when
/// any packet has no tag.
fn frame_runs(unit: &AssembledUnit) -> Option<Vec<FrameRun>> {
    let mut runs: Vec<FrameRun> = Vec::new();
    let mut previous_seq = None;
    for (index, (seq, packet)) in unit.packets.iter().enumerate() {
        let tag = packet.parsed.info.frame.clone()?;
        let contiguous = previous_seq.is_some_and(|p: u64| p + 1 == *seq);
        previous_seq = Some(*seq);
        match runs.last_mut() {
            Some(run) if run.tag.id == tag.id => {
                run.complete &= contiguous;
                run.tag.end = tag.end;
                run.packets.push(index);
            }
            _ => runs.push(FrameRun {
                // A frame whose first packet says it starts the frame is
                // whole at its start even after a gap.
                complete: tag.start,
                packets: vec![index],
                tag,
            }),
        }
    }
    for run in &mut runs {
        run.complete &= run.tag.end;
    }
    Some(runs)
}

/// True when the unit's packets say no later picture references it
/// (H.264 `nal_ref_idc == 0` on its slice), so losing it needs no key.
fn known_non_reference(unit: &AssembledUnit) -> bool {
    !unit.gap_before
        && unit
            .packets
            .iter()
            .filter_map(|(_, p)| p.parsed.info.reference)
            .fold(None, |acc: Option<bool>, r| Some(acc.unwrap_or(true) && !r))
            .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::LayerFilter;
    use crate::layering::{InterLayer, LayerFrame, LayerStructure};
    use crate::packetizer::{Packetizer, PacketizerConfig, UnitLayers};

    /// A low-overhead AV1 OBU (`kind`, optional `(temporal, spatial)`).
    fn obu(kind: u8, layer: Option<(u8, u8)>, len: usize) -> Vec<u8> {
        let mut out = vec![(kind << 3) | 0x02];
        if let Some((t, s)) = layer {
            out[0] |= 0x04;
            out.push((t << 5) | (s << 3));
        }
        out.extend(nc_common::leb128(len as u64));
        out.extend((0..len).map(|i| (i % 249) as u8 + 1));
        out
    }

    fn frame(spatial: u8, number: u64, references: &[u64], key: bool) -> LayerFrame {
        LayerFrame {
            spatial_id: spatial,
            temporal_id: 0,
            frame_number: number,
            references: references.to_vec(),
            key,
            inter_layer: spatial > 0,
            range: None,
        }
    }

    /// Two L2T1 temporal units (a key and an inter unit) as RTP packets.
    fn two_layer_stream() -> Vec<Vec<Vec<u8>>> {
        let structure = LayerStructure {
            spatial_sizes: vec![(320, 180), (640, 360)],
            temporal_layers: 1,
            inter_layer: InterLayer::Full,
        };
        let mut packetizer = Packetizer::new(PacketizerConfig::new(PayloadFormat::Av1)).unwrap();
        let units = [
            [frame(0, 0, &[], true), frame(1, 1, &[], true)],
            [frame(0, 2, &[0], false), frame(1, 3, &[1], false)],
        ];
        units
            .iter()
            .enumerate()
            .map(|(index, frames)| {
                let mut unit = obu(2, None, 0);
                if index == 0 {
                    unit.extend(obu(1, None, 12));
                }
                unit.extend(obu(6, Some((0, 0)), 2000));
                unit.extend(obu(6, Some((0, 1)), 3000));
                packetizer
                    .packetize(
                        &unit,
                        3000 * index as u32,
                        Some(UnitLayers {
                            structure: &structure,
                            frames,
                        }),
                    )
                    .unwrap()
            })
            .collect()
    }

    fn receive(packets: &[Vec<u8>]) -> Vec<ReceivedUnit> {
        let mut receiver = Depacketizer::new(DepacketizerConfig::new(PayloadFormat::Av1));
        for packet in packets {
            receiver.push(packet).unwrap();
        }
        receiver.flush();
        std::iter::from_fn(|| receiver.pop()).collect()
    }

    #[test]
    fn a_lost_enhancement_packet_leaves_the_base_layer_decodable() {
        let units = two_layer_stream();
        let mut packets: Vec<Vec<u8>> = units.concat();
        // The last packet of the second unit belongs to its enhancement frame.
        let lost = units[0].len() + units[1].len() - 1;
        packets.remove(lost);
        let received = receive(&packets);
        assert_eq!(received.len(), 2);
        assert!(received[0].complete && received[0].spatial_ids == vec![0, 1]);
        assert!(!received[1].complete && received[1].spatial_ids == vec![0]);
    }

    #[test]
    fn the_layer_filter_forwards_a_decodable_base_layer() {
        let packets: Vec<Vec<u8>> = two_layer_stream().concat();
        let mut filter = LayerFilter::new(1, 0, 0);
        let forwarded: Vec<Vec<u8>> = packets
            .iter()
            .filter_map(|p| filter.filter(p).unwrap())
            .collect();
        assert!(forwarded.len() < packets.len());
        let received = receive(&forwarded);
        assert_eq!(received.len(), 2);
        assert!(
            received
                .iter()
                .all(|u| u.complete && u.spatial_ids == vec![0])
        );
    }

    #[test]
    fn a_lost_reference_frame_stops_delivery_until_a_key() {
        let units = two_layer_stream();
        // The whole second unit's base frame is lost: its first packet.
        let mut packets: Vec<Vec<u8>> = units.concat();
        packets.remove(units[0].len());
        let mut receiver = Depacketizer::new(DepacketizerConfig::new(PayloadFormat::Av1));
        for packet in &packets {
            receiver.push(packet).unwrap();
        }
        receiver.flush();
        assert_eq!(receiver.pop().map(|u| u.spatial_ids), Some(vec![0, 1]));
        assert_eq!(
            receiver.pop(),
            None,
            "neither frame of the second unit is decodable"
        );
        assert!(receiver.take_keyframe_request());
        assert_eq!(receiver.take_lost_frames(), vec![65538, 65539]);
    }
}
