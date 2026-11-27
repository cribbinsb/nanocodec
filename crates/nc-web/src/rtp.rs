//! RTP in the browser (`nanocodec-rtp`, docs/handoff/rtp.md): `RtpSender`
//! packetizes a `BrowserEncoder`'s units with the encoder's layer
//! information; `RtpReceiver` takes RTP payloads (from the demo's network
//! emulator, a WebRTC `RTCRtpScriptTransform`, a WebTransport or WebSocket
//! relay) and hands the units that come out to `BrowserDecoder`, with the
//! frame numbers loss feedback names; `RtpLayerFilter` drops layers as a
//! selective forwarding server does, by the Dependency Descriptor.

use nc_encoder::Codec;
use nc_rtp::dependency_descriptor::{DependencyDescriptor, TemplateStructure};
use nc_rtp::{
    Depacketizer, DepacketizerConfig, LayerFilter, Packetizer, PacketizerConfig, PayloadFormat,
    RtpPacket, UnitLayers, rtp_timestamp,
};
use wasm_bindgen::prelude::*;

use crate::options::parse_codec;
use crate::{BrowserEncoder, js_error};

/// The RTP payload format of `codec` (AV2 has none yet).
pub(crate) fn payload_format(codec: Codec) -> Option<PayloadFormat> {
    match codec {
        Codec::H264 => Some(PayloadFormat::H264),
        Codec::Hevc => Some(PayloadFormat::Hevc),
        Codec::Vvc => Some(PayloadFormat::Vvc),
        Codec::Av1 => Some(PayloadFormat::Av1),
        Codec::Vp9 => Some(PayloadFormat::Vp9),
        Codec::Av2 => None,
    }
}

/// The payload format of the codec named `codec`.
fn format_of(codec: &str) -> Result<PayloadFormat, JsValue> {
    parse_codec(codec)
        .and_then(payload_format)
        .ok_or_else(|| js_error("No RTP payload format for this codec"))
}

/// A negotiated RFC 8285 extension id from JS (0: none).
fn extension_id(id: u8) -> Option<u8> {
    (id != 0).then_some(id)
}

/// Frame numbers as the Dependency Descriptor carries them (16 bits; the
/// depacketizer's unwrapped numbers start one wrap up), for JS.
fn numbers(ids: &[u64]) -> Vec<f64> {
    ids.iter()
        .map(|&id| f64::from(crate::wire_frame_number(id)))
        .collect()
}

/// Turns a `BrowserEncoder`'s units into RTP packets.
#[wasm_bindgen]
#[derive(Debug)]
pub struct RtpSender {
    packetizer: Packetizer,
    /// The packets of the latest `packetize`.
    packets: Vec<Vec<u8>>,
    /// `(spatial id, temporal id)` of each of those packets' frame, read
    /// back from its Dependency Descriptor (`(0, 0)` without one), so a
    /// network model can treat layers differently.
    layers: Vec<(u8, u8)>,
    /// The descriptor's template structure, as a receiver tracks it.
    structure: Option<TemplateStructure>,
}

#[wasm_bindgen]
impl RtpSender {
    /// A sender for `codec` ("h264", "hevc", "vvc", "av1", "vp9"), with
    /// RTP packets of at most `mtu` bytes. `dependency_descriptor_id`: the
    /// extension id of the Dependency Descriptor (0: none; any format may
    /// carry it, which gives every codec frame numbers and dependencies).
    /// # Errors
    /// Other codec names, an extension id outside 1-14, a tiny MTU.
    #[wasm_bindgen(constructor)]
    pub fn new(codec: &str, mtu: usize, dependency_descriptor_id: u8) -> Result<Self, JsValue> {
        let packetizer = Packetizer::new(PacketizerConfig {
            mtu,
            dependency_descriptor_id: extension_id(dependency_descriptor_id),
            ..PacketizerConfig::new(format_of(codec)?)
        })
        .map_err(js_error)?;
        Ok(Self {
            packetizer,
            packets: Vec::new(),
            layers: Vec::new(),
            structure: None,
        })
    }

    /// Packetize `unit`, the packet `encoder` just returned for a frame
    /// captured at `timestamp_us` (microseconds), with the encoder's layer
    /// information; returns the packet count (0 for a held or dropped
    /// frame), read with [`Self::packet`].
    /// # Errors
    /// A unit that does not parse as its format.
    pub fn packetize(
        &mut self,
        encoder: &BrowserEncoder,
        unit: &[u8],
        timestamp_us: f64,
    ) -> Result<usize, JsValue> {
        let (structure, frames) = encoder.rtp_layers();
        // Microsecond capture times are integers well inside f64's exact range.
        #[allow(clippy::cast_possible_truncation, reason = "integral microseconds")]
        let timestamp = rtp_timestamp(timestamp_us.round() as i64);
        self.packets = self
            .packetizer
            .packetize(
                unit,
                timestamp,
                Some(UnitLayers {
                    structure: &structure,
                    frames,
                }),
            )
            .map_err(js_error)?;
        let packets = std::mem::take(&mut self.packets);
        self.layers = packets.iter().map(|p| self.layer_of(p)).collect();
        self.packets = packets;
        Ok(self.packets.len())
    }

    /// Packet `index` of the latest [`Self::packetize`] (empty past the end).
    #[must_use]
    pub fn packet(&self, index: usize) -> Vec<u8> {
        self.packets.get(index).cloned().unwrap_or_default()
    }

    /// Spatial id of packet `index`'s frame.
    #[must_use]
    pub fn packet_spatial_id(&self, index: usize) -> u8 {
        self.layers.get(index).map_or(0, |&(spatial, _)| spatial)
    }

    /// Temporal id of packet `index`'s frame.
    #[must_use]
    pub fn packet_temporal_id(&self, index: usize) -> u8 {
        self.layers.get(index).map_or(0, |&(_, temporal)| temporal)
    }
}

impl RtpSender {
    /// The layer of `packet`'s frame, from its Dependency Descriptor.
    fn layer_of(&mut self, packet: &[u8]) -> (u8, u8) {
        let Some(id) = self.packetizer.config().dependency_descriptor_id else {
            return (0, 0);
        };
        let descriptor = RtpPacket::parse(packet).ok().and_then(|parsed| {
            DependencyDescriptor::parse(parsed.extension(id)?, self.structure.as_ref()).ok()
        });
        let Some(descriptor) = descriptor else {
            return (0, 0);
        };
        if let Some(structure) = &descriptor.structure {
            self.structure = Some(structure.clone());
        }
        self.structure
            .as_ref()
            .and_then(|structure| descriptor.resolve(structure).ok())
            .map_or((0, 0), |frame| (frame.spatial_id, frame.temporal_id))
    }
}

/// Turns RTP packets into units for `BrowserDecoder::decode`.
#[wasm_bindgen]
#[derive(Debug)]
pub struct RtpReceiver {
    depacketizer: Depacketizer,
    timestamp: u32,
    key: bool,
    frame_ids: Vec<u64>,
}

#[wasm_bindgen]
impl RtpReceiver {
    /// A receiver for `codec` ("h264", "hevc", "vvc", "av1", "vp9").
    /// `dependency_descriptor_id`: the negotiated extension id of the
    /// Dependency Descriptor (0: none). `deliver_after_loss`: keep
    /// delivering after a loss instead of waiting for a key frame.
    /// # Errors
    /// Rejects other codec names.
    #[wasm_bindgen(constructor)]
    pub fn new(
        codec: &str,
        dependency_descriptor_id: u8,
        deliver_after_loss: bool,
    ) -> Result<Self, JsValue> {
        Ok(Self {
            depacketizer: Depacketizer::new(DepacketizerConfig {
                dependency_descriptor_id: extension_id(dependency_descriptor_id),
                deliver_after_loss,
                ..DepacketizerConfig::new(format_of(codec)?)
            }),
            timestamp: 0,
            key: false,
            frame_ids: Vec::new(),
        })
    }

    /// Take one RTP packet (header included), in any order.
    /// # Errors
    /// A malformed packet (it is dropped; the stream goes on).
    pub fn push(&mut self, packet: &[u8]) -> Result<(), JsValue> {
        self.depacketizer.push(packet).map_err(js_error)
    }

    /// Playout deadline (the page's jitter buffer): packets of RTP
    /// timestamp `timestamp` or older are due, and a gap before one of
    /// them is loss now.
    pub fn set_playout_deadline(&mut self, timestamp: u32) {
        self.depacketizer.set_playout_deadline(timestamp);
    }

    /// End of stream: release what is held for reordering.
    pub fn flush(&mut self) {
        self.depacketizer.flush();
    }

    /// The next unit for `BrowserDecoder::decode`, if one is ready.
    pub fn pop(&mut self) -> Option<Vec<u8>> {
        let unit = self.depacketizer.pop()?;
        self.timestamp = unit.timestamp;
        self.key = unit.key;
        self.frame_ids = unit.frame_ids;
        Some(match unit.frames.as_slice() {
            [single] => single.clone(),
            frames => {
                nc_vp9::pack_superframe(&frames.iter().map(Vec::as_slice).collect::<Vec<_>>())
            }
        })
    }

    /// RTP timestamp of the last popped unit.
    #[must_use]
    pub fn timestamp(&self) -> u32 {
        self.timestamp
    }

    /// Whether the last popped unit is a key unit.
    #[must_use]
    pub fn key(&self) -> bool {
        self.key
    }

    /// The frame numbers of the last popped unit's layer frames, when its
    /// packets carry them (the Dependency Descriptor's 16-bit numbers): the
    /// encoder's frame numbers modulo 2^16, which a receiver acknowledges
    /// (`BrowserEncoder::acknowledge_frame`).
    #[must_use]
    pub fn frame_ids(&self) -> Vec<f64> {
        numbers(&self.frame_ids)
    }

    /// Frames found lost or undecodable since the last call, by frame
    /// number, for loss reports to the encoder.
    pub fn lost_frames(&mut self) -> Vec<f64> {
        numbers(&self.depacketizer.take_lost_frames())
    }

    /// True once after decoding stopped for want of a key frame: ask the
    /// sender for one (a PLI or FIR).
    pub fn keyframe_needed(&mut self) -> bool {
        self.depacketizer.take_keyframe_request()
    }

    /// Packets accepted so far.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn packets(&self) -> f64 {
        count(self.depacketizer.stats().packets)
    }

    /// Packets declared lost (their gap outlived the reorder window).
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn lost_packets(&self) -> f64 {
        count(self.depacketizer.stats().lost_packets)
    }

    /// Packets that arrived after their place was given up.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn late_packets(&self) -> f64 {
        count(self.depacketizer.stats().late_packets)
    }

    /// Units (or their layer frames) not delivered.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn dropped_units(&self) -> f64 {
        count(self.depacketizer.stats().dropped_units)
    }
}

/// A counter for JS.
#[allow(clippy::cast_precision_loss, reason = "counters stay below 2^53")]
fn count(value: u64) -> f64 {
    value as f64
}

/// Forwards the packets of one decode target, as a selective forwarding
/// server does (by the Dependency Descriptor alone).
#[wasm_bindgen]
#[derive(Debug)]
pub struct RtpLayerFilter {
    filter: LayerFilter,
}

#[wasm_bindgen]
impl RtpLayerFilter {
    /// Forward spatial layers up to `spatial` and temporal layers up to
    /// `temporal`, reading the descriptor at `dependency_descriptor_id`.
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new(dependency_descriptor_id: u8, spatial: u8, temporal: u8) -> Self {
        Self {
            filter: LayerFilter::new(dependency_descriptor_id, spatial, temporal),
        }
    }

    /// The packet to forward (renumbered), or `None` to drop it.
    /// # Errors
    /// A packet that does not parse.
    pub fn filter(&mut self, packet: &[u8]) -> Result<Option<Vec<u8>>, JsValue> {
        self.filter.filter(packet).map_err(js_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SourceFrame;

    #[test]
    fn packets_in_any_order_give_back_the_access_unit() {
        let mut unit = vec![0, 0, 0, 1, 0x65];
        unit.extend(std::iter::repeat_n(0x21u8, 3000));
        let packets = Packetizer::new(PacketizerConfig::new(PayloadFormat::H264))
            .unwrap()
            .packetize(&unit, 90, None)
            .unwrap();
        let mut receiver = RtpReceiver::new("h264", 0, false).unwrap();
        for packet in packets.iter().rev() {
            receiver.push(packet).unwrap();
        }
        receiver.flush();
        assert_eq!(receiver.pop(), Some(unit));
        assert!(receiver.key() && receiver.timestamp() == 90);
    }

    /// A small moving picture for `frame`.
    fn source(frame: usize) -> SourceFrame {
        let (width, height) = (64, 64);
        let mut source = SourceFrame::new(width, height).unwrap();
        let rgba: Vec<u8> = (0..width * height)
            .flat_map(|i| {
                let v = ((i % width + frame * 3) * 4 + i / width) as u8;
                [v, v.wrapping_mul(3), 255 - v, 255]
            })
            .collect();
        source.from_rgba(&rgba).unwrap();
        source
    }

    /// Every codec with a payload format: encoder -> sender -> receiver
    /// gives back the encoder's units, with the encoder's frame numbers;
    /// a lost inter frame is reported by number and its successor is not
    /// delivered.
    #[test]
    fn the_sender_and_receiver_carry_the_encoders_frames() {
        const DESCRIPTOR_ID: u8 = 3;
        for codec in ["h264", "hevc", "vvc", "av1", "vp9"] {
            let mut encoder = BrowserEncoder::new(
                codec, 64, 64, 30, 30, 30, "medium", None, false, None, None, None,
            )
            .unwrap();
            let mut sender = RtpSender::new(codec, 1200, DESCRIPTOR_ID).unwrap();
            let mut receiver = RtpReceiver::new(codec, DESCRIPTOR_ID, false).unwrap();
            let mut units = Vec::new();
            for frame in 0..4 {
                let unit = encoder.encode(&source(frame)).unwrap();
                let count = sender
                    .packetize(&encoder, &unit, frame as f64 * 33_333.0)
                    .unwrap();
                // Frame 2 is lost: frame 3 predicts from it.
                if frame != 2 {
                    for index in 0..count {
                        receiver.push(&sender.packet(index)).unwrap();
                    }
                }
                units.push(unit);
            }
            receiver.flush();
            let mut delivered = Vec::new();
            while let Some(unit) = receiver.pop() {
                delivered.push((unit, receiver.frame_ids()));
            }
            let summary = |list: &[(Vec<u8>, Vec<f64>)]| {
                list.iter()
                    .map(|(unit, ids)| (unit.len(), ids.clone()))
                    .collect::<Vec<_>>()
            };
            let expected = [(units[0].clone(), vec![0.0]), (units[1].clone(), vec![1.0])];
            assert_eq!(summary(&delivered), summary(&expected), "{codec}");
            assert!(delivered == expected, "{codec}: the units differ");
            assert_eq!(receiver.lost_frames(), [3.0], "{codec}");
            assert!(receiver.keyframe_needed(), "{codec}");
            // The loss report reaches the encoder: without a long-term
            // reference its repair is a key frame.
            encoder.report_lost_frame(3.0).unwrap();
            encoder.encode(&source(4)).unwrap();
            assert!(encoder.key(), "{codec}");
        }
    }

    /// Each packet's layer is read back from its descriptor: an L1T2
    /// stream alternates base and top-layer frames.
    #[test]
    fn packets_name_their_temporal_layer() {
        let mut encoder = BrowserEncoder::new(
            "h264",
            64,
            64,
            30,
            30,
            30,
            "medium",
            Some("layers.count=2\nlayers.adaptive=0".into()),
            false,
            None,
            None,
            None,
        )
        .unwrap();
        let mut sender = RtpSender::new("h264", 1200, 1).unwrap();
        let mut layers = Vec::new();
        for frame in 0..4 {
            let unit = encoder.encode(&source(frame)).unwrap();
            let count = sender
                .packetize(&encoder, &unit, frame as f64 * 33_333.0)
                .unwrap();
            layers.push((
                encoder.temporal_layer(),
                sender.packet_temporal_id(count - 1),
            ));
        }
        assert_eq!(layers, [(0, 0), (1, 1), (0, 0), (1, 1)]);
    }
}
