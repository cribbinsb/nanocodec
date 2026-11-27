//! The sender: one coded unit (access unit, temporal unit, VP9 picture) in,
//! RTP packets out, with sequence numbers, timestamp, marker bit and, for
//! layered streams, the Dependency Descriptor.
//!
//! The payload format modules cut the unit into payloads; this module adds
//! what every format shares (RFC 3550 §5.1 header fields, RFC 8285
//! extensions, the MTU budget) and the per-stream state the descriptors
//! need (`layering::DescriptorWriter`, the VP9 picture counters).

use std::collections::VecDeque;

use crate::av1;
use crate::error::{Result, RtpError};
use crate::layering::{DescriptorWriter, LayerFrame, LayerStructure};
use crate::nal::{self, h264::H264, hevc::Hevc, vvc::Vvc};
use crate::packet::{self, Extension, FIXED_HEADER_BYTES, MAX_PAYLOAD_TYPE, RtpHeader};
use crate::payload::OutPayload;
use crate::vp9::{self, Descriptor, FrameToSend, Vp9Mode};

/// The RTP clock of every video payload format here.
pub const CLOCK_RATE: u32 = 90_000;
/// Microseconds per second, for timestamp conversion.
const MICROS_PER_SECOND: i64 = 1_000_000;
/// Default packet size: below every common path MTU with room for SRTP and
/// TURN overheads (libwebrtc uses 1200 for the same reason).
pub const DEFAULT_MTU: usize = 1200;
/// The first dynamic payload type (RFC 3551 §3).
pub const DEFAULT_PAYLOAD_TYPE: u8 = 96;
/// RFC 8285 one-byte extension ids are 1-14.
const MAX_EXTENSION_ID: u8 = 14;
/// Pictures whose frame numbers the VP9 descriptor remembers for `P_DIFF`s.
const VP9_PICTURE_HISTORY: usize = 256;

/// The payload formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayloadFormat {
    /// H.264, RFC 6184.
    H264,
    /// HEVC, RFC 7798.
    Hevc,
    /// VVC, RFC 9328.
    Vvc,
    /// AV1, the AOM RTP payload format.
    Av1,
    /// VP9, RFC 9628.
    Vp9,
}

impl PayloadFormat {
    /// The SDP `a=rtpmap` encoding name.
    #[must_use]
    pub const fn encoding_name(self) -> &'static str {
        match self {
            Self::H264 => "H264",
            Self::Hevc => "H265",
            Self::Vvc => "H266",
            Self::Av1 => "AV1",
            Self::Vp9 => "VP9",
        }
    }
}

/// Packetizer settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PacketizerConfig {
    /// The payload format.
    pub format: PayloadFormat,
    /// The negotiated dynamic payload type (96-127 by convention).
    pub payload_type: u8,
    /// The stream's SSRC.
    pub ssrc: u32,
    /// Largest RTP packet (header, extensions and payload) in bytes.
    pub mtu: usize,
    /// Sequence number of the first packet.
    pub first_sequence: u16,
    /// RFC 8285 id of the Dependency Descriptor, if negotiated; it is
    /// written for units that come with layer information.
    pub dependency_descriptor_id: Option<u8>,
    /// VP9 descriptor mode.
    pub vp9_mode: Vp9Mode,
}

impl PacketizerConfig {
    /// Defaults for `format`: payload type 96, MTU 1200, and for AV1 the
    /// Dependency Descriptor as extension id 1.
    #[must_use]
    pub fn new(format: PayloadFormat) -> Self {
        Self {
            format,
            payload_type: DEFAULT_PAYLOAD_TYPE,
            ssrc: 0x6E63_0001,
            mtu: DEFAULT_MTU,
            first_sequence: 0,
            dependency_descriptor_id: (format == PayloadFormat::Av1).then_some(1),
            vp9_mode: Vp9Mode::Flexible,
        }
    }
}

/// The layer information of one unit, from the encoder.
#[derive(Debug, Clone, Copy)]
pub struct UnitLayers<'a> {
    /// The stream's layer structure.
    pub structure: &'a LayerStructure,
    /// The unit's layer frames, in decode order.
    pub frames: &'a [LayerFrame],
}

/// VP9 picture counters.
#[derive(Debug, Clone, Default)]
struct Vp9State {
    pictures: u64,
    tl0_picture_index: u8,
    /// `(frame_number, picture)` of recent layer frames.
    picture_of_frame: VecDeque<(u64, u64)>,
}

/// Turns coded units into RTP packets.
#[derive(Debug, Clone)]
pub struct Packetizer {
    config: PacketizerConfig,
    sequence: u16,
    descriptors: DescriptorWriter,
    vp9: Vp9State,
}

/// The 90 kHz RTP timestamp of a capture time in microseconds (wrapping).
#[must_use]
pub fn rtp_timestamp(timestamp_us: i64) -> u32 {
    (i128::from(timestamp_us) * i128::from(CLOCK_RATE) / i128::from(MICROS_PER_SECOND)) as u32
}

impl Packetizer {
    /// A packetizer for one RTP stream.
    ///
    /// # Errors
    /// A payload type over 127, an extension id outside 1-14, or an MTU too
    /// small for a header and one payload byte.
    pub fn new(config: PacketizerConfig) -> Result<Self> {
        if config.payload_type > MAX_PAYLOAD_TYPE {
            return Err(RtpError::Config(format!(
                "payload type {}",
                config.payload_type
            )));
        }
        if config
            .dependency_descriptor_id
            .is_some_and(|id| id == 0 || id > MAX_EXTENSION_ID)
        {
            return Err(RtpError::Config("extension id outside 1-14".into()));
        }
        if config.mtu <= FIXED_HEADER_BYTES + 16 {
            return Err(RtpError::Config(format!("MTU {} too small", config.mtu)));
        }
        Ok(Self {
            sequence: config.first_sequence,
            config,
            descriptors: DescriptorWriter::new(),
            vp9: Vp9State::default(),
        })
    }

    /// The settings.
    #[must_use]
    pub fn config(&self) -> &PacketizerConfig {
        &self.config
    }

    /// Sequence number the next packet will take.
    #[must_use]
    pub fn next_sequence(&self) -> u16 {
        self.sequence
    }

    /// Packetize one unit captured at RTP `timestamp`. `layers` describes
    /// its layer frames (needed for the Dependency Descriptor and VP9 layer
    /// indices; without it the unit is one frame and no descriptor is sent).
    ///
    /// # Errors
    /// A unit that does not parse as its format, or layer information that
    /// does not fit it.
    pub fn packetize(
        &mut self,
        unit: &[u8],
        timestamp: u32,
        layers: Option<UnitLayers<'_>>,
    ) -> Result<Vec<Vec<u8>>> {
        if unit.is_empty() {
            return Ok(Vec::new());
        }
        let descriptors = match (self.config.dependency_descriptor_id, layers) {
            (Some(_), Some(l)) => Some(self.descriptors.describe_unit(l.structure, l.frames)?),
            _ => None,
        };
        let structure = self.descriptors.structure().cloned();
        let encoded: Option<Vec<(Vec<u8>, Vec<u8>)>> = match (&descriptors, &structure) {
            (Some(list), Some(structure)) => Some(
                list.iter()
                    .map(|d| Ok((d.first.write(structure)?, d.rest.write(structure)?)))
                    .collect::<Result<_>>()?,
            ),
            _ => None,
        };
        let largest_extension = encoded.as_ref().map_or(0, |list| {
            list.iter()
                .map(|(first, _)| packet::extension_overhead(&[Extension { id: 1, data: first }]))
                .max()
                .unwrap_or(0)
        });
        let budget = self.config.mtu - FIXED_HEADER_BYTES - largest_extension;
        let payloads = self.payloads(unit, budget, layers)?;
        let count = payloads.len();
        let mut packets = Vec::with_capacity(count);
        for (index, payload) in payloads.into_iter().enumerate() {
            let header = RtpHeader {
                marker: index + 1 == count,
                payload_type: self.config.payload_type,
                sequence: self.sequence,
                timestamp,
                ssrc: self.config.ssrc,
            };
            self.sequence = self.sequence.wrapping_add(1);
            let descriptor_bytes = match (&descriptors, &structure) {
                (Some(list), Some(structure)) => {
                    let pair = list.get(payload.frame).ok_or_else(|| {
                        RtpError::Config("payload of an undescribed frame".into())
                    })?;
                    let mut descriptor = if payload.start_of_frame {
                        pair.first.clone()
                    } else {
                        pair.rest.clone()
                    };
                    descriptor.start_of_frame = payload.start_of_frame;
                    descriptor.end_of_frame = payload.end_of_frame;
                    Some(descriptor.write(structure)?)
                }
                _ => None,
            };
            let extensions: Vec<Extension<'_>> = descriptor_bytes
                .as_deref()
                .zip(self.config.dependency_descriptor_id)
                .map(|(data, id)| Extension { id, data })
                .into_iter()
                .collect();
            packets.push(packet::write_packet(&header, &extensions, &payload.bytes)?);
        }
        Ok(packets)
    }

    /// The payloads of one unit.
    fn payloads(
        &mut self,
        unit: &[u8],
        budget: usize,
        layers: Option<UnitLayers<'_>>,
    ) -> Result<Vec<OutPayload>> {
        match self.config.format {
            PayloadFormat::H264 => nal::packetize::<H264>(unit, budget),
            PayloadFormat::Hevc => nal::packetize::<Hevc>(unit, budget),
            PayloadFormat::Vvc => nal::packetize::<Vvc>(unit, budget),
            PayloadFormat::Av1 => {
                let key = layers.map_or_else(
                    || av1::has_sequence_header(unit),
                    |l| l.frames.first().is_some_and(|f| f.key && f.spatial_id == 0),
                );
                av1::packetize(unit, budget, key, |spatial| {
                    layers.and_then(|l| l.frames.iter().position(|f| f.spatial_id == spatial))
                })
            }
            PayloadFormat::Vp9 => {
                let frames = self.vp9_frames(unit, layers)?;
                vp9::packetize(&frames, budget)
            }
        }
    }

    /// The VP9 frames of one picture and their descriptors.
    fn vp9_frames<'u>(
        &mut self,
        unit: &'u [u8],
        layers: Option<UnitLayers<'_>>,
    ) -> Result<Vec<FrameToSend<'u>>> {
        let flexible = self.config.vp9_mode == Vp9Mode::Flexible;
        let picture = self.vp9.pictures;
        self.vp9.pictures += 1;
        let Some(layers) = layers else {
            // No layer information: the unit is one frame, described by its
            // own header only.
            return Ok(vec![FrameToSend {
                data: unit,
                index: 0,
                descriptor: Descriptor {
                    picture_id: Some(vp9::picture_id(picture)),
                    inter_picture: !vp9::is_key_frame(unit),
                    start: true,
                    ..Descriptor::default()
                },
            }]);
        };
        let base_tid0 = layers.frames.first().is_some_and(|f| f.temporal_id == 0);
        if base_tid0 {
            self.vp9.tl0_picture_index = self.vp9.tl0_picture_index.wrapping_add(1);
        }
        let mut frames = Vec::with_capacity(layers.frames.len());
        for (index, frame) in layers.frames.iter().enumerate() {
            let referenced: Vec<u64> = frame
                .references
                .iter()
                .filter_map(|r| {
                    self.vp9
                        .picture_of_frame
                        .iter()
                        .find(|&&(number, _)| number == *r)
                        .map(|&(_, p)| p)
                })
                .collect();
            let p_diffs = vp9::picture_diffs(picture, &referenced);
            // A frame whose references are all out of reach is sent as not
            // inter-predicted (docs/handoff/rtp.md "Sender rules").
            let inter_picture = !frame.key && (!flexible || !p_diffs.is_empty());
            let data = match &frame.range {
                Some(range) => unit.get(range.clone()).ok_or(RtpError::Config(
                    "layer frame range outside the unit".into(),
                ))?,
                None => unit,
            };
            let key_picture = frame.key && frame.spatial_id == 0;
            // `Z`: no frame of the layer above in this picture predicts
            // from this one (RFC 9628 §4.2), so a receiver of that layer
            // alone may go without it here.
            let not_upper_reference = !layers
                .frames
                .iter()
                .any(|f| f.spatial_id == frame.spatial_id + 1 && f.inter_layer);
            frames.push(FrameToSend {
                data,
                index,
                descriptor: Descriptor {
                    picture_id: Some(vp9::picture_id(picture)),
                    inter_picture,
                    layer: Some((
                        frame.temporal_id,
                        false,
                        frame.spatial_id,
                        frame.inter_layer,
                    )),
                    flexible,
                    tl0_picture_index: (!flexible).then_some(self.vp9.tl0_picture_index),
                    p_diffs: if inter_picture { p_diffs } else { Vec::new() },
                    scalability: key_picture.then(|| layers.structure.spatial_sizes.clone()),
                    not_upper_reference,
                    ..Descriptor::default()
                },
            });
            self.vp9
                .picture_of_frame
                .push_back((frame.frame_number, picture));
            if self.vp9.picture_of_frame.len() > VP9_PICTURE_HISTORY {
                self.vp9.picture_of_frame.pop_front();
            }
        }
        Ok(frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packet::RtpPacket;

    #[test]
    fn timestamps_are_90_khz() {
        assert_eq!(rtp_timestamp(1_000_000), 90_000);
        assert_eq!(rtp_timestamp(33_333), 2999);
    }

    #[test]
    fn h264_unit_gets_sequential_packets_and_one_marker() {
        let mut unit = vec![0, 0, 0, 1, 0x65];
        unit.extend(std::iter::repeat_n(7u8, 3000));
        let mut p = Packetizer::new(PacketizerConfig {
            first_sequence: 65535,
            ..PacketizerConfig::new(PayloadFormat::H264)
        })
        .unwrap();
        let packets = p.packetize(&unit, 1234, None).unwrap();
        assert_eq!(packets.len(), 3);
        let parsed: Vec<_> = packets
            .iter()
            .map(|b| RtpPacket::parse(b).unwrap())
            .collect();
        assert_eq!(
            parsed.iter().map(|p| p.header.sequence).collect::<Vec<_>>(),
            vec![65535, 0, 1]
        );
        assert_eq!(
            parsed.iter().map(|p| p.header.marker).collect::<Vec<_>>(),
            vec![false, false, true]
        );
        assert!(packets.iter().all(|b| b.len() <= DEFAULT_MTU));
    }

    #[test]
    fn bad_configs_are_refused() {
        let mut config = PacketizerConfig::new(PayloadFormat::Av1);
        config.dependency_descriptor_id = Some(15);
        assert!(Packetizer::new(config).is_err());
        let mut config = PacketizerConfig::new(PayloadFormat::Vp9);
        config.mtu = 20;
        assert!(Packetizer::new(config).is_err());
    }
}
