//! RTP behind the C ABI (safe Rust): a sender that packetizes an encoder
//! session's latest packet with the layer information only the session has
//! (the AV1 Dependency Descriptor, VP9 layer indices), and a receiver that
//! turns RTP packets back into units for a decoder (`nanocodec-rtp`,
//! docs/handoff/rtp.md).

#![forbid(unsafe_code)]

use nc_encoder::Codec;
use nc_rtp::{
    Depacketizer, DepacketizerConfig, Packetizer, PacketizerConfig, PayloadFormat, RtpError,
    UnitLayers, rtp_timestamp,
};

use crate::session::Session;

/// The RTP payload format of a C API codec.
fn payload_format(codec: Codec) -> Option<PayloadFormat> {
    match codec {
        Codec::H264 => Some(PayloadFormat::H264),
        Codec::Hevc => Some(PayloadFormat::Hevc),
        Codec::Av1 => Some(PayloadFormat::Av1),
        Codec::Vp9 => Some(PayloadFormat::Vp9),
        Codec::Vvc | Codec::Av2 => None,
    }
}

/// Sender settings (the C `nc_rtp_sender_config`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SenderConfig {
    /// The stream's codec.
    pub codec: Codec,
    /// RTP payload type.
    pub payload_type: u8,
    /// RTP SSRC.
    pub ssrc: u32,
    /// Largest packet in bytes; 0 = 1200.
    pub mtu: usize,
    /// Dependency Descriptor extension id (1-14); `None` = not sent.
    pub dependency_descriptor_id: Option<u8>,
}

/// Packetizes a session's packets.
#[derive(Debug)]
pub struct Sender {
    packetizer: Packetizer,
    packets: Vec<Vec<u8>>,
}

impl Sender {
    /// A sender.
    ///
    /// # Errors
    /// A codec without a payload format here, or settings out of range.
    pub fn new(config: SenderConfig) -> Result<Self, RtpError> {
        let format = payload_format(config.codec)
            .ok_or(RtpError::Unsupported("RTP for this codec in the C API"))?;
        let mut rtp = PacketizerConfig::new(format);
        rtp.payload_type = config.payload_type;
        rtp.ssrc = config.ssrc;
        if config.mtu > 0 {
            rtp.mtu = config.mtu;
        }
        rtp.dependency_descriptor_id = config.dependency_descriptor_id;
        Ok(Self {
            packetizer: Packetizer::new(rtp)?,
            packets: Vec::new(),
        })
    }

    /// Packetize the session's latest packet; returns the packet count
    /// (0 for a dropped or held frame).
    ///
    /// # Errors
    /// Layer information that does not fit the payload format.
    pub fn packetize(&mut self, session: &Session) -> Result<usize, RtpError> {
        let (structure, frames) = session.rtp_layers();
        let coded = session.packet();
        self.packets = if frames.is_empty() {
            Vec::new()
        } else {
            self.packetizer.packetize(
                &coded.data,
                rtp_timestamp(coded.metadata.timestamp_us),
                Some(UnitLayers {
                    structure: &structure,
                    frames: &frames,
                }),
            )?
        };
        Ok(self.packets.len())
    }

    /// Packet `index` of the latest [`Sender::packetize`].
    #[must_use]
    pub fn packet(&self, index: usize) -> Option<&[u8]> {
        self.packets.get(index).map(Vec::as_slice)
    }
}

/// A unit ready for a decoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadyUnit {
    /// Decoder input (VP9 layer frames packed into one superframe).
    pub data: Vec<u8>,
    /// RTP timestamp.
    pub timestamp: u32,
    /// Decoding can start here.
    pub keyframe: bool,
}

/// Turns RTP packets into decoder units.
#[derive(Debug)]
pub struct Receiver {
    depacketizer: Depacketizer,
    current: Option<ReadyUnit>,
}

impl Receiver {
    /// A receiver.
    ///
    /// # Errors
    /// A codec without a payload format here.
    pub fn new(
        codec: Codec,
        dependency_descriptor_id: Option<u8>,
        deliver_after_loss: bool,
    ) -> Result<Self, RtpError> {
        let format = payload_format(codec)
            .ok_or(RtpError::Unsupported("RTP for this codec in the C API"))?;
        Ok(Self {
            depacketizer: Depacketizer::new(DepacketizerConfig {
                dependency_descriptor_id,
                deliver_after_loss,
                ..DepacketizerConfig::new(format)
            }),
            current: None,
        })
    }

    /// Take one datagram.
    ///
    /// # Errors
    /// A packet that does not parse (dropped; the stream goes on).
    pub fn push(&mut self, datagram: &[u8]) -> Result<(), RtpError> {
        self.depacketizer.push(datagram)
    }

    /// End of stream: release what is held.
    pub fn flush(&mut self) {
        self.depacketizer.flush();
    }

    /// The next unit (borrowed until the next call), if one is ready.
    pub fn pop(&mut self) -> Option<&ReadyUnit> {
        let unit = self.depacketizer.pop()?;
        let data = match unit.frames.as_slice() {
            [single] => single.clone(),
            frames => {
                nc_vp9::pack_superframe(&frames.iter().map(Vec::as_slice).collect::<Vec<_>>())
            }
        };
        self.current = Some(ReadyUnit {
            data,
            timestamp: unit.timestamp,
            keyframe: unit.key,
        });
        self.current.as_ref()
    }

    /// True once after decoding stopped for want of a key frame: send a
    /// PLI/FIR, or call `nc_encoder_request_keyframe` on a local encoder.
    pub fn take_keyframe_request(&mut self) -> bool {
        self.depacketizer.take_keyframe_request()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::SessionConfig;
    use nc_common::RawFrame;

    const WIDTH: usize = 64;
    const HEIGHT: usize = 48;

    #[test]
    fn session_packets_round_trip_for_every_codec() {
        for codec in [Codec::H264, Codec::Hevc, Codec::Av1, Codec::Vp9] {
            let mut session = Session::new(&SessionConfig::new(codec, WIDTH, HEIGHT)).unwrap();
            let mut sender = Sender::new(SenderConfig {
                codec,
                payload_type: 96,
                ssrc: 1,
                mtu: 200,
                dependency_descriptor_id: (codec == Codec::Av1).then_some(1),
            })
            .unwrap();
            let mut receiver =
                Receiver::new(codec, (codec == Codec::Av1).then_some(1), false).unwrap();
            let mut sent = Vec::new();
            for index in 0..4u8 {
                let source: Vec<u8> = (0..WIDTH * HEIGHT * 3 / 2)
                    .map(|i| (i % 251) as u8 ^ (index * 9))
                    .collect();
                let frame = RawFrame::packed_i420(&source, WIDTH, HEIGHT).unwrap();
                sent.push(session.encode(&frame, None).unwrap().data.clone());
                let count = sender.packetize(&session).unwrap();
                assert!(count >= 1, "{codec:?}");
                for i in (0..count).rev() {
                    receiver.push(sender.packet(i).unwrap()).unwrap();
                }
            }
            receiver.flush();
            for (index, coded) in sent.iter().enumerate() {
                let unit = receiver.pop().expect("a whole unit");
                assert_eq!(unit.keyframe, index == 0, "{codec:?}");
                if codec != Codec::Av1 {
                    // AV1 comes back with the same OBUs, re-framed.
                    assert_eq!(&unit.data, coded, "{codec:?}");
                }
            }
        }
    }
}
