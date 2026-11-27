//! RTP reception in the browser: a page that receives RTP payloads (a
//! WebRTC `RTCRtpScriptTransform`, a WebTransport or WebSocket relay) feeds
//! them here and hands the units that come out to `BrowserDecoder`
//! (`nanocodec-rtp`, docs/handoff/rtp.md).

use nc_encoder::Codec;
use nc_rtp::{Depacketizer, DepacketizerConfig, PayloadFormat};
use wasm_bindgen::prelude::*;

use crate::js_error;
use crate::options::parse_codec;

/// Turns RTP packets into units for `BrowserDecoder::decode`.
#[wasm_bindgen]
#[derive(Debug)]
pub struct RtpReceiver {
    depacketizer: Depacketizer,
    timestamp: u32,
    key: bool,
}

#[wasm_bindgen]
impl RtpReceiver {
    /// A receiver for `codec` ("h264", "hevc", "vvc", "av1", "vp9").
    /// `dependency_descriptor_id`: the negotiated extension id of the AV1
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
        let format = match parse_codec(codec) {
            Some(Codec::H264) => PayloadFormat::H264,
            Some(Codec::Hevc) => PayloadFormat::Hevc,
            Some(Codec::Vvc) => PayloadFormat::Vvc,
            Some(Codec::Av1) => PayloadFormat::Av1,
            Some(Codec::Vp9) => PayloadFormat::Vp9,
            Some(Codec::Av2) | None => {
                return Err(js_error("No RTP payload format for this codec"));
            }
        };
        Ok(Self {
            depacketizer: Depacketizer::new(DepacketizerConfig {
                dependency_descriptor_id: (dependency_descriptor_id != 0)
                    .then_some(dependency_descriptor_id),
                deliver_after_loss,
                ..DepacketizerConfig::new(format)
            }),
            timestamp: 0,
            key: false,
        })
    }

    /// Take one RTP packet (header included), in any order.
    /// # Errors
    /// A malformed packet (it is dropped; the stream goes on).
    pub fn push(&mut self, packet: &[u8]) -> Result<(), JsValue> {
        self.depacketizer.push(packet).map_err(js_error)
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

    /// True once after decoding stopped for want of a key frame: ask the
    /// sender for one (a PLI or FIR).
    pub fn keyframe_needed(&mut self) -> bool {
        self.depacketizer.take_keyframe_request()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_rtp::{Packetizer, PacketizerConfig};

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
}
