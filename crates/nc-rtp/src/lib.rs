//! RTP payload formats for nanocodec streams: packetizers and
//! depacketizers for H.264 (RFC 6184), HEVC (RFC 7798), VVC (RFC 9328),
//! AV1 (the AOM RTP payload format, with its Dependency Descriptor header
//! extension for SVC forwarding) and VP9 (RFC 9628).
//!
//! The crate depends on nanocodec-common only: it works on coded units
//! (Annex-B access units, AV1 temporal units, VP9 frames) and on the layer
//! information an encoder gives about them ([`LayerStructure`],
//! [`LayerFrame`]), never on a codec crate. The design, and what was tested
//! against which independent implementation, is in `docs/handoff/rtp.md`.
//!
//! - [`Packetizer`]: a unit and its RTP timestamp in, RTP packets out
//!   (marker, sequence numbers, MTU, Dependency Descriptor).
//! - [`Depacketizer`]: RTP packets in, in any order and with losses,
//!   decoder-ready units out, plus keyframe requests and lost frames.
//! - [`LayerFilter`]: drop layers as a forwarding server would, by the
//!   Dependency Descriptor.
//! - [`dump`]: pcap and RFC 4571 capture files.
//! - [`dependency_descriptor`]: the descriptor's syntax, for other uses.
//!
//! ```
//! use nc_rtp::{Depacketizer, DepacketizerConfig, Packetizer, PacketizerConfig, PayloadFormat};
//!
//! // One H.264 access unit: an IDR slice NAL unit too big for one packet.
//! let mut unit = vec![0, 0, 0, 1, 0x65];
//! unit.extend(std::iter::repeat_n(0x11u8, 4000));
//!
//! let mut sender = Packetizer::new(PacketizerConfig::new(PayloadFormat::H264))?;
//! let packets = sender.packetize(&unit, 3000, None)?;
//! assert_eq!(packets.len(), 4); // FU-A fragments of at most 1200 bytes
//!
//! let mut receiver = Depacketizer::new(DepacketizerConfig::new(PayloadFormat::H264));
//! for packet in packets.iter().rev() {
//!     receiver.push(packet)?; // any order
//! }
//! receiver.flush(); // end of stream (otherwise the next unit's arrival releases it)
//! let received = receiver.pop().expect("a whole access unit");
//! assert_eq!(received.frames, vec![unit]);
//! # Ok::<(), nc_rtp::RtpError>(())
//! ```

#![forbid(unsafe_code)]

mod assembler;
mod av1;
mod depacketizer;
pub mod dependency_descriptor;
pub mod dump;
mod error;
mod filter;
mod layering;
mod nal;
mod packet;
mod packetizer;
mod payload;
mod seq;
mod vp9;

pub use depacketizer::{
    DEFAULT_REORDER_WINDOW, Depacketizer, DepacketizerConfig, ReceivedUnit, ReceiverStats,
};
pub use error::{Result, RtpError};
pub use filter::LayerFilter;
pub use layering::{
    DescriptorWriter, FrameDescriptors, InterLayer, LayerFrame, LayerStructure, MAX_SPATIAL_LAYERS,
    MAX_TEMPORAL_LAYERS,
};
pub use packet::{Extension, FIXED_HEADER_BYTES, RtpHeader, RtpPacket, write_packet};
pub use packetizer::{
    CLOCK_RATE, DEFAULT_MTU, DEFAULT_PAYLOAD_TYPE, Packetizer, PacketizerConfig, PayloadFormat,
    UnitLayers, rtp_timestamp,
};
pub use vp9::Vp9Mode;
