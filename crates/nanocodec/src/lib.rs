//! nanocodec: real-time, low-latency video encoders and matching decoders
//! for H.264, HEVC, VVC, AV1, VP9 and AV2, built on one shared engine.
//!
//! This crate is the front door of the workspace. It contains no codec
//! logic: it re-exports the shared engine ([`encoder`], with [`Encoder`] and
//! its configuration at the root), the picture types ([`common`]) and one
//! module per codec, each behind a cargo feature of the same name (`h264`,
//! `hevc`, `vvc`, `av1`, `vp9`, `av2`; all on by default). A codec module
//! holds that codec's [`CodecBackend`], which [`Encoder`] drives (the one
//! encode path of every codec), and its decoder.
//!
//! The encoders' streams are standard-conformant: 8-bit 4:2:0, forward
//! prediction only, one slice or tile per frame. The decoders also decode
//! other encoders' streams of the main profiles, but none decodes every
//! valid stream yet (e.g. interlaced H.264, some AV1 tools, AV2 10-bit and
//! tiles); the suites passed and every gap are listed in the FAQ
//! (`docs/FAQ.md#conformance`).
//!
//! # Example
//!
//! Encode a few frames of a moving gradient as H.264 under rate control and
//! decode each access unit:
//!
#![cfg_attr(feature = "h264", doc = "```")]
#![cfg_attr(not(feature = "h264"), doc = "```ignore")]
//! use nanocodec::h264::{Decoder, H264Backend};
//! use nanocodec::{
//!     Codec, Encoder, EncoderConfig, FrameRate, ParamOverrides, PlaneIndex, RateControlConfig,
//!     RawFrame, StreamSignal,
//! };
//!
//! const WIDTH: usize = 64;
//! const HEIGHT: usize = 48;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = EncoderConfig {
//!     codec: Codec::H264,
//!     width: WIDTH,
//!     height: HEIGHT,
//!     frame_rate: FrameRate { num: 30, den: 1 },
//!     quantizer: 32, // the starting QP; rate control chooses from there
//!     intra_period: 300,
//!     block_log2: 4, // 16x16 leaf blocks
//!     key_qp_offset: None,
//!     temporal_layers: 1,
//!     signal: StreamSignal::default(),
//!     params: ParamOverrides::default(), // the codec's Default preset
//! };
//! let mut encoder = Encoder::<H264Backend>::new(config)?;
//! encoder.set_rate_control(RateControlConfig::new(100_000))?;
//! let mut decoder = Decoder::new();
//!
//! let mut decoded = 0;
//! for index in 0..5 {
//!     // A packed I420 frame: a diagonal gradient that moves two pixels a frame.
//!     let mut source = vec![128u8; WIDTH * HEIGHT * 3 / 2];
//!     for (i, sample) in source[..WIDTH * HEIGHT].iter_mut().enumerate() {
//!         *sample = (16 + (i % WIDTH + i / WIDTH + 2 * index) % 220) as u8;
//!     }
//!     let frame = encoder.encode_raw(&RawFrame::packed_i420(&source, WIDTH, HEIGHT)?)?;
//!     if frame.data.is_empty() {
//!         continue; // dropped or held by rate control: nothing to send
//!     }
//!     let picture = decoder.decode(&frame.data)?;
//!     assert!(picture.plane(PlaneIndex::Y).view().row(0).len() >= WIDTH);
//!     decoded += 1;
//! }
//! assert!(decoded > 0);
//! # Ok(())
//! # }
//! ```
//!
//! [`preset_params`] gives the parameters of a named speed preset
//! (`veryfast` to `slow`) to put in [`EncoderConfig::params`].
#![forbid(unsafe_code)]

/// Picture buffers, borrowed raw frames, bit writing and errors
/// (`nanocodec-common`).
pub use nc_common as common;
/// The shared encoder engine: configuration, presets, rate control and the
/// [`CodecBackend`] trait (`nanocodec-encoder`).
pub use nc_encoder as encoder;

pub use nc_common::{Picture, PlaneIndex, RawFrame};
pub use nc_encoder::{
    Codec, CodecBackend, EncodedFrame, Encoder, EncoderConfig, FrameRate, FrameType,
    ParamOverrides, RateControlConfig, Speed, StreamSignal, preset_params,
};

/// AV1: `Av1Backend` and the decoder (`nanocodec-av1`).
#[cfg(feature = "av1")]
pub use nc_av1 as av1;
/// AV2: `Av2Backend` and the decoder (`nanocodec-av2`).
#[cfg(feature = "av2")]
pub use nc_av2 as av2;
/// H.264: `H264Backend` and the decoder (`nanocodec-h264`).
#[cfg(feature = "h264")]
pub use nc_h264 as h264;
/// HEVC: `HevcBackend` and the decoder (`nanocodec-hevc`).
#[cfg(feature = "hevc")]
pub use nc_hevc as hevc;
/// VP9: `Vp9Backend` and the decoder (`nanocodec-vp9`).
#[cfg(feature = "vp9")]
pub use nc_vp9 as vp9;
/// H.266/VVC: `VvcBackend` and the decoder (`nanocodec-vvc`).
#[cfg(feature = "vvc")]
pub use nc_vvc as vvc;
