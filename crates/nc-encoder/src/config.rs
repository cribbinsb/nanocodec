//! Encoder configuration: what to encode, independent of how.

use std::fmt;

use nc_common::ColourInfo;

use crate::params::{ParamError, ParamOverrides};

/// The output bitstream format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    /// H.264 / AVC.
    H264,
    /// H.265 / HEVC.
    Hevc,
    /// AV1.
    Av1,
    /// VP9 Profile 0.
    Vp9,
    /// AV2 final v1.0.0 (native subset).
    Av2,
    /// H.266 / VVC (real-time subset).
    Vvc,
}

/// A frame rate as an exact rational number (e.g. 30000/1001).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameRate {
    /// Frames per `den` seconds.
    pub num: u32,
    /// Time base denominator.
    pub den: u32,
}

/// How the sequence headers describe frame timing (H.264 VUI §E.1.1, HEVC
/// VUI §E.2.1, AV1 `timing_info()` §5.5.3). Timing never affects decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TimingSignal {
    /// The historical headers: H.264 signals `frame_rate` as a fixed rate,
    /// HEVC and AV1 carry no timing.
    #[default]
    Nominal,
    /// `frame_rate` is signalled as a constant rate in all three codecs.
    Constant,
    /// Frames arrive at a variable rate (camera timestamps, dropped frames):
    /// `frame_rate` is signalled as the clock tick, without a constant
    /// picture interval.
    Variable,
}

/// Informative signalling written to the sequence headers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StreamSignal {
    /// Colour description and range (H.264/HEVC VUI, AV1 `color_config`);
    /// the default signals nothing.
    pub colour: ColourInfo,
    /// Frame timing.
    pub timing: TimingSignal,
}

/// Everything the encoder needs to know about the stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncoderConfig {
    /// Output format.
    pub codec: Codec,
    /// Luma width in samples.
    pub width: usize,
    /// Luma height in samples.
    pub height: usize,
    /// Nominal frame rate (signalled in containers and timing info).
    pub frame_rate: FrameRate,
    /// Fixed quantiser in the codec's native scale: QP 0–51 for H.264 and
    /// HEVC, QP 0–63 for VVC, `base_q_idx` 1–255 for AV1, qindex 0–255 for VP9.
    pub quantizer: u32,
    /// Frames between intra (key/IDR) frames: frame `i` is a key frame when
    /// `i % intra_period == 0`, otherwise an inter frame (if the backend
    /// supports inter frames). 1 makes every frame intra.
    pub intra_period: u32,
    /// Temporal layers of the hierarchical-P structure (`structure.rs`):
    /// 1 is the flat chain (every frame predicted from the previous one),
    /// 2 and 3 are dyadic mini-GOPs of 2 and 4 frames whose top layer is
    /// droppable.
    pub temporal_layers: u8,
    /// log2 of the luma size every leaf block is coded at (bring-up: one
    /// fixed size; frame edges may force smaller blocks).
    pub block_log2: u8,
    /// How much finer than `quantizer` key frames are coded, in the codec's
    /// quantiser scale; `None` lets the backend choose. An experiment knob
    /// (backends that do not use it ignore it).
    pub key_qp_offset: Option<i32>,
    /// Informative stream signalling (colour, timing) for players; it never
    /// changes the coded pictures. The default keeps streams byte-identical
    /// to an encoder without these options.
    pub signal: StreamSignal,
    /// Heuristic parameters set away from their defaults (see
    /// `nc_encoder::engine_specs` and `nanocodec params`).
    pub params: ParamOverrides,
}

/// Why an [`EncoderConfig`] was rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    /// Width or height is zero or odd (4:2:0), or the backend's coded size
    /// is not a multiple of its minimum block size.
    Dimensions {
        /// Requested width.
        width: usize,
        /// Requested height.
        height: usize,
        /// The multiple both must be.
        multiple: usize,
    },
    /// The quantiser is outside the codec's range.
    Quantizer {
        /// Requested value.
        value: u32,
        /// Smallest accepted value.
        min: u32,
        /// Largest accepted value.
        max: u32,
    },
    /// The block size is not one the backend can code.
    BlockSize(u8),
    /// The frame rate has a zero numerator or denominator.
    FrameRate,
    /// The intra period is zero.
    IntraPeriod,
    /// The temporal layer count is unsupported (by the engine, or by this
    /// backend so far).
    TemporalLayers(u8),
    /// A parameter override is unknown or out of range.
    Param(ParamError),
    /// The colour description cannot describe 4:2:0 output (identity
    /// matrix).
    Colour,
    /// Intra refresh cannot be used with these settings.
    IntraRefresh(&'static str),
    /// A long-term reference cannot be used with these settings.
    LongTermReference(&'static str),
    /// `refs.second` is not possible with this configuration or backend.
    SecondReference(&'static str),
    /// The resolution cannot change mid-stream with this backend.
    Resolution(&'static str),
    /// Spatial scalability (`spatial.*`) cannot be used with these settings.
    SpatialScalability(&'static str),
    /// A codec-specific unsupported configuration.
    Unsupported(&'static str),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dimensions {
                width,
                height,
                multiple,
            } => {
                write!(
                    f,
                    "{width}x{height}: dimensions must be non-zero multiples of {multiple}"
                )
            }
            Self::Quantizer { value, min, max } => {
                write!(f, "quantizer {value} outside {min}..={max}")
            }
            Self::BlockSize(log2) => write!(f, "unsupported block size {}", 1u32 << log2),
            Self::FrameRate => write!(f, "frame rate must be a non-zero fraction"),
            Self::IntraPeriod => write!(f, "intra period must be at least 1"),
            Self::TemporalLayers(n) => write!(f, "{n} temporal layers are not supported"),
            Self::Unsupported(message) => f.write_str(message),
            Self::Param(e) => write!(f, "{e}"),
            Self::IntraRefresh(why) => write!(f, "intra refresh: {why}"),
            Self::LongTermReference(why) => write!(f, "long-term reference: {why}"),
            Self::SecondReference(why) => write!(f, "second reference: {why}"),
            Self::Resolution(why) => write!(f, "resolution change: {why}"),
            Self::SpatialScalability(why) => write!(f, "spatial scalability: {why}"),
            Self::Colour => write!(
                f,
                "the identity colour matrix needs 4:4:4 (output is 4:2:0)"
            ),
        }
    }
}

impl std::error::Error for ConfigError {}
