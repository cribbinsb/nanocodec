//! Browser bindings for independent nanocodec encoders and subset decoders.
//! The UI transfers captured frames (YUV planes, or RGBA) to workers, which
//! convert them per view (`source`); compressed bytes alone cross the
//! encoder/decoder boundary. No reconstruction-buffer shortcut is used.

#![forbid(unsafe_code)]

#[cfg(feature = "analysis")]
mod error_map;
mod options;
mod pixels;
mod source;

pub use source::SourceFrame;

use nc_common::decode::DecodeError;
use nc_common::outcome::{FrameOutcome, ReceivedFrames};
#[cfg(feature = "analysis")]
use nc_common::stream_stats::{self, StreamStats};
use nc_common::{ColourDescription, ColourInfo, Picture, PlaneIndex, RawFrame, ShownPicture};
use nc_dsp::DisplayMatrix;
use nc_dsp::display::display_scale_plane;
use nc_encoder::{
    Codec, EncodedFrame, Encoder, EncoderConfig, FrameRate, FrameType, ParamOverrides,
    RateControlConfig, RateControlError, RateControlStats,
};
use wasm_bindgen::prelude::*;

/// Largest packet the browser decoder accepts.
const MAX_PACKET_BYTES: usize = 16 * 1024 * 1024;
/// Leaf block size of the browser encoders: 16x16.
const BLOCK_LOG2: u8 = 4;
/// The AV2 backend accepts only 8x8 leaves (`docs/handoff/vp9-av2.md`).
const AV2_BLOCK_LOG2: u8 = 3;
/// VVC's fixed-partition leaf size, 32x32, as the CLI's default; its
/// planners choose leaves up to the CTU size (`vvc.ctu_size`).
const VVC_BLOCK_LOG2: u8 = 5;

/// The leaf block size the CLI uses by default for `codec`, so browser and
/// native encodes of the same input match.
fn block_log2(codec: Codec) -> u8 {
    match codec {
        Codec::Av2 => AV2_BLOCK_LOG2,
        Codec::Vvc => VVC_BLOCK_LOG2,
        Codec::H264 | Codec::Hevc | Codec::Av1 | Codec::Vp9 => BLOCK_LOG2,
    }
}

#[derive(Debug)]
enum EncoderKind {
    H264(Box<Encoder<nc_h264::H264Backend>>),
    Hevc(Box<Encoder<nc_hevc::HevcBackend>>),
    Av1(Box<Encoder<nc_av1::Av1Backend>>),
    Vp9(Box<Encoder<nc_vp9::Vp9Backend>>),
    Av2(Box<Encoder<nc_av2::Av2Backend>>),
    Vvc(Box<Encoder<nc_vvc::VvcBackend>>),
}

impl EncoderKind {
    fn set_rate_control(&mut self, config: RateControlConfig) -> Result<(), RateControlError> {
        match self {
            Self::H264(e) => e.set_rate_control(config),
            Self::Hevc(e) => e.set_rate_control(config),
            Self::Av1(e) => e.set_rate_control(config),
            Self::Vp9(e) => e.set_rate_control(config),
            Self::Av2(e) => e.set_rate_control(config),
            Self::Vvc(e) => e.set_rate_control(config),
        }
    }

    fn request_keyframe(&mut self) {
        match self {
            Self::H264(e) => e.request_keyframe(),
            Self::Hevc(e) => e.request_keyframe(),
            Self::Av1(e) => e.request_keyframe(),
            Self::Vp9(e) => e.request_keyframe(),
            Self::Av2(e) => e.request_keyframe(),
            Self::Vvc(e) => e.request_keyframe(),
        }
    }

    fn disable_rate_control(&mut self) {
        match self {
            Self::H264(e) => e.disable_rate_control(),
            Self::Hevc(e) => e.disable_rate_control(),
            Self::Av1(e) => e.disable_rate_control(),
            Self::Vp9(e) => e.disable_rate_control(),
            Self::Av2(e) => e.disable_rate_control(),
            Self::Vvc(e) => e.disable_rate_control(),
        }
    }

    fn rate_control_stats(&self) -> Option<RateControlStats> {
        match self {
            Self::H264(e) => e.rate_control_stats(),
            Self::Hevc(e) => e.rate_control_stats(),
            Self::Av1(e) => e.rate_control_stats(),
            Self::Vp9(e) => e.rate_control_stats(),
            Self::Av2(e) => e.rate_control_stats(),
            Self::Vvc(e) => e.rate_control_stats(),
        }
    }

    fn wants_look_ahead(&self) -> bool {
        match self {
            Self::H264(e) => e.wants_look_ahead(),
            Self::Hevc(e) => e.wants_look_ahead(),
            Self::Av1(e) => e.wants_look_ahead(),
            Self::Vp9(e) => e.wants_look_ahead(),
            Self::Av2(e) => e.wants_look_ahead(),
            Self::Vvc(e) => e.wants_look_ahead(),
        }
    }

    fn look_ahead(&mut self, next: &Picture) -> Result<(), nc_common::Error> {
        let frame = RawFrame::from_picture(next);
        match self {
            Self::H264(e) => e.look_ahead(&frame),
            Self::Hevc(e) => e.look_ahead(&frame),
            Self::Av1(e) => e.look_ahead(&frame),
            Self::Vp9(e) => e.look_ahead(&frame),
            Self::Av2(e) => e.look_ahead(&frame),
            Self::Vvc(e) => e.look_ahead(&frame),
        }
    }

    /// Encode `source`, one nominal frame interval after the previous frame
    /// (`timestamp_us` None) or captured at `timestamp_us`: rate control then
    /// budgets it for the real time since the previous frame (`Encoder::encode_at`).
    fn encode(&mut self, source: &Picture, timestamp_us: Option<i64>) -> EncodedFrame {
        match (self, timestamp_us) {
            (Self::H264(e), None) => e.encode(source),
            (Self::H264(e), Some(t)) => e.encode_at(source, t),
            (Self::Hevc(e), None) => e.encode(source),
            (Self::Hevc(e), Some(t)) => e.encode_at(source, t),
            (Self::Av1(e), None) => e.encode(source),
            (Self::Av1(e), Some(t)) => e.encode_at(source, t),
            (Self::Vp9(e), None) => e.encode(source),
            (Self::Vp9(e), Some(t)) => e.encode_at(source, t),
            (Self::Av2(e), None) => e.encode(source),
            (Self::Av2(e), Some(t)) => e.encode_at(source, t),
            (Self::Vvc(e), None) => e.encode(source),
            (Self::Vvc(e), Some(t)) => e.encode_at(source, t),
        }
    }
}

/// One browser encoder. A separate decoder instance consumes only its packets.
#[wasm_bindgen]
#[derive(Debug)]
pub struct BrowserEncoder {
    inner: EncoderKind,
    /// Visible size of every input.
    width: usize,
    height: usize,
    /// The colour the stream signals, which every input must have.
    matrix: DisplayMatrix,
    key: bool,
    quantizer: u32,
    /// The last input repeated the picture (`hold.rs`) or was dropped:
    /// nothing was coded, and the decoder keeps showing its last frame.
    repeat: bool,
}

fn js_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}

/// The demo's encoder presets: "default" (no overrides), "fast" (each
/// codec's Fast file, where one exists), "faster" (H.264, HEVC, AV1 and VP9), "fastest" (H.264 and VP9) and
/// "svc" and "svc-fast" (AV1 and VP9: two spatial layers, `<codec>-svc` and `<codec>-svc-fast`), "svc3" and
/// "svc3-fast" (AV1: three spatial layers, `av1-svc3` and `av1-svc3-fast`), the same
/// files as the CLI's `--preset`.
fn preset(codec: Codec, name: &str) -> Result<ParamOverrides, String> {
    let mut params = ParamOverrides::default();
    let text = match (name, codec) {
        ("default", _) => return Ok(params),
        ("fast", Codec::H264) => include_str!("../../../presets/h264-fast.toml"),
        ("faster", Codec::H264) => include_str!("../../../presets/h264-faster.toml"),
        ("fastest", Codec::H264) => include_str!("../../../presets/h264-fastest.toml"),
        ("fast", Codec::Hevc) => include_str!("../../../presets/hevc-fast.toml"),
        ("faster", Codec::Hevc) => include_str!("../../../presets/hevc-faster.toml"),
        ("fast", Codec::Av1) => include_str!("../../../presets/av1-fast.toml"),
        ("faster", Codec::Av1) => include_str!("../../../presets/av1-faster.toml"),
        ("svc", Codec::Av1) => include_str!("../../../presets/av1-svc.toml"),
        ("svc-fast", Codec::Av1) => include_str!("../../../presets/av1-svc-fast.toml"),
        ("svc3", Codec::Av1) => include_str!("../../../presets/av1-svc3.toml"),
        ("svc3-fast", Codec::Av1) => include_str!("../../../presets/av1-svc3-fast.toml"),
        ("fast", Codec::Vp9) => include_str!("../../../presets/vp9-fast.toml"),
        ("faster", Codec::Vp9) => include_str!("../../../presets/vp9-faster.toml"),
        ("fastest", Codec::Vp9) => include_str!("../../../presets/vp9-fastest.toml"),
        ("svc", Codec::Vp9) => include_str!("../../../presets/vp9-svc.toml"),
        ("svc-fast", Codec::Vp9) => include_str!("../../../presets/vp9-svc-fast.toml"),
        ("svc3", Codec::Vp9) => include_str!("../../../presets/vp9-svc3.toml"),
        ("svc3-fast", Codec::Vp9) => include_str!("../../../presets/vp9-svc3-fast.toml"),
        ("fast", Codec::Av2) => include_str!("../../../presets/av2-fast.toml"),
        ("fast", Codec::Vvc) => include_str!("../../../presets/vvc-fast.toml"),
        _ => return Err(format!("No {name} preset for this codec")),
    };
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let (name, value) = line.split_once('=').ok_or("invalid embedded preset")?;
        params.set(
            name.trim(),
            value.trim().parse::<f32>().map_err(|e| e.to_string())?,
        );
    }
    Ok(params)
}

/// The encoder's parameters: the preset, then the still-camera noise handling
/// when `still_camera` is set (`nc_encoder::still_camera_params`), then the
/// free-form `overrides` (`options::parse_overrides`), then film grain
/// synthesis (AV1 only) at its default strength when `film_grain` is set.
fn encoder_params(
    codec: Codec,
    preset_name: &str,
    overrides: &str,
    film_grain: bool,
    still_camera: bool,
) -> Result<ParamOverrides, String> {
    let overrides = options::parse_overrides(codec, overrides).map_err(|errors| {
        errors
            .iter()
            .map(|e| format!("parameter line {}: {}", e.line, e.message))
            .collect::<Vec<_>>()
            .join("; ")
    })?;
    let mut params = preset(codec, preset_name)?;
    if still_camera {
        params = params.overlaid_with(&nc_encoder::still_camera_params(codec));
    }
    let mut params = params.overlaid_with(&overrides);
    if film_grain {
        nc_av1::enable_film_grain(codec, None, &mut params).map_err(|e| e.to_string())?;
    }
    Ok(params)
}

#[wasm_bindgen]
impl BrowserEncoder {
    /// Create a fixed-quantizer encoder. Dimensions and buffering are bounded.
    /// `preset_name` is "default", "fast", (H.264, HEVC, AV1 and VP9) "faster" or
    /// (H.264 and VP9) "fastest";
    /// `overrides` (optional) holds `name=value` lines applied over the
    /// preset ([`check_parameters`] reports bad ones); `film_grain` (AV1 only)
    /// turns on film grain synthesis, which the decoder applies to its
    /// output. `colour` (optional, [`SourceFrame`] names, default `bt601`)
    /// is the matrix and range of the input, which the stream signals
    /// (AV2, whose subset has no colour syntax, signals nothing).
    /// `still_camera` (optional, default off) adds the still-camera noise
    /// handling over the preset, under `overrides`: it lowers the bitrate on
    /// still noisy scenes, is off in every preset, and on AV1 may run a long
    /// still stream slightly over target (`still_camera.rs` in nc-encoder).
    /// # Errors
    /// Rejects invalid codecs, presets, overrides, configurations, film
    /// grain outside AV1, unknown colours, or dimensions above1080p.
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments, reason = "one JS constructor call")]
    pub fn new(
        codec: &str,
        width: usize,
        height: usize,
        qp: u32,
        fps: u32,
        key_interval: u32,
        preset_name: &str,
        overrides: Option<String>,
        film_grain: bool,
        colour: Option<String>,
        still_camera: Option<bool>,
    ) -> Result<Self, JsValue> {
        if width > 1920 || height > 1080 || width < 64 || height < 64 {
            return Err(js_error("Use dimensions between64x64 and1920x1080"));
        }
        if fps == 0 || fps > 60 || key_interval == 0 || key_interval > fps.saturating_mul(30) {
            return Err(js_error("Invalid frame rate or keyframe interval"));
        }
        let codec = options::parse_codec(codec).ok_or_else(|| js_error("Unknown codec"))?;
        let colour = colour.unwrap_or_else(|| "bt601".to_owned());
        let matrix = source::parse_matrix(&colour).ok_or_else(|| js_error("Unsupported colour"))?;
        let overrides = overrides.unwrap_or_default();
        let config = EncoderConfig {
            codec,
            width,
            height,
            frame_rate: FrameRate { num: fps, den: 1 },
            quantizer: qp,
            intra_period: key_interval,
            block_log2: block_log2(codec),
            key_qp_offset: None,
            temporal_layers: 1,
            signal: nc_encoder::StreamSignal {
                colour: signalled_colour(codec, matrix),
                ..Default::default()
            },
            params: encoder_params(
                codec,
                preset_name,
                &overrides,
                film_grain,
                still_camera.unwrap_or(false),
            )
            .map_err(js_error)?,
        };
        let inner = match codec {
            Codec::H264 => EncoderKind::H264(Box::new(Encoder::new(config).map_err(js_error)?)),
            Codec::Hevc => EncoderKind::Hevc(Box::new(Encoder::new(config).map_err(js_error)?)),
            Codec::Av1 => EncoderKind::Av1(Box::new(Encoder::new(config).map_err(js_error)?)),
            Codec::Vp9 => EncoderKind::Vp9(Box::new(Encoder::new(config).map_err(js_error)?)),
            Codec::Av2 => EncoderKind::Av2(Box::new(Encoder::new(config).map_err(js_error)?)),
            Codec::Vvc => EncoderKind::Vvc(Box::new(Encoder::new(config).map_err(js_error)?)),
        };
        Ok(Self {
            inner,
            width,
            height,
            matrix,
            key: false,
            quantizer: qp,
            repeat: false,
        })
    }

    /// Encode one source frame as an access unit / AV1 temporal unit:
    /// empty when the input repeated the picture ([`BrowserEncoder::repeat`]).
    /// # Errors
    /// Rejects a frame of another size or colour than the configuration's.
    pub fn encode(&mut self, source: &SourceFrame) -> Result<Vec<u8>, JsValue> {
        self.encode_timed(source, None)
    }

    /// [`BrowserEncoder::encode`] for a frame captured at `timestamp_us`
    /// (microseconds, any origin): a live source whose frames may be skipped.
    /// Rate control budgets the frame for the time since the previous one, so
    /// the bitrate holds in real time.
    /// # Errors
    /// As [`BrowserEncoder::encode`].
    pub fn encode_at(
        &mut self,
        source: &SourceFrame,
        timestamp_us: f64,
    ) -> Result<Vec<u8>, JsValue> {
        // Microsecond capture times are integers well inside f64's exact range.
        #[allow(clippy::cast_possible_truncation, reason = "integral microseconds")]
        self.encode_timed(source, Some(timestamp_us.round() as i64))
    }

    fn encode_timed(
        &mut self,
        source: &SourceFrame,
        timestamp_us: Option<i64>,
    ) -> Result<Vec<u8>, JsValue> {
        let picture = self.input(source)?;
        let frame = self.inner.encode(picture, timestamp_us);
        self.repeat = frame.metadata.held || frame.metadata.dropped;
        self.key = frame.metadata.frame_type == FrameType::Key && !self.repeat;
        if !self.repeat {
            self.quantizer = frame.metadata.quantizer;
        }
        Ok(frame.data)
    }

    /// Whether the encoder wants the second frame before the first
    /// (`look_ahead`): only before the first frame, under rate control, for
    /// codecs whose rate model sizes the first key from start-up camera
    /// motion. The caller then holds frame 0 back one frame.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn wants_look_ahead(&self) -> bool {
        self.inner.wants_look_ahead()
    }

    /// Hand over the second source frame before encoding the first; both
    /// are still encoded in order afterwards.
    /// # Errors
    /// Rejects a frame of another size or colour than the configuration's.
    pub fn look_ahead(&mut self, source: &SourceFrame) -> Result<(), JsValue> {
        let picture = self.input(source)?;
        self.inner.look_ahead(picture).map_err(js_error)
    }

    /// The colour (`SourceFrame` names) the encoder expects and signals.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn colour(&self) -> String {
        source::matrix_name(self.matrix).to_owned()
    }

    /// Enable or update realtime bitrate control without restarting references.
    /// # Errors
    /// Rejects invalid targets, peak rates or buffer durations.
    pub fn set_rate_control(
        &mut self,
        target_kbps: u32,
        peak_kbps: u32,
        buffer_ms: u32,
    ) -> Result<(), JsValue> {
        let mut config = RateControlConfig::new(u64::from(target_kbps) * 1000);
        config.peak_bps = Some(u64::from(peak_kbps) * 1000);
        config.buffer_ms = buffer_ms;
        self.inner.set_rate_control(config).map_err(js_error)
    }

    /// Request a keyframe on the next input, preserving rate-control history.
    pub fn request_keyframe(&mut self) {
        self.inner.request_keyframe();
    }

    /// Return to the fixed quantizer supplied at construction.
    pub fn disable_rate_control(&mut self) {
        self.inner.disable_rate_control();
    }

    /// Quantizer reported for the latest frame, in the codec's native scale.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn quantizer(&self) -> u32 {
        self.quantizer
    }

    /// Fraction of the configured bitrate buffer currently occupied.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn buffer_fullness(&self) -> f64 {
        self.inner
            .rate_control_stats()
            .map_or(0.0, |stats| stats.buffer_fullness)
    }

    /// Number of frames that exceeded the modeled bitrate buffer capacity.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn buffer_overflows(&self) -> u32 {
        self.inner.rate_control_stats().map_or(0, |stats| {
            u32::try_from(stats.buffer_overflows).unwrap_or(u32::MAX)
        })
    }

    /// Whether the last input coded nothing (its packet is empty): it
    /// repeated the picture (frame hold) or rate control dropped it. Send
    /// nothing; the decoder keeps showing its last frame, which is the
    /// picture for this input.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn repeat(&self) -> bool {
        self.repeat
    }

    /// Whether the last encoded packet is a key frame.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn key(&self) -> bool {
        self.key
    }
}

impl BrowserEncoder {
    /// `source`'s picture, if it matches the configuration.
    fn input<'a>(&self, source: &'a SourceFrame) -> Result<&'a Picture, JsValue> {
        let picture = source.picture();
        if (picture.width(), picture.height()) != (self.width, self.height) {
            return Err(js_error(
                "Source frame size does not match the configuration",
            ));
        }
        if source.matrix() != self.matrix {
            return Err(js_error(
                "Source frame colour does not match the configuration",
            ));
        }
        Ok(picture)
    }
}

/// What a stream of `codec` signals for input of `matrix`: its matrix
/// (with the matching primaries and transfer) and range. AV2's subset has
/// no colour syntax and signals nothing.
fn signalled_colour(codec: Codec, matrix: DisplayMatrix) -> ColourInfo {
    if codec == Codec::Av2 {
        return ColourInfo::default();
    }
    let (description, full_range) = match matrix {
        DisplayMatrix::Bt601Limited => (ColourDescription::BT601_525, false),
        DisplayMatrix::Bt709Limited => (ColourDescription::BT709, false),
        DisplayMatrix::Bt601Full => (ColourDescription::BT601_525, true),
        DisplayMatrix::Bt709Full => (ColourDescription::BT709, true),
    };
    ColourInfo {
        description: Some(description),
        full_range,
    }
}

#[derive(Debug)]
enum DecoderKind {
    H264(Box<nc_h264::Decoder>),
    Hevc(Box<nc_hevc::Decoder>),
    Av1(Box<nc_av1::Decoder>),
    Vp9(Box<nc_vp9::stream::Decoder>),
    Av2(Box<nc_av2::Decoder>),
    Vvc(Box<nc_vvc::Decoder>),
}

/// An independent subset decoder; no encoder instance or reconstruction is needed.
#[wasm_bindgen]
#[derive(Debug)]
pub struct BrowserDecoder {
    /// Every decoder shows its latest frame in place (`shown`) until its
    /// next decode: no output copy.
    inner: DecoderKind,
    /// Display kernels for the RGBA conversion.
    dsp: nc_dsp::Dsp,
    /// The latest frame as RGBA or packed I420 (whichever the display asked
    /// for last), reused across frames so the display path allocates nothing
    /// per frame.
    display: Vec<u8>,
    /// The source frame `psnr` converts and compares against, reused
    /// across frames (no border: it is never predicted from).
    reference: Option<Picture>,
    /// Stream-size upscaled copy of a smaller decoded frame (spatial base
    /// layer or adaptive resolution), reused across frames when `psnr`
    /// scores against a full-resolution RGBA source.
    scaled_decoded: Option<Picture>,
    /// Frames decoded, for the transport's dependency signal (`decode_frame`).
    frames: ReceivedFrames,
    /// The latest picture's analysis records, packed for the page
    /// (`analysis_frame`), reused across frames.
    #[cfg(feature = "analysis")]
    analysis: Vec<i16>,
    /// The latest PSNR map's squared error per block (`error_map`), reused
    /// across frames.
    #[cfg(feature = "analysis")]
    error_sums: Vec<u32>,
    /// The latest frame's luma at the scored size (`scored_luma`), reused.
    #[cfg(feature = "analysis")]
    scored_luma: Vec<u8>,
}

/// One packet's outcome (`nc_common::outcome::FrameOutcome`): decoded, or
/// frozen. On a frozen packet the page keeps showing the picture it shows
/// (the decoder may no longer hold it) and sends a loss report naming
/// `report_index`.
#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub struct DecodeOutcome(FrameOutcome);

#[wasm_bindgen]
impl DecodeOutcome {
    /// Whether a new picture was decoded.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn decoded(&self) -> bool {
        self.0 == FrameOutcome::Decoded
    }

    /// Why the picture froze: `missing-reference` (an earlier frame was
    /// lost or refused), `missing-key` (no key frame yet) or `corrupt`
    /// (malformed or truncated packet); empty when decoded.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn reason(&self) -> String {
        match self.0 {
            FrameOutcome::Decoded => String::new(),
            FrameOutcome::Frozen { reason, .. } => reason.name().to_owned(),
        }
    }

    /// The coded-frame index to report lost (the missing reference when
    /// the dependency list named it, else this frame), or -1 when decoded.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn report_index(&self) -> f64 {
        match self.0 {
            FrameOutcome::Frozen {
                report: Some(index),
                ..
            } => index as f64,
            _ => -1.0,
        }
    }
}

/// Bytes of a packed `width` x `height` I420 frame.
fn packed_i420_len(width: usize, height: usize) -> usize {
    width * height + 2 * width.div_ceil(2) * height.div_ceil(2)
}

/// A frame index from JS (a non-negative integer below 2^53).
fn frame_index(value: f64) -> Result<u64, JsValue> {
    const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;
    if value.fract() == 0.0 && (0.0..=MAX_SAFE_INTEGER).contains(&value) {
        Ok(value as u64)
    } else {
        Err(js_error("Frame indices are non-negative integers"))
    }
}

#[wasm_bindgen]
impl BrowserDecoder {
    /// Create a decoder for per-frame Annex-B (H.264, HEVC, VVC), raw OBU
    /// (AV1, AV2) or VP9 frame packets.
    /// # Errors
    /// Rejects codec names other than h264, hevc, av1, vp9, av2 and vvc.
    #[wasm_bindgen(constructor)]
    pub fn new(codec: &str) -> Result<Self, JsValue> {
        let inner = match codec {
            "h264" => DecoderKind::H264(Box::default()),
            "hevc" => DecoderKind::Hevc(Box::default()),
            "av1" => DecoderKind::Av1(Box::default()),
            "vp9" => DecoderKind::Vp9(Box::default()),
            "av2" => DecoderKind::Av2(Box::default()),
            "vvc" => DecoderKind::Vvc(Box::default()),
            _ => return Err(js_error("Unknown codec")),
        };
        Ok(Self {
            inner,
            dsp: nc_dsp::Dsp::new(),
            display: Vec::new(),
            reference: None,
            scaled_decoded: None,
            frames: ReceivedFrames::default(),
            #[cfg(feature = "analysis")]
            analysis: Vec::new(),
            #[cfg(feature = "analysis")]
            error_sums: Vec::new(),
            #[cfg(feature = "analysis")]
            scored_luma: Vec::new(),
        })
    }

    #[cfg(feature = "analysis")]
    /// Switch coding-tool analysis on (from the next decoded picture) or
    /// off; returns whether this codec's decoder supports it (every
    /// codec). Only in the analysis build of the module; every
    /// picture's block records are then kept for `analysis_frame`.
    pub fn set_analysis(&mut self, on: bool) -> bool {
        let stats = || on.then(StreamStats::records_only);
        match &mut self.inner {
            DecoderKind::H264(d) => match stats() {
                Some(s) => d.collect_stats(s),
                None => drop(d.take_stats()),
            },
            DecoderKind::Hevc(d) => match stats() {
                Some(s) => d.collect_stats(s),
                None => drop(d.take_stats()),
            },
            DecoderKind::Av1(d) => match stats() {
                Some(s) => d.collect_stats(s),
                None => drop(d.take_stats()),
            },
            DecoderKind::Vp9(d) => match stats() {
                Some(s) => d.collect_stats(s),
                None => drop(d.take_stats()),
            },
            DecoderKind::Vvc(d) => match stats() {
                Some(s) => d.collect_stats(s),
                None => drop(d.take_stats()),
            },
            DecoderKind::Av2(d) => match stats() {
                Some(s) => d.collect_stats(s),
                None => drop(d.take_stats()),
            },
        }
        if !on {
            self.analysis = Vec::new();
        }
        true
    }

    #[cfg(feature = "analysis")]
    /// Pack the latest decoded picture's analysis records
    /// (`nc_common::stream_stats::pack_records`: `analysis_halfwords` `i16`
    /// values per record) into a reused buffer and return its byte offset
    /// in WASM memory, read as for `rgba_frame`; `analysis_length` is the
    /// number of `i16` values. Without analysis the buffer is empty.
    pub fn analysis_frame(&mut self) -> usize {
        let stats = match &self.inner {
            DecoderKind::H264(d) => d.stats(),
            DecoderKind::Hevc(d) => d.stats(),
            DecoderKind::Av1(d) => d.stats(),
            DecoderKind::Vp9(d) => d.stats(),
            DecoderKind::Vvc(d) => d.stats(),
            DecoderKind::Av2(d) => d.stats(),
        };
        match stats {
            Some(s) => stream_stats::pack_records(s.records(), &mut self.analysis),
            None => self.analysis.clear(),
        }
        self.analysis.as_ptr() as usize
    }

    #[cfg(feature = "analysis")]
    /// The number of `i16` values `analysis_frame` packed.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn analysis_length(&self) -> usize {
        self.analysis.len()
    }

    #[cfg(feature = "analysis")]
    /// `i16` values per packed analysis record.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn analysis_halfwords(&self) -> usize {
        stream_stats::PACKED_HALFWORDS
    }

    #[cfg(feature = "analysis")]
    /// The squared error of the latest decoded frame against the source frame
    /// the last `psnr` call scored it against, per block of the PSNR map's
    /// fixed grid (`error_map::block_sse`: luma blocks row by row, then Cb,
    /// then Cr), at the size `psnr` compared (`error_map_width` x
    /// `error_map_height`: a smaller spatial layer is upscaled to the stream
    /// size, as there). Returns the byte offset of the `u32` sums in WASM
    /// memory, read as for `rgba_frame`; `error_map_length` is their count.
    /// Only in the analysis build of the module.
    /// # Errors
    /// Rejects calls made before `psnr` has scored a decoded frame.
    pub fn error_map(&mut self) -> Result<usize, JsValue> {
        let mut sums = std::mem::take(&mut self.error_sums);
        let reference = self
            .reference
            .as_ref()
            .ok_or_else(|| js_error("No source frame scored yet"))?;
        let size = (reference.width(), reference.height());
        let shown = self.shown().ok_or_else(|| js_error("No decoded frame"))?;
        let decoded = if (shown.width(), shown.height()) == size {
            shown
        } else {
            self.scaled_decoded
                .as_ref()
                .filter(|p| (p.width(), p.height()) == size)
                .map(ShownPicture::whole)
                .ok_or_else(|| js_error("No scored frame of the source's size"))?
        };
        error_map::block_sse(&self.dsp, reference, decoded, &mut sums);
        self.error_sums = sums;
        Ok(self.error_sums.as_ptr() as usize)
    }

    #[cfg(feature = "analysis")]
    /// The number of `u32` sums `error_map` wrote.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn error_map_length(&self) -> usize {
        self.error_sums.len()
    }

    #[cfg(feature = "analysis")]
    /// Width of the picture `error_map` measured (the source's).
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn error_map_width(&self) -> usize {
        self.reference.as_ref().map_or(0, Picture::width)
    }

    #[cfg(feature = "analysis")]
    /// Height of the picture `error_map` measured (the source's).
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn error_map_height(&self) -> usize {
        self.reference.as_ref().map_or(0, Picture::height)
    }

    #[cfg(feature = "analysis")]
    /// Copy the luma of the latest decoded frame at the size the last `psnr`
    /// call compared (a smaller spatial layer upscaled, as there) into a
    /// reused buffer and return its byte offset in WASM memory
    /// (`error_map_width * error_map_height` bytes, read as for
    /// `rgba_frame`): the distorted plane `ms_ssim_luma` scores in the
    /// metrics worker. Only in the analysis build.
    /// # Errors
    /// Rejects calls made before `psnr` has scored a decoded frame.
    pub fn scored_luma(&mut self) -> Result<usize, JsValue> {
        let mut luma = std::mem::take(&mut self.scored_luma);
        let reference = self
            .reference
            .as_ref()
            .ok_or_else(|| js_error("No source frame scored yet"))?;
        let size = (reference.width(), reference.height());
        let shown = self.shown().ok_or_else(|| js_error("No decoded frame"))?;
        let decoded = if (shown.width(), shown.height()) == size {
            shown
        } else {
            self.scaled_decoded
                .as_ref()
                .filter(|p| (p.width(), p.height()) == size)
                .map(ShownPicture::whole)
                .ok_or_else(|| js_error("No scored frame of the source's size"))?
        };
        luma.clear();
        decoded
            .plane(PlaneIndex::Y)
            .rows()
            .for_each(|row| luma.extend_from_slice(row));
        self.scored_luma = luma;
        Ok(self.scored_luma.as_ptr() as usize)
    }

    /// Select the operating point (`0..=31` on AV1, `0..=2` on VP9; default `0`)
    /// before or between packets.
    /// # Errors
    /// Rejects non-zero operating points on codecs other than AV1 and VP9, and
    /// out-of-range operating points.
    pub fn set_operating_point(&mut self, op: u8) -> Result<(), JsValue> {
        if op > nc_av1::MAX_OPERATING_POINT {
            return Err(js_error("Operating point must be in 0..=31"));
        }
        match &mut self.inner {
            DecoderKind::Av1(d) => d.set_operating_point(op).map_err(js_error),
            DecoderKind::Vp9(d) => d.set_operating_point(op).map_err(js_error),
            _ if op == 0 => Ok(()),
            _ => Err(js_error(
                "Operating points are supported on AV1 and VP9 only",
            )),
        }
    }

    /// The `spatial_id` (`0..=3`) of the latest shown AV1 or VP9 frame (`0` on
    /// single-layer streams and other codecs).
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn spatial_id(&self) -> u8 {
        match &self.inner {
            DecoderKind::Av1(d) => d.shown_spatial_id().unwrap_or(0),
            DecoderKind::Vp9(d) => d.shown_spatial_id().unwrap_or(0),
            _ => 0,
        }
    }

    /// Sequence-header or stream maximum frame dimensions on AV1 and VP9, or
    /// the current frame dimensions on other codecs.
    fn stream_dimensions(&self) -> (usize, usize) {
        match &self.inner {
            DecoderKind::Av1(d) => d
                .max_frame_size()
                .unwrap_or_else(|| (self.width(), self.height())),
            DecoderKind::Vp9(d) => d
                .max_frame_size()
                .unwrap_or_else(|| (self.width(), self.height())),
            _ => (self.width(), self.height()),
        }
    }

    /// Maximum visible width of the stream (`max_frame_width` on AV1, or the
    /// current frame width on other codecs).
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn stream_width(&self) -> usize {
        self.stream_dimensions().0
    }

    /// Maximum visible height of the stream (`max_frame_height` on AV1, or the
    /// current frame height on other codecs).
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn stream_height(&self) -> usize {
        self.stream_dimensions().1
    }

    /// Decode one packet; the picture becomes the latest decoded frame.
    /// # Errors
    /// Every packet that does not decode, with its freeze reason
    /// (`missing-reference: ...`, `missing-key: ...`, `corrupt: ...`); a
    /// player uses `decode_frame` instead.
    pub fn decode(&mut self, packet: &[u8]) -> Result<(), JsValue> {
        let result = self.decode_packet(packet);
        match FrameOutcome::of(&result, None) {
            FrameOutcome::Decoded => Ok(()),
            FrameOutcome::Frozen { reason, .. } => Err(js_error(format!(
                "{}: {}",
                reason.name(),
                result.err().map(|e| e.to_string()).unwrap_or_default()
            ))),
        }
    }

    /// Decode coded frame `index` and report its outcome, never throwing
    /// for stream problems. `references` is the transport's dependency
    /// signal (the frames it predicts from; over RTP, the AV1 dependency
    /// descriptor), which RTP receivers should pass; else empty: H.264 and
    /// HEVC streams, and AV1 streams with frame ids (`av1.frame_ids`, or a
    /// long-term reference), then reveal a missing reference themselves; VP9
    /// and AV2 streams cannot (`docs/handoff/loss-resilience.md`).
    /// # Errors
    /// Rejects indices that are not non-negative integers.
    pub fn decode_frame(
        &mut self,
        packet: &[u8],
        index: f64,
        references: &[f64],
    ) -> Result<DecodeOutcome, JsValue> {
        let index = frame_index(index)?;
        let references = references
            .iter()
            .map(|&r| frame_index(r))
            .collect::<Result<Vec<u64>, JsValue>>()?;
        let mut frames = std::mem::take(&mut self.frames);
        let (outcome, _) = frames.receive(index, &references, || self.decode_packet(packet));
        self.frames = frames;
        Ok(DecodeOutcome(outcome))
    }

    /// One packet through this codec's decoder.
    fn decode_packet(&mut self, packet: &[u8]) -> Result<(), DecodeError> {
        if packet.len() > MAX_PACKET_BYTES {
            return Err(DecodeError::Unsupported(
                "packet exceeds the 16 MiB browser limit",
            ));
        }
        match &mut self.inner {
            DecoderKind::H264(d) => d.decode_shown(packet),
            DecoderKind::Hevc(d) => d.decode_shown(packet),
            DecoderKind::Av1(d) => d.decode_shown(packet),
            DecoderKind::Vp9(d) => d.decode_shown(packet),
            DecoderKind::Av2(d) => d.decode_shown(packet),
            DecoderKind::Vvc(d) => d.decode_shown(packet),
        }
    }

    /// The latest decoded frame's visible samples.
    fn shown(&self) -> Option<ShownPicture<'_>> {
        match &self.inner {
            DecoderKind::H264(d) => d.shown(),
            DecoderKind::Hevc(d) => d.shown(),
            DecoderKind::Av1(d) => d.shown(),
            DecoderKind::Vp9(d) => d.shown(),
            DecoderKind::Av2(d) => d.shown(),
            DecoderKind::Vvc(d) => d.shown(),
        }
    }

    /// Convert the latest frame to RGBA (`width * height * 4` bytes) with
    /// `colour` (the matrix and range of the samples, `SourceFrame` names)
    /// in a buffer this decoder reuses, and return its byte offset in the
    /// module's WASM memory. JS reads it through a view of `memory.buffer`,
    /// which is valid until the next call into the module (memory may grow);
    /// copying from it into a reused canvas buffer avoids a per-frame
    /// allocation. This is the CPU display path; `i420_frame` feeds a GPU
    /// conversion.
    /// # Errors
    /// Rejects unknown colours, and calls made before a frame has been
    /// decoded.
    pub fn rgba_frame(&mut self, colour: &str) -> Result<usize, JsValue> {
        let matrix = source::parse_matrix(colour).ok_or_else(|| js_error("Unsupported colour"))?;
        // Taken out so the frame can be borrowed while it is filled.
        let mut rgba = std::mem::take(&mut self.display);
        let picture = self.shown().ok_or_else(|| js_error("No decoded frame"))?;
        rgba.resize(picture.width() * picture.height() * 4, 0);
        pixels::to_rgba(&self.dsp, matrix, picture, &mut rgba);
        self.display = rgba;
        Ok(self.display.as_ptr() as usize)
    }

    /// Pack the latest frame's visible samples as I420 (Y, then Cb, then Cr,
    /// rows without padding: `packed_frame_len` bytes) into the reused display
    /// buffer and return its byte offset in WASM memory, read as for
    /// `rgba_frame`. A WebGL display uploads the three planes and converts
    /// them in a shader, moving 1.5 instead of 4 bytes per pixel.
    /// # Errors
    /// Rejects calls made before a frame has been decoded.
    pub fn i420_frame(&mut self) -> Result<usize, JsValue> {
        let mut packed = std::mem::take(&mut self.display);
        let picture = self.shown().ok_or_else(|| js_error("No decoded frame"))?;
        pixels::pack_i420(picture, &mut packed);
        self.display = packed;
        Ok(self.display.as_ptr() as usize)
    }

    /// PSNR-YUV of the latest decoded frame against the source frame it was
    /// coded from, packed I420 (`SourceFrame::packed`; 6:1:1 weighting).
    /// When the decoded frame is a smaller spatial layer or
    /// adaptive-resolution picture of the same stream, it is upscaled to the
    /// stream dimensions (`nc_dsp::display`) before scoring.
    /// # Errors
    /// Rejects calls made before a frame has been decoded, and a source
    /// whose byte count matches neither the decoded frame nor the stream.
    pub fn psnr(&mut self, i420: &[u8]) -> Result<f64, JsValue> {
        let (width, height) = (self.width(), self.height());
        if width == 0 {
            return Err(js_error("No decoded frame"));
        }
        let (stream_w, stream_h) = self.stream_dimensions();
        let (ref_w, ref_h) = if i420.len() == packed_i420_len(width, height) {
            (width, height)
        } else if stream_w > 0 && i420.len() == packed_i420_len(stream_w, stream_h) {
            (stream_w, stream_h)
        } else {
            return Err(js_error(
                "Source frame dimensions differ from the decoded frame",
            ));
        };
        let mut reference = match self.reference.take() {
            Some(p) if (p.width(), p.height()) == (ref_w, ref_h) => p,
            _ => Picture::new(ref_w, ref_h, 0).map_err(js_error)?,
        };
        pixels::unpack_i420(i420, &mut reference);
        let score = if (width, height) == (ref_w, ref_h) {
            let decoded = self.shown().ok_or_else(|| js_error("No decoded frame"))?;
            pixels::psnr(&self.dsp, &reference, decoded)
        } else {
            let mut scaled = match self.scaled_decoded.take() {
                Some(p) if (p.width(), p.height()) == (ref_w, ref_h) => p,
                _ => Picture::new(ref_w, ref_h, 0).map_err(js_error)?,
            };
            {
                let decoded = self.shown().ok_or_else(|| js_error("No decoded frame"))?;
                for plane in PlaneIndex::ALL {
                    display_scale_plane(
                        &decoded.plane(plane),
                        &mut scaled.plane_mut(plane).view_mut(),
                    );
                }
            }
            let score = pixels::psnr(&self.dsp, &reference, ShownPicture::whole(&scaled));
            self.scaled_decoded = Some(scaled);
            score
        };
        self.reference = Some(reference);
        Ok(score)
    }

    /// Visible width of the latest decoded frame, or zero before the first frame.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize {
        self.shown().map_or(0, |p| p.width())
    }

    /// Visible height of the latest decoded frame, or zero before the first frame.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize {
        self.shown().map_or(0, |p| p.height())
    }

    /// The latest decoded I420 samples, suitable for independent conformance checks.
    /// # Errors
    /// Rejects calls made before a frame has been decoded.
    pub fn decoded_i420(&self) -> Result<Vec<u8>, JsValue> {
        let picture = self.shown().ok_or_else(|| js_error("No decoded frame"))?;
        let mut bytes = Vec::new();
        pixels::pack_i420(picture, &mut bytes);
        Ok(bytes)
    }
}

#[cfg(feature = "analysis")]
/// MS-SSIM of two `width` x `height` luma planes (rows without padding), as
/// the evaluation scores it (`nc_dsp::ms_ssim`); `None` for a plane too
/// small for five scales. The demo's metrics worker calls it with a source
/// luma and `BrowserDecoder::scored_luma`. Only in the analysis build.
/// # Errors
/// Rejects planes whose length is not `width * height`.
#[wasm_bindgen]
pub fn ms_ssim_luma(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
) -> Result<Option<f64>, JsValue> {
    let view = |data| {
        nc_common::PlaneView::new(data, width, width, height)
            .filter(|_| data.len() == width * height)
    };
    let (Some(a), Some(b)) = (view(reference), view(distorted)) else {
        return Err(js_error("Planes must be width * height bytes"));
    };
    Ok(nc_dsp::ms_ssim::ms_ssim(&a, &b))
}

/// `codec`'s registered parameters, as a JSON array of `{name, default,
/// min, max, doc}` (defaults are the codec's own).
/// # Errors
/// Rejects unknown codec names.
#[wasm_bindgen]
pub fn parameter_registry(codec: &str) -> Result<String, JsValue> {
    let codec = options::parse_codec(codec).ok_or_else(|| js_error("Unknown codec"))?;
    Ok(options::registry_json(codec))
}

/// What the page may offer for `codec`, as JSON `{filmGrain, blockQp,
/// temporalLayers, secondReference}`.
/// # Errors
/// Rejects unknown codec names.
#[wasm_bindgen]
pub fn codec_options(codec: &str) -> Result<String, JsValue> {
    let codec = options::parse_codec(codec).ok_or_else(|| js_error("Unknown codec"))?;
    Ok(options::options_json(codec))
}

/// The bad lines of a `name=value` override list for `codec`, as a JSON
/// array of `{line, message}` (empty when every line is valid).
/// # Errors
/// Rejects unknown codec names.
#[wasm_bindgen]
pub fn check_parameters(codec: &str, text: &str) -> Result<String, JsValue> {
    let codec = options::parse_codec(codec).ok_or_else(|| js_error("Unknown codec"))?;
    Ok(match options::parse_overrides(codec, text) {
        Ok(_) => "[]".to_owned(),
        Err(errors) => options::errors_json(&errors),
    })
}

#[cfg(test)]
mod preset_tests {
    use super::*;
    use nc_av1::Av1Backend;
    use nc_encoder::CodecBackend;

    #[test]
    fn av1_fast_uses_adaptive_chroma_and_residual_rejection() {
        let fast = preset(Codec::Av1, "fast").unwrap();
        for (name, expected) in [
            ("av1.chroma_delta_q", -24.0),
            ("av1.chroma_noise_scale", 2.0),
            ("av1.transform_rd_skip", 2.0),
            ("av1.pre_tx_skip", 1.0),
        ] {
            let spec = Av1Backend::PARAM_SPECS
                .iter()
                .flat_map(|specs| specs.iter())
                .find(|s| s.name == name)
                .unwrap();
            assert_eq!(fast.value(spec), expected);
        }
        assert!(preset(Codec::Av1, "default").unwrap().is_empty());
    }

    #[test]
    fn film_grain_is_av1_only() {
        let fast = preset(Codec::Av1, "fast").unwrap();
        let grain = encoder_params(Codec::Av1, "fast", "", true, false).unwrap();
        let enable = nc_av1::Av1Backend::PARAM_SPECS
            .iter()
            .flat_map(|specs| specs.iter())
            .find(|s| s.name == "grain.enable")
            .unwrap();
        assert_eq!(fast.value(enable), 0.0);
        assert_eq!(grain.value(enable), 1.0);
        for codec in [Codec::H264, Codec::Hevc, Codec::Vvc, Codec::Vp9, Codec::Av2] {
            assert!(encoder_params(codec, "default", "", true, false).is_err());
        }
    }

    #[test]
    fn overrides_apply_over_the_preset() {
        let spec = |name| {
            nc_av1::Av1Backend::PARAM_SPECS
                .iter()
                .flat_map(|specs| specs.iter())
                .find(|s| s.name == name)
                .unwrap()
        };
        let params =
            encoder_params(Codec::Av1, "fast", "av1.chroma_delta_q = -12", false, false).unwrap();
        assert_eq!(params.value(spec("av1.chroma_delta_q")), -12.0);
        assert_eq!(
            params.value(spec("av1.pre_tx_skip")),
            1.0,
            "the rest of Fast stays"
        );
        let error =
            encoder_params(Codec::Av1, "default", "\nav1.nope=1", false, false).unwrap_err();
        assert!(
            error.starts_with("parameter line 2: unknown parameter"),
            "{error}"
        );
    }

    #[test]
    fn still_camera_lies_over_the_preset_under_overrides() {
        let on =
            encoder_params(Codec::Av1, "fast", "decision.noise_skip = 2", false, true).unwrap();
        let expected = nc_encoder::still_camera_params(Codec::Av1);
        assert_eq!(
            on.get("planner.noise_split"),
            expected.get("planner.noise_split")
        );
        assert_eq!(on.get("rate.skip_budget"), expected.get("rate.skip_budget"));
        assert_eq!(on.get("decision.noise_skip"), Some(2.0));
        let off = encoder_params(Codec::Av1, "fast", "", false, false).unwrap();
        assert_eq!(off, preset(Codec::Av1, "fast").unwrap());
    }

    #[test]
    fn h264_has_fast_and_faster_presets() {
        assert!(!preset(Codec::H264, "fast").unwrap().is_empty());
        assert!(!preset(Codec::H264, "faster").unwrap().is_empty());
        assert!(!preset(Codec::H264, "fastest").unwrap().is_empty());
        assert!(preset(Codec::Hevc, "fastest").is_err());
        assert!(!preset(Codec::Hevc, "faster").unwrap().is_empty());
        assert!(!preset(Codec::Vp9, "faster").unwrap().is_empty());
        assert!(!preset(Codec::Vp9, "fastest").unwrap().is_empty());
        assert!(preset(Codec::Vvc, "faster").is_err());
    }

    #[test]
    fn vvc_fast_is_the_cli_preset() {
        let fast = preset(Codec::Vvc, "fast").unwrap();
        for (name, expected) in [
            ("vvc.inter_intra_dc_only", 1.0),
            ("vvc.sao_skip_unchanged", 1.0),
            ("vvc.isp", 0.0),
        ] {
            let spec = nc_vvc::VvcBackend::PARAM_SPECS
                .iter()
                .flat_map(|specs| specs.iter())
                .find(|s| s.name == name)
                .unwrap();
            assert_eq!(fast.value(spec), expected);
        }
        assert!(preset(Codec::Vvc, "default").unwrap().is_empty());
        assert!(preset(Codec::Vvc, "faster").is_err());
    }
}

#[cfg(test)]
mod input_tests {
    use super::*;

    /// `rgba` as a source frame (BT.601, the RGBA path) and its packed I420.
    fn source_of(width: usize, height: usize, rgba: &[u8]) -> (SourceFrame, Vec<u8>) {
        let mut source = SourceFrame::new(width, height).unwrap();
        source.from_rgba(rgba).unwrap();
        let mut packed = Vec::new();
        pixels::pack_i420(ShownPicture::whole(source.picture()), &mut packed);
        (source, packed)
    }

    #[test]
    fn visible_input_matches_extended_borders_for_every_codec() {
        let (width, height) = (128, 72);
        for (codec, qp) in [
            ("h264", 32),
            ("hevc", 32),
            ("vvc", 32),
            ("av1", 140),
            ("vp9", 140),
            ("av2", 140),
        ] {
            let make = || {
                BrowserEncoder::new(
                    codec, width, height, qp, 30, 300, "fast", None, false, None, None,
                )
                .unwrap()
            };
            let (mut visible, mut extended) = (make(), make());
            let mut padded = Picture::new(width, height, 64).unwrap();
            for frame in 0..3 {
                let rgba: Vec<u8> = (0..4 * width * height)
                    .map(|i| (i * 17 + i / 37 + frame * 53) as u8)
                    .collect();
                let (source, _) = source_of(width, height, &rgba);
                pixels::from_rgba(&nc_dsp::Dsp::new(), &rgba, &mut padded);
                padded.extend_borders();
                assert_eq!(
                    visible.encode(&source).unwrap(),
                    extended.inner.encode(&padded, None).data,
                    "{codec} frame {frame}"
                );
            }
        }
    }

    #[test]
    fn av1_and_vp9_spatial_sparse_enhancement_score_psnr_on_both_layer_sizes() {
        // 1.5x scaling (192x108 enhancement, 128x72 base) with enhancement every 6 TUs.
        let (width, height) = (192, 108);
        let overrides = "spatial.layers=2\nspatial.base_scale=0.6666667\nspatial.enhancement_interval=6\nhold.enable=0";
        for codec in ["av1", "vp9"] {
            let mut encoder = BrowserEncoder::new(
                codec,
                width,
                height,
                140,
                30,
                30,
                "fast",
                Some(overrides.to_owned()),
                false,
                None,
                None,
            )
            .unwrap();
            let mut full_decoder = BrowserDecoder::new(codec).unwrap();
            let mut base_decoder = BrowserDecoder::new(codec).unwrap();
            base_decoder.set_operating_point(1).unwrap();

            for frame in 0..8 {
                let rgba: Vec<u8> = (0..4 * width * height)
                    .map(|i| (i * 13 + i / 29 + frame * 31) as u8)
                    .collect();
                let (source, packed) = source_of(width, height, &rgba);
                let packet = encoder.encode(&source).unwrap();
                full_decoder.decode(&packet).unwrap();
                base_decoder.decode(&packet).unwrap();

                let expected_full = matches!(frame, 0 | 6);
                if expected_full {
                    assert_eq!(
                        (full_decoder.width(), full_decoder.height()),
                        (192, 108),
                        "{codec} frame {frame}"
                    );
                    assert_eq!(full_decoder.spatial_id(), 1, "{codec} frame {frame}");
                } else {
                    assert_eq!(
                        (full_decoder.width(), full_decoder.height()),
                        (128, 72),
                        "{codec} frame {frame}"
                    );
                    assert_eq!(full_decoder.spatial_id(), 0, "{codec} frame {frame}");
                }
                assert_eq!(
                    (base_decoder.width(), base_decoder.height()),
                    (128, 72),
                    "{codec} frame {frame}"
                );
                assert_eq!(base_decoder.spatial_id(), 0, "{codec} frame {frame}");

                let worker_psnr = full_decoder.psnr(&packed).unwrap();
                assert!(
                    worker_psnr > 18.0,
                    "{codec} frame {frame} psnr {worker_psnr}"
                );
                assert!(base_decoder.psnr(&packed).unwrap() > 18.0);
            }
        }
    }

    #[test]
    fn av1_three_layer_l3t1_and_l3t3_operating_points_report_size_and_layer() {
        let (width, height) = (256, 144);
        let mut web_encoder = BrowserEncoder::new(
            "av1",
            width,
            height,
            140,
            30,
            30,
            "fast",
            Some("spatial.layers=3\nhold.enable=0".to_owned()),
            false,
            None,
            None,
        )
        .unwrap();
        let mut l3t1_decoders: Vec<BrowserDecoder> = (0..3u8)
            .map(|op| {
                let mut d = BrowserDecoder::new("av1").unwrap();
                d.set_operating_point(op).unwrap();
                d
            })
            .collect();
        for frame in 0..4 {
            let rgba: Vec<u8> = (0..4 * width * height)
                .map(|i| (i * 11 + i / 23 + frame * 29) as u8)
                .collect();
            let (source, packed) = source_of(width, height, &rgba);
            let packet = web_encoder.encode(&source).unwrap();
            for (op, dec) in l3t1_decoders.iter_mut().enumerate() {
                dec.decode(&packet).unwrap();
                let expected_sid = 2 - op as u8;
                let expected_dim = match expected_sid {
                    0 => (64, 36),
                    1 => (128, 72),
                    _ => (256, 144),
                };
                assert_eq!(dec.spatial_id(), expected_sid, "l3t1 op {op} frame {frame}");
                assert_eq!(
                    (dec.width(), dec.height()),
                    expected_dim,
                    "l3t1 op {op} frame {frame}"
                );
                assert_eq!((dec.stream_width(), dec.stream_height()), (width, height));
                assert!(dec.psnr(&packed).unwrap() > 16.0);
            }
        }

        // Full 9-operating-point L3T3 stream decoded through BrowserDecoder.
        let mut params = nc_encoder::ParamOverrides::default().coding_every_frame();
        params.set("spatial.layers", 3.0);
        params.set("layers.adaptive", 0.0);
        let config = EncoderConfig {
            codec: Codec::Av1,
            width,
            height,
            frame_rate: FrameRate { num: 30, den: 1 },
            quantizer: 140,
            intra_period: 30,
            block_log2: 4,
            key_qp_offset: None,
            temporal_layers: 3,
            signal: nc_encoder::StreamSignal::default(),
            params,
        };
        let mut l3t3_enc = Encoder::<nc_av1::Av1Backend>::new(config).unwrap();
        let mut l3t3_decoders: Vec<BrowserDecoder> = (0..9u8)
            .map(|op| {
                let mut d = BrowserDecoder::new("av1").unwrap();
                d.set_operating_point(op).unwrap();
                d
            })
            .collect();
        let mut pic = Picture::new(width, height, 0).unwrap();
        for frame in 0..5 {
            let rgba: Vec<u8> = (0..4 * width * height)
                .map(|i| (i * 11 + i / 23 + frame * 29) as u8)
                .collect();
            pixels::from_rgba(&nc_dsp::Dsp::new(), &rgba, &mut pic);
            let pkt = l3t3_enc.encode(&pic);
            for (op, dec) in l3t3_decoders.iter_mut().enumerate() {
                dec.decode(&pkt.data).unwrap();
                let expected_sid = 2 - (op as u8) / 3;
                let expected_dim = match expected_sid {
                    0 => (64, 36),
                    1 => (128, 72),
                    _ => (256, 144),
                };
                assert_eq!(dec.spatial_id(), expected_sid, "l3t3 op {op} frame {frame}");
                assert_eq!(
                    (dec.width(), dec.height()),
                    expected_dim,
                    "l3t3 op {op} frame {frame}"
                );
                assert_eq!((dec.stream_width(), dec.stream_height()), (width, height));
            }
        }
    }

    #[test]
    fn av1_three_layer_ksvc_extracted_streams_decode_in_browser() {
        let (width, height) = (256, 144);
        let mut ksvc_encoder = BrowserEncoder::new(
            "av1",
            width,
            height,
            140,
            30,
            30,
            "svc3",
            Some("spatial.inter_layer=2\nhold.enable=0".to_owned()),
            false,
            None,
            None,
        )
        .unwrap();
        let mut ksvc_extractors: Vec<nc_av1::Av1StreamExtractor> = (0..3u8)
            .map(|op| nc_av1::Av1StreamExtractor::new(op).unwrap())
            .collect();
        let mut ksvc_decoders: Vec<BrowserDecoder> = (0..3)
            .map(|_| BrowserDecoder::new("av1").unwrap())
            .collect();
        for frame in 0..4 {
            let rgba: Vec<u8> = (0..4 * width * height)
                .map(|i| (i * 17 + i / 19 + frame * 37) as u8)
                .collect();
            let (source, packed) = source_of(width, height, &rgba);
            let packet = ksvc_encoder.encode(&source).unwrap();
            for op in 0..3usize {
                let (rx_bytes, _) = ksvc_extractors[op]
                    .extract_temporal_unit(&packet)
                    .unwrap()
                    .unwrap();
                let expected_sid = (2 - op) as u8;
                if frame > 0 {
                    assert_eq!(
                        nc_av1::temporal_unit_layers(&rx_bytes).unwrap(),
                        vec![(0, expected_sid)]
                    );
                }
                ksvc_decoders[op].decode(&rx_bytes).unwrap();
                assert_eq!(ksvc_decoders[op].spatial_id(), expected_sid);
                assert!(ksvc_decoders[op].psnr(&packed).unwrap() > 16.0);
            }
        }
    }

    #[test]
    fn vp9_three_layer_and_ksvc_operating_points_decode_in_browser() {
        let (width, height) = (256, 128);
        let mut ksvc_encoder = BrowserEncoder::new(
            "vp9",
            width,
            height,
            140,
            30,
            30,
            "svc3",
            Some("spatial.inter_layer=2\nhold.enable=0".to_owned()),
            false,
            None,
            None,
        )
        .unwrap();
        let mut op_decoders: Vec<BrowserDecoder> = (0..3u8)
            .map(|op| {
                let mut d = BrowserDecoder::new("vp9").unwrap();
                d.set_operating_point(op).unwrap();
                d
            })
            .collect();
        let mut extractors: Vec<nc_vp9::Vp9StreamExtractor> = (0..3u8)
            .map(|op| nc_vp9::Vp9StreamExtractor::new(op).unwrap())
            .collect();
        let mut ext_decoders: Vec<BrowserDecoder> = (0..3)
            .map(|_| BrowserDecoder::new("vp9").unwrap())
            .collect();
        for frame in 0..4 {
            let rgba: Vec<u8> = (0..4 * width * height)
                .map(|i| (i * 17 + i / 19 + frame * 37) as u8)
                .collect();
            let (source, packed) = source_of(width, height, &rgba);
            let packet = ksvc_encoder.encode(&source).unwrap();
            for op in 0..3usize {
                let expected_sid = (2 - op) as u8;
                let expected_dim = match expected_sid {
                    0 => (64, 32),
                    1 => (128, 64),
                    _ => (256, 128),
                };
                op_decoders[op].decode(&packet).unwrap();
                assert_eq!(op_decoders[op].spatial_id(), expected_sid);
                assert_eq!(
                    (op_decoders[op].width(), op_decoders[op].height()),
                    expected_dim
                );
                assert_eq!(
                    (
                        op_decoders[op].stream_width(),
                        op_decoders[op].stream_height()
                    ),
                    (width, height)
                );
                assert!(op_decoders[op].psnr(&packed).unwrap() > 16.0);

                let (rx_bytes, _) = extractors[op].extract_packet(&packet).unwrap().unwrap();
                ext_decoders[op].decode(&rx_bytes).unwrap();
                assert_eq!(
                    (ext_decoders[op].width(), ext_decoders[op].height()),
                    expected_dim
                );
                if op == 0 {
                    assert!(ext_decoders[op].psnr(&packed).unwrap() > 16.0);
                }
            }
        }
    }
}
