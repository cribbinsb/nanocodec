//! The encoder information message: which encoder wrote a stream, its
//! version and its settings, as x264 and x265 write them (a
//! `user_data_unregistered` SEI), so a stream found later says how it was
//! made. It never changes decoding: every decoder may ignore it.
//!
//! The engine builds the payload once for every codec: [`ENCODER_INFO_UUID`]
//! then the settings text (`uuid_iso_iec_11578` + `user_data_payload_byte`
//! of H.264 §D.1.7, HEVC §D.2.7 and ITU-T H.274 `user_data_unregistered()`).
//! Each backend only wraps it in its own syntax, on every key frame that
//! carries the sequence headers ([`crate::CodecBackend::set_encoder_info`]):
//! the headers are repeated at every key so a receiver can start there, and
//! the message rides with them (about 100-250 bytes per key; x264 writes
//! its much longer one on the first frame only). AV1 and AV2 write it with
//! the first key frame's sequence header only (again after the settings
//! change, a rate change say): their message is a metadata
//! OBU of a private type, which dav1d logs at every occurrence ("Unknown
//! Metadata OBU type 6"), and libaom and SVT-AV1 write none. VP9 has no SEI
//! or metadata syntax, so its streams carry none.
//!
//! The text is deterministic (no time, host or commit), so identical
//! settings code identical streams on every build of one release: stream
//! hashes stay comparable across commits (`tools/dev/preset_identity.py`,
//! the verified-stream cache). The version is the release
//! (`CARGO_PKG_VERSION`); `nanocodec --version` names the commit.
//!
//! The settings: the codec, the preset the parameter overrides contain (the
//! largest built-in rung whose every setting they carry; `medium` has
//! none), the rate mode (target, buffer, peak and bounds, or the fixed
//! quantizer), the key period and the temporal and spatial layer counts.
//! `stream.encoder_info` = 0 writes nothing.
//!
//! **Why no parameter overrides.** Listing the overrides the preset does
//! not explain (as x264 lists its options) made every stream depend on
//! settings that do not change coding: the contract tests that two settings
//! code byte-identical streams (`debug.recode_identity`, non-base overlays
//! equal to the ordinary values) and a preset pin at its old default
//! (`tools/dev/preset_identity.py`) then differed in their key frames
//! alone (4 AV1 decoder tests failed, 2026-11-23). The preset name and the
//! stream-level settings identify an encode; parameter experiments do not
//! belong in a product stream.

use std::fmt::Write as _;

use crate::config::{Codec, EncoderConfig, StreamRate};
use crate::params::{ParamOverrides, SpatialParams, StreamParams};
use crate::presets::{preset_params, speeds};
use crate::rate_control::RateControlConfig;

/// The `uuid_iso_iec_11578` that marks nanocodec's message (a random
/// version-4 UUID, 72c2229f-9f14-4f78-b07b-0018d2fb5b43, fixed for good).
pub const ENCODER_INFO_UUID: [u8; 16] = [
    0x72, 0xc2, 0x22, 0x9f, 0x9f, 0x14, 0x4f, 0x78, 0xb0, 0x7b, 0x00, 0x18, 0xd2, 0xfb, 0x5b, 0x43,
];

/// The encoder name the message starts with.
pub const ENCODER_NAME: &str = "nanocodec";

/// The release the message names.
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The settings an encoder was created with, from which each message is
/// written; the preset is read from the caller's overrides, before the
/// backend's defaults are layered under them.
#[derive(Debug, Clone)]
pub(crate) struct EncoderInfo {
    codec: Codec,
    /// `None` when `stream.encoder_info` is 0.
    preset: Option<&'static str>,
    temporal_layers: u8,
    spatial_layers: u8,
    key_period: u32,
    /// The rate control in force, `None` at a fixed quantizer.
    rate: Option<RateControlConfig>,
}

impl EncoderInfo {
    /// The settings of `config`, whose overrides were `caller` before the
    /// backend's defaults were layered under them.
    pub fn new(config: &EncoderConfig, caller: &ParamOverrides) -> Self {
        let enabled = StreamParams::from_overrides(&config.params).encoder_info >= 0.5;
        // The range [1, 255] makes the cast exact.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let spatial_layers = SpatialParams::from_overrides(&config.params)
            .layers
            .round()
            .clamp(1., f32::from(u8::MAX)) as u8;
        Self {
            codec: config.codec,
            preset: enabled.then(|| preset_of(config.codec, spatial_layers, caller)),
            temporal_layers: config.temporal_layers,
            spatial_layers,
            key_period: config.intra_period,
            rate: None,
        }
    }

    /// Note the rate control now in force (`None`: a fixed quantizer).
    pub fn set_rate(&mut self, rate: Option<RateControlConfig>) {
        self.rate = rate;
    }

    /// The delivery rate of the rate control in force, `None` at a fixed
    /// quantizer.
    pub fn stream_rate(&self) -> Option<StreamRate> {
        self.rate.as_ref().map(RateControlConfig::stream_rate)
    }

    /// The message payload for a stream coded at fixed `quantizer` (when
    /// no rate control is in force), or `None` when it is off.
    pub fn payload(&self, quantizer: u32) -> Option<Vec<u8>> {
        let preset = self.preset?;
        let mut text = format!(
            "{ENCODER_NAME} {VERSION} - {} encoder - options: preset={preset}",
            codec_name(self.codec)
        );
        match &self.rate {
            Some(rate) => write_rate(&mut text, rate),
            None => {
                let _ = write!(text, " rc=cqp qp={quantizer}");
            }
        }
        let _ = write!(
            text,
            " keyint={} layers={}",
            self.key_period, self.temporal_layers
        );
        if self.spatial_layers > 1 {
            let _ = write!(text, " spatial={}", self.spatial_layers);
        }
        let mut payload = ENCODER_INFO_UUID.to_vec();
        payload.extend_from_slice(text.as_bytes());
        Some(payload)
    }
}

/// The rate settings: target and buffer always, the rest where set.
fn write_rate(text: &mut String, rate: &RateControlConfig) {
    let _ = write!(
        text,
        " rc=cbr bitrate={} buffer_ms={}",
        rate.target_bps, rate.buffer_ms
    );
    if let Some(peak) = rate.peak_bps {
        let _ = write!(text, " peak={peak}");
    }
    if let Some(min) = rate.min_quantizer {
        let _ = write!(text, " qmin={min}");
    }
    if let Some(max) = rate.max_quantizer {
        let _ = write!(text, " qmax={max}");
    }
    if rate.frame_drop {
        text.push_str(" frame_drop=1");
    }
    if rate.strict_buffer {
        text.push_str(" strict_buffer=1");
    }
}

/// The codec's name in the message.
const fn codec_name(codec: Codec) -> &'static str {
    match codec {
        Codec::H264 => "H.264",
        Codec::Hevc => "HEVC",
        Codec::Vvc => "VVC",
        Codec::Av1 => "AV1",
        Codec::Vp9 => "VP9",
        Codec::Av2 => "AV2",
    }
}

/// The largest built-in rung of `codec` all of whose settings `params`
/// carry (`medium`, which has none, at least).
fn preset_of(codec: Codec, spatial_layers: u8, params: &ParamOverrides) -> &'static str {
    let contained = |preset: &ParamOverrides| {
        preset
            .names()
            .all(|name| params.get(name).map(f32::to_bits) == preset.get(name).map(f32::to_bits))
    };
    speeds(codec)
        .into_iter()
        .filter_map(|speed| {
            let preset = preset_params(codec, spatial_layers, speed.name()).ok()?;
            contained(&preset).then(|| (speed.name(), preset.names().count()))
        })
        .max_by_key(|&(_, settings)| settings)
        .map_or("medium", |(name, _)| name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{FrameRate, StreamSignal};

    fn config(codec: Codec, params: ParamOverrides) -> EncoderConfig {
        EncoderConfig {
            codec,
            width: 64,
            height: 64,
            frame_rate: FrameRate { num: 30, den: 1 },
            quantizer: 30,
            intra_period: 60,
            temporal_layers: 1,
            block_log2: 4,
            key_qp_offset: None,
            signal: StreamSignal::default(),
            params,
        }
    }

    fn text(info: &EncoderInfo) -> String {
        let payload = info.payload(30).unwrap();
        assert_eq!(payload[..16], ENCODER_INFO_UUID);
        String::from_utf8(payload[16..].to_vec()).unwrap()
    }

    #[test]
    fn names_the_preset_whatever_else_is_overridden() {
        let mut params = preset_params(Codec::Hevc, 1, "fast").unwrap();
        params.set("scene.detect", 1.0);
        let info = {
            let c = config(Codec::Hevc, params);
            EncoderInfo::new(&c, &c.params)
        };
        assert_eq!(
            text(&info),
            "nanocodec 0.6.0 - HEVC encoder - options: preset=fast rc=cqp qp=30 keyint=60 layers=1"
                .replace("0.6.0", VERSION)
        );
    }

    #[test]
    fn default_settings_are_medium_and_rate_control_is_named() {
        let mut info = {
            let c = config(Codec::Av1, ParamOverrides::default());
            EncoderInfo::new(&c, &c.params)
        };
        info.set_rate(Some(RateControlConfig::new(600_000)));
        assert!(text(&info).ends_with(
            "AV1 encoder - options: preset=medium rc=cbr bitrate=600000 buffer_ms=1000 keyint=60 layers=1"
        ));
    }

    #[test]
    fn the_switch_turns_it_off() {
        let mut params = ParamOverrides::default();
        params.set("stream.encoder_info", 0.0);
        let c = config(Codec::H264, params);
        assert!(EncoderInfo::new(&c, &c.params).payload(30).is_none());
    }
}
