//! A codec-independent encoder session behind the C ABI (safe Rust).
//!
//! [`Session`] owns one engine instance for the configured codec and the
//! most recent packet, which the C caller borrows until its next call.

#![forbid(unsafe_code)]

use nc_common::RawFrame;
use nc_encoder::{
    Codec, ColourDescription, ColourInfo, ConfigError, EncodedFrame, Encoder, EncoderConfig,
    FrameRate, IntraRefresh, ParamOverrides, RateControlConfig, RateControlError, StreamSignal,
    TimingSignal,
};

/// Leaf block size the engine starts from (16x16), as the CLI's default.
const DEFAULT_BLOCK_LOG2: u8 = 4;
/// Bits per second in one kbps.
const BPS_PER_KBPS: u64 = 1000;

/// Everything needed to create a session (the C `nc_encoder_config`).
#[derive(Debug, Clone, PartialEq)]
pub struct SessionConfig {
    /// Output format.
    pub codec: Codec,
    /// Visible picture size.
    pub width: usize,
    /// Visible picture height.
    pub height: usize,
    /// Nominal frame rate.
    pub frame_rate: FrameRate,
    /// Fixed quantizer, or the rate controller's seed (codec native scale).
    pub quantizer: u32,
    /// Frames between key frames (ignored while intra refresh is on).
    pub intra_period: u32,
    /// Temporal layers (1–3).
    pub temporal_layers: u8,
    /// Frames per intra-refresh sweep; 0 = periodic keys.
    pub intra_refresh_period: u32,
    /// Frames between long-term reference marks; 0 = no LTR.
    pub ltr_interval: u32,
    /// Target bitrate in kbps; 0 = fixed quantizer.
    pub target_kbps: u32,
    /// Peak bitrate in kbps; 0 = the controller's default (twice the target).
    pub peak_kbps: u32,
    /// Rate-control buffer in milliseconds; 0 = the controller's default.
    pub buffer_ms: u32,
    /// Drop frames that would overflow the buffer at the maximum quantizer.
    pub frame_drop: bool,
    /// Colour and timing signalling.
    pub signal: StreamSignal,
    /// AV1 film grain synthesis (`nc_av1::enable_film_grain`); other codecs
    /// reject it.
    pub film_grain: bool,
    /// Its strength; `None` = the default.
    pub film_grain_strength: Option<f32>,
    /// Heuristic parameters set away from their defaults.
    pub params: ParamOverrides,
}

impl SessionConfig {
    /// A fixed-quantizer configuration with the CLI's defaults.
    #[must_use]
    pub fn new(codec: Codec, width: usize, height: usize) -> Self {
        Self {
            codec,
            width,
            height,
            frame_rate: FrameRate { num: 30, den: 1 },
            quantizer: default_quantizer(codec),
            intra_period: 300,
            temporal_layers: 1,
            intra_refresh_period: 0,
            ltr_interval: 0,
            target_kbps: 0,
            peak_kbps: 0,
            buffer_ms: 0,
            frame_drop: false,
            signal: StreamSignal::default(),
            film_grain: false,
            film_grain_strength: None,
            params: ParamOverrides::default(),
        }
    }

    /// The rate-control settings, or `None` for a fixed quantizer.
    fn rate_control(&self) -> Option<RateControlConfig> {
        (self.target_kbps > 0).then(|| {
            let mut config = RateControlConfig::new(u64::from(self.target_kbps) * BPS_PER_KBPS);
            if self.peak_kbps > 0 {
                config.peak_bps = Some(u64::from(self.peak_kbps) * BPS_PER_KBPS);
            }
            if self.buffer_ms > 0 {
                config.buffer_ms = self.buffer_ms;
            }
            config.frame_drop = self.frame_drop;
            config
        })
    }
}

/// The CLI's starting quantizer per codec (QP 32; AV1/VP9 qindex 180).
#[must_use]
pub fn default_quantizer(codec: Codec) -> u32 {
    if matches!(codec, Codec::Av1 | Codec::Vp9) {
        180
    } else {
        32
    }
}

/// A colour description from the three code points, or `None` when all
/// are zero (the C default: unsignalled).
#[must_use]
pub fn colour_info(code_points: [u8; 3], full_range: bool) -> ColourInfo {
    let [primaries, transfer, matrix] = code_points;
    ColourInfo {
        description: (code_points != [0; 3]).then_some(ColourDescription {
            primaries,
            transfer,
            matrix,
        }),
        full_range,
    }
}

/// The C timing code (0 nominal, 1 constant, 2 variable).
#[must_use]
pub fn timing_signal(code: u8) -> Option<TimingSignal> {
    match code {
        0 => Some(TimingSignal::Nominal),
        1 => Some(TimingSignal::Constant),
        2 => Some(TimingSignal::Variable),
        _ => None,
    }
}

/// Why a session call failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionError {
    /// The configuration was rejected.
    Config(ConfigError),
    /// Rate-control settings were rejected.
    RateControl(RateControlError),
    /// The frame's planes do not match the configured size.
    Frame,
}

impl From<ConfigError> for SessionError {
    fn from(e: ConfigError) -> Self {
        Self::Config(e)
    }
}

impl From<RateControlError> for SessionError {
    fn from(e: RateControlError) -> Self {
        Self::RateControl(e)
    }
}

/// One engine instance per codec.
#[derive(Debug)]
enum AnyEncoder {
    H264(Box<Encoder<nc_h264::H264Backend>>),
    Hevc(Box<Encoder<nc_hevc::HevcBackend>>),
    Av1(Box<Encoder<nc_av1::Av1Backend>>),
    Vp9(Box<Encoder<nc_vp9::Vp9Backend>>),
}

/// Calls `$method` on whichever engine `$encoder` holds.
macro_rules! dispatch {
    ($encoder:expr, $e:ident => $call:expr) => {
        match $encoder {
            AnyEncoder::H264($e) => $call,
            AnyEncoder::Hevc($e) => $call,
            AnyEncoder::Av1($e) => $call,
            AnyEncoder::Vp9($e) => $call,
        }
    };
}

/// An encoder session: the engine plus the last packet.
#[derive(Debug)]
pub struct Session {
    encoder: AnyEncoder,
    /// The rate-control settings in force (`None`: fixed quantizer).
    rate: Option<RateControlConfig>,
    packet: EncodedFrame,
}

impl Session {
    /// Create a session.
    ///
    /// # Errors
    ///
    /// A [`SessionError`] if the engine rejects the configuration.
    pub fn new(config: &SessionConfig) -> Result<Self, SessionError> {
        let mut params = config.params.clone();
        if config.intra_refresh_period > 0 {
            params.set("refresh.period", config.intra_refresh_period as f32);
        }
        if config.ltr_interval > 0 {
            params.set("ltr.interval", config.ltr_interval as f32);
        }
        if config.film_grain {
            nc_av1::enable_film_grain(config.codec, config.film_grain_strength, &mut params)?;
        }
        let engine = EncoderConfig {
            codec: config.codec,
            width: config.width,
            height: config.height,
            frame_rate: config.frame_rate,
            quantizer: config.quantizer,
            intra_period: config.intra_period,
            block_log2: DEFAULT_BLOCK_LOG2,
            key_qp_offset: None,
            temporal_layers: config.temporal_layers,
            signal: config.signal,
            params,
        };
        let mut encoder = match config.codec {
            Codec::H264 => AnyEncoder::H264(Box::new(Encoder::new(engine)?)),
            Codec::Hevc => AnyEncoder::Hevc(Box::new(Encoder::new(engine)?)),
            Codec::Av1 => AnyEncoder::Av1(Box::new(Encoder::new(engine)?)),
            Codec::Vp9 => AnyEncoder::Vp9(Box::new(Encoder::new(engine)?)),
            Codec::Av2 | Codec::Vvc => {
                return Err(ConfigError::Unsupported("the C API does not offer AV2 or VVC").into());
            }
        };
        let rate = config.rate_control();
        if let Some(rate) = rate {
            dispatch!(&mut encoder, e => e.set_rate_control(rate))?;
        }
        Ok(Self {
            encoder,
            rate,
            packet: EncodedFrame {
                data: Vec::new(),
                metadata: nc_encoder::FrameMetadata {
                    index: 0,
                    frame_type: nc_encoder::FrameType::Key,
                    temporal_layer: 0,
                    timestamp_us: 0,
                    dropped: true,
                    held: false,
                    quantizer: 0,
                    references: nc_encoder::ReferencedFrames::NONE,
                },
                layers: Vec::new(),
            },
        })
    }

    /// Encode `frame` (captured at `timestamp_us`, if known); the packet
    /// stays available through [`Session::packet`] until the next call.
    ///
    /// # Errors
    ///
    /// [`SessionError::Frame`] if the frame does not have the configured size.
    pub fn encode(
        &mut self,
        frame: &RawFrame<'_>,
        timestamp_us: Option<i64>,
    ) -> Result<&EncodedFrame, SessionError> {
        self.packet = dispatch!(&mut self.encoder, e => match timestamp_us {
            Some(t) => e.encode_raw_at(frame, t),
            None => e.encode_raw(frame),
        })
        .map_err(|_| SessionError::Frame)?;
        Ok(&self.packet)
    }

    /// Whether the engine wants the second frame before the first
    /// ([`Session::look_ahead`]; `Encoder::wants_look_ahead`).
    #[must_use]
    pub fn wants_look_ahead(&self) -> bool {
        dispatch!(&self.encoder, e => e.wants_look_ahead())
    }

    /// Hand over the second frame before the first is encoded
    /// (`Encoder::look_ahead`); it is still encoded in turn.
    ///
    /// # Errors
    ///
    /// [`SessionError::Frame`] if the frame does not have the configured size.
    pub fn look_ahead(&mut self, next: &RawFrame<'_>) -> Result<(), SessionError> {
        dispatch!(&mut self.encoder, e => e.look_ahead(next)).map_err(|_| SessionError::Frame)
    }

    /// The configured visible size.
    #[must_use]
    pub fn size(&self) -> (usize, usize) {
        let config = dispatch!(&self.encoder, e => e.config());
        (config.width, config.height)
    }

    /// The most recent packet.
    #[must_use]
    pub fn packet(&self) -> &EncodedFrame {
        &self.packet
    }

    /// Code the next frame as a key frame.
    pub fn request_keyframe(&mut self) {
        dispatch!(&mut self.encoder, e => e.request_keyframe());
    }

    /// Change the target (and peak) bitrate without restarting, or switch
    /// from a fixed quantizer to rate control.
    ///
    /// # Errors
    ///
    /// [`SessionError::RateControl`] for an invalid target.
    pub fn set_bitrate(&mut self, target_kbps: u32, peak_kbps: u32) -> Result<(), SessionError> {
        self.set_rate(target_kbps, peak_kbps, 0)
    }

    /// [`Session::set_bitrate`] with a new buffer size too (`buffer_ms`;
    /// 0 keeps the current one): a transport's new bandwidth estimate,
    /// applied from the next frame on (`RateController::reconfigure`).
    ///
    /// # Errors
    ///
    /// [`SessionError::RateControl`] for an invalid target or buffer.
    pub fn set_rate(
        &mut self,
        target_kbps: u32,
        peak_kbps: u32,
        buffer_ms: u32,
    ) -> Result<(), SessionError> {
        let target = u64::from(target_kbps) * BPS_PER_KBPS;
        let mut config = self.rate.unwrap_or_else(|| RateControlConfig::new(target));
        config.target_bps = target;
        config.peak_bps = (peak_kbps > 0).then(|| u64::from(peak_kbps) * BPS_PER_KBPS);
        if buffer_ms > 0 {
            config.buffer_ms = buffer_ms;
        }
        dispatch!(&mut self.encoder, e => e.set_rate_control(config))?;
        self.rate = Some(config);
        Ok(())
    }

    /// The receiver decoded coded frame `index`.
    pub fn acknowledge(&mut self, index: u64) {
        dispatch!(&mut self.encoder, e => e.acknowledge(index));
    }

    /// The receiver decoded spatial layer `spatial_id` (`0..=2`) of
    /// temporal unit `tu_index` (`FrameMetadata::index`).
    pub fn acknowledge_layer(&mut self, tu_index: u64, spatial_id: u8) {
        dispatch!(&mut self.encoder, e => e.acknowledge_layer(tu_index, spatial_id));
    }

    /// The receiver is missing coded frame `index`: the next frame recovers
    /// from the acknowledged long-term reference, or is a key frame.
    pub fn report_loss(&mut self, index: u64) {
        dispatch!(&mut self.encoder, e => e.report_loss(index));
    }

    /// The receiver is missing spatial layer `spatial_id` (`0..=2`) of
    /// temporal unit `tu_index` (`FrameMetadata::index`).
    pub fn report_layer_loss(&mut self, tu_index: u64, spatial_id: u8) {
        dispatch!(&mut self.encoder, e => e.report_layer_loss(tu_index, spatial_id));
    }

    /// Enable (period ≥ 2), change or disable (0) intra refresh.
    ///
    /// # Errors
    ///
    /// [`SessionError::Config`] when refresh is not possible here.
    pub fn set_intra_refresh(&mut self, period: u32) -> Result<(), SessionError> {
        let refresh = (period > 0).then_some(IntraRefresh { period });
        dispatch!(&mut self.encoder, e => e.set_intra_refresh(refresh))?;
        Ok(())
    }

    /// Code frames from the next one on at `width` x `height`: a key frame
    /// with new parameter sets (`Encoder::set_resolution`); frames passed
    /// to [`Session::encode`] must then have the new size.
    ///
    /// # Errors
    ///
    /// [`SessionError::Config`] for a size the codec rejects (the session
    /// is unchanged).
    pub fn set_resolution(&mut self, width: usize, height: usize) -> Result<(), SessionError> {
        dispatch!(&mut self.encoder, e => e.set_resolution(width, height))?;
        Ok(())
    }

    /// Untimed frames from the next one on are `den / num` seconds apart
    /// (`Encoder::set_frame_rate`); timestamped frames need nothing.
    ///
    /// # Errors
    ///
    /// [`SessionError::Config`] for a zero numerator or denominator.
    pub fn set_frame_rate(&mut self, num: u32, den: u32) -> Result<(), SessionError> {
        dispatch!(&mut self.encoder, e => e.set_frame_rate(FrameRate { num, den }))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Session, SessionConfig};
    use nc_common::RawFrame;
    use nc_encoder::Codec;

    #[test]
    fn encodes_nv12_frames_in_every_codec() {
        let (w, h) = (64, 48);
        let mut frame = vec![0u8; w * h * 3 / 2];
        for codec in [Codec::H264, Codec::Hevc, Codec::Av1] {
            let mut config = SessionConfig::new(codec, w, h);
            config.target_kbps = 200;
            let mut session = Session::new(&config).unwrap();
            for index in 0..4u8 {
                frame
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, v)| *v = (i as u8).wrapping_add(index));
                let raw = RawFrame::packed_nv12(&frame, w, h).unwrap();
                let packet = session
                    .encode(&raw, Some(i64::from(index) * 33_333))
                    .unwrap();
                assert_ne!(packet.data.len(), 0);
            }
            session.set_bitrate(100, 0).unwrap();
            session.set_rate(80, 160, 250).unwrap();
            assert!(session.set_rate(0, 0, 0).is_err());
            session.set_intra_refresh(5).unwrap();
            session.request_keyframe();
            let raw = RawFrame::packed_nv12(&frame, w, h).unwrap();
            let key = session.encode(&raw, None).unwrap();
            assert_eq!(key.metadata.frame_type, nc_encoder::FrameType::Key);
        }
    }

    #[test]
    fn resolution_change_starts_a_key_frame_at_the_new_size() {
        let (w, h, small_w, small_h) = (64, 48, 32, 24);
        let frame = vec![90u8; w * h * 3 / 2];
        let small = vec![60u8; small_w * small_h * 3 / 2];
        for codec in [Codec::H264, Codec::Hevc, Codec::Av1] {
            let mut config = SessionConfig::new(codec, w, h);
            config.target_kbps = 200;
            // The frames repeat one picture.
            config.params = nc_encoder::ParamOverrides::default().coding_every_frame();
            let mut session = Session::new(&config).unwrap();
            for _ in 0..3 {
                session
                    .encode(&RawFrame::packed_nv12(&frame, w, h).unwrap(), None)
                    .unwrap();
            }
            assert!(session.set_resolution(31, 24).is_err(), "odd width");
            session.set_resolution(small_w, small_h).unwrap();
            assert_eq!(session.size(), (small_w, small_h));
            let raw = RawFrame::packed_nv12(&small, small_w, small_h).unwrap();
            let packet = session.encode(&raw, None).unwrap();
            assert_eq!(packet.metadata.frame_type, nc_encoder::FrameType::Key);
            assert_eq!(packet.metadata.index, 3, "the frame index continues");
            let next = session.encode(&raw, None).unwrap();
            assert_eq!(next.metadata.frame_type, nc_encoder::FrameType::Inter);
            let old = RawFrame::packed_nv12(&frame, w, h).unwrap();
            assert!(session.encode(&old, None).is_err(), "old-size frame");
        }
    }

    #[test]
    fn frame_rate_change_spaces_untimed_frames() {
        let (w, h) = (64, 48);
        let frame = vec![50u8; w * h * 3 / 2];
        let mut session = Session::new(&SessionConfig::new(Codec::H264, w, h)).unwrap();
        let raw = RawFrame::packed_nv12(&frame, w, h).unwrap();
        let mut times = Vec::new();
        for index in 0..4 {
            if index == 2 {
                session.set_frame_rate(15, 1).unwrap();
            }
            times.push(session.encode(&raw, None).unwrap().metadata.timestamp_us);
        }
        assert_eq!(times, [0, 33_333, 100_000, 166_667]);
        assert!(session.set_frame_rate(0, 1).is_err());
    }

    #[test]
    fn film_grain_is_av1_only() {
        let (w, h) = (64, 48);
        let frame = vec![77u8; w * h * 3 / 2];
        let raw = RawFrame::packed_i420(&frame, w, h).unwrap();
        for codec in [Codec::H264, Codec::Hevc, Codec::Av1] {
            let mut config = SessionConfig::new(codec, w, h);
            config.film_grain = true;
            config.film_grain_strength = Some(0.5);
            match Session::new(&config) {
                Ok(mut session) => {
                    assert_eq!(codec, Codec::Av1);
                    assert_ne!(session.encode(&raw, None).unwrap().data.len(), 0);
                }
                Err(e) => assert_ne!(codec, Codec::Av1, "{e:?}"),
            }
        }
    }

    /// Without a long-term reference a loss costs one key frame; repeated
    /// or late reports of frames before it cost nothing more.
    #[test]
    fn stale_loss_reports_need_no_second_key() {
        let (w, h) = (64, 48);
        let mut session = Session::new(&SessionConfig::new(Codec::Av1, w, h)).unwrap();
        // Every frame differs, so none is held (`hold.rs`) and frame n has
        // index n.
        let encode = |session: &mut Session, n: u8| {
            let frame = vec![40 + 30 * n; w * h * 3 / 2];
            let raw = RawFrame::packed_i420(&frame, w, h).unwrap();
            session.encode(&raw, None).unwrap().metadata
        };
        for n in 0..3 {
            assert_eq!(encode(&mut session, n).index, u64::from(n));
        }
        session.report_loss(2);
        let key = encode(&mut session, 3);
        assert_eq!(key.frame_type, nc_encoder::FrameType::Key);
        assert_eq!(key.references.iter().count(), 0);
        session.report_loss(2);
        session.report_loss(1);
        let next = encode(&mut session, 4);
        assert_eq!(next.frame_type, nc_encoder::FrameType::Inter);
        assert_eq!(next.references.iter().collect::<Vec<_>>(), [3]);
    }

    #[test]
    fn loss_feedback_recovers_from_the_long_term_reference() {
        let (w, h) = (64, 48);
        let frame = vec![77u8; w * h * 3 / 2];
        let raw = RawFrame::packed_i420(&frame, w, h).unwrap();
        let mut config = SessionConfig::new(Codec::Av1, w, h);
        config.ltr_interval = 2;
        let mut session = Session::new(&config).unwrap();
        for index in 0..4 {
            session.encode(&raw, None).unwrap();
            session.acknowledge(index);
        }
        session.report_loss(4);
        let recovery = session.encode(&raw, None).unwrap();
        assert_eq!(recovery.metadata.frame_type, nc_encoder::FrameType::Inter);
        session.report_loss(5);
        session.acknowledge(4);
        // Losing the recovery frame too: the acknowledged LTR (frame 2,
        // marked every second frame) still recovers.
        let again = session.encode(&raw, None).unwrap();
        assert_eq!(again.metadata.frame_type, nc_encoder::FrameType::Inter);
    }

    #[test]
    fn spatial_layer_loss_repairs_s1_without_keyframe_and_s0_loss_forces_keyframe() {
        let (w, h) = (64, 48);
        for codec in [Codec::Av1, Codec::Vp9] {
            let mut config = SessionConfig::new(codec, w, h);
            config.params = nc_encoder::ParamOverrides::default().coding_every_frame();
            config.params.set("spatial.layers", 2.0);
            let mut session = Session::new(&config).unwrap();
            let encode = |session: &mut Session, n: u8| {
                let frame = vec![30 + 25 * n; w * h * 3 / 2];
                let raw = RawFrame::packed_i420(&frame, w, h).unwrap();
                session.encode(&raw, None).unwrap().clone()
            };
            let tu0 = encode(&mut session, 0);
            assert_eq!(tu0.metadata.frame_type, nc_encoder::FrameType::Key);
            let tu1 = encode(&mut session, 1);
            assert_eq!(tu1.metadata.frame_type, nc_encoder::FrameType::Inter);

            // S1 lost on TU 1 while S0 is acknowledged: TU 2 repairs S1 without an S0 keyframe.
            session.acknowledge_layer(1, 0);
            session.report_layer_loss(1, 1);
            let tu2 = encode(&mut session, 2);
            assert_eq!(tu2.metadata.frame_type, nc_encoder::FrameType::Inter);
            assert_eq!(tu2.layers.len(), 2);
            assert!(tu2.layers[1].enhancement_repair);
            assert_eq!(
                tu2.layers[1].references.iter().collect::<Vec<_>>(),
                vec![tu2.layers[0].picture_index]
            );

            // S0 lost on TU 2: TU 3 is a full key temporal unit.
            session.report_layer_loss(2, 0);
            let tu3 = encode(&mut session, 3);
            assert_eq!(tu3.metadata.frame_type, nc_encoder::FrameType::Key);
        }
    }

    #[test]
    fn av1_three_layer_loss_repairs_s2_from_s1_without_disturbing_s1() {
        let (w, h) = (96, 64);
        let mut config = SessionConfig::new(Codec::Av1, w, h);
        config.params = nc_encoder::ParamOverrides::default().coding_every_frame();
        config.params.set("spatial.layers", 3.0);
        let mut session = Session::new(&config).unwrap();
        let encode = |session: &mut Session, n: u8| {
            let frame = vec![24 + 20 * n; w * h * 3 / 2];
            let raw = RawFrame::packed_i420(&frame, w, h).unwrap();
            session.encode(&raw, None).unwrap().clone()
        };
        let _tu0 = encode(&mut session, 0);
        let _tu1 = encode(&mut session, 1);

        // S2 lost on TU 1 while S0 and S1 arrived: TU 2 repairs S2 from S1 only,
        // while S1 keeps its temporal reference!
        session.acknowledge_layer(1, 0);
        session.acknowledge_layer(1, 1);
        session.report_layer_loss(1, 2);
        let tu2 = encode(&mut session, 2);
        assert_eq!(tu2.metadata.frame_type, nc_encoder::FrameType::Inter);
        assert_eq!(tu2.layers.len(), 3);
        assert!(!tu2.layers[1].enhancement_repair);
        assert!(tu2.layers[2].enhancement_repair);
        assert_eq!(
            tu2.layers[2].references.iter().collect::<Vec<_>>(),
            vec![tu2.layers[1].picture_index]
        );
    }
}
