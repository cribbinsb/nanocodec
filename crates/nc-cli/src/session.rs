//! `nanocodec session`: the encoder and a receiver stepped frame by frame
//! over stdin/stdout, so a network model outside (`eval/nceval/losssim.py`)
//! can lose frames and return receiver feedback in a closed loop.
//!
//! The options are `encode`'s (no `--output`: nothing is written but the
//! receiver's pictures), plus `--display shown.yuv` (one stream-size picture
//! per input frame: what the viewer saw), `--receiver transport|bitstream`,
//! and optional `--operating-point N` (`AV1 §6.4.1`, default `0`).
//! Commands, one per line on stdin:
//!
//! ```text
//! ack N [S]       the receiver's acknowledgement of frame/TU N (or spatial
//!                 layer S in {0,1}) arrived
//! loss N [S]      its report that frame/TU N (or spatial layer S in {0,1})
//!                 is missing arrived
//! encode          code the next input frame; reply
//!                 "frame INDEX key|inter|held|dropped LAYER BYTES REFS [op=S:T layers=B0:B1 [repair=s1]]"
//! receive N A [E] frame N reaches the receiver: single-layer uses A in {0,1};
//!                 spatial streams may pass base arrival A and enhancement E
//!                 independently; reply "received N STATUS"
//! show            write the picture on screen (once per input frame)
//! ```
//!
//! A frame is whole or lost (one lost packet loses it): no partial frames.
//! STATUS is `good` (decoded, bit-exact with the encoder's reconstruction),
//! `lost`, a freeze reason (`nc_common::outcome::FreezeReason`: not
//! decoded, the receiver keeps showing its last good picture):
//! `missing-reference` (a reference was lost; until a key frame or a frame
//! whose references it holds arrives), `missing-key` (no key frame
//! decoded yet), `corrupt` (the decoder rejected the data), or `mismatch`
//! (decoded, but not what the encoder reconstructed: a decoder that
//! predicted from a picture it did not hold).
//!
//! Receivers: `transport` (the default) decodes a frame only when it holds
//! every frame the transport's dependency signal lists
//! (`FrameMetadata::references`, as the AV1 RTP Dependency Descriptor
//! carries). `bitstream` hands every arrived frame to the decoder and
//! relies on the stream alone, as a browser receiver without that signal
//! must: H.264 `frame_num` gaps and alternating long-term indices, HEVC
//! reference picture sets (keys after the first are CRA pictures, so POC
//! keeps counting), AV1 frame ids (which a session turns on). VP9 and AV2
//! (AVM) have no such syntax: their decoders cannot tell a lost reference,
//! and their receivers need the transport signal
//! (`docs/handoff/loss-resilience.md`).
//!
//! On spatial streams (`spatial.layers == 2`), a receiver subscribed to a base
//! operating point (`max_spatial_id == 0`) drops enhancement OBUs up front and
//! decodes `S0` only (`AV1 §6.2.1`, `§7.18`). On the full operating point,
//! losing an enhancement (`S1`) packet while `S0` arrives strips the lost `S1`
//! OBU, decodes `S0` bit-exactly (`STATUS = base-only`), and keeps playing the
//! base stream while `S1` stays frozen (`STATUS = enh-frozen`) until a key or
//! `S1` recovery unit arrives.

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, Write};
use std::path::PathBuf;

use nc_common::decode::DecodeError;
use nc_common::outcome::{FrameOutcome, ReceivedFrames};
use nc_common::{Picture, PlaneIndex, yuv};
use nc_encoder::{
    Codec, CodecBackend, EncodedFrame, Encoder, FrameType, LayerFrameMetadata, OperatingPointSpec,
};

use crate::decode::Decoder;
use crate::display::Display;
use crate::input::FrameReader;
use crate::{EncodeArgs, encoder_config, md5, stdio};

/// Sample value of the picture shown before anything was decoded.
const BLANK_SAMPLE: u8 = 128;

/// How the receiver decides a frame can be decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReceiverMode {
    Transport,
    Bitstream,
}

/// Run a session over stdin/stdout. `raw` holds the session options.
pub(crate) fn run(args: &EncodeArgs, raw: &[String]) -> Result<(), String> {
    let get = |flag: &str| {
        let i = raw.iter().position(|a| a == flag)?;
        raw.get(i + 1).cloned()
    };
    let display = PathBuf::from(get("--display").ok_or("session: missing --display")?);
    let mode = match get("--receiver").as_deref() {
        None | Some("transport") => ReceiverMode::Transport,
        Some("bitstream") => ReceiverMode::Bitstream,
        Some(other) => return Err(format!("bad --receiver {other} (transport, bitstream)")),
    };
    let operating_point = match get("--operating-point") {
        None => 0,
        Some(v) => {
            let op = crate::decode::parse_operating_point(&v, "session: ")?;
            if op != 0 && !matches!(args.codec, Codec::Av1 | Codec::Vp9) {
                return Err("session: --operating-point applies to AV1 and VP9 only".into());
            }
            op
        }
    };
    if !args.resizes.is_empty() || !args.rate_changes.is_empty() || args.timestamps.is_some() {
        return Err("session: --resize, --rate-change and --timestamps are not supported".into());
    }
    if [&args.input, &display]
        .into_iter()
        .any(|path| stdio::is_stdio(path))
    {
        return Err(
            "session: --input and --display cannot be '-' (stdin/stdout carry the session)".into(),
        );
    }
    let out = stdio::create(&display)?;
    let receiver = Receiver::new(
        args.codec,
        mode,
        operating_point,
        (args.width, args.height),
        out,
    );
    match args.codec {
        Codec::Av1 => serve::<nc_av1::Av1Backend>(args, operating_point, receiver),
        Codec::Vp9 => serve::<nc_vp9::Vp9Backend>(args, operating_point, receiver),
        Codec::Av2 => serve::<nc_av2::Av2Backend>(args, operating_point, receiver),
        Codec::H264 => serve::<nc_h264::H264Backend>(args, operating_point, receiver),
        Codec::Hevc => serve::<nc_hevc::HevcBackend>(args, operating_point, receiver),
        Codec::Vvc => serve::<nc_vvc::VvcBackend>(args, operating_point, receiver),
    }
}

/// Coded frame or multi-layer temporal unit awaiting `receive`.
enum Sent {
    Single {
        frame: EncodedFrame,
        recon_md5: String,
    },
    Spatial(Box<SpatialSent>),
}

struct SpatialSent {
    /// Prefix bitstreams stripped to `spatial_id <= k` for `k in 0..MAX_SPATIAL_LAYERS`.
    prefix_data: [Option<Vec<u8>>; nc_encoder::MAX_SPATIAL_LAYERS],
    /// Standalone single-layer bitstreams (`spatial_id == k`) for K-SVC / independent
    /// non-key temporal units where lower layers are not forwarded.
    single_data: [Option<Vec<u8>>; nc_encoder::MAX_SPATIAL_LAYERS],
    /// Coded spatial layers in OBU order (`0, 1[, 2]`): `(metadata, recon_md5)`.
    layers: Vec<(LayerFrameMetadata, String)>,
}

/// Answer commands until stdin ends.
fn serve<B: CodecBackend>(
    args: &EncodeArgs,
    operating_point: u8,
    mut receiver: Receiver,
) -> Result<(), String> {
    let mut config = encoder_config(args);
    if args.codec == Codec::Av1 {
        // A session returns loss feedback, so AV1 codes frame ids (unless
        // `--param av1.frame_ids=0`) and a bitstream receiver can tell a lost
        // reference without a long-term reference too.
        config.params = config
            .params
            .layered_over(&[(nc_av1::FRAME_IDS_PARAM, 1.0)]);
    }
    let mut encoder = Encoder::<B>::new(config).map_err(|e| e.to_string())?;
    if let Some(rate_control) = args.rate_control {
        encoder
            .set_rate_control(rate_control)
            .map_err(|error| error.to_string())?;
    }
    let spatial_op = resolve_operating_point(&encoder, operating_point)?;
    let mut input = FrameReader::new(
        stdio::open_raw(&args.input)?,
        args.input_format,
        args.width,
        args.height,
    );
    input.look_ahead_second(&mut encoder)?;
    let stdout = std::io::stdout();
    let mut reply = stdout.lock();
    let mut sent: HashMap<u64, Sent> = HashMap::new();
    for line in std::io::stdin().lock().lines() {
        let line = line.map_err(|e| e.to_string())?;
        let (command, value) = line.split_once(' ').unwrap_or((line.as_str(), ""));
        match command {
            "ack" => match parse_feedback(value, &line)? {
                (idx, Some(sid)) => encoder.acknowledge_layer(idx, sid),
                (idx, None) => encoder.acknowledge(idx),
            },
            "loss" => match parse_feedback(value, &line)? {
                (idx, Some(sid)) => encoder.report_layer_loss(idx, sid),
                (idx, None) => encoder.report_loss(idx),
            },
            "encode" => {
                let frame = input
                    .encode_next(&mut encoder, None)?
                    .ok_or("session: input ended")?;
                let (text, entry) = prepare_encoded(frame, &encoder, spatial_op)?;
                writeln!(reply, "{text}").map_err(|e| e.to_string())?;
                if let Some((idx, item)) = entry {
                    sent.insert(idx, item);
                }
            }
            "receive" => {
                let (index, arrived) = parse_receive(value, &line)?;
                let item = sent
                    .remove(&index)
                    .ok_or_else(|| format!("session: frame {index} was not sent"))?;
                let status = match item {
                    Sent::Single { frame, recon_md5 } => {
                        if arrived[0] {
                            receiver.decode_single(&frame, &recon_md5)?
                        } else {
                            "lost"
                        }
                    }
                    Sent::Spatial(tu) => receiver.decode_spatial(
                        &tu,
                        spatial_op.expect("spatial operating point resolved"),
                        arrived,
                    )?,
                };
                writeln!(reply, "received {index} {status}").map_err(|e| e.to_string())?;
            }
            "show" => receiver.show()?,
            _ => return Err(format!("session: unknown command '{line}'")),
        }
        reply.flush().map_err(|e| e.to_string())?;
    }
    receiver.out.flush().map_err(|e| e.to_string())
}

fn resolve_operating_point<B: CodecBackend>(
    encoder: &Encoder<B>,
    operating_point: u8,
) -> Result<Option<OperatingPointSpec>, String> {
    let Some(scfg) = encoder.spatial_config() else {
        let t_layers = encoder.config().temporal_layers;
        if operating_point >= t_layers {
            return Err(format!(
                "session: --operating-point {operating_point} out of range (0..{t_layers})"
            ));
        }
        return Ok(None);
    };
    scfg.operating_points
        .get(usize::from(operating_point))
        .copied()
        .map(Some)
        .ok_or_else(|| {
            format!(
                "session: --operating-point {operating_point} out of range (0..{})",
                scfg.operating_points.len()
            )
        })
}

fn parse_feedback(value: &str, line: &str) -> Result<(u64, Option<u8>), String> {
    let parts: Vec<&str> = value.split_whitespace().collect();
    let bad = || format!("session: bad command '{line}'");
    match parts.as_slice() {
        [idx] => Ok((idx.parse().map_err(|_| bad())?, None)),
        [idx, sid] => {
            let sid = sid.parse::<u8>().map_err(|_| bad())?;
            if usize::from(sid) >= nc_encoder::MAX_SPATIAL_LAYERS {
                return Err(bad());
            }
            Ok((idx.parse().map_err(|_| bad())?, Some(sid)))
        }
        _ => Err(bad()),
    }
}

fn parse_receive(
    value: &str,
    line: &str,
) -> Result<(u64, [bool; nc_encoder::MAX_SPATIAL_LAYERS]), String> {
    let parts: Vec<&str> = value.split_whitespace().collect();
    let bad = || format!("session: bad command '{line}'");
    let parse_bit = |s: &str| match s {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => Err(bad()),
    };
    match parts.as_slice() {
        [idx, a] => {
            let bit = parse_bit(a)?;
            Ok((idx.parse().map_err(|_| bad())?, [bit, bit, bit]))
        }
        [idx, a0, a1] => {
            let b0 = parse_bit(a0)?;
            let b1 = parse_bit(a1)?;
            Ok((idx.parse().map_err(|_| bad())?, [b0, b1, b1]))
        }
        [idx, a0, a1, a2] => Ok((
            idx.parse().map_err(|_| bad())?,
            [parse_bit(a0)?, parse_bit(a1)?, parse_bit(a2)?],
        )),
        _ => Err(bad()),
    }
}

fn prepare_encoded<B: CodecBackend>(
    frame: EncodedFrame,
    encoder: &Encoder<B>,
    spatial_op: Option<OperatingPointSpec>,
) -> Result<(String, Option<(u64, Sent)>), String> {
    let coded = !frame.metadata.dropped && !frame.metadata.held;
    if !coded {
        return Ok((describe(&frame), None));
    }
    let idx = frame.metadata.index;
    if frame.layers.is_empty() {
        let recon_md5 = md5::picture(encoder.reconstruction());
        let text = describe(&frame);
        return Ok((text, Some((idx, Sent::Single { frame, recon_md5 }))));
    }
    let op = spatial_op.ok_or("session: spatial frame without spatial operating point")?;
    let scfg = encoder
        .spatial_config()
        .ok_or("session: missing spatial config")?;
    let mut prefix_data: [Option<Vec<u8>>; nc_encoder::MAX_SPATIAL_LAYERS] =
        std::array::from_fn(|_| None);
    let mut single_data: [Option<Vec<u8>>; nc_encoder::MAX_SPATIAL_LAYERS] =
        std::array::from_fn(|_| None);
    for layer_meta in &frame.layers {
        let sid = layer_meta.spatial_id;
        let (stripped, single) = match encoder.config().codec {
            Codec::Av1 => {
                let prefix = if usize::from(sid) + 1 == frame.layers.len() {
                    frame.data.clone()
                } else {
                    nc_av1::filter_temporal_unit_layers(&frame.data, |_, s| s <= sid)
                        .map_err(|e| e.to_string())?
                };
                let only = if sid == 0 {
                    prefix.clone()
                } else {
                    nc_av1::filter_temporal_unit_layers(&frame.data, |_, s| s == sid)
                        .map_err(|e| e.to_string())?
                };
                (prefix, Some(only))
            }
            Codec::Vp9 => {
                let prefix = if sid == 0 && frame.layers.len() > 1 {
                    nc_vp9::strip_superframe_to_base(&frame.data).map_err(|e| e.to_string())?
                } else {
                    frame.data.clone()
                };
                let only = (sid == 0).then(|| prefix.clone());
                (prefix, only)
            }
            codec => {
                return Err(format!(
                    "session: {codec:?} does not support spatial layers"
                ));
            }
        };
        prefix_data[usize::from(sid)] = Some(stripped);
        single_data[usize::from(sid)] = single;
    }
    let mut layer_entries = Vec::with_capacity(frame.layers.len());
    for layer_meta in &frame.layers {
        let recon = encoder
            .layer_reconstruction(layer_meta.spatial_id)
            .ok_or_else(|| format!("session: missing S{} reconstruction", layer_meta.spatial_id))?;
        layer_entries.push((*layer_meta, md5::picture(recon)));
    }
    let text = format_spatial_reply(&frame, scfg, op);
    let sent = SpatialSent {
        prefix_data,
        single_data,
        layers: layer_entries,
    };
    Ok((text, Some((idx, Sent::Spatial(Box::new(sent))))))
}

fn format_spatial_reply(
    frame: &EncodedFrame,
    scfg: &nc_encoder::SpatialConfig,
    op: OperatingPointSpec,
) -> String {
    let num_layers = usize::from(scfg.layers);
    let mut layer_bytes = vec![0usize; num_layers];
    let mut layer_il = vec![0u8; num_layers];
    for layer_meta in &frame.layers {
        if let Some(slot) = layer_bytes.get_mut(usize::from(layer_meta.spatial_id)) {
            *slot = layer_meta.byte_len;
        }
        if let Some(slot) = layer_il.get_mut(usize::from(layer_meta.spatial_id)) {
            *slot = u8::from(layer_meta.inter_layer_ref);
        }
    }
    let layers_field = layer_bytes
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(":");
    let il_tag = if scfg.inter_layer == nc_encoder::InterLayerMode::On {
        String::new()
    } else {
        format!(
            " il={}",
            layer_il
                .iter()
                .map(u8::to_string)
                .collect::<Vec<_>>()
                .join(":")
        )
    };
    let mut repair_tag = String::new();
    if frame
        .layers
        .iter()
        .any(|l| l.spatial_id == 1 && l.enhancement_repair)
    {
        repair_tag.push_str(" repair=s1");
    }
    if frame
        .layers
        .iter()
        .any(|l| l.spatial_id == 2 && l.enhancement_repair)
    {
        repair_tag.push_str(" repair=s2");
    }
    format!(
        "{} op={}:{} layers={layers_field}{il_tag}{repair_tag}",
        describe(frame),
        op.max_spatial_id,
        op.max_temporal_id
    )
}

/// The base 6-token `encode` reply for `frame`.
fn describe(frame: &EncodedFrame) -> String {
    let meta = &frame.metadata;
    let kind = if meta.dropped {
        "dropped"
    } else if meta.held {
        "held"
    } else if meta.frame_type == FrameType::Key {
        "key"
    } else {
        "inter"
    };
    let references: Vec<String> = meta.references.iter().map(|i| i.to_string()).collect();
    let references = if references.is_empty() {
        "-".to_string()
    } else {
        references.join(",")
    };
    format!(
        "frame {} {kind} {} {} {references}",
        meta.index,
        meta.temporal_layer,
        frame.data.len()
    )
}

/// The receiving end: our decoder and the picture on screen.
struct Receiver {
    decoder: Decoder,
    mode: ReceiverMode,
    /// Frames decoded bit-exactly: those later frames may predict from.
    good: ReceivedFrames,
    /// Spatial `picture_index` values decoded bit-exactly.
    good_spatial: HashSet<u64>,
    /// The picture on screen (stream size), once one was decoded.
    shown: Option<Picture>,
    display: Display,
    stream: (usize, usize),
    out: stdio::Output,
}

impl Receiver {
    fn new(
        codec: Codec,
        mode: ReceiverMode,
        operating_point: u8,
        stream: (usize, usize),
        out: stdio::Output,
    ) -> Self {
        let mut decoder = Decoder::new(codec);
        decoder.skip_film_grain();
        if codec == Codec::Av1 {
            decoder
                .set_operating_point(operating_point)
                .expect("operating point validated by parse_operating_point");
        }
        Self {
            decoder,
            mode,
            good: ReceivedFrames::default(),
            good_spatial: HashSet::new(),
            shown: None,
            display: Display::default(),
            stream,
            out,
        }
    }

    fn decode_single(
        &mut self,
        frame: &EncodedFrame,
        recon_md5: &str,
    ) -> Result<&'static str, String> {
        let meta = &frame.metadata;
        let references: Vec<u64> = match self.mode {
            ReceiverMode::Transport => meta.references.iter().collect(),
            ReceiverMode::Bitstream => Vec::new(),
        };
        let decoder = &mut self.decoder;
        let (outcome, decoded) = self
            .good
            .receive(meta.index, &references, || decoder.decode(&frame.data));
        let decoded = match (outcome, decoded) {
            (FrameOutcome::Frozen { reason, .. }, _) => return Ok(reason.name()),
            (FrameOutcome::Decoded, Some(Some(decoded))) => decoded,
            // A packet that shows nothing (a VP9 hidden frame): nothing new.
            (FrameOutcome::Decoded, _) => return Ok("corrupt"),
        };
        let exact = decoded.md5() == recon_md5;
        if !exact {
            // Not a picture later frames may predict from.
            self.good.forget(meta.index);
        }
        // A mismatching picture is shown too: it is what the viewer sees.
        let decoded = decoded.to_picture()?;
        Self::present(&mut self.display, self.stream, &mut self.shown, &decoded)?;
        Ok(if exact { "good" } else { "mismatch" })
    }

    fn decode_spatial(
        &mut self,
        tu: &SpatialSent,
        op: OperatingPointSpec,
        arrived: [bool; nc_encoder::MAX_SPATIAL_LAYERS],
    ) -> Result<&'static str, String> {
        let (s0_meta, _) = &tu.layers[0];
        if s0_meta.temporal_layer > op.max_temporal_id {
            return Ok("skipped-op");
        }
        // Highest spatial layer in this TU belonging to the subscribed operating point.
        let target_idx = tu
            .layers
            .iter()
            .rposition(|(m, _)| {
                m.spatial_id <= op.max_spatial_id && m.temporal_layer <= op.max_temporal_id
            })
            .unwrap_or(0);
        let (target_meta, _) = &tu.layers[target_idx];
        let target_sid = target_meta.spatial_id;
        let s0_refs_ok = self.mode == ReceiverMode::Bitstream
            || s0_meta
                .references
                .iter()
                .all(|r| self.good_spatial.contains(&r));

        // Walk upward from S0..=target_idx to see whether the full prefix arrived
        // with intact references.
        let mut full_prefix_ok = arrived[0] && s0_refs_ok;
        let mut playable_idx = 0usize;
        let mut stopped_lost = false;
        if full_prefix_ok {
            for idx in 1..=target_idx {
                let (meta, _) = &tu.layers[idx];
                let sid = usize::from(meta.spatial_id);
                let refs_ok = self.mode == ReceiverMode::Bitstream
                    || meta.references.iter().all(|r| {
                        tu.layers[..idx].iter().any(|(lm, _)| lm.picture_index == r)
                            || self.good_spatial.contains(&r)
                    });
                if arrived[sid] && refs_ok {
                    playable_idx = idx;
                } else {
                    full_prefix_ok = false;
                    stopped_lost = !arrived[sid];
                    break;
                }
            }
        }

        if !full_prefix_ok
            && target_idx > 0
            && let Some(status) = self.try_decode_standalone_target(
                tu,
                target_idx,
                arrived[0] && s0_refs_ok,
                arrived[usize::from(target_sid)],
            )?
        {
            return Ok(status);
        }

        if !arrived[0] {
            if arrived[usize::from(target_sid)] {
                return Ok("missing-reference");
            }
            return Ok("lost");
        }
        if !s0_refs_ok {
            return Ok("missing-reference");
        }

        let playable_sid = tu.layers[playable_idx].0.spatial_id;
        let ok_status = if playable_idx == target_idx {
            "good"
        } else if playable_sid == 1 && target_sid == 2 {
            if stopped_lost { "s1-only" } else { "s2-frozen" }
        } else if stopped_lost {
            "base-only"
        } else {
            "enh-frozen"
        };

        self.decode_playable_prefix(tu, playable_idx, ok_status)
    }

    fn decode_playable_prefix(
        &mut self,
        tu: &SpatialSent,
        playable_idx: usize,
        ok_status: &'static str,
    ) -> Result<&'static str, String> {
        let (s0_meta, s0_md5) = &tu.layers[0];
        let playable_sid = tu.layers[playable_idx].0.spatial_id;
        let data = tu.prefix_data[usize::from(playable_sid)]
            .as_deref()
            .ok_or("session: missing stripped prefix data")?;
        if playable_idx == 0 {
            let decoded = match self.decoder.decode(data) {
                Ok(Some(decoded)) => decoded,
                Err(DecodeError::MissingReference) => return Ok("missing-reference"),
                Ok(None) | Err(_) => return Ok("error"),
            };
            let exact = decoded.md5() == *s0_md5;
            if exact {
                self.good_spatial.insert(s0_meta.picture_index);
            }
            let decoded = decoded.to_picture()?;
            Self::present(&mut self.display, self.stream, &mut self.shown, &decoded)?;
            return Ok(if exact { ok_status } else { "mismatch" });
        }

        let layers = match self.decoder.decode_all(data) {
            Ok(layers) if layers.len() == playable_idx + 1 => layers,
            Err(DecodeError::MissingReference) => return Ok("missing-reference"),
            Ok(_) | Err(_) => return Ok("error"),
        };
        let mut all_exact = true;
        for (shown_layer, (meta, expected_md5)) in layers.iter().zip(&tu.layers[..=playable_idx]) {
            all_exact &= shown_layer.md5() == *expected_md5;
            if all_exact {
                self.good_spatial.insert(meta.picture_index);
            }
        }
        let top_decoded = layers[playable_idx].to_picture()?;
        Self::present(
            &mut self.display,
            self.stream,
            &mut self.shown,
            &top_decoded,
        )?;
        Ok(if all_exact { ok_status } else { "mismatch" })
    }

    /// K-SVC / independent non-key unit or standalone `INTRA_ONLY` repair:
    /// when `target_idx > 0` does not depend on lower layers in this TU and
    /// the full prefix is unavailable (e.g. a selective forwarder sent only
    /// `{Sk}`, or an intermediate layer was lost), decode `{Sk}` standalone.
    fn try_decode_standalone_target(
        &mut self,
        tu: &SpatialSent,
        target_idx: usize,
        s0_playable: bool,
        target_arrived: bool,
    ) -> Result<Option<&'static str>, String> {
        let (s0_meta, s0_md5) = &tu.layers[0];
        let (target_meta, target_md5) = &tu.layers[target_idx];
        let target_sid = target_meta.spatial_id;
        let target_refs_ok = self.mode == ReceiverMode::Bitstream
            || target_meta
                .references
                .iter()
                .all(|r| self.good_spatial.contains(&r));
        let Some(single_bytes) = tu.single_data[usize::from(target_sid)].as_deref() else {
            return Ok(None);
        };
        if !target_arrived || target_meta.inter_layer_ref || !target_refs_ok {
            return Ok(None);
        }
        if s0_playable
            && let Some(s0_bytes) = tu.prefix_data[0].as_deref()
            && let Ok(Some(s0_dec)) = self.decoder.decode(s0_bytes)
            && s0_dec.md5() == *s0_md5
        {
            self.good_spatial.insert(s0_meta.picture_index);
        }
        let decoded = match self.decoder.decode(single_bytes) {
            Ok(Some(decoded)) => decoded,
            Err(DecodeError::MissingReference) => return Ok(Some("missing-reference")),
            Ok(None) | Err(_) => return Ok(Some("error")),
        };
        let exact = decoded.md5() == *target_md5;
        if exact {
            self.good_spatial.insert(target_meta.picture_index);
        }
        let decoded = decoded.to_picture()?;
        Self::present(&mut self.display, self.stream, &mut self.shown, &decoded)?;
        Ok(Some(if exact { "good" } else { "mismatch" }))
    }

    /// Put `decoded` on screen. Takes the screen fields rather than `self`:
    /// `decoded` may still borrow the decoder.
    fn present(
        display: &mut Display,
        stream: (usize, usize),
        shown: &mut Option<Picture>,
        decoded: &Picture,
    ) -> Result<(), String> {
        let picture = display.shown(decoded, stream)?;
        match shown.as_mut() {
            Some(shown) => shown.clone_from(picture),
            None => *shown = Some(picture.clone()),
        }
        Ok(())
    }

    /// Write the picture on screen (blank before the first decoded one).
    fn show(&mut self) -> Result<(), String> {
        if self.shown.is_none() {
            let (width, height) = self.stream;
            let mut blank = Picture::new(width, height, 0).map_err(|e| e.to_string())?;
            for plane in PlaneIndex::ALL {
                blank.plane_mut(plane).view_mut().fill(BLANK_SAMPLE);
            }
            self.shown = Some(blank);
        }
        let picture = self.shown.as_ref().expect("set above");
        yuv::write_frame(&mut self.out, picture).map_err(|e| e.to_string())
    }
}
