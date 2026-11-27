//! RTP from the command line (`nanocodec-rtp`, docs/handoff/rtp.md):
//!
//! - `encode ... --rtp OUT`: the encoder's units packetized as they are
//!   coded, with the layer information only the encoder has (the AV1
//!   Dependency Descriptor, VP9 layer indices and references);
//! - `packetize`: a stream file (IVF or Annex B) to RTP, payload format
//!   only (no descriptor: a file does not say what each frame references);
//! - `depacketize`: RTP back to a stream file, through the receiver's
//!   reorder window and loss handling.
//!
//! Captures are pcap (`.pcap`) or RFC 4571 framed streams (any other
//! name; the `rtpstreampay` / `rtpstreamdepay` format of `GStreamer`).

use std::io::Write;
use std::path::{Path, PathBuf};

use nc_encoder::{
    Codec, CodecBackend, EncodedFrame, Encoder, rtp_layer_frames, rtp_layer_structure,
};
use nc_rtp::dump::{self, CapturedPacket, PcapWriter};
use nc_rtp::{
    Depacketizer, DepacketizerConfig, LayerFilter, Packetizer, PacketizerConfig, PayloadFormat,
    UnitLayers, Vp9Mode, rtp_timestamp,
};

use crate::framing::{self, IvfWriter};
use crate::stdio::{self, Output};

/// IVF output of `depacketize` counts time in RTP clock ticks.
const IVF_RTP_TIME_BASE: u32 = nc_rtp::CLOCK_RATE;
/// `packetize` frame rate when an Annex-B input gives no timing.
const DEFAULT_FPS: u32 = 30;
/// Microseconds per second.
const MICROS: u64 = 1_000_000;

/// The RTP payload format of `codec`.
pub(crate) fn payload_format(codec: Codec) -> Result<PayloadFormat, String> {
    match codec {
        Codec::H264 => Ok(PayloadFormat::H264),
        Codec::Hevc => Ok(PayloadFormat::Hevc),
        Codec::Vvc => Ok(PayloadFormat::Vvc),
        Codec::Av1 => Ok(PayloadFormat::Av1),
        Codec::Vp9 => Ok(PayloadFormat::Vp9),
        Codec::Av2 => Err("AV2 has no RTP payload format".into()),
    }
}

/// Where packets go: a pcap capture or an RFC 4571 stream.
enum CaptureSink {
    Pcap(PcapWriter<Output>),
    Framed(Output),
}

impl CaptureSink {
    fn create(path: &Path) -> Result<Self, String> {
        let out = stdio::create(path)?;
        if path.extension().is_some_and(|e| e == "pcap") {
            Ok(Self::Pcap(PcapWriter::new(out).map_err(|e| e.to_string())?))
        } else {
            Ok(Self::Framed(out))
        }
    }

    fn write(&mut self, time_us: u64, packet: &[u8]) -> Result<(), String> {
        match self {
            Self::Pcap(writer) => writer.write(time_us, packet),
            Self::Framed(out) => dump::write_framed(out, packet),
        }
        .map_err(|e| e.to_string())
    }

    fn finish(self) -> Result<(), String> {
        let mut out = match self {
            Self::Pcap(writer) => writer.into_inner(),
            Self::Framed(out) => out,
        };
        out.flush().map_err(|e| e.to_string())
    }
}

/// Read every RTP packet (RTCP left out) of a capture.
fn read_capture(path: &Path) -> Result<Vec<CapturedPacket>, String> {
    let bytes = stdio::read_all(path)?;
    let pcap = bytes.len() >= 4
        && [
            [0xD4, 0xC3, 0xB2, 0xA1],
            [0xA1, 0xB2, 0xC3, 0xD4],
            [0x4D, 0x3C, 0xB2, 0xA1],
            [0xA1, 0xB2, 0x3C, 0x4D],
        ]
        .contains(&[bytes[0], bytes[1], bytes[2], bytes[3]]);
    let packets = if pcap {
        dump::read_pcap(bytes.as_slice())
    } else {
        dump::read_framed(bytes.as_slice())
    }
    .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(packets
        .into_iter()
        .filter(|p| !dump::is_rtcp(&p.data))
        .collect())
}

/// The Dependency Descriptor extension id a `--dd-id`-style option gives:
/// `none`, an id, or (absent) 1 for AV1 and none for the other codecs.
pub(crate) fn descriptor_id(
    codec: Codec,
    value: Option<&str>,
    flag: &str,
) -> Result<Option<u8>, String> {
    match value {
        None => Ok((codec == Codec::Av1).then_some(1)),
        Some("none") => Ok(None),
        Some(v) => v
            .parse::<u8>()
            .map(Some)
            .map_err(|_| format!("bad {flag} {v}")),
    }
}

/// `encode --rtp PATH [--rtp-mtu BYTES] [--rtp-dd-id ID|none]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RtpOptions {
    path: PathBuf,
    codec: Codec,
    mtu: usize,
    descriptor_id: Option<u8>,
}

impl RtpOptions {
    /// The `--rtp` options of `encode`, if `--rtp` is given.
    pub(crate) fn parse(args: &[String], codec: Codec) -> Result<Option<Self>, String> {
        let get = |flag: &str| {
            args.iter()
                .position(|a| a == flag)
                .and_then(|i| args.get(i + 1))
                .map(String::as_str)
        };
        let Some(path) = get("--rtp") else {
            return Ok(None);
        };
        let mtu = get("--rtp-mtu").map_or(Ok(nc_rtp::DEFAULT_MTU), |v| {
            v.parse().map_err(|_| format!("bad --rtp-mtu {v}"))
        })?;
        Ok(Some(Self {
            path: path.into(),
            codec,
            mtu,
            descriptor_id: descriptor_id(codec, get("--rtp-dd-id"), "--rtp-dd-id")?,
        }))
    }
}

/// Packetize one coded frame into `out`, when `--rtp` is on.
pub(crate) fn write<B: CodecBackend>(
    out: Option<&mut RtpOutput>,
    frame: &EncodedFrame,
    encoder: &Encoder<B>,
) -> Result<(), String> {
    out.map_or(Ok(()), |out| out.write(frame, encoder))
}

/// `encode --rtp OUT`: packets of every coded unit, written as coded.
pub(crate) struct RtpOutput {
    packetizer: Packetizer,
    sink: CaptureSink,
    /// Capture time of the first unit, so capture times start at zero.
    first_us: Option<i64>,
}

impl RtpOutput {
    /// Open the capture `--rtp` names.
    pub(crate) fn create(options: &RtpOptions) -> Result<Self, String> {
        let config = PacketizerConfig {
            mtu: options.mtu,
            dependency_descriptor_id: options.descriptor_id,
            ..PacketizerConfig::new(payload_format(options.codec)?)
        };
        Ok(Self {
            packetizer: Packetizer::new(config).map_err(|e| e.to_string())?,
            sink: CaptureSink::create(&options.path)?,
            first_us: None,
        })
    }

    /// Packetize one coded frame.
    fn write<B: CodecBackend>(
        &mut self,
        frame: &EncodedFrame,
        encoder: &Encoder<B>,
    ) -> Result<(), String> {
        let frames = rtp_layer_frames(frame);
        if frames.is_empty() {
            return Ok(());
        }
        let structure = rtp_layer_structure(encoder);
        let time = frame.metadata.timestamp_us;
        let first = *self.first_us.get_or_insert(time);
        let packets = self
            .packetizer
            .packetize(
                &frame.data,
                rtp_timestamp(time),
                Some(UnitLayers {
                    structure: &structure,
                    frames: &frames,
                }),
            )
            .map_err(|e| format!("--rtp: {e}"))?;
        let capture_us = u64::try_from(time - first).unwrap_or(0);
        for packet in &packets {
            self.sink.write(capture_us, packet)?;
        }
        Ok(())
    }

    pub(crate) fn finish(self) -> Result<(), String> {
        self.sink.finish()
    }
}

/// Command-line options of `packetize` and `depacketize`.
struct Options {
    codec: Codec,
    input: PathBuf,
    output: PathBuf,
    mtu: usize,
    payload_type: Option<u8>,
    fps: u32,
    vp9_mode: Vp9Mode,
    descriptor_id: Option<u8>,
    deliver_after_loss: bool,
    /// `--layers S:T`: forward only these layers first, as an SFU would.
    layers: Option<(u8, u8)>,
}

fn parse_options(args: &[String]) -> Result<Options, String> {
    let get = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let number = |flag: &str, default: u64| -> Result<u64, String> {
        get(flag).map_or(Ok(default), |v| {
            v.parse().map_err(|_| format!("bad value for {flag}: {v}"))
        })
    };
    let codec = crate::parse_codec(&get("--codec").ok_or("missing --codec")?)?;
    let payload_type = get("--payload-type")
        .map(|v| {
            v.parse::<u8>()
                .map_err(|_| format!("bad --payload-type {v}"))
        })
        .transpose()?;
    let descriptor_id = descriptor_id(codec, get("--dd-id").as_deref(), "--dd-id")?;
    Ok(Options {
        codec,
        input: get("--input").ok_or("missing --input")?.into(),
        output: get("--output").ok_or("missing --output")?.into(),
        mtu: usize::try_from(number("--mtu", nc_rtp::DEFAULT_MTU as u64)?)
            .map_err(|_| "bad --mtu")?,
        payload_type,
        fps: u32::try_from(number("--fps", u64::from(DEFAULT_FPS))?).map_err(|_| "bad --fps")?,
        vp9_mode: match get("--vp9-mode").as_deref() {
            None | Some("flexible") => Vp9Mode::Flexible,
            Some("non-flexible") => Vp9Mode::NonFlexible,
            Some(other) => return Err(format!("bad --vp9-mode {other}")),
        },
        descriptor_id,
        deliver_after_loss: args.iter().any(|a| a == "--deliver-after-loss"),
        layers: get("--layers")
            .map(|v| {
                let (s, t) = v.split_once(':').ok_or(format!("bad --layers {v} (S:T)"))?;
                Ok::<_, String>((
                    s.parse().map_err(|_| format!("bad --layers {v}"))?,
                    t.parse().map_err(|_| format!("bad --layers {v}"))?,
                ))
            })
            .transpose()?,
    })
}

/// `nanocodec packetize`: a stream file to an RTP capture.
pub(crate) fn run_packetize(args: &[String]) -> Result<(), String> {
    let options = parse_options(args)?;
    let mut config = PacketizerConfig::new(payload_format(options.codec)?);
    config.mtu = options.mtu;
    config.vp9_mode = options.vp9_mode;
    if let Some(pt) = options.payload_type {
        config.payload_type = pt;
    }
    let mut packetizer = Packetizer::new(config).map_err(|e| e.to_string())?;
    let mut sink = CaptureSink::create(&options.output)?;
    let mut input = stdio::open(&options.input)?;
    let mut units = 0u64;
    let mut send = |unit: &[u8], time_us: u64| -> Result<(), String> {
        let timestamp = rtp_timestamp(i64::try_from(time_us).unwrap_or(i64::MAX));
        let packets = packetizer
            .packetize(unit, timestamp, None)
            .map_err(|e| format!("unit {units}: {e}"))?;
        for packet in &packets {
            sink.write(time_us, packet)?;
        }
        units += 1;
        Ok(())
    };
    match options.codec {
        Codec::Av1 | Codec::Vp9 => {
            framing::read_ivf(&mut input, |header, frame| {
                if options.codec == Codec::Vp9 && nc_vp9::split_superframe(frame.packet).is_some() {
                    return Err("VP9 superframes (spatial layers) are not carried over RTP".into());
                }
                let rate = u64::from(u32::from_le_bytes(
                    header.bytes[16..20].try_into().expect("4 bytes"),
                ));
                let scale = u64::from(u32::from_le_bytes(
                    header.bytes[20..24].try_into().expect("4 bytes"),
                ));
                let pts = u64::from_le_bytes(frame.header[4..12].try_into().expect("8 bytes"));
                send(frame.packet, pts * scale * MICROS / rate)
            })?;
        }
        codec => {
            let frame_us = MICROS / u64::from(options.fps.max(1));
            let mut index = 0;
            framing::read_annex_b(codec, &mut input, |unit| {
                index += 1;
                send(unit, (index - 1) * frame_us)
            })?;
        }
    }
    sink.finish()?;
    eprintln!("packetize: {units} units");
    Ok(())
}

/// The decoder-ready stream output of `depacketize`.
enum StreamSink {
    Ivf(IvfWriter<Output>),
    AnnexB(Output),
}

/// `nanocodec depacketize`: an RTP capture to a stream file.
pub(crate) fn run_depacketize(args: &[String]) -> Result<(), String> {
    let options = parse_options(args)?;
    let format = payload_format(options.codec)?;
    let config = DepacketizerConfig {
        dependency_descriptor_id: options.descriptor_id,
        deliver_after_loss: options.deliver_after_loss,
        payload_type: options.payload_type,
        ..DepacketizerConfig::new(format)
    };
    let mut receiver = Depacketizer::new(config);
    let out = stdio::create(&options.output)?;
    let mut sink = match options.codec {
        Codec::Av1 | Codec::Vp9 => StreamSink::Ivf(
            IvfWriter::new(out, options.codec, 0, 0, IVF_RTP_TIME_BASE, 1, 0)
                .map_err(|e| e.to_string())?,
        ),
        _ => StreamSink::AnnexB(out),
    };
    let mut filter = match (options.layers, options.descriptor_id) {
        (Some((spatial, temporal)), Some(id)) => Some(LayerFilter::new(id, spatial, temporal)),
        (Some(_), None) => return Err("--layers needs the Dependency Descriptor (--dd-id)".into()),
        _ => None,
    };
    let mut first_timestamp = None;
    let mut rejected = 0u64;
    let mut write_ready =
        |receiver: &mut Depacketizer, sink: &mut StreamSink| -> Result<(), String> {
            while let Some(unit) = receiver.pop() {
                let first = *first_timestamp.get_or_insert(unit.timestamp);
                let pts = u64::from(unit.timestamp.wrapping_sub(first));
                let data = match unit.frames.as_slice() {
                    [single] => single.clone(),
                    many => {
                        nc_vp9::pack_superframe(&many.iter().map(Vec::as_slice).collect::<Vec<_>>())
                    }
                };
                match sink {
                    StreamSink::Ivf(writer) => writer.write_frame(&data, pts),
                    StreamSink::AnnexB(out) => out.write_all(&data),
                }
                .map_err(|e| e.to_string())?;
            }
            Ok(())
        };
    for packet in read_capture(&options.input)? {
        let data = match &mut filter {
            Some(filter) => match filter.filter(&packet.data).map_err(|e| e.to_string())? {
                Some(forwarded) => forwarded,
                None => continue,
            },
            None => packet.data,
        };
        if receiver.push(&data).is_err() {
            rejected += 1;
        }
        write_ready(&mut receiver, &mut sink)?;
    }
    receiver.flush();
    write_ready(&mut receiver, &mut sink)?;
    match sink {
        StreamSink::Ivf(writer) => writer.finish().map(drop),
        StreamSink::AnnexB(mut out) => out.flush(),
    }
    .map_err(|e| e.to_string())?;
    let stats = receiver.stats();
    eprintln!(
        "depacketize: {} packets ({} rejected, {} lost, {} late), {} units delivered, {} dropped",
        stats.packets,
        rejected,
        stats.lost_packets,
        stats.late_packets,
        stats.units,
        stats.dropped_units
    );
    Ok(())
}
