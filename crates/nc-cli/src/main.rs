//! `nanocodec`: the command-line encoder and subset decoder.
//!
//! ```text
//! nanocodec encode --codec av1|h264|hevc|vvc|vp9|av2 --input in.yuv --width W --height H
//!                  --fps F --qp Q --frames N --output out.{ivf,264,hevc,vvc}
//!                  [--recon recon.yuv] [--display-recon shown.yuv]
//!                  [--block 8|16|32|64] [--intra-period N] [--key-qp-offset D] [--preset NAME] [--params FILE]
//!                  [--param name=value]... [--film-grain [STRENGTH]]
//! nanocodec decode --codec av1|h264|hevc|vvc|vp9|av2 --input stream [--output decoded.yuv] [--discard 1]
//!                  [--frame-md5 frames.md5] [--film-grain 0|1]   (AV1: 0 shows frames grain-free)
//! nanocodec params      # list tunable parameters, defaults, ranges
//! nanocodec --version   # version and the commit it was built from (-dirty: local changes)
//! nanocodec session <encode options> --display shown.yuv [--receiver transport|bitstream]
//! ```
//!
//! `--bitrate kbps` enables realtime rate control; otherwise `--qp` is required.
//! `--qp` with bitrate is an initial seed, and no longer fixes frame quantizers.
//! Input is raw I420, or YV12/NV12 with `--input-format`. `--qp` is the codec's native quantiser (AV1/VP9:
//! `base_q_idx`; H.264, HEVC: QP). `--fps` accepts `30`, `29.97` or `30000/1001`.
//! AV1 and VP9 are written as IVF, H.264 and HEVC as raw Annex-B elementary streams.
//! `--key-qp-offset D` codes key frames D quantiser steps finer than `--qp`
//! (HEVC; an experiment knob, the backend picks a default otherwise).
//! `--recon` gets the reconstruction of every coded frame (what a decoder
//! outputs); `--display-recon` one picture per input frame, what a decoder
//! shows at that frame's time: a frame left uncoded (held as a repeat of the
//! picture, `hold.*`, or dropped by rate control) repeats the last one.
//! `--sse-log frames.csv --sse-reference ref.yuv` logs every shown
//! picture's per-plane squared error against the reference (`sse_log.rs`).
//! `--intra-period N` codes a key frame every N frames and inter frames in
//! between; the default is a key frame every 10 seconds of video.
//! Heuristic parameters: `--preset`, then `--params FILE`, then each
//! `--param name=value`, later settings winning (see `params.rs`).
//! `--film-grain [STRENGTH]` (AV1 only) turns on film grain synthesis
//! (`grain.enable`, and `grain.strength` when given), over those.
//! Any path option given as `-` is stdin (inputs) or stdout (outputs), one
//! of each at most (`stdio.rs`).

#![forbid(unsafe_code)]

mod colour;
mod decode;
mod display;
mod extract;
mod framing;
mod input;
mod md5;
#[cfg(test)]
mod outcome_tests;
mod params;
mod rate_control;
mod session;
mod spatial_stats;
mod sse_log;
#[cfg(feature = "analysis")]
mod stat;
mod stdio;
mod timestamps;

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use crate::framing::IvfWriter;
use crate::spatial_stats::SpatialTelemetry;
use nc_av1::Av1Backend;
use nc_av2::Av2Backend;
use nc_common::{Picture, yuv};

use crate::input::{FrameReader, InputFormat};
use nc_encoder::{
    Codec, CodecBackend, EncodedFrame, Encoder, EncoderConfig, FrameRate, MICROS_PER_SECOND,
    ParamOverrides, RateControlConfig, RateControlStats, ScalabilityMode, StreamSignal,
    TimingSignal,
};
use nc_h264::H264Backend;
use nc_hevc::HevcBackend;
use nc_vp9::Vp9Backend;

/// Default leaf block size during bring-up: 16x16.
const DEFAULT_BLOCK_LOG2: u8 = 4;
/// The AV2 backend accepts only 8x8 leaves (`docs/handoff/vp9-av2.md`).
const AV2_BLOCK_LOG2: u8 = 3;
/// VVC's fixed-partition leaf size (32x32, the smaller CTU); the default
/// planners choose leaves up to the CTU size (`vvc.ctu_size`).
const VVC_BLOCK_LOG2: u8 = 5;
/// Default key-frame interval in seconds when `--intra-period` is not given:
/// the live-stream setting the evaluation uses (10 s).
const DEFAULT_INTRA_PERIOD_SECONDS: u32 = 10;

/// `encode`/`session` path options read, any one of which may be stdin (`-`).
const ENCODE_INPUT_FLAGS: &[&str] = &["--input", "--sse-reference", "--timestamps", "--params"];
/// `encode`/`session` path options written, any one of which may be stdout (`-`).
const ENCODE_OUTPUT_FLAGS: &[&str] = &[
    "--output",
    "--recon",
    "--display-recon",
    "--layer-md5",
    "--spatial-stats",
    "--sse-log",
    "--rate-log",
    "--timestamps-out",
    "--display",
];

#[derive(Debug)]
struct EncodeArgs {
    codec: Codec,
    input: PathBuf,
    input_format: InputFormat,
    /// `--output`: required by `encode`, unused by `session`.
    output: Option<PathBuf>,
    recon: Option<PathBuf>,
    /// `--display-recon`: the picture shown at every input frame's time.
    display_recon: Option<PathBuf>,
    /// `--layer-md5`: CSV of per-layer coded picture dimensions, bytes and MD5.
    layer_md5: Option<PathBuf>,
    /// `--spatial-stats`: JSON summary of spatial reference and byte statistics.
    spatial_stats: Option<PathBuf>,
    /// `--sse-log` and its `--sse-reference`: the shown pictures' errors.
    sse_log: Option<(PathBuf, PathBuf)>,
    width: usize,
    height: usize,
    frame_rate: FrameRate,
    quantizer: u32,
    frames: u32,
    block_log2: u8,
    intra_period: u32,
    key_qp_offset: Option<i32>,
    temporal_layers: u8,
    signal: StreamSignal,
    params: ParamOverrides,
    rate_control: Option<RateControlConfig>,
    /// Mid-stream rate-control changes (`--rate-change`), in frame order.
    rate_changes: Vec<rate_control::RateChange>,
    /// Mid-stream resolution changes (`--resize FRAME:WxH`), in frame
    /// order: the input's frames have the new size from FRAME on.
    resizes: Vec<(u32, usize, usize)>,
    rate_log: Option<PathBuf>,
    /// Input frame times (`--timestamps`), microseconds.
    timestamps: Option<Vec<i64>>,
    timestamps_out: Option<PathBuf>,
    /// Loss simulation (`--simulate-loss N[:S]`): every coded frame is
    /// acknowledged at once, except frame `N` (or spatial layer `S` on TU `N`),
    /// which is reported lost and left out of every output.
    simulate_loss: Option<SimulatedLoss>,
}

/// `--simulate-loss FRAME[:SPATIAL_ID]`.
#[derive(Clone, Copy, Debug)]
struct SimulatedLoss {
    frame: u64,
    spatial_id: Option<u8>,
}

const USAGE: &str = concat!(
    "usage: nanocodec encode --codec av1|h264|hevc|vvc|vp9|av2 --input in.yuv --width W --height H\n",
    "       [--input-format i420|yv12|nv12]\n",
    "       --fps F (--qp Q | --bitrate kbps) --frames N --output out.{ivf,264,hevc,vvc}\n",
    "       [--recon recon.yuv] [--display-recon shown.yuv]\n",
    "       [--sse-log frames.csv --sse-reference ref.yuv]\n",
    "       [--block 8|16|32|64] [--intra-period N]\n",
    "       [--key-qp-offset D] [--temporal-layers 1|2|3]\n",
    "       [--scalability L1T1|L1T2|L1T3|L2T1|L2T2|L2T3|L3T1|L3T2|L3T3]\n",
    "       [--colour bt709|bt601|bt601-625|P,T,M] [--range full|limited]\n",
    "       [--timestamps in.txt] [--timestamps-out out.txt]\n",
    "       [--timing nominal|constant|variable] [--intra-refresh frames]\n",
    "       [--ltr-interval frames] [--simulate-loss frame]\n",
    "       [--preset NAME] [--params FILE] [--param name=value]...\n",
    "       [--film-grain [STRENGTH]]   (AV1 only)\n",
    "       [--buffer-ms ms] [--peak-bitrate kbps] [--min-qp Q] [--max-qp Q]\n",
    "       [--max-qp-step N] [--rate-log frames.csv] [--frame-drop]\n",
    "       [--rate-change FRAME:KBPS[:PEAK_KBPS[:BUFFER_MS]]]... [--resize FRAME:WxH]...\n",
    "       nanocodec decode --codec av1|h264|hevc|vvc|vp9|av2 --input stream [--output decoded.yuv] [--discard 1]\n",
    "                 [--frame-md5 frames.md5] [--film-grain 0|1] [--operating-point N]   (AV1)\n",
    "       nanocodec extract --codec av1|av2|h264|hevc|vvc --temporal-layers N --max-layer M\n",
    "                 --input stream (--output substream | --list)\n",
    "       nanocodec params [--codec av1|h264|hevc|vvc|vp9|av2]\n",
    "       nanocodec stat --codec h264|hevc|av1 --input stream [--json stats.json]   (analysis builds)\n",
    "       nanocodec session <encode options, no --output> --display shown.yuv\n",
    "                 [--receiver transport|bitstream]   (commands on stdin, session.rs)"
);

/// `nanocodec --version`: the package version and the commit the binary was
/// built from (`build.rs`; "-dirty": crates/ or presets/ had local changes).
const VERSION: &str = concat!(
    "nanocodec ",
    env!("CARGO_PKG_VERSION"),
    " ",
    env!("NC_GIT_COMMIT"),
    env!("NC_GIT_DIRTY")
);

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if matches!(args.first().map(String::as_str), Some("--version" | "-V")) {
        println!("{VERSION}");
        return ExitCode::SUCCESS;
    }
    if args.first().map(String::as_str) == Some("params") {
        let codec = args
            .iter()
            .position(|a| a == "--codec")
            .and_then(|i| args.get(i + 1));
        return match params::listing(codec.map(String::as_str)) {
            Ok(text) => {
                print!("{text}");
                ExitCode::SUCCESS
            }
            Err(message) => {
                eprintln!("nanocodec: {message}");
                ExitCode::FAILURE
            }
        };
    }
    if args.first().map(String::as_str) == Some("decode") {
        return match decode::run(&args[1..]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(message) => {
                eprintln!("nanocodec: {message}");
                ExitCode::FAILURE
            }
        };
    }
    #[cfg(feature = "analysis")]
    if args.first().map(String::as_str) == Some("stat") {
        return match stat::run(&args[1..]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(message) => {
                eprintln!("nanocodec: {message}");
                ExitCode::FAILURE
            }
        };
    }
    if args.first().map(String::as_str) == Some("extract") {
        return match extract::run(&args[1..]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(message) => {
                eprintln!("nanocodec: {message}");
                ExitCode::FAILURE
            }
        };
    }
    if args.first().map(String::as_str) == Some("session") {
        return match parse_encode(&args[1..]).and_then(|a| session::run(&a, &args[1..])) {
            Ok(()) => ExitCode::SUCCESS,
            Err(message) => {
                eprintln!("nanocodec: {message}");
                ExitCode::FAILURE
            }
        };
    }
    if args.first().map(String::as_str) != Some("encode") {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    }
    match parse_encode(&args[1..]).and_then(|a| encode(&a)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("nanocodec: {message}");
            ExitCode::FAILURE
        }
    }
}

fn parse_codec(name: &str) -> Result<Codec, String> {
    match name {
        "av1" => Ok(Codec::Av1),
        "vp9" => Ok(Codec::Vp9),
        "av2" => Ok(Codec::Av2),
        "hevc" => Ok(Codec::Hevc),
        "vvc" | "h266" => Ok(Codec::Vvc),
        "h264" => Ok(Codec::H264),
        other => Err(format!(
            "unsupported codec '{other}' (implemented: av1, h264, hevc, vvc, vp9, av2)"
        )),
    }
}

fn parse_block_log2(block: Option<&str>, codec: Codec) -> Result<u8, String> {
    match block {
        None if codec == Codec::Av2 => Ok(AV2_BLOCK_LOG2),
        None if codec == Codec::Vvc => Ok(VVC_BLOCK_LOG2),
        None => Ok(DEFAULT_BLOCK_LOG2),
        Some("8") => Ok(3),
        Some("16") => Ok(4),
        Some("32") => Ok(5),
        Some("64") => Ok(6),
        Some(other) => Err(format!("unsupported --block {other}")),
    }
}

fn parse_encode(args: &[String]) -> Result<EncodeArgs, String> {
    stdio::check_streams(args, ENCODE_INPUT_FLAGS, ENCODE_OUTPUT_FLAGS)?;
    let get = |flag: &str| -> Option<String> {
        let i = args.iter().position(|a| a == flag)?;
        args.get(i + 1).cloned()
    };
    let required = |v: Option<String>, flag: &str| v.ok_or_else(|| format!("missing {flag}"));
    let number = |v: String, flag: &str| {
        v.parse::<u64>()
            .map_err(|_| format!("bad value for {flag}: {v}"))
    };

    let codec = parse_codec(&required(get("--codec"), "--codec")?)?;
    let rate_control = rate_control::parse(args)?;
    let rate_changes = rate_control::changes(args, rate_control)?;
    let timestamps = get("--timestamps")
        .map(|path| {
            timestamps::read(stdio::open(Path::new(&path))?).map_err(|e| format!("{path}: {e}"))
        })
        .transpose()?;
    let block_log2 = parse_block_log2(get("--block").as_deref(), codec)?;
    let frame_rate = parse_fps(&required(get("--fps"), "--fps")?)?;
    Ok(EncodeArgs {
        codec,
        input: required(get("--input"), "--input")?.into(),
        input_format: get("--input-format")
            .map_or(Ok(InputFormat::I420), |v| InputFormat::parse(&v))?,
        output: get("--output").map(PathBuf::from),
        recon: get("--recon").map(PathBuf::from),
        display_recon: get("--display-recon").map(PathBuf::from),
        layer_md5: get("--layer-md5").map(PathBuf::from),
        spatial_stats: get("--spatial-stats").map(PathBuf::from),
        sse_log: match (get("--sse-log"), get("--sse-reference")) {
            (Some(log), Some(reference)) => Some((log.into(), reference.into())),
            (None, None) => None,
            _ => return Err("--sse-log and --sse-reference go together".into()),
        },
        width: number(required(get("--width"), "--width")?, "--width")? as usize,
        height: number(required(get("--height"), "--height")?, "--height")? as usize,
        frame_rate,
        quantizer: match get("--qp") {
            Some(value) => number(value, "--qp")? as u32,
            None if rate_control.is_some() => {
                if matches!(codec, Codec::Av1 | Codec::Vp9 | Codec::Av2) {
                    180
                } else {
                    32
                }
            }
            None => return Err("missing --qp (or specify --bitrate)".into()),
        },
        frames: number(required(get("--frames"), "--frames")?, "--frames")? as u32,
        block_log2,
        intra_period: match get("--intra-period") {
            None => default_intra_period(frame_rate),
            Some(v) => number(v, "--intra-period")? as u32,
        },
        key_qp_offset: get("--key-qp-offset")
            .map(|v| v.parse().map_err(|_| format!("bad --key-qp-offset {v}")))
            .transpose()?,
        temporal_layers: parse_temporal_layers(args)?,
        signal: StreamSignal {
            colour: colour::parse(get)?,
            timing: parse_timing(get("--timing").as_deref(), timestamps.is_some())?,
        },
        params: encode_params(args, codec)?,
        rate_control,
        rate_changes,
        resizes: parse_resizes(args)?,
        rate_log: get("--rate-log").map(PathBuf::from),
        timestamps,
        timestamps_out: get("--timestamps-out").map(PathBuf::from),
        simulate_loss: get("--simulate-loss")
            .as_deref()
            .map(parse_simulate_loss)
            .transpose()?,
    })
}

/// `--simulate-loss FRAME[:SPATIAL_ID]`.
fn parse_simulate_loss(spec: &str) -> Result<SimulatedLoss, String> {
    let bad = || format!("bad value for --simulate-loss: {spec} (expected FRAME or FRAME:LAYER)");
    if let Some((frame, layer)) = spec.split_once(':') {
        Ok(SimulatedLoss {
            frame: frame.parse::<u64>().map_err(|_| bad())?,
            spatial_id: Some(layer.parse::<u8>().map_err(|_| bad())?),
        })
    } else {
        Ok(SimulatedLoss {
            frame: spec.parse::<u64>().map_err(|_| bad())?,
            spatial_id: None,
        })
    }
}

/// `--scalability` when given, rejecting combination with `--temporal-layers`.
fn parse_scalability(args: &[String]) -> Result<Option<ScalabilityMode>, String> {
    let Some(spec) = params::flag_value(args, "--scalability") else {
        return Ok(None);
    };
    if args.iter().any(|a| a == "--temporal-layers") {
        return Err("--scalability and --temporal-layers cannot be combined".into());
    }
    ScalabilityMode::parse(&spec).map(Some).ok_or_else(|| {
        format!("unsupported --scalability '{spec}' (expected L1T1..L1T3, L2T1..L2T3, L3T1..L3T3)")
    })
}

/// Temporal layer count from `--scalability` or `--temporal-layers` (default 1).
fn parse_temporal_layers(args: &[String]) -> Result<u8, String> {
    if let Some(mode) = parse_scalability(args)? {
        return Ok(mode.temporal_layers());
    }
    if !args.iter().any(|a| a == "--temporal-layers") {
        return Ok(1);
    }
    let value = params::flag_value(args, "--temporal-layers")
        .ok_or_else(|| "missing --temporal-layers".to_string())?;
    value
        .parse::<u8>()
        .map_err(|_| format!("bad value for --temporal-layers: {value}"))
}

/// The heuristic parameters (`--preset`, `--params`, `--scalability`, `--param`)
/// with the options that stand for parameters applied over them.
fn encode_params(args: &[String], codec: Codec) -> Result<ParamOverrides, String> {
    let number = |v: String, flag: &str| {
        v.parse::<u64>()
            .map_err(|_| format!("bad value for {flag}: {v}"))
    };
    let mut params = params::overrides_from_args_with_scalability(args, parse_scalability(args)?)?;
    // `--intra-refresh N` is `--param refresh.period=N`.
    if let Some(period) = params::flag_value(args, "--intra-refresh") {
        params.set("refresh.period", number(period, "--intra-refresh")? as f32);
    }
    // `--ltr-interval N` is `--param ltr.interval=N`.
    if let Some(interval) = params::flag_value(args, "--ltr-interval") {
        params.set("ltr.interval", number(interval, "--ltr-interval")? as f32);
    }
    if let Some(FilmGrainFlag { strength }) = parse_film_grain(args)? {
        nc_av1::enable_film_grain(codec, strength, &mut params)
            .map_err(|e| format!("--film-grain: {e}"))?;
    }
    Ok(params)
}

/// Every `--resize FRAME:WxH`, in frame order.
fn parse_resizes(args: &[String]) -> Result<Vec<(u32, usize, usize)>, String> {
    let mut resizes = args
        .windows(2)
        .filter(|pair| pair[0] == "--resize")
        .map(|pair| {
            let spec = pair[1].as_str();
            let bad = || format!("bad --resize {spec} (FRAME:WxH)");
            let (frame, size) = spec.split_once(':').ok_or_else(bad)?;
            let (width, height) = size.split_once('x').ok_or_else(bad)?;
            Ok((
                frame.parse().map_err(|_| bad())?,
                width.parse().map_err(|_| bad())?,
                height.parse().map_err(|_| bad())?,
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    resizes.sort_unstable();
    Ok(resizes)
}

/// `--timing`; without it, timestamped input is signalled as variable-rate.
fn parse_timing(text: Option<&str>, timestamped: bool) -> Result<TimingSignal, String> {
    match text {
        None if timestamped => Ok(TimingSignal::Variable),
        None | Some("nominal") => Ok(TimingSignal::Nominal),
        Some("constant") => Ok(TimingSignal::Constant),
        Some("variable") => Ok(TimingSignal::Variable),
        Some(other) => Err(format!(
            "bad --timing {other} (nominal, constant, variable)"
        )),
    }
}

/// `--film-grain [STRENGTH]`, when given.
struct FilmGrainFlag {
    /// `None`: the default strength.
    strength: Option<f32>,
}

/// `--film-grain [STRENGTH]`; `None` without the flag.
fn parse_film_grain(args: &[String]) -> Result<Option<FilmGrainFlag>, String> {
    let Some(index) = args.iter().position(|a| a == "--film-grain") else {
        return Ok(None);
    };
    match args.get(index + 1).filter(|next| !next.starts_with("--")) {
        None => Ok(Some(FilmGrainFlag { strength: None })),
        Some(text) => match text.parse::<f32>() {
            Ok(strength) if strength.is_finite() && strength > 0.0 => Ok(Some(FilmGrainFlag {
                strength: Some(strength),
            })),
            _ => Err(format!(
                "bad --film-grain strength {text} (a positive number; 1 matches the measured source noise)"
            )),
        },
    }
}

/// `30`, `29.97` (NTSC rates map to /1001) or an explicit `num/den`.
/// A key frame every `DEFAULT_INTRA_PERIOD_SECONDS` of video, at least 1 frame.
fn default_intra_period(rate: FrameRate) -> u32 {
    let frames =
        u64::from(rate.num) * u64::from(DEFAULT_INTRA_PERIOD_SECONDS) / u64::from(rate.den.max(1));
    u32::try_from(frames).unwrap_or(u32::MAX).max(1)
}

fn parse_fps(text: &str) -> Result<FrameRate, String> {
    let bad = || format!("bad --fps {text}");
    if let Some((num, den)) = text.split_once('/') {
        return Ok(FrameRate {
            num: num.parse().map_err(|_| bad())?,
            den: den.parse().map_err(|_| bad())?,
        });
    }
    let fps: f64 = text.parse().map_err(|_| bad())?;
    let ntsc = fps * 1001.0 / 1000.0;
    if (ntsc - ntsc.round()).abs() < 1e-3 && (fps - fps.round()).abs() > 1e-3 {
        return Ok(FrameRate {
            num: (ntsc.round() as u32) * 1000,
            den: 1001,
        });
    }
    Ok(FrameRate {
        num: fps.round() as u32,
        den: 1,
    })
}

/// The encoder configuration `args` ask for.
fn encoder_config(args: &EncodeArgs) -> EncoderConfig {
    EncoderConfig {
        codec: args.codec,
        width: args.width,
        height: args.height,
        frame_rate: args.frame_rate,
        quantizer: args.quantizer,
        intra_period: args.intra_period,
        block_log2: args.block_log2,
        key_qp_offset: args.key_qp_offset,
        temporal_layers: args.temporal_layers,
        signal: args.signal,
        params: args.params.clone(),
    }
}

fn encode(args: &EncodeArgs) -> Result<(), String> {
    let config = encoder_config(args);
    let path = args.output.as_ref().ok_or("missing --output")?;
    let output = stdio::create(path)?;
    match args.codec {
        Codec::Av1 | Codec::Vp9 | Codec::Av2 => {
            let dims =
                |d: usize| u16::try_from(d).map_err(|_| "dimension too large for IVF".to_string());
            // Timestamped input keeps its exact times: a microsecond time
            // base. Otherwise one tick per nominal frame.
            let (base_den, base_num) = if args.timestamps.is_some() {
                (MICROS_PER_SECOND as u32, 1)
            } else {
                (args.frame_rate.num, args.frame_rate.den)
            };
            let ivf = IvfWriter::new(
                output,
                args.codec,
                dims(args.width)?,
                dims(args.height)?,
                base_den,
                base_num,
                args.frames,
            )
            .map_err(|e| e.to_string())?;
            if args.codec == Codec::Vp9 {
                run::<Vp9Backend, _>(args, config, ivf)
            } else if args.codec == Codec::Av2 {
                run::<Av2Backend, _>(args, config, ivf)
            } else {
                run::<Av1Backend, _>(args, config, ivf)
            }
        }
        Codec::Hevc => run::<HevcBackend, _>(args, config, AnnexB(output)),
        Codec::Vvc => run::<nc_vvc::VvcBackend, _>(args, config, AnnexB(output)),
        Codec::H264 => run::<H264Backend, _>(args, config, AnnexB(output)),
    }
}

/// One `--rate-log` row: the frame's type (or `Dropped`), quantizers, size,
/// the controller's buffer and rate statistics, and the coded picture size
/// (smaller than the stream while adaptive resolution codes at half size).
fn write_rate_row(
    log: &mut impl Write,
    index: u32,
    frame: &EncodedFrame,
    stats: &RateControlStats,
    (width, height): (usize, usize),
) -> std::io::Result<()> {
    let kind = if frame.metadata.dropped {
        "Dropped".to_string()
    } else if frame.metadata.held {
        "Held".to_string()
    } else {
        format!("{:?}", frame.metadata.frame_type)
    };
    writeln!(
        log,
        "{index},{kind},{},{},{},{:.6},{},{},{},{},{:.0},{:.3},{},{width},{height}",
        frame.metadata.quantizer,
        stats.current_quantizer,
        frame.data.len(),
        stats.buffer_fullness,
        stats.actual_bps,
        stats.total_bits,
        stats.buffer_overflows,
        stats.target_bps,
        stats.key_loan_bits,
        stats.residual_sigma,
        stats.recoded_frames
    )
}

/// Where coded frames go: a container or a raw elementary stream.
trait FrameSink {
    /// Append a frame presented at `pts` time-base ticks.
    fn write_frame(&mut self, data: &[u8], pts: u64) -> Result<(), String>;
    fn finish(self) -> Result<(), String>;
}

impl<W: Write> FrameSink for IvfWriter<W> {
    fn write_frame(&mut self, data: &[u8], pts: u64) -> Result<(), String> {
        IvfWriter::write_frame(self, data, pts).map_err(|e| e.to_string())
    }

    fn finish(self) -> Result<(), String> {
        IvfWriter::finish(self).map(drop).map_err(|e| e.to_string())
    }
}

/// A raw Annex-B stream: access units back to back (H.264, HEVC).
struct AnnexB<W: Write>(W);

impl<W: Write> FrameSink for AnnexB<W> {
    fn write_frame(&mut self, data: &[u8], _pts: u64) -> Result<(), String> {
        self.0.write_all(data).map_err(|e| e.to_string())
    }

    fn finish(mut self) -> Result<(), String> {
        self.0.flush().map_err(|e| e.to_string())
    }
}

/// The mid-stream changes (`--rate-change`, `--resize`) still to apply.
struct PendingChanges<'a> {
    rate: std::iter::Peekable<std::slice::Iter<'a, rate_control::RateChange>>,
    resize: std::iter::Peekable<std::slice::Iter<'a, (u32, usize, usize)>>,
}

impl<'a> PendingChanges<'a> {
    fn new(args: &'a EncodeArgs) -> Self {
        Self {
            rate: args.rate_changes.iter().peekable(),
            resize: args.resizes.iter().peekable(),
        }
    }

    /// Apply the changes due before input frame `index`.
    fn apply<B: CodecBackend, R: Read>(
        &mut self,
        index: u32,
        encoder: &mut Encoder<B>,
        input: &mut FrameReader<R>,
    ) -> Result<(), String> {
        while let Some(change) = self.rate.next_if(|change| change.frame <= index) {
            encoder
                .set_rate_control(change.config)
                .map_err(|error| error.to_string())?;
        }
        while let Some(&(_, width, height)) = self.resize.next_if(|resize| resize.0 <= index) {
            encoder
                .set_resolution(width, height)
                .map_err(|error| error.to_string())?;
            input.set_size(width, height);
        }
        Ok(())
    }
}

/// The reconstruction outputs: `--recon` (every coded frame),
/// `--display-recon` (the picture shown at every input frame's time),
/// `--layer-md5` (per-layer reconstruction hashes) and `--sse-log`.
struct ReconOutputs {
    coded: Option<stdio::Output>,
    shown: Option<stdio::Output>,
    layer_md5: Option<stdio::Output>,
    sse: Option<sse_log::SseLog>,
    /// Shown pictures coded smaller than the stream, scaled to it.
    display: display::Display,
}

impl ReconOutputs {
    fn create(args: &EncodeArgs) -> Result<Self, String> {
        let create = |path: &Option<PathBuf>| path.as_deref().map(stdio::create).transpose();
        let mut layer_md5 = create(&args.layer_md5)?;
        if let Some(out) = layer_md5.as_mut() {
            writeln!(
                out,
                "tu,spatial_id,temporal_layer,frame_type,width,height,bytes,md5,droppable,op_mask"
            )
            .map_err(|e| e.to_string())?;
        }
        Ok(Self {
            coded: create(&args.recon)?,
            shown: create(&args.display_recon)?,
            layer_md5,
            sse: args
                .sse_log
                .as_ref()
                .map(|(log, reference)| sse_log::SseLog::create(reference, log))
                .transpose()?,
            display: display::Display::default(),
        })
    }

    /// Write `picture`, the reconstruction after input frame `index`, which was
    /// `coded` (or not: held or dropped) or `lost` (simulated loss, in no
    /// output). An uncoded frame shows the last coded picture again. A
    /// picture coded smaller than the `stream` size (adaptive resolution)
    /// goes to `--recon` as decoded and to `--display-recon` as a player
    /// shows it, scaled to the stream size (`display.rs`).
    fn write(
        &mut self,
        index: u32,
        picture: &Picture,
        stream: (usize, usize),
        coded: bool,
        lost: bool,
    ) -> Result<(), String> {
        if let (true, Some(out)) = (coded, self.coded.as_mut()) {
            yuv::write_frame(out, picture).map_err(|e| e.to_string())?;
        }
        if lost || (self.shown.is_none() && self.sse.is_none()) {
            return Ok(());
        }
        let shown = self.display.shown(picture, stream)?;
        if let Some(out) = self.shown.as_mut() {
            yuv::write_frame(out, shown).map_err(|e| e.to_string())?;
        }
        if let Some(sse) = self.sse.as_mut() {
            sse.write(index, shown)?;
        }
        Ok(())
    }

    fn write_layers<B: CodecBackend>(
        &mut self,
        index: u32,
        frame: &EncodedFrame,
        encoder: &Encoder<B>,
    ) -> Result<(), String> {
        let Some(out) = self.layer_md5.as_mut() else {
            return Ok(());
        };
        if frame.layers.is_empty() {
            let pic = encoder.reconstruction();
            writeln!(
                out,
                "{index},0,{},{:?},{},{},{},{},0,1",
                frame.metadata.temporal_layer,
                frame.metadata.frame_type,
                pic.width(),
                pic.height(),
                frame.data.len(),
                md5::picture(pic)
            )
            .map_err(|e| e.to_string())?;
        } else {
            for layer in &frame.layers {
                let pic = encoder
                    .layer_reconstruction(layer.spatial_id)
                    .ok_or("missing spatial layer reconstruction")?;
                writeln!(
                    out,
                    "{index},{},{},{:?},{},{},{},{},{},{}",
                    layer.spatial_id,
                    layer.temporal_layer,
                    layer.frame_type,
                    pic.width(),
                    pic.height(),
                    layer.byte_len,
                    md5::picture(pic),
                    u8::from(layer.droppable),
                    layer.operating_point_mask
                )
                .map_err(|e| e.to_string())?;
            }
        }
        Ok(())
    }

    fn finish(self) -> Result<(), String> {
        for mut out in [self.coded, self.shown, self.layer_md5]
            .into_iter()
            .flatten()
        {
            out.flush().map_err(|e| e.to_string())?;
        }
        self.sse.map_or(Ok(()), sse_log::SseLog::finish)
    }
}

/// Encode every input frame with backend `B` into `sink`.
fn run<B: CodecBackend, S: FrameSink>(
    args: &EncodeArgs,
    config: EncoderConfig,
    mut sink: S,
) -> Result<(), String> {
    let mut encoder = Encoder::<B>::new(config).map_err(|e| e.to_string())?;
    if let Some(rate_control) = args.rate_control {
        encoder
            .set_rate_control(rate_control)
            .map_err(|error| error.to_string())?;
    }
    let mut rate_log = args.rate_log.as_deref().map(stdio::create).transpose()?;
    if let Some(log) = &mut rate_log {
        writeln!(log,"frame,type,planned_qp,qp,bytes,buffer_fullness,actual_bps,total_bits,buffer_overflows,target_bps,key_loan_bits,residual_sigma,recoded_frames,width,height").map_err(|error|error.to_string())?;
    }
    let mut input = FrameReader::new(
        stdio::open_raw(&args.input)?,
        args.input_format,
        args.width,
        args.height,
    );
    let mut timestamps_out = args
        .timestamps_out
        .as_ref()
        .map(|path| {
            timestamps::Writer::new(stdio::create(path)?)
                .map_err(|e| format!("{}: {e}", path.display()))
        })
        .transpose()?;
    let mut recon_out = ReconOutputs::create(args)?;
    let mut telemetry = SpatialTelemetry::default();
    input.look_ahead_second(&mut encoder)?;
    let mut pending = PendingChanges::new(args);
    for index in 0..args.frames {
        pending.apply(index, &mut encoder, &mut input)?;
        let timestamp = frame_timestamp(args.timestamps.as_deref(), index)?;
        let Some(mut frame) = input.encode_next(&mut encoder, timestamp)? else {
            return Err(format!("input ended after {index} frames"));
        };
        let pts = match (&args.timestamps, timestamp) {
            (Some(times), Some(t)) => u64::try_from(t - times[0]).unwrap_or(0),
            _ => u64::from(index),
        };
        let uncoded = frame.metadata.dropped || frame.metadata.held;
        let lost = apply_simulated_loss(args.simulate_loss, &mut frame, &mut encoder, uncoded)?;
        let coded = !uncoded && !lost;
        if coded {
            sink.write_frame(&frame.data, pts)?;
            telemetry.observe(&frame);
            recon_out.write_layers(index, &frame, &encoder)?;
        }
        if let (true, Some(out)) = (coded, timestamps_out.as_mut()) {
            out.write(frame.metadata.timestamp_us)
                .map_err(|e| e.to_string())?;
        }
        let shown_pic = frame
            .layers
            .last()
            .and_then(|l| encoder.layer_reconstruction(l.spatial_id))
            .unwrap_or_else(|| encoder.reconstruction());
        if let Some(log) = &mut rate_log {
            let stats = encoder
                .rate_control_stats()
                .ok_or("rate statistics unavailable")?;
            let size = (shown_pic.width(), shown_pic.height());
            write_rate_row(log, index, &frame, &stats, size).map_err(|error| error.to_string())?;
        }
        let stream = (encoder.config().width, encoder.config().height);
        recon_out.write(index, shown_pic, stream, coded, lost)?;
    }
    if let Some(log) = &mut rate_log {
        log.flush().map_err(|error| error.to_string())?;
    }
    if let Some(path) = args.spatial_stats.as_deref() {
        telemetry.write(path, &encoder)?;
    }
    recon_out.finish()?;
    if let Some(out) = timestamps_out {
        out.finish().map_err(|e| e.to_string())?;
    }
    sink.finish()
}

fn apply_simulated_loss<B: CodecBackend>(
    simulate_loss: Option<SimulatedLoss>,
    frame: &mut EncodedFrame,
    encoder: &mut Encoder<B>,
    uncoded: bool,
) -> Result<bool, String> {
    let Some(sim) = simulate_loss else {
        return Ok(false);
    };
    if uncoded {
        return Ok(false);
    }
    let tu = frame.metadata.index;
    if sim.frame != tu {
        encoder.acknowledge(tu);
        return Ok(false);
    }
    if let Some(lost_sid) = sim.spatial_id
        && lost_sid >= 1
        && !frame.layers.is_empty()
    {
        for layer in &frame.layers {
            if layer.spatial_id < lost_sid {
                encoder.acknowledge_layer(tu, layer.spatial_id);
            }
        }
        encoder.report_layer_loss(tu, lost_sid);
        if frame.layers.iter().any(|l| l.spatial_id >= lost_sid) {
            frame.data = match encoder.config().codec {
                Codec::Av1 => {
                    nc_av1::filter_temporal_unit_layers(&frame.data, |_, sid| sid < lost_sid)
                        .map_err(|e| e.to_string())?
                }
                Codec::Vp9 if lost_sid == 1 => {
                    nc_vp9::strip_superframe_to_base(&frame.data).map_err(|e| e.to_string())?
                }
                codec => return Err(format!("{codec:?} does not support spatial layer loss")),
            };
            frame.layers.retain(|l| l.spatial_id < lost_sid);
            if frame.layers.len() == 1 {
                frame.layers[0].byte_len = frame.data.len();
            } else if let Some(last) = frame.layers.last_mut() {
                last.byte_len = frame.data.len().saturating_sub(last.byte_offset);
            }
        }
        return Ok(false);
    }
    encoder.report_loss(tu);
    Ok(true)
}

fn frame_timestamp(times: Option<&[i64]>, index: u32) -> Result<Option<i64>, String> {
    match times {
        Some(list) => list.get(index as usize).copied().map(Some).ok_or_else(|| {
            format!(
                "--timestamps lists {} frames, fewer than --frames",
                list.len()
            )
        }),
        None => Ok(None),
    }
}
