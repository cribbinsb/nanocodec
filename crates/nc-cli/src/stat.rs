//! `nanocodec stat` (analysis builds: `--features analysis`): decode an
//! H.264 or HEVC stream with coding-tool
//! statistics enabled ([`nc_common::stream_stats`]) and write them as JSON,
//! to show which block sizes, modes, motion, transforms, QPs and filters an
//! encoder actually used. The input is Annex-B or an IVF file of Annex-B
//! access units; `eval/nceval/streamstats.py` summarises the output.

use std::io::{Cursor, Write};
use std::path::PathBuf;

use nc_common::stream_stats::StreamStats;
use nc_encoder::Codec;

use crate::framing::{is_ivf, read_ivf_annex_b};
use crate::stdio;

#[derive(Debug)]
struct StatArgs {
    codec: Codec,
    input: PathBuf,
    json: Option<PathBuf>,
}

fn parse(args: &[String]) -> Result<StatArgs, String> {
    let get = |flag: &str| {
        let i = args.iter().position(|a| a == flag)?;
        args.get(i + 1).cloned()
    };
    let codec = match get("--codec").as_deref() {
        Some("h264") => Codec::H264,
        Some("hevc") => Codec::Hevc,
        _ => return Err("stat needs --codec h264|hevc".into()),
    };
    let input = get("--input").ok_or("missing --input")?.into();
    Ok(StatArgs {
        codec,
        input,
        json: get("--json").map(PathBuf::from),
    })
}

/// Run `nanocodec stat`.
pub(super) fn run(args: &[String]) -> Result<(), String> {
    let args = parse(args)?;
    let data = stdio::read_all(&args.input)?;
    let stats = match args.codec {
        Codec::Hevc => hevc(&data)?,
        Codec::H264 => h264(&data)?,
        _ => unreachable!("parse accepts H.264 and HEVC only"),
    };
    let json = stats.to_json();
    match &args.json {
        Some(path) => stdio::write_all(path, json.as_bytes()),
        None => std::io::stdout()
            .write_all(json.as_bytes())
            .map_err(|e| e.to_string()),
    }
}

/// Call `visit` with the stream's Annex-B data: each IVF frame, or the
/// whole input.
fn for_each_packet(
    data: &[u8],
    mut visit: impl FnMut(&[u8]) -> Result<(), String>,
) -> Result<(), String> {
    if is_ivf(data) {
        read_ivf_annex_b(&mut Cursor::new(data), |_, frame| visit(frame.packet))?;
        Ok(())
    } else {
        visit(data)
    }
}

fn hevc(data: &[u8]) -> Result<StreamStats, String> {
    let mut decoder = nc_hevc::Decoder::new();
    decoder.collect_stats(StreamStats::default());
    for_each_packet(data, |packet| {
        decoder.push(packet).map_err(|e| e.to_string())?;
        while decoder.next_output().is_some() {}
        Ok(())
    })?;
    decoder.flush().map_err(|e| e.to_string())?;
    Ok(decoder.take_stats().unwrap_or_default())
}

fn h264(data: &[u8]) -> Result<StreamStats, String> {
    let mut decoder = nc_h264::Decoder::new();
    decoder.collect_stats(StreamStats::default());
    for_each_packet(data, |packet| {
        for nal in nc_common::decode::annex_b_units(packet).map_err(|e| e.to_string())? {
            decoder.decode_nal(nal).map_err(|e| e.to_string())?;
        }
        Ok(())
    })?;
    Ok(decoder.take_stats().unwrap_or_default())
}
