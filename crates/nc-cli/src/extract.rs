//! Temporal sub-stream extraction for trick play and seeking: keep the
//! access units of temporal layers up to `--max-layer` of a hierarchical-P
//! stream and drop the rest, without re-encoding (`docs/ARCHITECTURE.md` §6.1).
//!
//! The kept stream is itself conforming: H.264 drops by `nal_ref_idc`
//! (`nc_h264::access_unit_layer`), HEVC by `TemporalId` (sub-bitstream
//! extraction, HEVC §10), AV1 by the frame OBU's `temporal_id` (operating
//! point selection, AV1 §7.18; the sequence header keeps listing every
//! operating point, which stays valid since the dropped frames refresh no
//! reference slot the kept ones use), AV2 likewise by the frame OBU's
//! `temporal_id` (`nc_av2::temporal_unit_layer`). VVC uses NAL `TemporalId`
//! (`nc_vvc::access_unit_layer`), retaining each kept picture's prefix ALF APS.

use std::io::Cursor;
use std::path::PathBuf;

use nc_encoder::{Codec, MAX_TEMPORAL_LAYERS};

use crate::framing::{IVF_FRAME_COUNT_OFFSET, ivf_fourcc, read_annex_b, read_ivf};
use crate::stdio;

#[derive(Debug)]
struct ExtractArgs {
    codec: Codec,
    temporal_layers: u8,
    input: PathBuf,
    action: Action,
}

/// What `nanocodec extract` does with the stream.
#[derive(Debug)]
enum Action {
    /// Write the access units of layers up to `max_layer` to `output`.
    Keep { max_layer: u8, output: PathBuf },
    /// Strip a VP9 spatial stream to `operating_point` (optionally guided by a
    /// `--layer-md5` sidecar CSV for temporal operating points) and write IVF
    /// to `output`.
    Vp9Spatial {
        operating_point: u8,
        layer_md5: Option<PathBuf>,
        output: PathBuf,
    },
    /// Print each access unit's temporal layer, one per line (`--list`).
    List,
}

fn parse(args: &[String]) -> Result<ExtractArgs, String> {
    let get = |flag: &str| -> Result<String, String> {
        let i = args
            .iter()
            .position(|a| a == flag)
            .ok_or_else(|| format!("missing {flag}"))?;
        args.get(i + 1)
            .cloned()
            .ok_or_else(|| format!("{flag} needs a value"))
    };
    let layer = |flag: &str| -> Result<u8, String> {
        let value = get(flag)?;
        value
            .parse()
            .map_err(|_| format!("bad value for {flag}: {value}"))
    };
    let codec = match get("--codec")?.as_str() {
        "av1" => Codec::Av1,
        "av2" => Codec::Av2,
        "vp9" => Codec::Vp9,
        "h264" => Codec::H264,
        "hevc" => Codec::Hevc,
        "vvc" => Codec::Vvc,
        other => return Err(format!("unsupported extract codec '{other}'")),
    };
    let has_base_only = args.iter().any(|a| a == "--base-only");
    let has_op = args.iter().any(|a| a == "--operating-point");
    if codec == Codec::Vp9 && (has_base_only || has_op) {
        let operating_point = if has_op {
            crate::decode::parse_operating_point(&get("--operating-point")?, "")?
        } else {
            1
        };
        let layer_md5 = args
            .iter()
            .position(|a| a == "--layer-md5")
            .map(|i| {
                args.get(i + 1)
                    .cloned()
                    .map(PathBuf::from)
                    .ok_or_else(|| "--layer-md5 needs a value".to_string())
            })
            .transpose()?;
        return Ok(ExtractArgs {
            codec,
            temporal_layers: 1,
            input: get("--input")?.into(),
            action: Action::Vp9Spatial {
                operating_point,
                layer_md5,
                output: get("--output")?.into(),
            },
        });
    }
    if codec == Codec::Vp9 {
        return Err(NO_VP9_LAYERS.into());
    }
    let temporal_layers = layer("--temporal-layers")?;
    if !(1..=MAX_TEMPORAL_LAYERS).contains(&temporal_layers) {
        return Err("need 1 <= --temporal-layers <= 3".into());
    }
    let action = if args.iter().any(|a| a == "--list") {
        Action::List
    } else {
        let max_layer = layer("--max-layer")?;
        if max_layer >= temporal_layers {
            return Err("--max-layer must be below --temporal-layers".into());
        }
        Action::Keep {
            max_layer,
            output: get("--output")?.into(),
        }
    };
    Ok(ExtractArgs {
        codec,
        temporal_layers,
        input: get("--input")?.into(),
        action,
    })
}

/// VP9 temporal layer extraction without `--base-only` / `--operating-point`.
const NO_VP9_LAYERS: &str = "VP9 temporal layer extraction is not supported";

/// Run `nanocodec extract`.
pub(super) fn run(args: &[String]) -> Result<(), String> {
    let args = parse(args)?;
    let input = stdio::read_all(&args.input)?;
    match &args.action {
        Action::Keep { max_layer, output } => {
            let kept = extract(&args, *max_layer, &input)?;
            stdio::write_all(output, &kept)
        }
        Action::Vp9Spatial {
            operating_point,
            layer_md5,
            output,
        } => {
            let kept = extract_vp9_spatial(*operating_point, layer_md5.as_deref(), &input)?;
            stdio::write_all(output, &kept)
        }
        Action::List => {
            for layer in layers(&args, &input)? {
                println!("{layer}");
            }
            Ok(())
        }
    }
}

/// The temporal layer of one access unit (or IVF frame).
fn layer_of(codec: Codec, unit: &[u8], temporal_layers: u8) -> Result<u8, String> {
    match codec {
        Codec::H264 => nc_h264::access_unit_layer(unit, temporal_layers),
        Codec::Hevc => nc_hevc::access_unit_layer(unit),
        Codec::Vvc => nc_vvc::access_unit_layer(unit),
        Codec::Av1 => nc_av1::temporal_unit_layer(unit),
        Codec::Av2 => nc_av2::temporal_unit_layer(unit),
        Codec::Vp9 => return Err(NO_VP9_LAYERS.into()),
    }
    .map_err(|e| e.to_string())
}

/// The temporal layer of every access unit (IVF frame), in order.
fn layers(args: &ExtractArgs, input: &[u8]) -> Result<Vec<u8>, String> {
    let mut layers = Vec::new();
    let mut add = |unit: &[u8]| -> Result<(), String> {
        layers.push(layer_of(args.codec, unit, args.temporal_layers)?);
        Ok(())
    };
    match args.codec {
        Codec::Av1 | Codec::Av2 => read_ivf(&mut Cursor::new(input), |_, frame| add(frame.packet))?,
        Codec::H264 | Codec::Hevc | Codec::Vvc => {
            read_annex_b(args.codec, &mut Cursor::new(input), add)?
        }
        Codec::Vp9 => return Err(NO_VP9_LAYERS.into()),
    };
    Ok(layers)
}

fn extract(args: &ExtractArgs, max_layer: u8, input: &[u8]) -> Result<Vec<u8>, String> {
    let mut output = Vec::new();
    let keep = |unit: &[u8]| -> Result<bool, String> {
        Ok(layer_of(args.codec, unit, args.temporal_layers)? <= max_layer)
    };
    match args.codec {
        Codec::Vp9 => return Err(NO_VP9_LAYERS.into()),
        Codec::Av1 | Codec::Av2 => {
            let fourcc = ivf_fourcc(args.codec).expect("AV1 and AV2 streams are IVF");
            let mut kept = 0u32;
            read_ivf(&mut Cursor::new(input), |header, frame| {
                if header.fourcc() != fourcc {
                    return Err("extraction requires IVF of the stream's codec".into());
                }
                if output.is_empty() {
                    output.extend_from_slice(&header.bytes);
                }
                if keep(frame.packet)? {
                    output.extend_from_slice(frame.header);
                    output.extend_from_slice(frame.packet);
                    kept += 1;
                }
                Ok(())
            })?;
            let count = IVF_FRAME_COUNT_OFFSET..IVF_FRAME_COUNT_OFFSET + 4;
            output[count].copy_from_slice(&kept.to_le_bytes());
        }
        Codec::H264 | Codec::Hevc | Codec::Vvc => {
            read_annex_b(args.codec, &mut Cursor::new(input), |unit| {
                if keep(unit)? {
                    output.extend_from_slice(unit);
                }
                Ok(())
            })?;
        }
    }
    Ok(output)
}

#[derive(Clone, Copy)]
struct Vp9SidecarLayer {
    spatial_id: u8,
    width: usize,
    height: usize,
    op_mask: u32,
}

fn parse_vp9_sidecar(path: &std::path::Path) -> Result<Vec<Vec<Vp9SidecarLayer>>, String> {
    let text = String::from_utf8(stdio::read_all(path)?).map_err(|e| e.to_string())?;
    let mut tus: std::collections::BTreeMap<u32, Vec<Vp9SidecarLayer>> =
        std::collections::BTreeMap::new();
    for line in text.lines().skip(1) {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() < 8 {
            return Err(format!("malformed --layer-md5 row: {line}"));
        }
        let tu: u32 = cols[0].parse().map_err(|e| format!("bad tu: {e}"))?;
        let spatial_id: u8 = cols[1]
            .parse()
            .map_err(|e| format!("bad spatial_id: {e}"))?;
        let width: usize = cols[4].parse().map_err(|e| format!("bad width: {e}"))?;
        let height: usize = cols[5].parse().map_err(|e| format!("bad height: {e}"))?;
        let op_mask: u32 = cols
            .get(9)
            .and_then(|s| s.parse().ok())
            .unwrap_or(if spatial_id == 0 { 0b11 } else { 0b01 });
        tus.entry(tu).or_default().push(Vp9SidecarLayer {
            spatial_id,
            width,
            height,
            op_mask,
        });
    }
    Ok(tus.into_values().collect())
}

fn extract_vp9_spatial(
    operating_point: u8,
    layer_md5: Option<&std::path::Path>,
    input: &[u8],
) -> Result<Vec<u8>, String> {
    let sidecar = layer_md5.map(parse_vp9_sidecar).transpose()?;
    let fourcc = ivf_fourcc(Codec::Vp9).expect("VP9 streams are IVF");
    let mut output = Vec::new();
    let mut kept = 0u32;
    let mut tu_idx = 0usize;
    let mut base_width: Option<usize> = None;
    let mut first_size: Option<(usize, usize)> = None;
    read_ivf(&mut Cursor::new(input), |header, frame| {
        if header.fourcc() != fourcc {
            return Err("extraction requires a VP9 IVF stream".into());
        }
        if output.is_empty() {
            output.extend_from_slice(&header.bytes);
        }
        let idx = tu_idx;
        tu_idx += 1;
        let strip_to_base = if let Some(tus) = &sidecar {
            let layers = tus
                .get(idx)
                .ok_or_else(|| format!("--layer-md5 has fewer TUs than IVF ({idx})"))?;
            let bit = 1u32.checked_shl(u32::from(operating_point)).unwrap_or(0);
            let included: Vec<&Vp9SidecarLayer> =
                layers.iter().filter(|l| (l.op_mask & bit) != 0).collect();
            if included.is_empty() {
                return Ok(());
            }
            if let Some(s0) = layers.iter().find(|l| l.spatial_id == 0) {
                base_width = Some(s0.width);
            }
            let top = included.last().expect("non-empty");
            first_size.get_or_insert((top.width, top.height));
            top.spatial_id == 0
        } else {
            operating_point > 0
        };
        let packet = if strip_to_base {
            let stripped =
                nc_vp9::strip_superframe_to_base(frame.packet).map_err(|e| e.to_string())?;
            if let Ok(info) = nc_vp9::parse_uncompressed_header_info(&stripped, base_width)
                && let Some((w, h)) = info.coded_size
            {
                base_width = Some(w);
                first_size.get_or_insert((w, h));
            }
            stripped
        } else {
            frame.packet.to_vec()
        };
        let mut frame_hdr = *frame.header;
        let len = u32::try_from(packet.len()).map_err(|_| "VP9 frame too large".to_string())?;
        frame_hdr[..4].copy_from_slice(&len.to_le_bytes());
        output.extend_from_slice(&frame_hdr);
        output.extend_from_slice(&packet);
        kept += 1;
        Ok(())
    })?;
    if output.len() >= 32 {
        if let Some((w, h)) = first_size
            && let (Ok(w16), Ok(h16)) = (u16::try_from(w), u16::try_from(h))
        {
            output[12..14].copy_from_slice(&w16.to_le_bytes());
            output[14..16].copy_from_slice(&h16.to_le_bytes());
        }
        let count = IVF_FRAME_COUNT_OFFSET..IVF_FRAME_COUNT_OFFSET + 4;
        output[count].copy_from_slice(&kept.to_le_bytes());
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| (*s).into()).collect()
    }

    #[test]
    fn layer_limits_are_validated() {
        let base = ["--codec", "h264", "--input", "a", "--output", "b"];
        let with = |layers: &str, max: &str| {
            let mut v = strings(&base);
            v.extend(strings(&["--temporal-layers", layers, "--max-layer", max]));
            parse(&v)
        };
        assert!(with("3", "1").is_ok());
        let mut list = strings(&base);
        list.extend(strings(&["--temporal-layers", "3", "--list"]));
        assert!(matches!(parse(&list).unwrap().action, Action::List));
        assert!(with("3", "3").is_err());
        assert!(with("4", "0").is_err());
    }

    #[test]
    fn vvc_extraction_keeps_each_selected_pictures_alf_aps() {
        let mut packets = Vec::new();
        for layer in [0, 2, 1, 2] {
            let mut packet = Vec::new();
            for kind in [17, 0] {
                // prefix APS, then TRAIL slice
                packet.extend_from_slice(&[0, 0, 0, 1, 0, (kind << 3) | (layer + 1), 0x80]);
            }
            packets.push(packet);
        }
        let args = ExtractArgs {
            codec: Codec::Vvc,
            temporal_layers: 3,
            input: PathBuf::new(),
            action: Action::List,
        };
        let stream = packets.concat();
        assert_eq!(layers(&args, &stream).unwrap(), [0, 2, 1, 2]);
        assert_eq!(extract(&args, 0, &stream).unwrap(), packets[0]);
        assert_eq!(
            extract(&args, 1, &stream).unwrap(),
            [packets[0].clone(), packets[2].clone()].concat()
        );
    }
}
