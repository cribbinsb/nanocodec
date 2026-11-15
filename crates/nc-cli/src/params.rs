//! Parameter overrides from the command line: presets, parameter files and
//! single `--param name=value` settings, applied in that order (later
//! settings win).
//!
//! A parameter file is a TOML subset: one `name = value` per line, `#`
//! comments, blank lines. Presets are the codec-independent speed ladder
//! (`veryfast`, `faster`, `fast`, `medium` = the codec's Default), built in
//! by `nc_encoder::preset_params` (`presets.rs`); with `--scalability` they
//! take the codec's layered file where it has one. Tuned presets are written
//! by `eval/nceval/tune.py`.

use std::fmt::Write as _;
use std::path::Path;

use nc_av1::Av1Backend;
use nc_encoder::{
    Codec, CodecBackend, InterLayerMode, ParamOverrides, ScalabilityMode, Speed, engine_specs,
    preset_params, speeds,
};
use nc_h264::H264Backend;
use nc_hevc::HevcBackend;

/// The inter-layer prediction mode parameter a `--scalability` name sets.
const INTER_LAYER_PARAM: &str = "spatial.inter_layer";

/// Build the overrides of a `codec` encode from `args`: `--preset NAME`
/// (`medium` when absent; with `--scalability`, the rung's file for that many
/// spatial layers), then `named` (the named settings such as
/// `--still-camera`, `nc_encoder::still_camera_params`), `--params FILE`,
/// optional `--scalability`, and any number of `--param name=value`, later
/// settings winning.
pub fn overrides_from_args_with_scalability(
    args: &[String],
    codec: Codec,
    named: &ParamOverrides,
    scalability: Option<ScalabilityMode>,
) -> Result<ParamOverrides, String> {
    let preset = flag_value(args, "--preset").unwrap_or_else(|| Speed::Medium.name().to_owned());
    let spatial_layers = scalability.map_or(1, ScalabilityMode::spatial_layers);
    let mut overrides =
        preset_params(codec, spatial_layers, &preset).map_err(|e| format!("--preset: {e}"))?;
    overrides = overrides.overlaid_with(named);
    if let Some(path) = flag_value(args, "--params") {
        let text = crate::stdio::read_text(Path::new(&path))?;
        apply_file_text(&mut overrides, &text, &path)?;
    }
    if let Some(mode) = scalability {
        if mode.spatial_layers() >= 2 {
            overrides.set("spatial.layers", f32::from(mode.spatial_layers()));
            // A `_KEY` mode selects K-SVC; a full-SVC mode name undoes a
            // preset's K-SVC but keeps independent layers (`inter_layer = 0`).
            let inter_layer = overrides
                .get(INTER_LAYER_PARAM)
                .map(InterLayerMode::from_param);
            if mode.is_key_mode() {
                overrides.set(INTER_LAYER_PARAM, InterLayerMode::KeyOnly.param_value());
            } else if inter_layer == Some(InterLayerMode::KeyOnly) {
                overrides.set(INTER_LAYER_PARAM, InterLayerMode::On.param_value());
            }
        } else if overrides.contains("spatial.layers") {
            overrides.set("spatial.layers", 1.0);
        }
        if overrides.contains("layers.count") {
            overrides.set("layers.count", f32::from(mode.temporal_layers()));
        }
        if mode.temporal_layers() == 1 && overrides.contains("layers.adaptive") {
            overrides.set("layers.adaptive", 0.0);
        }
    }
    for (i, arg) in args.iter().enumerate() {
        if arg == "--param" {
            let setting = args.get(i + 1).ok_or("missing value after --param")?;
            apply_setting(&mut overrides, setting, "--param")?;
        }
    }
    Ok(overrides)
}

#[cfg(test)]
pub fn overrides_from_args(args: &[String], codec: Codec) -> Result<ParamOverrides, String> {
    overrides_from_args_with_scalability(args, codec, &ParamOverrides::default(), None)
}

/// `nanocodec params [--codec C]`: every parameter with its default, range
/// and doc. With a codec, a default is the value that codec actually encodes
/// with: its backend's `DEFAULT_PARAMS` layered over the engine defaults.
pub fn listing(codec: Option<&str>) -> Result<String, String> {
    let (codec_defaults, codec): (&[(&str, f32)], _) = match codec {
        None => (&[], None),
        Some("av1") => (Av1Backend::DEFAULT_PARAMS, Some(Codec::Av1)),
        Some("hevc") => (HevcBackend::DEFAULT_PARAMS, Some(Codec::Hevc)),
        Some("vvc" | "h266") => (nc_vvc::VvcBackend::DEFAULT_PARAMS, Some(Codec::Vvc)),
        Some("h264") => (H264Backend::DEFAULT_PARAMS, Some(Codec::H264)),
        Some("vp9") => (nc_vp9::Vp9Backend::DEFAULT_PARAMS, Some(Codec::Vp9)),
        Some("av2") => (nc_av2::Av2Backend::DEFAULT_PARAMS, Some(Codec::Av2)),
        Some(other) => return Err(format!("unknown codec '{other}'")),
    };
    let mut out = String::from("# name  default  [min, max]  what raising it trades\n");
    let backends = [
        Av1Backend::PARAM_SPECS,
        HevcBackend::PARAM_SPECS,
        H264Backend::PARAM_SPECS,
        nc_vp9::Vp9Backend::PARAM_SPECS,
        nc_av2::Av2Backend::PARAM_SPECS,
        nc_vvc::VvcBackend::PARAM_SPECS,
    ];
    let groups = engine_specs()
        .into_iter()
        .chain(backends.into_iter().flatten().copied());
    for spec in groups.flatten() {
        let default = codec_defaults
            .iter()
            .find(|(name, _)| *name == spec.name)
            .map_or(spec.default, |&(_, value)| value);
        let _ = writeln!(
            out,
            "{}  {}  [{}, {}]  {}",
            spec.name, default, spec.min, spec.max, spec.doc
        );
    }
    let _ = writeln!(
        out,
        "# nonbase.<name> = value: <name> on non-base inter frames only (engine decision.*, planner.*, analysis.*, and per-frame codec groups such as av1.*)"
    );
    let _ = writeln!(
        out,
        "# presets: {} (medium = the codec's Default; slow is reserved, not shipped yet)",
        codec
            .map_or(Speed::LADDER.to_vec(), speeds)
            .iter()
            .map(|s| s.name())
            .collect::<Vec<_>>()
            .join(", ")
    );
    Ok(out)
}

pub(crate) fn flag_value(args: &[String], flag: &str) -> Option<String> {
    let i = args.iter().position(|a| a == flag)?;
    args.get(i + 1).cloned()
}

fn apply_file_text(overrides: &mut ParamOverrides, text: &str, origin: &str) -> Result<(), String> {
    for (n, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        if !line.is_empty() {
            apply_setting(overrides, line, &format!("{origin}:{}", n + 1))?;
        }
    }
    Ok(())
}

fn apply_setting(
    overrides: &mut ParamOverrides,
    setting: &str,
    origin: &str,
) -> Result<(), String> {
    let (name, value) = setting
        .split_once('=')
        .ok_or_else(|| format!("{origin}: expected name=value, got '{setting}'"))?;
    let value: f32 = value
        .trim()
        .parse()
        .map_err(|_| format!("{origin}: bad number in '{setting}'"))?;
    overrides.set(name.trim(), value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn codec_listing_shows_backend_defaults() {
        let line = |codec| {
            listing(codec)
                .unwrap()
                .lines()
                .find(|l| l.starts_with("decision.codec_subpel_seed_lambdas  "))
                .map(str::to_owned)
                .unwrap()
        };
        assert!(line(None).starts_with("decision.codec_subpel_seed_lambdas  0.03  "));
        assert!(line(Some("vp9")).starts_with("decision.codec_subpel_seed_lambdas  0  "));
        assert!(line(Some("vvc")).starts_with("decision.codec_subpel_seed_lambdas  0.05  "));
        assert!(listing(Some("mpeg2")).is_err());
    }

    #[test]
    fn later_settings_win_and_files_parse() {
        let mut o = ParamOverrides::default();
        apply_file_text(&mut o, "# c\nplanner.lambda_scale = 3 # x\n\n", "t").unwrap();
        let o2 =
            overrides_from_args(&args(&["--param", "planner.lambda_scale=4"]), Codec::Av1).unwrap();
        assert_ne!(o, o2);
        assert!(
            overrides_from_args(&args(&["--preset", "medium"]), Codec::Av1)
                .unwrap()
                .is_empty()
        );
        // The codec-prefixed names before the ladder (2026-11-15) are gone.
        for old in ["default", "av1-fast", "nope"] {
            assert!(overrides_from_args(&args(&["--preset", old]), Codec::Av1).is_err());
        }
    }
    #[test]
    fn scalability_takes_the_layered_file_of_the_rung() {
        let layered = |mode: &str, preset: &str| {
            overrides_from_args_with_scalability(
                &args(&["--preset", preset]),
                Codec::Vp9,
                &ParamOverrides::default(),
                ScalabilityMode::parse(mode),
            )
            .unwrap()
        };
        assert_eq!(layered("L3T1", "fast").get("spatial.layers"), Some(3.0));
        assert_eq!(
            layered("L2T3", "medium").get("spatial.base_rate_share"),
            Some(0.3)
        );
        assert_eq!(
            layered("L1T1", "fast"),
            overrides_from_args(&args(&["--preset", "fast"]), Codec::Vp9).unwrap()
        );
    }
    #[test]
    fn named_settings_lie_over_the_preset_and_under_params() {
        let named = nc_encoder::still_camera_params(nc_encoder::Codec::Av1);
        let o = overrides_from_args_with_scalability(
            &args(&["--preset", "fast", "--param", "decision.noise_skip=2"]),
            Codec::Av1,
            &named,
            None,
        )
        .unwrap();
        assert_eq!(o.get("rate.skip_budget"), named.get("rate.skip_budget"));
        assert_eq!(
            o.get("planner.noise_split"),
            named.get("planner.noise_split")
        );
        assert_eq!(o.get("decision.noise_skip"), Some(2.0));
    }
    #[test]
    fn av1_fast_preserves_frozen_controls_and_accepts_chroma_override() {
        let mut expected = ParamOverrides::default();
        apply_file_text(
            &mut expected,
            include_str!("../../../presets/candidates/av1-joint-fast.toml"),
            "historical AV1 fast",
        )
        .unwrap();
        expected.set("av1.chroma_delta_q", -24.0);
        expected.set("av1.chroma_noise_scale", 2.0);
        expected.set("decision.codec_subpel_seed_lambdas", 0.1);
        expected.set("decision.codec_fullpel_probe", 1.0);
        expected.set("decision.codec_fullpel_probe_min_lambdas", 0.0094);
        // Modeled strong-luma CDEF cap (scale 3), adopted after TRAIN and follow-up VAL.
        expected.set("av1.cdef_per_superblock", 1.0);
        expected.set("av1.cdef_local_strong_luma", 1.0);
        expected.set("av1.cdef_local_strong_primary", 15.0);
        expected.set("av1.cdef_local_strong_variance_min", 0.0);
        expected.set("av1.cdef_local_noise_full_search_sigma", 1.0);
        expected.set("av1.cdef_local_strong_model_scale", 3.0);
        // Fast keeps the fixed RD guess when Default adopted the zero-residual test.
        expected.set("av1.pre_tx_skip_model", 0.0);
        // Fast keeps the engine motion search when the quality-first
        // Default dropped the margins and added the cheap refinement.
        expected.set("decision.subpel_move_bias", 1.1);
        expected.set("decision.subpel_move_bias_nonzero", 1.1);
        expected.set("decision.subpel_fractional_bias", 1.1);
        expected.set("decision.refine_cheap_vector", 0.0);
        expected.set("av1.inter_tx_split", 0.0);
        // Fast keeps deblocking off when the Default adopted it.
        expected.set("av1.loop_filter", 0.0);
        expected.set("planner.depth_removal_min_quantizer", 0.0);
        // Fast keeps the plain quantiser when the Default adopted the trellis.
        expected.set("quant.trellis", 0.0);
        // Fast keeps the plain key-frame intra set when the Default adopted
        // directions, angle deltas and filter intra.
        expected.set("av1.keyframe_directional", 0.0);
        expected.set("av1.keyframe_angle_deltas", 0.0);
        expected.set("av1.keyframe_filter_intra", 0.0);
        // Fast keeps inter-frame intra DC only when the Default adopted the
        // directional estimate and its intra search.
        expected.set("decision.intra_directional", 0.0);
        expected.set("av1.inter_intra_search", 0.0);
        // Fast codes 32x32 and 64x64 residuals with the low-frequency
        // forward (as HEVC Fast).
        expected.set("av1.partial_forward", 2.0);
        // Fast keeps its own chroma offset when the Default adopted the
        // engine chroma policy.
        expected.set("chroma.policy", 0.0);
        // Fast's P-frame deadzone (L3 2026-11-11: J -2.2 %, faster).
        expected.set("av1.inter_rounding_q7", 41.0);
        // Fast's key frames are DC-only, as SVT M12 (L3 2026-11-13).
        expected.set("av1.keyframe_intra_modes", 0.0);
        // Fast keeps both sub-sample seeds refined when the Default adopted
        // the seed gate (2026-11-14).
        expected.set("decision.seed_refine_ratio", 0.0);
        // Fast keeps budget-only debt correction when the Default adopted the
        // key-period ceiling (2026-11-14).
        expected.set("rate.period_ceiling", 0.0);
        // Fast keeps a single reference when the Default adopted the unified
        // structure's reference plan (2026-11-14).
        expected.set("structure.unified", 0.0);
        // Fast keeps its keys' budget sizing when the Default adopted the
        // key still boost (2026-11-14).
        expected.set("rate.key_still_boost", 0.0);
        // Fast keeps the static-scaled cadence when the Default adopted the
        // inherited cadence depth (2026-11-15).
        expected.set("structure.inherited_depth", 0.0);
        assert_eq!(
            overrides_from_args(&args(&["--preset", "fast"]), Codec::Av1).unwrap(),
            expected
        );
        expected.set("av1.chroma_delta_q", -12.0);
        assert_eq!(
            overrides_from_args(
                &args(&["--preset", "fast", "--param", "av1.chroma_delta_q=-12"]),
                Codec::Av1
            )
            .unwrap(),
            expected
        );
    }
}
