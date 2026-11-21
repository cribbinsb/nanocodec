//! The built-in presets: one codec-independent speed ladder, `veryfast`,
//! `faster`, `fast`, `medium`, `slow`, `slower` and `veryslow`, with x264's
//! familiar names, fastest first. The rungs of a codec are points on its own speed /
//! quality front, roughly equally spaced in log(speed) (owner, 2026-11-21;
//! docs/handoff/presets.md "The ladder"); a codec offers the rungs its front
//! supports. `medium` is each codec's Default (its backend `DEFAULT_PARAMS`
//! over the engine defaults, no overrides) and what an encode gets without a
//! preset; every other rung is a parameter file in the repository's
//! `presets/<codec>/` directory, built in here. A name is a position on that
//! codec's ladder, always in this order; a codec may lack some names (owner,
//! 2026-11-21: the names can change later). `slow`, `slower` and `veryslow`
//! are the quality-first rungs: more compression for more CPU, still low
//! delay. The slowest rung a codec has starts from a shared engine bundle
//! (`presets/slowest.toml`: the second reference on every frame) applied
//! before the codec's file, which adds the levers that pay for that codec; a
//! codec offers a rung by having its file. (Renames on 2026-11-21: `slow`
//! became `slower` in every codec, and where a codec then gained a rung between
//! `slow` and `slower`, its `slower` became `veryslow`.)
//!
//! The names are relative to our own Default for each codec: `veryfast` here is
//! not as fast as x264's `veryfast`, nor `medium` as slow as x264's `medium`.
//! The results pages give absolute speeds.
//!
//! Spatial scalability is a mode, not a speed: a codec may tune a rung for two
//! or three spatial layers (`presets/<codec>/l2-<rung>.toml`,
//! `l3-<rung>.toml`), which an encode with that many layers (`--scalability
//! L2T1`, ...) takes in place of the single-layer file. A rung without a
//! layered file codes its layers with the single-layer file.
//!
//! Every front end (CLI `--preset`, nc-web `BrowserEncoder`) asks this module.

use std::fmt;

use crate::config::Codec;
use crate::params::ParamOverrides;

/// A rung of the speed ladder, fastest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Speed {
    /// The fastest rung.
    Veryfast,
    /// Between `veryfast` and `fast`.
    Faster,
    /// The first rung below the Default's CPU.
    Fast,
    /// The codec's Default: no overrides.
    Medium,
    /// The first rung above the Default's CPU: more compression for more CPU.
    Slow,
    /// Between `slow` and `veryslow` (or a codec's slowest rung where it has no
    /// `veryslow`).
    Slower,
    /// The slowest rung: the most compression for the most CPU, still low delay.
    Veryslow,
}

impl Speed {
    /// Every rung, fastest first.
    pub const LADDER: [Self; 7] = [
        Self::Veryfast,
        Self::Faster,
        Self::Fast,
        Self::Medium,
        Self::Slow,
        Self::Slower,
        Self::Veryslow,
    ];

    /// The rung's name (`--preset NAME`).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Veryfast => "veryfast",
            Self::Faster => "faster",
            Self::Fast => "fast",
            Self::Medium => "medium",
            Self::Slow => "slow",
            Self::Slower => "slower",
            Self::Veryslow => "veryslow",
        }
    }

    /// The rung of a name, or `None` for any other name.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        Self::LADDER.into_iter().find(|speed| speed.name() == name)
    }
}

/// A built-in parameter file: the codec, spatial layer count and rung it
/// serves, and its text.
struct PresetFile {
    codec: Codec,
    spatial_layers: u8,
    speed: Speed,
    text: &'static str,
}

macro_rules! preset_file {
    ($codec:ident, $layers:literal, $speed:ident, $text:expr) => {
        PresetFile {
            codec: Codec::$codec,
            spatial_layers: $layers,
            speed: Speed::$speed,
            text: $text,
        }
    };
}

/// Every built-in parameter file (literal `include_str!` paths, which
/// `tools/dev/land_scope.py` reads to find the crates a preset reaches).
/// Single-layer `medium` has none: it is the
/// codec's Default.
#[rustfmt::skip] // one file per line reads as the table it is
const PRESET_FILES: &[PresetFile] = &[
    preset_file!(H264, 1, Slower, include_str!("../presets/h264/slower.toml")),
    preset_file!(H264, 1, Slow, include_str!("../presets/h264/slow.toml")),
    preset_file!(H264, 1, Veryfast, include_str!("../presets/h264/veryfast.toml")),
    preset_file!(H264, 1, Faster, include_str!("../presets/h264/faster.toml")),
    preset_file!(H264, 1, Fast, include_str!("../presets/h264/fast.toml")),
    preset_file!(Hevc, 1, Veryslow, include_str!("../presets/hevc/veryslow.toml")),
    preset_file!(Hevc, 1, Slower, include_str!("../presets/hevc/slower.toml")),
    preset_file!(Hevc, 1, Slow, include_str!("../presets/hevc/slow.toml")),
    preset_file!(Hevc, 1, Faster, include_str!("../presets/hevc/faster.toml")),
    preset_file!(Hevc, 1, Fast, include_str!("../presets/hevc/fast.toml")),
    preset_file!(Vvc, 1, Veryslow, include_str!("../presets/vvc/veryslow.toml")),
    preset_file!(Vvc, 1, Slower, include_str!("../presets/vvc/slower.toml")),
    preset_file!(Vvc, 1, Slow, include_str!("../presets/vvc/slow.toml")),
    preset_file!(Vvc, 1, Faster, include_str!("../presets/vvc/faster.toml")),
    preset_file!(Vvc, 1, Fast, include_str!("../presets/vvc/fast.toml")),
    preset_file!(Av1, 1, Slower, include_str!("../presets/av1/slower.toml")),
    preset_file!(Av1, 1, Slow, include_str!("../presets/av1/slow.toml")),
    preset_file!(Av1, 1, Veryfast, include_str!("../presets/av1/veryfast.toml")),
    preset_file!(Av1, 1, Faster, include_str!("../presets/av1/faster.toml")),
    preset_file!(Av1, 1, Fast, include_str!("../presets/av1/fast.toml")),
    preset_file!(Av1, 2, Fast, include_str!("../presets/av1/l2-fast.toml")),
    preset_file!(Av1, 2, Medium, include_str!("../presets/av1/l2-medium.toml")),
    preset_file!(Av1, 3, Fast, include_str!("../presets/av1/l3-fast.toml")),
    preset_file!(Av1, 3, Medium, include_str!("../presets/av1/l3-medium.toml")),
    preset_file!(Vp9, 1, Veryfast, include_str!("../presets/vp9/veryfast.toml")),
    preset_file!(Vp9, 1, Slower, include_str!("../presets/vp9/slower.toml")),
    preset_file!(Vp9, 1, Slow, include_str!("../presets/vp9/slow.toml")),
    preset_file!(Vp9, 1, Faster, include_str!("../presets/vp9/faster.toml")),
    preset_file!(Vp9, 1, Fast, include_str!("../presets/vp9/fast.toml")),
    preset_file!(Vp9, 2, Fast, include_str!("../presets/vp9/l2-fast.toml")),
    preset_file!(Vp9, 2, Medium, include_str!("../presets/vp9/l2-medium.toml")),
    preset_file!(Vp9, 3, Fast, include_str!("../presets/vp9/l3-fast.toml")),
    preset_file!(Vp9, 3, Medium, include_str!("../presets/vp9/l3-medium.toml")),
    preset_file!(Av2, 1, Slower, include_str!("../presets/av2/slower.toml")),
    preset_file!(Av2, 1, Slow, include_str!("../presets/av2/slow.toml")),
    preset_file!(Av2, 1, Faster, include_str!("../presets/av2/faster.toml")),
    preset_file!(Av2, 1, Fast, include_str!("../presets/av2/fast.toml")),
];

/// The parameters every codec's slowest rung starts from, applied before its file:
/// the levers that are the shared engine's, not syntax.
const SLOWEST_SHARED: &str = include_str!("../presets/slowest.toml");

/// Why a preset cannot be applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PresetError {
    /// The name is not a rung of the ladder.
    UnknownName(String),
    /// The codec has no file at that rung.
    NotOffered {
        /// The codec asked for.
        codec: Codec,
        /// The rung asked for.
        speed: Speed,
    },
    /// A built-in file has a line that is not `name = value`.
    BadLine(String),
}

impl fmt::Display for PresetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownName(name) => write!(
                f,
                "unknown preset '{name}' (presets: {})",
                names(&Speed::LADDER)
            ),
            Self::NotOffered { codec, speed } => write!(
                f,
                "{} has no '{}' preset (its presets: {})",
                codec_name(*codec),
                speed.name(),
                names(&speeds(*codec))
            ),
            Self::BadLine(line) => write!(f, "built-in preset line '{line}' is not name = value"),
        }
    }
}

impl std::error::Error for PresetError {}

/// The codec's command-line name.
const fn codec_name(codec: Codec) -> &'static str {
    match codec {
        Codec::H264 => "h264",
        Codec::Hevc => "hevc",
        Codec::Vvc => "vvc",
        Codec::Av1 => "av1",
        Codec::Vp9 => "vp9",
        Codec::Av2 => "av2",
    }
}

fn names(speeds: &[Speed]) -> String {
    speeds
        .iter()
        .map(|speed| speed.name())
        .collect::<Vec<_>>()
        .join(", ")
}

/// The rungs `codec` offers single-layer, fastest first (always `medium`).
#[must_use]
pub fn speeds(codec: Codec) -> Vec<Speed> {
    Speed::LADDER
        .into_iter()
        .filter(|&speed| speed == Speed::Medium || file(codec, 1, speed).is_some())
        .collect()
}

fn file(codec: Codec, spatial_layers: u8, speed: Speed) -> Option<&'static PresetFile> {
    PRESET_FILES.iter().find(|file| {
        file.codec == codec && file.spatial_layers == spatial_layers && file.speed == speed
    })
}

/// The parameters of the rung named `name` for `codec` with `spatial_layers`
/// spatial layers: the layered file where the codec has one for that layer
/// count, else the single-layer file (none for `medium`).
///
/// # Errors
/// An unknown name, or a rung the codec does not offer.
pub fn preset_params(
    codec: Codec,
    spatial_layers: u8,
    name: &str,
) -> Result<ParamOverrides, PresetError> {
    let speed = Speed::parse(name).ok_or_else(|| PresetError::UnknownName(name.to_owned()))?;
    let chosen = file(codec, spatial_layers, speed).or_else(|| file(codec, 1, speed));
    let text = match chosen {
        Some(file) => file.text,
        None if speed == Speed::Medium => "",
        None => return Err(PresetError::NotOffered { codec, speed }),
    };
    let slowest = speeds(codec).last() == Some(&speed) && speed != Speed::Medium;
    let shared = if slowest { SLOWEST_SHARED } else { "" };
    Ok(parse_file(shared)?.overlaid_with(&parse_file(text)?))
}

/// A parameter file: one `name = value` per line, `#` comments, blank lines.
fn parse_file(text: &str) -> Result<ParamOverrides, PresetError> {
    let mut params = ParamOverrides::default();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let bad = || PresetError::BadLine(line.to_owned());
        let (name, value) = line.split_once('=').ok_or_else(bad)?;
        params.set(name.trim(), value.trim().parse().map_err(|_| bad())?);
    }
    Ok(params)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_built_in_file_parses_and_is_not_empty() {
        for file in PRESET_FILES {
            let params = parse_file(file.text).unwrap();
            assert!(!params.is_empty(), "{:?} {}", file.codec, file.speed.name());
        }
    }

    #[test]
    fn the_slowest_rung_is_the_shared_bundle_under_the_codec_file() {
        let shared = parse_file(SLOWEST_SHARED).unwrap();
        let hevc = preset_params(Codec::Hevc, 1, "veryslow").unwrap();
        assert_eq!(hevc.get("refs.second"), Some(2.0));
        assert_eq!(hevc.get("decision.winner_walk_steps"), Some(3.0));
        assert_eq!(hevc.overlaid_with(&shared), hevc);
    }

    #[test]
    fn medium_is_the_default_and_layers_pick_their_file() {
        assert!(preset_params(Codec::Hevc, 1, "medium").unwrap().is_empty());
        let layered = preset_params(Codec::Av1, 2, "medium").unwrap();
        assert_eq!(layered.get("spatial.layers"), Some(2.0));
        // A rung without a layered file codes its layers with the single-layer file.
        assert_eq!(
            preset_params(Codec::Av1, 2, "faster").unwrap(),
            preset_params(Codec::Av1, 1, "faster").unwrap()
        );
    }

    #[test]
    fn old_and_missing_names_are_errors_listing_the_ladder() {
        let unknown = preset_params(Codec::Av1, 1, "av1-fast").unwrap_err();
        assert!(
            unknown
                .to_string()
                .contains("veryfast, faster, fast, medium, slow, slower, veryslow")
        );
        assert!(preset_params(Codec::H264, 1, "default").is_err());
        let missing = preset_params(Codec::Vvc, 1, "veryfast").unwrap_err();
        assert_eq!(
            missing.to_string(),
            "vvc has no 'veryfast' preset (its presets: faster, fast, medium, slow, slower, veryslow)"
        );
        assert_eq!(
            speeds(Codec::Hevc),
            [
                Speed::Faster,
                Speed::Fast,
                Speed::Medium,
                Speed::Slow,
                Speed::Slower,
                Speed::Veryslow
            ]
        );
    }
}
