//! The still-camera noise handling: one named setting (CLI `--still-camera`,
//! C `nc_encoder_config.still_camera`, nc-web `BrowserEncoder` `still_camera`)
//! for a camera that stands still and sees sensor noise. It turns on, over
//! the preset:
//!
//! - the noise skip (`decision.noise_skip`, `noise_skip.rs`) with chroma held
//!   to twice the luma noise (`decision.noise_skip_chroma_ratio`): still blocks
//!   whose change is the camera's noise are skipped instead of coded;
//! - the skip-aware budget (`rate.skip_budget`, `skip_budget.rs`, strength by
//!   bits per pixel): frames the noise skip mostly skipped get a smaller budget
//!   and the unspent part is forgiven, so a still noisy scene lowers the bitrate;
//! - for AV1, the noise split (`planner.noise_split`, `partition_plan.rs`): large
//!   still leaves that fail the noise test split so their quiet quarters can skip.
//!
//! Evidence: docs/analysis/static-noise-2026-11-14.md §§7-9. It is off in every
//! preset: on benchmark content the tuner's screens found it roughly neutral,
//! at +1.4-2.6% CPU. Known risk: on AV1 the pair can run a long still stream
//! slightly over target (chirla cam 7, 600 frames: 1.048 of target against the
//! Default's 1.032); periodic-key planning is being reworked to address it.
//!
//! This module is the one place that lists the parameters; every front end
//! asks it.

use crate::config::Codec;
use crate::params::ParamOverrides;

/// The parameters every codec gets, and their values.
const SHARED: &[(&str, f32)] = &[
    ("decision.noise_skip", 4.0),
    ("decision.noise_skip_chroma_ratio", 2.0),
    ("rate.skip_budget", 1.0),
];

/// The parameters only AV1 gets: its planner keeps large leaves that the
/// noise split can open up (the split was measured on AV1 only).
const AV1_ONLY: &[(&str, f32)] = &[("planner.noise_split", 1.0)];

/// The still-camera setting's parameters for `codec`, to lay over a preset
/// (explicit parameter overrides then still win).
#[must_use]
pub fn still_camera_params(codec: Codec) -> ParamOverrides {
    let mut params = ParamOverrides::default();
    let av1 = if codec == Codec::Av1 { AV1_ONLY } else { &[] };
    for &(name, value) in SHARED.iter().chain(av1) {
        params.set(name, value);
    }
    params
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::engine_specs;

    #[test]
    fn every_parameter_is_registered_and_in_range() {
        for codec in [
            Codec::H264,
            Codec::Hevc,
            Codec::Av1,
            Codec::Vp9,
            Codec::Av2,
            Codec::Vvc,
        ] {
            still_camera_params(codec)
                .validate(&engine_specs())
                .unwrap();
        }
    }

    #[test]
    fn the_noise_split_is_av1_only() {
        assert_eq!(
            still_camera_params(Codec::Av1).get("planner.noise_split"),
            Some(1.0)
        );
        assert_eq!(
            still_camera_params(Codec::H264).get("planner.noise_split"),
            None
        );
        assert_eq!(
            still_camera_params(Codec::H264).get("rate.skip_budget"),
            Some(1.0)
        );
    }
}
