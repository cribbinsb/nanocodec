//! One-frame start-up lookahead: the camera motion between the first two
//! source frames, measured before the first is coded, so rate control can
//! size the stream's first key for the scene (`rate.first_key_lookahead`).
//!
//! The caller hands the second frame over early ([`crate::Encoder::look_ahead`])
//! and still encodes it in turn; the cost is one frame of start-up latency
//! in the caller and one extra source analysis, once per stream.

use nc_analysis::Analyzer;
use nc_common::Picture;
use nc_dsp::Dsp;

use crate::config::EncoderConfig;
use crate::params::AnalysisParams;

/// Peak camera displacement (luma samples, `GlobalMotion::peak_displacement`)
/// from `first` to `next`, measured by a throwaway analyzer so the
/// encoder's own analysis state is untouched. `None` when the picture is too
/// small for the analysis pyramid. `(width, height)` is the coded picture
/// size, which `first` and `next` have.
pub(crate) fn startup_camera_motion(
    config: &EncoderConfig,
    (width, height): (usize, usize),
    first: &Picture,
    next: &Picture,
) -> Option<f64> {
    startup_scene_profile(config, (width, height), first, next).map(|(peak, _)| peak)
}

/// Peak camera displacement (`GlobalMotion::peak_displacement`, luma samples)
/// and stationary block fraction (`PreAnalysis::static_fraction`, `[0.0, 1.0]`)
/// between `first` and `next` from a single throwaway analysis pass.
pub(crate) fn startup_scene_profile(
    config: &EncoderConfig,
    (width, height): (usize, usize),
    first: &Picture,
    next: &Picture,
) -> Option<(f64, f32)> {
    let mut analyzer = Analyzer::new(width, height, Dsp::new()).ok()?;
    analyzer.set_search_config(AnalysisParams::from_overrides(&config.params).search_config());
    analyzer.analyze(first);
    let analysis = analyzer.analyze(next);
    let motion = analysis.motion.as_ref()?;
    Some((
        f64::from(motion.global.peak_displacement(width, height)),
        analysis.pre.static_fraction,
    ))
}
