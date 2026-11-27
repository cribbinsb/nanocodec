//! One-frame start-up lookahead: the camera motion and the share of
//! unchanged blocks between the first two source frames, measured before the
//! first is coded, so rate control can size the stream's first key for the
//! scene (`rate.first_key_lookahead`, `rate.key_still_boost`).
//!
//! The caller hands the second frame over early ([`crate::Encoder::look_ahead`])
//! and still encodes it in turn; the cost is one frame of start-up latency
//! in the caller and one extra source analysis, once per stream.
//! `rate.startup_lookahead` 0 turns it off in every codec and rate mode: the
//! first frame is then coded as it arrives, as encoders without lookahead
//! do (benchmark parity).

use nc_analysis::Analyzer;
use nc_common::Picture;
use nc_dsp::Dsp;

use crate::config::EncoderConfig;
use crate::params::{AnalysisParams, RateParams};

/// The start-up lookahead's state: off (`rate.startup_lookahead` 0), or on,
/// with the second source frame once the caller has handed it over.
#[derive(Debug)]
pub(crate) enum StartupLookahead {
    Off,
    On(Option<Box<Picture>>),
}

impl StartupLookahead {
    /// The lookahead `params` ask for, holding no frame yet.
    pub(crate) fn new(params: &RateParams) -> Self {
        if params.startup_lookahead >= 0.5 {
            Self::On(None)
        } else {
            Self::Off
        }
    }

    /// Whether the encoder may ask for the second frame at all.
    pub(crate) fn is_on(&self) -> bool {
        matches!(self, Self::On(_))
    }

    /// Keep `next`, the second source frame (ignored when off).
    pub(crate) fn hold(&mut self, next: Picture) {
        if let Self::On(held) = self {
            *held = Some(Box::new(next));
        }
    }

    /// The second source frame, if one was handed over and not yet taken.
    pub(crate) fn take(&mut self) -> Option<Picture> {
        match self {
            Self::Off => None,
            Self::On(held) => held.take().map(|next| *next),
        }
    }
}

/// Peak camera displacement (`GlobalMotion::peak_displacement`, luma samples)
/// and stationary block fraction (`PreAnalysis::static_fraction`, `[0.0, 1.0]`)
/// between `first` and `next` from a single throwaway analysis pass, so the
/// encoder's own analysis state is untouched. `None` when the picture is too
/// small for the analysis pyramid. `(width, height)` is the coded picture
/// size, which `first` and `next` have.
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
