//! Frame-level loop-filter effort: which inter frames are deblocked and which
//! search their CDEF strengths.
//!
//! SVT-AV1 real time (`enc_mode_config.c` `get_dlf_level_rtc` and the CDEF
//! levels, `deblocking_filter.c`; BSD-3-Clause, ideas only) does not filter
//! every frame alike: a non-base frame whose reference coded most of its area
//! as skip is not deblocked (`ref_skip_percentage > 75`), a frame the motion
//! search predicts well gets no luma deblocking (`rc_me_distortion` floor),
//! and non-base frames reuse their reference's CDEF strengths instead of
//! searching. Our backends filter every frame with their own fixed settings.
//!
//! Here the engine decides once per inter frame, from statistics it has
//! before coding:
//!
//! - the share of the previous frame's area its backend coded with no
//!   residual ([`CodecBackend::coded_skip_share`]; in a flat chain the
//!   previous frame is the reference);
//! - the frame's mean benefit proxy: the motion search's best 16x16 SAD
//!   plus weighted 8x8 spread, per sample, in quantizer sample steps
//!   (`effort_class.rs`). SVT's thresholds are absolute SADs; against the
//!   step, one value serves every quantizer, at fixed QP and under rate
//!   control.
//!
//! Each backend maps the decision onto its own syntax
//! ([`CodecBackend::set_filter_effort`]). Key frames and base frames (bar
//! the proxy test) keep the full filters; every `filter.*` default leaves
//! all frames at [`FilterEffort::FULL`], so streams are unchanged.
//!
//! [`CodecBackend::coded_skip_share`]: crate::CodecBackend::coded_skip_share
//! [`CodecBackend::set_filter_effort`]: crate::CodecBackend::set_filter_effort

use crate::frame_role::FrameRole;
use crate::params::FilterParams;

/// How much of its CDEF work a frame does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CdefEffort {
    /// The backend's own choice (searching where its settings say so).
    Search,
    /// Reuse the strengths the backend chose last, without a search.
    Reuse,
}

/// The engine's loop-filter decision for one frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FilterEffort {
    /// Whether the frame is deblocked (where the backend's settings
    /// deblock at all).
    pub deblock: bool,
    /// Whether the frame searches its CDEF strengths or reuses the last.
    pub cdef: CdefEffort,
}

impl FilterEffort {
    /// The backend's own filters, unchanged.
    pub const FULL: Self = Self {
        deblock: true,
        cdef: CdefEffort::Search,
    };
}

/// What the decision reads about one inter frame.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FrameActivity {
    pub role: FrameRole,
    /// The frame's mean benefit proxy (steps), when it has a motion search.
    pub proxy: Option<f32>,
    /// The previous coded frame's skipped share of its area, if known.
    pub previous_skip_share: Option<f32>,
    /// Consecutive frames just before this one decided [`CdefEffort::Reuse`].
    pub cdef_reused: u32,
}

/// The loop-filter effort of an inter frame with `activity`.
pub(crate) fn decide(params: &FilterParams, activity: FrameActivity) -> FilterEffort {
    let nonbase = activity.role == FrameRole::NonBase;
    let well_predicted = |limit: f32| limit > 0.0 && activity.proxy.is_some_and(|p| p < limit);
    // What it skips: the deblocking pass of a frame whose reference was
    // mostly skip, or whose prediction is well under the step: few block
    // edges carry residual for the filter to smooth. Why safe: base frames
    // keep it on the skip-share test (later frames lean on them), and the
    // share and proxy fall as soon as content moves. Effect: measured in
    // `docs/handoff/light-pipeline.md`.
    let skipped_reference = activity
        .previous_skip_share
        .is_some_and(|share| share >= params.deblock_skip_share && params.deblock_skip_share < 1.0);
    let deblock = !((nonbase && skipped_reference) || well_predicted(params.deblock_skip_proxy));
    // What it skips: the CDEF strength search of a well-predicted non-base
    // frame, which keeps the strengths chosen last (SVT's leaf frames).
    // Why safe: the strengths are still applied; base frames search afresh,
    // and so does any frame after `filter.cdef_max_reuse` reusing ones.
    let reuse_allowed = (activity.cdef_reused as f32) < params.cdef_max_reuse.round();
    let cdef = if nonbase && reuse_allowed && well_predicted(params.cdef_reuse_proxy) {
        CdefEffort::Reuse
    } else {
        CdefEffort::Search
    };
    FilterEffort { deblock, cdef }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn activity(role: FrameRole, proxy: f32, share: f32) -> FrameActivity {
        FrameActivity {
            role,
            proxy: Some(proxy),
            previous_skip_share: Some(share),
            cdef_reused: 0,
        }
    }

    #[test]
    fn defaults_keep_every_filter() {
        let params = FilterParams::default();
        for role in [FrameRole::Base, FrameRole::NonBase] {
            assert_eq!(
                decide(&params, activity(role, 0.0, 1.0)),
                FilterEffort::FULL
            );
        }
    }

    #[test]
    fn a_mostly_skipped_reference_turns_deblocking_off_on_nonbase_frames_only() {
        let params = FilterParams {
            deblock_skip_share: 0.75,
            ..FilterParams::default()
        };
        assert!(!decide(&params, activity(FrameRole::NonBase, 1.0, 0.8)).deblock);
        assert!(decide(&params, activity(FrameRole::NonBase, 1.0, 0.7)).deblock);
        assert!(decide(&params, activity(FrameRole::Base, 1.0, 0.8)).deblock);
    }

    #[test]
    fn well_predicted_frames_skip_deblocking_and_reuse_cdef() {
        let params = FilterParams {
            deblock_skip_proxy: 0.5,
            cdef_reuse_proxy: 0.5,
            ..FilterParams::default()
        };
        let easy = decide(&params, activity(FrameRole::NonBase, 0.2, 0.0));
        assert_eq!(
            easy,
            FilterEffort {
                deblock: false,
                cdef: CdefEffort::Reuse
            }
        );
        assert_eq!(
            decide(&params, activity(FrameRole::NonBase, 0.6, 0.0)),
            FilterEffort::FULL
        );
        // Base frames still search CDEF, and so does a frame after the
        // most reusing ones in a row.
        assert_eq!(
            decide(&params, activity(FrameRole::Base, 0.2, 0.0)).cdef,
            CdefEffort::Search
        );
        let long_reuse = FrameActivity {
            cdef_reused: 7,
            ..activity(FrameRole::NonBase, 0.2, 0.0)
        };
        assert_eq!(decide(&params, long_reuse).cdef, CdefEffort::Search);
    }
}
