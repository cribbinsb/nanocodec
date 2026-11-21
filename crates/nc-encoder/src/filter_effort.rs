//! Frame-level loop-filter effort: which inter frames are deblocked, which
//! search their CDEF strengths, and which units of a frame skip their SAO /
//! ALF statistics by usage ([`UsageGate`]).
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FilterEffort {
    /// Whether the frame is deblocked (where the backend's settings
    /// deblock at all).
    pub deblock: bool,
    /// Whether the frame searches its CDEF strengths or reuses the last.
    pub cdef: CdefEffort,
    /// Which units skip their filter statistics by usage (`None`: every
    /// unit the backend's own settings search).
    pub usage: Option<UsageGate>,
}

impl FilterEffort {
    /// The backend's own filters, unchanged.
    pub const FULL: Self = Self {
        deblock: true,
        cdef: CdefEffort::Search,
        usage: None,
    };
}

/// Filter-search effort by usage, for one inter frame: which units (CTBs)
/// keep the filter off without gathering statistics.
///
/// What it skips: the SAO / ALF statistics of a unit coded mostly without
/// residual (`quiet_share` of its area or more), optionally only after its
/// last `off_runs` searches ended off. Why safe: such a unit mostly copies
/// reference samples that were filtered already, so the filter rarely pays
/// there; measured 2026-11-21 (each codec's medium, CBR 600, 120 frames,
/// 5 classes; `--filter-stats` "quiet"), 74-92 % of those HEVC SAO CTBs,
/// 88-98 % of VVC SAO and 64-94 % of VVC ALF ones ended off when searched,
/// while the unit's own last outcome predicted little (P(off | off last)
/// within a few points of P(off)). Every unit is still searched on every
/// `refresh`-th frame (staggered across units) and on key frames, so a
/// unit that starts to need the filter recovers. Speed and J effect: in
/// `docs/handoff/filter-effort.md`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UsageGate {
    /// Smallest residual-free share of a unit's area that may skip.
    pub quiet_share: f32,
    /// Searches that must have ended off in a row before a unit may skip.
    pub off_runs: u8,
    /// Every unit searches once per this many frames.
    pub refresh: u64,
    /// The frame's index (the refresh phase).
    pub frame: u64,
}

impl UsageGate {
    /// Whether unit `unit`, with `residual_free` share of its area coded
    /// without residual and its last `off_run` searches off, skips.
    #[must_use]
    pub fn skips(&self, unit: usize, residual_free: f32, off_run: Option<u8>) -> bool {
        let refreshing = (self.frame + unit as u64).is_multiple_of(self.refresh.max(1));
        let off_enough = self.off_runs == 0 || off_run.is_some_and(|run| run >= self.off_runs);
        !refreshing && off_enough && residual_free >= self.quiet_share
    }
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

/// A frame's units that skip their filter statistics: those the codec's
/// own rule skips (`own`) and those the usage gate skips (`usage`).
#[must_use]
pub fn merge_skips(own: Option<Vec<bool>>, usage: Option<Vec<bool>>) -> Option<Vec<bool>> {
    match (own, usage) {
        (Some(mut own), Some(usage)) if own.len() == usage.len() => {
            for (own, usage) in own.iter_mut().zip(usage) {
                *own |= usage;
            }
            Some(own)
        }
        (own, usage) => own.or(usage),
    }
}

/// The usage gate of inter frame `frame` of `role` (`None`: off, or a
/// base frame while `filter.usage_base` is 0).
pub(crate) fn usage_gate(params: &FilterParams, role: FrameRole, frame: u64) -> Option<UsageGate> {
    let gated_role = role == FrameRole::NonBase || params.usage_base >= 0.5;
    (params.usage_quiet_share > 0.0 && gated_role).then(|| UsageGate {
        quiet_share: params.usage_quiet_share,
        off_runs: params.usage_off_runs.round() as u8,
        refresh: params.usage_refresh.round() as u64,
        frame,
    })
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
    FilterEffort {
        deblock,
        cdef,
        usage: None,
    }
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
                cdef: CdefEffort::Reuse,
                usage: None,
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

    #[test]
    fn the_usage_gate_skips_quiet_units_off_long_enough_outside_their_refresh() {
        let params = FilterParams {
            usage_quiet_share: 0.75,
            usage_off_runs: 1.0,
            usage_refresh: 4.0,
            ..FilterParams::default()
        };
        assert_eq!(
            usage_gate(&FilterParams::default(), FrameRole::NonBase, 1),
            None
        );
        assert_eq!(usage_gate(&params, FrameRole::Base, 1), None);
        let gate = usage_gate(&params, FrameRole::NonBase, 1).unwrap();
        assert!(gate.skips(0, 0.8, Some(1)));
        assert!(!gate.skips(0, 0.7, Some(1)));
        assert!(!gate.skips(0, 0.8, Some(0)));
        assert!(!gate.skips(0, 0.8, None));
        // Unit 3 refreshes at frame 1 (1 + 3 is a multiple of 4).
        assert!(!gate.skips(3, 1.0, Some(5)));
    }
}
