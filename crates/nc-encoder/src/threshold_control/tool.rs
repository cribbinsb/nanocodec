//! The tools the threshold controller adapts, and what the controller needs
//! to know about each: its switch and target among the `adapt.*`
//! parameters, how its multiplier acts, and which frames' state its
//! evidence moves. A new tool is one variant here, its two parameters, the
//! place that applies its multiplier and the place that records its
//! evidence; the controller's update is shared.

use crate::frame_role::FrameRole;
use crate::params::AdaptParams;

/// The tools the controller adapts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tool {
    /// The sub-sample refinement of the search seed
    /// (`decision.codec_subpel_seed_lambdas`).
    Subpel,
    /// The exact-filter ranking's prune margin (`decision.exact_rank_margin`).
    Rank,
    /// The integer-neighbour probes (`decision.codec_fullpel_probe_min_lambdas`).
    Probe,
    /// The interval between CDEF searches on non-base frames.
    Cdef,
    /// The planner's depth removal (`planner.depth_removal_qp_scale`).
    Depth,
    /// The motion search's zero-vector exits (`analysis.static_sad_per_pixel`,
    /// `one_position_sad_per_pixel`, `skip_hme_sad_per_pixel`, together).
    Motion,
    /// The bilinear ranking's refinement of its winner: the gate below
    /// which it is not refined (`decision.winner_refine_gate_scale`).
    Refine,
    /// The sub-sample walk's quarter stage after a half stage that did not
    /// move (`decision.subpel_walk`): run, or replaced by the step the
    /// half stage's costs predict (`decision.subpel_quarter_predict`).
    Quarter,
}

pub(super) const TOOL_COUNT: usize = 8;

/// What a tool's `adapt.min_runs` counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Size {
    /// The evidence's weight. For the tools that report a run and a win it
    /// is the number of runs. A CDEF search reports the squared error its
    /// choices are worth, which passes on any searched frame: every cycle
    /// with a search is judged.
    Weight,
    /// The number of observations, for tools whose weight is a cost.
    Observations,
}

/// How a tool's multiplier acts on the encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Actuator {
    /// It scales a gate's threshold, either way from the preset's.
    Threshold,
    /// It is an interval between runs of the tool, in frames: never below
    /// the preset's one run a frame.
    Interval,
    /// It is a switch between the tool (at the preset's multiplier) and
    /// its cheap alternative (at [`SWITCHED`]), which then replaces the
    /// tool everywhere bar sampled probes.
    Switch,
}

/// The multiplier at which a switched tool runs its cheap alternative.
pub(super) const SWITCHED: f32 = 2.0;

/// A tool's `adapt.*` settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Setting {
    /// Whether the controller adapts the tool.
    pub on: bool,
    /// The share of its evidence's weight the cheap alternative may lose
    /// while the multiplier stays where it is: the tool's own target times
    /// the preset's `adapt.aggression`.
    pub target: f32,
}

impl Tool {
    pub(crate) const ALL: [Self; TOOL_COUNT] = [
        Self::Subpel,
        Self::Rank,
        Self::Probe,
        Self::Cdef,
        Self::Depth,
        Self::Motion,
        Self::Refine,
        Self::Quarter,
    ];

    pub(super) fn actuator(self) -> Actuator {
        match self {
            Self::Cdef => Actuator::Interval,
            Self::Subpel | Self::Rank | Self::Probe | Self::Depth | Self::Motion | Self::Refine => {
                Actuator::Threshold
            }
            Self::Quarter => Actuator::Switch,
        }
    }

    pub(super) fn size(self) -> Size {
        match self {
            Self::Subpel | Self::Rank | Self::Probe | Self::Cdef | Self::Depth | Self::Motion => {
                Size::Weight
            }
            Self::Refine | Self::Quarter => Size::Observations,
        }
    }

    /// The role whose multiplier the evidence of a frame of `role` moves.
    /// The CDEF interval is a non-base setting that every searched frame
    /// measures, base frames included.
    pub(super) fn informs(self, role: FrameRole) -> FrameRole {
        match self {
            Self::Cdef => FrameRole::NonBase,
            _ => role,
        }
    }

    pub(super) fn setting(self, params: &AdaptParams) -> Setting {
        let (switch, target) = match self {
            Self::Subpel => (params.subpel, params.subpel_target),
            Self::Rank => (params.rank, params.rank_target),
            Self::Probe => (params.probe, params.probe_target),
            Self::Cdef => (params.cdef, params.cdef_target),
            Self::Depth => (params.depth, params.depth_target),
            Self::Motion => (params.motion, params.motion_target),
            Self::Refine => (params.refine, params.refine_target),
            Self::Quarter => (params.quarter, params.quarter_target),
        };
        Setting {
            on: switch >= 0.5,
            target: target * params.aggression,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_list_is_in_index_order() {
        for (index, tool) in Tool::ALL.into_iter().enumerate() {
            assert_eq!(tool as usize, index);
        }
    }

    #[test]
    fn every_tool_defaults_off() {
        let params = AdaptParams::default();
        assert!(Tool::ALL.iter().all(|tool| !tool.setting(&params).on));
    }
}
