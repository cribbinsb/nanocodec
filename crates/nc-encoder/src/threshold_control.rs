//! Adaptive tool thresholds: each gated tool's threshold moves per clip and
//! frame role by what the tool achieves on the blocks that run it.
//!
//! A preset fixes every gate (refine a seed only above so many λ of
//! residual, prune a candidate beyond a cost margin, search CDEF every so
//! many frames), but what a tool earns differs several-fold between clips
//! (docs/analysis/adaptive-effort-2026-11-16.md §3.3, §3.5). Which blocks
//! it pays on cannot be predicted before coding (§3.1-3.2), but a gated
//! tool measures its own value where it runs: it knows the outcome without
//! its extra work (the unrefined seed, the pruned candidate, the reused CDEF
//! choice) and with it, on the same block and reference. That paired
//! evidence is what the study found a controller needs (§5.4), and here it
//! costs no extra coding.
//!
//! Only blocks just inside the gate inform the threshold: those whose gate
//! value lies in the margin band between the threshold and `adapt.band`
//! times it, on the side where the tool runs. Each is a *run*, and a *win*
//! when the tool's work changed the decision (`evidence.rs`). Per tool and
//! frame role a multiplier on the preset's threshold (above 1: the tool
//! skips more) moves at the end of each cycle of `adapt.period` coded inter
//! frames, once the role has `adapt.min_runs` band runs: by `adapt.step`
//! towards less work when the band's win rate is below the tool's target
//! over `adapt.hysteresis`, towards more work when above the target times
//! it (`rule.rs`).
//! Raising a threshold moves the band to blocks where the tool wins more
//! often, so the multiplier settles where the marginal win rate meets the
//! target: the preset's price for that tool's work. Bounds keep it near the
//! preset (`adapt.min_scale`, `adapt.max_scale`; non-base frames may go as
//! far as `adapt.nonbase_max_scale`, nearly off, while the band above the
//! bound keeps reporting). Counts of coding decisions, updated between
//! frames in coding order: the same input gives the same stream.
//!
//! The controller is one update shared by every tool; what differs per tool
//! is listed in `tool.rs`. The decision's tools record their band evidence
//! in `decision.rs` ([`FrameEvidence`]); a codec's tool reports through
//! [`CodecBackend::filter_search_evidence`]. The design and measurements
//! are in `docs/handoff/light-pipeline.md` ("Adaptive tool thresholds") and
//! `docs/analysis/adaptive-thresholds-2026-11-23.md`.
//!
//! [`CodecBackend::filter_search_evidence`]: crate::CodecBackend::filter_search_evidence

mod evidence;
mod rule;
mod tool;

use crate::frame_role::FrameRole;
use nc_analysis::motion::SearchConfig;

use crate::params::{AdaptParams, DecisionParams, PlannerParams};

pub(crate) use evidence::{Band, FrameEvidence, Switch, Turn};
pub(crate) use tool::Tool;

use evidence::Evidence;
use rule::{PATIENCE, Rule, Travel, Verdict};
use tool::{Actuator, SWITCHED, TOOL_COUNT};

/// What one search of a codec's choices changed against the choices the
/// frame would have reused without it, weighted by what each choice is
/// worth (AV1 CDEF: the squared error the choice saves): `changed` of the
/// `total` value of all the search's choices sits in choices it changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SearchEvidence {
    /// The value of the choices the search changed.
    pub changed: u64,
    /// The value of all its choices.
    pub total: u64,
}

const ROLES: [FrameRole; ROLE_COUNT] = [FrameRole::Base, FrameRole::NonBase];
const ROLE_COUNT: usize = 2;

/// The multiplier at which a tool runs as the preset sets it.
const PRESET_SCALE: f32 = 1.0;

/// One tool's state on the frames of one role: its multiplier and the
/// evidence gathered since the multiplier was last judged.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Gate {
    scale: f32,
    evidence: Evidence,
}

impl Default for Gate {
    fn default() -> Self {
        Self {
            scale: PRESET_SCALE,
            evidence: Evidence::default(),
        }
    }
}

/// The controller: per tool and frame role, a multiplier and the evidence
/// gathered since it was last judged.
#[derive(Debug, Clone)]
pub(crate) struct ThresholdControl {
    params: AdaptParams,
    gates: [[Gate; TOOL_COUNT]; ROLE_COUNT],
    frames: u32,
}

impl ThresholdControl {
    /// The controller `params` asks for: `None` while every tool is off.
    pub fn new(params: AdaptParams) -> Option<Self> {
        let control = Self {
            params,
            gates: [[Gate::default(); TOOL_COUNT]; ROLE_COUNT],
            frames: 0,
        };
        Tool::ALL
            .iter()
            .any(|&tool| control.adapts(tool))
            .then_some(control)
    }

    /// The margin band the decision's evidence uses.
    pub fn band(&self) -> Band {
        Band(self.params.band)
    }

    /// The tools whose evidence the controller reads.
    pub fn adapted_tools(&self) -> impl Iterator<Item = Tool> + '_ {
        Tool::ALL.into_iter().filter(|&tool| self.adapts(tool))
    }

    fn adapts(&self, tool: Tool) -> bool {
        tool.setting(&self.params).on
    }

    fn scale(&self, role: FrameRole, tool: Tool) -> f32 {
        self.gates[role_index(role)][tool as usize].scale
    }

    /// The multiplier of `tool` on frames of `role`, while the controller
    /// adapts the tool.
    fn adapted(&self, role: FrameRole, tool: Tool) -> Option<f32> {
        self.adapts(tool).then(|| self.scale(role, tool))
    }

    /// `decision` and `planner` with this role's thresholds: each adapted
    /// gate moved by its multiplier in the direction that skips more above 1.
    pub fn apply(
        &self,
        role: FrameRole,
        decision: &mut DecisionParams,
        planner: &mut PlannerParams,
    ) {
        if let Some(scale) = self.adapted(role, Tool::Depth) {
            planner.depth_removal_qp_scale *= scale;
        }
        if let Some(scale) = self.adapted(role, Tool::Subpel) {
            decision.codec_subpel_seed_lambdas *= scale;
            decision.codec_subpel_seed_lambdas_large *= scale;
        }
        if let Some(scale) = self.adapted(role, Tool::Probe) {
            decision.codec_fullpel_probe_min_lambdas *= scale;
        }
        if let Some(scale) = self.adapted(role, Tool::Refine) {
            decision.winner_refine_gate_scale *= scale;
        }
        // The margin's excess over 1 is what prunes: a margin of 1 already
        // measures only candidates whose estimate beats the best.
        if let Some(scale) = self.adapted(role, Tool::Rank)
            && decision.exact_rank_margin > 1.0
        {
            let excess = decision.exact_rank_margin - 1.0;
            decision.exact_rank_margin = 1.0 + excess / scale;
        }
    }

    /// `search` with this role's zero-vector exits, when the controller
    /// adapts them: all three move together, keeping their order.
    pub fn apply_search(&self, role: FrameRole, search: &mut SearchConfig) {
        if let Some(scale) = self.adapted(role, Tool::Motion) {
            let gates = &mut search.gates;
            gates.static_per_pixel *= scale;
            gates.one_position_per_pixel *= scale;
            gates.skip_hme_per_pixel *= scale;
        }
    }

    /// How frames of `role` run the walk's quarter stage after a half
    /// stage that did not move.
    pub fn quarter(&self, role: FrameRole) -> Switch {
        match self.adapted(role, Tool::Quarter) {
            None => Switch::Preset,
            Some(scale) if scale > PRESET_SCALE => Switch::Cheap {
                period: self.params.probe_period.round() as u32,
            },
            Some(_) => Switch::Full,
        }
    }

    /// Whether the controller adapts the motion search's exits.
    pub fn adapts_motion(&self) -> bool {
        self.adapts(Tool::Motion)
    }

    /// How many non-base frames after a CDEF search reuse its choices
    /// before the next search, when the controller adapts CDEF: 0 while the
    /// multiplier is below 1.5, so the preset's searches stand.
    pub fn cdef_reuse_frames(&self) -> Option<u32> {
        self.adapted(FrameRole::NonBase, Tool::Cdef)
            .map(|scale| (scale.round() as u32).saturating_sub(1))
    }

    /// After coding an inter frame of `role`: its evidence, and what a
    /// codec search on it changed. A CDEF search's runs are its weighted
    /// total and its wins the changed part, so its win rate is the share of
    /// the choices' value that reusing them would have lost.
    pub fn observe(
        &mut self,
        role: FrameRole,
        frame: &FrameEvidence,
        cdef: Option<SearchEvidence>,
    ) {
        let mut evidence = frame.tools;
        if let Some(search) = cdef {
            evidence[Tool::Cdef as usize].add(Evidence {
                runs: search.total,
                wins: search.changed,
                samples: 1,
            });
        }
        for tool in Tool::ALL {
            self.gates[role_index(tool.informs(role))][tool as usize]
                .evidence
                .add(evidence[tool as usize]);
        }
        self.frames += 1;
        if self
            .frames
            .is_multiple_of(self.params.period.round().max(1.0) as u32)
        {
            self.end_cycle();
        }
    }

    /// Judges each multiplier that has enough evidence, and restarts that
    /// evidence unless the rule waits for more.
    fn end_cycle(&mut self) {
        for role in ROLES {
            for tool in Tool::ALL {
                let setting = tool.setting(&self.params);
                let gate = self.gates[role_index(role)][tool as usize];
                if !setting.on || gate.evidence.size(tool.size()) < self.params.min_runs {
                    continue;
                }
                let rule = Rule {
                    target: setting.target,
                    hysteresis: self.params.hysteresis,
                    confidence: self.params.confidence,
                    patience: PATIENCE * self.params.min_runs,
                };
                let verdict = rule.judge(gate.evidence);
                self.gates[role_index(role)][tool as usize] = Gate {
                    scale: self.travel(role, tool).moved(gate.scale, verdict),
                    evidence: match verdict {
                        Verdict::Wait => gate.evidence,
                        Verdict::Less | Verdict::Hold | Verdict::More => Evidence::default(),
                    },
                };
            }
        }
    }

    /// How the multiplier of `tool` moves on frames of `role`.
    fn travel(&self, role: FrameRole, tool: Tool) -> Travel {
        let max = match role {
            FrameRole::Base => self.params.max_scale,
            FrameRole::NonBase => self.params.nonbase_max_scale,
        };
        match tool.actuator() {
            Actuator::Threshold => Travel {
                step: self.params.step,
                min: self.params.min_scale,
                max,
            },
            Actuator::Interval => Travel {
                step: self.params.step,
                min: PRESET_SCALE,
                max,
            },
            // One step either way: the tool or its cheap alternative.
            Actuator::Switch => Travel {
                step: SWITCHED,
                min: PRESET_SCALE,
                max: SWITCHED,
            },
        }
    }
}

fn role_index(role: FrameRole) -> usize {
    match role {
        FrameRole::Base => 0,
        FrameRole::NonBase => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> AdaptParams {
        AdaptParams {
            subpel: 1.0,
            min_runs: 10.0,
            period: 1.0,
            ..AdaptParams::default()
        }
    }

    fn band_with(wins: u32, runs: u32) -> FrameEvidence {
        let mut band = FrameEvidence::default();
        band.clear(Some(Band(2.0)), Tool::ALL, Switch::Preset);
        for i in 0..runs {
            band.record_above(Tool::Subpel, 1.5, 1.0, i < wins);
        }
        band
    }

    #[test]
    fn every_tool_off_needs_no_controller() {
        assert!(ThresholdControl::new(AdaptParams::default()).is_none());
    }

    #[test]
    fn a_rarely_winning_tool_skips_more_and_a_winning_one_less() {
        let mut control = ThresholdControl::new(params()).expect("subpel adapts");
        control.observe(FrameRole::NonBase, &band_with(0, 20), None);
        assert_eq!(control.scale(FrameRole::NonBase, Tool::Subpel), 1.25);
        assert_eq!(control.scale(FrameRole::Base, Tool::Subpel), 1.0);
        control.observe(FrameRole::NonBase, &band_with(20, 20), None);
        control.observe(FrameRole::NonBase, &band_with(20, 20), None);
        assert_eq!(control.scale(FrameRole::NonBase, Tool::Subpel), 0.8);
        let mut decision = DecisionParams {
            codec_subpel_seed_lambdas: 0.1,
            ..DecisionParams::default()
        };
        control.apply(
            FrameRole::NonBase,
            &mut decision,
            &mut PlannerParams::default(),
        );
        assert!((decision.codec_subpel_seed_lambdas - 0.08).abs() < 1e-6);
    }

    #[test]
    fn too_little_evidence_carries_over() {
        let mut control = ThresholdControl::new(params()).expect("subpel adapts");
        control.observe(FrameRole::Base, &band_with(0, 6), None);
        assert_eq!(control.scale(FrameRole::Base, Tool::Subpel), 1.0);
        control.observe(FrameRole::Base, &band_with(0, 6), None);
        assert_eq!(control.scale(FrameRole::Base, Tool::Subpel), 1.25);
    }

    #[test]
    fn a_confident_controller_keeps_undecided_evidence() {
        let mut control = ThresholdControl::new(AdaptParams {
            confidence: 2.0,
            ..params()
        })
        .expect("subpel adapts");
        // 1 win in 20 at target 0.15 is under the band (0.1..0.225), but
        // not by two standard errors until 80 runs agree.
        control.observe(FrameRole::Base, &band_with(1, 20), None);
        control.observe(FrameRole::Base, &band_with(0, 20), None);
        control.observe(FrameRole::Base, &band_with(0, 20), None);
        assert_eq!(control.scale(FrameRole::Base, Tool::Subpel), 1.0);
        control.observe(FrameRole::Base, &band_with(0, 20), None);
        assert_eq!(control.scale(FrameRole::Base, Tool::Subpel), 1.25);
    }

    #[test]
    fn aggression_scales_every_target() {
        let mut control = ThresholdControl::new(AdaptParams {
            aggression: 2.0,
            ..params()
        })
        .expect("subpel adapts");
        // At the target 0.15 x 2 the band is 0.2..0.45: 3 wins in 20
        // tighten, and 5 in 20 hold where the plain target would loosen
        // (above 0.225).
        control.observe(FrameRole::Base, &band_with(3, 20), None);
        assert_eq!(control.scale(FrameRole::Base, Tool::Subpel), 1.25);
        control.observe(FrameRole::Base, &band_with(5, 20), None);
        assert_eq!(control.scale(FrameRole::Base, Tool::Subpel), 1.25);
    }

    #[test]
    fn a_cost_weighted_tool_needs_its_observations_not_its_weight() {
        let mut control = ThresholdControl::new(AdaptParams {
            refine: 1.0,
            min_runs: 3.0,
            period: 1.0,
            ..AdaptParams::default()
        })
        .expect("refine adapts");
        let mut frame = FrameEvidence::default();
        frame.clear(Some(Band(2.0)), Tool::ALL, Switch::Preset);
        frame.record_saving_above(Tool::Refine, 1.5, 1.0, (1000.0, 999.0));
        frame.record_saving_above(Tool::Refine, 1.5, 1.0, (1000.0, 1000.0));
        control.observe(FrameRole::Base, &frame, None);
        assert_eq!(control.scale(FrameRole::Base, Tool::Refine), 1.0);
        control.observe(FrameRole::Base, &frame, None);
        assert_eq!(control.scale(FrameRole::Base, Tool::Refine), 1.25);
        let mut decision = DecisionParams::default();
        control.apply(
            FrameRole::Base,
            &mut decision,
            &mut PlannerParams::default(),
        );
        assert_eq!(decision.winner_refine_gate_scale, 1.25);
    }

    #[test]
    fn a_switched_tool_goes_cheap_while_it_saves_little_and_back_when_probes_pay() {
        let mut control = ThresholdControl::new(AdaptParams {
            quarter: 1.0,
            quarter_target: 0.01,
            probe_period: 4.0,
            min_runs: 2.0,
            period: 1.0,
            ..AdaptParams::default()
        })
        .expect("quarter adapts");
        assert_eq!(control.quarter(FrameRole::Base), Switch::Full);
        let frame = |kept: f32| {
            let mut frame = FrameEvidence::default();
            frame.clear(Some(Band(2.0)), Tool::ALL, Switch::Full);
            frame.record_quarter(1000.0, kept);
            frame.record_quarter(1000.0, kept);
            frame
        };
        control.observe(FrameRole::Base, &frame(999.0), None);
        assert_eq!(
            control.quarter(FrameRole::Base),
            Switch::Cheap { period: 4 }
        );
        assert_eq!(control.quarter(FrameRole::NonBase), Switch::Full);
        control.observe(FrameRole::Base, &frame(990.0), None);
        assert_eq!(
            control.quarter(FrameRole::Base),
            Switch::Cheap { period: 4 }
        );
        control.observe(FrameRole::Base, &frame(900.0), None);
        assert_eq!(control.quarter(FrameRole::Base), Switch::Full);
    }

    #[test]
    fn multipliers_stay_in_their_bounds() {
        let mut control = ThresholdControl::new(params()).expect("subpel adapts");
        for _ in 0..50 {
            control.observe(FrameRole::Base, &band_with(0, 20), None);
            control.observe(FrameRole::NonBase, &band_with(20, 20), None);
        }
        assert_eq!(control.scale(FrameRole::Base, Tool::Subpel), 4.0);
        assert_eq!(control.scale(FrameRole::NonBase, Tool::Subpel), 0.5);
    }

    #[test]
    fn stable_cdef_choices_lengthen_the_interval_between_searches() {
        let mut control = ThresholdControl::new(AdaptParams {
            cdef: 1.0,
            period: 1.0,
            ..AdaptParams::default()
        })
        .expect("cdef adapts");
        assert_eq!(control.cdef_reuse_frames(), Some(0));
        let still = SearchEvidence {
            changed: 0,
            total: 240,
        };
        for _ in 0..4 {
            control.observe(FrameRole::Base, &FrameEvidence::default(), Some(still));
        }
        assert_eq!(control.cdef_reuse_frames(), Some(1));
        let moving = SearchEvidence {
            changed: 120,
            total: 240,
        };
        for _ in 0..8 {
            control.observe(FrameRole::Base, &FrameEvidence::default(), Some(moving));
        }
        assert_eq!(control.cdef_reuse_frames(), Some(0));
    }

    #[test]
    fn a_margin_moves_towards_one() {
        let mut control = ThresholdControl::new(AdaptParams {
            rank: 1.0,
            ..AdaptParams::default()
        })
        .expect("rank adapts");
        control.gates[role_index(FrameRole::Base)][Tool::Rank as usize].scale = 2.0;
        let mut decision = DecisionParams {
            exact_rank_margin: 1.1,
            ..DecisionParams::default()
        };
        control.apply(
            FrameRole::Base,
            &mut decision,
            &mut PlannerParams::default(),
        );
        assert!((decision.exact_rank_margin - 1.05).abs() < 1e-6);
    }
}
