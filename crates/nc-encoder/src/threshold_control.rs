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
//! when the tool's work changed the decision. Per tool and frame role a
//! multiplier on the preset's threshold (above 1: the tool skips more) moves
//! at the end of each cycle of `adapt.period` coded inter frames, once the
//! role has `adapt.min_runs` band runs: by `adapt.step` towards less work
//! when the band's win rate is below the tool's target over
//! `adapt.hysteresis`, towards more work when above the target times it.
//! Raising a threshold moves the band to blocks where the tool wins more
//! often, so the multiplier settles where the marginal win rate meets the
//! target: the preset's price for that tool's work. Bounds keep it near the
//! preset (`adapt.min_scale`, `adapt.max_scale`; non-base frames may go as
//! far as `adapt.nonbase_max_scale`, nearly off, while the band above the
//! bound keeps reporting). Counts of coding decisions, updated between
//! frames in coding order: the same input gives the same stream.
//!
//! The decision's tools record their band evidence in `decision.rs`
//! ([`BandEvidence`]); a codec's tool reports through
//! [`CodecBackend::filter_search_evidence`]. The design and measurements
//! are in `docs/handoff/light-pipeline.md` ("Adaptive tool thresholds").
//!
//! [`CodecBackend::filter_search_evidence`]: crate::CodecBackend::filter_search_evidence

use crate::frame_role::FrameRole;
use crate::params::{AdaptParams, DecisionParams};

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
}

const TOOL_COUNT: usize = 4;
const TOOLS: [Tool; TOOL_COUNT] = [Tool::Subpel, Tool::Rank, Tool::Probe, Tool::Cdef];
const ROLE_COUNT: usize = 2;

/// The multiplier at which a tool runs as the preset sets it.
const PRESET_SCALE: f32 = 1.0;

/// A tool's band runs and wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Evidence {
    runs: u64,
    wins: u64,
}

impl Evidence {
    fn add(&mut self, other: Self) {
        self.runs = self.runs.saturating_add(other.runs);
        self.wins = self.wins.saturating_add(other.wins);
    }
}

/// The margin band (`adapt.band`): how far inside its gate a block still
/// informs the gate's threshold, as a multiple of the threshold. A
/// threshold of 0 (the gate off) has no band.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Band(f32);

impl Band {
    /// Whether gate value `value` of a tool that runs where it is at least
    /// `threshold` lies in the band above the threshold.
    pub fn above(self, value: f32, threshold: f32) -> bool {
        threshold > 0.0 && value >= threshold && value < self.0 * threshold
    }

    /// Whether gate value `value` of a tool that runs where it is at most
    /// `threshold` lies in the band below the threshold.
    pub fn below(self, value: f32, threshold: f32) -> bool {
        threshold > 0.0 && value <= threshold && value > threshold / self.0
    }
}

/// The decision's band evidence for one coding of a frame, gathered block
/// by block while a controller wants it (`band` set); otherwise every
/// record is a single test, so default encodes pay nothing measurable.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct BandEvidence {
    band: Option<Band>,
    tools: [Evidence; TOOL_COUNT],
}

impl BandEvidence {
    /// A tool that runs where its gate value `value` is at least
    /// `threshold`: a run when `value` is in the band, won or not.
    pub fn record_above(&mut self, tool: Tool, value: f32, threshold: f32, won: bool) {
        if self.band.is_some_and(|band| band.above(value, threshold)) {
            self.add(tool, 1, u64::from(won));
        }
    }

    /// Adds `runs` band runs of `tool`, `wins` of them won.
    pub fn add(&mut self, tool: Tool, runs: u64, wins: u64) {
        self.tools[tool as usize].add(Evidence { runs, wins });
    }

    /// The band this evidence is gathered in, `None` while none is.
    pub fn band(&self) -> Option<Band> {
        self.band
    }

    /// Starts a frame's (or a recoding's) evidence afresh, gathered in
    /// `band` (`None`: not gathered).
    pub fn clear(&mut self, band: Option<Band>) {
        *self = Self {
            band,
            tools: [Evidence::default(); TOOL_COUNT],
        };
    }
}

/// The controller: a multiplier per tool and frame role, and the evidence
/// gathered since each last moved.
#[derive(Debug, Clone)]
pub(crate) struct ThresholdControl {
    params: AdaptParams,
    scales: [[f32; TOOL_COUNT]; ROLE_COUNT],
    evidence: [[Evidence; TOOL_COUNT]; ROLE_COUNT],
    frames: u32,
}

impl ThresholdControl {
    /// The controller `params` asks for: `None` while every tool is off.
    pub fn new(params: AdaptParams) -> Option<Self> {
        let control = Self {
            params,
            scales: [[PRESET_SCALE; TOOL_COUNT]; ROLE_COUNT],
            evidence: [[Evidence::default(); TOOL_COUNT]; ROLE_COUNT],
            frames: 0,
        };
        TOOLS
            .iter()
            .any(|&tool| control.adapts(tool))
            .then_some(control)
    }

    /// The margin band the decision's evidence uses.
    pub fn band(&self) -> Band {
        Band(self.params.band)
    }

    fn adapts(&self, tool: Tool) -> bool {
        let switch = match tool {
            Tool::Subpel => self.params.subpel,
            Tool::Rank => self.params.rank,
            Tool::Probe => self.params.probe,
            Tool::Cdef => self.params.cdef,
        };
        switch >= 0.5
    }

    fn target(&self, tool: Tool) -> f32 {
        match tool {
            Tool::Subpel => self.params.subpel_target,
            Tool::Rank => self.params.rank_target,
            Tool::Probe => self.params.probe_target,
            Tool::Cdef => self.params.cdef_target,
        }
    }

    fn scale(&self, role: FrameRole, tool: Tool) -> f32 {
        self.scales[role_index(role)][tool as usize]
    }

    /// `decision` with this role's thresholds: each adapted gate moved by
    /// its multiplier in the direction that skips more above 1.
    pub fn apply(&self, role: FrameRole, decision: &mut DecisionParams) {
        if self.adapts(Tool::Subpel) {
            let scale = self.scale(role, Tool::Subpel);
            decision.codec_subpel_seed_lambdas *= scale;
            decision.codec_subpel_seed_lambdas_large *= scale;
        }
        if self.adapts(Tool::Probe) {
            decision.codec_fullpel_probe_min_lambdas *= self.scale(role, Tool::Probe);
        }
        // The margin's excess over 1 is what prunes: a margin of 1 already
        // measures only candidates whose estimate beats the best.
        if self.adapts(Tool::Rank) && decision.exact_rank_margin > 1.0 {
            let excess = decision.exact_rank_margin - 1.0;
            decision.exact_rank_margin = 1.0 + excess / self.scale(role, Tool::Rank);
        }
    }

    /// How many non-base frames after a CDEF search reuse its choices
    /// before the next search, when the controller adapts CDEF: 0 while the
    /// multiplier is below 1.5, so the preset's searches stand.
    pub fn cdef_reuse_frames(&self) -> Option<u32> {
        self.adapts(Tool::Cdef)
            .then(|| (self.scale(FrameRole::NonBase, Tool::Cdef).round() as u32).saturating_sub(1))
    }

    /// After coding an inter frame of `role`: its band evidence, and what a
    /// codec search on it changed. The CDEF interval is a non-base setting,
    /// and every searched frame measures it; its runs are the search's
    /// weighted total and its wins the changed part, so its win rate is the
    /// share of the choices' value that reusing them would have lost.
    pub fn observe(&mut self, role: FrameRole, band: &BandEvidence, cdef: Option<SearchEvidence>) {
        let evidence = &mut self.evidence[role_index(role)];
        for tool in [Tool::Subpel, Tool::Rank, Tool::Probe] {
            evidence[tool as usize].add(band.tools[tool as usize]);
        }
        if let Some(search) = cdef {
            self.evidence[role_index(FrameRole::NonBase)][Tool::Cdef as usize].add(Evidence {
                runs: search.total,
                wins: search.changed,
            });
        }
        self.frames += 1;
        if self
            .frames
            .is_multiple_of(self.params.period.round().max(1.0) as u32)
        {
            self.end_cycle();
        }
    }

    /// Moves each multiplier that has enough evidence, and restarts that
    /// evidence.
    fn end_cycle(&mut self) {
        for role in [FrameRole::Base, FrameRole::NonBase] {
            let max = match role {
                FrameRole::Base => self.params.max_scale,
                FrameRole::NonBase => self.params.nonbase_max_scale,
            };
            for tool in TOOLS {
                let r = role_index(role);
                let t = tool as usize;
                let evidence = self.evidence[r][t];
                if !self.adapts(tool) || (evidence.runs as f32) < self.params.min_runs {
                    continue;
                }
                let rate = evidence.wins as f32 / evidence.runs as f32;
                let target = self.target(tool);
                let scale = &mut self.scales[r][t];
                if rate < target / self.params.hysteresis {
                    *scale *= self.params.step;
                } else if rate > target * self.params.hysteresis {
                    *scale /= self.params.step;
                }
                // The CDEF interval cannot fall below one search a frame.
                let min = match tool {
                    Tool::Cdef => PRESET_SCALE,
                    _ => self.params.min_scale,
                };
                *scale = scale.clamp(min, max);
                self.evidence[r][t] = Evidence::default();
            }
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

    fn band_with(wins: u32, runs: u32) -> BandEvidence {
        let mut band = BandEvidence::default();
        band.clear(Some(Band(2.0)));
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
    fn only_the_band_above_the_threshold_counts() {
        let mut band = BandEvidence::default();
        band.record_above(Tool::Subpel, 1.2, 1.0, true);
        assert_eq!(band.tools[Tool::Subpel as usize], Evidence::default());
        band.clear(Some(Band(2.0)));
        band.record_above(Tool::Subpel, 0.9, 1.0, true);
        band.record_above(Tool::Subpel, 2.5, 1.0, true);
        band.record_above(Tool::Subpel, 1.0, 0.0, true);
        band.record_above(Tool::Subpel, 1.2, 1.0, false);
        assert_eq!(
            band.tools[Tool::Subpel as usize],
            Evidence { runs: 1, wins: 0 }
        );
        let band = Band(2.0);
        assert!(band.below(0.08, 0.1));
        assert!(!band.below(0.04, 0.1));
        assert!(!band.below(0.12, 0.1));
        assert!(!band.below(0.0, 0.0));
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
        control.apply(FrameRole::NonBase, &mut decision);
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
            control.observe(FrameRole::Base, &BandEvidence::default(), Some(still));
        }
        assert_eq!(control.cdef_reuse_frames(), Some(1));
        let moving = SearchEvidence {
            changed: 120,
            total: 240,
        };
        for _ in 0..8 {
            control.observe(FrameRole::Base, &BandEvidence::default(), Some(moving));
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
        control.scales[role_index(FrameRole::Base)][Tool::Rank as usize] = 2.0;
        let mut decision = DecisionParams {
            exact_rank_margin: 1.1,
            ..DecisionParams::default()
        };
        control.apply(FrameRole::Base, &mut decision);
        assert!((decision.exact_rank_margin - 1.05).abs() < 1e-6);
    }
}
