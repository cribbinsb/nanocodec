//! A second reference: the last high-quality frame (`refs.second`,
//! `docs/ARCHITECTURE.md` §6.1).
//!
//! A flat chain predicts every frame from the previous one, so detail the
//! finer cadence base frames (`qp_cadence.rs`) or a key frame code is
//! re-predicted through each coarser frame after it and loses a little at
//! every step. Keeping the last such high-quality (HQ) frame as a second
//! reference lets each block predict from it directly where that is
//! cheaper (`VVenC`'s low-delay P GOP adds its base pictures as references:
//! −3.6% on its own engine, `docs/handoff/vvc.md` "qp-cascade"). Unlike
//! temporal layers no frame becomes disposable.
//!
//! The HQ frame lives in the long-term slot (`ltr::long_term_slot`): every
//! HQ frame also refreshes it, and each inter frame whose previous-frame
//! reference is younger than the HQ frame carries it as
//! [`FrameReferences::secondary`]. Mode decision then chooses the reference
//! per block (`encoder.rs`, `decide_inter_leaf`). Flat structures only so far,
//! and not beside a long-term reference with feedback or intra refresh.
//!
//! [`FrameSwitch`] (`refs.second_frame_min_share`, off by default) drops the
//! second reference from whole frames while the last frame that had it chose
//! it on too little of its area, offering it again every
//! `refs.second_frame_probe` frames so a scene can re-qualify: wins come in
//! runs of frames (static and panning scenes hold 92% of the chosen area).
//! Measured with screen 3.0, share 0.05, probe 8 (AV1 Default CBR, L3):
//! J -0.51% at +3.1% Jetson cycles (screen 3.0 alone: -1.64% at +14-16%),
//! the best J per cycle of the gates tried (`docs/handoff/tuning.md`).
//!
//! [`TierPlan`] (`structure.unified`, off by default) is the unified
//! structure's reference plan (`docs/handoff/unified-structure.md` §3): the
//! frame's tier decides whether it carries the second reference, with no
//! block statistics. It also runs beside temporal layers where the backend
//! allows (`CodecBackend::SECOND_REFERENCE_WITH_LAYERS`): the flat
//! mini-GOPs of an adaptive stream carry it.

use nc_analysis::link::{ReferenceLink, SlotMask};

use crate::params::{
    LtrParams, ParamOverrides, RefParams, RefreshParams, SpatialParams, StructureParams,
};
use crate::second_screen::SecondScreen;
use crate::structure::FrameReferences;

/// Whether `params` ask for a second reference: `refs.second`, or the
/// unified structure's plan (`structure.unified`).
#[must_use]
pub(crate) fn second_reference_enabled(params: &ParamOverrides) -> bool {
    RefParams::from_overrides(params).second >= 0.5 || TierPlan::new(params).is_some()
}

/// The unified structure's reference plan: which frames carry the second
/// reference, from their tier alone.
///
/// Only high-quality (HQ) frames carry it, and only where the QP cadence
/// codes them clearly finer than the frames between (its strength, the
/// plan's depth). The saving of the second reference is not additive over
/// frames: a block that predicts from the HQ frame on one frame of a
/// mini-GOP passes the detail on to the others, so one tier keeps about half
/// of the every-frame gain, and the HQ tier does it on the fewest frames.
/// Measured (AV1 Default CBR, probe build on 5dd01a46, draw 2026-11-14, cheap
/// ranking): every frame L2 J -1.96% at +21.7% Jetson cycles (full
/// decision); HQ frames only -0.70% at +4.2%; HQ frames at depth 0.75 and
/// above L3 (ladder, n=126) -0.53% [-0.76, -0.27] at +2.8%, the depth limit
/// removing the cost on a shaking camera and two thirds of it on a crowd
/// (`docs/handoff/unified-structure.md` §9).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TierPlan {
    /// `structure.secondary_min_depth`.
    min_depth: f64,
}

impl TierPlan {
    /// The plan `params` ask for (`structure.unified`), if any. The plan is
    /// a default way of coding, not a requested feature, so it steps aside
    /// for the features that need the long-term slot or a single reference
    /// themselves (a long-term reference with feedback, spatial layers,
    /// intra refresh) where an explicit `refs.second` is refused.
    pub(crate) fn new(params: &ParamOverrides) -> Option<Self> {
        let plan = StructureParams::from_overrides(params);
        let taken = LtrParams::from_overrides(params).interval > 0.0
            || RefreshParams::from_overrides(params).period > 0.0
            || SpatialParams::from_overrides(params).layers.round() > 1.0;
        (plan.unified >= 0.5 && !taken).then(|| Self {
            min_depth: f64::from(plan.secondary_min_depth),
        })
    }

    /// Whether a frame of this tier carries the second reference: an HQ
    /// frame (`high_quality`) at a cadence strength of `depth` (`None`
    /// before the frame is analysed: the plan is then not yet known, and
    /// the frame is treated as carrying it).
    fn offers(self, high_quality: bool, depth: Option<f64>) -> bool {
        high_quality && depth.is_none_or(|depth| depth >= self.min_depth)
    }
}

/// Whether the stream `params` configure keeps a frame in the long-term
/// slot (`ltr::long_term_slot`): a long-term reference (`ltr.interval`) or
/// the second reference. Backends size and signal their reference buffers
/// from it.
#[must_use]
pub fn uses_long_term_slot(params: &ParamOverrides) -> bool {
    LtrParams::from_overrides(params).interval > 0.0 || second_reference_enabled(params)
}

/// The second-reference state between frames.
#[derive(Debug, Clone)]
pub(crate) struct SecondReference {
    /// The slot holding the HQ frame; the slots below it are the structure's.
    slot: usize,
    /// Index of the frame in `slot`, once one is coded.
    held: Option<u64>,
    /// `refs.second_gate_lambdas`: blocks whose primary choice costs less
    /// per sample than this many λ do not try the second reference.
    pub(crate) gate_lambdas: f32,
    /// `refs.second_full`: the full mode decision against the second
    /// reference instead of the cheap ranking.
    pub(crate) full_decision: bool,
    /// `refs.second_refine_margin` of the cheap ranking.
    pub(crate) refine_margin: f32,
    /// `refs.second_compound_margin` gate for compound inter trials.
    pub(crate) compound_margin: f32,
    /// `refs.second_max_distance`: frames further from the HQ frame do not
    /// reference it.
    max_distance: u32,
    /// `refs.second_min_still`: frames whose still-block fraction is lower
    /// do not reference the HQ frame.
    min_still: f32,
    /// `refs.second_screen_ratio`: the blocks that try the HQ frame, when on.
    pub(crate) screen: Option<SecondScreen>,
    /// `refs.second_frame_min_share`: the frames that offer it, when on.
    switch: Option<FrameSwitch>,
    /// `structure.unified`: the tiers that carry it, when on.
    plan: Option<TierPlan>,
    /// No frame carries it for now (intra refresh running under the plan);
    /// HQ frames still refresh the slot, so it is current when released.
    suspended: bool,
}

/// The frame-level switch (module docs): the chosen share of the last frame
/// that offered the second reference, and the frames since one did.
#[derive(Debug, Clone)]
struct FrameSwitch {
    /// `refs.second_frame_min_share`.
    min_share: f32,
    /// `refs.second_frame_probe`.
    probe: u32,
    /// Chosen share of the last frame folded in (1 before any: on).
    share: f32,
    /// Frames coded without the second reference since that frame.
    skipped: u32,
    /// Luma area choosing the second reference, and offered it, in the frame
    /// being coded (folded in when the next frame is coded).
    chosen_area: u64,
    offered_area: u64,
}

impl FrameSwitch {
    fn new(params: &RefParams) -> Option<Self> {
        (params.second_frame_min_share > 0.0).then(|| Self {
            min_share: params.second_frame_min_share,
            probe: (params.second_frame_probe.round() as u32).max(1),
            share: 1.0,
            skipped: 0,
            chosen_area: 0,
            offered_area: 0,
        })
    }

    /// The share and skip count as of the next frame: the frame coded last
    /// counts once it offered the second reference to any block.
    fn current(&self) -> (f32, u32) {
        if self.offered_area > 0 {
            (self.chosen_area as f32 / self.offered_area as f32, 0)
        } else {
            (self.share, self.skipped)
        }
    }

    /// Whether the next frame offers the second reference.
    fn allows(&self) -> bool {
        let (share, skipped) = self.current();
        share >= self.min_share || skipped + 1 >= self.probe
    }

    /// The next frame is coded, `with_second` reference or not.
    fn coded(&mut self, with_second: bool) {
        (self.share, self.skipped) = self.current();
        (self.chosen_area, self.offered_area) = (0, 0);
        if !with_second {
            self.skipped += 1;
        }
    }
}

impl SecondReference {
    /// A second reference kept in `slot`, tried as `params` say.
    pub(crate) fn new(slot: usize, params: &RefParams) -> Self {
        Self {
            slot,
            held: None,
            gate_lambdas: params.second_gate_lambdas,
            full_decision: params.second_full >= 0.5,
            refine_margin: params.second_refine_margin,
            compound_margin: params.second_compound_margin,
            max_distance: params.second_max_distance.round() as u32,
            min_still: params.second_min_still,
            screen: SecondScreen::new(params),
            switch: FrameSwitch::new(params),
            plan: None,
            suspended: false,
        }
    }

    /// Whether the unified plan chooses the frames (`structure.unified`),
    /// not `refs.second` alone.
    pub(crate) fn is_planned(&self) -> bool {
        self.plan.is_some()
    }

    /// Stop offering the second reference (`suspended`), or offer it again.
    pub(crate) fn suspend(&mut self, suspended: bool) {
        self.suspended = suspended;
    }

    /// This second reference carried by the frames `plan` chooses.
    pub(crate) fn with_plan(self, plan: Option<TierPlan>) -> Self {
        Self { plan, ..self }
    }

    /// `references` (from [`SecondReference::apply`]) as the unified plan,
    /// when on, leaves them for a frame that is `high_quality` or not, at a
    /// cadence strength of `depth` when known.
    pub(crate) fn planned(
        &self,
        references: FrameReferences,
        high_quality: bool,
        depth: Option<f64>,
    ) -> FrameReferences {
        match self.plan {
            Some(plan) if self.suspended || !plan.offers(high_quality, depth) => FrameReferences {
                secondary: None,
                ..references
            },
            _ => references,
        }
    }

    /// `references` of frame `index` with the second reference: a
    /// high-quality frame (`high_quality`) also refreshes the HQ slot, and
    /// a frame whose reference is younger than the HQ frame may predict
    /// from it too, if it is at most `refs.second_max_distance` frames on
    /// and its `still_fraction` (when known) reaches `refs.second_min_still`.
    ///
    /// Both limits save the second reference's mode decision and index
    /// syntax where it may not pay, and both are off by default because
    /// they cost its gain where it pays most (H.264, CBR, period-8 cadence,
    /// `docs/handoff/engine.md` "Second reference"): distance 2 cut the
    /// P-frame instructions from +16% to +6% but `static_meva_hospital`
    /// went from -10.7% to +0.3% (it gains at distances 3-8), and a still
    /// minimum of 0.5 also switched it off there (a fixed camera with
    /// moving people has fewer still blocks than that).
    pub(crate) fn apply(
        &self,
        index: u64,
        (high_quality, still_fraction): (bool, Option<f32>),
        references: FrameReferences,
    ) -> FrameReferences {
        let refresh = if high_quality {
            SlotMask(references.refresh.0 | SlotMask::only(self.slot).0)
        } else {
            references.refresh
        };
        let primary_frame = index.saturating_sub(u64::from(references.reference.distance));
        let secondary = self
            .held
            .filter(|&held| {
                held < primary_frame
                    && references.refresh != SlotMask::ALL
                    && index - held <= u64::from(self.max_distance)
                    && still_fraction.is_none_or(|still| still >= self.min_still)
                    && self.switch.as_ref().is_none_or(FrameSwitch::allows)
            })
            .map(|held| ReferenceLink {
                slot: self.slot,
                distance: u32::try_from(index - held).unwrap_or(u32::MAX),
            });
        FrameReferences {
            refresh,
            secondary,
            ..references
        }
    }

    /// Frame `index` was coded with `references`.
    pub(crate) fn coded(&mut self, index: u64, references: &FrameReferences) {
        if references.refresh.contains(self.slot) {
            self.held = Some(index);
        }
        if let Some(switch) = self.switch.as_mut() {
            switch.coded(references.secondary.is_some());
        }
    }

    /// A leaf of `area` luma samples was offered the second reference and
    /// `chose` it (alone or in compound) or not: the frame switch's share.
    pub(crate) fn observe_leaf(&mut self, area: usize, chose: bool) {
        if let Some(switch) = self.switch.as_mut() {
            switch.offered_area += area as u64;
            if chose {
                switch.chosen_area += area as u64;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(distance: u32) -> FrameReferences {
        FrameReferences {
            reference: ReferenceLink { slot: 0, distance },
            ..FrameReferences::FLAT
        }
    }

    #[test]
    fn hq_frames_refresh_the_slot_and_later_frames_reference_them() {
        let mut second =
            SecondReference::new(1, &RefParams::from_overrides(&ParamOverrides::default()));
        // Frame 0: a key (refreshes every slot).
        let key = FrameReferences {
            refresh: SlotMask::ALL,
            ..flat(1)
        };
        let refs = second.apply(0, (true, None), key);
        assert_eq!(refs.secondary, None);
        second.coded(0, &refs);
        // Frame 1 references frame 0 as its previous frame: no second one.
        let refs = second.apply(1, (false, None), flat(1));
        assert_eq!(refs.secondary, None);
        second.coded(1, &refs);
        // Frame 2: the HQ frame 0 is older than its reference.
        let refs = second.apply(2, (false, None), flat(1));
        assert_eq!(
            refs.secondary,
            Some(ReferenceLink {
                slot: 1,
                distance: 2
            })
        );
        second.coded(2, &refs);
        // Frame 4 is HQ: it refreshes slot 1 and still references frame 0.
        let refs = second.apply(4, (true, None), flat(1));
        assert!(refs.refresh.contains(1) && refs.refresh.contains(0));
        assert_eq!(
            refs.secondary,
            Some(ReferenceLink {
                slot: 1,
                distance: 4
            })
        );
        second.coded(4, &refs);
        let refs = second.apply(5, (false, None), flat(1));
        assert_eq!(refs.secondary, None);
    }

    #[test]
    fn the_plan_gives_the_second_reference_to_deep_hq_frames_only() {
        let mut params = ParamOverrides::default();
        assert_eq!(TierPlan::new(&params), None);
        params.set("structure.unified", 1.0);
        params.set("structure.secondary_min_depth", 0.5);
        let second = SecondReference::new(1, &RefParams::from_overrides(&params))
            .with_plan(TierPlan::new(&params));
        let with_secondary = FrameReferences {
            secondary: Some(ReferenceLink {
                slot: 1,
                distance: 4,
            }),
            ..flat(1)
        };
        let carries = |high_quality, depth| {
            second
                .planned(with_secondary, high_quality, depth)
                .secondary
                .is_some()
        };
        assert!(carries(true, Some(0.5)));
        assert!(carries(true, None));
        assert!(!carries(true, Some(0.4)));
        assert!(!carries(false, Some(1.0)));
        // Suspended (intra refresh), no frame carries it.
        let mut suspended = second.clone();
        suspended.suspend(true);
        assert!(
            suspended
                .planned(with_secondary, true, Some(1.0))
                .secondary
                .is_none()
        );
        // The plan steps aside for intra refresh.
        let mut refresh = params.clone();
        refresh.set("refresh.period", 30.0);
        assert_eq!(TierPlan::new(&refresh), None);
        // Without a plan nothing changes.
        let plain = SecondReference::new(1, &RefParams::from_overrides(&params));
        assert!(
            plain
                .planned(with_secondary, false, Some(0.0))
                .secondary
                .is_some()
        );
    }

    #[test]
    fn frame_switch_drops_low_share_frames_and_probes() {
        let defaults = RefParams::from_overrides(&ParamOverrides::default());
        let mut second = SecondReference::new(
            1,
            &RefParams {
                second_frame_min_share: 0.1,
                second_frame_probe: 3.0,
                ..defaults
            },
        );
        let key = FrameReferences {
            refresh: SlotMask::ALL,
            ..flat(1)
        };
        second.coded(0, &second.apply(0, (true, None), key));
        second.coded(1, &second.apply(1, (false, None), flat(1)));
        // Frame 2 offers it; blocks choose it on 5% of the area.
        let refs = second.apply(2, (false, None), flat(1));
        assert!(refs.secondary.is_some());
        second.coded(2, &refs);
        second.observe_leaf(95, false);
        second.observe_leaf(5, true);
        // Frames 3 and 4 drop it; frame 5 is the probe.
        for index in 3..5 {
            let refs = second.apply(index, (false, None), flat(1));
            assert!(refs.secondary.is_none(), "frame {index}");
            second.coded(index, &refs);
        }
        let refs = second.apply(5, (false, None), flat(1));
        assert!(refs.secondary.is_some());
        second.coded(5, &refs);
        // The probe finds a 50% share: frame 6 keeps it.
        second.observe_leaf(50, true);
        second.observe_leaf(50, false);
        assert!(second.apply(6, (false, None), flat(1)).secondary.is_some());
    }

    #[test]
    fn distance_and_still_limits_drop_the_second_reference() {
        let defaults = RefParams::from_overrides(&ParamOverrides::default());
        let mut second = SecondReference::new(
            1,
            &RefParams {
                second_max_distance: 2.0,
                second_min_still: 0.5,
                ..defaults
            },
        );
        let key = FrameReferences {
            refresh: SlotMask::ALL,
            ..flat(1)
        };
        second.coded(0, &second.apply(0, (true, None), key));
        assert!(
            second
                .apply(2, (false, Some(0.6)), flat(1))
                .secondary
                .is_some()
        );
        assert!(
            second
                .apply(2, (false, Some(0.4)), flat(1))
                .secondary
                .is_none()
        );
        assert!(
            second
                .apply(3, (false, Some(0.9)), flat(1))
                .secondary
                .is_none()
        );
    }
}
