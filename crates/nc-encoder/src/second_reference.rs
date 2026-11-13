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

use nc_analysis::link::{ReferenceLink, SlotMask};

use crate::params::{LtrParams, ParamOverrides, RefParams};
use crate::structure::FrameReferences;

/// Whether `refs.second` is on in `params`.
#[must_use]
pub(crate) fn second_reference_enabled(params: &ParamOverrides) -> bool {
    RefParams::from_overrides(params).second >= 0.5
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
