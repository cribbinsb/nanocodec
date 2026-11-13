//! What the encoder tracks for a receiver that loses frames
//! (`docs/ARCHITECTURE.md` §6.1, `docs/handoff/loss-resilience.md`): which
//! coded frame each reference slot holds (the dependency signal a transport
//! carries, [`ReferencedFrames`]) and which loss reports still need a
//! repair.

use std::collections::VecDeque;

use nc_analysis::link::{MAX_REFERENCE_SLOTS, ReferenceLink};

use crate::structure::FrameReferences;

/// Recent non-reference frames remembered for loss reports; reports come
/// within a round trip, and an unknown (older) frame is treated as a
/// referenced one.
const NON_REFERENCE_MEMORY: usize = 64;

/// The coded frames one frame predicts from (`FrameMetadata::references`):
/// its reference and, with a second reference, that one too.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReferencedFrames([Option<u64>; 2]);

impl ReferencedFrames {
    /// No reference: a key frame, or a frame that is not coded.
    pub const NONE: ReferencedFrames = ReferencedFrames([None, None]);

    /// Construct from primary and optional secondary picture indices.
    #[must_use]
    pub const fn from_pair(primary: Option<u64>, secondary: Option<u64>) -> Self {
        Self([primary, secondary])
    }

    /// The referenced frames' indices (`FrameMetadata::index`).
    pub fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        self.0.iter().flatten().copied()
    }
}

/// The feedback state between frames.
#[derive(Debug, Clone)]
pub(crate) struct LossFeedback {
    /// The most recent coded frames no frame references (top temporal
    /// layer): losing one needs no recovery.
    non_reference: VecDeque<u64>,
    /// The coded frame each reference slot holds.
    slot_frames: [Option<u64>; MAX_REFERENCE_SLOTS],
    /// Index of the last key or loss-recovery frame: the receiver's
    /// pictures are whole again from it on, so a loss report of an earlier
    /// frame (late, or repeated while that frame was in flight) needs
    /// nothing.
    repaired_from: u64,
}

impl Default for LossFeedback {
    fn default() -> Self {
        Self {
            non_reference: VecDeque::with_capacity(NON_REFERENCE_MEMORY),
            slot_frames: [None; MAX_REFERENCE_SLOTS],
            repaired_from: 0,
        }
    }
}

impl LossFeedback {
    /// Whether losing frame `index` needs a repair: not a frame nothing
    /// references, nor one before the last key or recovery frame.
    pub(crate) fn needs_repair(&self, index: u64) -> bool {
        !self.non_reference.contains(&index) && index >= self.repaired_from
    }

    /// Frame `index` was coded with `references` (as a key frame if `key`,
    /// a loss-recovery frame if `recovery`): the frames it predicts from.
    pub(crate) fn coded(
        &mut self,
        index: u64,
        (key, recovery): (bool, bool),
        references: &FrameReferences,
    ) -> ReferencedFrames {
        let referenced = if key {
            ReferencedFrames::NONE
        } else {
            let slot = |link: Option<ReferenceLink>| link.and_then(|l| self.slot_frames[l.slot]);
            ReferencedFrames([slot(Some(references.reference)), slot(references.secondary)])
        };
        for (slot, frame) in self.slot_frames.iter_mut().enumerate() {
            if key || references.refresh.contains(slot) {
                *frame = Some(index);
            }
        }
        if key || recovery {
            self.repaired_from = index;
        }
        if references.refresh.is_empty() {
            if self.non_reference.len() == NON_REFERENCE_MEMORY {
                self.non_reference.pop_front();
            }
            self.non_reference.push_back(index);
        }
        referenced
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_analysis::link::SlotMask;

    fn flat(slot: usize, refresh: SlotMask) -> FrameReferences {
        FrameReferences {
            reference: ReferenceLink { slot, distance: 1 },
            refresh,
            ..FrameReferences::FLAT
        }
    }

    #[test]
    fn tracks_slots_repairs_and_non_reference_frames() {
        let mut feedback = LossFeedback::default();
        assert_eq!(
            feedback.coded(0, (true, false), &flat(0, SlotMask::ALL)),
            ReferencedFrames::NONE
        );
        let refs = feedback.coded(1, (false, false), &flat(0, SlotMask::only(0)));
        assert_eq!(refs.iter().collect::<Vec<_>>(), [0]);
        // A non-reference frame: losing it needs nothing.
        feedback.coded(2, (false, false), &flat(0, SlotMask::NONE));
        assert!(!feedback.needs_repair(2) && feedback.needs_repair(1));
        // A recovery frame from slot 1 (the key, frame 0) repairs frame 1.
        let refs = feedback.coded(3, (false, true), &flat(1, SlotMask::only(0)));
        assert_eq!(refs.iter().collect::<Vec<_>>(), [0]);
        assert!(!feedback.needs_repair(1) && feedback.needs_repair(3));
    }
}
