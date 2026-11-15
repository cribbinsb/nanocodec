//! Where the engine's reference slots live in the AV2 picture buffer, for
//! streams with a second reference (`refs.second`).
//!
//! The engine names its references by slot and refreshes a mask of slots;
//! a key frame refreshes all of them. AV2 stores a key frame in its first
//! slot only and empties the others (AVM `av2_frame_clears_multiple_inserted_in_one`),
//! so the engine's long-term slot, which holds the key frame as the second
//! reference until the next high-quality frame, has no AV2 slot of its own
//! after a key. This map keeps each engine slot's picture in some AV2 slot:
//! after a key every engine slot names AV2 slot 0, and an inter frame
//! writes one AV2 slot that no picture still held by another engine slot
//! occupies (reusing its own slot where it can).
//!
//! Only streams with a second reference use it: flat streams refresh every
//! slot implicitly, and long-term or layered streams keep naming the
//! engine's slots directly (their byte streams predate this map).

use crate::frame_header::REFERENCE_SLOTS;

const SLOTS: usize = REFERENCE_SLOTS as usize;

/// The AV2 slot of each engine slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PhysicalSlots {
    of: [u8; SLOTS],
}

impl PhysicalSlots {
    /// After a key frame: every engine slot holds it, in AV2 slot 0.
    pub(crate) const AFTER_KEY: Self = Self { of: [0; SLOTS] };

    /// The AV2 slot holding engine slot `slot`'s picture.
    pub(crate) fn of(self, slot: usize) -> u8 {
        self.of[slot]
    }

    /// The AV2 slot an inter frame refreshing engine slots `mask` writes,
    /// or `None` when it refreshes none: the AV2 slot of one of those
    /// engine slots that no other engine slot shares, else the first AV2
    /// slot no other engine slot occupies.
    pub(crate) fn target(self, mask: u8) -> Option<u8> {
        if mask == 0 {
            return None;
        }
        let refreshed = |slot: usize| mask & (1 << slot) != 0;
        let kept = (0..SLOTS)
            .filter(|&slot| !refreshed(slot))
            .fold(0_u16, |kept, slot| kept | 1 << self.of[slot]);
        let own = (0..SLOTS)
            .filter(|&slot| refreshed(slot))
            .map(|slot| self.of[slot])
            .find(|&physical| kept & (1 << physical) == 0);
        Some(own.unwrap_or_else(|| {
            (0..REFERENCE_SLOTS)
                .find(|&physical| kept & (1 << physical) == 0)
                .expect("at most seven engine slots are kept")
        }))
    }

    /// Record an inter frame that refreshed engine slots `mask` into AV2
    /// slot `target`.
    pub(crate) fn commit(&mut self, mask: u8, target: u8) {
        for (slot, physical) in self.of.iter_mut().enumerate() {
            if mask & (1 << slot) != 0 {
                *physical = target;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_key_frame_survives_in_its_slot_while_another_slot_holds_it() {
        // Engine slot 0 is the previous picture, slot 1 the second
        // reference; after a key both hold the key frame in AV2 slot 0.
        let mut slots = PhysicalSlots::AFTER_KEY;
        let first = slots.target(0b01).unwrap();
        assert_ne!(first, 0, "slot 1 still holds the key frame in AV2 slot 0");
        slots.commit(0b01, first);
        assert_eq!((slots.of(0), slots.of(1)), (first, 0));
        // The next frame reuses its own slot.
        assert_eq!(slots.target(0b01), Some(first));
        // A high-quality frame refreshes both engine slots.
        let both = slots.target(0b11).unwrap();
        slots.commit(0b11, both);
        assert_eq!(slots.of(0), slots.of(1));
        assert_eq!(slots.target(0), None);
    }
}
