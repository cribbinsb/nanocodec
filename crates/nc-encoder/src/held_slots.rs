//! A backend's own copies of reference pictures, by reference slot, for the
//! slots whose normative picture is not the engine's reconstruction (AV1
//! superres and coded-size switches, AV1/VP9 spatial layers predicting
//! scaled from the layer below), and the scaled-prediction size rule those
//! codecs share.

use crate::{MAX_REFERENCE_SLOTS, SlotMask};

/// What each reference slot holds beside the engine's picture (`None`: the
/// engine's reconstruction is the slot's normative picture).
#[derive(Debug, Clone)]
pub struct HeldSlots<T> {
    slots: [Option<T>; MAX_REFERENCE_SLOTS],
}

impl<T> Default for HeldSlots<T> {
    fn default() -> Self {
        Self {
            slots: std::array::from_fn(|_| None),
        }
    }
}

impl<T: Clone> HeldSlots<T> {
    /// What slot `slot` holds, if anything.
    #[must_use]
    pub fn held(&self, slot: usize) -> Option<&T> {
        self.slots[slot].as_ref()
    }

    /// Every slot, for a backend that fills the empty ones itself.
    pub fn slots_mut(&mut self) -> &mut [Option<T>] {
        &mut self.slots
    }

    /// A frame was committed: every slot it refreshes (all of them with
    /// `all`, as for a key frame) now holds `value`, or nothing.
    pub fn commit(&mut self, all: bool, refresh: SlotMask, value: Option<&T>) {
        for (slot, held) in self.slots.iter_mut().enumerate() {
            if all || refresh.contains(slot) {
                match (held.as_mut(), value) {
                    (Some(existing), Some(value)) => existing.clone_from(value),
                    (_, value) => *held = value.cloned(),
                }
            }
        }
    }
}

/// Whether a frame of `frame` size may predict from a reference of
/// `reference` size: the reference at most twice the frame's size and at
/// least 1/16 of it, per dimension (AV1 §7.11.3.3 / §7.20, VP9 §7.2 and
/// libvpx `valid_ref_frame_size`: `2 * width >= ref_width`,
/// `width <= 16 * ref_width`).
#[must_use]
pub fn scaled_reference_allowed(reference: (usize, usize), frame: (usize, usize)) -> bool {
    let fits = |r: usize, f: usize| 2 * f >= r && f <= 16 * r;
    fits(reference.0, frame.0) && fits(reference.1, frame.1)
}
