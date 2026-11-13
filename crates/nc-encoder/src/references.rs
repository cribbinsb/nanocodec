//! The reconstructions the encoder keeps: one per reference slot of the
//! temporal structure (`structure.rs`), plus the picture being coded.
//!
//! Slots hold indices into a small pool, so a key frame that refreshes
//! every slot, or a frame that refreshes none (a droppable top-layer frame),
//! costs no picture copies. A flat chain uses one slot and two pictures,
//! alternating exactly like a swap of "current" and "previous".

use nc_analysis::link::{MAX_REFERENCE_SLOTS, SlotMask};
use nc_common::Picture;
use nc_dsp::Dsp;

use crate::source_scale::scale_picture;

/// Reconstructed pictures by reference slot.
#[derive(Debug)]
pub(crate) struct ReferencePool {
    pictures: Vec<Picture>,
    /// The picture each slot holds.
    slots: [usize; MAX_REFERENCE_SLOTS],
    /// Slots in use (the rest are never read or refreshed).
    slot_count: usize,
    /// The picture being coded; no slot holds it.
    current: usize,
    /// The most recently finished picture.
    last: usize,
}

impl ReferencePool {
    /// A pool for `slot_count` slots of `width` x `height` pictures with
    /// `border` samples of border.
    ///
    /// # Errors
    ///
    /// If the picture dimensions are invalid.
    pub(crate) fn new(
        width: usize,
        height: usize,
        border: usize,
        slot_count: usize,
    ) -> nc_common::Result<Self> {
        debug_assert!((1..=MAX_REFERENCE_SLOTS).contains(&slot_count));
        let pictures = (0..=slot_count)
            .map(|_| Picture::new(width, height, border))
            .collect::<nc_common::Result<Vec<_>>>()?;
        Ok(Self {
            pictures,
            slots: [0; MAX_REFERENCE_SLOTS],
            slot_count,
            current: 1,
            last: 0,
        })
    }

    /// The reference in `slot`, the one in `secondary` if any, and the
    /// picture to reconstruct into.
    pub(crate) fn frames(
        &mut self,
        slot: usize,
        secondary: Option<usize>,
    ) -> (&Picture, Option<&Picture>, &mut Picture) {
        let (reference, current) = (self.slots[slot], self.current);
        let second = secondary.map(|s| self.slots[s]);
        debug_assert!(reference != current && second != Some(current));
        let mut primary = None;
        let mut other = None;
        let mut recon = None;
        for (i, picture) in self.pictures.iter_mut().enumerate() {
            if i == current {
                recon = Some(picture);
            } else {
                let picture: &Picture = picture;
                if i == reference {
                    primary = Some(picture);
                }
                if Some(i) == second {
                    other = Some(picture);
                }
            }
        }
        (
            primary.expect("the slot holds a picture"),
            other,
            recon.expect("the current picture exists"),
        )
    }

    /// Finish the current picture: extend its borders for prediction, make
    /// it the reference of every slot in `refresh`, and pick a free picture
    /// for the next frame.
    pub(crate) fn finish(&mut self, refresh: SlotMask, dsp: &Dsp) {
        self.pictures[self.current].extend_borders_with(dsp);
        for slot in 0..self.slot_count {
            if refresh.contains(slot) {
                self.slots[slot] = self.current;
            }
        }
        self.last = self.current;
        let held = &self.slots[..self.slot_count];
        self.current = (0..self.pictures.len())
            .find(|i| !held.contains(i))
            .expect("one picture more than slots, so one is always free");
    }

    /// Mutable access to the scratch picture being prepared, before [`Self::finish`]
    /// or [`Self::store_proxy`].
    pub(crate) fn current_mut(&mut self) -> &mut Picture {
        &mut self.pictures[self.current]
    }

    /// Extend the border of [`Self::current_mut`], place it in every slot of
    /// `refresh` without changing [`Self::last`], and select a free picture for
    /// the next frame (used to inject an upscaled inter-layer reference proxy).
    pub(crate) fn store_proxy(&mut self, refresh: SlotMask) {
        self.pictures[self.current].extend_borders();
        for slot in 0..self.slot_count {
            if refresh.contains(slot) {
                self.slots[slot] = self.current;
            }
        }
        let held = &self.slots[..self.slot_count];
        self.current = (0..self.pictures.len())
            .find(|i| !held.contains(i))
            .expect("one picture more than slots, so one is always free");
    }

    /// The most recently finished picture.
    pub(crate) fn last(&self) -> &Picture {
        &self.pictures[self.last]
    }

    /// The picture each slot in use holds, indexed by slot (`None` for
    /// slots past the ones in use).
    pub(crate) fn slot_pictures(&self) -> [Option<&Picture>; MAX_REFERENCE_SLOTS] {
        std::array::from_fn(|slot| {
            (slot < self.slot_count).then(|| &self.pictures[self.slots[slot]])
        })
    }

    /// This pool at `width` x `height` with `border`: every picture
    /// resampled (`source_scale::scale_picture`) and its border extended,
    /// the slots unchanged. The pictures are proxies of the references
    /// for the engine's decisions after a coded-size switch.
    ///
    /// # Errors
    ///
    /// If the picture dimensions are invalid.
    pub(crate) fn resampled(
        &self,
        width: usize,
        height: usize,
        border: usize,
    ) -> nc_common::Result<Self> {
        let pictures = self
            .pictures
            .iter()
            .map(|old| {
                let mut new = Picture::new(width, height, border)?;
                scale_picture(old, &mut new);
                new.extend_borders();
                Ok(new)
            })
            .collect::<nc_common::Result<Vec<_>>>()?;
        Ok(Self { pictures, ..*self })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mark(pool: &mut ReferencePool, slot: usize, value: u8) {
        let (_, _, current) = pool.frames(slot, None);
        current
            .plane_mut(nc_common::PlaneIndex::Y)
            .view_mut()
            .row_mut(0)[0] = value;
    }

    fn first_sample(picture: &Picture) -> u8 {
        picture.plane(nc_common::PlaneIndex::Y).view().row(0)[0]
    }

    #[test]
    fn non_reference_frames_keep_slots_and_become_last() {
        let mut pool = ReferencePool::new(16, 16, 0, 2).unwrap();
        mark(&mut pool, 0, 10);
        pool.finish(SlotMask::ALL, &Dsp::new());
        mark(&mut pool, 0, 20);
        pool.finish(SlotMask::NONE, &Dsp::new());
        assert_eq!(first_sample(pool.last()), 20);
        assert_eq!(first_sample(pool.frames(0, None).0), 10);
        assert_eq!(first_sample(pool.frames(1, None).0), 10);
        mark(&mut pool, 0, 30);
        pool.finish(SlotMask::only(1), &Dsp::new());
        assert_eq!(first_sample(pool.frames(0, None).0), 10);
        assert_eq!(first_sample(pool.frames(1, None).0), 30);
    }
}
