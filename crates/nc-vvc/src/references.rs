//! VVC syntax and POC distances for the shared engine's reference slots.
//! This does not schedule pictures: `FramePlan` owns layers, selection and refresh.
//!
//! A picture's local RPL0 (§7.3.10, `ref_pic_list_struct`) lists the
//! short-term pictures it predicts from or keeps: the previous one, the T0
//! picture a top-layer picture keeps for the next base picture, and with
//! the second reference (`refs.second`) the last high-quality picture held
//! in the engine's long-term slot. A picture absent from a picture's RPLs
//! leaves the DPB (§8.3.2), so that picture stays listed (inactive) on every
//! frame while it may be used again; it is active (`NumRefIdxActive[0]` 2,
//! `ref_idx_l0` 1) on frames whose plan carries it as `secondary`.
use nc_encoder::FrameReferences;

use crate::motion::{ActiveReferences, MAX_ACTIVE_REFERENCES};

/// The POC distance up to which a listed picture may stay in the DPB with
/// `poc_bits` POC LSBs: the pictures of a coded video sequence span less
/// than `MaxPicOrderCntLsb / 2` (§8.3.1), the current picture and the
/// previous one included. With `headers::POC_BITS` the engine's second
/// reference stays within 64 frames; a held one widens them
/// (`headers::LONG_POC_BITS`, `CodecBackend::SECOND_REFERENCE_REACH`).
fn max_listed_distance(poc_bits: u32) -> u32 {
    (1 << (poc_bits - 1)) - 2
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Syntax {
    /// The listed pictures' POC distances, nearest first.
    pub distances: [u32; 2],
    /// Listed pictures (`num_ref_entries[0]`).
    pub count: usize,
    /// Active ones (`NumRefIdxActive[0]`): the first `active` entries.
    pub active: usize,
    pub non_reference: bool,
    /// Bits of `ph_pic_order_cnt_lsb` (the SPS's).
    pub poc_bits: u32,
}
impl Syntax {
    pub const FLAT: Self = Self {
        distances: [1, 0],
        count: 1,
        active: 1,
        non_reference: false,
        poc_bits: crate::headers::POC_BITS,
    };

    /// The active references, for motion vector prediction.
    pub fn active_references(&self) -> ActiveReferences {
        let mut distances = [1; MAX_ACTIVE_REFERENCES];
        distances[..self.active].copy_from_slice(&self.distances[..self.active]);
        ActiveReferences {
            distances,
            count: self.active,
        }
    }
}

#[derive(Debug)]
pub(crate) struct References {
    pub plan: FrameReferences,
    pub syntax: Syntax,
    /// The POC of the picture in each slot (only its LSBs are coded).
    pocs: Vec<u16>,
    /// Bits of the POC LSBs.
    poc_bits: u32,
    /// The engine's long-term slot when the second reference is on.
    second_slot: Option<usize>,
}
impl References {
    /// References for `layers` temporal layers, with the second reference's
    /// slot when `second` is on, and `poc_bits` POC LSBs.
    pub fn new(layers: u8, second: bool, poc_bits: u32) -> Self {
        let slots = nc_encoder::reference_slots(layers);
        let second_slot = second.then(|| nc_encoder::long_term_slot(layers));
        Self {
            plan: FrameReferences::FLAT,
            syntax: Syntax::FLAT,
            pocs: vec![0; slots + usize::from(second)],
            poc_bits,
            second_slot,
        }
    }
    pub fn begin(&mut self, plan: FrameReferences, poc: u16, inter: bool) {
        self.plan = plan;
        let slot = plan.reference.slot;
        // POC advances only on commit, unlike source-frame distances after drops.
        let lsb_mask = (1u32 << self.poc_bits) - 1;
        let distance_to = |slot: usize| u32::from(poc.wrapping_sub(self.pocs[slot])) & lsb_mask;
        let distance = if inter { distance_to(slot) } else { 1 };
        self.syntax = Syntax {
            distances: [distance, 0],
            count: 1,
            active: 1,
            non_reference: plan.refresh.is_empty(),
            poc_bits: self.poc_bits,
        };
        // A top-layer picture using T1 must retain T0 for the next base picture.
        // It remains inactive, so every coded block still has reference index zero.
        if inter && slot != 0 && self.pocs[slot] != self.pocs[0] {
            self.syntax.distances[1] = distance_to(0);
            self.syntax.count = 2;
        }
        if let Some(second) = self.second_slot.filter(|_| inter) {
            let held = distance_to(second);
            debug_assert_eq!(self.syntax.count, 1, "no second reference with layers");
            if self.pocs[second] != self.pocs[slot] && held <= max_listed_distance(self.poc_bits) {
                self.syntax.distances[1] = held;
                self.syntax.count = 2;
                if let Some(link) = plan.secondary {
                    debug_assert_eq!(link.slot, second);
                    self.syntax.active = 2;
                }
            } else {
                debug_assert!(plan.secondary.is_none(), "an unlisted second reference");
            }
        }
    }
    pub fn commit(&mut self, poc: u16) {
        for slot in 0..self.pocs.len() {
            if self.plan.refresh.contains(slot) {
                self.pocs[slot] = poc;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_encoder::{ReferenceLink, SlotMask};

    fn plan(slot: usize, refresh: SlotMask) -> FrameReferences {
        FrameReferences {
            reference: ReferenceLink { slot, distance: 1 },
            refresh,
            ..FrameReferences::FLAT
        }
    }

    #[test]
    fn disposable_and_recode_attempts_do_not_replace_reference_metadata() {
        let mut refs = References::new(3, false, crate::headers::POC_BITS);
        refs.begin(plan(0, SlotMask::ALL), 0, false);
        refs.commit(0);
        refs.begin(plan(0, SlotMask::NONE), 1, true);
        refs.commit(1);
        refs.begin(plan(0, SlotMask::only(1)), 2, true);
        assert_eq!(refs.syntax.distances[0], 2);
        refs.commit(2);
        refs.begin(plan(1, SlotMask::NONE), 3, true);
        assert_eq!(refs.syntax.distances, [1, 3]);
        assert_eq!(refs.syntax.count, 2);
        assert_eq!(refs.syntax.active, 1);
        // Recode the same picture, then discard it without commit.
        refs.begin(plan(1, SlotMask::NONE), 3, true);
        assert_eq!(refs.syntax.distances, [1, 3]);
        refs.begin(plan(0, SlotMask::only(0)), 3, true);
        assert_eq!(refs.syntax.distances[0], 3);
        assert_eq!(refs.syntax.count, 1);
    }

    #[test]
    fn syntax_distances_cross_poc_lsb_wrap() {
        let mut refs = References::new(3, false, crate::headers::POC_BITS);
        refs.begin(plan(0, SlotMask::ALL), 252, false);
        refs.commit(252);
        refs.begin(plan(0, SlotMask::only(1)), 254, true);
        refs.commit(254);
        refs.begin(plan(0, SlotMask::only(0)), 0, true);
        assert_eq!(refs.syntax.distances[0], 4);
    }

    #[test]
    fn the_second_reference_stays_listed_and_is_active_when_planned() {
        let second_slot = nc_encoder::long_term_slot(1);
        let mut refs = References::new(1, true, crate::headers::POC_BITS);
        // A key refreshes every slot, the second reference's too.
        refs.begin(plan(0, SlotMask::ALL), 0, false);
        refs.commit(0);
        // The next picture predicts from the key: one entry.
        refs.begin(plan(0, SlotMask::only(0)), 1, true);
        assert_eq!((refs.syntax.count, refs.syntax.active), (1, 1));
        refs.commit(1);
        // Then the key is listed, inactive unless the plan uses it.
        refs.begin(plan(0, SlotMask::only(0)), 2, true);
        assert_eq!(refs.syntax.distances, [1, 2]);
        assert_eq!((refs.syntax.count, refs.syntax.active), (2, 1));
        let with_second = FrameReferences {
            secondary: Some(ReferenceLink {
                slot: second_slot,
                distance: 2,
            }),
            ..plan(0, SlotMask::only(0))
        };
        refs.begin(with_second, 2, true);
        assert_eq!((refs.syntax.count, refs.syntax.active), (2, 2));
        assert_eq!(
            refs.syntax.active_references(),
            ActiveReferences {
                distances: [1, 2],
                count: 2
            }
        );
    }
}
