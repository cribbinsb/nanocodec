//! VVC syntax and POC distances for the shared engine's reference slots.
//! This does not schedule pictures: `FramePlan` owns layers, selection and refresh.
//!
//! A picture's local RPL0 (§7.3.10, `ref_pic_list_struct`) lists the
//! short-term pictures it predicts from or keeps: the previous one, the T0
//! picture a top-layer picture keeps for the next base picture, and the
//! picture in the engine's long-term slot: the second reference's last
//! high-quality picture (`refs.second`) or the long-term reference
//! (`ltr.interval`, `nc_encoder` `ltr.rs`). A picture absent from a
//! picture's RPLs leaves the DPB (§8.3.2), so that picture stays listed
//! (inactive) on every frame while it may be used again; it is active
//! (`NumRefIdxActive[0]` 2, `ref_idx_l0` 1) on frames whose plan carries it
//! as `secondary`, and the only entry of a loss-recovery picture, which
//! predicts from it and lists none of the short-term pictures it replaces
//! (the receiver may have lost them).
//!
//! The long-term slot's picture is listed as a short-term entry (as HEVC's
//! backend does), not as a long-term one (§7.4.11 `st_ref_pic_flag` 0):
//! the POC LSBs are widened to reach it (`headers::LONG_POC_BITS`), and
//! short-term marking keeps the vector scaling and temporal candidates of
//! every other reference, so one motion-vector prediction serves both.
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

/// RPL0 entries a picture lists: its reference, the T0 picture and the
/// long-term slot's picture.
pub(crate) const MAX_LISTED: usize = 3;

#[derive(Debug, Clone, Copy)]
pub(crate) struct Syntax {
    /// The listed pictures' POC distances, nearest first.
    pub distances: [u32; MAX_LISTED],
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
        distances: [1, 0, 0],
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
    /// The engine's long-term slot, when the second or the long-term
    /// reference uses it.
    long_term_slot: Option<usize>,
}
impl References {
    /// References for `layers` temporal layers, with the engine's long-term
    /// slot when `long_term` (the second or the long-term reference uses
    /// it), and `poc_bits` POC LSBs.
    pub fn new(layers: u8, long_term: bool, poc_bits: u32) -> Self {
        let slots = nc_encoder::reference_slots(layers);
        let long_term_slot = long_term.then(|| nc_encoder::long_term_slot(layers));
        Self {
            plan: FrameReferences::FLAT,
            syntax: Syntax::FLAT,
            pocs: vec![0; slots + usize::from(long_term)],
            poc_bits,
            long_term_slot,
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
            distances: [distance, 0, 0],
            count: 1,
            active: 1,
            non_reference: plan.refresh.is_empty(),
            poc_bits: self.poc_bits,
        };
        if !inter {
            return;
        }
        // A top-layer picture using T1 must retain T0 for the next base
        // picture. It remains inactive, so every coded block still has
        // reference index zero. A recovery picture replaces T0 itself.
        let base = (slot != 0 && self.pocs[slot] != self.pocs[0] && !plan.refresh.contains(0))
            .then(|| distance_to(0));
        let long_term = self.long_term_slot.map(distance_to);
        if let Some(base) = base {
            self.push(base);
        }
        if let Some(held) = long_term {
            let listed = &self.syntax.distances[..self.syntax.count];
            if !listed.contains(&held) && held <= max_listed_distance(self.poc_bits) {
                self.push(held);
                if let Some(link) = plan.secondary {
                    debug_assert_eq!(Some(link.slot), self.long_term_slot);
                    debug_assert_eq!(self.syntax.count, 2, "no second reference with layers");
                    self.syntax.active = 2;
                }
            } else {
                debug_assert!(plan.secondary.is_none(), "an unlisted second reference");
            }
        }
    }

    /// List one more (inactive) picture, `distance` back: every listed
    /// picture is older than the ones before it (the long-term slot's is
    /// marked on a base picture, T0's at the latest), as the cumulative
    /// deltas `headers::slice_layered` writes need.
    fn push(&mut self, distance: u32) {
        let count = self.syntax.count;
        debug_assert!(
            distance > self.syntax.distances[count - 1],
            "RPL0 entries in POC order"
        );
        self.syntax.distances[count] = distance;
        self.syntax.count = count + 1;
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
        assert_eq!(refs.syntax.distances[..2], [1, 3]);
        assert_eq!(refs.syntax.count, 2);
        assert_eq!(refs.syntax.active, 1);
        // Recode the same picture, then discard it without commit.
        refs.begin(plan(1, SlotMask::NONE), 3, true);
        assert_eq!(refs.syntax.distances[..2], [1, 3]);
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
        assert_eq!(refs.syntax.distances[..2], [1, 2]);
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

    #[test]
    fn the_long_term_reference_stays_listed_and_recovers_alone() {
        let layers = 3;
        let long_term = nc_encoder::long_term_slot(layers);
        let mut refs = References::new(layers, true, crate::headers::LONG_POC_BITS);
        refs.begin(plan(0, SlotMask::ALL), 0, false);
        refs.commit(0);
        // A T0 picture refreshes slot 0, the long-term slot keeps the key.
        refs.begin(plan(0, SlotMask::only(0)), 4, true);
        assert_eq!(refs.syntax.count, 1, "the key is the reference");
        refs.commit(4);
        refs.begin(plan(0, SlotMask::only(1)), 6, true);
        assert_eq!(refs.syntax.distances[..2], [2, 6]);
        refs.commit(6);
        // A top-layer picture lists its reference, T0 and the long-term one.
        refs.begin(plan(1, SlotMask::NONE), 7, true);
        assert_eq!(refs.syntax.distances, [1, 3, 7]);
        assert_eq!((refs.syntax.count, refs.syntax.active), (3, 1));
        // A recovery picture predicts from the long-term slot and lists none
        // of the short-term pictures it replaces.
        let recovery = FrameReferences {
            reference: ReferenceLink {
                slot: long_term,
                distance: 9,
            },
            refresh: SlotMask((1 << long_term) - 1),
            ..FrameReferences::FLAT
        };
        refs.begin(recovery, 9, true);
        assert_eq!((refs.syntax.distances[0], refs.syntax.count), (9, 1));
    }
}
