//! VVC syntax and POC distances for the shared engine's reference slots.
//! This does not schedule pictures: `FramePlan` owns layers, selection and refresh.
//!
//! A picture's local RPL0 (§7.3.10, `ref_pic_list_struct`) lists the
//! pictures it predicts from or keeps: the previous one, the T0 picture a
//! top-layer picture keeps for the next base picture, and the picture in
//! the engine's long-term slot: the second reference's last high-quality
//! picture (`refs.second`) or the long-term reference (`ltr.interval`,
//! `nc_encoder` `ltr.rs`). A picture absent from a picture's RPLs leaves
//! the DPB (§8.3.2), so that picture stays listed (inactive) on every frame
//! while it may be used again; it is active (`NumRefIdxActive[0]` 2,
//! `ref_idx_l0` 1) on frames whose plan carries it as `secondary`, and the
//! only entry of a loss-recovery picture, which predicts from it and lists
//! none of the short-term pictures it replaces (the receiver may have lost
//! them).
//!
//! The long-term reference is a long-term entry (`sps_long_term_ref_pics_flag`,
//! `st_ref_pic_flag` 0, its POC LSBs in `poc_lsb_lt`; §7.3.9, §7.4.11)
//! once no short-term entry names it, so it reaches back further than the
//! short-term distances the POC LSBs allow. The second reference's picture
//! stays a short-term entry: short-term marking keeps the vector scaling
//! and temporal candidates of every other reference; a long-held one widens
//! the POC LSBs instead (`headers::LONG_POC_BITS`).
use nc_encoder::FrameReferences;

use crate::motion::{ActiveReferences, MAX_ACTIVE_REFERENCES};

/// The POC distance up to which a short-term entry may stay in the DPB
/// with `poc_bits` POC LSBs: the pictures of a coded video sequence span
/// less than `MaxPicOrderCntLsb / 2` (§8.3.1), the current picture and the
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
    /// Which listed pictures are long-term entries (`st_ref_pic_flag` 0).
    pub long_term: [bool; MAX_LISTED],
    /// Listed pictures (`num_ref_entries[0]`).
    pub count: usize,
    /// Active ones (`NumRefIdxActive[0]`): the first `active` entries.
    pub active: usize,
    pub non_reference: bool,
    /// Bits of `ph_pic_order_cnt_lsb` (the SPS's).
    pub poc_bits: u32,
    /// `sps_long_term_ref_pics_flag`: every entry codes `st_ref_pic_flag`.
    pub long_term_refs: bool,
}
impl Syntax {
    pub const FLAT: Self = Self {
        distances: [1, 0, 0],
        long_term: [false; MAX_LISTED],
        count: 1,
        active: 1,
        non_reference: false,
        poc_bits: crate::headers::POC_BITS,
        long_term_refs: false,
    };

    /// The active references, for motion vector prediction.
    pub fn active_references(&self) -> ActiveReferences {
        let mut distances = [1; MAX_ACTIVE_REFERENCES];
        distances[..self.active].copy_from_slice(&self.distances[..self.active]);
        let mut long_term = [false; MAX_ACTIVE_REFERENCES];
        long_term[..self.active].copy_from_slice(&self.long_term[..self.active]);
        ActiveReferences {
            distances,
            long_term,
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
    /// The long-term slot's picture is a long-term entry once no
    /// short-term one names it (`sps_long_term_ref_pics_flag`).
    long_term_refs: bool,
    /// POC of the picture the stream has named as a long-term entry, once
    /// a committed picture has: it stays one (§8.3.2 marks it long-term).
    long_term_poc: Option<u16>,
    /// The current picture names the long-term slot's picture as a
    /// long-term entry.
    names_long_term: bool,
    /// The current picture is an inter picture (an intra one is an IDR).
    inter: bool,
}
impl References {
    /// References for `layers` temporal layers, with the engine's long-term
    /// slot when `long_term` (the second or the long-term reference uses
    /// it), the SPS's POC LSBs, and its picture a long-term entry with the
    /// SPS's `sps_long_term_ref_pics_flag` (the long-term reference).
    pub fn new(layers: u8, long_term: bool, tools: &crate::headers::Tools) -> Self {
        let (poc_bits, long_term_refs) = (tools.poc_bits, tools.long_term_refs);
        let slots = nc_encoder::reference_slots(layers);
        let long_term_slot = long_term.then(|| nc_encoder::long_term_slot(layers));
        Self {
            plan: FrameReferences::FLAT,
            syntax: Syntax {
                long_term_refs,
                ..Syntax::FLAT
            },
            pocs: vec![0; slots + usize::from(long_term)],
            poc_bits,
            long_term_slot,
            long_term_refs,
            long_term_poc: None,
            names_long_term: false,
            inter: false,
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
            long_term: [false; MAX_LISTED],
            count: 1,
            active: 1,
            non_reference: plan.refresh.is_empty(),
            poc_bits: self.poc_bits,
            long_term_refs: self.long_term_refs,
        };
        self.names_long_term = false;
        self.inter = inter;
        if !inter {
            return;
        }
        // A top-layer picture using T1 must retain T0 for the next base
        // picture. It remains inactive, so every coded block still has
        // reference index zero. A recovery picture replaces T0 itself.
        let base = (slot != 0 && self.pocs[slot] != self.pocs[0] && !plan.refresh.contains(0))
            .then(|| distance_to(0));
        let Some(long_term_slot) = self.long_term_slot else {
            if let Some(base) = base {
                self.push(base, false);
            }
            return;
        };
        let held = self.pocs[long_term_slot];
        let held_distance = distance_to(long_term_slot);
        self.names_long_term = self.long_term_refs
            && (self.long_term_poc == Some(held)
                || !(slot != long_term_slot && self.pocs[slot] == held
                    || base.is_some() && self.pocs[0] == held));
        let names_long_term = self.names_long_term;
        let is_held = |slot_poc: u16| names_long_term && slot_poc == held;
        self.syntax.long_term[0] = is_held(self.pocs[slot]);
        if let Some(base) = base {
            self.push(base, is_held(self.pocs[0]));
        }
        let distance = held_distance;
        if self.syntax.distances[..self.syntax.count].contains(&distance) {
            debug_assert!(plan.secondary.is_none(), "an unlisted second reference");
        } else if self.names_long_term {
            // Identified by its POC LSBs (no MSB cycle, §7.4.11): no
            // other picture of the DPB shares them within one LSB cycle.
            debug_assert!(
                distance < 1 << self.poc_bits,
                "a long-term entry within one LSB cycle"
            );
            self.push(distance, true);
        } else if distance <= max_listed_distance(self.poc_bits) {
            self.push(distance, false);
            if let Some(link) = plan.secondary {
                debug_assert_eq!(Some(link.slot), self.long_term_slot);
                debug_assert_eq!(self.syntax.count, 2, "no second reference with layers");
                self.syntax.active = 2;
            }
        } else {
            debug_assert!(plan.secondary.is_none(), "an unlisted second reference");
        }
    }

    /// List one more (inactive) picture, `distance` back, a long-term entry
    /// when `long_term`: every listed picture is older than the ones before
    /// it (the long-term slot's is marked on a base picture, T0's at the
    /// latest), as the cumulative deltas `headers::slice_layered` writes
    /// need.
    fn push(&mut self, distance: u32, long_term: bool) {
        let count = self.syntax.count;
        debug_assert!(
            distance > self.syntax.distances[count - 1],
            "RPL0 entries in POC order"
        );
        self.syntax.distances[count] = distance;
        self.syntax.long_term[count] = long_term;
        self.syntax.count = count + 1;
    }
    pub fn commit(&mut self, poc: u16) {
        // An IDR picture's empty lists release every picture (§8.3.2).
        self.long_term_poc = if self.names_long_term {
            self.long_term_slot.map(|slot| self.pocs[slot])
        } else if self.inter {
            self.long_term_poc
        } else {
            None
        };
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
    use crate::headers::Tools;
    use nc_encoder::{ReferenceLink, SlotMask};

    /// An LTR stream's tools: 8-bit POC LSBs, long-term entries.
    fn long_term_tools() -> Tools {
        Tools {
            long_term_refs: true,
            ..Tools::default()
        }
    }

    fn plan(slot: usize, refresh: SlotMask) -> FrameReferences {
        FrameReferences {
            reference: ReferenceLink { slot, distance: 1 },
            refresh,
            ..FrameReferences::FLAT
        }
    }

    #[test]
    fn disposable_and_recode_attempts_do_not_replace_reference_metadata() {
        let mut refs = References::new(3, false, &Tools::default());
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
        let mut refs = References::new(3, false, &Tools::default());
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
        let mut refs = References::new(1, true, &Tools::default());
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
                long_term: [false; 2],
                count: 2
            }
        );
    }

    #[test]
    fn the_long_term_reference_stays_listed_and_recovers_alone() {
        let layers = 3;
        let long_term = nc_encoder::long_term_slot(layers);
        let mut refs = References::new(layers, true, &long_term_tools());
        refs.begin(plan(0, SlotMask::ALL), 0, false);
        refs.commit(0);
        // A T0 picture refreshes slot 0, the long-term slot keeps the key:
        // the key is its short-term reference.
        refs.begin(plan(0, SlotMask::only(0)), 4, true);
        assert_eq!(refs.syntax.count, 1, "the key is the reference");
        assert!(!refs.syntax.long_term[0]);
        refs.commit(4);
        // Then no short-term entry names the key: it is a long-term entry.
        refs.begin(plan(0, SlotMask::only(1)), 6, true);
        assert_eq!(refs.syntax.distances[..2], [2, 6]);
        assert_eq!(refs.syntax.long_term[..2], [false, true]);
        refs.commit(6);
        // A top-layer picture lists its reference, T0 and the long-term one.
        refs.begin(plan(1, SlotMask::NONE), 7, true);
        assert_eq!(refs.syntax.distances, [1, 3, 7]);
        assert_eq!(refs.syntax.long_term, [false, false, true]);
        assert_eq!((refs.syntax.count, refs.syntax.active), (3, 1));
        refs.commit(7);
        // A recovery picture predicts from the long-term slot and lists none
        // of the short-term pictures it replaces; its reference is the
        // long-term picture, further back than short-term entries reach
        // with 8-bit POC LSBs.
        let recovery = FrameReferences {
            reference: ReferenceLink {
                slot: long_term,
                distance: 200,
            },
            refresh: SlotMask((1 << long_term) - 1),
            ..FrameReferences::FLAT
        };
        refs.begin(recovery, 200, true);
        assert_eq!((refs.syntax.distances[0], refs.syntax.count), (200, 1));
        assert!(refs.syntax.long_term[0]);
        assert!(refs.syntax.active_references().long_term[0]);
    }

    #[test]
    fn a_picture_named_long_term_stays_long_term() {
        let mut refs = References::new(1, true, &long_term_tools());
        refs.begin(plan(0, SlotMask::ALL), 0, false);
        refs.commit(0);
        // A recovery right after the key predicts from it while slot 0
        // still holds it: the recovery replaces slot 0, so the key is named
        // long-term ...
        let long_term = nc_encoder::long_term_slot(1);
        let recovery = FrameReferences {
            reference: ReferenceLink {
                slot: long_term,
                distance: 1,
            },
            refresh: SlotMask::only(0),
            ..FrameReferences::FLAT
        };
        refs.begin(recovery, 1, true);
        assert!(refs.syntax.long_term[0]);
        refs.commit(1);
        // ... and stays so on the next picture.
        refs.begin(plan(0, SlotMask::only(0)), 2, true);
        assert_eq!(refs.syntax.distances[..2], [1, 2]);
        assert_eq!(refs.syntax.long_term[..2], [false, true]);
        // A new key releases it.
        refs.begin(plan(0, SlotMask::ALL), 0, false);
        refs.commit(0);
        refs.begin(plan(0, SlotMask::only(0)), 1, true);
        assert_eq!((refs.syntax.count, refs.syntax.long_term[0]), (1, false));
    }
}
