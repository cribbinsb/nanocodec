//! VVC syntax and collocated distances for the shared engine's reference slots.
//! This does not schedule pictures: `FramePlan` owns layers, selection and refresh.
use nc_encoder::FrameReferences;

#[derive(Debug, Clone, Copy)]
pub(crate) struct Syntax {
    pub distances: [u32; 2],
    pub count: usize,
    pub non_reference: bool,
}
impl Syntax {
    pub const FLAT: Self = Self {
        distances: [1, 0],
        count: 1,
        non_reference: false,
    };
}

#[derive(Debug)]
pub(crate) struct References {
    pub plan: FrameReferences,
    pub syntax: Syntax,
    pocs: Vec<u8>,
    distances: Vec<u32>,
}
impl References {
    pub fn new(layers: u8) -> Self {
        let slots = nc_encoder::reference_slots(layers);
        Self {
            plan: FrameReferences::FLAT,
            syntax: Syntax::FLAT,
            pocs: vec![0; slots],
            distances: vec![1; slots],
        }
    }
    pub fn begin(&mut self, plan: FrameReferences, poc: u8, inter: bool) {
        self.plan = plan;
        let slot = plan.reference.slot;
        // POC advances only on commit, unlike source-frame distances after drops.
        let distance = if inter {
            u32::from(poc.wrapping_sub(self.pocs[slot]))
        } else {
            1
        };
        self.syntax = Syntax {
            distances: [distance, 0],
            count: 1,
            non_reference: plan.refresh.is_empty(),
        };
        // A top-layer picture using T1 must retain T0 for the next base picture.
        // It remains inactive, so every coded block still has reference index zero.
        if inter && slot != 0 && self.pocs[slot] != self.pocs[0] {
            self.syntax.distances[1] = u32::from(poc.wrapping_sub(self.pocs[0]));
            self.syntax.count = 2;
        }
    }
    pub fn collocated_distance(&self) -> u32 {
        self.distances[self.plan.reference.slot]
    }
    pub fn commit(&mut self, poc: u8) {
        for slot in 0..self.pocs.len() {
            if self.plan.refresh.contains(slot) {
                self.pocs[slot] = poc;
                self.distances[slot] = self.syntax.distances[0];
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
        let mut refs = References::new(3);
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
        assert_eq!(refs.collocated_distance(), 2);
        // Recode the same picture, then discard it without commit.
        refs.begin(plan(1, SlotMask::NONE), 3, true);
        assert_eq!(refs.syntax.distances, [1, 3]);
        refs.begin(plan(0, SlotMask::only(0)), 3, true);
        assert_eq!(refs.syntax.distances[0], 3);
        assert_eq!(refs.syntax.count, 1);
    }

    #[test]
    fn syntax_distances_cross_poc_lsb_wrap() {
        let mut refs = References::new(3);
        refs.begin(plan(0, SlotMask::ALL), 252, false);
        refs.commit(252);
        refs.begin(plan(0, SlotMask::only(1)), 254, true);
        refs.commit(254);
        refs.begin(plan(0, SlotMask::only(0)), 0, true);
        assert_eq!(refs.syntax.distances[0], 4);
    }
}
