//! The direction search of intra CUs in P pictures: a cheap one
//! (`vvc.coarse_inter_intra`: the core modes and MPMs;
//! `vvc.inter_intra_dc_only`: DC alone), except in pictures coded mostly
//! intra, which keep the ranked search.
//!
//! Most P-frame intra CUs repair occlusions or brightness changes, where the
//! four core modes and the MPMs suffice. When fast motion defeats the motion
//! search, a P picture is coded mostly intra and behaves like a key picture:
//! there the coarse set loses detail where it shows most. On
//! `bodycam_antare_0815` (150 f, CBR 105/211 kbps) the coarse set's RD cost
//! is 3.7-5.7% above the ranked search's on every CU size alike, but the
//! frames whose worst-1% VMAF NEG fell (f54-f59: worst NEG 11.2 -> 8.2) were
//! 56-99% intra; on `crowd_mot17_09` and `bodycam_bc1` (300 kbps) no P picture
//! reaches 30% intra, so the gate keeps the ranked search on about 1% / 4%
//! of their intra area and the whole saving stays (diagnosis in
//! docs/handoff/vvc.md, "vvc-fast").
use crate::intra::DirectionSearch;
use nc_encoder::Block;

/// Intra share of the P picture being coded (and of the last one), which
/// decides each P-frame intra CU's direction search.
#[derive(Debug, Clone, Copy)]
pub(crate) struct InterIntraSearch {
    /// The search outside intra-dense pictures (`vvc.inter_angular_*`).
    ranked: DirectionSearch,
    /// The cheap search where not intra-dense (`vvc.coarse_inter_intra`,
    /// `vvc.inter_intra_dc_only`); none keeps the ranked one everywhere.
    cheap: Option<DirectionSearch>,
    /// `vvc.coarse_intra_dense_share`: the intra share above which a
    /// picture is intra-dense.
    dense_share: f32,
    /// Luma samples of the picture's intra CUs coded so far.
    intra_samples: usize,
    /// Intra share of the last committed P picture (0 after a key picture).
    previous_share: f32,
}

impl InterIntraSearch {
    pub(crate) fn new(
        ranked: DirectionSearch,
        cheap: Option<DirectionSearch>,
        dense_share: f32,
    ) -> Self {
        Self {
            ranked,
            cheap,
            dense_share,
            intra_samples: 0,
            previous_share: 0.,
        }
    }

    /// Restart the picture's count (also on a recode of the same picture).
    pub(crate) fn begin_picture(&mut self) {
        self.intra_samples = 0;
    }

    /// The search for intra CU `block`: ranked once the intra CUs so far
    /// cover more than the dense share of the CTUs coded so far (at least
    /// one CTU row, so the first CUs do not decide alone), or covered more
    /// of the previous P picture.
    pub(crate) fn search(&self, block: &Block, width: usize, ctu: usize) -> DirectionSearch {
        let Some(cheap) = self.cheap else {
            return self.ranked;
        };
        let rows_above = block.y / ctu * ctu * width;
        let coded = (rows_above + (block.x / ctu + 1) * ctu * ctu).max(ctu * width);
        let share = self.intra_samples as f32 / coded as f32;
        if share > self.dense_share || self.previous_share > self.dense_share {
            self.ranked
        } else {
            cheap
        }
    }

    /// Count intra CU `block` (after its search).
    pub(crate) fn add(&mut self, block: &Block) {
        self.intra_samples += block.size() * block.size();
    }

    /// The coded picture is final: remember its share for the next one.
    pub(crate) fn commit(&mut self, inter: bool, picture_samples: usize) {
        self.previous_share = if inter {
            self.intra_samples as f32 / picture_samples as f32
        } else {
            0.
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RANKED: DirectionSearch = DirectionSearch::Ranked {
        angular_candidates: 3,
        coarse_step: 4,
        rd_candidates: 0,
        rd_margin_permille: 0,
    };
    const COARSE_SEARCH: DirectionSearch = DirectionSearch::CoreAndMpm { rd_candidates: 0 };
    const COARSE: Option<DirectionSearch> = Some(COARSE_SEARCH);
    const WIDTH: usize = 256;
    const CTU: usize = 64;

    #[test]
    fn coarse_until_the_picture_is_intra_dense() {
        let mut search = InterIntraSearch::new(RANKED, COARSE, 0.3);
        search.begin_picture();
        let first = Block::square(0, 0, 6);
        assert_eq!(search.search(&first, WIDTH, CTU), COARSE_SEARCH);
        // Two of the first row's four CTUs intra: half the row.
        search.add(&first);
        search.add(&Block::square(64, 0, 6));
        assert_eq!(search.search(&Block::square(128, 0, 6), WIDTH, CTU), RANKED);
        // Second row, first CTU: 2 of 5 CTUs coded so far are intra.
        assert_eq!(search.search(&Block::square(0, 64, 6), WIDTH, CTU), RANKED);
        let mut sparse = InterIntraSearch::new(RANKED, COARSE, 0.45);
        sparse.add(&first);
        sparse.add(&Block::square(64, 0, 6));
        assert_eq!(
            sparse.search(&Block::square(0, 64, 6), WIDTH, CTU),
            COARSE_SEARCH
        );
        // The next picture starts ranked from the previous share (2/8).
        search.commit(true, WIDTH * 128);
        search.begin_picture();
        assert_eq!(search.search(&first, WIDTH, CTU), COARSE_SEARCH);
        let mut dense = InterIntraSearch::new(RANKED, COARSE, 0.2);
        dense.add(&first);
        dense.add(&Block::square(64, 0, 6));
        dense.commit(true, WIDTH * 128);
        dense.begin_picture();
        assert_eq!(dense.search(&first, WIDTH, CTU), RANKED);
        // A key picture resets it.
        dense.commit(false, WIDTH * 128);
        assert_eq!(dense.search(&first, WIDTH, CTU), COARSE_SEARCH);
    }

    #[test]
    fn ranked_without_the_coarse_lever_or_at_full_share() {
        let mut search = InterIntraSearch::new(RANKED, None, 0.3);
        assert_eq!(search.search(&Block::square(0, 0, 4), WIDTH, CTU), RANKED);
        search = InterIntraSearch::new(RANKED, COARSE, 1.0);
        search.add(&Block::square(0, 0, 6));
        search.commit(true, CTU * CTU);
        assert_eq!(
            search.search(&Block::square(0, 0, 4), WIDTH, CTU),
            COARSE_SEARCH
        );
    }
}
