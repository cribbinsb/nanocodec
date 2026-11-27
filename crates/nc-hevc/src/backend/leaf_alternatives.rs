//! What else a key-frame intra CU could be coded with, for the engine's
//! leaf search by coded cost (`nc_encoder` `leaf_rd.rs`,
//! `decision.intra_rd_modes` and `intra_rd_chroma`).
//!
//! - Luma: the modes the SATD search scored, after its choice, by SATD.
//!   The search picks by prediction SATD (or, with
//!   `hevc.intra_rd_candidates`, by its first transform block priced in
//!   the coefficient domain); the leaf search codes the CU.
//! - Chroma: planar, vertical, horizontal and DC instead of the luma mode
//!   (`intra_chroma_pred_mode` 0..3, HEVC §7.4.9.2 and Table 8-2), which
//!   the CU otherwise always follows (DM).
//!
//! The transform tree is already decided by coded cost
//! (`intra.rs` `code_luma_tree`), so there is no transform axis.

use nc_common::Picture;
use nc_encoder::{Block, IntraChoice};

use super::{HevcBackend, SliceKind};
use crate::predict;

/// The explicit chroma modes (`intra_chroma_pred_mode` 0..3).
const EXPLICIT_CHROMA_MODES: u8 = predict::CHROMA_DM;

impl HevcBackend {
    /// [`nc_encoder::CodecBackend::intra_alternatives`]: key pictures
    /// only, and not in the syntax stress test (it draws its own modes).
    pub(super) fn key_intra_alternatives(
        &mut self,
        block: &Block,
        source: &Picture,
        recon: &Picture,
    ) -> IntraChoice {
        self.key_luma_ranking = None;
        if self.slice != SliceKind::Intra || self.random.is_some() {
            return IntraChoice::default();
        }
        let mpm = self.most_probable_modes(block.x, block.y);
        let mut ranking = Vec::new();
        self.choose_intra_mode(
            (block.x, block.y, block.log2_size()),
            source,
            recon,
            mpm,
            None,
            Some(&mut ranking),
        );
        let luma = ranking.len().saturating_sub(1) as u8;
        self.key_luma_ranking = Some((*block, ranking));
        IntraChoice {
            luma,
            transform: 0,
            chroma: EXPLICIT_CHROMA_MODES,
        }
    }

    /// [`nc_encoder::CodecBackend::intra_alternative_estimate`]: the SATD
    /// cost (with √λ · mode bits) the mode search gave candidate `rank` of
    /// the ranking left by [`Self::key_intra_alternatives`].
    pub(super) fn key_alternative_estimate(&self, rank: u8) -> Option<f32> {
        let (_, ranking) = self.key_luma_ranking.as_ref()?;
        Some(ranking.get(usize::from(rank))?.1)
    }

    /// The luma mode and `intra_chroma_pred_mode` that `choice` names for
    /// `block`: the ranking left by [`Self::key_intra_alternatives`] (or
    /// `searched`, the mode search's own result, without one) and the
    /// explicit chroma mode before the alternative's number.
    pub(super) fn leaf_alternative(
        &self,
        block: &Block,
        choice: IntraChoice,
        searched: Option<u8>,
    ) -> (Option<u8>, u8) {
        // A ranking is a key picture's: a P picture's CU at the same place
        // must not read the last key's.
        let luma = self
            .key_luma_ranking
            .as_ref()
            .filter(|(ranked, _)| ranked == block && self.slice == SliceKind::Intra)
            .and_then(|(_, ranking)| Some(ranking.get(usize::from(choice.luma))?.0))
            .or(searched);
        let chroma = match choice.chroma {
            0 => predict::CHROMA_DM,
            alternative => alternative - 1,
        };
        (luma, chroma)
    }
}
