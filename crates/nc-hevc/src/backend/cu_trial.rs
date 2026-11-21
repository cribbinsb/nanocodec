//! Whole-CU coding trials for the engine's CU pricing
//! (`nc_encoder` `cu_pricing.rs`, `CodecBackend::begin_cu_trial`): a CU is
//! coded by the real coder, priced, and can be undone.
//!
//! The noted state and the price are the shared snapshot's
//! (`coded_snapshot.rs`): the CU's three planes with chroma weighted by the
//! inverse chroma λ ratio, plus λ times its bits at the slice's initial
//! context states.
//!
//! An intra CU coded while a trial is open (the priced alternative, or the
//! ranking's intra choice being priced) takes the full (key-frame) luma
//! mode search instead of the coarse P-frame one: in the shadow log the
//! full search's intra was worth 0.86 points of the worst six's coded cost
//! on the gated leaves against 0.59 for the coarse search's, and the
//! pricing pays for one intra CU per priced leaf at most.

use nc_common::Picture;
use nc_encoder::{Block, Mv};

use super::coded_snapshot::CodedSnapshot;
use super::{HevcBackend, InterIntraSearch};
use crate::syntax::Item;

/// The coded state of one CU before a trial (buffers reused trial to
/// trial).
#[derive(Debug, Default)]
pub(super) struct CuTrial {
    snapshot: CodedSnapshot,
    /// The P-frame intra search the trial replaced, while one is open.
    search: Option<InterIntraSearch>,
}

impl HevcBackend {
    /// `CodecBackend::begin_cu_trial`: note `block`'s coded state.
    pub(super) fn save_cu_trial(&mut self, block: &Block, recon: &Picture) -> bool {
        if self.random.is_some() || self.still_skipped.is_some() {
            return false;
        }
        let mut trial = std::mem::take(&mut self.cu_trial);
        self.save_coded(block, recon, &mut trial.snapshot);
        if let Some(search) = self.trial_intra_search
            && trial.search.is_none()
        {
            trial.search = Some(std::mem::replace(&mut self.inter_intra, search));
        }
        self.cu_trial = trial;
        true
    }

    /// `CodecBackend::end_cu_trial`: the P-frame intra search is back.
    pub(super) fn finish_cu_trial(&mut self) {
        if let Some(search) = self.cu_trial.search.take() {
            self.inter_intra = search;
        }
    }

    /// `CodecBackend::undo_cu_trial`: put the noted state back.
    pub(super) fn restore_cu_trial(&mut self, recon: &mut Picture) {
        let trial = std::mem::take(&mut self.cu_trial);
        self.restore_coded(recon, &trial.snapshot);
        self.cu_trial = trial;
    }

    /// `CodecBackend::cu_trial_cost`: the price of the CU just coded.
    pub(super) fn coded_cu_cost(&self, block: &Block, source: &Picture, recon: &Picture) -> f32 {
        debug_assert!(matches!(self.items.last(), Some(Item::Cu(_))));
        self.coded_cost(block, self.items.len() - 1, source, recon)
    }

    /// `CodecBackend::cu_trial_skip_vector`: merge candidate 0 where it
    /// predicts from the primary reference (`cu_skip_flag` with
    /// `merge_idx` 0, HEVC §7.3.8.5).
    pub(super) fn first_merge_vector(&mut self, block: &Block) -> Option<Mv> {
        let first = self.predictors(block).merge.get(0);
        (first.ref_idx == 0).then_some(first.mv)
    }
}
