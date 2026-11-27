//! Whole-CU coding trials for the engine's CU pricing (`nc_encoder`
//! `cu_pricing.rs`, `CodecBackend::begin_cu_trial`): a CU is coded by the
//! real coder, priced, and can be undone. HEVC's `backend/cu_trial.rs` does
//! the same over its own coded state.
//!
//! The noted state and the price are the shared snapshot's
//! (`coded_snapshot.rs`): the CU's three planes with chroma weighted by the
//! inverse chroma λ ratio, plus λ times its bits at the slice's initial
//! context states.
//!
//! With `vvc.trial_full_intra`, an intra CU coded while a trial is open
//! (the priced alternative, or the ranking's intra choice being priced)
//! takes the key pictures' direction search instead of the P-picture one,
//! as HEVC's trials do (`hevc.trial_full_intra`).
//!
//! Only square CUs are tried (rectangular leaves come only from
//! `vvc.mtt`, off by default).

use nc_common::Picture;
use nc_encoder::Block;

use super::VvcBackend;
use super::coded_snapshot::CodedSnapshot;

/// The coded state of one CU before a trial (buffers reused trial to
/// trial).
#[derive(Debug, Default)]
pub(super) struct CuTrial {
    snapshot: CodedSnapshot,
    /// `vvc.trial_full_intra`.
    full_intra_search: bool,
    /// Whether a trial is open (begun and not yet ended).
    open: bool,
}

impl CuTrial {
    pub(super) fn new(full_intra_search: bool) -> Self {
        Self {
            full_intra_search,
            ..Self::default()
        }
    }

    /// Whether intra CUs take the key pictures' search now.
    pub(super) fn full_intra(&self) -> bool {
        self.open && self.full_intra_search
    }

    /// `CodecBackend::end_cu_trial`: the leaf's pricing is over.
    pub(super) fn end(&mut self) {
        self.open = false;
    }
}

impl VvcBackend {
    /// `CodecBackend::begin_cu_trial`: note `block`'s coded state.
    pub(super) fn save_cu_trial(&mut self, block: &Block, recon: &Picture) -> bool {
        if block.width() != block.height() {
            return false;
        }
        let mut trial = std::mem::take(&mut self.cu_trial);
        self.save_coded(block, recon, &mut trial.snapshot);
        trial.open = true;
        self.cu_trial = trial;
        true
    }

    /// `CodecBackend::undo_cu_trial`: put the noted state back.
    pub(super) fn restore_cu_trial(&mut self, recon: &mut Picture) {
        let mut trial = std::mem::take(&mut self.cu_trial);
        self.restore_coded(recon, &mut trial.snapshot);
        self.cu_trial = trial;
    }

    /// `CodecBackend::cu_trial_cost`: the price of the CU just coded.
    pub(super) fn coded_cu_cost(&self, block: &Block, source: &Picture, recon: &Picture) -> f32 {
        debug_assert!(matches!(
            self.items.last(),
            Some(crate::syntax::Item::Leaf { .. })
        ));
        self.coded_cost(block, self.items.len() - 1, source, recon)
    }
}
