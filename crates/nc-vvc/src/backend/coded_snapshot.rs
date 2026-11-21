//! The coded state over one rectangle of the picture, noted so that coding
//! it can be undone and priced: the shared core of the whole-CU trials
//! (`cu_trial.rs`, `CodecBackend::begin_cu_trial`) and the partition
//! search's node trials (`node_trial.rs`, `CodecBackend::begin_node_trial`).
//!
//! A snapshot notes everything coding the rectangle's CUs changes: its
//! reconstruction in all planes, its motion-grid cells with the history
//! list and affine models, the recorded items, the quantization group and
//! the block-QP map's cells (deblocking reads them: a kept coding put back
//! over a lost one's QPs filtered with the wrong QP, 2026-11-21), the
//! per-CU predictors and the P-picture intra count the coarse intra search
//! reads. The ranking's cached luma prediction is dropped, so every coding
//! after an undo predicts afresh.
//!
//! The price of what was coded since a snapshot is the squared error of the
//! three planes over its rectangle (chroma weighted by `2^((QpY − QpC) / 3)`,
//! the inverse of the chroma λ ratio the residual decisions use) plus λ
//! times the bits of the items recorded since, priced by the slice writer's
//! own syntax walk (`coding_tree::code` and `syntax::block` over
//! `coeff::BitCost`) at the slice's initial context states, as every rate
//! estimate of the backend's trials.

use nc_common::{Picture, PlaneIndex};
use nc_encoder::Block;

use super::VvcBackend;
use crate::coeff::BitCost;
use crate::inter_intra::InterIntraSearch;
use crate::motion::{CuSnapshot, Predictors};
use crate::quant_group::QuantGroup;
use crate::syntax;

/// The coded state of one rectangle (buffers reused snapshot to snapshot).
#[derive(Debug, Default)]
pub(super) struct CodedSnapshot {
    block: Option<Block>,
    planes: [Vec<u8>; 3],
    grid: CuSnapshot,
    items: usize,
    /// The items recorded since the snapshot this one follows (a snapshot
    /// taken after a trial, [`VvcBackend::save_coded_after`]).
    added_items: Vec<syntax::Item>,
    group: Option<QuantGroup>,
    /// The block-QP map's cells over the rectangle (with block QP).
    qp_cells: Vec<u8>,
    current: Option<(Block, Predictors)>,
    inter_intra: Option<InterIntraSearch>,
}

impl CodedSnapshot {
    /// The rectangle noted.
    pub(super) fn block(&self) -> Block {
        self.block.expect("a snapshot was taken")
    }

    /// The number of items recorded when the snapshot was taken.
    pub(super) fn items(&self) -> usize {
        self.items
    }

    /// Note only `block` and the number of `items` recorded: what a
    /// priced-only trial needs (`CodecBackend::begin_node_pricing`).
    pub(super) fn mark(&mut self, block: &Block, items: usize) {
        self.block = Some(*block);
        self.items = items;
    }
}

/// `(x, y, width, height)` of `block` in `plane` (4:2:0).
pub(super) fn plane_rect(plane: PlaneIndex, block: &Block) -> (usize, usize, usize, usize) {
    let shift = usize::from(plane != PlaneIndex::Y);
    (
        block.x >> shift,
        block.y >> shift,
        block.width() >> shift,
        block.height() >> shift,
    )
}

impl VvcBackend {
    /// Note the block-QP map's cells over `block` (none without block QP).
    fn save_qp_cells(&self, block: &Block, saved: &mut Vec<u8>) {
        match self.block_qp.as_ref() {
            Some(block_qp) => block_qp.map.save(block, saved),
            None => saved.clear(),
        }
    }

    /// Note the coded state of `block` (inside the picture) in `snapshot`.
    pub(super) fn save_coded(
        &mut self,
        block: &Block,
        recon: &Picture,
        snapshot: &mut CodedSnapshot,
    ) {
        for (plane, saved) in PlaneIndex::ALL.into_iter().zip(&mut snapshot.planes) {
            let (x, y, w, h) = plane_rect(plane, block);
            let view = recon.plane(plane).view();
            saved.clear();
            for row in 0..h {
                saved.extend_from_slice(&view.row(y + row)[x..x + w]);
            }
        }
        self.motion.save_cu(block, &mut snapshot.grid);
        snapshot.block = Some(*block);
        snapshot.items = self.items.len();
        snapshot.added_items.clear();
        snapshot.group = self.quant_group();
        self.save_qp_cells(block, &mut snapshot.qp_cells);
        snapshot.current = self.current;
        snapshot.inter_intra = Some(self.inter_intra_search);
        // A coding after an undo must predict afresh: the cache matched the
        // ranking's winner, which the first coding may already have taken.
        self.cached_luma = None;
    }

    /// Note the coded state of `before`'s rectangle as coded since
    /// `before` was taken in `snapshot`, so that [`Self::restore_coded`]
    /// can put this coding back after others were tried.
    pub(super) fn save_coded_after(
        &self,
        before: &CodedSnapshot,
        recon: &Picture,
        snapshot: &mut CodedSnapshot,
    ) {
        let block = before.block();
        for (plane, saved) in PlaneIndex::ALL.into_iter().zip(&mut snapshot.planes) {
            let (x, y, w, h) = plane_rect(plane, &block);
            let view = recon.plane(plane).view();
            saved.clear();
            for row in 0..h {
                saved.extend_from_slice(&view.row(y + row)[x..x + w]);
            }
        }
        self.motion
            .save_cu_after(&block, &before.grid, &mut snapshot.grid);
        snapshot.block = Some(block);
        snapshot.items = before.items;
        snapshot.added_items.clear();
        snapshot
            .added_items
            .extend_from_slice(&self.items[before.items..]);
        snapshot.group = self.quant_group();
        self.save_qp_cells(&block, &mut snapshot.qp_cells);
        snapshot.current = self.current;
        snapshot.inter_intra = Some(self.inter_intra_search);
    }

    /// Put back the coded state `snapshot` noted.
    pub(super) fn restore_coded(&mut self, recon: &mut Picture, snapshot: &CodedSnapshot) {
        let block = snapshot.block();
        for (plane, saved) in PlaneIndex::ALL.into_iter().zip(&snapshot.planes) {
            let (x, y, w, _) = plane_rect(plane, &block);
            let mut view = recon.plane_mut(plane).view_mut();
            for (row, samples) in saved.chunks_exact(w).enumerate() {
                view.row_mut(y + row)[x..x + w].copy_from_slice(samples);
            }
        }
        self.motion.restore_cu(&block, &snapshot.grid);
        self.items.truncate(snapshot.items);
        self.items.extend_from_slice(&snapshot.added_items);
        self.restore_quant_group(snapshot.group);
        if let Some(block_qp) = self.block_qp.as_mut() {
            block_qp.map.restore(&block, &snapshot.qp_cells);
        }
        self.current = snapshot.current;
        if let Some(search) = snapshot.inter_intra {
            self.inter_intra_search = search;
        }
        self.cached_luma = None;
    }

    /// The price of `block` as coded by the items from index `first` on.
    pub(super) fn coded_cost(
        &self,
        block: &Block,
        first: usize,
        source: &Picture,
        recon: &Picture,
    ) -> f32 {
        let mut contexts = self.contexts.clone();
        let mut bits = BitCost::default();
        for item in &self.items[first..] {
            match item {
                &syntax::Item::Split {
                    allowed,
                    contexts: split_contexts,
                    split,
                } => {
                    crate::coding_tree::code(
                        &mut bits,
                        &mut contexts.tree,
                        allowed,
                        split_contexts,
                        split,
                    )
                    .expect("an allowed split");
                }
                syntax::Item::Leaf {
                    contexts: neighbours,
                    coding,
                } => syntax::block(
                    &mut bits,
                    &mut contexts,
                    &self.dsp,
                    (self.inter, self.references.syntax.active, self.tools),
                    neighbours,
                    coding,
                ),
                syntax::Item::Ctu { .. } => unreachable!("a trial lies inside one CTU"),
            }
        }
        let [luma, cb, cr] = PlaneIndex::ALL.map(|plane| {
            let (x, y, w, h) = plane_rect(plane, block);
            nc_encoder::block_sse_rect(&self.dsp, (source, recon), plane, (x, y, w, h)) as f32
        });
        let lambda = self.lambda;
        let chroma_weight = lambda / self.quantizers().chroma_lambda(lambda);
        luma + chroma_weight * (cb + cr) + lambda * bits.0
    }
}
