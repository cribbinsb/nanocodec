//! The coded state over one square of the picture, noted so that coding it
//! can be undone and priced: the shared core of the whole-CU trials
//! (`cu_trial.rs`, `CodecBackend::begin_cu_trial`) and the partition
//! search's node trials (`node_trial.rs`, `CodecBackend::begin_node_trial`).
//! VVC's `backend/coded_snapshot.rs` does the same over its own coded state.
//!
//! A snapshot notes everything coding the square's CUs changes: its
//! reconstruction in all planes, its grid cells, the recorded items,
//! transform units and levels, the CU and TU counters, the quantization
//! group and the per-CU hints `encode_block` consumes. The ranking's cached
//! luma prediction is dropped, as trials overwrite the buffer it lives in.
//! A snapshot taken *after* a trial ([`HevcBackend::save_coded_after`])
//! also keeps the items, units and levels the trial added, so that coding
//! can be put back after others were tried: the stores are truncated to the
//! same marks first, so the indices the kept items hold stay valid.
//!
//! The price of what was coded since a snapshot is the squared error of the
//! three planes over its square (chroma weighted by HM's `2^((QpY − QpC) /
//! 3)`, the inverse of the chroma λ ratio SAO uses) plus λ times the bits of
//! the split flags and CUs recorded since, estimated by the slice writer's
//! own syntax at the slice's initial context states (as every rate estimate
//! of the backend's own trials).

use nc_common::{Picture, PlaneIndex};
use nc_encoder::Block;

use super::{EarlyMergeHint, HevcBackend, Predictors, plane_rect};
use crate::cabac::BinSink;
use crate::grid::Cell;
use crate::quant_group::QuantGroup;
use crate::rate::BitEstimator;
use crate::syntax::{
    Item, LevelTail, TransformSyntax, TransformUnit, estimate_cu_bits, split_ctx_inc,
    split_flag_present,
};

/// The coded state of one square (buffers reused snapshot to snapshot).
#[derive(Debug, Default)]
pub(super) struct CodedSnapshot {
    block: Option<Block>,
    planes: [Vec<u8>; 3],
    cells: Vec<Cell>,
    items: usize,
    tus: usize,
    levels: (usize, usize),
    /// What the trial before this snapshot added (a snapshot taken after a
    /// trial only): its items, units and levels.
    added_items: Vec<Item>,
    added_tus: Vec<TransformUnit>,
    added_levels: LevelTail,
    next_cu: u32,
    next_tu: u32,
    group: Option<QuantGroup>,
    current: Option<Predictors>,
    early_merge: Option<EarlyMergeHint>,
}

impl CodedSnapshot {
    /// The square noted.
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

impl HevcBackend {
    /// Note the reconstruction and grid cells of `block` in `snapshot`.
    fn save_square(&self, block: &Block, recon: &Picture, snapshot: &mut CodedSnapshot) {
        for (plane, saved) in PlaneIndex::ALL.into_iter().zip(&mut snapshot.planes) {
            let (x, y, n) = plane_rect(plane, block.x, block.y, block.log2_size());
            let view = recon.plane(plane).view();
            saved.clear();
            for row in 0..n {
                saved.extend_from_slice(&view.row(y + row)[x..x + n]);
            }
        }
        self.grid.save_square(block, &mut snapshot.cells);
        snapshot.block = Some(*block);
        snapshot.next_cu = self.next_cu;
        snapshot.next_tu = self.next_tu;
        snapshot.group = self.block_qp.as_ref().map(|b| b.group);
        snapshot.current = self.current;
        snapshot.early_merge = self.early_merge;
    }

    /// Note the coded state of `block` (inside the picture) in `snapshot`.
    pub(super) fn save_coded(
        &mut self,
        block: &Block,
        recon: &Picture,
        snapshot: &mut CodedSnapshot,
    ) {
        self.save_square(block, recon, snapshot);
        snapshot.items = self.items.len();
        snapshot.tus = self.tu_store.len();
        snapshot.levels = self.level_store.borrow().mark();
        snapshot.added_items.clear();
        snapshot.added_tus.clear();
        snapshot.added_levels.clear();
        // The ranking's cached luma prediction lives in `inter_pred`, which
        // every trial overwrites (the skip vector's prediction, the inter
        // fallback's): a coding after an undo must predict afresh, or it
        // reconstructs from another vector's samples than the decoder.
        self.cached_luma = None;
    }

    /// Note the coded state of `before`'s square as coded since `before`
    /// was taken in `snapshot`, so that [`Self::restore_coded`] can put this
    /// coding back after others were tried.
    pub(super) fn save_coded_after(
        &self,
        before: &CodedSnapshot,
        recon: &Picture,
        snapshot: &mut CodedSnapshot,
    ) {
        self.save_square(&before.block(), recon, snapshot);
        snapshot.items = before.items;
        snapshot.tus = before.tus;
        snapshot.levels = before.levels;
        snapshot.added_items.clear();
        snapshot
            .added_items
            .extend_from_slice(&self.items[before.items..]);
        snapshot.added_tus.clear();
        snapshot
            .added_tus
            .extend_from_slice(&self.tu_store[before.tus..]);
        self.level_store
            .borrow()
            .save_after(before.levels, &mut snapshot.added_levels);
    }

    /// Put back the coded state `snapshot` noted.
    pub(super) fn restore_coded(&mut self, recon: &mut Picture, snapshot: &CodedSnapshot) {
        let block = snapshot.block();
        for (plane, saved) in PlaneIndex::ALL.into_iter().zip(&snapshot.planes) {
            let (x, y, n) = plane_rect(plane, block.x, block.y, block.log2_size());
            let mut view = recon.plane_mut(plane).view_mut();
            for (row, samples) in saved.chunks_exact(n).enumerate() {
                view.row_mut(y + row)[x..x + n].copy_from_slice(samples);
            }
        }
        self.grid.restore_square(&block, &snapshot.cells);
        self.items.truncate(snapshot.items);
        self.items.extend_from_slice(&snapshot.added_items);
        self.tu_store.truncate(snapshot.tus);
        self.tu_store.extend_from_slice(&snapshot.added_tus);
        self.level_store
            .borrow_mut()
            .restore_after(snapshot.levels, &snapshot.added_levels);
        self.next_cu = snapshot.next_cu;
        self.next_tu = snapshot.next_tu;
        if let (Some(block_qp), Some(group)) = (self.block_qp.as_mut(), snapshot.group) {
            block_qp.group = group;
        }
        self.current = snapshot.current;
        self.early_merge = snapshot.early_merge;
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
        let mut bits = 0.0;
        for item in &self.items[first..] {
            match item {
                Item::Split { node, split } => {
                    if split_flag_present(&self.grid, node) {
                        let mut estimate = BitEstimator::new(&self.bin_costs);
                        let ctx_inc = split_ctx_inc(&self.grid, node);
                        estimate.encode_bin(&mut contexts.split_cu_flag[ctx_inc], *split);
                        bits += estimate.bits;
                    }
                }
                Item::Cu(cu) => {
                    bits += estimate_cu_bits(
                        &self.bin_costs,
                        &mut contexts,
                        &self.grid,
                        (&self.level_store.borrow(), &self.tu_store),
                        TransformSyntax {
                            scans: &self.scans,
                            max_log2: self.info.max_transform_log2,
                            inter_depth: self.info.inter_transform_depth,
                            intra_depth: self.info.intra_transform_depth,
                        },
                        self.info.merge_candidates,
                        cu,
                    );
                }
                Item::Ctu { .. } => unreachable!("a trial lies inside one CTU"),
            }
        }
        let [luma, cb, cr] = PlaneIndex::ALL.map(|plane| {
            let rect = plane_rect(plane, block.x, block.y, block.log2_size());
            self.reconstruction_sse(source, recon, plane, rect) as f32
        });
        let chroma_weight = (1.0 / self.chroma_lambda_ratio()) as f32;
        luma + chroma_weight * (cb + cr) + self.rd_lambda() * bits
    }
}
