//! Partition trials for the engine's RD partition search (`nc_encoder`
//! `rd_split.rs`, `CodecBackend::begin_node_trial`): a node's partition
//! symbols and blocks are coded by the real coder, priced over the node,
//! and can be undone, so the engine can compare a leaf and a quad split by
//! what they cost to code. HEVC's and VVC's `backend/node_trial.rs` are the
//! same stack over their recorded syntax.
//!
//! AV1 codes symbols as it goes, adapting its CDFs, so a snapshot holds
//! the coder's state as well as the picture's:
//! - **Symbols.** From a frame's first node trial on, the writer logs its
//!   symbols (`SymbolWriter::start_logging`, coded at the frame's end, as
//!   frames with per-block CDEF indices are), so a trial's symbols are a
//!   run of the log after a mark:
//!   undone by truncation, kept as a copy of the run, and priced as the sum
//!   of `-log2` of their logged probabilities, which is what the adapted
//!   CDFs make them cost.
//! - **CDFs.** An undo journal (`nc_encoder::undo_journal`, kept by the
//!   writer) records each CDF's state before it adapts, from the first
//!   full trial until the outermost one ends; a trial is a mark in it.
//!   An undo exchanges the entries after the mark back newest first; a
//!   kept coding's undone entries (each then holding the state its write
//!   produced) are set aside in the trial and redone if that coding wins.
//!   It replaced a copy of the whole `CdfContext` (tens of KiB with its
//!   symbol-cost caches) per trial and undo, which was most of the
//!   search's added instructions (docs/handoff/av1.md).
//! - **The square.** Its reconstruction in all planes, mode-info cells (and
//!   secondary vectors), coefficient and transform-size contexts above and
//!   left, CDEF coded-unit masks and `cdef_idx` slot, segment ids, the
//!   superblock quantizer state and the per-block hints `encode_block`
//!   consumes. The block caches keyed by position (the ranking's luma
//!   prediction, the intra levels, the transform-split trial) are dropped,
//!   so every coding after an undo predicts afresh.
//!
//! A trial that is only priced (a node with one candidate,
//! `CodecBackend::begin_node_pricing`) notes only its node and mark: on
//! AV1 Slow the copies were most of the search's added instructions.
//!
//! The price is the squared error of the three planes over the node
//! (chroma weighted by the squared luma-to-chroma step ratio, the inverse
//! of the chroma λ scaling of the trellis) plus λ times the bits.

use nc_common::{Picture, PlaneIndex};
use nc_encoder::undo_journal::KeptRun;
use nc_encoder::{Block, CodecBackend};

use super::Av1Backend;
use super::delta_q::DeltaQ;
use super::{CurrentBlock, MI_LOG2};
use crate::cdf::CdfState;
use crate::modeinfo::SavedCells;
use crate::range_coder::{LiteralSlot, SymbolTail};

/// The coded state of one node (buffers reused trial to trial).
#[derive(Debug, Default)]
struct Snapshot {
    block: Option<Block>,
    planes: [Vec<u8>; 3],
    cells: SavedCells,
    /// Coefficient contexts above and left, per plane.
    coeff_contexts: [Vec<u8>; 3],
    txfm_contexts: Vec<u8>,
    cdef_coded: Vec<u8>,
    cdef_slot: Option<LiteralSlot>,
    segment_ids: Vec<u8>,
    delta_q: DeltaQ,
    /// The symbol log's length at the trial's start.
    mark: usize,
    /// The symbols the trial logged (a snapshot taken after it only).
    symbols: SymbolTail,
    current: Option<CurrentBlock>,
    luma_copy: Option<(usize, usize)>,
}

impl Snapshot {
    fn block(&self) -> Block {
        self.block.expect("a snapshot was taken")
    }
}

/// One node trial: the state before it and its kept coding.
#[derive(Debug, Default)]
struct NodeTrial {
    before: Snapshot,
    kept: Snapshot,
    /// Whether the trial can be undone (not only priced).
    full: bool,
    /// The CDF journal's mark at the trial's start.
    cdf_mark: usize,
    /// The kept coding's CDF writes, set aside at the undo after it.
    kept_cdfs: KeptRun<CdfState>,
    /// The kept coding's CDFs are still the backend's own (not yet set
    /// aside in `kept_cdfs`).
    kept_cdfs_current: bool,
}

/// The open node trials, innermost last.
#[derive(Debug, Default)]
pub(super) struct NodeTrials {
    /// Trials; the first `open` are in use, the rest are spare buffers.
    snapshots: Vec<NodeTrial>,
    open: usize,
    /// How many of the open trials are full (the CDF journal records while
    /// any is).
    full_open: usize,
}

/// `(x, y, side)` of square `block` in `plane` (4:2:0).
fn plane_square(plane: PlaneIndex, block: &Block) -> (usize, usize, usize) {
    let shift = usize::from(plane != PlaneIndex::Y);
    (block.x >> shift, block.y >> shift, block.width() >> shift)
}

impl Av1Backend {
    /// The index of the 64x64 CDEF filter block holding `block`.
    fn cdef_slot_index(&self, block: &Block) -> usize {
        block.y / 64 * self.grids.width.div_ceil(64) + block.x / 64
    }

    /// Note the coded state of `block` in `snapshot` (the symbol mark
    /// aside).
    fn save_node(&self, block: &Block, recon: &Picture, snapshot: &mut Snapshot) {
        for (plane, saved) in PlaneIndex::ALL.into_iter().zip(&mut snapshot.planes) {
            let (x, y, n) = plane_square(plane, block);
            let view = recon.plane(plane).view();
            saved.clear();
            for row in 0..n {
                saved.extend_from_slice(&view.row(y + row)[x..x + n]);
            }
        }
        let (mi_col, mi_row) = (block.x >> MI_LOG2, block.y >> MI_LOG2);
        let side_mi = block.width() >> MI_LOG2;
        self.grids
            .mode_info
            .save_square((mi_row, mi_col), side_mi, &mut snapshot.cells);
        for (plane, saved) in snapshot.coeff_contexts.iter_mut().enumerate() {
            let (x4, y4, units) = Self::plane_units(block, plane);
            saved.clear();
            saved.extend_from_slice(&self.grids.contexts.above[plane][x4..x4 + units]);
            saved.extend_from_slice(&self.grids.contexts.left[plane][y4..y4 + units]);
        }
        self.grids
            .txfm
            .save(mi_col, mi_row, side_mi, &mut snapshot.txfm_contexts);
        self.grids
            .cdef_map
            .save_coded((mi_col, mi_row), side_mi, &mut snapshot.cdef_coded);
        snapshot.cdef_slot = self.grids.cdef_slots[self.cdef_slot_index(block)];
        self.segments
            .save_square((block.x, block.y), block.width(), &mut snapshot.segment_ids);
        snapshot.delta_q.clone_from(&self.delta_q);
        snapshot.current = self.current;
        snapshot.luma_copy = self.luma_copy;
        snapshot.block = Some(*block);
    }

    /// Put back the coded state `snapshot` noted.
    fn restore_node(&mut self, recon: &mut Picture, snapshot: &Snapshot) {
        let block = snapshot.block();
        for (plane, saved) in PlaneIndex::ALL.into_iter().zip(&snapshot.planes) {
            let (x, y, n) = plane_square(plane, &block);
            let mut view = recon.plane_mut(plane).view_mut();
            for (row, samples) in saved.chunks_exact(n).enumerate() {
                view.row_mut(y + row)[x..x + n].copy_from_slice(samples);
            }
        }
        let (mi_col, mi_row) = (block.x >> MI_LOG2, block.y >> MI_LOG2);
        let side_mi = block.width() >> MI_LOG2;
        self.grids
            .mode_info
            .restore_square((mi_row, mi_col), side_mi, &snapshot.cells);
        for (plane, saved) in snapshot.coeff_contexts.iter().enumerate() {
            let (x4, y4, units) = Self::plane_units(&block, plane);
            let (above, left) = saved.split_at(units);
            self.grids.contexts.above[plane][x4..x4 + units].copy_from_slice(above);
            self.grids.contexts.left[plane][y4..y4 + units].copy_from_slice(left);
        }
        self.grids
            .txfm
            .restore(mi_col, mi_row, side_mi, &snapshot.txfm_contexts);
        self.grids
            .cdef_map
            .restore_coded((mi_col, mi_row), side_mi, &snapshot.cdef_coded);
        let slot = self.cdef_slot_index(&block);
        self.grids.cdef_slots[slot] = snapshot.cdef_slot;
        self.segments
            .restore_square((block.x, block.y), block.width(), &snapshot.segment_ids);
        self.delta_q.clone_from(&snapshot.delta_q);
        self.writer.restore_after(snapshot.mark, &snapshot.symbols);
        self.current = snapshot.current;
        self.luma_copy = snapshot.luma_copy;
        self.drop_block_caches();
    }

    /// Forget the block caches keyed by position: after an undo the block
    /// at a position may be coded from other neighbours.
    fn drop_block_caches(&mut self) {
        self.scratch.cached_luma = None;
        self.scratch.cached_intra = None;
        self.scratch.tree_trial = None;
    }

    /// `CodecBackend::begin_node_trial`: note `node`'s coded state.
    pub(super) fn open_node_trial(&mut self, node: &Block, recon: &Picture) {
        let mut trials = std::mem::take(&mut self.node_trials);
        if trials.open == trials.snapshots.len() {
            trials.snapshots.push(NodeTrial::default());
        }
        let trial = &mut trials.snapshots[trials.open];
        trial.full = true;
        trial.kept_cdfs_current = false;
        if trials.full_open == 0 {
            self.writer.start_journal(&self.cdfs);
        }
        trial.cdf_mark = self.writer.journal_mut().mark();
        let before = &mut trial.before;
        self.save_node(node, recon, before);
        self.writer.start_logging();
        before.mark = self.writer.mark();
        before.symbols.clear();
        trials.open += 1;
        trials.full_open += 1;
        self.node_trials = trials;
        self.drop_block_caches();
    }

    /// `CodecBackend::begin_node_pricing`: note only what the price needs
    /// (the node and the symbol mark): a priced-only trial is never undone
    /// or kept, so it copies no square and marks no CDF journal.
    pub(super) fn open_node_pricing(&mut self, node: &Block) {
        let mut trials = std::mem::take(&mut self.node_trials);
        if trials.open == trials.snapshots.len() {
            trials.snapshots.push(NodeTrial::default());
        }
        let trial = &mut trials.snapshots[trials.open];
        trial.full = false;
        let before = &mut trial.before;
        before.block = Some(*node);
        self.writer.start_logging();
        before.mark = self.writer.mark();
        trials.open += 1;
        self.node_trials = trials;
    }

    /// The innermost open trial.
    fn innermost_node_trial(&self) -> &NodeTrial {
        let trials = &self.node_trials;
        &trials.snapshots[trials.open.checked_sub(1).expect("a node trial is open")]
    }

    /// `CodecBackend::node_trial_cost`: the price of the innermost trial's
    /// node as coded since it began.
    pub(super) fn node_trial_price(&self, source: &Picture, recon: &Picture) -> f32 {
        let before = &self.innermost_node_trial().before;
        let block = before.block();
        let [luma, cb, cr] = PlaneIndex::ALL.map(|plane| {
            nc_encoder::block_sse(
                &self.dsp,
                (source, recon),
                plane,
                plane_square(plane, &block),
            ) as f32
        });
        let step_ratio = self.steps.ac as f32 / self.chroma_steps.ac as f32;
        let chroma_weight = step_ratio * step_ratio;
        let bits = self.writer.bits_since(before.mark);
        luma + chroma_weight * (cb + cr) + self.lambda() * bits
    }

    /// `CodecBackend::undo_node_trial`: put the innermost trial's state
    /// back.
    pub(super) fn undo_node(&mut self, recon: &mut Picture) {
        let mut trials = std::mem::take(&mut self.node_trials);
        let trial = &mut trials.snapshots[trials.open - 1];
        debug_assert!(trial.full, "a priced-only trial is never undone");
        let journal = self.writer.journal_mut();
        if std::mem::take(&mut trial.kept_cdfs_current) {
            journal.undo_keeping(trial.cdf_mark, &mut *self.cdfs, &mut trial.kept_cdfs);
        } else {
            journal.undo(trial.cdf_mark, &mut *self.cdfs);
        }
        self.restore_node(recon, &trial.before);
        self.node_trials = trials;
    }

    /// `CodecBackend::keep_node_candidate`: note the innermost trial's
    /// coding as its best.
    pub(super) fn keep_node(&mut self, recon: &Picture) {
        let mut trials = std::mem::take(&mut self.node_trials);
        let NodeTrial {
            before,
            kept,
            kept_cdfs_current,
            ..
        } = &mut trials.snapshots[trials.open - 1];
        self.save_node(&before.block(), recon, kept);
        *kept_cdfs_current = true;
        kept.mark = before.mark;
        self.writer.save_after(before.mark, &mut kept.symbols);
        self.node_trials = trials;
    }

    /// `CodecBackend::restore_node_candidate`: put the innermost trial's
    /// kept coding back.
    pub(super) fn restore_kept_node(&mut self, recon: &mut Picture) {
        let mut trials = std::mem::take(&mut self.node_trials);
        let trial = &mut trials.snapshots[trials.open - 1];
        if !std::mem::take(&mut trial.kept_cdfs_current) {
            // The backend's CDFs are a lost coding's: undo it, redo the
            // kept one.
            let journal = self.writer.journal_mut();
            journal.undo(trial.cdf_mark, &mut *self.cdfs);
            journal.redo(&mut trial.kept_cdfs, &mut *self.cdfs);
        }
        self.restore_node(recon, &trial.kept);
        self.node_trials = trials;
    }

    /// `CodecBackend::end_node_trial`: close the innermost trial.
    pub(super) fn close_node_trial(&mut self) {
        let trials = &mut self.node_trials;
        trials.open = trials.open.checked_sub(1).expect("a node trial is open");
        if trials.snapshots[trials.open].full {
            trials.full_open -= 1;
            if trials.full_open == 0 {
                self.writer.stop_journal();
            }
        }
    }

    /// `CodecBackend::last_cu_skipped`: the block last coded is a skipped
    /// inter block (`skip` with a signalled vector, AV1 §5.11.11).
    pub(super) fn last_block_skipped(&self) -> bool {
        self.last_coded.is_some_and(|(row, col)| {
            let info = self.grids.mode_info.at(row, col);
            info.skip && info.is_inter()
        })
    }
}
