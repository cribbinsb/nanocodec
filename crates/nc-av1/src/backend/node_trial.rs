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
//! - **The node** (a square, or a half of one). Its reconstruction in all planes, mode-info cells (and
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
//! The trial's summary (`CodecBackend::node_trial_summary`: coded transform
//! blocks and their non-zero levels) is counted as the coefficients are
//! written, since AV1 keeps no syntax record to read it back from: each trial
//! notes its parent's count and starts from nothing, an undo starts it again,
//! a kept coding notes its count with it, and the trial's end adds its count
//! to the parent's.
//!
//! The price is the squared error of the three planes over the node
//! (chroma weighted by the squared luma-to-chroma step ratio, the inverse
//! of the chroma λ scaling of the trellis) plus λ times the bits.

use nc_common::{Picture, PlaneIndex};
use nc_encoder::trial_pixels::TrialPixels;
use nc_encoder::undo_journal::KeptRun;
use nc_encoder::{Block, CodecBackend, NodeTrialSummary};

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
    pixels: TrialPixels,
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
    /// Before a trial: its parent's summary so far; kept: the kept coding's.
    summary: NodeTrialSummary,
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
    /// What the coding since the innermost trial began coded.
    coded: NodeTrialSummary,
}

impl NodeTrials {
    /// Note a coded transform block of `nonzero()` levels (`luma_log2`: its
    /// longer side's log2 when it is a luma block). Counted only while a
    /// trial is open: the engine reads a summary only inside one (after its
    /// `node_trial_cost`), so outside trials the count is skipped, a pass
    /// over each coded block's levels (AV1 Veryfast codes no trials).
    pub(super) fn note_coded_block(
        &mut self,
        luma_log2: Option<u8>,
        nonzero: impl FnOnce() -> usize,
    ) {
        if self.open > 0 {
            self.coded.add_block(luma_log2, nonzero());
        }
    }

    /// Start a trial's count, noting its parent's in `before`.
    fn begin_summary(&mut self, before: usize) {
        self.snapshots[before].before.summary = std::mem::take(&mut self.coded);
    }
}

/// `(x, y, width, height)` of `block` in `plane` (4:2:0).
fn plane_rect(plane: PlaneIndex, block: &Block) -> (usize, usize, usize, usize) {
    let shift = usize::from(plane != PlaneIndex::Y);
    (
        block.x >> shift,
        block.y >> shift,
        block.width() >> shift,
        block.height() >> shift,
    )
}

impl Av1Backend {
    /// The index of the 64x64 CDEF filter block holding `block`.
    fn cdef_slot_index(&self, block: &Block) -> usize {
        block.y / 64 * self.grids.width.div_ceil(64) + block.x / 64
    }

    /// Note the coded state of `block` in `snapshot` (the symbol mark
    /// aside).
    fn save_node(&self, block: &Block, recon: &Picture, snapshot: &mut Snapshot) {
        snapshot.pixels.save(
            &self.dsp,
            recon,
            (block.x, block.y, block.width(), block.height()),
        );
        let (mi_col, mi_row) = (block.x >> MI_LOG2, block.y >> MI_LOG2);
        let size_mi = (block.width() >> MI_LOG2, block.height() >> MI_LOG2);
        self.grids
            .mode_info
            .save_rect((mi_row, mi_col), size_mi, &mut snapshot.cells);
        for (plane, saved) in snapshot.coeff_contexts.iter_mut().enumerate() {
            let (above, left) = self.context_slices(plane, Self::plane_units(block, plane));
            saved.clear();
            saved.extend_from_slice(above);
            saved.extend_from_slice(left);
        }
        self.grids
            .txfm
            .save((mi_col, mi_row), size_mi, &mut snapshot.txfm_contexts);
        self.grids
            .cdef_map
            .save_coded((mi_col, mi_row), size_mi.1, &mut snapshot.cdef_coded);
        snapshot.cdef_slot = self.grids.cdef_slots[self.cdef_slot_index(block)];
        self.segments.save_rect(block, &mut snapshot.segment_ids);
        snapshot.delta_q.clone_from(&self.delta_q);
        snapshot.current = self.current;
        snapshot.luma_copy = self.luma_copy;
        snapshot.block = Some(*block);
    }

    /// Put back the coded state `snapshot` noted.
    fn restore_node(&mut self, recon: &mut Picture, snapshot: &Snapshot) {
        let block = snapshot.block();
        snapshot.pixels.restore(
            &self.dsp,
            recon,
            (block.x, block.y, block.width(), block.height()),
        );
        let (mi_col, mi_row) = (block.x >> MI_LOG2, block.y >> MI_LOG2);
        let size_mi = (block.width() >> MI_LOG2, block.height() >> MI_LOG2);
        self.grids
            .mode_info
            .restore_rect((mi_row, mi_col), size_mi, &snapshot.cells);
        for (plane, saved) in snapshot.coeff_contexts.iter().enumerate() {
            let span = Self::plane_units(&block, plane);
            let (above, left) = saved.split_at(span.w4);
            self.grids.contexts.above[plane][span.x4..span.x4 + span.w4].copy_from_slice(above);
            self.grids.contexts.left[plane][span.y4..span.y4 + span.h4].copy_from_slice(left);
        }
        self.grids
            .txfm
            .restore((mi_col, mi_row), size_mi, &snapshot.txfm_contexts);
        self.grids
            .cdef_map
            .restore_coded((mi_col, mi_row), size_mi.1, &snapshot.cdef_coded);
        let slot = self.cdef_slot_index(&block);
        self.grids.cdef_slots[slot] = snapshot.cdef_slot;
        self.segments.restore_rect(&block, &snapshot.segment_ids);
        self.delta_q.clone_from(&snapshot.delta_q);
        self.writer.restore_after(snapshot.mark, &snapshot.symbols);
        self.current = snapshot.current;
        self.luma_copy = snapshot.luma_copy;
        self.drop_block_caches();
    }

    /// Forget the block caches keyed by position: after an undo the block
    /// at a position may be coded from other neighbours. That includes the
    /// key-frame luma ranking, which `ranked_key_luma` would otherwise reuse
    /// for a later block at the same position coded without the leaf search.
    fn drop_block_caches(&mut self) {
        self.scratch.cached_luma = None;
        self.scratch.cached_intra = None;
        self.scratch.tree_trial = None;
        self.scratch.key_luma_ranking = None;
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
        trials.begin_summary(trials.open);
        trials.open += 1;
        trials.full_open += 1;
        self.node_trials = trials;
        self.drop_block_caches();
    }

    /// `CodecBackend::begin_node_pricing`: note only what the price needs
    /// (the node and the symbol mark): a priced-only trial is never undone
    /// or kept, so it copies no block and marks no CDF journal.
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
        trials.begin_summary(trials.open);
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
        self.node_trial_split_price(source, recon).cost()
    }

    /// `CodecBackend::node_trial_parts`.
    pub(super) fn node_trial_split_price(
        &self,
        source: &Picture,
        recon: &Picture,
    ) -> nc_encoder::NodeTrialParts {
        let before = &self.innermost_node_trial().before;
        let block = before.block();
        let [luma, cb, cr] = PlaneIndex::ALL.map(|plane| {
            nc_encoder::block_sse_rect(&self.dsp, (source, recon), plane, plane_rect(plane, &block))
                as f32
        });
        let step_ratio = self.steps.ac as f32 / self.chroma_steps.ac as f32;
        let chroma_weight = step_ratio * step_ratio;
        let bits = self.writer.bits_since(before.mark);
        nc_encoder::NodeTrialParts {
            distortion: luma + chroma_weight * (cb + cr),
            bits,
            lambda: self.lambda(),
        }
    }

    /// `CodecBackend::node_trial_summary`.
    pub(super) fn node_trial_coded_summary(&self) -> NodeTrialSummary {
        self.node_trials.coded
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
        trials.coded = NodeTrialSummary::default();
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
        kept.summary = trials.coded;
        self.writer.save_after(before.mark, &mut kept.symbols);
        self.node_trials = trials;
    }

    /// `CodecBackend::restore_node_candidate`: put the innermost trial's
    /// kept coding back.
    pub(super) fn restore_kept_node(&mut self, recon: &mut Picture) {
        let mut trials = std::mem::take(&mut self.node_trials);
        let trial = &mut trials.snapshots[trials.open - 1];
        if !std::mem::replace(&mut trial.kept_cdfs_current, true) {
            // The backend's CDFs are a lost coding's: undo it, redo the
            // kept one, which an undo after this sets aside again.
            let journal = self.writer.journal_mut();
            journal.undo(trial.cdf_mark, &mut *self.cdfs);
            journal.redo(&mut trial.kept_cdfs, &mut *self.cdfs);
        }
        self.restore_node(recon, &trial.kept);
        trials.coded = trial.kept.summary;
        self.node_trials = trials;
    }

    /// `CodecBackend::end_node_trial`: close the innermost trial.
    pub(super) fn close_node_trial(&mut self) {
        let trials = &mut self.node_trials;
        trials.open = trials.open.checked_sub(1).expect("a node trial is open");
        trials.coded = trials.snapshots[trials.open]
            .before
            .summary
            .merged(trials.coded);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::SUPERBLOCK_LOG2;
    use nc_encoder::{BlockDecision, FramePlan, FrameType, ParamOverrides, PredictionMode};

    const SIDE: usize = 64;

    /// Code `node` as a leaf, or split `depth` times, with DC blocks.
    fn code_node(
        backend: &mut Av1Backend,
        node: Block,
        depth: u8,
        source: &Picture,
        recon: &mut Picture,
    ) {
        backend.write_partition(&node, depth > 0);
        if depth == 0 {
            let decision = BlockDecision::new(PredictionMode::IntraDc);
            backend.encode_block(&node, &decision, source, None, recon);
            return;
        }
        let half = node.size() / 2;
        for (dx, dy) in [(0, 0), (half, 0), (0, half), (half, half)] {
            let child = Block::square(node.x + dx, node.y + dy, node.log2_width - 1);
            code_node(backend, child, depth - 1, source, recon);
        }
    }

    /// `text` without the CDFs' symbol-cost caches, which pricing fills
    /// lazily between codings (a cache of the probabilities, not state).
    fn without_cost_caches(text: &str) -> String {
        const CACHE: &str = "costs: SymbolCosts {";
        let mut rest = text;
        let mut kept = String::with_capacity(text.len());
        while let Some(start) = rest.find(CACHE) {
            kept.push_str(&rest[..start]);
            let end = rest[start..].find('}').expect("a cache's closing brace");
            rest = &rest[start + end + 1..];
        }
        kept.push_str(rest);
        kept
    }

    /// Everything a node trial restores, as text, with the picture.
    fn state(backend: &Av1Backend, recon: &Picture) -> (String, Vec<u8>) {
        let mut samples = Vec::new();
        nc_common::yuv::write_frame(&mut samples, recon).unwrap();
        let coded = format!(
            "{:?}",
            (
                &*backend.cdfs,
                backend.writer.mark(),
                &backend.grids,
                &backend.segments,
                &backend.delta_q,
            )
        );
        (without_cost_caches(&coded), samples)
    }

    /// A node trial can be coded and undone any number of times, each undo
    /// putting back the state it began with, and its kept coding, not the
    /// last, comes back whether or not the coding after it was undone.
    #[test]
    fn node_trials_undo_every_coding_and_restore_a_kept_one() {
        /// The split depth of the coding kept.
        const KEPT: u8 = 1;
        let mut backend = Av1Backend::new(&nc_encoder::EncoderConfig {
            codec: nc_encoder::Codec::Av1,
            width: SIDE,
            height: SIDE,
            frame_rate: nc_encoder::FrameRate { num: 30, den: 1 },
            quantizer: 120,
            intra_period: 300,
            block_log2: SUPERBLOCK_LOG2,
            key_qp_offset: None,
            temporal_layers: 1,
            signal: nc_encoder::StreamSignal::default(),
            params: ParamOverrides::default(),
        })
        .unwrap();
        let mut source = Picture::new(SIDE, SIDE, 0).unwrap();
        for plane in PlaneIndex::ALL {
            let mut view = source.plane_mut(plane).view_mut();
            for y in 0..view.height() {
                for (x, sample) in view.row_mut(y).iter_mut().enumerate() {
                    *sample = ((x * 23 + y * 17 + plane as usize * 47) ^ (x * y / 7)) as u8;
                }
            }
        }
        backend.begin_frame(&FramePlan {
            index: 0,
            frame_type: FrameType::Key,
            quantizer: 120,
            references: nc_encoder::FrameReferences::FLAT,
            lambda_scale: 1.0,
            block_qp: false,
        });
        backend.begin_superblock(0, 0);
        let node = Block::square(0, 0, SUPERBLOCK_LOG2);
        let mut recon = Picture::new(SIDE, SIDE, 0).unwrap();
        backend.begin_node_trial(&node, &recon);
        let start = state(&backend, &recon);
        let mut kept = None;
        for depth in [0, KEPT, 2] {
            code_node(&mut backend, node, depth, &source, &mut recon);
            let coded = state(&backend, &recon);
            assert_ne!(coded, start, "depth {depth} coded nothing");
            if depth == KEPT {
                backend.keep_node_candidate(&recon);
                kept = Some(coded);
            }
            backend.undo_node_trial(&mut recon);
            assert!(state(&backend, &recon) == start, "undo {depth}");
        }
        let kept = kept.unwrap();
        backend.restore_node_candidate(&mut recon);
        assert!(state(&backend, &recon) == kept, "restored");
        backend.undo_node_trial(&mut recon);
        code_node(&mut backend, node, 0, &source, &mut recon);
        backend.restore_node_candidate(&mut recon);
        assert!(state(&backend, &recon) == kept, "over a coding");
        backend.end_node_trial();
    }
}
