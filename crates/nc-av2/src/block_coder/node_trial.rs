//! Partition trials for the engine's RD partition search (`nc_encoder`
//! `rd_split.rs`, `CodecBackend::begin_node_trial`): a node's partition
//! symbols and blocks are coded by the real coder, priced over the node,
//! and can be undone, so the engine can compare a leaf and a quad split by
//! what they cost to code. The design is AV1's (`nc-av1`
//! `backend/node_trial.rs`) over AV2's own writer and tile state.
//!
//! AV2 codes symbols as it goes, adapting its CDFs, so a snapshot holds the
//! coder's state as well as the picture's:
//! - **Symbols.** From a frame's first node trial on, the writer records
//!   its operations ([`Writer::record`], the log frames with deferred CDEF
//!   or CCSO symbols keep from their start), coded after the symbols coded
//!   before when the tile finishes. A trial's symbols are the run of the log
//!   after a mark: undone by truncation, kept as a copy of the run, priced
//!   as estimates price them ([`Writer::bits_since`]).
//! - **CDFs.** The tile's models are copied into a buffer kept per trial
//!   level when the trial begins. AV2 tries at most a leaf and a quad
//!   split. At the only undo, that buffer becomes live and the first
//!   coding's models take its place. A second swap restores the first
//!   coding's models if it wins.
//! - **The square** (everything later blocks and the loop filters read):
//!   its reconstruction in all planes; mode information, coefficient
//!   contexts, motion cells, the MV bank (its updates follow the coding
//!   order, AVM `update_ref_mv_bank`) and the deblocking records
//!   ([`crate::block_state::BlockState::save_square`]); partition contexts and the halves still
//!   to split ([`crate::partition::PartitionWriter::save_square`]); the superblock quantizer
//!   state and the block's qindex (AVM `read_delta_q_params`); the filter
//!   block's CDEF index slot (AVM `read_cdef`) and the deferred CDEF and
//!   CCSO symbols. Caches keyed by position (the engine's luma prediction,
//!   the final intra screen, the ranked intra luma) are dropped, so every
//!   coding after an undo predicts afresh.
//!
//! A trial that is only priced (a node with one candidate,
//! `CodecBackend::begin_node_pricing`) notes only its node and marks.
//!
//! The price is the squared error of the three planes over the node, chroma
//! weighted as the block decisions weigh it ([`super::CoderOptions::chroma_weight`]),
//! plus the frame's λ times the bits.

use super::{BlockCoder, DeferredSymbol};
use crate::block_state::SavedSquare;
use crate::cdf_models::TileCdfs;
use crate::delta_q::SuperblockQindex;
use crate::entropy::{SymbolRun, Writer};
use crate::partition::SavedPartitions;
use nc_common::{Picture, PlaneIndex};
use nc_encoder::trial_pixels::TrialPixels;

/// A square node: top-left luma sample and side.
type Node = ((usize, usize), usize);

/// The coded state of one node (buffers reused trial to trial).
#[derive(Debug, Default)]
struct Snapshot {
    node: Option<Node>,
    pixels: TrialPixels,
    square: SavedSquare,
    partitions: SavedPartitions,
    superblock_qindex: SuperblockQindex,
    block_qindex: u8,
    cdef_index: Option<u8>,
    /// The deferred symbols' count at the trial's start, and those listed
    /// after it (a kept coding's).
    deferred_mark: usize,
    deferred: Vec<DeferredSymbol>,
    /// The symbol log's length at the trial's start, and the operations
    /// after it (a kept coding's).
    mark: usize,
    symbols: SymbolRun,
}

impl Snapshot {
    fn node(&self) -> Node {
        self.node.expect("a snapshot was taken")
    }

    /// The deblocking mark the trial's square was saved against.
    fn edge_mark(&self) -> usize {
        self.square.edge_mark()
    }
}

/// One node trial: the state before it and its kept coding.
#[derive(Debug, Default)]
struct NodeTrial {
    before: Snapshot,
    kept: Snapshot,
    /// The models before the trial, then the first candidate's models
    /// after swapping the original models back for the second candidate.
    other_cdfs: Option<Box<TileCdfs>>,
    /// Whether the trial was undone once: the swap restores the starting
    /// models only at the first undo (module docs), so a second is refused.
    undone: bool,
}

/// The open node trials, innermost last.
#[derive(Debug, Default)]
pub(super) struct NodeTrials {
    /// Trials; the first `open` are in use, the rest are spare buffers.
    snapshots: Vec<NodeTrial>,
    open: usize,
}

impl NodeTrials {
    /// A free trial for the next level, opened.
    fn push(&mut self) -> &mut NodeTrial {
        if self.open == self.snapshots.len() {
            self.snapshots.push(NodeTrial::default());
        }
        self.open += 1;
        &mut self.snapshots[self.open - 1]
    }

    /// The innermost open trial.
    fn innermost(&mut self) -> &mut NodeTrial {
        let open = self.open.checked_sub(1).expect("a node trial is open");
        &mut self.snapshots[open]
    }
}

/// `(x, y, side)` of square `node` in `plane` (4:2:0).
fn plane_square(plane: PlaneIndex, ((x, y), size): Node) -> (usize, usize, usize) {
    let shift = usize::from(plane != PlaneIndex::Y);
    (x >> shift, y >> shift, size >> shift)
}

/// Mark the start of a trial in `writer`'s log (recording from now on).
fn start_mark(writer: &mut Writer) -> usize {
    writer.record();
    writer.mark()
}

impl BlockCoder {
    /// Note the coded state of `node` in `snapshot` (the marks aside).
    fn save_node(&self, node: Node, recon: &Picture, snapshot: &mut Snapshot, edge_mark: usize) {
        let ((x, y), size) = node;
        snapshot
            .pixels
            .save(&self.quantizer.dsp, recon, (x, y, size, size));
        self.state
            .save_square((x, y), size, edge_mark, &mut snapshot.square);
        self.partitions
            .save_square((x, y), size, &mut snapshot.partitions);
        snapshot.superblock_qindex = self.superblock_qindex;
        snapshot.block_qindex = self.block_qindex;
        snapshot.cdef_index = self.cdef_indices.get(x, y);
        snapshot.node = Some(node);
    }

    /// Put back the coded state `snapshot` noted.
    fn restore_node(&mut self, recon: &mut Picture, snapshot: &Snapshot) {
        let ((x, y), size) = snapshot.node();
        snapshot
            .pixels
            .restore(&self.quantizer.dsp, recon, (x, y, size, size));
        self.state.restore_square((x, y), size, &snapshot.square);
        self.partitions
            .restore_square((x, y), size, &snapshot.partitions);
        self.superblock_qindex = snapshot.superblock_qindex;
        self.quantize_at(snapshot.block_qindex);
        self.cdef_indices.set(x, y, snapshot.cdef_index);
        self.deferred.truncate(snapshot.deferred_mark);
        self.deferred.extend_from_slice(&snapshot.deferred);
        self.writer.restore_after(snapshot.mark, &snapshot.symbols);
        self.drop_block_caches();
    }

    /// Forget the caches keyed by block position: after an undo the block
    /// at a position may be coded from other neighbours.
    fn drop_block_caches(&mut self) {
        self.cached_luma = None;
        self.final_intra_block = None;
        self.ranked_luma.winner = None;
    }

    /// `CodecBackend::begin_node_trial`: note `node`'s coded state.
    pub fn begin_node_trial(&mut self, node: Node, recon: &Picture) {
        let mut trials = std::mem::take(&mut self.node_trials);
        let trial = trials.push();
        trial.other_cdfs.clone_from(self.writer.models_box());
        trial.undone = false;
        let before = &mut trial.before;
        let edge_mark = self.state.edges.mark();
        self.save_node(node, recon, before, edge_mark);
        before.deferred_mark = self.deferred.len();
        before.deferred.clear();
        before.mark = start_mark(&mut self.writer);
        before.symbols = SymbolRun::default();
        self.node_trials = trials;
        self.drop_block_caches();
    }

    /// `CodecBackend::begin_node_pricing`: note only what the price needs
    /// (the node and the symbol mark); a priced-only trial is never undone
    /// or kept.
    pub fn begin_node_pricing(&mut self, node: Node) {
        let mark = start_mark(&mut self.writer);
        let before = &mut self.node_trials.push().before;
        before.node = Some(node);
        before.mark = mark;
    }

    /// `CodecBackend::node_trial_cost`: the price of the innermost trial's
    /// node as coded since it began.
    pub fn node_trial_cost(&mut self, source: &Picture, recon: &Picture) -> f32 {
        let (distortion, bits, lambda) = self.node_trial_terms(source, recon);
        (distortion + lambda * bits) as f32
    }

    /// `CodecBackend::node_trial_parts` (the cost is summed in f64 and
    /// rounded once, so its parts are rounded separately here).
    pub fn node_trial_parts(
        &mut self,
        source: &Picture,
        recon: &Picture,
    ) -> nc_encoder::NodeTrialParts {
        let (distortion, bits, lambda) = self.node_trial_terms(source, recon);
        nc_encoder::NodeTrialParts {
            distortion: distortion as f32,
            bits: bits as f32,
            lambda: lambda as f32,
        }
    }

    /// The innermost trial's weighted distortion, bits and λ.
    fn node_trial_terms(&mut self, source: &Picture, recon: &Picture) -> (f64, f64, f64) {
        let before = &self.node_trials.innermost().before;
        let (node, mark) = (before.node(), before.mark);
        let [luma, cb, cr] = PlaneIndex::ALL.map(|plane| {
            nc_encoder::block_sse(
                &self.quantizer.dsp,
                (source, recon),
                plane,
                plane_square(plane, node),
            ) as f64
        });
        let bits = f64::from(self.writer.bits_since(mark));
        (
            luma + self.chroma_weight * (cb + cr),
            bits,
            self.lambda_at(self.qindex),
        )
    }

    /// `CodecBackend::undo_node_trial`: put the innermost trial's state
    /// back.
    pub fn undo_node_trial(&mut self, recon: &mut Picture) {
        let mut trials = std::mem::take(&mut self.node_trials);
        let trial = trials.innermost();
        debug_assert!(
            !std::mem::replace(&mut trial.undone, true),
            "one undo per AV2 node trial: a second would restore the first candidate's CDFs"
        );
        std::mem::swap(self.writer.models_box(), &mut trial.other_cdfs);
        self.restore_node(recon, &trial.before);
        self.node_trials = trials;
    }

    /// `CodecBackend::keep_node_candidate`: note the innermost trial's
    /// coding as its best.
    pub fn keep_node_candidate(&mut self, recon: &Picture) {
        let mut trials = std::mem::take(&mut self.node_trials);
        let NodeTrial { before, kept, .. } = trials.innermost();
        self.save_node(before.node(), recon, kept, before.edge_mark());
        kept.deferred_mark = before.deferred_mark;
        kept.deferred.clear();
        kept.deferred
            .extend_from_slice(&self.deferred[before.deferred_mark..]);
        kept.mark = before.mark;
        self.writer.save_after(before.mark, &mut kept.symbols);
        self.node_trials = trials;
    }

    /// `CodecBackend::restore_node_candidate`: put the innermost trial's
    /// kept coding back.
    pub fn restore_node_candidate(&mut self, recon: &mut Picture) {
        let mut trials = std::mem::take(&mut self.node_trials);
        let trial = trials.innermost();
        std::mem::swap(self.writer.models_box(), &mut trial.other_cdfs);
        self.restore_node(recon, &trial.kept);
        self.node_trials = trials;
    }

    /// `CodecBackend::end_node_trial`: close the innermost trial.
    pub fn end_node_trial(&mut self) {
        self.node_trials.open = self
            .node_trials
            .open
            .checked_sub(1)
            .expect("a node trial is open");
    }

    /// `CodecBackend::last_cu_skipped`: the block last coded is an inter
    /// block without residual (`skip_txfm`, AVM `read_skip_txfm`).
    pub fn last_block_skipped(&self) -> bool {
        self.last_inter_skipped
    }
}
