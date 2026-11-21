//! Partition trials for the engine's RD partition search (`nc_encoder`
//! `rd_split.rs`, `CodecBackend::begin_node_trial`): a node's partition
//! symbols and blocks are coded by the real coder, priced over the node, and
//! can be undone, so the engine can compare a leaf and a quad split by what
//! they cost to code. HEVC's and VVC's `backend/node_trial.rs` are the same
//! stack over their own coded state.
//!
//! A snapshot notes everything coding the node's blocks changes and a later
//! stage reads:
//! - the reconstruction of the node in all three planes;
//! - the mode-info cells of the node (the loop filter, the segment map,
//!   motion-vector prediction and the skip / reference / filter contexts of
//!   later blocks read them);
//! - the above / left entropy contexts (coefficient nonzero flags and
//!   partition contexts, VP9 §7.4.7 / §9.3);
//! - the frame's symbol counts, which choose the forward coefficient updates
//!   at `end_frame` and drive backward adaptation at `commit_frame`
//!   (VP9 §8.4.2);
//! - the tile recording (truncated to a mark; a kept coding also keeps the
//!   bits it recorded) and its segment-id statistics, which give the frame's
//!   segment-tree probabilities;
//! - the block quantizers' segment assignments (`SegmentQuantizers`), which
//!   fill the header's segment features at `end_frame`.
//!
//! Per-block caches (the reference vectors of the block being decided) are
//! dropped, as an undone coding leaves other neighbours. The frame's
//! segmentation, header, probabilities and the last frame's segment map are
//! only read while a frame is coded.
//!
//! Trials nest: a quad split's children open their own while the parent's
//! stays open, so the snapshots form a stack (buffers kept for reuse when a
//! trial ends). Each trial also keeps its best coding so far (a snapshot
//! taken after it), put back when a later candidate lost.
//! The entropy contexts and counts are copied once. Undo exchanges that
//! copy with the first candidate's state so it is available if the second loses.
//!
//! The price of what was coded since a snapshot is the squared error of the
//! three planes over the node (chroma weighted as the backend's residual RD
//! weighs it, [`Vp9Backend::distortion_weight`]) plus λ times the estimated
//! bits recorded since (`TileRecorder::estimated_bits_since`).

use nc_common::{Picture, PlaneIndex};
use nc_encoder::segment_quantizers::SegmentQuantizers;
use nc_encoder::{Block, CodecBackend};

use super::{MI_LOG2, TrialPlanes, Vp9Backend, copy_block_in, copy_block_out};
use crate::contexts::EntropyContexts;
use crate::probs::Counts;
use crate::recorder::{RecordedRun, RecorderMark};
use crate::types::ModeInfo;

/// The coded state of one node (buffers reused snapshot to snapshot).
#[derive(Debug, Default)]
struct CodedSnapshot {
    node: Option<Block>,
    /// The node's samples, plane after plane.
    planes: Vec<u8>,
    cells: Vec<ModeInfo>,
    recorder: RecorderMark,
    /// What was recorded after the trial's mark (a snapshot taken after a
    /// trial only).
    added: RecordedRun,
    segments: Option<SegmentQuantizers>,
}

impl CodedSnapshot {
    fn node(&self) -> Block {
        self.node.expect("a snapshot was taken")
    }
}

/// One node trial: the state before it and its kept coding.
#[derive(Debug, Default)]
struct NodeTrial {
    before: CodedSnapshot,
    kept: CodedSnapshot,
    // The starting state until undo, then the first candidate's state.
    other_contexts: Option<EntropyContexts>,
    other_counts: Option<Box<Counts>>,
}

impl NodeTrial {
    fn swap_entropy(&mut self, backend: &mut Vp9Backend) {
        std::mem::swap(
            &mut backend.grids.contexts,
            self.other_contexts.as_mut().expect("a snapshot was taken"),
        );
        std::mem::swap(
            &mut backend.counts,
            self.other_counts.as_mut().expect("a snapshot was taken"),
        );
    }
}

/// The open node trials, innermost last.
#[derive(Debug, Default)]
pub(super) struct NodeTrials {
    /// Trials; the first `open` are in use, the rest are spare buffers.
    trials: Vec<NodeTrial>,
    open: usize,
}

impl NodeTrials {
    /// Open a trial, reusing a spare buffer.
    fn push(&mut self) -> &mut NodeTrial {
        if self.open == self.trials.len() {
            self.trials.push(NodeTrial::default());
        }
        self.open += 1;
        &mut self.trials[self.open - 1]
    }

    fn innermost(&self) -> &NodeTrial {
        &self.trials[self.open.checked_sub(1).expect("a node trial is open")]
    }
}

/// The node's 8x8 position and side in 8x8 units.
fn units_of(node: &Block) -> ((usize, usize), usize) {
    let shift = usize::from(MI_LOG2);
    ((node.y >> shift, node.x >> shift), node.size() >> shift)
}

impl Vp9Backend {
    /// Note the coded state of `node` (inside the picture) in `snapshot`,
    /// all but the recorder mark.
    fn save_state(&self, node: &Block, recon: &Picture, snapshot: &mut CodedSnapshot) {
        let samples = node.size() * node.size() * 3 / 2;
        snapshot.planes.resize(samples, 0);
        copy_block_out(
            &self.dsp,
            node,
            recon,
            &mut snapshot.planes,
            TrialPlanes::All,
        );
        let (position, units) = units_of(node);
        snapshot.cells.clear();
        self.grids
            .grid
            .save_square(position, units, &mut snapshot.cells);
        snapshot.segments.clone_from(&self.block_qp);
        snapshot.node = Some(*node);
    }

    /// Put back the coded state `snapshot` noted, the recording included.
    fn restore_state(&mut self, recon: &mut Picture, snapshot: &CodedSnapshot) {
        let node = snapshot.node();
        copy_block_in(&self.dsp, &node, &snapshot.planes, recon, TrialPlanes::All);
        let (position, units) = units_of(&node);
        self.grids
            .grid
            .restore_square(position, units, &snapshot.cells);
        self.block_qp.clone_from(&snapshot.segments);
        self.writer
            .restore_after(snapshot.recorder, &snapshot.added);
        self.refs = None;
    }

    /// `CodecBackend::begin_node_trial`: note `node`'s coded state.
    pub(super) fn open_node_trial(&mut self, node: &Block, recon: &Picture) {
        let mut trials = std::mem::take(&mut self.node_trials);
        let trial = trials.push();
        self.save_state(node, recon, &mut trial.before);
        match &mut trial.other_contexts {
            Some(contexts) => contexts.copy_from(&self.grids.contexts),
            None => trial.other_contexts = Some(self.grids.contexts.clone()),
        }
        trial
            .other_counts
            .get_or_insert_with(Box::default)
            .copy_from(&self.counts);
        trial.before.recorder = self.writer.mark();
        // Nothing is recorded after the mark yet: undoing truncates to it.
        self.writer
            .save_after(trial.before.recorder, &mut trial.before.added);
        self.node_trials = trials;
        // The cached reference vectors belong to the block being decided.
        self.refs = None;
    }

    /// `CodecBackend::begin_node_pricing`: note only the node and the
    /// recording (a priced-only trial is never undone or kept).
    pub(super) fn open_node_pricing(&mut self, node: &Block) {
        let mark = self.writer.mark();
        let trial = self.node_trials.push();
        trial.before.node = Some(*node);
        trial.before.recorder = mark;
    }

    /// `CodecBackend::node_trial_cost`: the price of the innermost trial's
    /// node as coded since it began.
    pub(super) fn node_trial_price(&self, source: &Picture, recon: &Picture) -> f32 {
        let before = &self.node_trials.innermost().before;
        let node = before.node();
        let distortion: f32 = PlaneIndex::ALL
            .into_iter()
            .map(|plane| {
                let ss = usize::from(plane != PlaneIndex::Y);
                let rect = (node.x >> ss, node.y >> ss, node.size() >> ss);
                let sse = nc_encoder::block_sse(&self.dsp, (source, recon), plane, rect);
                self.distortion_weight(plane) * sse as f32
            })
            .sum();
        // Outside `encode_block` λ is the frame quantizer's.
        distortion + self.lambda() * self.writer.estimated_bits_since(before.recorder)
    }

    /// `CodecBackend::undo_node_trial`: put the innermost trial's state
    /// back.
    pub(super) fn undo_node(&mut self, recon: &mut Picture) {
        let mut trials = std::mem::take(&mut self.node_trials);
        let trial = &mut trials.trials[trials.open - 1];
        trial.swap_entropy(self);
        self.restore_state(recon, &trial.before);
        self.node_trials = trials;
    }

    /// `CodecBackend::keep_node_candidate`: note the innermost trial's
    /// coding as its best.
    pub(super) fn keep_node(&mut self, recon: &Picture) {
        let mut trials = std::mem::take(&mut self.node_trials);
        let open = trials.open;
        let NodeTrial { before, kept, .. } = &mut trials.trials[open - 1];
        self.save_state(&before.node(), recon, kept);
        kept.recorder = before.recorder;
        self.writer.save_after(before.recorder, &mut kept.added);
        self.node_trials = trials;
    }

    /// `CodecBackend::restore_node_candidate`: put the innermost trial's
    /// kept coding back.
    pub(super) fn restore_kept_node(&mut self, recon: &mut Picture) {
        let mut trials = std::mem::take(&mut self.node_trials);
        let trial = &mut trials.trials[trials.open - 1];
        trial.swap_entropy(self);
        self.restore_state(recon, &trial.kept);
        self.node_trials = trials;
    }

    /// `CodecBackend::end_node_trial`: close the innermost trial.
    pub(super) fn close_node_trial(&mut self) {
        self.node_trials.open = self
            .node_trials
            .open
            .checked_sub(1)
            .expect("a node trial is open");
    }
}
