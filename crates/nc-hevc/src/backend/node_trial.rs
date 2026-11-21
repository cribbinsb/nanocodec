//! Partition trials for the engine's RD partition search (`nc_encoder`
//! `rd_split.rs`, `CodecBackend::begin_node_trial`): a node's split flags
//! and CUs are coded by the real coder, priced over the node, and can be
//! undone, so the engine can compare a leaf and a quad split by what they
//! cost to code. VVC's `backend/node_trial.rs` is the same stack over its
//! own snapshot.
//!
//! Trials nest: a quad split's children open their own while the parent's
//! stays open, so the snapshots form a stack (buffers kept for reuse when
//! a trial ends). Each trial also keeps its best coding so far (a snapshot
//! taken after it), put back when a later candidate lost. Snapshot and
//! price are the shared ones (`coded_snapshot.rs`), over the node's square
//! and the items recorded since its trial began.

use nc_common::Picture;
use nc_encoder::Block;

use super::HevcBackend;
use super::coded_snapshot::CodedSnapshot;
use crate::syntax::{CuKind, Item};

/// One node trial: the state before it and its kept coding.
#[derive(Debug, Default)]
struct NodeTrial {
    before: CodedSnapshot,
    kept: CodedSnapshot,
}

/// The open node trials, innermost last.
#[derive(Debug, Default)]
pub(super) struct NodeTrials {
    /// Trials; the first `open` are in use, the rest are spare buffers.
    snapshots: Vec<NodeTrial>,
    open: usize,
}

impl HevcBackend {
    /// `CodecBackend::begin_node_trial`: note `node`'s coded state.
    pub(super) fn open_node_trial(&mut self, node: &Block, recon: &Picture) {
        let mut trials = std::mem::take(&mut self.node_trials);
        if trials.open == trials.snapshots.len() {
            trials.snapshots.push(NodeTrial::default());
        }
        self.save_coded(node, recon, &mut trials.snapshots[trials.open].before);
        trials.open += 1;
        self.node_trials = trials;
    }

    /// `CodecBackend::begin_node_pricing`: note only the node and the
    /// items recorded (a priced-only trial is never undone or kept).
    pub(super) fn open_node_pricing(&mut self, node: &Block) {
        let mut trials = std::mem::take(&mut self.node_trials);
        if trials.open == trials.snapshots.len() {
            trials.snapshots.push(NodeTrial::default());
        }
        trials.snapshots[trials.open]
            .before
            .mark(node, self.items.len());
        trials.open += 1;
        self.node_trials = trials;
    }

    /// The innermost open trial's snapshot.
    fn innermost_node_trial(&self) -> &CodedSnapshot {
        let trials = &self.node_trials;
        &trials.snapshots[trials.open.checked_sub(1).expect("a node trial is open")].before
    }

    /// `CodecBackend::node_trial_cost`: the price of the innermost trial's
    /// node as coded since it began.
    pub(super) fn node_trial_price(&self, source: &Picture, recon: &Picture) -> f32 {
        let snapshot = self.innermost_node_trial();
        self.coded_cost(&snapshot.block(), snapshot.items(), source, recon)
    }

    /// `CodecBackend::undo_node_trial`: put the innermost trial's state
    /// back.
    pub(super) fn undo_node(&mut self, recon: &mut Picture) {
        let trials = std::mem::take(&mut self.node_trials);
        self.restore_coded(recon, &trials.snapshots[trials.open - 1].before);
        self.node_trials = trials;
    }

    /// `CodecBackend::keep_node_candidate`: note the innermost trial's
    /// coding as its best.
    pub(super) fn keep_node(&mut self, recon: &Picture) {
        let mut trials = std::mem::take(&mut self.node_trials);
        let trial = &mut trials.snapshots[trials.open - 1];
        self.save_coded_after(&trial.before, recon, &mut trial.kept);
        self.node_trials = trials;
    }

    /// `CodecBackend::restore_node_candidate`: put the innermost trial's
    /// kept coding back.
    pub(super) fn restore_kept_node(&mut self, recon: &mut Picture) {
        let trials = std::mem::take(&mut self.node_trials);
        self.restore_coded(recon, &trials.snapshots[trials.open - 1].kept);
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

    /// `CodecBackend::last_cu_skipped`: the CU last recorded is coded with
    /// `cu_skip_flag` (HEVC §7.3.8.5).
    pub(super) fn last_cu_was_skip(&self) -> bool {
        matches!(
            self.items.last(),
            Some(Item::Cu(cu)) if matches!(cu.kind, CuKind::Skip { .. })
        )
    }
}
