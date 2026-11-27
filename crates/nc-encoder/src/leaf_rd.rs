//! Key-frame leaves by coded cost (`decision.intra_rd_modes`,
//! `intra_rd_transforms`, `intra_rd_chroma`): a leaf is coded with its
//! backend's alternatives and keeps the cheapest coded one.
//!
//! Why: every backend chooses an intra leaf's mode and transform by an
//! estimate (prediction SATD, or one transform priced without neighbour
//! contexts, luma only), because that is what a P frame's few intra blocks
//! can afford. A key frame is one frame in a key interval and every frame
//! after it inherits its quality, and the estimate is where the strong
//! intra encoders differ from us: they decide by the coded cost (libaom
//! `av1_rd_pick_intra_sby_mode` / `av1_pick_uniform_tx_size_type_yrd`,
//! SVT-AV1's full-loop mode decision stage, `VVenC`
//! `IntraSearch::estIntraPredLumaQT`).
//!
//! How: the backend declares what else it could code the leaf with
//! ([`CodecBackend::intra_alternatives`]): its next-ranked luma modes,
//! other transform layouts, other chroma modes. The search settles one
//! axis at a time, luma first: each alternative is coded through the real
//! leaf path inside a node trial (`rd_split.rs` uses the same trials for
//! partitions) and priced with [`CodecBackend::node_trial_cost`]: all
//! planes, reconstruction error, entropy-coded bits with the real
//! contexts. One axis at a time costs the sum of the alternatives, not
//! their product.
//!
//! Which alternatives (`decision.intra_rd_margin`): only those the
//! backend's own ranking left within that margin of its best estimate, as
//! SVT-AV1 passes a candidate to its full-cost stage only within a cost
//! deviation of the best of the stage before (`mds3_cand_base_th`,
//! `nic_pruning`; idea, no code). What it skips: coding modes the estimate
//! already puts far behind. Why safe: they keep the backend's choice.
//! Measured in docs/analysis/intra-cheap-2026-11-21.md.
//!
//! What it costs: one more coding of the leaf per alternative tried, on
//! key frames only. Measured in
//! docs/analysis/key-frame-rdo-2026-11-21.md.

use nc_common::Picture;

use super::Encoder;
use crate::backend::{BlockDecision, CodecBackend, IntraAxis, IntraChoice};
use crate::params::DecisionParams;
use crate::partition::Block;

/// Whether the key-frame leaf search runs on the leaves being coded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum LeafSearch {
    /// On every key leaf coded.
    #[default]
    Active,
    /// Not now: the partition search is coding its trials with the
    /// backend's own leaf choices (`planner.key_rd_leaf_pass`).
    Suspended,
}

/// Per mille (`decision.intra_rd_margin`).
const MARGIN_UNIT: f32 = 1000.0;

/// How many alternatives per axis the search may try (`decision.intra_rd_*`,
/// rounded).
fn limits(params: &DecisionParams) -> IntraChoice {
    IntraChoice {
        luma: params.intra_rd_modes.round() as u8,
        transform: params.intra_rd_transforms.round() as u8,
        chroma: params.intra_rd_chroma.round() as u8,
    }
}

impl<B: CodecBackend> Encoder<B> {
    /// Whether key-frame leaves are searched: a limit is set, the backend
    /// has node trials, the partition search is not coding trials without
    /// it (`planner.key_rd_leaf_pass`), and nothing else constrains or
    /// draws the leaves (intra refresh, random decisions, spatial layers).
    pub(super) fn leaf_rd_applies(&self) -> bool {
        B::NODE_TRIALS
            && self.leaf_search == LeafSearch::Active
            && limits(&self.frame_decision_params) != IntraChoice::default()
            && self.window.is_none()
            && self.random.is_none()
            && self.spatial.is_none()
    }

    /// Code key-frame leaf `node` as the cheapest of `decision` and the
    /// backend's alternatives to it; returns what was coded.
    pub(super) fn code_key_leaf_by_rd(
        &mut self,
        node: &Block,
        decision: BlockDecision,
        source: &Picture,
        recon: &mut Picture,
    ) -> BlockDecision {
        let limits = limits(&self.frame_decision_params);
        let offered = self
            .backend
            .intra_alternatives(node, &decision, source, recon);
        let tried = |offered: IntraChoice, axis: IntraAxis| offered.get(axis).min(limits.get(axis));
        if IntraAxis::ALL.iter().all(|&axis| tried(offered, axis) == 0) {
            self.backend
                .encode_block(node, &decision, source, None, recon);
            return decision;
        }
        self.backend.begin_node_trial(node, recon);
        self.backend
            .encode_block(node, &decision, source, None, recon);
        let mut best = (self.backend.node_trial_cost(source, recon), decision);
        // Whether the coded state is the best candidate's, and whether the
        // backend holds a kept copy of it.
        let (mut best_is_coded, mut best_is_kept) = (true, false);
        for axis in IntraAxis::ALL {
            if limits.get(axis) == 0 {
                continue;
            }
            // The alternatives of this axis are those of the choices made
            // on the axes before it, asked on the state before the leaf.
            let offered = if axis == IntraAxis::Luma {
                offered
            } else {
                if !best_is_coded {
                    self.backend.restore_node_candidate(recon);
                }
                // Kept again even when it was: a backend may hand its kept
                // copy back when it restores.
                self.backend.keep_node_candidate(recon);
                best_is_kept = true;
                self.backend.undo_node_trial(recon);
                best_is_coded = false;
                self.backend
                    .intra_alternatives(node, &best.1, source, recon)
            };
            for alternative in 1..=tried(offered, axis) {
                if axis == IntraAxis::Luma && self.estimate_rules_out(alternative) {
                    // The ranking is ascending: the rest are no closer.
                    break;
                }
                if best_is_coded && !best_is_kept {
                    self.backend.keep_node_candidate(recon);
                    best_is_kept = true;
                }
                self.backend.undo_node_trial(recon);
                if let Some(labeller) = self.labels.as_mut() {
                    labeller.add_work(node.width() * node.height());
                }
                let candidate = BlockDecision {
                    intra: best.1.intra.with(axis, alternative),
                    ..best.1
                };
                self.backend
                    .encode_block(node, &candidate, source, None, recon);
                let cost = self.backend.node_trial_cost(source, recon);
                best_is_coded = cost < best.0;
                if best_is_coded {
                    best = (cost, candidate);
                    best_is_kept = false;
                }
            }
        }
        if !best_is_coded {
            self.backend.restore_node_candidate(recon);
        }
        self.backend.end_node_trial();
        best.1
    }

    /// Whether the backend's own estimate puts luma alternative
    /// `alternative` further than `decision.intra_rd_margin` behind the
    /// better of its choice and its first alternative (a backend's choice
    /// may come from a finer measure than the ranking of the rest).
    fn estimate_rules_out(&self, alternative: u8) -> bool {
        let margin = self.frame_decision_params.intra_rd_margin;
        if margin <= 0.0 {
            return false;
        }
        let estimate = |rank| self.backend.intra_alternative_estimate(rank);
        let (Some(chosen), Some(first), Some(this)) =
            (estimate(0), estimate(1), estimate(alternative))
        else {
            return false;
        };
        this * MARGIN_UNIT > chosen.min(first) * margin
    }
}
