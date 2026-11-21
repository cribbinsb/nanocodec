//! RD partition search (`planner.rd_split`): each node of an inter picture
//! is decided by coding its candidate partitions and keeping the cheapest
//! coded one, not by the planner's variance model alone.
//!
//! Why: the planner (`partition_plan.rs`, SVT-AV1's VLPD0) ranks partitions
//! by the variance of their whole-sample prediction: it misses the
//! transform, chroma and the real mode decision. Shapes and small blocks it
//! favours code worse (VVC multi-type tree: fixed QP J +4.6 %, +1.2 % with
//! the luma-only coded confirmation `planner.rect_rd`; small and
//! rectangular CUs code chroma 0.1-0.3 dB worse,
//! docs/analysis/tools-that-hurt-2026-11-20.md). A luma-only coded estimate
//! at planning time split more and lost (`planner.split_rd`, 2026-11-14).
//! `VVenC` decides its splits by coding them (`EncCu::xCompressCU`), as
//! SVT-AV1's PD1 refines PD0's depth by full mode decision.
//!
//! How: the plan proposes; at each node the search codes the planned
//! choice first, then the other candidates near it, each through the real
//! leaf path (mode decision and the codec's coder), and prices each over
//! the node with the codec ([`CodecBackend::node_trial_cost`]: all planes,
//! entropy-coded bits). A candidate is undone after pricing
//! ([`CodecBackend::undo_node_trial`]); the best so far is noted
//! ([`CodecBackend::keep_node_candidate`]) and put back when a later one
//! loses ([`CodecBackend::restore_node_candidate`]), so no node is coded
//! twice for its own sake.
//!
//! Candidates and pruning (the depth window as SVT's PD1 around PD0, the
//! rest as `VVenC`'s mode control):
//! - a quad split at most `rd_split_deeper` levels below a planned leaf, a
//!   leaf at most `rd_split_shallower` levels above the planned leaves;
//!   the backend's rectangular shapes where a leaf is tried
//!   (`rd_split_shapes`: everywhere, only at the plan's leaves, or only the
//!   planned shape), where the motion search separates their parts
//!   (`rd_split_shape_gate`, the planner's `rect_motion_gate` test);
//! - static nodes (pre-analysis) and nodes the planner may not change
//!   (the picture edge) keep the plan's choice;
//! - when the planned leaf codes as a skip CU, nothing smaller is tried
//!   (`rd_split_skip_exit`): smaller CUs cannot code less residual, and
//!   their extra partition and mode bits rarely buy prediction there;
//! - a split or shape stops as soon as its coded parts cost the best so far
//!   (exact: costs are nonnegative);
//! - a ternary shape only where the binary shape of its direction came
//!   within `rd_split_tt_margin` of the best (`VVenC`'s fast ternary rule);
//! - with `rd_split_margin`, a planned node tries the other way (split or
//!   leaf) only where the planner's cost of it was within that multiple of
//!   its choice's (`partition_plan::NodeMargins`): a classifier-style gate
//!   on the variance model's confidence, as Kvazaar's and Amestoy et al.'s
//!   (ICASSP 2019) learned trees search only their uncertain nodes;
//! - with `rd_split_context`, the context gate (`depth_context.rs`): a
//!   planned leaf tries the split only where a neighbour or the co-located
//!   leaf of the last frame is at least `-rd_split_deepen_context` levels
//!   smaller, a planned split tries the leaf only where the neighbours' and
//!   co-located levels relative to the node sum to at least
//!   `rd_split_merge_context`. What it skips: alternatives where the
//!   surroundings were coded at the planned size or coarser (for a split)
//!   or finer (for a leaf). Why safe: they only keep the plan. Measured
//!   (docs/handoff/rd-split-gate.md; fixed QP, quick draw, against the
//!   Default without the search): HEVC Default the full search J -5.4 % at
//!   +86 % P instructions, the gate -1 / -1 -4.2 % at +40 %; AV1 -4.7 % /
//!   +128 % and -3.5 % / +61 %; VVC +77 % and +40 % P instructions. The gate
//!   reads leaves the gated search coded itself, so `rd_split_context_period`
//!   searches one superblock in that many without it.
//!
//! Key frames (`planner.key_rd_split`): the same search around the key
//! plan (`key_plan.rs`), whose leaf cost predicts DC from the source and
//! codes one transform, so it misses what a block's own direction, its
//! chroma and the reconstructed neighbours change. A key node has no
//! motion or pre-analysis: it tries the leaf and the quad split inside
//! the depth window (`key_rd_deeper`, `key_rd_shallower`, no smaller than
//! `intra_min_log2`), with the exact bound as its only pruning. Why a
//! key can afford it: one frame in a key interval, inherited by all the
//! others. The strong intra encoders decide partitions this way (libaom
//! `av1_rd_pick_partition`, SVT-AV1 PD1, `VVenC` `xCompressCU`); measured
//! in docs/analysis/key-frame-rdo-2026-11-21.md.
//!
//! The key margin gate (`planner.key_rd_margin`, the margin gate above on
//! the key plan's margins): only nodes whose estimate nearly tied leaf
//! and split try the alternative. What it skips: the trials where the
//! estimate was decisive. Why that is safe: on the key RD labels (AV1,
//! HEVC, VP9 medium, 918 one-frame runs) the estimate's wrong decisions
//! sit at small margins: ratios under 1.05 hold about half the search's
//! gain, under 1.1 about three quarters, for a third to a half of the
//! extra coding (docs/analysis/intra-selection-2026-11-21.md). SVT-AV1's
//! PD1 depth refinement keeps a depth for the same reason, where PD0's
//! cost deviation to it is under a threshold (`DepthRefinementCtrls`).

use nc_common::{Picture, PlaneIndex};
use nc_dsp::BlockSize;

use super::{Encoder, FrameInputs, FramePictures, search_vector};
use crate::backend::CodecBackend;
use crate::decision::{is_static, n_variance};
use crate::depth_context::NodeContext;
use crate::params::PlannerParams;
use crate::partition::Block;
use crate::partition_plan::{Partition, motion_separates};
use crate::reuse::{CodedBlock, ReuseTerm};
use crate::shape::{MAX_SHAPE_PARTS, PartitionShape};
use crate::threshold_control::BandEvidence;

/// The search's settings (`planner.rd_split*`).
#[derive(Debug, Clone, Copy)]
struct RdSplit {
    deeper: u8,
    shallower: u8,
    shapes: ShapeReach,
    shape_gate: f32,
    skip_exit: bool,
    tt_margin: f32,
    /// `planner.rd_split_margin` (0: no gate).
    margin: f32,
    /// The context gate's thresholds (`planner.rd_split_context`), `None`
    /// also on a superblock its refresh rotation searches ungated.
    context: Option<ContextGate>,
    /// Whether coded leaf sizes are recorded for the gate (it is on).
    record_depths: bool,
    /// `planner.rd_split_deepen_residual` (0: no gate).
    deepen_residual: f32,
    /// `planner.rd_split_merge_skipped`.
    merge_skipped: bool,
    /// `planner.rd_split_frame_period`: one inter frame in this many is
    /// searched.
    frame_period: u64,
    /// `planner.inter_min_log2` (key frames: `planner.intra_min_log2`): no
    /// split below it.
    min_log2: u8,
    /// Optional fraction of the estimated SATD tail that contributes to
    /// the prune test after a completed prefix.
    tail_satd: f32,
}

impl RdSplit {
    /// The search's settings, or `None` while `planner.rd_split` is off.
    fn of(params: &PlannerParams) -> Option<Self> {
        (params.rd_split >= 0.5).then(|| Self {
            deeper: params.rd_split_deeper.round() as u8,
            shallower: params.rd_split_shallower.round() as u8,
            shapes: match params.rd_split_shapes.round() as u8 {
                0 => ShapeReach::Planned,
                1 => ShapeReach::PlannedLeaves,
                _ => ShapeReach::Leaves,
            },
            shape_gate: params.rd_split_shape_gate,
            skip_exit: params.rd_split_skip_exit >= 0.5,
            tt_margin: params.rd_split_tt_margin,
            margin: params.rd_split_margin,
            frame_period: params.rd_split_frame_period.round().max(1.0) as u64,
            deepen_residual: params.rd_split_deepen_residual,
            merge_skipped: params.rd_split_merge_skipped >= 0.5,
            record_depths: params.rd_split_context >= 0.5,
            context: (params.rd_split_context >= 0.5).then(|| ContextGate {
                deepen: params.rd_split_deepen_context.round() as i8,
                merge: params.rd_split_merge_context.round() as i8,
                period: params.rd_split_context_period.round() as u64,
            }),
            min_log2: params.inter_min_log2.round() as u8,
            tail_satd: params.rd_split_tail_satd,
        })
    }

    /// The key-frame search's settings, or `None` while
    /// `planner.key_rd_split` is off: the depth window and the margin gate
    /// (on the key plan's margins), since every other gate of the inter
    /// search reads motion, residual or co-located leaves a key frame does
    /// not have.
    fn of_key(params: &PlannerParams) -> Option<Self> {
        (params.key_rd_split >= 0.5).then(|| Self {
            deeper: params.key_rd_deeper.round() as u8,
            shallower: params.key_rd_shallower.round() as u8,
            shapes: ShapeReach::Planned,
            shape_gate: 0.0,
            skip_exit: false,
            tt_margin: 0.0,
            margin: params.key_rd_margin,
            frame_period: 1,
            deepen_residual: 0.0,
            merge_skipped: false,
            record_depths: false,
            context: None,
            min_log2: params.intra_min_log2.round() as u8,
            tail_satd: 0.0,
        })
    }
}

/// The context gate (module docs; `planner.rd_split_deepen_context` and
/// `planner.rd_split_merge_context`).
#[derive(Debug, Clone, Copy)]
struct ContextGate {
    /// A planned leaf tries the split where the smaller relative level is
    /// at most this.
    deepen: i8,
    /// A planned split tries the leaf where the relative levels sum to at
    /// least this.
    merge: i8,
    /// One superblock in this many is searched ungated (0: none).
    period: u64,
}

/// Relative levels beyond this count as this (the gate's thresholds were
/// fitted on levels clamped so).
const CONTEXT_LEVELS: i8 = 2;

impl ContextGate {
    /// Whether a planned leaf in `context` tries the split.
    fn deepens(self, context: NodeContext) -> bool {
        context.neighbours.min(context.co_located) <= self.deepen
    }

    /// Whether a planned split in `context` tries the leaf.
    fn merges(self, context: NodeContext) -> bool {
        let clamp = |v: i8| v.clamp(-CONTEXT_LEVELS, CONTEXT_LEVELS);
        clamp(context.neighbours) + clamp(context.co_located) >= self.merge
    }
}

/// Which nodes try rectangular shapes besides a planned one
/// (`planner.rd_split_shapes`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShapeReach {
    /// None.
    Planned,
    /// Nodes the plan codes as a leaf or a shape.
    PlannedLeaves,
    /// Every node where a leaf is tried.
    Leaves,
}

/// Where a node stands against the plan.
#[derive(Clone, Copy)]
enum Planned<'p> {
    /// The plan's partition of the node.
    Plan(&'p Partition),
    /// This many levels below a planned leaf (at least 1).
    Below(u8),
}

/// One way to code a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Candidate {
    Leaf,
    Split,
    Shape(PartitionShape),
}

/// Most candidates of one node: leaf, split and four shapes.
const MAX_CANDIDATES: usize = 2 + PartitionShape::ALL.len();

/// The candidates of one node, the planned choice first.
struct Candidates {
    list: [Candidate; MAX_CANDIDATES],
    len: usize,
}

impl Candidates {
    fn push(&mut self, candidate: Candidate) {
        if !self.list[..self.len].contains(&candidate) {
            self.list[self.len] = candidate;
            self.len += 1;
        }
    }

    fn as_slice(&self) -> &[Candidate] {
        &self.list[..self.len]
    }
}

/// Leaves the search has coded and how many of them were skips
/// (`planner.rd_split_merge_skipped`).
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct LeafTally {
    coded: u64,
    skipped: u64,
}

/// Whether `candidate` is left out before it is coded: a ternary shape
/// whose binary shape of the same direction came above `bound` by more
/// than `rd_split_tt_margin`, or, with `rd_split_merge_skipped`, a leaf or
/// shape after a planned split whose coded leaves were not all skips
/// (`split_coded_residual`). What the latter skips: the leaf and shapes
/// where the split's parts code residual the leaf would have to match.
/// Measured: docs/handoff/rd-split-gate.md.
fn pruned(
    candidate: Candidate,
    rd: RdSplit,
    bound: f32,
    binary: [f32; 2],
    split_coded_residual: bool,
) -> bool {
    match candidate {
        Candidate::Shape(shape)
            if shape.part_count() == 3
                && rd.tt_margin < 1.0
                && binary[usize::from(!shape.is_horizontal())] > bound * (1.0 + rd.tt_margin) =>
        {
            true
        }
        Candidate::Leaf | Candidate::Shape(_) => rd.merge_skipped && split_coded_residual,
        Candidate::Split => false,
    }
}

/// The engine's own state a node trial must undo with the codec's.
struct EngineMark {
    block_offsets: usize,
    reuse: Option<usize>,
    evidence: BandEvidence,
}

/// The engine's own state after a node's best candidate so far, put back
/// with the codec's kept coding.
#[derive(Default)]
struct EngineKept {
    block_offsets: Vec<(f64, f64)>,
    reuse: Vec<CodedBlock>,
    evidence: BandEvidence,
}

impl<B: CodecBackend> Encoder<B> {
    /// Whether the RD partition search may run on this frame: the backend
    /// has node trials and nothing constrains the leaves (no intra
    /// refresh, random decisions or spatial layers).
    fn rd_split_applies(&self) -> bool {
        B::NODE_TRIALS && self.window.is_none() && self.random.is_none() && self.spatial.is_none()
    }

    /// Code superblock `sb` as planned (`plan`, on inter frames), or by the
    /// RD partition search around the plan where `planner_params` turn it
    /// on and the frame allows it.
    pub(super) fn code_superblock(
        &mut self,
        sb: Block,
        plan: Option<&Partition>,
        pictures: FramePictures<'_>,
        inter: Option<&FrameInputs<'_>>,
        planner_params: &PlannerParams,
    ) {
        let rd = match inter {
            Some(_) => RdSplit::of(planner_params),
            None => RdSplit::of_key(planner_params),
        }
        .filter(|_| self.rd_split_applies());
        let rd = match rd {
            Some(rd) if inter.is_some() && !self.searches_this_frame(&rd) => {
                self.depths.hold();
                None
            }
            rd => rd.map(|rd| self.refreshed(rd, &sb)),
        };
        match (plan, rd) {
            (Some(plan), Some(rd)) => {
                if rd.record_depths {
                    self.depths
                        .fit(self.layer.coded_width, self.layer.coded_height);
                }
                self.search_node(sb, Planned::Plan(plan), pictures, inter, (rd, false));
            }
            _ => self.code_node(sb, plan, pictures, inter),
        }
    }

    /// Whether this frame is one the search runs on
    /// (`planner.rd_split_frame_period`). What it skips: the search on the
    /// other inter frames, which code their plan. Why: the search's partitions
    /// improve the references the frames between draw on.
    fn searches_this_frame(&self, rd: &RdSplit) -> bool {
        self.next_index
            .saturating_sub(1)
            .is_multiple_of(rd.frame_period)
    }

    /// `rd` for superblock `sb`: without the context gate on the
    /// superblocks its refresh rotation re-checks this frame
    /// (`planner.rd_split_context_period`). What it costs: an ungated
    /// search on one superblock in `period`. Why: the gate reads leaves the
    /// gated search itself coded, so a size it keeps agreeing with would
    /// otherwise never be re-checked.
    fn refreshed(&self, mut rd: RdSplit, sb: &Block) -> RdSplit {
        if let Some(gate) = rd.context.filter(|gate| gate.period > 0) {
            let log2 = sb.log2_size();
            let columns = self.layer.coded_width.div_ceil(1 << log2);
            let index = (sb.y >> log2) * columns + (sb.x >> log2);
            if (index as u64 + self.depths.frames()).is_multiple_of(gate.period) {
                rd.context = None;
            }
        }
        rd
    }

    /// Code node `node` (`frame`: its inter inputs, `None` on a key frame)
    /// by the RD partition search around `plan`; returns the coded cost when another node needs it for a
    /// pruning bound. The backend keeps the chosen coding and each node
    /// records its own depth, so no second partition tree is needed.
    #[allow(
        clippy::too_many_lines,
        reason = "the candidate loop also owns rollback and commit"
    )]
    fn search_node(
        &mut self,
        node: Block,
        planned: Planned<'_>,
        pictures: FramePictures<'_>,
        frame: Option<&FrameInputs<'_>>,
        (rd, cost_needed): (RdSplit, bool),
    ) -> f32 {
        let (width, height) = (self.layer.coded_width, self.layer.coded_height);
        if node.x >= width || node.y >= height {
            return 0.0; // outside: no syntax
        }
        let FramePictures {
            source,
            reference,
            secondary,
            recon,
        } = pictures;
        macro_rules! pictures {
            () => {
                FramePictures {
                    source,
                    reference,
                    secondary,
                    recon: &mut *recon,
                }
            };
        }
        if !node.is_inside(width, height) {
            // The picture edge forces the split (the plan splits it too).
            let split = self.code_split(
                node,
                planned,
                pictures!(),
                frame,
                rd,
                (f32::INFINITY, cost_needed),
            );
            return split.expect("an unbounded split is never abandoned");
        }
        let candidates = self.candidates(&node, planned, frame, rd);
        if candidates.len == 1 {
            self.backend.begin_node_pricing(&node, recon);
        } else {
            self.backend.begin_node_trial(&node, recon);
        }
        let mark = self.engine_mark();
        let mut kept = EngineKept::default();
        let mut best: Option<(f32, Candidate)> = None;
        // Whether the coded state is the best candidate's.
        let mut best_is_coded = false;
        // The binary shapes' costs (horizontal, vertical), for the ternary
        // rule.
        let mut binary = [f32::INFINITY; 2];
        // Whether every leaf the planned split coded is a skip
        // (`planner.rd_split_merge_skipped`; true until it is coded).
        let tally = self.rd_leaf_tally;
        let mut split_all_skipped = true;
        let list = candidates.as_slice();
        for (i, &candidate) in list.iter().enumerate() {
            let bound = best.as_ref().map_or(f32::INFINITY, |b| b.0);
            if pruned(candidate, rd, bound, binary, i > 0 && !split_all_skipped) {
                continue;
            }
            if i > 0 {
                self.backend.undo_node_trial(recon);
                self.undo_engine(&mark);
            }
            let coded = match candidate {
                Candidate::Leaf => {
                    self.code_leaf_candidate(node, planned, pictures!(), frame);
                    true
                }
                Candidate::Split => self
                    .code_split(node, planned, pictures!(), frame, rd, (bound, false))
                    .is_some(),
                Candidate::Shape(shape) => self.code_shape(
                    node,
                    shape,
                    planned,
                    pictures!(),
                    frame,
                    (rd.tail_satd, bound),
                ),
            };
            best_is_coded = false;
            if i == 0 && candidate == Candidate::Split {
                let now = self.rd_leaf_tally;
                split_all_skipped = now.skipped - tally.skipped == now.coded - tally.coded;
            }
            if !coded {
                continue;
            }
            // A forced choice still opens the backend's pricing scope. Some
            // writers start recording symbols there. Its price only matters
            // if an ancestor uses it to decide whether to code another part.
            let cost = if cost_needed || list.len() > 1 {
                self.backend.node_trial_cost(source, recon)
            } else {
                0.0
            };
            if let Candidate::Shape(shape) = candidate
                && shape.part_count() == 2
            {
                binary[usize::from(!shape.is_horizontal())] = cost;
            }
            if cost >= bound {
                continue;
            }
            best = Some((cost, candidate));
            best_is_coded = true;
            let skip_exit = i == 0
                && candidate == Candidate::Leaf
                && rd.skip_exit
                && self.backend.last_cu_skipped();
            if skip_exit {
                break;
            }
            if i + 1 < list.len() {
                self.backend.keep_node_candidate(recon);
                self.keep_engine(&mark, &mut kept);
            }
        }
        let (cost, candidate) = best.expect("the first candidate is unbounded");
        if !best_is_coded {
            self.backend.restore_node_candidate(recon);
            self.restore_engine(&mark, &kept);
        }
        self.backend.end_node_trial();
        self.record_leaves(&node, candidate, rd);
        cost
    }

    /// The candidates of `node` (inside the picture), the planned choice
    /// first.
    fn candidates(
        &self,
        node: &Block,
        planned: Planned<'_>,
        frame: Option<&FrameInputs<'_>>,
        rd: RdSplit,
    ) -> Candidates {
        let log2 = node.log2_size();
        let can_split = log2 > self.backend.min_block_log2() && log2 > rd.min_log2;
        let can_leaf = log2 <= self.backend.max_leaf_log2();
        let mut candidates = Candidates {
            list: [Candidate::Leaf; MAX_CANDIDATES],
            len: 0,
        };
        // The margin gate: the planner's choice stands alone where the
        // variance model preferred it by at least `rd.margin` (what it
        // skips and its measured effect: docs/handoff/hevc.md "RD
        // partition search").
        let uncertain = rd.margin <= 0.0
            || self
                .plan_margins
                .as_ref()
                .is_none_or(|margins| margins.get(node) < rd.margin);
        // The context gate (module docs).
        let context = rd.context.map(|gate| (gate, self.depths.of(node)));
        let deepens = context.is_none_or(|(gate, context)| gate.deepens(context))
            && self.residual_deepens(node, planned, rd);
        let merges = context.is_none_or(|(gate, context)| gate.merges(context));
        let (leaf, split, below) = match planned {
            Planned::Plan(Partition::Leaf(_)) => {
                candidates.push(Candidate::Leaf);
                (true, can_split && rd.deeper > 0 && uncertain && deepens, 0)
            }
            Planned::Plan(Partition::Shaped(shape, _)) => {
                candidates.push(Candidate::Shape(*shape));
                (can_leaf, can_split && rd.deeper > 0, 0)
            }
            Planned::Plan(plan @ Partition::Split(_)) => {
                candidates.push(Candidate::Split);
                let height = log2 - plan.smallest_side_log2(log2);
                (
                    can_leaf && height <= rd.shallower && uncertain && merges,
                    true,
                    0,
                )
            }
            Planned::Below(levels) => (can_leaf, can_split && levels < rd.deeper, levels),
        };
        let pre = frame.and_then(|frame| frame.pre);
        if below == 0 && pre.is_some_and(|pre| is_static::<B>(pre, node)) {
            // A static node keeps the plan, as the planner never splits one.
            if candidates.len == 0 {
                candidates.push(Candidate::Leaf);
            }
            return candidates;
        }
        if leaf {
            candidates.push(Candidate::Leaf);
        }
        if split {
            candidates.push(Candidate::Split);
        }
        let planned_leaf = matches!(
            planned,
            Planned::Plan(Partition::Leaf(_) | Partition::Shaped(..))
        );
        let shapes = match rd.shapes {
            ShapeReach::Planned => false,
            ShapeReach::PlannedLeaves => planned_leaf,
            ShapeReach::Leaves => leaf,
        };
        // Rectangular leaves are inter syntax in every backend that has
        // them (`CodecBackend::RECTANGULAR_LEAVES`).
        if let Some(frame) = frame.filter(|_| shapes && B::RECTANGULAR_LEAVES) {
            for shape in self.legal_shapes(node).iter() {
                let parts = shape.parts(node);
                if rd.shape_gate <= 0.0
                    || motion_separates(frame.motion, &parts[..shape.part_count()], rd.shape_gate)
                {
                    candidates.push(Candidate::Shape(shape));
                }
            }
        }
        candidates
    }

    /// The residual gate (`planner.rd_split_deepen_residual`): whether a
    /// planned leaf's residual variance per sample at its search vector
    /// reaches the threshold in λ. What it skips: the split below leaves
    /// whose prediction already leaves little residual, where smaller
    /// blocks have little left to predict better. Measured:
    /// docs/handoff/rd-split-gate.md.
    fn residual_deepens(&self, node: &Block, planned: Planned<'_>, rd: RdSplit) -> bool {
        let Planned::Plan(Partition::Leaf(Some(stats))) = planned else {
            return true;
        };
        if rd.deepen_residual <= 0.0 {
            return true;
        }
        let area = (node.width() * node.height()) as f32;
        let per_sample = n_variance::<B>(stats.search, node) as f32 / area;
        per_sample >= rd.deepen_residual * self.backend.lambda()
    }

    /// Code `node` as one leaf, with the plan's statistics if it planned
    /// one.
    fn code_leaf_candidate(
        &mut self,
        node: Block,
        planned: Planned<'_>,
        pictures: FramePictures<'_>,
        frame: Option<&FrameInputs<'_>>,
    ) {
        let stats = match planned {
            Planned::Plan(Partition::Leaf(stats)) => stats.as_ref(),
            _ => None,
        };
        self.backend.write_partition(&node, false);
        self.code_leaf(node, stats, pictures, frame);
        self.tally_leaf();
    }

    /// Code `node` split into quadrants, each searched; `None` once the
    /// coded quadrants cost `bound` (the rest is left uncoded).
    fn code_split(
        &mut self,
        node: Block,
        planned: Planned<'_>,
        pictures: FramePictures<'_>,
        frame: Option<&FrameInputs<'_>>,
        rd: RdSplit,
        (bound, return_cost): (f32, bool),
    ) -> Option<f32> {
        let FramePictures {
            source,
            reference,
            secondary,
            recon,
        } = pictures;
        self.backend.write_partition(&node, true);
        let mut spent = 0.0;
        let children = node.quadrants();
        for (i, child) in children.iter().copied().enumerate() {
            if spent >= bound
                || (i > 0
                    && rd.tail_satd > 0.0
                    && spent >= 0.75 * bound
                    && self.satd_tail_loses(
                        &children,
                        i,
                        frame,
                        (source, reference),
                        (rd.tail_satd, spent, bound),
                    ))
            {
                return None;
            }
            let child_planned = match planned {
                Planned::Plan(Partition::Split(plans)) => Planned::Plan(&plans[i]),
                Planned::Plan(_) => Planned::Below(1),
                Planned::Below(levels) => Planned::Below(levels + 1),
            };
            let pictures = FramePictures {
                source,
                reference,
                secondary,
                recon: &mut *recon,
            };
            // The caller prices a competing split as a whole. Child prices
            // are needed only for pruning the unencoded suffix or when a
            // forced edge split must return its accumulated cost.
            let cost_needed = return_cost || (bound.is_finite() && i + 1 < children.len());
            spent += self.search_node(child, child_planned, pictures, frame, (rd, cost_needed));
        }
        Some(spent)
    }

    /// Code `node` as the leaves of `shape`; `false` once the coded leaves
    /// cost `bound`.
    fn code_shape(
        &mut self,
        node: Block,
        shape: PartitionShape,
        planned: Planned<'_>,
        pictures: FramePictures<'_>,
        frame: Option<&FrameInputs<'_>>,
        (tail_satd, bound): (f32, f32),
    ) -> bool {
        let FramePictures {
            source,
            reference,
            secondary,
            recon,
        } = pictures;
        let unplanned = [None; MAX_SHAPE_PARTS];
        let stats = match planned {
            Planned::Plan(Partition::Shaped(s, stats)) if *s == shape => stats.as_ref(),
            _ => &unplanned,
        };
        self.backend.write_shape(&node, shape);
        let mut spent = 0.0;
        let parts = shape.parts(&node);
        let part_count = shape.part_count();
        let parts = &parts[..part_count];
        for (i, (part, part_stats)) in parts.iter().copied().zip(stats).enumerate() {
            if spent >= bound
                || (i > 0
                    && tail_satd > 0.0
                    && spent >= 0.75 * bound
                    && self.satd_tail_loses(
                        parts,
                        i,
                        frame,
                        (source, reference),
                        (tail_satd, spent, bound),
                    ))
            {
                return false;
            }
            let pictures = FramePictures {
                source,
                reference,
                secondary,
                recon: &mut *recon,
            };
            self.backend.begin_node_pricing(&part, pictures.recon);
            self.code_leaf(part, part_stats.as_ref(), pictures, frame);
            self.tally_leaf();
            if bound.is_finite() && i + 1 < part_count {
                spent += self.backend.node_trial_cost(source, recon);
            }
            self.backend.end_node_trial();
        }
        true
    }

    /// Heuristic for a split or shape whose coded prefix already spends
    /// most of the budget. Scale the remaining parts' sampled luma SATD
    /// by the actual coded cost per unit of SATD in the prefix.
    #[inline(never)]
    fn satd_tail_loses(
        &self,
        parts: &[Block],
        done: usize,
        frame: Option<&FrameInputs<'_>>,
        (source, reference): (&Picture, &Picture),
        (fraction, spent, bound): (f32, f32, f32),
    ) -> bool {
        // The estimate predicts from the reference: inter frames only.
        let Some(frame) = frame else {
            return false;
        };
        let source = source.plane(PlaneIndex::Y);
        let reference = reference.plane(PlaneIndex::Y);
        let mut prefix = 0u32;
        let mut tail = 0u32;
        for (i, part) in parts.iter().enumerate() {
            // Sample the centre at the analysis vector. This is a proxy
            // for cost, including when the coder ultimately picks another
            // vector or intra.
            let side = 8.min(part.width().min(part.height()));
            let x = (part.x + (part.width() - side) / 2) as isize;
            let y = (part.y + (part.height() - side) / 2) as isize;
            let (dx, dy) = search_vector::<B>(frame.motion, part).full_pel();
            let Some(src) = source.region(x, y, side, side) else {
                return false;
            };
            let Some(pred) = reference.region(x + dx as isize, y + dy as isize, side, side) else {
                return false;
            };
            let size = BlockSize::new(side, side).expect("SATD block size");
            let satd = self.dsp.satd(size, &src, &pred) * (part.area() / (side * side)) as u32;
            if i < done {
                prefix += satd;
            } else {
                tail += satd;
            }
        }
        prefix != 0 && fraction * spent * tail as f32 >= (bound - spent) * prefix as f32
    }

    /// Record `node`'s coded leaves for the context gate. A split's
    /// children recorded their own when searched; a leaf or shape records
    /// the node (an earlier split candidate's children may have recorded
    /// smaller ones).
    fn record_leaves(&mut self, node: &Block, candidate: Candidate, rd: RdSplit) {
        if !rd.record_depths {
            return;
        }
        let log2 = node.log2_size();
        let smallest = match candidate {
            Candidate::Leaf => log2,
            Candidate::Shape(shape) => shape.log2_min_side(log2),
            Candidate::Split => return,
        };
        self.depths.record(node, smallest);
    }

    /// Count the leaf just coded (`LeafTally`).
    fn tally_leaf(&mut self) {
        self.rd_leaf_tally.coded += 1;
        self.rd_leaf_tally.skipped += u64::from(self.backend.last_cu_skipped());
    }

    fn engine_mark(&self) -> EngineMark {
        EngineMark {
            block_offsets: self.block_offsets.len(),
            reuse: self.layer.reuse.as_ref().map(ReuseTerm::coded_count),
            evidence: self.decision_scratch.evidence,
        }
    }

    fn keep_engine(&self, mark: &EngineMark, kept: &mut EngineKept) {
        kept.block_offsets.clear();
        kept.block_offsets
            .extend_from_slice(&self.block_offsets[mark.block_offsets..]);
        kept.reuse.clear();
        if let (Some(term), Some(count)) = (self.layer.reuse.as_ref(), mark.reuse) {
            kept.reuse.extend_from_slice(term.coded_after(count));
        }
        kept.evidence = self.decision_scratch.evidence;
    }

    fn restore_engine(&mut self, mark: &EngineMark, kept: &EngineKept) {
        self.block_offsets.truncate(mark.block_offsets);
        self.block_offsets.extend_from_slice(&kept.block_offsets);
        if let (Some(term), Some(count)) = (self.layer.reuse.as_mut(), mark.reuse) {
            term.replace_coded_after(count, &kept.reuse);
        }
        self.decision_scratch.evidence = kept.evidence;
    }

    fn undo_engine(&mut self, mark: &EngineMark) {
        self.block_offsets.truncate(mark.block_offsets);
        if let (Some(term), Some(count)) = (self.layer.reuse.as_mut(), mark.reuse) {
            term.replace_coded_after(count, &[]);
        }
        self.decision_scratch.evidence = mark.evidence;
    }
}
