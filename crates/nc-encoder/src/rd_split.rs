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
//!
//! Its value per direction (`planner.key_rd_merge_margin`): planned splits
//! trying the leaf may take another margin than planned leaves trying the
//! split, since a plan errs one way more than the other (HEVC, VP9 and VVC
//! plans over-split). Measured (one-frame keys, window 1/2): VP9 deepen
//! 1.05 with merge 1.5 J -7.1 % at 1.73x the key's CPU against -6.9 % at
//! 2.03x for 1.2 both ways; VVC, HEVC, AV1 and AV2 within 0.1 % and 0.05x
//! of the single value (docs/analysis/intra-cheap-2026-11-21.md 4.9).
//!
//! The fitted key-trial gate (`planner.key_rd_gate`, `trial_gate.rs`):
//! where set, a regression tree on the key plan's margin, direction, size,
//! leaf estimate and λ decides instead of the two margins.
//!
//! Leaf first (`planner.key_rd_leaf_first`): where the plan split and the
//! leaf is tried, the leaf is coded first, so the split runs under the
//! leaf's cost as its bound. What it skips: the rest of a planned split
//! once its coded quadrants cost the leaf. Why safe: the bound is exact;
//! only an exact cost tie now keeps the leaf where it kept the split.
//! Measured (one key under search 2/2, hardware cycles on Zen 5): AV1
//! -2.2 % [-2.4, -2.1], VP9 -0.3 %, AV2 -0.2 %, VVC -0.1 %, HEVC +0.1 %.
//!
//! Key frames, the leaf pass (`planner.key_rd_leaf_pass`): the key-frame
//! leaf search (`leaf_rd.rs`) leaves the partition trials, which code the
//! backend's own leaf choice; the chosen partition is then coded once more
//! with the leaf search on. What it skips: the alternatives of every leaf
//! of every losing candidate. What it costs: one more coding of the
//! superblock, and a partition chosen under the estimate-ranked modes.
//! Measured (one-frame keys, HEVC, search 2/2 with four margin-gated
//! alternatives): J -10.2 % at 3.4x the key's CPU against -11.8 % at 5.0x
//! with the leaf search inside the trials
//! (docs/analysis/intra-cheap-2026-11-21.md).

use nc_common::{Picture, PlaneIndex};
use nc_dsp::{BlockSize, Dsp};

use super::leaf_rd::LeafSearch;
use super::{Encoder, FrameInputs, FramePictures, search_vector};
use crate::backend::CodecBackend;
use crate::decision::{is_static, n_variance};
use crate::depth_context::NodeContext;
use crate::labels;
use crate::params::PlannerParams;
use crate::partition::Block;
use crate::partition_plan::{Partition, motion_separates};
use crate::reuse::{CodedBlock, ReuseTerm};
use crate::shape::{MAX_SHAPE_PARTS, PartitionShape};
use crate::stop_inputs::{StopEvidence, StopInputs};
use crate::stop_value::StopNetwork;
use crate::threshold_control::FrameEvidence;
use crate::trial_gate::{KeyNodeEvidence, key_trial_score};
use nc_analysis::linear_threshold::LinearThreshold;

/// The search's settings (`planner.rd_split*`).
#[derive(Debug, Clone, Copy)]
struct RdSplit {
    deeper: u8,
    shallower: u8,
    shapes: ShapeReach,
    shape_gate: f32,
    skip_exit: bool,
    tt_margin: f32,
    /// `planner.rd_split_margin` (0: no gate); on key frames
    /// `planner.key_rd_margin`, for planned leaves trying the split.
    margin: f32,
    /// The margin for planned splits trying the leaf: `margin`, or on key
    /// frames `planner.key_rd_merge_margin` where it is set.
    merge_margin: f32,
    /// `planner.key_rd_gate` on key frames (0: off, the margins decide):
    /// the fitted trial gate's least score for a trial (`trial_gate.rs`).
    gate: u16,
    /// `planner.key_rd_leaf_first`.
    order: TrialOrder,
    /// The context gate's thresholds (`planner.rd_split_context`), `None`
    /// also on a superblock its refresh rotation searches ungated.
    context: Option<ContextGate>,
    /// Whether coded leaf sizes are recorded for the gate (it is on).
    record_depths: bool,
    /// `planner.rd_split_deepen_residual` (0: no gate).
    deepen_residual: f32,
    /// The post-trial stop (`planner.rd_split_stop_*`), `None` while off.
    stop: Option<FirstTrialStop>,
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
    /// Whether the search reports the partition it chose.
    tree: TreeReport,
}

/// Which candidate of a planned split is coded first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrialOrder {
    /// The planned split, then the leaf.
    PlannedFirst,
    /// The leaf where it is tried, then the split under the leaf's cost as
    /// its bound (`planner.key_rd_leaf_first`).
    LeafFirst,
}

/// Whether a search reports its chosen partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TreeReport {
    /// No: the coded state is the result.
    None,
    /// Yes: `planner.key_rd_leaf_pass` codes it again with the leaf search.
    ForLeafPass,
    /// Yes: the partition transplant log files it (`transplant.rs`).
    ForRecord,
}

impl TreeReport {
    fn wanted(self) -> bool {
        self != Self::None
    }
}

/// What the search of one node reports.
struct Searched {
    /// The node's coded cost, where the caller asked for it.
    cost: f32,
    /// The partition chosen (`RdSplit::tree` only).
    tree: Option<Partition>,
}

impl RdSplit {
    /// An inter search with no candidates beyond the plan can use the
    /// ordinary walk when it has no depth map to update. A planned shape
    /// can still offer a square even with all square alternatives gated.
    fn follows_plan<B: CodecBackend>(
        self,
        plan: &Partition,
        sb_log2: u8,
        params: &PlannerParams,
    ) -> bool {
        let squares_closed =
            (self.deeper == 0 && self.shallower == 0) || (self.margin > 0.0 && self.margin <= 1.0);
        !self.record_depths
            && squares_closed
            && (!B::RECTANGULAR_LEAVES
                || (self.shapes == ShapeReach::Planned
                    && (params.rect_min_log2.round() >= f32::from(sb_log2) || !plan.has_shape())))
    }

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
            merge_margin: params.rd_split_margin,
            gate: 0,
            order: TrialOrder::PlannedFirst,
            frame_period: params.rd_split_frame_period.round().max(1.0) as u64,
            deepen_residual: params.rd_split_deepen_residual,
            stop: FirstTrialStop::of(params),
            merge_skipped: params.rd_split_merge_skipped >= 0.5,
            record_depths: params.rd_split_context >= 0.5,
            context: (params.rd_split_context >= 0.5).then(|| ContextGate {
                deepen: params.rd_split_deepen_context.round() as i8,
                merge: params.rd_split_merge_context.round() as i8,
                period: params.rd_split_context_period.round() as u64,
                shift: ContextShift::of(params),
            }),
            min_log2: params.inter_min_log2.round() as u8,
            tail_satd: params.rd_split_tail_satd,
            tree: TreeReport::None,
        })
    }

    /// The key-frame search's settings, or `None` while
    /// `planner.key_rd_split` is off: the depth window and the margin gate
    /// (on the key plan's margins), since every other gate of the inter
    /// search reads motion, residual or co-located leaves a key frame does
    /// not have.
    /// `params` are the key frame's, already at its key interval
    /// (`key_interval::key_search_params`).
    fn of_key(params: &PlannerParams) -> Option<Self> {
        (params.key_rd_split >= 0.5).then(|| Self {
            deeper: params.key_rd_deeper.round() as u8,
            shallower: params.key_rd_shallower.round() as u8,
            shapes: if params.key_rd_shapes >= 0.5 {
                ShapeReach::Leaves
            } else {
                ShapeReach::Planned
            },
            shape_gate: 0.0,
            skip_exit: false,
            tt_margin: 0.0,
            margin: params.key_rd_margin,
            merge_margin: if params.key_rd_merge_margin > 0.0 {
                params.key_rd_merge_margin
            } else {
                params.key_rd_margin
            },
            gate: params.key_rd_gate.round() as u16,
            order: if params.key_rd_leaf_first >= 0.5 {
                TrialOrder::LeafFirst
            } else {
                TrialOrder::PlannedFirst
            },
            frame_period: 1,
            deepen_residual: 0.0,
            stop: None,
            merge_skipped: false,
            record_depths: false,
            context: None,
            min_log2: params.intra_min_log2.round() as u8,
            tail_satd: 0.0,
            tree: if params.key_rd_leaf_pass >= 0.5 {
                TreeReport::ForLeafPass
            } else {
                TreeReport::None
            },
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
    /// The thresholds' shifts by the node's features (`None`: fixed).
    shift: Option<ContextShift>,
}

/// The context gate's thresholds as linear functions of the node's features
/// (`planner.rd_split_{deepen,merge}_shift_*`, `linear_threshold.rs`). What
/// it changes: which planned nodes try their alternative, by the node's
/// size, the planner's confidence and its residual besides the depth
/// context. Why safe: all terms 0 is the fixed gate exactly (no shift is
/// built and no feature computed); a tuned shift is adopted only on the
/// held-out screens.
#[derive(Debug, Clone, Copy)]
struct ContextShift {
    deepen: Option<LinearThreshold<DEEPEN_FEATURES>>,
    merge: Option<LinearThreshold<MERGE_FEATURES>>,
}

/// Deepen features: neighbours, co-located, size, margin, residual.
const DEEPEN_FEATURES: usize = 5;
/// Merge features: neighbours, co-located, size, margin.
const MERGE_FEATURES: usize = 4;
/// The node size (log2) the size feature is measured from.
const SHIFT_SIZE_ORIGIN: i8 = 4;
/// ln(planner margin) is clamped to at most this.
const SHIFT_MARGIN_MAX: f32 = 2.0;
/// log2(residual per sample / λ) is clamped to +- this.
const SHIFT_RESIDUAL_LEVELS: f32 = 4.0;

impl ContextShift {
    fn of(params: &PlannerParams) -> Option<Self> {
        let deepen = LinearThreshold::new(
            params.rd_split_deepen_shift_bias,
            [
                params.rd_split_deepen_shift_neighbours,
                params.rd_split_deepen_shift_co_located,
                params.rd_split_deepen_shift_size,
                params.rd_split_deepen_shift_margin,
                params.rd_split_deepen_shift_residual,
            ],
        );
        let merge = LinearThreshold::new(
            params.rd_split_merge_shift_bias,
            [
                params.rd_split_merge_shift_neighbours,
                params.rd_split_merge_shift_co_located,
                params.rd_split_merge_shift_size,
                params.rd_split_merge_shift_margin,
            ],
        );
        (deepen.is_some() || merge.is_some()).then_some(Self { deepen, merge })
    }
}

/// Whether the context gate's shifts or the post-trial stop read the
/// planner's margins, which it then records (`encoder.rs`).
pub(super) fn context_reads_margins(params: &PlannerParams) -> bool {
    (params.rd_split_context >= 0.5
        && (params.rd_split_deepen_shift_margin != 0.0
            || params.rd_split_merge_shift_margin != 0.0))
        || params.rd_split_stop_margin != 0.0
        || stop_reads_inputs(params)
}

/// Whether the stop network runs, which reads the planner's trial bytes at
/// each node (the planner then records them, `encoder.rs`).
pub(super) fn stop_reads_inputs(params: &PlannerParams) -> bool {
    StopNetwork::of(params).is_some()
}

/// The post-trial stop (`planner.rd_split_stop_*`): once a planned inter
/// node's planned candidate is coded, its alternatives (the ones the gates
/// admitted) are skipped where `bias + w · features > 0`, a linear score on
/// that coding's outcome ([`StopEvidence`]). What it skips: trials
/// after a planned candidate whose own coding already says the plan holds
/// (libaom's post-trial partition pruners read the same: the coded
/// candidate's rate, distortion and coefficients). Why safe: all terms 0
/// never stops (today's search exactly); it only removes trials, and a
/// tuned setting is adopted only on the held-out screens. The stop network
/// (`stop_value.rs`, `planner.rd_split_stop_net_*`) adds a learned term to
/// the score, on all the search knows at the node; its terms 0 leave the
/// linear score exactly.
#[derive(Debug, Clone, Copy)]
struct FirstTrialStop {
    linear: Option<LinearThreshold<STOP_FEATURES>>,
    network: Option<StopNetwork>,
}

/// [`stop_features`]' count.
const STOP_FEATURES: usize = 10;
/// Samples of the 8x8 block the per-area features are counted per.
const STOP_UNIT_AREA: f32 = 64.0;

impl FirstTrialStop {
    fn of(params: &PlannerParams) -> Option<Self> {
        let linear = LinearThreshold::new(
            params.rd_split_stop_bias,
            [
                params.rd_split_stop_cost,
                params.rd_split_stop_bits,
                params.rd_split_stop_distortion,
                params.rd_split_stop_skip,
                params.rd_split_stop_nonzero,
                params.rd_split_stop_planned_split,
                params.rd_split_stop_size,
                params.rd_split_stop_margin,
                params.rd_split_stop_neighbours,
                params.rd_split_stop_co_located,
            ],
        );
        let network = StopNetwork::of(params);
        (linear.is_some() || network.is_some()).then_some(Self { linear, network })
    }

    fn stops(self, dsp: &Dsp, outcome: &StopEvidence) -> bool {
        let linear = self
            .linear
            .map_or(0.0, |linear| linear.at(&stop_features(outcome)));
        let network = self
            .network
            .map_or(0.0, |network| network.score(dsp, &StopInputs::new(outcome)));
        linear + network > 0.0
    }
}

/// The post-trial stop's linear features of the planned candidate's coding:
/// ln(1 + cost per sample / λ), ln(1 + bits per 8x8), ln(1 + distortion per
/// sample / λ) (0 without parts), skip, ln(1 + levels per 8x8), planned
/// split, size, ln margin, and the clamped context levels.
fn stop_features(e: &StopEvidence) -> [f32; STOP_FEATURES] {
    let per_lambda = |v: f32| (v / e.area / e.lambda.max(f32::MIN_POSITIVE)).ln_1p();
    let per_unit = |v: f32| (v * STOP_UNIT_AREA / e.area).ln_1p();
    let (bits, distortion) = e
        .parts
        .map_or((0.0, 0.0), |p| (per_unit(p.bits), per_lambda(p.distortion)));
    let nonzero_levels = e.summary.map_or(0, |s| s.nonzero_levels);
    let [neighbours, co_located] = clamped(e.context);
    [
        per_lambda(e.cost),
        bits,
        distortion,
        f32::from(u8::from(e.skip)),
        per_unit(f32::from(nonzero_levels)),
        f32::from(u8::from(e.planned_split)),
        f32::from(e.log2 as i8 - SHIFT_SIZE_ORIGIN),
        e.margin.max(1.0).ln().min(SHIFT_MARGIN_MAX),
        neighbours,
        co_located,
    ]
}

/// What a shifted context gate reads at a node besides its depth context.
#[derive(Debug, Clone, Copy)]
struct NodeFeatures {
    /// The node's log2 size minus [`SHIFT_SIZE_ORIGIN`].
    size: f32,
    /// ln of the planner's margin, in `[0, SHIFT_MARGIN_MAX]` (0 unknown).
    margin: f32,
    /// log2 of the planned leaf's residual variance per sample over λ,
    /// within +-[`SHIFT_RESIDUAL_LEVELS`] (0 where not a planned leaf).
    residual: f32,
}

/// Relative levels beyond this count as this (the gate's thresholds were
/// fitted on levels clamped so).
const CONTEXT_LEVELS: i8 = 2;

impl ContextGate {
    /// Whether a planned leaf in `context` tries the split; `features` are
    /// read only by a shifted threshold.
    fn deepens(self, context: NodeContext, features: impl FnOnce() -> NodeFeatures) -> bool {
        let level = context.neighbours.min(context.co_located);
        match self.shift.and_then(|shift| shift.deepen) {
            None => level <= self.deepen,
            Some(shift) => {
                let f = features();
                let [neighbours, co_located] = clamped(context);
                let threshold = f32::from(self.deepen)
                    + shift.at(&[neighbours, co_located, f.size, f.margin, f.residual]);
                f32::from(level) <= threshold
            }
        }
    }

    /// Whether a planned split in `context` tries the leaf.
    fn merges(self, context: NodeContext, features: impl FnOnce() -> NodeFeatures) -> bool {
        let [neighbours, co_located] = clamped(context);
        match self.shift.and_then(|shift| shift.merge) {
            None => neighbours + co_located >= f32::from(self.merge),
            Some(shift) => {
                let f = features();
                let threshold =
                    f32::from(self.merge) + shift.at(&[neighbours, co_located, f.size, f.margin]);
                neighbours + co_located >= threshold
            }
        }
    }
}

/// The depth context's levels clamped to +-[`CONTEXT_LEVELS`].
fn clamped(context: NodeContext) -> [f32; 2] {
    [context.neighbours, context.co_located]
        .map(|v| f32::from(v.clamp(-CONTEXT_LEVELS, CONTEXT_LEVELS)))
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

/// What the search's gates allow at one node (`Encoder::gates`), as the
/// label dump's bits (`labels::GATE_*`).
#[derive(Debug, Clone, Copy)]
struct Gates(u8);

impl Gates {
    fn of(allowed: [(bool, u8); 5]) -> Self {
        Self(
            allowed
                .iter()
                .filter(|(on, _)| *on)
                .fold(0, |bits, (_, bit)| bits | bit),
        )
    }
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
pub(super) struct EngineMark {
    block_offsets: usize,
    /// The label dump's leaf records (`labels.rs`), while it is on.
    label_leaves: usize,
    reuse: Option<usize>,
    evidence: FrameEvidence,
}

/// The engine's own state after a node's best candidate so far, put back
/// with the codec's kept coding.
#[derive(Default)]
struct EngineKept {
    block_offsets: Vec<(f64, f64)>,
    label_leaves: Vec<labels::LeafLabel>,
    reuse: Vec<CodedBlock>,
    evidence: FrameEvidence,
}

impl<B: CodecBackend> Encoder<B> {
    /// Whether the RD partition search may run on this frame: the backend
    /// has node trials and nothing constrains the leaves (no intra
    /// refresh, random decisions or spatial layers).
    pub(super) fn rd_split_applies(&self) -> bool {
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
            // A transplanted partition is coded as it is (`transplant.rs`).
            Some(_) if self.replays_partitions() => None,
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
            (Some(plan), Some(rd))
                if inter.is_some()
                    && self.labels.is_none()
                    && !self.records_partitions()
                    && rd.follows_plan::<B>(plan, sb.log2_size(), planner_params) =>
            {
                self.code_node(sb, Some(plan), pictures, inter);
            }
            (Some(plan), Some(rd)) => {
                if rd.record_depths {
                    self.depths
                        .fit(self.layer.coded_width, self.layer.coded_height);
                }
                if inter.is_none() && rd.tree.wanted() && self.leaf_rd_applies() {
                    self.search_then_code_leaves(sb, plan, pictures, rd);
                } else {
                    let record = inter.is_some() && self.records_partitions();
                    let rd = RdSplit {
                        tree: if record {
                            TreeReport::ForRecord
                        } else {
                            TreeReport::None
                        },
                        ..rd
                    };
                    let searched =
                        self.search_node(sb, Planned::Plan(plan), pictures, inter, (rd, false));
                    if let Some(tree) = searched.tree.filter(|_| record) {
                        self.record_partition(&sb, &tree);
                    }
                }
            }
            (plan, _) => {
                if let Some(plan) = plan.filter(|_| inter.is_some()) {
                    self.record_partition(&sb, plan);
                }
                self.code_node(sb, plan, pictures, inter);
            }
        }
    }

    /// Key-frame node `node` by the partition search without the leaf
    /// search, then coded once more as the chosen partition with it
    /// (`planner.key_rd_leaf_pass`, module docs). A node across the picture
    /// edge is the forced split of its children, each handled so.
    fn search_then_code_leaves(
        &mut self,
        node: Block,
        plan: &Partition,
        pictures: FramePictures<'_>,
        rd: RdSplit,
    ) {
        let (width, height) = (self.layer.coded_width, self.layer.coded_height);
        if node.x >= width || node.y >= height {
            return; // outside: no syntax
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
            let Partition::Split(plans) = plan else {
                // The key plan splits the picture edge; anything else is
                // searched in one pass.
                let rd = RdSplit {
                    tree: TreeReport::None,
                    ..rd
                };
                self.search_node(node, Planned::Plan(plan), pictures!(), None, (rd, false));
                return;
            };
            self.backend.write_partition(&node, true);
            for (child, plan) in node.quadrants().into_iter().zip(plans.iter()) {
                self.search_then_code_leaves(child, plan, pictures!(), rd);
            }
            return;
        }
        self.backend.begin_node_trial(&node, recon);
        let mark = self.engine_mark();
        self.leaf_search = LeafSearch::Suspended;
        let searched = self.search_node(node, Planned::Plan(plan), pictures!(), None, (rd, false));
        self.leaf_search = LeafSearch::Active;
        self.backend.undo_node_trial(recon);
        self.undo_engine(&mark);
        let tree = searched.tree.expect("the leaf pass asks for the tree");
        self.code_node(node, Some(&tree), pictures!(), None);
        self.backend.end_node_trial();
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
    ) -> Searched {
        let (width, height) = (self.layer.coded_width, self.layer.coded_height);
        if node.x >= width || node.y >= height {
            // Outside: no syntax.
            return Searched {
                cost: 0.0,
                tree: rd.tree.wanted().then_some(Partition::Leaf(None)),
            };
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
            let (cost, children) = split.expect("an unbounded split is never abandoned");
            return Searched {
                cost,
                tree: children.map(Partition::Split),
            };
        }
        let candidates = self.candidates(&node, planned, frame, rd);
        // The label dump (`labels.rs`) also codes the candidates the
        // preset's gates leave out; they are recorded, never chosen.
        let extras = self
            .labels
            .as_ref()
            .is_some_and(|l| l.records_nodes())
            .then(|| self.ungated_extras(&node, planned, frame, rd, &candidates));
        let full = extras.is_some();
        let preset_count = candidates.len;
        if preset_count == 1 && !cost_needed && !full {
            let choice = candidates.list[0];
            let (coded, children) = self.code_candidate(
                node,
                choice,
                planned,
                pictures!(),
                frame,
                (rd, f32::INFINITY),
            );
            debug_assert!(coded, "the only candidate is unbounded");
            self.record_leaves(&node, choice, rd);
            return Searched {
                cost: 0.0,
                tree: rd.tree.wanted().then(|| candidate_tree(choice, children)),
            };
        }
        let count = preset_count + extras.as_ref().map_or(0, |e| e.len);
        // Candidate `i` and whether the preset codes it (`false`: an extra).
        let at = |i: usize| match &extras {
            Some(extras) if i >= preset_count => (extras.list[i - preset_count], false),
            _ => (candidates.list[i], true),
        };
        if count == 1 {
            self.backend.begin_node_pricing(&node, recon);
        } else {
            self.backend.begin_node_trial(&node, recon);
        }
        let label = self.open_node_label(&node, planned, candidates.as_slice()[0], (rd, frame));
        let mark = self.engine_mark();
        let mut kept = EngineKept::default();
        let mut best: Option<(f32, Candidate)> = None;
        // The chosen split's partitions (`RdSplit::tree`).
        let mut best_children = None;
        // Whether the coded state is the best candidate's.
        let mut best_is_coded = false;
        // The binary shapes' costs (horizontal, vertical), for the ternary
        // rule.
        let mut binary = [f32::INFINITY; 2];
        // Whether every leaf the planned split coded is a skip
        // (`planner.rd_split_merge_skipped`; true until it is coded).
        let tally = self.rd_leaf_tally;
        let mut split_all_skipped = true;
        // Whether the preset stopped here (the skip exit): the dump codes
        // the rest, the preset's choice is made.
        let mut stopped = false;
        for i in 0..count {
            let (candidate, preset) = at(i);
            let bound = best.as_ref().map_or(f32::INFINITY, |b| b.0);
            let admitted =
                preset && !pruned(candidate, rd, bound, binary, i > 0 && !split_all_skipped);
            let eligible = admitted && !stopped;
            if admitted {
                self.record_admitted(label, candidate);
            }
            if !eligible && !full {
                continue;
            }
            // The dump codes every candidate in full.
            if i > 0 {
                self.backend.undo_node_trial(recon);
                self.undo_engine(&mark);
            }
            let started = self.label_mark();
            // The preset's own candidates are coded as the preset codes them,
            // bound and all, so every side effect the search keeps (the second
            // reference's usage, the depth map, backend state outside the node
            // trial) is the preset's. A candidate the preset abandons at its
            // bound is recorded so, its bound a lower bound of its cost: coding
            // it on in full left state behind (AV2 slow, 2026-11-22). The dump's
            // extra codings (candidates the preset leaves out) put the side
            // effects back after them.
            let side = (!eligible).then(|| self.dump_side_state(&node));
            let bound_for = |eligible: bool| if eligible { bound } else { f32::INFINITY };
            let (coded, children) = self.code_candidate(
                node,
                candidate,
                planned,
                pictures!(),
                frame,
                (rd, bound_for(eligible)),
            );
            if let Some(side) = side {
                self.restore_dump_side_state(&node, &side);
            }
            if full && eligible && !coded {
                self.record_abandoned(label, candidate, (bound, started));
            }
            best_is_coded = false;
            if i == 0 && candidate == Candidate::Split {
                let now = self.rd_leaf_tally;
                split_all_skipped = now.skipped - tally.skipped == now.coded - tally.coded;
            }
            if !coded {
                continue;
            }
            let cost = self.backend.node_trial_cost(source, recon);
            let parts = label.and_then(|_| self.backend.node_trial_parts(source, recon));
            let summary = label.and_then(|_| self.backend.node_trial_summary());
            self.record_candidate(
                label,
                candidate,
                (cost, started),
                (eligible, parts, summary),
            );
            if i == 0 {
                self.record_first_candidate(label, &mark);
            }
            if !eligible {
                continue;
            }
            if let Candidate::Shape(shape) = candidate
                && shape.part_count() == 2
            {
                binary[usize::from(!shape.is_horizontal())] = cost;
            }
            if cost >= bound {
                continue;
            }
            best = Some((cost, candidate));
            best_children = children;
            best_is_coded = true;
            let skip_exit = i == 0
                && candidate == Candidate::Leaf
                && rd.skip_exit
                && self.backend.last_cu_skipped();
            let stop_point = i == 0 && frame.is_some() && !matches!(planned, Planned::Below(_));
            let evidence = (stop_point && (rd.stop.is_some() || label.is_some()))
                .then(|| self.stop_evidence(&node, planned, cost, source, recon));
            if let Some(evidence) = &evidence {
                self.record_stop_inputs(label, evidence);
            }
            let post_trial_stop = rd
                .stop
                .zip(evidence.as_ref())
                .is_some_and(|(stop, evidence)| stop.stops(&self.dsp, evidence));
            if skip_exit || post_trial_stop {
                stopped = true;
                if !full {
                    break;
                }
            }
            if i + 1 < count {
                self.backend.keep_node_candidate(recon);
                self.keep_engine(&mark, &mut kept);
            }
        }
        let (cost, candidate) = best.expect("the first candidate is unbounded");
        self.close_node_label(label, candidate);
        if !best_is_coded {
            self.backend.restore_node_candidate(recon);
            self.restore_engine(&mark, &kept);
        }
        self.backend.end_node_trial();
        self.record_leaves(&node, candidate, rd);
        let tree = rd
            .tree
            .wanted()
            .then(|| candidate_tree(candidate, best_children));
        Searched { cost, tree }
    }

    /// The fitted key-trial gate's verdict on a planned key node
    /// (`planner.key_rd_gate`, `trial_gate.rs`): whether it tries its
    /// alternative; `None` where the gate is off or the node was not planned.
    fn fitted_gate(&self, node: &Block, planned: Planned<'_>, rd: RdSplit) -> Option<bool> {
        let margins = self.plan_margins.as_ref().filter(|_| rd.gate > 0)?;
        let planned_split = match planned {
            Planned::Plan(Partition::Split(_)) => true,
            Planned::Plan(_) => false,
            Planned::Below(_) => return None,
        };
        let evidence = KeyNodeEvidence {
            margin: margins.get(node),
            planned_split,
            log2: node.log2_size(),
            estimate: margins.estimate(node),
            area: node.area() as f32,
            lambda: self.backend.lambda(),
        };
        Some(key_trial_score(&evidence) >= rd.gate)
    }

    /// Code `candidate` of `node` under `bound`; returns whether it was coded
    /// in full (a split or shape stops at the bound) and a split's children.
    #[allow(clippy::type_complexity, reason = "the search's own result pair")]
    fn code_candidate(
        &mut self,
        node: Block,
        candidate: Candidate,
        planned: Planned<'_>,
        pictures: FramePictures<'_>,
        frame: Option<&FrameInputs<'_>>,
        (rd, bound): (RdSplit, f32),
    ) -> (bool, Option<Box<[Partition; 4]>>) {
        match candidate {
            Candidate::Leaf => {
                self.code_leaf_candidate(node, planned, pictures, frame);
                (true, None)
            }
            Candidate::Split => {
                match self.code_split(node, planned, pictures, frame, rd, (bound, false)) {
                    Some((_, children)) => (true, children),
                    None => (false, None),
                }
            }
            Candidate::Shape(shape) => (
                self.code_shape(node, shape, planned, pictures, frame, (rd.tail_satd, bound)),
                None,
            ),
        }
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
        let pre = frame.and_then(|frame| frame.pre);
        let static_node = !matches!(planned, Planned::Below(1..))
            && pre.is_some_and(|pre| is_static::<B>(pre, node));
        let (leaf, split) = match planned {
            Planned::Plan(Partition::Leaf(_)) => {
                candidates.push(Candidate::Leaf);
                let split = !static_node
                    && can_split
                    && rd.deeper > 0
                    && self.margin_allows(node, planned, rd, rd.margin)
                    && rd.context.is_none_or(|gate| {
                        gate.deepens(self.depths.of(node), || self.node_features(node, planned))
                    })
                    && self.residual_deepens(node, planned, rd);
                (true, split)
            }
            Planned::Plan(Partition::Shaped(shape, _)) => {
                candidates.push(Candidate::Shape(*shape));
                (can_leaf, can_split && rd.deeper > 0)
            }
            Planned::Plan(plan @ Partition::Split(_)) => {
                let leaf = !static_node
                    && can_leaf
                    && rd.shallower > 0
                    && self.margin_allows(node, planned, rd, rd.merge_margin)
                    && log2 - plan.smallest_side_log2(log2) <= rd.shallower
                    && rd.context.is_none_or(|gate| {
                        gate.merges(self.depths.of(node), || self.node_features(node, planned))
                    });
                if !(leaf && rd.order == TrialOrder::LeafFirst) {
                    candidates.push(Candidate::Split);
                }
                (leaf, true)
            }
            Planned::Below(levels) => (can_leaf, can_split && levels < rd.deeper),
        };
        if static_node {
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
        // The backend declares its rectangles (`CodecBackend::
        // RECTANGULAR_LEAVES`); on inter frames the motion gate may keep
        // them out, key frames (`planner.key_rd_shapes`) try them all.
        if shapes && B::RECTANGULAR_LEAVES {
            for shape in self.legal_shapes(node).iter() {
                let parts = shape.parts(node);
                let gated = frame.is_some_and(|frame| {
                    rd.shape_gate > 0.0
                        && !motion_separates(
                            frame.motion,
                            &parts[..shape.part_count()],
                            rd.shape_gate,
                        )
                });
                if !gated {
                    candidates.push(Candidate::Shape(shape));
                }
            }
        }
        candidates
    }

    /// The gates' verdicts at `node` under `rd`.
    fn gates(&self, node: &Block, planned: Planned<'_>, rd: RdSplit) -> Gates {
        // The fitted key-trial gate replaces both margins where it is set.
        let (deepen, merge) = match self.fitted_gate(node, planned, rd) {
            Some(trial) => (trial, trial),
            None => (
                self.within_margin(node, rd.margin),
                self.within_margin(node, rd.merge_margin),
            ),
        };
        // The context gate (module docs).
        let context = rd.context.map(|gate| (gate, self.depths.of(node)));
        Gates::of([
            (deepen, labels::GATE_MARGIN_DEEPEN),
            (merge, labels::GATE_MARGIN_MERGE),
            (
                context.is_none_or(|(gate, context)| {
                    gate.deepens(context, || self.node_features(node, planned))
                }),
                labels::GATE_CONTEXT_DEEPEN,
            ),
            (
                self.residual_deepens(node, planned, rd),
                labels::GATE_RESIDUAL_DEEPEN,
            ),
            (
                context.is_none_or(|(gate, context)| {
                    gate.merges(context, || self.node_features(node, planned))
                }),
                labels::GATE_CONTEXT_MERGE,
            ),
        ])
    }

    /// Only eligible planned nodes need the verdict for their other choice.
    fn margin_allows(&self, node: &Block, planned: Planned<'_>, rd: RdSplit, margin: f32) -> bool {
        self.fitted_gate(node, planned, rd)
            .unwrap_or_else(|| self.within_margin(node, margin))
    }

    /// The planner's stored margin is at least 1, including unknown nodes.
    fn within_margin(&self, node: &Block, margin: f32) -> bool {
        margin <= 0.0
            || (margin > 1.0
                && self
                    .plan_margins
                    .as_ref()
                    .is_none_or(|margins| margins.get(node) < margin))
    }

    /// The planned candidate's coding of `node` at `cost`, and what the
    /// search knew of the node, for the post-trial stop and its network.
    fn stop_evidence(
        &mut self,
        node: &Block,
        planned: Planned<'_>,
        cost: f32,
        source: &Picture,
        recon: &Picture,
    ) -> StopEvidence {
        let margins = self.plan_margins.as_ref();
        StopEvidence {
            log2: node.log2_size(),
            area: node.area() as f32,
            lambda: self.backend.lambda(),
            cost,
            parts: self.backend.node_trial_parts(source, recon),
            skip: self.backend.last_cu_skipped(),
            summary: self.backend.node_trial_summary(),
            planned_split: matches!(planned, Planned::Plan(Partition::Split(_))),
            margin: margins.map_or(1.0, |margins| margins.get(node)),
            context: self.depths.of(node),
            trial: margins.and_then(|margins| margins.inputs(node)),
        }
    }

    /// The features a shifted context gate reads at `node`.
    fn node_features(&self, node: &Block, planned: Planned<'_>) -> NodeFeatures {
        let margin = self
            .plan_margins
            .as_ref()
            .map_or(1.0, |margins| margins.get(node));
        let residual = match planned {
            Planned::Plan(Partition::Leaf(Some(stats))) => {
                let area = (node.width() * node.height()) as f32;
                let per_sample = n_variance::<B>(stats.search, node) as f32 / area;
                (per_sample / self.backend.lambda())
                    .log2()
                    .clamp(-SHIFT_RESIDUAL_LEVELS, SHIFT_RESIDUAL_LEVELS)
            }
            _ => 0.0,
        };
        NodeFeatures {
            size: f32::from(node.log2_size() as i8 - SHIFT_SIZE_ORIGIN),
            margin: margin.max(1.0).ln().min(SHIFT_MARGIN_MAX),
            residual,
        }
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
    /// coded quadrants cost `bound` (the rest is left uncoded), else the
    /// cost and, for the leaf pass, the quadrants' chosen partitions.
    fn code_split(
        &mut self,
        node: Block,
        planned: Planned<'_>,
        pictures: FramePictures<'_>,
        frame: Option<&FrameInputs<'_>>,
        rd: RdSplit,
        (bound, return_cost): (f32, bool),
    ) -> Option<(f32, Option<Box<[Partition; 4]>>)> {
        let FramePictures {
            source,
            reference,
            secondary,
            recon,
        } = pictures;
        self.backend.write_partition(&node, true);
        let mut spent = 0.0;
        let mut trees = rd
            .tree
            .wanted()
            .then(|| Box::new(std::array::from_fn(|_| Partition::Leaf(None))));
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
            let searched =
                self.search_node(child, child_planned, pictures, frame, (rd, cost_needed));
            spent += searched.cost;
            if let (Some(trees), Some(tree)) = (trees.as_mut(), searched.tree) {
                trees[i] = tree;
            }
        }
        Some((spent, trees))
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
            let price = bound.is_finite() && i + 1 < part_count;
            if price {
                self.backend.begin_node_pricing(&part, pictures.recon);
            }
            self.code_leaf(part, part_stats.as_ref(), pictures, frame);
            self.tally_leaf();
            if price {
                spent += self.backend.node_trial_cost(source, recon);
                self.backend.end_node_trial();
            }
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

    pub(super) fn engine_mark(&self) -> EngineMark {
        EngineMark {
            block_offsets: self.block_offsets.len(),
            label_leaves: self.labels.as_ref().map_or(0, |l| l.current.leaves.len()),
            reuse: self.layer.reuse.as_ref().map(ReuseTerm::coded_count),
            evidence: self.decision_scratch.evidence,
        }
    }

    fn keep_engine(&self, mark: &EngineMark, kept: &mut EngineKept) {
        kept.block_offsets.clear();
        kept.block_offsets
            .extend_from_slice(&self.block_offsets[mark.block_offsets..]);
        kept.label_leaves.clear();
        if let Some(labeller) = self.labels.as_ref() {
            kept.label_leaves
                .extend_from_slice(&labeller.current.leaves[mark.label_leaves..]);
        }
        kept.reuse.clear();
        if let (Some(term), Some(count)) = (self.layer.reuse.as_ref(), mark.reuse) {
            kept.reuse.extend_from_slice(term.coded_after(count));
        }
        kept.evidence = self.decision_scratch.evidence;
    }

    fn restore_engine(&mut self, mark: &EngineMark, kept: &EngineKept) {
        self.block_offsets.truncate(mark.block_offsets);
        self.block_offsets.extend_from_slice(&kept.block_offsets);
        if let Some(labeller) = self.labels.as_mut() {
            labeller.current.leaves.truncate(mark.label_leaves);
            labeller
                .current
                .leaves
                .extend_from_slice(&kept.label_leaves);
        }
        if let (Some(term), Some(count)) = (self.layer.reuse.as_mut(), mark.reuse) {
            term.replace_coded_after(count, &kept.reuse);
        }
        self.decision_scratch.evidence = kept.evidence;
    }

    pub(super) fn undo_engine(&mut self, mark: &EngineMark) {
        self.block_offsets.truncate(mark.block_offsets);
        if let Some(labeller) = self.labels.as_mut() {
            labeller.current.leaves.truncate(mark.label_leaves);
        }
        if let (Some(term), Some(count)) = (self.layer.reuse.as_mut(), mark.reuse) {
            term.replace_coded_after(count, &[]);
        }
        self.decision_scratch.evidence = mark.evidence;
    }
}

/// The search's state outside the node trial (`Encoder::dump_side_state`).
struct DumpSide {
    usage: Option<crate::second_reference::Usage>,
    depths: crate::depth_context::DepthSnapshot,
}

/// The selected tree when the caller needs to walk it again.
fn candidate_tree(candidate: Candidate, children: Option<Box<[Partition; 4]>>) -> Partition {
    match (candidate, children) {
        (Candidate::Split, Some(children)) => Partition::Split(children),
        (Candidate::Shape(shape), _) => Partition::Shaped(shape, Box::new([None; MAX_SHAPE_PARTS])),
        _ => Partition::Leaf(None),
    }
}

/// The label dump's slot of `candidate` (`labels.rs`).
fn label_slot(candidate: Candidate) -> u8 {
    match candidate {
        Candidate::Leaf => labels::SLOT_LEAF,
        Candidate::Split => labels::SLOT_SPLIT,
        Candidate::Shape(shape) => labels::shape_slot(shape),
    }
}

/// The label dump's hooks into the search (`labels.rs`): each does nothing
/// while the dump is off.
impl<B: CodecBackend> Encoder<B> {
    /// The candidates of `node` the preset's gates leave out of `preset`:
    /// those of the same depth window with no margin, fitted, context, residual or
    /// motion gate, and every shape where a leaf is tried.
    fn ungated_extras(
        &self,
        node: &Block,
        planned: Planned<'_>,
        frame: Option<&FrameInputs<'_>>,
        rd: RdSplit,
        preset: &Candidates,
    ) -> Candidates {
        let ungated = RdSplit {
            margin: 0.0,
            merge_margin: 0.0,
            context: None,
            deepen_residual: 0.0,
            shape_gate: 0.0,
            gate: 0,
            shapes: ShapeReach::Leaves,
            ..rd
        };
        let mut extras = Candidates {
            list: [Candidate::Leaf; MAX_CANDIDATES],
            len: 0,
        };
        for &candidate in self.candidates(node, planned, frame, ungated).as_slice() {
            if !preset.as_slice().contains(&candidate) {
                extras.push(candidate);
            }
        }
        extras
    }

    /// What the search keeps of a coding beyond the node trial's undo (the
    /// second reference's usage, `node`'s cells of the depth map), saved
    /// before a coding the preset would not make.
    fn dump_side_state(&self, node: &Block) -> DumpSide {
        DumpSide {
            usage: self
                .layer
                .second
                .as_ref()
                .map(crate::second_reference::SecondReference::usage),
            depths: self.depths.save(node),
        }
    }

    /// Put back what [`Self::dump_side_state`] saved.
    fn restore_dump_side_state(&mut self, node: &Block, side: &DumpSide) {
        if let (Some(second), Some(usage)) = (self.layer.second.as_mut(), side.usage) {
            second.set_usage(usage);
        }
        self.depths.restore(node, &side.depths);
    }

    /// The dump's clock and work counters (0 while it is off).
    pub(super) fn label_mark(&self) -> labels::LabelMark {
        self.labels.as_ref().map_or_else(Default::default, |l| {
            l.mark(self.decision_scratch.search_positions())
        })
    }

    /// Open the record of searched `node`, whose preset tries `first` first.
    /// Its record holds the planner's margin, the depth context and the
    /// gates' verdicts under the preset's `rd`.
    fn open_node_label(
        &mut self,
        node: &Block,
        planned: Planned<'_>,
        first: Candidate,
        (rd, frame): (RdSplit, Option<&FrameInputs<'_>>),
    ) -> Option<u32> {
        if !self.labels.as_ref().is_some_and(|l| l.records_nodes()) {
            return None;
        }
        let (slot, margin) = match planned {
            Planned::Plan(_) => (
                label_slot(first),
                self.plan_margins.as_ref().map_or(f32::NAN, |m| m.get(node)),
            ),
            Planned::Below(_) => (labels::SLOT_NONE, f32::NAN),
        };
        let gates = self.gates(node, planned, rd).0;
        let plan_costs = match planned {
            Planned::Plan(_) => self.plan_margins.as_ref().and_then(|m| m.costs(node)),
            Planned::Below(_) => None,
        };
        let search = labels::node_search(frame.and_then(|f| f.motion), node);
        let pre = labels::node_pre(frame.and_then(|f| f.pre), frame.and_then(|f| f.still), node);
        let inputs = match planned {
            Planned::Plan(_) => self
                .plan_margins
                .as_ref()
                .and_then(|m| m.inputs(node))
                .map(|inputs| inputs.0),
            Planned::Below(_) => None,
        };
        let context = self.depths.of(node);
        let labeller = self.labels.as_mut()?;
        let final_context = labeller.depths.of(node);
        let index = labeller.open_node(node, slot);
        let record = labeller.node_mut(index);
        record.plan_costs = plan_costs.map_or([f32::NAN; 2], |(leaf, split)| [leaf, split]);
        record.search = search;
        record.pre = pre;
        record.final_context = [final_context.neighbours, final_context.co_located];
        record.plan_margin = margin;
        record.gates = gates;
        record.context = [context.neighbours, context.co_located];
        record.inputs = inputs;
        Some(index)
    }

    /// Note `candidate`'s coded `cost` and its time and work since `started`.
    fn record_candidate(
        &mut self,
        label: Option<u32>,
        candidate: Candidate,
        (cost, started): (f32, labels::LabelMark),
        (eligible, parts, summary): (
            bool,
            Option<crate::NodeTrialParts>,
            Option<crate::NodeTrialSummary>,
        ),
    ) {
        let now = self.label_mark();
        let (Some(index), Some(labeller)) = (label, self.labels.as_mut()) else {
            return;
        };
        let (ns, work, positions) = started.until(now);
        let record = labeller.node_mut(index);
        let slot = usize::from(label_slot(candidate));
        record.cost[slot] = cost;
        record.ns[slot] = ns;
        record.work[slot] = work;
        record.positions[slot] = positions;
        record.set_summary(slot, summary);
        if let Some(parts) = parts {
            record.distortion[slot] = parts.distortion;
            record.bits[slot] = parts.bits;
        }
        if eligible {
            record.eligible |= 1 << slot;
        }
    }

    /// Note that `candidate` was abandoned at the preset's `bound`: its cost
    /// is at least that.
    fn record_abandoned(
        &mut self,
        label: Option<u32>,
        candidate: Candidate,
        (bound, started): (f32, labels::LabelMark),
    ) {
        self.record_candidate(label, candidate, (bound, started), (true, None, None));
        if let (Some(index), Some(labeller)) = (label, self.labels.as_mut()) {
            labeller.node_mut(index).abandoned |= 1 << label_slot(candidate);
        }
    }

    /// Note what the first candidate's coded leaves (since `mark`) were.
    fn record_first_candidate(&mut self, label: Option<u32>, mark: &EngineMark) {
        let (Some(index), Some(labeller)) = (label, self.labels.as_mut()) else {
            return;
        };
        let leaves = &labeller.current.leaves[mark.label_leaves..];
        let count = leaves.len() as f32;
        let share = |test: fn(&labels::LeafLabel) -> bool| {
            if leaves.is_empty() {
                f32::NAN
            } else {
                leaves.iter().filter(|l| test(l)).count() as f32 / count
            }
        };
        let first = [
            count,
            share(|l| l.skip || l.mode == labels::MODE_STATIC_SKIP),
            share(|l| l.mode == labels::MODE_INTRA),
        ];
        labeller.node_mut(index).first = first;
    }

    /// Note that the preset's gates and bounds admitted `candidate` at
    /// `label`'s node (`NodeLabel::admitted`).
    fn record_admitted(&mut self, label: Option<u32>, candidate: Candidate) {
        if let (Some(index), Some(labeller)) = (label, self.labels.as_mut()) {
            labeller.node_mut(index).admitted |= 1 << label_slot(candidate);
        }
    }

    /// Record the stop network's inputs at `label`'s node.
    fn record_stop_inputs(&mut self, label: Option<u32>, evidence: &StopEvidence) {
        if let (Some(index), Some(labeller)) = (label, self.labels.as_mut()) {
            labeller.node_mut(index).stop_inputs = Some(StopInputs::new(evidence).0);
        }
    }

    /// Close `label`, which chose `chosen`.
    fn close_node_label(&mut self, label: Option<u32>, chosen: Candidate) {
        if let (Some(index), Some(labeller)) = (label, self.labels.as_mut()) {
            labeller.node_mut(index).chosen = label_slot(chosen);
            labeller.close_node();
        }
    }
}
