//! Decision labels for learned encoder models (the nn dataset), off by
//! default ([`Encoder::enable_labels`](crate::Encoder::enable_labels)).
//!
//! While on, the engine records per superblock what a model could read
//! before coding it (the context, cheap source and pre-analysis
//! statistics, a downscaled luma patch and the motion-compensated residual
//! patch at the search vector) and what coding it cost (its coded J and wall
//! time); and per node of the RD partition search (`rd_split.rs`), the coded
//! cost and time of every candidate, each coded in full: no bound and no
//! early exit. The search still chooses exactly as the preset would, among
//! the candidates the preset would have coded, so the stream is the
//! preset's own (`docs/analysis/nn-dataset-schema.md`; the dump tooling
//! checks it byte for byte).
//!
//! Records of a coding that rate control re-codes are dropped with it: only
//! the coding that is kept is reported. The front end drains the records
//! after every frame ([`Encoder::take_labels`](crate::Encoder::take_labels))
//! and writes them (`nc-cli` `--label-dump`).

use nc_analysis::MotionAnalysis;
pub use nc_analysis::cell_stats::{self, CELL_FEATURES, CellStats};
use nc_analysis::cell_stats::{CELL, cell_stats};
use nc_analysis::preanalysis::PreAnalysis;
use nc_analysis::still::StillMap;
use nc_common::{Picture, PlaneIndex};

use crate::NodeTrialSummary;
use crate::depth_context::DepthContext;
use crate::mv::Mv;
use crate::params::ParamOverrides;
use crate::partition::Block;
use crate::shape::PartitionShape;

/// Side of the luma patches, in patch cells: the superblock is averaged
/// down (or sampled up) to this many cells per side.
pub const PATCH_SIDE: usize = 32;
/// Cells of one patch.
pub const PATCH_CELLS: usize = PATCH_SIDE * PATCH_SIDE;
/// Candidate slots of a node: leaf, quad split, then each shape of
/// [`PartitionShape::ALL`] in its order.
pub const CANDIDATE_SLOTS: usize = 2 + PartitionShape::ALL.len();
/// The leaf's slot.
pub const SLOT_LEAF: u8 = 0;
/// The quad split's slot.
pub const SLOT_SPLIT: u8 = 1;
/// No candidate (a node without a planned choice).
pub const SLOT_NONE: u8 = u8::MAX;
/// Inputs of the trial value network per node (`trial_inputs.rs`).
pub const TRIAL_INPUTS: usize = crate::trial_inputs::TRIAL_INPUT_COUNT;
/// Inputs of the zero-gate network per superblock (`nc_analysis::gate_net`).
pub const GATE_INPUTS: usize = nc_analysis::gate_net::GATE_INPUT_COUNT;
/// Inputs of the stop network per node (`stop_inputs.rs`).
pub const STOP_INPUTS: usize = crate::stop_inputs::STOP_INPUT_COUNT;
/// [`NodeLabel::gates`]: the margin gate lets a planned leaf try the split.
pub const GATE_MARGIN_DEEPEN: u8 = 1;
/// The margin gate lets a planned split try the leaf.
pub const GATE_MARGIN_MERGE: u8 = 2;
/// The context gate lets a planned leaf try the split.
pub const GATE_CONTEXT_DEEPEN: u8 = 4;
/// The context gate lets a planned split try the leaf.
pub const GATE_CONTEXT_MERGE: u8 = 8;
/// The residual gate (`planner.rd_split_deepen_residual`) lets a planned leaf
/// try the split.
pub const GATE_RESIDUAL_DEEPEN: u8 = 16;
/// [`LeafLabel::mode`]: intra.
pub const MODE_INTRA: u8 = 0;
/// Inter, from the primary reference.
pub const MODE_INTER: u8 = 1;
/// Inter, from the secondary reference.
pub const MODE_SECONDARY: u8 = 2;
/// Inter, compound of both references.
pub const MODE_COMPOUND: u8 = 3;
/// Pre-analysis' static skip (coded at the frame quantizer, no decision).
pub const MODE_STATIC_SKIP: u8 = 4;
/// Inter, low-delay bi-prediction from the primary reference at two vectors.
pub const MODE_PRIMARY_PAIR: u8 = 5;
/// [`NodeLabel::nonzero`] and [`NodeLabel::tx_blocks`] of a slot that was
/// not coded, or whose backend keeps no [`NodeTrialSummary`].
pub const NO_COUNT: u16 = u16::MAX;
/// [`NodeLabel::max_tx`] likewise.
pub const NO_TX: u8 = u8::MAX;
/// Cells per side of the superblocks' motion grids ([`SuperblockFeatures`]).
pub const MOTION_GRID: usize = 8;
/// Pre-analysis and motion statistics are on an 8x8 grid.
const GRID_LOG2: u8 = 3;
/// Quarter samples per luma sample (encoder vectors).
const QUARTER: isize = 4;

/// What enables the dump.
#[derive(Debug, Clone)]
pub struct LabelConfig {
    /// A monotonic clock in nanoseconds, supplied by the front end so the
    /// engine itself never reads time.
    pub clock: fn() -> u64,
    /// The tools labelled per superblock. Each superblock is also coded once
    /// more as the preset codes it (tool slot 0), so every tool's J and time
    /// compare with the preset's own on the same state.
    pub tools: Vec<LabelTool>,
}

/// One tool labelled per superblock: the superblock is coded with these
/// parameters on top of the preset's (the tool flipped from the preset's
/// setting), the rest of the frame as the preset codes it.
#[derive(Debug, Clone)]
pub struct LabelTool {
    /// Its name in the shard.
    pub name: String,
    /// The overrides. Engine groups (`decision.`, `planner.`) apply to the
    /// superblock directly; anything else must be one of the backend's
    /// [`CodecBackend::SUPERBLOCK_PARAMS`](crate::CodecBackend::SUPERBLOCK_PARAMS).
    pub overrides: ParamOverrides,
}

/// Parameter groups the engine reads per superblock.
pub(crate) const ENGINE_PREFIXES: [&str; 2] = ["decision.", "planner."];

impl LabelTool {
    /// Its engine and backend overrides.
    pub(crate) fn split(&self) -> (ParamOverrides, ParamOverrides) {
        let mut engine = ParamOverrides::default();
        let mut backend = ParamOverrides::default();
        for name in self.overrides.names() {
            let value = self.overrides.get(name).expect("a listed name has a value");
            if ENGINE_PREFIXES.iter().any(|p| name.starts_with(p)) {
                engine.set(name, value);
            } else {
                backend.set(name, value);
            }
        }
        (engine, backend)
    }
}

/// The context of one coded frame.
#[derive(Debug, Clone, Copy)]
pub struct FrameContext {
    /// Input frame index.
    pub index: u64,
    /// Whether it is a key frame.
    pub key: bool,
    /// Its temporal layer.
    pub temporal_layer: u8,
    /// The frame quantizer, in the codec's own units.
    pub quantizer: u32,
    /// The mode-decision λ of the frame quantizer (engine cost units).
    pub lambda: f32,
    /// Pre-analysis' noise estimate (σ, luma samples; 0 without analysis).
    pub noise_sigma: f32,
    /// Coded picture size.
    pub width: u32,
    /// Coded picture size.
    pub height: u32,
    /// The codec's superblock size and smallest block (log2).
    pub superblock_log2: u8,
    /// The codec's smallest block (log2).
    pub min_block_log2: u8,
    /// Frames since the last key frame.
    pub key_distance: u32,
    /// Rate control at the frame's start (NaN without it): buffer fullness
    /// (share of the buffer), the frame's bit budget, the level quantizer
    /// and the cadence offset of its plan.
    pub rate: [f32; 4],
}

/// Cheap statistics of a superblock, known before it is coded.
#[derive(Debug, Clone, Copy, Default)]
pub struct SuperblockFeatures {
    /// Source luma mean.
    pub luma_mean: f32,
    /// Source luma variance (per sample).
    pub luma_variance: f32,
    /// Mean absolute horizontal and vertical luma gradient.
    pub gradient: [f32; 2],
    /// Gradient structure-tensor coherence (0 isotropic, 1 one direction).
    pub coherence: f32,
    /// Mean of the 8x8 luma variances and their spread (std).
    pub block_variance: [f32; 2],
    /// Share of 8x8 cells pre-analysis marks static.
    pub static_share: f32,
    /// Share of 8x8 cells the still map marks still, and their mean run.
    pub still: [f32; 2],
    /// Zero-vector SAD per sample against the last source frame.
    pub zero_sad: f32,
    /// The motion search's best 8x8 SAD per sample.
    pub best_sad: f32,
    /// Mean 8x8 motion vector (luma samples) and its spread (std of x, y).
    pub motion: [f32; 4],
    /// The search vector of the superblock (quarter samples).
    pub search_vector: [i16; 2],
    /// Residual at the search vector: mean square per sample.
    pub residual_energy: f32,
    /// Against the frame's second reference, where it carries one (0 otherwise):
    /// the residual at the search vector scaled to its distance, and at the
    /// zero vector, mean square per sample (format 7).
    pub second_residual: [f32; 2],
    /// The motion search's best 8x8 SAD per sample, pooled to a
    /// [`MOTION_GRID`] square over the superblock (0 without a search).
    pub block_sad: [[f32; MOTION_GRID]; MOTION_GRID],
    /// Its 8x8 vectors (luma samples), pooled the same way: x, y.
    pub block_mv: [[[f32; 2]; MOTION_GRID]; MOTION_GRID],
}

/// One superblock's record.
#[derive(Debug, Clone)]
pub struct SuperblockLabel {
    /// Position (luma samples) and size (log2).
    pub x: u32,
    /// Position (luma samples).
    pub y: u32,
    /// Whether it lies wholly inside the coded picture.
    pub inside: bool,
    /// Its features.
    pub features: SuperblockFeatures,
    /// Source luma, [`PATCH_SIDE`] square, row-major.
    pub patch: Box<[u8; PATCH_CELLS]>,
    /// Luma residual at the search vector, as `patch` (0 on key frames).
    pub residual: Box<[i16; PATCH_CELLS]>,
    /// Per-8x8-cell statistics (`nc_analysis::cell_stats`), the superblock's
    /// cells pooled (averaged) or placed top-left into a [`MOTION_GRID`]
    /// square: [row][column][feature].
    pub cell_stats: Box<CellGrid>,
    /// Coded J over the superblock (NaN where it is not priced: picture
    /// edges, codecs without node trials).
    pub cost: f32,
    /// Wall time of coding it, nanoseconds (with the dump's full search).
    pub ns: u64,
    /// Per tool slot (0: the preset; then [`LabelConfig::tools`]): the
    /// superblock's coded J and wall time coded so, the rest of the frame as
    /// the preset codes it (NaN and 0 where not priced).
    pub tool_cost: Vec<f32>,
    /// As `tool_cost`, nanoseconds.
    pub tool_ns: Vec<u64>,
    /// As `tool_cost`, in work units (`NodeLabel::work`).
    pub tool_work: Vec<u64>,
    /// As `tool_cost`, in search positions (`NodeLabel::positions`).
    pub tool_positions: Vec<u32>,
    /// The zero-gate network's inputs at the superblock, exactly as the
    /// motion search computed them (`nc_analysis::gate_net`; `None` on key
    /// frames and where the analysis superblock is not this one).
    pub gate_inputs: Option<[u8; GATE_INPUTS]>,
    /// The motion search's zero-vector SAD of the superblock, then its best
    /// total SAD at 8x8, 16x16, 32x32 and 64x64 (`SbStats::me_dist`): what
    /// searching gained over the zero vector, the gate network's label.
    pub gate_sads: [u32; 5],
}

/// A superblock's grid of cell statistics.
pub type CellGrid = [[CellStats; MOTION_GRID]; MOTION_GRID];

/// One searched node's record.
#[derive(Debug, Clone, Copy)]
pub struct NodeLabel {
    /// Index of its superblock in the frame's records.
    pub superblock: u32,
    /// Index of its parent node in the frame's records (-1: none).
    pub parent: i32,
    /// Position (luma samples) and size (log2).
    pub x: u32,
    /// Position.
    pub y: u32,
    /// Size (log2).
    pub log2: u8,
    /// The candidate the preset tries first (its plan), or [`SLOT_NONE`].
    pub planned: u8,
    /// The candidate chosen.
    pub chosen: u8,
    /// Coded J of each candidate slot (NaN: not coded).
    pub cost: [f32; CANDIDATE_SLOTS],
    /// Wall time of coding each slot (its whole subtree), nanoseconds.
    pub ns: [u64; CANDIDATE_SLOTS],
    /// Each slot's cost in its parts (`CodecBackend::node_trial_parts`;
    /// NaN where not coded or the backend does not split it): the weighted
    /// distortion and the bits, `cost = distortion + lambda · bits`.
    pub distortion: [f32; CANDIDATE_SLOTS],
    /// As `distortion`: the bits.
    pub bits: [f32; CANDIDATE_SLOTS],
    /// Bit `slot` set: the preset abandoned that candidate at its bound (the
    /// best cost so far); its `cost` is that bound, a lower bound of its cost.
    pub abandoned: u8,
    /// Deterministic work of coding each slot (its whole subtree), in
    /// [`Labeller::work`] units: the same on any machine, unlike `ns`.
    pub work: [u64; CANDIDATE_SLOTS],
    /// The other deterministic work counter of each slot's whole subtree:
    /// the block predictions its leaf decisions measured (the motion
    /// search's positions, `decision.rs` `DecisionScratch::search_positions`).
    pub positions: [u32; CANDIDATE_SLOTS],
    /// What each slot's coding holds ([`NodeTrialSummary`], read as its
    /// cost is): non-zero levels ([`NO_COUNT`]: not coded or not known).
    pub nonzero: [u16; CANDIDATE_SLOTS],
    /// As `nonzero`: transform blocks with coefficients.
    pub tx_blocks: [u16; CANDIDATE_SLOTS],
    /// As `nonzero`: the largest coded luma transform's log2 side
    /// ([`NO_TX`]: not known).
    pub max_tx: [u8; CANDIDATE_SLOTS],
    /// Bit `slot` set: the preset would have coded that candidate (the
    /// choice is the cheapest of these).
    pub eligible: u8,
    /// Bit `slot` set: the preset's gates and bounds admitted that
    /// candidate, whether or not a stop then skipped it (`eligible` without
    /// the skip exit and the post-trial stop): what a stop's training reads.
    pub admitted: u8,
    /// The planner's margin at the node: its rejected cost over its chosen
    /// one, at least 1 (1 where unknown; NaN below a planned leaf).
    pub plan_margin: f32,
    /// The preset's gates' verdicts (`GATE_*` bits; a gate that is off
    /// allows).
    pub gates: u8,
    /// The depth context (`depth_context.rs`): the left/above neighbours'
    /// and the co-located leaf log2 relative to the node's.
    pub context: [i8; 2],
    /// The trial value network's integer inputs at the node, exactly as the
    /// planner computes them (`trial_inputs.rs`), where the planner weighed
    /// it both ways (`None`: key frames, static nodes, nodes below a
    /// planned leaf).
    pub inputs: Option<[u8; TRIAL_INPUTS]>,
    /// The stop network's inputs at a planned inter node once its planned
    /// candidate was coded, exactly as the search computes them
    /// (`stop_inputs.rs`; `None` elsewhere).
    pub stop_inputs: Option<[u8; STOP_INPUTS]>,
    /// The planner's own cost estimates of the node as a leaf and as a
    /// split (engine cost units; NaN where it did not weigh both).
    pub plan_costs: [f32; 2],
    /// The motion search at the node's size (16, 32, 64: the aggregated
    /// fields; zero elsewhere and on key frames): vector (luma samples),
    /// whole SAD and the sum of its quarters' own SADs, per sample.
    pub search: [f32; 4],
    /// Pre-analysis over the node's 8x8 cells: static share, still share,
    /// zero-vector SAD per sample, mean and max 8x8 variance.
    pub pre: [f32; 5],
    /// What the first candidate coded (the plan) left, known the moment it
    /// is coded: its leaves' count, their skip share (the backend's skip CUs
    /// and static skips) and intra share.
    pub first: [f32; 3],
    /// The depth context from the dump's own map of final leaves, on every
    /// rung (`context` is the encoder's, kept only where its gate runs):
    /// left/above neighbours' and co-located leaf log2 minus the node's,
    /// from the superblocks coded before this one.
    pub final_context: [i8; 2],
}

/// One coded leaf of the frame's final coding.
#[derive(Debug, Clone, Copy)]
pub struct LeafLabel {
    /// Index of its superblock in the frame's records.
    pub superblock: u32,
    /// Position (luma samples) and size (log2 width, height).
    pub x: u32,
    /// Position.
    pub y: u32,
    /// log2 width and height.
    pub log2: [u8; 2],
    /// `MODE_*`.
    pub mode: u8,
    /// Whether the backend coded it as a skip CU.
    pub skip: bool,
    /// Its vector into the primary reference (quarter samples; 0 intra).
    pub mv: [i16; 2],
    /// Its block quantizer offset.
    pub qp_offset: i16,
    /// Block predictions its leaf decision measured (`NodeLabel::positions`;
    /// saturating).
    pub positions: u16,
}

/// One frame's records.
#[derive(Debug, Clone)]
pub struct FrameLabels {
    /// Its context.
    pub context: FrameContext,
    /// Its superblocks, in coding order.
    pub superblocks: Vec<SuperblockLabel>,
    /// Its searched nodes, parents before children.
    pub nodes: Vec<NodeLabel>,
    /// Its final coded leaves, in coding order.
    pub leaves: Vec<LeafLabel>,
}

impl FrameLabels {
    fn new(context: FrameContext) -> Self {
        Self {
            context,
            superblocks: Vec::new(),
            nodes: Vec::new(),
            leaves: Vec::new(),
        }
    }
}

impl Default for FrameContext {
    fn default() -> Self {
        Self {
            index: 0,
            key: false,
            temporal_layer: 0,
            quantizer: 0,
            lambda: 0.0,
            noise_sigma: 0.0,
            width: 0,
            height: 0,
            superblock_log2: 0,
            min_block_log2: 0,
            key_distance: 0,
            rate: [f32::NAN; 4],
        }
    }
}

/// The dump's state in the encoder.
#[derive(Debug)]
pub(crate) struct Labeller {
    pub config: LabelConfig,
    /// The coding in progress.
    pub current: FrameLabels,
    /// Frames finished and not yet drained.
    pub done: Vec<FrameLabels>,
    /// The nodes being searched, innermost last.
    pub open: Vec<u32>,
    /// The next frame's noise estimate (set before its coding).
    pub noise_sigma: f32,
    /// Work done so far, a deterministic cost counter: luma samples of every
    /// leaf decided and coded (`Encoder::code_leaf`) and of every extra
    /// coding of a leaf by the key-frame leaf search. Its weights against
    /// cycles are fitted once on a quiet machine.
    pub work: u64,
    /// While a tool trial codes a superblock: the search codes as the preset
    /// does and records no nodes.
    pub in_tool_trial: bool,
    /// The dump's own map of final leaf sizes, on every rung.
    pub depths: DepthContext,
    /// The backend overrides of the frame's regular and light superblocks,
    /// under a tool's backend overrides.
    pub backend_regular: ParamOverrides,
    /// As `backend_regular`, for light superblocks (`None`: none this frame).
    pub backend_light: Option<ParamOverrides>,
}

impl Labeller {
    pub fn new(config: LabelConfig) -> Self {
        Self {
            config,
            current: FrameLabels::new(FrameContext::default()),
            done: Vec::new(),
            open: Vec::new(),
            noise_sigma: 0.0,
            work: 0,
            in_tool_trial: false,
            depths: DepthContext::default(),
            backend_regular: ParamOverrides::default(),
            backend_light: None,
        }
    }

    /// Start a coding of a frame: whatever an earlier coding of it
    /// recorded is dropped.
    pub fn begin_coding(&mut self, context: FrameContext) {
        self.current = FrameLabels::new(context);
        self.open.clear();
        self.depths
            .fit(context.width as usize, context.height as usize);
        self.depths.clear_current();
    }

    /// The frame's kept coding is final.
    pub fn finish_frame(&mut self) {
        let context = self.current.context;
        self.done.push(std::mem::replace(
            &mut self.current,
            FrameLabels::new(context),
        ));
        self.depths.commit();
    }

    /// Note the leaves superblock `sb` finally coded in the dump's own depth
    /// map (`NodeLabel::final_context`).
    pub fn record_final_leaves(&mut self, sb: u32) {
        for leaf in self
            .current
            .leaves
            .iter()
            .rev()
            .take_while(|l| l.superblock == sb)
        {
            let side = leaf.log2[0].min(leaf.log2[1]);
            let (w, h) = (1usize << leaf.log2[0], 1usize << leaf.log2[1]);
            for dy in (0..h).step_by(1 << side) {
                for dx in (0..w).step_by(1 << side) {
                    let cell = Block::square(leaf.x as usize + dx, leaf.y as usize + dy, side);
                    self.depths.record(&cell, side);
                }
            }
        }
    }

    /// Whether the search records its nodes (and codes every candidate):
    /// not inside a tool trial.
    pub fn records_nodes(&self) -> bool {
        !self.in_tool_trial
    }

    /// Note `samples` luma samples of coding work.
    pub fn add_work(&mut self, samples: usize) {
        self.work += samples as u64;
    }

    /// The clock and the work counters now; `positions` is the engine's
    /// search position count (`DecisionScratch::search_positions`).
    pub fn mark(&self, positions: u64) -> LabelMark {
        LabelMark {
            ns: self.now(),
            work: self.work,
            positions,
        }
    }

    pub fn now(&self) -> u64 {
        (self.config.clock)()
    }

    /// Open a node record; returns its index.
    pub fn open_node(&mut self, node: &Block, planned: u8) -> u32 {
        let index = self.current.nodes.len() as u32;
        let superblock = self.current.superblocks.len().saturating_sub(1) as u32;
        self.current.nodes.push(NodeLabel {
            superblock,
            parent: self.open.last().map_or(-1, |&p| p as i32),
            x: node.x as u32,
            y: node.y as u32,
            log2: node.log2_size(),
            planned,
            chosen: SLOT_NONE,
            cost: [f32::NAN; CANDIDATE_SLOTS],
            ns: [0; CANDIDATE_SLOTS],
            distortion: [f32::NAN; CANDIDATE_SLOTS],
            bits: [f32::NAN; CANDIDATE_SLOTS],
            work: [0; CANDIDATE_SLOTS],
            positions: [0; CANDIDATE_SLOTS],
            nonzero: [NO_COUNT; CANDIDATE_SLOTS],
            tx_blocks: [NO_COUNT; CANDIDATE_SLOTS],
            max_tx: [NO_TX; CANDIDATE_SLOTS],
            abandoned: 0,
            eligible: 0,
            admitted: 0,
            plan_margin: f32::NAN,
            gates: 0,
            context: [0; 2],
            inputs: None,
            stop_inputs: None,
            plan_costs: [f32::NAN; 2],
            search: [0.0; 4],
            pre: [0.0; 5],
            first: [f32::NAN; 3],
            final_context: [0; 2],
        });
        self.open.push(index);
        index
    }

    pub fn node_mut(&mut self, index: u32) -> &mut NodeLabel {
        &mut self.current.nodes[index as usize]
    }

    pub fn close_node(&mut self) {
        self.open.pop();
    }
}

/// The dump's clock and work counters at a point of the coding
/// ([`Labeller::mark`]); a span's cost is the difference of two.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct LabelMark {
    pub ns: u64,
    pub work: u64,
    pub positions: u64,
}

impl LabelMark {
    /// Time, work and positions from `self` to `later`.
    pub fn until(self, later: Self) -> (u64, u64, u32) {
        (
            later.ns.saturating_sub(self.ns),
            later.work.saturating_sub(self.work),
            u32::try_from(later.positions.saturating_sub(self.positions)).unwrap_or(u32::MAX),
        )
    }
}

impl NodeLabel {
    /// Note `slot`'s coded content (`None`: the backend keeps none).
    pub(crate) fn set_summary(&mut self, slot: usize, summary: Option<NodeTrialSummary>) {
        if let Some(summary) = summary {
            self.nonzero[slot] = summary.nonzero_levels;
            self.tx_blocks[slot] = summary.coded_blocks;
            self.max_tx[slot] = summary.max_luma_tx_log2;
        }
    }
}

/// The slot of a candidate shape.
pub(crate) fn shape_slot(shape: PartitionShape) -> u8 {
    let index = PartitionShape::ALL
        .iter()
        .position(|&s| s == shape)
        .expect("every shape is in ALL");
    SLOT_SPLIT + 1 + index as u8
}

/// The inputs a superblock's features read.
pub(crate) struct FeatureInputs<'a> {
    pub source: &'a Picture,
    /// The primary reference and the superblock's search vector (inter).
    pub reference: Option<(&'a Picture, Mv)>,
    /// The second reference and the search vector scaled to its distance.
    pub secondary: Option<(&'a Picture, Mv)>,
    pub pre: Option<&'a PreAnalysis>,
    pub still: Option<&'a StillMap>,
    pub motion: Option<&'a MotionAnalysis>,
}

/// The record of superblock `sb` before it is coded (cost and time unset).
pub(crate) fn superblock_record(
    inputs: &FeatureInputs<'_>,
    sb: &Block,
    inside: bool,
    tools: usize,
) -> SuperblockLabel {
    let mut features = SuperblockFeatures::default();
    let mut patch = Box::new([0u8; PATCH_CELLS]);
    let mut residual = Box::new([0i16; PATCH_CELLS]);
    let side = sb.width();
    let luma = inputs.source.plane(PlaneIndex::Y);
    if let Some(view) = luma.region(sb.x as isize, sb.y as isize, side, side) {
        let rows: Vec<&[u8]> = (0..side).map(|y| &view.row(y)[..side]).collect();
        source_statistics(&rows, &mut features);
        downscale(
            side,
            |x, y| i32::from(rows[y][x]),
            |i, v| patch[i] = v as u8,
        );
        if let Some((reference, mv)) = inputs.reference {
            let rx = sb.x as isize + isize::from(mv.x).div_euclid(QUARTER);
            let ry = sb.y as isize + isize::from(mv.y).div_euclid(QUARTER);
            features.search_vector = [mv.x, mv.y];
            if let Some(r) = reference.plane(PlaneIndex::Y).region(rx, ry, side, side) {
                let diff = |x: usize, y: usize| i32::from(rows[y][x]) - i32::from(r.row(y)[x]);
                let mut energy = 0f64;
                for y in 0..side {
                    for x in 0..side {
                        energy += f64::from(diff(x, y).pow(2));
                    }
                }
                features.residual_energy = (energy / (side * side) as f64) as f32;
                downscale(side, diff, |i, v| residual[i] = v as i16);
            }
        }
        if let Some((secondary, mv)) = inputs.secondary {
            let plane = secondary.plane(PlaneIndex::Y);
            let energy_at = |mv: Mv| {
                let rx = sb.x as isize + isize::from(mv.x).div_euclid(QUARTER);
                let ry = sb.y as isize + isize::from(mv.y).div_euclid(QUARTER);
                plane.region(rx, ry, side, side).map_or(0.0, |r| {
                    let energy: f64 = (0..side)
                        .flat_map(|y| (0..side).map(move |x| (x, y)))
                        .map(|(x, y)| {
                            f64::from((i32::from(rows[y][x]) - i32::from(r.row(y)[x])).pow(2))
                        })
                        .sum();
                    (energy / (side * side) as f64) as f32
                })
            };
            features.second_residual = [energy_at(mv), energy_at(Mv::ZERO)];
        }
    }
    grid_statistics(inputs, sb, &mut features);
    let cell_stats = cell_grid(inputs, sb);
    // The analysis superblock, where it is this superblock.
    let gate = inputs
        .motion
        .filter(|_| sb.width() == nc_analysis::motion::SB)
        .and_then(|motion| {
            let side = nc_analysis::motion::SB;
            motion
                .sb
                .try_get((sb.x / side) as isize, (sb.y / side) as isize)
        });
    SuperblockLabel {
        cell_stats,
        x: sb.x as u32,
        y: sb.y as u32,
        inside,
        features,
        patch,
        residual,
        cost: f32::NAN,
        ns: 0,
        tool_cost: vec![f32::NAN; tools + 1],
        tool_ns: vec![0; tools + 1],
        tool_work: vec![0; tools + 1],
        tool_positions: vec![0; tools + 1],
        gate_inputs: gate
            .and_then(|stats| stats.gate_inputs)
            .map(|inputs| inputs.0),
        gate_sads: gate.map_or([0; 5], |stats| {
            let [d8, d16, d32, d64] = stats.me_dist;
            [stats.zz_sad, d8, d16, d32, d64]
        }),
    }
}

/// The cell statistics of superblock `sb` (`SuperblockLabel::cell_stats`).
fn cell_grid(inputs: &FeatureInputs<'_>, sb: &Block) -> Box<CellGrid> {
    let mut grid = Box::new([[[0u16; CELL_FEATURES]; MOTION_GRID]; MOTION_GRID]);
    let side = sb.width();
    let luma = inputs.source.plane(PlaneIndex::Y);
    // One sample of margin for the central differences.
    let Some(view) = luma.region(sb.x as isize - 1, sb.y as isize - 1, side + 2, side + 2) else {
        return grid;
    };
    let cells = (side / CELL).max(1);
    let per = (cells / MOTION_GRID).max(1);
    let mut sums = vec![[0u32; CELL_FEATURES]; MOTION_GRID * MOTION_GRID];
    for cy in 0..cells {
        for cx in 0..cells {
            let (x, y) = (sb.x + cx * CELL, sb.y + cy * CELL);
            let references = inputs.reference.and_then(|(reference, mv)| {
                let plane = reference.plane(PlaneIndex::Y);
                let (mx, my) = (
                    isize::from(mv.x).div_euclid(QUARTER),
                    isize::from(mv.y).div_euclid(QUARTER),
                );
                let zero = plane.region(x as isize, y as isize, CELL, CELL)?;
                let moved = plane.region(x as isize + mx, y as isize + my, CELL, CELL)?;
                Some((zero, moved))
            });
            let stats = cell_stats(
                &view,
                (cx * CELL, cy * CELL),
                references.as_ref().map(|(z, m)| (z, m)),
            );
            let slot = &mut sums
                [(cy / per).min(MOTION_GRID - 1) * MOTION_GRID + (cx / per).min(MOTION_GRID - 1)];
            for (total, v) in slot.iter_mut().zip(stats) {
                *total += u32::from(v);
            }
        }
    }
    let count = (per * per) as u32;
    for (i, total) in sums.iter().enumerate() {
        for (k, v) in total.iter().enumerate() {
            grid[i / MOTION_GRID][i % MOTION_GRID][k] =
                u16::try_from(v / count).unwrap_or(u16::MAX);
        }
    }
    grid
}

/// Mean, variance, gradients and coherence of the source rows.
fn source_statistics(rows: &[&[u8]], f: &mut SuperblockFeatures) {
    let n = rows.len();
    let (mut sum, mut sum_sq) = (0f64, 0f64);
    let (mut gxx, mut gyy, mut gxy, mut ax, mut ay) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (y, row) in rows.iter().enumerate() {
        for (x, &v) in row.iter().enumerate() {
            let v = f64::from(v);
            sum += v;
            sum_sq += v * v;
            if x + 1 < n && y + 1 < n {
                let dx = f64::from(row[x + 1]) - v;
                let dy = f64::from(rows[y + 1][x]) - v;
                gxx += dx * dx;
                gyy += dy * dy;
                gxy += dx * dy;
                ax += dx.abs();
                ay += dy.abs();
            }
        }
    }
    let area = (n * n) as f64;
    let mean = sum / area;
    f.luma_mean = mean as f32;
    f.luma_variance = (sum_sq / area - mean * mean) as f32;
    let inner = ((n - 1) * (n - 1)).max(1) as f64;
    f.gradient = [(ax / inner) as f32, (ay / inner) as f32];
    let energy = gxx + gyy;
    f.coherence = if energy > 0.0 {
        (((gxx - gyy).powi(2) + 4.0 * gxy * gxy).sqrt() / energy) as f32
    } else {
        0.0
    };
}

/// Average `side`-square samples `at(x, y)` down to the patch (or repeat
/// them up to it), writing cell `i` with `put(i, value)`; rounded to
/// nearest.
fn downscale(side: usize, at: impl Fn(usize, usize) -> i32, mut put: impl FnMut(usize, i32)) {
    let factor = (side / PATCH_SIDE).max(1);
    let count = (factor * factor) as i32;
    for py in 0..PATCH_SIDE {
        for px in 0..PATCH_SIDE {
            let (x0, y0) = (px * side / PATCH_SIDE, py * side / PATCH_SIDE);
            let mut total = 0i32;
            for y in y0..y0 + factor {
                for x in x0..x0 + factor {
                    total += at(x, y);
                }
            }
            put(
                py * PATCH_SIDE + px,
                (2 * total + count).div_euclid(2 * count),
            );
        }
    }
}

/// Mean and standard deviation of `values`.
fn mean_std(values: &[f32]) -> [f32; 2] {
    if values.is_empty() {
        return [0.0, 0.0];
    }
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    [mean, var.sqrt()]
}

/// Statistics of the 8x8 analysis cells under the superblock.
fn grid_statistics(inputs: &FeatureInputs<'_>, sb: &Block, f: &mut SuperblockFeatures) {
    let cells = 1usize << sb.log2_size().saturating_sub(GRID_LOG2);
    let (col0, row0) = (sb.x >> GRID_LOG2, sb.y >> GRID_LOG2);
    let samples = (1usize << (2 * GRID_LOG2)) as f32;
    let mut variances = Vec::new();
    let (mut statics, mut zero_sad, mut total) = (0usize, 0f32, 0usize);
    let (mut stills, mut runs) = (0usize, 0f32);
    let (mut best_sad, mut mvx, mut mvy) = (0f32, Vec::new(), Vec::new());
    for row in row0..row0 + cells {
        for col in col0..col0 + cells {
            if let Some(pre) = inputs.pre
                && let Some(b) = pre.blocks.try_get(col as isize, row as isize)
            {
                total += 1;
                variances.push(b.sum_sq_dev as f32 / samples);
                statics += usize::from(b.is_static);
                zero_sad += f32::from(b.sad_prev.unwrap_or(0)) / samples;
            }
            if let Some(still) = inputs.still
                && let Some(&s) = still.blocks.try_get(col as isize, row as isize)
            {
                stills += usize::from(s);
                runs += f32::from(*still.runs.get(col, row));
            }
            if let Some(motion) = inputs.motion
                && let Some(m) = motion.fine.try_get(col as isize, row as isize)
            {
                best_sad += m.sad as f32 / samples;
                mvx.push(f32::from(m.mv.x));
                mvy.push(f32::from(m.mv.y));
            }
        }
    }
    let share = |count: usize| {
        if total == 0 {
            0.0
        } else {
            count as f32 / total as f32
        }
    };
    f.block_variance = mean_std(&variances);
    f.static_share = share(statics);
    f.still = [
        share(stills),
        if total == 0 { 0.0 } else { runs / total as f32 },
    ];
    f.zero_sad = if total == 0 {
        0.0
    } else {
        zero_sad / total as f32
    };
    if !mvx.is_empty() {
        let (x, y) = (mean_std(&mvx), mean_std(&mvy));
        f.motion = [x[0], y[0], x[1], y[1]];
        f.best_sad = best_sad / mvx.len() as f32;
    }
    if let Some(motion) = inputs.motion {
        pool_motion(motion, (col0, row0, cells), f);
    }
}

/// The 8x8 motion field over `cells` square cells from (`col0`, `row0`),
/// averaged down (or repeated up) to the [`MOTION_GRID`] square.
fn pool_motion(
    motion: &MotionAnalysis,
    (col0, row0, cells): (usize, usize, usize),
    f: &mut SuperblockFeatures,
) {
    let samples = (1usize << (2 * GRID_LOG2)) as f32;
    let per = (cells / MOTION_GRID).max(1);
    for gy in 0..MOTION_GRID {
        for gx in 0..MOTION_GRID {
            let (c0, r0) = (
                col0 + gx * cells / MOTION_GRID,
                row0 + gy * cells / MOTION_GRID,
            );
            let (mut sad, mut mv, mut n) = (0f32, [0f32; 2], 0f32);
            for r in r0..r0 + per {
                for c in c0..c0 + per {
                    if let Some(m) = motion.fine.try_get(c as isize, r as isize) {
                        sad += m.sad as f32 / samples;
                        mv[0] += f32::from(m.mv.x);
                        mv[1] += f32::from(m.mv.y);
                        n += 1.0;
                    }
                }
            }
            if n > 0.0 {
                f.block_sad[gy][gx] = sad / n;
                f.block_mv[gy][gx] = [mv[0] / n, mv[1] / n];
            }
        }
    }
}

/// Pre-analysis over `node`'s 8x8 cells ([`NodeLabel::pre`]).
pub(crate) fn node_pre(
    pre: Option<&PreAnalysis>,
    still: Option<&StillMap>,
    node: &Block,
) -> [f32; 5] {
    let cells = 1usize << node.log2_size().saturating_sub(GRID_LOG2);
    let (col0, row0) = (node.x >> GRID_LOG2, node.y >> GRID_LOG2);
    let samples = (1usize << (2 * GRID_LOG2)) as f32;
    let (mut n, mut statics, mut stills, mut zero, mut var_sum, mut var_max) =
        (0f32, 0f32, 0f32, 0f32, 0f32, 0f32);
    for row in row0..row0 + cells {
        for col in col0..col0 + cells {
            let Some(b) = pre.and_then(|p| p.blocks.try_get(col as isize, row as isize)) else {
                continue;
            };
            n += 1.0;
            statics += f32::from(u8::from(b.is_static));
            zero += f32::from(b.sad_prev.unwrap_or(0)) / samples;
            let var = b.sum_sq_dev as f32 / samples;
            var_sum += var;
            var_max = var_max.max(var);
            if let Some(&s) = still.and_then(|s| s.blocks.try_get(col as isize, row as isize)) {
                stills += f32::from(u8::from(s));
            }
        }
    }
    if n == 0.0 {
        return [0.0; 5];
    }
    [statics / n, stills / n, zero / n, var_sum / n, var_max]
}

/// The motion search's aggregated field at `node`'s size ([`NodeLabel::search`]).
pub(crate) fn node_search(motion: Option<&MotionAnalysis>, node: &Block) -> [f32; 4] {
    let Some(field) = motion.and_then(|m| {
        m.aggregated
            .iter()
            .find(|f| f.size == node.width() && node.width() == node.height())
    }) else {
        return [0.0; 4];
    };
    let Some(b) = field.blocks.try_get(
        (node.x / field.size) as isize,
        (node.y / field.size) as isize,
    ) else {
        return [0.0; 4];
    };
    let area = (field.size * field.size) as f32;
    [
        f32::from(b.mv.x),
        f32::from(b.mv.y),
        b.whole_sad as f32 / area,
        b.split_sad as f32 / area,
    ]
}
