//! The label dump's columns (`--label-dump`, `label_dump.rs`): the one place
//! a column is declared. Each entry names the column, its numpy type and row
//! shape, and how a record writes its row; the writer emits every entry as
//! `<table>/<name>.npy` and lists them in `labels.json`, which is the schema
//! the reader (`eval/nceval/nndata`) checks each shard against. Adding a column
//! is one line here (and the value in `nc_encoder::labels`).

use nc_encoder::labels::cell_stats;
use nc_encoder::labels::{
    CANDIDATE_SLOTS, CellGrid, FrameContext, GATE_INPUTS, LeafLabel, MOTION_GRID, NodeLabel,
    PATCH_SIDE, STOP_INPUTS, SuperblockLabel, TRIAL_INPUTS,
};

/// A row shape entry standing for the shard's tool-slot count (`labels.json`
/// `tools`: the preset, then each `--label-tool`).
pub const TOOLS: usize = usize::MAX;

/// A value written little-endian into a column row.
pub trait Le {
    fn put(&self, out: &mut Vec<u8>);
}

macro_rules! le_number {
    ($($t:ty),*) => {$(
        impl Le for $t {
            fn put(&self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
        }
    )*};
}

le_number!(u8, i8, i16, u16, u32, i32, u64, f32);

impl Le for bool {
    fn put(&self, out: &mut Vec<u8>) {
        out.push(u8::from(*self));
    }
}

impl<T: Le, const N: usize> Le for [T; N] {
    fn put(&self, out: &mut Vec<u8>) {
        for v in self {
            v.put(out);
        }
    }
}

impl<T: Le> Le for Vec<T> {
    fn put(&self, out: &mut Vec<u8>) {
        for v in self {
            v.put(out);
        }
    }
}

impl<T: Le> Le for Box<T> {
    fn put(&self, out: &mut Vec<u8>) {
        (**self).put(out);
    }
}

/// Where a record sits in the shard: its frame, and the shard rows the
/// frame's superblock and node indices start at.
#[derive(Debug, Clone, Copy, Default)]
pub struct Place {
    pub frame: u32,
    pub sb_base: u32,
    pub node_base: i32,
}

/// The cell statistics of the source alone (`nc_analysis::cell_stats`): the
/// same for every rung and rate of a clip, so a released version stores them
/// once per clip (`nceval.nndata` patch store).
const SOURCE_FEATURES: [usize; 12] = [
    cell_stats::MEAN,
    cell_stats::VARIANCE,
    cell_stats::GRADIENT,
    cell_stats::HOG,
    cell_stats::HOG + 1,
    cell_stats::HOG + 2,
    cell_stats::HOG + 3,
    cell_stats::HOG + 4,
    cell_stats::HOG + 5,
    cell_stats::HOG + 6,
    cell_stats::HOG + 7,
    cell_stats::NOISE_FLOOR,
];
/// The cell statistics against the reference: they depend on the coding.
const MOTION_FEATURES: [usize; 3] = [
    cell_stats::ZERO_SAD,
    cell_stats::VECTOR_SAD,
    cell_stats::RESIDUAL_ENERGY,
];

/// The `features` of every cell of `grid`, in grid order.
fn cells(grid: &CellGrid, features: &[usize]) -> Vec<u16> {
    grid.iter()
        .flatten()
        .flat_map(|cell| features.iter().map(|&f| cell[f]))
        .collect()
}

/// One column of a table whose records are `R`.
pub struct Spec<R> {
    pub name: &'static str,
    /// numpy `descr`.
    pub dtype: &'static str,
    /// Per-row shape ([`TOOLS`] for the tool count).
    pub shape: &'static [usize],
    pub put: fn(&R, Place, &mut Vec<u8>),
}

/// `col!(name, dtype, [shape], |record, place| value)`.
macro_rules! col {
    ($name:literal, $dtype:literal, [$($d:expr),*], |$r:pat_param, $p:pat_param| $value:expr) => {
        Spec {
            name: $name,
            dtype: $dtype,
            shape: &[$($d),*],
            put: |$r, $p, out| Le::put(&$value, out),
        }
    };
}

pub const FRAME: &[Spec<FrameContext>] = &[
    col!("index", "<u4", [], |r, _| u32::try_from(r.index)
        .unwrap_or(u32::MAX)),
    col!("key", "|u1", [], |r, _| r.key),
    col!("temporal_layer", "|u1", [], |r, _| r.temporal_layer),
    col!("key_distance", "<u4", [], |r, _| r.key_distance),
    col!("quantizer", "<u4", [], |r, _| r.quantizer),
    col!("lambda", "<f4", [], |r, _| r.lambda),
    col!("noise_sigma", "<f4", [], |r, _| r.noise_sigma),
    col!("rate", "<f4", [4], |r, _| r.rate),
    col!("width", "<u4", [], |r, _| r.width),
    col!("height", "<u4", [], |r, _| r.height),
    col!("superblock_log2", "|u1", [], |r, _| r.superblock_log2),
    col!("min_block_log2", "|u1", [], |r, _| r.min_block_log2),
];

pub const SB: &[Spec<SuperblockLabel>] = &[
    col!("frame", "<u4", [], |_, p| p.frame),
    col!("x", "<u4", [], |r, _| r.x),
    col!("y", "<u4", [], |r, _| r.y),
    col!("inside", "|u1", [], |r, _| r.inside),
    col!("cost", "<f4", [], |r, _| r.cost),
    col!("ns", "<u8", [], |r, _| r.ns),
    col!("luma_mean", "<f4", [], |r, _| r.features.luma_mean),
    col!("luma_variance", "<f4", [], |r, _| r.features.luma_variance),
    col!("gradient", "<f4", [2], |r, _| r.features.gradient),
    col!("coherence", "<f4", [], |r, _| r.features.coherence),
    col!("block_variance", "<f4", [2], |r, _| r
        .features
        .block_variance),
    col!("static_share", "<f4", [], |r, _| r.features.static_share),
    col!("still", "<f4", [2], |r, _| r.features.still),
    col!("zero_sad", "<f4", [], |r, _| r.features.zero_sad),
    col!("best_sad", "<f4", [], |r, _| r.features.best_sad),
    col!("motion", "<f4", [4], |r, _| r.features.motion),
    col!("search_vector", "<i2", [2], |r, _| r.features.search_vector),
    col!("residual_energy", "<f4", [], |r, _| r
        .features
        .residual_energy),
    col!("second_residual", "<f4", [2], |r, _| r
        .features
        .second_residual),
    col!("block_sad", "<f4", [MOTION_GRID, MOTION_GRID], |r, _| r
        .features
        .block_sad),
    col!("block_mv", "<f4", [MOTION_GRID, MOTION_GRID, 2], |r, _| r
        .features
        .block_mv),
    col!(
        "cell_source",
        "<u2",
        [MOTION_GRID, MOTION_GRID, SOURCE_FEATURES.len()],
        |r, _| { cells(&r.cell_stats, &SOURCE_FEATURES) }
    ),
    col!(
        "cell_motion",
        "<u2",
        [MOTION_GRID, MOTION_GRID, MOTION_FEATURES.len()],
        |r, _| { cells(&r.cell_stats, &MOTION_FEATURES) }
    ),
    col!("has_gate_inputs", "|u1", [], |r, _| r.gate_inputs.is_some()),
    col!("gate_inputs", "|u1", [GATE_INPUTS], |r, _| r
        .gate_inputs
        .unwrap_or([0; GATE_INPUTS])),
    col!("gate_sads", "<u4", [5], |r, _| r.gate_sads),
    col!("patch", "|u1", [PATCH_SIDE, PATCH_SIDE], |r, _| r.patch),
    col!("residual", "<i2", [PATCH_SIDE, PATCH_SIDE], |r, _| r
        .residual),
    col!("tool_cost", "<f4", [TOOLS], |r, _| r.tool_cost),
    col!("tool_ns", "<u8", [TOOLS], |r, _| r.tool_ns),
    col!("tool_work", "<u8", [TOOLS], |r, _| r.tool_work),
    col!("tool_positions", "<u4", [TOOLS], |r, _| r.tool_positions),
];

pub const NODE: &[Spec<NodeLabel>] = &[
    col!("frame", "<u4", [], |_, p| p.frame),
    col!("sb", "<u4", [], |r, p| p.sb_base + r.superblock),
    col!("parent", "<i4", [], |r, p| if r.parent < 0 {
        -1
    } else {
        p.node_base + r.parent
    }),
    col!("x", "<u4", [], |r, _| r.x),
    col!("y", "<u4", [], |r, _| r.y),
    col!("log2", "|u1", [], |r, _| r.log2),
    col!("planned", "|u1", [], |r, _| r.planned),
    col!("chosen", "|u1", [], |r, _| r.chosen),
    col!("eligible", "|u1", [], |r, _| r.eligible),
    col!("admitted", "|u1", [], |r, _| r.admitted),
    col!("abandoned", "|u1", [], |r, _| r.abandoned),
    col!("plan_margin", "<f4", [], |r, _| r.plan_margin),
    col!("plan_costs", "<f4", [2], |r, _| r.plan_costs),
    col!("gates", "|u1", [], |r, _| r.gates),
    col!("context", "|i1", [2], |r, _| r.context),
    col!("final_context", "|i1", [2], |r, _| r.final_context),
    col!("has_inputs", "|u1", [], |r, _| r.inputs.is_some()),
    col!("inputs", "|u1", [TRIAL_INPUTS], |r, _| r
        .inputs
        .unwrap_or_default()),
    col!("has_stop_inputs", "|u1", [], |r, _| r.stop_inputs.is_some()),
    col!("stop_inputs", "|u1", [STOP_INPUTS], |r, _| r
        .stop_inputs
        .unwrap_or([0; STOP_INPUTS])),
    col!("search", "<f4", [4], |r, _| r.search),
    col!("pre", "<f4", [5], |r, _| r.pre),
    col!("first", "<f4", [3], |r, _| r.first),
    col!("cost", "<f4", [CANDIDATE_SLOTS], |r, _| r.cost),
    col!("ns", "<u8", [CANDIDATE_SLOTS], |r, _| r.ns),
    col!("distortion", "<f4", [CANDIDATE_SLOTS], |r, _| r.distortion),
    col!("bits", "<f4", [CANDIDATE_SLOTS], |r, _| r.bits),
    col!("cand_work", "<u8", [CANDIDATE_SLOTS], |r, _| r.work),
    col!("cand_positions", "<u4", [CANDIDATE_SLOTS], |r, _| r
        .positions),
    col!("nonzero", "<u2", [CANDIDATE_SLOTS], |r, _| r.nonzero),
    col!("tx_blocks", "<u2", [CANDIDATE_SLOTS], |r, _| r.tx_blocks),
    col!("max_tx", "|u1", [CANDIDATE_SLOTS], |r, _| r.max_tx),
];

pub const LEAF: &[Spec<LeafLabel>] = &[
    col!("frame", "<u4", [], |_, p| p.frame),
    col!("sb", "<u4", [], |r, p| p.sb_base + r.superblock),
    col!("x", "<u4", [], |r, _| r.x),
    col!("y", "<u4", [], |r, _| r.y),
    col!("log2", "|u1", [2], |r, _| r.log2),
    col!("mode", "|u1", [], |r, _| r.mode),
    col!("skip", "|u1", [], |r, _| r.skip),
    col!("mv", "<i2", [2], |r, _| r.mv),
    col!("qp_offset", "<i2", [], |r, _| r.qp_offset),
    col!("positions", "<u2", [], |r, _| r.positions),
];
