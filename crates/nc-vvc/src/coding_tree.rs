//! H.266 §7.3.11.4 `coding_tree` split syntax of our subset: the allowed
//! splits of a node (§6.4.1-6.4.3), their contexts (§9.3.4.2.2-9.3.4.2.3)
//! and the `split_cu_flag`, `split_qt_flag`, `mtt_split_cu_vertical_flag`
//! and `mtt_split_cu_binary_flag` bins, shared by the encoder and our
//! decoder's own-stream path.
//!
//! The subset: quadtree nodes are square; inter slices may add one level of
//! multi-type tree (binary or ternary splits of a quadtree leaf,
//! `sps_max_mtt_hierarchy_depth_inter_slice` 0 or 1), intra slices none.
//! Across the picture edge the encoder always splits by quadtree. Rules
//! checked against `VVdeC`/`VVenC` (BSD-3-Clause-Clear, `THIRD_PARTY.md`)
//! and the general decoder's `decoder_blocks/tree.rs`.
use crate::{coeff::Bins, contexts::Context, headers};
use nc_common::decode::{DecodeError, Result};
use nc_encoder::{Block, PartitionShape};

/// The SPS partition constraints of inter slices (§7.4.3.4); intra slices
/// have no multi-type tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Partitioning {
    /// `sps_max_mtt_hierarchy_depth_inter_slice`: 0 or 1.
    pub mtt_depth: u8,
    /// log2 of `MaxBtSizeY` minus log2 of `MinQtSizeY`
    /// (`sps_log2_diff_max_bt_min_qt_inter_slice`; coded when `mtt_depth` > 0).
    pub bt_diff: u8,
    /// `sps_log2_diff_max_tt_min_qt_inter_slice`.
    pub tt_diff: u8,
    /// log2 of `MinCbSizeY` (`sps_log2_min_luma_coding_block_size_minus2`
    /// + 2), for both slice types.
    pub min_cb_log2: u8,
}

impl Partitioning {
    /// Quadtree only.
    pub(crate) const QUADTREE: Self = Self {
        mtt_depth: 0,
        bt_diff: 0,
        tt_diff: 0,
        min_cb_log2: QUADTREE_MIN_CB_LOG2,
    };

    /// The configuration of `vvc.mtt` `mode` (0 quadtree, 1 binary, 2
    /// binary and ternary splits of quadtree leaves, one level) in CTUs of
    /// `ctu_log2`: splits up to the CTU size (§7.4.3.4 bounds the
    /// differences by `CtbLog2SizeY` − `MinQtLog2SizeInterY`), and
    /// `MaxTtSizeY` = `MinQtSizeY` rules every ternary split out of mode 1.
    /// With a multi-type tree `MinCbSizeY` is 8 (our smallest CU anyway):
    /// at 4, 8x8 nodes would code a `split_cu_flag` for binary splits we
    /// never make.
    pub(crate) fn of_mode(mode: u8, ctu_log2: u8) -> Self {
        let largest = ctu_log2 - headers::MIN_QT_LOG2;
        match mode {
            0 => Self::QUADTREE,
            1 => Self {
                mtt_depth: 1,
                bt_diff: largest,
                tt_diff: 0,
                min_cb_log2: MTT_MIN_CB_LOG2,
            },
            _ => Self {
                mtt_depth: 1,
                bt_diff: largest,
                tt_diff: largest,
                min_cb_log2: MTT_MIN_CB_LOG2,
            },
        }
    }

    /// The constraints of intra slices: no multi-type tree.
    pub(crate) fn intra(self) -> Self {
        Self {
            mtt_depth: 0,
            bt_diff: 0,
            tt_diff: 0,
            ..self
        }
    }

    fn max_bt(self) -> usize {
        1 << (headers::MIN_QT_LOG2 + self.bt_diff)
    }
    fn max_tt(self) -> usize {
        1 << (headers::MIN_QT_LOG2 + self.tt_diff)
    }
}

/// The largest `vvc.mtt` mode.
pub(crate) const MAX_MODE: u8 = 2;

/// The 64x64 virtual pipeline data unit that §6.4.2/§6.4.3 keep binary
/// and ternary splits within.
const VPDU_SIZE: usize = 64;
/// log2 of `MinCbSizeY` without and with a multi-type tree.
const QUADTREE_MIN_CB_LOG2: u8 = 2;
const MTT_MIN_CB_LOG2: u8 = 3;

/// A coding-tree node: its block and multi-type tree depth (0 for quadtree
/// nodes, 1 for the leaves of a shape).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Node {
    pub block: Block,
    pub mtt_depth: u8,
}

/// How a node is split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TreeSplit {
    None,
    Quad,
    Shape(PartitionShape),
}

/// The splits allowed at a node (§6.4.1-§6.4.3), and whether it may stay
/// unsplit (false across the picture edge).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)] // independently available split modes
pub(crate) struct Allowed {
    pub none: bool,
    pub quad: bool,
    pub horizontal: bool,
    pub vertical: bool,
    pub triple_horizontal: bool,
    pub triple_vertical: bool,
}

impl Allowed {
    /// The splits of `node` in a slice whose multi-type tree follows
    /// `limits`, in a `width` x `height` coded picture. A multi-type tree
    /// leaf's parent was a quadtree node (depth 1 is the deepest we code).
    pub(crate) fn of(node: &Node, limits: Partitioning, (width, height): (usize, usize)) -> Self {
        let block = &node.block;
        let (w, h) = (block.width(), block.height());
        let quad = node.mtt_depth == 0 && w > (1 << headers::MIN_QT_LOG2);
        let mtt = node.mtt_depth < limits.mtt_depth;
        let (max_bt, max_tt) = (limits.max_bt(), limits.max_tt());
        let right_inside = block.x + w <= width;
        let bottom_inside = block.y + h <= height;
        if !right_inside || !bottom_inside {
            // §6.4.2: across the edge only binary splits towards it (and
            // the quadtree) are allowed, and the node must split.
            let binary = w <= max_bt && h <= max_bt && w <= VPDU_SIZE && h <= VPDU_SIZE && mtt;
            let mut quad = quad || !binary;
            let horizontal = binary && !bottom_inside && (right_inside || !quad);
            let vertical = binary && bottom_inside && !right_inside;
            quad |= !horizontal && !vertical;
            return Self {
                none: false,
                quad,
                horizontal,
                vertical,
                triple_horizontal: false,
                triple_vertical: false,
            };
        }
        let min_cb = 1 << limits.min_cb_log2;
        let mtt = mtt
            && (w > min_cb || h > min_cb)
            && ((w <= max_bt && h <= max_bt) || (w <= max_tt && h <= max_tt));
        let binary = mtt && w <= max_bt && h <= max_bt;
        let triple = mtt && w <= max_tt && h <= max_tt && w <= VPDU_SIZE && h <= VPDU_SIZE;
        Self {
            none: true,
            quad,
            horizontal: binary && h > min_cb && (w <= VPDU_SIZE || h > VPDU_SIZE),
            vertical: binary && w > min_cb && (w > VPDU_SIZE || h <= VPDU_SIZE),
            triple_horizontal: triple && h > 2 * min_cb,
            triple_vertical: triple && w > 2 * min_cb,
        }
    }

    fn horizontal_count(self) -> usize {
        usize::from(self.horizontal) + usize::from(self.triple_horizontal)
    }

    fn vertical_count(self) -> usize {
        usize::from(self.vertical) + usize::from(self.triple_vertical)
    }

    /// Whether the node codes any split bin.
    pub(crate) fn signals(self) -> bool {
        let (hor, ver) = (self.horizontal_count(), self.vertical_count());
        let any = self.quad || hor + ver > 0;
        // `split_cu_flag`, `split_qt_flag`, the direction, or binary vs ternary.
        (self.none && any)
            || (self.quad && hor + ver > 0)
            || (hor > 0 && ver > 0)
            || hor == 2
            || ver == 2
    }

    /// Whether `split` is one of the allowed ones.
    pub(crate) fn permits(self, split: TreeSplit) -> bool {
        match split {
            TreeSplit::None => self.none,
            TreeSplit::Quad => self.quad,
            TreeSplit::Shape(PartitionShape::Horizontal) => self.horizontal,
            TreeSplit::Shape(PartitionShape::Vertical) => self.vertical,
            TreeSplit::Shape(PartitionShape::HorizontalTriple) => self.triple_horizontal,
            TreeSplit::Shape(PartitionShape::VerticalTriple) => self.triple_vertical,
        }
    }
}

/// A coded neighbour's coding-unit size, for the split contexts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NeighbourCu {
    pub width: usize,
    pub height: usize,
    /// Its quadtree depth (`CqtDepth`).
    pub qt_depth: u8,
}

/// The context indices of a node's split bins that depend on its
/// neighbours, fixed when the node is visited (§9.3.4.2.2, §9.3.4.2.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SplitContexts {
    split: usize,
    quad: usize,
    direction: usize,
    binary_base: usize,
}

impl SplitContexts {
    /// The contexts of `node` (quadtree depth `qt_depth`) with `allowed`
    /// splits, given its left (x − 1, y) and above (x, y − 1) neighbours.
    pub(crate) fn new(
        node: &Node,
        qt_depth: u8,
        allowed: Allowed,
        [left, above]: [Option<NeighbourCu>; 2],
    ) -> Self {
        let (w, h) = (node.block.width(), node.block.height());
        let (hor, ver) = (allowed.horizontal_count(), allowed.vertical_count());
        // ctxSetIdx: (allowed binary/ternary splits + 2 · quad − 1) / 2.
        let total = 2 * usize::from(allowed.quad) + hor + ver;
        let split = usize::from(left.is_some_and(|n| n.height < h))
            + usize::from(above.is_some_and(|n| n.width < w))
            + SPLIT_SET_OFFSET[total];
        let quad = usize::from(left.is_some_and(|n| n.qt_depth > qt_depth))
            + usize::from(above.is_some_and(|n| n.qt_depth > qt_depth))
            + if qt_depth >= 2 { 3 } else { 0 };
        let direction = match ver.cmp(&hor) {
            std::cmp::Ordering::Greater => 4,
            std::cmp::Ordering::Less => 3,
            std::cmp::Ordering::Equal => match (left, above) {
                (Some(left), Some(above)) => match (w / above.width).cmp(&(h / left.height)) {
                    std::cmp::Ordering::Equal => 0,
                    std::cmp::Ordering::Less => 1,
                    std::cmp::Ordering::Greater => 2,
                },
                _ => 0,
            },
        };
        Self {
            split,
            quad,
            direction,
            binary_base: usize::from(node.mtt_depth <= 1),
        }
    }
}

/// `split_cu_flag` context offset (3 · ctxSetIdx) by the count of allowed
/// splits (quadtree counted twice).
const SPLIT_SET_OFFSET: [usize; 7] = [0, 0, 0, 3, 3, 6, 6];

/// The split contexts (Tables 51-54).
#[derive(Debug, Clone)]
pub(crate) struct TreeContexts {
    pub split: [Context; 9],
    pub quad: [Context; 6],
    pub direction: [Context; 5],
    pub binary: [Context; 4],
}

impl TreeContexts {
    /// Initial states at `qp` for an intra (I) or inter (P) slice.
    pub(crate) fn new(qp: u32, inter: bool) -> Self {
        let init = |values: (&[u8], &[u8]), windows: &[u8], i: usize| {
            let value = if inter { values.1[i] } else { values.0[i] };
            Context::new(value, windows[i], qp)
        };
        Self {
            split: std::array::from_fn(|i| init(SPLIT_INIT, &SPLIT_WINDOW, i)),
            quad: std::array::from_fn(|i| init(QUAD_INIT, &QUAD_WINDOW, i)),
            direction: std::array::from_fn(|i| init(DIRECTION_INIT, &DIRECTION_WINDOW, i)),
            binary: std::array::from_fn(|i| init(BINARY_INIT, &BINARY_WINDOW, i)),
        }
    }
}

/// (I, P) initial values and adaptation windows (`VVdeC` `Contexts.cpp`
/// `SplitFlag`, `SplitQtFlag`, `SplitHvFlag`, `Split12Flag`).
const SPLIT_INIT: (&[u8], &[u8]) = (
    &[19, 28, 38, 27, 29, 38, 20, 30, 31],
    &[11, 35, 53, 12, 6, 30, 13, 15, 31],
);
const SPLIT_WINDOW: [u8; 9] = [12, 13, 8, 8, 13, 12, 5, 9, 9];
const QUAD_INIT: (&[u8], &[u8]) = (&[27, 6, 15, 25, 19, 37], &[20, 14, 23, 18, 19, 6]);
const QUAD_WINDOW: [u8; 6] = [0, 8, 8, 12, 12, 8];
const DIRECTION_INIT: (&[u8], &[u8]) = (&[43, 42, 29, 27, 44], &[43, 35, 37, 34, 52]);
const DIRECTION_WINDOW: [u8; 5] = [9, 8, 9, 8, 5];
const BINARY_INIT: (&[u8], &[u8]) = (&[36, 45, 36, 45], &[43, 37, 21, 22]);
const BINARY_WINDOW: [u8; 4] = [12, 13, 12, 13];

/// Code (or read) the split of a node with `allowed` splits and contexts
/// `ctx`: `split` is the encoder's choice (ignored when decoding). Returns
/// the coded split.
///
/// # Errors
///
/// A decoded split outside the allowed set, or a reader error.
pub(crate) fn code(
    io: &mut impl Bins,
    c: &mut TreeContexts,
    allowed: Allowed,
    ctx: SplitContexts,
    split: TreeSplit,
) -> Result<TreeSplit> {
    let (hor, ver) = (allowed.horizontal_count(), allowed.vertical_count());
    if 2 * usize::from(allowed.quad) + hor + ver == 0 {
        return if allowed.none {
            Ok(TreeSplit::None)
        } else {
            Err(DecodeError::Malformed("VVC node cannot split"))
        };
    }
    if allowed.none && !io.bin(&mut c.split[ctx.split], split != TreeSplit::None)? {
        return Ok(TreeSplit::None);
    }
    if allowed.quad
        && (hor + ver == 0 || io.bin(&mut c.quad[ctx.quad], split == TreeSplit::Quad)?)
    {
        return Ok(TreeSplit::Quad);
    }
    let shape = match split {
        TreeSplit::Shape(shape) => Some(shape),
        _ => None,
    };
    let vertical = if ver == 0 {
        false
    } else if hor == 0 {
        true
    } else {
        io.bin(
            &mut c.direction[ctx.direction],
            shape.is_some_and(|s| !s.is_horizontal()),
        )?
    };
    let (binary, triple) = if vertical {
        (allowed.vertical, allowed.triple_vertical)
    } else {
        (allowed.horizontal, allowed.triple_horizontal)
    };
    let binary = binary
        && (!triple
            || io.bin(
                &mut c.binary[ctx.binary_base + 2 * usize::from(vertical)],
                shape.is_some_and(|s| s.part_count() == 2),
            )?);
    Ok(TreeSplit::Shape(match (vertical, binary) {
        (false, true) => PartitionShape::Horizontal,
        (true, true) => PartitionShape::Vertical,
        (false, false) => PartitionShape::HorizontalTriple,
        (true, false) => PartitionShape::VerticalTriple,
    }))
}

/// Bits of coding `split` (estimated from the context states as they
/// stand; the states are not changed).
pub(crate) fn bits(
    c: &TreeContexts,
    allowed: Allowed,
    ctx: SplitContexts,
    split: TreeSplit,
) -> f32 {
    let mut cost = crate::coeff::BitCost::default();
    let mut c = c.clone();
    code(&mut cost, &mut c, allowed, ctx, split).expect("pricing an allowed split");
    cost.0
}
