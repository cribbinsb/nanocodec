//! Transform-size syntax for frames with `TX_MODE_SELECT` (AV1 §5.11.15–17):
//! the variable inter transform tree (`txfm_split`), the intra `tx_depth`,
//! and the above/left transform-size contexts both use, shared by the
//! encoder and our decoder.
//!
//! In our subset blocks are square (8x8 to 64x64), so the largest transform
//! of a block is the block itself and the tree is a quadtree of at most
//! [`MAX_VARTX_DEPTH`] levels. The context rules follow libaom
//! `av1/common/av1_common_int.h` (`txfm_partition_context`,
//! `get_tx_size_context`, BSD-2-Clause, see `THIRD_PARTY.md`).

use crate::modeinfo::{MI_LOG2, ModeInfo};

/// `MAX_VARTX_DEPTH`: inter transform tree levels below the block.
pub const MAX_VARTX_DEPTH: u8 = 2;
/// `MAX_TX_DEPTH`: intra `tx_depth` levels.
const MAX_TX_DEPTH: u8 = 2;
/// Smallest transform (4x4), log2 of its side.
pub const TX_4X4_LOG2: u8 = 2;
/// Largest transform (64x64), log2 of its side: the reset value of the
/// contexts (`TX_SIZES_LARGEST`).
const TX_64X64_LOG2: u8 = 6;
/// `TX_SIZES`: transform side classes 4..=64.
const TX_SIZES: u8 = 5;
/// Superblock side in 4x4 units (the left context covers one superblock).
const SB_MI: usize = 16;

/// Above/left transform sides (in samples) of the coded neighbours, per 4x4
/// column of the frame and per 4x4 row of the current superblock row
/// (libaom `above_txfm_context` / `left_txfm_context`).
#[derive(Debug, Clone)]
pub struct TxfmContexts {
    above: Vec<u8>,
    left: [u8; SB_MI],
}

impl TxfmContexts {
    /// Contexts for a frame `mi_cols` 4x4 units wide, reset.
    #[must_use]
    pub fn new(mi_cols: usize) -> Self {
        Self {
            above: vec![1 << TX_64X64_LOG2; mi_cols],
            left: [1 << TX_64X64_LOG2; SB_MI],
        }
    }

    /// Start of a frame (tile): the above row holds the largest size.
    pub fn start_frame(&mut self) {
        self.above.fill(1 << TX_64X64_LOG2);
    }

    /// Start of a superblock row: the left column holds the largest size.
    pub fn start_sb_row(&mut self) {
        self.left.fill(1 << TX_64X64_LOG2);
    }

    /// Record a `side` x `side` transform (or, for a skipped inter block, the
    /// block) covering 4x4 units `mi_col..`, `mi_row..` (`side_mi` units).
    pub fn fill(&mut self, mi_col: usize, mi_row: usize, side_mi: usize, side: u8) {
        let end = (mi_col + side_mi).min(self.above.len());
        self.above[mi_col..end].fill(side);
        let row = mi_row % SB_MI;
        self.left[row..(row + side_mi).min(SB_MI)].fill(side);
    }

    /// Width of the transform stored above 4x4 column `mi_col`.
    pub(crate) fn above(&self, mi_col: usize) -> u8 {
        self.above[mi_col]
    }

    /// Height of the transform stored left of 4x4 row `mi_row`.
    pub(crate) fn left(&self, mi_row: usize) -> u8 {
        self.left[mi_row % SB_MI]
    }
}

/// `txfm_split` context of a `tx_log2` transform at 4x4 unit (`mi_col`,
/// `mi_row`) of a block of side `block_log2` (libaom
/// `txfm_partition_context`).
#[must_use]
pub fn split_context(
    contexts: &TxfmContexts,
    mi_col: usize,
    mi_row: usize,
    block_log2: u8,
    tx_log2: u8,
) -> usize {
    let side = 1_u16 << tx_log2;
    let above = usize::from(u16::from(contexts.above(mi_col)) < side);
    let left = usize::from(u16::from(contexts.left(mi_row)) < side);
    // Square-transform classes: 0 = 4x4 .. 4 = 64x64.
    let max_class = block_log2 - TX_4X4_LOG2;
    let class = tx_log2 - TX_4X4_LOG2;
    let category = usize::from(class != max_class && max_class > 1)
        + usize::from(TX_SIZES - 1 - max_class) * 2;
    category * 3 + above + left
}

/// `tx_size` CDF category of a block (libaom `bsize_to_tx_size_cat`): 0 for
/// 8x8 up to 3 for 64x64.
#[must_use]
pub fn tx_size_category(block_log2: u8) -> usize {
    usize::from(block_log2 - TX_4X4_LOG2 - 1)
}

/// Number of `tx_depth` symbols of a block (libaom `bsize_to_max_depth` + 1).
#[must_use]
pub fn tx_depth_symbols(block_log2: u8) -> usize {
    usize::from((block_log2 - TX_4X4_LOG2).min(MAX_TX_DEPTH)) + 1
}

/// `tx_depth` context of an intra block (libaom `get_tx_size_context`):
/// whether the above/left neighbours' transforms (or inter blocks) are at
/// least as large as this block.
#[must_use]
pub fn tx_size_context(
    contexts: &TxfmContexts,
    above: Option<ModeInfo>,
    left: Option<ModeInfo>,
    mi_col: usize,
    mi_row: usize,
    block_log2: u8,
) -> usize {
    let side = 1_u16 << block_log2;
    let neighbour = |info: Option<ModeInfo>, stored: u8| {
        info.map(|n| {
            if n.is_inter() {
                usize::from((n.size_mi() << MI_LOG2) >= usize::from(side))
            } else {
                usize::from(u16::from(stored) >= side)
            }
        })
    };
    let above = neighbour(above, contexts.above(mi_col));
    let left = neighbour(left, contexts.left(mi_row));
    match (above, left) {
        (Some(a), Some(l)) => a + l,
        (Some(n), None) | (None, Some(n)) => n,
        (None, None) => 0,
    }
}

/// The inter transform tree of one block: whether the block splits into
/// four quadrant transforms, and whether each quadrant splits again
/// (raster order). [`MAX_VARTX_DEPTH`] levels at most; transforms never go
/// below 4x4.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TxSplit {
    pub split: bool,
    pub quadrants: [bool; 4],
}

impl TxSplit {
    /// One transform covering the block.
    pub const NONE: Self = Self {
        split: false,
        quadrants: [false; 4],
    };

    /// The leaf transforms of a block of side `block_log2`, in coding order
    /// (recursive raster, AV1 §5.11.35 `transform_tree`): (x offset, y
    /// offset, log2 side) in luma samples.
    #[must_use]
    pub fn leaves(self, block_log2: u8) -> Vec<(usize, usize, u8)> {
        if !self.split {
            return vec![(0, 0, block_log2)];
        }
        let half = block_log2 - 1;
        let step = 1_usize << half;
        let mut out = Vec::with_capacity(16);
        for (i, &quadrant) in self.quadrants.iter().enumerate() {
            let (qx, qy) = ((i % 2) * step, (i / 2) * step);
            if quadrant && half > TX_4X4_LOG2 {
                let sub = half - 1;
                let s = 1_usize << sub;
                for j in 0..4 {
                    out.push((qx + (j % 2) * s, qy + (j / 2) * s, sub));
                }
            } else {
                out.push((qx, qy, half));
            }
        }
        out
    }

    /// Whether a split flag is coded for a `tx_log2` transform at `depth`
    /// (AV1 §5.11.17: not at [`MAX_VARTX_DEPTH`], and 4x4 cannot split).
    #[must_use]
    pub fn flag_coded(depth: u8, tx_log2: u8) -> bool {
        depth < MAX_VARTX_DEPTH && tx_log2 > TX_4X4_LOG2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_contexts_follow_libaom_categories() {
        let contexts = TxfmContexts::new(32);
        // 64x64 block: the 64 transform is category 0, its 32 children 1.
        assert_eq!(split_context(&contexts, 0, 0, 6, 6), 0);
        assert_eq!(split_context(&contexts, 0, 0, 6, 5), 3);
        // 8x8 block (max class 1): category (5 - 1 - 1) * 2 = 6.
        assert_eq!(split_context(&contexts, 0, 0, 3, 3), 18);
    }

    #[test]
    fn leaves_follow_the_recursive_raster_order() {
        let tree = TxSplit {
            split: true,
            quadrants: [false, true, false, false],
        };
        let leaves = tree.leaves(5);
        assert_eq!(
            leaves,
            vec![
                (0, 0, 4),
                (16, 0, 3),
                (24, 0, 3),
                (16, 8, 3),
                (24, 8, 3),
                (0, 16, 4),
                (16, 16, 4)
            ]
        );
        assert_eq!(TxSplit::NONE.leaves(6), vec![(0, 0, 6)]);
    }

    #[test]
    fn tx_depth_alphabet_and_category() {
        assert_eq!((tx_size_category(3), tx_depth_symbols(3)), (0, 2));
        assert_eq!((tx_size_category(6), tx_depth_symbols(6)), (3, 3));
    }
}
