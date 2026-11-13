//! What later macroblocks (and the deblocking filter) need to know about a
//! coded macroblock, and the neighbourhood of the macroblock being coded.

use nc_encoder::Mv;

use crate::predict::ChromaMode;

/// How a macroblock was coded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MbKind {
    /// `I_16x16`.
    Intra16x16,
    /// `I_NxN` with 4x4 transforms (`Intra_4x4` prediction).
    Intra4x4,
    /// `I_NxN` with 8x8 transforms (`Intra_8x8` prediction).
    Intra8x8,
    /// `P_L0_16x16`, `P_L0_L0_16x8`, `P_L0_L0_8x16` or `P_8x8` (all
    /// `P_L0_8x8` sub-macroblocks), every partition with the reference
    /// index [`MbInfo::ref_idx`].
    Inter,
    /// `P_Skip`: predicted vector, no residual.
    Skip,
}

/// How an inter macroblock is partitioned for motion (`mb_type`, Table 7-13;
/// `P_8x8` always with four `P_L0_8x8` sub-macroblocks).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MbPartition {
    P16x16,
    P16x8,
    P8x16,
    P8x8,
}

impl MbPartition {
    /// Number of partitions.
    pub const fn count(self) -> usize {
        match self {
            Self::P16x16 => 1,
            Self::P16x8 | Self::P8x16 => 2,
            Self::P8x8 => 4,
        }
    }

    /// Partition `part` as `(x8, y8, w8, h8)` in 8x8 units.
    pub const fn rect(self, part: usize) -> (usize, usize, usize, usize) {
        match self {
            Self::P16x16 => (0, 0, 2, 2),
            Self::P16x8 => (0, part, 2, 1),
            Self::P8x16 => (part, 0, 1, 2),
            Self::P8x8 => (part % 2, part / 2, 1, 1),
        }
    }
}

/// Per-macroblock state kept for the rest of the picture.
#[derive(Debug, Clone, Copy)]
pub struct MbInfo {
    /// How the macroblock was coded.
    pub kind: MbKind,
    /// `CodedBlockPatternLuma`: bit `b8` set if that 8x8 block has residual.
    pub cbp_luma: u8,
    /// `CodedBlockPatternChroma` (0, 1 or 2).
    pub cbp_chroma: u8,
    /// `coded_block_flag` of the `Intra16x16DCLevel` block.
    pub luma_dc_coded: bool,
    /// `coded_block_flag` of each luma 4x4 block (AC for `I_16x16`), bit
    /// `by * 4 + bx`. With the 8x8 transform each 4x4 block takes the flag
    /// of its 8x8 block (§9.3.3.1.1.9). For inter macroblocks this is also
    /// "has non-zero coefficients", which the deblocking filter needs.
    pub luma_coded: u16,
    /// `coded_block_flag` of the two chroma DC blocks.
    pub chroma_dc_coded: [bool; 2],
    /// `coded_block_flag` of each chroma AC 4x4 block, bit `by * 2 + bx`.
    pub chroma_ac_coded: [u8; 2],
    /// The motion vector of each 8x8 quadrant, raster (zero for intra).
    pub mvs: [Mv; 4],
    /// `|mvd|` per component of each 8x8 quadrant, for the mvd context of
    /// later blocks (zero for skipped and intra macroblocks, §9.3.3.1.1.7).
    pub abs_mvd: [[u16; 2]; 4],
    /// `transform_size_8x8_flag` (0 where it is not coded).
    pub transform_8x8: bool,
    /// `intra_chroma_pred_mode` (DC for inter and skipped macroblocks).
    pub chroma_pred_mode: ChromaMode,
    /// `Intra4x4PredMode` per 4x4 block, raster (`by * 4 + bx`), or each
    /// 4x4 block's `Intra8x8PredMode`; only meaningful for `I_NxN`.
    pub intra_modes: [u8; 16],
    /// `QP_Y` (§7.4.5): the predicted QP when no `mb_qp_delta` was coded.
    /// Deblocking averages it across each edge (§8.7.2.2).
    pub qp: u8,
    /// `ref_idx_l0` of every partition (0 for skipped and intra
    /// macroblocks): one per macroblock, as the engine chooses one
    /// reference per macroblock.
    pub ref_idx: u8,
}

impl MbInfo {
    /// True for intra macroblocks.
    pub fn is_intra(&self) -> bool {
        matches!(
            self.kind,
            MbKind::Intra16x16 | MbKind::Intra4x4 | MbKind::Intra8x8
        )
    }

    /// True for `I_NxN` (`Intra_4x4` or `Intra_8x8` prediction).
    pub fn is_intra_nxn(&self) -> bool {
        matches!(self.kind, MbKind::Intra4x4 | MbKind::Intra8x8)
    }

    /// True for `P_Skip`.
    pub fn is_skip(&self) -> bool {
        self.kind == MbKind::Skip
    }

    /// Whether the luma 4x4 block at `(bx, by)` was coded.
    pub fn luma_block_coded(&self, bx: usize, by: usize) -> bool {
        (self.luma_coded >> (by * 4 + bx)) & 1 != 0
    }
}

/// The macroblocks around the current one, `None` where unavailable
/// (outside the picture; there is one slice, so nothing else is).
#[derive(Debug, Clone, Copy, Default)]
pub struct Neighbours<'a> {
    /// A: left.
    pub left: Option<&'a MbInfo>,
    /// B: above.
    pub above: Option<&'a MbInfo>,
    /// C: above right.
    pub above_right: Option<&'a MbInfo>,
    /// D: above left.
    pub above_left: Option<&'a MbInfo>,
}

impl Neighbours<'_> {
    /// `predIntra4x4PredMode` of block `(bx, by)` of the current macroblock,
    /// whose blocks decoded so far have the modes in `own` (raster), per
    /// §8.3.1.1: DC if a neighbouring block is unavailable, the neighbour's
    /// mode if it is an `I_NxN` block, DC (2) for any other macroblock,
    /// and the smaller of the two. It is also `predIntra8x8PredMode` of the
    /// 8x8 block whose top-left 4x4 block is `(bx, by)` (§8.3.2.1 reads the
    /// same neighbouring blocks: of an `Intra_4x4` neighbour the 4x4 block
    /// next to that corner, of an `Intra_8x8` one its 8x8 block).
    pub fn predicted_intra_4x4_mode(&self, bx: usize, by: usize, own: &[u8; 16]) -> u8 {
        const DC: u8 = 2;
        let of = |mb: Option<&MbInfo>, nbx: usize, nby: usize| {
            mb.map(|m| {
                if m.is_intra_nxn() {
                    m.intra_modes[nby * 4 + nbx]
                } else {
                    DC
                }
            })
        };
        let a = if bx > 0 {
            Some(own[by * 4 + bx - 1])
        } else {
            of(self.left, 3, by)
        };
        let b = if by > 0 {
            Some(own[(by - 1) * 4 + bx])
        } else {
            of(self.above, bx, 3)
        };
        match (a, b) {
            (Some(a), Some(b)) => a.min(b),
            _ => DC,
        }
    }
}

impl<'a> Neighbours<'a> {
    /// The neighbourhood of macroblock `(col, row)` in a picture `cols` wide,
    /// from the already coded macroblocks in `mbs` (raster order). Inlined:
    /// it runs about three times per macroblock, and out of line it cost
    /// ~150 instructions per call.
    #[inline]
    pub fn of(mbs: &'a [Option<MbInfo>], cols: usize, col: usize, row: usize) -> Self {
        let index = row * cols + col;
        // Every neighbour precedes the macroblock in raster order, so only
        // the coded ones are read; `get` folds the picture-edge test of the
        // row above into its bounds check.
        let coded = &mbs[..index];
        let get = |i: Option<usize>| i.and_then(|i| coded.get(i)).and_then(Option::as_ref);
        let above = index.checked_sub(cols);
        let (has_left, has_right) = (col > 0, col + 1 < cols);
        Self {
            left: get(has_left.then(|| index - 1)),
            above: get(above),
            above_right: get(above.filter(|_| has_right).map(|a| a + 1)),
            above_left: get(above.filter(|_| has_left).map(|a| a - 1)),
        }
    }
}
