//! Block geometry and per-block mode info of the full VP9 decoder: the 13
//! block sizes with their lookup tables, prediction modes, reference
//! frames, and what each decoded block stores for its neighbours, the loop
//! filter and the next frame (VP9 §6.4, §7.4; libvpx `vp9_common_data.c`,
//! `vp9_blockd.h`, BSD-3-Clause, see `THIRD_PARTY.md`).

use crate::transform::TxSize;
use crate::types::{IntraMode, Mv};

/// `BLOCK_SIZE`: every partition shape from 4x4 to 64x64.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub(super) enum BlockSize {
    B4x4 = 0,
    B4x8,
    B8x4,
    #[default]
    B8x8,
    B8x16,
    B16x8,
    B16x16,
    B16x32,
    B32x16,
    B32x32,
    B32x64,
    B64x32,
    B64x64,
}

/// `b_width_log2_lookup` / `b_height_log2_lookup`: sides in log2 4x4 units.
const WIDTH_LOG2: [u8; 13] = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4];
const HEIGHT_LOG2: [u8; 13] = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4];
/// `size_group_lookup`: the y-mode probability group of inter frames.
const SIZE_GROUP: [usize; 13] = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3];
/// `max_txsize_lookup`.
const MAX_TX: [TxSize; 13] = [
    TxSize::Tx4,
    TxSize::Tx4,
    TxSize::Tx4,
    TxSize::Tx8,
    TxSize::Tx8,
    TxSize::Tx8,
    TxSize::Tx16,
    TxSize::Tx16,
    TxSize::Tx16,
    TxSize::Tx32,
    TxSize::Tx32,
    TxSize::Tx32,
    TxSize::Tx32,
];
/// Largest transform of the 4:2:0 chroma block (`uv_txsize_lookup` cap:
/// `max_txsize_lookup[ss_size_lookup[bsize][1][1]]`); sub-8x8 blocks
/// code 4x4 chroma.
const UV_MAX_TX: [TxSize; 13] = [
    TxSize::Tx4,
    TxSize::Tx4,
    TxSize::Tx4,
    TxSize::Tx4,
    TxSize::Tx4,
    TxSize::Tx4,
    TxSize::Tx8,
    TxSize::Tx8,
    TxSize::Tx8,
    TxSize::Tx16,
    TxSize::Tx16,
    TxSize::Tx16,
    TxSize::Tx32,
];

impl BlockSize {
    pub(super) const ALL: [Self; 13] = [
        Self::B4x4,
        Self::B4x8,
        Self::B8x4,
        Self::B8x8,
        Self::B8x16,
        Self::B16x8,
        Self::B16x16,
        Self::B16x32,
        Self::B32x16,
        Self::B32x32,
        Self::B32x64,
        Self::B64x32,
        Self::B64x64,
    ];

    /// The square block of side `8 << log2_8x8` (8x8 .. 64x64).
    pub(super) fn square(log2_8x8: u8) -> Self {
        [Self::B8x8, Self::B16x16, Self::B32x32, Self::B64x64][usize::from(log2_8x8)]
    }

    /// Width in log2 4x4 units.
    pub(super) fn width_log2(self) -> u8 {
        WIDTH_LOG2[self as usize]
    }

    /// Height in log2 4x4 units.
    pub(super) fn height_log2(self) -> u8 {
        HEIGHT_LOG2[self as usize]
    }

    /// Width in 4x4 units.
    pub(super) fn width_4x4(self) -> usize {
        1 << self.width_log2()
    }

    /// Height in 4x4 units.
    pub(super) fn height_4x4(self) -> usize {
        1 << self.height_log2()
    }

    /// Width in 8x8 mode-info units (`num_8x8_blocks_wide_lookup`).
    pub(super) fn width_mi(self) -> usize {
        (self.width_4x4() / 2).max(1)
    }

    /// Height in 8x8 mode-info units.
    pub(super) fn height_mi(self) -> usize {
        (self.height_4x4() / 2).max(1)
    }

    /// Smaller than 8x8: coded with per-4x4 modes and vectors.
    pub(super) fn is_sub8x8(self) -> bool {
        self < Self::B8x8
    }

    pub(super) fn size_group(self) -> usize {
        SIZE_GROUP[self as usize]
    }

    pub(super) fn max_tx(self) -> TxSize {
        MAX_TX[self as usize]
    }

    /// The 4:2:0 chroma transform of a block with luma transform `tx`
    /// (`get_uv_tx_size`).
    pub(super) fn uv_tx(self, tx: TxSize) -> TxSize {
        tx.min(UV_MAX_TX[self as usize])
    }

    /// `partition_context_lookup`: (above, left) partition context bits;
    /// a bit set for each larger square size the block is smaller than.
    pub(super) fn partition_context(self) -> (u8, u8) {
        const ALL_SIZES: u8 = 0xf;
        (
            (ALL_SIZES << self.width_log2()) & ALL_SIZES,
            (ALL_SIZES << self.height_log2()) & ALL_SIZES,
        )
    }

    /// `subsize_lookup[partition][self]` of a square block.
    pub(super) fn subsize(self, partition: Partition) -> Self {
        let index = self as usize;
        match partition {
            Partition::None => self,
            Partition::Horizontal => Self::ALL[index - 1],
            Partition::Vertical => Self::ALL[index - 2],
            Partition::Split => Self::ALL[index - 3],
        }
    }
}

/// `PARTITION_TYPE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Partition {
    None = 0,
    Horizontal = 1,
    Vertical = 2,
    Split = 3,
}

impl Partition {
    pub(super) fn from_index(index: usize) -> Self {
        [Self::None, Self::Horizontal, Self::Vertical, Self::Split][index]
    }
}

/// `MV_REFERENCE_FRAME`.
pub(super) type RefFrame = i8;
/// No second reference (`NO_REF_FRAME`).
pub(super) const NO_REF: RefFrame = -1;
pub(super) const INTRA_FRAME: RefFrame = 0;
pub(super) const LAST_FRAME: RefFrame = 1;
pub(super) const GOLDEN_FRAME: RefFrame = 2;
pub(super) const ALTREF_FRAME: RefFrame = 3;
/// References an inter frame names (LAST, GOLDEN, ALTREF).
pub(super) const REFS_PER_FRAME: usize = 3;

/// `PREDICTION_MODE`: the ten intra modes, then the four inter modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PredMode {
    Intra(IntraMode),
    NearestMv,
    NearMv,
    ZeroMv,
    NewMv,
}

impl PredMode {
    /// The inter mode of `vp9_inter_mode_tree` leaf `index`
    /// (`INTER_OFFSET`: NEAREST, NEAR, ZERO, NEW).
    pub(super) fn inter(index: usize) -> Self {
        [Self::NearestMv, Self::NearMv, Self::ZeroMv, Self::NewMv][index]
    }

    /// `mode_2_counter`: weight in the inter-mode context.
    pub(super) fn counter(self) -> usize {
        match self {
            Self::Intra(_) => 9,
            Self::NearestMv | Self::NearMv => 0,
            Self::ZeroMv => 3,
            Self::NewMv => 1,
        }
    }

    /// The intra mode, DC for inter modes.
    pub(super) fn intra(self) -> IntraMode {
        match self {
            Self::Intra(mode) => mode,
            _ => IntraMode::Dc,
        }
    }
}

/// `SWITCHABLE_FILTERS`: the interpolation-filter "type" of an intra block
/// (and of a block without a switchable filter) in the filter context.
pub(super) const NO_SWITCHABLE_FILTER: u8 = 3;

/// One decoded block (libvpx `MODE_INFO`), referenced by every 8x8 unit
/// it covers.
#[derive(Debug, Clone, Copy)]
pub(super) struct BlockInfo {
    pub size: BlockSize,
    /// Effective skip: coded skip, or an inter block of 8x8 or more whose
    /// coefficients were all zero (`decode_block`).
    pub skip: bool,
    pub tx_size: TxSize,
    pub segment_id: u8,
    pub seg_id_predicted: bool,
    /// `ref_frame[0..2]`: `INTRA_FRAME` or a reference, and `NO_REF` or the
    /// second reference of a compound block.
    pub ref_frame: [RefFrame; 2],
    /// Luma mode (sub-8x8: the last 4x4 block's).
    pub mode: PredMode,
    /// Modes of the four 4x4 blocks of a sub-8x8 block (`bmi[].as_mode`).
    pub sub_modes: [PredMode; 4],
    pub uv_mode: IntraMode,
    /// `interp_filter` as libvpx numbers it (`EIGHTTAP` 0, `_SMOOTH` 1,
    /// `_SHARP` 2, `BILINEAR` 3), [`NO_SWITCHABLE_FILTER`] when intra.
    pub filter: u8,
    /// Vectors per reference (sub-8x8: the last 4x4 block's).
    pub mv: [Mv; 2],
    /// Vectors of the four 4x4 blocks of a sub-8x8 block.
    pub sub_mvs: [[Mv; 2]; 4],
}

impl Default for BlockInfo {
    fn default() -> Self {
        Self {
            size: BlockSize::B8x8,
            skip: false,
            tx_size: TxSize::Tx4,
            segment_id: 0,
            seg_id_predicted: false,
            ref_frame: [INTRA_FRAME, NO_REF],
            mode: PredMode::Intra(IntraMode::Dc),
            sub_modes: [PredMode::Intra(IntraMode::Dc); 4],
            uv_mode: IntraMode::Dc,
            filter: NO_SWITCHABLE_FILTER,
            mv: [Mv::ZERO; 2],
            sub_mvs: [[Mv::ZERO; 2]; 4],
        }
    }
}

impl BlockInfo {
    pub(super) fn is_inter(&self) -> bool {
        self.ref_frame[0] > INTRA_FRAME
    }

    pub(super) fn is_compound(&self) -> bool {
        self.ref_frame[1] > INTRA_FRAME
    }

    /// `get_y_mode`: the mode of 4x4 block `index` (raster in the 8x8).
    pub(super) fn y_mode(&self, index: usize) -> PredMode {
        if self.size.is_sub8x8() {
            self.sub_modes[index]
        } else {
            self.mode
        }
    }
}

/// What the next frame reads of this one (`MV_REF`).
#[derive(Debug, Clone, Copy)]
pub(super) struct MvRef {
    pub ref_frame: [RefFrame; 2],
    pub mv: [Mv; 2],
}

impl Default for MvRef {
    fn default() -> Self {
        Self {
            ref_frame: [INTRA_FRAME, NO_REF],
            mv: [Mv::ZERO; 2],
        }
    }
}
