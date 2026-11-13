//! VP9 block geometry, prediction modes, motion vectors and the per-8x8
//! mode-info grid shared by the encoder backend, the decoder and the loop
//! filter (VP9 §6.4 mode info, libvpx `vp9_blockd.h` / `vp9_enums.h`).
//!
//! Only square blocks of 8x8..64x64 are coded (no sub-8x8 or rectangular
//! partitions), so a block size is its log2 side in luma samples.

use crate::transform::TxSize;

/// Luma samples per mode-info unit (VP9 `MI_SIZE`).
pub(crate) const MI_SIZE: usize = 8;
/// log2 of [`MI_SIZE`].
pub(crate) const MI_LOG2: u8 = 3;
/// log2 of the superblock side (64x64).
pub(crate) const SUPERBLOCK_LOG2: u8 = 6;
/// Mode-info units per superblock side.
pub(crate) const MI_PER_SUPERBLOCK: usize = 1 << (SUPERBLOCK_LOG2 - MI_LOG2);
/// Smallest coded block: 8x8 (sub-8x8 partitions are never used).
pub(crate) const MIN_BLOCK_LOG2: u8 = MI_LOG2;

/// libvpx `BLOCK_SIZE` index of a square block (`BLOCK_8X8` = 3 .. `BLOCK_64X64` = 12).
pub(crate) const fn block_size_index(log2: u8) -> usize {
    3 * (log2 as usize - 2)
}

/// `size_group_lookup` of a square block: 8x8 -> 1, 16x16 -> 2, 32x32 and 64x64 -> 3.
pub(crate) const fn size_group(log2: u8) -> usize {
    match log2 {
        3 => 1,
        4 => 2,
        _ => 3,
    }
}

/// The largest transform of a square block (`max_txsize_lookup`).
pub(crate) fn max_tx_size(log2: u8) -> TxSize {
    TxSize::from_log2(log2.min(TxSize::Tx32.log2()))
}

/// Chroma transform of a 4:2:0 block (`uv_txsize_lookup[bsize][tx][1][1]`):
/// the luma transform, capped at the chroma block size.
pub(crate) fn uv_tx_size(block_log2: u8, tx: TxSize) -> TxSize {
    TxSize::from_log2(tx.log2().min(block_log2 - 1).max(TxSize::Tx4.log2()))
}

/// Intra prediction modes in VP9 order (`PREDICTION_MODE` 0..9).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum IntraMode {
    #[default]
    Dc = 0,
    V = 1,
    H = 2,
    D45 = 3,
    D135 = 4,
    D117 = 5,
    D153 = 6,
    D207 = 7,
    D63 = 8,
    Tm = 9,
}

impl IntraMode {
    /// All modes by index.
    pub(crate) const ALL: [IntraMode; 10] = [
        Self::Dc,
        Self::V,
        Self::H,
        Self::D45,
        Self::D135,
        Self::D117,
        Self::D153,
        Self::D207,
        Self::D63,
        Self::Tm,
    ];

    pub(crate) fn from_index(index: usize) -> Self {
        Self::ALL[index]
    }
}

/// Single-reference inter modes (`INTER_OFFSET(mode)` = mode - `NEARESTMV`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InterMode {
    Nearest = 0,
    Near = 1,
    Zero = 2,
    New = 3,
}

impl InterMode {
    /// Weight of this mode in the neighbour mode counter (`mode_2_counter`).
    pub(crate) const fn counter(self) -> usize {
        match self {
            Self::Nearest | Self::Near => 0,
            Self::Zero => 3,
            Self::New => 1,
        }
    }
}

/// `mode_2_counter` of any intra mode.
pub(crate) const INTRA_COUNTER: usize = 9;

/// A VP9 motion vector in 1/8 luma samples (`MV`: row first).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Mv {
    pub row: i32,
    pub col: i32,
}

impl Mv {
    pub(crate) const ZERO: Mv = Mv { row: 0, col: 0 };
}

/// What one coded block decided, stored at every 8x8 unit it covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ModeInfo {
    /// log2 of the block side in luma samples (3..=6).
    pub size_log2: u8,
    /// `Some` for an inter block (`LAST_FRAME` unless [`Self::secondary_ref`]
    /// is true, in which case `GOLDEN_FRAME`).
    pub inter: Option<(InterMode, Mv)>,
    /// True when an inter block predicts from the secondary reference
    /// (`GOLDEN_FRAME`); false for `LAST_FRAME` and intra blocks.
    pub secondary_ref: bool,
    /// Luma intra mode of an intra block.
    pub y_mode: IntraMode,
    /// Chroma intra mode of an intra block.
    pub uv_mode: IntraMode,
    /// The effective skip flag: coded skip, or an inter block whose
    /// coefficients were all zero (libvpx `decode_block` sets `mi->skip`
    /// then, and later skip contexts and the loop filter read it).
    pub skip: bool,
    /// Luma transform size.
    pub tx_size: TxSize,
    /// Segment id (0 unless the frame enables segmentation).
    pub segment_id: u8,
    /// `seg_id_predicted`: the id was taken from the last segment map.
    pub segment_predicted: bool,
    /// Interpolation filter of an inter block.
    pub filter: nc_dsp::Vp9Filter,
}

impl Default for ModeInfo {
    fn default() -> Self {
        Self {
            size_log2: MIN_BLOCK_LOG2,
            inter: None,
            secondary_ref: false,
            y_mode: IntraMode::Dc,
            uv_mode: IntraMode::Dc,
            skip: false,
            tx_size: TxSize::Tx4,
            segment_id: 0,
            segment_predicted: false,
            filter: nc_dsp::Vp9Filter::Regular,
        }
    }
}

impl ModeInfo {
    pub(crate) fn is_inter(&self) -> bool {
        self.inter.is_some()
    }

    /// `mode_2_counter` of the block.
    pub(crate) fn mode_counter(&self) -> usize {
        self.inter.map_or(INTRA_COUNTER, |(mode, _)| mode.counter())
    }
}

/// The frame's mode info, one entry per 8x8 unit, raster order. Entries
/// not yet coded in the current frame are stale; every reader only looks
/// above or left of the block being coded.
#[derive(Debug, Clone)]
pub(crate) struct ModeInfoGrid {
    pub cols: usize,
    pub rows: usize,
    cells: Vec<ModeInfo>,
}

impl ModeInfoGrid {
    pub(crate) fn new(cols: usize, rows: usize) -> Self {
        Self {
            cols,
            rows,
            cells: vec![ModeInfo::default(); cols * rows],
        }
    }

    pub(crate) fn at(&self, row: usize, col: usize) -> &ModeInfo {
        &self.cells[row * self.cols + col]
    }

    /// The unit at `(row + dr, col + dc)` if it lies inside the frame
    /// (libvpx `is_inside` with one tile).
    pub(crate) fn offset(&self, row: usize, col: usize, dr: isize, dc: isize) -> Option<&ModeInfo> {
        let r = row.checked_add_signed(dr)?;
        let c = col.checked_add_signed(dc)?;
        (r < self.rows && c < self.cols).then(|| self.at(r, c))
    }

    /// Above and left neighbours of the unit at `(row, col)`.
    pub(crate) fn neighbours(
        &self,
        row: usize,
        col: usize,
    ) -> (Option<&ModeInfo>, Option<&ModeInfo>) {
        (
            row.checked_sub(1).map(|r| self.at(r, col)),
            col.checked_sub(1).map(|c| self.at(row, c)),
        )
    }

    /// Store `info` over the block's units, clipped to the frame.
    pub(crate) fn fill(&mut self, row: usize, col: usize, info: ModeInfo) {
        let units = 1usize << (info.size_log2 - MI_LOG2);
        for r in row..(row + units).min(self.rows) {
            let start = r * self.cols + col;
            let end = r * self.cols + (col + units).min(self.cols);
            self.cells[start..end].fill(info);
        }
    }
}
