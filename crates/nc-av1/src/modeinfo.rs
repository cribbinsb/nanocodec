//! Per-4x4 ("mode info") state of already coded blocks.
//!
//! AV1 derives most contexts, and the reference motion-vector list, from the
//! blocks above and to the left. Every coded block writes its state into the
//! 4x4 cells it covers; later blocks read it back.

/// AV1 "mode info" unit: 4x4 luma samples.
pub const MI_LOG2: usize = 2;

/// A motion vector in AV1 units: eighths of a luma sample, row first
/// (AV1 `Mv[]`).
///
/// Components are `i16`: a conforming vector lies strictly between
/// [`MV_LOW`] and [`MV_UPP`] (AV1 §6.10.25 `is_mv_valid`), and the
/// mode-info grid stores one per 4x4 cell (10 bytes a cell instead of 16).
/// Arithmetic on components widens to `i32` first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Av1Mv {
    /// Vertical component, 1/8 luma sample.
    pub row: i16,
    /// Horizontal component, 1/8 luma sample.
    pub col: i16,
}

/// Exclusive lower bound of a valid vector component (AV1 `MV_LOW`).
pub const MV_LOW: i32 = -(1 << 14);

/// Exclusive upper bound of a valid vector component (AV1 `MV_UPP`).
pub const MV_UPP: i32 = 1 << 14;

impl Av1Mv {
    /// The zero vector (also the `GLOBALMV` vector: global motion is never
    /// signalled, so every reference has the identity model).
    pub const ZERO: Av1Mv = Av1Mv { row: 0, col: 0 };

    /// The vector `(row, col)`, or `None` unless both components are valid
    /// (strictly between [`MV_LOW`] and [`MV_UPP`]).
    #[must_use]
    pub fn checked(row: i32, col: i32) -> Option<Self> {
        let valid = |v: i32| (MV_LOW + 1..MV_UPP).contains(&v);
        (valid(row) && valid(col)).then_some(Self {
            row: row as i16,
            col: col as i16,
        })
    }
}

/// How a coded block was predicted, as far as later contexts care.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BlockMode {
    /// Intra (or not coded yet).
    #[default]
    Intra,
    /// Inter, `NEARESTMV`.
    Nearest,
    /// Inter, `NEARMV`.
    Near,
    /// Inter, `GLOBALMV`.
    Global,
    /// Inter, `NEWMV`.
    New,
}

/// State of one 4x4 cell.
#[derive(Debug, Clone, Copy, Default)]
pub struct ModeInfo {
    /// log2 of the covering block's width and height in 4x4 units.
    pub size_mi: Log2Dims,
    /// The block's `skip` flag.
    pub skip: bool,
    /// Prediction mode.
    pub mode: BlockMode,
    /// Intra blocks: the luma intra mode symbol. Inter blocks: their
    /// reference frames ([`InterRefs::code`]). One byte serves both so a
    /// cell holds 9 bytes of data whatever the stream uses. Inter blocks read
    /// it through [`Self::refs`].
    pub mode_detail: u8,
    /// The block's vector (inter blocks only; list 0 for compound blocks).
    pub mv: Av1Mv,
    /// log2 of the width and height in samples of the luma transform
    /// covering this cell (the block's own unless an inter transform tree
    /// split it).
    pub tx: Log2Dims,
}

/// A width and a height, log2 (each below 16), packed in one byte so a
/// cell stays small.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Log2Dims(u8);

impl Log2Dims {
    const HEIGHT_SHIFT: u32 = 4;
    const WIDTH_MASK: u8 = (1 << Self::HEIGHT_SHIFT) - 1;

    /// `width_log2` x `height_log2`.
    #[must_use]
    pub const fn new(width_log2: u8, height_log2: u8) -> Self {
        debug_assert!(width_log2 <= Self::WIDTH_MASK && height_log2 <= Self::WIDTH_MASK);
        Self(width_log2 | (height_log2 << Self::HEIGHT_SHIFT))
    }

    /// A square of side `1 << log2`.
    #[must_use]
    pub const fn square(log2: u8) -> Self {
        Self::new(log2, log2)
    }

    /// log2 of the width.
    #[must_use]
    pub const fn width(self) -> u8 {
        self.0 & Self::WIDTH_MASK
    }

    /// log2 of the height.
    #[must_use]
    pub const fn height(self) -> u8 {
        self.0 >> Self::HEIGHT_SHIFT
    }
}

/// The reference frames of an inter block (AV1 §5.11.25 `RefFrame[0..2]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterRefs {
    /// `LAST_FRAME` alone (the primary reference).
    Last,
    /// `LAST2_FRAME` alone (the secondary or inter-layer reference).
    Last2,
    /// Compound `[LAST_FRAME, LAST2_FRAME]`.
    LastLast2,
}

impl InterRefs {
    /// The value stored in [`ModeInfo::mode_detail`].
    #[must_use]
    pub const fn code(self) -> u8 {
        self as u8
    }

    const fn from_code(code: u8) -> Self {
        match code {
            1 => Self::Last2,
            2 => Self::LastLast2,
            _ => Self::Last,
        }
    }

    /// The reference of a single-reference block: `LAST2_FRAME` when
    /// `secondary`, else `LAST_FRAME`.
    #[must_use]
    pub const fn single(secondary: bool) -> Self {
        if secondary { Self::Last2 } else { Self::Last }
    }
}

impl ModeInfo {
    /// The cell of a `1 << log2_width` x `1 << log2_height` block coded
    /// with one transform, otherwise default.
    #[must_use]
    pub fn block(log2_width: u8, log2_height: u8) -> Self {
        let mi = MI_LOG2 as u8;
        Self {
            size_mi: Log2Dims::new(log2_width - mi, log2_height - mi),
            tx: Log2Dims::new(log2_width, log2_height),
            ..Self::default()
        }
    }

    /// Inter (predicted from `LAST_FRAME` / `LAST2_FRAME`) rather than intra.
    #[must_use]
    pub fn is_inter(&self) -> bool {
        self.mode != BlockMode::Intra
    }

    /// The reference frames of an inter block; `None` on intra blocks.
    #[must_use]
    pub fn refs(&self) -> Option<InterRefs> {
        self.is_inter()
            .then(|| InterRefs::from_code(self.mode_detail))
    }

    /// Compound `[LAST_FRAME, LAST2_FRAME]` inter block (`AV1 §5.11.25`).
    #[must_use]
    pub fn is_compound(&self) -> bool {
        self.refs() == Some(InterRefs::LastLast2)
    }

    /// Single-reference inter block predicted from `LAST2_FRAME`.
    #[must_use]
    pub fn is_last2_only(&self) -> bool {
        self.refs() == Some(InterRefs::Last2)
    }

    /// Inter block that uses `LAST_FRAME` (alone or in a compound pair).
    #[must_use]
    pub fn uses_last(&self) -> bool {
        matches!(self.refs(), Some(InterRefs::Last | InterRefs::LastLast2))
    }

    /// Inter block that uses `LAST2_FRAME` (alone or in a compound pair).
    #[must_use]
    pub fn uses_last2(&self) -> bool {
        matches!(self.refs(), Some(InterRefs::Last2 | InterRefs::LastLast2))
    }

    /// Coded with `NEWMV` (libaom `have_newmv_in_inter_mode`, which feeds
    /// the reference-list mode contexts).
    #[must_use]
    pub fn is_new_mv(&self) -> bool {
        self.mode == BlockMode::New
    }

    /// Width of the covering block, in 4x4 units.
    #[must_use]
    pub fn width_mi(&self) -> usize {
        1 << self.size_mi.width()
    }

    /// Height of the covering block, in 4x4 units.
    #[must_use]
    pub fn height_mi(&self) -> usize {
        1 << self.size_mi.height()
    }
}

/// The mode-info grid of one frame (`mi_rows` x `mi_cols` cells).
#[derive(Debug, Clone)]
pub struct ModeInfoGrid {
    cells: Vec<ModeInfo>,
    /// Secondary motion vector (`RefFrame[1] = LAST2_FRAME`) for cells with
    /// [`ModeInfo::is_compound`], kept in a separate table so [`ModeInfo`] stays
    /// 10 bytes per 4x4 cell and single-reference frames never touch `mv2_cells`.
    mv2_cells: Vec<Av1Mv>,
    /// Columns, in 4x4 units.
    pub mi_cols: usize,
    /// Rows, in 4x4 units.
    pub mi_rows: usize,
}

/// A square of a [`ModeInfoGrid`] noted by [`ModeInfoGrid::save_rect`].
#[derive(Debug, Default)]
pub struct SavedCells {
    cells: Vec<ModeInfo>,
    mv2: Vec<Av1Mv>,
}

impl ModeInfoGrid {
    /// A grid of default (intra, unskipped) cells.
    #[must_use]
    pub fn new(mi_cols: usize, mi_rows: usize) -> Self {
        Self {
            cells: vec![ModeInfo::default(); mi_cols * mi_rows],
            mv2_cells: Vec::new(),
            mi_cols,
            mi_rows,
        }
    }

    /// The share of the cells coded with `skip` set.
    #[must_use]
    pub fn skip_share(&self) -> f32 {
        let skipped = self.cells.iter().filter(|cell| cell.skip).count();
        skipped as f32 / self.cells.len().max(1) as f32
    }

    /// Reset every cell (start of a frame: no information crosses frames).
    pub fn clear(&mut self) {
        self.cells.fill(ModeInfo::default());
    }

    /// The cell at (`row`, `col`).
    #[must_use]
    pub fn at(&self, row: usize, col: usize) -> &ModeInfo {
        &self.cells[row * self.mi_cols + col]
    }

    /// The secondary vector (`RefFrame[1] = LAST2_FRAME`) at (`row`, `col`) for
    /// a compound inter cell.
    #[must_use]
    pub fn mv2_at(&self, row: usize, col: usize) -> Av1Mv {
        self.mv2_cells
            .get(row * self.mi_cols + col)
            .copied()
            .unwrap_or(Av1Mv::ZERO)
    }

    /// The cells of row `row`.
    #[cfg(test)]
    #[must_use]
    pub fn row(&self, row: usize) -> &[ModeInfo] {
        &self.cells[row * self.mi_cols..(row + 1) * self.mi_cols]
    }

    /// Record `info` for the `side` x `side` cells from (`row`, `col`),
    /// clipped to the frame (a transform inside a larger block).
    pub fn fill_square(&mut self, row: usize, col: usize, side: usize, info: ModeInfo) {
        let grid = (self.mi_cols, self.mi_rows);
        nc_common::cell_grid::fill_square(&mut self.cells, grid, (row, col), side, info);
    }

    /// Record `info` for the cells of its block (`info.width_mi()` x
    /// `info.height_mi()`) whose top-left cell is (`row`, `col`), clipped to
    /// the frame.
    pub fn fill(&mut self, row: usize, col: usize, info: ModeInfo) {
        let grid = (self.mi_cols, self.mi_rows);
        let size = (info.width_mi(), info.height_mi());
        nc_common::cell_grid::fill_rect(&mut self.cells, grid, (row, col), size, info);
    }

    /// Note the `side` x `side` cells from (`row`, `col`), clipped to the
    /// frame, with their secondary vectors (a coding trial,
    /// `backend/node_trial.rs`).
    pub fn save_rect(
        &self,
        (row, col): (usize, usize),
        (width, height): (usize, usize),
        saved: &mut SavedCells,
    ) {
        let grid = (self.mi_cols, self.mi_rows);
        nc_common::cell_grid::save_rect(
            &self.cells,
            grid,
            (row, col),
            (width, height),
            &mut saved.cells,
        );
        saved.mv2.clear();
        if !self.mv2_cells.is_empty() {
            nc_common::cell_grid::save_rect(
                &self.mv2_cells,
                grid,
                (row, col),
                (width, height),
                &mut saved.mv2,
            );
        }
    }

    /// Put back the cells [`Self::save_rect`] noted of the same block.
    pub fn restore_rect(
        &mut self,
        (row, col): (usize, usize),
        (width, height): (usize, usize),
        saved: &SavedCells,
    ) {
        let grid = (self.mi_cols, self.mi_rows);
        nc_common::cell_grid::restore_rect(
            &mut self.cells,
            grid,
            (row, col),
            (width, height),
            &saved.cells,
        );
        if !saved.mv2.is_empty() {
            nc_common::cell_grid::restore_rect(
                &mut self.mv2_cells,
                grid,
                (row, col),
                (width, height),
                &saved.mv2,
            );
        } else if !self.mv2_cells.is_empty() {
            // The table was created inside the trial: its block was zero.
            nc_common::cell_grid::fill_rect(
                &mut self.mv2_cells,
                grid,
                (row, col),
                (width, height),
                Av1Mv::ZERO,
            );
        }
    }

    /// Record secondary vector `mv2` for the `side` x `side` compound cells from
    /// (`row`, `col`), clipped to the frame.
    pub fn fill_mv2(&mut self, row: usize, col: usize, side: usize, mv2: Av1Mv) {
        if self.mv2_cells.len() < self.cells.len() {
            self.mv2_cells.resize(self.cells.len(), Av1Mv::ZERO);
        }
        let grid = (self.mi_cols, self.mi_rows);
        nc_common::cell_grid::fill_square(&mut self.mv2_cells, grid, (row, col), side, mv2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_info_cell_stays_compact() {
        // One cell per 4x4 block: 57,600 at 720p, read for every block's
        // contexts and vector list.
        assert_eq!(size_of::<ModeInfo>(), 10);
    }

    #[test]
    fn checked_vectors_are_strictly_inside_the_range() {
        assert_eq!(
            Av1Mv::checked(MV_UPP - 1, MV_LOW + 1),
            Some(Av1Mv {
                row: (MV_UPP - 1) as i16,
                col: (MV_LOW + 1) as i16
            })
        );
        assert_eq!(Av1Mv::checked(MV_UPP, 0), None);
        assert_eq!(Av1Mv::checked(0, MV_LOW), None);
    }
}
