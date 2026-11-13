//! The decoder's motion store: the references, vectors, inter mode,
//! interpolation filters and size of the block covering each 4x4 unit
//! (AV1 §6.10 `RefFrames`, `Mvs`, `YModes`, `InterpFilters`, `MiSizes`),
//! as the reference MV stack (`super::mv_stack`, §7.10.2), the filter
//! contexts, OBMC and warped motion read them from blocks decoded
//! earlier.
//!
//! The stack reads only rows 1 to 5 above a block and columns 1 to 5 to
//! its left (§7.10.2.2–3 scan rows -1, -3, -5, shifted by one for blocks
//! at odd positions, and the corners next to row/column -1). Any unit it
//! reads therefore lies within [`CANDIDATE_REACH`] units of the bottom or
//! right edge of the block covering it, so a block writes only those
//! units: a 64x64 block writes 5 rows and 5 columns, not 16x16 cells.
//! Units inside a block are never read again in that frame, so the store
//! is reused across frames without clearing.
use super::block_size::BlockSize;
use crate::backend::Av1Mv;
use nc_dsp::av1_mc::filter::{Av1Filter, Av1Filters};

/// `NONE` in `RefFrame[1]` (single prediction, §6.10.24).
pub(crate) const NONE_FRAME: i8 = -1;
/// `INTRA_FRAME` in `RefFrame[0]`.
pub(crate) const INTRA_FRAME: i8 = 0;
/// Rows (columns) above (left of) a block the MV stack reads.
const CANDIDATE_REACH: usize = 5;

/// The inter `YMode` of a block (§6.10.22): the four single-reference
/// modes, then the eight compound modes in `compound_mode` symbol order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[expect(
    clippy::enum_variant_names,
    reason = "the specification's names (NEARESTMV, NEAR_NEWMV, ...)"
)]
pub(crate) enum InterMode {
    #[default]
    NearestMv,
    NearMv,
    GlobalMv,
    NewMv,
    NearestNearestMv,
    NearNearMv,
    NearestNewMv,
    NewNearestMv,
    NearNewMv,
    NewNearMv,
    GlobalGlobalMv,
    NewNewMv,
}

impl InterMode {
    /// The compound modes in `compound_mode` symbol order.
    pub(crate) const COMPOUND: [Self; 8] = [
        Self::NearestNearestMv,
        Self::NearNearMv,
        Self::NearestNewMv,
        Self::NewNearestMv,
        Self::NearNewMv,
        Self::NewNearMv,
        Self::GlobalGlobalMv,
        Self::NewNewMv,
    ];

    /// A mode that codes a vector difference (libaom
    /// `have_newmv_in_inter_mode`): counted for the `new_mv` context.
    pub(crate) fn has_new_mv(self) -> bool {
        matches!(
            self,
            Self::NewMv
                | Self::NearestNewMv
                | Self::NewNearestMv
                | Self::NearNewMv
                | Self::NewNearMv
                | Self::NewNewMv
        )
    }
    /// `GLOBALMV` or `GLOBAL_GLOBALMV`.
    pub(crate) fn is_global(self) -> bool {
        matches!(self, Self::GlobalMv | Self::GlobalGlobalMv)
    }
    /// `has_nearmv()` (§5.11.23): `drl_mode` picks among the near
    /// candidates.
    pub(crate) fn has_near_mv(self) -> bool {
        matches!(
            self,
            Self::NearMv | Self::NearNearMv | Self::NearNewMv | Self::NewNearMv
        )
    }
    /// The single-reference mode of list `list` (§5.11.26 `get_mode`).
    pub(crate) fn single(self, list: usize) -> Self {
        match (self, list) {
            (Self::NearestNearestMv, _) | (Self::NearestNewMv, 0) | (Self::NewNearestMv, 1) => {
                Self::NearestMv
            }
            (Self::NearNearMv, _) | (Self::NearNewMv, 0) | (Self::NewNearMv, 1) => Self::NearMv,
            (Self::GlobalGlobalMv, _) => Self::GlobalMv,
            (
                Self::NewNewMv
                | Self::NearestNewMv
                | Self::NearNewMv
                | Self::NewNearestMv
                | Self::NewNearMv,
                _,
            ) => Self::NewMv,
            (single, _) => single,
        }
    }
}

/// Fields of [`MotionCell`]'s packed `info`: the two properties of the
/// inter mode later blocks ask about, and vertical/horizontal filters.
const GLOBAL_MODE_BIT: u16 = 1 << 6;
const NEW_MV_MODE_BIT: u16 = 1 << 7;
const FILTER_Y_SHIFT: u32 = 10;
const FILTER_X_SHIFT: u32 = 12;
const FILTER_MASK: u16 = 3;

/// One 4x4 unit: the block covering it, in 14 bytes padded to 16 so that
/// recording a block stores one 128-bit vector per unit (three scalar
/// stores per unit at 12 bytes cost ~40M instructions on static 720p).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C, align(16))]
pub(crate) struct MotionCell {
    /// `Mvs[][][list]`; zero for unused lists.
    pub mv: [Av1Mv; 2],
    /// `RefFrames[][][list]`: [`INTRA_FRAME`] and [`NONE_FRAME`] for
    /// intra blocks, [`INTRA_FRAME`] second for inter-intra.
    pub refs: [i8; 2],
    /// Mode properties and filters, packed (`*_SHIFT`, `*_BIT`).
    info: u16,
    /// Direct dimensions reuse padding without growing the grid. Neighbour
    /// scans read these repeatedly; no log2 unpacking is needed per visit.
    width_mi: u8,
    height_mi: u8,
}

impl MotionCell {
    /// An inter block of `size`.
    pub(crate) fn inter(
        size: BlockSize,
        refs: [i8; 2],
        mode: InterMode,
        mv: [Av1Mv; 2],
        filters: Av1Filters,
    ) -> Self {
        let info = if mode.is_global() { GLOBAL_MODE_BIT } else { 0 }
            | if mode.has_new_mv() {
                NEW_MV_MODE_BIT
            } else {
                0
            }
            | ((filters.y as u16) << FILTER_Y_SHIFT)
            | ((filters.x as u16) << FILTER_X_SHIFT);
        Self {
            mv,
            refs,
            info,
            width_mi: size.width_mi() as u8,
            height_mi: size.height_mi() as u8,
        }
    }
    /// An intra block of `size`.
    pub(crate) fn intra(size: BlockSize) -> Self {
        Self::inter(
            size,
            [INTRA_FRAME, NONE_FRAME],
            InterMode::default(),
            [Av1Mv::ZERO; 2],
            Av1Filters::both(Av1Filter::Regular),
        )
    }
    /// A plain intra block of a frame with intra block copy: no reference
    /// at all, so no stack takes it as a candidate.
    pub(crate) fn no_reference(size: BlockSize) -> Self {
        Self::inter(
            size,
            [NONE_FRAME, NONE_FRAME],
            InterMode::default(),
            [Av1Mv::ZERO; 2],
            Av1Filters::both(Av1Filter::Regular),
        )
    }
    pub(crate) fn is_inter(&self) -> bool {
        self.refs[0] > INTRA_FRAME
    }
    /// `Num_4x4_Blocks_Wide` of the block.
    pub(crate) fn width_mi(&self) -> usize {
        usize::from(self.width_mi)
    }
    /// `Num_4x4_Blocks_High` of the block.
    pub(crate) fn height_mi(&self) -> usize {
        usize::from(self.height_mi)
    }
    /// [`InterMode::has_new_mv`] of the block's mode.
    pub(crate) fn has_new_mv(&self) -> bool {
        self.info & NEW_MV_MODE_BIT != 0
    }
    /// [`InterMode::is_global`] of the block's mode.
    pub(crate) fn is_global(&self) -> bool {
        self.info & GLOBAL_MODE_BIT != 0
    }
    /// The block's interpolation filters (`InterpFilters`).
    pub(crate) fn filters(&self) -> Av1Filters {
        let filter = |shift: u32| Av1Filter::ALL[usize::from((self.info >> shift) & FILTER_MASK)];
        Av1Filters {
            x: filter(FILTER_X_SHIFT),
            y: filter(FILTER_Y_SHIFT),
        }
    }
}

/// The motion store of one frame, in 4x4 units.
#[derive(Debug, Default)]
pub(crate) struct MotionGrid {
    cells: Vec<MotionCell>,
    mi_cols: usize,
    mi_rows: usize,
}

impl MotionGrid {
    /// Size the store for a frame of `mi_cols` x `mi_rows` units. Every
    /// unit the MV stack reads is written by this frame first, so a store
    /// of the same size is kept as it is.
    pub(crate) fn start_frame(&mut self, mi_cols: usize, mi_rows: usize) {
        if (self.mi_cols, self.mi_rows) != (mi_cols, mi_rows) {
            self.cells = vec![MotionCell::intra(BlockSize::BLOCK_4X4); mi_cols * mi_rows];
            self.mi_cols = mi_cols;
            self.mi_rows = mi_rows;
        }
    }
    /// The unit at (`mi_row`, `mi_col`), inside the frame.
    pub(crate) fn at(&self, mi_row: usize, mi_col: usize) -> &MotionCell {
        &self.cells[mi_row * self.mi_cols + mi_col]
    }
    /// The position of unit (`mi_row`, `mi_col`) for [`Self::cell`]:
    /// a scan along a row (column) steps it by 1 ([`Self::stride`]).
    pub(crate) fn index(&self, mi_row: usize, mi_col: usize) -> usize {
        mi_row * self.mi_cols + mi_col
    }
    /// Positions between vertically adjacent units.
    pub(crate) fn stride(&self) -> usize {
        self.mi_cols
    }
    /// The unit at position `index` ([`Self::index`]), inside the frame.
    pub(crate) fn cell(&self, index: usize) -> &MotionCell {
        &self.cells[index]
    }
    /// Record the block `cell` at (`mi_col`, `mi_row`): the units of it
    /// within [`CANDIDATE_REACH`] of its bottom or right edge, inside the
    /// frame.
    pub(crate) fn fill(&mut self, (mi_col, mi_row): (usize, usize), cell: MotionCell) {
        let w4 = cell.width_mi().min(self.mi_cols - mi_col);
        let h4 = cell.height_mi().min(self.mi_rows - mi_row);
        let inner_rows = h4.saturating_sub(CANDIDATE_REACH);
        let right = w4.saturating_sub(CANDIDATE_REACH);
        let start = mi_row * self.mi_cols + mi_col;
        let mut rows = self.cells[start..].chunks_mut(self.mi_cols).take(h4);
        for row in rows.by_ref().take(inner_rows) {
            row[right..w4].fill(cell);
        }
        for row in rows {
            row[..w4].fill(cell);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_dimensions_preserve_grid_store_layout() {
        assert_eq!(std::mem::size_of::<MotionCell>(), 16);
        assert_eq!(std::mem::align_of::<MotionCell>(), 16);
        for width_log2 in 2..=7 {
            for height_log2 in 2..=7 {
                let Some(size) = BlockSize::new(width_log2, height_log2) else {
                    continue;
                };
                for cell in [MotionCell::intra(size), MotionCell::no_reference(size)] {
                    assert_eq!(cell.width_mi(), size.width() / 4);
                    assert_eq!(cell.height_mi(), size.height() / 4);
                }
            }
        }
    }

    /// The flag bits answer the specification's mode sets, and the
    /// compound modes split into the single modes `get_mode` names.
    #[test]
    fn mode_flags_and_lists_follow_the_spec() {
        use InterMode::*;
        for mode in [NearestMv, NearMv, GlobalMv, NewMv]
            .into_iter()
            .chain(InterMode::COMPOUND)
        {
            let lists = [mode.single(0), mode.single(1)];
            let new = lists.contains(&NewMv);
            let global = lists.contains(&GlobalMv);
            assert_eq!(mode.has_new_mv(), new, "{mode:?}");
            assert_eq!(mode.is_global(), global, "{mode:?}");
        }
        assert_eq!([NearNewMv.single(0), NearNewMv.single(1)], [NearMv, NewMv]);
        assert_eq!(
            [NewNearestMv.single(0), NewNearestMv.single(1)],
            [NewMv, NearestMv]
        );
        assert!(NearNearMv.has_near_mv() && !NewNewMv.has_near_mv());
    }
}
