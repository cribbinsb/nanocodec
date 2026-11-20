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
///
/// Within a tile, blocks decode superblock row by superblock row (`16` or `32`
/// 4x4 rows high), and later blocks never read more than 6 rows above the
/// current superblock row (`mv_stack` reaches `-6`, OBMC/warp/sub-8x8 chroma
/// reach `-1`). Indexing rows modulo `ring_mask + 1` (`32` rows for 64x64
/// superblocks, `64` rows for 128x128) keeps the entire store inside L2 cache
/// (~240 KiB at 1080p instead of 2.1 MiB in DRAM).
///
/// As in dav1d's `refmvs` (`src/refmvs.c` `splat_mv`, BSD-2-Clause,
/// Copyright (c) 2020, `VideoLAN` and dav1d authors; see `THIRD_PARTY.md`),
/// rows carry [`MAX_BLOCK_MI`] spare units past the frame's right edge and
/// a block is written unclipped to the frame: units outside the frame are
/// never read, so a block's runs need no clipping. A block below the
/// frame's bottom edge stays in its superblock's half of the ring, which
/// holds nothing else.
#[derive(Debug, Default)]
pub(crate) struct MotionGrid {
    cells: Vec<MotionCell>,
    /// Units per ring row: the frame's columns and the spare ones.
    stride: usize,
    ring_mask: usize,
}

/// The widest block, in 4x4 units: the spare units each row carries.
const MAX_BLOCK_MI: usize = 32;

impl MotionGrid {
    /// Size the store for a frame of `mi_cols` units across and
    /// superblock height `sb_mi` (`16` or `32` units). Every unit a reader
    /// visits is written by this frame first, so an existing buffer of the
    /// same stride and ring height is reused in place.
    pub(crate) fn start_frame(&mut self, mi_cols: usize, sb_mi: usize) {
        let ring_rows = if sb_mi <= 16 { 32 } else { 64 };
        let ring_mask = ring_rows - 1;
        let stride = mi_cols + MAX_BLOCK_MI;
        if (self.stride, self.ring_mask) != (stride, ring_mask) {
            self.cells = vec![MotionCell::intra(BlockSize::BLOCK_4X4); stride * ring_rows];
            self.stride = stride;
            self.ring_mask = ring_mask;
        }
    }
    /// The unit at (`mi_row`, `mi_col`), inside the active superblock window.
    #[inline]
    pub(crate) fn at(&self, mi_row: usize, mi_col: usize) -> &MotionCell {
        &self.cells[self.index(mi_row, mi_col)]
    }
    /// The position of unit (`mi_row`, `mi_col`) for [`Self::cell`]:
    /// a horizontal scan along a row steps it by 1.
    #[inline]
    pub(crate) fn index(&self, mi_row: usize, mi_col: usize) -> usize {
        (mi_row & self.ring_mask) * self.stride + mi_col
    }
    /// The unit at position `index` ([`Self::index`]), inside the ring.
    #[inline]
    pub(crate) fn cell(&self, index: usize) -> &MotionCell {
        &self.cells[index]
    }
    /// Record the block `cell` at (`mi_col`, `mi_row`): the units of it
    /// within [`CANDIDATE_REACH`] of its bottom or right edge, unclipped.
    ///
    /// One plain fill per row (a `splat_slice` jump table per row cost 31
    /// more instructions a block on `aom720_good4_1000k`; fills of a
    /// constant length per block width had their cells split into one
    /// store per field instead of one 128-bit store). A block's rows are
    /// contiguous in the ring (it lies inside one superblock, whose height
    /// divides the ring's), so the block's area is sliced once and walked
    /// by rows.
    #[allow(
        clippy::extra_unused_type_parameters,
        reason = "one copy per decoder instance, kept inlined (docs/handoff/high-bit-depth.md)"
    )]
    #[inline]
    pub(crate) fn fill<T>(&mut self, (mi_col, mi_row): (usize, usize), cell: MotionCell) {
        let (w4, h4) = (cell.width_mi(), cell.height_mi());
        let stride = self.stride;
        let base = self.index(mi_row, mi_col);
        let area = &mut self.cells[base..base + (h4 - 1) * stride + w4];
        let right = w4.saturating_sub(CANDIDATE_REACH);
        let mut rows = area.chunks_mut(stride);
        for row in rows.by_ref().take(h4.saturating_sub(CANDIDATE_REACH)) {
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
