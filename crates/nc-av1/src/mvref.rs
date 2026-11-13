//! The reference motion-vector list ("ref MV stack") of a block (AV1
//! §7.10.2 `find_mv_stack`): single-reference (`LAST_FRAME` or
//! `LAST2_FRAME`) and compound `[LAST_FRAME, LAST2_FRAME]`.
//!
//! Ported from libaom `setup_ref_mv_list` (`av1/common/mvref_common.c`,
//! BSD-2-Clause, Alliance for Open Media) for the subset this encoder
//! produces, where it is exactly equivalent:
//! - references `LAST_FRAME` and (spatial SVC, compound) `LAST2_FRAME`,
//!   both with the same sign bias;
//! - no temporal candidates (`use_ref_frame_mvs = 0`);
//! - identity global motion for every reference (`GLOBALMV` = zero);
//! - square blocks of at least 8x8, one tile, 64x64 superblocks.
//!
//! The list decides which vectors `NEARESTMV` / `NEARMV` denote, the
//! predictor `NEWMV` is coded against, and the contexts of the inter-mode
//! symbols, so the encoder must build it exactly as a decoder does. One
//! neighbour scan serves every list kind ([`Collect`]); each kind gets its own
//! monomorphised copy, so a frame with `LAST_FRAME` only runs the scan it
//! ran before compound prediction existed.

#[cfg(test)]
use crate::modeinfo::BlockMode;
use crate::modeinfo::{Av1Mv, InterRefs, MI_LOG2, ModeInfo, ModeInfoGrid};

/// Longest list kept (libaom `MAX_REF_MV_STACK_SIZE`).
pub const MAX_REF_MV_STACK_SIZE: usize = 8;
/// Weight bonus that marks candidates from the nearest row/column
/// (libaom `REF_CAT_LEVEL`).
pub const REF_CAT_LEVEL: u16 = 640;
/// Outer rows/columns scanned (libaom `MVREF_ROW_COLS`).
const MVREF_ROW_COLS: i32 = 3;
/// Vectors may point this far beyond the frame, in 1/8 samples (libaom
/// `MV_BORDER`: 16 samples).
const MV_BORDER: i32 = 16 << 3;
/// Superblock side in 4x4 units.
const SB_MI: i32 = 16;
/// 8x8 / 16x16 block sides in 4x4 units.
const MI_8X8: i32 = 2;
const MI_16X16: i32 = 4;
/// Blocks at least this wide (tall) in 4x4 units, i.e. 64x64, scan the
/// neighbouring row (column) in steps of at least 16x16 (AV1 §7.10.2.2/3
/// `useStep16 = bw4 >= 16`); smaller blocks step by the candidates' size.
const STEP_16_MIN_MI: i32 = 16;
/// Bit positions of the sub-contexts packed into `mode_context` (libaom
/// `GLOBALMV_OFFSET`, `REFMV_OFFSET`, `NEWMV_CTX_MASK`, `REFMV_CTX_MASK`).
const GLOBALMV_OFFSET: u32 = 3;
const REFMV_OFFSET: u32 = 4;
const NEWMV_CTX_MASK: u16 = 7;
const GLOBALMV_CTX_MASK: u16 = 1;
const REFMV_CTX_MASK: u16 = 15;
/// Vectors kept for `NEARESTMV` / `NEARMV` (libaom `MAX_MV_REF_CANDIDATES`).
const MAX_MV_REF_CANDIDATES: usize = 2;

/// The reference MV list of one block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MvStack {
    /// Candidate vectors, most likely first.
    pub mvs: [Av1Mv; MAX_REF_MV_STACK_SIZE],
    /// Their weights (used for the `drl_mode` contexts).
    pub weights: [u16; MAX_REF_MV_STACK_SIZE],
    /// Number of candidates found (`NumMvFound`).
    pub count: usize,
    /// Packed contexts of `new_mv`, `zero_mv` and `ref_mv`.
    mode_context: u16,
}

impl MvStack {
    /// Context of the `new_mv` symbol (`NewMvContext`).
    #[must_use]
    pub fn new_mv_context(&self) -> usize {
        usize::from(self.mode_context & NEWMV_CTX_MASK)
    }

    /// Context of the `zero_mv` symbol (`ZeroMvContext`).
    #[must_use]
    pub fn zero_mv_context(&self) -> usize {
        usize::from((self.mode_context >> GLOBALMV_OFFSET) & GLOBALMV_CTX_MASK)
    }

    /// Context of the `ref_mv` symbol (`RefMvContext`).
    #[must_use]
    pub fn ref_mv_context(&self) -> usize {
        usize::from((self.mode_context >> REFMV_OFFSET) & REFMV_CTX_MASK)
    }

    /// The vector of list position `index` as the decoder sees it: positions
    /// past the candidates found hold the global (zero) vector.
    #[must_use]
    pub fn vector(&self, index: usize) -> Av1Mv {
        if index < self.count {
            self.mvs[index]
        } else {
            Av1Mv::ZERO
        }
    }

    /// Context of the `drl_mode` symbol at list position `index` (libaom
    /// `av1_drl_ctx`).
    #[must_use]
    pub fn drl_context(&self, index: usize) -> usize {
        let strong = |i: usize| self.weights[i] >= REF_CAT_LEVEL;
        match (strong(index), strong(index + 1)) {
            (true, false) => 1,
            (false, false) => 2,
            _ => 0,
        }
    }
}

/// Position and size of the block the list is built for, in 4x4 units.
#[derive(Debug, Clone, Copy)]
struct Target {
    mi_row: i32,
    mi_col: i32,
    /// Side length (blocks are square).
    size: i32,
}

/// Which references a list is built for: what a neighbour contributes.
pub trait Collect {
    /// One list entry: a vector, or a compound pair.
    type Entry: Copy + PartialEq;
    /// The entry past the candidates found (the global, zero, vector).
    const ZERO: Self::Entry;
    /// The frame may hold compound neighbours (it has `LAST2_FRAME`).
    const LAYERED: bool;
    /// The entry neighbour `cand` at (`r`, `c`) contributes, if it predicts
    /// from this list's references (libaom `add_ref_mv_candidate`).
    fn entry(&self, grid: &ModeInfoGrid, at: (usize, usize), cand: ModeInfo)
    -> Option<Self::Entry>;
}

/// A single-reference `LAST_FRAME` list in a frame that has no other
/// reference: every inter neighbour counts.
pub struct LastOnly;

impl Collect for LastOnly {
    type Entry = Av1Mv;
    const ZERO: Av1Mv = Av1Mv::ZERO;
    const LAYERED: bool = false;

    #[inline]
    fn entry(&self, _: &ModeInfoGrid, _: (usize, usize), cand: ModeInfo) -> Option<Av1Mv> {
        cand.is_inter().then_some(cand.mv)
    }
}

/// A single-reference list in a frame with `LAST_FRAME` and `LAST2_FRAME`:
/// for `LAST2_FRAME` when `last2`, else `LAST_FRAME`.
pub struct SingleRef {
    /// The list is for `LAST2_FRAME`.
    pub last2: bool,
}

impl Collect for SingleRef {
    type Entry = Av1Mv;
    const ZERO: Av1Mv = Av1Mv::ZERO;
    const LAYERED: bool = true;

    #[inline]
    fn entry(&self, grid: &ModeInfoGrid, (r, c): (usize, usize), cand: ModeInfo) -> Option<Av1Mv> {
        match (self.last2, cand.refs()?) {
            (false, InterRefs::Last | InterRefs::LastLast2) | (true, InterRefs::Last2) => {
                Some(cand.mv)
            }
            (true, InterRefs::LastLast2) => Some(grid.mv2_at(r, c)),
            (false, InterRefs::Last2) | (true, InterRefs::Last) => None,
        }
    }
}

/// A compound `[LAST_FRAME, LAST2_FRAME]` list: only compound neighbours
/// with the same reference pair count.
pub struct CompoundPair;

impl Collect for CompoundPair {
    type Entry = [Av1Mv; 2];
    const ZERO: [Av1Mv; 2] = [Av1Mv::ZERO; 2];
    const LAYERED: bool = true;

    #[inline]
    fn entry(
        &self,
        grid: &ModeInfoGrid,
        (r, c): (usize, usize),
        cand: ModeInfo,
    ) -> Option<[Av1Mv; 2]> {
        cand.is_compound().then(|| [cand.mv, grid.mv2_at(r, c)])
    }
}

/// Candidates of one list, in scan order until sorted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Candidates<E> {
    entries: [E; MAX_REF_MV_STACK_SIZE],
    weights: [u16; MAX_REF_MV_STACK_SIZE],
    count: usize,
}

/// Running state of the scan (libaom's `refmv_count`, match counts, etc.).
struct Scan<'a, C: Collect> {
    grid: &'a ModeInfoGrid,
    target: Target,
    collect: C,
    list: Candidates<C::Entry>,
    newmv_count: u32,
}

/// What the neighbour scan leaves for the list's extension.
struct Scanned<E> {
    list: Candidates<E>,
    mode_context: u16,
    /// Furthest row and column above / left of the block that may be scanned.
    max_offsets: (i32, i32),
}

/// Scan the neighbours of `target` for `collect`'s list (AV1 §7.10.2.1 to
/// §7.10.2.9): nearest row, column and top-right, then the outer area; sort
/// both groups by weight.
fn scan_neighbours<C: Collect>(
    grid: &ModeInfoGrid,
    target: Target,
    collect: C,
) -> Scanned<C::Entry> {
    let mut scan = Scan {
        grid,
        target,
        collect,
        list: Candidates {
            entries: [C::ZERO; MAX_REF_MV_STACK_SIZE],
            weights: [0; MAX_REF_MV_STACK_SIZE],
            count: 0,
        },
        newmv_count: 0,
    };
    let Target {
        mi_row,
        mi_col,
        size,
    } = target;
    let (mi_rows, mi_cols) = (grid.mi_rows as i32, grid.mi_cols as i32);

    // Furthest row/column above/left that may be scanned (one tile:
    // `find_valid_{row,col}_offset` clamp to the frame).
    let max_row_offset = if mi_row > 0 {
        (-(MVREF_ROW_COLS << 1)).clamp(-mi_row, mi_rows - mi_row - 1)
    } else {
        0
    };
    let max_col_offset = if mi_col > 0 {
        (-(MVREF_ROW_COLS << 1)).clamp(-mi_col, mi_cols - mi_col - 1)
    } else {
        0
    };
    let mut processed_rows = 0;
    let mut processed_cols = 0;
    let mut row_matches = 0u32;
    let mut col_matches = 0u32;

    // Nearest row and column, and the top-right 8x8.
    if max_row_offset.abs() >= 1 {
        row_matches += scan.row(-1, max_row_offset, &mut processed_rows, true);
    }
    if max_col_offset.abs() >= 1 {
        col_matches += scan.col(-1, max_col_offset, &mut processed_cols, true);
    }
    if has_top_right(mi_row, mi_col, size) {
        row_matches += scan.single(-1, size, true);
    }
    let nearest_match = u32::from(row_matches > 0) + u32::from(col_matches > 0);
    let nearest_count = scan.list.count;
    for weight in &mut scan.list.weights[..nearest_count] {
        *weight += REF_CAT_LEVEL;
    }

    // Outer area: top-left 8x8, then rows/columns 3 and 5 away.
    row_matches += scan.single(-1, -1, false);
    for idx in 2..=MVREF_ROW_COLS {
        let offset = -(idx << 1) + 1;
        if offset.abs() <= max_row_offset.abs() && offset.abs() > processed_rows {
            row_matches += scan.row(offset, max_row_offset, &mut processed_rows, false);
        }
        if offset.abs() <= max_col_offset.abs() && offset.abs() > processed_cols {
            col_matches += scan.col(offset, max_col_offset, &mut processed_cols, false);
        }
    }
    let ref_match = u32::from(row_matches > 0) + u32::from(col_matches > 0);

    let mut list = scan.list;
    sort_by_weight(&mut list, 0, nearest_count);
    let count = list.count;
    sort_by_weight(&mut list, nearest_count, count);
    Scanned {
        list,
        mode_context: mode_context(nearest_match, ref_match, scan.newmv_count),
        max_offsets: (max_row_offset, max_col_offset),
    }
}

/// The scan target of the square block whose top-left luma sample is
/// (`x`, `y`) and side is `1 << log2_size` samples.
fn target_of(x: usize, y: usize, log2_size: u8) -> Target {
    Target {
        mi_row: (y >> MI_LOG2) as i32,
        mi_col: (x >> MI_LOG2) as i32,
        size: 1 << (usize::from(log2_size) - MI_LOG2),
    }
}

/// Build the reference MV list of the square block whose top-left luma
/// sample is (`x`, `y`) and side is `1 << log2_size` samples, for the
/// single reference `collect` names ([`LastOnly`] or [`SingleRef`]).
#[must_use]
pub fn find_mv_stack<C: Collect<Entry = Av1Mv>>(
    grid: &ModeInfoGrid,
    x: usize,
    y: usize,
    log2_size: u8,
    collect: C,
) -> MvStack {
    let target = target_of(x, y, log2_size);
    let scanned = scan_neighbours(grid, target, collect);
    let mut stack = MvStack {
        mvs: scanned.list.entries,
        weights: scanned.list.weights,
        count: scanned.list.count,
        mode_context: scanned.mode_context,
    };
    extend_single::<C>(grid, target, scanned.max_offsets, &mut stack);
    for mv in &mut stack.mvs[..stack.count] {
        *mv = clamp_mv_ref(*mv, target, grid);
    }
    stack
}

impl<C: Collect> Scan<'_, C> {
    /// Scan the row `row_offset` above the block (libaom `scan_row_mbmi`).
    /// Returns the number of matching candidates. `count_new` adds NEWMV
    /// candidates to the `new_mv` context count (nearest row only).
    fn row(
        &mut self,
        row_offset: i32,
        max_row_offset: i32,
        processed_rows: &mut i32,
        count_new: bool,
    ) -> u32 {
        let Target {
            mi_row,
            mi_col,
            size,
        } = self.target;
        let end = size.min(self.grid.mi_cols as i32 - mi_col).min(SB_MI);
        // Rows beyond the nearest look one 4x4 column to the right.
        let col_offset = i32::from(row_offset.abs() > 1);
        let use_step_16 = size >= STEP_16_MIN_MI;
        let mut matches = 0;
        let mut i = 0;
        while i < end {
            let (r, c) = (
                (mi_row + row_offset) as usize,
                (mi_col + col_offset + i) as usize,
            );
            let cand = *self.grid.at(r, c);
            let cand_size = cand.size_mi() as i32;
            let mut len = size.min(cand_size);
            if use_step_16 {
                len = len.max(MI_16X16);
            } else if row_offset.abs() > 1 {
                len = len.max(MI_8X8);
            }
            let mut weight = 2;
            if size >= MI_8X8 && size <= cand_size {
                let inc = (-max_row_offset + row_offset + 1).min(cand_size);
                weight = weight.max(inc);
                *processed_rows = inc - row_offset - 1;
            }
            matches += self.add((r, c), cand, (len * weight) as u16, count_new);
            i += len;
        }
        matches
    }

    /// Scan the column `col_offset` left of the block (libaom
    /// `scan_col_mbmi`).
    fn col(
        &mut self,
        col_offset: i32,
        max_col_offset: i32,
        processed_cols: &mut i32,
        count_new: bool,
    ) -> u32 {
        let Target {
            mi_row,
            mi_col,
            size,
        } = self.target;
        let end = size.min(self.grid.mi_rows as i32 - mi_row).min(SB_MI);
        let row_offset = i32::from(col_offset.abs() > 1);
        let use_step_16 = size >= STEP_16_MIN_MI;
        let mut matches = 0;
        let mut i = 0;
        while i < end {
            let (r, c) = (
                (mi_row + row_offset + i) as usize,
                (mi_col + col_offset) as usize,
            );
            let cand = *self.grid.at(r, c);
            let cand_size = cand.size_mi() as i32;
            let mut len = size.min(cand_size);
            if use_step_16 {
                len = len.max(MI_16X16);
            } else if col_offset.abs() > 1 {
                len = len.max(MI_8X8);
            }
            let mut weight = 2;
            if size >= MI_8X8 && size <= cand_size {
                let inc = (-max_col_offset + col_offset + 1).min(cand_size);
                weight = weight.max(inc);
                *processed_cols = inc - col_offset - 1;
            }
            matches += self.add((r, c), cand, (len * weight) as u16, count_new);
            i += len;
        }
        matches
    }

    /// Look at the single 8x8 at (`row_offset`, `col_offset`) from the block
    /// (libaom `scan_blk_mbmi`), if it lies in the frame.
    fn single(&mut self, row_offset: i32, col_offset: i32, count_new: bool) -> u32 {
        let row = self.target.mi_row + row_offset;
        let col = self.target.mi_col + col_offset;
        if row < 0 || col < 0 || row >= self.grid.mi_rows as i32 || col >= self.grid.mi_cols as i32
        {
            return 0;
        }
        let (r, c) = (row as usize, col as usize);
        let cand = *self.grid.at(r, c);
        self.add((r, c), cand, (2 * MI_8X8) as u16, count_new)
    }

    /// Add one neighbour's entry with `weight` (libaom
    /// `add_ref_mv_candidate`). Returns 1 if it matched the references.
    #[inline]
    fn add(&mut self, at: (usize, usize), cand: ModeInfo, weight: u16, count_new: bool) -> u32 {
        let Some(entry) = self.collect.entry(self.grid, at, cand) else {
            return 0;
        };
        let list = &mut self.list;
        if let Some(i) = list.entries[..list.count].iter().position(|&e| e == entry) {
            list.weights[i] += weight;
        } else if list.count < MAX_REF_MV_STACK_SIZE {
            list.entries[list.count] = entry;
            list.weights[list.count] = weight;
            list.count += 1;
        }
        if cand.is_new_mv() && count_new {
            self.newmv_count += 1;
        }
        1
    }
}

/// Whether the 8x8 above-right of the block is already coded (libaom
/// `has_top_right` for square blocks).
fn has_top_right(mi_row: i32, mi_col: i32, size: i32) -> bool {
    let mask_row = mi_row & (SB_MI - 1);
    let mask_col = mi_col & (SB_MI - 1);
    if size > SB_MI {
        return false;
    }
    // In a split, every quadrant but the bottom-right has its top right.
    let mut has_tr = !((mask_row & size) != 0 && (mask_col & size) != 0);
    // Moving up the quadtree: a right-hand quadrant whose parent is itself
    // a bottom-right quadrant has no decoded block to its top right.
    let mut bs = size;
    while bs < SB_MI {
        if mask_col & bs == 0 {
            break;
        }
        if (mask_col & (2 * bs)) != 0 && (mask_row & (2 * bs)) != 0 {
            has_tr = false;
            break;
        }
        bs <<= 1;
    }
    has_tr
}

/// Pack the `new_mv` / `zero_mv` / `ref_mv` contexts (libaom
/// `setup_ref_mv_list`, the `switch (nearest_match)`). The zero-MV context
/// is always 0: it is only set by temporal candidates, which are off.
fn mode_context(nearest_match: u32, ref_match: u32, newmv_count: u32) -> u16 {
    match nearest_match {
        0 => {
            let new = u16::from(ref_match >= 1);
            let refmv = match ref_match {
                0 => 0,
                1 => 1,
                _ => 2,
            };
            new | (refmv << REFMV_OFFSET)
        }
        1 => {
            let new = if newmv_count > 0 { 2 } else { 3 };
            let refmv = match ref_match {
                0 => 0,
                1 => 3,
                _ => 4,
            };
            new | (refmv << REFMV_OFFSET)
        }
        _ => {
            let new = if newmv_count >= 1 { 4 } else { 5 };
            new | (5 << REFMV_OFFSET)
        }
    }
}

/// Bubble-sort entries `[start, end)` by descending weight, exactly as
/// libaom does (the order of equal weights matters).
fn sort_by_weight<E>(list: &mut Candidates<E>, start: usize, end: usize) {
    let mut len = end;
    while len > start {
        let mut last_swap = start;
        for idx in start + 1..len {
            if list.weights[idx - 1] < list.weights[idx] {
                list.weights.swap(idx - 1, idx);
                list.entries.swap(idx - 1, idx);
                last_swap = idx;
            }
        }
        len = last_swap;
    }
}

/// With fewer than two candidates, take vectors of any inter neighbour in the
/// nearest row and column (libaom's single-reference extension; every
/// reference here has the same sign bias, so vectors are used as they are).
fn extend_single<C: Collect>(
    grid: &ModeInfoGrid,
    target: Target,
    (max_row_offset, max_col_offset): (i32, i32),
    stack: &mut MvStack,
) {
    let Target {
        mi_row,
        mi_col,
        size,
    } = target;
    let span = |limit: usize, pos: i32| size.min(SB_MI).min(limit as i32 - pos);
    let mi_size = span(grid.mi_cols, mi_col).min(span(grid.mi_rows, mi_row));
    let push_mv = |mv: Av1Mv, stack: &mut MvStack| {
        if stack.count < MAX_REF_MV_STACK_SIZE && !stack.mvs[..stack.count].contains(&mv) {
            stack.mvs[stack.count] = mv;
            stack.weights[stack.count] = 2;
            stack.count += 1;
        }
    };
    let push = |(r, c): (usize, usize), cand: ModeInfo, stack: &mut MvStack| {
        if cand.is_inter() {
            push_mv(cand.mv, stack);
            if C::LAYERED && cand.is_compound() {
                push_mv(grid.mv2_at(r, c), stack);
            }
        }
    };
    let mut idx = 0;
    while max_row_offset.abs() >= 1 && idx < mi_size && stack.count < MAX_MV_REF_CANDIDATES {
        let (r, c) = ((mi_row - 1) as usize, (mi_col + idx) as usize);
        let cand = *grid.at(r, c);
        push((r, c), cand, stack);
        idx += cand.size_mi() as i32;
    }
    let mut idx = 0;
    while max_col_offset.abs() >= 1 && idx < mi_size && stack.count < MAX_MV_REF_CANDIDATES {
        let (r, c) = ((mi_row + idx) as usize, (mi_col - 1) as usize);
        let cand = *grid.at(r, c);
        push((r, c), cand, stack);
        idx += cand.size_mi() as i32;
    }
}

/// The reference MV list of one compound block (`[LAST_FRAME, LAST2_FRAME]`,
/// `AV1 §7.10.2`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompoundMvStack {
    /// Candidate vector pairs `[mv0, mv1]`, most likely first.
    pub mvs: [[Av1Mv; 2]; MAX_REF_MV_STACK_SIZE],
    /// Their weights (used for the `drl_mode` contexts).
    pub weights: [u16; MAX_REF_MV_STACK_SIZE],
    /// Number of candidates found (`NumMvFound`).
    pub count: usize,
    /// Packed contexts of `new_mv`, `zero_mv` and `ref_mv`.
    mode_context: u16,
}

impl CompoundMvStack {
    /// Context of `new_mv` (`NewMvContext`).
    #[must_use]
    pub fn new_mv_context(&self) -> usize {
        usize::from(self.mode_context & NEWMV_CTX_MASK)
    }

    /// Context of `ref_mv` (`RefMvContext`).
    #[must_use]
    pub fn ref_mv_context(&self) -> usize {
        usize::from((self.mode_context >> REFMV_OFFSET) & REFMV_CTX_MASK)
    }

    /// Context of `compound_mode` (`CompoundModeContext`, `AV1 §7.10.3`).
    #[must_use]
    pub fn compound_mode_context(&self) -> usize {
        const MAP: [[u8; 5]; 3] = [[0, 1, 1, 1, 1], [1, 2, 3, 4, 4], [4, 4, 5, 6, 7]];
        usize::from(MAP[self.ref_mv_context() >> 1][self.new_mv_context().min(4)])
    }

    /// The vector pair of list position `index` as the decoder sees it:
    /// positions past the candidates found hold `[Av1Mv::ZERO, Av1Mv::ZERO]`.
    #[must_use]
    pub fn vector(&self, index: usize) -> [Av1Mv; 2] {
        if index < self.count {
            self.mvs[index]
        } else {
            [Av1Mv::ZERO, Av1Mv::ZERO]
        }
    }

    /// Reference vector pair used to code/decode `NEW_NEWMV` (`index`),
    /// `NEAR_NEWMV` (`index`), or `NEW_NEARMV` (`index`): when `count <= 1`,
    /// position `0` is used (`AV1 §7.10.2.12`).
    #[must_use]
    pub fn new_base_vector(&self, index: usize) -> [Av1Mv; 2] {
        self.vector(if self.count <= 1 { 0 } else { index })
    }

    /// Context of the `drl_mode` symbol at list position `index` (`AV1 §7.10.3`).
    #[must_use]
    pub fn drl_context(&self, index: usize) -> usize {
        let strong = |i: usize| self.weights[i] >= REF_CAT_LEVEL;
        match (strong(index), strong(index + 1)) {
            (true, false) => 1,
            (false, false) => 2,
            _ => 0,
        }
    }
}

/// Build the compound reference MV list (`[LAST_FRAME, LAST2_FRAME]`) of the
/// square block whose top-left luma sample is (`x`, `y`) and side is
/// `1 << log2_size` samples (`AV1 §7.10.2`).
#[must_use]
pub fn find_compound_mv_stack(
    grid: &ModeInfoGrid,
    x: usize,
    y: usize,
    log2_size: u8,
) -> CompoundMvStack {
    let target = target_of(x, y, log2_size);
    let scanned = scan_neighbours(grid, target, CompoundPair);
    let mut stack = CompoundMvStack {
        mvs: scanned.list.entries,
        weights: scanned.list.weights,
        count: scanned.list.count,
        mode_context: scanned.mode_context,
    };
    if stack.count < MAX_MV_REF_CANDIDATES {
        extend_compound(grid, target, scanned.max_offsets, &mut stack);
    }
    for pair in &mut stack.mvs[..stack.count] {
        pair[0] = clamp_mv_ref(pair[0], target, grid);
        pair[1] = clamp_mv_ref(pair[1], target, grid);
    }
    stack
}

/// Compound extension (`AV1 §7.10.2.10` `extra_search` for compound blocks):
/// collects up to 2 single-reference vectors per reference (`LAST_FRAME`,
/// `LAST2_FRAME`) from the nearest row and column and pairs them into
/// `stack` until `stack.count == 2` (`MAX_MV_REF_CANDIDATES`).
fn extend_compound(
    grid: &ModeInfoGrid,
    target: Target,
    (max_row_offset, max_col_offset): (i32, i32),
    stack: &mut CompoundMvStack,
) {
    let Target {
        mi_row,
        mi_col,
        size,
    } = target;
    let span = |limit: usize, pos: i32| size.min(SB_MI).min(limit as i32 - pos);
    let mi_size = span(grid.mi_cols, mi_col).min(span(grid.mi_rows, mi_row));
    let mut same = [[Av1Mv::ZERO; MAX_MV_REF_CANDIDATES]; 2];
    let mut same_count = [0usize; 2];
    let mut other = [[Av1Mv::ZERO; MAX_MV_REF_CANDIDATES]; 2];
    let mut other_count = [0usize; 2];
    let collect = |(r, c): (usize, usize),
                   cand: ModeInfo,
                   same: &mut [[Av1Mv; MAX_MV_REF_CANDIDATES]; 2],
                   same_count: &mut [usize; 2],
                   other: &mut [[Av1Mv; MAX_MV_REF_CANDIDATES]; 2],
                   other_count: &mut [usize; 2]| {
        if !cand.is_inter() {
            return;
        }
        let cand_lists: [(bool, Av1Mv); 2] = if cand.is_compound() {
            [(false, cand.mv), (true, grid.mv2_at(r, c))]
        } else {
            let last2 = cand.is_last2_only();
            [(last2, cand.mv), (last2, cand.mv)]
        };
        let list_count = 1 + usize::from(cand.is_compound());
        for &(cand_secondary, cand_mv) in &cand_lists[..list_count] {
            for (target_list, want_secondary) in [false, true].into_iter().enumerate() {
                if cand_secondary == want_secondary {
                    if same_count[target_list] < MAX_MV_REF_CANDIDATES {
                        same[target_list][same_count[target_list]] = cand_mv;
                        same_count[target_list] += 1;
                    }
                } else if other_count[target_list] < MAX_MV_REF_CANDIDATES {
                    other[target_list][other_count[target_list]] = cand_mv;
                    other_count[target_list] += 1;
                }
            }
        }
    };
    let mut idx = 0;
    while max_row_offset.abs() >= 1 && idx < mi_size {
        let (r, c) = ((mi_row - 1) as usize, (mi_col + idx) as usize);
        let cand = *grid.at(r, c);
        collect(
            (r, c),
            cand,
            &mut same,
            &mut same_count,
            &mut other,
            &mut other_count,
        );
        idx += cand.size_mi() as i32;
    }
    let mut idx = 0;
    while max_col_offset.abs() >= 1 && idx < mi_size {
        let (r, c) = ((mi_row + idx) as usize, (mi_col - 1) as usize);
        let cand = *grid.at(r, c);
        collect(
            (r, c),
            cand,
            &mut same,
            &mut same_count,
            &mut other,
            &mut other_count,
        );
        idx += cand.size_mi() as i32;
    }
    let mut combined = [[Av1Mv::ZERO; 2]; MAX_MV_REF_CANDIDATES];
    for target_list in 0..2 {
        let found = same[target_list][..same_count[target_list]]
            .iter()
            .chain(&other[target_list][..other_count[target_list]])
            .copied()
            .chain(std::iter::repeat(Av1Mv::ZERO));
        for (pair, mv) in combined.iter_mut().zip(found) {
            pair[target_list] = mv;
        }
    }
    if stack.count == 1 {
        let next = if combined[0] == stack.mvs[0] {
            combined[1]
        } else {
            combined[0]
        };
        stack.mvs[1] = next;
        stack.weights[1] = 2;
        stack.count = 2;
    } else {
        for pair in combined {
            stack.mvs[stack.count] = pair;
            stack.weights[stack.count] = 2;
            stack.count += 1;
        }
    }
}

/// Clamp a candidate so the block it points at stays within `MV_BORDER` of
/// the frame (libaom `clamp_mv_ref`). Always inlined: single and compound
/// lists both call it, and the single-reference list must not pay a call.
#[allow(clippy::inline_always)]
#[inline(always)]
fn clamp_mv_ref(mv: Av1Mv, target: Target, grid: &ModeInfoGrid) -> Av1Mv {
    let px = |mi: i32| mi * (1 << MI_LOG2) * 8;
    let to_left = -px(target.mi_col);
    let to_right = px(grid.mi_cols as i32 - target.size - target.mi_col);
    let to_top = -px(target.mi_row);
    let to_bottom = px(grid.mi_rows as i32 - target.size - target.mi_row);
    let block = px(target.size);
    // Clamping only moves a component towards zero's side of its bounds,
    // so the result still fits the component type.
    let clamp = |v: i16, low: i32, high: i32| i32::from(v).clamp(low, high) as i16;
    Av1Mv {
        row: clamp(
            mv.row,
            to_top - block - MV_BORDER,
            to_bottom + block + MV_BORDER,
        ),
        col: clamp(
            mv.col,
            to_left - block - MV_BORDER,
            to_right + block + MV_BORDER,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inter(size_log2_mi: u8, row: i16, col: i16, new_mv: bool) -> ModeInfo {
        ModeInfo {
            size_log2_mi,
            skip: false,
            mode_detail: InterRefs::Last.code(),
            mode: if new_mv {
                BlockMode::New
            } else {
                BlockMode::Nearest
            },
            mv: Av1Mv { row, col },
            tx_log2: size_log2_mi + MI_LOG2 as u8,
        }
    }

    #[test]
    fn top_left_block_has_an_empty_list() {
        let grid = ModeInfoGrid::new(32, 32);
        let stack = find_mv_stack(&grid, 0, 0, 4, LastOnly);
        assert_eq!(stack.count, 0);
        assert_eq!(stack.vector(0), Av1Mv::ZERO);
        assert_eq!(stack.new_mv_context(), 0);
        assert_eq!(stack.ref_mv_context(), 0);
    }

    #[test]
    fn identical_neighbours_merge_into_one_heavy_candidate() {
        let mut grid = ModeInfoGrid::new(32, 32);
        // 16x16 blocks (4x4 cells) above and to the left of the block at
        // luma (16, 16), all with the same NEWMV vector.
        for (r, c) in [(0, 0), (0, 4), (0, 8), (4, 0)] {
            grid.fill(r, c, inter(2, 8, -16, true));
        }
        let stack = find_mv_stack(&grid, 16, 16, 4, LastOnly);
        assert_eq!(stack.count, 1);
        assert_eq!(stack.mvs[0], Av1Mv { row: 8, col: -16 });
        assert!(stack.weights[0] >= REF_CAT_LEVEL);
        // Both nearest row and column match, with NEWMV neighbours.
        assert_eq!(stack.new_mv_context(), 4);
        assert_eq!(stack.ref_mv_context(), 5);
    }

    /// A 32x32 block steps through 8x8 neighbours one by one: only 64-wide
    /// blocks step by 16x16 (AV1 §7.10.2.2 `useStep16 = bw4 >= 16`). With
    /// the old 16x16 threshold, the 8x8 candidates at cells 2 and 6 were
    /// skipped and the stack disagreed with libaom and dav1d.
    #[test]
    fn blocks_below_64_scan_every_8x8_neighbour() {
        let mut grid = ModeInfoGrid::new(32, 32);
        // Four 8x8 blocks (2x2 cells) above the 32x32 block at luma (0, 32),
        // each with its own vector.
        for i in 0..4 {
            grid.fill(6, 2 * i, inter(1, 8 * (i as i16 + 1), 0, false));
        }
        let stack = find_mv_stack(&grid, 0, 32, 5, LastOnly);
        assert_eq!(stack.count, 4);
        let mut found: Vec<i16> = stack.mvs[..stack.count].iter().map(|m| m.row).collect();
        found.sort_unstable();
        assert_eq!(found, [8, 16, 24, 32]);
    }

    #[test]
    fn secondary_ref_filters_direct_scan_and_uses_other_ref_for_extension() {
        let mut grid = ModeInfoGrid::new(32, 32);
        let mut secondary = inter(2, -12, 24, true);
        secondary.mode_detail = InterRefs::Last2.code();
        grid.fill(0, 4, secondary);
        grid.fill(4, 0, inter(2, 16, -8, true));

        let primary_stack = find_mv_stack(&grid, 16, 16, 4, SingleRef { last2: false });
        assert_eq!(primary_stack.mvs[0], Av1Mv { row: 16, col: -8 });
        // `extend_single` (§7.10.2.10) appends the other reference's vector.
        assert_eq!(primary_stack.mvs[1], Av1Mv { row: -12, col: 24 });

        let secondary_stack = find_mv_stack(&grid, 16, 16, 4, SingleRef { last2: true });
        assert_eq!(secondary_stack.mvs[0], Av1Mv { row: -12, col: 24 });
        assert_eq!(secondary_stack.mvs[1], Av1Mv { row: 16, col: -8 });
    }

    #[test]
    fn top_right_availability_follows_coding_order() {
        // 16x16 blocks (size 4) inside a 64x64 superblock.
        assert!(has_top_right(0, 0, 4)); // top-left quadrant
        assert!(has_top_right(0, 4, 4)); // top row: the SB row above is coded
        assert!(has_top_right(4, 0, 4)); // bottom-left of a split
        assert!(!has_top_right(4, 4, 4)); // bottom-right of a split
        assert!(!has_top_right(4, 12, 4)); // right edge of the SB, lower row
    }
}
