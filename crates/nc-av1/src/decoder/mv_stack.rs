//! The reference motion-vector stack of a block (AV1 §7.10.2
//! `find_mv_stack`), read from the decoder's [`MotionGrid`]: single and
//! compound references, blocks of every size, global motion candidates,
//! temporal candidates from the motion field (§7.10.2.5–6), the extra
//! search of §7.10.2.12 and the contexts of the inter mode symbols.
//!
//! The scan follows libaom `setup_ref_mv_list` (`av1/common/mvref_common.c`,
//! BSD-2-Clause, see `THIRD_PARTY.md`), whose merged row/column weights
//! (`processed_rows`) give the same stack as the spec's per-row scans.
use super::availability::TileBounds;
use super::block_size::BlockSize;
use super::frame_header::{GlobalMotion, TOTAL_REFS, WARPEDMODEL_PREC_BITS, WarpType};
use super::motion::{INTRA_FRAME, MotionCell, MotionGrid};
use super::motion_field::{MotionField, ProjectedMv, project_mv};
use crate::backend::Av1Mv;

/// `MAX_REF_MV_STACK_SIZE`.
const MAX_REF_MV_STACK_SIZE: usize = 8;
/// `REF_CAT_LEVEL`: weight bonus of the candidates next to the block.
const REF_CAT_LEVEL: u16 = 640;
/// `MVREF_ROW_COLS`: outer rows and columns scanned (at -1, -3, -5).
const MVREF_ROW_COLS: i32 = 3;
/// `MV_BORDER`: vectors may point 16 samples beyond the frame (1/8 units).
const MV_BORDER: i32 = 16 << 3;
/// `MAX_MV_REF_CANDIDATES`: vectors the extra search guarantees.
const MAX_MV_REF_CANDIDATES: usize = 2;
/// 4x4 units of an 8x8, 16x16 and 64x64 side.
const MI_8X8: i32 = 2;
const MI_16X16: i32 = 4;
const MI_64X64: i32 = 16;
/// Weight of a single 8x8 corner candidate (`2 * Num_4x4_Blocks_Wide[BLOCK_8X8]`).
const CORNER_WEIGHT: u16 = 2 * MI_8X8 as u16;
/// Weight of a candidate added by the extra search.
const EXTRA_WEIGHT: u16 = 2;
/// Packed `mode_context` fields (libaom `GLOBALMV_OFFSET`, `REFMV_OFFSET`
/// and masks).
const GLOBALMV_OFFSET: u32 = 3;
const REFMV_OFFSET: u32 = 4;
const NEWMV_CTX_MASK: u16 = 7;
const GLOBALMV_CTX_MASK: u16 = 1;
const REFMV_CTX_MASK: u16 = 15;
/// Fraction bits of a global translation beyond a 1/8-sample vector.
const GM_TRANS_ONLY_PREC_DIFF: u32 = WARPEDMODEL_PREC_BITS - 3;
/// Eighths per sample.
const SUBPEL_PER_SAMPLE: i32 = 8;
/// Weight of a temporal candidate.
const TEMPORAL_WEIGHT: u16 = 2;
/// A temporal candidate this far (1/8 samples) from the global vector in
/// a component sets `ZeroMvContext`.
const TEMPORAL_GLOBAL_DISTANCE: i32 = 16;
/// The superblock the temporal extension stays inside (64x64), 4x4
/// units.
const TEMPORAL_SUPERBLOCK_MI: i32 = 16;

/// What a frame's MV stacks depend on beyond the motion store.
#[derive(Debug, Clone)]
pub(crate) struct MvFrame {
    pub mi_cols: usize,
    pub mi_rows: usize,
    pub allow_high_precision_mv: bool,
    pub force_integer_mv: bool,
    /// `gm_params` per reference name.
    pub global_motion: [GlobalMotion; TOTAL_REFS],
    /// `RefFrameSignBias` per reference name.
    pub sign_bias: [bool; TOTAL_REFS],
    /// `use_ref_frame_mvs`: temporal candidates from the motion field.
    pub use_ref_frame_mvs: bool,
    /// Order-hint distance from the frame to each reference, clamped to
    /// the range temporal candidates are projected over (§7.9.3).
    pub projection_distances: [i32; TOTAL_REFS],
}

/// The block a stack is built for.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MvBlock {
    pub mi_col: usize,
    pub mi_row: usize,
    pub size: BlockSize,
    /// `RefFrame[0..2]`; `RefFrame[1]` is `NONE_FRAME` for single
    /// prediction.
    pub refs: [i8; 2],
    /// The unit above-right of the block is decoded (the spec's "has been
    /// written for this frame" test of `scan_point(-1, bw4)`).
    pub top_right_decoded: bool,
    /// The tile holding the block: candidates outside it are not read.
    pub tile: TileBounds,
}

/// The reference MV stack of one block.
#[derive(Debug, Clone)]
pub(crate) struct MvStack {
    /// `RefStackMv`: candidate vectors (second list for compound).
    mvs: [[Av1Mv; 2]; MAX_REF_MV_STACK_SIZE],
    /// `WeightStack`.
    weights: [u16; MAX_REF_MV_STACK_SIZE],
    /// `NumMvFound`.
    pub count: usize,
    /// `GlobalMvs`.
    global: [Av1Mv; 2],
    /// Packed `NewMvContext`, `ZeroMvContext`, `RefMvContext`.
    mode_context: u16,
    /// The block's reach beyond the frame, applied as candidates are read.
    bounds: MvBounds,
}

/// The range a stack candidate is clamped to (§7.10.2.14 `clamp_mv_row`
/// / `clamp_mv_col`), in 1/8 samples.
#[derive(Debug, Clone, Copy)]
struct MvBounds {
    rows: (i32, i32),
    cols: (i32, i32),
}

impl MvBounds {
    /// The bounds of `block`.
    fn new(frame: &MvFrame, block: &MvBlock) -> Self {
        let to_subpel = |units: i32| units * 4 * SUBPEL_PER_SAMPLE;
        let (mi_row, mi_col) = (block.mi_row as i32, block.mi_col as i32);
        let (w4, h4) = (block.size.width_mi() as i32, block.size.height_mi() as i32);
        Self {
            rows: (
                -to_subpel(mi_row + h4) - MV_BORDER,
                to_subpel(frame.mi_rows as i32 - mi_row) + MV_BORDER,
            ),
            cols: (
                -to_subpel(mi_col + w4) - MV_BORDER,
                to_subpel(frame.mi_cols as i32 - mi_col) + MV_BORDER,
            ),
        }
    }
    fn clamp(self, mv: Av1Mv) -> Av1Mv {
        Av1Mv {
            row: i32::from(mv.row).clamp(self.rows.0, self.rows.1) as i16,
            col: i32::from(mv.col).clamp(self.cols.0, self.cols.1) as i16,
        }
    }
}

impl MvStack {
    /// `NewMvContext`.
    pub(crate) fn new_mv_context(&self) -> usize {
        usize::from(self.mode_context & NEWMV_CTX_MASK)
    }
    /// `ZeroMvContext`.
    pub(crate) fn zero_mv_context(&self) -> usize {
        usize::from((self.mode_context >> GLOBALMV_OFFSET) & GLOBALMV_CTX_MASK)
    }
    /// `RefMvContext`.
    pub(crate) fn ref_mv_context(&self) -> usize {
        usize::from((self.mode_context >> REFMV_OFFSET) & REFMV_CTX_MASK)
    }
    /// The `drl_mode` context of list position `index` (§8.3.2
    /// `DrlCtxStack`).
    pub(crate) fn drl_context(&self, index: usize) -> usize {
        let strong = |i: usize| self.weights[i] >= REF_CAT_LEVEL;
        match (strong(index), strong(index + 1)) {
            (true, false) => 1,
            (false, false) => 2,
            _ => 0,
        }
    }
    /// The vector of list `list` at stack position `index`, clamped to
    /// the block's reach (§7.10.2.14); positions past the candidates found
    /// hold the global vector (libaom `av1_get_ref_mv_from_stack`). The
    /// spec clamps every candidate when the stack is built; a block reads
    /// at most a few, so they are clamped here instead.
    pub(crate) fn vector(&self, index: usize, list: usize) -> Av1Mv {
        if index < self.count {
            self.bounds.clamp(self.mvs[index][list])
        } else {
            self.global[list]
        }
    }
    /// `GlobalMvs[list]`.
    pub(crate) fn global(&self, list: usize) -> Av1Mv {
        self.global[list]
    }
}

/// `Round2Signed(value, bits)`.
fn round2_signed(value: i32, bits: u32) -> i32 {
    let half = 1 << (bits - 1);
    if value < 0 {
        -((-value + half) >> bits)
    } else {
        (value + half) >> bits
    }
}

/// `lower_mv_precision` (§7.10.2.10).
fn lower_precision(frame: &MvFrame, mv: Av1Mv) -> Av1Mv {
    if frame.force_integer_mv {
        return integer_precision(mv);
    }
    if frame.allow_high_precision_mv {
        return mv;
    }
    let even = |v: i16| if v & 1 == 0 { v } else { v - v.signum() };
    Av1Mv {
        row: even(mv.row),
        col: even(mv.col),
    }
}

/// A vector pair as one integer, so a stack search compares each entry
/// once instead of component by component.
#[inline]
fn packed([a, b]: [Av1Mv; 2]) -> u64 {
    let bits = |v: i16| u64::from(v as u16);
    bits(a.row) | bits(a.col) << 16 | bits(b.row) << 32 | bits(b.col) << 48
}

/// `lower_mv_precision` with `force_integer_mv` (§7.10.2.10): round
/// each component to a whole sample, halves toward zero.
fn integer_precision(mv: Av1Mv) -> Av1Mv {
    let round = |v: i16| {
        let v = i32::from(v);
        let fraction = v % SUBPEL_PER_SAMPLE;
        let mut whole = v - fraction;
        if fraction.abs() > SUBPEL_PER_SAMPLE / 2 {
            whole += fraction.signum() * SUBPEL_PER_SAMPLE;
        }
        whole as i16
    };
    Av1Mv {
        row: round(mv.row),
        col: round(mv.col),
    }
}

/// The global motion vector of reference `name` at the block (§7.10.2.1
/// `setup_global_mv`, libaom `gm_get_motion_vector`).
fn global_mv(frame: &MvFrame, block: &MvBlock, name: i8) -> Av1Mv {
    if name <= INTRA_FRAME {
        return Av1Mv::ZERO;
    }
    let gm = &frame.global_motion[name as usize];
    let mv = match gm.kind {
        WarpType::Identity => return Av1Mv::ZERO,
        // The spec puts the horizontal translation in the row (libaom
        // keeps it: crbug.com/aomedia/3328).
        WarpType::Translation => Av1Mv {
            row: (gm.params[0] >> GM_TRANS_ONLY_PREC_DIFF) as i16,
            col: (gm.params[1] >> GM_TRANS_ONLY_PREC_DIFF) as i16,
        },
        WarpType::RotZoom | WarpType::Affine => {
            let m = &gm.params;
            let one = 1 << WARPEDMODEL_PREC_BITS;
            let x = (block.mi_col * 4 + block.size.width() / 2) as i32 - 1;
            let y = (block.mi_row * 4 + block.size.height() / 2) as i32 - 1;
            let xc = (m[2] - one) * x + m[3] * y + m[0];
            let yc = m[4] * x + (m[5] - one) * y + m[1];
            let to_mv = |c: i32| {
                if frame.allow_high_precision_mv {
                    round2_signed(c, WARPEDMODEL_PREC_BITS - 3)
                } else {
                    round2_signed(c, WARPEDMODEL_PREC_BITS - 2) * 2
                }
            };
            Av1Mv {
                row: to_mv(yc) as i16,
                col: to_mv(xc) as i16,
            }
        }
    };
    if frame.force_integer_mv {
        integer_precision(mv)
    } else {
        mv
    }
}

/// The running state of one stack's scan, building the stack in place.
struct Scan<'a, 's> {
    grid: &'a MotionGrid,
    field: &'a MotionField,
    frame: &'a MvFrame,
    block: MvBlock,
    /// Block size in 4x4 units.
    w4: i32,
    h4: i32,
    /// Units of the rows above (columns left) the scans cover: the block's
    /// side inside the frame, at most 64 samples.
    row_span: i32,
    col_span: i32,
    stack: &'s mut MvStack,
    /// `RefFrame[1]` is a reference.
    compound: bool,
    /// The global model of `RefFrame[list]` is more than a translation:
    /// large `GLOBALMV` candidates then contribute the block's own global
    /// vector (checked once per block, not per candidate).
    warped_global: [bool; 2],
    /// `NewMvCount` of the nearest candidates.
    new_mv_count: u32,
    /// The temporal scan set `ZeroMvContext`.
    zero_mv: bool,
}

/// What every temporal candidate of a block shares: the distances from
/// the frame to the block's references (`get_mv_projection` numerators)
/// and how many references it has.
#[derive(Clone, Copy)]
struct TemporalProjection {
    numerators: [i32; 2],
    compound: bool,
}

/// The reference MV stack of `block` (§7.10.2).
#[expect(
    clippy::inline_always,
    reason = "once per inter block from one call site; out of line it cost ~13M instructions on static 720p"
)]
#[inline(always)]
pub(crate) fn find_mv_stack(
    grid: &MotionGrid,
    field: &MotionField,
    frame: &MvFrame,
    block: MvBlock,
) -> MvStack {
    // One call site, so `global_mv` stays inlined.
    let global = block.refs.map(|name| global_mv(frame, &block, name));
    let (w4, h4) = (block.size.width_mi() as i32, block.size.height_mi() as i32);
    let mut stack = MvStack {
        mvs: [[Av1Mv::ZERO; 2]; MAX_REF_MV_STACK_SIZE],
        weights: [0; MAX_REF_MV_STACK_SIZE],
        count: 0,
        global,
        mode_context: 0,
        bounds: MvBounds::new(frame, &block),
    };
    Scan {
        grid,
        field,
        frame,
        block,
        w4,
        h4,
        row_span: w4
            .min(frame.mi_cols as i32 - block.mi_col as i32)
            .min(MI_64X64),
        col_span: h4
            .min(frame.mi_rows as i32 - block.mi_row as i32)
            .min(MI_64X64),
        stack: &mut stack,
        compound: block.refs[1] > INTRA_FRAME,
        warped_global: block.refs.map(|name| {
            name > INTRA_FRAME && frame.global_motion[name as usize].kind > WarpType::Translation
        }),
        new_mv_count: 0,
        zero_mv: false,
    }
    .run();
    stack
}

impl<'a> Scan<'a, '_> {
    #[expect(
        clippy::inline_always,
        reason = "one call site, part of `find_mv_stack`, which is always inlined"
    )]
    #[inline(always)]
    fn run(&mut self) {
        let (mi_row, mi_col) = (self.block.mi_row as i32, self.block.mi_col as i32);
        let (mi_rows, mi_cols) = (self.frame.mi_rows as i32, self.frame.mi_cols as i32);
        let tile = self.block.tile;
        // Blocks 4 units high (wide) at odd rows (columns) look one unit
        // closer, and only as far as -4 (§7.10.2.2 `deltaRow` adjustment).
        let row_adj = i32::from(self.h4 < MI_8X8 && mi_row & 1 == 1);
        let col_adj = i32::from(self.w4 < MI_8X8 && mi_col & 1 == 1);
        let reach = |side: i32, adj: i32| {
            if side < MI_8X8 {
                -(2 << 1) + adj
            } else {
                -(MVREF_ROW_COLS << 1) + adj
            }
        };
        // Rows and columns stay inside the tile (libaom
        // `find_valid_row_offset` / `find_valid_col_offset`).
        let (row_start, row_end) = (tile.row_start as i32, tile.row_end as i32);
        let (col_start, col_end) = (tile.col_start as i32, tile.col_end as i32);
        let max_row_offset = if mi_row > row_start {
            reach(self.h4, row_adj).clamp(row_start - mi_row, row_end - mi_row - 1)
        } else {
            0
        };
        let max_col_offset = if mi_col > col_start {
            reach(self.w4, col_adj).clamp(col_start - mi_col, col_end - mi_col - 1)
        } else {
            0
        };
        let mut processed_rows = 0;
        let mut processed_cols = 0;
        let mut row_matches = 0;
        let mut col_matches = 0;

        // The nearest row and column, and the top-right corner.
        if max_row_offset != 0 {
            row_matches += self.row(-1, max_row_offset, &mut processed_rows, true);
        }
        if max_col_offset != 0 {
            col_matches += self.col(-1, max_col_offset, &mut processed_cols, true);
        }
        if self.w4.max(self.h4) <= MI_64X64 && self.block.top_right_decoded {
            row_matches += self.corner(-1, self.w4, true);
        }
        let nearest_match = u32::from(row_matches > 0) + u32::from(col_matches > 0);
        let nearest_count = self.stack.count;
        for weight in &mut self.stack.weights[..nearest_count] {
            *weight += REF_CAT_LEVEL;
        }
        if self.frame.use_ref_frame_mvs {
            self.temporal();
        }

        // The top-left corner, then rows and columns 3 and 5 away.
        row_matches += self.corner(-1, -1, false);
        for idx in 2..=MVREF_ROW_COLS {
            let row_offset = -(idx << 1) + 1 + row_adj;
            let col_offset = -(idx << 1) + 1 + col_adj;
            if row_offset.abs() <= max_row_offset.abs() && row_offset.abs() > processed_rows {
                row_matches += self.row(row_offset, max_row_offset, &mut processed_rows, false);
            }
            if col_offset.abs() <= max_col_offset.abs() && col_offset.abs() > processed_cols {
                col_matches += self.col(col_offset, max_col_offset, &mut processed_cols, false);
            }
        }
        let ref_match = u32::from(row_matches > 0) + u32::from(col_matches > 0);
        self.stack.mode_context = mode_context(nearest_match, ref_match, self.new_mv_count)
            | (u16::from(self.zero_mv) << GLOBALMV_OFFSET);

        let count = self.stack.count;
        sort_by_weight(self.stack, 0, nearest_count);
        sort_by_weight(self.stack, nearest_count, count);
        if self.stack.count < MAX_MV_REF_CANDIDATES {
            let span = |side: i32, frame: i32, at: i32| side.min(MI_64X64).min(frame - at);
            let mi_size = span(self.w4, mi_cols, mi_col).min(span(self.h4, mi_rows, mi_row));
            let (above, left) = (max_row_offset != 0, max_col_offset != 0);
            if self.compound {
                self.extra_compound(mi_size, above, left);
            } else {
                self.extra_single(mi_size, above, left);
            }
        }
    }

    /// Visit the candidates along the row above and then the column left
    /// of the block, `mi_size` units of each, one per neighbouring block
    /// (the extra search's order), while `visit` returns true.
    #[expect(
        clippy::inline_always,
        reason = "one call site per search; part of the extra search, inlined with it"
    )]
    #[inline(always)]
    fn scan_edges(
        &mut self,
        mi_size: i32,
        above: bool,
        left: bool,
        mut visit: impl FnMut(&mut Self, &MotionCell) -> bool,
    ) {
        let grid = self.grid;
        let (mi_row, mi_col) = (self.block.mi_row, self.block.mi_col);
        let size = mi_size as usize;
        let mut i = 0;
        while above && i < size {
            let cand = grid.at(mi_row - 1, mi_col + i);
            if !visit(self, cand) {
                return;
            }
            i += cand.width_mi();
        }
        i = 0;
        while left && i < size {
            let cand = grid.at(mi_row + i, mi_col - 1);
            if !visit(self, cand) {
                return;
            }
            i += cand.height_mi();
        }
    }

    /// The temporal scan (§7.10.2.5, libaom's `allow_ref_frame_mvs` part
    /// of `setup_ref_mv_list`): the motion field over the block, then
    /// three positions below and right of it. As in dav1d's
    /// `dav1d_refmvs_find` (BSD-2-Clause), the field is walked in 8x8
    /// units along its rows, with the tile bounds applied to the walk, and
    /// what every candidate shares is worked out once per block.
    fn temporal(&mut self) {
        let block = self.block;
        let tile = block.tile;
        let projection = TemporalProjection {
            numerators: block
                .refs
                .map(|name| self.frame.projection_distances[name.max(0) as usize]),
            compound: self.compound,
        };
        // Each 8x8 unit is read at its odd 4x4 unit, inside the tile and
        // the block's first 64 samples; blocks of 64 or more step by 16.
        let units = |at: usize, side: i32, end: usize| {
            let first = at | 1;
            let end = (first + side.min(MI_64X64) as usize).min(end);
            end.saturating_sub(first).div_ceil(2)
        };
        let step = |side: i32| if side >= MI_64X64 { 2 } else { 1 };
        let (x8, y8) = (block.mi_col >> 1, block.mi_row >> 1);
        let columns = x8..x8 + units(block.mi_col, self.w4, tile.col_end);
        let rows = units(block.mi_row, self.h4, tile.row_end);
        // The unit at the block's corner also sets `ZeroMvContext`: when it
        // is empty, or when there is none.
        let field = self.field;
        let (row_step, column_step) = (step(self.h4), step(self.w4));
        let mut y = y8;
        while y < y8 + rows {
            let units = &field.row(y)[columns.clone()];
            let mut x = 0;
            while x < units.len() {
                let first = (x, y) == (0, y8);
                if units[x].offset != 0 {
                    self.add_temporal(projection, units[x], first);
                } else if first {
                    self.zero_mv = true;
                }
                x += column_step;
            }
            y += row_step;
        }
        if rows == 0 || columns.is_empty() {
            self.zero_mv = true;
        }
        let (w4, h4) = (self.w4, self.h4);
        let extend = (MI_8X8..MI_64X64).contains(&h4) && (MI_8X8..MI_64X64).contains(&w4);
        if !extend {
            return;
        }
        let (row0, col0) = (block.mi_row | 1, block.mi_col | 1);
        let (down, right) = (h4.max(MI_8X8), w4.max(MI_8X8));
        let (mi_row, mi_col) = (block.mi_row as i32, block.mi_col as i32);
        let inside_superblock = |offset: i32, at: i32| {
            (0..TEMPORAL_SUPERBLOCK_MI).contains(&((at & (TEMPORAL_SUPERBLOCK_MI - 1)) + offset))
        };
        for (row, col) in [(down, -2), (down, right), (down - 2, right)] {
            let at_row = row0 as isize + row as isize;
            let at_col = col0 as isize + col as isize;
            if inside_superblock(row, mi_row)
                && inside_superblock(col, mi_col)
                && tile.is_inside(at_row, at_col)
            {
                let unit = self.field.row(at_row as usize >> 1)[at_col as usize >> 1];
                if unit.offset != 0 {
                    self.add_temporal(projection, unit, false);
                }
            }
        }
    }

    /// Add the temporal candidate `unit`, a unit of the field something
    /// projected to (§7.10.2.6, libaom `add_tpl_ref_mv`; `first`, the unit
    /// at the block's corner, also sets `ZeroMvContext`).
    #[inline(never)]
    fn add_temporal(&mut self, projection: TemporalProjection, unit: ProjectedMv, first: bool) {
        let to_reference = |list: usize| {
            lower_precision(
                self.frame,
                project_mv(unit.mv, projection.numerators[list], unit.offset),
            )
        };
        let mut pair = [to_reference(0), Av1Mv::ZERO];
        if projection.compound {
            pair[1] = to_reference(1);
        }
        if first {
            let far = |list: usize| {
                let global = self.stack.global[list];
                (i32::from(pair[list].row) - i32::from(global.row)).abs()
                    >= TEMPORAL_GLOBAL_DISTANCE
                    || (i32::from(pair[list].col) - i32::from(global.col)).abs()
                        >= TEMPORAL_GLOBAL_DISTANCE
            };
            if far(0) || (projection.compound && far(1)) {
                self.zero_mv = true;
            }
        }
        self.push(pair, TEMPORAL_WEIGHT);
    }

    /// The candidate at `(row_offset, col_offset)` from the block.
    fn cell(&self, row_offset: i32, col_offset: i32) -> &'a MotionCell {
        let row = (self.block.mi_row as i32 + row_offset) as usize;
        let col = (self.block.mi_col as i32 + col_offset) as usize;
        self.grid.at(row, col)
    }

    /// Scan row `row_offset` above the block (libaom `scan_row_mbmi`);
    /// returns 1 when a candidate matched. `count_new` counts `NEWMV`
    /// candidates (nearest row only).
    #[expect(
        clippy::inline_always,
        reason = "a scan line holds ~1 candidate; out of line its set-up cost ~30M instructions on crowd 720p"
    )]
    #[inline(always)]
    fn row(
        &mut self,
        row_offset: i32,
        max_row_offset: i32,
        processed_rows: &mut i32,
        count_new: bool,
    ) -> u32 {
        let mi_col = self.block.mi_col;
        // Outer rows start one unit right, except for 4-wide blocks at
        // odd columns.
        let col_offset = if row_offset.abs() > 1 {
            usize::from(!(self.w4 < MI_8X8 && mi_col & 1 == 1))
        } else {
            0
        };
        let grid = self.grid;
        let start = grid.index(
            (self.block.mi_row as i32 + row_offset) as usize,
            mi_col + col_offset,
        );
        let use_step_16 = self.w4 >= MI_64X64;
        let mut matched = 0;
        let mut i = 0;
        while i < self.row_span {
            let cand = grid.cell(start + i as usize);
            let (cand_w4, cand_h4) = (cand.width_mi() as i32, cand.height_mi() as i32);
            let mut len = self.w4.min(cand_w4);
            if use_step_16 {
                len = len.max(MI_16X16);
            } else if row_offset.abs() > 1 {
                len = len.max(MI_8X8);
            }
            let mut weight = 2;
            if self.w4 >= MI_8X8 && self.w4 <= cand_w4 {
                let inc = (-max_row_offset + row_offset + 1).min(cand_h4);
                weight = weight.max(inc);
                *processed_rows = inc - row_offset - 1;
            }
            matched |= self.add(cand, (len * weight) as u16, count_new);
            i += len;
        }
        matched
    }

    /// Scan column `col_offset` left of the block (libaom
    /// `scan_col_mbmi`).
    #[expect(
        clippy::inline_always,
        reason = "a scan line holds ~1 candidate; out of line its set-up cost ~30M instructions on crowd 720p"
    )]
    #[inline(always)]
    fn col(
        &mut self,
        col_offset: i32,
        max_col_offset: i32,
        processed_cols: &mut i32,
        count_new: bool,
    ) -> u32 {
        let mi_row = self.block.mi_row;
        let row_offset = if col_offset.abs() > 1 {
            usize::from(!(self.h4 < MI_8X8 && mi_row & 1 == 1))
        } else {
            0
        };
        let grid = self.grid;
        let start = grid.index(
            mi_row + row_offset,
            (self.block.mi_col as i32 + col_offset) as usize,
        );
        let stride = grid.stride();
        let use_step_16 = self.h4 >= MI_64X64;
        let mut matched = 0;
        let mut i = 0;
        while i < self.col_span {
            let cand = grid.cell(start + i as usize * stride);
            let (cand_w4, cand_h4) = (cand.width_mi() as i32, cand.height_mi() as i32);
            let mut len = self.h4.min(cand_h4);
            if use_step_16 {
                len = len.max(MI_16X16);
            } else if col_offset.abs() > 1 {
                len = len.max(MI_8X8);
            }
            let mut weight = 2;
            if self.h4 >= MI_8X8 && self.h4 <= cand_h4 {
                let inc = (-max_col_offset + col_offset + 1).min(cand_w4);
                weight = weight.max(inc);
                *processed_cols = inc - col_offset - 1;
            }
            matched |= self.add(cand, (len * weight) as u16, count_new);
            i += len;
        }
        matched
    }

    /// The single candidate at (`row_offset`, `col_offset`), if inside
    /// the tile (libaom `scan_blk_mbmi`).
    #[expect(
        clippy::inline_always,
        reason = "a scan line holds ~1 candidate; out of line its set-up cost ~30M instructions on crowd 720p"
    )]
    #[inline(always)]
    fn corner(&mut self, row_offset: i32, col_offset: i32, count_new: bool) -> u32 {
        let row = self.block.mi_row as i32 + row_offset;
        let col = self.block.mi_col as i32 + col_offset;
        if !self.block.tile.is_inside(row as isize, col as isize) {
            return 0;
        }
        let cand = self.cell(row_offset, col_offset);
        self.add(cand, CORNER_WEIGHT, count_new)
    }

    /// The vector of list `list` of `cand` as this block sees it: the
    /// block's own global vector for large global-motion candidates
    /// (libaom `is_global_mv_block`).
    fn candidate_mv(&self, cand: &MotionCell, list: usize, target: usize) -> Av1Mv {
        if self.warped_global[target]
            && cand.is_global()
            && cand.width_mi().min(cand.height_mi()) >= MI_8X8 as usize
        {
            self.stack.global[target]
        } else {
            cand.mv[list]
        }
    }

    /// Add `cand` with `weight` (§7.10.2.7 `add_ref_mv_candidate`):
    /// returns 1 when it predicts from the block's references.
    #[expect(
        clippy::inline_always,
        reason = "per candidate; out of line it cost ~45M instructions on static 720p"
    )]
    #[inline(always)]
    fn add(&mut self, cand: &MotionCell, weight: u16, count_new: bool) -> u32 {
        // An intra candidate holds no reference in either list, so it
        // matches none of the block's.
        if self.compound {
            return self.add_compound(cand, weight, count_new);
        }
        let mut matched = 0;
        for list in 0..2 {
            if cand.refs[list] == self.block.refs[0] {
                let mv = self.candidate_mv(cand, list, 0);
                self.push([mv, Av1Mv::ZERO], weight);
                matched = 1;
                if count_new && cand.has_new_mv() {
                    self.new_mv_count += 1;
                }
            }
        }
        matched
    }

    /// [`Self::add`] for a compound block: the candidate must use both
    /// references, in order.
    #[inline(never)]
    fn add_compound(&mut self, cand: &MotionCell, weight: u16, count_new: bool) -> u32 {
        if cand.refs != self.block.refs {
            return 0;
        }
        let pair = [self.candidate_mv(cand, 0, 0), self.candidate_mv(cand, 1, 1)];
        self.push(pair, weight);
        if count_new && cand.has_new_mv() {
            self.new_mv_count += 1;
        }
        1
    }

    /// Add `weight` to the entry holding `mvs`, or append it.
    fn push(&mut self, mvs: [Av1Mv; 2], weight: u16) {
        let stack = &mut *self.stack;
        let key = packed(mvs);
        if let Some(i) = stack.mvs[..stack.count]
            .iter()
            .position(|&m| packed(m) == key)
        {
            stack.weights[i] += weight;
        } else if stack.count < MAX_REF_MV_STACK_SIZE {
            stack.mvs[stack.count] = mvs;
            stack.weights[stack.count] = weight;
            stack.count += 1;
        }
    }

    /// The extra search of a single-reference block (§7.10.2.12, libaom
    /// `process_single_ref_mv_candidate`): vectors of any reference,
    /// negated when its sign bias differs, until two are found.
    #[expect(
        clippy::inline_always,
        reason = "one call site; out of line it cost ~5M instructions on static 720p"
    )]
    #[inline(always)]
    fn extra_single(&mut self, mi_size: i32, above: bool, left: bool) {
        let name = self.block.refs[0] as usize;
        self.scan_edges(mi_size, above, left, |scan, cand| {
            if scan.stack.count >= MAX_MV_REF_CANDIDATES {
                return false;
            }
            for list in 0..2 {
                let cand_ref = cand.refs[list];
                if cand_ref <= INTRA_FRAME {
                    continue;
                }
                let mut mv = cand.mv[list];
                if scan.frame.sign_bias[cand_ref as usize] != scan.frame.sign_bias[name] {
                    mv = negate(mv);
                }
                let stack = &mut *scan.stack;
                if !stack.mvs[..stack.count].iter().any(|m| m[0] == mv) {
                    stack.mvs[stack.count] = [mv, Av1Mv::ZERO];
                    stack.weights[stack.count] = EXTRA_WEIGHT;
                    stack.count += 1;
                }
            }
            true
        });
    }

    /// The extra search of a compound block (§7.10.2.12, libaom
    /// `process_compound_ref_mv_candidate`): per list, vectors of the
    /// same reference first, then of other references, then the global
    /// vector, combined into pairs until the stack holds two.
    // Out of line: our own streams code no compound blocks.
    #[cold]
    #[inline(never)]
    fn extra_compound(&mut self, mi_size: i32, above: bool, left: bool) {
        let refs = self.block.refs;
        let mut same = [[Av1Mv::ZERO; 2]; 2];
        let mut same_count = [0; 2];
        let mut other = [[Av1Mv::ZERO; 2]; 2];
        let mut other_count = [0; 2];
        let bias = self.frame.sign_bias;
        self.scan_edges(mi_size, above, left, |_, cand| {
            for list in 0..2 {
                let cand_ref = cand.refs[list];
                for target in 0..2 {
                    if cand_ref == refs[target] && same_count[target] < 2 {
                        same[target][same_count[target]] = cand.mv[list];
                        same_count[target] += 1;
                    } else if cand_ref > INTRA_FRAME && other_count[target] < 2 {
                        let mut mv = cand.mv[list];
                        if bias[cand_ref as usize] != bias[refs[target] as usize] {
                            mv = negate(mv);
                        }
                        other[target][other_count[target]] = mv;
                        other_count[target] += 1;
                    }
                }
            }
            true
        });
        let mut combined = [[Av1Mv::ZERO; 2]; MAX_MV_REF_CANDIDATES];
        for target in 0..2 {
            let found = same[target][..same_count[target]]
                .iter()
                .chain(&other[target][..other_count[target]])
                .copied()
                .chain(std::iter::repeat(self.stack.global[target]));
            for (pair, mv) in combined.iter_mut().zip(found) {
                pair[target] = mv;
            }
        }
        let stack = &mut *self.stack;
        if stack.count == 1 {
            let next = if combined[0] == stack.mvs[0] {
                combined[1]
            } else {
                combined[0]
            };
            stack.mvs[1] = next;
            stack.weights[1] = EXTRA_WEIGHT;
            stack.count = 2;
        } else {
            for pair in combined {
                stack.mvs[stack.count] = pair;
                stack.weights[stack.count] = EXTRA_WEIGHT;
                stack.count += 1;
            }
        }
    }
}

fn negate(mv: Av1Mv) -> Av1Mv {
    Av1Mv {
        row: -mv.row,
        col: -mv.col,
    }
}

/// Pack the `new_mv`, `zero_mv` and `ref_mv` contexts (§7.10.2.11,
/// libaom's `switch (nearest_match)`). `ZeroMvContext` is 0 without
/// temporal candidates.
fn mode_context(nearest_match: u32, ref_match: u32, new_mv_count: u32) -> u16 {
    let (new_ctx, ref_ctx) = match nearest_match {
        0 => (u16::from(ref_match >= 1), ref_match.min(2) as u16),
        1 => (
            if new_mv_count > 0 { 2 } else { 3 },
            match ref_match {
                0 => 0,
                1 => 3,
                _ => 4,
            },
        ),
        _ => (if new_mv_count >= 1 { 4 } else { 5 }, 5),
    };
    new_ctx | (ref_ctx << REFMV_OFFSET)
}

/// Stable sort of stack entries `start..end` by decreasing weight
/// (§7.10.2.11 `sorting`; ties keep their order). The spec's bubble sort
/// and this insertion sort both give the one stable order; insertion
/// sort does less work on the short, mostly ordered runs found here.
fn sort_by_weight(stack: &mut MvStack, start: usize, end: usize) {
    let weights = &mut stack.weights[start..end];
    let mvs = &mut stack.mvs[start..end];
    for i in 1..weights.len() {
        let mut j = i;
        while j > 0 && weights[j - 1] < weights[j] {
            weights.swap(j - 1, j);
            mvs.swap(j - 1, j);
            j -= 1;
        }
    }
}
