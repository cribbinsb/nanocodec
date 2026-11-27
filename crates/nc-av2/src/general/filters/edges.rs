//! Which edges the deblocking filter crosses, and how far (dav2d
//! `lf_mask.c`: `dav2d_create_db_mask` and its helpers), recorded block by
//! block as the block layer decodes them.
//!
//! dav2d keeps one bit mask per filter level and 256x256 area; here each
//! 4x4 unit of a plane holds a small bit set per edge direction: bit `k`
//! says an edge of level `k` starts at the unit's left (vertical edges) or
//! top (horizontal edges) side, [`SUB_PU`] that the edge is a sub-PU edge
//! whose thresholds the filter scales down. The filter takes the highest
//! level set, as dav2d's kernels take the highest mask. A level `k` edge
//! lets the filter modify up to `LUMA_WIDTHS[k]` (`CHROMA_WIDTHS[k]`)
//! samples on each side ([`super::deblock`]).
//!
//! An outer block edge takes the smaller of the two sides' transform
//! levels; the above and left contexts hold the level of the transform
//! touching each unit's edge (dav2d `tx_lpf_y`, `tx_lpf_uv`).

use crate::general::block::size::BlockSize;
use crate::general::block::transform::{TxPartition, TxSize};

/// Bit of a unit's edge set marking a sub-PU edge (dav2d mask index 4).
pub const SUB_PU: u8 = 1 << 4;
/// Bits of the levels in a unit's edge set.
pub const LEVELS: u8 = SUB_PU - 1;
/// Largest edge level of luma.
pub const MAX_LUMA_LEVEL: u8 = 3;
/// Largest edge level of chroma.
pub const MAX_CHROMA_LEVEL: u8 = 2;
/// The sub-PU level of a block without sub-PU edges (dav2d
/// `subpu_flt_lvl`'s `max_lvl`): no sub-PU edges are added.
pub const NO_SUB_PU: u8 = 3;
/// Context value of an edge with no block before it in the tile (dav2d
/// `reset_context`): the largest level, so the first block's own level
/// decides.
pub const RESET_LUMA_CONTEXT: u8 = MAX_LUMA_LEVEL;
/// See [`RESET_LUMA_CONTEXT`].
pub const RESET_CHROMA_CONTEXT: u8 = MAX_CHROMA_LEVEL;

/// Edge direction: vertical edges separate columns (dav2d "cols"),
/// horizontal edges rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Between horizontally adjacent units.
    Vertical = 0,
    /// Between vertically adjacent units.
    Horizontal = 1,
}

/// The edges of one plane (luma, or both chroma planes, which share
/// them), per 4x4 unit of that plane and direction.
#[derive(Debug, Clone, Default)]
pub struct EdgeMap {
    /// Width in 4x4 units of the plane.
    pub columns: usize,
    /// Height in 4x4 units.
    pub rows: usize,
    /// Edge sets, row-major, per [`Direction`].
    sets: [Vec<u8>; 2],
}

impl EdgeMap {
    /// A map without edges for a plane `columns` x `rows` units.
    #[must_use]
    pub fn new(columns: usize, rows: usize) -> Self {
        Self {
            columns,
            rows,
            sets: [vec![0; columns * rows], vec![0; columns * rows]],
        }
    }

    /// The edge set of unit (`column`, `row`) in `direction`.
    #[must_use]
    pub fn get(&self, direction: Direction, column: usize, row: usize) -> u8 {
        self.sets[direction as usize][row * self.columns + column]
    }

    /// Mark a level `level` edge at the unit, or a sub-PU one.
    fn set(&mut self, direction: Direction, column: u32, row: u32, bits: u8) {
        let (column, row) = (column as usize, row as usize);
        if column < self.columns && row < self.rows {
            self.sets[direction as usize][row * self.columns + column] |= bits;
        }
    }

    /// Cap the edge at unit (`column`, `row`) in `direction` at level
    /// `limit`: the sum of its set levels (its level, as one is set at
    /// most), at most `limit`, becomes its only level, level 0 when none
    /// was set (dav2d's tile-boundary fix-ups in
    /// `dav2d_deblock_slice_cols`). The sub-PU mark stays.
    pub fn cap(&mut self, direction: Direction, column: usize, row: usize, limit: u8) {
        if column >= self.columns || row >= self.rows {
            return;
        }
        let set = &mut self.sets[direction as usize][row * self.columns + column];
        let level: u8 = (1..=MAX_LUMA_LEVEL)
            .filter(|&level| *set >> level & 1 != 0)
            .sum();
        *set = (*set & !LEVELS) | 1 << level.min(limit);
    }

    /// Remove every horizontal edge of row `line` (`direction`
    /// horizontal) or vertical edge of column `line`: AVM's
    /// `set_lpf_parameters` leaves edges on tile boundaries unfiltered in a
    /// sequence that keeps loop filters within tiles.
    pub fn clear_line(&mut self, direction: Direction, line: usize) {
        let columns = self.columns;
        let sets = &mut self.sets[direction as usize];
        match direction {
            Direction::Horizontal if line < self.rows => {
                sets[line * columns..(line + 1) * columns].fill(0);
            }
            Direction::Vertical if line < columns => {
                sets[line..]
                    .iter_mut()
                    .step_by(columns)
                    .for_each(|set| *set = 0);
            }
            _ => {}
        }
    }

    /// Move every horizontal edge of row `line` (`direction`
    /// horizontal) or vertical edge of column `line` from level `from` to
    /// level `to` (dav2d's crop of long edges at the frame's bottom and
    /// right).
    pub fn lower_line(&mut self, direction: Direction, line: usize, from: u8, to: u8) {
        let columns = self.columns;
        let sets = &mut self.sets[direction as usize];
        let lower = |set: &mut u8| {
            if *set & (1 << from) != 0 {
                *set = (*set & !(1 << from)) | 1 << to;
            }
        };
        match direction {
            Direction::Horizontal => sets[line * columns..(line + 1) * columns]
                .iter_mut()
                .for_each(lower),
            Direction::Vertical => sets[line..].iter_mut().step_by(columns).for_each(lower),
        }
    }
}

/// What the mask of one block's plane needs (dav2d `create_db_mask`'s
/// arguments).
#[derive(Debug, Clone, Copy)]
#[allow(clippy::struct_excessive_bools, reason = "the block's flags")]
pub struct BlockEdges {
    /// The block's tree size in luma 4x4 units.
    pub size: BlockSize,
    /// Its top-left luma 4x4 unit row (chroma blocks: in luma units).
    pub row: u32,
    /// Its top-left luma 4x4 unit column.
    pub column: u32,
    /// The plane is chroma (4:2:0: both sides halved).
    pub chroma: bool,
    /// The block codes transforms here (intra, or a residual).
    pub transformed: bool,
    /// Lossless segment.
    pub lossless: bool,
    /// The transform size of the plane (each part's, for a partition).
    pub tx: TxSize,
    /// The luma partition (`None` for chroma).
    pub partition: TxPartition,
    /// dav2d `subpu_flt_lvl`: log2 of the sub-PU edge spacing in 4x4
    /// units, [`NO_SUB_PU`] for none.
    pub sub_pu: u8,
    /// Sub-PU edges inside a block mark their threshold scaling (dav2d
    /// `ds_subpu_mask`: not in TIP frame mode 2).
    pub scaled_sub_pu: bool,
}

/// The block's extent in plane units, clipped to the frame.
struct Extent {
    column: u32,
    row: u32,
    width: u32,
    height: u32,
}

/// Record one block's edges in `map` (dav2d `dav2d_create_db_mask`),
/// reading and updating the transform levels of the above and left
/// contexts (`above`, `left`: the block's own entries, plane units).
pub fn record_block(
    map: &mut EdgeMap,
    edges: &BlockEdges,
    frame: (u32, u32),
    above: &mut [u8],
    left: &mut [u8],
) {
    let ss = u32::from(edges.chroma);
    let (bw4, bh4) = edges.size.units();
    let (bw_log2, bh_log2) = edges.size.log2_units();
    let extent = Extent {
        column: edges.column >> ss,
        row: edges.row >> ss,
        width: (frame.0 - edges.column).min(bw4) >> ss,
        height: (frame.1 - edges.row).min(bh4) >> ss,
    };
    let max_level = if edges.chroma {
        MAX_CHROMA_LEVEL
    } else {
        MAX_LUMA_LEVEL
    };
    let sub_pu = edges.sub_pu;
    let limit = |level: u8| level.saturating_sub(ss as u8).min(max_level);
    let (width_level, height_level);
    if edges.transformed {
        let tx = edges.tx;
        mask_partition(
            map,
            &extent,
            (edges.partition, tx),
            (limit(sub_pu), limit(sub_pu)),
            above,
            left,
        );
        width_level = sub_pu.min(tx.w);
        height_level = sub_pu.min(tx.h);
    } else {
        let (h, v) = if edges.lossless {
            (0, 0)
        } else {
            (limit(sub_pu.min(bw_log2)), limit(sub_pu.min(bh_log2)))
        };
        outer_left(map, &extent, extent.row, extent.height, h, left);
        outer_top(map, &extent, extent.column, extent.width, v, above);
        width_level = sub_pu;
        height_level = sub_pu;
    }
    if sub_pu != NO_SUB_PU {
        let halve = |level: u8| level - u8::from(edges.chroma && level != 0);
        let spacing = (1_u32 << sub_pu) >> ss;
        mask_sub_pu(
            map,
            &extent,
            (halve(width_level), halve(height_level)),
            spacing,
            edges.scaled_sub_pu,
        );
    }
}

/// The left block edge of rows `row..row + count` (dav2d
/// `mask_outer_edge_l`): each takes the smaller of `level` and the left
/// neighbour's, which then becomes `level`.
fn outer_left(
    map: &mut EdgeMap,
    extent: &Extent,
    row: u32,
    count: u32,
    level: u8,
    left: &mut [u8],
) {
    let first = (row - extent.row) as usize;
    for (y, context) in left[first..first + count as usize].iter_mut().enumerate() {
        let edge = level.min(*context);
        map.set(
            Direction::Vertical,
            extent.column,
            row + y as u32,
            1 << edge,
        );
        *context = level;
    }
}

/// The top block edge of columns `column..column + count` (dav2d
/// `mask_outer_edge_t`).
fn outer_top(
    map: &mut EdgeMap,
    extent: &Extent,
    column: u32,
    count: u32,
    level: u8,
    above: &mut [u8],
) {
    let first = (column - extent.column) as usize;
    for (x, context) in above[first..first + count as usize].iter_mut().enumerate() {
        let edge = level.min(*context);
        map.set(
            Direction::Horizontal,
            column + x as u32,
            extent.row,
            1 << edge,
        );
        *context = level;
    }
}

/// Inner vertical transform edges at columns `start, start + step, ..`
/// of the block, over the rows `rows` selects (dav2d `mask_inner_edges_v`).
fn inner_vertical(
    map: &mut EdgeMap,
    extent: &Extent,
    (start, step): (u32, u32),
    level: u8,
    rows: impl Fn(u32) -> bool,
) {
    for x in (start..extent.width).step_by(step as usize) {
        for y in (0..extent.height).filter(|&y| rows(y)) {
            map.set(
                Direction::Vertical,
                extent.column + x,
                extent.row + y,
                1 << level,
            );
        }
    }
}

/// Inner horizontal transform edges at rows `start, start + step, ..`
/// over the columns `columns` selects (dav2d `mask_inner_edges_h`).
fn inner_horizontal(
    map: &mut EdgeMap,
    extent: &Extent,
    (start, step): (u32, u32),
    level: u8,
    columns: impl Fn(u32) -> bool,
) {
    for y in (start..extent.height).step_by(step as usize) {
        for x in (0..extent.width).filter(|&x| columns(x)) {
            map.set(
                Direction::Horizontal,
                extent.column + x,
                extent.row + y,
                1 << level,
            );
        }
    }
}

/// The edges of a block's transform partition (dav2d `mask_edges_part`):
/// `limits` caps the levels (the sub-PU level, by direction).
fn mask_partition(
    map: &mut EdgeMap,
    extent: &Extent,
    (partition, tx): (TxPartition, TxSize),
    (h_limit, v_limit): (u8, u8),
    above: &mut [u8],
    left: &mut [u8],
) {
    let (tw4, th4) = tx.units();
    let (w4, h4) = (extent.width, extent.height);
    let wl = h_limit.min(tx.w);
    let hl = v_limit.min(tx.h);
    let all = |_: u32| true;
    match partition {
        TxPartition::Horizontal5 => {
            // Two quarter-height parts, a half-height part of twice the
            // width across the middle, two more quarter-height parts.
            outer_top(map, extent, extent.column, w4, hl, above);
            outer_left(map, extent, extent.row, th4.min(h4), wl, left);
            if h4 > th4 {
                let middle = (wl + 1).min(h_limit);
                outer_left(
                    map,
                    extent,
                    extent.row + th4,
                    (2 * th4).min(h4 - th4),
                    middle,
                    left,
                );
                if h4 > 3 * th4 {
                    outer_left(
                        map,
                        extent,
                        extent.row + 3 * th4,
                        th4.min(h4 - 3 * th4),
                        wl,
                        left,
                    );
                }
            }
            inner_horizontal(map, extent, (th4, 2 * th4), hl, all);
            let middle = th4..3 * th4;
            inner_vertical(map, extent, (tw4, tw4), wl, |y| !middle.contains(&y));
        }
        TxPartition::Vertical5 => {
            outer_left(map, extent, extent.row, h4, wl, left);
            outer_top(map, extent, extent.column, tw4.min(w4), hl, above);
            if w4 > tw4 {
                let middle = (hl + 1).min(v_limit);
                outer_top(
                    map,
                    extent,
                    extent.column + tw4,
                    (2 * tw4).min(w4 - tw4),
                    middle,
                    above,
                );
                if w4 > 3 * tw4 {
                    outer_top(
                        map,
                        extent,
                        extent.column + 3 * tw4,
                        tw4.min(w4 - 3 * tw4),
                        hl,
                        above,
                    );
                }
            }
            inner_vertical(map, extent, (tw4, 2 * tw4), wl, all);
            let middle = tw4..3 * tw4;
            inner_horizontal(map, extent, (th4, th4), hl, |x| !middle.contains(&x));
        }
        _ => {
            outer_left(map, extent, extent.row, h4, wl, left);
            outer_top(map, extent, extent.column, w4, hl, above);
            inner_vertical(map, extent, (tw4, tw4), wl, all);
            inner_horizontal(map, extent, (th4, th4), hl, all);
        }
    }
}

/// Sub-PU edges every `spacing` units inside the block (dav2d
/// `mask_subpu_edges`). A sub-PU edge where the block had no edge of the
/// same level is marked [`SUB_PU`] when `scaled`, except at offsets that
/// are multiples of 16 units (dav2d tests the offset against the 4-bit
/// mask `ds_sub_pu_mask`).
fn mask_sub_pu(
    map: &mut EdgeMap,
    extent: &Extent,
    (h_level, v_level): (u8, u8),
    spacing: u32,
    scaled: bool,
) {
    const SCALE_MASK: u32 = 15;
    if spacing == 0 {
        return;
    }
    let mark = |map: &mut EdgeMap, direction, column, row, offset: u32, level: u8| {
        let before = map.get(direction, column as usize, row as usize);
        let mut bits = 1 << level;
        if scaled && offset & SCALE_MASK != 0 && before & (1 << level) == 0 {
            bits |= SUB_PU;
        }
        map.set(direction, column, row, bits);
    };
    for x in (spacing..extent.width).step_by(spacing as usize) {
        for y in 0..extent.height {
            let (column, row) = (extent.column + x, extent.row + y);
            mark(map, Direction::Vertical, column, row, x, h_level);
        }
    }
    for y in (spacing..extent.height).step_by(spacing as usize) {
        for x in 0..extent.width {
            let (column, row) = (extent.column + x, extent.row + y);
            mark(map, Direction::Horizontal, column, row, y, v_level);
        }
    }
}
