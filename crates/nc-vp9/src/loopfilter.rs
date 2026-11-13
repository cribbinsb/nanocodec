//! VP9 in-loop deblocking of a finished frame (VP9 §8.8; libvpx
//! `vp9_loopfilter.c` `vp9_filter_block_plane_non420` and
//! `vpx_dsp/loopfilter.c`, BSD-3-Clause, see `THIRD_PARTY.md`).
//!
//! Edge masks are built per superblock from the mode-info grid: every
//! transform edge is filtered except inside skipped inter blocks, with the
//! filter width from the transform size (16-wide on 16x16/32x32 edges,
//! 8-wide on 8x8 edges and every 32-sample boundary, 4-wide otherwise).
//! Superblocks run in raster order, each plane filtering all vertical edges
//! before the horizontal ones. Each block filters with its segment's level
//! (the frame level without segmentation; reference/mode deltas are off);
//! a block whose level is 0 filters none of its edges.

use nc_common::{Picture, Plane, PlaneIndex};
use nc_dsp::{Vp9FilterLimits as Limits, Vp9FilterWidth as Width};

use crate::segmentation::MAX_SEGMENTS;
use crate::transform::TxSize;
use crate::types::{MI_PER_SUPERBLOCK, MI_SIZE, ModeInfo, ModeInfoGrid, uv_tx_size};

/// Largest filter level (6 bits).
pub(crate) const MAX_LEVEL: u8 = 63;

/// Filter the `units` x 8 lines across the edge at plane sample `(x, y)`;
/// lines beyond the plane are skipped.
fn filter_edge(
    dsp: &nc_dsp::Dsp,
    plane: &mut Plane,
    (x, y): (usize, usize),
    (vertical, units): (bool, usize),
    width: Width,
    limits: Limits,
) {
    let (plane_w, plane_h) = (plane.width(), plane.height());
    let mut view = plane.view_mut();
    let stride = view.stride();
    let span = units * MI_SIZE;
    let (step, along, lines) = if vertical {
        (1, stride, span.min(plane_h - y))
    } else {
        (stride, 1, span.min(plane_w - x))
    };
    dsp.vp9_filter_edge(
        view.data_mut(),
        y * stride + x,
        step,
        along,
        lines,
        width,
        limits,
    );
}

/// Edges of one 8x8 unit (in plane samples) in one direction.
#[derive(Default, Clone, Copy)]
struct UnitEdges {
    /// Filter at the unit's left/top edge, if any.
    edge: Option<Width>,
    /// 4-wide filter at the internal edge 4 samples in (4x4 transforms).
    internal: bool,
}

/// Masks of one direction for a unit, as `vp9_filter_block_plane_non420`
/// builds them. `c` is the unit index along the direction inside the
/// superblock (in plane 8x8 units), `skip_border` drops the internal
/// edge of a chroma unit in the frame's last 8x8 column/row.
fn unit_edges(
    info: &ModeInfo,
    tx: TxSize,
    c: usize,
    block_edge: bool,
    skip_border: bool,
) -> UnitEdges {
    let skip_inter = info.skip && info.is_inter();
    let skip_this = skip_inter && !block_edge;
    let edge = if skip_this {
        None
    } else {
        match tx {
            TxSize::Tx32 if c.is_multiple_of(4) => Some(if skip_border {
                Width::Eight
            } else {
                Width::Sixteen
            }),
            TxSize::Tx16 if c.is_multiple_of(2) => Some(if skip_border {
                Width::Eight
            } else {
                Width::Sixteen
            }),
            TxSize::Tx32 | TxSize::Tx16 => None,
            // Force 8-wide filtering on 32-sample boundaries.
            TxSize::Tx8 => Some(Width::Eight),
            TxSize::Tx4 => Some(if c.is_multiple_of(4) {
                Width::Eight
            } else {
                Width::Four
            }),
        }
    };
    UnitEdges {
        edge,
        internal: !skip_inter && tx == TxSize::Tx4 && !skip_border,
    }
}

/// Deblock `recon` in place with frame filter `level`, adjusted per block
/// by `segment_levels` (each segment's level) when segmentation is on.
pub(crate) fn apply(
    dsp: &nc_dsp::Dsp,
    recon: &mut Picture,
    grid: &ModeInfoGrid,
    (level, sharpness): (u8, u8),
    segment_levels: Option<[u8; MAX_SEGMENTS]>,
) {
    if level == 0 {
        return;
    }
    let levels = segment_levels.unwrap_or([level; MAX_SEGMENTS]);
    let limits = levels.map(|level| Limits::with_sharpness(level, sharpness));
    for sb_row in (0..grid.rows).step_by(MI_PER_SUPERBLOCK) {
        for sb_col in (0..grid.cols).step_by(MI_PER_SUPERBLOCK) {
            for plane in PlaneIndex::ALL {
                filter_superblock(
                    dsp,
                    recon.plane_mut(plane),
                    plane,
                    grid,
                    (sb_row, sb_col),
                    (&levels, &limits),
                );
            }
        }
    }
}

fn filter_superblock(
    dsp: &nc_dsp::Dsp,
    plane: &mut Plane,
    index: PlaneIndex,
    grid: &ModeInfoGrid,
    (sb_row, sb_col): (usize, usize),
    (levels, limits): (&[u8; MAX_SEGMENTS], &[Limits; MAX_SEGMENTS]),
) {
    let ss = usize::from(index != PlaneIndex::Y);
    let step = 1 << ss;
    let rows = (0..MI_PER_SUPERBLOCK)
        .step_by(step)
        .filter(|&r| sb_row + r < grid.rows);
    let cols = || {
        (0..MI_PER_SUPERBLOCK)
            .step_by(step)
            .filter(move |&c| sb_col + c < grid.cols)
    };
    // Masks per (r, c) in mode-info units relative to the superblock.
    let mut vertical = [[UnitEdges::default(); MI_PER_SUPERBLOCK]; MI_PER_SUPERBLOCK];
    let mut horizontal = vertical;
    let mut segment = [[0u8; MI_PER_SUPERBLOCK]; MI_PER_SUPERBLOCK];
    for r in rows.clone() {
        for c in cols() {
            let (row, col) = (sb_row + r, sb_col + c);
            let info = grid.at(row, col);
            if levels[usize::from(info.segment_id)] == 0 {
                // `build_masks` / the non-420 path: a level-0 block filters
                // none of its edges.
                continue;
            }
            segment[r][c] = info.segment_id;
            let units = 1usize << (info.size_log2 - 3);
            let tx = if ss == 0 {
                info.tx_size
            } else {
                uv_tx_size(info.size_log2, info.tx_size)
            };
            let block_left = col % units == 0;
            let block_top = row % units == 0;
            let skip_col = ss == 1 && col == grid.cols - 1;
            let skip_row = ss == 1 && row == grid.rows - 1;
            vertical[r][c] = unit_edges(info, tx, c >> ss, block_left, skip_col);
            horizontal[r][c] = unit_edges(info, tx, r >> ss, block_top, skip_row);
            // The internal 4x4 edges of both directions follow the column
            // rule of the source (`mask_4x4_int`), then the row border.
            horizontal[r][c].internal = vertical[r][c].internal && !skip_row;
        }
    }
    let to_plane = |units: usize| (units * MI_SIZE) >> ss;
    // Fixed arrays, not vectors: this runs per superblock and plane.
    let (row_units, row_count) = collect_units(rows);
    let (col_units, col_count) = collect_units(cols());
    let (rows, cols) = (&row_units[..row_count], &col_units[..col_count]);
    // Exact batching: the pixels of an 8-line stripe are only touched by
    // that stripe's edges, filtered left to right (vertical) or top to
    // bottom (horizontal); so a run of identical edges along the other
    // axis is one kernel call, in the same per-line order.
    let key = |edges: &[[UnitEdges; MI_PER_SUPERBLOCK]; MI_PER_SUPERBLOCK], r: usize, c: usize| {
        (edges[r][c].edge, edges[r][c].internal, segment[r][c])
    };
    for &c in cols {
        let x = to_plane(sb_col + c);
        for run in runs(rows, |&r| key(&vertical, r, c)) {
            let (r, units) = (run[0], run.len());
            let y = to_plane(sb_row + r);
            let edges = vertical[r][c];
            let limits = limits[usize::from(segment[r][c])];
            if let Some(width) = edges.edge
                && x != 0
            {
                filter_edge(dsp, plane, (x, y), (true, units), width, limits);
            }
            if edges.internal {
                filter_edge(dsp, plane, (x + 4, y), (true, units), Width::Four, limits);
            }
        }
    }
    for &r in rows {
        let y = to_plane(sb_row + r);
        for run in runs(cols, |&c| key(&horizontal, r, c)) {
            let (c, units) = (run[0], run.len());
            let x = to_plane(sb_col + c);
            let edges = horizontal[r][c];
            let limits = limits[usize::from(segment[r][c])];
            if let Some(width) = edges.edge
                && y != 0
            {
                filter_edge(dsp, plane, (x, y), (false, units), width, limits);
            }
            if edges.internal {
                filter_edge(dsp, plane, (x, y + 4), (false, units), Width::Four, limits);
            }
        }
    }
}
/// The mode-info unit offsets of `units` (at most a superblock's) in a
/// fixed array, with their count.
fn collect_units(units: impl Iterator<Item = usize>) -> ([usize; MI_PER_SUPERBLOCK], usize) {
    let mut out = [0; MI_PER_SUPERBLOCK];
    let mut count = 0;
    for unit in units {
        out[count] = unit;
        count += 1;
    }
    (out, count)
}

/// Maximal runs of consecutive `items` with equal `key`.
fn runs<'a, T, K: PartialEq>(
    items: &'a [T],
    key: impl Fn(&T) -> K + 'a,
) -> impl Iterator<Item = &'a [T]> + 'a {
    items.chunk_by(move |a, b| key(a) == key(b))
}
