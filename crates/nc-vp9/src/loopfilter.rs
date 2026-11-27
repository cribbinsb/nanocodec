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
#[derive(Clone, Copy)]
struct UnitEdges {
    /// Filter at the unit's left/top edge, if any.
    edge: Option<Width>,
    /// 4-wide filter at the internal edge 4 samples in (4x4 transforms).
    internal: bool,
}

/// A unit's edges in one direction and its segment, packed into one byte:
/// the edge runs compare keys, a byte rather than three fields (with the
/// plane-unit loops below, VP9 Faster x86 instructions -1.3 to -1.5 %).
#[derive(Default, Clone, Copy, PartialEq, Eq)]
struct EdgeKey(u8);

impl EdgeKey {
    /// Bits of the edge's filter width (0: no edge).
    const WIDTH_MASK: u8 = 0b11;
    /// The internal 4-wide edge.
    const INTERNAL: u8 = 1 << 2;
    /// The segment id's first bit.
    const SEGMENT_SHIFT: u32 = 3;

    fn new(edges: UnitEdges, segment: u8) -> Self {
        let width = match edges.edge {
            None => 0,
            Some(Width::Four) => 1,
            Some(Width::Eight) => 2,
            Some(Width::Sixteen) => 3,
        };
        let internal = if edges.internal { Self::INTERNAL } else { 0 };
        Self(width | internal | (segment << Self::SEGMENT_SHIFT))
    }

    fn edge(self) -> Option<Width> {
        match self.0 & Self::WIDTH_MASK {
            0 => None,
            1 => Some(Width::Four),
            2 => Some(Width::Eight),
            _ => Some(Width::Sixteen),
        }
    }

    fn internal(self) -> bool {
        self.0 & Self::INTERNAL != 0
    }

    fn segment(self) -> usize {
        usize::from(self.0 >> Self::SEGMENT_SHIFT)
    }
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
    // The superblock's plane 8x8 units inside the frame; unit (i, j) takes
    // the mode info of mode-info unit (i << ss, j << ss).
    let rows = (grid.rows - sb_row)
        .min(MI_PER_SUPERBLOCK)
        .div_ceil(1 << ss);
    let cols = (grid.cols - sb_col)
        .min(MI_PER_SUPERBLOCK)
        .div_ceil(1 << ss);
    // Packed keys, vertical edges by column (`[j][i]`) and horizontal ones
    // by row (`[i][j]`), so each run search reads one contiguous slice.
    // A level-0 block filters none of its edges (`build_masks` / the
    // non-420 path) and keeps the empty key.
    let mut vertical = [[EdgeKey::default(); MI_PER_SUPERBLOCK]; MI_PER_SUPERBLOCK];
    let mut horizontal = vertical;
    for i in 0..rows {
        let row = sb_row + (i << ss);
        let skip_row = ss == 1 && row == grid.rows - 1;
        for j in 0..cols {
            let col = sb_col + (j << ss);
            let info = grid.at(row, col);
            let segment = info.segment_id;
            if levels[usize::from(segment)] == 0 {
                continue;
            }
            let units = 1usize << (info.size_log2 - 3);
            let tx = if ss == 0 {
                info.tx_size
            } else {
                uv_tx_size(info.size_log2, info.tx_size)
            };
            let skip_col = ss == 1 && col == grid.cols - 1;
            let column_edges = unit_edges(info, tx, j, col % units == 0, skip_col);
            let mut row_edges = unit_edges(info, tx, i, row % units == 0, skip_row);
            // The internal 4x4 edges of both directions follow the column
            // rule of the source (`mask_4x4_int`), then the row border.
            row_edges.internal = column_edges.internal && !skip_row;
            vertical[j][i] = EdgeKey::new(column_edges, segment);
            horizontal[i][j] = EdgeKey::new(row_edges, segment);
        }
    }
    let (x0, y0) = ((sb_col * MI_SIZE) >> ss, (sb_row * MI_SIZE) >> ss);
    // Exact batching: the pixels of an 8-line stripe are only touched by
    // that stripe's edges, filtered left to right (vertical) or top to
    // bottom (horizontal); so a run of identical edges along the other
    // axis is one kernel call, in the same per-line order.
    for (j, column) in vertical[..cols].iter().enumerate() {
        let x = x0 + j * MI_SIZE;
        let mut i = 0;
        for run in column[..rows].chunk_by(|a, b| a == b) {
            let (key, units, y) = (run[0], run.len(), y0 + i * MI_SIZE);
            i += units;
            let limits = limits[key.segment()];
            if let Some(width) = key.edge()
                && x != 0
            {
                filter_edge(dsp, plane, (x, y), (true, units), width, limits);
            }
            if key.internal() {
                filter_edge(dsp, plane, (x + 4, y), (true, units), Width::Four, limits);
            }
        }
    }
    for (i, row) in horizontal[..rows].iter().enumerate() {
        let y = y0 + i * MI_SIZE;
        let mut j = 0;
        for run in row[..cols].chunk_by(|a, b| a == b) {
            let (key, units, x) = (run[0], run.len(), x0 + j * MI_SIZE);
            j += units;
            let limits = limits[key.segment()];
            if let Some(width) = key.edge()
                && y != 0
            {
                filter_edge(dsp, plane, (x, y), (false, units), width, limits);
            }
            if key.internal() {
                filter_edge(dsp, plane, (x, y + 4), (false, units), Width::Four, limits);
            }
        }
    }
}
