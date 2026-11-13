//! The HEVC deblocking filter (HEVC §8.7.2) for one slice and no tiles,
//! with the PPS `beta`/`tc` offsets and chroma QP offset. Each edge piece
//! filters at the mean `QpY` of its two sides (§8.7.2.5.3), read from the
//! grid (one QP per picture unless block QP is on).
//!
//! All vertical edges of the picture are filtered first, then all
//! horizontal edges on the result (§8.7.2.1). Edges lie on the 8x8 luma
//! grid where a CU (= prediction unit) or a luma transform block ends;
//! luma is filtered in 4-sample segments, chroma (4:2:0) only on the 8x8
//! chroma grid and only where the boundary strength is 2.
//!
//! The boundary strengths are derived here; the decisions and filtering of
//! the samples are `nc_dsp` kernels (`Dsp::hevc_deblock_luma_edge`, 8 luma
//! lines at a time, and `Dsp::hevc_deblock_chroma_edge`).

use nc_common::{Picture, PlaneIndex};
use nc_dsp::Dsp;

use crate::grid::{Cell, Grid};
use crate::quant::chroma_qp;

/// Luma edges are on this grid (`8x8`).
const EDGE_SPACING: usize = 8;
/// Luma decisions and filtering work on 4-sample segments.
const SEGMENT: usize = 4;
/// Chroma edges are filtered on the 8x8 chroma grid: every 16 luma samples.
const CHROMA_EDGE_SPACING_LUMA: usize = 16;
/// Chroma lines per kernel call (16 luma lines: four luma segments).
const CHROMA_PIECE: usize = 8;
/// Only this boundary strength filters chroma (§8.7.2.5.5).
const CHROMA_BS: u8 = 2;
/// A motion vector difference of one whole sample (quarter units) or more
/// across an edge gives boundary strength 1 (§8.7.2.4).
const MV_DIFF_THRESHOLD: i32 = 4;

/// `β′` by Q (Table 8-12 of the deblocking clause, 8-bit: β = β′).
#[rustfmt::skip]
const BETA_TABLE: [i32; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64,
];
/// `tC′` by Q (8-bit: tC = tC′).
#[rustfmt::skip]
const TC_TABLE: [i32; 54] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3,
    3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 22, 24,
];

/// Edge direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    /// A vertical edge: samples across it lie along a row.
    Vertical,
    /// A horizontal edge: samples across it lie along a column.
    Horizontal,
}

/// What the filter strength depends on besides the boundary strengths and
/// the blocks' QPs.
#[derive(Debug, Clone, Copy)]
pub struct Strength {
    /// `cQpPicOffset` (`pps_cb_qp_offset`, equal for Cr).
    pub chroma_qp_offset: i32,
    /// `slice_beta_offset_div2`.
    pub beta_offset_div2: i32,
    /// `slice_tc_offset_div2`.
    pub tc_offset_div2: i32,
}

impl Strength {
    /// `β` at `qPL` `qp` (§8.7.2.5.3).
    fn beta(&self, qp: i32) -> u8 {
        BETA_TABLE[(qp + 2 * self.beta_offset_div2).clamp(0, 51) as usize] as u8
    }

    /// `tC` of a segment of boundary strength `bs` at QP `qp` (0 for none:
    /// the segment is not filtered).
    fn tc(&self, bs: u8, qp: i32) -> u8 {
        if bs == 0 {
            return 0;
        }
        let index = qp + 2 * (i32::from(bs) - 1) + 2 * self.tc_offset_div2;
        TC_TABLE[index.clamp(0, 53) as usize] as u8
    }
}

/// Deblock `recon` in place using the coding information in `grid`.
pub fn deblock(recon: &mut Picture, grid: &Grid, strength: &Strength) {
    let dsp = Dsp::new();
    for dir in [Direction::Vertical, Direction::Horizontal] {
        let strengths = EdgeStrengths::new(grid, dir);
        if strengths.any_edge {
            filter_luma(&dsp, recon, dir, &strengths, strength);
            filter_chroma(&dsp, recon, dir, &strengths, strength);
        }
    }
}

/// Boundary strength of every 8-sample edge piece on the 8x8 grid in one
/// direction. The grid's cells are 8x8, so both 4-sample segments of a
/// piece share their strength; storing one value per cell (rather than
/// per segment) lets the filters skip unfiltered pieces with one load.
struct EdgeStrengths {
    /// `bS` of the left (vertical) or top (horizontal) edge of each cell,
    /// indexed `[row * cols + col]`; 0 on the picture boundary.
    bs: Vec<u8>,
    /// `qPL = (QpQ + QpP + 1) >> 1` of the same edge pieces (§8.7.2.5.3).
    qp: Vec<i8>,
    cols: usize,
    rows: usize,
    /// Some piece has a nonzero strength. Mostly skipped pictures often
    /// have none, so their filtering passes are skipped outright.
    any_edge: bool,
}

impl EdgeStrengths {
    /// Most edge pieces lie inside one CU and one transform block (bS 0),
    /// so the buffers start at 0 and only pieces between two units are
    /// written, their `qPL` only where bS is nonzero (the filters read it
    /// nowhere else). Exact; HEVC Faster x86 instructions -0.2..-0.4%
    /// (callgrind, chart clips).
    fn new(grid: &Grid, dir: Direction) -> Self {
        let (cols, rows) = (grid.cols(), grid.rows());
        let mut edges = Self {
            bs: vec![0u8; cols * rows],
            qp: vec![0i8; cols * rows],
            cols,
            rows,
            any_edge: false,
        };
        let mut any = 0;
        match dir {
            Direction::Vertical => {
                for row in 0..rows {
                    let cells = grid.row(row);
                    for col in 1..cols {
                        any |= edges.set(row * cols + col, &cells[col - 1], &cells[col]);
                    }
                }
            }
            Direction::Horizontal => {
                for row in 1..rows {
                    let (above, below) = (grid.row(row - 1), grid.row(row));
                    for col in 0..cols {
                        any |= edges.set(row * cols + col, &above[col], &below[col]);
                    }
                }
            }
        }
        edges.any_edge = any != 0;
        edges
    }

    /// Set the piece at `index` between cells `p` and `q`; returns its bS.
    #[inline]
    fn set(&mut self, index: usize, p: &Cell, q: &Cell) -> u8 {
        if p.cu == q.cu && p.tu == q.tu {
            return 0;
        }
        let bs = strength(p, q);
        if bs != 0 {
            self.bs[index] = bs;
            self.qp[index] = ((i16::from(p.qp) + i16::from(q.qp) + 1) >> 1) as i8;
        }
        bs
    }

    /// `bS` of the edge piece of the cell containing luma sample `(x, y)`.
    fn at(&self, x: usize, y: usize) -> u8 {
        self.bs[(y / EDGE_SPACING) * self.cols + x / EDGE_SPACING]
    }

    /// `qPL` of the edge piece of the cell containing luma sample `(x, y)`.
    fn qp_at(&self, x: usize, y: usize) -> i32 {
        i32::from(self.qp[(y / EDGE_SPACING) * self.cols + x / EDGE_SPACING])
    }
}

/// `bS` between the blocks containing `p0` and `q0` (§8.7.2.4).
fn strength(p: &Cell, q: &Cell) -> u8 {
    let transform_edge = p.tu != q.tu;
    let prediction_edge = p.cu != q.cu;
    if !transform_edge && !prediction_edge {
        return 0;
    }
    if p.intra || q.intra {
        return 2;
    }
    if transform_edge && (p.luma_coded || q.luma_coded) {
        return 1;
    }
    // One reference picture and one vector per prediction unit: only the
    // vector difference can make the motion "different".
    let differs = (i32::from(p.mv.x) - i32::from(q.mv.x)).abs() >= MV_DIFF_THRESHOLD
        || (i32::from(p.mv.y) - i32::from(q.mv.y)).abs() >= MV_DIFF_THRESHOLD;
    u8::from(prediction_edge && differs)
}

fn filter_luma(
    dsp: &Dsp,
    recon: &mut Picture,
    dir: Direction,
    bs: &EdgeStrengths,
    strength: &Strength,
) {
    let mut view = recon.plane_mut(PlaneIndex::Y).view_mut();
    let vertical = dir == Direction::Vertical;
    // Each kernel call covers both 4-sample segments of one 8-sample piece,
    // which lie in the same pair of 8x8 cells and so share bS and qPL.
    for row in 0..bs.rows {
        let range = row * bs.cols..(row + 1) * bs.cols;
        for (col, (&b, &qp)) in bs.bs[range.clone()].iter().zip(&bs.qp[range]).enumerate() {
            let qp = i32::from(qp);
            let tc = strength.tc(b, qp);
            if tc != 0 {
                let at = (col * EDGE_SPACING, row * EDGE_SPACING);
                let beta = strength.beta(qp);
                dsp.hevc_deblock_luma_edge(&mut view, at, vertical, [tc, tc], beta);
            }
        }
    }
}

fn filter_chroma(
    dsp: &Dsp,
    recon: &mut Picture,
    dir: Direction,
    bs: &EdgeStrengths,
    strength: &Strength,
) {
    let (w, h) = (bs.cols * EDGE_SPACING, bs.rows * EDGE_SPACING);
    // QpC is the chroma QP of qPL plus cQpPicOffset (§8.7.2.5.5); chroma
    // edges are filtered only at boundary strength 2.
    let tc_at = |x: usize, y: usize| {
        strength.tc(
            CHROMA_BS,
            chroma_qp(bs.qp_at(x, y) + strength.chroma_qp_offset),
        )
    };
    let vertical = dir == Direction::Vertical;
    // Kernel calls cover up to CHROMA_PIECE chroma lines (two per luma
    // segment) of an edge on the chroma grid.
    let piece = 2 * CHROMA_PIECE;
    let (xs, ys) = if vertical {
        (
            (CHROMA_EDGE_SPACING_LUMA..w).step_by(CHROMA_EDGE_SPACING_LUMA),
            (0..h).step_by(piece),
        )
    } else {
        (
            (0..w).step_by(piece),
            (CHROMA_EDGE_SPACING_LUMA..h).step_by(CHROMA_EDGE_SPACING_LUMA),
        )
    };
    let mut calls = Vec::new();
    for y in ys {
        for x in xs.clone() {
            let along = if vertical { h - y } else { w - x };
            let segments = along.min(piece) / SEGMENT;
            let mut tcs = [0u8; 4];
            for (s, t) in tcs.iter_mut().enumerate().take(segments) {
                let (sx, sy) = if vertical {
                    (x, y + s * SEGMENT)
                } else {
                    (x + s * SEGMENT, y)
                };
                if bs.at(sx, sy) == CHROMA_BS {
                    *t = tc_at(sx, sy);
                }
            }
            if tcs != [0; 4] {
                calls.push(((x / 2, y / 2), 2 * segments, tcs));
            }
        }
    }
    for plane in [PlaneIndex::U, PlaneIndex::V] {
        let mut view = recon.plane_mut(plane).view_mut();
        for &(at, lines, tcs) in &calls {
            dsp.hevc_deblock_chroma_edge(&mut view, at, vertical, lines, tcs);
        }
    }
}
