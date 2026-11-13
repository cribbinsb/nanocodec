//! H.266 §8.8 deblocking control through the shared engine's frame-filter hook.
//! Threshold tables/strength rules checked against `VVenC` (BSD-3-Clause-Clear;
//! `THIRD_PARTY.md`). Pixel arithmetic lives in `nc-dsp`.
use crate::chroma_qp::ChromaQpTable;
use crate::motion::MotionGrid;
use crate::quant_group::QpMap;
use nc_common::{Picture, PlaneIndex};

/// §8.8.3 (maxFilterLengthP/Q): a side of at least 32 luma samples uses the long (7-sample) filter,
/// except the P side at a horizontal CTU boundary (line-buffer limit).
const LARGE_BLOCK: usize = 32;
const LONG_REACH: usize = 7;
/// §8.8.3.3: the long filter reaches at most 5 samples into an affine CU.
const AFFINE_LONG_REACH: usize = 5;
/// §8.8.3.3: affine sub-block edges (not transform edges) every 8
/// samples take reach 2 next to the CU's first sub-block edge or within 8
/// samples of a transform edge or the CU's end, else 3.
const SUB_BLOCK_NEAR_REACH: usize = 2;
/// Strength of a sub-block edge whose two sides move apart (§8.8.3.5).
const MOTION_STRENGTH: u8 = 1;
/// Edge segment length: luma decisions are made per 4 lines (§8.8.3).
const SEGMENT: usize = 4;
/// Smallest CU side (the minimum quadtree node).
const MIN_CU: usize = 8;
const SHORT_REACH: usize = 3;
/// §8.8.3.3: beside a transform block at most this many samples across,
/// both sides take the one-sample weak filter (intra sub-partitions).
const NARROW_BLOCK: usize = 4;
const NARROW_REACH: usize = 1;
/// Lines of one luma decision segment and of an edge visit (two segments).
const EDGE_LINES: usize = 2 * SEGMENT;
pub(crate) const TC: [i32; 66] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 4, 4, 4, 5, 5, 5, 5, 7, 7, 8, 9,
    10, 10, 11, 13, 14, 15, 17, 19, 21, 24, 25, 29, 33, 36, 41, 45, 51, 57, 64, 71, 80, 89, 100,
    112, 125, 141, 157, 177, 198, 222, 250, 280, 314, 352, 395,
];
pub(crate) const BETA: [i32; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66,
    68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88,
];

/// Deblock `picture`. `qp` holds the luma QP and the chroma QP (§8.8.3.6.5:
/// with one slice QP both sides of every chroma edge share the mapped chroma
/// QP, which therefore indexes the chroma thresholds).
/// The QPs deblocking filters at: the picture's, or with block QP each
/// edge's mean of its two sides (§8.8.3.6): luma `(QpQ + QpP + 1) >> 1` of
/// the CUs' `QpY`, chroma the same mean of their chroma QPs (each `QpY`
/// through the SPS table; mapping the luma mean instead differs from
/// `VVdeC` and `FFmpeg` where the sides' QPs differ).
#[derive(Debug, Clone, Copy)]
pub(crate) struct DeblockQp<'a> {
    /// Luma and chroma QP of the picture.
    pub(crate) frame: [u32; 2],
    /// Every CU's `QpY` and the chroma QP table, when CUs carry their own.
    pub(crate) blocks: Option<(&'a QpMap, &'a ChromaQpTable)>,
}

impl DeblockQp<'_> {
    /// Luma and chroma QP of the edge at luma (`x`, `y`) between the block
    /// above or left of it and the one at it.
    fn edge(&self, (x, y): (usize, usize), horizontal: bool) -> [u32; 2] {
        let Some((map, chroma)) = self.blocks else {
            return self.frame;
        };
        let (px, py) = if horizontal { (x, y - 1) } else { (x - 1, y) };
        let (p, q) = (map.at(px, py), map.at(x, y));
        [
            (p + q).div_ceil(2),
            (chroma.map(p) + chroma.map(q)).div_ceil(2),
        ]
    }

    /// Luma QP of an edge inside the CU at luma (`x`, `y`).
    fn inside(&self, (x, y): (usize, usize)) -> u32 {
        self.blocks.map_or(self.frame[0], |(map, _)| map.at(x, y))
    }
}

pub(crate) fn apply(
    dsp: &nc_dsp::Dsp,
    picture: &mut Picture,
    grid: &MotionGrid,
    qp: &DeblockQp<'_>,
) {
    for horizontal in [false, true] {
        // Edges lie on TU boundaries, and TUs are at least 8 a side, so
        // only every second 4-sample column (vertical edges) or row
        // (horizontal edges) can hold one. Along each line of positions the
        // walk jumps from one transform unit to the next: a vertical edge is
        // a unit's left side, a horizontal one its top side, so the positions
        // inside a unit are never visited.
        // Lines go in pairs of segments: both halves of an 8-line run lie
        // between the same two transform units.
        for y in (0..picture.height()).step_by(MIN_CU) {
            let mut x = 0;
            while x < picture.width() {
                let Some(tu) = grid.transform_unit(x, y) else {
                    x += MIN_CU;
                    continue;
                };
                let block_end = (tu.x + tu.width()).min(picture.width());
                let affine = grid.affine_cu(x, y);
                let inside = qp.inside((x, y));
                if !horizontal {
                    edge(dsp, picture, grid, qp, (x, y), horizontal);
                    if let Some(cu) = &affine {
                        for x in (x + MIN_CU..block_end).step_by(MIN_CU) {
                            sub_block_edge(dsp, picture, grid, inside, cu, (x, y), horizontal);
                        }
                    }
                } else if y == tu.y {
                    for x in (x..block_end).step_by(MIN_CU) {
                        edge(dsp, picture, grid, qp, (x, y), horizontal);
                    }
                } else if let Some(cu) = &affine {
                    for x in (x..block_end).step_by(MIN_CU) {
                        sub_block_edge(dsp, picture, grid, inside, cu, (x, y), horizontal);
                    }
                }
                if let Some(split) = grid.narrow_sub_partitions(x, y) {
                    sub_partition_edges(picture, inside, (&tu, split), y, horizontal);
                }
                x = tu.x + tu.width();
            }
        }
    }
}

/// `tC` (§8.8.3.6.2, 8-bit) at `qp` for boundary strength `bs` (1 or 2).
fn tc(qp: u32, bs: u8) -> i32 {
    (TC[qp as usize + 2 * (usize::from(bs) - 1)] + 2) >> 2
}

/// Filter the luma of the edge between two affine sub-blocks at `(x, y)`
/// inside CU `cu` (8 lines; §8.8.3.2: sub-block edges on the 8-sample
/// grid that are not transform edges), segment by segment where the
/// vectors across it differ by half a sample or more.
fn sub_block_edge(
    dsp: &nc_dsp::Dsp,
    picture: &mut Picture,
    grid: &MotionGrid,
    qp: u32,
    cu: &nc_encoder::Block,
    (x, y): (usize, usize),
    horizontal: bool,
) {
    // Every transform unit of a CU has the same size.
    let tu = grid
        .transform_unit(cu.x, cu.y)
        .expect("a coded CU has transform units");
    let (offset, size, transform) = if horizontal {
        (y - cu.y, cu.height(), tu.height())
    } else {
        (x - cu.x, cu.width(), tu.width())
    };
    if offset.is_multiple_of(transform) {
        return; // a transform edge: `edge` filters it
    }
    let transform_edge = |at: usize| at < size && at.is_multiple_of(transform);
    let near = offset == MIN_CU
        || transform_edge(offset - MIN_CU)
        || offset + MIN_CU >= size
        || transform_edge(offset + MIN_CU);
    let reach = if near {
        SUB_BLOCK_NEAR_REACH
    } else {
        SHORT_REACH
    };
    let beta = BETA[qp as usize];
    let mut view = picture.plane_mut(PlaneIndex::Y).view_mut();
    for (segment, moved) in grid
        .segment_motion((x, y), horizontal)
        .into_iter()
        .enumerate()
    {
        if moved {
            let along = SEGMENT * segment;
            let at = if horizontal {
                (x + along, y)
            } else {
                (x, y + along)
            };
            let tc = tc(qp, MOTION_STRENGTH);
            dsp.vvc_deblock_luma_segment(&mut view, at, !horizontal, beta, tc, [reach; 2]);
        }
    }
}

/// Filter the planes across the 8 luma lines (two segments) from `(x, y)`
/// and their chroma counterparts, when a coded block edge with non-zero
/// strength lies there.
fn edge(
    dsp: &nc_dsp::Dsp,
    picture: &mut Picture,
    grid: &MotionGrid,
    qp: &DeblockQp<'_>,
    (x, y): (usize, usize),
    horizontal: bool,
) {
    let Some(edge) = grid.deblocking_edge(x, y, horizontal) else {
        return;
    };
    let qp = qp.edge((x, y), horizontal);
    let boundary = horizontal && y.is_multiple_of(grid.ctu_size());
    luma_edge(dsp, picture, qp[0], (x, y), horizontal, &edge, boundary);
    for plane in [PlaneIndex::U, PlaneIndex::V] {
        let bs = edge.chroma[plane as usize - 1];
        if bs == 0 {
            continue;
        }
        let tc = tc(qp[1], bs);
        let beta = BETA[qp[1] as usize];
        let perp = if horizontal { y } else { x };
        let large = edge.sizes[1].iter().all(|&s| s >= 16);
        if !perp.is_multiple_of(16) || (bs == 1 && !large) {
            continue;
        }
        if tc == 0 {
            continue;
        }
        let mut view = picture.plane_mut(plane).view_mut();
        dsp.vvc_deblock_chroma_edge(
            &mut view,
            (x / 2, y / 2),
            !horizontal,
            EDGE_LINES / 2,
            beta,
            tc,
            large,
            boundary,
        );
    }
}

/// The luma of transform edge `edge` at `(x, y)`: per 4-line segment at its
/// own strength, with the reaches its transform sizes and CUs allow.
fn luma_edge(
    dsp: &nc_dsp::Dsp,
    picture: &mut Picture,
    qp: u32,
    (x, y): (usize, usize),
    horizontal: bool,
    edge: &crate::motion::Edge,
    boundary: bool,
) {
    if edge.luma == [0; 2] {
        return;
    }
    let beta = BETA[qp as usize];
    let sizes = edge.sizes[0];
    if sizes.iter().any(|&s| s <= NARROW_BLOCK) {
        // Only beside intra sub-partitions: intra, one strength.
        narrow_luma_edge(picture, (x, y), horizontal, beta, tc(qp, edge.luma[0]));
        return;
    }
    let reach = |side: usize| {
        let long = sizes[side] >= LARGE_BLOCK && !(side == 0 && boundary);
        match (long, edge.affine[side]) {
            (false, _) => SHORT_REACH,
            (true, false) => LONG_REACH,
            (true, true) => AFFINE_LONG_REACH,
        }
    };
    let lengths = [reach(0), reach(1)];
    let mut view = picture.plane_mut(PlaneIndex::Y).view_mut();
    if edge.luma[0] == edge.luma[1] {
        let tc = tc(qp, edge.luma[0]);
        dsp.vvc_deblock_luma_edge(&mut view, (x, y), !horizontal, beta, tc, lengths);
        return;
    }
    for (segment, bs) in edge.luma.into_iter().enumerate() {
        if bs > 0 {
            let along = SEGMENT * segment;
            let at = if horizontal {
                (x + along, y)
            } else {
                (x, y + along)
            };
            dsp.vvc_deblock_luma_segment(&mut view, at, !horizontal, beta, tc(qp, bs), lengths);
        }
    }
}

/// The luma edges between the intra sub-partitions `split` of `cu`
/// narrower than a grid cell (§8.8.3.3: transform block edges on the
/// 4-sample grid, intra strength 2, luma only, one sample each side), on the 8 lines from row
/// `y` (vertical edges), or all of them on the CU's first row band
/// (horizontal edges).
fn sub_partition_edges(
    picture: &mut Picture,
    qp: u32,
    (cu, split): (&nc_encoder::Block, crate::isp::Split),
    y: usize,
    horizontal: bool,
) {
    const INTRA_STRENGTH: u8 = 2;
    let tc = tc(qp, INTRA_STRENGTH);
    let beta = BETA[qp as usize];
    // An 8x8 CU's 2-sample sub-partitions meet off the grid between grid
    // edges: only the grid ones are filtered.
    let on_grid = |part: &&nc_encoder::Block| {
        (part.x - cu.x).is_multiple_of(crate::isp::DEBLOCK_GRID)
            && (part.y - cu.y).is_multiple_of(crate::isp::DEBLOCK_GRID)
    };
    let parts = crate::isp::parts(cu, split);
    let inner = parts[1..].iter().filter(on_grid);
    match (split, horizontal) {
        (crate::isp::Split::Vertical, false) => {
            for part in inner {
                narrow_luma_edge(picture, (part.x, y), false, beta, tc);
            }
        }
        (crate::isp::Split::Horizontal, true) if y == cu.y => {
            for part in inner {
                for x in (cu.x..cu.x + cu.width()).step_by(EDGE_LINES) {
                    narrow_luma_edge(picture, (x, part.y), true, beta, tc);
                }
            }
        }
        _ => {}
    }
}

/// The one-sample weak luma filter across the 8 lines from `(x, y)`, two
/// segments deciding separately (`nc_dsp::vvc_deblock_luma` at reach 1).
fn narrow_luma_edge(
    picture: &mut Picture,
    (x, y): (usize, usize),
    horizontal: bool,
    beta: i32,
    tc: i32,
) {
    for segment in [0, SEGMENT] {
        let (sx, sy) = if horizontal {
            (x + segment, y)
        } else {
            (x, y + segment)
        };
        let lines = gather::<SEGMENT, 8>(picture, PlaneIndex::Y, sx, sy, horizontal);
        // p3..q3 in the middle of the kernel's p7..q7 rows; the outer
        // samples are never read at reach 1.
        let mut rows = lines.map(|line| {
            let mut row = [0; 16];
            row[4..12].copy_from_slice(&line);
            row
        });
        nc_dsp::vvc_deblock_luma(&mut rows, beta, tc, [NARROW_REACH; 2]);
        let lines = rows.map(|row| std::array::from_fn::<u8, 8, _>(|i| row[4 + i]));
        store(picture, PlaneIndex::Y, sx, sy, horizontal, &lines);
    }
}

/// The `L` lines of `W` samples across the edge at `(x, y)`, line by line.
/// Each picture row is fetched once (a horizontal edge's lines are columns).
pub(crate) fn gather<const L: usize, const W: usize>(
    picture: &Picture,
    plane: PlaneIndex,
    x: usize,
    y: usize,
    horizontal: bool,
) -> [[u8; W]; L] {
    let view = picture.plane(plane).view();
    let mut samples = [[0; W]; L];
    if horizontal {
        for i in 0..W {
            let row = &view.row(y + i - W / 2)[x..x + L];
            for (line, &value) in samples.iter_mut().zip(row) {
                line[i] = value;
            }
        }
    } else {
        for (line, samples) in samples.iter_mut().enumerate() {
            samples.copy_from_slice(&view.row(y + line)[x - W / 2..x + W / 2]);
        }
    }
    samples
}

/// Write back what [`gather`] read.
pub(crate) fn store<const L: usize, const W: usize>(
    picture: &mut Picture,
    plane: PlaneIndex,
    x: usize,
    y: usize,
    horizontal: bool,
    samples: &[[u8; W]; L],
) {
    let mut view = picture.plane_mut(plane).view_mut();
    if horizontal {
        for i in 0..W {
            let row = &mut view.row_mut(y + i - W / 2)[x..x + L];
            for (value, line) in row.iter_mut().zip(samples) {
                *value = line[i];
            }
        }
    } else {
        for (line, samples) in samples.iter().enumerate() {
            view.row_mut(y + line)[x - W / 2..x + W / 2].copy_from_slice(samples);
        }
    }
}
