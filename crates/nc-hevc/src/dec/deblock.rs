//! The deblocking filter for whole pictures of any slice and tile
//! structure (H.265 §8.7.2): edge selection with the slice, tile and
//! disable flags (§8.7.2.3), boundary strengths (§8.7.2.4), per-edge QP
//! and offsets (§8.7.2.5.3), and the PCM / lossless exemptions
//! (`nDp = nDq = 0`, §8.7.2.5.7).
//!
//! The specification filters the vertical edges of the whole picture first,
//! then the horizontal ones (§8.7.2.1). The decoder filters bands of CTB
//! rows as they are decoded (`loop_filter`): a band's vertical edges, then
//! its horizontal edges (those whose q side lies in the band), visiting
//! only the edge pieces decoding listed (`picture::EdgePieces`). That is the
//! same: a vertical edge reads and writes only its own band's rows, and a
//! horizontal edge at the band's top reads the rows above it after their
//! vertical edges were filtered with their own band. Decisions and sample
//! filtering are nc-dsp kernels; exempt samples are restored after the
//! kernel ran.
//!
//! At high bit depth `QpY` reaches down to `−QpBdOffsetY` and `β`, `tC`
//! scale by `2^(BitDepth − 8)` (eq. 8-330, 8-332, 8-353); the tables below
//! are built per slice at the picture's depths.

#![allow(
    clippy::many_single_char_names,
    reason = "H.265 sample and block variables keep their specification names (x, y, w, h, n)"
)]

use nc_common::{BitDepths, Picture, PlaneViewMut};
use nc_dsp::{DepthKernels, Dsp, HEVC_DEBLOCK_BATCH_PIECES, HevcLumaPiece, Pixel};

use super::params::qp_bd_offset;
use super::picture::{
    BlockInfo, Blocks, EDGE_GRID, GRID_LOG2, PictureState, PuMotion, RefPic, flag,
};
use crate::quant::chroma_qp_table;

/// Luma lines of an edge piece: one kernel call (two 4-line segments).
const LUMA_LINES: usize = EDGE_GRID;
/// Chroma lines of an edge piece (4:2:0).
const CHROMA_LINES: usize = LUMA_LINES / 2;
/// Chroma edges lie on the 8x8 chroma grid: every 16 luma samples.
const CHROMA_GRID_LUMA: usize = 16;
/// Only intra edges (bS 2) filter chroma (§8.7.2.5.5).
const CHROMA_BS: u8 = 2;
/// A vector difference of one sample (quarter units) gives bS 1.
const MV_THRESHOLD: i32 = 4;
/// Samples a filter may modify on each side of an edge.
const MODIFIED: usize = 3;

/// `β′` by Q (Table 8-12).
#[rustfmt::skip]
const BETA: [u8; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64,
];
/// `tC′` by Q (Table 8-12).
#[rustfmt::skip]
const TC: [u8; 54] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3,
    3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 22, 24,
];

/// Deblock CTB row `row` of the picture, whose blocks are all decoded, as
/// are the rows above it (see the module docs). `chroma_offsets` are
/// `pps_cb_qp_offset` and `pps_cr_qp_offset`; `tiles_cross` is
/// `loop_filter_across_tiles_enabled_flag`.
pub(super) fn deblock_band<T: Pixel>(
    dsp: &Dsp,
    pic: &mut PictureState<T>,
    chroma_offsets: [i32; 2],
    tiles_cross: bool,
    row: usize,
) {
    if pic.slices.iter().all(|s| s.deblock_disabled) {
        return;
    }
    let depths = pic.depths;
    let (samples, blocks) = pic.split();
    let tables: Vec<SliceTables> = blocks
        .slices
        .iter()
        .map(|s| SliceTables::new(s.beta_offset_div2, s.tc_offset_div2, chroma_offsets, depths))
        .collect();
    let kernels = [dsp.at(depths.luma), dsp.at(depths.chroma)];
    // One slice with deblocking on and no tile restriction (every stream
    // nanocodec writes): no per-segment slice or tile checks are needed.
    let uniform = blocks.slices.len() == 1
        && !blocks.slices[0].deblock_disabled
        && (tiles_cross || !blocks.layout.tiled);
    for vertical in [true, false] {
        let pieces = if vertical {
            &blocks.edges.vertical[row]
        } else {
            &blocks.edges.horizontal[row]
        };
        let filter = Filter {
            vertical,
            tiles_cross,
            uniform,
        };
        filter_pieces(kernels, samples, &blocks, &tables, filter, pieces);
    }
}

/// `QpY` range of 8-bit video.
const EIGHT_BIT_QPS: usize = 52;
/// Range of `QpL` (the average of two luma QPs) at the deepest supported
/// depth: `−QpBdOffsetY..=51`.
const QP_COUNT: usize = EIGHT_BIT_QPS + super::params::MAX_QP_BD_OFFSET as usize;

/// One slice's filter parameters by `QpL + QpBdOffsetY` (§8.7.2.5.3,
/// §8.7.2.5.5), scaled to the depths: `β`, luma `tC` for bS 1 and 2, and
/// chroma `tC` (bS 2) for Cb and Cr.
struct SliceTables {
    /// `QpBdOffsetY`: the table index of `QpL` 0.
    qp_offset: i32,
    beta: [i16; QP_COUNT],
    tc: [[i16; QP_COUNT]; 2],
    tc_chroma: [[i16; QP_COUNT]; 2],
}

impl SliceTables {
    fn new(
        beta_offset_div2: i32,
        tc_offset_div2: i32,
        chroma_offsets: [i32; 2],
        depths: BitDepths,
    ) -> Self {
        let qp_offset = qp_bd_offset(depths.luma);
        // Table entries at `Clip3(0, max, q)`, scaled by `2^(BitDepth − 8)`.
        let at = |table: &[u8], q: i32, scale: u32| {
            i16::from(table[q.clamp(0, table.len() as i32 - 1) as usize]) << scale
        };
        let (luma, chroma) = (depths.luma.above_eight(), depths.chroma.above_eight());
        let qp = |index: usize| index as i32 - qp_offset;
        Self {
            qp_offset,
            beta: std::array::from_fn(|i| at(&BETA, qp(i) + 2 * beta_offset_div2, luma)),
            tc: std::array::from_fn(|b| {
                std::array::from_fn(|i| at(&TC, qp(i) + 2 * b as i32 + 2 * tc_offset_div2, luma))
            }),
            tc_chroma: std::array::from_fn(|c| {
                std::array::from_fn(|i| {
                    let qpc = chroma_qp_table(qp(i) + chroma_offsets[c]);
                    at(&TC, qpc + 2 + 2 * tc_offset_div2, chroma)
                })
            }),
        }
    }

    /// The table index of `QpL`.
    fn index(&self, qp: i32) -> usize {
        (qp + self.qp_offset) as usize
    }
}

/// What the strength derivation checks besides the blocks.
#[derive(Debug, Clone, Copy)]
struct Filter {
    vertical: bool,
    tiles_cross: bool,
    /// Every segment lies in one deblocked slice without tile limits.
    uniform: bool,
}

/// bS of the segment between blocks `p` and `q` (indices `pi`, `qi`): 0
/// unless an edge runs there that the slice and tile flags let filter.
#[allow(
    clippy::inline_always,
    reason = "per-segment work of the strength pass"
)]
#[inline(always)]
fn segment(
    b: &Blocks<'_>,
    filter: Filter,
    (pi, p): (usize, BlockInfo),
    (qi, q): (usize, BlockInfo),
) -> u8 {
    let (tu_edge, pu_edge) = if filter.vertical {
        (flag::TU_EDGE_LEFT, flag::PU_EDGE_LEFT)
    } else {
        (flag::TU_EDGE_TOP, flag::PU_EDGE_TOP)
    };
    if q.flags & (tu_edge | pu_edge) == 0 {
        return 0;
    }
    if !filter.uniform {
        let slice = &b.slices[usize::from(q.slice - 1)];
        if slice.deblock_disabled || (p.slice != q.slice && !slice.loop_filter_across_slices) {
            return 0;
        }
        if !filter.tiles_cross && b.layout.tiled && tile_boundary(b, pi, qi) {
            return 0;
        }
    }
    strength(b, (p, pi), (q, qi), q.flags & tu_edge != 0)
}

/// Whether the blocks `pi` and `qi` lie in different tiles.
fn tile_boundary(b: &Blocks<'_>, pi: usize, qi: usize) -> bool {
    let at = |i: usize| ((i % b.stride) << GRID_LOG2, (i / b.stride) << GRID_LOG2);
    let ((px, py), (qx, qy)) = (at(pi), at(qi));
    let layout = b.layout;
    layout.tile_of[layout.ctb_of(px, py)] != layout.tile_of[layout.ctb_of(qx, qy)]
}

/// A vector difference of a sample or more in either component.
fn far(a: nc_encoder::Mv, b: nc_encoder::Mv) -> bool {
    (i32::from(a.x) - i32::from(b.x)).abs() >= MV_THRESHOLD
        || (i32::from(a.y) - i32::from(b.y)).abs() >= MV_THRESHOLD
}

/// §8.7.2.4 for one segment.
#[allow(clippy::inline_always, reason = "per-edge work of the strength pass")]
#[inline(always)]
fn strength(
    b: &Blocks<'_>,
    (p, pi): (BlockInfo, usize),
    (q, qi): (BlockInfo, usize),
    tu_edge: bool,
) -> u8 {
    if (p.flags | q.flags) & flag::INTRA != 0 {
        return 2;
    }
    if tu_edge && (p.flags | q.flags) & flag::CODED != 0 {
        return 1;
    }
    if b.motion[pi] == b.motion[qi] {
        // One prediction block (both sides decoded inter in this picture,
        // so their table indices are current): the same motion, bS 0.
        // Most transform edges inside prediction blocks end here.
        return 0;
    }
    let (mp, mq) = (b.motion_at(pi), b.motion_at(qi));
    if p.slice == q.slice && mp.ref_idx == mq.ref_idx {
        // Same lists and indices: the same pictures in the same roles.
        if mp == mq {
            return 0;
        }
        if !(mp.uses(0) && mp.uses(1)) {
            // One vector each, to the same picture.
            let l = usize::from(!mp.uses(0));
            return u8::from(far(mp.mv[l], mq.mv[l]));
        }
    }
    let refs = |blk: BlockInfo, m: &PuMotion| -> [Option<u64>; 2] {
        let lists = &b.slices[usize::from(blk.slice - 1)].lists;
        [0, 1].map(|l| {
            m.uses(l)
                .then(|| lists[l].get(m.ref_idx[l] as usize).map(|r: &RefPic| r.id))
                .flatten()
        })
    };
    u8::from(motion_differs(refs(p, &mp), &mp, refs(q, &mq), &mq))
}

/// The prediction part of §8.7.2.4: different pictures or vector counts,
/// or a vector difference of a sample or more.
fn motion_differs(
    rp: [Option<u64>; 2],
    mp: &PuMotion,
    rq: [Option<u64>; 2],
    mq: &PuMotion,
) -> bool {
    let count = |r: &[Option<u64>; 2]| r.iter().flatten().count();
    if count(&rp) != count(&rq) {
        return true;
    }
    if count(&rp) == 1 {
        let (lp, lq) = (usize::from(rp[0].is_none()), usize::from(rq[0].is_none()));
        return rp[lp] != rq[lq] || far(mp.mv[lp], mq.mv[lq]);
    }
    let (p0, p1, q0, q1) = (rp[0], rp[1], rq[0], rq[1]);
    if !((p0 == q0 && p1 == q1) || (p0 == q1 && p1 == q0)) {
        return true;
    }
    if p0 == p1 {
        // Both vectors refer to one picture: either pairing may match.
        (far(mp.mv[0], mq.mv[0]) || far(mp.mv[1], mq.mv[1]))
            && (far(mp.mv[0], mq.mv[1]) || far(mp.mv[1], mq.mv[0]))
    } else if p0 == q0 {
        far(mp.mv[0], mq.mv[0]) || far(mp.mv[1], mq.mv[1])
    } else {
        far(mp.mv[0], mq.mv[1]) || far(mp.mv[1], mq.mv[0])
    }
}

/// Samples of the exempt side(s) of an edge, to restore after filtering.
struct Saved<T: Pixel> {
    samples: Vec<(usize, usize, T)>,
}

impl<T: Pixel> Saved<T> {
    fn take(
        view: &PlaneViewMut<'_, T>,
        (x, y): (usize, usize),
        vertical: bool,
        lines: usize,
        sides: [bool; 2],
    ) -> Self {
        let mut samples = Vec::new();
        for line in 0..lines {
            for k in 0..MODIFIED {
                for (side, &exempt) in sides.iter().enumerate() {
                    if !exempt {
                        continue;
                    }
                    // side 0: p (before the edge), 1: q.
                    let offset = if side == 0 {
                        -(k as isize) - 1
                    } else {
                        k as isize
                    };
                    let (sx, sy) = if vertical {
                        ((x as isize + offset) as usize, y + line)
                    } else {
                        (x + line, (y as isize + offset) as usize)
                    };
                    samples.push((sx, sy, view.as_view().row(sy)[sx]));
                }
            }
        }
        Self { samples }
    }

    fn restore(self, view: &mut PlaneViewMut<'_, T>) {
        for (x, y, v) in self.samples {
            view.row_mut(y)[x] = v;
        }
    }
}

fn exempt(b: BlockInfo) -> bool {
    b.flags & flag::NO_FILTER != 0
}

/// The edge pieces `pieces` of one direction (see `picture::EdgePieces`):
/// each piece's two segment strengths (§8.7.2.4), then its eight luma
/// lines and, on the chroma grid with an intra side, its four chroma lines
/// of each chroma plane. Pieces are 8 samples apart and the filters change
/// at most 3 samples on either side, so their order does not matter.
fn filter_pieces<T: Pixel>(
    [luma_kernels, chroma_kernels]: [DepthKernels<'_>; 2],
    samples: &mut Picture<T>,
    b: &Blocks<'_>,
    tables: &[SliceTables],
    filter: Filter,
    pieces: &[u32],
) {
    let vertical = filter.vertical;
    // From a piece's first q-side block: to its second, and to its p side.
    let (along, across) = if vertical {
        (b.stride, 1)
    } else {
        (1, b.stride)
    };
    let [luma, cb, cr] = samples.planes_mut();
    let mut luma = luma.view_mut();
    let mut chroma = [cb.view_mut(), cr.view_mut()];
    let mut luma_batch = LumaBatch::new();
    let mut pending: Option<ChromaRun> = None;
    for &piece in pieces {
        let q0 = piece as usize;
        let q1 = q0 + along;
        let (p, q) = (b.info[q0 - across], b.info[q0]);
        let s = [
            segment(b, filter, (q0 - across, p), (q0, q)),
            segment(
                b,
                filter,
                (q1 - across, b.info[q1 - across]),
                (q1, b.info[q1]),
            ),
        ];
        if s == [0, 0] {
            continue;
        }
        // Both segments' blocks lie in the same two coding units (pieces
        // are 8-aligned, coding blocks at least 8x8): one QpL and one set
        // of exemptions serve the piece.
        let (x, y) = ((q0 % b.stride) << GRID_LOG2, (q0 / b.stride) << GRID_LOG2);
        let t = &tables[usize::from(q.slice - 1)];
        let qp = t.index((i32::from(p.qp) + i32::from(q.qp) + 1) >> 1);
        let exempt_sides = [exempt(p), exempt(q)];
        let tc = s.map(|v| {
            if v == 0 {
                0
            } else {
                t.tc[usize::from(v) - 1][qp]
            }
        });
        let beta = t.beta[qp];
        if tc != [0, 0] && beta != 0 {
            if exempt_sides.contains(&true) {
                luma_batch.flush(luma_kernels, &mut luma, vertical);
                filter_luma(
                    luma_kernels,
                    &mut luma,
                    (x, y),
                    vertical,
                    (tc, beta),
                    exempt_sides,
                );
            } else {
                luma_batch.push(
                    luma_kernels,
                    &mut luma,
                    vertical,
                    HevcLumaPiece {
                        at: (x, y),
                        tc,
                        beta,
                    },
                );
            }
        }
        let across_position = if vertical { x } else { y };
        if across_position.is_multiple_of(CHROMA_GRID_LUMA) && s.contains(&CHROMA_BS) {
            // Only intra (bS 2) segments filter chroma (§8.7.2.5.5).
            let tc = t
                .tc_chroma
                .map(|table| s.map(|v| if v == CHROMA_BS { table[qp] } else { 0 }));
            let at = (x / 2, y / 2);
            if exempt_sides.contains(&true) {
                for (view, tc) in chroma.iter_mut().zip(tc) {
                    filter_chroma(chroma_kernels, view, at, vertical, tc, exempt_sides);
                }
            } else {
                // Pieces of one edge are mostly listed one after another:
                // two adjacent ones share an eight-line kernel call.
                let joined = pending
                    .as_mut()
                    .is_some_and(|run| run.join(at, vertical, tc));
                if !joined && let Some(run) = pending.replace(ChromaRun::new(at, tc)) {
                    run.filter(chroma_kernels, &mut chroma, vertical);
                }
            }
        }
    }
    luma_batch.flush(luma_kernels, &mut luma, vertical);
    if let Some(run) = pending {
        run.filter(chroma_kernels, &mut chroma, vertical);
    }
}

/// Disjoint non-exempt 8-line luma pieces queued for batched filtering.
struct LumaBatch {
    pieces: [HevcLumaPiece; HEVC_DEBLOCK_BATCH_PIECES],
    len: usize,
}

impl LumaBatch {
    const fn new() -> Self {
        Self {
            pieces: [HevcLumaPiece {
                at: (0, 0),
                tc: [0, 0],
                beta: 0,
            }; HEVC_DEBLOCK_BATCH_PIECES],
            len: 0,
        }
    }

    #[inline]
    fn push<T: Pixel>(
        &mut self,
        kernels: DepthKernels<'_>,
        view: &mut PlaneViewMut<'_, T>,
        vertical: bool,
        piece: HevcLumaPiece,
    ) {
        self.pieces[self.len] = piece;
        self.len += 1;
        if self.len == HEVC_DEBLOCK_BATCH_PIECES {
            kernels.hevc_deblock_luma_pieces(view, vertical, &self.pieces);
            self.len = 0;
        }
    }

    #[inline]
    fn flush<T: Pixel>(
        &mut self,
        kernels: DepthKernels<'_>,
        view: &mut PlaneViewMut<'_, T>,
        vertical: bool,
    ) {
        if self.len > 0 {
            kernels.hevc_deblock_luma_pieces(view, vertical, &self.pieces[..self.len]);
            self.len = 0;
        }
    }
}

/// Chroma lines of one or two adjacent pieces of an edge without exempt
/// sides, filtered together.
struct ChromaRun {
    /// First q sample (chroma).
    at: (usize, usize),
    /// Per plane, `tC` of each two-line segment.
    tc: [[i16; 4]; 2],
    lines: usize,
}

impl ChromaRun {
    fn new(at: (usize, usize), tc: [[i16; 2]; 2]) -> Self {
        Self {
            at,
            tc: tc.map(|t| [t[0], t[1], 0, 0]),
            lines: CHROMA_LINES,
        }
    }

    /// Append the piece at `at` if it continues this one-piece run.
    fn join(&mut self, at: (usize, usize), vertical: bool, tc: [[i16; 2]; 2]) -> bool {
        let next = if vertical {
            (self.at.0, self.at.1 + CHROMA_LINES)
        } else {
            (self.at.0 + CHROMA_LINES, self.at.1)
        };
        if self.lines != CHROMA_LINES || at != next {
            return false;
        }
        for (run, piece) in self.tc.iter_mut().zip(tc) {
            run[2..].copy_from_slice(&piece);
        }
        self.lines += CHROMA_LINES;
        true
    }

    fn filter<T: Pixel>(
        self,
        kernels: DepthKernels<'_>,
        views: &mut [PlaneViewMut<'_, T>; 2],
        vertical: bool,
    ) {
        for (view, tc) in views.iter_mut().zip(self.tc) {
            // tC 0 (low QP) leaves the samples unchanged.
            if tc != [0; 4] {
                kernels.hevc_deblock_chroma_edge(view, self.at, vertical, self.lines, tc);
            }
        }
    }
}

/// Eight luma lines across the edge whose first q sample is `(x, y)`,
/// restoring the samples of exempt sides.
fn filter_luma<T: Pixel>(
    kernels: DepthKernels<'_>,
    view: &mut PlaneViewMut<'_, T>,
    (x, y): (usize, usize),
    vertical: bool,
    (tc, beta): ([i16; 2], i16),
    exempt_sides: [bool; 2],
) {
    let saved = (exempt_sides[0] || exempt_sides[1])
        .then(|| Saved::take(view, (x, y), vertical, LUMA_LINES, exempt_sides));
    kernels.hevc_deblock_luma_edge(view, (x, y), vertical, tc, beta);
    if let Some(saved) = saved {
        saved.restore(view);
    }
}

/// Four chroma lines (two segments, `tc` each) across the edge whose first
/// q sample is chroma `(x, y)`, restoring the samples of exempt sides.
fn filter_chroma<T: Pixel>(
    kernels: DepthKernels<'_>,
    view: &mut PlaneViewMut<'_, T>,
    (x, y): (usize, usize),
    vertical: bool,
    tc: [i16; 2],
    exempt_sides: [bool; 2],
) {
    if tc == [0, 0] {
        // tC 0 (low QP) leaves the samples unchanged.
        return;
    }
    let saved = (exempt_sides[0] || exempt_sides[1])
        .then(|| Saved::take(view, (x, y), vertical, CHROMA_LINES, exempt_sides));
    kernels.hevc_deblock_chroma_edge(view, (x, y), vertical, CHROMA_LINES, [tc[0], tc[1], 0, 0]);
    if let Some(saved) = saved {
        saved.restore(view);
    }
}
