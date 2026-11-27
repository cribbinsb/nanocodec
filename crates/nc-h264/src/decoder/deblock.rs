//! The deblocking filter of a decoded picture (§8.7) with per-slice
//! control: `disable_deblocking_filter_idc` 1 (off) and 2 (no filtering
//! across slice edges), per-slice filter offsets and chroma QP offsets.
//!
//! Boundary strengths (§8.7.2.1) come from the macroblock states; the
//! sample filters of each edge (§8.7.2.3, §8.7.2.4) are the nc-dsp kernels.
//! Chroma: 4:2:0 edges on luma edges 0 and 2; 4:2:2 the same vertically and
//! all four horizontal edges (filtered as two 8x8 halves by the 4:2:0
//! kernel); 4:4:4 Cb and Cr with the luma filter on the luma edges (§8.7:
//! `chromaStyleFilteringFlag` 0 when `ChromaArrayType` is 3).
//! A field picture is filtered as a picture of its own rows; its
//! macroblocks are field macroblocks, so horizontal intra edges take bS 3,
//! not 4, and vertical vector differences count in field rows (a
//! difference of 2 quarter field samples is 4 quarter frame samples).
//!
//! The filter runs a macroblock row behind decoding ([`RowDeblocker`]),
//! while the row is still in cache: filtering row `r` changes samples of
//! rows `r - 1` and `r` only, and the intra prediction of row `r + 1`
//! (which must see unfiltered samples) has read row `r` by the time row
//! `r + 1` is complete. Rows are filtered in order, so the result is the
//! standard's whole-picture raster order.

mod mbaff;

use core::ops::Range;

use nc_common::{BitDepths, PlaneViewMut};
use nc_dsp::{Dsp, H264EdgeParams, H264MbEdges, Pixel};
use nc_encoder::Mv;

use super::Planes;
use super::chroma::ChromaArray;
use super::mbstate::{MbKind, MbState};
use super::transform::chroma_qp_at;
use crate::deblock::{DEBLOCK_IDC_NO_SLICE_EDGES, DeblockControl};
use crate::tables::{DEBLOCK_ALPHA, DEBLOCK_BETA, DEBLOCK_TC0};
pub(super) use mbaff::filter_mbaff;

/// Vector difference (quarter samples) from which `bS` is 1 (§8.7.2.1).
const MV_THRESHOLD: i32 = 4;
/// The vertical one between field macroblocks (quarter field samples).
const FIELD_MV_THRESHOLD_Y: i32 = 2;
/// Largest `indexA` / `indexB` (eqs. 8-461, 8-462).
const MAX_INDEX: i32 = 51;
/// Luma and chroma QPs run from `−QpBdOffset` (−36 at 14 bits) to 51; the
/// threshold table is indexed by their average plus [`QP_BIAS`].
const QP_BIAS: i32 = 36;
const QP_COUNT: usize = 52 + QP_BIAS as usize;
/// `bS` of macroblock edges and internal edges next to intra samples.
const BS_INTRA_MB_EDGE: u8 = 4;
const BS_INTRA: u8 = 3;
const BS_COEFFICIENTS: u8 = 2;

/// The deblocking parameters of one slice.
#[derive(Debug, Clone)]
pub(super) struct SliceFilter {
    pub control: DeblockControl,
    pub chroma_qp_offset: [i8; 2],
    /// A B slice: its macroblocks may predict from list 1.
    pub b_slice: bool,
    /// An SP or SI slice: its macroblocks filter as intra ones (§8.7.2.1).
    pub switching: bool,
    /// `QpBdOffsetC`: chroma QPs start at its negative.
    qp_bd_offset_c: i32,
    /// The thresholds at each average QP (biased by [`QP_BIAS`]), under
    /// `control`'s offsets.
    thresholds: [Thresholds; QP_COUNT],
}

impl SliceFilter {
    pub fn new(
        control: DeblockControl,
        chroma_qp_offset: [i8; 2],
        b_slice: bool,
        switching: bool,
        depths: BitDepths,
    ) -> Self {
        Self {
            control,
            chroma_qp_offset,
            b_slice,
            switching,
            qp_bd_offset_c: 6 * depths.chroma.above_eight() as i32,
            thresholds: core::array::from_fn(|index| {
                Thresholds::at_average(index as i32 - QP_BIAS, control)
            }),
        }
    }

    /// The thresholds of an edge between QPs `qp_p` and `qp_q` (eq.
    /// 8-460: their rounded average), by table.
    fn thresholds(&self, qp_p: i32, qp_q: i32) -> Thresholds {
        self.thresholds[(((qp_p + qp_q + 1) >> 1) + QP_BIAS) as usize]
    }

    /// `QPc` of component `comp` of a macroblock with `QPY` `qp`.
    fn chroma_qp(&self, qp: i32, comp: usize) -> i32 {
        chroma_qp_at(qp, self.chroma_qp_offset[comp], self.qp_bd_offset_c)
    }
}

/// Boundary strengths of one macroblock: per direction (vertical edges,
/// horizontal edges), per luma edge, per 4-sample segment.
type Strengths = [[[u8; 4]; 4]; 2];

/// One edge's thresholds (§8.7.2.2): `alpha`, `beta` and `tC0` by bS.
#[derive(Debug, Clone, Copy)]
struct Thresholds {
    alpha: u8,
    beta: u8,
    /// `tC0` indexed by bS (entries 0 and 4..8 unused, 0): eight entries
    /// so a bS masked to 3 bits indexes it without a bounds check.
    tc0_by_bs: [u8; 8],
}

impl Thresholds {
    /// The thresholds at the average QP `average` (eqs. 8-461, 8-462), at
    /// 8 bits (the kernels scale them to the depth).
    fn at_average(average: i32, control: DeblockControl) -> Self {
        let index =
            |offset_div2: i8| (average + 2 * i32::from(offset_div2)).clamp(0, MAX_INDEX) as usize;
        let index_a = index(control.alpha_offset_div2);
        let [tc0_1, tc0_2, tc0_3] = DEBLOCK_TC0[index_a];
        Self {
            alpha: DEBLOCK_ALPHA[index_a],
            beta: DEBLOCK_BETA[index(control.beta_offset_div2)],
            tc0_by_bs: [0, tc0_1, tc0_2, tc0_3, 0, 0, 0, 0],
        }
    }

    /// `tC0` of each segment's bS (unused for 0 and 4): a lookup by bS,
    /// not a branch per segment.
    fn tc0s(self, bs: [u8; 4]) -> [u8; 4] {
        bs.map(|b| self.tc0_by_bs[usize::from(b & 7)])
    }
}

/// Deblocking of a picture's macroblock rows as they complete.
#[derive(Debug, Default)]
pub(super) struct RowDeblocker {
    /// Rows filtered so far.
    done: usize,
    /// Macroblocks decoded so far, while they arrive in raster order.
    in_order: usize,
    /// A macroblock arrived out of raster order (not in conforming
    /// Constrained High streams): rows are filtered at the end only.
    scattered: bool,
}

impl RowDeblocker {
    /// Note that macroblock `addr` was decoded; returns the last row that
    /// can be filtered now (every row up to the one before `addr`'s is
    /// complete), if `addr` completes a row.
    pub fn decoded(&mut self, addr: usize, width_mbs: usize) -> Option<usize> {
        if self.scattered || addr != self.in_order {
            self.scattered = true;
            return None;
        }
        self.in_order += 1;
        let row = addr / width_mbs;
        (addr % width_mbs == width_mbs - 1 && row > 0).then(|| row - 1)
    }

    /// Filter every row not yet filtered up to `last` (inclusive).
    pub fn filter_through<T: Pixel>(
        &mut self,
        last: usize,
        planes: &mut Planes<'_, T>,
        ctx: &DeblockContext<'_>,
    ) {
        if last >= self.done {
            deblock_rows(ctx, planes, self.done..last + 1);
            self.done = last + 1;
        }
    }
}

/// What the filter of one picture reads.
pub(super) struct DeblockContext<'a> {
    pub dsp: &'a Dsp,
    pub mbs: &'a [MbState],
    pub width_mbs: usize,
    /// `MbState::slice` of `slices[0]`.
    pub first_slice: u32,
    pub slices: &'a [SliceFilter],
    /// `ChromaArrayType`.
    pub chroma: ChromaArray,
    pub depths: BitDepths,
    /// A field picture.
    pub field: bool,
}

/// Filter the macroblock rows `rows` of `picture`: one instance for
/// pictures with B slices so far, one (list 0 only) for the others, each
/// for frames and fields.
fn deblock_rows<T: Pixel>(
    ctx: &DeblockContext<'_>,
    planes: &mut Planes<'_, T>,
    rows: Range<usize>,
) {
    let b_slice = ctx.slices.iter().any(|s| s.b_slice);
    let switching = ctx.slices.iter().any(|s| s.switching);
    match (b_slice, ctx.field, switching) {
        (true, false, false) => deblock_rows_of::<true, false, false, T>(ctx, planes, rows),
        (true, false, true) => deblock_rows_of::<true, false, true, T>(ctx, planes, rows),
        (true, true, false) => deblock_rows_of::<true, true, false, T>(ctx, planes, rows),
        (true, true, true) => deblock_rows_of::<true, true, true, T>(ctx, planes, rows),
        (false, false, false) => deblock_rows_of::<false, false, false, T>(ctx, planes, rows),
        (false, false, true) => deblock_rows_of::<false, false, true, T>(ctx, planes, rows),
        (false, true, false) => deblock_rows_of::<false, true, false, T>(ctx, planes, rows),
        (false, true, true) => deblock_rows_of::<false, true, true, T>(ctx, planes, rows),
    }
}

/// [`deblock_rows`] of a picture with (`BI`) or without B slices, a field
/// (`FIELD`) or a frame, and with (`SW`) or without SP / SI slices.
#[inline(never)]
fn deblock_rows_of<const BI: bool, const FIELD: bool, const SW: bool, T: Pixel>(
    ctx: &DeblockContext<'_>,
    planes: &mut Planes<'_, T>,
    rows: Range<usize>,
) {
    let DeblockContext {
        dsp,
        mbs,
        width_mbs,
        first_slice,
        slices,
        chroma,
        depths,
        field: _,
    } = *ctx;
    // Folds to the 8-bit constants in the 8-bit instance.
    let depths = depths.of::<T>();
    if slices.iter().all(|s| !s.control.filters()) {
        return;
    }
    let [luma, cb, cr] = planes;
    // 4:2:2 chroma has horizontal edges on every luma edge, which need
    // their strengths even where the 8x8 transform leaves luma unfiltered.
    let all_horizontal = chroma == ChromaArray::Yuv422;
    for addr in rows.start * width_mbs..rows.end * width_mbs {
        let at = (first_slice, slices);
        let Some(bs) = strengths::<BI, FIELD, SW, T>(mbs, width_mbs, at, addr, all_horizontal)
        else {
            continue;
        };
        let mb = &mbs[addr];
        let filter = &slices[(mb.slice - first_slice) as usize];
        let (mx, my) = (addr % width_mbs, addr / width_mbs);
        // QP_p of edge 0 of each direction: the left / upper neighbour
        // (only read where that edge has a strength, so it exists).
        let neighbour_qp = [addr.wrapping_sub(1), addr.wrapping_sub(width_mbs)]
            .map(|n| mbs.get(n).map_or(0, filter_qp));
        let qp = filter_qp(mb);
        // With the 8x8 transform luma edges 1 and 3 are not filtered (their
        // strengths are 0 unless 4:2:2 chroma needs them).
        let luma_mask = if mb.transform_8x8 {
            TRANSFORM_8X8_EDGES
        } else {
            LUMA_EDGES
        };
        let luma_edges = mb_edges::<T>(&bs, luma_mask, |edge_0, dir| {
            let qp_p = if edge_0 { neighbour_qp[dir] } else { qp };
            filter.thresholds(qp_p, qp)
        });
        dsp.at(depths.luma)
            .h264_deblock_mb_luma(luma, (mx * 16, my * 16), &luma_edges);
        match chroma {
            ChromaArray::Yuv420 => {}
            ChromaArray::Mono => continue,
            ChromaArray::Yuv422 | ChromaArray::Yuv444 => {
                let at = (mx, my);
                other_chroma::<T>(ctx, [&mut *cb, &mut *cr], (at, mb), &bs, neighbour_qp);
                continue;
            }
        }
        // 4:2:0 chroma edges lie on luma edges 0 and 2 and take their bS
        // (§8.7.2.1); QPs map through each component's offset.
        let chroma = |comp: usize| {
            // Mapped once per macroblock, not per edge.
            let q = filter.chroma_qp(qp, comp);
            let p = neighbour_qp.map(|n| filter.chroma_qp(n, comp));
            mb_edges::<T>(&bs, CHROMA_EDGES, |edge_0, dir| {
                filter.thresholds(if edge_0 { p[dir] } else { q }, q)
            })
        };
        let cb_edges = chroma(0);
        let position = (mx * 8, my * 8);
        let kernels = dsp.at(depths.chroma);
        if filter.chroma_qp_offset[0] == filter.chroma_qp_offset[1] {
            // Both planes share bS and thresholds: one call filters both.
            kernels.h264_deblock_mb_chroma(cb, cr, position, &cb_edges);
        } else {
            kernels.h264_deblock_mb_chroma_plane(cb, position, &cb_edges);
            kernels.h264_deblock_mb_chroma_plane(cr, position, &chroma(1));
        }
    }
}

/// Luma edges of a macroblock: all four of each direction.
const LUMA_EDGES: u8 = 0b1111;
/// The luma edges of an 8x8 transform: 0 and 2.
const TRANSFORM_8X8_EDGES: u8 = 0b0101;
/// 4:2:0 chroma edges: luma edges 0 and 2 of each direction.
const CHROMA_EDGES: u8 = 0b0101;
/// Chroma rows of a 4:2:2 macroblock filtered per call of the 8x8 kernel.
const HALF_422: usize = 8;

/// The chroma of a 4:2:2 or 4:4:4 macroblock at (`mx`, `my`) with
/// strengths `bs`, out of line (the 4:2:0 path stays inlined).
#[inline(never)]
fn other_chroma<T: Pixel>(
    ctx: &DeblockContext<'_>,
    [cb, cr]: [&mut PlaneViewMut<'_, T>; 2],
    ((mx, my), mb): ((usize, usize), &MbState),
    bs: &Strengths,
    neighbour_qp: [i32; 2],
) {
    let filter = &ctx.slices[(mb.slice - ctx.first_slice) as usize];
    let qp = filter_qp(mb);
    let kernels = ctx.dsp.at(ctx.depths.of::<T>().chroma);
    for (comp, plane) in [cb, cr].into_iter().enumerate() {
        let q = filter.chroma_qp(qp, comp);
        let p = neighbour_qp.map(|n| filter.chroma_qp(n, comp));
        if ctx.chroma == ChromaArray::Yuv444 {
            // The luma filter at the chroma QP on the luma edges (§8.7.2.3
            // with chromaStyleFilteringFlag 0).
            let mask = if mb.transform_8x8 {
                TRANSFORM_8X8_EDGES
            } else {
                LUMA_EDGES
            };
            let edges = mb_edges::<T>(bs, mask, |edge_0, dir| {
                filter.thresholds(if edge_0 { p[dir] } else { q }, q)
            });
            kernels.h264_deblock_mb_luma(plane, (mx * 16, my * 16), &edges);
            continue;
        }
        // 4:2:2: an 8x16 block. Each 8-row half is a 4:2:0 macroblock to the
        // kernel: vertical edges 0 and 2 with each luma segment (4 rows)
        // as two of the kernel's 2-row segments, horizontal edges at rows 0
        // and 4 of the half on luma edges 0 and 1 (upper half) or 2 and 3.
        for half in 0..2 {
            let rows = |seg: [u8; 4]| {
                let (a, b) = (seg[2 * half], seg[2 * half + 1]);
                [a, a, b, b]
            };
            let mut half_bs = [[[0u8; 4]; 4]; 2];
            for edge in [0, 2] {
                half_bs[0][edge] = rows(bs[0][edge]);
            }
            half_bs[1][0] = bs[1][2 * half];
            half_bs[1][2] = bs[1][2 * half + 1];
            let edges = mb_edges::<T>(&half_bs, CHROMA_EDGES, |edge_0, dir| {
                // Only the macroblock's own top edge has the neighbour's QP.
                let outer = edge_0 && (dir == 0 || half == 0);
                filter.thresholds(if outer { p[dir] } else { q }, q)
            });
            let at = (mx * 8, my * 16 + half * HALF_422);
            kernels.h264_deblock_mb_chroma_plane(plane, at, &edges);
        }
    }
}

/// The edges `edges` (a mask of edge indices, per direction) of a
/// macroblock with strengths `bs` as nc-dsp's macroblock form, with the
/// thresholds `thresholds(is_edge_0, direction)`. Edges whose bS is all
/// 0 are left out of the mask (filtered as nothing).
#[allow(
    clippy::extra_unused_type_parameters,
    reason = "one copy per sample type keeps the 8-bit instance inlined (high-bit-depth handoff)"
)]
fn mb_edges<T>(
    bs: &Strengths,
    edges: u8,
    thresholds: impl Fn(bool, usize) -> Thresholds,
) -> H264MbEdges {
    let mut out = H264MbEdges::default();
    for (dir, (direction_bs, direction_out)) in bs.iter().zip(&mut out.edges).enumerate() {
        let (outer, inner) = (thresholds(true, dir), thresholds(false, dir));
        for (edge, (&edge_bs, params)) in direction_bs.iter().zip(direction_out).enumerate() {
            let t = if edge == 0 { outer } else { inner };
            *params = H264EdgeParams {
                bs: edge_bs,
                tc0: t.tc0s(edge_bs),
                alpha: t.alpha,
                beta: t.beta,
            };
            let filtered = (edges >> edge) & 1 != 0 && edge_bs != [0; 4];
            out.mask |= u8::from(filtered) << (4 * dir + edge);
        }
    }
    out
}

/// `QP_Y` as the filter sees it: 0 for `I_PCM` (§8.7.2.2).
fn filter_qp(mb: &MbState) -> i32 {
    if mb.kind == MbKind::IPcm {
        0
    } else {
        i32::from(mb.qp)
    }
}

/// The boundary strengths of macroblock `addr`, or `None` when it is not
/// filtered or every strength is 0.
fn strengths<const BI: bool, const FIELD: bool, const SW: bool, T>(
    mbs: &[MbState],
    width_mbs: usize,
    (first_slice, slices): (u32, &[SliceFilter]),
    addr: usize,
    all_horizontal: bool,
) -> Option<Strengths> {
    let mb = &mbs[addr];
    let control = slices[(mb.slice - first_slice) as usize].control;
    if !control.filters() {
        return None;
    }
    let (mx, my) = (addr % width_mbs, addr / width_mbs);
    // §8.7: a macroblock edge is filtered unless it is a picture edge, or
    // a slice edge with disable_deblocking_filter_idc 2.
    let edge_neighbour = |available: bool, n: usize| -> Option<&MbState> {
        let neighbour = available.then(|| &mbs[n])?;
        (control.disable_idc != DEBLOCK_IDC_NO_SLICE_EDGES || neighbour.slice == mb.slice)
            .then_some(neighbour)
    };
    let (left_addr, above_addr) = (addr.wrapping_sub(1), addr.wrapping_sub(width_mbs));
    let left = edge_neighbour(mx > 0, left_addr);
    let above = edge_neighbour(my > 0, above_addr);
    let mut out = [[[0u8; 4]; 4]; 2];
    let q_uniform = is_uniform_inter(mb);
    let [vertical, horizontal] = &mut out;
    direction_strengths::<0, BI, FIELD, T>(mb, left, q_uniform, false, vertical);
    direction_strengths::<1, BI, FIELD, T>(mb, above, q_uniform, all_horizontal, horizontal);
    if SW {
        switching_strengths::<FIELD>(&mut out, (mb, [left, above]), (first_slice, slices));
    }
    (out != [[[0; 4]; 4]; 2]).then_some(out)
}

/// Raise the strengths of a macroblock's edges that have a macroblock of
/// an SP or SI slice on either side to the intra ones (§8.7.2.1: bS 4 on
/// macroblock edges, but 3 on the horizontal ones of a field, 3 inside).
fn switching_strengths<const FIELD: bool>(
    out: &mut Strengths,
    (mb, neighbours): (&MbState, [Option<&MbState>; 2]),
    (first_slice, slices): (u32, &[SliceFilter]),
) {
    let switching = |m: &MbState| slices[(m.slice - first_slice) as usize].switching;
    let inside = switching(mb);
    for (dir, (edges, neighbour)) in out.iter_mut().zip(neighbours).enumerate() {
        if inside {
            for edge in &mut edges[1..] {
                *edge = [BS_INTRA; 4];
            }
        }
        if neighbour.is_some_and(|n| inside || switching(n)) {
            let strong = !FIELD || dir == 0;
            edges[0] = [if strong { BS_INTRA_MB_EDGE } else { BS_INTRA }; 4];
        }
    }
}

/// The strengths of the four luma edges of direction `DIR` (0 vertical) of
/// `mb`, with `neighbour` across edge 0; only edge 0 when `mb` is a
/// uniform inter macroblock. `all`: every edge, even those the 8x8
/// transform leaves unfiltered in luma (4:2:2 chroma filters them).
fn direction_strengths<const DIR: usize, const BI: bool, const FIELD: bool, T>(
    mb: &MbState,
    neighbour: Option<&MbState>,
    q_uniform: bool,
    all: bool,
    out: &mut [[u8; 4]; 4],
) {
    if let Some(p_mb) = neighbour {
        out[0] = edge_strengths::<DIR, BI, FIELD, T>(p_mb, mb, 0);
    }
    if q_uniform {
        return;
    }
    // With the 8x8 transform the luma edges 1 and 3 are not transform
    // block edges and are not filtered (§8.7).
    let step = if mb.transform_8x8 && !all { 2 } else { 1 };
    for edge in (step..4).step_by(step) {
        out[edge] = edge_strengths::<DIR, BI, FIELD, T>(mb, mb, edge);
    }
}

/// An inter macroblock without luma coefficients whose 4x4 blocks share one
/// vector and reference: its internal edges are bS 0 (most of a P
/// picture).
fn is_uniform_inter(mb: &MbState) -> bool {
    mb.single_motion && mb.nonzero == 0
}

/// The four segment strengths of luma edge `edge` of direction `dir` of
/// `mb` (`p_mb` across it: the neighbour for edge 0, else `mb`). The
/// coefficient rule is evaluated for all four segments at once on the
/// blocks' coded-bit masks; the motion rule only where it decides, and
/// once for the whole edge when both sides have a single vector.
#[allow(
    clippy::extra_unused_type_parameters,
    reason = "one copy per sample type keeps the 8-bit instance inlined (high-bit-depth handoff)"
)]
fn edge_strengths<const DIR: usize, const BI: bool, const FIELD: bool, T>(
    p_mb: &MbState,
    mb: &MbState,
    edge: usize,
) -> [u8; 4] {
    if p_mb.is_intra() || mb.is_intra() {
        // bS 4 needs a macroblock edge between frame macroblocks, or a
        // vertical one.
        let strong = edge == 0 && (!FIELD || DIR == 0);
        return [if strong { BS_INTRA_MB_EDGE } else { BS_INTRA }; 4];
    }
    let p_edge = (edge + 3) % 4;
    let coded = if DIR == 0 {
        column_bits(mb.nonzero, edge) | column_bits(p_mb.nonzero, p_edge)
    } else {
        row_bits(mb.nonzero, edge) | row_bits(p_mb.nonzero, p_edge)
    };
    if coded == SEGMENTS_ALL {
        return [BS_COEFFICIENTS; 4];
    }
    // Once per edge: whether either side predicts from list 1.
    let bi = BI && (p_mb.uses_l1() | mb.uses_l1());
    if p_mb.single_motion && mb.single_motion {
        // One motion each side: one motion test for the edge, and the
        // segments combine it with their coded bits arithmetically.
        let moved = u32::from(motion_differs::<FIELD>(p_mb, (0, 0), mb, (0, 0), bi));
        // One byte per segment: bS 2 where coded, else the motion bit.
        let coded = spread_bits(coded);
        let bs =
            coded * u32::from(BS_COEFFICIENTS) + ((moved * SEGMENT_ONES) & !coded & SEGMENT_ONES);
        return bs.to_le_bytes();
    }
    core::array::from_fn(|seg| {
        if (coded >> seg) & 1 != 0 {
            BS_COEFFICIENTS
        } else {
            let (q, p) = if DIR == 0 {
                ((edge, seg), (p_edge, seg))
            } else {
                ((seg, edge), (seg, p_edge))
            };
            u8::from(motion_differs::<FIELD>(p_mb, p, mb, q, bi))
        }
    })
}

/// All four segments of an edge.
const SEGMENTS_ALL: u8 = 0b1111;
/// 1 in each byte of a little-endian `u32` holding four segment bS.
const SEGMENT_ONES: u32 = 0x0101_0101;

/// Bits 0..3 of `bits` as bytes 0..3 (0 or 1) of a little-endian `u32`:
/// the multiply copies bit `i` to bit `8 · i` (and elsewhere), the mask
/// keeps one bit per byte.
fn spread_bits(bits: u8) -> u32 {
    (u32::from(bits & 0xf) * 0x0020_4081) & SEGMENT_ONES
}

/// The coded bits (`MbState::nonzero`, bit `y * 4 + x`) of 4x4 row `y`,
/// by column.
fn row_bits(nonzero: u16, y: usize) -> u8 {
    ((nonzero >> (4 * y)) & 0xf) as u8
}

/// The coded bits of 4x4 column `x`, by row.
fn column_bits(nonzero: u16, x: usize) -> u8 {
    let column = u32::from((nonzero >> x) & 0x1111);
    // Rows 0..3 sit at bits 0, 4, 8, 12; one multiply by 2^0 + 2^3 + 2^6 +
    // 2^9 places them at bits 9..12 (every partial product lands on its own
    // bit, so nothing carries).
    ((column * COLUMN_GATHER) >> 9) as u8 & 0xf
}

/// See [`column_bits`].
const COLUMN_GATHER: u32 = 0x249;

/// bS 1 rule of inter blocks without coefficients (§8.7.2.1): different
/// reference pictures or numbers of vectors, or vectors a sample or more
/// apart (in field rows between field macroblocks, `FIELD`). `bi`: either
/// macroblock predicts from list 1 (else both have one list 0 vector per
/// block, the inlined common case).
#[allow(
    clippy::inline_always,
    reason = "measured: keeps the P path at the base instruction count (callgrind)"
)]
#[inline(always)]
fn motion_differs<const FIELD: bool>(
    p_mb: &MbState,
    p: (usize, usize),
    q_mb: &MbState,
    q: (usize, usize),
    bi: bool,
) -> bool {
    if bi {
        return bi_motion_differs::<FIELD>(p_mb, p, q_mb, q);
    }
    let (p_b8, q_b8) = ((p.1 / 2) * 2 + p.0 / 2, (q.1 / 2) * 2 + q.0 / 2);
    (p_mb.ref_pic[0][p_b8] != q_mb.ref_pic[0][q_b8])
        | far::<FIELD>(p_mb.mvs[0][p.1 * 4 + p.0], q_mb.mvs[0][q.1 * 4 + q.0])
}

/// [`motion_differs`] where either side may use both lists.
#[inline(never)]
fn bi_motion_differs<const FIELD: bool>(
    p_mb: &MbState,
    p: (usize, usize),
    q_mb: &MbState,
    q: (usize, usize),
) -> bool {
    let far = far::<FIELD>;
    let (p_block, q_block) = (p.1 * 4 + p.0, q.1 * 4 + q.0);
    let (p_b8, q_b8) = ((p.1 / 2) * 2 + p.0 / 2, (q.1 / 2) * 2 + q.0 / 2);
    // Pictures (0: none) and vectors of each side's two lists.
    let pics = |m: &MbState, b8: usize| [m.ref_pic[0][b8], m.ref_pic[1][b8]];
    let mvs = |m: &MbState, block: usize| [m.mvs[0][block], m.mvs[1][block]];
    let ([p0, p1], [q0, q1]) = (pics(p_mb, p_b8), pics(q_mb, q_b8));
    // Vectors of an unused list are not read (stale in P macroblocks).
    let ([pv0, pv1], [qv0, qv1]) = (mvs(p_mb, p_block), mvs(q_mb, q_block));
    // The same set of reference pictures, counted with repetition.
    if !((p0 == q0 && p1 == q1) || (p0 == q1 && p1 == q0)) {
        return true;
    }
    if p0 == 0 || p1 == 0 {
        // One vector each (the unused list's picture is 0 on both sides).
        let (pv, qv) = (
            if p0 == 0 { pv1 } else { pv0 },
            if q0 == 0 { qv1 } else { qv0 },
        );
        return far(pv, qv);
    }
    if p0 != p1 {
        // Two different pictures: compare the vectors that use the same one.
        return if p0 == q0 {
            far(pv0, qv0) | far(pv1, qv1)
        } else {
            far(pv0, qv1) | far(pv1, qv0)
        };
    }
    // Both vectors on one picture: bS 1 only when both pairings differ.
    (far(pv0, qv0) | far(pv1, qv1)) & (far(pv0, qv1) | far(pv1, qv0))
}

/// Whether two vectors are a sample or more apart in either component
/// (vertically a field row between field macroblocks, `FIELD`). Bitwise,
/// not short-circuit: no branch per term.
#[allow(
    clippy::inline_always,
    reason = "measured: keeps the P path at the base instruction count (callgrind)"
)]
#[inline(always)]
fn far<const FIELD: bool>(a: Mv, b: Mv) -> bool {
    let limit_y = if FIELD {
        FIELD_MV_THRESHOLD_Y
    } else {
        MV_THRESHOLD
    };
    ((i32::from(a.x) - i32::from(b.x)).abs() >= MV_THRESHOLD)
        | ((i32::from(a.y) - i32::from(b.y)).abs() >= limit_y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_gathers_match_their_loops() {
        for nonzero in [0u16, 0xffff, 0x1234, 0x8421, 0x0f0f, 0xa5a5] {
            for x in 0..4 {
                let expected =
                    (0..4).fold(0u8, |m, y| m | (((nonzero >> (4 * y + x)) & 1) as u8) << y);
                assert_eq!(column_bits(nonzero, x), expected);
            }
        }
        for bits in 0..16u8 {
            let expected = u32::from_le_bytes(core::array::from_fn(|i| (bits >> i) & 1));
            assert_eq!(spread_bits(bits), expected);
        }
    }
}
