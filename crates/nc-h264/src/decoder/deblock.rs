//! The deblocking filter of a decoded picture (§8.7) with per-slice
//! control: `disable_deblocking_filter_idc` 1 (off) and 2 (no filtering
//! across slice edges), per-slice filter offsets and chroma QP offsets.
//!
//! Boundary strengths (§8.7.2.1) come from the macroblock states; the
//! sample filters of each edge (§8.7.2.3, §8.7.2.4) are the nc-dsp kernels.
//!
//! The filter runs a macroblock row behind decoding ([`RowDeblocker`]),
//! while the row is still in cache: filtering row `r` changes samples of
//! rows `r - 1` and `r` only, and the intra prediction of row `r + 1`
//! (which must see unfiltered samples) has read row `r` by the time row
//! `r + 1` is complete. Rows are filtered in order, so the result is the
//! standard's whole-picture raster order.

use core::ops::Range;

use nc_dsp::{Dsp, H264EdgeParams, H264MbEdges};
use nc_encoder::Mv;

use super::Planes;
use super::mbstate::{MbKind, MbState};
use crate::deblock::{DEBLOCK_IDC_NO_SLICE_EDGES, DeblockControl};
use crate::tables::{DEBLOCK_ALPHA, DEBLOCK_BETA, DEBLOCK_TC0, chroma_qp};

/// Vector difference (quarter samples) from which `bS` is 1 (§8.7.2.1).
const MV_THRESHOLD: i32 = 4;
/// Largest `indexA` / `indexB` (eqs. 8-461, 8-462).
const MAX_INDEX: i32 = 51;
/// QPs (luma and chroma) 0..=51, and so their averages.
const QP_COUNT: usize = 52;
/// `bS` of macroblock edges and internal edges next to intra samples.
const BS_INTRA_MB_EDGE: u8 = 4;
const BS_INTRA: u8 = 3;
const BS_COEFFICIENTS: u8 = 2;

/// The deblocking parameters of one slice.
#[derive(Debug, Clone)]
pub(super) struct SliceFilter {
    pub control: DeblockControl,
    pub chroma_qp_offset: [i8; 2],
    /// The thresholds at each average QP, under `control`'s offsets.
    thresholds: [Thresholds; QP_COUNT],
}

impl SliceFilter {
    pub fn new(control: DeblockControl, chroma_qp_offset: [i8; 2]) -> Self {
        Self {
            control,
            chroma_qp_offset,
            thresholds: core::array::from_fn(|average| Thresholds::at_average(average, control)),
        }
    }

    /// The thresholds of an edge between QPs `qp_p` and `qp_q` (eq.
    /// 8-460: their rounded average), by table.
    fn thresholds(&self, qp_p: u8, qp_q: u8) -> Thresholds {
        self.thresholds[(usize::from(qp_p) + usize::from(qp_q) + 1) >> 1]
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
    /// The thresholds at the average QP `average` (eqs. 8-461, 8-462).
    fn at_average(average: usize, control: DeblockControl) -> Self {
        let average = average as i32;
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
    pub fn filter_through(
        &mut self,
        last: usize,
        planes: &mut Planes<'_>,
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
    pub monochrome: bool,
}

/// Filter the macroblock rows `rows` of `picture`.
fn deblock_rows(ctx: &DeblockContext<'_>, planes: &mut Planes<'_>, rows: Range<usize>) {
    let DeblockContext {
        dsp,
        mbs,
        width_mbs,
        first_slice,
        slices,
        monochrome,
    } = *ctx;
    if slices.iter().all(|s| !s.control.filters()) {
        return;
    }
    let [luma, cb, cr] = planes;
    for addr in rows.start * width_mbs..rows.end * width_mbs {
        let Some(bs) = strengths(mbs, width_mbs, (first_slice, slices), addr) else {
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
        let luma_edges = mb_edges(&bs, LUMA_EDGES, |edge_0, dir| {
            let qp_p = if edge_0 { neighbour_qp[dir] } else { qp };
            filter.thresholds(qp_p, qp)
        });
        dsp.h264_deblock_mb_luma(luma, (mx * 16, my * 16), &luma_edges);
        if monochrome {
            continue;
        }
        // 4:2:0 chroma edges lie on luma edges 0 and 2 and take their bS
        // (§8.7.2.1); QPs map through each component's offset.
        let chroma = |comp: usize| {
            let offset = filter.chroma_qp_offset[comp];
            mb_edges(&bs, CHROMA_EDGES, |edge_0, dir| {
                let qp_p = if edge_0 { neighbour_qp[dir] } else { qp };
                filter.thresholds(chroma_qp(qp_p, offset), chroma_qp(qp, offset))
            })
        };
        let cb_edges = chroma(0);
        let position = (mx * 8, my * 8);
        if filter.chroma_qp_offset[0] == filter.chroma_qp_offset[1] {
            // Both planes share bS and thresholds: one call filters both.
            dsp.h264_deblock_mb_chroma(cb, cr, position, &cb_edges);
        } else {
            dsp.h264_deblock_mb_chroma_plane(cb, position, &cb_edges);
            dsp.h264_deblock_mb_chroma_plane(cr, position, &chroma(1));
        }
    }
}

/// Luma edges of a macroblock: all four of each direction.
const LUMA_EDGES: u8 = 0b1111;
/// 4:2:0 chroma edges: luma edges 0 and 2 of each direction.
const CHROMA_EDGES: u8 = 0b0101;

/// The edges `edges` (a mask of edge indices, per direction) of a
/// macroblock with strengths `bs` as nc-dsp's macroblock form, with the
/// thresholds `thresholds(is_edge_0, direction)`. Edges whose bS is all
/// 0 are left out of the mask (filtered as nothing).
fn mb_edges(
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
fn filter_qp(mb: &MbState) -> u8 {
    if mb.kind == MbKind::IPcm { 0 } else { mb.qp }
}

/// The boundary strengths of macroblock `addr`, or `None` when it is not
/// filtered or every strength is 0.
fn strengths(
    mbs: &[MbState],
    width_mbs: usize,
    (first_slice, slices): (u32, &[SliceFilter]),
    addr: usize,
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
    direction_strengths::<0>(mb, left, q_uniform, vertical);
    direction_strengths::<1>(mb, above, q_uniform, horizontal);
    (out != [[[0; 4]; 4]; 2]).then_some(out)
}

/// The strengths of the four luma edges of direction `DIR` (0 vertical) of
/// `mb`, with `neighbour` across edge 0; only edge 0 when `mb` is a
/// uniform inter macroblock.
fn direction_strengths<const DIR: usize>(
    mb: &MbState,
    neighbour: Option<&MbState>,
    q_uniform: bool,
    out: &mut [[u8; 4]; 4],
) {
    if let Some(p_mb) = neighbour {
        out[0] = edge_strengths::<DIR>(p_mb, mb, 0);
    }
    if q_uniform {
        return;
    }
    // With the 8x8 transform the luma edges 1 and 3 are not transform
    // block edges and are not filtered (§8.7).
    let step = if mb.transform_8x8 { 2 } else { 1 };
    for edge in (step..4).step_by(step) {
        out[edge] = edge_strengths::<DIR>(mb, mb, edge);
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
fn edge_strengths<const DIR: usize>(p_mb: &MbState, mb: &MbState, edge: usize) -> [u8; 4] {
    if p_mb.is_intra() || mb.is_intra() {
        return [if edge == 0 {
            BS_INTRA_MB_EDGE
        } else {
            BS_INTRA
        }; 4];
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
    if p_mb.single_motion && mb.single_motion {
        // One vector each side: one motion test for the edge, and the
        // segments combine it with their coded bits arithmetically.
        let moved = u32::from(motion_differs(p_mb, (0, 0), mb, (0, 0)));
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
            u8::from(motion_differs(p_mb, p, mb, q))
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
/// reference pictures, or vectors a sample or more apart.
fn motion_differs(p_mb: &MbState, p: (usize, usize), q_mb: &MbState, q: (usize, usize)) -> bool {
    let (p_mv, q_mv) = (p_mb.mvs[p.1 * 4 + p.0], q_mb.mvs[q.1 * 4 + q.0]);
    let ref_of = |m: &MbState, (x, y): (usize, usize)| m.ref_pic[(y / 2) * 2 + x / 2];
    // Bitwise, not short-circuit: no branch per term.
    let far = |a: Mv, b: Mv| {
        ((i32::from(a.x) - i32::from(b.x)).abs() >= MV_THRESHOLD)
            | ((i32::from(a.y) - i32::from(b.y)).abs() >= MV_THRESHOLD)
    };
    (ref_of(p_mb, p) != ref_of(q_mb, q)) | far(p_mv, q_mv)
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
