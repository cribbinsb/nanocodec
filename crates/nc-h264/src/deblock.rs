//! The H.264 deblocking filter (§8.7), applied to a whole reconstructed
//! picture in place after its last macroblock.
//!
//! Macroblocks are filtered in raster order; within one, the four vertical
//! luma edges left to right, then the four horizontal edges top to bottom,
//! then the same for each chroma plane (two edges each way for 4:2:0). Each
//! edge is filtered in 4-sample segments with a boundary strength `bS`
//! derived per segment (§8.7.2.1). The picture's own left and top edges are
//! not filtered. The slice header's [`DeblockControl`] can switch the filter
//! off or move its thresholds (`FilterOffsetA`/`FilterOffsetB`, §8.7.2.2);
//! normal encodes use [`DeblockControl::DEFAULT`] (on, zero offsets).
//!
//! The boundary strengths and thresholds are derived here into one
//! [`H264MbEdges`] per macroblock and plane type; the sample filtering
//! (§8.7.2.3, §8.7.2.4) is `nc_dsp::Dsp::h264_deblock_mb_luma` /
//! `h264_deblock_mb_chroma`, one call per macroblock.

use nc_common::{Picture, PlaneIndex};
use nc_dsp::{Dsp, H264EdgeParams, H264MbEdges};
use nc_encoder::Mv;

use crate::mbinfo::MbInfo;
use crate::tables::{DEBLOCK_ALPHA, DEBLOCK_BETA, DEBLOCK_TC0, chroma_qp};

/// Vertical motion difference (quarter samples) from which bS is 1
/// (§8.7.2.1: "greater than or equal to 4 in units of quarter luma frame
/// samples"); the same threshold applies horizontally.
const MV_DIFF_STRENGTH_THRESHOLD: i32 = 4;

/// `disable_deblocking_filter_idc` values (§7.4.3).
pub const DEBLOCK_IDC_ON: u8 = 0;
/// The filter is off for the slice.
pub const DEBLOCK_IDC_OFF: u8 = 1;
/// On, except across slice edges (with one slice per picture: everywhere).
pub const DEBLOCK_IDC_NO_SLICE_EDGES: u8 = 2;
/// Range of `slice_alpha_c0_offset_div2` and `slice_beta_offset_div2`:
/// -6..=6 (§7.4.3).
pub const DEBLOCK_OFFSET_DIV2_MAX: i8 = 6;

/// The slice header's deblocking control (§7.3.3, §7.4.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeblockControl {
    /// `disable_deblocking_filter_idc`.
    pub disable_idc: u8,
    /// `slice_alpha_c0_offset_div2` (`FilterOffsetA / 2`).
    pub alpha_offset_div2: i8,
    /// `slice_beta_offset_div2` (`FilterOffsetB / 2`).
    pub beta_offset_div2: i8,
}

impl DeblockControl {
    /// Filter on everywhere with zero offsets.
    pub const DEFAULT: Self = Self {
        disable_idc: DEBLOCK_IDC_ON,
        alpha_offset_div2: 0,
        beta_offset_div2: 0,
    };

    /// Filter off (`disable_deblocking_filter_idc` 1, no offsets coded).
    pub const OFF: Self = Self {
        disable_idc: DEBLOCK_IDC_OFF,
        ..Self::DEFAULT
    };

    /// Whether the slice is filtered at all. With one slice per picture,
    /// idc 2 filters exactly the edges idc 0 does.
    pub fn filters(self) -> bool {
        self.disable_idc != DEBLOCK_IDC_OFF
    }

    /// Whether the offsets are coded: not when the filter is off (§7.3.3).
    pub fn offsets_coded(self) -> bool {
        self.filters()
    }
}

/// Which way an edge runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    /// A vertical edge between horizontally adjacent samples.
    Vertical,
    /// A horizontal edge between vertically adjacent samples.
    Horizontal,
}

/// Deblock `recon` (coded size, `mb_cols` x `mb_rows` macroblocks) using
/// the per-macroblock state `mbs`, including each macroblock's `QP_Y`.
/// `chroma_offset` is the PPS `chroma_qp_index_offset` (§8.7.2.2: `QPc` of
/// each side); `control` the slice header's deblocking control.
pub fn deblock_picture(
    dsp: &Dsp,
    recon: &mut Picture,
    mbs: &[Option<MbInfo>],
    mb_cols: usize,
    chroma_offset: i8,
    control: DeblockControl,
) {
    if !control.filters() {
        return;
    }
    // Boundary strengths of every macroblock first (`None`: all zero), then
    // each plane type filtered in one pass, one call per macroblock. Planes
    // do not interact, and within a plane the order (macroblocks in raster
    // order, vertical edges before horizontal ones) is the standard's.
    let strengths: Vec<Option<MbStrengths>> = (0..mbs.len())
        .map(|index| mb_strengths(mbs, mb_cols, index))
        .collect();
    let thresholds = ThresholdTable::new(control, chroma_offset);
    let mut luma = recon.plane_mut(PlaneIndex::Y).view_mut();
    for (index, (bs, mb)) in strengths.iter().zip(mbs).enumerate() {
        let (Some(bs), Some(mb)) = (bs, mb) else {
            continue;
        };
        let edges = luma_edges(
            bs,
            mb.qp,
            |dir| edge_qps(mbs, mb_cols, index, dir, 0),
            &thresholds,
        );
        if edges.mask != 0 {
            let (col, row) = (index % mb_cols, index / mb_cols);
            dsp.h264_deblock_mb_luma(&mut luma, (col * MB_SIZE, row * MB_SIZE), &edges);
        }
    }
    let [_, u, v] = recon.planes_mut();
    let (mut u, mut v) = (u.view_mut(), v.view_mut());
    for (index, (bs, mb)) in strengths.iter().zip(mbs).enumerate() {
        let (Some(bs), Some(mb)) = (bs, mb) else {
            continue;
        };
        let neighbour_qps = |dir| edge_qps(mbs, mb_cols, index, dir, 0);
        let edges = chroma_edges(bs, mb.qp, neighbour_qps, &thresholds);
        if edges.mask != 0 {
            let (col, row) = (index % mb_cols, index / mb_cols);
            let pos = (col * MB_CHROMA_SIZE, row * MB_CHROMA_SIZE);
            dsp.h264_deblock_mb_chroma(&mut u, &mut v, pos, &edges);
        }
    }
}

/// Luma macroblock size, and 4:2:0 chroma.
const MB_SIZE: usize = 16;
const MB_CHROMA_SIZE: usize = 8;

/// The luma edges of a macroblock with boundary strengths `bs` and `QP_Y`
/// `qp`: internal edges share the macroblock's thresholds, looked up once;
/// edge 0 averages with the neighbour's QP (`edge0_qps(dir)` gives the p and
/// q `QP_Y` across it). Edges whose bS is all zero (the picture's own edges,
/// and with the 8x8 transform internal edges 1 and 3, which are not
/// transform block edges: §8.7) are left out of the mask.
fn luma_edges(
    bs: &MbStrengths,
    qp: u8,
    edge0_qps: impl Fn(Direction) -> (u8, u8),
    thresholds: &ThresholdTable,
) -> H264MbEdges {
    let internal = thresholds.at(qp);
    let mut edges = H264MbEdges::default();
    for (d, dir) in [Direction::Vertical, Direction::Horizontal]
        .into_iter()
        .enumerate()
    {
        for (e, &edge_bs) in bs[d].iter().enumerate() {
            if edge_bs == [0; 4] {
                continue;
            }
            let t = if e == 0 {
                thresholds.at(average_qp(edge0_qps(dir)))
            } else {
                internal
            };
            edges.mask |= 1 << (4 * d + e);
            edges.edges[d][e] = t.edge(edge_bs);
        }
    }
    edges
}

/// The 4:2:0 chroma edges of a macroblock (both planes): edges 0 and 2 of
/// the luma grid, with the luma bS (chroma line k takes segment k / 2) and
/// thresholds at the average of each side's `QPc` (§8.7.2.2, chromaEdgeFlag
/// 1).
fn chroma_edges(
    bs: &MbStrengths,
    qp: u8,
    edge0_qps: impl Fn(Direction) -> (u8, u8),
    thresholds: &ThresholdTable,
) -> H264MbEdges {
    let internal = thresholds.at(thresholds.chroma_qp(qp));
    let mut edges = H264MbEdges::default();
    for (d, dir) in [Direction::Vertical, Direction::Horizontal]
        .into_iter()
        .enumerate()
    {
        for e in [0, 2] {
            let edge_bs = bs[d][e];
            if edge_bs == [0; 4] {
                continue;
            }
            let t = if e == 0 {
                let (p, q) = edge0_qps(dir);
                thresholds.at(average_qp((
                    thresholds.chroma_qp(p),
                    thresholds.chroma_qp(q),
                )))
            } else {
                internal
            };
            edges.mask |= 1 << (4 * d + e);
            edges.edges[d][e] = t.edge(edge_bs);
        }
    }
    edges
}

/// `QP_Y` of the p and q sides of luma edge `edge` of macroblock `index`:
/// the neighbouring macroblock's across the macroblock edge (edge 0), the
/// macroblock's own inside it.
fn edge_qps(
    mbs: &[Option<MbInfo>],
    mb_cols: usize,
    index: usize,
    dir: Direction,
    edge: usize,
) -> (u8, u8) {
    let qp_of = |i: usize| mbs[i].as_ref().map_or(0, |mb| mb.qp);
    let q = qp_of(index);
    let p_index = match (edge, dir) {
        (0, Direction::Vertical) if !index.is_multiple_of(mb_cols) => index - 1,
        (0, Direction::Horizontal) if index >= mb_cols => index - mb_cols,
        _ => index, // internal edge, or a picture edge (never filtered)
    };
    (qp_of(p_index), q)
}

/// `qPav = (qPp + qPq + 1) >> 1` (§8.7.2.2).
fn average_qp((p, q): (u8, u8)) -> u8 {
    ((u16::from(p) + u16::from(q) + 1) >> 1) as u8
}

/// Boundary strengths of one macroblock: per direction (vertical edges,
/// then horizontal), per luma edge, per 4-sample segment.
type MbStrengths = [[[u8; 4]; 4]; 2];

/// [`MbStrengths`] of macroblock `index`, or `None` when every one is zero.
fn mb_strengths(mbs: &[Option<MbInfo>], mb_cols: usize, index: usize) -> Option<MbStrengths> {
    let mb = mbs[index]
        .as_ref()
        .expect("every macroblock is coded before deblocking");
    let (col, row) = (index % mb_cols, index / mb_cols);
    let left = (col > 0).then(|| mbs[index - 1].as_ref()).flatten();
    let above = (row > 0).then(|| mbs[index - mb_cols].as_ref()).flatten();
    let uniform_inter = is_uniform_inter(mb);
    // A uniform inter macroblock whose left and top neighbours are uniform
    // inter with the same vector (or absent) has bS 0 on every edge: most
    // of a P picture.
    if uniform_inter {
        let quiet = |n: Option<&MbInfo>| {
            n.is_none_or(|n| is_uniform_inter(n) && motion_strength(mb, 0, n, 0) == 0)
        };
        if quiet(left) && quiet(above) {
            return None;
        }
    }
    Some([Direction::Vertical, Direction::Horizontal].map(|dir| {
        let neighbour = if dir == Direction::Vertical {
            left
        } else {
            above
        };
        core::array::from_fn(|edge| {
            // These are not 8x8 transform boundaries. Luma skips them and
            // chroma only uses strengths from edges 0 and 2.
            if mb.transform_8x8 && edge % 2 == 1 {
                return [0; 4];
            }
            // The internal edges of an inter macroblock with no coded luma
            // and one vector are all bS 0 (§8.7.2.1: same vector, no
            // coefficients on either side), so they are not derived one by
            // one.
            if edge > 0 && uniform_inter {
                return [0; 4];
            }
            // Between two uniform inter macroblocks with no coded luma every
            // segment of edge 0 has the same bS: only their single vectors
            // differ or not.
            if edge == 0
                && uniform_inter
                && let Some(n) = neighbour.filter(|n| is_uniform_inter(n))
            {
                return [motion_strength(mb, 0, n, 0); 4];
            }
            edge_strengths(mb, neighbour, dir, edge)
        })
    }))
}

/// An inter macroblock with no coded luma and one vector: all its internal
/// edges have bS 0 and its edges to another such macroblock one bS.
fn is_uniform_inter(mb: &MbInfo) -> bool {
    !mb.is_intra() && mb.luma_coded == 0 && mb.mvs.iter().all(|&mv| mv == mb.mvs[0])
}

/// bS 1 if quadrant `q_quadrant` of `q` and `p_quadrant` of `p` predict
/// from different reference pictures or with vectors at least one sample
/// apart, else 0 (§8.7.2.1). Every partition of a macroblock uses its one
/// reference index, and the indices of the one slice name distinct
/// pictures, so comparing indices compares pictures.
fn motion_strength(q: &MbInfo, q_quadrant: usize, p: &MbInfo, p_quadrant: usize) -> u8 {
    if q.ref_idx != p.ref_idx {
        return 1;
    }
    mv_strength(q.mvs[q_quadrant], p.mvs[p_quadrant])
}

/// bS 1 if two vectors differ by at least one sample in either component,
/// else 0 (§8.7.2.1, same reference picture).
fn mv_strength(q_mv: Mv, p_mv: Mv) -> u8 {
    let (dx, dy) = (
        i32::from(q_mv.x) - i32::from(p_mv.x),
        i32::from(q_mv.y) - i32::from(p_mv.y),
    );
    u8::from(dx.abs() >= MV_DIFF_STRENGTH_THRESHOLD || dy.abs() >= MV_DIFF_STRENGTH_THRESHOLD)
}

/// bS of the four segments of luma edge `edge` (0..=3) of `mb` (§8.7.2.1).
/// `neighbour` is the macroblock across edge 0; `None` at the picture edge
/// (bS 0: not filtered). The intra tests and the vector comparisons are made
/// once per edge (vectors are per 8x8 quadrant, so per half edge), not per
/// segment; only the coded-coefficient test is per segment.
fn edge_strengths(mb: &MbInfo, neighbour: Option<&MbInfo>, dir: Direction, edge: usize) -> [u8; 4] {
    let p_mb = if edge == 0 {
        let Some(n) = neighbour else {
            return [0; 4];
        };
        n
    } else {
        mb
    };
    if mb.is_intra() || p_mb.is_intra() {
        return [if edge == 0 { 4 } else { 3 }; 4];
    }
    // The p side's blocks: the neighbour's last column or row across the
    // macroblock edge, the previous one inside it.
    let p_edge = if edge == 0 { 3 } else { edge - 1 };
    // (bx, by) of the block at `line` (edge column or row) and `seg`.
    let block = |line: usize, seg: usize| match dir {
        Direction::Vertical => (line, seg),
        Direction::Horizontal => (seg, line),
    };
    let quadrant = |(bx, by): (usize, usize)| (by / 2) * 2 + bx / 2;
    // Segments 0-1 and 2-3 each lie in one quadrant on either side.
    let mv_bs = [0, 2].map(|seg| {
        motion_strength(
            mb,
            quadrant(block(edge, seg)),
            p_mb,
            quadrant(block(p_edge, seg)),
        )
    });
    core::array::from_fn(|seg| {
        let (q, p) = (block(edge, seg), block(p_edge, seg));
        if mb.luma_block_coded(q.0, q.1) || p_mb.luma_block_coded(p.0, p.1) {
            2
        } else {
            mv_bs[seg / 2]
        }
    })
}

/// Thresholds of one edge at its average QP `qPav` (§8.7.2.2).
#[derive(Debug, Clone, Copy)]
struct Thresholds {
    alpha: u8,
    beta: u8,
    /// `tC0` by bS (0..=4): the `indexA` row of Table 8-17 for bS 1..=3, 0
    /// where bS is 0 or 4 (unused there). Padded to a power of two so a
    /// masked bS indexes it without a bounds check.
    tc0_by_bs: [u8; TC0_SLOTS],
}

/// Number of `QP_Y` (and `qPav`) values, 0..=51.
const QP_COUNT: usize = 52;

/// [`Thresholds`] of every `qPav` and `QPc` of every `QP_Y` for one slice's
/// deblocking control, built once per picture: each macroblock looks its
/// edges' thresholds up instead of clipping indices into three tables.
struct ThresholdTable {
    by_qp: [Thresholds; QP_COUNT],
    /// `QPc` of each `QP_Y` with the PPS `chroma_qp_index_offset`.
    chroma_qp: [u8; QP_COUNT],
}

impl ThresholdTable {
    fn new(control: DeblockControl, chroma_offset: i8) -> Self {
        Self {
            by_qp: core::array::from_fn(|qp| Thresholds::at(qp as u8, control)),
            chroma_qp: core::array::from_fn(|qp| chroma_qp(qp as u8, chroma_offset)),
        }
    }

    /// Thresholds of an edge at `qPav` `qp` (luma or chroma, §8.7.2.2).
    fn at(&self, qp: u8) -> Thresholds {
        self.by_qp[usize::from(qp)]
    }

    /// `QPc` of `QP_Y` `qp`.
    fn chroma_qp(&self, qp: u8) -> u8 {
        self.chroma_qp[usize::from(qp)]
    }
}

/// Entries of [`Thresholds::tc0_by_bs`]: bS 0..=4, padded to a power of two.
const TC0_SLOTS: usize = 8;

/// Largest `indexA` / `indexB` (§8.7.2.2, eqs. 8-461 and 8-462).
const MAX_THRESHOLD_INDEX: i32 = 51;

impl Thresholds {
    fn at(qp: u8, control: DeblockControl) -> Self {
        // indexA = Clip3(0, 51, qPav + FilterOffsetA), indexB likewise with
        // FilterOffsetB; each offset is twice its `_div2` syntax element.
        let index = |offset_div2: i8| {
            (i32::from(qp) + 2 * i32::from(offset_div2)).clamp(0, MAX_THRESHOLD_INDEX) as usize
        };
        let index_a = index(control.alpha_offset_div2);
        let [tc1, tc2, tc3] = DEBLOCK_TC0[index_a];
        Self {
            alpha: DEBLOCK_ALPHA[index_a],
            beta: DEBLOCK_BETA[index(control.beta_offset_div2)],
            tc0_by_bs: [0, tc1, tc2, tc3, 0, 0, 0, 0],
        }
    }

    /// The edge parameters of bS `bs` at these thresholds; `tC0` is 0
    /// where bS is 0 or 4 (unused there).
    fn edge(self, bs: [u8; 4]) -> H264EdgeParams {
        H264EdgeParams {
            bs,
            tc0: bs.map(|b| self.tc0_by_bs[usize::from(b) & (TC0_SLOTS - 1)]),
            alpha: self.alpha,
            beta: self.beta,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_move_and_clip_the_threshold_indices() {
        let control = |alpha_offset_div2, beta_offset_div2| DeblockControl {
            alpha_offset_div2,
            beta_offset_div2,
            ..DeblockControl::DEFAULT
        };
        // indexA = Clip3(0, 51, qPav + 2 * alpha_div2), indexB likewise (§8.7.2.2).
        let t = Thresholds::at(30, control(3, -2));
        let [tc1, tc2, tc3] = DEBLOCK_TC0[36];
        assert_eq!(t.tc0_by_bs, [0, tc1, tc2, tc3, 0, 0, 0, 0]);
        assert_eq!(t.alpha, DEBLOCK_ALPHA[36]);
        assert_eq!(t.beta, DEBLOCK_BETA[26]);
        let high = Thresholds::at(48, control(6, 6));
        assert_eq!(
            (high.alpha, high.beta),
            (DEBLOCK_ALPHA[51], DEBLOCK_BETA[51])
        );
        let low = Thresholds::at(4, control(-6, -6));
        assert_eq!((low.alpha, low.beta), (DEBLOCK_ALPHA[0], DEBLOCK_BETA[0]));
    }

    /// bS of segment `seg`, segment by segment as §8.7.2.1 states it: the
    /// reference [`edge_strengths`] must match.
    /// bS of segment `seg` of luma edge `edge` (0..=3) of `mb` (§8.7.2.1).
    /// `neighbour` is the macroblock across edge 0; `None` at the picture edge
    /// (bS 0: not filtered).
    fn boundary_strength(
        mb: &MbInfo,
        neighbour: Option<&MbInfo>,
        dir: Direction,
        edge: usize,
        seg: usize,
    ) -> u8 {
        // Blocks on the q side (current macroblock) and p side, as (bx, by).
        let q_blk = match dir {
            Direction::Vertical => (edge, seg),
            Direction::Horizontal => (seg, edge),
        };
        let (p_mb, p_blk) = if edge == 0 {
            let Some(n) = neighbour else {
                return 0;
            };
            (
                n,
                match dir {
                    Direction::Vertical => (3, seg),
                    Direction::Horizontal => (seg, 3),
                },
            )
        } else {
            (
                mb,
                match dir {
                    Direction::Vertical => (edge - 1, seg),
                    Direction::Horizontal => (seg, edge - 1),
                },
            )
        };
        if mb.is_intra() || p_mb.is_intra() {
            return if edge == 0 { 4 } else { 3 };
        }
        if mb.luma_block_coded(q_blk.0, q_blk.1) || p_mb.luma_block_coded(p_blk.0, p_blk.1) {
            return 2;
        }
        // Vectors of the 8x8 quadrants holding the q and p blocks.
        mv_strength(
            mb.mvs[(q_blk.1 / 2) * 2 + q_blk.0 / 2],
            p_mb.mvs[(p_blk.1 / 2) * 2 + p_blk.0 / 2],
        )
    }

    #[test]
    fn edge_strengths_match_the_per_segment_rule() {
        use crate::mbinfo::MbKind;
        use crate::predict::ChromaMode;
        let mv = |x, y| Mv { x, y };
        let vectors = [mv(0, 0), mv(3, 0), mv(4, 0), mv(0, -4), mv(9, 2)];
        let mut state = 0x2545_f491_u32;
        let mut next = |n: u32| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state % n
        };
        for _ in 0..2000 {
            let mut random_mb = || MbInfo {
                kind: match next(4) {
                    0 => MbKind::Intra16x16,
                    1 => MbKind::Skip,
                    _ => MbKind::Inter,
                },
                cbp_luma: 0,
                cbp_chroma: 0,
                luma_dc_coded: false,
                luma_coded: if next(2) == 0 {
                    0
                } else {
                    next(1 << 16) as u16
                },
                chroma_dc_coded: [false; 2],
                chroma_ac_coded: [0; 2],
                mvs: core::array::from_fn(|_| vectors[next(5) as usize]),
                abs_mvd: [[0; 2]; 4],
                transform_8x8: false,
                chroma_pred_mode: ChromaMode::Dc,
                intra_modes: [2; 16],
                qp: 0,
                ref_idx: 0,
            };
            let (mb, neighbour) = (random_mb(), random_mb());
            for dir in [Direction::Vertical, Direction::Horizontal] {
                for edge in 0..4 {
                    for n in [None, Some(&neighbour)] {
                        let expected: [u8; 4] =
                            core::array::from_fn(|seg| boundary_strength(&mb, n, dir, edge, seg));
                        assert_eq!(edge_strengths(&mb, n, dir, edge), expected);
                    }
                }
            }
        }
    }
}
