//! The deblocking filter of MBAFF frames (§8.7): macroblocks in address
//! order, pair by pair, each filtered in its own rows (a field macroblock in
//! its field's rows of the pair). Where a frame and a field macroblock
//! meet (`mixedModeEdgeFlag`, §8.7.2.1) the edge is filtered as the
//! samples lie: the left edge line by line, each line with the bS and QP of
//! the left macroblock it reaches (Table 6-4); the top edge in field rows
//! (a frame macroblock under a field pair twice, once per field, §8.7.1).
//! Such edges take bS 1 where neither side is intra or coded, and
//! horizontal intra ones bS 3.
//!
//! Rare enough to stay scalar where it differs: the mixed left edges go
//! through the per-line reference filters of nc-dsp.

use nc_dsp::{H264EdgeParams, Pixel};

use super::super::Planes;
use super::super::mbstate::MbState;
use super::super::structure::{Parity, field_rows};
use super::{
    BS_COEFFICIENTS, BS_INTRA, BS_INTRA_MB_EDGE, CHROMA_EDGES, DEBLOCK_IDC_NO_SLICE_EDGES,
    DeblockContext, LUMA_EDGES, SliceFilter, Strengths, Thresholds, column_bits,
    direction_strengths, edge_strengths, filter_qp, is_uniform_inter, mb_edges, row_bits,
};

/// bS of a mixed edge between inter macroblocks without coefficients.
const BS_MIXED: u8 = 1;
/// Luma lines of a macroblock edge, chroma lines of a 4:2:0 one (MBAFF
/// frames are 4:2:0 or 4:0:0 here: `slice::check_mbaff`).
const LUMA_LINES: usize = 16;
const CHROMA_LINES: usize = 8;

/// The two macroblocks of a pair (top, bottom).
type Pair<'a> = (&'a MbState, &'a MbState);

/// What lies above a macroblock's top edge.
#[derive(Clone, Copy)]
enum Above<'a> {
    /// A macroblock coded alike, in the current macroblock's rows.
    Alike(&'a MbState),
    /// A frame macroblock above a field macroblock: its rows of the
    /// field's parity.
    FrameOverField(&'a MbState),
    /// A field pair above a frame macroblock: each field macroblock above
    /// the current one's rows of its parity.
    FieldsOverFrame(Pair<'a>),
}

/// Filter every macroblock of an MBAFF frame (`planes` the frame).
pub(in super::super) fn filter_mbaff<T: Pixel>(
    planes: &mut Planes<'_, T>,
    ctx: &DeblockContext<'_>,
) {
    if ctx.slices.iter().all(|s| !s.control.filters()) {
        return;
    }
    let bi = ctx.slices.iter().any(|s| s.b_slice);
    for addr in 0..ctx.mbs.len() {
        match (bi, ctx.mbs[addr].field) {
            (true, true) => filter_mb::<true, true, T>(planes, ctx, addr),
            (true, false) => filter_mb::<true, false, T>(planes, ctx, addr),
            (false, true) => filter_mb::<false, true, T>(planes, ctx, addr),
            (false, false) => filter_mb::<false, false, T>(planes, ctx, addr),
        }
    }
}

/// Filter macroblock `addr`, a field macroblock (`FIELD`) or not, of a
/// frame with (`BI`) or without B slices.
fn filter_mb<const BI: bool, const FIELD: bool, T: Pixel>(
    planes: &mut Planes<'_, T>,
    ctx: &DeblockContext<'_>,
    addr: usize,
) {
    let mbs = ctx.mbs;
    let mb = &mbs[addr];
    let filter = &ctx.slices[(mb.slice - ctx.first_slice) as usize];
    if !filter.control.filters() {
        return;
    }
    let width = ctx.width_mbs;
    let (pair, bottom) = (addr / 2, addr % 2 == 1);
    let (px, py) = (pair % width, pair / width);
    // §8.7: a macroblock edge is filtered unless it is a picture edge, or a
    // slice edge with disable_deblocking_filter_idc 2.
    let allowed = |p: Pair<'_>| {
        filter.control.disable_idc != DEBLOCK_IDC_NO_SLICE_EDGES || p.0.slice == mb.slice
    };
    let pair_at = |index: usize| (&mbs[2 * index], &mbs[2 * index + 1]);
    let left = (px > 0).then(|| pair_at(pair - 1)).filter(|&p| allowed(p));
    let above = match (FIELD, bottom) {
        // The bottom frame macroblock's top neighbour is its pair's top.
        (false, true) => Some(Above::Alike(&mbs[addr - 1])),
        _ => (py > 0)
            .then(|| pair_at(pair - width))
            .filter(|&p| allowed(p))
            .map(|(top, below)| match (FIELD, top.field) {
                (false, false) => Above::Alike(below),
                (false, true) => Above::FieldsOverFrame((top, below)),
                (true, true) => Above::Alike(if bottom { below } else { top }),
                (true, false) => Above::FrameOverField(below),
            }),
    };
    let left_alike = left
        .filter(|p| p.0.field == FIELD)
        .map(|(top, below)| if bottom { below } else { top });
    // Internal edges and the edges to neighbours coded alike, in the
    // macroblock's own rows.
    let mut bs: Strengths = [[[0; 4]; 4]; 2];
    let q_uniform = is_uniform_inter(mb);
    {
        let [vertical, horizontal] = &mut bs;
        direction_strengths::<0, BI, FIELD, T>(mb, None, q_uniform, false, vertical);
        direction_strengths::<1, BI, FIELD, T>(mb, None, q_uniform, false, horizontal);
    }
    if let Some(p) = left_alike {
        bs[0][0] = edge_strengths::<0, BI, FIELD, T>(p, mb, 0);
    }
    let above_alike = match above {
        Some(Above::Alike(p)) => Some(p),
        _ => None,
    };
    if let Some(p) = above_alike {
        bs[1][0] = edge_strengths::<1, BI, FIELD, T>(p, mb, 0);
    }
    let qp = filter_qp(mb);
    let neighbour_qp = [left_alike, above_alike].map(|n| n.map_or(0, filter_qp));
    let parity = if bottom { Parity::Bottom } else { Parity::Top };
    let (luma_at, chroma_at) = if FIELD {
        ((16 * px, 16 * py), (8 * px, 8 * py))
    } else {
        let row = 2 * py + usize::from(bottom);
        ((16 * px, 16 * row), (8 * px, 8 * row))
    };
    let rows_of = FIELD.then_some(parity);
    let edges = Edges {
        ctx,
        filter,
        mb,
        qp,
        neighbour_qp,
    };
    // Vertical edges, left to right: a mixed left edge line by line.
    {
        let mut own = own_rows(planes, rows_of);
        if let Some(left) = left.filter(|p| p.0.field != FIELD) {
            edges.mixed_left::<FIELD, T>(&mut own, (luma_at, chroma_at), left, bottom);
        }
        edges.filter_direction(&mut own, (luma_at, chroma_at), &bs, 0);
    }
    // Horizontal edges, top to bottom: a frame macroblock under a field
    // pair filters its top edge once in each field's rows first.
    if let Some(Above::FieldsOverFrame((top, below))) = above {
        for (field, p) in [(Parity::Top, top), (Parity::Bottom, below)] {
            let mut rows = field_rows(planes, field);
            edges.mixed_top(&mut rows, ((16 * px, 16 * py), (8 * px, 8 * py)), p);
        }
    }
    let mut own = own_rows(planes, rows_of);
    if let Some(Above::FrameOverField(p)) = above {
        edges.mixed_top(&mut own, (luma_at, chroma_at), p);
    }
    edges.filter_direction(&mut own, (luma_at, chroma_at), &bs, 1);
}

/// The rows a macroblock is filtered in: the frame, or one field's rows.
fn own_rows<'p, T: Pixel>(planes: &'p mut Planes<'_, T>, field: Option<Parity>) -> Planes<'p, T> {
    match field {
        Some(parity) => field_rows(planes, parity),
        None => planes.each_mut().map(|plane| {
            let (width, height) = (plane.width(), plane.height());
            plane
                .sub_view_mut(0, 0, width, height)
                .expect("the whole view")
        }),
    }
}

/// One macroblock's filtering context.
struct Edges<'a> {
    ctx: &'a DeblockContext<'a>,
    filter: &'a SliceFilter,
    mb: &'a MbState,
    qp: i32,
    /// `QP_Y` of the left and the upper neighbour coded alike (0 where
    /// there is none: their edges then have no strength).
    neighbour_qp: [i32; 2],
}

impl Edges<'_> {
    /// The edges of direction `dir` (0 vertical) with strengths `bs` in
    /// the macroblock's rows (`at`: its luma and chroma positions).
    fn filter_direction<T: Pixel>(
        &self,
        planes: &mut Planes<'_, T>,
        (luma_at, chroma_at): ((usize, usize), (usize, usize)),
        bs: &Strengths,
        dir: usize,
    ) {
        let mut direction: Strengths = [[[0; 4]; 4]; 2];
        direction[dir] = bs[dir];
        let filter = self.filter;
        let luma = mb_edges::<T>(&direction, LUMA_EDGES, |edge_0, d| {
            filter.thresholds(
                if edge_0 {
                    self.neighbour_qp[d]
                } else {
                    self.qp
                },
                self.qp,
            )
        });
        let depths = self.ctx.depths.of::<T>();
        let [y, cb, cr] = planes;
        self.ctx
            .dsp
            .at(depths.luma)
            .h264_deblock_mb_luma(y, luma_at, &luma);
        if self.ctx.chroma.is_mono() {
            return;
        }
        for (comp, plane) in [cb, cr].into_iter().enumerate() {
            let q = filter.chroma_qp(self.qp, comp);
            let p = self.neighbour_qp.map(|n| filter.chroma_qp(n, comp));
            let chroma = mb_edges::<T>(&direction, CHROMA_EDGES, |edge_0, d| {
                filter.thresholds(if edge_0 { p[d] } else { q }, q)
            });
            self.ctx
                .dsp
                .at(depths.chroma)
                .h264_deblock_mb_chroma_plane(plane, chroma_at, &chroma);
        }
    }

    /// The left edge between this macroblock and a pair coded the other
    /// way (`left`), line by line (Table 6-4 gives each line's macroblock
    /// and row; bS 4 by intra, 2 by coefficients, else 1).
    fn mixed_left<const FIELD: bool, T: Pixel>(
        &self,
        planes: &mut Planes<'_, T>,
        (luma_at, chroma_at): ((usize, usize), (usize, usize)),
        (top, below): Pair<'_>,
        bottom: bool,
    ) {
        let mb = self.mb;
        let filter = self.filter;
        // Line `k` of the current macroblock: the left macroblock and its
        // row (a frame macroblock's rows alternate between the fields; a
        // field macroblock's upper half lies beside the top frame one).
        let left_of = |k: usize, lines: usize| -> (&MbState, usize) {
            if FIELD {
                let frame_row = 2 * k + usize::from(bottom);
                let in_top = frame_row < lines;
                (if in_top { top } else { below }, frame_row % lines)
            } else {
                let row = k / 2 + lines / 2 * usize::from(bottom);
                (if k.is_multiple_of(2) { top } else { below }, row)
            }
        };
        let luma_bs = |k: usize| -> (u8, &MbState) {
            let (p, row) = left_of(k, LUMA_LINES);
            let bs = if p.is_intra() || mb.is_intra() {
                BS_INTRA_MB_EDGE
            } else if (column_bits(mb.nonzero, 0) >> (k / 4)) & 1 != 0
                || (column_bits(p.nonzero, 3) >> (row / 4)) & 1 != 0
            {
                BS_COEFFICIENTS
            } else {
                BS_MIXED
            };
            (bs, p)
        };
        let depths = self.ctx.depths.of::<T>();
        let [y, cb, cr] = planes;
        let luma_kernels = self.ctx.dsp.at(depths.luma);
        for k in 0..LUMA_LINES {
            let (bs, p) = luma_bs(k);
            let t = filter.thresholds(filter_qp(p), self.qp);
            let at = (luma_at.0, luma_at.1 + k);
            luma_kernels.h264_deblock_luma_line(y, at, true, &line_params(t, bs));
        }
        if self.ctx.chroma.is_mono() {
            return;
        }
        let chroma_kernels = self.ctx.dsp.at(depths.chroma);
        for (comp, plane) in [cb, cr].into_iter().enumerate() {
            for k in 0..CHROMA_LINES {
                // The luma line whose bS a chroma line takes (JM's
                // mapping: a field macroblock beside frame ones every
                // other line, else pairs of lines of each 4).
                let luma_line = if FIELD { 2 * k } else { 4 * (k / 2) + k % 2 };
                let (bs, _) = luma_bs(luma_line);
                let (p, _) = left_of(k, CHROMA_LINES);
                let (qp_p, qp_q) = (
                    filter.chroma_qp(filter_qp(p), comp),
                    filter.chroma_qp(self.qp, comp),
                );
                let t = filter.thresholds(qp_p, qp_q);
                let at = (chroma_at.0, chroma_at.1 + k);
                chroma_kernels.h264_deblock_chroma_line(plane, at, true, &line_params(t, bs));
            }
        }
    }

    /// A top edge in field rows between this macroblock and `p`, coded the
    /// other way (`planes` the rows of the field, `at` the current
    /// macroblock's first row in them): bS 3 by intra, 2 by coefficients,
    /// else 1.
    fn mixed_top<T: Pixel>(
        &self,
        planes: &mut Planes<'_, T>,
        at: ((usize, usize), (usize, usize)),
        p: &MbState,
    ) {
        let mb = self.mb;
        let coded = row_bits(mb.nonzero, 0) | row_bits(p.nonzero, 3);
        let segment: [u8; 4] = core::array::from_fn(|seg| {
            if p.is_intra() || mb.is_intra() {
                BS_INTRA
            } else if (coded >> seg) & 1 != 0 {
                BS_COEFFICIENTS
            } else {
                BS_MIXED
            }
        });
        let mut bs: Strengths = [[[0; 4]; 4]; 2];
        bs[1][0] = segment;
        let edges = Edges {
            neighbour_qp: [0, filter_qp(p)],
            ..*self
        };
        edges.filter_direction(planes, at, &bs, 1);
    }
}

/// One line's parameters: bS `bs` with thresholds `t`.
fn line_params(t: Thresholds, bs: u8) -> H264EdgeParams {
    H264EdgeParams {
        bs: [bs; 4],
        tc0: t.tc0s([bs; 4]),
        alpha: t.alpha,
        beta: t.beta,
    }
}
