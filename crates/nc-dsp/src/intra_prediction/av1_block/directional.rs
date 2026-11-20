//! Directional prediction of any block shape with the intra edge filter and
//! upsampling (AV1 §7.11.2.4; libaom `build_directional_and_filter_intra_
//! predictors` and `av1_dr_prediction_z{1,2,3}_c`).
//!
//! After the edges are prepared, a block goes whole to a [`ZoneBlock`]
//! kernel where the ISA has one and the block fits (up to 16x16, as dav1d's
//! `ipred_z{1,2,3}` predict small blocks); otherwise upsampled lines are
//! predicted sample by sample ([`ZoneBlock::predict`]) and full-resolution
//! ones by runs along rows or columns (`zone1`, `zone2`, `zone3`).

use nc_common::{PlaneViewMut, Sample};

use super::edge::{self, LINE_LEN, LINE_ORIGIN};
use super::{BlockEdges, EdgeFilterCtx};
use crate::intra_prediction::IntraEdges;
use crate::intra_prediction::directional::DERIVATIVE;
#[cfg(test)]
use crate::intra_prediction::directional::interpolate_run;

/// Angle of pure vertical prediction (degrees).
pub(super) const VERTICAL: u16 = 90;
/// Angle of pure horizontal prediction (degrees).
pub(super) const HORIZONTAL: u16 = 180;
/// Directional angles lie strictly between zero and this.
const MAX_ANGLE: u16 = 270;
/// The corner is filtered only for blocks with at least this `w + h`.
const CORNER_FILTER_MIN_WH: usize = 24;
/// Fractional bits of an edge position.
const POSITION_BITS: u32 = 6;
/// Interpolation weights sum to `1 << WEIGHT_BITS`.
const WEIGHT_BITS: u32 = 5;
/// Mask of the fractional position, before dropping one bit to a weight.
const POSITION_MASK: i32 = (1 << POSITION_BITS) - 1;

/// One edge line, upsampled or not.
struct Line<T: Sample> {
    samples: [T; LINE_LEN],
    /// log2 of the resolution factor: 1 if upsampled.
    upsample: u32,
}

impl<T: Sample> Line<T> {
    /// Edge sample at (possibly negative) index `i`.
    fn at(&self, i: i32) -> i32 {
        self.samples[(LINE_ORIGIN as i32 + i) as usize].to_i32()
    }

    /// A full-resolution run: `out[k]` interpolates samples `base + k` and
    /// `base + k + 1` at `position`'s weight while `base + k < max_base`,
    /// and is sample `max_base` from there on (as [`Self::interpolate`]
    /// and the zones' clamp, sample by sample).
    fn run(
        &self,
        base: i32,
        max_base: i32,
        position: i32,
        out: &mut [T],
        run_fn: &impl Fn(&[T], usize, &mut [T]),
    ) {
        debug_assert_eq!(self.upsample, 0);
        let inside = usize::try_from(max_base.saturating_sub(base))
            .unwrap_or(0)
            .min(out.len());
        let weight = ((position & POSITION_MASK) >> 1) as usize;
        let (interpolated, clamped) = out.split_at_mut(inside);
        if inside > 0 {
            let first = (LINE_ORIGIN as i32 + base) as usize;
            run_fn(&self.samples[first..=first + inside], weight, interpolated);
        }
        if !clamped.is_empty() {
            clamped.fill(T::from_i32(self.at(max_base)));
        }
    }

    /// The line from edge index `-margin` on, for a [`ZoneBlock`].
    fn from(&self, margin: usize) -> &[T] {
        &self.samples[LINE_ORIGIN - margin..]
    }

    /// Repeat sample `max_base` once past it, so that an interpolation at
    /// `max_base` (any weight) yields that sample: the whole-block kernels
    /// then clamp the index instead of selecting per sample.
    fn pad_after(&mut self, max_base: i32) {
        let last = LINE_ORIGIN + max_base as usize;
        self.samples[last + 1] = self.samples[last];
    }
}

/// Samples a whole-block kernel holds of each edge line (one 32-byte
/// table: two `pshufb` halves).
pub(crate) const ZONE_TABLE: usize = 32;
/// Widest and tallest block a whole-block kernel predicts (sixteen 16-bit
/// lanes: one row of 16, two of 8 or four of 4 per vector).
pub(crate) const ZONE_MAX_SIDE: usize = 16;
/// Zone-2 lines start this many samples before edge index 0: the corner
/// at -1 and, upsampled, -2 (`minBaseX`, `minBaseY` of §7.11.2.4).
pub(crate) const ZONE2_MARGIN: usize = 2;

/// Where a directional prediction reads its edges (§7.11.2.4 steps 7–9),
/// with the derivative and resolution of each line it reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Zone {
    /// Angles below 90: the above row; positions from `max_base` on
    /// repeat that sample.
    Above {
        /// `dx` of the angle.
        dx: i32,
        /// log2 of the line's resolution factor (`upsampleAbove`).
        upsample: u32,
        /// `maxBaseX`.
        max_base: i32,
    },
    /// Angles between 90 and 180: the above row where the projection lands
    /// on it, else the left column.
    Both {
        /// `dx` of the angle.
        dx: i32,
        /// `dy` of the angle.
        dy: i32,
        /// `upsampleAbove`.
        upsample_above: u32,
        /// `upsampleLeft`.
        upsample_left: u32,
    },
    /// Angles above 180: the left column (zone 1 transposed).
    Left {
        /// `dy` of the angle.
        dy: i32,
        /// `upsampleLeft`.
        upsample: u32,
        /// `maxBaseY`.
        max_base: i32,
    },
}

/// One block's directional prediction from its prepared (filtered,
/// upsampled) edge lines, for the whole-block kernels: `above[k]` is
/// `AboveRow[k - margin]` and `left[k]` is `LeftCol[k - margin]`, with
/// `margin` [`ZONE2_MARGIN`] in zone 2 and 0 in zones 1 and 3 (which read
/// no index below 0). A zone 1 or 3 line repeats its `max_base` sample once
/// past it ([`Line::pad_after`]).
#[derive(Debug, Clone, Copy)]
pub(crate) struct ZoneBlock<'a, T: Sample> {
    /// Geometry of the prediction.
    pub zone: Zone,
    /// `AboveRow` from `-margin` (unused in zone 3).
    pub above: &'a [T],
    /// `LeftCol` from `-margin` (unused in zone 1).
    pub left: &'a [T],
}

impl<T: Sample> ZoneBlock<'_, T> {
    /// Whether a whole-block kernel predicts this zone on a `w`x`h` block:
    /// every sample it reads, and the repeated one, lies in the first
    /// [`ZONE_TABLE`] of its line, and a row fits a vector. Zone 2 reads at
    /// most `2w - 1` above (upsampled, `w <= 12`) or `w` samples, and as
    /// many left.
    pub(crate) fn fits(&self, (w, h): (usize, usize)) -> bool {
        let lines = match self.zone {
            Zone::Above { max_base, .. } | Zone::Left { max_base, .. } => {
                (max_base as usize) + 1 < ZONE_TABLE
            }
            Zone::Both { .. } => true,
        };
        lines && w <= ZONE_MAX_SIDE && h <= ZONE_MAX_SIDE
    }

    /// The prediction sample by sample: the scalar reference of the
    /// whole-block kernels, at any size and resolution.
    pub(crate) fn predict(&self, dst: &mut PlaneViewMut<'_, T>) {
        let (w, h) = (dst.width(), dst.height());
        for r in 0..h {
            for (c, pixel) in dst.row_mut(r)[..w].iter_mut().enumerate() {
                *pixel = self.sample(r as i32, c as i32);
            }
        }
    }

    /// Sample `(r, c)` (§7.11.2.4 steps 7–9).
    fn sample(&self, r: i32, c: i32) -> T {
        match self.zone {
            Zone::Above {
                dx,
                upsample,
                max_base,
            } => along(self.above, (r + 1) * dx, c, upsample, max_base),
            Zone::Left {
                dy,
                upsample,
                max_base,
            } => along(self.left, (c + 1) * dy, r, upsample, max_base),
            Zone::Both {
                dx,
                dy,
                upsample_above,
                upsample_left,
            } => {
                let margin = ZONE2_MARGIN as i32;
                let x = (c << POSITION_BITS) - (r + 1) * dx;
                let base_x = x >> (POSITION_BITS - upsample_above);
                if base_x >= -(1 << upsample_above) {
                    interpolate(self.above, base_x + margin, x, upsample_above)
                } else {
                    let y = (r << POSITION_BITS) - (c + 1) * dy;
                    let base_y = y >> (POSITION_BITS - upsample_left);
                    interpolate(self.left, base_y + margin, y, upsample_left)
                }
            }
        }
    }
}

/// Zone 1 or 3 sample `step` along the line at `position`: interpolated
/// below `max_base`, that sample from there on.
fn along<T: Sample>(line: &[T], position: i32, step: i32, upsample: u32, max_base: i32) -> T {
    let base = (position >> (POSITION_BITS - upsample)) + (step << upsample);
    if base < max_base {
        interpolate(line, base, position, upsample)
    } else {
        line[max_base as usize]
    }
}

/// The interpolation between `line[index]` and `line[index + 1]` at the
/// 1/64 position `position` (on a line of resolution `1 << upsample`).
fn interpolate<T: Sample>(line: &[T], index: i32, position: i32, upsample: u32) -> T {
    let shift = ((position << upsample) & POSITION_MASK) >> 1;
    let index = index as usize;
    let sum =
        line[index].to_i32() * ((1 << WEIGHT_BITS) - shift) + line[index + 1].to_i32() * shift;
    T::from_i32((sum + (1 << (WEIGHT_BITS - 1))) >> WEIGHT_BITS)
}

/// A whole-block directional kernel ([`ZoneBlock::predict`]'s contract,
/// for blocks that [`ZoneBlock::fits`]).
pub(crate) type ZoneKernel<'k, T> = &'k dyn Fn(&ZoneBlock<'_, T>, &mut PlaneViewMut<'_, T>);

/// Predict `dst` from `edges` at `angle` degrees, filtering and upsampling
/// the edges first when `edge_filter` is set.
#[cfg(test)]
pub(super) fn predict<T: Sample>(
    edges: &BlockEdges<T>,
    angle: u16,
    edge_filter: Option<EdgeFilterCtx>,
    dst: &mut PlaneViewMut<'_, T>,
) {
    predict_with_run(edges, angle, edge_filter, dst, interpolate_run, None);
}

/// Predict `dst` from `edges` at `angle` degrees with caller-supplied
/// kernels: `zones` for the blocks it [fits](ZoneBlock::fits), else
/// [`ZoneBlock::predict`] for upsampled lines and runs of `run_fn` for
/// full-resolution ones.
pub(crate) fn predict_with_run<T: Sample>(
    edges: &BlockEdges<T>,
    angle: u16,
    edge_filter: Option<EdgeFilterCtx>,
    dst: &mut PlaneViewMut<'_, T>,
    run_fn: impl Fn(&[T], usize, &mut [T]),
    zones: Option<ZoneKernel<'_, T>>,
) {
    assert!(angle > 0 && angle < MAX_ANGLE, "directional angle {angle}");
    let (w, h) = (edges.width, edges.height);
    if angle == VERTICAL {
        for r in 0..h {
            dst.row_mut(r)[..w].copy_from_slice(&edges.square.above[..w]);
        }
        return;
    }
    if angle == HORIZONTAL {
        for r in 0..h {
            dst.row_mut(r)[..w].fill(edges.square.left[r]);
        }
        return;
    }
    let span = w + h;
    if angle < VERTICAL {
        let mut above = line(&edges.square.above, edges.square.corner, span);
        if let Some(ctx) = edge_filter {
            let smooth = ctx.smooth();
            let delta = i32::from(angle) - i32::from(VERTICAL);
            if edges.avail.have_above {
                let strength = edge::filter_strength(span, delta, smooth);
                edge::filter(&mut above.samples, edges.above_inside + h + 1, strength);
            }
            if edge::use_upsample(span, delta, smooth) {
                edge::upsample(&mut above.samples, span, edges.depth);
                above.upsample = 1;
            }
        }
        let dx = DERIVATIVE[usize::from(angle)];
        let max_base = ((span as i32) - 1) << above.upsample;
        above.pad_after(max_base);
        let block = ZoneBlock {
            zone: Zone::Above {
                dx,
                upsample: above.upsample,
                max_base,
            },
            above: above.from(0),
            left: &[],
        };
        if !whole_block(&block, above.upsample != 0, zones, dst) {
            zone1(&above, dx, w, h, dst, &run_fn);
        }
        return;
    }
    if angle > HORIZONTAL {
        let mut left = line(&edges.square.left, edges.square.corner, span);
        if let Some(ctx) = edge_filter {
            let smooth = ctx.smooth();
            let delta = i32::from(angle) - i32::from(HORIZONTAL);
            if edges.avail.have_left {
                let strength = edge::filter_strength(span, delta, smooth);
                edge::filter(&mut left.samples, edges.left_inside + w + 1, strength);
            }
            if edge::use_upsample(span, delta, smooth) {
                edge::upsample(&mut left.samples, span, edges.depth);
                left.upsample = 1;
            }
        }
        let dy = DERIVATIVE[usize::from(MAX_ANGLE - angle)];
        let max_base = ((span as i32) - 1) << left.upsample;
        left.pad_after(max_base);
        let block = ZoneBlock {
            zone: Zone::Left {
                dy,
                upsample: left.upsample,
                max_base,
            },
            above: &[],
            left: left.from(0),
        };
        if !whole_block(&block, left.upsample != 0, zones, dst) {
            zone3(&left, dy, w, h, dst, &run_fn);
        }
        return;
    }
    let mut above = line(&edges.square.above, edges.square.corner, span);
    let mut left = line(&edges.square.left, edges.square.corner, span);
    if let Some(ctx) = edge_filter {
        prepare_edges(edges, angle, ctx.smooth(), &mut above, &mut left);
    }
    let (dx, dy) = (
        DERIVATIVE[usize::from(HORIZONTAL - angle)],
        DERIVATIVE[usize::from(angle - VERTICAL)],
    );
    let block = ZoneBlock {
        zone: Zone::Both {
            dx,
            dy,
            upsample_above: above.upsample,
            upsample_left: left.upsample,
        },
        above: above.from(ZONE2_MARGIN),
        left: left.from(ZONE2_MARGIN),
    };
    let upsampled = above.upsample != 0 || left.upsample != 0;
    if !whole_block(&block, upsampled, zones, dst) {
        zone2(&above, &left, dx, dy, dst, &run_fn);
    }
}

/// Predict `block` whole: with `kernel` where it fits, else sample by
/// sample if a line is `upsampled` (the runs serve full-resolution lines
/// only). False if neither applies and the caller is to run the zone.
fn whole_block<T: Sample>(
    block: &ZoneBlock<'_, T>,
    upsampled: bool,
    kernel: Option<ZoneKernel<'_, T>>,
    dst: &mut PlaneViewMut<'_, T>,
) -> bool {
    match kernel {
        Some(kernel) if block.fits((dst.width(), dst.height())) => kernel(block, dst),
        _ if upsampled => block.predict(dst),
        _ => return false,
    }
    true
}

/// Outcome of preparing a square block's edges for the square directional kernels.
pub(super) enum PreparedSquare<T: Sample> {
    /// Edge filtering and upsampling are both inactive; borrow `edges.square` directly.
    Unchanged,
    /// Filtered without upsampling into a square `IntraEdges`.
    Filtered(IntraEdges<T>),
    /// At least one active edge upsamples (`w + h <= 16`); use `predict_with_run`.
    Upsampled,
}

/// Prepare a square block's edges (§7.11.2.4) for `angle` without constructing
/// temporary `Line` buffers when filtering or upsampling is inactive.
pub(super) fn prepared_square<T: Sample>(
    edges: &BlockEdges<T>,
    angle: u16,
    ctx: EdgeFilterCtx,
) -> PreparedSquare<T> {
    let n = edges.width;
    debug_assert_eq!(n, edges.height);
    if angle == VERTICAL || angle == HORIZONTAL {
        return PreparedSquare::Unchanged;
    }
    let block_wh = 2 * n;
    let smooth = ctx.smooth();
    let (ang, vert, horz) = (i32::from(angle), i32::from(VERTICAL), i32::from(HORIZONTAL));
    let (need_above, need_left) = (angle < HORIZONTAL, angle > VERTICAL);
    if (need_above && edge::use_upsample(block_wh, ang - vert, smooth))
        || (need_left && edge::use_upsample(block_wh, ang - horz, smooth))
    {
        return PreparedSquare::Upsampled;
    }
    let corner_f = need_above && need_left && block_wh >= CORNER_FILTER_MIN_WH;
    let above_s = if need_above && edges.avail.have_above {
        edge::filter_strength(block_wh, ang - vert, smooth)
    } else {
        0
    };
    let left_s = if need_left && edges.avail.have_left {
        edge::filter_strength(block_wh, ang - horz, smooth)
    } else {
        0
    };
    if !corner_f && above_s == 0 && left_s == 0 {
        return PreparedSquare::Unchanged;
    }
    let corner = if corner_f {
        edge::filtered_corner(
            edges.square.left[0],
            edges.square.corner,
            edges.square.above[0],
        )
    } else {
        edges.square.corner
    };
    let mut square = IntraEdges {
        above: [T::default(); super::EDGE_LEN],
        left: [T::default(); super::EDGE_LEN],
        corner,
        dc: edges.square.dc,
        size: n,
    };
    if need_above {
        square.above[..2 * n].copy_from_slice(&edges.square.above[..2 * n]);
        if above_s != 0 {
            let num_px = edges.above_inside + if angle < VERTICAL { n } else { 0 } + 1;
            edge::filter_edge(
                corner,
                &edges.square.above,
                &mut square.above,
                num_px,
                above_s,
            );
        }
    }
    if need_left {
        square.left[..2 * n].copy_from_slice(&edges.square.left[..2 * n]);
        if left_s != 0 {
            let num_px = edges.left_inside + if angle > HORIZONTAL { n } else { 0 } + 1;
            edge::filter_edge(corner, &edges.square.left, &mut square.left, num_px, left_s);
        }
    }
    PreparedSquare::Filtered(square)
}

/// An edge line: the corner at index -1, then `len` edge samples.
fn line<T: Sample>(edge: &[T], corner: T, len: usize) -> Line<T> {
    let mut samples = [T::default(); LINE_LEN];
    samples[LINE_ORIGIN - 1] = corner;
    samples[LINE_ORIGIN..LINE_ORIGIN + len].copy_from_slice(&edge[..len]);
    Line {
        samples,
        upsample: 0,
    }
}

/// The edge preparation of §7.11.2.4 when `enable_intra_edge_filter` is 1:
/// corner filter (zone 2 only), edge filters of the available edges, then
/// upsampling where the angle and block size call for it.
fn prepare_edges<T: Sample>(
    edges: &BlockEdges<T>,
    angle: u16,
    smooth: bool,
    above: &mut Line<T>,
    left: &mut Line<T>,
) {
    let (w, h) = (edges.width, edges.height);
    let block_wh = w + h;
    let angle = i32::from(angle);
    let (vertical, horizontal) = (i32::from(VERTICAL), i32::from(HORIZONTAL));
    let needs_right = angle < vertical;
    let needs_bottom = angle > horizontal;
    if angle != vertical && angle != horizontal {
        if angle > vertical && angle < horizontal && block_wh >= CORNER_FILTER_MIN_WH {
            edge::filter_corner(&mut above.samples, &mut left.samples);
        }
        if edges.avail.have_above {
            let strength = edge::filter_strength(block_wh, angle - vertical, smooth);
            let num_px = edges.above_inside + if needs_right { h } else { 0 } + 1;
            edge::filter(&mut above.samples, num_px, strength);
        }
        if edges.avail.have_left {
            let strength = edge::filter_strength(block_wh, angle - horizontal, smooth);
            let num_px = edges.left_inside + if needs_bottom { w } else { 0 } + 1;
            edge::filter(&mut left.samples, num_px, strength);
        }
    }
    if edge::use_upsample(block_wh, angle - vertical, smooth) {
        edge::upsample(
            &mut above.samples,
            w + if needs_right { h } else { 0 },
            edges.depth,
        );
        above.upsample = 1;
    }
    if edge::use_upsample(block_wh, angle - horizontal, smooth) {
        edge::upsample(
            &mut left.samples,
            h + if needs_bottom { w } else { 0 },
            edges.depth,
        );
        left.upsample = 1;
    }
}

/// Angles below 90: from the above row only; positions past its last
/// sample repeat it. At full resolution each row is one run at one weight
/// (samples `base..` up to `max_base`, then that sample repeated).
fn zone1<T: Sample>(
    above: &Line<T>,
    dx: i32,
    w: usize,
    h: usize,
    dst: &mut PlaneViewMut<'_, T>,
    run_fn: &impl Fn(&[T], usize, &mut [T]),
) {
    let max_base = ((w + h) as i32) - 1;
    for r in 0..h {
        let position = (r as i32 + 1) * dx;
        above.run(
            position >> POSITION_BITS,
            max_base,
            position,
            &mut dst.row_mut(r)[..w],
            run_fn,
        );
    }
}

/// Angles between 90 and 180: from the above row where the projection
/// lands on it, otherwise from the left column. Along a row the above
/// projection moves a whole sample per column, so at full resolution the
/// columns reaching the above row (`x >> 6 >= -1`, from column
/// `ceil(((r + 1) dx - 64) / 64)` on) are one run at one weight.
fn zone2<T: Sample>(
    above: &Line<T>,
    left: &Line<T>,
    dx: i32,
    dy: i32,
    dst: &mut PlaneViewMut<'_, T>,
    run_fn: &impl Fn(&[T], usize, &mut [T]),
) {
    debug_assert!(above.upsample == 0 && left.upsample == 0);
    let (w, h) = (dst.width(), dst.height());
    let unit = 1 << POSITION_BITS;
    // Row `r`'s first `splits[r]` samples project left of the corner and
    // come from the left column, the rest from the above row; the split
    // grows down the block.
    let mut splits = [0usize; super::MAX_SIDE];
    for (r, split) in splits[..h].iter_mut().enumerate() {
        let reach = (r as i32 + 1) * dx - unit;
        if reach > 0 {
            *split = ((reach + unit - 1) >> POSITION_BITS).min(w as i32) as usize;
        }
    }
    for (r, &split) in splits[..h].iter().enumerate() {
        if split < w {
            let x = ((split as i32) << POSITION_BITS) - (r as i32 + 1) * dx;
            let row = &mut dst.row_mut(r)[split..w];
            above.run(x >> POSITION_BITS, i32::MAX, x, row, run_fn);
        }
    }
    // The left part of column `c` (rows from the first whose split passes
    // it) is one run along the left column at one weight, as in zone 3,
    // written down the column; sample by sample it was a branch and a
    // gather per sample.
    let stride = dst.stride();
    let data = dst.data_mut();
    let mut column = [T::default(); super::MAX_SIDE];
    for c in 0..splits[h - 1] {
        let first = splits[..h].partition_point(|&split| split <= c);
        let y = ((first as i32) << POSITION_BITS) - (c as i32 + 1) * dy;
        let column = &mut column[..h - first];
        left.run(y >> POSITION_BITS, i32::MAX, y, column, run_fn);
        for (r, &sample) in column.iter().enumerate() {
            data[(first + r) * stride + c] = sample;
        }
    }
}

/// Angles above 180: from the left column only; positions past its last
/// sample repeat it. At full resolution each column is one run (zone 1 on
/// the left column), written down the column.
fn zone3<T: Sample>(
    left: &Line<T>,
    dy: i32,
    w: usize,
    h: usize,
    dst: &mut PlaneViewMut<'_, T>,
    run_fn: &impl Fn(&[T], usize, &mut [T]),
) {
    let max_base = ((w + h) as i32) - 1;
    let stride = dst.stride();
    let data = dst.data_mut();
    let mut column = [T::default(); super::MAX_SIDE];
    for c in 0..w {
        let position = (c as i32 + 1) * dy;
        left.run(
            position >> POSITION_BITS,
            max_base,
            position,
            &mut column[..h],
            run_fn,
        );
        for (r, &sample) in column[..h].iter().enumerate() {
            data[r * stride + c] = sample;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intra_prediction::IntraMode;
    use crate::intra_prediction::av1_block::{EdgeAvail, IntraKind};
    use nc_common::BitDepth;
    use nc_common::Plane;

    /// The scalar whole-block prediction of `zone` from full lines.
    fn reference<'a>(zone: Zone, above: &'a Line<u8>, left: &'a Line<u8>) -> ZoneBlock<'a, u8> {
        let margin = if matches!(zone, Zone::Both { .. }) {
            ZONE2_MARGIN
        } else {
            0
        };
        ZoneBlock {
            zone,
            above: above.from(margin),
            left: left.from(margin),
        }
    }

    /// The full-resolution runs of every zone equal the zones sample by
    /// sample ([`ZoneBlock::predict`]), for every block shape and reachable angle, on random lines.
    #[test]
    fn zone_runs_match_the_zones_sample_by_sample() {
        let shapes = [
            (4, 4),
            (8, 8),
            (16, 16),
            (32, 32),
            (64, 64),
            (4, 8),
            (8, 4),
            (8, 16),
            (16, 8),
            (16, 32),
            (32, 16),
            (32, 64),
            (64, 32),
            (4, 16),
            (16, 4),
            (8, 32),
            (32, 8),
            (16, 64),
            (64, 16),
        ];
        let mut state = 0x9e37_79b9_u32;
        let mut random_line = || {
            let mut samples = [0u8; LINE_LEN];
            for sample in &mut samples {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *sample = (state >> 24) as u8;
            }
            Line {
                samples,
                upsample: 0,
            }
        };
        for (w, h) in shapes {
            let (above, left) = (random_line(), random_line());
            for angle in (3..270).filter(|a| a % 90 != 0) {
                let mut runs = Plane::new(w, h, 0).unwrap();
                let mut samples = Plane::new(w, h, 0).unwrap();
                match angle {
                    a if a < VERTICAL => {
                        let dx = DERIVATIVE[usize::from(a)];
                        zone1(&above, dx, w, h, &mut runs.view_mut(), &interpolate_run);
                        let max_base = (w + h) as i32 - 1;
                        let zone = Zone::Above {
                            dx,
                            upsample: 0,
                            max_base,
                        };
                        reference(zone, &above, &left).predict(&mut samples.view_mut());
                    }
                    a if a < HORIZONTAL => {
                        let dx = DERIVATIVE[usize::from(HORIZONTAL - a)];
                        let dy = DERIVATIVE[usize::from(a - VERTICAL)];
                        zone2(
                            &above,
                            &left,
                            dx,
                            dy,
                            &mut runs.view_mut(),
                            &interpolate_run,
                        );
                        let zone = Zone::Both {
                            dx,
                            dy,
                            upsample_above: 0,
                            upsample_left: 0,
                        };
                        reference(zone, &above, &left).predict(&mut samples.view_mut());
                    }
                    a => {
                        let dy = DERIVATIVE[usize::from(MAX_ANGLE - a)];
                        zone3(&left, dy, w, h, &mut runs.view_mut(), &interpolate_run);
                        let max_base = (w + h) as i32 - 1;
                        let zone = Zone::Left {
                            dy,
                            upsample: 0,
                            max_base,
                        };
                        reference(zone, &above, &left).predict(&mut samples.view_mut());
                    }
                }
                for y in 0..h {
                    assert_eq!(
                        runs.view().row(y)[..w],
                        samples.view().row(y)[..w],
                        "{w}x{h} angle {angle}"
                    );
                }
            }
        }
    }

    /// Square blocks with the edge filter go through [`prepared_square`]
    /// and the square kernels unless an edge is upsampled; either way the
    /// prediction equals the per-sample zones of [`predict`], at every
    /// reachable angle, both filter types and every availability.
    #[test]
    fn prepared_squares_predict_as_the_zones() {
        const MAX: usize = 64;
        let picture = 3 * MAX;
        let mut plane = Plane::new(picture, picture, 0).unwrap();
        let mut view = plane.view_mut();
        let mut state = 0x1234_5678_u32;
        for y in 0..picture {
            for sample in &mut view.row_mut(y)[..picture] {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                // Smooth gradients with noise, and some flat runs, so both
                // filter strengths and upsampling occur.
                *sample = ((y * 3 + (state >> 28) as usize) & 0xff) as u8;
            }
        }
        for side in [4, 8, 16, 32, 64] {
            for avail_bits in 0..16_u8 {
                let avail = EdgeAvail {
                    have_above: avail_bits & 1 != 0,
                    have_left: avail_bits & 2 != 0,
                    have_above_right: avail_bits & 4 != 0,
                    have_below_left: avail_bits & 8 != 0,
                };
                let edges =
                    BlockEdges::read(&plane, (MAX, MAX), (side, side), avail, BitDepth::EIGHT);
                for smooth in [false, true] {
                    let ctx = EdgeFilterCtx {
                        smooth_above: smooth,
                        smooth_left: false,
                    };
                    for mode in IntraMode::ALL {
                        let Some(nominal) = mode.nominal_angle() else {
                            continue;
                        };
                        for delta in -3_i16..=3 {
                            let angle = (nominal as i16 + 3 * delta) as u16;
                            let mut zones = Plane::new(side, side, 0).unwrap();
                            predict(&edges, angle, Some(ctx), &mut zones.view_mut());
                            let mut routed = Plane::new(side, side, 0).unwrap();
                            let kind = IntraKind::Directional {
                                angle,
                                edge_filter: Some(ctx),
                            };
                            edges.predict(kind, &mut routed.view_mut());
                            for y in 0..side {
                                assert_eq!(
                                    routed.view().row(y)[..side],
                                    zones.view().row(y)[..side],
                                    "side {side} angle {angle} smooth {smooth} {avail:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
