//! Directional prediction of any block shape with the intra edge filter and
//! upsampling (AV1 §7.11.2.4; libaom `build_directional_and_filter_intra_
//! predictors` and `av1_dr_prediction_z{1,2,3}_c`).

use nc_common::PlaneViewMut;

use super::edge::{self, LINE_LEN, LINE_ORIGIN};
use super::{BlockEdges, EdgeFilterCtx};
use crate::intra_prediction::directional::DERIVATIVE;

/// Angle of pure vertical prediction (degrees).
const VERTICAL: u16 = 90;
/// Angle of pure horizontal prediction (degrees).
const HORIZONTAL: u16 = 180;
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
struct Line {
    samples: [u8; LINE_LEN],
    /// log2 of the resolution factor: 1 if upsampled.
    upsample: u32,
}

impl Line {
    /// Edge sample at (possibly negative) index `i`.
    fn at(&self, i: i32) -> i32 {
        i32::from(self.samples[(LINE_ORIGIN as i32 + i) as usize])
    }

    /// The interpolation between samples `base` and `base + 1` at the 1/64
    /// position `position` (already scaled to the line's resolution).
    fn interpolate(&self, base: i32, position: i32) -> u8 {
        let shift = ((position << self.upsample) & POSITION_MASK) >> 1;
        let sum = self.at(base) * ((1 << WEIGHT_BITS) - shift) + self.at(base + 1) * shift;
        ((sum + (1 << (WEIGHT_BITS - 1))) >> WEIGHT_BITS) as u8
    }
}

/// Predict `dst` from `edges` at `angle` degrees, filtering and upsampling
/// the edges first when `edge_filter` is set.
pub(super) fn predict(
    edges: &BlockEdges,
    angle: u16,
    edge_filter: Option<EdgeFilterCtx>,
    dst: &mut PlaneViewMut<'_>,
) {
    assert!(angle > 0 && angle < MAX_ANGLE, "directional angle {angle}");
    let (w, h) = (edges.width, edges.height);
    let mut above = line(&edges.square.above, edges.square.corner, w + h);
    let mut left = line(&edges.square.left, edges.square.corner, w + h);
    if let Some(ctx) = edge_filter {
        prepare_edges(edges, angle, ctx.smooth(), &mut above, &mut left);
    }
    match angle {
        VERTICAL => (0..h).for_each(|r| {
            dst.row_mut(r)[..w].copy_from_slice(&above.samples[LINE_ORIGIN..LINE_ORIGIN + w]);
        }),
        HORIZONTAL => (0..h).for_each(|r| dst.row_mut(r)[..w].fill(left.samples[LINE_ORIGIN + r])),
        a if a < VERTICAL => zone1(&above, DERIVATIVE[usize::from(a)], w, h, dst),
        a if a < HORIZONTAL => zone2(
            &above,
            &left,
            DERIVATIVE[usize::from(HORIZONTAL - a)],
            DERIVATIVE[usize::from(a - VERTICAL)],
            dst,
        ),
        a => zone3(&left, DERIVATIVE[usize::from(MAX_ANGLE - a)], w, h, dst),
    }
}

/// An edge line: the corner at index -1, then `len` edge samples.
fn line(edge: &[u8], corner: u8, len: usize) -> Line {
    let mut samples = [0; LINE_LEN];
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
fn prepare_edges(edges: &BlockEdges, angle: u16, smooth: bool, above: &mut Line, left: &mut Line) {
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
        edge::upsample(&mut above.samples, w + if needs_right { h } else { 0 });
        above.upsample = 1;
    }
    if edge::use_upsample(block_wh, angle - horizontal, smooth) {
        edge::upsample(&mut left.samples, h + if needs_bottom { w } else { 0 });
        left.upsample = 1;
    }
}

/// Angles below 90: from the above row only; positions past its last
/// sample repeat it.
fn zone1(above: &Line, dx: i32, w: usize, h: usize, dst: &mut PlaneViewMut<'_>) {
    let max_base = (((w + h) as i32) - 1) << above.upsample;
    let frac_bits = POSITION_BITS - above.upsample;
    for r in 0..h {
        let position = (r as i32 + 1) * dx;
        let row = &mut dst.row_mut(r)[..w];
        for (c, pixel) in row.iter_mut().enumerate() {
            let base = (position >> frac_bits) + ((c as i32) << above.upsample);
            *pixel = if base < max_base {
                above.interpolate(base, position)
            } else {
                above.at(max_base) as u8
            };
        }
    }
}

/// Angles between 90 and 180: from the above row where the projection
/// lands on it, otherwise from the left column.
fn zone2(above: &Line, left: &Line, dx: i32, dy: i32, dst: &mut PlaneViewMut<'_>) {
    let (w, h) = (dst.width(), dst.height());
    let min_base_above = -(1 << above.upsample);
    for r in 0..h {
        let row = &mut dst.row_mut(r)[..w];
        for (c, pixel) in row.iter_mut().enumerate() {
            let x = ((c as i32) << POSITION_BITS) - (r as i32 + 1) * dx;
            let base_x = x >> (POSITION_BITS - above.upsample);
            *pixel = if base_x >= min_base_above {
                above.interpolate(base_x, x)
            } else {
                let y = ((r as i32) << POSITION_BITS) - (c as i32 + 1) * dy;
                left.interpolate(y >> (POSITION_BITS - left.upsample), y)
            };
        }
    }
}

/// Angles above 180: from the left column only; positions past its last
/// sample repeat it.
fn zone3(left: &Line, dy: i32, w: usize, h: usize, dst: &mut PlaneViewMut<'_>) {
    let max_base = (((w + h) as i32) - 1) << left.upsample;
    let frac_bits = POSITION_BITS - left.upsample;
    for c in 0..w {
        let position = (c as i32 + 1) * dy;
        for r in 0..h {
            let base = (position >> frac_bits) + ((r as i32) << left.upsample);
            dst.row_mut(r)[c] = if base < max_base {
                left.interpolate(base, position)
            } else {
                left.at(max_base) as u8
            };
        }
    }
}
