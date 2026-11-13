//! Directional prediction of any block shape with the intra edge filter and
//! upsampling (AV1 §7.11.2.4; libaom `build_directional_and_filter_intra_
//! predictors` and `av1_dr_prediction_z{1,2,3}_c`).

use nc_common::PlaneViewMut;

use super::edge::{self, LINE_LEN, LINE_ORIGIN};
use super::{BlockEdges, EdgeFilterCtx};
use crate::intra_prediction::IntraEdges;
use crate::intra_prediction::directional::{DERIVATIVE, interpolate_run};

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

    /// A full-resolution run: `out[k]` interpolates samples `base + k` and
    /// `base + k + 1` at `position`'s weight while `base + k < max_base`,
    /// and is sample `max_base` from there on (as [`Self::interpolate`]
    /// and the zones' clamp, sample by sample).
    fn run(&self, base: i32, max_base: i32, position: i32, out: &mut [u8]) {
        debug_assert_eq!(self.upsample, 0);
        let inside = usize::try_from(max_base.saturating_sub(base))
            .unwrap_or(0)
            .min(out.len());
        let weight = ((position & POSITION_MASK) >> 1) as usize;
        let (interpolated, clamped) = out.split_at_mut(inside);
        if inside > 0 {
            let first = (LINE_ORIGIN as i32 + base) as usize;
            interpolate_run(&self.samples[first..=first + inside], weight, interpolated);
        }
        if !clamped.is_empty() {
            clamped.fill(self.at(max_base) as u8);
        }
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

/// A square block's edges after the edge preparation of §7.11.2.4 for
/// `angle` (filtered corner and edges), in the square kernels' layout, or
/// `None` when it upsamples an edge (the square kernels interpolate at the
/// edges' own resolution only). The square kernels then predict exactly as
/// [`predict`]: every zone reads `AboveRow` / `LeftCol` up to index
/// `2n - 1` and repeats that sample beyond it, and filtering leaves the
/// corner the same in both lines.
pub(super) fn prepared_square(
    edges: &BlockEdges,
    angle: u16,
    ctx: EdgeFilterCtx,
) -> Option<IntraEdges> {
    let n = edges.width;
    debug_assert_eq!(n, edges.height);
    let mut above = line(&edges.square.above, edges.square.corner, 2 * n);
    let mut left = line(&edges.square.left, edges.square.corner, 2 * n);
    prepare_edges(edges, angle, ctx.smooth(), &mut above, &mut left);
    if above.upsample != 0 || left.upsample != 0 {
        return None;
    }
    let corner = above.samples[LINE_ORIGIN - 1];
    debug_assert_eq!(corner, left.samples[LINE_ORIGIN - 1]);
    let mut square = IntraEdges {
        above: edges.square.above,
        left: edges.square.left,
        corner,
        dc: edges.square.dc,
        size: n,
    };
    square.above[..2 * n].copy_from_slice(&above.samples[LINE_ORIGIN..][..2 * n]);
    square.left[..2 * n].copy_from_slice(&left.samples[LINE_ORIGIN..][..2 * n]);
    Some(square)
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
/// sample repeat it. At full resolution each row is one run at one weight
/// (samples `base..` up to `max_base`, then that sample repeated).
fn zone1(above: &Line, dx: i32, w: usize, h: usize, dst: &mut PlaneViewMut<'_>) {
    if above.upsample != 0 {
        return zone1_samples(above, dx, w, h, dst);
    }
    let max_base = ((w + h) as i32) - 1;
    for r in 0..h {
        let position = (r as i32 + 1) * dx;
        above.run(
            position >> POSITION_BITS,
            max_base,
            position,
            &mut dst.row_mut(r)[..w],
        );
    }
}

/// [`zone1`] sample by sample, at any resolution.
fn zone1_samples(above: &Line, dx: i32, w: usize, h: usize, dst: &mut PlaneViewMut<'_>) {
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
/// lands on it, otherwise from the left column. Along a row the above
/// projection moves a whole sample per column, so at full resolution the
/// columns reaching the above row (`x >> 6 >= -1`, from column
/// `ceil(((r + 1) dx - 64) / 64)` on) are one run at one weight.
fn zone2(above: &Line, left: &Line, dx: i32, dy: i32, dst: &mut PlaneViewMut<'_>) {
    if above.upsample != 0 {
        return zone2_samples(above, left, dx, dy, dst);
    }
    let (w, h) = (dst.width(), dst.height());
    let unit = 1 << POSITION_BITS;
    for r in 0..h {
        let reach = (r as i32 + 1) * dx - unit;
        let split = if reach <= 0 {
            0
        } else {
            ((reach + unit - 1) >> POSITION_BITS).min(w as i32) as usize
        };
        let row = &mut dst.row_mut(r)[..w];
        for (c, pixel) in row[..split].iter_mut().enumerate() {
            *pixel = zone2_sample(above, left, dx, dy, (r, c));
        }
        if split < w {
            let x = ((split as i32) << POSITION_BITS) - (r as i32 + 1) * dx;
            above.run(x >> POSITION_BITS, i32::MAX, x, &mut row[split..]);
        }
    }
}

/// [`zone2`] sample by sample, at any resolution.
fn zone2_samples(above: &Line, left: &Line, dx: i32, dy: i32, dst: &mut PlaneViewMut<'_>) {
    let (w, h) = (dst.width(), dst.height());
    for r in 0..h {
        for (c, pixel) in dst.row_mut(r)[..w].iter_mut().enumerate() {
            *pixel = zone2_sample(above, left, dx, dy, (r, c));
        }
    }
}

/// Zone-2 sample `(r, c)`: from the above row where its projection lands
/// on it (or on the corner), else from the left column.
fn zone2_sample(above: &Line, left: &Line, dx: i32, dy: i32, (r, c): (usize, usize)) -> u8 {
    let min_base_above = -(1 << above.upsample);
    let x = ((c as i32) << POSITION_BITS) - (r as i32 + 1) * dx;
    let base_x = x >> (POSITION_BITS - above.upsample);
    if base_x >= min_base_above {
        above.interpolate(base_x, x)
    } else {
        let y = ((r as i32) << POSITION_BITS) - (c as i32 + 1) * dy;
        left.interpolate(y >> (POSITION_BITS - left.upsample), y)
    }
}

/// Angles above 180: from the left column only; positions past its last
/// sample repeat it. At full resolution each column is one run (zone 1 on
/// the left column), written down the column.
fn zone3(left: &Line, dy: i32, w: usize, h: usize, dst: &mut PlaneViewMut<'_>) {
    if left.upsample != 0 {
        return zone3_samples(left, dy, w, h, dst);
    }
    let max_base = ((w + h) as i32) - 1;
    let stride = dst.stride();
    let data = dst.data_mut();
    let mut column = [0u8; super::MAX_SIDE];
    for c in 0..w {
        let position = (c as i32 + 1) * dy;
        left.run(
            position >> POSITION_BITS,
            max_base,
            position,
            &mut column[..h],
        );
        for (r, &sample) in column[..h].iter().enumerate() {
            data[r * stride + c] = sample;
        }
    }
}

/// [`zone3`] sample by sample, at any resolution.
fn zone3_samples(left: &Line, dy: i32, w: usize, h: usize, dst: &mut PlaneViewMut<'_>) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intra_prediction::IntraMode;
    use crate::intra_prediction::av1_block::{EdgeAvail, IntraKind};
    use nc_common::Plane;

    /// The full-resolution runs of every zone equal the zones sample by
    /// sample, for every block shape and reachable angle, on random lines.
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
                        zone1(&above, dx, w, h, &mut runs.view_mut());
                        zone1_samples(&above, dx, w, h, &mut samples.view_mut());
                    }
                    a if a < HORIZONTAL => {
                        let dx = DERIVATIVE[usize::from(HORIZONTAL - a)];
                        let dy = DERIVATIVE[usize::from(a - VERTICAL)];
                        zone2(&above, &left, dx, dy, &mut runs.view_mut());
                        zone2_samples(&above, &left, dx, dy, &mut samples.view_mut());
                    }
                    a => {
                        let dy = DERIVATIVE[usize::from(MAX_ANGLE - a)];
                        zone3(&left, dy, w, h, &mut runs.view_mut());
                        zone3_samples(&left, dy, w, h, &mut samples.view_mut());
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
                let edges = BlockEdges::read(&plane, MAX, MAX, side, side, avail);
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
