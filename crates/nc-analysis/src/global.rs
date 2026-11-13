//! Global (camera) motion: a translation + zoom model fitted to the 64x64
//! motion field (240 points at 720p: plenty for three parameters; fitting
//! the 32x32 field cost 0.03 ms/frame more on the Orin for no measurable
//! change in search quality).
//!
//! Model, in luma samples, with (x, y) measured from the picture centre:
//!
//! ```text
//!   mv_x = tx + zoom * x
//!   mv_y = ty + zoom * y
//! ```
//!
//! This covers pans, tilts and dolly/zoom moves (drones, handheld). The fit
//! is least squares over textured blocks only (flat blocks match
//! anywhere, so their vectors are noise), repeated a few times with outliers
//! (independently moving objects) removed. The result seeds the next
//! frame's hierarchical search; AV1 could later signal it as a global
//! motion model.

use crate::grid::Grid;
use crate::motion::AggregatedField;
use crate::mv::Mv;
use crate::preanalysis::BlockStats;

/// Minimum mean variance (per sample) of a block's area for its vector to
/// be trusted. Provisional.
const MIN_TEXTURE_VARIANCE: f64 = 4.0;
/// Refits with outliers removed.
const ROBUST_ITERATIONS: usize = 3;
/// Points further than this many median residuals from the fit are outliers...
const OUTLIER_MEDIAN_FACTOR: f64 = 2.5;
/// ...but residuals below this (in luma samples) are never outliers.
const OUTLIER_MIN_RESIDUAL: f64 = 2.0;
/// Fewer inliers than this and the model is not trusted.
const MIN_POINTS: usize = 8;

/// A fitted global motion model.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GlobalMotion {
    /// Horizontal translation, luma samples.
    pub tx: f32,
    /// Vertical translation, luma samples.
    pub ty: f32,
    /// Zoom: displacement per sample of distance from the centre.
    pub zoom: f32,
    /// Fraction of the trusted blocks the final fit explains (0 if no model).
    pub inlier_fraction: f32,
    centre_x: f32,
    centre_y: f32,
}

impl GlobalMotion {
    /// No camera motion.
    pub const NONE: GlobalMotion = GlobalMotion {
        tx: 0.0,
        ty: 0.0,
        zoom: 0.0,
        inlier_fraction: 0.0,
        centre_x: 0.0,
        centre_y: 0.0,
    };

    /// The same model with its displacement multiplied by `factor` (the
    /// motion over a different frame distance, assuming constant speed).
    #[must_use]
    pub fn scaled(self, factor: f32) -> GlobalMotion {
        GlobalMotion {
            tx: self.tx * factor,
            ty: self.ty * factor,
            zoom: self.zoom * factor,
            ..self
        }
    }

    /// The model's largest displacement over a `width` x `height` picture,
    /// in luma samples: its length at the farthest corner (translation and
    /// zoom both grow towards the corners).
    #[must_use]
    pub fn peak_displacement(&self, width: usize, height: usize) -> f32 {
        let (w, h) = (width as f32, height as f32);
        [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h)]
            .into_iter()
            .map(|(x, y)| {
                let mx = self.tx + self.zoom * (x - self.centre_x);
                let my = self.ty + self.zoom * (y - self.centre_y);
                mx.hypot(my)
            })
            .fold(0.0, f32::max)
    }

    /// The model's vector at luma position (`x`, `y`), rounded to whole samples.
    #[must_use]
    pub fn mv_at(&self, x: f32, y: f32) -> Mv {
        let mx = self.tx + self.zoom * (x - self.centre_x);
        let my = self.ty + self.zoom * (y - self.centre_y);
        Mv::new(mx.round() as i16, my.round() as i16)
    }
}

/// One observation: block centre relative to the picture centre, and its vector.
#[derive(Debug, Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
    u: f64,
    v: f64,
}

/// Fit the model to `field` (a 16x16 or larger field, vectors in luma
/// samples) using the 8x8 `stats` grid to judge texture. `width` x
/// `height` is the luma size.
#[must_use]
pub fn fit(
    field: &AggregatedField,
    stats: &Grid<BlockStats>,
    width: usize,
    height: usize,
) -> GlobalMotion {
    let (centre_x, centre_y) = (width as f64 / 2.0, height as f64 / 2.0);
    let size = field.size as f64;
    let per = field.size / 8;
    let mut points: Vec<Point> = field
        .blocks
        .positions()
        .filter(|&(c, r)| is_textured(stats, c, r, per))
        .map(|(c, r)| {
            let mv = field.blocks.get(c, r).mv;
            Point {
                x: (c as f64 + 0.5) * size - centre_x,
                y: (r as f64 + 0.5) * size - centre_y,
                u: f64::from(mv.x),
                v: f64::from(mv.y),
            }
        })
        .collect();
    let trusted = points.len();
    if trusted < MIN_POINTS {
        return GlobalMotion::NONE;
    }

    let mut model = solve(&points);
    for _ in 0..ROBUST_ITERATIONS {
        let mut residuals: Vec<f64> = points.iter().map(|p| residual(model, p)).collect();
        let median = median(&mut residuals);
        let limit = OUTLIER_MIN_RESIDUAL.max(OUTLIER_MEDIAN_FACTOR * median);
        points.retain(|p| residual(model, p) <= limit);
        if points.len() < MIN_POINTS {
            return GlobalMotion::NONE;
        }
        model = solve(&points);
    }
    let (tx, ty, zoom) = model;
    GlobalMotion {
        tx: tx as f32,
        ty: ty as f32,
        zoom: zoom as f32,
        inlier_fraction: points.len() as f32 / trusted as f32,
        centre_x: centre_x as f32,
        centre_y: centre_y as f32,
    }
}

/// Whether the 8x8 blocks covered by block (`c`, `r`) (`per` 8x8 blocks
/// per edge) are textured enough on average.
fn is_textured(stats: &Grid<BlockStats>, c: usize, r: usize, per: usize) -> bool {
    // Integer sums: every per-block variance is a multiple of 1/64, so the
    // mean compares exactly as it would in floating point.
    let (mut sum_sq_dev, mut count) = (0u64, 0u64);
    for row in r * per..(r + 1) * per {
        for b in stats.row_run(c * per, row, per) {
            sum_sq_dev += u64::from(b.sum_sq_dev);
            count += 1;
        }
    }
    count > 0 && sum_sq_dev as f64 >= MIN_TEXTURE_VARIANCE * 64.0 * count as f64
}

/// Least-squares (tx, ty, zoom). From the normal equations: with
/// n points, X = sum x, Y = sum y, R = sum (x^2 + y^2),
/// P = sum (x u + y v), U = sum u, V = sum v:
/// zoom = (P - (U X + V Y) / n) / (R - (X^2 + Y^2) / n),
/// tx = (U - zoom X) / n, ty = (V - zoom Y) / n.
fn solve(points: &[Point]) -> (f64, f64, f64) {
    let n = points.len() as f64;
    let (mut sx, mut sy, mut su, mut sv, mut r, mut p) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for q in points {
        sx += q.x;
        sy += q.y;
        su += q.u;
        sv += q.v;
        r += q.x * q.x + q.y * q.y;
        p += q.x * q.u + q.y * q.v;
    }
    let denominator = r - (sx * sx + sy * sy) / n;
    let zoom = if denominator.abs() > f64::EPSILON {
        (p - (su * sx + sv * sy) / n) / denominator
    } else {
        0.0
    };
    ((su - zoom * sx) / n, (sv - zoom * sy) / n, zoom)
}

fn residual((tx, ty, zoom): (f64, f64, f64), p: &Point) -> f64 {
    (tx + zoom * p.x - p.u).abs() + (ty + zoom * p.y - p.v).abs()
}

fn median(values: &mut [f64]) -> f64 {
    let mid = values.len() / 2;
    values.select_nth_unstable_by(mid, f64::total_cmp);
    values[mid]
}

#[cfg(test)]
mod tests {
    use super::GlobalMotion;

    #[test]
    fn peak_displacement_is_largest_at_a_corner() {
        let pan = GlobalMotion {
            tx: 3.0,
            ty: -4.0,
            ..GlobalMotion::NONE
        };
        assert!((pan.peak_displacement(64, 32) - 5.0).abs() < 1e-6);
        // Zoom about the centre of a 200 x 100 picture: corners at 100, 50.
        let zoom = GlobalMotion {
            zoom: 0.01,
            centre_x: 100.0,
            centre_y: 50.0,
            ..GlobalMotion::NONE
        };
        assert!((zoom.peak_displacement(200, 100) - 1.0_f32.hypot(0.5)).abs() < 1e-6);
    }
}
