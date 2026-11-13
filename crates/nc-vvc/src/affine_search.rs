//! Encoder search of affine motion (§8.5.5) for an inter CU the engine
//! chose with one vector: the sub-block merge candidates ranked by luma
//! SATD plus λ-weighted bits, and an affine AMVP model refined from a start
//! model by Gauss-Newton steps on the prediction's gradients (the optical
//! flow fit of VTM's affine motion estimation, written anew). The caller
//! confirms the winner against the vector with full luma RD.
use crate::affine::Model;
use crate::motion::Mv16;
use nc_common::{Picture, PlaneIndex, PlaneView};
use nc_dsp::{BlockSize, Dsp, Filter};
use nc_encoder::Block;

/// Sixteenths per luma sample.
const SIXTEENTHS: f64 = 16.;
/// Control points are coded in quarter samples (no affine AMVR): their
/// sixteenth-sample values are multiples of this.
const QUARTER: i32 = 4;
/// Largest SATD block: the CU is measured in tiles of at most this side.
const SATD_TILE: usize = 32;
/// The Gauss-Newton fit samples every this many rows and columns.
const FIT_STEP: usize = 2;
/// The search predicts in blocks of this side (`Frame::predict_coarse`).
const SEARCH_BLOCK: usize = 8;
/// Search blocks of the largest (64x64) CU.
const MAX_SEARCH_BLOCKS: usize = 64;

/// A CU's luma, its reference and the kernels to predict and measure it.
pub(crate) struct Frame<'a> {
    pub dsp: &'a Dsp,
    pub source: &'a Picture,
    pub reference: &'a Picture,
    pub cu: Block,
}

impl Frame<'_> {
    /// The affine luma prediction of the CU at `model`.
    pub(crate) fn predict(&self, model: &Model, out: &mut [u8]) {
        let field = model.field(self.cu.width(), self.cu.height());
        let reference = self.reference.plane(PlaneIndex::Y);
        let mut out = crate::inter::packed(out, (self.cu.width(), self.cu.height()));
        crate::affine::predict(
            self.dsp,
            reference,
            PlaneIndex::Y,
            &self.cu,
            &field,
            &mut out,
        );
    }

    /// Luma SATD of `prediction` against the source CU.
    pub(crate) fn satd(&self, prediction: &[u8]) -> u32 {
        let (w, h) = (self.cu.width(), self.cu.height());
        let (tw, th) = (w.min(SATD_TILE), h.min(SATD_TILE));
        let size = BlockSize::new(tw, th).expect("CU tiles are kernel sizes");
        let source = self.source.plane(PlaneIndex::Y);
        let mut total = 0;
        for y in (0..h).step_by(th) {
            for x in (0..w).step_by(tw) {
                let original = source
                    .region((self.cu.x + x) as isize, (self.cu.y + y) as isize, tw, th)
                    .expect("CUs lie inside the coded picture");
                let predicted = PlaneView::new(&prediction[y * w + x..], w, tw, th)
                    .expect("prediction covers the CU");
                total += self.dsp.satd(size, &original, &predicted);
            }
        }
        total
    }

    /// The search's approximation of `model`'s luma prediction: on CUs at
    /// least [`SEARCH_BLOCK`] a side, one vector per 8x8 block (the model at
    /// its centre) through the 8-tap filter instead of four 4x4 sub-blocks
    /// through the 6-tap one, a quarter of the kernel calls. Ranking
    /// candidates and the Gauss-Newton fit tolerate the difference; the
    /// winner is confirmed with the exact prediction.
    pub(crate) fn predict_coarse(&self, model: &Model, out: &mut [u8]) {
        let (w, h) = (self.cu.width(), self.cu.height());
        if w < SEARCH_BLOCK || h < SEARCH_BLOCK {
            self.predict(model, out);
            return;
        }
        let field = model.field(w, h);
        let per_row = w / SEARCH_BLOCK;
        let mut vectors = [Mv16::ZERO; MAX_SEARCH_BLOCKS];
        let vectors = &mut vectors[..per_row * (h / SEARCH_BLOCK)];
        for (i, mv) in vectors.iter_mut().enumerate() {
            *mv = field.at_block(SEARCH_BLOCK, i % per_row, i / per_row);
        }
        // VVC's luma table at quarter phases is HEVC's: one filter serves
        // every vector of the batch.
        crate::inter::predict_sub_blocks(
            self.dsp,
            Filter::VvcLuma,
            self.reference.plane(PlaneIndex::Y),
            (self.cu.x, self.cu.y),
            w,
            SEARCH_BLOCK,
            vectors,
            &mut crate::inter::packed(out, (w, h)),
        );
    }

    /// SATD of the search's prediction of `model` (`scratch` holds it
    /// afterwards).
    pub(crate) fn model_satd(&self, model: &Model, scratch: &mut [u8]) -> u32 {
        self.predict_coarse(model, scratch);
        self.satd(scratch)
    }
}

/// Whether `model` moves its sub-blocks differently (else a vector codes
/// it as well, with the better 8-tap filter).
pub(crate) fn is_affine(model: &Model) -> bool {
    model.cp[1] != model.cp[0] || (model.six && model.cp[2] != model.cp[0])
}

/// Refine `start` by up to `iterations` Gauss-Newton steps of its
/// parameters (4, or 6 when `start.six`), each rounded to quarter-sample
/// control points, while each lowers `cost` (SATD and the coding bits it
/// prices); returns the last model that did and its cost. `known` is the start's
/// search prediction and SATD when the caller has them (a translation's is
/// its vector's).
pub(crate) fn refine(
    frame: &Frame<'_>,
    start: Model,
    known: Option<(&[u8], u32)>,
    iterations: usize,
    cost: impl Fn(&Model, u32) -> f32,
) -> (Model, f32) {
    let mut model = round_to_quarter(start);
    let (mut prediction, satd) = match known {
        Some((prediction, satd)) if model == start => (prediction.to_vec(), satd),
        _ => {
            let mut prediction = vec![0u8; frame.cu.area()];
            let satd = frame.model_satd(&model, &mut prediction);
            (prediction, satd)
        }
    };
    let mut best = (model, cost(&model, satd));
    for _ in 0..iterations {
        let Some(next) = step(frame, &model, &prediction) else {
            break;
        };
        if next == model {
            break;
        }
        model = next;
        let satd = frame.model_satd(&model, &mut prediction);
        let candidate = cost(&model, satd);
        // A step that does not pay ends the fit: the linearisation has
        // stopped describing the residual.
        if candidate >= best.1 {
            break;
        }
        best = (model, candidate);
    }
    best
}

/// `model` with its control points on the quarter-sample grid.
pub(crate) fn round_to_quarter(model: Model) -> Model {
    Model {
        cp: model.cp.map(|v| v.round(QUARTER.ilog2())),
        ..model
    }
}

/// One Gauss-Newton step from `model`, whose prediction is `prediction`:
/// the parameter change that best explains the residual by the
/// prediction's gradients, least squares (`None` if degenerate). The fit
/// samples every [`FIT_STEP`]th row and column inside the CU's border
/// (central differences): a quarter of the work for nearly the same
/// solution, which is rounded to quarter samples anyway.
fn step(frame: &Frame<'_>, model: &Model, prediction: &[u8]) -> Option<Model> {
    let (w, h) = (frame.cu.width(), frame.cu.height());
    let delta = if model.six {
        fit::<6>(frame, prediction, |gx, gy, px, py| {
            [gx, gy, gx * px, gy * px, gx * py, gy * py]
        })?
    } else {
        let four = fit::<4>(frame, prediction, |gx, gy, px, py| {
            [gx, gy, gx * px + gy * py, gy * px - gx * py]
        })?;
        [four[0], four[1], four[2], four[3], 0., 0.]
    };
    Some(apply(model, &delta, (w, h)))
}

/// The least-squares parameter change of an `N`-parameter model: `terms`
/// maps a sample's gradients `(gx, gy)` and position to how each parameter
/// moves its prediction.
fn fit<const N: usize>(
    frame: &Frame<'_>,
    prediction: &[u8],
    terms: impl Fn(f64, f64, f64, f64) -> [f64; N],
) -> Option<[f64; N]> {
    let cu = &frame.cu;
    let (w, h) = (cu.width(), cu.height());
    let mut normal = [[0f64; N]; N];
    let mut rhs = [0f64; N];
    let source = frame.source.plane(PlaneIndex::Y).view();
    let line = |y: usize| &prediction[y * w..(y + 1) * w];
    for y in (1..h - 1).step_by(FIT_STEP) {
        let original = &source.row(cu.y + y)[cu.x..cu.x + w];
        let (above, here, below) = (line(y - 1), line(y), line(y + 1));
        let py = y as f64 + 0.5;
        for x in (1..w - 1).step_by(FIT_STEP) {
            let gx = f64::from(i16::from(here[x + 1]) - i16::from(here[x - 1])) / 2.;
            let gy = f64::from(i16::from(below[x]) - i16::from(above[x])) / 2.;
            let t = terms(gx, gy, x as f64 + 0.5, py);
            let residual = f64::from(i16::from(original[x]) - i16::from(here[x]));
            for i in 0..N {
                rhs[i] += t[i] * residual;
                for j in i..N {
                    normal[i][j] += t[i] * t[j];
                }
            }
        }
    }
    // Only the upper triangle was summed: mirror it.
    let upper = normal;
    for (i, row) in normal.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate().take(i) {
            *value = upper[j][i];
        }
    }
    solve(&mut normal, &mut rhs)
}

/// `model` moved by parameter change `delta` (samples, and samples per
/// sample), its control points rounded to quarter samples.
fn apply(model: &Model, delta: &[f64; 6], (width, height): (usize, usize)) -> Model {
    let (width, height) = (width as f64, height as f64);
    let [v0, v1, v2] = model
        .cp
        .map(|v| (f64::from(v.x) / SIXTEENTHS, f64::from(v.y) / SIXTEENTHS));
    let origin = (v0.0 + delta[0], v0.1 + delta[1]);
    // The vector change per sample along x (4-parameter: rotation and
    // zoom), and for 6 parameters along y.
    let along_x = (
        (v1.0 - v0.0) / width + delta[2],
        (v1.1 - v0.1) / width + delta[3],
    );
    let quantise = |(x, y): (f64, f64)| {
        let quarter = |v: f64| (v * SIXTEENTHS / f64::from(QUARTER)).round() as i32 * QUARTER;
        Mv16 {
            x: quarter(x),
            y: quarter(y),
        }
        .clip()
    };
    // A 4-parameter model leaves its third control point zero.
    let cp2 = if model.six {
        let along_y = (
            (v2.0 - v0.0) / height + delta[4],
            (v2.1 - v0.1) / height + delta[5],
        );
        quantise((origin.0 + along_y.0 * height, origin.1 + along_y.1 * height))
    } else {
        Mv16::ZERO
    };
    Model {
        cp: [
            quantise(origin),
            quantise((origin.0 + along_x.0 * width, origin.1 + along_x.1 * width)),
            cp2,
        ],
        six: model.six,
    }
}

/// Solve the `N`x`N` system `normal · x = rhs` by Gaussian elimination
/// with partial pivoting; `None` when it is singular.
fn solve<const N: usize>(normal: &mut [[f64; N]; N], rhs: &mut [f64; N]) -> Option<[f64; N]> {
    const SINGULAR: f64 = 1e-9;
    for col in 0..N {
        let pivot =
            (col..N).max_by(|&a, &b| normal[a][col].abs().total_cmp(&normal[b][col].abs()))?;
        if normal[pivot][col].abs() < SINGULAR {
            return None;
        }
        normal.swap(col, pivot);
        rhs.swap(col, pivot);
        let pivot_row = normal[col];
        for row in col + 1..N {
            let factor = normal[row][col] / pivot_row[col];
            for (target, &p) in normal[row][col..].iter_mut().zip(&pivot_row[col..]) {
                *target -= factor * p;
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    let mut x = [0f64; N];
    for row in (0..N).rev() {
        let tail: f64 = (row + 1..N).map(|k| normal[row][k] * x[k]).sum();
        x[row] = (rhs[row] - tail) / normal[row][row];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_recovers_a_known_solution() {
        let mut normal = [[4., 1., 0.], [1., 3., 1.], [0., 1., 2.]];
        let x = [1., -2., 0.5];
        let mut rhs = [0f64; 3];
        for i in 0..3 {
            rhs[i] = (0..3).map(|j| normal[i][j] * x[j]).sum();
        }
        let solved = solve(&mut normal, &mut rhs).expect("regular system");
        for i in 0..3 {
            assert!((solved[i] - x[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn a_zero_step_keeps_the_model() {
        let model = Model {
            cp: [Mv16 { x: 8, y: -4 }, Mv16 { x: 24, y: 4 }, Mv16::ZERO],
            six: false,
        };
        assert_eq!(apply(&model, &[0.; 6], (16, 16)), model);
    }
}
