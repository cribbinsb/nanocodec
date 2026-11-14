//! MS-SSIM of two 8-bit luma planes, exactly as the evaluation scores it
//! (`eval/nceval/msssim.py`, equal to libvmaf's `float_ms_ssim`), so the
//! browser demo (nc-web, analysis builds) and native tools show the same
//! number the project's J score uses.
//!
//! The definition, step by step (libvmaf `feature/ms_ssim.c`,
//! `iqa/ssim_tools.c`, `iqa/convolve.c`, `iqa/decimate.c`, v3.2.1):
//!
//! - five scales; each scale's SSIM terms are taken only where the 11x11
//!   Gaussian window (sigma 1.5, separable taps) lies inside the plane;
//! - variances are clamped at 0, contrast uses `sqrt(var_x * var_y)`, and a
//!   negative covariance on a flat window counts as 0;
//! - between scales both planes pass the 9x9 9/7 biorthogonal low-pass (the
//!   2-D table) with symmetric borders and keep every second sample, giving
//!   `ceil(w / 2) x ceil(h / 2)`;
//! - the score is the product over scales of `|mean c|^beta * |mean s|^gamma`,
//!   times `|mean l|^alpha` at the coarsest scale.
//!
//! Filters run in `f32` and the means are summed in `f64`, as in the
//! evaluation's version; scores agree with it far below its 6 printed decimals.
//!
//! The filter tables and exponents are libvmaf's (BSD+Patent, Copyright
//! 2016-2020 Netflix, Inc.; `THIRD_PARTY.md`), reimplemented here.

use nc_common::PlaneView;

/// 11-tap Gaussian window (sigma 1.5), libvmaf's `g_gaussian_window_h/_v`.
const GAUSSIAN_TAPS: [f32; WINDOW] = [
    0.001_028, 0.007_599, 0.036_001, 0.109_361, 0.213_006, 0.266_012, 0.213_006, 0.109_361,
    0.036_001, 0.007_599, 0.001_028,
];
/// Window side in samples.
const WINDOW: usize = 11;
/// Side of the decimation low-pass table.
const LOW_PASS: usize = 9;
/// The 9x9 9/7 biorthogonal low-pass applied before each decimation,
/// libvmaf's `g_lpf` (its decimation uses this 2-D table, not separable taps).
#[rustfmt::skip]
const LOW_PASS_2D: [[f32; LOW_PASS]; LOW_PASS] = [
    [0.000_714, -0.000_450, -0.002_090, 0.007_132, 0.016_114, 0.007_132, -0.002_090, -0.000_450, 0.000_714],
    [-0.000_450, 0.000_283, 0.001_316, -0.004_490, -0.010_146, -0.004_490, 0.001_316, 0.000_283, -0.000_450],
    [-0.002_090, 0.001_316, 0.006_115, -0.020_867, -0.047_149, -0.020_867, 0.006_115, 0.001_316, -0.002_090],
    [0.007_132, -0.004_490, -0.020_867, 0.071_207, 0.160_885, 0.071_207, -0.020_867, -0.004_490, 0.007_132],
    [0.016_114, -0.010_146, -0.047_149, 0.160_885, 0.363_505, 0.160_885, -0.047_149, -0.010_146, 0.016_114],
    [0.007_132, -0.004_490, -0.020_867, 0.071_207, 0.160_885, 0.071_207, -0.020_867, -0.004_490, 0.007_132],
    [-0.002_090, 0.001_316, 0.006_115, -0.020_867, -0.047_149, -0.020_867, 0.006_115, 0.001_316, -0.002_090],
    [-0.000_450, 0.000_283, 0.001_316, -0.004_490, -0.010_146, -0.004_490, 0.001_316, 0.000_283, -0.000_450],
    [0.000_714, -0.000_450, -0.002_090, 0.007_132, 0.016_114, 0.007_132, -0.002_090, -0.000_450, 0.000_714],
];
/// Per-scale exponents (Wang et al. 2003): luminance only at the coarsest scale.
const ALPHAS: [f64; SCALES] = [0.0, 0.0, 0.0, 0.0, 0.1333];
const BETAS: [f64; SCALES] = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333];
const GAMMAS: [f64; SCALES] = BETAS;
/// Number of scales.
const SCALES: usize = 5;
/// The 8-bit sample range.
const DYNAMIC_RANGE: f32 = 255.0;
const C1: f32 = (0.01 * DYNAMIC_RANGE) * (0.01 * DYNAMIC_RANGE);
const C2: f32 = (0.03 * DYNAMIC_RANGE) * (0.03 * DYNAMIC_RANGE);
const C3: f32 = C2 / 2.0;

/// A plane of `f32` samples, row-major without padding.
struct Plane {
    width: usize,
    height: usize,
    data: Vec<f32>,
}

impl Plane {
    fn from_view(view: &PlaneView<'_>) -> Self {
        let data = view
            .rows()
            .flat_map(|row| row.iter().map(|&s| f32::from(s)))
            .collect();
        Self {
            width: view.width(),
            height: view.height(),
            data,
        }
    }

    fn at(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }
}

/// MS-SSIM of `reference` and `distorted` (luma planes of one size), or
/// `None` when the plane is too small for five scales (the coarsest must
/// still hold the 11x11 window).
///
/// # Panics
///
/// If the planes differ in size.
#[must_use]
pub fn ms_ssim(reference: &PlaneView<'_>, distorted: &PlaneView<'_>) -> Option<f64> {
    assert!(
        (reference.width(), reference.height()) == (distorted.width(), distorted.height()),
        "ms_ssim: planes differ in size"
    );
    if reference.width().min(reference.height()) >> (SCALES - 1) < WINDOW {
        return None;
    }
    let (mut x, mut y) = (Plane::from_view(reference), Plane::from_view(distorted));
    let mut score = 1.0;
    for scale in 0..SCALES {
        let [luminance, contrast, structure] = ssim_terms(&x, &y);
        score *= luminance.abs().powf(ALPHAS[scale])
            * contrast.abs().powf(BETAS[scale])
            * structure.abs().powf(GAMMAS[scale]);
        if scale + 1 < SCALES {
            x = decimate(&x);
            y = decimate(&y);
        }
    }
    Some(score)
}

/// The window sums of one scale: the Gaussian-weighted means of x, y, x², y²
/// and xy at every position where the window fits, as planes of the valid size.
struct Moments {
    len: usize,
    mu_x: Vec<f32>,
    mu_y: Vec<f32>,
    xx: Vec<f32>,
    yy: Vec<f32>,
    xy: Vec<f32>,
}

/// The five window means of `x` and `y`: one horizontal pass computing all
/// five products per row, then one vertical pass, over the valid positions.
fn moments(x: &Plane, y: &Plane) -> Moments {
    let (plane_width, plane_height) = (x.width, x.height);
    let width = plane_width - WINDOW + 1;
    let height = plane_height - WINDOW + 1;
    // Horizontal pass: rows[product][row * width + column].
    let mut rows: [Vec<f32>; 5] = std::array::from_fn(|_| vec![0.0; width * plane_height]);
    let mut products: [Vec<f32>; 5] = std::array::from_fn(|_| vec![0.0; plane_width]);
    for row in 0..plane_height {
        let (xs, ys) = (
            &x.data[row * plane_width..][..plane_width],
            &y.data[row * plane_width..][..plane_width],
        );
        for (i, (&a, &b)) in xs.iter().zip(ys).enumerate() {
            products[0][i] = a;
            products[1][i] = b;
            products[2][i] = a * a;
            products[3][i] = b * b;
            products[4][i] = a * b;
        }
        for (out, product) in rows.iter_mut().zip(&products) {
            let out = &mut out[row * width..][..width];
            for (k, &tap) in GAUSSIAN_TAPS.iter().enumerate() {
                for (o, &v) in out.iter_mut().zip(&product[k..k + width]) {
                    *o += tap * v;
                }
            }
        }
    }
    // Vertical pass.
    let vertical = |rows: &[f32]| {
        let mut out = vec![0.0f32; width * height];
        for row in 0..height {
            let target = &mut out[row * width..][..width];
            for (k, &tap) in GAUSSIAN_TAPS.iter().enumerate() {
                for (o, &v) in target.iter_mut().zip(&rows[(row + k) * width..][..width]) {
                    *o += tap * v;
                }
            }
        }
        out
    };
    let [mu_x, mu_y, xx, yy, xy] = rows.map(|r| vertical(&r));
    Moments {
        len: width * height,
        mu_x,
        mu_y,
        xx,
        yy,
        xy,
    }
}

/// Mean luminance, contrast and structure terms over the valid positions.
fn ssim_terms(x: &Plane, y: &Plane) -> [f64; 3] {
    let m = moments(x, y);
    let mut sums = [0.0f64; 3];
    for i in 0..m.len {
        let (mx, my) = (m.mu_x[i], m.mu_y[i]);
        let var_x = (m.xx[i] - mx * mx).max(0.0);
        let var_y = (m.yy[i] - my * my).max(0.0);
        let sigma_xy = (var_x * var_y).sqrt();
        let mut covariance = m.xy[i] - mx * my;
        if covariance < 0.0 && sigma_xy <= 0.0 {
            covariance = 0.0;
        }
        sums[0] += f64::from((2.0 * mx * my + C1) / (mx * mx + my * my + C1));
        sums[1] += f64::from((2.0 * sigma_xy + C2) / (var_x + var_y + C2));
        sums[2] += f64::from((covariance + C3) / (sigma_xy + C3));
    }
    let count = m.len as f64;
    sums.map(|sum| sum / count)
}

/// The sample at `index` of a plane `size` long with symmetric borders (`-1 - i` before, `2 * size - 1 - i` after).
fn symmetric(index: isize, size: usize) -> usize {
    let size = size as isize;
    let folded = if index < 0 {
        -1 - index
    } else if index >= size {
        2 * size - 1 - index
    } else {
        index
    };
    folded as usize
}

/// The low-pass of `plane` at every even position: `ceil(w / 2) x ceil(h / 2)`.
/// Positions whose 9x9 neighbourhood lies inside the plane read it directly;
/// the others fold their indices at the borders (`symmetric`).
fn decimate(plane: &Plane) -> Plane {
    let (width, height) = (plane.width.div_ceil(2), plane.height.div_ceil(2));
    let half = LOW_PASS / 2;
    let inside = |centre: usize, size: usize| centre >= half && centre + half < size;
    let mut data = Vec::with_capacity(width * height);
    for oy in 0..height {
        let cy = 2 * oy;
        for ox in 0..width {
            let cx = 2 * ox;
            let mut sum = 0.0f32;
            if inside(cx, plane.width) && inside(cy, plane.height) {
                for (ky, taps) in LOW_PASS_2D.iter().enumerate() {
                    let row = &plane.data[(cy + ky - half) * plane.width + cx - half..][..LOW_PASS];
                    sum += taps.iter().zip(row).map(|(t, v)| t * v).sum::<f32>();
                }
            } else {
                for (ky, taps) in LOW_PASS_2D.iter().enumerate() {
                    let sy = symmetric(cy as isize + ky as isize - half as isize, plane.height);
                    for (kx, tap) in taps.iter().enumerate() {
                        sum += tap
                            * plane.at(
                                symmetric(cx as isize + kx as isize - half as isize, plane.width),
                                sy,
                            );
                    }
                }
            }
            data.push(sum);
        }
    }
    Plane {
        width,
        height,
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_common::Plane as SamplePlane;

    fn plane(width: usize, height: usize, sample: impl Fn(usize, usize) -> u8) -> SamplePlane {
        let mut plane = SamplePlane::new(width, height, 0).unwrap();
        let mut view = plane.view_mut();
        for y in 0..height {
            for (x, s) in view.row_mut(y).iter_mut().enumerate() {
                *s = sample(x, y);
            }
        }
        plane
    }

    #[test]
    fn identical_planes_score_one_and_small_planes_none() {
        let a = plane(192, 176, |x, y| ((x * 7 + y * 13) % 251) as u8);
        assert!((ms_ssim(&a.view(), &a.view()).unwrap() - 1.0).abs() < 1e-6);
        let small = plane(160, 160, |_, _| 0);
        assert_eq!(ms_ssim(&small.view(), &small.view()), None);
    }

    #[test]
    fn matches_the_evaluation_on_a_known_pair() {
        // eval/nceval/msssim.py on these planes (python3 -c, 2026-11-14): 0.950_068_352.
        let a = plane(192, 176, |x, y| ((x * 7 + y * 13) % 251) as u8);
        let b = plane(192, 176, |x, y| {
            (((x * 7 + y * 13) % 251) as u8).wrapping_add(((x * y) % 23) as u8)
        });
        let score = ms_ssim(&a.view(), &b.view()).unwrap();
        assert!((score - EVALUATION_SCORE).abs() < 1e-5, "{score}");
    }

    const EVALUATION_SCORE: f64 = 0.950_068_352;

    #[test]
    fn symmetric_borders_repeat_the_edge_sample() {
        assert_eq!(symmetric(-1, 5), 0);
        assert_eq!(symmetric(-2, 5), 1);
        assert_eq!(symmetric(5, 5), 4);
        assert_eq!(symmetric(6, 5), 3);
    }
}
