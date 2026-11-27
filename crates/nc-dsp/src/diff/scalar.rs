//! Scalar reference residual and variance kernels.

use super::{DiffFns, Prediction, ResidualRows, Variance, VarianceEnergy, bilinear_taps};

pub(crate) fn diff_fns() -> DiffFns {
    DiffFns {
        residual: [
            residual::<4, false>,
            residual::<8, false>,
            residual::<16, false>,
            residual::<32, false>,
            residual::<64, false>,
        ],
        residual_sad: [
            residual::<4, true>,
            residual::<8, true>,
            residual::<16, true>,
            residual::<32, true>,
            residual::<64, true>,
        ],
        reconstruct: [
            reconstruct::<4>,
            reconstruct::<8>,
            reconstruct::<16>,
            reconstruct::<32>,
            reconstruct::<64>,
        ],
        variance: [
            variance::<4>,
            variance::<8>,
            variance::<16>,
            variance::<32>,
            variance::<64>,
        ],
        variance_energy: [
            variance_energy::<4>,
            variance_energy::<8>,
            variance_energy::<16>,
            variance_energy::<32>,
            variance_energy::<64>,
        ],
        block_sums: [
            block_sums::<4>,
            block_sums::<8>,
            block_sums::<16>,
            block_sums::<32>,
            block_sums::<64>,
        ],
        halfpel_cross: None,
        sub_pixel_variance: [
            sub_pixel_variance::<4>,
            sub_pixel_variance::<8>,
            sub_pixel_variance::<16>,
            sub_pixel_variance::<32>,
            sub_pixel_variance::<64>,
        ],
        sse_8x8_row,
        plane_sse,
        variance_8x8_row: None,
    }
}

fn residual<const W: usize, const SUM_ABS: bool>(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    out: &mut [i16],
    out_stride: usize,
    h: usize,
) -> u32 {
    let mut sum = 0;
    for r in 0..h {
        for c in 0..W {
            out[r * out_stride + c] =
                i16::from(src[r * src_stride + c]) - i16::from(pred[r * pred_stride + c]);
            if SUM_ABS {
                sum += u32::from(out[r * out_stride + c].unsigned_abs());
            }
        }
    }
    sum
}

/// The reference `ReconstructFn`: the clamped exact sum (equal to the
/// SIMD kernels' saturating 16-bit add and saturating store).
fn reconstruct<const W: usize>(
    prediction: Prediction<'_>,
    residual: ResidualRows<'_>,
    dst: &mut [u8],
    dst_stride: usize,
    h: usize,
) {
    for r in 0..h {
        for c in 0..W {
            let at = r * dst_stride + c;
            let predicted = match prediction {
                Prediction::InPlace => dst[at],
                Prediction::Block(block, stride) => block[r * stride + c],
            };
            let delta = match residual {
                ResidualRows::Samples(samples, stride) => samples[r * stride + c],
                ResidualRows::Constant(dc) => dc,
            };
            dst[at] = (i32::from(predicted) + i32::from(delta)).clamp(0, 255) as u8;
        }
    }
}

fn variance<const W: usize>(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    h: usize,
) -> Variance {
    let (mut sse, mut sum) = (0u32, 0i32);
    for r in 0..h {
        for c in 0..W {
            let d = i32::from(src[r * src_stride + c]) - i32::from(pred[r * pred_stride + c]);
            sum += d;
            sse += (d * d) as u32;
        }
    }
    Variance { sse, sum }
}

fn block_sums<const W: usize>(src: &[u8], src_stride: usize, h: usize) -> Variance {
    let (mut sse, mut sum) = (0u32, 0i32);
    for r in 0..h {
        for &x in &src[r * src_stride..r * src_stride + W] {
            sum += i32::from(x);
            sse += u32::from(x) * u32::from(x);
        }
    }
    Variance { sse, sum }
}

fn variance_energy<const W: usize>(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    h: usize,
) -> VarianceEnergy {
    let (mut sse, mut sum, mut pred_sq, mut pred_sum) = (0u32, 0i32, 0u32, 0i32);
    for r in 0..h {
        for c in 0..W {
            let p = i32::from(pred[r * pred_stride + c]);
            let d = i32::from(src[r * src_stride + c]) - p;
            sum += d;
            sse += (d * d) as u32;
            pred_sum += p;
            pred_sq += (p * p) as u32;
        }
    }
    VarianceEnergy {
        residual: Variance { sse, sum },
        prediction: Variance {
            sse: pred_sq,
            sum: pred_sum,
        },
    }
}

/// One libaom bilinear tap pair applied to two samples.
fn bilinear(a: i32, b: i32, (f0, f1): (i16, i16)) -> i32 {
    (a * i32::from(f0) + b * i32::from(f1) + 64) >> 7
}

fn sub_pixel_variance<const W: usize>(
    rf: &[u8],
    ref_stride: usize,
    x_offset: usize,
    y_offset: usize,
    src: &[u8],
    src_stride: usize,
    h: usize,
) -> Variance {
    let (fx, fy) = (bilinear_taps(x_offset), bilinear_taps(y_offset));
    let first: Vec<i32> = (0..=h)
        .flat_map(|r| {
            (0..W).map(move |c| {
                let row = &rf[r * ref_stride..];
                bilinear(i32::from(row[c]), i32::from(row[c + 1]), fx)
            })
        })
        .collect();
    let mut pred = vec![0u8; W * h];
    for r in 0..h {
        for c in 0..W {
            pred[r * W + c] = bilinear(first[r * W + c], first[(r + 1) * W + c], fy) as u8;
        }
    }
    variance::<W>(src, src_stride, &pred, W, h)
}

/// Sum of squared differences of each 8x8 block along a row (the reference
/// for every ISA).
pub(crate) fn sse_8x8_row(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    out: &mut [u32],
) {
    for (i, sse) in out.iter_mut().enumerate() {
        *sse = (0..8)
            .map(|y| {
                let (s, p) = (
                    &src[y * src_stride + 8 * i..],
                    &pred[y * pred_stride + 8 * i..],
                );
                s[..8]
                    .iter()
                    .zip(&p[..8])
                    .map(|(&a, &b)| u32::from(a.abs_diff(b)).pow(2))
                    .sum::<u32>()
            })
            .sum();
    }
}

/// Total squared error, in raster order, between same-sized planes.
fn plane_sse(a: &nc_common::PlaneView<'_>, b: &nc_common::PlaneView<'_>) -> u64 {
    a.rows()
        .zip(b.rows())
        .flat_map(|(a, b)| a.iter().zip(b))
        .map(|(&a, &b)| u64::from(a.abs_diff(b)).pow(2))
        .sum()
}
