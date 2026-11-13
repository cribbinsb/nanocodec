//! Sixteen-byte bilinear prediction and dot-product distortion accumulation.
//! Half-pixel filtering uses the exact rounded byte average; other offsets
//! retain both normative rounding stages. Installed only with dotprod support.
//! Five paired Orin runs: HEVC CPU -5.91%/-5.66%, AV1 -7.02%/-6.28%
//! on bodycam/crowd, with identical streams and independent decoder gates.

use super::{Variance, VarianceEnergy};
use crate::simd::neon_dot::udot as dot;
use core::arch::aarch64::{
    uint8x16_t, vabdq_u8, vaddvq_u32, vcombine_u8, vdup_n_u8, vdupq_n_u8, vdupq_n_u32,
    vget_high_u8, vget_low_u8, vhaddq_u8, vld1q_u8, vmlal_u8, vmull_u8, vrhaddq_u8, vrshrn_n_u16,
};

#[inline]
#[target_feature(enable = "neon")]
unsafe fn blend(a: uint8x16_t, b: uint8x16_t, offset: usize) -> uint8x16_t {
    // The quarter positions the encoders use (offsets 2, 4, 6 of 8) are
    // exact byte averages: with t = (a + b) >> 1, (6a + 2b + 4) >> 3 =
    // (a + t + 1) >> 1 and (2a + 6b + 4) >> 3 = (b + t + 1) >> 1 (checked
    // over all byte pairs; the dropped half of t never changes the result).
    // Jetson 16x16 per call: offset (2, 6) 142 -> 54 ns, (2, 0) 72 -> 37 ns.
    match offset {
        2 => return vrhaddq_u8(a, vhaddq_u8(a, b)),
        4 => return vrhaddq_u8(a, b),
        6 => return vrhaddq_u8(b, vhaddq_u8(a, b)),
        _ => {}
    }
    // Divide the bilinear coefficients by16: (a*(8-k)+b*k+4)>>3.
    // The largest unrounded sum is2040, so unsigned16 cannot overflow.
    let left = vdup_n_u8((8 - offset) as u8);
    let right = vdup_n_u8(offset as u8);
    let low = vmlal_u8(vmull_u8(vget_low_u8(a), left), vget_low_u8(b), right);
    let high = vmlal_u8(vmull_u8(vget_high_u8(a), left), vget_high_u8(b), right);
    vcombine_u8(vrshrn_n_u16::<3>(low), vrshrn_n_u16::<3>(high))
}

/// Whole-pel variance with exact W-byte rows: unlike subpel, neither input
/// needs a filter border or an extra bottom row. Only16/32/64 widths use this
/// table entry; narrower widths and CPUs without dotprod retain generic NEON.
#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn variance<const W: usize>(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    h: usize,
) -> Variance {
    assert!(matches!(W, 16 | 32 | 64));
    assert!(super::generic::covers(src.len(), src_stride, h, W));
    assert!(super::generic::covers(pred.len(), pred_stride, h, W));
    // Keep independent sums for the 16-byte columns and walk rows once.
    // W is constant, so the inner loop unrolls without a candidate-order
    // change (only exact integer sums are reassociated).
    let mut squares = [vdupq_n_u32(0); 4];
    let mut source_sum = squares;
    let mut pred_sum = squares;
    let ones = vdupq_n_u8(1);
    // SAFETY: the checked windows cover every exact-width row; all partial
    // sums are bounded by the full 64x64 sum, below 2^32.
    unsafe {
        for r in 0..h {
            for c in 0..W / 16 {
                let source = vld1q_u8(src.as_ptr().add(r * src_stride + c * 16));
                let prediction = vld1q_u8(pred.as_ptr().add(r * pred_stride + c * 16));
                let difference = vabdq_u8(source, prediction);
                squares[c] = dot(squares[c], difference, difference);
                source_sum[c] = dot(source_sum[c], source, ones);
                pred_sum[c] = dot(pred_sum[c], prediction, ones);
            }
        }
    }
    let total = |values: [core::arch::aarch64::uint32x4_t; 4]| {
        values[..W / 16].iter().map(|&v| vaddvq_u32(v)).sum::<u32>()
    };
    Variance {
        sse: total(squares),
        sum: total(source_sum) as i32 - total(pred_sum) as i32,
    }
}

/// Whole-pel [`VarianceEnergy`]: [`variance`]'s three dot-product sums
/// plus the prediction's sum of squares, five vector operations per 16
/// samples. Jetson 16x16: 1.19x [`variance`] (the generic 8-lane kernel
/// took about 2.8x). Only 16/32/64 widths use this table entry, as for
/// [`variance`].
#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn variance_energy<const W: usize>(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    h: usize,
) -> VarianceEnergy {
    assert!(matches!(W, 16 | 32 | 64));
    assert!(super::generic::covers(src.len(), src_stride, h, W));
    assert!(super::generic::covers(pred.len(), pred_stride, h, W));
    let mut squares = vdupq_n_u32(0);
    let mut source_sum = vdupq_n_u32(0);
    let mut pred_sum = vdupq_n_u32(0);
    let mut pred_squares = vdupq_n_u32(0);
    let ones = vdupq_n_u8(1);
    // SAFETY: both exact-width windows are checked above; every 16-byte
    // load ends within its row. A 64x64 block's squares stay below
    // 4096 * 255^2 < 2^32.
    unsafe {
        for c in (0..W).step_by(16) {
            for r in 0..h {
                let source = vld1q_u8(src.as_ptr().add(r * src_stride + c));
                let prediction = vld1q_u8(pred.as_ptr().add(r * pred_stride + c));
                let difference = vabdq_u8(source, prediction);
                squares = dot(squares, difference, difference);
                source_sum = dot(source_sum, source, ones);
                pred_sum = dot(pred_sum, prediction, ones);
                pred_squares = dot(pred_squares, prediction, prediction);
            }
        }
    }
    let prediction_sum = vaddvq_u32(pred_sum);
    VarianceEnergy {
        residual: Variance {
            sse: vaddvq_u32(squares),
            sum: vaddvq_u32(source_sum) as i32 - prediction_sum as i32,
        },
        prediction: Variance {
            sse: vaddvq_u32(pred_squares),
            sum: prediction_sum as i32,
        },
    }
}

/// Whole-block sums of `src`'s own samples with two dot products per 16
/// samples (sum against ones, squares against itself). As the psy AC
/// energy (`nc_encoder::psy::ac_energy`, formerly a variance pass against
/// zeros): h264-fast −0.4% (crowd) / −0.8% (bodycam) Jetson cycles. Only
/// 16/32/64 widths use this table entry, as for [`variance`].
#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn block_sums<const W: usize>(
    src: &[u8],
    src_stride: usize,
    h: usize,
) -> Variance {
    assert!(matches!(W, 16 | 32 | 64));
    assert!(super::generic::covers(src.len(), src_stride, h, W));
    let mut squares = vdupq_n_u32(0);
    let mut sum = vdupq_n_u32(0);
    let ones = vdupq_n_u8(1);
    // SAFETY: the exact-width window is checked above; every 16-byte load
    // ends within its row.
    unsafe {
        for c in (0..W).step_by(16) {
            for r in 0..h {
                let x = vld1q_u8(src.as_ptr().add(r * src_stride + c));
                squares = dot(squares, x, x);
                sum = dot(sum, x, ones);
            }
        }
    }
    Variance {
        sse: vaddvq_u32(squares),
        sum: vaddvq_u32(sum) as i32,
    }
}

/// The checked reference window holds W+1 bytes per row for h+1 rows;
/// source holds W bytes per row for h rows. W is16,32 or64.
#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn subpel<const W: usize>(
    rf: &[u8],
    ref_stride: usize,
    xo: usize,
    yo: usize,
    src: &[u8],
    src_stride: usize,
    h: usize,
) -> Variance {
    assert!(xo < 8 && yo < 8 && h > 0);
    assert!(rf.len() > h * ref_stride + W);
    assert!(src.len() >= (h - 1) * src_stride + W);
    // SAFETY: the assertions cover every16-byte load, including the one-byte
    // shifted horizontal taps and the extra bottom row. ISA is enabled here.
    unsafe {
        match (xo, yo) {
            // Whole-sample: no filter at all (it ran the multiply blend at
            // offset 0, 72 ns instead of 27).
            (0, 0) => block::<W, false, false>(rf, ref_stride, xo, yo, src, src_stride, h),
            (_, 0) => block::<W, true, false>(rf, ref_stride, xo, yo, src, src_stride, h),
            (0, _) => block::<W, false, true>(rf, ref_stride, xo, yo, src, src_stride, h),
            _ => block::<W, true, true>(rf, ref_stride, xo, yo, src, src_stride, h),
        }
    }
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn block<const W: usize, const H: bool, const V: bool>(
    rf: &[u8],
    rs: usize,
    xo: usize,
    yo: usize,
    src: &[u8],
    ss: usize,
    h: usize,
) -> Variance {
    let mut squares = vdupq_n_u32(0);
    let mut source_sum = vdupq_n_u32(0);
    let mut pred_sum = vdupq_n_u32(0);
    let ones = vdupq_n_u8(1);
    // Keep each column's previous filtered row in a register across rows.
    // SAFETY: entry points validate both windows; c+16 never exceeds W.
    unsafe {
        for c in (0..W).step_by(16) {
            let mut previous = vdupq_n_u8(0);
            if V {
                previous = vld1q_u8(rf.as_ptr().add(c));
                if H {
                    previous = blend(previous, vld1q_u8(rf.as_ptr().add(c + 1)), xo);
                }
            }
            for r in 0..h {
                let index = (r + usize::from(V)) * rs + c;
                let mut next = vld1q_u8(rf.as_ptr().add(index));
                if H {
                    next = blend(next, vld1q_u8(rf.as_ptr().add(index + 1)), xo);
                }
                let prediction = if V { blend(previous, next, yo) } else { next };
                previous = next;
                let source = vld1q_u8(src.as_ptr().add(r * ss + c));
                let difference = vabdq_u8(source, prediction);
                squares = dot(squares, difference, difference);
                source_sum = dot(source_sum, source, ones);
                pred_sum = dot(pred_sum, prediction, ones);
            }
        }
    }
    Variance {
        sse: vaddvq_u32(squares),
        sum: vaddvq_u32(source_sum) as i32 - vaddvq_u32(pred_sum) as i32,
    }
}

#[cfg(test)]
mod tests {
    use super::subpel;

    #[test]
    fn ordinary_variance_exact_tight_unaligned_and_extreme_windows() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        let scalar = crate::diff::scalar::diff_fns();
        for (index, w) in [(2, 16), (3, 32), (4, 64)] {
            for h in [0, 1, 2, 7, 15, 16, 31, 32, 63, 64] {
                for extra in [0, 5] {
                    let ss = w + extra;
                    let ps = w + extra * 2;
                    let slen = if h == 0 { 0 } else { (h - 1) * ss + w };
                    let plen = if h == 0 { 0 } else { (h - 1) * ps + w };
                    for pattern in 0..6 {
                        let src: Vec<u8> = (0..=slen)
                            .map(|i| match pattern {
                                0 => 0,
                                1 => 255,
                                2 => (i % 2 * 255) as u8,
                                3 => ((i / ss + i % ss) % 2 * 255) as u8,
                                _ => ((i * 73 + i / 17) % 256) as u8,
                            })
                            .collect();
                        let pred: Vec<u8> = (0..plen + 3)
                            .map(|i| match pattern {
                                0 => 255,
                                1 => 0,
                                2 => ((i + 1) % 2 * 255) as u8,
                                3 => ((i / ps + i % ps + 1) % 2 * 255) as u8,
                                4 => ((i * 19 + i / 13) % 256) as u8,
                                _ => 128,
                            })
                            .collect();
                        let (src, pred) = (&src[1..], &pred[3..]);
                        // SAFETY: exact strided windows include every row,
                        // with no extra filtering border even on the last row.
                        let (actual, expected) = unsafe {
                            let actual = match w {
                                16 => super::variance::<16>(src, ss, pred, ps, h),
                                32 => super::variance::<32>(src, ss, pred, ps, h),
                                _ => super::variance::<64>(src, ss, pred, ps, h),
                            };
                            (actual, (scalar.variance[index])(src, ss, pred, ps, h))
                        };
                        assert_eq!(actual, expected, "W{w} H{h} extra{extra} pattern{pattern}");
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "bounded native kernel timing"]
    fn ordinary_variance_native_screen() {
        use crate::simd::neon::I16x8;
        type Kernel = unsafe fn(&[u8], usize, &[u8], usize, usize) -> super::Variance;
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        for (w, old, new) in [
            (
                16,
                crate::diff::generic::variance::<I16x8, 16> as Kernel,
                super::variance::<16> as Kernel,
            ),
            (
                32,
                crate::diff::generic::variance::<I16x8, 32> as Kernel,
                super::variance::<32> as Kernel,
            ),
            (
                64,
                crate::diff::generic::variance::<I16x8, 64> as Kernel,
                super::variance::<64> as Kernel,
            ),
        ] {
            for h in [w / 2, w] {
                let a: Vec<u8> = (0..w * h).map(|i| (i * 73) as u8).collect();
                let b: Vec<u8> = (0..w * h).map(|i| (i * 101 + 17) as u8).collect();
                for round in 0..3 {
                    for (name, f) in if round % 2 == 0 {
                        [("before", old), ("after", new)]
                    } else {
                        [("after", new), ("before", old)]
                    } {
                        let start = std::time::Instant::now();
                        for _ in 0..100_000 {
                            // SAFETY: feature checked and both exact windows cover W*h.
                            std::hint::black_box(unsafe {
                                f(std::hint::black_box(&a), w, std::hint::black_box(&b), w, h)
                            });
                        }
                        println!(
                            "variance W{w} H{h} round{round} {name} ns {}",
                            start.elapsed().as_nanos()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn all_offsets_tight_windows_match_scalar() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        let scalar = crate::diff::scalar::diff_fns();
        for (index, w) in [(2, 16), (3, 32), (4, 64)] {
            for h in [1, 4, 16, 64] {
                for pattern in 0..4 {
                    let rf: Vec<u8> = (0..(h + 1) * (w + 1))
                        .map(|i| match pattern {
                            0 => 255,
                            1 => 0,
                            2 => (i % 2 * 255) as u8,
                            _ => ((i * 73 + i / 17) % 256) as u8,
                        })
                        .collect();
                    let src: Vec<u8> = (0..h * w)
                        .map(|i| {
                            if pattern == 0 {
                                0
                            } else {
                                ((i * 101 + 255) % 256) as u8
                            }
                        })
                        .collect();
                    for xo in 0..8 {
                        for yo in 0..8 {
                            // SAFETY: feature checked; both tightly sized windows
                            // cover their complete blocks and extra filter taps.
                            let (got, want) = unsafe {
                                let got = match w {
                                    16 => subpel::<16>(&rf, w + 1, xo, yo, &src, w, h),
                                    32 => subpel::<32>(&rf, w + 1, xo, yo, &src, w, h),
                                    _ => subpel::<64>(&rf, w + 1, xo, yo, &src, w, h),
                                };
                                (
                                    got,
                                    (scalar.sub_pixel_variance[index])(
                                        &rf,
                                        w + 1,
                                        xo,
                                        yo,
                                        &src,
                                        w,
                                        h,
                                    ),
                                )
                            };
                            assert_eq!(got, want, "{w}x{h} pattern{pattern} offset{xo},{yo}");
                        }
                    }
                }
            }
        }
    }
}
