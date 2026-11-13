//! VP9 interpolation benchmarks for identity, one-axis and separable filtering.
use crate::bench::{BenchResult, time};
use nc_dsp::{Dsp, Vp9Filter};

pub(crate) fn bench(dsp: &Dsp, results: &mut Vec<BenchResult>) {
    for side in [4, 8, 16, 32, 64] {
        let stride = side + 7;
        let src: Vec<u8> = (0..stride * stride)
            .map(|i| (i * 37 + i / stride * 19) as u8)
            .collect();
        let mut dst = vec![0; side * side];
        for (kind, names) in [
            (
                Vp9Filter::Regular,
                ["vp9_mc_copy", "vp9_mc_h", "vp9_mc_v", "vp9_mc_2d"],
            ),
            (
                Vp9Filter::Smooth,
                [
                    "vp9_smooth_copy",
                    "vp9_smooth_h",
                    "vp9_smooth_v",
                    "vp9_smooth_2d",
                ],
            ),
            (
                Vp9Filter::Sharp,
                [
                    "vp9_sharp_copy",
                    "vp9_sharp_h",
                    "vp9_sharp_v",
                    "vp9_sharp_2d",
                ],
            ),
            (
                Vp9Filter::Bilinear,
                [
                    "vp9_bilinear_copy",
                    "vp9_bilinear_h",
                    "vp9_bilinear_v",
                    "vp9_bilinear_2d",
                ],
            ),
        ] {
            for (name, phase) in names.into_iter().zip([(0, 0), (7, 0), (0, 7), (7, 9)]) {
                results.push(BenchResult {
                    kernel: name,
                    size: (side, side),
                    isa: dsp.isa(),
                    ns_per_call: time(|| {
                        dsp.vp9_convolve(kind, &src, stride, phase, (side, side), &mut dst);
                        u32::from(dst[0])
                    }),
                });
            }
        }
        let dst_stride = side + 13;
        let mut dst = vec![127; (side - 1) * dst_stride + side];
        for (kernel, phase, average) in [
            ("vp9_put_strided", (7, 9), false),
            ("vp9_avg_strided_copy", (0, 0), true),
            ("vp9_avg_strided_h", (7, 0), true),
            ("vp9_avg_strided_v", (0, 7), true),
            ("vp9_avg_strided_2d", (7, 9), true),
        ] {
            results.push(BenchResult {
                kernel,
                size: (side, side),
                isa: dsp.isa(),
                ns_per_call: time(|| {
                    if average {
                        dsp.vp9_convolve_avg_into(
                            Vp9Filter::Regular,
                            &src,
                            stride,
                            phase,
                            (side, side),
                            &mut dst,
                            dst_stride,
                        );
                    } else {
                        dsp.vp9_convolve_into(
                            Vp9Filter::Regular,
                            &src,
                            stride,
                            phase,
                            (side, side),
                            &mut dst,
                            dst_stride,
                        );
                    }
                    u32::from(dst[0])
                }),
            });
        }
    }
}
