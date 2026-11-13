//! Kernel micro-benchmarks: nanoseconds per call for each kernel, block size
//! and ISA, on cache-resident random data.
//!
//! These numbers compare implementations of one kernel on one machine.
//! Encoder speed claims come from the end-to-end harness in `eval/`, never
//! from here.

use std::hint::black_box;
use std::time::{Duration, Instant};

use nc_common::{PlaneView, PlaneViewMut};
use nc_dsp::{
    Av1Quantizer, Av2Kernel, Av2TxSize, Av2TxType, BlockSize, DctSize, Dsp, Filter, GridBest,
    GridRows, H264Quantizer, HevcQuantizer, HevcTransform, Isa, Moments, Vp9IntraEdges,
    Vp9IntraMode, Vp9QuantStep,
};

use crate::rng::Rng;

/// Block sizes benchmarked (a representative subset of all 25).
const SIZES: [(usize, usize); 9] = [
    (4, 4),
    (8, 8),
    (16, 16),
    (16, 8),
    (32, 8),
    (64, 8),
    (32, 32),
    (64, 64),
    (8, 32),
];
/// Output size of the downscale benchmark (input is twice this).
const DOWNSCALE_OUT: usize = 32;
/// Output block of the deinterleave benchmark: 64 Cb and 64 Cr samples
/// from 128 interleaved bytes per row, 32 rows.
const DEINTERLEAVE_OUT: (usize, usize) = (64, 32);
/// Target duration of one measurement.
const TARGET: Duration = Duration::from_millis(20);
/// Measurements per kernel; the fastest is reported (least disturbed).
const REPEATS: usize = 5;
/// Stride of the benchmark planes: room for four 64-wide candidates.
const STRIDE: usize = 4 * 64 + 32;
/// Rows of the benchmark planes: a 64-high block plus its prediction window.
const ROWS: usize = 64 + 8;

/// One timing.
#[derive(Debug, Clone)]
pub struct BenchResult {
    /// Kernel name (`sad`, `sad_x4`, `sad_quads`, `satd`, `moments`,
    /// `downscale_2x`, `deinterleave_uv`, ...).
    pub kernel: &'static str,
    /// Measured width and height (input dimensions for row benchmarks).
    pub size: (usize, usize),
    /// ISA measured.
    pub isa: Isa,
    /// Best observed nanoseconds per call.
    pub ns_per_call: f64,
}

/// Time only AV1 edge/run filters, using the same cases and timing procedure
/// as the full suite. This keeps iterative filter tuning bounded.
#[must_use]
pub fn bench_av1_runs(isas: &[Isa]) -> Vec<BenchResult> {
    let mut results = Vec::new();
    for &isa in isas {
        if let Some(dsp) = Dsp::with_isa(isa) {
            crate::bench_av1_runs::bench(&dsp, &mut results);
        }
    }
    results
}

/// Time every kernel for each ISA in `isas` that is supported here.
#[must_use]
pub fn bench(isas: &[Isa]) -> Vec<BenchResult> {
    let mut rng = Rng::new(0x5EED);
    let mut src = vec![0u8; STRIDE * ROWS];
    let mut rf = vec![0u8; STRIDE * ROWS];
    rng.fill(&mut src);
    rng.fill(&mut rf);

    let mut results = Vec::new();
    for &isa in isas {
        let Some(dsp) = Dsp::with_isa(isa) else {
            continue;
        };
        for (w, h) in SIZES {
            let size = BlockSize::new(w, h).expect("benchmark sizes are valid");
            let s = PlaneView::new(&src, STRIDE, w, h).expect("plane covers block");
            let refs = [0, 1, 2, 3]
                .map(|i| PlaneView::new(&rf[i * 64..], STRIDE, w, h).expect("plane covers block"));
            let mut record = |kernel, ns_per_call| {
                results.push(BenchResult {
                    kernel,
                    size: (size.width(), size.height()),
                    isa,
                    ns_per_call,
                });
            };
            record("sad", time(|| dsp.sad(size, &s, &refs[0])));
            record("sad_x4", time(|| dsp.sad_x4(size, &s, &refs)[0]));
            record("satd", time(|| dsp.satd(size, &s, &refs[0])));
            bench_size_specific(&dsp, (w, h), &s, &refs, &rf, &mut record);
        }
        bench_satd_quads(&dsp, &src, &rf, &mut results);
        let eight = BlockSize::new(8, 8).expect("8x8 is valid");
        let block = PlaneView::new(&src, STRIDE, 8, 8).expect("plane covers block");
        let mut out = [Moments::default(); 1];
        let moments = time(|| {
            dsp.moments_8x8_row(&block, &mut out);
            out[0].sum_sq
        });
        results.push(BenchResult {
            kernel: "moments",
            size: (eight.width(), eight.height()),
            isa,
            ns_per_call: moments,
        });
        let other = PlaneView::new(&rf, STRIDE, 8, 8).expect("plane covers block");
        let mut sse = [0u32; 1];
        let sse_8x8 = time(|| {
            dsp.sse_8x8_row(&block, &other, &mut sse);
            sse[0]
        });
        results.push(BenchResult {
            kernel: "sse_8x8_row",
            size: (eight.width(), eight.height()),
            isa,
            ns_per_call: sse_8x8,
        });
        // Downscale: `size` is the *output* block (input is twice as large).
        let out_size = BlockSize::new(DOWNSCALE_OUT, DOWNSCALE_OUT).expect("valid size");
        let input = PlaneView::new(&src, STRIDE, 2 * DOWNSCALE_OUT, 2 * DOWNSCALE_OUT)
            .expect("plane covers input");
        let mut out = vec![0u8; DOWNSCALE_OUT * DOWNSCALE_OUT];
        let downscale = time(|| {
            let mut view = PlaneViewMut::new(&mut out, DOWNSCALE_OUT, DOWNSCALE_OUT, DOWNSCALE_OUT)
                .expect("buffer fits");
            dsp.downscale_2x(&input, &mut view);
            u32::from(out[0])
        });
        results.push(BenchResult {
            kernel: "downscale_2x",
            size: (out_size.width(), out_size.height()),
            isa,
            ns_per_call: downscale,
        });
        results.push(bench_deinterleave(&dsp, isa, &src));
        bench_codec(&dsp, isa, &src, &rf, &mut results);
        bench_av1_wide(&dsp, &mut results);
        crate::bench_av1_runs::bench(&dsp, &mut results);
        crate::check_vp9_inverse::bench(&dsp, &mut results);
        crate::check_av1_inverse::bench(&dsp, &mut results);
        crate::bench_rows::bench_rows(&dsp, &mut results);
        crate::bench_memory::bench(&dsp, &mut results);
        crate::bench_vp9_mc::bench(&dsp, &mut results);
    }
    results
}

/// Prediction (sub-sample offsets 5/16 and 9/16), residual, variance and
/// transform kernels.
fn bench_codec(dsp: &Dsp, isa: Isa, src: &[u8], rf: &[u8], results: &mut Vec<BenchResult>) {
    let mut push = |kernel, size: BlockSize, ns_per_call| {
        results.push(BenchResult {
            kernel,
            size: (size.width(), size.height()),
            isa,
            ns_per_call,
        });
    };
    for (w, h) in SIZES {
        let size = BlockSize::new(w, h).expect("benchmark sizes are valid");
        let s = PlaneView::new(src, STRIDE, w, h).expect("plane covers block");
        let window = PlaneView::new(rf, STRIDE, w + 7, h + 7).expect("plane covers window");
        let mut out = vec![0u8; w * h];
        let mut predict = |filter, fx, fy| {
            time(|| {
                let mut dst = PlaneViewMut::new(&mut out, w, w, h).expect("buffer fits");
                dsp.predict(filter, size, &window, fx, fy, &mut dst);
                u32::from(out[0])
            })
        };
        push("av1_predict_2d", size, predict(Filter::Av1, 5, 9));
        push("av1_predict_h", size, predict(Filter::Av1, 5, 0));
        push("av1_smooth_2d", size, predict(Filter::Av1Smooth, 5, 9));
        push("av1_sharp_2d", size, predict(Filter::Av1Sharp, 5, 9));
        push("hevc_luma_2d", size, predict(Filter::HevcLuma, 1, 2));
        push("hevc_chroma_2d", size, predict(Filter::HevcChroma, 3, 5));
        push("h264_luma_j", size, predict(Filter::H264Luma, 2, 2));
        push("h264_luma_f", size, predict(Filter::H264Luma, 2, 1));
        push("h264_chroma", size, predict(Filter::H264Chroma, 3, 5));
        let p = PlaneView::new(rf, STRIDE, w, h).expect("plane covers block");
        let mut residual = vec![0i16; w * h];
        push(
            "residual",
            size,
            time(|| {
                dsp.residual(size, &s, &p, &mut residual, w);
                residual[0] as u32
            }),
        );
        push("variance", size, time(|| dsp.variance(size, &s, &p).sse));
        push("block_sums", size, time(|| dsp.block_sums(size, &s).sse));
        push(
            "variance_energy",
            size,
            time(|| dsp.variance_energy(size, &s, &p).residual.sse),
        );
        let window = PlaneView::new(rf, STRIDE, w + 1, h + 1).expect("plane covers window");
        push(
            "sub_pixel_variance",
            size,
            time(|| dsp.sub_pixel_variance(size, &window, 3, 5, &s).sse),
        );
    }
    bench_av1_transforms(dsp, src, &mut push);
    bench_av1_64(dsp, src, &mut push);
    bench_av1_sparse(dsp, src, &mut push);
    bench_vp9_intra(dsp, src, &mut push);
    bench_hevc_intra(dsp, src, &mut push);
    bench_hevc(dsp, src, &mut push);
    bench_av2(dsp, src, &mut push);
    bench_vp9_quantize(dsp, src, &mut push);
    bench_h264(dsp, src, &mut push);
    crate::bench_extra::bench_extra(dsp, &mut push);
    crate::bench_filters::bench_filters(dsp, &mut push);
    crate::bench_av1_intra::bench_av1_intra(dsp, src, &mut push);
    crate::bench_av2_intra::bench_av2_intra(dsp, src, &mut push);
    crate::bench_vp9_filter::bench(dsp, results);
}

/// The AV1 transforms 4x4 to 32x32 and the quantisers, chained on one
/// residual: forward, quantise, dequantise, gain, inverse.
fn bench_av1_transforms(
    dsp: &Dsp,
    src: &[u8],
    push: &mut impl FnMut(&'static str, BlockSize, f64),
) {
    let q = Av1Quantizer {
        dc_step: 120,
        ac_step: 140,
        rounding_q7: 48,
        max_level: 8191,
        log_scale: 0,
    };
    for dct in DctSize::ALL {
        let n = dct.side();
        let size = BlockSize::new(n, n).expect("DCT sizes are block sizes");
        let residual: Vec<i16> = src[..n * n].iter().map(|&v| i16::from(v) - 128).collect();
        let mut coeffs = vec![0i32; n * n];
        push(
            "av1_forward_dct",
            size,
            time(|| {
                dsp.av1_forward_dct(dct, &residual, n, &mut coeffs);
                coeffs[0] as u32
            }),
        );
        let mut raster = vec![0i32; n * n];
        push(
            "transpose_coefficients",
            size,
            time(|| {
                dsp.transpose_coefficients(dct, &coeffs, &mut raster);
                raster[1] as u32
            }),
        );
        let mut levels = coeffs.clone();
        push(
            "av1_quantize",
            size,
            time(|| {
                levels.copy_from_slice(&coeffs);
                dsp.av1_quantize(&mut levels, &q)
            }),
        );
        push(
            "av1_quantize_from",
            size,
            time(|| dsp.av1_quantize_from(&coeffs, &q, &mut levels)),
        );
        let mut dequantized = vec![0i32; n * n];
        push(
            "av1_dequantize",
            size,
            time(|| {
                dsp.av1_dequantize(&levels, &q, &mut dequantized);
                dequantized[0] as u32
            }),
        );
        push(
            "av1_coefficient_gain",
            size,
            time(|| dsp.av1_coefficient_gain(&coeffs, &dequantized) as u32),
        );
        let mut raster_out = vec![0i32; n * n];
        push(
            "av1_forward_dct_raster",
            size,
            time(|| {
                dsp.av1_forward_dct_raster(dct, &residual, n, &mut raster_out);
                raster_out[1] as u32
            }),
        );
        let mut recon = vec![128u8; n * n];
        push(
            "av1_inverse_dct_add",
            size,
            time(|| {
                let mut dst = PlaneViewMut::new(&mut recon, n, n, n).expect("buffer fits");
                dsp.av1_inverse_dct_add(dct, &dequantized, &mut dst);
                u32::from(recon[0])
            }),
        );
    }
}

/// HEVC transforms and quantisers: DST 4x4 is reported as `hevc_*_dst`.
/// The AV1 64x64 transforms (top-left 32x32 coefficients).
fn bench_av1_64(dsp: &Dsp, src: &[u8], push: &mut impl FnMut(&'static str, BlockSize, f64)) {
    let size_64 = BlockSize::new(64, 64).expect("64x64 is a block size");
    let residual: Vec<i16> = src[..64 * 64].iter().map(|&v| i16::from(v) - 128).collect();
    let mut coeffs = vec![0i32; 32 * 32];
    push(
        "av1_forward_dct_64",
        size_64,
        time(|| {
            dsp.av1_forward_dct_64(&residual, 64, &mut coeffs);
            coeffs[0] as u32
        }),
    );
    let mut recon = vec![128u8; 64 * 64];
    push(
        "av1_inverse_dct_add_64",
        size_64,
        time(|| {
            let mut dst = PlaneViewMut::new(&mut recon, 64, 64, 64).expect("buffer fits");
            dsp.av1_inverse_dct_add_64(&coeffs, &mut dst);
            u32::from(recon[0])
        }),
    );
}

fn bench_av2(dsp: &Dsp, src: &[u8], push: &mut impl FnMut(&'static str, BlockSize, f64)) {
    for transform in Av2TxSize::ALL {
        let n = transform.side();
        let size = BlockSize::new(n, n).expect("transform sizes are block sizes");
        let residual: Vec<i16> = src[..n * n].iter().map(|&v| i16::from(v) - 128).collect();
        let mut coeffs = vec![0i32; transform.coded_area()];
        push(
            "av2_forward_dct",
            size,
            time(|| {
                dsp.av2_forward(transform, &residual, n, &mut coeffs);
                coeffs[0] as u32
            }),
        );
        let mut out = vec![0i32; n * n];
        push(
            "av2_inverse_dct",
            size,
            time(|| {
                dsp.av2_inverse(transform, &coeffs, &mut out);
                out[0] as u32
            }),
        );
        if transform == Av2TxSize::Tx64 {
            continue;
        }
        // ADST where it exists (up to 16x16), else the identity.
        let kernel = if Av2Kernel::Adst.supports(transform) {
            Av2Kernel::Adst
        } else {
            Av2Kernel::Identity
        };
        let tx_type = Av2TxType {
            vertical: kernel,
            horizontal: Av2Kernel::Dct,
        };
        push(
            "av2_forward_typed",
            size,
            time(|| {
                dsp.av2_forward_typed(transform, tx_type, &residual, n, &mut coeffs);
                coeffs[0] as u32
            }),
        );
        push(
            "av2_inverse_typed",
            size,
            time(|| {
                dsp.av2_inverse_typed(transform, tx_type, &coeffs, &mut out);
                out[0] as u32
            }),
        );
    }
}

fn bench_vp9_quantize(dsp: &Dsp, src: &[u8], push: &mut impl FnMut(&'static str, BlockSize, f64)) {
    /// A mid-range VP9 qindex's steps and the inter deadzone.
    const DC_STEP: i32 = 120;
    const AC_STEP: i32 = 140;
    const INTER_DIVISOR: i32 = 6;
    for side in [8, 16, 32] {
        let size = BlockSize::new(side, side).expect("transform sizes are block sizes");
        let shift = u32::from(side == 32);
        let dc = Vp9QuantStep::new(DC_STEP, INTER_DIVISOR, shift);
        let ac = Vp9QuantStep::new(AC_STEP, INTER_DIVISOR, shift);
        // Transform-like magnitudes: residual bytes scaled up.
        let coefficients: Vec<i32> = src[..side * side]
            .iter()
            .map(|&v| (i32::from(v) - 128) * 8)
            .collect();
        let mut levels = vec![0i32; side * side];
        push(
            "vp9_quantize",
            size,
            time(|| u32::from(dsp.vp9_quantize(&coefficients, &dc, &ac, &mut levels))),
        );
    }
}

fn bench_hevc(dsp: &Dsp, src: &[u8], push: &mut impl FnMut(&'static str, BlockSize, f64)) {
    for transform in HevcTransform::ALL {
        let n = transform.side();
        let size = BlockSize::new(n, n).expect("transform sizes are block sizes");
        let (forward_name, inverse_name) = if transform == HevcTransform::Dst4 {
            ("hevc_forward_dst", "hevc_inverse_dst")
        } else {
            ("hevc_forward_dct", "hevc_inverse_dct")
        };
        let residual: Vec<i16> = src[..n * n].iter().map(|&v| i16::from(v) - 128).collect();
        let mut coeffs = vec![0i32; n * n];
        push(
            forward_name,
            size,
            time(|| {
                dsp.hevc_forward(transform, &residual, n, &mut coeffs);
                coeffs[0] as u32
            }),
        );
        let q = HevcQuantizer::new(32, transform.log2(), false);
        let mut levels = coeffs.clone();
        if transform != HevcTransform::Dst4 {
            push(
                "hevc_quantize",
                size,
                time(|| {
                    levels.copy_from_slice(&coeffs);
                    dsp.hevc_quantize(&mut levels, &q)
                }),
            );
        }
        let mut dequantized = vec![0i32; n * n];
        if transform != HevcTransform::Dst4 {
            push(
                "hevc_dequantize",
                size,
                time(|| {
                    dsp.hevc_dequantize(&levels, &q, &mut dequantized);
                    dequantized[0] as u32
                }),
            );
        }
        dsp.hevc_dequantize(&levels, &q, &mut dequantized);
        let mut out = vec![0i32; n * n];
        push(
            inverse_name,
            size,
            time(|| {
                dsp.hevc_inverse(transform, &dequantized, &mut out);
                out[0] as u32
            }),
        );
        bench_hevc_inverse_add(dsp, transform, src, push);
        // Every coefficient nonzero: the worst case of the zero-skipping
        // inverses (quantized blocks above are mostly zero).
        let dense: Vec<i32> = src[..n * n]
            .iter()
            .map(|&v| (i32::from(v) - 128) * DENSE_COEFFICIENT_SCALE)
            .collect();
        push(
            "hevc_inverse_dense",
            size,
            time(|| {
                dsp.hevc_inverse(transform, &dense, &mut out);
                out[0] as u32
            }),
        );
    }
}

/// A 720p picture pitch exposes the reconstruction access pattern separately
/// from tightly packed transform scratch.
const HEVC_BENCH_PICTURE_STRIDE: usize = 1280;

fn bench_hevc_inverse_add(
    dsp: &Dsp,
    transform: HevcTransform,
    src: &[u8],
    push: &mut impl FnMut(&'static str, BlockSize, f64),
) {
    let n = transform.side();
    if transform == HevcTransform::Dst4 {
        return;
    }
    let size = BlockSize::new(n, n).unwrap();
    let mut out = vec![0; n * n];
    for (corner, packed, picture) in [
        (
            1,
            ("hevc_inverse_add_dc", "hevc_inverse_split_dc"),
            ("hevc_add_dc_picture", "hevc_split_dc_picture"),
        ),
        (
            4,
            ("hevc_inverse_add_low4", "hevc_inverse_split_low4"),
            ("hevc_add_low4_picture", "hevc_split_low4_picture"),
        ),
        (
            8,
            ("hevc_inverse_add_low8", "hevc_inverse_split_low8"),
            ("hevc_add_low8_picture", "hevc_split_low8_picture"),
        ),
        (
            n,
            ("hevc_inverse_add_full", "hevc_inverse_split_full"),
            ("hevc_add_full_picture", "hevc_split_full_picture"),
        ),
    ] {
        let input: Vec<i32> = (0..n * n)
            .map(|i| {
                if i / n < corner && i % n < corner {
                    (i32::from(src[i]) - 128) * 16
                } else {
                    0
                }
            })
            .collect();
        for (stride, (fused_name, split_name)) in
            [(n, packed), (HEVC_BENCH_PICTURE_STRIDE, picture)]
        {
            if stride != n && n < 16 {
                continue;
            }
            let mut pixels = vec![128; (n - 1) * stride + n];
            push(
                fused_name,
                size,
                time(|| {
                    dsp.hevc_inverse_add(
                        transform,
                        &input,
                        (corner, corner),
                        &mut PlaneViewMut::new(&mut pixels, stride, n, n).unwrap(),
                    );
                    u32::from(pixels[0])
                }),
            );
            push(
                split_name,
                size,
                time(|| {
                    dsp.hevc_inverse(transform, &input, &mut out);
                    for (row, residual) in pixels.chunks_mut(stride).zip(out.chunks(n)) {
                        for (p, &r) in row.iter_mut().zip(residual) {
                            *p = (i32::from(*p) + r).clamp(0, 255) as u8;
                        }
                    }
                    u32::from(pixels[0])
                }),
            );
        }
    }
}

/// Scale of the dense inverse-transform benchmark's coefficients (source
/// samples minus 128, so up to about ±2^12).
const DENSE_COEFFICIENT_SCALE: i32 = 32;

/// H.264 transforms and quantisers. 4x4 kernels run on a whole 16x16
/// macroblock (16 blocks) and are reported at 16x16.
fn bench_h264(dsp: &Dsp, src: &[u8], push: &mut impl FnMut(&'static str, BlockSize, f64)) {
    let mb = BlockSize::new(16, 16).expect("valid");
    let residual: Vec<i16> = src[..256].iter().map(|&v| i16::from(v) - 128).collect();
    let mut coeffs = [[0i32; 16]; 16];
    push(
        "h264_forward_4x4",
        mb,
        time(|| {
            dsp.h264_forward_4x4(&residual, 16, 4, &mut coeffs);
            coeffs[0][0] as u32
        }),
    );
    let q = H264Quantizer::new(30, false);
    let mut levels = coeffs;
    push(
        "h264_quantize_4x4",
        mb,
        time(|| {
            levels = coeffs;
            dsp.h264_quantize_4x4(&mut levels, &q)
        }),
    );
    let mut scaled = [[0i32; 16]; 16];
    push(
        "h264_dequantize_4x4",
        mb,
        time(|| {
            dsp.h264_dequantize_4x4(&levels, &q, &mut scaled);
            scaled[0][0] as u32
        }),
    );
    let mut out = [[0i32; 16]; 16];
    push(
        "h264_inverse_4x4",
        mb,
        time(|| {
            dsp.h264_inverse_4x4(&scaled, &mut out);
            out[0][0] as u32
        }),
    );
    let dc: [i32; 16] = core::array::from_fn(|i| coeffs[i][0]);
    let b4 = BlockSize::new(4, 4).expect("valid");
    push(
        "h264_hadamard_4x4",
        b4,
        time(|| dsp.h264_hadamard_4x4(&dc)[0] as u32),
    );
    let b8 = BlockSize::new(8, 8).expect("valid");
    let mut c8 = [0i32; 64];
    push(
        "h264_forward_8x8",
        b8,
        time(|| {
            dsp.h264_forward_8x8(&residual, 16, &mut c8);
            c8[0] as u32
        }),
    );
    let mut l8 = c8;
    push(
        "h264_quantize_8x8",
        b8,
        time(|| {
            l8 = c8;
            dsp.h264_quantize_8x8(&mut l8, &q)
        }),
    );
    let mut d8 = [0i32; 64];
    let mut mb_levels = [[0; 64]; 4];
    push(
        "h264_forward_quantize_mb",
        mb,
        time(|| dsp.h264_forward_quantize_mb(&residual, 16, 0, &q, &mut mb_levels)[0]),
    );
    dsp.h264_dequantize_8x8(&l8, &q, &mut d8);
    let mut o8 = [0i32; 64];
    push(
        "h264_inverse_8x8",
        b8,
        time(|| {
            dsp.h264_inverse_8x8(&d8, &mut o8);
            o8[0] as u32
        }),
    );
}

/// Kernels that exist for one block size only.
fn bench_size_specific(
    dsp: &Dsp,
    (w, h): (usize, usize),
    s: &PlaneView<'_>,
    refs: &[PlaneView<'_>; 4],
    rf: &[u8],
    record: &mut impl FnMut(&'static str, f64),
) {
    if (w, h) == (8, 8) {
        let mut out = [[0u32; 4]; 1];
        record(
            "sad_quads",
            time(|| {
                dsp.sad_quads_8x8_row(s, &refs[0], &mut out);
                out[0][0]
            }),
        );
    }
    if (w, h) == (16, 16) {
        record(
            "sad_quads",
            time(|| dsp.sad_quads_16x16_at(s, &refs[0], (0, 0))[0]),
        );
        // One HME window row: 33 positions.
        let wide = PlaneView::new(rf, STRIDE, 16 + 32, 16).expect("plane covers scan");
        let mut out = [0u32; 33];
        record(
            "sad_scan_x33",
            time(|| {
                dsp.sad_scan_16x16_even_rows_at(s, &wide, (0, 0), &mut out);
                out[0]
            }),
        );
    }
    if (w, h) == (64, 64) {
        let mut best = GridBest::default();
        let visible = [u32::MAX; 64];
        let wide = PlaneView::new(rf, STRIDE, 4 * 64, 64).expect("four candidate grids");
        for (single, batched, rows) in [
            ("sad_grid_best", "sad_grid_best_x4", GridRows::All),
            (
                "sad_grid_best_even",
                "sad_grid_best_x4_even",
                GridRows::Even,
            ),
        ] {
            record(
                single,
                time(|| {
                    dsp.sad_grid_best_64x64_at(s, &refs[0], (0, 0), &visible, rows, &mut best, 1);
                    best.sad[84]
                }),
            );
            record(
                batched,
                time(|| {
                    dsp.sad_grid_best_x4_at(
                        s,
                        &wide,
                        [(0, 0), (1, 0), (2, 0), (3, 0)],
                        &visible,
                        rows,
                        &mut best,
                        [1, 2, 3, 4],
                    );
                    best.sad[84]
                }),
            );
        }
    }
}

/// Best-of-`REPEATS` nanoseconds per call of `f`.
pub(crate) fn time(mut f: impl FnMut() -> u32) -> f64 {
    let calls = calibrate(&mut f);
    (0..REPEATS)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..calls {
                black_box(f());
            }
            start.elapsed().as_nanos() as f64 / calls as f64
        })
        .fold(f64::INFINITY, f64::min)
}

/// Number of calls that take roughly `TARGET`.
fn calibrate(f: &mut impl FnMut() -> u32) -> u64 {
    let mut calls = 16u64;
    loop {
        let start = Instant::now();
        for _ in 0..calls {
            black_box(f());
        }
        let elapsed = start.elapsed();
        if elapsed >= TARGET / 4 || calls >= 1 << 30 {
            let scale = TARGET.as_secs_f64() / elapsed.as_secs_f64().max(1e-9);
            return ((calls as f64 * scale) as u64).max(1);
        }
        calls *= 4;
    }
}

fn bench_av1_sparse(dsp: &Dsp, src: &[u8], push: &mut impl FnMut(&'static str, BlockSize, f64)) {
    for side in [32, 64] {
        for (limit, full_name, sparse_name) in [
            (8, "av1_inverse_full_low8", "av1_inverse_sparse_low8"),
            (16, "av1_inverse_full_low16", "av1_inverse_sparse_low16"),
            (32, "av1_inverse_full_low32", "av1_inverse_sparse_low32"),
        ] {
            let mut coeffs = vec![0_i32; 1024];
            for c in 0..limit {
                for r in 0..limit {
                    coeffs[c * 32 + r] = (i32::from(src[c * 64 + r]) - 128) * 16;
                }
            }
            for sparse in [false, true] {
                let mut recon = vec![128_u8; side * side];
                push(
                    if sparse { sparse_name } else { full_name },
                    BlockSize::new(side, side).unwrap(),
                    time(|| {
                        let mut dst = PlaneViewMut::new(&mut recon, side, side, side).unwrap();
                        if sparse {
                            dsp.av1_inverse_dct_add_sparse(side, &coeffs, limit, &mut dst);
                        } else if side == 64 {
                            dsp.av1_inverse_dct_add_64(&coeffs, &mut dst);
                        } else {
                            dsp.av1_inverse_dct_add(
                                DctSize::from_side(side).unwrap(),
                                &coeffs,
                                &mut dst,
                            );
                        }
                        u32::from(recon[0])
                    }),
                );
            }
        }
    }
}

/// Time NV12 chroma deinterleaving of one [`DEINTERLEAVE_OUT`] block.
fn bench_deinterleave(dsp: &Dsp, isa: Isa, src: &[u8]) -> BenchResult {
    let (w, h) = DEINTERLEAVE_OUT;
    let input = PlaneView::new(src, STRIDE, 2 * w, h).expect("plane covers input");
    let (mut u, mut v) = (vec![0u8; w * h], vec![0u8; w * h]);
    let ns_per_call = time(|| {
        let mut u_view = PlaneViewMut::new(&mut u, w, w, h).expect("buffer fits");
        let mut v_view = PlaneViewMut::new(&mut v, w, w, h).expect("buffer fits");
        dsp.deinterleave_uv(&input, &mut u_view, &mut v_view);
        u32::from(u[0]) + u32::from(v[0])
    });
    BenchResult {
        kernel: "deinterleave_uv",
        size: (w, h),
        isa,
        ns_per_call,
    }
}

fn bench_hevc_intra(dsp: &Dsp, src: &[u8], push: &mut impl FnMut(&'static str, BlockSize, f64)) {
    for n in [4, 8, 16, 32] {
        let size = BlockSize::new(n, n).expect("transform sizes are block sizes");
        let mut out = vec![0u8; n * n];
        let refs = &src[..=3 * n];
        // A fractional angle (every line blends) and a negative one.
        for (name, angle) in [
            ("hevc_intra_angular_p13", 13),
            ("hevc_intra_angular_m21", -21),
        ] {
            push(
                name,
                size,
                time(|| {
                    dsp.hevc_intra_angular(refs, n, angle, &mut out);
                    u32::from(out[n * n - 1])
                }),
            );
        }
        let edges = nc_dsp::HevcPlanarEdges {
            above: &src[..n],
            left: &src[n..2 * n],
            top_right: src[2 * n],
            bottom_left: src[2 * n + 1],
        };
        push(
            "hevc_intra_planar",
            size,
            time(|| {
                dsp.hevc_intra_planar(&edges, n, &mut out);
                u32::from(out[n * n - 1])
            }),
        );
    }
}

fn bench_vp9_intra(dsp: &Dsp, src: &[u8], push: &mut impl FnMut(&'static str, BlockSize, f64)) {
    let mut edges = Vp9IntraEdges {
        above: [0; 65],
        left: [0; 32],
        has_above: true,
        has_left: true,
    };
    edges.above.copy_from_slice(&src[..65]);
    edges.left.copy_from_slice(&src[65..97]);
    for side in [4, 8, 16, 32] {
        let size = BlockSize::new(side, side).expect("transform sizes are block sizes");
        let mut out = vec![0u8; side * side];
        let mut edges = edges;
        if side > 4 {
            let last = edges.above[side];
            edges.above[side + 1..=2 * side].fill(last);
        }
        for (name, mode) in [
            ("vp9_intra_dc", Vp9IntraMode::Dc),
            ("vp9_intra_tm", Vp9IntraMode::Tm),
            ("vp9_intra_d45", Vp9IntraMode::D45),
            ("vp9_intra_d63", Vp9IntraMode::D63),
            ("vp9_intra_d117", Vp9IntraMode::D117),
            ("vp9_intra_d153", Vp9IntraMode::D153),
            ("vp9_intra_d207", Vp9IntraMode::D207),
        ] {
            push(
                name,
                size,
                time(|| {
                    dsp.vp9_intra_predict(mode, side, &edges, &mut out);
                    u32::from(out[side * side - 1])
                }),
            );
        }
    }
}

/// 128-sample prediction uses the shared 64-sample kernel family directly.
fn bench_av1_wide(dsp: &Dsp, results: &mut Vec<BenchResult>) {
    use nc_dsp::av1_mc::{
        convolve::Av1McAxis,
        filter::{Av1Filter, Av1Filters},
    };
    for (w, h) in [(128, 128), (128, 4), (4, 128)] {
        let ss = w + 7;
        let source: Vec<u8> = (0..ss * (h + 7))
            .map(|i| (i * 37 + i / ss * 17) as u8)
            .collect();
        let window = PlaneView::new(&source, ss, w + 7, h + 7).expect("window");
        let ds = w + 13;
        let mut output = vec![0; (h - 1) * ds + w];
        for (kernel, fx, fy) in [
            ("av1_wide_copy", 0, 0),
            ("av1_wide_h", 7, 0),
            ("av1_wide_v", 0, 9),
            ("av1_wide_2d", 7, 9),
        ] {
            results.push(BenchResult {
                kernel,
                size: (w, h),
                isa: dsp.isa(),
                ns_per_call: time(|| {
                    let mut view = PlaneViewMut::new(&mut output, ds, w, h).expect("output");
                    dsp.av1_predict(
                        Av1Filters::both(Av1Filter::Regular),
                        &window,
                        (w, h),
                        (Av1McAxis::unscaled(fx), Av1McAxis::unscaled(fy)),
                        &mut view,
                    );
                    u32::from(output[0])
                }),
            });
        }
    }
}

/// Paired costs and independent calls use identical inputs and normalization.
fn bench_satd_quads(dsp: &Dsp, src: &[u8], rf: &[u8], results: &mut Vec<BenchResult>) {
    for (ss, rs, names) in [
        (STRIDE, STRIDE, ["satd_8x8_quads", "satd_8x8_quads_split"]),
        // Actual partition seed: picture region against a packed prediction.
        (1280, 16, ["satd_quads_picture", "satd_quads_picture_split"]),
    ] {
        let source = PlaneView::new(src, ss, 16, 16).unwrap();
        let reference = PlaneView::new(rf, rs, 16, 16).unwrap();
        for (index, kernel) in names.into_iter().enumerate() {
            let ns_per_call = time(|| {
                if index == 0 {
                    black_box(dsp.satd_8x8_quads(&source, &reference))
                        .into_iter()
                        .sum()
                } else {
                    let eight = BlockSize::new(8, 8).unwrap();
                    (0..4)
                        .map(|q| {
                            let (x, y) = (8 * (q % 2), 8 * (q / 2));
                            dsp.satd(
                                eight,
                                &source.sub_view(x, y, 8, 8).unwrap(),
                                &reference.sub_view(x, y, 8, 8).unwrap(),
                            )
                        })
                        .sum()
                }
            });
            results.push(BenchResult {
                kernel,
                size: (16, 16),
                isa: dsp.isa(),
                ns_per_call,
            });
        }
    }
}
