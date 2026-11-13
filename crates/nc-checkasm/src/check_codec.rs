//! Bit-exactness checks of the codec kernels (prediction, AV1 compound
//! prediction and average, residual,
//! variance, AV1 transforms and quantisation) against scalar; the HEVC and
//! H.264 transforms are in `check_codec_txfm`.
//!
//! Buffers end exactly at their last sample, so an over-read or over-write
//! leaves the allocation; outputs are compared in full, including the
//! untouched gaps between rows.

use nc_common::{PlaneView, PlaneViewMut};
use nc_dsp::{Av1Quantizer, BlockSize, DctSize, Dsp, Filter};

use crate::check::{CheckReport, MAX_EXTRA_STRIDE, PATTERNS, TestBlock, fill};
use crate::rng::Rng;

/// Every prediction filter.
const FILTERS: [Filter; 10] = [
    Filter::Av1,
    Filter::Av1Smooth,
    Filter::Av1Sharp,
    Filter::HevcLuma,
    Filter::HevcChroma,
    Filter::H264Luma,
    Filter::H264Chroma,
    Filter::VvcLuma,
    Filter::VvcAffineLuma,
    Filter::VvcChroma,
];
/// The filters with compound-precision prediction: AV1's families and
/// HEVC's 14-bit `predSamplesLX`.
const COMPOUND_FILTERS: [Filter; 5] = [
    Filter::Av1,
    Filter::Av1Smooth,
    Filter::Av1Sharp,
    Filter::HevcLuma,
    Filter::HevcChroma,
];
/// Magnitude bound of a compound prediction sample: 255 in the compound
/// domain (x16) with the sharp filter's overshoot, below `2^14`.
const COMPOUND_MAGNITUDE: usize = 1 << 13;
/// Sub-sample positions of the sub-pixel variance (1/8).
const VARIANCE_POSITIONS: usize = 8;
/// Largest quantiser step tried (AV1 8-bit `ac_q` at qindex 255 is 1828).
const MAX_STEP: usize = 1828;
/// Largest coefficient magnitude quantised (beyond any transform output).
const MAX_COEFF: usize = 1 << 20;
/// Largest level cap tried: nc-av1 caps at the level whose dequantised
/// value saturates, `2^(15 + dqDenom) / step`, at most 32768 at step 4.
const MAX_LEVEL: usize = 1 << 15;
/// Largest `log2(dqDenom)` (2, for 64x64 transforms).
const MAX_LOG_SCALE: usize = 2;

pub(crate) fn check_codec_kernels(
    candidate: &Dsp,
    reference: &Dsp,
    iterations: usize,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    for size in BlockSize::all() {
        for _ in 0..iterations {
            for filter in FILTERS {
                check_predict(candidate, reference, filter, size, rng, report);
            }
            for filter in COMPOUND_FILTERS {
                check_predict_compound(candidate, reference, filter, size, rng, report);
            }
            check_average_compound(candidate, reference, size, rng, report);
            check_residuals(candidate, reference, size, rng, report);
        }
    }
    check_hevc_prep_phases(candidate, reference, report);
    check_hevc_bi(candidate, reference, rng, report);
    check_h264_predictor_phases(candidate, reference, rng, report);
    check_av1_wide_prediction(candidate, reference, rng, report);
    check_inverse_narrow_boundaries(candidate, reference, rng, report);
    for size in DctSize::ALL {
        for _ in 0..iterations {
            check_transforms(candidate, reference, size, rng, report);
        }
    }
    for _ in 0..iterations {
        check_transforms_64(candidate, reference, rng, report);
        check_forward_low(candidate, reference, rng, report);
        check_forward_bounded(candidate, reference, rng, report);
        check_sparse_inverse(candidate, reference, rng, report);
    }
    for _ in 0..iterations {
        check_quantize(candidate, reference, rng, report);
        check_quantize_from(candidate, reference, rng, report);
    }
    crate::check_codec_txfm::check_codec_txfm(candidate, reference, iterations, rng, report);
    crate::check_av1_intra::check_av1_intra(candidate, reference, iterations, rng, report);
    crate::check_av1_intra::check_av1_directional(candidate, reference, iterations, rng, report);
    crate::check_av2_intra::check_av2_intra(candidate, reference, iterations, rng, report);
}

/// Count a case; on mismatch, describe it (lazily: most cases pass).
pub(crate) fn record(
    report: &mut CheckReport,
    isa: nc_dsp::Isa,
    same: bool,
    what: impl FnOnce() -> String,
) {
    report.cases += 1;
    if !same {
        report.failures.push(format!("{isa} {}", what()));
    }
}

/// A destination buffer of `w`x`h` at a random stride, exactly long enough,
/// pre-filled with `fill_value` or random data.
fn destination(w: usize, h: usize, rng: &mut Rng) -> (Vec<u8>, usize) {
    let stride = w + rng.below(MAX_EXTRA_STRIDE + 1);
    let mut data = vec![0u8; (h - 1) * stride + w];
    rng.fill(&mut data);
    (data, stride)
}

fn check_predict(
    candidate: &Dsp,
    reference: &Dsp,
    filter: Filter,
    size: BlockSize,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let (w, h) = (size.width(), size.height());
    let extra = filter.taps() - 1;
    let pattern = PATTERNS[rng.below(PATTERNS.len())];
    let window = TestBlock::new(w + extra, h + extra, 1, rng);
    let other = TestBlock::new(w + extra, h + extra, 1, rng);
    let (data, _) = fill(pattern, &window, &other, rng);
    let view = window.view(&data, 0);
    // Zero offsets in either direction take separate paths: make them common.
    let pick = |rng: &mut Rng| {
        if rng.below(3) == 0 {
            0
        } else {
            rng.below(filter.positions())
        }
    };
    let (fx, fy) = (pick(rng), pick(rng));
    let (start, stride) = destination(w, h, rng);
    let run = |dsp: &Dsp| {
        let mut out = start.clone();
        let mut dst = PlaneViewMut::new(&mut out, stride, w, h).expect("buffer fits");
        dsp.predict(filter, size, &view, fx, fy, &mut dst);
        out
    };
    let same = run(candidate) == run(reference);
    record(report, candidate.isa(), same, || {
        format!(
            "predict {filter:?} {w}x{h} {pattern:?} frac=({fx},{fy}) stride={}",
            window.stride
        )
    });
}

fn check_predict_compound(
    candidate: &Dsp,
    reference: &Dsp,
    filter: Filter,
    size: BlockSize,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let (w, h) = (size.width(), size.height());
    let extra = filter.taps() - 1;
    let pattern = PATTERNS[rng.below(PATTERNS.len())];
    let window = TestBlock::new(w + extra, h + extra, 1, rng);
    let other = TestBlock::new(w + extra, h + extra, 1, rng);
    let (data, _) = fill(pattern, &window, &other, rng);
    let view = window.view(&data, 0);
    let pick = |rng: &mut Rng| {
        if rng.below(3) == 0 {
            0
        } else {
            rng.below(filter.positions())
        }
    };
    let (fx, fy) = (pick(rng), pick(rng));
    let stride = w + rng.below(MAX_EXTRA_STRIDE + 1);
    let start: Vec<i16> = (0..(h - 1) * stride + w)
        .map(|_| rng.below(1 << 15) as i16)
        .collect();
    let run = |dsp: &Dsp| {
        let mut out = start.clone();
        dsp.predict_compound(filter, size, &view, (fx, fy), &mut out, stride);
        out
    };
    let same = run(candidate) == run(reference);
    record(report, candidate.isa(), same, || {
        format!(
            "predict_compound {filter:?} {w}x{h} {pattern:?} frac=({fx},{fy}) stride={}",
            window.stride
        )
    });
}

fn check_average_compound(
    candidate: &Dsp,
    reference: &Dsp,
    size: BlockSize,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let (w, h) = (size.width(), size.height());
    let src_stride = w + rng.below(MAX_EXTRA_STRIDE + 1);
    let len = (h - 1) * src_stride + w;
    // Both signs and both clipping ends: the sum of two predictions spans
    // well beyond 0..255 after rounding.
    let sample =
        |rng: &mut Rng| rng.below(2 * COMPOUND_MAGNITUDE) as i16 - COMPOUND_MAGNITUDE as i16 / 4;
    let p0: Vec<i16> = (0..len).map(|_| sample(rng)).collect();
    let p1: Vec<i16> = (0..len).map(|_| sample(rng)).collect();
    let (start, stride) = destination(w, h, rng);
    let run = |dsp: &Dsp| {
        let mut out = start.clone();
        let mut dst = PlaneViewMut::new(&mut out, stride, w, h).expect("buffer fits");
        dsp.average_compound(size, (&p0, &p1), src_stride, &mut dst);
        out
    };
    let same = run(candidate) == run(reference);
    record(report, candidate.isa(), same, || {
        format!("average_compound {w}x{h}")
    });
}

fn check_residuals(
    candidate: &Dsp,
    reference: &Dsp,
    size: BlockSize,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let (w, h) = (size.width(), size.height());
    let isa = candidate.isa();
    let pattern = PATTERNS[rng.below(PATTERNS.len())];
    let src = TestBlock::new(w, h, 1, rng);
    let pred = TestBlock::new(w + 1, h + 1, 1, rng);
    let (src_data, pred_data) = fill(pattern, &src, &pred, rng);
    let (s, p) = (src.view(&src_data, 0), pred.view(&pred_data, 0));

    let out_stride = w + rng.below(MAX_EXTRA_STRIDE + 1);
    let out_len = (h - 1) * out_stride + w;
    let run_residual = |dsp: &Dsp| {
        let mut out = vec![0x5A5Ai16; out_len];
        dsp.residual(size, &s, &p, &mut out, out_stride);
        out
    };
    let same = run_residual(candidate) == run_residual(reference);
    record(report, isa, same, || {
        format!("residual {w}x{h} {pattern:?}")
    });

    let expected = run_residual(reference);
    let expected_sum: u32 = expected
        .chunks(out_stride)
        .map(|row| {
            row[..w]
                .iter()
                .map(|v| u32::from(v.unsigned_abs()))
                .sum::<u32>()
        })
        .sum();
    let mut fused = vec![0x5A5Ai16; out_len];
    let sum = candidate.residual_sad(size, &s, &p, &mut fused, out_stride);
    record(
        report,
        isa,
        fused == expected && sum == expected_sum,
        || format!("residual_sad {w}x{h} {pattern:?}"),
    );

    // Residuals as an inverse transform produces them: mostly small, some
    // large enough to clip either way.
    let residual: Vec<i16> = (0..out_len)
        .map(|_| match rng.below(4) {
            0 => rng.below(4001) as i16 - 2000,
            _ => rng.below(65) as i16 - 32,
        })
        .collect();
    let (start, dst_stride) = destination(w, h, rng);
    let run_add = |dsp: &Dsp| {
        let mut out = start.clone();
        let mut dst = PlaneViewMut::new(&mut out, dst_stride, w, h).expect("buffer fits");
        dsp.add_residual(size, &mut dst, &residual, out_stride);
        out
    };
    let same = run_add(candidate) == run_add(reference);
    record(report, isa, same, || format!("add_residual {w}x{h}"));

    let (got, want) = (
        candidate.variance(size, &s, &p),
        reference.variance(size, &s, &p),
    );
    record(report, isa, got == want, || {
        format!("variance {w}x{h} {pattern:?}: got {got:?}, want {want:?}")
    });

    let (got, want) = (
        candidate.block_sums(size, &s),
        reference.block_sums(size, &s),
    );
    record(report, isa, got == want, || {
        format!("block_sums {w}x{h} {pattern:?}: got {got:?}, want {want:?}")
    });

    let (got, want) = (
        candidate.variance_energy(size, &s, &p),
        reference.variance_energy(size, &s, &p),
    );
    record(report, isa, got == want, || {
        format!("variance_energy {w}x{h} {pattern:?}: got {got:?}, want {want:?}")
    });

    let (xo, yo) = (rng.below(VARIANCE_POSITIONS), rng.below(VARIANCE_POSITIONS));
    let window =
        PlaneView::new(&pred_data[pred.start..], pred.stride, w + 1, h + 1).expect("covers");
    let got = candidate.sub_pixel_variance(size, &window, xo, yo, &s);
    let want = reference.sub_pixel_variance(size, &window, xo, yo, &s);
    record(report, isa, got == want, || {
        format!(
            "sub_pixel_variance {w}x{h} {pattern:?} offsets=({xo},{yo}): got {got:?}, want {want:?}"
        )
    });
}

fn check_transforms(
    candidate: &Dsp,
    reference: &Dsp,
    size: DctSize,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let n = size.side();
    let isa = candidate.isa();
    let stride = n + rng.below(MAX_EXTRA_STRIDE + 1);
    let len = (n - 1) * stride + n;
    let mode = rng.below(4);
    let residual: Vec<i16> = (0..len)
        .map(|i| match mode {
            0 => rng.below(511) as i16 - 255,
            1 => {
                if rng.below(2) == 0 {
                    255
                } else {
                    -255
                }
            }
            2 => {
                if ((i % stride) + i / stride).is_multiple_of(2) {
                    255
                } else {
                    -255
                }
            }
            _ => rng.below(21) as i16 - 10,
        })
        .collect();
    let run_forward = |dsp: &Dsp| {
        let mut out = vec![0i32; n * n];
        dsp.av1_forward_dct(size, &residual, stride, &mut out);
        out
    };
    let same = run_forward(candidate) == run_forward(reference);
    record(report, isa, same, || {
        format!("av1_forward_dct {n}x{n} mode={mode} stride={stride}")
    });

    let coeffs: Vec<i32> = (0..n * n)
        .map(|_| match mode {
            0 => rng.below(4001) as i32 - 2000,
            1 => rng.below(1 << 16) as i32 - (1 << 15),
            2 => rng.below(1 << 22) as i32 - (1 << 21),
            _ => {
                if rng.below(8) == 0 {
                    rng.below(20001) as i32 - 10000
                } else {
                    0
                }
            }
        })
        .collect();
    let (start, dst_stride) = destination(n, n, rng);
    let run_inverse = |dsp: &Dsp| {
        let mut out = start.clone();
        let mut dst = PlaneViewMut::new(&mut out, dst_stride, n, n).expect("buffer fits");
        dsp.av1_inverse_dct_add(size, &coeffs, &mut dst);
        out
    };
    let same = run_inverse(candidate) == run_inverse(reference);
    record(report, isa, same, || {
        format!("av1_inverse_dct_add {n}x{n} mode={mode}")
    });
}

/// The 64x64 transforms, which keep (forward) or take (inverse) only the
/// top-left 32x32 coefficients.
fn check_transforms_64(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    const N: usize = 64;
    const CODED: usize = 32 * 32;
    let isa = candidate.isa();
    let stride = N + rng.below(MAX_EXTRA_STRIDE + 1);
    let mode = rng.below(3);
    let residual: Vec<i16> = (0..(N - 1) * stride + N)
        .map(|_| match mode {
            0 => rng.below(511) as i16 - 255,
            1 => {
                if rng.below(2) == 0 {
                    255
                } else {
                    -255
                }
            }
            _ => rng.below(21) as i16 - 10,
        })
        .collect();
    let run_forward = |dsp: &Dsp| {
        let mut out = vec![0i32; CODED];
        dsp.av1_forward_dct_64(&residual, stride, &mut out);
        out
    };
    let same = run_forward(candidate) == run_forward(reference);
    record(report, isa, same, || {
        format!("av1_forward_dct_64 mode={mode} stride={stride}")
    });

    let coeffs: Vec<i32> = (0..CODED)
        .map(|_| match mode {
            0 => rng.below(4001) as i32 - 2000,
            1 => rng.below(1 << 22) as i32 - (1 << 21),
            _ => {
                if rng.below(8) == 0 {
                    rng.below(20001) as i32 - 10000
                } else {
                    0
                }
            }
        })
        .collect();
    let (start, dst_stride) = destination(N, N, rng);
    let run_inverse = |dsp: &Dsp| {
        let mut out = start.clone();
        let mut dst = PlaneViewMut::new(&mut out, dst_stride, N, N).expect("buffer fits");
        dsp.av1_inverse_dct_add_64(&coeffs, &mut dst);
        out
    };
    let same = run_inverse(candidate) == run_inverse(reference);
    record(report, isa, same, || {
        format!("av1_inverse_dct_add_64 mode={mode}")
    });
}

fn check_quantize(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    let isa = candidate.isa();
    let q = Av1Quantizer {
        dc_step: 4 + rng.below(MAX_STEP - 3) as i32,
        ac_step: 4 + rng.below(MAX_STEP - 3) as i32,
        rounding_q7: rng.below(128) as i32,
        max_level: 1 + rng.below(MAX_LEVEL) as i32,
        log_scale: rng.below(MAX_LOG_SCALE + 1) as u32,
    };
    let len = 1 + rng.below(1024);
    let coeffs: Vec<i32> = (0..len)
        .map(|_| match rng.below(3) {
            0 => rng.below(2 * MAX_COEFF + 1) as i32 - MAX_COEFF as i32,
            1 => rng.below(8 * q.ac_step as usize + 1) as i32 - 4 * q.ac_step,
            _ => 0,
        })
        .collect();
    let run = |dsp: &Dsp| {
        let mut levels = coeffs.clone();
        let nonzero = dsp.av1_quantize(&mut levels, &q);
        (levels, nonzero)
    };
    let same = run(candidate) == run(reference);
    record(report, isa, same, || {
        format!("av1_quantize len={len} {q:?}")
    });

    // Mostly realistic levels, plus any i32 (the dequantiser masks the
    // product, so it must be exact for every level).
    let levels: Vec<i32> = (0..len)
        .map(|_| match rng.below(8) {
            0 => rng.next_u64() as i32,
            _ => rng.below(16383) as i32 - 8191,
        })
        .collect();
    let run_dq = |dsp: &Dsp| {
        let mut out = vec![0i32; len];
        dsp.av1_dequantize(&levels, &q, &mut out);
        out
    };
    let same = run_dq(candidate) == run_dq(reference);
    record(report, isa, same, || {
        format!("av1_dequantize len={len} {q:?}")
    });
    let dequantized = run_dq(reference);
    record(
        report,
        isa,
        candidate.av1_coefficient_gain(&coeffs, &dequantized)
            == reference.av1_coefficient_gain(&coeffs, &dequantized),
        || format!("av1_coefficient_gain len={len}"),
    );
}

/// Coefficient amplitude classes for the large inverse transforms: the
/// guarded 16-bit NEON graphs run when a transform's input L1 norm is small
/// (the realistic classes) and fall back to 32-bit arithmetic otherwise.
#[derive(Debug, Clone, Copy)]
enum InverseAmplitude {
    /// Out-of-range and huge values mixed with moderate ones.
    Extreme,
    /// Typical dequantised residual coefficients.
    Small,
    /// Rows and columns near the 16-bit guard's limit.
    Medium,
}

impl InverseAmplitude {
    const ALL: [Self; 3] = [Self::Extreme, Self::Small, Self::Medium];

    fn sample(self, rng: &mut Rng) -> i32 {
        match self {
            Self::Extreme => match rng.below(4) {
                0 => -32_769,
                1 => 32_768,
                2 => rng.below(1 << 22) as i32 - (1 << 21),
                _ => rng.below(4001) as i32 - 2000,
            },
            Self::Small => {
                if rng.below(3) == 0 {
                    rng.below(257) as i32 - 128
                } else {
                    0
                }
            }
            Self::Medium => rng.below(2801) as i32 - 1400,
        }
    }
}

/// Sparse kernels always compare against the unchanged full scalar
/// transform; at `limit` 32 the candidate's full transform is checked too.
fn check_sparse_inverse(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for side in [32, 64] {
        for limit in [8, 16, 32] {
            for amplitude in InverseAmplitude::ALL {
                let mut coefficients = vec![0_i32; 1024];
                for c in 0..limit {
                    for r in 0..limit {
                        coefficients[c * 32 + r] = amplitude.sample(rng);
                    }
                }
                let (start, stride) = destination(side, side, rng);
                let full = |dsp: &Dsp| {
                    let mut out = start.clone();
                    let mut dst =
                        PlaneViewMut::new(&mut out, stride, side, side).expect("buffer fits");
                    if side == 64 {
                        dsp.av1_inverse_dct_add_64(&coefficients, &mut dst);
                    } else {
                        dsp.av1_inverse_dct_add(
                            DctSize::from_side(side).unwrap(),
                            &coefficients,
                            &mut dst,
                        );
                    }
                    out
                };
                let expected = full(reference);
                let mut actual = start.clone();
                let mut dst =
                    PlaneViewMut::new(&mut actual, stride, side, side).expect("buffer fits");
                candidate.av1_inverse_dct_add_sparse(side, &coefficients, limit, &mut dst);
                record(report, candidate.isa(), actual == expected, || {
                    format!("av1_inverse_dct_add_sparse {side} low{limit} {amplitude:?}")
                });
                if limit == 32 {
                    record(report, candidate.isa(), full(candidate) == expected, || {
                        format!("av1_inverse_dct_add {side}x{side} {amplitude:?}")
                    });
                }
            }
        }
    }
}

/// Exercise both sides of the narrow inverse guard, input clamping, and
/// full-range predictions through the public dispatch API on every ISA, for
/// the full-size 16-bit drivers (8x8 and 16x16).
fn check_inverse_narrow_boundaries(
    candidate: &Dsp,
    reference: &Dsp,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    check_inverse_boundaries::<8>(candidate, reference, rng, report);
    check_inverse_boundaries::<16>(candidate, reference, rng, report);
}

fn check_inverse_boundaries<const SIDE: usize>(
    candidate: &Dsp,
    reference: &Dsp,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    const GUARD: i32 = 23_000;
    const PATTERNS: usize = 4;
    let size = DctSize::from_side(SIDE).expect("square DCT size");
    for magnitude in [GUARD - 1, GUARD, GUARD + 1, i32::from(i16::MAX), i32::MAX] {
        for pattern in 0..PATTERNS {
            let mut coefficients = vec![0; SIDE * SIDE];
            for column in 0..SIDE {
                let value = magnitude / SIDE as i32
                    + if column == 0 {
                        magnitude % SIDE as i32
                    } else {
                        0
                    };
                let value = if (column + pattern).is_multiple_of(2) {
                    value
                } else {
                    -value
                };
                match pattern {
                    0 | 1 => coefficients[column * SIDE] = value,
                    2 => coefficients[column * SIDE + column] = magnitude,
                    _ => coefficients[column * SIDE..(column + 1) * SIDE].fill(-magnitude),
                }
            }
            let (start, stride) = destination(SIDE, SIDE, rng);
            let run = |dsp: &Dsp| {
                let mut output = start.clone();
                let mut dst =
                    PlaneViewMut::new(&mut output, stride, SIDE, SIDE).expect("buffer fits");
                dsp.av1_inverse_dct_add(size, &coefficients, &mut dst);
                output
            };
            record(
                report,
                candidate.isa(),
                run(candidate) == run(reference),
                || format!("av1_inverse{SIDE} boundary magnitude={magnitude} pattern={pattern}"),
            );
        }
    }
}

/// Low-frequency forwards (16, 32 and 64 points) against the reference's
/// full forward with the high frequencies zeroed; guard values past the
/// coded square stay untouched.
fn check_forward_low(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    const N: usize = 64;
    let isa = candidate.isa();
    let stride = N + rng.below(MAX_EXTRA_STRIDE + 1);
    let mode = rng.below(3);
    let residual: Vec<i16> = (0..(N - 1) * stride + N)
        .map(|_| match mode {
            0 => rng.below(511) as i16 - 255,
            1 => {
                if rng.below(2) == 0 {
                    255
                } else {
                    -255
                }
            }
            _ => rng.below(21) as i16 - 10,
        })
        .collect();
    for side in [16, 32, 64] {
        let coded = side.min(32) * side.min(32);
        let run_low = |dsp: &Dsp| {
            let mut out = vec![i32::MIN; coded + 3];
            dsp.av1_forward_dct_low(side, &residual, stride, &mut out[..coded]);
            out
        };
        let same = run_low(candidate) == run_low(reference);
        record(report, isa, same, || {
            format!("av1_forward_dct_low {side} mode={mode} stride={stride}")
        });
    }
    for side in [16, 32] {
        let run_raster = |dsp: &Dsp| {
            let mut out = vec![i32::MIN; side * side + 3];
            dsp.av1_forward_dct_raster_low(side, &residual, stride, &mut out[..side * side]);
            out
        };
        let same = run_raster(candidate) == run_raster(reference);
        record(report, isa, same, || {
            format!("av1_forward_dct_raster_low {side} mode={mode} stride={stride}")
        });
    }
}

/// The out-of-place AV1 quantiser against the reference's in-place one;
/// the source stays unchanged and guard values past the levels untouched.
fn check_quantize_from(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    let isa = candidate.isa();
    let q = Av1Quantizer {
        dc_step: 4 + rng.below(MAX_STEP - 3) as i32,
        ac_step: 4 + rng.below(MAX_STEP - 3) as i32,
        rounding_q7: rng.below(128) as i32,
        max_level: 1 + rng.below(MAX_LEVEL) as i32,
        log_scale: rng.below(MAX_LOG_SCALE + 1) as u32,
    };
    let len = 1 + rng.below(1024);
    let coeffs: Vec<i32> = (0..len)
        .map(|_| match rng.below(3) {
            0 => rng.below(2 * MAX_COEFF + 1) as i32 - MAX_COEFF as i32,
            1 => rng.below(8 * q.ac_step as usize + 1) as i32 - 4 * q.ac_step,
            _ => 0,
        })
        .collect();
    let mut expected = coeffs.clone();
    let expected_nonzero = reference.av1_quantize(&mut expected, &q);
    expected.extend([i32::MIN; 3]);
    let source = coeffs.clone();
    let mut levels = vec![i32::MIN; len + 3];
    let nonzero = candidate.av1_quantize_from(&source, &q, &mut levels[..len]);
    let same = nonzero == expected_nonzero && levels == expected && source == coeffs;
    record(report, isa, same, || {
        format!("av1_quantize_from len={len} {q:?}")
    });
}

/// The bounded raster forwards (16 and 32 points) against the reference:
/// the same decision and coefficients, and a "zero" decision only when the
/// full forward's coefficients are all within the limit.
fn check_forward_bounded(
    candidate: &Dsp,
    reference: &Dsp,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    const N: usize = 32;
    let isa = candidate.isa();
    let stride = N + rng.below(MAX_EXTRA_STRIDE + 1);
    let mode = rng.below(4);
    let residual: Vec<i16> = (0..(N - 1) * stride + N)
        .map(|_| match mode {
            0 => rng.below(511) as i16 - 255,
            1 => {
                if rng.below(2) == 0 {
                    255
                } else {
                    -255
                }
            }
            2 => rng.below(21) as i16 - 10,
            _ => rng.below(5) as i16 - 2,
        })
        .collect();
    let zero_limit = rng.below(4096) as u32;
    for size in [DctSize::N16, DctSize::N32] {
        let area = size.side() * size.side();
        let run = |dsp: &Dsp| {
            let mut out = vec![i32::MIN; area + 3];
            let coded = dsp.av1_forward_dct_raster_unless_zero(
                size,
                &residual,
                stride,
                zero_limit,
                &mut out[..area],
            );
            (coded, out)
        };
        let (coded, out) = run(candidate);
        let mut full = vec![0i32; area];
        reference.av1_forward_dct_raster(size, &residual, stride, &mut full);
        let sound = coded || full.iter().all(|c| c.unsigned_abs() <= zero_limit);
        let exact = !coded || out[..area] == full[..];
        let same = (coded, out) == run(reference) && sound && exact;
        record(report, isa, same, || {
            format!("av1_forward_dct_raster_unless_zero {size:?} mode={mode} limit={zero_limit}")
        });
    }
}

/// Wide tiles retain the full block's tap selection and source origin. The
/// scalar Dsp deliberately uses the independent full-block equation here.
fn check_av1_wide_prediction(
    candidate: &Dsp,
    reference: &Dsp,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    use nc_dsp::av1_mc::{
        convolve::Av1McAxis,
        filter::{Av1Filter, Av1Filters},
    };
    for (w, h) in [
        (128, 128),
        (128, 64),
        (64, 128),
        (128, 4),
        (4, 128),
        (128, 8),
        (8, 128),
        (128, 32),
        (32, 128),
        (128, 96),
        (2, 128),
    ] {
        let ss = w + 12;
        let mut src = vec![0u8; (h + 6) * ss + w + 7];
        for impulse in [false, true] {
            if impulse {
                src.fill(0);
                src[(h / 2 + 3) * ss + w / 2 + 3] = 255;
            } else {
                rng.fill(&mut src);
            }
            let window = PlaneView::new(&src, ss, w + 7, h + 7).expect("window");
            for filter in [Av1Filter::Regular, Av1Filter::Smooth, Av1Filter::Sharp] {
                for (fx, fy) in [(0, 0), (1, 0), (0, 15), (7, 8), (15, 1)] {
                    let mut x = Av1McAxis::unscaled(fx);
                    let mut y = Av1McAxis::unscaled(fy);
                    x.position += 31;
                    y.position += 17;
                    let ds = w + 13;
                    let run = |dsp: &Dsp| {
                        let mut out = vec![0xA5u8; (h - 1) * ds + w];
                        let mut view = PlaneViewMut::new(&mut out, ds, w, h).expect("output");
                        dsp.av1_predict(
                            Av1Filters::both(filter),
                            &window,
                            (w, h),
                            (x, y),
                            &mut view,
                        );
                        out
                    };
                    record(
                        report,
                        candidate.isa(),
                        run(candidate) == run(reference),
                        || {
                            format!(
                                "AV1 wide put {w}x{h} {filter:?} phase=({fx},{fy}) impulse={impulse}"
                            )
                        },
                    );
                }
            }
        }
    }
}

/// Exercise every HEVC fraction, including negative lobes and unrounded 2-D
/// outputs. Tight final rows and sentinel row gaps catch vector store overruns.
/// HEVC default bi-prediction ([`Dsp::hevc_predict_bi`]) against the
/// reference ISA for every filter, width, phase pair and a range of
/// heights, over byte patterns that include the filters' extreme
/// overshoot, with first predictions across the whole centred range; the
/// reference itself must equal the unfused path (prep, then
/// [`Dsp::hevc_weighted_prediction_centered`] with default weights).
fn check_hevc_bi(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    /// Centred HEVC predictions lie in -25022..=24958 at 8-bit depth.
    const FIRST_RANGE: usize = 25022 + 24958 + 1;
    const GUARD: u8 = 0xA5;
    for filter in [Filter::HevcLuma, Filter::HevcChroma] {
        for width in [4, 8, 16, 32, 64] {
            for height in [4, 8, 16, 64] {
                let size = BlockSize::new(width, height).expect("valid shape");
                let taps = filter.taps();
                let stride = width + taps + rng.below(4);
                let rows = height + taps - 1;
                let extreme = rng.below(2) == 0;
                let source: Vec<u8> = (0..(rows - 1) * stride + width + taps - 1)
                    .map(|i| {
                        if extreme {
                            if (i + i / stride * 3) % 7 < 3 { 255 } else { 0 }
                        } else {
                            rng.below(256) as u8
                        }
                    })
                    .collect();
                let window = PlaneView::new(&source, stride, width + taps - 1, rows)
                    .expect("prediction window");
                let first_stride = width + rng.below(3);
                let first: Vec<i16> = (0..(height - 1) * first_stride + width)
                    .map(|_| (rng.below(FIRST_RANGE) as i32 - 25022) as i16)
                    .collect();
                let out_stride = width + 5;
                for fx in 0..filter.positions() {
                    for fy in 0..filter.positions() {
                        let run = |dsp: &Dsp| {
                            let mut out = vec![GUARD; (height - 1) * out_stride + width];
                            let mut view =
                                PlaneViewMut::new(&mut out, out_stride, width, height).unwrap();
                            dsp.hevc_predict_bi(
                                filter,
                                size,
                                &window,
                                (fx, fy),
                                (&first, first_stride),
                                &mut view,
                            );
                            out
                        };
                        let got = run(reference);
                        record(report, candidate.isa(), run(candidate) == got, || {
                            format!("HEVC bi {filter:?} {width}x{height} ({fx},{fy})")
                        });
                        // The unfused path, on the reference ISA.
                        let mut second = vec![0i16; width * height];
                        reference.predict_compound(
                            filter,
                            size,
                            &window,
                            (fx, fy),
                            &mut second,
                            width,
                        );
                        let packed: Vec<i16> = (0..height)
                            .flat_map(|r| {
                                first[r * first_stride..r * first_stride + width].to_vec()
                            })
                            .collect();
                        let mut want = vec![GUARD; (height - 1) * out_stride + width];
                        let mut view =
                            PlaneViewMut::new(&mut want, out_stride, width, height).unwrap();
                        let weights = nc_dsp::HevcWeights {
                            w0: 1,
                            w1: 1,
                            round: 64,
                            shift: 7,
                        };
                        reference.hevc_weighted_prediction_centered(
                            weights, &packed, &second, &mut view,
                        );
                        record(report, reference.isa(), got == want, || {
                            format!(
                                "HEVC bi reference vs unfused {filter:?} {width}x{height} ({fx},{fy})"
                            )
                        });
                    }
                }
            }
        }
    }
}

fn check_hevc_prep_phases(candidate: &Dsp, reference: &Dsp, report: &mut CheckReport) {
    const HEIGHT: usize = 8;
    const GUARD: i16 = 0x5A5A;
    for filter in [Filter::HevcLuma, Filter::HevcChroma] {
        for width in [4, 8, 16, 32, 64] {
            let size = BlockSize::new(width, HEIGHT).expect("valid shape");
            let stride = width + filter.taps() + 2;
            let rows = HEIGHT + filter.taps() - 1;
            let source: Vec<u8> = (0..(rows - 1) * stride + width + filter.taps() - 1)
                .map(|i| if (i + i / stride * 3) % 7 < 3 { 255 } else { 0 })
                .collect();
            let window = PlaneView::new(&source, stride, width + filter.taps() - 1, rows)
                .expect("tight prediction window");
            let out_stride = width + 3;
            for fx in 0..filter.positions() {
                for fy in 0..filter.positions() {
                    let run = |dsp: &Dsp| {
                        let mut out = vec![GUARD; (HEIGHT - 1) * out_stride + width];
                        dsp.predict_compound(filter, size, &window, (fx, fy), &mut out, out_stride);
                        out
                    };
                    record(
                        report,
                        candidate.isa(),
                        run(candidate) == run(reference),
                        || format!("HEVC prep all phases {filter:?} {width}x{HEIGHT} ({fx},{fy})"),
                    );
                }
            }
        }
    }
}

/// Every sub-sample position of every H.264 block shape through
/// [`Dsp::h264_predictor`] (kernels resolved per shape and phase), on tight
/// random windows (the window ends at its last sample) and destinations.
fn check_h264_predictor_phases(
    candidate: &Dsp,
    reference: &Dsp,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    const LUMA: [(usize, usize); 7] = [(16, 16), (16, 8), (8, 16), (8, 8), (8, 4), (4, 8), (4, 4)];
    const CHROMA: [(usize, usize); 5] = [(8, 8), (8, 4), (4, 8), (4, 4), (16, 16)];
    for (filter, shapes) in [
        (Filter::H264Luma, &LUMA[..]),
        (Filter::H264Chroma, &CHROMA[..]),
    ] {
        for &(w, h) in shapes {
            let size = BlockSize::new(w, h).expect("valid shape");
            let extra = filter.taps() - 1;
            let window_stride = w + extra + rng.below(MAX_EXTRA_STRIDE + 1);
            let mut source = vec![0u8; (h + extra - 1) * window_stride + w + extra];
            rng.fill(&mut source);
            let (start, stride) = destination(w, h, rng);
            let (cand, refr) = (
                candidate.h264_predictor(filter, size),
                reference.h264_predictor(filter, size),
            );
            for fx in 0..filter.positions() {
                for fy in 0..filter.positions() {
                    let run = |predictor: &nc_dsp::H264Predictor| {
                        let mut out = start.clone();
                        predictor.predict(&source, window_stride, fx, fy, &mut out, stride);
                        out
                    };
                    record(report, candidate.isa(), run(&cand) == run(&refr), || {
                        format!("h264_predictor {filter:?} {w}x{h} frac=({fx},{fy})")
                    });
                }
            }
        }
    }
}
