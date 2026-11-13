//! Bit-exactness checks for AV1 inter prediction through its general entry
//! points (`Dsp::av1_predict`, `Dsp::av1_put` and
//! `Dsp::av1_predict_compound`): scaled, at arbitrary `Av1McAxis` steps,
//! and unscaled with every pair of filter families per direction
//! (`dual_filter`, the mixed-family kernels) and 128-sample edges (tiled
//! kernels); and arbitrary-ratio polyphase resampling
//! (`Dsp::resample_plane` / `Dsp::resample_plane_with`).

use nc_common::{PlaneView, PlaneViewMut};
use nc_dsp::av1_mc::convolve::{AV1_MAX_STEP, AV1_UNSCALED_STEP, Av1McAxis};
use nc_dsp::av1_mc::filter::{Av1Filter, Av1Filters};
use nc_dsp::{Dsp, PlaneResampler};

use crate::check::{CheckReport, MAX_EXTRA_STRIDE, PATTERNS, TestBlock, fill};
use crate::rng::Rng;

const SCALED_SIZES: [(usize, usize); 22] = [
    (2, 2),
    (2, 4),
    (4, 2),
    (4, 4),
    (4, 8),
    (8, 4),
    (4, 16),
    (16, 4),
    (8, 8),
    (8, 16),
    (16, 8),
    (12, 8),
    (24, 16),
    (16, 16),
    (32, 16),
    (16, 64),
    (64, 16),
    (32, 32),
    (64, 64),
    (128, 64),
    (128, 128),
    (7, 5),
];

/// Unscaled block sizes: every kernel width and height class, the 128-sample
/// edges, the 2-sample edges of sub-8x8 chroma (padded to the 4-sample
/// kernels) and a shape no kernel covers.
const UNSCALED_SIZES: [(usize, usize); 20] = [
    (4, 4),
    (4, 8),
    (8, 4),
    (4, 16),
    (16, 4),
    (8, 8),
    (8, 32),
    (32, 8),
    (16, 16),
    (16, 64),
    (64, 16),
    (64, 64),
    (128, 64),
    (64, 128),
    (128, 128),
    (2, 2),
    (2, 4),
    (4, 2),
    (2, 8),
    (8, 2),
];

/// Fixed steps covering 2x upscale (`512`), 1.5x upscale (`683`, 480p->720p),
/// 1.25x upscale (`819`), 4x upscale (`256`), 1.5x downscale (`1536`) and the
/// normative ceiling (`AV1_MAX_STEP = 2048`).
const FIXED_STEPS: [u32; 6] = [256, 512, 683, 819, 1536, AV1_MAX_STEP];

const RESAMPLE_PAIRS: [((usize, usize), (usize, usize)); 8] = [
    ((128, 72), (86, 48)),
    ((64, 36), (43, 24)),
    ((80, 48), (64, 38)),
    ((96, 54), (48, 27)),
    ((40, 24), (60, 36)),
    ((33, 19), (17, 11)),
    ((16, 12), (16, 12)),
    ((3, 2), (2, 3)),
];

/// Production 720p<->480p resampler sizes exercised once per `nc-checkasm` run.
const FULL_RESAMPLE_PAIRS: [((usize, usize), (usize, usize)); 2] =
    [((1280, 720), (854, 480)), ((854, 480), (1280, 720))];

pub(crate) fn check(
    candidate: &Dsp,
    reference: &Dsp,
    iterations: usize,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    check_scaled_prediction(candidate, reference, iterations, rng, report);
    check_unscaled_prediction(candidate, reference, iterations, rng, report);
    check_resample(candidate, reference, iterations, rng, report);
}

fn check_scaled_prediction(
    candidate: &Dsp,
    reference: &Dsp,
    iterations: usize,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    for &(w, h) in &SCALED_SIZES {
        let rounds = if w == 128 {
            FIXED_STEPS.len()
        } else {
            iterations.clamp(FIXED_STEPS.len(), 16)
        };
        for round in 0..rounds {
            let x = pick_axis(round, rng);
            let y = pick_axis(round + 3, rng);
            let (span_x, span_y) = (x.span(w), y.span(h));
            let pattern = PATTERNS[round % PATTERNS.len()];
            let win_block = TestBlock::new(span_x, span_y, 1, rng);
            let dummy = TestBlock::new(span_x, span_y, 1, rng);
            let (win_data, _) = fill(pattern, &win_block, &dummy, rng);
            let window = win_block.view(&win_data, 0);

            let filters = Av1Filters {
                x: Av1Filter::ALL[rng.below(Av1Filter::ALL.len())],
                y: Av1Filter::ALL[rng.below(Av1Filter::ALL.len())],
            };
            let dst_stride = w + rng.below(MAX_EXTRA_STRIDE + 1);
            let dst_len = (h - 1) * dst_stride + w;

            let run_u8 = |dsp: &Dsp| {
                let mut out = vec![0xA5u8; dst_len];
                let mut view =
                    PlaneViewMut::new(&mut out, dst_stride, w, h).expect("destination fits");
                dsp.av1_predict(filters, &window, (w, h), (x, y), &mut view);
                out
            };
            let (got_u8, want_u8) = (run_u8(candidate), run_u8(reference));
            report.cases += 1;
            if got_u8 != want_u8 {
                report.failures.push(format!(
                    "{} av1_predict scaled {w}x{h} {filters:?} x={x:?} y={y:?} {pattern:?}: outputs differ",
                    candidate.isa()
                ));
            }

            let run_i16 = |dsp: &Dsp| {
                let mut out = vec![0x55AAu16 as i16; dst_len];
                dsp.av1_predict_compound(filters, &window, (w, h), (x, y), &mut out, dst_stride);
                out
            };
            let (got_i16, want_i16) = (run_i16(candidate), run_i16(reference));
            report.cases += 1;
            if got_i16 != want_i16 {
                report.failures.push(format!(
                    "{} av1_predict_compound scaled {w}x{h} {filters:?} x={x:?} y={y:?} {pattern:?}: outputs differ",
                    candidate.isa()
                ));
            }
        }
    }
}

/// Unscaled prediction of every filter-family pair: whole-sample,
/// one-direction and two-direction phases, then random ones.
fn check_unscaled_prediction(
    candidate: &Dsp,
    reference: &Dsp,
    iterations: usize,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    const FIXED_PHASES: [(usize, usize); 4] = [(0, 0), (5, 0), (0, 11), (8, 8)];
    let phases = 16;
    for &(w, h) in &UNSCALED_SIZES {
        for fx in Av1Filter::ALL {
            for fy in Av1Filter::ALL {
                let filters = Av1Filters { x: fx, y: fy };
                let rounds = FIXED_PHASES.len() + iterations.min(4);
                for round in 0..rounds {
                    let (frac_x, frac_y) = FIXED_PHASES
                        .get(round)
                        .copied()
                        .unwrap_or_else(|| (rng.below(phases), rng.below(phases)));
                    let axes = (Av1McAxis::unscaled(frac_x), Av1McAxis::unscaled(frac_y));
                    let (span_x, span_y) = (axes.0.span(w), axes.1.span(h));
                    let pattern = PATTERNS[round % PATTERNS.len()];
                    let win_block = TestBlock::new(span_x, span_y, 1, rng);
                    let dummy = TestBlock::new(span_x, span_y, 1, rng);
                    let (win_data, _) = fill(pattern, &win_block, &dummy, rng);
                    let window = win_block.view(&win_data, 0);
                    let dst_stride = w + rng.below(MAX_EXTRA_STRIDE + 1);
                    let dst_len = (h - 1) * dst_stride + w;
                    let run_u8 = |dsp: &Dsp, put: bool| {
                        let mut out = vec![0xA5u8; dst_len];
                        let mut view = PlaneViewMut::new(&mut out, dst_stride, w, h)
                            .expect("destination fits");
                        if put {
                            dsp.av1_put(filters, &window, (w, h), (frac_x, frac_y), &mut view);
                        } else {
                            dsp.av1_predict(filters, &window, (w, h), axes, &mut view);
                        }
                        out
                    };
                    let want = run_u8(reference, false);
                    for (put, name) in [(false, "av1_predict"), (true, "av1_put")] {
                        report.cases += 1;
                        if run_u8(candidate, put) != want {
                            report.failures.push(format!(
                                "{} {name} unscaled {w}x{h} {filters:?} phase ({frac_x}, {frac_y}) {pattern:?}: outputs differ",
                                candidate.isa()
                            ));
                        }
                    }
                    let run_i16 = |dsp: &Dsp| {
                        let mut out = vec![0x55AAu16 as i16; dst_len];
                        dsp.av1_predict_compound(
                            filters,
                            &window,
                            (w, h),
                            axes,
                            &mut out,
                            dst_stride,
                        );
                        out
                    };
                    report.cases += 1;
                    if run_i16(candidate) != run_i16(reference) {
                        report.failures.push(format!(
                            "{} av1_predict_compound unscaled {w}x{h} {filters:?} phase ({frac_x}, {frac_y}) {pattern:?}: outputs differ",
                            candidate.isa()
                        ));
                    }
                }
            }
        }
    }
}

fn pick_axis(round: usize, rng: &mut Rng) -> Av1McAxis {
    let step = if round < FIXED_STEPS.len() {
        FIXED_STEPS[round]
    } else {
        1 + rng.below(AV1_MAX_STEP as usize) as u32
    };
    let position = match round % 4 {
        0 => 0,
        1 => AV1_UNSCALED_STEP - 1,
        _ => rng.below(AV1_UNSCALED_STEP as usize) as u32,
    };
    Av1McAxis { position, step }
}

fn check_resample(
    candidate: &Dsp,
    reference: &Dsp,
    iterations: usize,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    for &(in_size, out_size) in &RESAMPLE_PAIRS {
        let mut plan = PlaneResampler::new(in_size, out_size);
        for &pattern in &PATTERNS {
            check_one_resample(candidate, reference, &mut plan, pattern, rng, report);
        }
    }
    for (i, &(in_size, out_size)) in FULL_RESAMPLE_PAIRS.iter().enumerate() {
        let mut plan = PlaneResampler::new(in_size, out_size);
        let pattern = PATTERNS[i % PATTERNS.len()];
        check_one_resample(candidate, reference, &mut plan, pattern, rng, report);
    }
    let random_rounds = iterations.clamp(8, 32);
    for i in 0..random_rounds {
        let in_size = (1 + rng.below(96), 1 + rng.below(40));
        let out_size = (1 + rng.below(96), 1 + rng.below(40));
        let mut plan = PlaneResampler::new(in_size, out_size);
        let pattern = PATTERNS[i % PATTERNS.len()];
        check_one_resample(candidate, reference, &mut plan, pattern, rng, report);
    }
}

fn check_one_resample(
    candidate: &Dsp,
    reference: &Dsp,
    plan: &mut PlaneResampler,
    pattern: crate::check::Pattern,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let (in_w, in_h) = plan.in_size();
    let (out_w, out_h) = plan.out_size();
    let src_block = TestBlock::new(in_w, in_h, 1, rng);
    let dummy = TestBlock::new(in_w, in_h, 1, rng);
    let (src_data, _) = fill(pattern, &src_block, &dummy, rng);
    let src_view: PlaneView<'_> = src_block.view(&src_data, 0);
    let dst_stride = out_w + rng.below(MAX_EXTRA_STRIDE + 1);
    let dst_len = (out_h - 1) * dst_stride + out_w;

    let mut run = |dsp: &Dsp| {
        let mut out = vec![0xA5u8; dst_len];
        let mut view =
            PlaneViewMut::new(&mut out, dst_stride, out_w, out_h).expect("destination fits");
        dsp.resample_plane_with(plan, &src_view, &mut view);
        out
    };
    let (got, want) = (run(candidate), run(reference));
    report.cases += 1;
    if got != want {
        report.failures.push(format!(
            "{} resample_plane {in_w}x{in_h}->{out_w}x{out_h} {pattern:?} src_stride={} dst_stride={dst_stride}: outputs differ",
            candidate.isa(),
            src_block.stride
        ));
    }
}
