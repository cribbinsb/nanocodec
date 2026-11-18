//! The `u16` kernels of `Dsp::at` (`nc_dsp` `high_depth`) against the
//! scalar table at 10 bits: interpolation (uni, centred, default bi) for
//! every filter family the table has and every width, weighted
//! prediction, the HEVC inverse transform add with random extents, and
//! AV1 unscaled prediction (final and compound precision) of every kernel
//! side and filter pair, AV1 CDEF of a picture (interior and edge
//! blocks), and VP9 prediction (every filter, width and phase, written and
//! averaged) and deblocking (every width and direction).
//! Samples are random and extreme (0 and 1023 runs, the filters' largest
//! overshoot).
use crate::{check::CheckReport, rng::Rng};
use nc_common::{BitDepth, Picture, Plane, PlaneIndex, PlaneView, PlaneViewMut};
use nc_dsp::av1_film_grain::{Av1FilmGrainParams, Av1ScalingPoints};
use nc_dsp::av1_mc::blend::{Av1CompoundPred, Av1Mask, Av1ObmcEdge, av1_distance_weights};
use nc_dsp::av1_mc::convolve::Av1McAxis;
use nc_dsp::av1_mc::filter::{Av1Filter, Av1Filters};
use nc_dsp::av1_mc::warp::av1_setup_shear;
use nc_dsp::av1_restoration::{
    Av1LoopRestoration, Av1LrFilter, Av1LrScratch, Av1SgrProj, Av1StripeBoundaries, Av1Wiener,
    av1_lr_unit_count,
};
use nc_dsp::intra_prediction::av1_block::{BlockEdges, EdgeAvail, EdgeFilterCtx, IntraKind};
use nc_dsp::intra_prediction::filter::FilterIntraMode;
use nc_dsp::{
    Av1EdgeFilter, Av1FilterLength, Av1TxSize, Av1TxType, BlockSize, CdefLines, CdefSets,
    CdefStrengths, CdefUnit, DepthKernels, Dsp, EdgeRun, Filter, HevcTransform, HevcWeights,
    VP9_TAPS, Vp9Filter, Vp9FilterLimits, Vp9FilterWidth,
};

/// Depths checked for AV1 high-bit-depth kernels (10-bit and 12-bit).
fn av1_depths() -> [BitDepth; 2] {
    [BitDepth::TEN, BitDepth::new(12).expect("12-bit")]
}
/// The depth checked for HEVC/VVC Main 10 kernels.
const DEPTH: BitDepth = BitDepth::TEN;
/// Block sides tried.
const SIDES: [usize; 5] = [4, 8, 16, 32, 64];
/// Filters with `u16` kernels.
const FILTERS: [Filter; 5] = [
    Filter::HevcLuma,
    Filter::HevcChroma,
    Filter::VvcLuma,
    Filter::VvcAffineLuma,
    Filter::VvcChroma,
];
/// Rounds per shape.
const ROUNDS: usize = 3;

fn samples_at(depth: BitDepth, len: usize, round: usize, rng: &mut Rng) -> Vec<u16> {
    let max = depth.max() as usize;
    (0..len)
        .map(|i| match round % 4 {
            0 => rng.below(max + 1) as u16,
            1 => {
                if (i / 3).is_multiple_of(2) {
                    max as u16
                } else {
                    0
                }
            }
            2 => {
                if rng.below(2) == 0 {
                    max as u16
                } else {
                    0
                }
            }
            _ => ((i * 97 + rng.below(17)) % (max + 1)) as u16,
        })
        .collect()
}

fn samples(len: usize, round: usize, rng: &mut Rng) -> Vec<u16> {
    samples_at(DEPTH, len, round, rng)
}

/// AV1 block sides with `u16` kernels (2 narrow, 4..=64 direct, 128 tiled).
const AV1_SIDES: [usize; 7] = [2, 4, 8, 16, 32, 64, 128];
/// Taps around an AV1 block (the window is the block plus seven).
const AV1_EXTRA: usize = 7;

/// AV1 unscaled, scaled, warp, and compound/OBMC/inter-intra blends at 10 & 12 bits.
#[allow(clippy::too_many_lines)]
fn check_av1_mc(
    depth: BitDepth,
    (cand, refer): (DepthKernels<'_>, DepthKernels<'_>),
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let pick = |rng: &mut Rng| Av1Filter::ALL[rng.below(Av1Filter::ALL.len())];
    for (i, &w) in AV1_SIDES.iter().enumerate() {
        let mut h = AV1_SIDES[(i + rng.below(3)) % AV1_SIDES.len()];
        if w.min(h) == 2 && w.max(h) > 16 {
            h = 8;
        }
        if (w, h) == (2, 2) {
            h = 4;
        }
        let (span_w, span_h) = (w + AV1_EXTRA, h + AV1_EXTRA);
        for mode in 0..4 {
            let source = samples_at(depth, span_w * span_h, mode, rng);
            let window = PlaneView::new(&source, span_w, span_w, span_h).expect("window");
            let filters = Av1Filters {
                x: pick(rng),
                y: pick(rng),
            };
            let (fx, fy) = match mode {
                0 => (0, 0),
                1 => (1 + rng.below(15), 0),
                2 => (0, 1 + rng.below(15)),
                _ => (1 + rng.below(15), 1 + rng.below(15)),
            };
            let axes = (Av1McAxis::unscaled(fx), Av1McAxis::unscaled(fy));
            let mut put = [vec![0u16; w * h], vec![0u16; w * h]];
            let mut prep = [vec![0i16; w * h], vec![0i16; w * h]];
            for (k, kernels) in [cand, refer].into_iter().enumerate() {
                if w >= 4 && h >= 4 {
                    kernels.av1_put_samples(
                        filters,
                        (&source, span_w),
                        (w, h),
                        (fx, fy),
                        (&mut put[k], w),
                    );
                    kernels.av1_prep_samples(
                        filters,
                        (&source, span_w),
                        (w, h),
                        (fx, fy),
                        (&mut prep[k], w),
                    );
                } else {
                    let mut view = PlaneViewMut::new(&mut put[k], w, w, h).expect("view");
                    kernels.av1_predict(filters, &window, (w, h), axes, &mut view);
                    kernels.av1_predict_compound(filters, &window, (w, h), axes, (&mut prep[k], w));
                }
            }
            report.cases += 2;
            if put[0] != put[1] || prep[0] != prep[1] {
                report.failures.push(format!(
                    "u16 AV1 unscaled {depth:?} {w}x{h} {filters:?} ({fx},{fy}): differs"
                ));
            }
        }
    }
    // Scaled MC + compound blends + diff-weighted mask + mask blend + interintra + OBMC.
    for &(w, h) in &[(4, 4), (8, 8), (12, 8), (16, 16), (32, 16), (64, 32)] {
        let ax = Av1McAxis {
            position: rng.below(1024) as u32,
            step: (512 + rng.below(1537)) as u32,
        };
        let ay = Av1McAxis {
            position: rng.below(1024) as u32,
            step: (512 + rng.below(1537)) as u32,
        };
        let (sw, sh) = (ax.span(w), ay.span(h));
        let src0 = samples_at(depth, sw * sh, 0, rng);
        let src1 = samples_at(depth, sw * sh, 2, rng);
        let win0 = PlaneView::new(&src0, sw, sw, sh).expect("win0");
        let win1 = PlaneView::new(&src1, sw, sw, sh).expect("win1");
        let filters = Av1Filters {
            x: pick(rng),
            y: pick(rng),
        };
        let mut s_put = [vec![0u16; w * h], vec![0u16; w * h]];
        let mut p0 = [vec![0i16; w * h], vec![0i16; w * h]];
        let mut p1 = [vec![0i16; w * h], vec![0i16; w * h]];
        for (k, kernels) in [cand, refer].into_iter().enumerate() {
            let mut v = PlaneViewMut::new(&mut s_put[k], w, w, h).expect("v");
            kernels.av1_predict(filters, &win0, (w, h), (ax, ay), &mut v);
            kernels.av1_predict_compound(filters, &win0, (w, h), (ax, ay), (&mut p0[k], w));
            kernels.av1_predict_compound(filters, &win1, (w, h), (ax, ay), (&mut p1[k], w));
        }
        report.cases += 3;
        if s_put[0] != s_put[1] || p0[0] != p0[1] || p1[0] != p1[1] {
            report.failures.push(format!(
                "u16 AV1 scaled {depth:?} {w}x{h} {filters:?}: differs"
            ));
        }
        let cp0 = Av1CompoundPred {
            data: &p0[1],
            stride: w,
        };
        let cp1 = Av1CompoundPred {
            data: &p1[1],
            stride: w,
        };
        let mut avg = [vec![0u16; w * h], vec![0u16; w * h]];
        let mut dist = [vec![0u16; w * h], vec![0u16; w * h]];
        let mut dmask = [vec![0u8; w * h], vec![0u8; w * h]];
        let weights = av1_distance_weights(1 + rng.below(20) as i32, 1 + rng.below(20) as i32);
        let inv = rng.below(2) == 1;
        for (k, kernels) in [cand, refer].into_iter().enumerate() {
            let mut va = PlaneViewMut::new(&mut avg[k], w, w, h).expect("va");
            kernels.av1_average_compound((&p0[1], &p1[1]), (w, h), &mut va);
            let mut vd = PlaneViewMut::new(&mut dist[k], w, w, h).expect("vd");
            kernels.av1_blend_distance((cp0, cp1), weights, (w, h), &mut vd);
            kernels.av1_difference_weight_mask((cp0, cp1), inv, (w, h), &mut dmask[k]);
        }
        report.cases += 3;
        if avg[0] != avg[1] || dist[0] != dist[1] || dmask[0] != dmask[1] {
            report.failures.push(format!(
                "u16 AV1 compound avg/dist/diffmask {depth:?} {w}x{h}: differs"
            ));
        }
        // Mask blend (all 4 sub_x/sub_y combinations) + interintra + OBMC.
        let mask_raw: Vec<u8> = (0..4 * w * h).map(|_| rng.below(65) as u8).collect();
        for (sx, sy) in [(false, false), (true, false), (false, true), (true, true)] {
            let mstride = if sx { 2 * w } else { w };
            let mask = Av1Mask {
                data: &mask_raw,
                stride: mstride,
                sub_x: sx,
                sub_y: sy,
            };
            let mut m_out = [vec![0u16; w * h], vec![0u16; w * h]];
            let intra_s = samples_at(depth, w * h, 1, rng);
            let inter_s = samples_at(depth, w * h, 2, rng);
            let intra_v = PlaneView::new(&intra_s, w, w, h).expect("intra");
            let mut ii_out = [inter_s.clone(), inter_s];
            for (k, kernels) in [cand, refer].into_iter().enumerate() {
                let mut vm = PlaneViewMut::new(&mut m_out[k], w, w, h).expect("vm");
                kernels.av1_blend_mask((cp0, cp1), mask, (w, h), &mut vm);
                let mut vii = PlaneViewMut::new(&mut ii_out[k], w, w, h).expect("vii");
                kernels.av1_blend_interintra(&intra_v, mask, (w, h), &mut vii);
            }
            report.cases += 2;
            if m_out[0] != m_out[1] || ii_out[0] != ii_out[1] {
                report.failures.push(format!(
                    "u16 AV1 mask/interintra {depth:?} {w}x{h} sub ({sx},{sy}): differs"
                ));
            }
        }
        if w.is_power_of_two() && w <= 32 && h.is_power_of_two() && h <= 32 {
            let obmc_s = samples_at(depth, w * h, 0, rng);
            let base_s = samples_at(depth, w * h, 3, rng);
            let obmc_v = PlaneView::new(&obmc_s, w, w, h).expect("obmc");
            for edge in [Av1ObmcEdge::Above, Av1ObmcEdge::Left] {
                let mut o_out = [base_s.clone(), base_s.clone()];
                for (k, kernels) in [cand, refer].into_iter().enumerate() {
                    let mut vo = PlaneViewMut::new(&mut o_out[k], w, w, h).expect("vo");
                    kernels.av1_obmc_blend(&obmc_v, edge, (w, h), &mut vo);
                }
                report.cases += 1;
                if o_out[0] != o_out[1] {
                    report
                        .failures
                        .push(format!("u16 AV1 OBMC {depth:?} {w}x{h} {edge:?}: differs"));
                }
            }
        }
    }
    // Affine 8x8 warp (put & compound).
    let (rw, rh) = (48, 48);
    let ref_s = samples_at(depth, rw * rh, 0, rng);
    let ref_v = PlaneView::new(&ref_s, rw, rw, rh).expect("ref");
    let model = [
        (rng.below(4096) as i32 - 2048) * 8,
        (rng.below(4096) as i32 - 2048) * 8,
        (1 << 16) + (rng.below(512) as i32 - 256) * 8,
        (rng.below(512) as i32 - 256) * 8,
        (rng.below(512) as i32 - 256) * 8,
        (1 << 16) + (rng.below(512) as i32 - 256) * 8,
    ];
    if let Some(shear) = av1_setup_shear(&model) {
        let (bw, bh) = (16, 16);
        let mut w_put = [vec![0u16; bw * bh], vec![0u16; bw * bh]];
        let mut w_prep = [vec![0i16; bw * bh], vec![0i16; bw * bh]];
        for (k, kernels) in [cand, refer].into_iter().enumerate() {
            let mut vw = PlaneViewMut::new(&mut w_put[k], bw, bw, bh).expect("vw");
            kernels.av1_warp(
                &ref_v,
                (&model, shear),
                (8, 8),
                (bw, bh),
                (false, false),
                &mut vw,
            );
            kernels.av1_warp_compound(
                &ref_v,
                (&model, shear),
                (8, 8),
                (bw, bh),
                (false, false),
                (&mut w_prep[k], bw),
            );
        }
        report.cases += 2;
        if w_put[0] != w_put[1] || w_prep[0] != w_prep[1] {
            report
                .failures
                .push(format!("u16 AV1 warp {depth:?}: differs"));
        }
    }
}

/// VP9 interpolation filters.
const VP9_FILTERS: [Vp9Filter; 4] = [
    Vp9Filter::Regular,
    Vp9Filter::Smooth,
    Vp9Filter::Sharp,
    Vp9Filter::Bilinear,
];

/// VP9 prediction at 10 bits (whole-row kernels of every width; the
/// height and phases random, zero phases included).
fn check_vp9_mc(
    (cand, refer): (DepthKernels<'_>, DepthKernels<'_>),
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    for kind in VP9_FILTERS {
        for (i, &w) in SIDES.iter().enumerate() {
            let h = SIDES[(i + rng.below(3)) % SIDES.len()];
            let (span_w, span_h) = (w + VP9_TAPS - 1, h + VP9_TAPS - 1);
            for round in 0..ROUNDS {
                let source = samples(span_w * span_h, round, rng);
                let phase = (rng.below(16) * rng.below(2), rng.below(16) * rng.below(2));
                let base = samples(w * h, round, rng);
                let average = rng.below(2) == 0;
                let mut out = [base.clone(), base];
                for (k, kernels) in [cand, refer].into_iter().enumerate() {
                    kernels.vp9_convolve(
                        kind,
                        (&source, span_w),
                        phase,
                        (w, h),
                        (&mut out[k], w),
                        average,
                    );
                }
                report.cases += 1;
                if out[0] != out[1] {
                    report.failures.push(format!(
                        "u16 VP9 {kind:?} {w}x{h} {phase:?} average {average}: prediction differs"
                    ));
                }
            }
        }
    }
}

/// VP9 deblocking at 10 bits (every width, both directions, random levels
/// and line counts).
fn check_vp9_deblock(
    (cand, refer): (DepthKernels<'_>, DepthKernels<'_>),
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    const STRIDE: usize = 48;
    for width in [
        Vp9FilterWidth::Four,
        Vp9FilterWidth::Eight,
        Vp9FilterWidth::Sixteen,
    ] {
        for vertical in [false, true] {
            for round in 0..ROUNDS * 4 {
                let base = samples(STRIDE * STRIDE, round % ROUNDS, rng);
                let limits =
                    Vp9FilterLimits::with_sharpness(1 + rng.below(63) as u8, rng.below(8) as u8);
                let lines = 1 + rng.below(24);
                let (step, along) = if vertical { (1, STRIDE) } else { (STRIDE, 1) };
                let at = (8 * STRIDE + 8, step, along);
                let mut out = [base.clone(), base];
                for (k, kernels) in [cand, refer].into_iter().enumerate() {
                    kernels.vp9_filter_edge(&mut out[k], at, lines, width, limits);
                }
                report.cases += 1;
                if out[0] != out[1] {
                    report.failures.push(format!(
                        "u16 VP9 deblocking {width:?} vertical {vertical} {lines} lines: differs"
                    ));
                }
            }
        }
    }
}

/// Luma sizes of the CDEF test pictures: `48x40` (interior + all edges + dual
/// 4x4 chroma pairs) and `16x16` (pure edge blocks on every side).
const CDEF_PICTURES: [(usize, usize); 2] = [(48, 40), (16, 16)];
/// CDEF luma units.
const CDEF_UNIT: usize = 8;

/// AV1 CDEF at 10 and 12 bits: interior, frame edges, multi-band streaming,
/// direction search, pri+sec / pri-only / sec-only / luma-only / chroma-only.
fn check_cdef(
    depth: BitDepth,
    (cand, refer): (DepthKernels<'_>, DepthKernels<'_>),
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let secondary = [0, 1, 2, 4];
    for &(w, h) in &CDEF_PICTURES {
        for round in 0..8 {
            let mut source = Picture::<u16>::new(w, h, 0).expect("picture");
            for plane in PlaneIndex::ALL {
                let mut view = source.plane_mut(plane).view_mut();
                let (pw, ph) = (view.width(), view.height());
                let values = samples_at(depth, pw * ph, round, rng);
                for (y, row) in values.chunks_exact(pw).enumerate() {
                    view.row_mut(y)[..pw].copy_from_slice(row);
                }
            }
            // Cover pri+sec, pri-only, sec-only, luma-only, chroma-only, and
            // multi-set pictures in every iteration.
            let sets_raw = [
                CdefStrengths {
                    luma_primary: 1 + rng.below(15) as u8,
                    luma_secondary: secondary[1 + rng.below(3)],
                    chroma_primary: 1 + rng.below(15) as u8,
                    chroma_secondary: secondary[1 + rng.below(3)],
                },
                CdefStrengths {
                    luma_primary: 1 + rng.below(15) as u8,
                    luma_secondary: 0,
                    chroma_primary: 1 + rng.below(15) as u8,
                    chroma_secondary: 0,
                },
                CdefStrengths {
                    luma_primary: 0,
                    luma_secondary: secondary[1 + rng.below(3)],
                    chroma_primary: 0,
                    chroma_secondary: secondary[1 + rng.below(3)],
                },
                CdefStrengths {
                    luma_primary: if round % 2 == 0 {
                        1 + rng.below(15) as u8
                    } else {
                        0
                    },
                    luma_secondary: secondary[rng.below(4)],
                    chroma_primary: if round % 2 == 1 {
                        1 + rng.below(15) as u8
                    } else {
                        0
                    },
                    chroma_secondary: if round % 2 == 1 {
                        secondary[1 + rng.below(3)]
                    } else {
                        0
                    },
                },
            ];
            let damping = 3 + rng.below(4) as u8;
            let sets = CdefSets::new(&sets_raw, damping);
            let band_h = if round % 2 == 0 { 16 } else { h };
            let mut out = [source.clone(), source];
            for (k, kernels) in [cand, refer].into_iter().enumerate() {
                let mut lines = CdefLines::<u16>::new();
                lines.start(&out[k]);
                let mut top = 0;
                while top < h {
                    let end = (top + band_h).min(h);
                    let units: Vec<CdefUnit> = (top / CDEF_UNIT..end / CDEF_UNIT)
                        .flat_map(|uy| {
                            (0..w / CDEF_UNIT).map(move |ux| CdefUnit {
                                x: ux * CDEF_UNIT,
                                y: uy * CDEF_UNIT,
                                set: ((ux + uy + round) & 3) as u8,
                            })
                        })
                        .collect();
                    kernels.cdef_filter_band(&mut lines, &mut out[k], top..end, &units, &sets);
                    top = end;
                }
            }
            report.cases += 1;
            let same = PlaneIndex::ALL.into_iter().all(|plane| {
                out[0].plane(plane).view().data() == out[1].plane(plane).view().data()
            });
            if !same {
                report.failures.push(format!(
                    "u16 AV1 CDEF {depth:?} {w}x{h} damping {damping} round {round}: differs"
                ));
            }
        }
    }
}

fn check_av1_blends_u8(
    (cand, refer): (nc_dsp::DepthKernels<'_>, nc_dsp::DepthKernels<'_>),
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    for (w, h) in [(4, 4), (8, 8), (16, 8), (32, 16), (64, 32), (128, 64)] {
        let p0: Vec<i16> = (0..w * h).map(|_| rng.below(14000) as i16 - 5000).collect();
        let p1: Vec<i16> = (0..w * h).map(|_| rng.below(14000) as i16 - 5000).collect();
        let (cp0, cp1) = (
            Av1CompoundPred {
                data: &p0,
                stride: w,
            },
            Av1CompoundPred {
                data: &p1,
                stride: w,
            },
        );
        let weights = av1_distance_weights(1 + rng.below(20) as i32, 1 + rng.below(20) as i32);
        let inv = rng.below(2) == 1;
        let mut dist = [vec![0u8; w * h], vec![0u8; w * h]];
        let mut dmask = [vec![0u8; w * h], vec![0u8; w * h]];
        for (k, kernels) in [cand, refer].into_iter().enumerate() {
            let mut vd = PlaneViewMut::new(&mut dist[k], w, w, h).expect("vd");
            kernels.av1_blend_distance((cp0, cp1), weights, (w, h), &mut vd);
            kernels.av1_difference_weight_mask((cp0, cp1), inv, (w, h), &mut dmask[k]);
        }
        report.cases += 2;
        if dist[0] != dist[1] || dmask[0] != dmask[1] {
            report
                .failures
                .push(format!("u8 AV1 dist/diffmask {w}x{h}: differs"));
        }
        let mask_raw: Vec<u8> = (0..4 * w * h).map(|_| rng.below(65) as u8).collect();
        for (sx, sy) in [(false, false), (true, false), (false, true), (true, true)] {
            let mask = Av1Mask {
                data: &mask_raw,
                stride: if sx { 2 * w } else { w },
                sub_x: sx,
                sub_y: sy,
            };
            let mut m_out = [vec![0u8; w * h], vec![0u8; w * h]];
            let intra_s: Vec<u8> = (0..w * h).map(|_| rng.below(256) as u8).collect();
            let inter_s: Vec<u8> = (0..w * h).map(|_| rng.below(256) as u8).collect();
            let intra_v = PlaneView::new(&intra_s, w, w, h).expect("intra");
            let mut ii_out = [inter_s.clone(), inter_s];
            for (k, kernels) in [cand, refer].into_iter().enumerate() {
                let mut vm = PlaneViewMut::new(&mut m_out[k], w, w, h).expect("vm");
                kernels.av1_blend_mask((cp0, cp1), mask, (w, h), &mut vm);
                let mut vii = PlaneViewMut::new(&mut ii_out[k], w, w, h).expect("vii");
                kernels.av1_blend_interintra(&intra_v, mask, (w, h), &mut vii);
            }
            report.cases += 2;
            if m_out[0] != m_out[1] || ii_out[0] != ii_out[1] {
                report.failures.push(format!(
                    "u8 AV1 mask/interintra {w}x{h} ({sx},{sy}): differs"
                ));
            }
        }
        if w <= 32 && h <= 32 {
            let obmc_s: Vec<u8> = (0..w * h).map(|_| rng.below(256) as u8).collect();
            let base_s: Vec<u8> = (0..w * h).map(|_| rng.below(256) as u8).collect();
            let obmc_v = PlaneView::new(&obmc_s, w, w, h).expect("obmc");
            for edge in [Av1ObmcEdge::Above, Av1ObmcEdge::Left] {
                let mut o_out = [base_s.clone(), base_s.clone()];
                for (k, kernels) in [cand, refer].into_iter().enumerate() {
                    let mut vo = PlaneViewMut::new(&mut o_out[k], w, w, h).expect("vo");
                    kernels.av1_obmc_blend(&obmc_v, edge, (w, h), &mut vo);
                }
                report.cases += 1;
                if o_out[0] != o_out[1] {
                    report
                        .failures
                        .push(format!("u8 AV1 OBMC {w}x{h} {edge:?}: differs"));
                }
            }
        }
    }
}

fn scan_positions(size: Av1TxSize, kind: Av1TxType) -> Vec<usize> {
    use Av1TxType::{HAdst, HDct, HFlipadst, VAdst, VDct, VFlipadst};
    let (cw, ch) = (size.coded_width(), size.coded_height());
    let mut positions: Vec<usize> = (0..cw * ch).collect();
    match kind {
        HDct | HAdst | HFlipadst => {}
        VDct | VAdst | VFlipadst => positions.sort_by_key(|&p| (p % ch, p / ch)),
        _ => positions.sort_by_key(|&p| (p / ch + p % ch, p % ch)),
    }
    positions
}

fn check_av1_inverse_high(
    depth: BitDepth,
    (cand, refer): (DepthKernels<'_>, DepthKernels<'_>),
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let max_c = (1i32 << (7 + depth.bits())) - 1;
    let min_c = -(1i32 << (7 + depth.bits()));
    for size in Av1TxSize::ALL {
        let (w, h) = (size.width(), size.height());
        let stride = w + 3;
        let area = size.coded_area();
        for kind in Av1TxType::ALL.into_iter().filter(|k| k.fits(size)) {
            let scan = scan_positions(size, kind);
            for pattern in 0..5usize {
                let mut coeffs = vec![0i32; area];
                let eob = match pattern {
                    0 => {
                        coeffs[0] = rng.below((2 * max_c + 1) as usize) as i32 + min_c;
                        1
                    }
                    1 => {
                        let eob = (2 + rng.below(35)).min(area);
                        for &pos in &scan[..eob] {
                            coeffs[pos] = rng.below(2048) as i32 - 1024;
                        }
                        eob
                    }
                    2 => {
                        for c in &mut coeffs {
                            *c = if rng.below(4) == 0 {
                                rng.below(4000) as i32 - 2000
                            } else {
                                0
                            };
                        }
                        area
                    }
                    3 => {
                        for c in &mut coeffs {
                            *c = rng.below((2 * max_c + 1) as usize) as i32 + min_c;
                        }
                        area
                    }
                    _ => {
                        for (i, c) in coeffs.iter_mut().enumerate() {
                            *c = if i % 3 == 0 {
                                900_000
                            } else if i % 2 == 0 {
                                max_c
                            } else {
                                min_c
                            };
                        }
                        area
                    }
                };
                let base = samples_at(depth, stride * (h + 2), pattern, rng);
                let mut out = [base.clone(), base];
                for (k, (kernels, used_eob)) in [(cand, eob), (refer, area)].into_iter().enumerate()
                {
                    let mut cbuf = coeffs.clone();
                    let mut view = PlaneViewMut::new(&mut out[k], stride, w, h).expect("plane");
                    kernels.av1_inverse_add(size, kind, &mut cbuf, used_eob, &mut view);
                }
                report.cases += 1;
                if out[0] != out[1] {
                    report.failures.push(format!(
                        "u16 AV1 inverse {depth:?} {w}x{h} {kind:?} pattern {pattern} eob {eob}: differs"
                    ));
                }
            }
        }
    }
}

/// AV1 loop restoration (`Wiener` 5/7-tap and `SgrProj` sets `0..16`) at 10 & 12 bits.
fn check_av1_restoration_high(
    depth: BitDepth,
    (cand, refer): (DepthKernels<'_>, DepthKernels<'_>),
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    const EXTREME_TAPS: [[i8; 3]; 4] = [[10, 8, 46], [-5, -23, -17], [10, -23, 46], [-5, 8, -17]];
    let max_u16 = depth.max() as u16;
    let mut scratch = [Av1LrScratch::default(), Av1LrScratch::default()];
    // Deterministic 5-tap / 7-tap Wiener axes + all 16 SgrProj sets.
    for (w, h) in [(1_usize, 1_usize), (17, 61), (33, 70), (97, 135)] {
        for pattern in 0..4 {
            let cdef: Vec<u16> = if pattern == 3 {
                vec![max_u16; w * h]
            } else {
                samples_at(depth, w * h, pattern, rng)
            };
            let deblocked = samples_at(depth, w * h, pattern + 1, rng);
            let deb_view = PlaneView::new(&deblocked, w, w, h).expect("deb");
            for sub_y in [false, true] {
                let bounds = Av1StripeBoundaries::save_at(&deb_view, sub_y, None, depth);
                let unit_size = 64 >> u32::from(sub_y);
                let (cols, rows) = (
                    av1_lr_unit_count(unit_size, w),
                    av1_lr_unit_count(unit_size, h),
                );
                let filters: Vec<Av1LrFilter> = (0..cols * rows)
                    .map(|u| match (u + pattern) % 6 {
                        0 => Av1LrFilter::None,
                        1 => Av1LrFilter::Wiener(Av1Wiener {
                            horizontal: [0, EXTREME_TAPS[u % 4][1], EXTREME_TAPS[u % 4][2]],
                            vertical: [
                                0,
                                EXTREME_TAPS[(u + 1) % 4][1],
                                EXTREME_TAPS[(u + 1) % 4][2],
                            ],
                        }),
                        2 => Av1LrFilter::Wiener(Av1Wiener {
                            horizontal: [0, -23, 46],
                            vertical: EXTREME_TAPS[u % 4],
                        }),
                        3 => Av1LrFilter::Wiener(Av1Wiener {
                            horizontal: EXTREME_TAPS[u % 4],
                            vertical: [0, 8, -17],
                        }),
                        _ => Av1LrFilter::SgrProj(Av1SgrProj {
                            set: ((u * 5 + rng.below(16)) % 16) as u8,
                            xqd: [
                                (rng.below(128) as i32 - 96) as i8,
                                (rng.below(128) as i32 - 32) as i8,
                            ],
                        }),
                    })
                    .collect();
                let lr = Av1LoopRestoration {
                    unit_size,
                    columns: cols,
                    rows,
                    filters: &filters,
                };
                let mut planes = [cdef.clone(), cdef.clone()];
                for (k, kernels) in [cand, refer].into_iter().enumerate() {
                    let mut view = PlaneViewMut::new(&mut planes[k], w, w, h).expect("plane");
                    kernels.av1_loop_restoration(&lr, &mut view, (&bounds, sub_y), &mut scratch[k]);
                }
                report.cases += 1;
                if planes[0] != planes[1] {
                    report.failures.push(format!(
                        "u16 AV1 loop restoration {depth:?} {w}x{h} sub_y={sub_y} pat={pattern}: differs"
                    ));
                }
            }
        }
    }
    // Explicitly exercise every SgrProj set 0..16 on max-value samples (12-bit 65584 edge case).
    let (w, h) = (35_usize, 19_usize);
    let src: Vec<u16> = (0..w * h)
        .map(|i| if i % 11 == 0 { max_u16 - 1 } else { max_u16 })
        .collect();
    let deb_view = PlaneView::new(&src, w, w, h).expect("deb");
    let bounds = Av1StripeBoundaries::save_at(&deb_view, false, None, depth);
    for set in 0..16_u8 {
        let filters = [Av1LrFilter::SgrProj(Av1SgrProj {
            set,
            xqd: [-64, 64],
        })];
        let lr = Av1LoopRestoration {
            unit_size: 64,
            columns: 1,
            rows: 1,
            filters: &filters,
        };
        let mut planes = [src.clone(), src.clone()];
        for (k, kernels) in [cand, refer].into_iter().enumerate() {
            let mut view = PlaneViewMut::new(&mut planes[k], w, w, h).expect("plane");
            kernels.av1_loop_restoration(&lr, &mut view, (&bounds, false), &mut scratch[k]);
        }
        report.cases += 1;
        if planes[0] != planes[1] {
            report.failures.push(format!(
                "u16 AV1 SGR {depth:?} set {set} max-sample edge case: differs"
            ));
        }
    }
}

/// AV1 loop filter (deblocking) runs (`Four`, `Six`, `Eight`, `Fourteen`) at 10 & 12 bits.
fn check_av1_deblock_high(
    depth: BitDepth,
    (cand, refer): (DepthKernels<'_>, DepthKernels<'_>),
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let max_val = depth.max();
    let shift = depth.above_eight();
    let side = 64_usize;
    let stride = 72_usize;
    for length in [
        Av1FilterLength::Four,
        Av1FilterLength::Six,
        Av1FilterLength::Eight,
        Av1FilterLength::Fourteen,
    ] {
        for level in [1_u8, 8, 24, 48, 63] {
            let filter = Av1EdgeFilter {
                length,
                limit: (level >> 1).max(1),
                blimit: 2 * (level + 2) + (level >> 1).max(1),
                thresh: level >> 4,
            };
            for vertical in [true, false] {
                for pat in 0..4 {
                    let base: Vec<u16> = (0..stride * side)
                        .map(|i| match pat {
                            // Smooth near max_val (exercises 12-bit 14-tap 65528 unsigned sum).
                            0 => (max_val - ((rng.below(3) as i32) << shift)).max(0) as u16,
                            // Smooth near mid-grey with mixed flat / HEV steps across the edge.
                            1 => {
                                let step = if (if vertical { i % stride } else { i / stride }) >= 16
                                {
                                    (rng.below(6) as i32) << shift
                                } else {
                                    0
                                };
                                ((max_val / 2) + step - ((rng.below(3) as i32) << shift))
                                    .clamp(0, max_val) as u16
                            }
                            // Random full-range samples.
                            2 => rng.below(max_val as usize + 1) as u16,
                            // Near zero with small steps (exercises negative narrow clamping).
                            _ => ((rng.below(5) as i32) << shift).min(max_val) as u16,
                        })
                        .collect();
                    // Adjacent 4-line runs (merged to 36 = 32 + 4 / 16 + 16 + 4) plus 12-line & 4-line runs.
                    let mut runs = Vec::new();
                    for seg in 0..9 {
                        let along = 4 + seg * 4;
                        let (x, y) = if vertical { (16, along) } else { (along, 16) };
                        runs.push(EdgeRun {
                            x,
                            y,
                            lines: 4,
                            filter,
                        });
                    }
                    for &(off, len) in &[(8_usize, 12_usize), (28, 8), (44, 4)] {
                        let (x, y) = if vertical { (32, off) } else { (off, 32) };
                        runs.push(EdgeRun {
                            x,
                            y,
                            lines: len,
                            filter,
                        });
                    }
                    let mut planes = [base.clone(), base];
                    for (k, kernels) in [cand, refer].into_iter().enumerate() {
                        let mut view =
                            PlaneViewMut::new(&mut planes[k], stride, side, side).expect("plane");
                        kernels.av1_loop_filter_runs(&mut view, vertical, &runs);
                    }
                    report.cases += 1;
                    if planes[0] != planes[1] {
                        report.failures.push(format!(
                            "u16 AV1 deblock {depth:?} {length:?} lvl={level} vert={vertical} pat={pat}: differs"
                        ));
                    }
                }
            }
        }
    }
}

const INTRA_SHAPES: [(usize, usize); 15] = [
    (4, 4),
    (4, 8),
    (8, 4),
    (4, 16),
    (16, 4),
    (8, 8),
    (8, 16),
    (16, 8),
    (8, 32),
    (32, 8),
    (16, 16),
    (16, 32),
    (32, 16),
    (32, 32),
    (64, 64),
];

fn random_avail(rng: &mut Rng) -> EdgeAvail {
    EdgeAvail {
        have_above: rng.below(4) != 0,
        have_left: rng.below(4) != 0,
        have_above_right: rng.below(2) == 0,
        have_below_left: rng.below(2) == 0,
    }
}

fn intra_kinds() -> [IntraKind; 15] {
    let ef_off = EdgeFilterCtx {
        smooth_above: false,
        smooth_left: false,
    };
    let ef_on = EdgeFilterCtx {
        smooth_above: true,
        smooth_left: true,
    };
    [
        IntraKind::Dc,
        IntraKind::Paeth,
        IntraKind::Smooth,
        IntraKind::SmoothVertical,
        IntraKind::SmoothHorizontal,
        IntraKind::Directional {
            angle: 36,
            edge_filter: None,
        },
        IntraKind::Directional {
            angle: 67,
            edge_filter: Some(ef_on),
        },
        IntraKind::Directional {
            angle: 90,
            edge_filter: None,
        },
        IntraKind::Directional {
            angle: 113,
            edge_filter: None,
        },
        IntraKind::Directional {
            angle: 135,
            edge_filter: Some(ef_off),
        },
        IntraKind::Directional {
            angle: 157,
            edge_filter: Some(ef_on),
        },
        IntraKind::Directional {
            angle: 180,
            edge_filter: None,
        },
        IntraKind::Directional {
            angle: 203,
            edge_filter: None,
        },
        IntraKind::Directional {
            angle: 225,
            edge_filter: Some(ef_on),
        },
        IntraKind::Directional {
            angle: 261,
            edge_filter: None,
        },
    ]
}

fn check_av1_intra_high(
    depth: BitDepth,
    (cand, refer): (DepthKernels<'_>, DepthKernels<'_>),
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let mut plane = Plane::<u16>::new(192, 192, 0).expect("plane");
    for &(w, h) in &INTRA_SHAPES {
        for round in 0..2 {
            let raw = samples_at(depth, 192 * 192, round, rng);
            plane.view_mut().data_mut()[..raw.len()].copy_from_slice(&raw);
            let edges = BlockEdges::read(&plane, (16, 16), (w, h), random_avail(rng), depth);
            for kind in intra_kinds() {
                let mut out = [vec![0u16; w * h], vec![0u16; w * h]];
                for (k, kernels) in [cand, refer].into_iter().enumerate() {
                    let mut view = PlaneViewMut::new(&mut out[k], w, w, h).expect("dst");
                    kernels.av1_intra_predict_block(&edges, kind, &mut view);
                }
                report.cases += 1;
                if out[0] != out[1] {
                    report
                        .failures
                        .push(format!("u16 AV1 intra {depth:?} {w}x{h} {kind:?}: differs"));
                }
            }
            if w <= 32 && h <= 32 {
                for mode in FilterIntraMode::ALL {
                    let mut out = [vec![0u16; w * h], vec![0u16; w * h]];
                    for (k, kernels) in [cand, refer].into_iter().enumerate() {
                        let mut view = PlaneViewMut::new(&mut out[k], w, w, h).expect("dst");
                        kernels.av1_intra_predict_filter_block(&edges, mode, &mut view);
                    }
                    report.cases += 1;
                    if out[0] != out[1] {
                        report.failures.push(format!(
                            "u16 AV1 filter intra {depth:?} {w}x{h} {mode:?}: differs"
                        ));
                    }
                }
                let luma = plane.region(16, 16, 2 * w, 2 * h).expect("luma");
                let extent = (
                    (2 * w).saturating_sub(2 * rng.below(2)).max(2),
                    (2 * h).saturating_sub(2 * rng.below(2)).max(2),
                );
                let dc = edges.dc_value();
                for alpha in [-16, -7, -1, 0, 1, 9, 16] {
                    let mut out = [vec![0u16; w * h], vec![0u16; w * h]];
                    for (k, kernels) in [cand, refer].into_iter().enumerate() {
                        let mut view = PlaneViewMut::new(&mut out[k], w, w, h).expect("dst");
                        kernels.av1_cfl_predict(&luma, extent, dc, alpha, &mut view);
                    }
                    report.cases += 1;
                    if out[0] != out[1] {
                        report.failures.push(format!(
                            "u16 AV1 CfL {depth:?} {w}x{h} alpha={alpha}: differs"
                        ));
                    }
                }
            }
        }
    }
}

fn check_av1_intra_u8_rect_and_cfl(
    (cand, refer): (DepthKernels<'_>, DepthKernels<'_>),
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let mut plane = Plane::<u8>::new(192, 192, 0).expect("plane");
    for &(w, h) in &INTRA_SHAPES {
        let raw: Vec<u8> = (0..192 * 192).map(|_| rng.byte()).collect();
        plane.view_mut().data_mut()[..raw.len()].copy_from_slice(&raw);
        let edges = BlockEdges::read(&plane, (16, 16), (w, h), random_avail(rng), BitDepth::EIGHT);
        if w != h {
            for kind in intra_kinds() {
                let mut out = [vec![0u8; w * h], vec![0u8; w * h]];
                for (k, kernels) in [cand, refer].into_iter().enumerate() {
                    let mut view = PlaneViewMut::new(&mut out[k], w, w, h).expect("dst");
                    kernels.av1_intra_predict_block(&edges, kind, &mut view);
                }
                report.cases += 1;
                if out[0] != out[1] {
                    report
                        .failures
                        .push(format!("u8 rect intra {w}x{h} {kind:?}: differs"));
                }
            }
        }
        if w <= 32 && h <= 32 {
            let luma = plane.region(16, 16, 2 * w, 2 * h).expect("luma");
            let extent = ((2 * w).max(2), (2 * h).saturating_sub(2).max(2));
            let dc = edges.dc_value();
            for alpha in [-16, -5, 0, 5, 16] {
                let mut out = [vec![0u8; w * h], vec![0u8; w * h]];
                for (k, kernels) in [cand, refer].into_iter().enumerate() {
                    let mut view = PlaneViewMut::new(&mut out[k], w, w, h).expect("dst");
                    kernels.av1_cfl_predict(&luma, extent, dc, alpha, &mut view);
                }
                report.cases += 1;
                if out[0] != out[1] {
                    report
                        .failures
                        .push(format!("u8 CfL {w}x{h} alpha={alpha}: differs"));
                }
            }
        }
    }
}

fn check_av1_grain_high(
    depth: BitDepth,
    (cand, refer): (DepthKernels<'_>, DepthKernels<'_>),
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    for &(w, h) in &[(64, 64), (70, 38), (98, 66)] {
        for round in 0..4 {
            let mut src = Picture::<u16>::new(128, 96, 0).expect("src");
            for plane in [PlaneIndex::Y, PlaneIndex::U, PlaneIndex::V] {
                let len = src.plane(plane).view().data().len();
                let vals = samples_at(depth, len, round, rng);
                src.plane_mut(plane)
                    .view_mut()
                    .data_mut()
                    .copy_from_slice(&vals);
            }
            let pts = Av1ScalingPoints {
                count: 3,
                value: {
                    let mut v = [0u8; 14];
                    v[0] = 16;
                    v[1] = 128;
                    v[2] = 240;
                    v
                },
                scaling: {
                    let mut s = [0u8; 14];
                    s[0] = rng.byte();
                    s[1] = 255;
                    s[2] = rng.byte();
                    s
                },
            };
            let uv_pts = Av1ScalingPoints {
                count: 2,
                value: {
                    let mut v = [0u8; 14];
                    v[0] = 0;
                    v[1] = 255;
                    v
                },
                scaling: {
                    let mut s = [0u8; 14];
                    s[0] = rng.byte();
                    s[1] = rng.byte();
                    s
                },
            };
            let params = Av1FilmGrainParams {
                grain_seed: 0x1357 ^ (round as u16 * 97),
                y_points: pts,
                chroma_scaling_from_luma: round % 2 == 0,
                cb_points: uv_pts,
                cr_points: uv_pts,
                grain_scaling_minus_8: (round % 4) as u8,
                ar_coeff_lag: (round % 4) as u8,
                ar_coeffs_y_plus_128: core::array::from_fn(|_| 120 + (rng.below(17) as u8)),
                ar_coeffs_cb_plus_128: core::array::from_fn(|_| 120 + (rng.below(17) as u8)),
                ar_coeffs_cr_plus_128: core::array::from_fn(|_| 120 + (rng.below(17) as u8)),
                ar_coeff_shift_minus_6: (round % 4) as u8,
                grain_scale_shift: (round % 2) as u8,
                cb_mult: (rng.byte(), rng.byte(), rng.below(512) as u16),
                cr_mult: (0, 255, 511),
                overlap: round != 1,
                clip_to_restricted_range: round >= 2,
                mc_identity: false,
            };
            let mut out = [
                Picture::<u16>::new(128, 96, 0).expect("out0"),
                Picture::<u16>::new(128, 96, 0).expect("out1"),
            ];
            let (sy, su, sv) = (
                src.plane(PlaneIndex::Y).view(),
                src.plane(PlaneIndex::U).view(),
                src.plane(PlaneIndex::V).view(),
            );
            for (k, kernels) in [cand, refer].into_iter().enumerate() {
                let [oy, ou, ov] = out[k].planes_mut();
                let (mut vy, mut vu, mut vv) = (oy.view_mut(), ou.view_mut(), ov.view_mut());
                kernels.av1_add_film_grain(
                    &params,
                    [&sy, &su, &sv],
                    (w, h),
                    [&mut vy, &mut vu, &mut vv],
                );
            }
            report.cases += 1;
            for p in [PlaneIndex::Y, PlaneIndex::U, PlaneIndex::V] {
                if out[0].plane(p).view().data() != out[1].plane(p).view().data() {
                    report.failures.push(format!(
                        "u16 AV1 film grain {depth:?} {w}x{h} round={round} plane={p:?}: differs"
                    ));
                }
            }
        }
    }
}

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    check_sao_high(candidate, reference, report);
    check_av1_blends_u8(
        (candidate.at(BitDepth::EIGHT), reference.at(BitDepth::EIGHT)),
        rng,
        report,
    );
    check_av1_intra_u8_rect_and_cfl(
        (candidate.at(BitDepth::EIGHT), reference.at(BitDepth::EIGHT)),
        rng,
        report,
    );
    for depth in av1_depths() {
        let (cand, refer) = (candidate.at(depth), reference.at(depth));
        check_av1_mc(depth, (cand, refer), rng, report);
        check_av1_inverse_high(depth, (cand, refer), rng, report);
        check_av1_restoration_high(depth, (cand, refer), rng, report);
        check_av1_deblock_high(depth, (cand, refer), rng, report);
        check_av1_intra_high(depth, (cand, refer), rng, report);
        check_av1_grain_high(depth, (cand, refer), rng, report);
        check_cdef(depth, (cand, refer), rng, report);
    }
    let (cand, refer) = (candidate.at(DEPTH), reference.at(DEPTH));
    check_vp9_mc((cand, refer), rng, report);
    check_vp9_deblock((cand, refer), rng, report);
    check_hevc_vvc_high((cand, refer), rng, report);
    crate::check_vvc_high::check(candidate, reference, rng, report);
}

/// The HEVC and VVC Main 10 kernels.
fn check_hevc_vvc_high(
    (cand, refer): (DepthKernels<'_>, DepthKernels<'_>),
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    for filter in FILTERS {
        let span_extra = filter.taps() - 1;
        for side in SIDES {
            let size = BlockSize::new(side, side).expect("square kernel size");
            let span = side + span_extra;
            for round in 0..ROUNDS {
                let source = samples(span * span, round, rng);
                let window = PlaneView::new(&source, span, span, span).expect("window");
                let frac = (rng.below(filter.positions()), rng.below(filter.positions()));
                let mut prep = [vec![0i16; side * side], vec![0i16; side * side]];
                cand.predict_compound(filter, size, &window, frac, &mut prep[0], side);
                refer.predict_compound(filter, size, &window, frac, &mut prep[1], side);
                let mut uni = [vec![0u16; side * side], vec![0u16; side * side]];
                let mut bi = uni.clone();
                for (k, kernels) in [cand, refer].into_iter().enumerate() {
                    let mut uni_view =
                        PlaneViewMut::new(&mut uni[k], side, side, side).expect("view");
                    kernels.predict(filter, size, &window, frac, &mut uni_view);
                    let mut bi_view =
                        PlaneViewMut::new(&mut bi[k], side, side, side).expect("view");
                    kernels.hevc_predict_bi(
                        filter,
                        size,
                        &window,
                        frac,
                        (&prep[1], side),
                        &mut bi_view,
                    );
                }
                report.cases += 3;
                if prep[0] != prep[1] || uni[0] != uni[1] || bi[0] != bi[1] {
                    report.failures.push(format!(
                        "u16 {filter:?} {side}x{side} frac {frac:?} round {round}: prediction differs"
                    ));
                }
                let weights = HevcWeights {
                    w0: rng.below(384) as i16 - 128,
                    w1: rng.below(384) as i16 - 128,
                    round: (rng.below(256) as i32 - 128) << 6,
                    shift: 5 + rng.below(9) as u32,
                };
                let mut weighted = [vec![0u16; side * side], vec![0u16; side * side]];
                for (k, kernels) in [cand, refer].into_iter().enumerate() {
                    let mut view =
                        PlaneViewMut::new(&mut weighted[k], side, side, side).expect("view");
                    kernels
                        .hevc_weighted_prediction_centered(weights, &prep[0], &prep[1], &mut view);
                }
                report.cases += 1;
                if weighted[0] != weighted[1] {
                    report.failures.push(format!(
                        "u16 weighted prediction {side}x{side} {weights:?} differs"
                    ));
                }
            }
        }
    }
    for transform in HevcTransform::ALL {
        let n = transform.side();
        for round in 0..ROUNDS * 4 {
            let (rows, columns) = (1 + rng.below(n), 1 + rng.below(n));
            let coeffs: Vec<i16> = (0..n * n)
                .map(|i| {
                    let inside = i / n < rows && i % n < columns;
                    match (inside, round % 3) {
                        (false, _) => 0,
                        (true, 0) => rng.below(65_536) as i16,
                        (true, 1) => i16::MAX,
                        _ => rng.below(512) as i16 - 256,
                    }
                })
                .collect();
            let base = samples(n * n, round, rng);
            let mut out = [base.clone(), base];
            for (k, kernels) in [cand, refer].into_iter().enumerate() {
                let mut view = PlaneViewMut::new(&mut out[k], n, n, n).expect("view");
                kernels.hevc_inverse_add(transform, &coeffs, (rows, columns), &mut view);
            }
            report.cases += 1;
            if out[0] != out[1] {
                report.failures.push(format!(
                    "u16 {transform:?} inverse add, extent {rows}x{columns}, differs"
                ));
            }
        }
    }
}

/// Shared HEVC/VVC SAO: exact windows, every vector tail, saturated offsets
/// and the 16-bit scalar fallback where samples no longer fit signed words.
fn check_sao_high(candidate: &Dsp, reference: &Dsp, report: &mut CheckReport) {
    for bits in [10, 12, 15, 16] {
        let depth = BitDepth::new(bits).expect("sample depth");
        for width in [9, 10, 15, 16, 17, 18, 31, 32, 33] {
            for height in [4, 9, 16] {
                let ss = width + 3;
                let ds = width + 7;
                for pattern in 0..4 {
                    let source: Vec<u16> = (0usize..1 + (height - 1) * ss + width)
                        .map(|i| match pattern {
                            0 => 0,
                            1 => depth.max() as u16,
                            2 => {
                                if i.is_multiple_of(2) {
                                    0
                                } else {
                                    depth.max() as u16
                                }
                            }
                            _ => ((i * 191 + 73) as u32 & depth.max() as u32) as u16,
                        })
                        .collect();
                    let input =
                        PlaneView::new(&source[1..], ss, width, height).expect("exact source");
                    for class in 0..4 {
                        for offsets in [[0; 4], [-31, 15, -7, 23], [i16::MIN, -1, 1, i16::MAX]] {
                            let guard = 0xa5a5;
                            let mut a = vec![guard; 3 + (height - 1) * ds + width + 5];
                            let mut b = a.clone();
                            for (dsp, out) in [(candidate, &mut a), (reference, &mut b)] {
                                let len = (height - 1) * ds + width;
                                let mut view =
                                    PlaneViewMut::new(&mut out[3..3 + len], ds, width, height)
                                        .expect("exact destination");
                                dsp.at(depth).sao_edge_apply(
                                    class,
                                    offsets,
                                    &input,
                                    &mut view,
                                    (1, 1, width - 1, height - 1),
                                );
                            }
                            report.cases += 1;
                            if a != b {
                                report.failures.push(format!("u16 SAO {bits}bit {width}x{height} class{class} pattern{pattern} offsets{offsets:?}"));
                            }
                        }
                    }
                }
            }
        }
    }
}
