//! High-bit-depth (`u16`, VVC Main 10) kernel checks against scalar reference.

use nc_common::{BitDepth, PlaneView, PlaneViewMut};
use nc_dsp::{
    DepthKernels, Dsp, Filter, HevcPlanarEdges, HevcTransform, HevcWeights, VVC_ALF_BLOCK,
    VVC_ALF_REACH, VVC_ALF_TAPS, VvcGeometric, VvcIntraMode, VvcLfnst, VvcLmcs, VvcRprFilter,
    VvcScaledBlock, VvcTransformKind, vvc_alf_clip_value, vvc_alf_max_group_sum, vvc_deblock_beta,
    vvc_deblock_tc,
};

use crate::{check::CheckReport, rng::Rng};

const ALF_WIDTHS: [usize; 14] = [1, 3, 4, 7, 8, 12, 16, 20, 31, 32, 35, 64, 68, 128];

fn sample_row(rng: &mut Rng, depth: BitDepth, pattern: usize, len: usize) -> Vec<u16> {
    let max = depth.max() as u16;
    (0..len)
        .map(|x| match pattern {
            0 => 0,
            1 => max,
            2 => {
                if rng.below(2) == 0 {
                    0
                } else {
                    max
                }
            }
            3 => (max / 3) + u16::from(x >= len / 2) * (max / 3),
            4 => {
                if x % 2 == 0 {
                    max
                } else {
                    0
                }
            }
            _ => rng.below(usize::from(max) + 1) as u16,
        })
        .collect()
}

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    check_dmvr_and_bdof_u8(candidate, reference, rng, report);
    for depth in [
        BitDepth::EIGHT,
        BitDepth::new(9).expect("9b"),
        BitDepth::TEN,
    ] {
        let (cand, refer) = (candidate.at(depth), reference.at(depth));
        check_alf_filters(depth, (cand, refer), rng, report);
        check_alf_laplacians_and_classify(depth, (cand, refer), rng, report);
        check_high_precision(depth, (cand, refer), rng, report);
        check_weight(depth, (cand, refer), rng, report);
        check_refinement_and_gpm(depth, (cand, refer), rng, report);
        check_rpr(depth, (cand, refer), rng, report);
        check_transforms_and_recon(depth, (cand, refer), rng, report);
        check_lmcs(depth, (cand, refer), rng, report);
        check_sao(depth, (cand, refer), rng, report);
        check_deblock(depth, (cand, refer), rng, report);
        check_vvc_intra(depth, (cand, refer), rng, report);
        check_hevc_intra(depth, (cand, refer), rng, report);
    }
    let d12 = BitDepth::new(12).expect("12b");
    check_sao(d12, (candidate.at(d12), reference.at(d12)), rng, report);
    check_hevc_intra(d12, (candidate.at(d12), reference.at(d12)), rng, report);
}

#[allow(clippy::too_many_lines)]
fn check_alf_filters(
    depth: BitDepth,
    (cand, refer): (DepthKernels<'_>, DepthKernels<'_>),
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let clips: [i16; 4] = core::array::from_fn(|i| vvc_alf_clip_value(i, depth));
    for width in ALF_WIDTHS {
        for pattern in 0..6 {
            let slack = if pattern >= 3 { 16 } else { 0 };
            let data: Vec<Vec<u16>> = (0..10)
                .map(|_| sample_row(rng, depth, pattern, width + 2 * VVC_ALF_REACH + slack))
                .collect();
            let block_rows: [[&[u16]; 7]; VVC_ALF_BLOCK] = core::array::from_fn(|r| {
                core::array::from_fn(|i| {
                    let dy = i as isize - 3;
                    let dy = if pattern % 2 == 0 {
                        dy
                    } else {
                        dy.clamp(-(r as isize), r as isize)
                    };
                    data[(r as isize + 3 + dy) as usize].as_slice()
                })
            });
            let blocks = width.div_ceil(VVC_ALF_BLOCK);
            let coefficients: Vec<[i16; VVC_ALF_TAPS]> = (0..blocks)
                .map(|_| core::array::from_fn(|_| rng.below(256) as i16 - 128))
                .collect();
            let limits: Vec<[i16; VVC_ALF_TAPS]> = (0..blocks)
                .map(|_| core::array::from_fn(|_| clips[rng.below(4)]))
                .collect();
            for shifts in [[7; VVC_ALF_BLOCK], [10, 7, 9, 10], [1, 7, 10, 16]] {
                let mut out = [
                    core::array::from_fn::<_, VVC_ALF_BLOCK, _>(|_| vec![0u16; width]),
                    core::array::from_fn::<_, VVC_ALF_BLOCK, _>(|_| vec![0u16; width]),
                ];
                for (k, kernels) in [cand, refer].into_iter().enumerate() {
                    kernels.vvc_alf_filter_block_row(
                        &block_rows,
                        &coefficients,
                        &shifts,
                        &mut out[k].each_mut().map(Vec::as_mut_slice),
                    );
                }
                report.cases += 1;
                if out[0] != out[1] {
                    report.failures.push(format!(
                        "u16 vvc_alf_filter_block_row {depth:?} w={width} pat={pattern} shifts={shifts:?}"
                    ));
                }
            }
            for shifts in [[7; VVC_ALF_BLOCK], [10, 7, 7, 10]] {
                let mut out = [
                    core::array::from_fn::<_, VVC_ALF_BLOCK, _>(|_| vec![0u16; width]),
                    core::array::from_fn::<_, VVC_ALF_BLOCK, _>(|_| vec![0u16; width]),
                ];
                for (k, kernels) in [cand, refer].into_iter().enumerate() {
                    kernels.vvc_alf_filter_clipped_block_row(
                        &block_rows,
                        &coefficients,
                        &limits,
                        &shifts,
                        &mut out[k].each_mut().map(Vec::as_mut_slice),
                    );
                }
                report.cases += 1;
                if out[0] != out[1] {
                    report.failures.push(format!(
                        "u16 vvc_alf_filter_clipped_block_row {depth:?} w={width} pat={pattern} shifts={shifts:?}"
                    ));
                }
            }
            let single_rows = block_rows[0];
            let chroma_c: [i16; 6] = core::array::from_fn(|_| rng.below(256) as i16 - 128);
            let chroma_l: [i16; 6] = core::array::from_fn(|_| clips[rng.below(4)]);
            for shift in [7, 10] {
                let mut clipped = [vec![0u16; width], vec![0u16; width]];
                let mut chroma = [vec![0u16; width], vec![0u16; width]];
                for (k, kernels) in [cand, refer].into_iter().enumerate() {
                    kernels.vvc_alf_filter_clipped_row(
                        &single_rows,
                        &coefficients,
                        &limits,
                        shift,
                        &mut clipped[k],
                    );
                    kernels.vvc_alf_filter_chroma_row(
                        &single_rows,
                        chroma_c,
                        chroma_l,
                        shift,
                        &mut chroma[k],
                    );
                }
                report.cases += 2;
                if clipped[0] != clipped[1] {
                    report.failures.push(format!(
                        "u16 vvc_alf_filter_clipped_row {depth:?} w={width} pat={pattern} shift={shift}"
                    ));
                }
                if chroma[0] != chroma[1] {
                    report.failures.push(format!(
                        "u16 vvc_alf_filter_chroma_row {depth:?} w={width} pat={pattern} shift={shift}"
                    ));
                }
            }
            let luma_data: Vec<Vec<u16>> = (0..4)
                .map(|_| sample_row(rng, depth, pattern, 2 * width + 2 + slack))
                .collect();
            let cross_rows: [&[u16]; 4] = core::array::from_fn(|r| luma_data[r].as_slice());
            let cross_c: [i16; 7] = core::array::from_fn(|k| match pattern {
                2 if k % 2 == 0 => -64,
                1 | 2 => 64,
                _ => rng.below(129) as i16 - 64,
            });
            let base = sample_row(rng, depth, (pattern + 2) % 6, width);
            let mut cross_out = [base.clone(), base];
            for (k, kernels) in [cand, refer].into_iter().enumerate() {
                kernels.vvc_alf_cross_row(&cross_rows, cross_c, &mut cross_out[k]);
            }
            report.cases += 1;
            if cross_out[0] != cross_out[1] {
                report.failures.push(format!(
                    "u16 vvc_alf_cross_row {depth:?} w={width} pat={pattern}"
                ));
            }
        }
    }
}

fn check_alf_laplacians_and_classify(
    depth: BitDepth,
    (cand, refer): (DepthKernels<'_>, DepthKernels<'_>),
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    for groups in [1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 33] {
        for pattern in 0..6 {
            let data: Vec<Vec<u16>> = (0..4)
                .map(|_| sample_row(rng, depth, pattern, 4 * groups + 2))
                .collect();
            let rows: [&[u16]; 4] = core::array::from_fn(|r| data[r].as_slice());
            let mut out = [vec![[0u16; 4]; groups], vec![[0u16; 4]; groups]];
            for (k, kernels) in [cand, refer].into_iter().enumerate() {
                kernels.vvc_alf_laplacians(&rows, &mut out[k]);
            }
            report.cases += 1;
            if out[0] != out[1] {
                report.failures.push(format!(
                    "u16 vvc_alf_laplacians {depth:?} groups={groups} pat={pattern}"
                ));
            }
        }
    }
    let max_sum = vvc_alf_max_group_sum(depth);
    for blocks in [1, 3, 4, 7, 8, 9, 15, 16, 17, 32, 35] {
        for pattern in 0..9 {
            let groups: Vec<[u16; 4]> = (0..=blocks)
                .map(|i| {
                    core::array::from_fn(|d| match pattern {
                        0 => 0,
                        1 => max_sum,
                        // Exercise unsigned 32-bit cross-product > 2^31 (at 10-bit max_sum = 32736,
                        // block sums reach 65472 and products reach ~4.286e9 > i32::MAX).
                        2 => match (i + d) % 4 {
                            0 => max_sum,
                            1 => max_sum - (rng.below(64) as u16).min(max_sum),
                            2 => max_sum / 2 + (rng.below(128) as u16).min(max_sum / 2),
                            _ => (max_sum / 2).saturating_sub(rng.below(128) as u16),
                        },
                        3..=6 if d == pattern - 3 => rng.below(usize::from(max_sum) + 1) as u16,
                        3..=6 => rng.below(64) as u16,
                        _ => rng.below(usize::from(max_sum) + 1) as u16,
                    })
                })
                .collect();
            for boundary in [false, true] {
                let mut out = [vec![[0u8; 2]; blocks], vec![[0u8; 2]; blocks]];
                for (k, kernels) in [cand, refer].into_iter().enumerate() {
                    kernels.vvc_alf_classify::<u16>(&groups, boundary, &mut out[k]);
                }
                report.cases += 1;
                if out[0] != out[1] {
                    report.failures.push(format!(
                        "u16 vvc_alf_classify {depth:?} blocks={blocks} pat={pattern} boundary={boundary}"
                    ));
                }
            }
        }
    }
}

fn check_dmvr_and_bdof_u8(
    candidate: &Dsp,
    reference: &Dsp,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    for size in [(8, 8), (8, 16), (16, 8), (16, 16)] {
        let len = (size.0 + 4) * (size.1 + 4);
        for pat in 0..4 {
            let a: Vec<i16> = (0..len)
                .map(|_| match pat {
                    0 => rng.below(1024) as i16,
                    1 => (rng.below(65536) as i32 - 32768) as i16,
                    _ => (rng.below(2048) as i32 - 512) as i16,
                })
                .collect();
            let b: Vec<i16> = (0..len)
                .map(|i| match pat {
                    2 => a[i].saturating_add(rng.below(9) as i16 - 4),
                    _ => (rng.below(2048) as i32 - 512) as i16,
                })
                .collect();
            let got = candidate.vvc_dmvr(size, [&a, &b]);
            let exp = reference.vvc_dmvr(size, [&a, &b]);
            report.cases += 1;
            if got != exp {
                report
                    .failures
                    .push(format!("vvc_dmvr size={size:?} pat={pat}"));
            }
        }
    }
    for size in [(4, 4), (8, 4), (4, 8), (8, 8), (12, 8), (16, 12), (16, 16)] {
        let len = (size.0 + 2) * (size.1 + 2);
        for pat in 0..3 {
            let a: Vec<i16> = (0..len)
                .map(|_| (rng.below(20000) as i32 - 2000) as i16)
                .collect();
            let b: Vec<i16> = (0..len)
                .map(|_| (rng.below(20000) as i32 - 2000) as i16)
                .collect();
            let mut out = [vec![0u8; size.0 * size.1], vec![0u8; size.0 * size.1]];
            candidate.vvc_bdof(size, [&a, &b], &mut out[0]);
            reference.vvc_bdof(size, [&a, &b], &mut out[1]);
            report.cases += 1;
            if out[0] != out[1] {
                report
                    .failures
                    .push(format!("u8 vvc_bdof size={size:?} pat={pat}"));
            }
        }
    }
}

fn check_high_precision(
    depth: BitDepth,
    (cand, refer): (DepthKernels<'_>, DepthKernels<'_>),
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let filters = [
        (Filter::HevcLuma, false, 4, 8),
        (Filter::HevcChroma, false, 8, 4),
        (Filter::VvcLuma, false, 16, 8),
        (Filter::VvcLuma, true, 16, 8),
        (Filter::VvcAffineLuma, false, 16, 8),
        (Filter::VvcChroma, false, 32, 4),
    ];
    for &(filter, alt_half, positions, taps) in &filters {
        for &(w, h) in &[
            (2, 4),
            (4, 2),
            (4, 4),
            (6, 8),
            (8, 8),
            (12, 16),
            (16, 16),
            (24, 8),
            (32, 16),
            (64, 8),
        ] {
            let stride = w + taps - 1 + 4;
            let rows = h + taps - 1;
            let half = positions / 2;
            let phase_list = [
                (0, 0),
                (half, 0),
                (0, half),
                (half, half),
                (rng.below(positions), rng.below(positions)),
            ];
            for pat in [2, 4, 5] {
                let buf = sample_row(rng, depth, pat, stride * rows);
                let win = PlaneView::new(&buf, stride, w + taps - 1, rows).expect("window");
                for &phases in &phase_list {
                    let mut out = [vec![0i16; w * h], vec![0i16; w * h]];
                    cand.predict_high_precision(
                        filter,
                        (w, h),
                        &win,
                        phases,
                        alt_half,
                        &mut out[0],
                    );
                    refer.predict_high_precision(
                        filter,
                        (w, h),
                        &win,
                        phases,
                        alt_half,
                        &mut out[1],
                    );
                    report.cases += 1;
                    if out[0] != out[1] {
                        report.failures.push(format!(
                            "u16 predict_high_precision {depth:?} {filter:?} alt={alt_half} {w}x{h} phases={phases:?} pat={pat}"
                        ));
                    }
                }
            }
        }
    }
}

fn check_weight(
    depth: BitDepth,
    (cand, refer): (DepthKernels<'_>, DepthKernels<'_>),
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    for &(w, h) in &[
        (2, 4),
        (4, 4),
        (6, 8),
        (8, 8),
        (12, 8),
        (16, 8),
        (32, 4),
        (36, 4),
        (64, 4),
    ] {
        let a: Vec<i16> = (0..w * h)
            .map(|_| (rng.below(40000) as i32 - 7000) as i16)
            .collect();
        let b: Vec<i16> = (0..w * h)
            .map(|_| (rng.below(40000) as i32 - 7000) as i16)
            .collect();
        let single = 14 - depth.bits();
        let shift = single + 3;
        for weights in [
            HevcWeights {
                w0: 1,
                w1: 0,
                round: 1 << (single - 1),
                shift: single,
            },
            HevcWeights {
                w0: 0,
                w1: 1,
                round: 1 << (single - 1),
                shift: single,
            },
            HevcWeights {
                w0: 1,
                w1: 1,
                round: 1 << single,
                shift: single + 1,
            },
            HevcWeights {
                w0: 4,
                w1: 4,
                round: 1 << (shift - 1),
                shift,
            },
            HevcWeights {
                w0: -2,
                w1: 10,
                round: 1 << (shift - 1),
                shift,
            },
            HevcWeights {
                w0: 63,
                w1: -31,
                round: -1024,
                shift: 10,
            },
        ] {
            let stride = w + 4;
            let mut d0 = vec![0u16; stride * h];
            let mut d1 = vec![0u16; stride * h];
            cand.hevc_weighted_prediction(
                weights,
                &a,
                &b,
                &mut PlaneViewMut::new(&mut d0, stride, w, h).expect("dst"),
            );
            refer.hevc_weighted_prediction(
                weights,
                &a,
                &b,
                &mut PlaneViewMut::new(&mut d1, stride, w, h).expect("dst"),
            );
            report.cases += 1;
            if d0 != d1 {
                report.failures.push(format!(
                    "u16 hevc_weighted_prediction {depth:?} {w}x{h} {weights:?}"
                ));
            }
        }
    }
}

fn check_refinement_and_gpm(
    depth: BitDepth,
    (cand, refer): (DepthKernels<'_>, DepthKernels<'_>),
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    for &(w, h) in &[(8, 8), (12, 12), (16, 16), (20, 20), (32, 16), (36, 12)] {
        let stride = w + 4;
        let buf = sample_row(rng, depth, 5, stride * (h + 1));
        let win = PlaneView::new(&buf, stride, w + 1, h + 1).expect("bilinear win");
        for phases in [
            (0, 0),
            (8, 0),
            (0, 8),
            (8, 8),
            (15, 15),
            (rng.below(16), rng.below(16)),
        ] {
            let mut out = [vec![0i16; w * h], vec![0i16; w * h]];
            cand.vvc_dmvr_bilinear((w, h), &win, phases, &mut out[0]);
            refer.vvc_dmvr_bilinear((w, h), &win, phases, &mut out[1]);
            report.cases += 1;
            if out[0] != out[1] {
                report.failures.push(format!(
                    "u16 vvc_dmvr_bilinear {depth:?} {w}x{h} phases={phases:?}"
                ));
            }
        }
    }
    for size in [(4, 4), (8, 4), (4, 8), (8, 8), (12, 8), (16, 12), (16, 16)] {
        let len = (size.0 + 2) * (size.1 + 2);
        for pat in 0..3 {
            let a: Vec<i16> = (0..len)
                .map(|_| (rng.below(28000) as i32 - 6000) as i16)
                .collect();
            let b: Vec<i16> = (0..len)
                .map(|_| (rng.below(28000) as i32 - 6000) as i16)
                .collect();
            let mut out = [vec![0u16; size.0 * size.1], vec![0u16; size.0 * size.1]];
            cand.vvc_bdof(size, [&a, &b], &mut out[0]);
            refer.vvc_bdof(size, [&a, &b], &mut out[1]);
            report.cases += 1;
            if out[0] != out[1] {
                report
                    .failures
                    .push(format!("u16 vvc_bdof {depth:?} size={size:?} pat={pat}"));
            }
            let (pw, ph) = (size.0 + 2, size.1 + 2);
            let stride = pw + 5;
            let ref_buf = sample_row(rng, depth, pat + 2, stride * ph);
            let win = PlaneView::new(&ref_buf, stride, pw, ph).expect("bdof pad win");
            let interior: Vec<i16> = (0..size.0 * size.1)
                .map(|_| rng.next_u64() as i16)
                .collect();
            let mut pad = [vec![-12345i16; pw * ph + 2], vec![-12345i16; pw * ph + 2]];
            cand.vvc_bdof_pad(size, &win, &interior, &mut pad[0][1..=pw * ph]);
            refer.vvc_bdof_pad(size, &win, &interior, &mut pad[1][1..=pw * ph]);
            report.cases += 1;
            if pad[0] != pad[1] {
                report.failures.push(format!(
                    "u16 vvc_bdof_pad {depth:?} size={size:?} pat={pat}"
                ));
            }
        }
    }
    for &(cu_w, cu_h) in &[(8, 8), (16, 8), (16, 16), (32, 16), (64, 32)] {
        for chroma in [false, true] {
            let sub = usize::from(chroma);
            let (w, h) = (cu_w >> sub, cu_h >> sub);
            let a: Vec<i16> = (0..w * h)
                .map(|_| (rng.below(36000) as i32 - 6000) as i16)
                .collect();
            let b: Vec<i16> = (0..w * h)
                .map(|_| (rng.below(36000) as i32 - 6000) as i16)
                .collect();
            for split in [0u8, 13, 27, 41, rng.below(64) as u8] {
                let mask = VvcGeometric::new((cu_w, cu_h), split);
                for origin in [(0, 0), (4, 8), (600, 12)] {
                    let mut out = [vec![0u16; w * h], vec![0u16; w * h]];
                    cand.vvc_geometric_blend(mask, origin, (w, h), chroma, (&a, &b), &mut out[0]);
                    refer.vvc_geometric_blend(mask, origin, (w, h), chroma, (&a, &b), &mut out[1]);
                    report.cases += 1;
                    if out[0] != out[1] {
                        report.failures.push(format!(
                            "u16 vvc_geometric_blend {depth:?} cu={cu_w}x{cu_h} chroma={chroma} split={split} origin={origin:?}"
                        ));
                    }
                }
            }
        }
    }
}

fn check_rpr(
    depth: BitDepth,
    (cand, refer): (DepthKernels<'_>, DepthKernels<'_>),
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let ref_w = 80;
    let ref_h = 64;
    let buf = sample_row(rng, depth, 5, ref_w * ref_h);
    let view = PlaneView::new(&buf, ref_w, ref_w, ref_h).expect("rpr view");
    for &(luma, affine) in &[(true, false), (true, true), (false, false)] {
        for filter in [
            VvcRprFilter::Regular,
            VvcRprFilter::AlternativeHalf,
            VvcRprFilter::Ratio1p5,
            VvcRprFilter::Ratio2,
        ] {
            for &(w, h) in &[(2, 4), (4, 4), (8, 8), (12, 8), (16, 16), (36, 8)] {
                let block = VvcScaledBlock {
                    size: (w, h),
                    origin: (
                        rng.below(40 * 1024) as i32 - 2048,
                        rng.below(24 * 1024) as i32 - 2048,
                    ),
                    step: (900 + rng.below(1100) as i32, 900 + rng.below(1100) as i32),
                    bounds: [0, 0, ref_w as i32 - 1, ref_h as i32 - 1],
                };
                let mut out = [vec![0i16; w * h], vec![0i16; w * h]];
                cand.vvc_predict_scaled(&view, &block, [filter; 2], (luma, affine), &mut out[0]);
                refer.vvc_predict_scaled(&view, &block, [filter; 2], (luma, affine), &mut out[1]);
                report.cases += 1;
                if out[0] != out[1] {
                    report.failures.push(format!(
                        "u16 vvc_predict_scaled {depth:?} luma={luma} affine={affine} {filter:?} {w}x{h}"
                    ));
                }
            }
        }
    }
}

fn check_transforms_and_recon(
    depth: BitDepth,
    (cand, refer): (DepthKernels<'_>, DepthKernels<'_>),
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let kinds_32 = [
        VvcTransformKind::Dct2,
        VvcTransformKind::Dst7,
        VvcTransformKind::Dct8,
    ];
    for &(w, h) in &[
        (4, 4),
        (8, 4),
        (4, 16),
        (8, 8),
        (16, 8),
        (16, 16),
        (32, 16),
        (32, 32),
        (64, 8),
        (16, 64),
        (64, 64),
    ] {
        let kh_list: &[VvcTransformKind] = if w == 64 {
            &[VvcTransformKind::Dct2]
        } else {
            &kinds_32
        };
        let kv_list: &[VvcTransformKind] = if h == 64 {
            &[VvcTransformKind::Dct2]
        } else {
            &kinds_32
        };
        for &kh in kh_list {
            for &kv in kv_list {
                for sparse in [true, false] {
                    let levels: Vec<i32> = (0..w * h)
                        .map(|i| {
                            if sparse && (i % w >= 3 || i / w >= 3) {
                                0
                            } else {
                                rng.below(513) as i32 - 256
                            }
                        })
                        .collect();
                    let qp = 22 + depth.above_eight() as u8 * 6;
                    let mut out = [vec![0i16; w * h], vec![0i16; w * h]];
                    cand.vvc_inverse_primary((w, h), [kh, kv], &levels, qp, false, &mut out[0]);
                    refer.vvc_inverse_primary((w, h), [kh, kv], &levels, qp, false, &mut out[1]);
                    report.cases += 1;
                    if out[0] != out[1] {
                        report.failures.push(format!(
                            "u16 vvc_inverse_primary {depth:?} {w}x{h} {kh:?}x{kv:?} sparse={sparse}"
                        ));
                    }
                }
            }
        }
        let lfnst = VvcLfnst {
            index: 1,
            direction: 18,
            prediction_size: (w, h),
        };
        let levels: Vec<i32> = (0..w * h)
            .map(|i| {
                if i % w < 4 && i / w < 4 {
                    rng.below(257) as i32 - 128
                } else {
                    0
                }
            })
            .collect();
        let mut out = [vec![0i16; w * h], vec![0i16; w * h]];
        cand.vvc_inverse_lfnst((w, h), lfnst, &levels, 24, true, &mut out[0]);
        refer.vvc_inverse_lfnst((w, h), lfnst, &levels, 24, true, &mut out[1]);
        report.cases += 1;
        if out[0] != out[1] {
            report
                .failures
                .push(format!("u16 vvc_inverse_lfnst {depth:?} {w}x{h}"));
        }
    }
    check_hevc_inverse_and_residual(depth, (cand, refer), rng, report);
}

fn check_hevc_inverse_and_residual(
    depth: BitDepth,
    (cand, refer): (DepthKernels<'_>, DepthKernels<'_>),
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    for tx in [
        HevcTransform::Dst4,
        HevcTransform::Dct4,
        HevcTransform::Dct8,
        HevcTransform::Dct16,
        HevcTransform::Dct32,
    ] {
        let n = match tx {
            HevcTransform::Dst4 | HevcTransform::Dct4 => 4,
            HevcTransform::Dct8 => 8,
            HevcTransform::Dct16 => 16,
            HevcTransform::Dct32 => 32,
        };
        let coeffs: Vec<i16> = (0..n * n)
            .map(|_| (rng.below(16384) as i32 - 8192) as i16)
            .collect();
        let stride = n + 4;
        let base = sample_row(rng, depth, 5, stride * n);
        let mut d0 = base.clone();
        let mut d1 = base;
        cand.hevc_inverse_add(
            tx,
            &coeffs,
            (n, n),
            &mut PlaneViewMut::new(&mut d0, stride, n, n).expect("dst"),
        );
        refer.hevc_inverse_add(
            tx,
            &coeffs,
            (n, n),
            &mut PlaneViewMut::new(&mut d1, stride, n, n).expect("dst"),
        );
        report.cases += 1;
        if d0 != d1 {
            report
                .failures
                .push(format!("u16 hevc_inverse_add {depth:?} {tx:?}"));
        }
    }
    for &(w, h) in &[(1, 16), (2, 8), (4, 4), (8, 8), (16, 16), (32, 8), (64, 4)] {
        let base = sample_row(rng, depth, 5, w * h);
        let res: Vec<i16> = (0..w * h)
            .map(|i| match i % 5 {
                0 => i16::MIN,
                1 => i16::MAX,
                _ => (rng.below(4096) as i32 - 2048) as i16,
            })
            .collect();
        let mut p0 = base.clone();
        let mut p1 = base;
        cand.vvc_add_residual((w, h), &mut p0, &res);
        refer.vvc_add_residual((w, h), &mut p1, &res);
        report.cases += 1;
        if p0 != p1 {
            report
                .failures
                .push(format!("u16 vvc_add_residual {depth:?} {w}x{h}"));
        }
    }
}

fn check_lmcs(
    depth: BitDepth,
    (cand, refer): (DepthKernels<'_>, DepthKernels<'_>),
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let shift = depth.above_eight();
    let mut deltas = [0i16; 16];
    deltas[2] = (4 << shift) as i16;
    deltas[5] = -(4 << shift) as i16;
    deltas[9] = (8 << shift) as i16;
    deltas[12] = -(8 << shift) as i16;
    let model = VvcLmcs::<u16>::new_at([1, 14], deltas, 0, depth).expect("valid LMCS model");
    for len in [4, 7, 16, 31, 32, 35, 68, 128] {
        let base = sample_row(rng, depth, 5, len);
        let (mut f0, mut f1) = (base.clone(), base.clone());
        cand.vvc_forward_lmcs(&model, &mut f0);
        refer.vvc_forward_lmcs(&model, &mut f1);
        report.cases += 1;
        if f0 != f1 {
            report
                .failures
                .push(format!("u16 vvc_forward_lmcs {depth:?} len={len}"));
        }
        let (mut i0, mut i1) = (base.clone(), base);
        cand.vvc_inverse_lmcs(
            &model,
            &mut PlaneViewMut::new(&mut i0, len, len, 1).expect("lmcs"),
        );
        refer.vvc_inverse_lmcs(
            &model,
            &mut PlaneViewMut::new(&mut i1, len, len, 1).expect("lmcs"),
        );
        report.cases += 1;
        if i0 != i1 {
            report
                .failures
                .push(format!("u16 vvc_inverse_lmcs {depth:?} len={len}"));
        }
    }
}

fn check_sao(
    depth: BitDepth,
    (cand, refer): (DepthKernels<'_>, DepthKernels<'_>),
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let lim = 1i32 << (depth.bits().min(10) - 5);
    for w in [3, 8, 15, 16, 31, 32, 35, 64, 67] {
        let h = 6;
        let (pw, ph, stride) = (w + 2, h + 2, w + 5);
        for pat in [2, 4, 5] {
            let src = sample_row(rng, depth, pat, stride * ph);
            let input = PlaneView::new(&src, stride, pw, ph).expect("sao in");
            let offs: [i16; 4] =
                std::array::from_fn(|_| (rng.below((2 * lim) as usize) as i32 - lim) as i16);
            for class in 0..4u8 {
                let (mut d0, mut d1) = (src.clone(), src.clone());
                cand.sao_edge_apply(
                    class,
                    offs,
                    &input,
                    &mut PlaneViewMut::new(&mut d0, stride, pw, ph).expect("sao out"),
                    (1, 1, 1 + w, 1 + h),
                );
                refer.sao_edge_apply(
                    class,
                    offs,
                    &input,
                    &mut PlaneViewMut::new(&mut d1, stride, pw, ph).expect("sao out"),
                    (1, 1, 1 + w, 1 + h),
                );
                report.cases += 1;
                if d0 != d1 {
                    report
                        .failures
                        .push(format!("u16 sao_edge_apply {depth:?} class={class} w={w}"));
                }
            }
            for masked in [false, true] {
                let pos = (28 + rng.below(4)) as u8;
                let exempt: Vec<bool> = if masked {
                    (0..w * h).map(|_| rng.below(5) == 0).collect()
                } else {
                    Vec::new()
                };
                let (mut b0, mut b1) = (src.clone(), src.clone());
                cand.sao_band_apply_masked(
                    pos,
                    offs,
                    &mut PlaneViewMut::new(&mut b0, stride, pw, ph).expect("band"),
                    (1, 1, 1 + w, 1 + h),
                    &exempt,
                );
                refer.sao_band_apply_masked(
                    pos,
                    offs,
                    &mut PlaneViewMut::new(&mut b1, stride, pw, ph).expect("band"),
                    (1, 1, 1 + w, 1 + h),
                    &exempt,
                );
                report.cases += 1;
                if b0 != b1 {
                    report.failures.push(format!(
                        "u16 sao_band_apply {depth:?} w={w} masked={masked}"
                    ));
                }
            }
        }
    }
}

const DEBLOCK_REACHES: [[usize; 2]; 11] = [
    [1, 1],
    [2, 2],
    [3, 3],
    [3, 5],
    [3, 7],
    [5, 3],
    [5, 5],
    [5, 7],
    [7, 3],
    [7, 5],
    [7, 7],
];

fn deblock_plane(rng: &mut Rng, depth: BitDepth, mode: usize, len: usize) -> Vec<u16> {
    let max = depth.max() as u32;
    let mut level = max / 2;
    (0..len)
        .map(|i| {
            if mode == 0 {
                return sample_row(rng, depth, 2, 1)[0];
            }
            if i % 9 == 0 {
                level = rng.below(max as usize + 1) as u32;
            }
            (level + rng.below(9) as u32).min(max) as u16
        })
        .collect()
}

fn check_deblock(
    depth: BitDepth,
    (cand, refer): (DepthKernels<'_>, DepthKernels<'_>),
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let side = 32;
    for (bp, tcp) in [(18, 12), (52, 95), (88, 395)] {
        let (beta, tc) = (vvc_deblock_beta(bp, depth), vvc_deblock_tc(tcp, depth));
        for mode in 0..2 {
            let base = deblock_plane(rng, depth, mode, side * side);
            for lengths in DEBLOCK_REACHES {
                for vertical in [false, true] {
                    let (mut p0, mut p1) = (base.clone(), base.clone());
                    cand.vvc_deblock_luma_edge(
                        &mut PlaneViewMut::new(&mut p0, side, side, side).expect("luma8"),
                        (8, 8),
                        vertical,
                        beta,
                        tc,
                        lengths,
                    );
                    refer.vvc_deblock_luma_edge(
                        &mut PlaneViewMut::new(&mut p1, side, side, side).expect("luma8"),
                        (8, 8),
                        vertical,
                        beta,
                        tc,
                        lengths,
                    );
                    let (mut s0, mut s1) = (base.clone(), base.clone());
                    cand.vvc_deblock_luma_segment(
                        &mut PlaneViewMut::new(&mut s0, side, side, side).expect("luma4"),
                        (8, 12),
                        vertical,
                        beta,
                        tc,
                        lengths,
                    );
                    refer.vvc_deblock_luma_segment(
                        &mut PlaneViewMut::new(&mut s1, side, side, side).expect("luma4"),
                        (8, 12),
                        vertical,
                        beta,
                        tc,
                        lengths,
                    );
                    report.cases += 2;
                    if p0 != p1 || s0 != s1 {
                        report.failures.push(format!(
                            "u16 vvc_deblock_luma_edge/seg {depth:?} {lengths:?} vert={vertical}"
                        ));
                    }
                }
                let mut e0: [[u16; 16]; 4] =
                    std::array::from_fn(|r| std::array::from_fn(|c| base[r * 16 + c]));
                let mut e1 = e0;
                cand.vvc_deblock_luma(&mut e0, beta, tc, lengths);
                refer.vvc_deblock_luma(&mut e1, beta, tc, lengths);
                report.cases += 1;
                if e0 != e1 {
                    report
                        .failures
                        .push(format!("u16 vvc_deblock_luma {depth:?} {lengths:?}"));
                }
            }
            check_deblock_chroma(depth, (cand, refer), &base, (beta, tc), report);
        }
    }
}

fn check_deblock_chroma(
    depth: BitDepth,
    (cand, refer): (DepthKernels<'_>, DepthKernels<'_>),
    base: &[u16],
    (beta, tc): (i32, i32),
    report: &mut CheckReport,
) {
    let side = 32;
    for (large, boundary) in [(false, false), (true, false), (true, true)] {
        for (lines, vertical) in [(2, false), (4, true), (6, false), (8, true)] {
            let (mut c0, mut c1) = (base.to_vec(), base.to_vec());
            cand.vvc_deblock_chroma_edge(
                &mut PlaneViewMut::new(&mut c0, side, side, side).expect("chr"),
                (8, 8),
                vertical,
                lines,
                beta,
                tc,
                large,
                boundary,
            );
            refer.vvc_deblock_chroma_edge(
                &mut PlaneViewMut::new(&mut c1, side, side, side).expect("chr"),
                (8, 8),
                vertical,
                lines,
                beta,
                tc,
                large,
                boundary,
            );
            report.cases += 1;
            if c0 != c1 {
                report.failures.push(format!(
                    "u16 vvc_deblock_chroma_edge {depth:?} lines={lines} vert={vertical}"
                ));
            }
        }
        let mut g0: [[u16; 8]; 2] =
            std::array::from_fn(|r| std::array::from_fn(|c| base[r * 8 + c]));
        let mut g1 = g0;
        cand.vvc_deblock_chroma(&mut g0, beta, tc, large, boundary);
        refer.vvc_deblock_chroma(&mut g1, beta, tc, large, boundary);
        report.cases += 1;
        if g0 != g1 {
            report.failures.push(format!(
                "u16 vvc_deblock_chroma {depth:?} large={large} bnd={boundary}"
            ));
        }
    }
}

fn check_vvc_intra(
    depth: BitDepth,
    (cand, refer): (DepthKernels<'_>, DepthKernels<'_>),
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let modes = [
        VvcIntraMode::Planar,
        VvcIntraMode::Dc,
        VvcIntraMode::Horizontal,
        VvcIntraMode::Vertical,
    ];
    for side in [4, 8, 16, 32] {
        for pat in [2, 4, 5] {
            let top: Vec<i32> = sample_row(rng, depth, pat, 2 * side)
                .into_iter()
                .map(i32::from)
                .collect();
            let left: Vec<i32> = sample_row(rng, depth, pat, 2 * side)
                .into_iter()
                .map(i32::from)
                .collect();
            let corner = i32::from(sample_row(rng, depth, pat, 1)[0]);
            for mode in modes.into_iter().chain((2..=66).flat_map(|direction| {
                [false, true].map(|luma| VvcIntraMode::Angular { direction, luma })
            })) {
                let (mut a0, mut a1) = (
                    vec![0xA5A5u16; side * side + 2],
                    vec![0xA5A5u16; side * side + 2],
                );
                cand.vvc_intra_predict(side, mode, &top, &left, corner, &mut a0[1..=side * side]);
                refer.vvc_intra_predict(side, mode, &top, &left, corner, &mut a1[1..=side * side]);
                let (mut o0, mut o1) = (vec![0u16; side * side], vec![0u16; side * side]);
                let t0 = cand.vvc_intra_predict_oriented(side, mode, &top, &left, corner, &mut o0);
                let t1 = refer.vvc_intra_predict_oriented(side, mode, &top, &left, corner, &mut o1);
                report.cases += 2;
                if a0 != a1 || t0 != t1 || o0 != o1 {
                    report.failures.push(format!(
                        "u16 vvc_intra_predict {depth:?} {side}x{side} {mode:?} pat={pat}"
                    ));
                }
            }
        }
    }
    check_vvc_intra_rect(depth, (cand, refer), rng, report);
}

fn check_vvc_intra_rect(
    depth: BitDepth,
    (cand, refer): (DepthKernels<'_>, DepthKernels<'_>),
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let rect_sizes = [
        (2, 4),
        (4, 2),
        (4, 8),
        (8, 4),
        (8, 16),
        (16, 8),
        (32, 8),
        (8, 32),
        (64, 16),
        (16, 64),
        (64, 64),
    ];
    let base_modes = [
        VvcIntraMode::Planar,
        VvcIntraMode::Dc,
        VvcIntraMode::Horizontal,
        VvcIntraMode::Vertical,
    ];
    for &(w, h) in &rect_sizes {
        let top: Vec<i32> = sample_row(rng, depth, 5, 2 * w)
            .into_iter()
            .map(i32::from)
            .collect();
        let left: Vec<i32> = sample_row(rng, depth, 5, 2 * h)
            .into_iter()
            .map(i32::from)
            .collect();
        let corner = i32::from(sample_row(rng, depth, 5, 1)[0]);
        for mode in base_modes.into_iter().chain((2..=66).flat_map(|direction| {
            [false, true].map(|luma| VvcIntraMode::Angular { direction, luma })
        })) {
            // 2xN and Nx2 are chroma-only in VVC (PDPC luma shifts assume >= 4).
            if matches!(mode, VvcIntraMode::Angular { luma: true, .. }) && w.min(h) < 4 {
                continue;
            }
            let (mut a0, mut a1) = (vec![0xA5A5u16; w * h + 2], vec![0xA5A5u16; w * h + 2]);
            cand.vvc_intra_predict_rect((w, h), mode, &top, &left, corner, &mut a0[1..=w * h]);
            refer.vvc_intra_predict_rect((w, h), mode, &top, &left, corner, &mut a1[1..=w * h]);
            report.cases += 1;
            if a0 != a1 {
                report.failures.push(format!(
                    "u16 vvc_intra_predict_rect {depth:?} {w}x{h} {mode:?}"
                ));
            }
        }
    }
    check_vvc_mrl_and_isp(depth, (cand, refer), rng, report);
}

fn check_vvc_mrl_and_isp(
    depth: BitDepth,
    (cand, refer): (DepthKernels<'_>, DepthKernels<'_>),
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    for &(w, h) in &[(4, 4), (4, 8), (8, 4), (8, 16), (16, 8), (32, 16), (64, 8)] {
        let top: Vec<i32> = sample_row(rng, depth, 5, 2 * w + 2)
            .into_iter()
            .map(i32::from)
            .collect();
        let left: Vec<i32> = sample_row(rng, depth, 5, 2 * h + 2)
            .into_iter()
            .map(i32::from)
            .collect();
        let corner = i32::from(sample_row(rng, depth, 5, 1)[0]);
        for line in [1u8, 2] {
            for direction in 1..=66u8 {
                let (mut a0, mut a1) = (vec![0xA5A5u16; w * h + 2], vec![0xA5A5u16; w * h + 2]);
                cand.vvc_mrl_predict(
                    (w, h),
                    direction,
                    line,
                    &top,
                    &left,
                    corner,
                    &mut a0[1..=w * h],
                );
                refer.vvc_mrl_predict(
                    (w, h),
                    direction,
                    line,
                    &top,
                    &left,
                    corner,
                    &mut a1[1..=w * h],
                );
                report.cases += 1;
                if a0 != a1 {
                    report.failures.push(format!(
                        "u16 vvc_mrl_predict {depth:?} {w}x{h} line={line} dir={direction}"
                    ));
                }
            }
        }
    }
    for &(cu, (w, h)) in &[
        ((16, 4), (16, 1)),
        ((8, 8), (8, 2)),
        ((8, 4), (4, 4)),
        ((4, 16), (4, 4)),
        ((16, 16), (16, 4)),
        ((16, 16), (4, 16)),
        ((32, 16), (32, 4)),
        ((64, 32), (16, 32)),
    ] {
        let top: Vec<i32> = sample_row(rng, depth, 5, 2 * w)
            .into_iter()
            .map(i32::from)
            .collect();
        let left: Vec<i32> = sample_row(rng, depth, 5, 2 * h)
            .into_iter()
            .map(i32::from)
            .collect();
        let corner = i32::from(sample_row(rng, depth, 5, 1)[0]);
        for direction in 0..=66u8 {
            let (mut a0, mut a1) = (vec![0xA5A5u16; w * h + 2], vec![0xA5A5u16; w * h + 2]);
            cand.vvc_intra_predict_isp(
                cu,
                (w, h),
                direction,
                &top,
                &left,
                corner,
                &mut a0[1..=w * h],
            );
            refer.vvc_intra_predict_isp(
                cu,
                (w, h),
                direction,
                &top,
                &left,
                corner,
                &mut a1[1..=w * h],
            );
            report.cases += 1;
            if a0 != a1 {
                report.failures.push(format!(
                    "u16 vvc_intra_predict_isp {depth:?} cu={cu:?} {w}x{h} dir={direction}"
                ));
            }
        }
    }
}

fn check_hevc_intra(
    depth: BitDepth,
    (cand, refer): (DepthKernels<'_>, DepthKernels<'_>),
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    for n in [4, 8, 16, 32] {
        for pat in [2, 4, 5] {
            let samples = sample_row(rng, depth, pat, 2 * n + 2);
            let edges = HevcPlanarEdges {
                above: &samples[..n],
                left: &samples[n..2 * n],
                top_right: samples[2 * n],
                bottom_left: samples[2 * n + 1],
            };
            let (mut p0, mut p1) = (vec![0xA5A5u16; n * n + 2], vec![0xA5A5u16; n * n + 2]);
            cand.hevc_intra_planar(&edges, n, &mut p0[1..=n * n]);
            refer.hevc_intra_planar(&edges, n, &mut p1[1..=n * n]);
            report.cases += 1;
            if p0 != p1 {
                report
                    .failures
                    .push(format!("u16 hevc_intra_planar {depth:?} {n}x{n} pat={pat}"));
            }
            let refs = sample_row(rng, depth, pat, 3 * n + 1);
            for angle in -32..=32 {
                let (mut a0, mut a1) = (vec![0xA5A5u16; n * n + 2], vec![0xA5A5u16; n * n + 2]);
                cand.hevc_intra_angular(&refs, n, angle, &mut a0[1..=n * n]);
                refer.hevc_intra_angular(&refs, n, angle, &mut a1[1..=n * n]);
                report.cases += 1;
                if a0 != a1 {
                    report.failures.push(format!(
                        "u16 hevc_intra_angular {depth:?} {n}x{n} angle={angle} pat={pat}"
                    ));
                }
            }
        }
    }
}
