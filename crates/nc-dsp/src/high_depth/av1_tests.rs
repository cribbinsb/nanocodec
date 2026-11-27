//! The AV1 `u16` routes at 8 bits equal the 8-bit kernels (one source for
//! both depths), family by family, and stay inside the depth at 10 bits.

use nc_common::{BitDepth, Picture, PlaneIndex, PlaneView, PlaneViewMut, Sample};

use super::av1::Av1Pixel;
use crate::av1_film_grain::{Av1FilmGrainParams, Av1ScalingPoints};
use crate::av1_mc::blend::{Av1CompoundPred, Av1Mask};
use crate::av1_mc::convolve::{AV1_UNSCALED_STEP, Av1McAxis};
use crate::av1_mc::filter::{AV1_FILTER_TAPS, Av1Filter, Av1Filters};
use crate::av1_mc::warp::av1_setup_shear;
use crate::av1_restoration::{
    Av1LoopRestoration, Av1LrFilter, Av1LrScratch, Av1SgrProj, Av1StripeBoundaries, Av1Wiener,
};
use crate::intra_prediction::av1_block::{BlockEdges, EdgeAvail, EdgeFilterCtx, IntraKind};
use crate::intra_prediction::filter::FilterIntraMode;
use crate::{
    Av1EdgeFilter, Av1FilterLength, Av1TxSize, Av1TxType, CdefLines, CdefSets, CdefStrengths,
    CdefUnit, DepthKernels, Dsp, EdgeRun,
};

/// A deterministic pattern of samples up to `max`, with flat runs and
/// edges (so the filters' decisions take both ways).
pub(super) fn pattern(len: usize, seed: usize, max: u32) -> Vec<u32> {
    (0..len)
        .map(|i| {
            let v = (i.wrapping_mul(2_654_435_761) + seed * 40_503) >> 7;
            match (i / 11 + seed) % 9 {
                0 => max,
                1 | 2 => max / 2 + (seed as u32 % 3),
                _ => (v as u32) % (max + 1),
            }
        })
        .collect()
}

pub(super) fn samples<T: Sample>(values: &[u32]) -> Vec<T> {
    values.iter().map(|&v| T::from_i32(v as i32)).collect()
}

pub(super) fn widened(values: &[u8]) -> Vec<u16> {
    values.iter().map(|&v| u16::from(v)).collect()
}

/// A picture filled with `pattern(.., seed, max)`.
fn picture<T: Sample>((width, height): (usize, usize), seed: usize, max: u32) -> Picture<T> {
    let mut picture = Picture::<T>::new(width, height, 16).unwrap();
    for (index, plane) in PlaneIndex::ALL.into_iter().enumerate() {
        let mut view = picture.plane_mut(plane).view_mut();
        let (w, h) = (view.width(), view.height());
        let values = samples::<T>(&pattern(w * h, seed + index, max));
        for (y, row) in values.chunks_exact(w).enumerate() {
            view.row_mut(y)[..w].copy_from_slice(row);
        }
    }
    picture.extend_borders();
    picture
}

/// Every plane's samples, widened.
fn contents<T: Sample>(picture: &Picture<T>) -> Vec<u16> {
    PlaneIndex::ALL
        .into_iter()
        .flat_map(|plane| {
            let view = picture.plane(plane).view();
            (0..view.height())
                .flat_map(move |y| {
                    view.row(y)[..view.width()]
                        .iter()
                        .map(|&s| s.to_i32() as u16)
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Run `f` with the 8-bit route and with the `u16` route at 8 bits.
fn both(dsp: &Dsp) -> DepthKernels<'_> {
    dsp.at(BitDepth::EIGHT)
}

#[test]
fn inverse_transforms_at_eight_bits_match() {
    let dsp = Dsp::new();
    let k = both(&dsp);
    for (t, &tx) in Av1TxSize::ALL.iter().enumerate() {
        for kind in Av1TxType::ALL.into_iter().filter(|kind| kind.fits(tx)) {
            for seed in 0..3 {
                let area = tx.coded_area();
                // Moderate coefficients: within every intermediate range,
                // so both clamping orders agree.
                let coeffs: Vec<i32> = pattern(area, seed + t, 600)
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        if i > 24 + seed * 40 {
                            0
                        } else {
                            v as i32 - 300
                        }
                    })
                    .collect();
                let (w, h) = (tx.width(), tx.height());
                let prediction = pattern(w * h, seed, 255);
                let mut out8 = samples::<u8>(&prediction);
                let mut out16 = samples::<u16>(&prediction);
                let mut c8: Vec<i16> = coeffs.iter().map(|&c| c as i16).collect();
                let mut c16 = coeffs.clone();
                k.av1_inverse_add::<u8>(
                    tx,
                    kind,
                    &mut c8,
                    area,
                    &mut PlaneViewMut::new(&mut out8, w, w, h).unwrap(),
                );
                k.av1_inverse_add::<u16>(
                    tx,
                    kind,
                    &mut c16,
                    area,
                    &mut PlaneViewMut::new(&mut out16, w, w, h).unwrap(),
                );
                assert_eq!(widened(&out8), out16, "{tx:?} {kind:?} seed {seed}");
            }
        }
    }
}

#[test]
fn intra_predictions_at_eight_bits_match() {
    let dsp = Dsp::new();
    let k = both(&dsp);
    let size = (96, 96);
    let p8 = picture::<u8>(size, 3, 255);
    let p16 = picture::<u16>(size, 3, 255);
    let origin = (32, 32);
    for (w, h) in [(4, 4), (8, 8), (16, 8), (4, 16), (32, 32), (8, 32)] {
        for avail_bits in [0b1111_u8, 0b0011, 0b0101, 0b0000] {
            let avail = EdgeAvail {
                have_above: avail_bits & 1 != 0,
                have_left: avail_bits & 2 != 0,
                have_above_right: avail_bits & 4 != 0,
                have_below_left: avail_bits & 8 != 0,
            };
            let e8 = BlockEdges::read(
                p8.plane(PlaneIndex::Y),
                origin,
                (w, h),
                avail,
                BitDepth::EIGHT,
            );
            let e16 = BlockEdges::read(
                p16.plane(PlaneIndex::Y),
                origin,
                (w, h),
                avail,
                BitDepth::EIGHT,
            );
            let mut kinds = vec![
                IntraKind::Dc,
                IntraKind::Smooth,
                IntraKind::SmoothVertical,
                IntraKind::SmoothHorizontal,
                IntraKind::Paeth,
            ];
            for angle in [36, 45, 81, 90, 113, 135, 157, 180, 203, 222] {
                for edge_filter in [None, Some(EdgeFilterCtx::default())] {
                    kinds.push(IntraKind::Directional { angle, edge_filter });
                }
            }
            for kind in kinds {
                let mut o8 = vec![0u8; w * h];
                let mut o16 = vec![0u16; w * h];
                k.av1_intra_predict_block(
                    &e8,
                    kind,
                    &mut PlaneViewMut::new(&mut o8, w, w, h).unwrap(),
                );
                k.av1_intra_predict_block(
                    &e16,
                    kind,
                    &mut PlaneViewMut::new(&mut o16, w, w, h).unwrap(),
                );
                assert_eq!(widened(&o8), o16, "{w}x{h} {kind:?} {avail:?}");
            }
            if w <= 32 && h <= 32 {
                for mode in FilterIntraMode::ALL {
                    let mut o8 = vec![0u8; w * h];
                    let mut o16 = vec![0u16; w * h];
                    k.av1_intra_predict_filter_block(
                        &e8,
                        mode,
                        &mut PlaneViewMut::new(&mut o8, w, w, h).unwrap(),
                    );
                    k.av1_intra_predict_filter_block(
                        &e16,
                        mode,
                        &mut PlaneViewMut::new(&mut o16, w, w, h).unwrap(),
                    );
                    assert_eq!(widened(&o8), o16, "{w}x{h} {mode:?}");
                }
            }
        }
    }
}

#[test]
fn inter_predictions_and_blends_at_eight_bits_match() {
    let dsp = Dsp::new();
    let k = both(&dsp);
    let (w, h) = (16, 8);
    let filters = [
        Av1Filters::both(Av1Filter::Regular),
        Av1Filters::both(Av1Filter::Sharp),
        Av1Filters {
            x: Av1Filter::Smooth,
            y: Av1Filter::Bilinear,
        },
    ];
    let axes = [
        (Av1McAxis::unscaled(0), Av1McAxis::unscaled(0)),
        (Av1McAxis::unscaled(5), Av1McAxis::unscaled(11)),
        (
            Av1McAxis {
                position: 300,
                step: AV1_UNSCALED_STEP * 3 / 2,
            },
            Av1McAxis {
                position: 700,
                step: AV1_UNSCALED_STEP / 2 + 3,
            },
        ),
    ];
    // Kernel sides (one tiled from 64), dual filters, 2-sample sides and
    // scaled steps.
    let sizes = [(w, h), (4, 4), (64, 128), (8, 32), (2, 8)];
    for ((seed, filters), (w, h)) in filters
        .into_iter()
        .enumerate()
        .flat_map(|f| sizes.map(|s| (f, s)))
    {
        for axes in axes {
            let span = (
                axes.0.span(w).max(w + AV1_FILTER_TAPS),
                axes.1.span(h).max(h + AV1_FILTER_TAPS),
            );
            let source = pattern(span.0 * span.1, seed, 255);
            let (s8, s16) = (samples::<u8>(&source), samples::<u16>(&source));
            let v8 = PlaneView::new(&s8, span.0, span.0, span.1).unwrap();
            let v16 = PlaneView::new(&s16, span.0, span.0, span.1).unwrap();
            let mut o8 = vec![0u8; w * h];
            let mut o16 = vec![0u16; w * h];
            k.av1_predict(
                filters,
                &v8,
                (w, h),
                axes,
                &mut PlaneViewMut::new(&mut o8, w, w, h).unwrap(),
            );
            k.av1_predict(
                filters,
                &v16,
                (w, h),
                axes,
                &mut PlaneViewMut::new(&mut o16, w, w, h).unwrap(),
            );
            assert_eq!(widened(&o8), o16, "{filters:?} {axes:?}");
            let (mut c8, mut c16) = (vec![0i16; w * h], vec![0i16; w * h]);
            k.av1_predict_compound(filters, &v8, (w, h), axes, (&mut c8, w));
            k.av1_predict_compound(filters, &v16, (w, h), axes, (&mut c16, w));
            assert_eq!(c8, c16, "{filters:?} {axes:?} compound");
        }
    }
}

#[test]
fn compound_blends_at_eight_bits_match() {
    let dsp = Dsp::new();
    let k = both(&dsp);
    let (w, h) = (16, 8);
    let p0: Vec<i16> = pattern(w * h, 1, 4000)
        .iter()
        .map(|&v| v as i16 - 500)
        .collect();
    let p1: Vec<i16> = pattern(w * h, 2, 4000)
        .iter()
        .map(|&v| v as i16 - 300)
        .collect();
    let mask: Vec<u8> = pattern(w * h, 3, 64).iter().map(|&v| v as u8).collect();
    let pred = |data| Av1CompoundPred { data, stride: w };
    let m = Av1Mask {
        data: &mask,
        stride: w,
        sub_x: false,
        sub_y: false,
    };
    let (mut o8, mut o16) = (vec![0u8; w * h], vec![0u16; w * h]);
    k.av1_blend_mask(
        (pred(&p0), pred(&p1)),
        m,
        (w, h),
        &mut PlaneViewMut::new(&mut o8, w, w, h).unwrap(),
    );
    k.av1_blend_mask(
        (pred(&p0), pred(&p1)),
        m,
        (w, h),
        &mut PlaneViewMut::new(&mut o16, w, w, h).unwrap(),
    );
    assert_eq!(widened(&o8), o16, "mask blend");
    k.av1_average_compound(
        (&p0, &p1),
        (w, h),
        &mut PlaneViewMut::new(&mut o8, w, w, h).unwrap(),
    );
    k.av1_average_compound(
        (&p0, &p1),
        (w, h),
        &mut PlaneViewMut::new(&mut o16, w, w, h).unwrap(),
    );
    assert_eq!(widened(&o8), o16, "average");
}

#[test]
fn warps_at_eight_bits_match() {
    let dsp = Dsp::new();
    let k = both(&dsp);
    let size = (64, 48);
    let (r8, r16) = (picture::<u8>(size, 4, 255), picture::<u16>(size, 4, 255));
    let model = [
        1 << 10,
        -3 << 9,
        (1 << 16) + 600,
        400,
        -300,
        (1 << 16) - 500,
    ];
    let shear = av1_setup_shear(&model).expect("a valid shear");
    for (position, sub) in [
        ((8, 8), (false, false)),
        ((40, 32), (false, false)),
        ((4, 12), (true, true)),
    ] {
        let (w, h) = (16, 16);
        let v8 = r8.plane(PlaneIndex::Y).view();
        let v16 = r16.plane(PlaneIndex::Y).view();
        let (mut o8, mut o16) = (vec![0u8; w * h], vec![0u16; w * h]);
        k.av1_warp(
            &v8,
            (&model, shear),
            position,
            (w, h),
            sub,
            &mut PlaneViewMut::new(&mut o8, w, w, h).unwrap(),
        );
        k.av1_warp(
            &v16,
            (&model, shear),
            position,
            (w, h),
            sub,
            &mut PlaneViewMut::new(&mut o16, w, w, h).unwrap(),
        );
        assert_eq!(widened(&o8), o16, "warp at {position:?}");
        let (mut c8, mut c16) = (vec![0i16; w * h], vec![0i16; w * h]);
        k.av1_warp_compound(&v8, (&model, shear), position, (w, h), sub, (&mut c8, w));
        k.av1_warp_compound(&v16, (&model, shear), position, (w, h), sub, (&mut c16, w));
        assert_eq!(c8, c16, "compound warp at {position:?}");
    }
}

#[test]
fn loop_filters_at_eight_bits_match() {
    let dsp = Dsp::new();
    let k = both(&dsp);
    let size = (64, 64);
    for seed in 0..4 {
        let (mut p8, mut p16) = (
            picture::<u8>(size, seed, 255),
            picture::<u16>(size, seed, 255),
        );
        for vertical in [true, false] {
            let runs: Vec<EdgeRun<Av1EdgeFilter>> = [
                (Av1FilterLength::Four, 8),
                (Av1FilterLength::Six, 16),
                (Av1FilterLength::Eight, 24),
                (Av1FilterLength::Fourteen, 40),
            ]
            .into_iter()
            .map(|(length, at)| {
                let (x, y) = if vertical { (at, 8) } else { (8, at) };
                EdgeRun {
                    x,
                    y,
                    lines: 16,
                    filter: Av1EdgeFilter {
                        length,
                        limit: 40 + seed as u8,
                        blimit: 120,
                        thresh: 6,
                    },
                }
            })
            .collect();
            let mut lanes = crate::LoopFilterLanes::default();
            k.av1_loop_filter_runs(
                &mut p8.plane_mut(PlaneIndex::Y).view_mut(),
                vertical,
                &runs,
                &mut lanes,
            );
            k.av1_loop_filter_runs(
                &mut p16.plane_mut(PlaneIndex::Y).view_mut(),
                vertical,
                &runs,
                &mut lanes,
            );
        }
        assert_eq!(contents(&p8), contents(&p16), "seed {seed}");
    }
}

#[test]
fn cdef_at_eight_bits_matches() {
    let dsp = Dsp::new();
    let k = both(&dsp);
    let size = (64, 48);
    let sets = CdefSets::new(
        &[
            CdefStrengths {
                luma_primary: 9,
                luma_secondary: 2,
                chroma_primary: 4,
                chroma_secondary: 1,
            },
            CdefStrengths {
                luma_primary: 0,
                luma_secondary: 4,
                chroma_primary: 15,
                chroma_secondary: 0,
            },
        ],
        5,
    );
    let units: Vec<CdefUnit> = (0..size.1 / 8)
        .flat_map(|y| {
            (0..size.0 / 8).map(move |x| CdefUnit {
                x: x * 8,
                y: y * 8,
                set: ((x + y) % 2) as u8,
            })
        })
        .collect();
    let (mut p8, mut p16) = (picture::<u8>(size, 6, 255), picture::<u16>(size, 6, 255));
    let (mut l8, mut l16) = (CdefLines::<u8>::new(), CdefLines::<u16>::new());
    l8.start(&p8);
    l16.start(&p16);
    k.cdef_filter_band(&mut l8, &mut p8, 0..size.1, &units, &sets);
    k.cdef_filter_band(&mut l16, &mut p16, 0..size.1, &units, &sets);
    assert_eq!(contents(&p8), contents(&p16));
}

#[test]
fn loop_restoration_at_eight_bits_matches() {
    let dsp = Dsp::new();
    let k = both(&dsp);
    let (w, h) = (96, 80);
    let filters = [
        Av1LrFilter::Wiener(Av1Wiener {
            vertical: [3, -7, 15],
            horizontal: [-5, 8, 46],
        }),
        Av1LrFilter::SgrProj(Av1SgrProj {
            set: 3,
            xqd: [-32, 31],
        }),
        Av1LrFilter::SgrProj(Av1SgrProj {
            set: 12,
            xqd: [0, 60],
        }),
        Av1LrFilter::None,
    ];
    let restoration = Av1LoopRestoration {
        unit_size: 32,
        columns: 3,
        rows: 2,
        filters: &[
            filters[0], filters[1], filters[2], filters[2], filters[3], filters[0],
        ],
    };
    let (p8, p16) = (
        picture::<u8>((w, h), 7, 255),
        picture::<u16>((w, h), 7, 255),
    );
    let b8 = Av1StripeBoundaries::save_at(
        &p8.plane(PlaneIndex::Y).view(),
        false,
        None,
        BitDepth::EIGHT,
    );
    let b16 = Av1StripeBoundaries::save_at(
        &p16.plane(PlaneIndex::Y).view(),
        false,
        None,
        BitDepth::EIGHT,
    );
    let (mut q8, mut q16) = (p8, p16);
    k.av1_loop_restoration(
        &restoration,
        &mut q8.plane_mut(PlaneIndex::Y).view_mut(),
        (&b8, false),
        &mut Av1LrScratch::default(),
    );
    k.av1_loop_restoration(
        &restoration,
        &mut q16.plane_mut(PlaneIndex::Y).view_mut(),
        (&b16, false),
        &mut Av1LrScratch::default(),
    );
    assert_eq!(contents(&q8), contents(&q16));
}

#[test]
fn film_grain_at_eight_bits_matches() {
    let dsp = Dsp::new();
    let k = both(&dsp);
    let size = (64, 48);
    let mut points = Av1ScalingPoints::<14> {
        count: 3,
        ..Default::default()
    };
    points.value[..3].copy_from_slice(&[0, 128, 255]);
    points.scaling[..3].copy_from_slice(&[20, 80, 40]);
    let mut chroma = Av1ScalingPoints::<14> {
        count: 2,
        ..Default::default()
    };
    chroma.value[..2].copy_from_slice(&[16, 240]);
    chroma.scaling[..2].copy_from_slice(&[60, 30]);
    let mut params = Av1FilmGrainParams {
        grain_seed: 4321,
        y_points: points,
        cb_points: chroma,
        cr_points: chroma,
        grain_scaling_minus_8: 2,
        ar_coeff_lag: 2,
        ar_coeff_shift_minus_6: 1,
        cb_mult: (140, 120, 280),
        cr_mult: (100, 150, 240),
        overlap: true,
        ..Default::default()
    };
    params.ar_coeffs_y_plus_128 = core::array::from_fn(|i| 120 + (i as u8 % 17));
    params.ar_coeffs_cb_plus_128 = core::array::from_fn(|i| 131 - (i as u8 % 9));
    params.ar_coeffs_cr_plus_128 = core::array::from_fn(|i| 125 + (i as u8 % 5));
    for clip in [false, true] {
        params.clip_to_restricted_range = clip;
        let (i8_, i16_) = (picture::<u8>(size, 9, 255), picture::<u16>(size, 9, 255));
        let (mut o8, mut o16) = (
            Picture::<u8>::new(size.0, size.1, 0).unwrap(),
            Picture::<u16>::new(size.0, size.1, 0).unwrap(),
        );
        film_grain(k, &params, &i8_, size, &mut o8);
        film_grain(k, &params, &i16_, size, &mut o16);
        assert_eq!(contents(&o8), contents(&o16), "restricted range {clip}");
    }
}

/// [`DepthKernels::av1_add_film_grain`] of whole pictures.
fn film_grain<T: Av1Pixel>(
    k: DepthKernels<'_>,
    params: &Av1FilmGrainParams,
    input: &Picture<T>,
    size: (usize, usize),
    output: &mut Picture<T>,
) {
    let views = PlaneIndex::ALL.map(|plane| input.plane(plane).view());
    let [y, u, v] = output.planes_mut();
    k.av1_add_film_grain(
        params,
        [&views[0], &views[1], &views[2]],
        size,
        [&mut y.view_mut(), &mut u.view_mut(), &mut v.view_mut()],
    );
}

#[test]
fn ten_bit_av1_outputs_stay_inside_the_depth() {
    let dsp = Dsp::new();
    let ten = dsp.at(BitDepth::TEN);
    let size = (64, 48);
    let p = picture::<u16>(size, 2, 1023);
    let edges = BlockEdges::read(
        p.plane(PlaneIndex::Y),
        (16, 16),
        (16, 16),
        EdgeAvail {
            have_above: true,
            have_left: true,
            have_above_right: true,
            have_below_left: true,
        },
        BitDepth::TEN,
    );
    for mode in FilterIntraMode::ALL {
        let mut out = vec![0u16; 256];
        ten.av1_intra_predict_filter_block(
            &edges,
            mode,
            &mut PlaneViewMut::new(&mut out, 16, 16, 16).unwrap(),
        );
        assert!(out.iter().all(|&v| v <= 1023), "{mode:?}");
    }
    let tx = Av1TxSize {
        width_log2: 3,
        height_log2: 3,
    };
    let mut coeffs = vec![(1 << 17) - 1; 64];
    let mut out = vec![1000u16; 64];
    ten.av1_inverse_add::<u16>(
        tx,
        Av1TxType::Idtx,
        &mut coeffs,
        64,
        &mut PlaneViewMut::new(&mut out, 8, 8, 8).unwrap(),
    );
    assert!(out.iter().all(|&v| v <= 1023));
}
