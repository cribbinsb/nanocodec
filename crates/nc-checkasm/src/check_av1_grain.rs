//! AV1 grain row exactness: clipped extremes, signed rounding, tails and guards.
use crate::{check::CheckReport, rng::Rng};
use nc_common::{PlaneView, PlaneViewMut};
use nc_dsp::Dsp;
use nc_dsp::av1_film_grain::{Av1FilmGrainParams, Av1GrainScaling};

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for len in [
        0usize, 1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
    ] {
        for pattern in 0..5 {
            let source: Vec<u8> = (0..=len)
                .map(|i| match pattern {
                    1 => 0,
                    2 => 255,
                    3 => (i % 2 * 255) as u8,
                    _ => rng.byte(),
                })
                .collect();
            let luma: Vec<u8> = (0..=2 * len).map(|_| rng.byte()).collect();
            let grain: Vec<i8> = (0..=len)
                .map(|i| match pattern {
                    1 => -128,
                    2 => 127,
                    3 => {
                        if i % 2 == 0 {
                            -128
                        } else {
                            127
                        }
                    }
                    _ => rng.byte().wrapping_sub(128) as i8,
                })
                .collect();
            let scaling = core::array::from_fn(|i| match pattern {
                1 => 0,
                2 => 255,
                3 => i as u8,
                _ => rng.byte(),
            });
            for shift in 8..=11 {
                let scaling = Av1GrainScaling::new(&scaling, shift);
                for bounds in [[0, 255], [16, 235], [16, 240]] {
                    let mut expected = vec![0xa5; len + 4];
                    let mut actual = expected.clone();
                    reference.av1_grain_luma_row(
                        &source[1..],
                        &grain[1..],
                        &scaling,
                        bounds,
                        &mut expected[1..=len],
                    );
                    candidate.av1_grain_luma_row(
                        &source[1..],
                        &grain[1..],
                        &scaling,
                        bounds,
                        &mut actual[1..=len],
                    );
                    compare(&actual, &expected, "luma", report);
                    for coefficients in [
                        [0, 64, 0],
                        [-128, -128, -256],
                        [127, 127, 255],
                        [127, -128, 0],
                        [-128, 127, 0],
                        [1, 1, -1],
                    ] {
                        for odd in [false, true] {
                            let count = 2 * len - usize::from(odd && len > 0);
                            reference.av1_grain_chroma_row(
                                &source[1..],
                                &luma[1..=count],
                                &grain[1..],
                                &scaling,
                                bounds,
                                coefficients,
                                &mut expected[1..=len],
                            );
                            candidate.av1_grain_chroma_row(
                                &source[1..],
                                &luma[1..=count],
                                &grain[1..],
                                &scaling,
                                bounds,
                                coefficients,
                                &mut actual[1..=len],
                            );
                            compare(&actual, &expected, "chroma", report);
                        }
                    }
                }
            }
            check_blend(candidate, reference, &grain, len, rng, report);
        }
    }
    check_luma_planes(candidate, reference, rng, report);
}

fn check_luma_planes(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    // Multiple stripes, partial bottom stripes, and a right-edge fallback.
    for (width, height) in [(32usize, 2usize), (96, 34), (65, 35)] {
        let y_stride = width + 19;
        let y: Vec<_> = (0..y_stride * height).map(|_| rng.byte()).collect();
        let y = PlaneView::new(&y, y_stride, width, height).expect("valid luma");
        let c_width = width.div_ceil(2);
        let c_height = height.div_ceil(2);
        let uv = vec![134; c_width * c_height];
        let uv = PlaneView::new(&uv, c_width, c_width, c_height).expect("valid chroma");
        for (codec, overlap) in [(0, false), (0, true), (1, true), (2, true)] {
            let mut params = Av1FilmGrainParams {
                grain_seed: 47641,
                overlap,
                clip_to_restricted_range: overlap,
                ..Default::default()
            };
            params.y_points.count = 2;
            params.y_points.value[1] = 255;
            params.y_points.scaling[..2].copy_from_slice(&[21, 52]);
            let apply = |dsp: &Dsp| {
                let mut pixels = vec![0xa5; y_stride * height + 1];
                let mut cb = vec![0xa5; c_width * c_height];
                let mut cr = cb.clone();
                let output = [
                    &mut PlaneViewMut::new(&mut pixels, y_stride, width, height).unwrap(),
                    &mut PlaneViewMut::new(&mut cb, c_width, c_width, c_height).unwrap(),
                    &mut PlaneViewMut::new(&mut cr, c_width, c_width, c_height).unwrap(),
                ];
                if codec == 0 {
                    dsp.av1_add_film_grain(&params, [&y, &uv, &uv], (width, height), output);
                } else {
                    dsp.av2_add_film_grain(
                        &params,
                        codec == 1,
                        [&y, &uv, &uv],
                        (width, height),
                        output,
                    );
                }
                pixels
            };
            compare(&apply(candidate), &apply(reference), "luma plane", report);
        }
    }
}
fn compare<T: PartialEq>(actual: &[T], expected: &[T], kind: &str, report: &mut CheckReport) {
    report.cases += 1;
    if actual != expected {
        report.failures.push(format!(
            "AV1 grain {kind} mismatch, guarded length {}",
            actual.len()
        ));
    }
}

fn check_blend(
    candidate: &Dsp,
    reference: &Dsp,
    grain: &[i8],
    len: usize,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    for weights in [[27, 17], [17, 27], [23, 22]] {
        let mut expected: Vec<i8> = (0..len + 4)
            .map(|i| {
                if i == 0 || i > len {
                    0x5a
                } else {
                    rng.byte().wrapping_sub(128) as i8
                }
            })
            .collect();
        let mut actual = expected.clone();
        reference.av1_grain_blend_row(&grain[1..], &mut expected[1..=len], weights);
        candidate.av1_grain_blend_row(&grain[1..], &mut actual[1..=len], weights);
        compare(&actual, &expected, "overlap", report);
    }
}
