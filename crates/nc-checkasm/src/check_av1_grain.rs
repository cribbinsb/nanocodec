//! AV1 grain row exactness: clipped extremes, signed rounding, tails and guards.
use crate::{check::CheckReport, rng::Rng};
use nc_dsp::Dsp;
use nc_dsp::av1_film_grain::Av1GrainScaling;

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for len in [0usize, 1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65] {
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
            let grain: Vec<i16> = (0..=len)
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
                    _ => i16::from(rng.byte()) - 128,
                })
                .collect();
            let scaling = core::array::from_fn(|i| match pattern {
                1 => 0,
                2 => 255,
                3 => i as i32,
                _ => i32::from(rng.byte()),
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
    grain: &[i16],
    len: usize,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    for weights in [[27, 17], [17, 27], [23, 22]] {
        let mut expected: Vec<i16> = (0..len + 4)
            .map(|i| {
                if i == 0 || i > len {
                    1234
                } else {
                    i16::from(rng.byte()) - 128
                }
            })
            .collect();
        let mut actual = expected.clone();
        reference.av1_grain_blend_row(&grain[1..], &mut expected[1..=len], weights);
        candidate.av1_grain_blend_row(&grain[1..], &mut actual[1..=len], weights);
        compare(&actual, &expected, "overlap", report);
    }
}
