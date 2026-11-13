//! VVC refinement exactness across phase/tail, search, flow and clipping branches.
use crate::{check::CheckReport, rng::Rng};
use nc_common::PlaneView;
use nc_dsp::Dsp;

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    bilinear(candidate, reference, rng, report);
    dmvr(candidate, reference, rng, report);
    optical(candidate, reference, rng, report);
}
fn compare<T: PartialEq>(actual: &T, expected: &T, kind: &str, report: &mut CheckReport) {
    report.cases += 1;
    if actual != expected {
        report
            .failures
            .push(format!("VVC refinement {kind} differs from scalar"));
    }
}
fn bilinear(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for (w, h) in [
        (0, 0),
        (0, 3),
        (3, 0),
        (1, 1),
        (4, 7),
        (7, 4),
        (8, 8),
        (12, 12),
        (16, 16),
        (20, 20),
        (21, 5),
    ] {
        let stride = w + 4;
        for pattern in 0..4 {
            let mut bytes = vec![0; stride * (h + 1) + 1];
            for (i, v) in bytes.iter_mut().enumerate() {
                *v = match pattern {
                    0 => 0,
                    1 => 255,
                    2 => {
                        if i % 2 == 0 {
                            0
                        } else {
                            255
                        }
                    }
                    _ => rng.byte(),
                };
            }
            let source = PlaneView::new(&bytes[1..], stride, w + 1, h + 1).unwrap();
            for fx in 0..16 {
                for fy in 0..16 {
                    let mut expected = vec![12345; w * h + 3];
                    let mut actual = expected.clone();
                    reference.vvc_dmvr_bilinear(
                        (w, h),
                        &source,
                        (fx, fy),
                        &mut expected[1..=w * h],
                    );
                    candidate.vvc_dmvr_bilinear((w, h), &source, (fx, fy), &mut actual[1..=w * h]);
                    compare(&actual, &expected, "bilinear phases/tail/guard", report);
                }
            }
        }
    }
}
fn dmvr(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for size in [(8, 8), (8, 16), (16, 8), (16, 16)] {
        let (w, h) = size;
        let stride = w + 4;
        let count = stride * (h + 4);
        for pattern in 0..48 {
            let first: Vec<i16> = (0..=count)
                .map(|_| match pattern {
                    0..=7 => 512,
                    8 => i16::MIN,
                    9 => i16::MAX,
                    10 => rng.next_u64() as i16,
                    _ => rng.below(1024) as i16,
                })
                .collect();
            let second: Vec<i16> = (0..=count)
                .map(|i| match pattern {
                    0..=7 => 512 + pattern as i16,
                    8 => i16::MAX,
                    9 => i16::MIN,
                    10 => rng.next_u64() as i16,
                    11 => first[i],
                    _ => rng.below(1024) as i16,
                })
                .collect();
            compare(
                &candidate.vvc_dmvr(size, [&first[1..], &second[1..]]),
                &reference.vvc_dmvr(size, [&first[1..], &second[1..]]),
                "search thresholds/full-range",
                report,
            );
        }
        // Shifted common texture forces every integer search displacement,
        // including edge optima where fractional refinement must be skipped.
        let texture: Vec<i16> = (0..48 * 48).map(|_| rng.below(1024) as i16).collect();
        for dy in -2..=2 {
            for dx in -2..=2 {
                let first: Vec<i16> = (0..count)
                    .map(|i| texture[(i / stride + 8) * 48 + i % stride + 8])
                    .collect();
                let second: Vec<i16> = (0..count)
                    .map(|i| {
                        texture[((i / stride + 8) as i32 + 2 * dy) as usize * 48
                            + ((i % stride + 8) as i32 + 2 * dx) as usize]
                    })
                    .collect();
                compare(
                    &candidate.vvc_dmvr(size, [&first, &second]),
                    &reference.vvc_dmvr(size, [&first, &second]),
                    "integer displacement/tie order",
                    report,
                );
            }
        }
    }
}
fn optical(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for w in [4, 8, 12, 16] {
        for h in [4, 8, 12, 16] {
            let count = (w + 2) * (h + 2);
            for pattern in 0..32 {
                let sample = |rng: &mut Rng, index: usize| match pattern {
                    0 => 0,
                    1 => i16::MAX,
                    2 => i16::MIN,
                    3 => {
                        if index.is_multiple_of(2) {
                            i16::MIN
                        } else {
                            i16::MAX
                        }
                    }
                    4 => ((index % 19) as i16 - 9) * 1024,
                    _ => rng.next_u64() as i16,
                };
                let first: Vec<i16> = (0..=count).map(|i| sample(rng, i)).collect();
                let second: Vec<i16> = (0..=count)
                    .map(|i| {
                        if pattern == 5 {
                            first[i]
                        } else {
                            sample(rng, i + 1)
                        }
                    })
                    .collect();
                let mut expected = vec![0xa5; w * h + 3];
                let mut actual = expected.clone();
                reference.vvc_bdof(
                    (w, h),
                    [&first[1..], &second[1..]],
                    &mut expected[1..=w * h],
                );
                candidate.vvc_bdof((w, h), [&first[1..], &second[1..]], &mut actual[1..=w * h]);
                compare(
                    &actual,
                    &expected,
                    "BDOF gradients/flow/clipping/guard",
                    report,
                );
            }
        }
    }
    for pattern in 0..128 {
        let samples = core::array::from_fn(|i| match pattern % 4 {
            0 => {
                if i % 2 == 0 {
                    i16::MIN
                } else {
                    i16::MAX
                }
            }
            1 => 0,
            _ => rng.next_u64() as i16,
        });
        let corrections = core::array::from_fn(|i| match pattern % 4 {
            0 => [-31, 31],
            1 => [0, 0],
            2 => [if i % 2 == 0 { -31 } else { 31 }; 2],
            _ => [rng.below(63) as i32 - 31, rng.below(63) as i32 - 31],
        });
        let mut expected = [0; 16];
        let mut actual = [0; 16];
        reference.vvc_prof_refine(&samples, &corrections, &mut expected);
        candidate.vvc_prof_refine(&samples, &corrections, &mut actual);
        compare(
            &actual,
            &expected,
            "PROF gradient/correction saturation",
            report,
        );
    }
}
