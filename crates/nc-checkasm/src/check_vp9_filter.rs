//! VP9 edge-filter differential cases, including partial lines and mixed masks.

use crate::{check::CheckReport, check_codec::record, rng::Rng};
use nc_dsp::{Dsp, Vp9FilterLimits, Vp9FilterWidth};

const STRIDE: usize = 67;
const SIDE: usize = 64;
const ORIGIN: usize = 16 * STRIDE + 16;

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    exact_footprints(candidate, reference, report);
    lazy_outer_taps(candidate, reference, report);
    for pattern in 0..14 {
        let mut input = vec![0; STRIDE * SIDE + 2];
        for (i, pixel) in input.iter_mut().enumerate() {
            let row = i / STRIDE;
            let col = i % STRIDE;
            *pixel = match pattern {
                0 => rng.byte(),
                1 => 0,
                2 => 255,
                3 => {
                    if (row + col).is_multiple_of(2) {
                        0
                    } else {
                        255
                    }
                }
                4 => 120 + ((row + col) % 3) as u8,
                5 => 120 + u8::from(col >= 16) * 4,
                6 => 120 + u8::from(row >= 16) * 4,
                7 => 120 + ((row + col) % 2) as u8,
                8 => {
                    if (row / 4 + col / 4).is_multiple_of(2) {
                        rng.byte()
                    } else {
                        128
                    }
                }
                9 => ((row * 11 + col * 13) % 256) as u8,
                10 => 120 + u8::from(!(12..20).contains(&col)) * 8,
                11 => 120 + u8::from(!(12..20).contains(&row)) * 8,
                12 => ((row + col) % 7) as u8,
                _ => 255 - ((row + col) % 7) as u8,
            };
        }
        for width in [
            Vp9FilterWidth::Four,
            Vp9FilterWidth::Eight,
            Vp9FilterWidth::Sixteen,
        ] {
            for (step, along) in [(STRIDE, 1), (1, STRIDE), (1, 1), (2, 3)] {
                for lines in [0, 1, 4, 7, 8, 9, 15, 16, 17, 24, 31, 32] {
                    for (level, sharpness) in [0, 1, 15, 16, 31, 32, 47, 63, 83, 84, 255]
                        .into_iter()
                        .flat_map(|level| [0, 3, 7].map(move |sharpness| (level, sharpness)))
                    {
                        let limits = Vp9FilterLimits::with_sharpness(level, sharpness);
                        let mut expected = input.clone();
                        let mut actual = input.clone();
                        reference.vp9_filter_edge(
                            &mut expected[1..input.len() - 1],
                            ORIGIN,
                            step,
                            along,
                            lines,
                            width,
                            limits,
                        );
                        candidate.vp9_filter_edge(
                            &mut actual[1..input.len() - 1],
                            ORIGIN,
                            step,
                            along,
                            lines,
                            width,
                            limits,
                        );
                        record(report, candidate.isa(), expected == actual, || {
                            format!(
                                "VP9 filter {width:?} pattern={pattern} step={step} along={along} lines={lines} level={level} sharpness={sharpness}"
                            )
                        });
                    }
                }
            }
        }
    }
}

fn exact_footprints(candidate: &Dsp, reference: &Dsp, report: &mut CheckReport) {
    for width in [
        Vp9FilterWidth::Four,
        Vp9FilterWidth::Eight,
        Vp9FilterWidth::Sixteen,
    ] {
        let reach = if width == Vp9FilterWidth::Sixteen {
            8
        } else {
            4
        };
        for (step, along) in [(31, 1), (1, 31)] {
            for lines in [8, 16, 24, 32] {
                if step != 1 && lines > step {
                    continue;
                }
                let edge = reach * step;
                let last = edge + (lines - 1) * along + (reach - 1) * step;
                let input: Vec<u8> = (0..last + 2).map(|i| 120 + (i % 2) as u8).collect();
                let mut expected = input.clone();
                let mut actual = input;
                let limits = Vp9FilterLimits::new(63);
                reference.vp9_filter_edge(
                    &mut expected[1..],
                    edge,
                    step,
                    along,
                    lines,
                    width,
                    limits,
                );
                candidate.vp9_filter_edge(
                    &mut actual[1..],
                    edge,
                    step,
                    along,
                    lines,
                    width,
                    limits,
                );
                record(report, candidate.isa(), expected == actual, || {
                    format!("VP9 exact footprint {width:?} step={step} lines={lines}")
                });
            }
        }
    }
}

fn lazy_outer_taps(candidate: &Dsp, reference: &Dsp, report: &mut CheckReport) {
    // Only p3..q3 exists, but every line rejects filtering before p7..q7.
    const STRIDE: usize = 8;
    let input: Vec<u8> = (0..STRIDE * 8)
        .map(|i| if i.is_multiple_of(2) { 0 } else { 255 })
        .collect();
    let mut expected = input.clone();
    let mut actual = input;
    let limits = Vp9FilterLimits::new(1);
    reference.vp9_filter_edge(
        &mut expected,
        4,
        1,
        STRIDE,
        8,
        Vp9FilterWidth::Sixteen,
        limits,
    );
    candidate.vp9_filter_edge(
        &mut actual,
        4,
        1,
        STRIDE,
        8,
        Vp9FilterWidth::Sixteen,
        limits,
    );
    record(report, candidate.isa(), expected == actual, || {
        "VP9 rejected outer taps unavailable".into()
    });
}
