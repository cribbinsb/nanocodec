//! VVC 64-point transform blocks (`Dsp::vvc_forward64`, `Dsp::vvc_inverse_dct64`)
//! against the scalar reference: the forward's 2x2-mean residual kernel on
//! random, extreme and nearly flat blocks at a source stride wider than the
//! block, through both forward modes and zero limits; the inverse on dense,
//! extreme and low-frequency levels.
use crate::{check::CheckReport, rng::Rng};
use nc_dsp::{Dsp, HevcForwardMode, HevcForwardStaging};

/// The transform block's side, and its coded region's area (32x32).
const SIDE: usize = 64;
const CODED_AREA: usize = 32 * 32;
/// A source stride wider than the block, as a picture's.
const SOURCE_STRIDE: usize = SIDE + 16;
/// Random blocks per pattern.
const BLOCKS: usize = 8;

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    forward(candidate, reference, rng, report);
    inverse(candidate, reference, rng, report);
}

fn forward(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    let modes = [
        HevcForwardMode::EXACT,
        HevcForwardMode {
            staging: HevcForwardStaging::Rows,
            zero_columns: true,
            low_frequencies: false,
        },
    ];
    for pattern in 0..4 {
        for _ in 0..BLOCKS {
            let mut source = vec![0u8; SOURCE_STRIDE * SIDE];
            let mut prediction = vec![0u8; SIDE * SIDE];
            match pattern {
                0 => {
                    rng.fill(&mut source);
                    rng.fill(&mut prediction);
                }
                // Extremes: the largest residual sums of either sign.
                1 => {
                    for (i, s) in source.iter_mut().enumerate() {
                        *s = if i.is_multiple_of(3) { 255 } else { 0 };
                    }
                    for (i, p) in prediction.iter_mut().enumerate() {
                        *p = if i.is_multiple_of(3) { 0 } else { 255 };
                    }
                }
                // Nearly flat: rounding of the 2x2 means.
                _ => {
                    let base = rng.byte() / 2 + 64;
                    for s in source.iter_mut().chain(prediction.iter_mut()) {
                        *s = base + rng.byte() % 3;
                    }
                }
            }
            let source_view =
                nc_common::PlaneView::new(&source, SOURCE_STRIDE, SIDE, SIDE).unwrap();
            let prediction_view = nc_common::PlaneView::new(&prediction, SIDE, SIDE, SIDE).unwrap();
            for mode in modes {
                for limit in [0, rng.below(64) as u32, rng.below(4096) as u32] {
                    let run = |dsp: &Dsp| {
                        let mut out = vec![0x5A5A; CODED_AREA];
                        let coded = dsp.vvc_forward64(
                            &source_view,
                            &prediction_view,
                            limit,
                            mode,
                            &mut out,
                        );
                        (coded, if coded { out } else { Vec::new() })
                    };
                    report.cases += 1;
                    if run(candidate) != run(reference) {
                        report.failures.push(format!(
                            "{} vvc_forward64 pattern={pattern} mode={mode:?} limit={limit}",
                            candidate.isa()
                        ));
                    }
                }
            }
        }
    }
}

fn inverse(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for pattern in 0..3 {
        for qp in [4, 22, 37, 51, 63] {
            for dependent in [false, true] {
                let levels: Vec<i32> = (0..CODED_AREA)
                    .map(|i| match pattern {
                        0 => rng.below(2001) as i32 - 1000,
                        1 => {
                            if i % 2 == 0 {
                                32767
                            } else {
                                -32768
                            }
                        }
                        _ if i % 32 + i / 32 < 6 => rng.below(41) as i32 - 20,
                        _ => 0,
                    })
                    .collect();
                let run = |dsp: &Dsp| {
                    let mut out = vec![0x5A5A_i16; SIDE * SIDE];
                    dsp.vvc_inverse_dct64(&levels, qp, dependent, &mut out);
                    out
                };
                report.cases += 1;
                if run(candidate) != run(reference) {
                    report.failures.push(format!(
                        "{} vvc_inverse_dct64 pattern={pattern} qp={qp} dependent={dependent}",
                        candidate.isa()
                    ));
                }
            }
        }
    }
}
