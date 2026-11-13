//! VVC rectangular inverse scaling/DCT checks through the shared HEVC kernels.
use crate::{check::CheckReport, rng::Rng};
use nc_dsp::Dsp;

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    const GUARD: i32 = 0x1234_5678;
    for width in [2, 4, 8, 16, 32] {
        for height in [2, 4, 8, 16, 32] {
            for qp in 0..=63 {
                for dependent in [false, true] {
                    let count = width * height;
                    let levels: Vec<_> = (0..count)
                        .map(|i| match i % 5 {
                            0 => -32768,
                            1 => 32767,
                            2 => 0,
                            _ => rng.below(65536) as i32 - 32768,
                        })
                        .collect();
                    let mut expected = vec![GUARD; count + 4];
                    let mut actual = expected.clone();
                    reference.vvc_inverse_dct(
                        width,
                        height,
                        &levels,
                        qp,
                        dependent,
                        &mut expected[..count],
                    );
                    candidate.vvc_inverse_dct(
                        width,
                        height,
                        &levels,
                        qp,
                        dependent,
                        &mut actual[..count],
                    );
                    if !dependent {
                        for bdpcm in 0..=2 {
                            let mut skip_expected = vec![GUARD; count + 4];
                            let mut skip_actual = skip_expected.clone();
                            reference.vvc_inverse_skip(
                                (width, height),
                                &levels,
                                qp,
                                bdpcm,
                                &mut skip_expected[..count],
                            );
                            candidate.vvc_inverse_skip(
                                (width, height),
                                &levels,
                                qp,
                                bdpcm,
                                &mut skip_actual[..count],
                            );
                            report.cases += 1;
                            if skip_actual != skip_expected
                                || skip_actual[count..].iter().any(|&v| v != GUARD)
                            {
                                report.failures.push(format!(
                                    "{} vvc_inverse_skip {width}x{height} qp={qp} bdpcm={bdpcm}",
                                    candidate.isa()
                                ));
                            }
                        }
                    }
                    report.cases += 1;
                    if actual != expected || actual[count..].iter().any(|&v| v != GUARD) {
                        report.failures.push(format!(
                            "{} vvc_inverse_dct {width}x{height} qp={qp} dependent={dependent}",
                            candidate.isa()
                        ));
                    }
                }
            }
        }
    }
}
