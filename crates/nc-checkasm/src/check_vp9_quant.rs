//! VP9 quantization against the scalar reference: random steps, deadzones and
//! shifts, block lengths with and without a vector tail, and random, extreme,
//! zero and threshold coefficients.
use crate::{check::CheckReport, rng::Rng};
use nc_dsp::{Dsp, Vp9QuantStep};

/// The 16-bit coefficients' extremes.
const MAX_MAGNITUDE: i16 = i16::MAX;
const MIN_COEFFICIENT: i16 = i16::MIN;
/// VP9's quantizer step range.
const MIN_STEP: usize = 4;
const MAX_STEP: usize = 1828;
/// Deadzone divisors the VP9 encoder uses (intra, inter, nearest).
const DIVISORS: [i32; 3] = [3, 6, 2];
/// Random parameter sets per block length.
const RANDOM_SETS: usize = 24;

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    check_largest_magnitude(candidate, reference, rng, report);
    for len in [1, 7, 16, 17, 64, 256, 1024] {
        for _ in 0..RANDOM_SETS {
            let step = |rng: &mut Rng| (MIN_STEP + rng.below(MAX_STEP - MIN_STEP + 1)) as i32;
            let divisor = DIVISORS[rng.below(DIVISORS.len())];
            let shift = rng.below(2) as u32;
            let (dc_step, ac_step) = (step(rng), step(rng));
            let dc = Vp9QuantStep::new(dc_step, divisor, shift);
            let ac = Vp9QuantStep::new(ac_step, divisor, shift);
            for pattern in 0..5 {
                let coefficients: Vec<i16> = (0..len)
                    .map(|i| match pattern {
                        0 => rng.next_u64() as i16,
                        1 => {
                            if i % 2 == 0 {
                                MAX_MAGNITUDE
                            } else {
                                MIN_COEFFICIENT
                            }
                        }
                        2 => 0,
                        // Around the first nonzero threshold of either step.
                        3 => {
                            let s = if i == 0 { dc_step } else { ac_step };
                            let t = (s - s / divisor) >> shift;
                            (t + rng.below(5) as i32 - 2) as i16
                        }
                        _ => rng.below(64) as i16 - 32,
                    })
                    .collect();
                let run = |dsp: &Dsp| {
                    let mut levels = vec![0x5A5A; len];
                    let nonzero = dsp.vp9_quantize(&coefficients, &dc, &ac, &mut levels);
                    (nonzero, levels)
                };
                report.cases += 1;
                if run(candidate) != run(reference) {
                    report.failures.push(format!(
                        "{} vp9_quantize len={len} steps={dc_step},{ac_step} divisor={divisor} shift={shift} pattern={pattern}",
                        candidate.isa()
                    ));
                }
            }
        }
    }
}

/// `vp9_largest_magnitude` on random, single-spike and empty blocks of
/// lengths with and without a vector tail.
fn check_largest_magnitude(
    candidate: &Dsp,
    reference: &Dsp,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    for len in [0, 1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 33, 64, 256, 1024] {
        for pattern in 0..4 {
            let mut coefficients: Vec<i16> = (0..len)
                .map(|_| match pattern {
                    0 => rng.next_u64() as i16,
                    1 => rng.below(7) as i16 - 3,
                    _ => 0,
                })
                .collect();
            if pattern == 3 && len > 0 {
                // One spike anywhere, including the scalar tail.
                coefficients[rng.below(len)] = MIN_COEFFICIENT;
            }
            report.cases += 1;
            if candidate.vp9_largest_magnitude(&coefficients)
                != reference.vp9_largest_magnitude(&coefficients)
            {
                report.failures.push(format!(
                    "{} vp9_largest_magnitude len={len} pattern={pattern}",
                    candidate.isa()
                ));
            }
        }
    }
}
