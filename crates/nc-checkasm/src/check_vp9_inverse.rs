//! VP9 inverse DCT/ADST checks, including full-range wrapping and final-round
//! boundaries, plus per-size/pass benchmarks against the unchanged scalar.

use crate::{BenchResult, CheckReport, rng::Rng};
use nc_dsp::{Dsp, Vp9Pass};

const PASSES: [(Vp9Pass, Vp9Pass); 4] = [
    (Vp9Pass::Dct, Vp9Pass::Dct),
    (Vp9Pass::Dct, Vp9Pass::Adst),
    (Vp9Pass::Adst, Vp9Pass::Dct),
    (Vp9Pass::Adst, Vp9Pass::Adst),
];
const GUARD: i32 = 0x1234_5678;
const MAX_SIDE: usize = 32;
const PATTERNS: usize = 12;

/// The fused inverse + add against the scalar oracle: coefficients limited
/// to a `(rows, columns)` extent (as the decoder passes it, including the
/// DC-only and zero-corner classes), random predictions in a strided
/// picture, residuals that clip both ways.
pub(crate) fn check_add(
    candidate: &Dsp,
    reference: &Dsp,
    iterations: usize,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    for log2 in 3..=5_u8 {
        let side = 1_usize << log2;
        let stride = side + 3;
        for &(columns, rows) in &PASSES[..if log2 == 5 { 1 } else { 4 }] {
            for nonzero_rows in [1, 2, 4, 5, 8, 9, side / 2 + 1, side] {
                for nonzero_columns in [1, 3, 4, 8, 12, 16, side] {
                    let extent = (nonzero_rows.min(side), nonzero_columns.min(side));
                    for pattern in 0..3 {
                        for _ in 0..iterations.max(1) {
                            let input = sparse_block(side, extent, pattern, rng);
                            let mut expected: Vec<u8> =
                                (0..=side * stride).map(|_| rng.next_u64() as u8).collect();
                            let mut actual = expected.clone();
                            reference.vp9_inverse_add(
                                log2,
                                (columns, rows),
                                &input,
                                (side, side),
                                (&mut expected[1..], stride),
                            );
                            candidate.vp9_inverse_add(
                                log2,
                                (columns, rows),
                                &input,
                                extent,
                                (&mut actual[1..], stride),
                            );
                            report.cases += 1;
                            if actual != expected {
                                report.failures.push(format!(
                                    "{} vp9_inverse_add {side}x{side} {columns:?}/{rows:?} extent={extent:?} pattern={pattern}",
                                    candidate.isa()
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
}

/// A raster block whose coefficients outside `(rows, columns)` are zero:
/// moderate levels, full-range i16 words, or sparse values.
fn sparse_block(
    side: usize,
    (rows, columns): (usize, usize),
    pattern: usize,
    rng: &mut Rng,
) -> Vec<i32> {
    let mut input = vec![0; side * side];
    for row in input.chunks_exact_mut(side).take(rows) {
        for value in &mut row[..columns] {
            *value = match pattern {
                0 => (rng.below(512) as i32 - 256) * 8,
                1 => i32::from(rng.next_u64() as i16),
                _ => {
                    if rng.below(8) == 0 {
                        rng.below(4096) as i32 - 2048
                    } else {
                        0
                    }
                }
            };
        }
    }
    input
}

pub(crate) fn check(
    candidate: &Dsp,
    reference: &Dsp,
    iterations: usize,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    for log2 in 3..=5 {
        let samples = 1 << (2 * log2);
        for &(columns, rows) in &PASSES[..if log2 == 5 { 1 } else { 4 }] {
            for pattern in 0..PATTERNS {
                for _ in 0..iterations.max(1) {
                    let mut input = vec![0; samples + 1];
                    for (i, value) in input[1..].iter_mut().enumerate() {
                        *value = match pattern {
                            0 => rng.next_u64() as i32,
                            1 => i32::from(rng.next_u64() as i16),
                            2 => (rng.below(4096) as i32 - 2048) * 8,
                            3 => i32::MAX,
                            4 => i32::MIN,
                            5 => i32::from(i16::MAX),
                            6 => i32::from(i16::MIN),
                            7 => {
                                if i % 2 == 0 {
                                    32767
                                } else {
                                    -32768
                                }
                            }
                            8 => {
                                if i == 0 {
                                    32767
                                } else {
                                    0
                                }
                            }
                            9 => {
                                if i == samples - 1 {
                                    -32768
                                } else {
                                    0
                                }
                            }
                            10 => 65535,
                            _ => 0,
                        };
                    }
                    let mut expected = vec![GUARD; samples + 2];
                    let mut actual = expected.clone();
                    reference.vp9_inverse_transform(
                        log2,
                        columns,
                        rows,
                        &input[1..],
                        &mut expected[1..=samples],
                    );
                    candidate.vp9_inverse_transform(
                        log2,
                        columns,
                        rows,
                        &input[1..],
                        &mut actual[1..=samples],
                    );
                    report.cases += 1;
                    if actual != expected {
                        report.failures.push(format!(
                            "vp9_inverse log2={log2} {columns:?}/{rows:?} pattern={pattern}"
                        ));
                    }
                }
            }
        }
    }
}

pub(crate) fn bench(dsp: &Dsp, results: &mut Vec<BenchResult>) {
    let mut rng = Rng::new(0x5EED);
    let input: Vec<i32> = (0..MAX_SIDE * MAX_SIDE)
        .map(|_| (i32::from(rng.byte()) - 128) * 8)
        .collect();
    let mut output = vec![0; MAX_SIDE * MAX_SIDE];
    for log2 in 3..=5 {
        let side = 1 << log2;
        for (index, &(columns, rows)) in PASSES[..if log2 == 5 { 1 } else { 4 }].iter().enumerate()
        {
            let name = [
                "vp9_inverse_dd",
                "vp9_inverse_da",
                "vp9_inverse_ad",
                "vp9_inverse_aa",
            ][index];
            results.push(BenchResult {
                kernel: name,
                size: (side, side),
                isa: dsp.isa(),
                ns_per_call: crate::bench::time(|| {
                    dsp.vp9_inverse_transform(log2, columns, rows, &input, &mut output);
                    output[0] as u32
                }),
            });
        }
    }
}
