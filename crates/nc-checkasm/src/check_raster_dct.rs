//! The raster forward DCT (VP9's layout) at every DCT size against the
//! scalar column-major one, transposed by index, with guarded outputs.
use crate::{check::CheckReport, rng::Rng};
use nc_dsp::{DctSize, Dsp};

/// Guard value around the destination, to catch out-of-range writes.
const GUARD: i16 = 0x5A5A;
/// Random blocks per size.
const RANDOM_BLOCKS: usize = 16;
/// Largest residual magnitude of 8-bit samples (the forward's contract).
const MAX_RESIDUAL: i16 = 255;

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for size in DctSize::ALL {
        let n = size.side();
        let mut residuals: Vec<Vec<i16>> = (0..RANDOM_BLOCKS)
            .map(|_| {
                (0..n * n)
                    .map(|_| i16::from(rng.byte()) - i16::from(rng.byte()))
                    .collect()
            })
            .collect();
        residuals.push(vec![MAX_RESIDUAL; n * n]);
        residuals.push(vec![-MAX_RESIDUAL; n * n]);
        for residual in &residuals {
            let mut column_major = vec![0; n * n];
            reference.av1_forward_dct(size, residual, n, &mut column_major);
            let want: Vec<i16> = (0..n * n)
                .map(|i| column_major[(i % n) * n + i / n])
                .collect();
            let mut got = vec![GUARD; n * n + 2];
            candidate.av1_forward_dct_raster(size, residual, n, &mut got[1..=n * n]);
            report.cases += 1;
            if got[1..=n * n] != want[..] || got[0] != GUARD || got[n * n + 1] != GUARD {
                report.failures.push(format!(
                    "{} av1_forward_dct_raster {n}x{n}",
                    candidate.isa()
                ));
            }
        }
    }
    check_range_edges(candidate, reference, report);
}

/// Constant32-point residual125 has a row L1 of32000;126 crosses the
/// narrow guard and128 crosses signed-word output range. Include both
/// signs, padded strides and zero limits above u16 to exercise saturation
/// and the exact wide fallback, including an untouched skipped output.
fn check_range_edges(candidate: &Dsp, reference: &Dsp, report: &mut CheckReport) {
    const AMPLITUDES: [i16; 8] = [0, 1, 124, 125, 126, 127, 128, 255];
    const LIMITS: [u32; 9] = [
        0,
        1,
        31_999,
        32_000,
        32_007,
        32_767,
        65_535,
        65_536,
        u32::MAX,
    ];
    for size in [DctSize::N16, DctSize::N32] {
        let n = size.side();
        for stride in [n, n + 7] {
            for amplitude in AMPLITUDES {
                for sign in [-1, 1] {
                    let input = vec![amplitude * sign; n * stride + 1];
                    for limit in LIMITS {
                        let mut want = vec![GUARD; n * n + 2];
                        let mut got = want.clone();
                        let expected = reference.av1_forward_dct_raster_unless_zero(
                            size,
                            &input[1..],
                            stride,
                            limit,
                            &mut want[1..=n * n],
                        );
                        let coded = candidate.av1_forward_dct_raster_unless_zero(
                            size,
                            &input[1..],
                            stride,
                            limit,
                            &mut got[1..=n * n],
                        );
                        report.cases += 1;
                        if coded != expected || got != want {
                            report.failures.push(format!("{} forward range {n}x{n} residual={} stride={stride} limit={limit}",candidate.isa(),amplitude*sign));
                        }
                    }
                    if n == 32 {
                        for raster in [false, true] {
                            let run = |dsp: &Dsp| {
                                let mut out = vec![GUARD; n * n + 2];
                                if raster {
                                    dsp.av1_forward_dct_raster_low(
                                        n,
                                        &input[1..],
                                        stride,
                                        &mut out[1..=n * n],
                                    );
                                } else {
                                    dsp.av1_forward_dct_low(
                                        n,
                                        &input[1..],
                                        stride,
                                        &mut out[1..=n * n],
                                    );
                                }
                                out
                            };
                            report.cases += 1;
                            if run(candidate) != run(reference) {
                                report.failures.push(format!(
                                    "{} low32 range residual={} stride={stride} raster={raster}",
                                    candidate.isa(),
                                    amplitude * sign,
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
}
