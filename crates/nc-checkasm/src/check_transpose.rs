//! Coefficient transposes (column-major <-> raster) at every DCT size, against
//! the direct index definition and the scalar table, with guarded outputs;
//! and the raster forward DCT against the scalar column-major one.
use crate::{check::CheckReport, rng::Rng};
use nc_dsp::{DctSize, Dsp};

/// Guard value around the destination, to catch out-of-range writes.
const GUARD: i32 = 0x5A5A_5A5A;
/// Random blocks per size.
const RANDOM_BLOCKS: usize = 16;

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for size in DctSize::ALL {
        let n = size.side();
        let mut blocks: Vec<Vec<i32>> = (0..RANDOM_BLOCKS)
            .map(|_| (0..n * n).map(|_| rng.next_u64() as i32).collect())
            .collect();
        blocks.push((0..n * n).map(|i| i as i32).collect());
        blocks.push(vec![i32::MIN; n * n]);
        blocks.push(
            (0..n * n)
                .map(|i| if i % 2 == 0 { i32::MAX } else { i32::MIN })
                .collect(),
        );
        for src in &blocks {
            let want: Vec<i32> = (0..n * n).map(|i| src[(i % n) * n + i / n]).collect();
            for (dsp, label) in [(candidate, "candidate"), (reference, "reference")] {
                let mut got = vec![GUARD; n * n + 2];
                dsp.transpose_coefficients(size, src, &mut got[1..=n * n]);
                report.cases += 1;
                if got[1..=n * n] != want[..] || got[0] != GUARD || got[n * n + 1] != GUARD {
                    report.failures.push(format!(
                        "{} transpose_coefficients {n}x{n} ({label})",
                        candidate.isa()
                    ));
                }
            }
        }
        check_raster_dct(candidate, reference, size, rng, report);
    }
}

/// Residuals of 8-bit samples (the forward kernel's contract), plus extremes.
fn check_raster_dct(
    candidate: &Dsp,
    reference: &Dsp,
    size: DctSize,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    const MAX_RESIDUAL: i16 = 255;
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
        let want: Vec<i32> = (0..n * n)
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
