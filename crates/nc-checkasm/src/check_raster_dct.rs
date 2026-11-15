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
}
