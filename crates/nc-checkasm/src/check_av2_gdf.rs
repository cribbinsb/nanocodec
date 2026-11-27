//! AV2 GDF corrections for every trained table and complete or partial stripes.

use nc_common::BitDepth;
use nc_dsp::Dsp;
use nc_dsp::av2_dec::gdf::{
    GDF_BORDER, GDF_ERROR_STRIDE, GDF_MAX_SIDE, GdfWindow, gdf_add, gdf_add_with_dsp, gdf_error,
    gdf_error_with_dsp,
};
use nc_dsp::av2_dec::gdf_tables::{INTER, INTRA};

use crate::{CheckReport, rng::Rng};

pub(crate) fn check(dsp: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for (width, height) in [(8, 8), (16, 8), (24, 16), (32, 56), (48, 24), (64, 64)] {
        let stride = width + 2 * GDF_BORDER + 3;
        for pattern in 0..4 {
            let samples: Vec<u16> = (0..(height + 2 * GDF_BORDER) * stride)
                .map(|i| {
                    let (x, y) = (i % stride, i / stride);
                    match pattern {
                        0 => rng.below(1024) as u16,
                        1 => (((x + y) & 1) * 1023) as u16,
                        2 => (512 + rng.below(21)) as u16,
                        _ => ((41 * x + 77 * y) & 1023) as u16,
                    }
                })
                .collect();
            let window = GdfWindow {
                samples: &samples,
                stride,
                width,
                height,
            };
            for (set, params) in INTRA.iter().chain(INTER.iter().flatten()).enumerate() {
                let mut expected = [73; GDF_ERROR_STRIDE * GDF_MAX_SIDE];
                let mut actual = expected;
                gdf_error(&window, params, &mut expected);
                gdf_error_with_dsp(dsp, &window, params, &mut actual);
                report.cases += 1;
                if let Some(sample) = actual.iter().zip(&expected).position(|(a, b)| a != b) {
                    let (x, y) = (sample % GDF_ERROR_STRIDE, sample / GDF_ERROR_STRIDE);
                    report.failures.push(format!(
                        "AV2 GDF mismatch: {width}x{height}, pattern {pattern}, table {set}, ({x},{y}): {} != {}",
                        actual[sample], expected[sample],
                    ));
                }
            }
        }
    }
    // Exercise every signed-byte correction, including exact rounding
    // ties and -128, whose magnitude is unsigned.
    for width in [8, 16, 24, 32, 40, 48, 56, 64] {
        let size = (width, 64);
        let stride = width + 9;
        let errors: Vec<_> = (0..GDF_ERROR_STRIDE * size.1)
            .map(|i| (37 * i + i / GDF_ERROR_STRIDE) as i8)
            .collect();
        for scale in 1..=4 {
            let mut expected: Vec<_> = (0..stride * size.1).map(|_| rng.byte()).collect();
            let mut actual = expected.clone();
            gdf_add(&mut expected, stride, &errors, size, scale, BitDepth::EIGHT);
            gdf_add_with_dsp(dsp, &mut actual, stride, &errors, size, scale);
            report.cases += 1;
            if actual != expected {
                report.failures.push(format!(
                    "AV2 GDF application mismatch: width {width}, scale {scale}"
                ));
            }
        }
    }
}
