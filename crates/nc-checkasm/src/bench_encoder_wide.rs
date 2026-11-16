//! Focused forward-transform and intra timings for encoder ISA tuning.
//! Uses the full suite's time function and scalar/candidate dispatch tables.
use crate::bench::{BenchResult, time};
use nc_dsp::{BlockSize, DctSize, Dsp, HevcForwardMode, HevcForwardStaging, HevcTransform, Isa};

pub(crate) fn bench(dsp: &Dsp, isa: Isa, source: &[u8], out: &mut Vec<BenchResult>) {
    let mut push = |kernel, size: BlockSize, ns_per_call| {
        out.push(BenchResult {
            kernel,
            size: (size.width(), size.height()),
            isa,
            ns_per_call,
        });
    };
    for (side, dct, hevc) in [
        (16, DctSize::N16, HevcTransform::Dct16),
        (32, DctSize::N32, HevcTransform::Dct32),
    ] {
        let size = BlockSize::new(side, side).expect("transform side");
        let residual: Vec<i16> = source[..side * side]
            .iter()
            .map(|&v| i16::from(v) - 128)
            .collect();
        let mut coefficients = vec![0i16; side * side];
        push(
            "av1_forward_dct",
            size,
            time(|| {
                dsp.av1_forward_dct(dct, &residual, side, &mut coefficients);
                coefficients[0] as u32
            }),
        );
        push(
            "av1_forward_dct_raster",
            size,
            time(|| {
                dsp.av1_forward_dct_raster(dct, &residual, side, &mut coefficients);
                coefficients[1] as u32
            }),
        );
        push(
            "av1_forward_bounded",
            size,
            time(|| {
                u32::from(dsp.av1_forward_dct_raster_unless_zero(
                    dct,
                    &residual,
                    side,
                    0,
                    &mut coefficients,
                ))
            }),
        );
        push(
            "av1_forward_low",
            size,
            time(|| {
                dsp.av1_forward_dct_low(side, &residual, side, &mut coefficients);
                coefficients[0] as u32
            }),
        );
        push(
            "av1_forward_raster_low",
            size,
            time(|| {
                dsp.av1_forward_dct_raster_low(side, &residual, side, &mut coefficients);
                coefficients[1] as u32
            }),
        );
        push(
            "hevc_forward_dct",
            size,
            time(|| {
                dsp.hevc_forward(hevc, &residual, side, &mut coefficients);
                coefficients[1] as u32
            }),
        );
        for (name, staging) in [
            ("hevc_forward_staged_rows", HevcForwardStaging::Rows),
            (
                "hevc_forward_staged_both",
                HevcForwardStaging::RowsAndColumns,
            ),
        ] {
            push(
                name,
                size,
                time(|| {
                    u32::from(dsp.hevc_forward_staged_unless_zero(
                        hevc,
                        &residual,
                        side,
                        0,
                        HevcForwardMode::staged(staging),
                        &mut coefficients,
                    ))
                }),
            );
        }
    }
    crate::bench_av1_intra::bench_av1_intra(dsp, source, &mut push);
}
