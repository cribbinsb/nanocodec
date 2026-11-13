//! VP9 scaled prediction (`Dsp::vp9_convolve_scaled_into`) against the scalar
//! oracle `vp9_convolve_scaled`: every filter, the half-size step's both
//! phase classes, other steps, odd heights, averaging and strided output.
use crate::{check::CheckReport, rng::Rng};
use nc_dsp::{Dsp, VP9_TAPS, VP9_TAPS_BEFORE, Vp9Filter, Vp9ScaledScratch, vp9_convolve_scaled};

const KINDS: [Vp9Filter; 4] = [
    Vp9Filter::Regular,
    Vp9Filter::Smooth,
    Vp9Filter::Sharp,
    Vp9Filter::Bilinear,
];

pub(crate) fn check(candidate: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    let mut scratch = Vp9ScaledScratch::default();
    let sizes = [
        (4, 4),
        (4, 7),
        (8, 8),
        (16, 8),
        (8, 16),
        (32, 32),
        (64, 64),
        (64, 33),
    ];
    for (w, h) in sizes {
        for step in [
            (8, 8),
            (8, 12),
            (16, 8),
            (10, 10),
            (12, 16),
            (32, 32),
            (24, 5),
        ] {
            for kind in KINDS {
                for _ in 0..12 {
                    let phase = (rng.below(16), rng.below(16));
                    let average = rng.below(2) == 1;
                    one(
                        candidate,
                        kind,
                        (w, h),
                        step,
                        phase,
                        average,
                        rng,
                        &mut scratch,
                        report,
                    );
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments, reason = "one case's full description")]
fn one(
    dsp: &Dsp,
    kind: Vp9Filter,
    (w, h): (usize, usize),
    (x_step, y_step): (usize, usize),
    (x0, y0): (usize, usize),
    average: bool,
    rng: &mut Rng,
    scratch: &mut Vp9ScaledScratch,
    report: &mut CheckReport,
) {
    let columns = (((w - 1) * x_step + x0) >> 4) + VP9_TAPS;
    let rows = (((h - 1) * y_step + y0) >> 4) + VP9_TAPS;
    let stride = columns + rng.below(9);
    let extreme = rng.below(3) == 0;
    // The window with the three rows above and four below the footprint.
    let tall: Vec<u8> = (0..(rows + VP9_TAPS - 1) * stride)
        .map(|_| {
            if extreme {
                [0, 255][rng.below(2)]
            } else {
                rng.byte()
            }
        })
        .collect();
    let footprint = &tall[VP9_TAPS_BEFORE * stride..];
    let mut packed = vec![0; w * h];
    vp9_convolve_scaled(
        kind,
        footprint,
        stride,
        (x0, x_step),
        (y0, y_step),
        (w, h),
        &mut packed,
    );
    let dst_stride = w + rng.below(7);
    let before: Vec<u8> = (0..h * dst_stride).map(|_| rng.byte()).collect();
    let mut want = before.clone();
    for (row, line) in packed.chunks_exact(w).zip(want.chunks_mut(dst_stride)) {
        for (out, &p) in line.iter_mut().zip(row) {
            *out = if average {
                ((u16::from(*out) + u16::from(p) + 1) >> 1) as u8
            } else {
                p
            };
        }
    }
    let mut got = before;
    dsp.vp9_convolve_scaled_into(
        kind,
        (&tall, stride),
        (x0, x_step),
        (y0, y_step),
        (w, h),
        (&mut got, dst_stride),
        average,
        scratch,
    );
    report.cases += 1;
    if got != want {
        report.failures.push(format!(
            "{} vp9_convolve_scaled_into {kind:?} {w}x{h} step={x_step},{y_step} phase={x0},{y0} average={average}",
            dsp.isa()
        ));
    }
}
