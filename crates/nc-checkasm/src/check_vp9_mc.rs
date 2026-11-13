//! All VP9 interpolation phases, exact windows, short rows and adversarial pixels.
use crate::{check::CheckReport, rng::Rng};
use nc_dsp::{Dsp, Vp9Filter};

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for (w, h) in [
        (1, 1),
        (4, 4),
        (4, 8),
        (8, 4),
        (8, 8),
        (12, 7),
        (16, 16),
        (32, 16),
        (64, 64),
        (7, 5),
    ] {
        for extra in [0, 13] {
            let stride = w + 7 + extra;
            let mut src = vec![0; 1 + (h + 6) * stride + w + 7];
            for pattern in 0..6 {
                fill_pattern(&mut src, stride, pattern, rng);
                for kind in [
                    Vp9Filter::Regular,
                    Vp9Filter::Smooth,
                    Vp9Filter::Sharp,
                    Vp9Filter::Bilinear,
                ] {
                    for fy in 0..16 {
                        for fx in 0..16 {
                            for dst_stride in [w, w + 5] {
                                let [got, want] = [candidate, reference].map(|dsp| {
                                    convolve(
                                        dsp,
                                        kind,
                                        (&src[1..], stride),
                                        (fx, fy),
                                        (w, h),
                                        dst_stride,
                                    )
                                });
                                report.cases += 1;
                                if got != want {
                                    report.failures.push(format!("{} vp9_convolve_into {kind:?} {w}x{h} phase={fx},{fy} stride={stride} dst_stride={dst_stride} pattern={pattern}", candidate.isa()));
                                }
                            }
                            let mut got = vec![0xA5; w * h + 2];
                            let mut want = got.clone();
                            candidate.vp9_convolve(
                                kind,
                                &src[1..],
                                stride,
                                (fx, fy),
                                (w, h),
                                &mut got[1..=w * h],
                            );
                            reference.vp9_convolve(
                                kind,
                                &src[1..],
                                stride,
                                (fx, fy),
                                (w, h),
                                &mut want[1..=w * h],
                            );
                            check_into(
                                candidate,
                                kind,
                                &src[1..],
                                stride,
                                (fx, fy),
                                (w, h),
                                &want[1..=w * h],
                                report,
                            );
                            report.cases += 1;
                            if got != want {
                                report.failures.push(format!("{} vp9_convolve {kind:?} {w}x{h} phase={fx},{fy} stride={stride} pattern={pattern}",candidate.isa()));
                            }
                        }
                    }
                }
            }
        }
    }
}

fn fill_pattern(src: &mut [u8], stride: usize, pattern: usize, rng: &mut Rng) {
    for (i, p) in src.iter_mut().enumerate() {
        *p = match pattern {
            0 => rng.byte(),
            1 => 0,
            2 => 255,
            3 => {
                if (i + i / stride).is_multiple_of(2) {
                    255
                } else {
                    0
                }
            }
            4 => {
                if i % stride < stride / 2 {
                    255
                } else {
                    0
                }
            }
            _ => {
                if (i / stride).is_multiple_of(2) {
                    255
                } else {
                    0
                }
            }
        };
    }
}

/// The packed scalar result also independently defines final byte averaging.
fn check_into(
    dsp: &Dsp,
    kind: Vp9Filter,
    src: &[u8],
    stride: usize,
    phase: (usize, usize),
    (w, h): (usize, usize),
    packed: &[u8],
    report: &mut CheckReport,
) {
    let dst_stride = w + 11;
    let len = (h - 1) * dst_stride + w;
    for average in [false, true] {
        let mut got: Vec<u8> = (0..len + 2).map(|i| (i * 53 + 7) as u8).collect();
        let mut want = got.clone();
        for y in 0..h {
            for x in 0..w {
                let out = &mut want[1 + y * dst_stride + x];
                let pixel = packed[y * w + x];
                *out = if average {
                    ((u16::from(*out) + u16::from(pixel) + 1) >> 1) as u8
                } else {
                    pixel
                };
            }
        }
        if average {
            dsp.vp9_convolve_avg_into(
                kind,
                src,
                stride,
                phase,
                (w, h),
                &mut got[1..=len],
                dst_stride,
            );
        } else {
            dsp.vp9_convolve_into(
                kind,
                src,
                stride,
                phase,
                (w, h),
                &mut got[1..=len],
                dst_stride,
            );
        }
        report.cases += 1;
        if got != want {
            report.failures.push(format!(
                "{} vp9_convolve_into {kind:?} {w}x{h} phase={phase:?} average={average}",
                dsp.isa()
            ));
        }
    }
}

/// One prediction into a destination of `dst_stride` whose gaps and
/// surrounding bytes start as a marker, so stray stores show.
fn convolve(
    dsp: &Dsp,
    kind: Vp9Filter,
    (src, stride): (&[u8], usize),
    phase: (usize, usize),
    (w, h): (usize, usize),
    dst_stride: usize,
) -> Vec<u8> {
    let mut dst = vec![0xA5; h * dst_stride + 2];
    let end = 1 + (h - 1) * dst_stride + w;
    dsp.vp9_convolve_into(
        kind,
        src,
        stride,
        phase,
        (w, h),
        &mut dst[1..end],
        dst_stride,
    );
    dst
}
