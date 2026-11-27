//! AV1 forward transforms of every size and type (`Dsp::av1_forward`)
//! against the scalar reference: random, small, flat-extreme and alternating
//! 8-bit residuals at a padded stride, plus per-size benchmarks of ADST,
//! FLIPADST, IDTX and rectangular DCT.

use crate::{BenchResult, CheckReport, rng::Rng};
use nc_dsp::{Av1TxSize, Av1TxType, Dsp};

/// Residual patterns per (size, type).
const PATTERNS: usize = 4;
/// Extra columns right of the residual block to exercise arbitrary strides.
const STRIDE_PAD: usize = 3;
/// Sentinel value placed before and after the output buffer to detect OOB writes.
const GUARD: i16 = 0x5A5A;

fn residual_sample(pattern: usize, index: usize, rng: &mut Rng) -> i16 {
    match pattern {
        0 => (rng.below(511) as i16) - 255,
        1 => (rng.below(65) as i16) - 32,
        2 => {
            if rng.below(2) == 0 {
                255
            } else {
                -255
            }
        }
        _ => {
            if index.is_multiple_of(2) {
                255
            } else {
                -255
            }
        }
    }
}

pub(crate) fn check(
    candidate: &Dsp,
    reference: &Dsp,
    iterations: usize,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let iters = iterations.max(1);
    for size in Av1TxSize::ALL {
        let (w, h) = (size.width(), size.height());
        let stride = w + STRIDE_PAD;
        let coded = size.coded_area();
        for kind in Av1TxType::ALL.into_iter().filter(|kind| kind.fits(size)) {
            for pattern in 0..PATTERNS {
                for _ in 0..iters {
                    let residual: Vec<i16> = (0..stride * h)
                        .map(|i| residual_sample(pattern, i, rng))
                        .collect();
                    let mut expected = vec![GUARD; coded + 2];
                    let mut actual = vec![GUARD; coded + 2];
                    reference.av1_forward(size, kind, &residual, stride, &mut expected[1..=coded]);
                    candidate.av1_forward(size, kind, &residual, stride, &mut actual[1..=coded]);
                    report.cases += 1;
                    if actual != expected {
                        report
                            .failures
                            .push(format!("av1_forward {w}x{h} {kind:?} pattern={pattern}"));
                    }
                }
            }
        }
    }
}

/// Benchmarked kinds beyond square `DCT_DCT` (covered by `av1_forward_dct`).
const BENCH_KINDS: [(Av1TxType, &str); 4] = [
    (Av1TxType::AdstAdst, "av1_forward_adst_adst"),
    (Av1TxType::FlipadstAdst, "av1_forward_flipadst_adst"),
    (Av1TxType::Idtx, "av1_forward_idtx"),
    (Av1TxType::DctDct, "av1_forward_rect_dct"),
];

pub(crate) fn bench(dsp: &Dsp, results: &mut Vec<BenchResult>) {
    let mut rng = Rng::new(0x1F0D);
    let residual: Vec<i16> = (0..64 * 64)
        .map(|_| (rng.below(511) as i16) - 255)
        .collect();
    let mut out = vec![0i16; 32 * 32];
    for size in Av1TxSize::ALL {
        let (w, h) = (size.width(), size.height());
        for (kind, name) in BENCH_KINDS {
            if !kind.fits(size) {
                continue;
            }
            if kind == Av1TxType::DctDct && w == h {
                continue;
            }
            results.push(BenchResult {
                kernel: name,
                size: (w, h),
                isa: dsp.isa(),
                ns_per_call: crate::bench::time(|| {
                    dsp.av1_forward(size, kind, &residual, w, &mut out);
                    u32::from(out[0].cast_unsigned())
                }),
            });
        }
    }
}
