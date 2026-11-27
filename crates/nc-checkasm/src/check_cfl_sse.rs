//! Bit-exactness and benchmarks of the `CfL` prediction SSE (`Dsp::cfl_sse`)
//! and fused slope search (`Dsp::cfl_best_alpha`) against scalar: every
//! square and 2:1 rectangular shape, every slope, random and extreme sources,
//! AC values and DC (the full `MAX_AC` range, exercising the 16-bit product
//! and clamps), at random strides.

use nc_dsp::Dsp;
use nc_dsp::cfl_sse::{MAX_AC, MAX_ALPHA, SHAPES};

use crate::BenchResult;
use crate::check::CheckReport;
use crate::check_codec::record;
use crate::rng::Rng;

/// Most extra samples past a row in the source buffer.
const MAX_STRIDE_PAD: usize = 24;

/// See the module docs.
pub(crate) fn check_cfl_sse(
    candidate: &Dsp,
    reference: &Dsp,
    iterations: usize,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    for _ in 0..iterations {
        for (w, h) in SHAPES {
            let stride = w + rng.below(MAX_STRIDE_PAD + 1);
            let extreme = rng.below(2) == 0;
            let mut src = vec![0u8; (h - 1) * stride + w];
            if extreme {
                for s in &mut src {
                    *s = if rng.below(2) == 0 { 0 } else { 255 };
                }
            } else {
                rng.fill(&mut src);
            }
            let span = 2 * usize::try_from(MAX_AC).unwrap() + 1;
            let ac: Vec<i16> = (0..w * h)
                .map(|_| {
                    if extreme {
                        [MAX_AC, -MAX_AC][rng.below(2)]
                    } else {
                        (rng.below(span) as i32 - i32::from(MAX_AC)) as i16
                    }
                })
                .collect();
            let dc = rng.byte();
            let view = nc_common::PlaneView::new(&src, stride, w, h).unwrap();
            for alpha in -MAX_ALPHA..=MAX_ALPHA {
                let got = candidate.cfl_sse(&view, &ac, dc, alpha);
                let want = reference.cfl_sse(&view, &ac, dc, alpha);
                record(report, candidate.isa(), got == want, || {
                    format!(
                        "cfl_sse {w}x{h} stride {stride} dc {dc} alpha {alpha}: {got} != {want}"
                    )
                });
            }
            let got_best = candidate.cfl_best_alpha(&view, &ac, dc);
            let want_best = reference.cfl_best_alpha(&view, &ac, dc);
            record(report, candidate.isa(), got_best == want_best, || {
                format!(
                    "cfl_best_alpha {w}x{h} stride {stride} dc {dc}: {got_best:?} != {want_best:?}"
                )
            });
        }
    }
}

pub(crate) fn bench(dsp: &Dsp, results: &mut Vec<BenchResult>) {
    let mut rng = Rng::new(0xCF15);
    let mut src = vec![0u8; 16 * 32];
    rng.fill(&mut src);
    let ac: Vec<i16> = (0..16 * 16)
        .map(|_| (rng.below(2049) as i32 - 1024) as i16)
        .collect();
    for (w, h) in SHAPES {
        let view = nc_common::PlaneView::new(&src, w + 4, w, h).unwrap();
        results.push(BenchResult {
            kernel: "cfl_best_alpha",
            size: (w, h),
            isa: dsp.isa(),
            ns_per_call: crate::bench::time(|| {
                let (dc_sse, best_sse, alpha) = dsp.cfl_best_alpha(&view, &ac[..w * h], 128);
                dc_sse ^ best_sse ^ u32::from(alpha.cast_unsigned())
            }),
        });
    }
}
