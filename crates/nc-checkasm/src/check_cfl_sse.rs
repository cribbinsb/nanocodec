//! Bit-exactness of the `CfL` prediction SSE (`Dsp::cfl_sse`, the AV1
//! encoder's slope search) against scalar: every side, every slope, random
//! and extreme sources, AC values and DC (the full `MAX_AC` range, so the
//! 16-bit product and the clamps are exercised), at random strides.

use nc_dsp::Dsp;
use nc_dsp::cfl_sse::{MAX_AC, MAX_ALPHA, SIDES};

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
        for side in SIDES {
            let stride = side + rng.below(MAX_STRIDE_PAD + 1);
            let extreme = rng.below(2) == 0;
            let mut src = vec![0u8; (side - 1) * stride + side];
            if extreme {
                for s in &mut src {
                    *s = if rng.below(2) == 0 { 0 } else { 255 };
                }
            } else {
                rng.fill(&mut src);
            }
            let span = 2 * usize::try_from(MAX_AC).unwrap() + 1;
            let ac: Vec<i16> = (0..side * side)
                .map(|_| {
                    if extreme {
                        [MAX_AC, -MAX_AC][rng.below(2)]
                    } else {
                        (rng.below(span) as i32 - i32::from(MAX_AC)) as i16
                    }
                })
                .collect();
            let dc = rng.byte();
            let view = nc_common::PlaneView::new(&src, stride, side, side).unwrap();
            for alpha in -MAX_ALPHA..=MAX_ALPHA {
                let got = candidate.cfl_sse(&view, &ac, dc, alpha);
                let want = reference.cfl_sse(&view, &ac, dc, alpha);
                record(report, candidate.isa(), got == want, || {
                    format!(
                        "cfl_sse side {side} stride {stride} dc {dc} alpha {alpha}: {got} != {want}"
                    )
                });
            }
        }
    }
}
