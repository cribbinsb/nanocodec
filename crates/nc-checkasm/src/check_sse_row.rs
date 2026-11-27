//! Exact plane and cell SSE with independent strides and short tails.

use nc_common::PlaneView;
use nc_dsp::{Dsp, Variance};

use crate::check::CheckReport;
use crate::check_codec::record;

const GUARD: u32 = 0xa5a5_a5a5;

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, report: &mut CheckReport) {
    for cells in [0, 1, 2, 3, 4, 7, 8, 9, 31, 32, 159, 160] {
        let width = 8 * cells.max(1);
        for extra in [0, 5] {
            let (ss, ps) = (width + extra, width + 2 * extra);
            for pattern in 0..4 {
                let mut source = vec![0; 1 + 7 * ss + width];
                let mut prediction = vec![0; 3 + 7 * ps + width];
                for y in 0..8 {
                    for x in 0..width {
                        let (a, b) = match pattern {
                            0 => (255, 0),
                            1 => (0, 255),
                            2 => (((x + y) % 2 * 255) as u8, ((x + y + 1) % 2 * 255) as u8),
                            _ => ((x * 17 + y * 41) as u8, (x * 73 + y * 61) as u8),
                        };
                        source[1 + y * ss + x] = a;
                        prediction[3 + y * ps + x] = b;
                    }
                }
                let src = PlaneView::new(&source[1..], ss, width, 8).expect("exact source");
                let pred =
                    PlaneView::new(&prediction[3..], ps, width, 8).expect("exact prediction");
                let mut got = vec![GUARD; cells + 8];
                let mut want = got.clone();
                candidate.sse_8x8_row(&src, &pred, &mut got[3..3 + cells]);
                reference.sse_8x8_row(&src, &pred, &mut want[3..3 + cells]);
                record(report, candidate.isa(), got == want, || {
                    format!(
                        "SSE cell row n{cells} padding{extra} pattern{pattern}: guarded output differs"
                    )
                });
                let guard = Variance {
                    sse: GUARD,
                    sum: -123_456,
                };
                let mut got = vec![guard; cells + 8];
                let mut want = got.clone();
                candidate.variance_8x8_row(&src, &pred, &mut got[3..3 + cells]);
                reference.variance_8x8_row(&src, &pred, &mut want[3..3 + cells]);
                record(report, candidate.isa(), got == want, || {
                    format!(
                        "variance cell row n{cells} padding{extra} pattern{pattern}: guarded output differs"
                    )
                });
            }
        }
    }

    // The wide cases cover the SIMD accumulator's row limit and sums beyond
    // u32 across several rows of maximum differences.
    for (w, h) in [
        (1, 1),
        (13, 7),
        (63, 5),
        (641, 17),
        (32_769, 3),
        (66_052, 2),
    ] {
        let (ss, ps) = (w + 5, w + 13);
        for pattern in [false, true] {
            let mut src = vec![0; 1 + (h - 1) * ss + w];
            let mut pred = vec![255; 3 + (h - 1) * ps + w];
            if pattern {
                for (i, value) in src.iter_mut().enumerate() {
                    *value = (17 * i + i / ss * 41) as u8;
                }
                for (i, value) in pred.iter_mut().enumerate() {
                    *value = (73 * i + i / ps * 61) as u8;
                }
            }
            let src = PlaneView::new(&src[1..], ss, w, h).expect("exact source");
            let pred = PlaneView::new(&pred[3..], ps, w, h).expect("exact prediction");
            let got = candidate.sse(&src, &pred);
            let want = reference.sse(&src, &pred);
            record(report, candidate.isa(), got == want, || {
                format!("plane SSE {w}x{h} pattern {pattern}: got {got}, want {want}")
            });
        }
    }
}
