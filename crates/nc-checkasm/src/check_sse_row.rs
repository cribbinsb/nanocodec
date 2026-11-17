//! Exact final rows, unaligned independent strides and guarded SSE-cell outputs.

use nc_common::PlaneView;
use nc_dsp::Dsp;

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
            }
        }
    }
}
