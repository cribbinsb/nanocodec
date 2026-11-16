//! Narrow statistics on exact-size, independently strided and aligned inputs.
//! Signed extrema, nonconstant rows and the four-row ICL tail exercise packed
//! row assembly and separate residual/prediction reductions against scalar.

use nc_common::PlaneView;
use nc_dsp::{BlockSize, Dsp};

use crate::check::CheckReport;
use crate::check_codec::record;

const GUARD: u8 = 0xa5;
const HEIGHTS: [usize; 5] = [4, 8, 16, 32, 64];
const PATTERNS: usize = 6;

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, report: &mut CheckReport) {
    for w in [4, 8] {
        for h in HEIGHTS {
            let size = BlockSize::new(w, h).expect("fixed valid block sizes");
            for (ss, ps) in [(w, w), (w + 3, w + 5), (1280, 1293)] {
                for (so, po) in [(0, 0), (1, 7), (7, 1)] {
                    for pattern in 0..PATTERNS {
                        let slen = (h - 1) * ss + w;
                        let plen = (h - 1) * ps + w;
                        let mut source = vec![GUARD; so + slen];
                        let mut prediction = vec![GUARD; po + plen];
                        for y in 0..h {
                            for x in 0..w {
                                let (s, p) = pixels(x, y, h, pattern);
                                source[so + y * ss + x] = s;
                                prediction[po + y * ps + x] = p;
                            }
                        }
                        let src = PlaneView::new(&source[so..], ss, w, h).expect("source");
                        let pred = PlaneView::new(&prediction[po..], ps, w, h).expect("prediction");
                        let same = candidate.variance(size, &src, &pred)
                            == reference.variance(size, &src, &pred)
                            && candidate.variance_energy(size, &src, &pred)
                                == reference.variance_energy(size, &src, &pred)
                            && candidate.block_sums(size, &src) == reference.block_sums(size, &src)
                            && candidate.block_sums(size, &pred)
                                == reference.block_sums(size, &pred);
                        record(report, candidate.isa(), same, || {
                            format!(
                                "packed narrow statistics {w}x{h} pattern {pattern} strides {ss}/{ps} offsets {so}/{po}"
                            )
                        });
                    }
                }
            }
        }
    }
}

fn pixels(x: usize, y: usize, h: usize, pattern: usize) -> (u8, u8) {
    match pattern {
        0 => (255, 0),
        1 => (0, 255),
        2 => {
            if (x + y).is_multiple_of(2) {
                (255, 0)
            } else {
                (0, 255)
            }
        }
        3 => ((y * 41 + x * 17) as u8, (y * 61 + x * 73 + 5) as u8),
        4 => {
            if y == h - 1 {
                (255, 0)
            } else {
                (128, 128)
            }
        }
        _ => ((17 + 3 * y + x) as u8, (2 * y + 7 * x) as u8),
    }
}
