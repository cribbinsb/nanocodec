//! AV1 masked compound blend exactness (`av1_blend_mask`): every width
//! from 1 to 128 (whole vectors and tails), every subsampling, compound
//! samples at the ends of their range and every weight.

use crate::{check::CheckReport, rng::Rng};
use nc_common::PlaneViewMut;
use nc_dsp::Dsp;
use nc_dsp::av1_mc::blend::{Av1CompoundPred, Av1Mask, av1_blend_mask};

/// Rows per case.
const ROWS: usize = 4;

fn compound(rng: &mut Rng, pattern: usize) -> i16 {
    match pattern {
        0 => i16::MIN,
        1 => i16::MAX,
        _ => (rng.below(1 << 16) as i32 - (1 << 15)) as i16,
    }
}

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for w in 1..=128 {
        for (sub_x, sub_y) in [(false, false), (true, false), (false, true), (true, true)] {
            let pattern = rng.below(4);
            let p0: Vec<i16> = (0..w * ROWS).map(|_| compound(rng, pattern)).collect();
            let p1: Vec<i16> = (0..w * ROWS)
                .map(|_| compound(rng, 3 - pattern.min(3)))
                .collect();
            let (mw, mh) = (w << u32::from(sub_x), ROWS << u32::from(sub_y));
            let weights: Vec<u8> = (0..mw * mh).map(|_| rng.below(65) as u8).collect();
            let blend = |dsp: &Dsp| {
                let mut out = vec![0u8; w * ROWS];
                av1_blend_mask(
                    dsp,
                    Av1CompoundPred {
                        data: &p0,
                        stride: w,
                    },
                    Av1CompoundPred {
                        data: &p1,
                        stride: w,
                    },
                    Av1Mask {
                        data: &weights,
                        stride: mw,
                        sub_x,
                        sub_y,
                    },
                    (w, ROWS),
                    &mut PlaneViewMut::new(&mut out, w, w, ROWS).unwrap(),
                );
                out
            };
            report.cases += 1;
            if blend(candidate) != blend(reference) {
                report.failures.push(format!(
                    "AV1 mask blend mismatch: width {w}, sub ({sub_x}, {sub_y})"
                ));
            }
        }
    }
}
