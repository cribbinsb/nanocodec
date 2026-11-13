//! VP9 intra prediction against the scalar reference: every mode and side,
//! all edge-availability combinations, random, extreme and alternating
//! edges, guarded outputs.
use crate::{check::CheckReport, rng::Rng};
use nc_dsp::{Dsp, Vp9IntraEdges, Vp9IntraMode};

/// Guard byte around the output, to catch out-of-range writes.
const GUARD: u8 = 0xA5;
/// Random edge sets per (mode, side, availability).
const RANDOM_EDGES: usize = 8;

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for side in [4, 8, 16, 32] {
        for mode in Vp9IntraMode::ALL {
            for availability in 0..4 {
                for pattern in 0..RANDOM_EDGES + 3 {
                    let mut edges = Vp9IntraEdges {
                        above: [0; 65],
                        left: [0; 32],
                        has_above: availability & 1 != 0,
                        has_left: availability & 2 != 0,
                    };
                    let mut sample = |i: usize| match pattern {
                        p if p < RANDOM_EDGES => rng.byte(),
                        p if p == RANDOM_EDGES => u8::MAX,
                        p if p == RANDOM_EDGES + 1 => 0,
                        _ => {
                            if i.is_multiple_of(2) {
                                u8::MAX
                            } else {
                                0
                            }
                        }
                    };
                    for (i, s) in edges.above.iter_mut().enumerate() {
                        *s = sample(i);
                    }
                    for (i, s) in edges.left.iter_mut().enumerate() {
                        *s = sample(i + 1);
                    }
                    // VP9's above-right rule (the kernel contract): real
                    // samples only at 4x4, else the last above repeated.
                    if side > 4 {
                        let last = edges.above[side];
                        edges.above[side + 1..=2 * side].fill(last);
                    }
                    let run = |dsp: &Dsp| {
                        let mut out = vec![GUARD; side * side + 2];
                        dsp.vp9_intra_predict(mode, side, &edges, &mut out[1..=side * side]);
                        out
                    };
                    report.cases += 1;
                    if run(candidate) != run(reference) {
                        report.failures.push(format!(
                            "{} vp9_intra_predict {mode:?} {side}x{side} availability={availability} pattern={pattern}",
                            candidate.isa()
                        ));
                    }
                }
            }
        }
    }
}
