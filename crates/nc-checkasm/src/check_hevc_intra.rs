//! HEVC intra kernels against the scalar reference: angular lines at every
//! angle from −32 to 32 and planar, every side, random, extreme and
//! alternating references, guarded outputs.
use crate::{check::CheckReport, rng::Rng};
use nc_dsp::{Dsp, HevcPlanarEdges};

/// Guard byte around the output, to catch out-of-range writes.
const GUARD: u8 = 0xA5;
/// Random reference sets per (side, angle).
const RANDOM_REFS: usize = 4;
/// Reference patterns: random sets, then all 255, all 0, alternating.
const PATTERNS: usize = RANDOM_REFS + 3;

fn references(len: usize, pattern: usize, rng: &mut Rng) -> Vec<u8> {
    (0..len)
        .map(|i| match pattern {
            p if p < RANDOM_REFS => rng.byte(),
            p if p == RANDOM_REFS => u8::MAX,
            p if p == RANDOM_REFS + 1 => 0,
            _ => {
                if i.is_multiple_of(2) {
                    u8::MAX
                } else {
                    0
                }
            }
        })
        .collect()
}

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for n in [4, 8, 16, 32] {
        for pattern in 0..PATTERNS {
            for angle in -32..=32 {
                let refs = references(3 * n + 1, pattern, rng);
                let run = |dsp: &Dsp| {
                    let mut out = vec![GUARD; n * n + 2];
                    dsp.hevc_intra_angular(&refs, n, angle, &mut out[1..=n * n]);
                    out
                };
                report.cases += 1;
                if run(candidate) != run(reference) {
                    report.failures.push(format!(
                        "{} hevc_intra_angular {n}x{n} angle={angle} pattern={pattern}",
                        candidate.isa()
                    ));
                }
            }
            let samples = references(2 * n + 2, pattern, rng);
            let edges = HevcPlanarEdges {
                above: &samples[..n],
                left: &samples[n..2 * n],
                top_right: samples[2 * n],
                bottom_left: samples[2 * n + 1],
            };
            let run = |dsp: &Dsp| {
                let mut out = vec![GUARD; n * n + 2];
                dsp.hevc_intra_planar(&edges, n, &mut out[1..=n * n]);
                out
            };
            report.cases += 1;
            if run(candidate) != run(reference) {
                report.failures.push(format!(
                    "{} hevc_intra_planar {n}x{n} pattern={pattern}",
                    candidate.isa()
                ));
            }
        }
    }
}
