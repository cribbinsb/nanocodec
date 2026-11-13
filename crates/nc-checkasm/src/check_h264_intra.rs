//! H.264 `Intra_8x8` prediction against the scalar reference: every mode
//! the neighbours allow, every availability combination, random, extreme
//! and alternating neighbours.
use crate::{check::CheckReport, rng::Rng};
use nc_dsp::{Dsp, H264Intra8x8Edges, h264_intra_8x8_mode};

/// Random neighbour sets per availability combination.
const RANDOM_EDGES: usize = 8;
/// Neighbour patterns: random sets, then all 255, all 0, alternating.
const PATTERNS: usize = RANDOM_EDGES + 3;
/// Availability flags: top, top-right, left, corner.
const AVAILABILITY_COMBINATIONS: u8 = 16;

fn sample(pattern: usize, i: usize, rng: &mut Rng) -> u8 {
    match pattern {
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
    }
}

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for flags in 0..AVAILABILITY_COMBINATIONS {
        for pattern in 0..PATTERNS {
            let mut i = 0;
            let mut next = || {
                i += 1;
                sample(pattern, i, rng)
            };
            let edges = H264Intra8x8Edges {
                top: core::array::from_fn(|_| next()),
                left: core::array::from_fn(|_| next()),
                corner: next(),
                has_top: flags & 1 != 0,
                has_top_right: flags & 2 != 0,
                has_left: flags & 4 != 0,
                has_corner: flags & 8 != 0,
            };
            let refs = edges.filtered();
            for mode in (0..h264_intra_8x8_mode::COUNT).filter(|&m| refs.allows(m)) {
                let run = |dsp: &Dsp| {
                    let mut out = [0u8; 64];
                    dsp.h264_intra_8x8(&refs, mode, &mut out);
                    out
                };
                report.cases += 1;
                if run(candidate) != run(reference) {
                    report.failures.push(format!(
                        "{} h264_intra_8x8 mode={mode} flags={flags:#06b} pattern={pattern}",
                        candidate.isa()
                    ));
                }
            }
        }
    }
}
