//! VVC primary transforms of every kind pair (DCT-II, DST-VII, DCT-VIII:
//! `Dsp::vvc_forward_primary`, `Dsp::vvc_inverse_primary`) against the
//! scalar reference: every 4- to 32-point shape, extreme and sparse
//! levels (the 32-point zero-out), residual extremes, guard values after
//! the block.
use crate::{check::CheckReport, rng::Rng};
use nc_dsp::{Dsp, VvcTransformKind};

const SIDES: [usize; 4] = [4, 8, 16, 32];
const KINDS: [VvcTransformKind; 3] = [
    VvcTransformKind::Dct2,
    VvcTransformKind::Dst7,
    VvcTransformKind::Dct8,
];
const GUARD: i32 = 0x1234_5678;
/// Level patterns: full-range noise, extremes, low-frequency only.
const PATTERNS: usize = 3;

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for width in SIDES {
        for height in SIDES {
            for kinds in KINDS.into_iter().flat_map(|h| KINDS.map(|v| [h, v])) {
                inverse(candidate, reference, rng, report, (width, height), kinds);
                forward(candidate, reference, rng, report, (width, height), kinds);
            }
        }
    }
}

fn inverse(
    candidate: &Dsp,
    reference: &Dsp,
    rng: &mut Rng,
    report: &mut CheckReport,
    (width, height): (usize, usize),
    kinds: [VvcTransformKind; 2],
) {
    let count = width * height;
    for pattern in 0..PATTERNS {
        for qp in [0, 4, 22, 37, 51, 63] {
            for dependent in [false, true] {
                let levels: Vec<i32> = (0..count)
                    .map(|i| match pattern {
                        0 => rng.below(65536) as i32 - 32768,
                        1 => [32767, -32768][i % 2],
                        _ if i % width < 5 && i / width < 3 => rng.below(201) as i32 - 100,
                        _ => 0,
                    })
                    .collect();
                let mut expected = vec![GUARD; count + 4];
                let mut actual = expected.clone();
                let size = (width, height);
                reference.vvc_inverse_primary(size, kinds, &levels, qp, dependent, &mut expected);
                candidate.vvc_inverse_primary(size, kinds, &levels, qp, dependent, &mut actual);
                report.cases += 1;
                if actual != expected || actual[count..].iter().any(|&v| v != GUARD) {
                    report.failures.push(format!(
                        "{} vvc_inverse_primary {width}x{height} {kinds:?} qp={qp} dependent={dependent} pattern={pattern}",
                        candidate.isa()
                    ));
                }
            }
        }
    }
}

fn forward(
    candidate: &Dsp,
    reference: &Dsp,
    rng: &mut Rng,
    report: &mut CheckReport,
    (width, height): (usize, usize),
    kinds: [VvcTransformKind; 2],
) {
    let count = width * height;
    for pattern in 0..PATTERNS {
        let residual: Vec<i16> = (0..count)
            .map(|i| match pattern {
                0 => rng.below(511) as i16 - 255,
                1 => [255, -255][(i + i / width) % 2],
                _ => 255,
            })
            .collect();
        let mut expected = vec![GUARD; count + 4];
        let mut actual = expected.clone();
        reference.vvc_forward_primary((width, height), kinds, &residual, &mut expected);
        candidate.vvc_forward_primary((width, height), kinds, &residual, &mut actual);
        report.cases += 1;
        if actual != expected || actual[count..].iter().any(|&v| v != GUARD) {
            report.failures.push(format!(
                "{} vvc_forward_primary {width}x{height} {kinds:?} pattern={pattern}",
                candidate.isa()
            ));
        }
    }
}
