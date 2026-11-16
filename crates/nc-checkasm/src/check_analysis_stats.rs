//! Block-parallel statistics: exact slices, every vector boundary and odd
//! chroma pairs, inclusive thresholds, key frames and even-row SAD shapes.

use crate::check::CheckReport;
use crate::check_codec::record;
use nc_dsp::{AnalysisBlock, AnalysisChanges, Dsp, Moments};

const LENGTHS: [usize; 18] = [
    0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 159, 160, 161,
];

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, report: &mut CheckReport) {
    for len in LENGTHS {
        for pattern in 0..8 {
            let moments: Vec<_> = (0..len).map(|i| moment(i, pattern)).collect();
            let luma: Vec<_> = (0..len).map(|i| quad(i, pattern)).collect();
            let u: Vec<_> = (0..len.div_ceil(2))
                .map(|i| quad(2 * i + 1, pattern))
                .collect();
            let v: Vec<_> = (0..len.div_ceil(2))
                .map(|i| quad(3 * i + 2, pattern))
                .collect();
            for bottom in [false, true] {
                for limits in [(0, 0), (640, 320), (16320, 4080), (u32::MAX, u32::MAX)] {
                    let c = AnalysisChanges {
                        luma: &luma,
                        chroma: [&u, &v],
                        bottom,
                        block_limit: limits.0,
                        quarter_limit: limits.1,
                    };
                    compare(candidate, reference, &moments, Some(c), report, pattern);
                }
            }
            compare(candidate, reference, &moments, None, report, pattern);
        }
    }
}

fn compare(
    candidate: &Dsp,
    reference: &Dsp,
    m: &[Moments],
    c: Option<AnalysisChanges<'_>>,
    report: &mut CheckReport,
    pattern: usize,
) {
    let guard = AnalysisBlock {
        mean: 197,
        sum: 127,
        sum_sq_dev: 34567,
        sad_prev: Some(4321),
        is_static: true,
    };
    let mut actual = vec![guard; m.len() + 2];
    let mut expected = actual.clone();
    let a = candidate.analysis_stats_row(m, c, &mut actual[1..=m.len()]);
    let b = reference.analysis_stats_row(m, c, &mut expected[1..=m.len()]);
    record(
        report,
        candidate.isa(),
        a == b && actual == expected,
        || {
            format!(
                "analysis stats len {} pattern {pattern} changes {c:?}",
                m.len()
            )
        },
    );
}

fn moment(block: usize, pattern: usize) -> Moments {
    let mut m = Moments::default();
    for i in 0..64 {
        let value = match pattern {
            0 => 0,
            1 => 255,
            2 => ((i + block) % 2 * 255) as u32,
            _ => ((block * 31 + i * 17 + pattern * 53) % 256) as u32,
        };
        m.sum += value;
        m.sum_sq += value * value;
    }
    m
}

fn quad(block: usize, pattern: usize) -> [u16; 4] {
    core::array::from_fn(|q| match pattern {
        0 => 0,
        1 => 4080,
        2 => {
            if (block + q).is_multiple_of(2) {
                4080
            } else {
                0
            }
        }
        3 => [160, 160, 160, 160][q],
        4 => [320, 0, 0, 0][(block + q) % 4],
        5 => [321, 0, 0, 0][(block + q) % 4],
        6 => [320, 320, 1, 0][(block + q) % 4],
        _ => ((block * 191 + q * 701) % 4081) as u16,
    })
}
