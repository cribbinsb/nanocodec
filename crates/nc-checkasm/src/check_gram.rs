//! The Gram-rows kernel (`Dsp::gram_rows`, the AV1 Wiener restoration
//! statistics) against the scalar reference: every row count, lengths from
//! one vector to the documented limit, and values at the magnitude bound
//! (the largest 32-bit lane sums).
use crate::{check::CheckReport, rng::Rng};
use nc_dsp::{
    Dsp, GRAM_MAX_MAGNITUDE, GRAM_MAX_ROWS, GRAM_ROW_ALIGN, GRAM_ROW_MAX_LEN, gram_terms,
};

/// Row lengths: one and a few vectors, a restoration unit row, the limit.
const LENGTHS: [usize; 5] = [
    GRAM_ROW_ALIGN,
    3 * GRAM_ROW_ALIGN,
    64,
    256,
    GRAM_ROW_MAX_LEN,
];

/// A value of a test row: random, or both extremes of the bound.
fn value(rng: &mut Rng, pattern: usize) -> i16 {
    let bound = GRAM_MAX_MAGNITUDE;
    match pattern {
        0 => bound,
        1 => -bound,
        2 => {
            if rng.below(2) == 0 {
                bound
            } else {
                -bound
            }
        }
        _ => rng.below(2 * bound as usize + 1) as i16 - bound,
    }
}

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for n in 1..=GRAM_MAX_ROWS {
        for len in LENGTHS {
            for pattern in 0..4 {
                let rows: Vec<Vec<i16>> = (0..n)
                    .map(|_| (0..len).map(|_| value(rng, pattern)).collect())
                    .collect();
                let views: Vec<&[i16]> = rows.iter().map(Vec::as_slice).collect();
                // Accumulation starts from what `out` already holds.
                let start: Vec<i64> = (0..gram_terms(n))
                    .map(|_| i64::from(rng.next_u64() as i32))
                    .collect();
                let (mut got, mut want) = (start.clone(), start);
                candidate.gram_rows(&views, &mut got);
                reference.gram_rows(&views, &mut want);
                report.cases += 1;
                if got != want {
                    report.failures.push(format!(
                        "{} gram_rows n={n} len={len} pattern={pattern}",
                        candidate.isa()
                    ));
                }
            }
        }
    }
}
