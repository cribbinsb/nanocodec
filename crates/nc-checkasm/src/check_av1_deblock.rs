//! AV1 loop filter edges against the scalar kernel: every filter length,
//! both orientations, 4..64 lines, edges at the plane borders (the fallback
//! paths), thresholds through the byte-exact limit, and pixel patterns that
//! reach every decision (rejected, narrow with and without high variance,
//! flat, flat2). Whole planes are compared.
use crate::{check::CheckReport, rng::Rng};
use nc_common::PlaneViewMut;
use nc_dsp::{AV1_EDGE_LINES, AV1_EDGE_REACH, Av1EdgeFilter, Av1FilterLength, Dsp};

/// Plane size: room for 64-line edges and every reach.
const SIDE: usize = 80;
/// Random threshold sets per (length, orientation, lines, pattern).
const THRESHOLD_SETS: usize = 3;
const LENGTHS: [Av1FilterLength; 4] = [
    Av1FilterLength::Four,
    Av1FilterLength::Six,
    Av1FilterLength::Eight,
    Av1FilterLength::Fourteen,
];

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    // Keep existing kernel inputs stable for baseline instruction-count comparisons.
    check_lines(candidate, reference, &mut rng.clone(), report);
    for length in LENGTHS {
        for vertical in [false, true] {
            for lines in [4, 8, 12, 16, 64] {
                // Edge positions: the smallest legal offset, a tight far side,
                // and a roomy interior position.
                let reach = length.reach();
                for offset in [reach, SIDE - reach, 2 * AV1_EDGE_REACH + 2] {
                    for pattern in 0..6 {
                        for set in 0..THRESHOLD_SETS {
                            let filter = thresholds(length, set, rng);
                            let plane = fill(pattern, rng);
                            let run = |dsp: &Dsp| {
                                let mut out = plane.clone();
                                let mut dst =
                                    PlaneViewMut::new(&mut out, SIDE, SIDE, SIDE).expect("plane");
                                let origin = if vertical { (offset, 3) } else { (3, offset) };
                                dsp.av1_loop_filter_edge(&mut dst, origin, vertical, lines, filter);
                                out
                            };
                            report.cases += 1;
                            if run(candidate) != run(reference) {
                                report.failures.push(format!(
                                    "{} av1_loop_filter_edge {length:?} vertical={vertical} lines={lines} offset={offset} pattern={pattern} {filter:?}",
                                    candidate.isa()
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Typical, minimal and extreme thresholds (`blimit` 254 is the largest the
/// byte kernels handle exactly; 255 takes their fallback).
fn thresholds(length: Av1FilterLength, set: usize, rng: &mut Rng) -> Av1EdgeFilter {
    let (limit, blimit, thresh) = match set {
        0 => (
            1 + rng.below(63) as u8,
            rng.below(256) as u8,
            rng.below(64) as u8,
        ),
        1 => (0, 0, 0),
        _ => (u8::MAX, if rng.below(2) == 0 { 254 } else { 255 }, u8::MAX),
    };
    Av1EdgeFilter {
        length,
        limit,
        blimit,
        thresh,
    }
}

/// Pixel patterns: random, smooth with a small step (flat, flat2), smooth
/// with a large step (narrow), near-threshold noise, alternating extremes,
/// and smooth with noise on the outer taps only (flat but not flat2).
fn fill(pattern: usize, rng: &mut Rng) -> Vec<u8> {
    (0..SIDE * SIDE)
        .map(|i| {
            let (x, y) = (i % SIDE, i / SIDE);
            let base = 100 + (x + y) as i32 / 16;
            let value = match pattern {
                0 => i32::from(rng.byte()),
                1 => base + i32::from((x + y) % 2 == 0) + rng.below(2) as i32,
                2 => base + if (x / 4 + y / 4) % 2 == 0 { 40 } else { 0 },
                3 => base + rng.below(9) as i32 - 4,
                4 => {
                    if (x + y) % 2 == 0 {
                        255
                    } else {
                        0
                    }
                }
                _ => {
                    base + if (x % 16 < 2 || x % 16 > 13) || (y % 16 < 2 || y % 16 > 13) {
                        rng.below(20) as i32
                    } else {
                        0
                    }
                }
            };
            value.clamp(0, 255) as u8
        })
        .collect()
}

/// Mixed masks, lengths and thresholds, including equal runs and four-line
/// tails. Compare both scalar batching and individual SIMD calls; whole-plane
/// comparison catches writes to masked segments and padding.
fn check_lines(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    const LINE_SIDE: usize = 2 * SIDE;
    for vertical in [false, true] {
        for count in [0, 1, 2, 3, 4, 7, 16, 32] {
            for pattern in 0..6 {
                for mode in 0..4 {
                    let repeated = thresholds(LENGTHS[rng.below(4)], 0, rng);
                    let segments: Vec<_> = (0..count)
                        .map(|i| match mode {
                            0 => None,
                            2 if i % 3 == 0 => None,
                            1 | 2 => Some(repeated),
                            _ => Some(thresholds(LENGTHS[rng.below(4)], i % THRESHOLD_SETS, rng)),
                        })
                        .collect();
                    let reach = segments
                        .iter()
                        .flatten()
                        .map(|f| f.length.reach())
                        .max()
                        .unwrap_or(1);
                    for offset in [reach, LINE_SIDE - reach, 2 * AV1_EDGE_REACH + 2] {
                        let origin = if vertical { (offset, 3) } else { (3, offset) };
                        let plane: Vec<_> = fill(pattern, rng)
                            .into_iter()
                            .cycle()
                            .take(LINE_SIDE * LINE_SIDE)
                            .collect();
                        let run = |dsp: &Dsp, batch: bool| {
                            let mut out = plane.clone();
                            let mut dst =
                                PlaneViewMut::new(&mut out, LINE_SIDE, LINE_SIDE, LINE_SIDE)
                                    .expect("plane");
                            if batch {
                                dsp.av1_loop_filter_line(&mut dst, origin, vertical, &segments);
                            } else {
                                for (i, filter) in segments.iter().enumerate() {
                                    if let Some(filter) = filter {
                                        let pos = if vertical {
                                            (origin.0, origin.1 + AV1_EDGE_LINES * i)
                                        } else {
                                            (origin.0 + AV1_EDGE_LINES * i, origin.1)
                                        };
                                        dsp.av1_loop_filter_edge(
                                            &mut dst,
                                            pos,
                                            vertical,
                                            AV1_EDGE_LINES,
                                            *filter,
                                        );
                                    }
                                }
                            }
                            out
                        };
                        let got = run(candidate, true);
                        report.cases += 2;
                        if got != run(reference, true) || got != run(candidate, false) {
                            report.failures.push(format!("{} av1_loop_filter_line vertical={vertical} count={count} pattern={pattern} mode={mode} offset={offset}", candidate.isa()));
                        }
                    }
                }
            }
        }
    }
}
