//! AV1 loop restoration exactness: whole planes restored in place by the
//! candidate's and the scalar reference's row kernels
//! (`Av1LoopRestoration::apply`), over odd sizes, both stripe offsets,
//! extreme sample patterns and the extreme legal Wiener taps (the 16-bit
//! horizontal sums' worst cases), every self-guided set and random weights.

use crate::{check::CheckReport, rng::Rng};
use nc_common::{PlaneView, PlaneViewMut};
use nc_dsp::Dsp;
use nc_dsp::av1_restoration::{
    Av1LoopRestoration, Av1LrFilter, Av1LrScratch, Av1SgrProj, Av1StripeBoundaries, Av1Wiener,
    av1_lr_unit_count,
};

/// Plane sizes: narrower than a vector, between, around unit and stripe
/// sizes.
const SIZES: [(usize, usize); 8] = [
    (1, 1),
    (5, 9),
    (17, 61),
    (33, 70),
    (64, 64),
    (97, 135),
    (200, 47),
    (455, 140),
];

/// Wiener taps at the edges of their coded ranges (AV1 §5.11.57:
/// `[-5, 10]`, `[-23, 8]`, `[-17, 46]`): all positive at the maximum, all
/// negative at the minimum, and the mixed extreme.
const EXTREME_TAPS: [[i8; 3]; 4] = [[10, 8, 46], [-5, -23, -17], [10, -23, 46], [-5, 8, -17]];

/// Sample patterns: random, flat, alternating extremes, and an alternation
/// by column that maximises the Wiener sums' magnitude.
fn sample(pattern: usize, x: usize, y: usize, rng: &mut Rng) -> u8 {
    match pattern {
        0 => rng.byte(),
        1 => 128,
        2 => ((x + y) % 2 * 255) as u8,
        3 => [255, 0, 255, 0, 255, 255, 0][x % 7],
        _ => (x * 3 + y * 5) as u8,
    }
}

fn random_tap(rng: &mut Rng, (low, high): (i32, i32)) -> i8 {
    (low + rng.below((high - low + 1) as usize) as i32) as i8
}

fn random_filter(rng: &mut Rng) -> Av1LrFilter {
    const RANGES: [(i32, i32); 3] = [(-5, 10), (-23, 8), (-17, 46)];
    match rng.below(5) {
        0 => Av1LrFilter::None,
        1 => Av1LrFilter::Wiener(Av1Wiener {
            vertical: EXTREME_TAPS[rng.below(EXTREME_TAPS.len())],
            horizontal: EXTREME_TAPS[rng.below(EXTREME_TAPS.len())],
        }),
        2 => Av1LrFilter::Wiener(Av1Wiener {
            vertical: RANGES.map(|r| random_tap(rng, r)),
            horizontal: RANGES.map(|r| random_tap(rng, r)),
        }),
        _ => Av1LrFilter::SgrProj(Av1SgrProj {
            set: rng.below(16) as u8,
            xqd: [random_tap(rng, (-96, 31)), random_tap(rng, (-32, 95))],
        }),
    }
}

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    // Reused across cases, as decoders reuse them: stale contents from a
    // larger plane must not leak into a result.
    let mut scratch = [Av1LrScratch::default(), Av1LrScratch::default()];
    for (width, height) in SIZES {
        for pattern in 0..5 {
            for sub_y in [false, true] {
                let cdef: Vec<u8> = (0..width * height)
                    .map(|i| sample(pattern, i % width, i / width, rng))
                    .collect();
                let deblocked: Vec<u8> = (0..width * height)
                    .map(|i| sample((pattern + 1) % 5, i % width, i / width, rng))
                    .collect();
                let deblocked = PlaneView::new(&deblocked, width, width, height).unwrap();
                let boundaries = Av1StripeBoundaries::save(&deblocked, sub_y, None);
                // Luma units are 64 to 256 samples (AV1 §5.9.20), chroma halved.
                let unit_size = (64 << rng.below(3)) >> u32::from(sub_y);
                let (columns, rows) = (
                    av1_lr_unit_count(unit_size, width),
                    av1_lr_unit_count(unit_size, height),
                );
                let filters: Vec<Av1LrFilter> =
                    (0..columns * rows).map(|_| random_filter(rng)).collect();
                let restoration = Av1LoopRestoration {
                    unit_size,
                    columns,
                    rows,
                    filters: &filters,
                };
                let restore = |dsp: &Dsp, scratch: &mut Av1LrScratch| {
                    let mut plane = cdef.clone();
                    let mut view = PlaneViewMut::new(&mut plane, width, width, height).unwrap();
                    restoration.apply(dsp, &mut view, &boundaries, sub_y, scratch);
                    plane
                };
                report.cases += 1;
                let [candidate_scratch, reference_scratch] = &mut scratch;
                if restore(candidate, candidate_scratch) != restore(reference, reference_scratch) {
                    report.failures.push(format!(
                        "AV1 loop restoration mismatch: {width}x{height}, pattern {pattern}, \
                         sub_y {sub_y}, units of {unit_size}, filters {filters:?}"
                    ));
                }
            }
        }
    }
}
