//! The `u16` kernels of `Dsp::at` (`nc_dsp` `high_depth`) against the
//! scalar table at 10 bits: interpolation (uni, centred, default bi) for
//! every filter family the table has and every width, weighted
//! prediction, and the HEVC inverse transform add with random extents.
//! Samples are random and extreme (0 and 1023 runs, the filters' largest
//! overshoot).
use crate::{check::CheckReport, rng::Rng};
use nc_common::{BitDepth, PlaneView, PlaneViewMut};
use nc_dsp::{BlockSize, Dsp, Filter, HevcTransform, HevcWeights};

/// The depth checked: Main 10.
const DEPTH: BitDepth = BitDepth::TEN;
/// Block sides tried.
const SIDES: [usize; 5] = [4, 8, 16, 32, 64];
/// Filters with `u16` kernels.
const FILTERS: [Filter; 5] = [
    Filter::HevcLuma,
    Filter::HevcChroma,
    Filter::VvcLuma,
    Filter::VvcAffineLuma,
    Filter::VvcChroma,
];
/// Rounds per shape.
const ROUNDS: usize = 3;

fn samples(len: usize, round: usize, rng: &mut Rng) -> Vec<u16> {
    let max = DEPTH.max() as usize;
    (0..len)
        .map(|i| match round {
            0 => rng.below(max + 1) as u16,
            1 => {
                if (i / 3).is_multiple_of(2) {
                    max as u16
                } else {
                    0
                }
            }
            _ => {
                if rng.below(2) == 0 {
                    max as u16
                } else {
                    0
                }
            }
        })
        .collect()
}

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    let (cand, refer) = (candidate.at(DEPTH), reference.at(DEPTH));
    for filter in FILTERS {
        let span_extra = filter.taps() - 1;
        for side in SIDES {
            let size = BlockSize::new(side, side).expect("square kernel size");
            let span = side + span_extra;
            for round in 0..ROUNDS {
                let source = samples(span * span, round, rng);
                let window = PlaneView::new(&source, span, span, span).expect("window");
                let frac = (rng.below(filter.positions()), rng.below(filter.positions()));
                let mut prep = [vec![0i16; side * side], vec![0i16; side * side]];
                cand.predict_compound(filter, size, &window, frac, &mut prep[0], side);
                refer.predict_compound(filter, size, &window, frac, &mut prep[1], side);
                let mut uni = [vec![0u16; side * side], vec![0u16; side * side]];
                let mut bi = uni.clone();
                for (k, kernels) in [cand, refer].into_iter().enumerate() {
                    let mut uni_view =
                        PlaneViewMut::new(&mut uni[k], side, side, side).expect("view");
                    kernels.predict(filter, size, &window, frac, &mut uni_view);
                    let mut bi_view =
                        PlaneViewMut::new(&mut bi[k], side, side, side).expect("view");
                    kernels.hevc_predict_bi(
                        filter,
                        size,
                        &window,
                        frac,
                        (&prep[1], side),
                        &mut bi_view,
                    );
                }
                report.cases += 3;
                if prep[0] != prep[1] || uni[0] != uni[1] || bi[0] != bi[1] {
                    report.failures.push(format!(
                        "u16 {filter:?} {side}x{side} frac {frac:?} round {round}: prediction differs"
                    ));
                }
                let weights = HevcWeights {
                    w0: rng.below(384) as i16 - 128,
                    w1: rng.below(384) as i16 - 128,
                    round: (rng.below(256) as i32 - 128) << 6,
                    shift: 5 + rng.below(9) as u32,
                };
                let mut weighted = [vec![0u16; side * side], vec![0u16; side * side]];
                for (k, kernels) in [cand, refer].into_iter().enumerate() {
                    let mut view =
                        PlaneViewMut::new(&mut weighted[k], side, side, side).expect("view");
                    kernels
                        .hevc_weighted_prediction_centered(weights, &prep[0], &prep[1], &mut view);
                }
                report.cases += 1;
                if weighted[0] != weighted[1] {
                    report.failures.push(format!(
                        "u16 weighted prediction {side}x{side} {weights:?} differs"
                    ));
                }
            }
        }
    }
    for transform in HevcTransform::ALL {
        let n = transform.side();
        for round in 0..ROUNDS * 4 {
            let (rows, columns) = (1 + rng.below(n), 1 + rng.below(n));
            let coeffs: Vec<i16> = (0..n * n)
                .map(|i| {
                    let inside = i / n < rows && i % n < columns;
                    match (inside, round % 3) {
                        (false, _) => 0,
                        (true, 0) => rng.below(65_536) as i16,
                        (true, 1) => i16::MAX,
                        _ => rng.below(512) as i16 - 256,
                    }
                })
                .collect();
            let base = samples(n * n, round, rng);
            let mut out = [base.clone(), base];
            for (k, kernels) in [cand, refer].into_iter().enumerate() {
                let mut view = PlaneViewMut::new(&mut out[k], n, n, n).expect("view");
                kernels.hevc_inverse_add(transform, &coeffs, (rows, columns), &mut view);
            }
            report.cases += 1;
            if out[0] != out[1] {
                report.failures.push(format!(
                    "u16 {transform:?} inverse add, extent {rows}x{columns}, differs"
                ));
            }
        }
    }
}
