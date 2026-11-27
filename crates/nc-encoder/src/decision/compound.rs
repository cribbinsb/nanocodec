//! Two-hypothesis inter trials: the equal-weight average of two block
//! predictions, either from the frame's two references
//! ([`ReferenceChoice::Compound`](crate::backend::ReferenceChoice), AV1
//! §7.11.3.15) or from the primary reference at two vectors
//! ([`ReferenceChoice::PrimaryPair`](crate::backend::ReferenceChoice):
//! low-delay bi-prediction, HEVC/VVC generalised B slices whose two lists
//! hold past pictures only), with the search for the pair's second vector.
//!
//! Both halves are predicted at the codec filter family's compound precision
//! and averaged with its rounding ([`Dsp::average_compound_for`]), so a
//! candidate is ranked on the samples the codec reconstructs.

use nc_common::{Picture, PlaneViewMut};
use nc_dsp::{Dsp, Filter, Variance};

use super::{
    Choice, DecidedMode, DecisionScratch, Measurer, PricingHint, Psy, dc_error, holds, n_variance,
};
use crate::backend::{CodecBackend, ReferenceChoice};
use crate::mv::{MV_UNITS_PER_SAMPLE, Mv};
use crate::params::DecisionParams;
use crate::partition::Block;
use crate::refresh::RefreshWindow;
use nc_analysis::preanalysis::PreAnalysis;

/// One quarter-sample step, the second vector's refinement step.
const QUARTER: i16 = 1;
/// One whole sample in vector units, the second vector's coarse seed step.
const SAMPLE: i16 = MV_UNITS_PER_SAMPLE as i16;

/// The filter the compound halves are predicted with: the backend's luma
/// filter where it has compound precision, else AV1's. A codec without
/// compound-precision kernels (VP9 averages two 8-bit predictions) is ranked
/// by AV1's compound prediction, as before it exposed its filter.
fn compound_filter<B: CodecBackend>(backend: &B) -> Filter {
    backend
        .luma_filter()
        .filter(|filter| filter.has_compound())
        .unwrap_or(Filter::Av1)
}

/// Grow `buffer` to hold at least `len` values.
fn reserve<T: Copy + Default>(buffer: &mut Vec<T>, len: usize) {
    if buffer.len() < len {
        buffer.resize(len, T::default());
    }
}

/// The average of compound predictions `first` and `second` of the
/// `scorer`'s block, written to `average`, and its residual statistics
/// against the source with the psychovisual penalty, and with `energy` the
/// average's AC energy.
fn score_average(
    dsp: &Dsp,
    filter: Filter,
    scorer: &Measurer<'_>,
    (first, second): (&[i16], &[i16]),
    average: &mut [u8],
    (psy, energy): (Option<Psy>, bool),
) -> (Variance, f32, u64) {
    let (w, h) = scorer.extent();
    let mut view = PlaneViewMut::new(average, w, w, h).expect("block fits");
    dsp.average_compound_for(filter, scorer.size, (first, second), w, &mut view);
    if psy.is_some() || energy {
        let (stats, prediction) = scorer.stats_and_energy(&view.as_view());
        (
            stats,
            psy.map_or(0.0, |psy| psy.penalty(prediction)),
            prediction,
        )
    } else {
        (scorer.residual_stats(&view.as_view()), 0.0, 0)
    }
}

/// The blur term of a two-hypothesis candidate (`decision.bi_pair_psy`):
/// averaging two predictions low-passes them, which the squared error does
/// not see but VMAF and MS-SSIM do (traffic, 505 kbps: equal PSNR, 8 % less
/// reconstructed gradient energy). The AC energy the pair's prediction loses
/// against the single-reference prediction (`single_energy`) is priced at
/// `strength` squared-error units per unit of energy.
#[derive(Debug, Clone, Copy)]
struct Blur {
    strength: f32,
    single_energy: u64,
}

impl Blur {
    fn penalty(self, pair_energy: u64) -> f32 {
        self.strength * self.single_energy.saturating_sub(pair_energy) as f32
    }
}

/// The AC energy of the uni-prediction of the measurer's block at `mv`
/// (its compound-precision prediction averaged with itself: the codec's
/// uni-prediction samples), using `a` and `out` as scratch.
fn uni_energy(
    dsp: &Dsp,
    filter: Filter,
    measurer: &Measurer<'_>,
    mv: Mv,
    (a, out): (&mut [i16], &mut [u8]),
) -> u64 {
    measurer.predict_compound_into(filter, mv, a);
    let (w, h) = measurer.extent();
    let mut view = PlaneViewMut::new(out, w, w, h).expect("block fits");
    dsp.average_compound_for(filter, measurer.size, (a, a), w, &mut view);
    crate::psy::ac_energy(dsp, measurer.size, &view.as_view())
}

/// Evaluate the equal-weight compound prediction averaging `primary_reference`
/// at `primary_mv` and `secondary_reference` at `secondary_mv`.
///
/// Called only when both single-reference choices are inter and within
/// `refs.second_compound_margin` of each other, so compound is tried only on
/// blocks where the two references are genuinely competitive.
pub(crate) fn choose_compound_mode<B: CodecBackend>(
    dsp: &Dsp,
    backend: &mut B,
    block: &Block,
    source: &Picture,
    primary_reference: &Picture,
    secondary_reference: &Picture,
    (primary_mv, secondary_mv, include_dc): (Mv, Mv, bool),
    scratch: &mut DecisionScratch,
    (params, source_moments): (&DecisionParams, Option<&PreAnalysis>),
    refresh: Option<&RefreshWindow>,
) -> Option<Choice> {
    if refresh.is_some_and(|w| !w.allows(block, primary_mv) || !w.allows(block, secondary_mv)) {
        return None;
    }
    let DecisionScratch {
        measure,
        compound_a,
        compound_b,
        ..
    } = scratch;
    let lambda = backend.lambda() * params.lambda_scale;
    let primary_measurer = Measurer::new::<B>(dsp, source, primary_reference, block)
        .with_refresh(refresh)
        .with_prediction_energy_and_rows(params, source_moments);
    let secondary_measurer = Measurer::new::<B>(dsp, source, secondary_reference, block)
        .with_refresh(refresh)
        .with_row_step(params.distortion_row_step);
    let (w, h) = primary_measurer.extent();
    let area = w * h;
    reserve(compound_a, area);
    reserve(compound_b, area);
    let mut pred_buf = measure.prediction.borrow_mut();
    reserve(&mut pred_buf, area);
    let filter = compound_filter(backend);
    primary_measurer.predict_compound_into(filter, primary_mv, &mut compound_a[..area]);
    secondary_measurer.predict_compound_into(filter, secondary_mv, &mut compound_b[..area]);
    // The compound candidate is one block prediction per reference.
    measure.count_positions(2);
    let (stats, penalty, _) = score_average(
        dsp,
        filter,
        &primary_measurer,
        (&compound_a[..area], &compound_b[..area]),
        &mut pred_buf[..area],
        (Psy::of(params, &primary_measurer), false),
    );
    let distortion = n_variance::<B>(stats, block) as f32;
    let bits = backend.compound_mode_bits(block, primary_mv, secondary_mv);
    let mut cost = distortion + lambda * bits + penalty;
    if include_dc && cost.is_finite() {
        cost += dc_error::<B>(params, stats, block);
    }
    cost.is_finite().then_some(Choice {
        mode: DecidedMode::inter(primary_mv),
        cost,
        pricing: PricingHint::NONE,
    })
}

/// Low-delay bi-prediction of `block` from `reference` alone: the
/// equal-weight average of the prediction at `first_mv` (the
/// single-reference winner) and at a second vector, with that second
/// vector. `None` when no second vector gives a finite cost.
///
/// The second vector is searched with the first prediction fixed (the
/// iterative bi-prediction search of the JVET reference encoders, one pass,
/// simplified for real time): the codec's predicted vectors, the zero
/// vector and the first vector's whole- and quarter-sample neighbours are
/// ranked by the averaged prediction's cost, then the winner is refined by
/// up to `decision.bi_pair_steps` quarter-sample crosses. Every position is
/// one compound prediction and average, so the trial costs about
/// `13 + 4 · steps` block predictions.
pub(crate) fn choose_primary_pair<B: CodecBackend>(
    dsp: &Dsp,
    backend: &mut B,
    block: &Block,
    source: &Picture,
    reference: &Picture,
    (first_mv, include_dc): (Mv, bool),
    scratch: &mut DecisionScratch,
    (params, source_moments): (&DecisionParams, Option<&PreAnalysis>),
    refresh: Option<&RefreshWindow>,
) -> Option<(Choice, Mv, f32)> {
    if refresh.is_some_and(|w| !w.allows(block, first_mv)) {
        return None;
    }
    let DecisionScratch {
        candidates,
        measure,
        compound_a,
        compound_b,
        ..
    } = scratch;
    second_vector_seeds(backend, block, first_mv, candidates);
    let lambda = backend.lambda() * params.lambda_scale;
    let measurer = Measurer::new::<B>(dsp, source, reference, block)
        .with_refresh(refresh)
        .with_prediction_energy_and_rows(params, source_moments);
    let (w, h) = measurer.extent();
    let area = w * h;
    reserve(compound_a, area);
    reserve(compound_b, area);
    let mut pred_buf = measure.prediction.borrow_mut();
    reserve(&mut pred_buf, area);
    let filter = compound_filter(backend);
    let blur = (params.bi_pair_psy > 0.0).then(|| Blur {
        strength: params.bi_pair_psy,
        single_energy: uni_energy(
            dsp,
            filter,
            &measurer,
            first_mv,
            (&mut compound_a[..area], &mut pred_buf[..area]),
        ),
    });
    measurer.predict_compound_into(filter, first_mv, &mut compound_a[..area]);
    measure.count_positions(1);
    let psy = Psy::of(params, &measurer);
    let mut cost_of = |second_mv: Mv| -> Option<(f32, Variance, f32)> {
        // The pair at one vector is the single prediction, already ranked.
        if second_mv == first_mv || refresh.is_some_and(|w| !w.allows(block, second_mv)) {
            return None;
        }
        measurer.predict_compound_into(filter, second_mv, &mut compound_b[..area]);
        measure.count_positions(1);
        let (stats, penalty, energy) = score_average(
            dsp,
            filter,
            &measurer,
            (&compound_a[..area], &compound_b[..area]),
            &mut pred_buf[..area],
            (psy, blur.is_some()),
        );
        let blurred = blur.map_or(0.0, |blur| blur.penalty(energy));
        let bits = backend.primary_pair_mode_bits(block, first_mv, second_mv);
        let cost = n_variance::<B>(stats, block) as f32 + lambda * bits + penalty + blurred;
        cost.is_finite().then_some((cost, stats, blurred))
    };
    let mut best: Option<(f32, Variance, Mv, f32)> = None;
    for &mv in candidates.iter() {
        if let Some((cost, stats, blurred)) = cost_of(mv)
            && best.is_none_or(|(b, ..)| cost < b)
        {
            best = Some((cost, stats, mv, blurred));
        }
    }
    let (mut cost, mut stats, mut second_mv, mut blurred) = best?;
    // What it skips: further crosses once one finds nothing cheaper, and
    // all of them past `decision.bi_pair_steps`. Why safe: the pair's cost
    // surface around the ranked winner is smooth at quarter-sample steps;
    // the bound caps the trial's work. Effect: not yet measured (the trial
    // is off by default).
    for _ in 0..params.bi_pair_steps.round() as usize {
        let centre = second_mv;
        for (dx, dy) in [(QUARTER, 0), (-QUARTER, 0), (0, QUARTER), (0, -QUARTER)] {
            let mv = step(centre, dx, dy);
            if let Some((c, s, b)) = cost_of(mv)
                && c < cost
            {
                (cost, stats, second_mv, blurred) = (c, s, mv, b);
            }
        }
        if second_mv == centre {
            break;
        }
    }
    if include_dc {
        cost += dc_error::<B>(params, stats, block);
    }
    Some((
        Choice {
            mode: DecidedMode::inter(first_mv),
            cost,
            pricing: PricingHint::NONE,
        },
        second_mv,
        blurred,
    ))
}

/// The second vector's ranked seeds into `candidates`: zero, the codec's
/// predicted vectors and `first_mv`'s whole- and quarter-sample neighbours.
fn second_vector_seeds<B: CodecBackend>(
    backend: &mut B,
    block: &Block,
    first_mv: Mv,
    candidates: &mut Vec<Mv>,
) {
    candidates.clear();
    candidates.push(Mv::ZERO);
    backend.mv_candidates(block, candidates);
    for (dx, dy) in [
        (SAMPLE, 0),
        (-SAMPLE, 0),
        (0, SAMPLE),
        (0, -SAMPLE),
        (QUARTER, 0),
        (-QUARTER, 0),
        (0, QUARTER),
        (0, -QUARTER),
    ] {
        let mv = step(first_mv, dx, dy);
        if !holds(candidates, mv) {
            candidates.push(mv);
        }
    }
}

/// `mv` moved by `(dx, dy)` vector units, saturating at the vector range.
fn step(mv: Mv, dx: i16, dy: i16) -> Mv {
    Mv {
        x: mv.x.saturating_add(dx),
        y: mv.y.saturating_add(dy),
    }
}

/// The cheapest of the backend's two-hypothesis merge candidates of `block`
/// (`CodecBackend::pair_merge_candidates`): each averages `primary_reference`
/// at its first vector with its second half, from `primary_reference`
/// ([`ReferenceChoice::PrimaryPair`]) or `secondary_reference`
/// ([`ReferenceChoice::Compound`], skipped without one). Merged motion costs
/// a few bits, so these are tried as they are, with no search.
pub(crate) fn choose_pair_merge<B: CodecBackend>(
    dsp: &Dsp,
    backend: &mut B,
    block: &Block,
    source: &Picture,
    (primary_reference, secondary_reference): (&Picture, Option<&Picture>),
    single_mv: Mv,
    scratch: &mut DecisionScratch,
    (params, source_moments): (&DecisionParams, Option<&PreAnalysis>),
    refresh: Option<&RefreshWindow>,
) -> Option<(Choice, ReferenceChoice, f32)> {
    let DecisionScratch {
        measure,
        compound_a,
        compound_b,
        pairs,
        ..
    } = scratch;
    pairs.clear();
    backend.pair_merge_candidates(block, pairs);
    if pairs.is_empty() {
        return None;
    }
    let lambda = backend.lambda() * params.lambda_scale;
    let first = Measurer::new::<B>(dsp, source, primary_reference, block)
        .with_refresh(refresh)
        .with_prediction_energy_and_rows(params, source_moments);
    let secondary = secondary_reference.map(|reference| {
        Measurer::new::<B>(dsp, source, reference, block)
            .with_refresh(refresh)
            .with_row_step(params.distortion_row_step)
    });
    let (w, h) = first.extent();
    let area = w * h;
    reserve(compound_a, area);
    reserve(compound_b, area);
    let mut pred_buf = measure.prediction.borrow_mut();
    reserve(&mut pred_buf, area);
    let filter = compound_filter(backend);
    let psy = Psy::of(params, &first);
    let blur = (params.bi_pair_psy > 0.0).then(|| Blur {
        strength: params.bi_pair_psy,
        single_energy: uni_energy(
            dsp,
            filter,
            &first,
            single_mv,
            (&mut compound_a[..area], &mut pred_buf[..area]),
        ),
    });
    let mut best: Option<(Choice, ReferenceChoice, f32)> = None;
    for &(first_mv, second) in pairs.iter() {
        let (second_measurer, second_mv) = match second {
            ReferenceChoice::PrimaryPair { second_mv } => (&first, second_mv),
            ReferenceChoice::Compound { secondary_mv } => match &secondary {
                Some(measurer) => (measurer, secondary_mv),
                None => continue,
            },
            ReferenceChoice::Primary | ReferenceChoice::Secondary => continue,
        };
        if refresh.is_some_and(|w| !w.allows(block, first_mv) || !w.allows(block, second_mv)) {
            continue;
        }
        first.predict_compound_into(filter, first_mv, &mut compound_a[..area]);
        second_measurer.predict_compound_into(filter, second_mv, &mut compound_b[..area]);
        measure.count_positions(2);
        let (stats, penalty, energy) = score_average(
            dsp,
            filter,
            &first,
            (&compound_a[..area], &compound_b[..area]),
            &mut pred_buf[..area],
            (psy, blur.is_some()),
        );
        let blurred = blur.map_or(0.0, |blur| blur.penalty(energy));
        let bits = match second {
            ReferenceChoice::Compound { .. } => {
                backend.compound_mode_bits(block, first_mv, second_mv)
            }
            _ => backend.primary_pair_mode_bits(block, first_mv, second_mv),
        };
        let cost = n_variance::<B>(stats, block) as f32 + lambda * bits + penalty + blurred;
        if cost.is_finite() && best.is_none_or(|(b, ..)| cost < b.cost) {
            best = Some((
                Choice {
                    mode: DecidedMode::inter(first_mv),
                    cost,
                    pricing: PricingHint::NONE,
                },
                second,
                blurred,
            ));
        }
    }
    best
}
