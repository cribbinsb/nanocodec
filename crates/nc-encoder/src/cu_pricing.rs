//! Whole-CU pricing of an inter-frame leaf (`decision.cu_price_*`): where
//! the ranking's choice is in doubt, the codec codes the leaf both ways, in
//! all planes with its real coder, and the cheaper coded CU is kept.
//!
//! Why: the ranking (`decision.rs`) and its DC intra test compare uncoded
//! luma costs. A shadow log of HEVC Default that re-coded every P-frame CU
//! every way (docs/handoff/hevc.md "VC-1K: shadow RD ceiling") found the
//! choice the full coder prefers saves 5.9 % of the coded cost (all planes,
//! CABAC bits) at our partition on the six worst VC-1K clips, spread over
//! many transitions: inter -> intra 2.0 points, intra -> inter 1.2,
//! other vectors 0.9, other transform trees 0.6. Chroma carries about a
//! third of it, so the pricing is in all three planes, unlike the luma-only
//! final stage (`final_stage.rs`).
//!
//! Which leaves are priced (the two places the log found most gain per
//! coding):
//! - an inter leaf whose winner leaves more than `cu_price_error_lambdas` λ
//!   per sample of (mean-removed) prediction error, and whose estimated
//!   intra cost is within `cu_price_intra_ratio` times inter's, is also
//!   coded as intra: these flips are not near the DC test's margin (half
//!   their gain is where DC lost by over 0.1 λ per sample), but the error
//!   separates them; the ratio drops textured leaves intra never wins
//!   (replayed at 0.1 λ: ratio 4 keeps the gain and prices 6.3 % of the
//!   worst six's area against 9.7 %, 5.6 % of six fresh clips' against
//!   30.7 %); with `cu_price_screen_ratio` the codec's intra screen
//!   ([`CodecBackend::final_intra_screen`], from the reconstruction) must
//!   also come within that multiple of inter's cost;
//! - an intra leaf that beat inter by less than `cu_price_intra_lambdas` λ
//!   per sample is also coded at the codec's skip vector (with residual
//!   where it pays): 69 % of the intra -> inter gain is within 0.05 λ.
//!
//! The alternative is coded first and undone, then the ranking's choice is
//! coded and kept if it is the cheaper (so the common case codes one extra
//! CU); otherwise it is undone and the alternative coded again. Costs come
//! from the codec ([`CodecBackend::cu_trial_cost`]), which also restores
//! its own state ([`CodecBackend::undo_cu_trial`]).

use nc_common::Picture;

use crate::backend::{BlockDecision, CodecBackend, PredictionMode, ReferenceChoice};
use crate::decision::{PricingHint, leaf_area};
use crate::params::DecisionParams;
use crate::partition::Block;

/// Whether the codec's intra screen of `block` (uncoded luma error from
/// the reconstruction + λ · mode bits) comes within
/// `cu_price_screen_ratio` of inter's cost (always, with the screen off or
/// without one).
fn intra_screen_passes<B: CodecBackend>(
    backend: &mut B,
    (source, recon): (&Picture, &Picture),
    block: &Block,
    hint: PricingHint,
    params: &DecisionParams,
) -> bool {
    if params.cu_price_screen_ratio <= 0.0 {
        return true;
    }
    backend
        .final_intra_screen(source, recon, block)
        .is_none_or(|(sse, bits)| {
            sse + backend.lambda() * bits < params.cu_price_screen_ratio * hint.inter_cost
        })
}

/// The alternative to price against `decision`, if its hint passes a gate.
fn alternative<B: CodecBackend>(
    backend: &mut B,
    block: &Block,
    decision: &BlockDecision,
    hint: PricingHint,
    params: &DecisionParams,
    pictures: (&Picture, &Picture),
) -> Option<BlockDecision> {
    let per_sample = backend.lambda() * leaf_area::<B>(block) as f32;
    match decision.mode {
        PredictionMode::Inter(_)
            if params.cu_price_error_lambdas > 0.0
                && hint.winner_error > params.cu_price_error_lambdas * per_sample
                && (params.cu_price_intra_ratio <= 0.0
                    || hint.intra_ratio < params.cu_price_intra_ratio)
                && intra_screen_passes(backend, pictures, block, hint, params) =>
        {
            Some(BlockDecision {
                mode: PredictionMode::IntraDc,
                reference: ReferenceChoice::Primary,
                intra_fallback: None,
                ..*decision
            })
        }
        PredictionMode::IntraDc
            if params.cu_price_intra_lambdas > 0.0
                && hint.intra_margin < params.cu_price_intra_lambdas * per_sample =>
        {
            let skip = backend.cu_trial_skip_vector(block)?;
            Some(BlockDecision {
                mode: PredictionMode::Inter(skip),
                reference: ReferenceChoice::Primary,
                intra_fallback: None,
                ..*decision
            })
        }
        _ => None,
    }
}

/// Code leaf `block` as `decision`, or as its priced alternative where that
/// codes cheaper; returns what was coded. `reference` is the picture
/// `decision` predicts from, `primary` the primary reference (the
/// alternatives' inter reference).
pub(crate) fn code_leaf<B: CodecBackend>(
    backend: &mut B,
    block: &Block,
    decision: BlockDecision,
    hint: PricingHint,
    params: &DecisionParams,
    (source, reference, primary): (&Picture, Option<&Picture>, &Picture),
    recon: &mut Picture,
) -> BlockDecision {
    let alternative = alternative(backend, block, &decision, hint, params, (source, recon))
        .filter(|_| backend.begin_cu_trial(block, recon));
    let Some(alternative) = alternative else {
        backend.encode_block(block, &decision, source, reference, recon);
        return decision;
    };
    let alternative_reference = match alternative.mode {
        PredictionMode::Inter(_) => Some(primary),
        PredictionMode::IntraDc => reference,
    };
    backend.encode_block(block, &alternative, source, alternative_reference, recon);
    let alternative_cost = backend.cu_trial_cost(block, source, recon);
    backend.undo_cu_trial(recon);
    backend.encode_block(block, &decision, source, reference, recon);
    let coded = if backend.cu_trial_cost(block, source, recon) <= alternative_cost {
        decision
    } else {
        backend.undo_cu_trial(recon);
        backend.encode_block(block, &alternative, source, alternative_reference, recon);
        alternative
    };
    backend.end_cu_trial();
    coded
}
