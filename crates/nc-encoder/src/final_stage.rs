//! The final decision stage of an inter leaf (`decision.final_stage`): the
//! cheap ranking's winner, the codec's skip vector and the codec's best intra
//! prediction decided by coded cost, where the ranking cannot tell them apart.
//!
//! Why: the ranking (`decision.rs`) orders inter vectors by mean-removed
//! residual variance + λ · mode bits, and tries DC intra against it by the
//! source's AC energy. Neither prices the residual's coding. On flat, dark or
//! noisy video (VC-1K phone calls) the winner, the skip vector and intra
//! often sit within 0.05 λ per sample of each other on that metric, and the
//! coded-cost-best choice (luma SSE + CABAC bits) saves 6.3 % of the luma
//! cost of our choice on the six worst VC-1K clips, 2.5 % on six good fresh
//! clips; winner -> intra, winner -> skip vector and intra -> skip vector
//! carry 5.3 of the 6.3 points (`docs/handoff/rd-mode-decision.md`).
//!
//! How it stays cheap (SVT-AV1's MDS staging, x264's "RD only on the close
//! candidates"): two levels, luma only.
//! 1. **Screen**, nearly free: each candidate's luma SSE with no residual
//!    plus λ · its mode bits. The winner's and the skip vector's SSE come
//!    from the ranking's own measurements; intra's from the codec's
//!    whole-block intra predictor ([`CodecBackend::final_intra_screen`]),
//!    tried only where the source's AC energy is below
//!    `final_intra_energy_ratio` times the winner's screen cost.
//! 2. **Price**, only when an alternative screens below the winner or
//!    within `final_margin_lambdas` λ per sample above it: the winner and
//!    those alternatives are priced by the codec's coded luma cost
//!    ([`CodecBackend::final_coded_cost`]: transform, quantisation,
//!    reconstruction, entropy-coded bits; chroma uncoded), and the cheapest
//!    wins. The screen never switches by itself, and a codec without a
//!    pricing keeps the winner.
//!
//! The returned [`Choice`] carries a cost in the ranking's units (what the
//! secondary-reference decision compares against), not the coded cost.

use nc_common::Picture;

use crate::backend::{CodecBackend, FinalCandidate, PredictionMode};
use crate::decision::{Choice, DecidedMode, is_square_leaf, leaf_area};
use crate::mv::Mv;
use crate::params::DecisionParams;
use crate::partition::Block;

/// The winner, the skip vector and intra.
const MAX_CANDIDATES: usize = 3;

/// One inter finalist as the ranking measured it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct InterFinalist {
    pub mv: Mv,
    /// Luma squared error of its prediction (no residual).
    pub sse: f32,
    /// Its cost in the ranking's units.
    pub rank_cost: f32,
}

/// What the stage decides between.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Finalists {
    /// The ranking's (walked) winner.
    pub winner: InterFinalist,
    /// The codec's skip vector, when it differs from the winner.
    pub skip: Option<InterFinalist>,
    /// The source block's AC energy (squared deviation from its mean).
    pub source_energy: f32,
    /// The ranking-unit cost intra is reported with when it wins.
    pub intra_rank_cost: f32,
    /// Whether the winner's residual passes the ranking's intra gate
    /// (`decision.intra_gate_*`): intra is screened only then.
    pub intra_gate_open: bool,
}

/// A screened candidate.
#[derive(Debug, Clone, Copy)]
struct Screened {
    candidate: FinalCandidate,
    /// Luma squared error with no residual.
    sse: f32,
    cost: f32,
}

/// The final choice for inter leaf `block` among `finalists`: inter from
/// `reference`, intra from the reconstruction `recon` (none: no intra).
pub(crate) fn decide<B: CodecBackend>(
    backend: &mut B,
    (source, reference, recon): (&Picture, &Picture, Option<&Picture>),
    block: &Block,
    params: &DecisionParams,
    finalists: Finalists,
) -> Choice {
    let lambda = backend.lambda();
    let mut screened = [None::<Screened>; MAX_CANDIDATES];
    let inter_screen = |backend: &B, f: InterFinalist| Screened {
        candidate: FinalCandidate::Inter(f.mv),
        sse: f.sse,
        cost: f.sse + lambda * backend.mode_bits(block, &PredictionMode::Inter(f.mv)),
    };
    let winner = inter_screen(backend, finalists.winner);
    screened[0] = Some(winner);
    screened[1] = finalists.skip.map(|s| inter_screen(backend, s));
    let ratio = params.final_intra_energy_ratio;
    // A source block with more AC energy than a few times the winner's cost
    // is far from any whole-block intra prediction's reach: its screen would
    // cost a prediction per mode for nothing.
    let intra_plausible = finalists.intra_gate_open
        && (ratio <= 0.0 || finalists.source_energy < ratio * winner.cost);
    if let Some(recon) = recon.filter(|_| intra_plausible && is_square_leaf::<B>(block)) {
        screened[2] = backend
            .final_intra_screen(source, recon, block)
            .map(|(sse, bits)| Screened {
                candidate: FinalCandidate::Intra,
                sse,
                cost: sse + lambda * bits,
            });
    }
    let lambda_area = lambda * leaf_area::<B>(block) as f32;
    // Only an alternative that screens below, or within the margin above,
    // the winner can be worth pricing; the winner stays unless the coded
    // cost overturns it. (Letting the screen alone switch to an alternative
    // that screened lower lost J +1.9 % on the VC-1K worst six and two
    // static clips: the uncoded error cannot see what the winner's
    // residual would buy.)
    let close_to_winner = winner.cost + params.final_margin_lambdas * lambda_area;
    let close = |s: &Screened| s.cost <= close_to_winner;
    // A prediction whose whole error is below `final_price_min_lambdas` λ
    // per sample codes no residual worth its bits: its screen cost is its
    // coded cost, up to the header bits' estimate.
    let uncoded = params.final_price_min_lambdas * lambda_area;
    let chosen = if screened[1..].iter().flatten().any(&close) {
        price_close(
            backend,
            (source, reference),
            block,
            &screened,
            (close, uncoded),
        )
    } else {
        winner.candidate
    };
    let mode = match chosen {
        FinalCandidate::Inter(mv) => DecidedMode::inter(mv),
        FinalCandidate::Intra => DecidedMode::INTRA,
    };
    let cost = match chosen {
        FinalCandidate::Inter(mv) if mv == finalists.winner.mv => finalists.winner.rank_cost,
        FinalCandidate::Inter(_) => finalists
            .skip
            .map_or(finalists.winner.rank_cost, |s| s.rank_cost),
        FinalCandidate::Intra => finalists.intra_rank_cost,
    };
    Choice { mode, cost }
}

/// The candidate of the lowest cost (the first on ties: the winner).
fn cheapest(list: impl Iterator<Item = Screened>) -> FinalCandidate {
    list.fold(None::<Screened>, |best, s| match best {
        Some(b) if b.cost <= s.cost => Some(b),
        _ => Some(s),
    })
    .expect("the winner is always screened")
    .candidate
}

/// The cheapest by the codec's coded cost of the `close` screened
/// candidates, those with less error than `uncoded` at their screen cost;
/// the winner (the first) if the codec prices none of them.
fn price_close<B: CodecBackend>(
    backend: &mut B,
    (source, reference): (&Picture, &Picture),
    block: &Block,
    screened: &[Option<Screened>; MAX_CANDIDATES],
    (close, uncoded): (impl Fn(&Screened) -> bool, f32),
) -> FinalCandidate {
    let mut priced = [None::<Screened>; MAX_CANDIDATES];
    for (slot, s) in priced.iter_mut().zip(screened) {
        let Some(s) = s.filter(|s| close(s)) else {
            continue;
        };
        if s.sse < uncoded {
            *slot = Some(s);
            continue;
        }
        match backend.final_coded_cost(source, reference, block, s.candidate) {
            Some(cost) => *slot = Some(Screened { cost, ..s }),
            None => {
                return screened[0]
                    .expect("the winner is always screened")
                    .candidate;
            }
        }
    }
    cheapest(priced.iter().flatten().copied())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BlockDecision;
    use std::cell::Cell;

    /// Mode bits 0 and λ 1, so screen costs are the given SSEs; coded
    /// costs from `prices`, counted.
    struct Priced {
        prices: Vec<(FinalCandidate, f32)>,
        intra: Option<(f32, f32)>,
        calls: Cell<usize>,
    }

    impl CodecBackend for Priced {
        fn new(_: &crate::config::EncoderConfig) -> Result<Self, crate::config::ConfigError> {
            unreachable!("built directly")
        }
        fn superblock_log2(&self) -> u8 {
            4
        }
        fn min_block_log2(&self) -> u8 {
            4
        }
        fn mode_bits(&self, _: &Block, _: &PredictionMode) -> f32 {
            0.0
        }
        fn final_intra_screen(
            &mut self,
            _: &Picture,
            _: &Picture,
            _: &Block,
        ) -> Option<(f32, f32)> {
            self.intra
        }
        fn final_coded_cost(
            &mut self,
            _: &Picture,
            _: &Picture,
            _: &Block,
            candidate: FinalCandidate,
        ) -> Option<f32> {
            self.calls.set(self.calls.get() + 1);
            self.prices
                .iter()
                .find(|(c, _)| *c == candidate)
                .map(|&(_, cost)| cost)
        }
        fn begin_frame(&mut self, _: &crate::backend::FramePlan) {}
        fn begin_superblock(&mut self, _: usize, _: usize) {}
        fn write_partition(&mut self, _: &Block, _: bool) {}
        fn encode_block(
            &mut self,
            _: &Block,
            _: &BlockDecision,
            _: &Picture,
            _: Option<&Picture>,
            _: &mut Picture,
        ) {
        }
        fn end_frame(&mut self) -> Vec<u8> {
            Vec::new()
        }
    }

    const WINNER: Mv = Mv { x: 4, y: 0 };
    const SKIP: Mv = Mv { x: 0, y: 0 };

    fn finalists(winner_sse: f32, skip_sse: f32) -> Finalists {
        let finalist = |mv, sse| InterFinalist {
            mv,
            sse,
            rank_cost: sse,
        };
        Finalists {
            winner: finalist(WINNER, winner_sse),
            skip: Some(finalist(SKIP, skip_sse)),
            source_energy: 0.0,
            intra_rank_cost: winner_sse,
            intra_gate_open: true,
        }
    }

    fn decide_with(backend: &mut Priced, margin: f32, f: Finalists) -> DecidedMode {
        let picture = Picture::new(16, 16, 0).unwrap();
        let mut overrides = crate::params::ParamOverrides::default();
        overrides.set("decision.final_margin_lambdas", margin);
        let params = DecisionParams::from_overrides(&overrides);
        let block = Block::square(0, 0, 4);
        decide(
            backend,
            (&picture, &picture, Some(&picture)),
            &block,
            &params,
            f,
        )
        .mode
    }

    #[test]
    fn close_candidates_are_decided_by_coded_cost() {
        let mut backend = Priced {
            prices: vec![
                (FinalCandidate::Inter(WINNER), 900.0),
                (FinalCandidate::Inter(SKIP), 500.0),
            ],
            intra: None,
            calls: Cell::new(0),
        };
        // Screen: winner 100, skip 110, within 0.1 λ per sample (25.6).
        assert_eq!(
            decide_with(&mut backend, 0.1, finalists(100.0, 110.0)),
            DecidedMode::inter(SKIP)
        );
        assert_eq!(backend.calls.get(), 2);
    }

    #[test]
    fn a_clear_screen_winner_is_not_priced() {
        let mut backend = Priced {
            prices: vec![(FinalCandidate::Inter(SKIP), 0.0)],
            intra: Some((5000.0, 0.0)),
            calls: Cell::new(0),
        };
        assert_eq!(
            decide_with(&mut backend, 0.1, finalists(100.0, 1000.0)),
            DecidedMode::inter(WINNER)
        );
        assert_eq!(backend.calls.get(), 0);
    }

    #[test]
    fn a_lower_screen_alone_never_switches() {
        let mut backend = Priced {
            prices: vec![
                (FinalCandidate::Inter(WINNER), 100.0),
                (FinalCandidate::Inter(SKIP), 200.0),
            ],
            intra: None,
            calls: Cell::new(0),
        };
        // The skip vector screens far below the winner but costs more coded.
        assert_eq!(
            decide_with(&mut backend, 0.0, finalists(1000.0, 10.0)),
            DecidedMode::inter(WINNER)
        );
        assert_eq!(backend.calls.get(), 2);
    }

    #[test]
    fn intra_wins_by_coded_cost() {
        let mut backend = Priced {
            prices: vec![
                (FinalCandidate::Inter(WINNER), 900.0),
                (FinalCandidate::Intra, 300.0),
            ],
            intra: Some((101.0, 0.0)),
            calls: Cell::new(0),
        };
        let mut f = finalists(100.0, 0.0);
        f.skip = None;
        assert_eq!(decide_with(&mut backend, 0.1, f), DecidedMode::INTRA);
    }
}
