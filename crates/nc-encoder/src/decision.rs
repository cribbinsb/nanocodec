//! Per-block mode decision for inter frames.
//!
//! Follows SVT-AV1's real-time "light PD1" pass
//! (`docs/reference/svt-av1-m12-rtc.md` §3): the motion search vector is
//! refined to quarter-sample precision with SVT's fixed-stage search, then a
//! handful of candidates are ranked by the *variance* of their luma residual
//! (mean removed) plus λ·(signalling bits), and exactly one survivor goes on
//! to the transform. Candidates: the refined motion search result, the
//! vectors the codec can signal cheaply, the zero vector, and DC intra when
//! inter prediction is poor.
//!
//! Prediction here is luma only and approximate — whole samples copied,
//! sub-sample positions bilinear (libaom/SVT's `svf` kernels, nc-dsp) — and
//! identical for all three codecs: every one of them extends the reference
//! picture by replicating its edge samples. Codec-exact prediction (chroma,
//! the codec's interpolation filters) happens in the backend once the mode
//! is chosen.

use nc_analysis::Grid;
use nc_analysis::preanalysis::PreAnalysis;
use nc_analysis::still::StillMap;
use nc_common::{Picture, Plane, PlaneIndex, PlaneView, PlaneViewMut};
use nc_dsp::{BlockSize, Dsp, Filter, SUBPEL_VARIANCE_POSITIONS, Variance, WindowVariance};
use std::cell::{Cell, RefCell};

use crate::backend::{CodecBackend, PredictionMode};
use crate::final_stage::{self, Finalists, InterFinalist};
use crate::intra_estimate;
use crate::mv::{MV_UNITS_PER_SAMPLE, Mv};
use crate::params::DecisionParams;
use crate::partition::Block;
use crate::partition_plan::LeafStats;
use crate::psy::{ac_energy, psy_penalty};
use crate::quarter_predict::{QuarterPredict, predicted_step};
use crate::rd_refine::{RdWalk, rd_refine_vector};
use crate::refresh::RefreshWindow;
use crate::threshold_control::{Band, FrameEvidence, Tool, Turn};
use nc_analysis::linear_threshold::{
    LinearThreshold, block_size_feature, flag_feature, log_lambda_feature,
};

mod compound;
pub(crate) use compound::{choose_compound_mode, choose_pair_merge, choose_primary_pair};

// The decision's tunable thresholds are `DecisionParams` (`params.rs`); the
// notes below explain their hand-tuned defaults.
//
// `intra_gate_variance` (64): residual variance per luma sample above which
// DC intra is also tried. SVT skips intra in P frames unless motion search
// did badly (`skip_intra`, about 1.2 SAD per sample); a standard deviation
// of 8 is the equivalent per-block gate here.
// `skip_no_refine_lambdas` (0.25): residual variance, in λ per luma sample,
// below which the codec's skip vector ([`CodecBackend::skip_vector`])
// winning the candidate ranking is kept without sub-sample refinement.
// H.264 (crowd 720p, QP 36): 5.6M fewer instructions per frame and BD-rate
// -8.4% -> -8.7% vs openh264-baseline on the quick clips; 0.5 λ/sample
// breaks low-light (+24%), so 0.25.
// `subpel_move_bias`: a sub-sample step is taken only if it costs less than
// the current best divided by this. SVT's `bias_fp = 110` (moving off a
// position must win by 10%) is the legacy 1.1 that H.264, HEVC and VP9
// keep; the engine default is AV1's measured 1.0 (no margin).

/// The fixed-stage search: half-sample then quarter-sample crosses, in
/// quarter-sample units.
const SUBPEL_STEPS: [i16; 2] = [2, 1];

/// When the fixed-stage search runs its quarter-sample stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QuarterStage {
    /// After every half-sample stage.
    Always,
    /// Only around a half-sample winner
    /// (`decision.subpel_quarter_needs_half_move`).
    AfterHalfMove,
    /// Never: vectors stop at half samples (`decision.subpel_half_only`).
    Never,
}

impl QuarterStage {
    fn of(params: &DecisionParams) -> Self {
        if params.subpel_half_only >= 0.5 {
            Self::Never
        } else if params.subpel_quarter_needs_half_move >= 0.5 {
            Self::AfterHalfMove
        } else {
            Self::Always
        }
    }

    /// Whether the quarter stage is skipped after a half stage that left
    /// the vector at `centre`, refined from `start`.
    fn skips(self, centre: Mv, start: Mv) -> bool {
        match self {
            Self::Always => false,
            Self::AfterHalfMove => centre == start,
            Self::Never => true,
        }
    }
}
/// What the fixed-stage search predicts where it skips its quarter stage.
fn quarter_predict(params: &DecisionParams) -> QuarterPredict {
    QuarterPredict::of(
        params.subpel_quarter_predict,
        params.subpel_quarter_predict_strength,
    )
}

/// The adaptive switch of the walk's quarter stage (`threshold_control.rs`,
/// `adapt.quarter`) as one refinement sees it: asked at each opportunity
/// (a half stage that did not move), and told what a measured quarter stage
/// kept of the cost before it.
trait QuarterSwitch {
    fn turn(&mut self) -> Turn;
    fn record(&mut self, before: f32, kept: f32);
}

/// No switch: the quarter stage runs as the preset says.
struct PresetQuarter;

impl QuarterSwitch for PresetQuarter {
    fn turn(&mut self) -> Turn {
        Turn::Preset
    }

    fn record(&mut self, _before: f32, _kept: f32) {}
}

/// The frame's switch, in the decision's gathering instance only: without
/// `GATHER` it is [`PresetQuarter`], so that instance compiles as before.
struct FrameQuarter<'a, const GATHER: bool>(&'a mut FrameEvidence);

impl<const GATHER: bool> QuarterSwitch for FrameQuarter<'_, GATHER> {
    fn turn(&mut self) -> Turn {
        if GATHER {
            self.0.quarter_turn()
        } else {
            Turn::Preset
        }
    }

    fn record(&mut self, before: f32, kept: f32) {
        if GATHER {
            self.0.record_quarter(before, kept);
        }
    }
}

/// Area from which `decision.codec_subpel_seed_lambdas_large` applies (32x32).
const LARGE_BLOCK_AREA: f32 = 32.0 * 32.0;
/// Integer-probe order matches the decision-neutral omitted-candidate oracle.
const INTEGER_PROBE_CROSS: [(i16, i16); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const CROSS: [(i16, i16); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

/// The `(cost, distortion)` of candidate vectors during sub-sample
/// refinement: one vector at a time, or the four vectors of one refinement
/// stage (a cross around a fixed centre), which a measurer can evaluate in
/// one batched kernel call.
trait RefineCost {
    /// The cost and distortion of `mv`.
    fn cost(&mut self, mv: Mv) -> (f32, f32);

    /// [`Self::cost`] of each of `mvs`, in order.
    fn cost_four(&mut self, mvs: [Mv; 4]) -> [(f32, f32); 4] {
        mvs.map(|mv| self.cost(mv))
    }

    /// [`Self::cost_four`], except that a vector whose distortion alone
    /// cannot win (`distortion * bias >= best`) may cost `f32::INFINITY`:
    /// costs are distortion plus a rate that is never negative, so its cost
    /// times `bias` is at least `best` as well.
    fn cost_four_below(&mut self, mvs: [Mv; 4], (_best, _bias): (f32, f32)) -> [(f32, f32); 4] {
        self.cost_four(mvs)
    }

    /// [`Self::cost_four_below`] of one vector.
    fn cost_below(&mut self, mv: Mv, (_best, _bias): (f32, f32)) -> (f32, f32) {
        self.cost(mv)
    }
}

impl<F: FnMut(Mv) -> (f32, f32)> RefineCost for F {
    fn cost(&mut self, mv: Mv) -> (f32, f32) {
        self(mv)
    }
}

/// A [`RefineCost`] with a batched form for whole stages.
#[derive(Clone, Copy)]
struct BatchedCost<'a, One, Four> {
    one: &'a One,
    four: &'a Four,
}

impl<One, Four> RefineCost for BatchedCost<'_, One, Four>
where
    One: Fn(Mv) -> (f32, f32),
    Four: Fn([Mv; 4]) -> [(f32, f32); 4],
{
    fn cost(&mut self, mv: Mv) -> (f32, f32) {
        (self.one)(mv)
    }

    fn cost_four(&mut self, mvs: [Mv; 4]) -> [(f32, f32); 4] {
        (self.four)(mvs)
    }
}

/// The bilinear ranking's refinement cost: distortion plus `rate`, with the
/// four vectors of a stage measured in one batch and priced only where
/// their distortion alone can still win
/// ([`RefineCost::cost_four_below`]): a vector's mode bits are a backend
/// rate estimate per vector, and most of a stage's vectors already lose on
/// distortion.
struct RateAfterDistortion<'a, One, Four, Rate> {
    distortion: &'a One,
    distortion_four: &'a Four,
    rate: Rate,
}

impl<One, Four, Rate> RefineCost for RateAfterDistortion<'_, One, Four, Rate>
where
    One: Fn(Mv) -> f32,
    Four: Fn([Mv; 4]) -> [f32; 4],
    Rate: FnMut(Mv) -> f32,
{
    fn cost(&mut self, mv: Mv) -> (f32, f32) {
        let distortion = (self.distortion)(mv);
        (distortion + (self.rate)(mv), distortion)
    }

    fn cost_four(&mut self, mvs: [Mv; 4]) -> [(f32, f32); 4] {
        let distortions = (self.distortion_four)(mvs);
        [0, 1, 2, 3].map(|i| (distortions[i] + (self.rate)(mvs[i]), distortions[i]))
    }

    fn cost_four_below(&mut self, mvs: [Mv; 4], (best, bias): (f32, f32)) -> [(f32, f32); 4] {
        let distortions = (self.distortion_four)(mvs);
        [0, 1, 2, 3].map(|i| {
            let distortion = distortions[i];
            if distortion * bias >= best {
                (f32::INFINITY, distortion)
            } else {
                (distortion + (self.rate)(mvs[i]), distortion)
            }
        })
    }

    fn cost_below(&mut self, mv: Mv, (best, bias): (f32, f32)) -> (f32, f32) {
        let distortion = (self.distortion)(mv);
        if distortion * bias >= best {
            (f32::INFINITY, distortion)
        } else {
            (distortion + (self.rate)(mv), distortion)
        }
    }
}

/// The four vectors `step` units from `centre` in `directions`.
fn cross_around(centre: Mv, step: i16, directions: [(i16, i16); 4]) -> [Mv; 4] {
    directions.map(|(dx, dy)| Mv {
        x: centre.x + dx * step,
        y: centre.y + dy * step,
    })
}

/// Size of the pre-analysis statistics grid (8x8 luma blocks).
const PRE_ANALYSIS_BLOCK_LOG2: u8 = 3;
/// Whether every 8x8 block covered by `block` is static in pre-analysis
/// (unchanged since the previous *source* frame, noise-aware; see
/// `nc_analysis::preanalysis`). Blocks reaching outside the analysed picture
/// are never static.
#[inline]
pub(crate) fn is_static<B: CodecBackend>(pre: &PreAnalysis, block: &Block) -> bool {
    covers_all::<B, _>(&pre.blocks, block, |b| b.is_static)
}

/// Whether every 8x8 block covered by `block` is still (static for several
/// frames with a static neighbourhood; see `nc_analysis::still`). Blocks
/// reaching outside the analysed picture are never still.
#[inline]
pub(crate) fn is_still<B: CodecBackend>(still: &StillMap, block: &Block) -> bool {
    covers_all::<B, _>(&still.blocks, block, |&b| b)
}

/// Whether `test` holds for every cell of the 8x8 analysis `grid` that
/// `block` covers (false if `block` reaches outside the grid).
#[inline]
fn covers_all<B: CodecBackend, T>(
    grid: &Grid<T>,
    block: &Block,
    test: impl Fn(&T) -> bool,
) -> bool {
    let (width, height) = leaf_extent::<B>(block);
    let columns = (width >> PRE_ANALYSIS_BLOCK_LOG2).max(1);
    let rows = (height >> PRE_ANALYSIS_BLOCK_LOG2).max(1);
    let (col0, row0) = (
        block.x >> PRE_ANALYSIS_BLOCK_LOG2,
        block.y >> PRE_ANALYSIS_BLOCK_LOG2,
    );
    // Whole rows of the grid at a time (a per-cell `try_get` cost ~47
    // instructions); a row clipped by the grid edge fails.
    (0..rows).all(|r| {
        let run = grid.row_run(col0, row0 + r, columns);
        run.len() == columns && run.iter().all(&test)
    })
}

/// The mode mode decision found for one block on an inter frame: the
/// codec-facing [`PredictionMode`] and, for intra that won only through the
/// directional estimate, the inter vector it beat
/// ([`crate::BlockDecision::intra_fallback`]). Packed as a vector and a kind
/// (6 bytes, as `PredictionMode`) so the fallback costs the hot decision
/// path nothing with the estimate off.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DecidedMode {
    /// The inter vector for [`ModeKind::Inter`] and
    /// [`ModeKind::IntraOverInter`], zero for [`ModeKind::Intra`].
    mv: Mv,
    kind: ModeKind,
}

/// What [`DecidedMode`] holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModeKind {
    Inter,
    Intra,
    /// Intra that beat inter vector `mv` only through the open-loop
    /// directional estimate.
    IntraOverInter,
}

impl DecidedMode {
    /// Inter prediction at `mv`.
    pub(crate) const fn inter(mv: Mv) -> Self {
        Self {
            mv,
            kind: ModeKind::Inter,
        }
    }

    /// Intra prediction.
    pub(crate) const INTRA: Self = Self {
        mv: Mv::ZERO,
        kind: ModeKind::Intra,
    };

    /// Intra that won over inter vector `mv` only through the directional
    /// estimate.
    const fn intra_over(mv: Mv) -> Self {
        Self {
            mv,
            kind: ModeKind::IntraOverInter,
        }
    }

    /// The mode as the backend codes it.
    pub(crate) const fn prediction(self) -> PredictionMode {
        match self.kind {
            ModeKind::Inter => PredictionMode::Inter(self.mv),
            ModeKind::Intra | ModeKind::IntraOverInter => PredictionMode::IntraDc,
        }
    }

    /// Whether the mode is inter.
    pub(crate) fn is_inter(self) -> bool {
        self.kind == ModeKind::Inter
    }

    /// The inter vector an estimate-won intra mode beat, else `None`.
    pub(crate) fn intra_fallback(self) -> Option<Mv> {
        (self.kind == ModeKind::IntraOverInter).then_some(self.mv)
    }
}

/// The ranking's margins whole-CU pricing (`cu_pricing.rs`) gates on, per
/// leaf, in the ranking's units (not yet per sample or λ).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PricingHint {
    /// The inter winner's mean-removed prediction error (`n_variance`)
    /// where inter won the intra test; NaN otherwise.
    pub winner_error: f32,
    /// How far intra's cost was below inter's where intra won; NaN
    /// otherwise.
    pub intra_margin: f32,
    /// Where inter won: intra's estimated cost (DC, or the directional
    /// estimate where it ran) over inter's; NaN otherwise.
    pub intra_ratio: f32,
    /// Where inter won: its cost with the residual's mean error
    /// (distortion + λ · bits, as the intra test compares); NaN otherwise.
    pub inter_cost: f32,
}

impl PricingHint {
    /// Nothing to gate on: no pricing.
    pub(crate) const NONE: Self = Self {
        winner_error: f32::NAN,
        intra_margin: f32::NAN,
        intra_ratio: f32::NAN,
        inter_cost: f32::NAN,
    };
}

/// The engine's choice for one block on an inter frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Choice {
    pub mode: DecidedMode,
    /// The cost `mode` won with (distortion + λ · bits; for intra, the
    /// intra cost it beat inter with). `-inf` when a cached planner
    /// measurement was accepted without ranking: nothing need be compared.
    pub cost: f32,
    /// What whole-CU pricing (`cu_pricing.rs`) gates on.
    pub pricing: PricingHint,
}

/// Buffers mode decision reuses from block to block, owned by the encoder:
/// the candidate list, and the exact-filter prediction of one candidate
/// (before, a zero-initialised array per sub-sample candidate showed up as
/// memset in callgrind).
#[derive(Debug, Default)]
pub(crate) struct DecisionScratch {
    candidates: Vec<Mv>,
    /// Refinement-only seeds ([`CodecBackend::mv_seeds`]).
    seeds: Vec<Mv>,
    measure: MeasureScratch,
    /// Compound `InterRound1 = 7` `i16` prediction buffers reused across blocks.
    compound_a: Vec<i16>,
    compound_b: Vec<i16>,
    /// Two-hypothesis merge candidates (`CodecBackend::pair_merge_candidates`).
    pairs: Vec<(Mv, crate::backend::ReferenceChoice)>,
    /// Bilinear distortions already known for ranking candidates (NEW, the
    /// refined cheap vector, the cheap vectors), for `ExactPrune`.
    bilinear: Vec<(Mv, f32)>,
    /// What the gated tools achieved just inside their gates this frame
    /// (`threshold_control.rs`), read by the encoder after the frame.
    pub(crate) evidence: FrameEvidence,
}

/// What a leaf decision's measurers share ([`Measurer::with_scratch`]).
#[derive(Debug, Default)]
pub(crate) struct MeasureScratch {
    /// The exact-filter prediction of one candidate.
    prediction: RefCell<Vec<u8>>,
    /// Block predictions measured by every leaf decision so far: a
    /// deterministic count of the motion search's work, which the label dump
    /// (`labels.rs`) and a learned gate read as differences between two
    /// marks. One per vector measured (whole-sample, sub-sample or a compound
    /// half), whatever its kernel.
    positions: Cell<u64>,
}

impl MeasureScratch {
    /// Note `count` block predictions measured.
    #[inline]
    fn count_positions(&self, count: u64) {
        self.positions.set(self.positions.get() + count);
    }
}

impl DecisionScratch {
    /// The candidate list alone, for decisions that make no measurements.
    pub(crate) fn candidates_mut(&mut self) -> &mut Vec<Mv> {
        &mut self.candidates
    }

    /// Block predictions the leaf decisions have measured so far.
    #[inline]
    pub(crate) fn search_positions(&self) -> u64 {
        self.measure.positions.get()
    }
}

/// Choose the mode of an inter-frame block: rank the zero vector, the
/// vectors the codec signals cheaply and the whole-sample motion search
/// result by residual variance + λ · bits, refine only the winner to
/// quarter samples, and fall back to DC intra when even that predicts badly.
/// `candidates` is scratch space reused across blocks. `planned` holds the
/// residual statistics the partition planner already measured for this
/// block (at `search_mv` and the zero vector), if any.
/// `analysis` carries the pre-analysis where its moments cover the
/// source samples exactly, and the static verdict over the same extent.
pub(crate) fn choose_inter_mode<B: CodecBackend>(
    dsp: &Dsp,
    backend: &mut B,
    block: &Block,
    (source, reference, recon): (&Picture, &Picture, Option<&Picture>),
    search_mv: Mv,
    planned: Option<&LeafStats>,
    scratch: &mut DecisionScratch,
    analysis: (Option<&PreAnalysis>, bool),
    params: &DecisionParams,
    refresh: Option<&RefreshWindow>,
) -> Choice {
    // Two instances: while no controller gathers band evidence
    // (`threshold_control.rs`) the decision compiles without it, so
    // default encodes run exactly the code they ran before it existed.
    if scratch.evidence.band().is_some() {
        choose_inter_mode_with::<B, true>(
            dsp,
            backend,
            block,
            (source, reference, recon),
            search_mv,
            planned,
            scratch,
            analysis,
            params,
            refresh,
        )
    } else {
        choose_inter_mode_with::<B, false>(
            dsp,
            backend,
            block,
            (source, reference, recon),
            search_mv,
            planned,
            scratch,
            analysis,
            params,
            refresh,
        )
    }
}

/// [`choose_inter_mode`]; with `GATHER` it also records the gated tools'
/// band evidence in `scratch`.
#[allow(clippy::too_many_lines)]
fn choose_inter_mode_with<B: CodecBackend, const GATHER: bool>(
    dsp: &Dsp,
    backend: &mut B,
    block: &Block,
    (source, reference, recon): (&Picture, &Picture, Option<&Picture>),
    search_mv: Mv,
    planned: Option<&LeafStats>,
    scratch: &mut DecisionScratch,
    analysis: (Option<&PreAnalysis>, bool),
    params: &DecisionParams,
    refresh: Option<&RefreshWindow>,
) -> Choice {
    let DecisionScratch {
        candidates,
        seeds,
        measure,
        bilinear,
        evidence,
        ..
    } = scratch;
    candidates.clear();
    seeds.clear();
    bilinear.clear();
    // Unchanged since the previous source frame (noise-aware pre-analysis):
    // its true motion is zero, so a sub-sample walk only fits noise.
    let (source_moments, static_block) = analysis;
    candidates.push(Mv::ZERO);
    let skip_vector = backend.ranking_vectors(block, candidates);
    backend.mv_seeds(block, seeds);
    // `decision.max_cheap_vectors` (default: all) keeps the codec's first
    // few predicted vectors: each costs a measurement, a rate estimate and,
    // at a sub-sample position, an exact prediction. A vector left out can
    // still be coded, as NEW. Measured effect at the parameter.
    candidates.truncate(1 + params.max_cheap_vectors as usize);
    // Zero and the codec's cheap (predicted) vectors; the rest are NEW.
    let cheap = candidates.len();
    candidates.push(search_mv);
    let lambda = backend.lambda() * params.lambda_scale;
    if let (Some(&mv), Some(stats)) = (candidates.get(1), planned) {
        let cached = if mv == search_mv {
            Some(stats.search)
        } else if mv == Mv::ZERO {
            Some(stats.zero)
        } else {
            None
        };
        if mv.is_full_pel()
            && refresh.is_none_or(|w| w.allows(block, mv))
            && cached.is_some_and(|stats| backend.prefer_cached_inter(block, mv, stats, lambda))
        {
            return Choice {
                mode: DecidedMode::inter(mv),
                cost: f32::NEG_INFINITY,
                pricing: PricingHint::NONE,
            };
        }
    }
    let measurer = Measurer::new::<B>(dsp, source, reference, block)
        .with_refresh(refresh)
        .with_scratch(measure)
        .with_prediction_energy(params.psy_strength > 0.0)
        .with_row_step(params.distortion_row_step)
        .with_source_moments(source_moments);
    // The planner already measured the block at the search vector and at
    // zero; every other vector is measured here. (Caching every measured
    // vector was tried: revisits are too rare to pay for the lookups.)
    let known_stats = |mv: Mv| match planned {
        _ if !measurer.allows(mv) => Some(FORBIDDEN_STATS),
        Some(p) if mv == search_mv => Some(p.search),
        Some(p) if mv == Mv::ZERO => Some(p.zero),
        _ => None,
    };
    let stats_of = |mv: Mv| known_stats(mv).unwrap_or_else(|| measurer.measure(mv));
    // `stats_of` for the four vectors of one refinement stage, measured in
    // one batch.
    let stats_of_four = |mvs: [Mv; 4]| {
        let mut stats = measurer.measure_four(mvs);
        for (stats, mv) in stats.iter_mut().zip(mvs) {
            if let Some(p) = planned.filter(|_| measurer.allows(mv)) {
                if mv == search_mv {
                    *stats = p.search;
                } else if mv == Mv::ZERO {
                    *stats = p.zero;
                }
            }
        }
        stats
    };

    if let Some(filter) = backend.luma_filter() {
        // SVT light-PD1 (`docs/reference/svt-av1-m12-rtc.md` §3): NEW is the
        // search vector refined on its own (bilinear, SVT's `svf`), then
        // zero, the codec's cheap vectors and NEW are ranked with the
        // codec's own interpolation filter. Ranking every candidate by
        // bilinear variance (the path below) mis-ranks sub-sample vectors on
        // clean content. Quick clips, 10 s IP, PSNR-YUV BD vs svtav1-m12, AV1:
        // −8.9 points overall (bodycam −36, low-light −4, dashcam −3, static
        // −2); with bilinear ranking the same flow was +5.8 points worse than
        // the path below. Cost: +7.8 M instructions/frame (crowd 720p).
        // NEW is refined on distortion alone, as SVT's `svf` search is: with
        // λ·bits in the cost, moving off a search vector that equals NEAREST
        // was charged a whole motion-vector difference and rarely happened
        // (bodycam: NEW on 4% of the area against SVT's 15%). Rate enters in
        // the ranking below. Fast check vs svtav1-m12 (quick clips, 10 s IP,
        // PSNR-YUV): AV1 +35.0% → +33.6% (bodycam −4.7, dashcam −2.3 points),
        // HEVC −4.9% → −5.3%.
        // Refinement uses no rate term. Do not ask the backend to price a
        // vector here only to discard that work afterward.
        let distortion = |stats: Variance| {
            let distortion = n_variance::<B>(stats, block) as f32;
            (distortion, distortion)
        };
        let distortion_of = |mv: Mv| distortion(stats_of(mv));
        let distortion_of_four = |mvs: [Mv; 4]| stats_of_four(mvs).map(distortion);
        let refine_cost = BatchedCost {
            one: &distortion_of,
            four: &distortion_of_four,
        };
        // Kept for the ranking, which measures the search vector (a NEW
        // candidate) again otherwise: without planner statistics (H.264)
        // that was a second variance pass per block. With the psy term on,
        // the same pass also gives the reference block's AC energy.
        let (search_stats, search_energy) = match known_stats(search_mv) {
            Some(stats) => (stats, None),
            None if search_mv.is_full_pel() => measurer.measure_with_energy(search_mv),
            None => (measurer.measure(search_mv), None),
        };
        let start = distortion(search_stats);
        let area = leaf_area::<B>(block) as f32;
        let integer = integer_probe_worth_trying(start.0, lambda, area, params)
            .then(|| best_integer_neighbor::<B>(search_mv, start.0, block, stats_of))
            .flatten();
        // The best cheap vector (NEAREST/NEAR, often inherited at integer
        // positions) is measured before either seed is refined, so that
        // `decision.seed_refine_ratio` can compare the two seeds.
        let best_cheap = (params.refine_cheap_vector >= 0.5)
            .then(|| {
                best_cheap_vector(
                    &candidates[1..cheap],
                    seeds,
                    search_mv,
                    bilinear,
                    distortion_of,
                )
            })
            .flatten();
        let seed_gate = SeedGate {
            ratio: params.seed_refine_ratio,
        };
        let seed_refined = seed_gate.refines(start.0, best_cheap.map(|(d, _)| d));
        let (_, new_distortion, new) = if seed_refined {
            refine_codec_seed(
                (start.0, start.1, search_mv),
                lambda,
                area,
                params,
                refine_cost,
                FrameQuarter::<GATHER>(&mut *evidence),
            )
        } else {
            (start.0, start.1, search_mv)
        };
        bilinear.push((new, new_distortion));
        candidates.push(new);
        if let Some((mv, _)) = integer {
            candidates.push(mv);
        }
        // What the gated tools found where their gates were close
        // (`threshold_control.rs`), noted before the ranking while a
        // controller gathers it: a refinement or probe wins when the vector
        // only it found wins the ranking.
        let gathers_gates = evidence.gathers(Tool::Subpel) || evidence.gathers(Tool::Probe);
        let gates = (GATHER && gathers_gates).then(|| {
            GateOutcomes::of(
                (seed_refined, new, search_mv, integer.map(|(mv, _)| mv)),
                (start, lambda, area),
                params,
            )
        });
        // The NEW seed is the motion-search vector; the cheap vectors are
        // otherwise ranked only where they stand. Refining the best of them
        // yields a NEW candidate with a small vector difference. Work: one
        // sub-sample refinement (at most 8 more evaluations) per block, and
        // one more exact ranking candidate when it moves. Skipped (as for
        // the NEW seed) when the cheap vector already predicts within
        // `refine_cheap_min_lambdas` λ per sample: a sub-sample move cannot
        // repay its vector bits there.
        let worth = |distortion: f32| {
            distortion >= params.refine_cheap_min_lambdas * lambda * area
                && seed_gate.refines(distortion, Some(start.0))
        };
        let best = best_cheap.map(|(distortion, mv)| {
            if worth(distortion) {
                let (_, refined_distortion, refined) = refine_codec_seed(
                    (distortion, distortion, mv),
                    lambda,
                    area,
                    params,
                    refine_cost,
                    FrameQuarter::<GATHER>(&mut *evidence),
                );
                bilinear.push((refined, refined_distortion));
                refined
            } else {
                mv
            }
        });
        if let Some(best) = best.filter(|&mv| !holds(candidates, mv)) {
            candidates.push(best);
        }
        // The optional probe already measured this integer predictor exactly.
        let known_stats = |mv| match integer {
            Some((key, stats)) if mv == key => Some(stats),
            _ if mv == search_mv => Some(search_stats),
            _ => known_stats(mv),
        };
        let stats_of = |mv| known_stats(mv).unwrap_or_else(|| measurer.measure(mv));
        // Whole-sample vectors the ranking measures itself also take the
        // reference block's AC energy from that pass where the psy term
        // will need it.
        let stats_and_energy = |mv| {
            if mv == search_mv {
                return (search_stats, search_energy);
            }
            known_stats(mv).map_or_else(|| measurer.measure_with_energy(mv), |stats| (stats, None))
        };
        let new_rate_scale = params.new_rate_scale;
        let rank_band = evidence
            .band()
            .filter(|_| GATHER && evidence.gathers(Tool::Rank))
            .map(RankBand::new);
        let prune = ExactPrune {
            margin: params.exact_rank_margin,
            known: bilinear,
            band: rank_band.as_ref(),
        };
        let psy = Psy::of(params, &measurer);
        let ((inter_cost, inter_distortion, inter_mv), inter_stats) = if B::CACHE_EXACT_LUMA {
            rank_cached_exact::<B, GATHER>(
                backend,
                block,
                (&measurer, psy, params.rank_dc_weight),
                candidates,
                (cheap, filter, lambda, new_rate_scale),
                &prune,
                stats_and_energy,
            )
        } else {
            let mode_bits = backend.inter_mode_pricer(block);
            let mut best_cost: Option<f32> = None;
            rank_with_stats(candidates, |i, mv| {
                let bits = mode_bits(backend, block, mv);
                let scale = if i < cheap { 1.0 } else { new_rate_scale };
                let rate = lambda * scale * bits;
                // Isolated legal-rate P probes: HEVC/AV1 −1.2%, parity identical
                // (zero BD-rate change).
                // Integer prediction is identical under every codec filter. Reuse
                // the planner's search/zero statistics instead of measuring those
                // same samples again during exact candidate ranking.
                let exact = !mv.is_full_pel();
                if exact && prune.skips(mv, rate, best_cost) {
                    return PRUNED;
                }
                let mut score = |stats, pixels: Option<&[u8]>| {
                    let distortion = ranking_distortion::<B>(stats, block, params.rank_dc_weight);
                    let cost = distortion + rate;
                    if exact {
                        prune.observe(mv, rate, best_cost, cost);
                    }
                    if best_cost.is_none_or(|best| cost < best) {
                        best_cost = Some(cost);
                        if let Some(pixels) = pixels {
                            backend.cache_exact_luma(block, mv, filter, pixels, stats);
                        }
                    }
                    (cost, distortion, stats)
                };
                if !exact {
                    score(stats_of(mv), None)
                } else if B::CACHE_UNWEIGHTED_LUMA && measurer.allows(mv) {
                    measurer.with_exact_prediction(mv, filter, |stats, pixels, _| {
                        score(stats, Some(pixels))
                    })
                } else {
                    score(measurer.measure_exact(mv, filter), None)
                }
            })
        };
        // Neither the codec's skip vector (`decision.winner_walk_keeps_skip`)
        // nor a static block is walked: on a still camera, moving them to a
        // sub-sample neighbour blurs the scene over the GOP (skip vector:
        // pool_chad_3_086, fixed QP, J +7.2 % walked, -0.3 % kept; fixed QP,
        // draw fixedqp-2026-11-20, walked: H.264 Default +0.2 %, HEVC Slow
        // +0.2 %; static blocks: the long windows of final-check,
        // docs/handoff/rd-depth.md). The AV1 family walks it (its codecs'
        // `DEFAULT_PARAMS`).
        let walk = winner_walk(params).filter(|_| {
            (params.winner_walk_keeps_skip < 0.5 || Some(inter_mv) != skip_vector)
                && !static_block
                && inter_distortion
                    >= params.winner_walk_min_lambdas * lambda * leaf_area::<B>(block) as f32
        });
        let ((inter_cost, inter_distortion, inter_mv), inter_stats) = match walk {
            Some(walk) => walk_winner(
                backend,
                block,
                (&measurer, filter, params.rank_dc_weight),
                (lambda * new_rate_scale, walk),
                ((inter_cost, inter_distortion, inter_mv), inter_stats),
            ),
            None => ((inter_cost, inter_distortion, inter_mv), inter_stats),
        };
        if let Some(gates) = gates {
            gates.record(inter_mv, evidence);
        }
        if let Some(rank_band) = &rank_band {
            rank_band.report(inter_mv, evidence);
        }
        let inter = InterChoice {
            cost: inter_cost,
            distortion: inter_distortion,
            dc: dc_error::<B>(params, inter_stats, block),
            mv: inter_mv,
        };
        if !final_stage_on(backend, params) {
            return intra_or_inter(backend, (&measurer, source), block, lambda, params, inter);
        }
        let finish = Finish {
            pictures: (source, reference, recon),
            skip_vector,
            filter: Some(filter),
            winner_sse: inter_stats.sse as f32,
        };
        return finish.decide(backend, &measurer, block, lambda, params, inter);
    }

    let distortion_of = |mv: Mv| n_variance::<B>(stats_of(mv), block) as f32;
    let distortion_of_four =
        |mvs: [Mv; 4]| stats_of_four(mvs).map(|stats| n_variance::<B>(stats, block) as f32);
    let mode_bits = backend.inter_mode_pricer(block);
    let mut refine_cost = RateAfterDistortion {
        distortion: &distortion_of,
        distortion_four: &distortion_of_four,
        rate: |mv: Mv| lambda * mode_bits(backend, block, mv),
    };
    let best = rank(candidates, |mv| refine_cost.cost(mv));
    // Only the winner is refined: refining every candidate (or the search
    // vector before ranking) costs up to 8 more evaluations each.
    let area = leaf_area::<B>(block) as f32;
    // The codec's skip vector winning with a negligible residual is final:
    // refining it would cost up to 8 evaluations to move a block that is
    // about to be coded as skip anyway.
    let negligible = effective_skip_no_refine(lambda, params) * lambda * area;
    // The winner is not refined below the codec's variance gate
    // (`CodecBackend::subpel_skip_variance`) nor below the engine's seed gate
    // (`decision.codec_subpel_seed_lambdas`, the exact-filter path's gate on
    // its NEW seed), whichever is higher. What it skips: up to 8 bilinear
    // evaluations on a winner whose residual is already a small fraction of
    // λ per sample, where a sub-sample move rarely repays its vector bits.
    // VP9 Veryfast, four quick clips, 60 f, CBR 600: refinement ran on every
    // block and moved 11 % of them. At 0.05 (VP9 Veryfast): Jetson cycles
    // -3.0 %, x86 instructions -3.55 %; pool L3 J -1.30 % at k 0.6, quality
    // +0.85 %; final-check quality +0.48 %, no guard
    // (docs/handoff/fast-decision.md).
    let skip_below = backend
        .subpel_skip_variance()
        .max(effective_subpel_seed_gate(area, lambda, best.2, params) * lambda)
        * area
        * params.winner_refine_gate_scale;
    let exact_k = (params.exact_finalists.round() as usize).min(MAX_EXACT_FINALISTS);
    let exact_filter = (exact_k > 0).then(|| backend.exact_luma_filter()).flatten();
    let mut exact_winner_stats: Option<Variance> = None;
    let (inter_cost, inter_distortion, inter_mv) =
        if Some(best.2) == skip_vector && best.1 < negligible {
            best
        } else if let Some(filter) = exact_filter {
            let (refined, finalists, searched) = refine_subpel_finalists(
                best,
                skip_below,
                (
                    params.subpel_move_bias,
                    params.subpel_move_bias_nonzero,
                    params.subpel_fractional_bias,
                ),
                (QuarterStage::of(params), quarter_predict(params)),
                exact_k,
                refine_cost,
            );
            if GATHER && evidence.gathers(Tool::Refine) {
                evidence.record_saving_above(Tool::Refine, best.1, skip_below, (best.0, refined.0));
            }
            if searched {
                let (reranked, stats) = rerank_exact_finalists::<B>(
                    backend,
                    &measurer,
                    block,
                    filter,
                    lambda,
                    refined.0,
                    finalists.entries(),
                    &stats_of,
                );
                exact_winner_stats = Some(stats);
                reranked
            } else {
                refined
            }
        } else {
            let refined = refine_subpel(
                best,
                skip_below,
                (
                    params.subpel_move_bias,
                    params.subpel_move_bias_nonzero,
                    params.subpel_fractional_bias,
                ),
                (QuarterStage::of(params), quarter_predict(params)),
                refine_cost,
            );
            // What refining a winner just above the gate saved
            // (`threshold_control.rs`, `adapt.refine`).
            if GATHER && evidence.gathers(Tool::Refine) {
                evidence.record_saving_above(Tool::Refine, best.1, skip_below, (best.0, refined.0));
            }
            refined
        };

    // Only the DC-aware intra decision needs the winner's residual mean:
    // the planner's vectors are cached, others are measured once more.
    let dc = if params.intra_test_dc >= 0.5 {
        let stats = exact_winner_stats.unwrap_or_else(|| stats_of(inter_mv));
        dc_error::<B>(params, stats, block)
    } else {
        0.0
    };
    let inter = InterChoice {
        cost: inter_cost,
        distortion: inter_distortion,
        dc,
        mv: inter_mv,
    };
    if !final_stage_on(backend, params) {
        return intra_or_inter(backend, (&measurer, source), block, lambda, params, inter);
    }
    // The final stage needs the winner's prediction error (the planner's
    // vectors are cached).
    let finish = Finish {
        pictures: (source, reference, recon),
        skip_vector,
        filter: exact_filter.filter(|_| exact_winner_stats.is_some()),
        winner_sse: exact_winner_stats.unwrap_or_else(|| stats_of(inter_mv)).sse as f32,
    };
    finish.decide(backend, &measurer, block, lambda, params, inter)
}

/// Whether the final decision stage decides this leaf: asked for, and the
/// codec can price its candidates (a codec without a pricing keeps the
/// ranking's winner or DC intra, as with the stage off).
fn final_stage_on<B: CodecBackend>(backend: &B, params: &DecisionParams) -> bool {
    params.final_stage >= 0.5 && backend.prices_final_candidates()
}

/// What the end of the inter decision needs besides the ranking's winner:
/// the pictures, the codec's skip vector, the exact filter (if the codec
/// ranks with one) and the winner's prediction error.
struct Finish<'p> {
    pictures: (&'p Picture, &'p Picture, Option<&'p Picture>),
    skip_vector: Option<Mv>,
    filter: Option<Filter>,
    winner_sse: f32,
}

impl Finish<'_> {
    /// The leaf's choice under `decision.final_stage`: the final stage
    /// (`final_stage.rs`), or the ranking's winner or DC intra
    /// ([`intra_or_inter`]) where the winner's error is past
    /// `decision.final_max_winner_lambdas`. Out of line, so the decision
    /// with the stage off runs the code it ran before the stage existed.
    #[inline(never)]
    fn decide<B: CodecBackend>(
        self,
        backend: &mut B,
        measurer: &Measurer<'_>,
        block: &Block,
        lambda: f32,
        params: &DecisionParams,
        inter: InterChoice,
    ) -> Choice {
        let (source, reference, recon) = self.pictures;
        let ranked = intra_or_inter(backend, (measurer, source), block, lambda, params, inter);
        // The stage settles near-ties of flat, well-predicted blocks; where
        // the winner's error is large the residual dominates whatever is
        // chosen, and pricing it (many coefficients) costs most.
        let max_winner =
            params.final_max_winner_lambdas * backend.lambda() * leaf_area::<B>(block) as f32;
        // Intra won through the directional estimate (`decision.intra_directional`),
        // which the stage's whole-block intra screen cannot price: it
        // stands, and the backend's own direction search confirms it.
        // Without the stage's intra candidate (`decision.final_intra_candidate`
        // 0) intra stays the ranking's decision.
        let ranked_intra = !ranked.mode.is_inter() && params.final_intra_candidate < 0.5;
        if (max_winner > 0.0 && self.winner_sse > max_winner)
            || ranked.mode.intra_fallback().is_some()
            || ranked_intra
        {
            return ranked;
        }
        let skip = self
            .skip_vector
            .filter(|&mv| {
                params.final_skip_candidate >= 0.5 && mv != inter.mv && measurer.allows(mv)
            })
            .map(|mv| {
                let stats = match self.filter {
                    Some(filter) if !mv.is_full_pel() => measurer.measure_exact(mv, filter),
                    _ => measurer.measure(mv),
                };
                let bits = backend.mode_bits(block, &PredictionMode::Inter(mv));
                InterFinalist {
                    mv,
                    sse: stats.sse as f32,
                    rank_cost: ranking_distortion::<B>(stats, block, params.rank_dc_weight)
                        + lambda * bits,
                }
            });
        let finalists = Finalists {
            winner: InterFinalist {
                mv: inter.mv,
                sse: self.winner_sse,
                rank_cost: inter.cost,
            },
            skip,
            source_energy: measurer.source_energy() as f32,
            intra_rank_cost: inter.cost + inter.dc,
            // Below the gate the inter residual is too small for intra to
            // pay (and the screen to be worth its predictions).
            intra_gate_open: params.final_intra_candidate >= 0.5
                && inter.distortion + inter.dc
                    > final_intra_gate(params, lambda) * leaf_area::<B>(block) as f32,
        };
        let choice = final_stage::decide(
            backend,
            (source, reference, recon),
            block,
            params,
            finalists,
        );
        // Where the stage agrees with the ranking, whole-CU pricing
        // (`cu_pricing.rs`) keeps the ranking's margins to gate on.
        if choice.mode == ranked.mode {
            Choice {
                pricing: ranked.pricing,
                ..choice
            }
        } else {
            choice
        }
    }
}

/// The best inter mode of `block` predicted from a second reference
/// (`second_reference.rs`), ranked cheaply: the zero vector, the codec's
/// predicted vectors for that reference and `seed`, each measured once at
/// distortion + λ · bits (+ the psy term), with the codec's filter where it
/// has one, and no intra trial. The winner is refined to sub-sample
/// precision only when it comes within `refine_margin` of `primary_cost`
/// (the cost to beat), so refinement is spent on blocks the second
/// reference may win. The cost terms are the primary ranking's, so the two
/// costs compare directly. `source_moments` (as for the primary) give the
/// source block's energy without another pass.
///
/// The full decision against the second reference (refinement included)
/// cost +23% H.264 P-frame instructions for `J_real` −1.2% (5 TRAIN clips,
/// CBR; `docs/handoff/engine.md` "Second reference").
pub(crate) fn choose_secondary_mode<B: CodecBackend>(
    dsp: &Dsp,
    backend: &mut B,
    block: &Block,
    source: &Picture,
    reference: &Picture,
    seed: Mv,
    (primary_cost, refine_margin, include_dc): (f32, f32, bool),
    scratch: &mut DecisionScratch,
    (params, source_moments): (&DecisionParams, Option<&PreAnalysis>),
    refresh: Option<&RefreshWindow>,
) -> Option<Choice> {
    let DecisionScratch {
        candidates,
        measure,
        ..
    } = scratch;
    candidates.clear();
    candidates.push(Mv::ZERO);
    backend.mv_candidates(block, candidates);
    candidates.truncate(1 + params.max_cheap_vectors as usize);
    let cheap = candidates.len();
    if !holds(candidates, seed) {
        candidates.push(seed);
    }
    let lambda = backend.lambda() * params.lambda_scale;
    let measurer = Measurer::new::<B>(dsp, source, reference, block)
        .with_refresh(refresh)
        .with_scratch(measure)
        .with_prediction_energy(params.psy_strength > 0.0)
        .with_row_step(params.distortion_row_step)
        .with_source_moments(source_moments);
    let psy = Psy::of(params, &measurer);
    let ((mut cost, _, mut mv), mut winner_stats) = match backend.luma_filter() {
        Some(filter) => {
            // No pruning, so no margin evidence either.
            let prune = ExactPrune {
                margin: 0.0,
                known: &[],
                band: None,
            };
            rank_cached_exact::<B, false>(
                backend,
                block,
                (&measurer, psy, params.rank_dc_weight),
                candidates,
                (cheap, filter, lambda, params.new_rate_scale),
                &prune,
                |mv| measurer.measure_with_energy(mv),
            )
        }
        None => rank_with_stats(candidates, |_, mv| {
            let stats = measurer.measure(mv);
            let distortion = n_variance::<B>(stats, block) as f32;
            let penalty = psy.map_or(0.0, |psy| psy.penalty(measurer.prediction_energy(mv, None)));
            let rate = lambda * backend.mode_bits(block, &PredictionMode::Inter(mv));
            (distortion + rate + penalty, distortion, stats)
        }),
    };
    if cost.is_finite() && cost < primary_cost * (1.0 + refine_margin) {
        // Refinement on distortion alone, as for the primary's NEW seed;
        // the refined vector is then costed like the candidates.
        let distortion = |stats: Variance| {
            let distortion = n_variance::<B>(stats, block) as f32;
            (distortion, distortion)
        };
        let one = |mv: Mv| distortion(measurer.measure(mv));
        let four = |mvs: [Mv; 4]| measurer.measure_four(mvs).map(distortion);
        let start = one(mv);
        let (_, _, refined) = refine_codec_subpel(
            (start.0, start.1, mv),
            params,
            BatchedCost {
                one: &one,
                four: &four,
            },
            PresetQuarter,
        );
        if refined != mv {
            let stats = match backend.luma_filter() {
                Some(filter) => measurer.measure_exact(refined, filter),
                None => measurer.measure(refined),
            };
            let penalty = psy.map_or(0.0, |psy| {
                psy.penalty(measurer.prediction_energy(refined, None))
            });
            let refined_cost = n_variance::<B>(stats, block) as f32
                + lambda
                    * params.new_rate_scale
                    * backend.mode_bits(block, &PredictionMode::Inter(refined))
                + penalty;
            if refined_cost < cost {
                (cost, mv, winner_stats) = (refined_cost, refined, stats);
            }
        }
    }
    if include_dc && cost.is_finite() {
        cost += dc_error::<B>(params, winner_stats, block);
    }
    cost.is_finite().then_some(Choice {
        mode: DecidedMode::inter(mv),
        cost,
        pricing: PricingHint::NONE,
    })
}

/// Optional work/quality tradeoff, using already measured seed variance.
/// A saved13f TRAIN Antare oracle places the90%-retained-RD-gain quantile at
/// 0.009413; fixed0.0094 skips56.02% probe area in that diagnostic. This is
/// an empirical candidate gate, not a proof that other predictors cannot win.
fn integer_probe_worth_trying(
    seed_variance: f32,
    lambda: f32,
    area: f32,
    params: &DecisionParams,
) -> bool {
    params.codec_fullpel_probe >= 0.5
        && (params.codec_fullpel_probe_min_lambdas <= 0.0
            || seed_variance >= lambda * area * params.codec_fullpel_probe_min_lambdas)
}

/// The source ME seed can miss an integer predictor on the coded reference.
/// A decision-neutral13f TRAIN oracle found omitted integer winners on11.53%
/// of Antare area (0.986% summed inter-RD potential, not a bitrate gain).
/// Keep the original candidate pool/refinement and try only one extra survivor.
fn best_integer_neighbor<B: CodecBackend>(
    start: Mv,
    start_distortion: f32,
    block: &Block,
    mut stats_of: impl FnMut(Mv) -> Variance,
) -> Option<(Mv, Variance)> {
    // Variance is nonnegative. Strict ties keep the existing predictor, so
    // a zero-variance seed needs no probes (common on flat/static blocks).
    if start_distortion == 0.0 {
        return None;
    }
    let mut best = None;
    let mut distortion = start_distortion;
    for (dx, dy) in INTEGER_PROBE_CROSS {
        // Saturation could turn a full-pel coordinate into a fractional one.
        let (Some(x), Some(y)) = (
            start.x.checked_add(dx * MV_UNITS_PER_SAMPLE as i16),
            start.y.checked_add(dy * MV_UNITS_PER_SAMPLE as i16),
        ) else {
            continue;
        };
        let mv = Mv { x, y };
        let stats = stats_of(mv);
        let candidate = n_variance::<B>(stats, block) as f32;
        if candidate < distortion {
            distortion = candidate;
            best = Some((mv, stats));
            if candidate == 0.0 {
                return best;
            }
        }
    }
    best
}

/// Optional approximation, not a literal SVT threshold. Pinned SVT light-PD1
/// omits subpel for small/intra-bordered blocks (`product_coding_loop.c:2721`);
/// its VLPD0 variance model (:8241) compares residual variance/sample to lambda.
/// Here a cached ME seed below a tunable noise floor bypasses only refinement:
/// codec-filter ranking and intra/skip decisions still run normally.
/// At 0.05, five paired native runs reduced AV1 CPU 5.04%/3.86% and HEVC
/// 6.09%/2.09% (bodycam/crowd). Four-natural TRAIN full-frame VMAF v1
/// BD-rate changed +1.24% AV1/-0.09% HEVC; PSNR +1.00%/+0.92%.
/// The default stays disabled: this trades motion accuracy for speed.
/// Evidence: docs/handoff/overnight.md and its frozen seed-study paths.
fn refine_codec_seed(
    start: (f32, f32, Mv),
    lambda: f32,
    area: f32,
    params: &DecisionParams,
    cost: impl RefineCost,
    quarter: impl QuarterSwitch,
) -> (f32, f32, Mv) {
    let gate = effective_subpel_seed_gate(area, lambda, start.2, params);
    if gate > 0.0 && start.1 < gate * lambda * area {
        start
    } else {
        refine_codec_subpel(start, params, cost, quarter)
    }
}

/// The seed gate of a block of `area` samples, in λ of residual per sample:
/// large blocks may take a stricter one.
fn seed_gate_threshold(area: f32, params: &DecisionParams) -> f32 {
    if area >= LARGE_BLOCK_AREA {
        params
            .codec_subpel_seed_lambdas
            .max(params.codec_subpel_seed_lambdas_large)
    } else {
        params.codec_subpel_seed_lambdas
    }
}

/// `seed_gate_threshold` plus its linear shift (`decision.subpel_seed_shift_*`,
/// `linear_threshold.rs`) at the block's λ, size and whether the seed is the
/// zero vector, clamped at 0. The frame's constant features are already in
/// the bias (`DecisionParams::with_frame_shift_features`). With every term 0
/// it is the fixed gate.
fn effective_subpel_seed_gate(area: f32, lambda: f32, mv: Mv, params: &DecisionParams) -> f32 {
    let base = seed_gate_threshold(area, params);
    let Some(shift) = LinearThreshold::new(
        params.subpel_seed_shift_bias,
        [
            params.subpel_seed_shift_lambda,
            params.subpel_seed_shift_size,
            params.subpel_seed_shift_zero_mv,
        ],
    ) else {
        return base;
    };
    let features = [
        log_lambda_feature(lambda),
        block_size_feature(area),
        flag_feature(mv == Mv::ZERO),
    ];
    (base + shift.at(&features)).max(0.0)
}

/// `skip_no_refine_lambdas` plus its linear shift
/// (`decision.skip_no_refine_shift_*`, `linear_threshold.rs`) at the block's
/// λ, clamped at 0; the frame's constant features are already in the bias.
/// With every term 0 it is the fixed gate.
fn effective_skip_no_refine(lambda: f32, params: &DecisionParams) -> f32 {
    let base = params.skip_no_refine_lambdas;
    let Some(shift) = LinearThreshold::new(
        params.skip_no_refine_shift_bias,
        [params.skip_no_refine_shift_lambda],
    ) else {
        return base;
    };
    (base + shift.at(&[log_lambda_feature(lambda)])).max(0.0)
}

/// SVT's quarter-pel loop walks around the updated winner, rather than
/// evaluating one fixed cross. Its first-move bias is relative to the ME
/// starting vector, not to absolute zero, and disappears after the first
/// accepted move (`md_subpel_search_fixed_stage`, pinned SVT4.3.0).
fn refine_codec_subpel(
    start: (f32, f32, Mv),
    params: &DecisionParams,
    mut cost: impl RefineCost,
    mut quarter: impl QuarterSwitch,
) -> (f32, f32, Mv) {
    if params.subpel_walk < 0.5 {
        return refine_subpel(
            start,
            0.0,
            (
                params.subpel_move_bias,
                params.subpel_move_bias_nonzero,
                params.subpel_fractional_bias,
            ),
            (QuarterStage::of(params), quarter_predict(params)),
            cost,
        );
    }
    let svt = params.subpel_walk >= 1.5;
    let directions = if svt {
        [(1, 0), (-1, 0), (0, 1), (0, -1)]
    } else {
        CROSS
    };
    let bias_percent = (params.subpel_move_bias * 100.0).round() as u64;
    let mut best = start;
    // A coordinate step may return to its previous centre. Reuse exact
    // distortion pairs rather than running motion compensation a second time.
    // Two legal TRAIN probes: -0.22% P instructions; exact synthetic streams.
    let mut previous_best = start;
    // The half stage's costs in `CROSS` order, and a measured quarter
    // stage's cost before it with the step its cheap alternative predicts.
    let mut half_stage = None;
    let mut measured = None;
    for step in SUBPEL_STEPS {
        let stage_origin = best.2;
        let old_bias = if stage_origin == Mv::ZERO {
            params.subpel_move_bias
        } else {
            params.subpel_move_bias_nonzero
        };
        // The half-sample cross uses the DSP's left, right, above, below
        // order. The SVT walk still visits the same costs in its own order.
        let half = (step != 1).then(|| cost.cost_four(cross_around(stage_origin, step, CROSS)));
        // A quarter stage after a half stage that did not move is the
        // adaptive switch's opportunity (`adapt.quarter`): run and
        // measured, or replaced by the step the half stage's costs predict.
        if step == 1
            && stage_origin == start.2
            && let Some(half) = half_stage
        {
            let bias = if svt {
                params.subpel_move_bias
            } else {
                old_bias
            };
            let predict = quarter_predict(params);
            match quarter.turn() {
                Turn::Preset => {}
                Turn::Cheap => return predicted_quarter(best, (half, bias), predict, &mut cost),
                Turn::Measure => measured = Some((best.0, predicted_vector(best, half, predict))),
            }
        }
        if let Some(half) = half {
            half_stage = Some(half);
        }
        for (i, (dx, dy)) in directions.into_iter().enumerate() {
            let centre = if step == 1 { best.2 } else { stage_origin };
            let mv = Mv {
                x: centre.x + dx * step,
                y: centre.y + dy * step,
            };
            let (c, d) = if mv == previous_best.2 {
                (previous_best.0, previous_best.1)
            } else if let Some(half) = half {
                half[if svt { i ^ 1 } else { i }]
            } else {
                cost.cost(mv)
            };
            let biased = if svt {
                if best.2 == start.2 {
                    ((c as u64 * bias_percent) / 100) as f32
                } else {
                    c
                }
            } else {
                c * old_bias
            };
            if biased < best.0 {
                previous_best = best;
                best = (c, d, mv);
            }
        }
    }
    // The cheap alternative keeps the whole gain where it predicts the
    // vector the quarter stage found, and none of it otherwise (it may keep
    // part: an upper bound of its loss).
    if let Some((before, predicted)) = measured {
        let kept = if predicted == Some(best.2) {
            before
        } else {
            best.0
        };
        quarter.record(before, kept);
    }
    best
}

/// The vector [`predicted_quarter`] would move the unmoved half-stage
/// winner `best` to under `predict`, before any probe of it.
fn predicted_vector(
    best: (f32, f32, Mv),
    half: [(f32, f32); 4],
    predict: QuarterPredict,
) -> Option<Mv> {
    if predict == QuarterPredict::Off {
        return None;
    }
    let (index, _) = predicted_step(best.1, half.map(|(_, d)| d))?;
    let (dx, dy) = CROSS[index];
    Some(Mv {
        x: best.2.x + dx,
        y: best.2.y + dy,
    })
}

/// Whether `list` holds `mv`. Candidate lists are a few vectors long and
/// where (or whether) a vector repeats depends on the content, so an
/// early-exit search mispredicts its exit about once per call; this scans
/// the whole list without branches.
/// The cheap vector (`cheap_vectors`, then the codec's refinement-only
/// `seeds`) of least bilinear distortion other than `search_mv`, as
/// `(distortion, vector)`; every distinct vector is measured once and noted
/// in `bilinear`. Merge lists repeat vectors (HEVC `crowd_mot17_09` QP 36:
/// 3.7% of all bilinear measurements were such repeats); the first of equal
/// minima wins either way.
fn best_cheap_vector(
    cheap_vectors: &[Mv],
    seeds: &[Mv],
    search_mv: Mv,
    bilinear: &mut Vec<(Mv, f32)>,
    distortion_of: impl Fn(Mv) -> (f32, f32),
) -> Option<(f32, Mv)> {
    // Whether the `i`-th vector of cheap vectors then seeds repeats an
    // earlier one.
    let measured_before = |i: usize, mv: Mv| {
        let seen_seeds = i.saturating_sub(cheap_vectors.len());
        holds(&cheap_vectors[..i.min(cheap_vectors.len())], mv) || holds(&seeds[..seen_seeds], mv)
    };
    cheap_vectors
        .iter()
        .chain(seeds.iter())
        .enumerate()
        .filter(|&(i, &mv)| mv != search_mv && !measured_before(i, mv))
        .map(|(_, &mv)| {
            let distortion = distortion_of(mv).0;
            bilinear.push((mv, distortion));
            (distortion, mv)
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
}

/// Speed heuristic (`decision.seed_refine_ratio`): which of the two
/// sub-sample seeds of the exact-filter ranking (the motion-search vector
/// and the best cheap vector) are refined. A seed whose whole-sample
/// distortion is more than `ratio` times the other's is ranked where it
/// stands: its refinement (up to eight bilinear evaluations) rarely ends
/// below the better seed's. VVC Default (chart clips, 60 f, CBR 600, P
/// pictures; docs/handoff/vvc.md "vvc-default-prune"): where the search
/// vector is at least 1.1 times the best cheap vector's distortion, its
/// refinement wins 1.2-5.6% of the leaves (unrefined 0%); below 0.9 times,
/// the refined cheap vector wins 0.5-3.5%. 0 refines both. VVC Default,
/// Jetson cycles (chart clips, 150 f, CBR 600) / L1 J (2026-11-14 draw,
/// 24 clips): 1.1 -1.6% / -0.09% [-0.19, +0.14], 2 -0.0% / -0.02%. Other
/// codecs not measured.
#[derive(Debug, Clone, Copy)]
struct SeedGate {
    ratio: f32,
}

impl SeedGate {
    /// Whether a seed of whole-sample `distortion` is refined against the
    /// other seed's distortion `other` (`None`: no other seed).
    fn refines(self, distortion: f32, other: Option<f32>) -> bool {
        self.ratio <= 0.0 || other.is_none_or(|other| distortion <= self.ratio * other)
    }
}

fn holds(list: &[Mv], mv: Mv) -> bool {
    list.iter().fold(false, |found, candidate| {
        found | (candidate.bits() == mv.bits())
    })
}

/// The cheapest of `candidates` (duplicates measured once) by `cost`, which
/// gives (cost, distortion) of a vector. Returns (cost, distortion, vector).
fn rank(candidates: &[Mv], mut cost: impl FnMut(Mv) -> (f32, f32)) -> (f32, f32, Mv) {
    rank_with_stats(candidates, |_, mv| {
        let (c, d) = cost(mv);
        (c, d, Variance::default())
    })
    .0
}

/// [`rank`] for costs that also report the vector's residual statistics;
/// returns the winner and its statistics. `cost` also gets the index of the
/// vector's first occurrence, so callers can tell which part of the list
/// (e.g. the codec's cheap vectors at its start) the vector belongs to
/// without searching it.
fn rank_with_stats(
    candidates: &[Mv],
    mut cost: impl FnMut(usize, Mv) -> (f32, f32, Variance),
) -> ((f32, f32, Mv), Variance) {
    let mut best: Option<((f32, f32, Mv), Variance)> = None;
    for (i, &mv) in candidates.iter().enumerate() {
        if holds(&candidates[..i], mv) {
            continue;
        }
        let (c, d, stats) = cost(i, mv);
        if best.is_none_or(|((b, _, _), _)| c < b) {
            best = Some(((c, d, mv), stats));
        }
    }
    best.expect("the zero vector is always a candidate")
}

/// Skips a sub-sample candidate's exact (codec-filter) prediction, about
/// 3K instructions for a 16x16 AV1 block, when its bilinear cost is already
/// above `margin` times the best exact cost so far
/// (`decision.exact_rank_margin`; 0, off, by default; Fast presets set it). The bilinear distortion
/// estimates the exact one closely, so a candidate far behind on it rarely
/// wins; measured effect at the parameter.
///
/// While a controller gathers evidence (`threshold_control.rs`) it also
/// notes the candidates just inside the margin in `band`.
struct ExactPrune<'a> {
    margin: f32,
    /// Candidates' bilinear distortions, where already measured.
    known: &'a [(Mv, f32)],
    band: Option<&'a RankBand>,
}

impl ExactPrune<'_> {
    fn known_distortion(&self, mv: Mv) -> Option<f32> {
        self.known
            .iter()
            .find(|(known, _)| *known == mv)
            .map(|&(_, distortion)| distortion)
    }

    fn skips(&self, mv: Mv, rate: f32, best_cost: Option<f32>) -> bool {
        self.margin > 0.0
            && best_cost.is_some_and(|best| {
                self.known_distortion(mv)
                    .is_some_and(|distortion| distortion + rate > self.margin * best)
            })
    }

    /// Notes an exact prediction of `mv` that cost `cost` against the best
    /// cost before it: a band run when its estimate's excess over that best
    /// was just inside the margin's (which must exceed 1 to prune at all).
    fn observe(&self, mv: Mv, rate: f32, best_cost: Option<f32>, cost: f32) {
        let (Some(band), Some(best)) = (self.band, best_cost) else {
            return;
        };
        let Some(distortion) = self.known_distortion(mv).filter(|_| best > 0.0) else {
            return;
        };
        let excess = (distortion + rate) / best - 1.0;
        if band.band.below(excess, self.margin - 1.0) {
            band.runs.set(band.runs.get() + 1);
            if cost < best {
                band.leader.set(Some(mv));
            }
        }
    }
}

/// One gated tool's outcome on a block (`threshold_control.rs`): its gate
/// value and threshold, and the vector only it found, if any.
#[derive(Debug, Clone, Copy)]
struct GateOutcome {
    value: f32,
    threshold: f32,
    found: Option<Mv>,
}

/// The decision's gated tools that ran on one block.
#[derive(Debug, Clone, Copy)]
struct GateOutcomes {
    subpel: Option<GateOutcome>,
    probe: Option<GateOutcome>,
}

impl GateOutcomes {
    /// The gated tools' outcomes on a block of `area` samples at `lambda`
    /// whose search seed cost `start` (whole-sample distortion, then the
    /// refinement's): the refined seed `new` (when the seed was refined at
    /// all) and the probed vector `probed`. Kept out of line: only a
    /// controller asks for it.
    #[cold]
    #[inline(never)]
    fn of(
        (seed_refined, new, search_mv, probed): (bool, Mv, Mv, Option<Mv>),
        (start, lambda, area): ((f32, f32), f32, f32),
        params: &DecisionParams,
    ) -> Self {
        let lambda_area = lambda * area;
        Self {
            subpel: seed_refined.then(|| GateOutcome {
                value: start.1 / lambda_area,
                threshold: seed_gate_threshold(area, params),
                found: (new != search_mv).then_some(new),
            }),
            probe: (params.codec_fullpel_probe >= 0.5).then(|| GateOutcome {
                value: start.0 / lambda_area,
                threshold: params.codec_fullpel_probe_min_lambdas,
                found: probed,
            }),
        }
    }

    /// Records each tool that ran, won when its vector is `winner`.
    #[cold]
    #[inline(never)]
    fn record(self, winner: Mv, evidence: &mut FrameEvidence) {
        for (tool, outcome) in [(Tool::Subpel, self.subpel), (Tool::Probe, self.probe)] {
            if let Some(o) = outcome {
                evidence.record_above(tool, o.value, o.threshold, o.found == Some(winner));
            }
        }
    }
}

/// The exact-rank margin's band evidence of one block: the exact
/// predictions whose estimate was just inside the margin, and the last of
/// them that took the lead, which wins if it is still the winner.
struct RankBand {
    band: Band,
    runs: Cell<u32>,
    leader: Cell<Option<Mv>>,
}

impl RankBand {
    fn new(band: Band) -> Self {
        Self {
            band,
            runs: Cell::new(0),
            leader: Cell::new(None),
        }
    }

    /// Adds the block's band runs to `evidence`, with a win when `winner`
    /// is the band candidate that last took the lead.
    #[cold]
    #[inline(never)]
    fn report(&self, winner: Mv, evidence: &mut FrameEvidence) {
        let runs = self.runs.get();
        if runs > 0 {
            let won = self.leader.get() == Some(winner);
            evidence.add(Tool::Rank, u64::from(runs), u64::from(won));
        }
    }
}

/// The ranking entry of a pruned candidate: never the winner (the zero
/// vector, ranked first, is always finite).
const PRUNED: (f32, f32, Variance) = (f32::INFINITY, f32::INFINITY, FORBIDDEN_STATS);

/// [`rank_cached_exact`]'s cost of `mv` with residual `stats` (and, for an
/// exact sub-sample prediction, its `pixels` and their AC `energy` if
/// measured): distortion plus `rate`, plus the psy penalty while the
/// candidate can still win; an improving candidate becomes the cached best
/// and its pixels the backend's cached prediction.
// Always inlined: as a call it costs H.264 about 0.3% of its instructions
// (callgrind, H.264 Fast crowd 720p), more than its whole body.
#[inline(always)]
#[allow(
    clippy::inline_always,
    reason = "measured: a call costs more than the body"
)]
fn consider_exact<B: CodecBackend>(
    backend: &mut B,
    (block, mv, filter): (&Block, Mv, Filter),
    (measurer, psy, dc_weight): (&Measurer<'_>, Option<Psy>, f32),
    rate: f32,
    cached_best_cost: &mut Option<f32>,
    (stats, pixels, energy): (Variance, Option<&[u8]>, Option<u64>),
) -> (f32, f32, Variance) {
    let distortion = ranking_distortion::<B>(stats, block, dc_weight);
    let base = distortion + rate;
    // The penalty is never negative: a candidate already at or above
    // the best cost cannot win, so its prediction energy (a variance
    // pass, ~3% of P-frame instructions when always measured) is not
    // needed. The ranking is unchanged.
    let cost = match psy {
        Some(psy) if cached_best_cost.is_none_or(|best| base < best) => {
            let energy = energy.unwrap_or_else(|| measurer.prediction_energy(mv, pixels));
            base + psy.penalty(energy)
        }
        _ => base,
    };
    // Mirror rank's strict comparison, including first-candidate
    // and tie behavior; integer winners need no filtered cache.
    if cached_best_cost.is_none_or(|best| cost < best) {
        *cached_best_cost = Some(cost);
        if let Some(pixels) = pixels {
            backend.cache_exact_luma(block, mv, filter, pixels, stats);
        }
    }
    (cost, distortion, stats)
}

/// [`rank_with_stats`] by exact cost ([`consider_exact`]), caching the
/// improving candidates' exact pixels in the backend in that same order. The
/// first `cheap` candidates are the codec's cheap vectors (rate scale 1).
/// With `GATHER`, `prune` notes the margin's band evidence.
fn rank_cached_exact<B: CodecBackend, const GATHER: bool>(
    backend: &mut B,
    block: &Block,
    (measurer, psy, dc_weight): (&Measurer<'_>, Option<Psy>, f32),
    candidates: &[Mv],
    (cheap, filter, lambda, new_rate_scale): (usize, Filter, f32, f32),
    prune: &ExactPrune<'_>,
    mut stats_and_energy: impl FnMut(Mv) -> (Variance, Option<u64>),
) -> ((f32, f32, Mv), Variance) {
    let mode_bits = backend.inter_mode_pricer(block);
    let mut cached_best_cost = None;
    rank_with_stats(candidates, |i, mv| {
        let bits = mode_bits(backend, block, mv);
        let scale = if i < cheap { 1.0 } else { new_rate_scale };
        let rate = lambda * scale * bits;
        let exact = !mv.is_full_pel() && measurer.allows(mv);
        if exact && prune.skips(mv, rate, cached_best_cost) {
            return PRUNED;
        }
        let best_before = cached_best_cost;
        let mut consider = |stats: Variance, pixels: Option<&[u8]>, energy: Option<u64>| {
            consider_exact(
                backend,
                (block, mv, filter),
                (measurer, psy, dc_weight),
                rate,
                &mut cached_best_cost,
                (stats, pixels, energy),
            )
        };
        if exact {
            let ranked = measurer.with_exact_prediction(mv, filter, |stats, pixels, energy| {
                consider(stats, Some(pixels), energy)
            });
            if GATHER {
                prune.observe(mv, rate, best_before, ranked.0);
            }
            ranked
        } else {
            let (stats, energy) = stats_and_energy(mv);
            consider(stats, None, energy)
        }
    })
}

/// The walk of `decision.winner_walk_steps` (`None` when 0).
fn winner_walk(params: &DecisionParams) -> Option<RdWalk> {
    let steps = params.winner_walk_steps.round() as usize;
    (steps > 0).then_some(RdWalk {
        square: params.winner_walk_square >= 0.5,
        steps,
    })
}

/// The ranking's winner `start` (with its statistics) walked to the best of
/// its quarter-sample neighbours by distortion + `rate_lambda` · bits on the
/// codec's exact prediction (`decision.winner_walk_*`).
///
/// Why: the seeds (the search vector and the best cheap vector) are refined
/// before the ranking, on bilinear distortion alone, under the seed gates and
/// move biases; the vector that wins the exact ranking, often a cheap
/// (neighbour) vector, is never refined. On VC-1K phone calls our H.264 coded
/// 59 % whole-sample vectors where x264 codes 14 %; walking the winner's
/// neighbours recovered as much as an x264-style coded-cost walk did
/// (docs/handoff/rd-depth.md). The walk leaves the psy term out, as the seed
/// refinement does: with it, the smoother sub-sample predictions lose to the
/// whole-sample copy and the walk recovers about a third of the gain (VC-1K
/// worst six, fixed QP: J -0.4 % against -1.5 %). The returned cost keeps the
/// start's psy penalty, so intra is weighed against the same scale as before.
fn walk_winner<B: CodecBackend>(
    backend: &mut B,
    block: &Block,
    (measurer, filter, dc_weight): (&Measurer<'_>, Filter, f32),
    (rate_lambda, walk): (f32, RdWalk),
    (start, start_stats): ((f32, f32, Mv), Variance),
) -> ((f32, f32, Mv), Variance) {
    let start_rate = rate_lambda * backend.mode_bits(block, &PredictionMode::Inter(start.2));
    let start_walk_cost = start.1 + start_rate;
    // The start's psy penalty (0 without the term).
    let psy_penalty = start.0 - start_walk_cost;
    let mut best = ((start_walk_cost, start.1, start.2), start_stats);
    let mut cached_best_cost = Some(start_walk_cost);
    rd_refine_vector(start.2, start_walk_cost, walk, |mv| {
        if !measurer.allows(mv) {
            return None;
        }
        let rate = rate_lambda * backend.mode_bits(block, &PredictionMode::Inter(mv));
        let (cost, distortion, stats) = if B::CACHE_EXACT_LUMA {
            let mut consider = |stats: Variance, pixels: Option<&[u8]>| {
                consider_exact(
                    backend,
                    (block, mv, filter),
                    (measurer, None, dc_weight),
                    rate,
                    &mut cached_best_cost,
                    (stats, pixels, None),
                )
            };
            if mv.is_full_pel() {
                consider(measurer.measure(mv), None)
            } else {
                measurer.with_exact_prediction(mv, filter, |stats, pixels, _| {
                    consider(stats, Some(pixels))
                })
            }
        } else if B::CACHE_UNWEIGHTED_LUMA && !mv.is_full_pel() {
            measurer.with_exact_prediction(mv, filter, |stats, pixels, _| {
                let distortion = ranking_distortion::<B>(stats, block, dc_weight);
                let cost = distortion + rate;
                if cost < best.0.0 {
                    backend.cache_exact_luma(block, mv, filter, pixels, stats);
                }
                (cost, distortion, stats)
            })
        } else {
            let stats = measurer.measure_exact(mv, filter);
            let distortion = ranking_distortion::<B>(stats, block, dc_weight);
            (distortion + rate, distortion, stats)
        };
        if cost < best.0.0 {
            best = ((cost, distortion, mv), stats);
        }
        Some(cost)
    });
    let ((cost, distortion, mv), stats) = best;
    ((cost + psy_penalty, distortion, mv), stats)
}

/// The psychovisual term of the inter ranking for one block
/// (`decision.psy_strength`, [`crate::psy`]).
#[derive(Debug, Clone, Copy)]
struct Psy {
    strength: f32,
    source_energy: u64,
}

impl Psy {
    /// The term for the measurer's block, or `None` when it is off.
    fn of(params: &DecisionParams, measurer: &Measurer<'_>) -> Option<Self> {
        (params.psy_strength > 0.0).then(|| Self {
            strength: params.psy_strength,
            source_energy: measurer.source_energy(),
        })
    }

    /// The penalty of a prediction with energy `prediction_energy`.
    fn penalty(self, prediction_energy: u64) -> f32 {
        psy_penalty(self.strength, self.source_energy, prediction_energy)
    }
}

/// The best inter candidate, as intra is compared against it.
#[derive(Debug, Clone, Copy)]
struct InterChoice {
    /// Distortion + λ · bits.
    cost: f32,
    /// Mean-removed residual distortion (`n_variance`).
    distortion: f32,
    /// The residual's mean (DC) error, `sse - n_variance`, when
    /// `decision.intra_test_dc` is on (else 0).
    dc: f32,
    mv: Mv,
}

/// The ranking distortion of inter vectors: mean-removed variance plus
/// `dc_weight` (`decision.rank_dc_weight`) of the residual's DC error.
/// Variance alone is right between vectors that differ in texture, but on
/// a moving smooth gradient the zero vector's residual is mostly DC, so it
/// ranks nearly as well as the true motion; where the codec pays for that DC
/// (H.264: one DC per 4x4 or 8x8 block), the encoder then codes it.
fn ranking_distortion<B: CodecBackend>(stats: Variance, block: &Block, dc_weight: f32) -> f32 {
    let variance = n_variance::<B>(stats, block);
    variance as f32 + dc_weight * (u64::from(stats.sse) - variance) as f32
}

/// The residual DC error of `stats` over `block` (`sse - n_variance`, the
/// part mean-removed ranking ignores), or 0 unless `decision.intra_test_dc`.
fn dc_error<B: CodecBackend>(params: &DecisionParams, stats: Variance, block: &Block) -> f32 {
    if params.intra_test_dc >= 0.5 {
        (u64::from(stats.sse) - n_variance::<B>(stats, block)) as f32
    } else {
        0.0
    }
}

/// DC intra when the best inter vector predicts badly and intra costs less,
/// otherwise that vector.
///
/// Inter candidates are ranked by mean-removed variance, which is right
/// between inter vectors but blind to a brightness change (exposure jump,
/// flash, fade): the residual is then mostly DC, so its variance stays small
/// and intra is never tried. With `decision.intra_test_dc` the gate and the
/// comparison add the winner's DC error. On `bodycam_antare_0815` (exposure
/// jumps; SVT codes 40-64% intra on those frames) HEVC Fast Q41 rate -5.8% at
/// +0.35dB PSNR-YUV; crowd09 Q47 -1.2% rate at +0.20dB.
#[inline]
fn intra_or_inter<B: CodecBackend>(
    backend: &B,
    (measurer, source): (&Measurer<'_>, &Picture),
    block: &Block,
    lambda: f32,
    params: &DecisionParams,
    inter: InterChoice,
) -> Choice {
    let area = leaf_area::<B>(block) as f32;
    // Rectangular leaves are inter only (`CodecBackend::partition_shapes`).
    if is_square_leaf::<B>(block) && inter.distortion + inter.dc > intra_gate(params, lambda) * area
    {
        // The DC prediction's residual is the source about its mean: its
        // distortion is the source's AC energy.
        let intra_distortion = measurer.source_energy() as f32;
        // The DC prediction keeps none of the source's AC energy: its whole
        // energy is the psy penalty ([`crate::psy`]).
        let mode_bits = backend.mode_bits(block, &PredictionMode::IntraDc);
        let dc_cost = intra_distortion * (1. + params.psy_strength) + lambda * mode_bits;
        let inter_cost = inter.cost + inter.dc;
        if dc_cost < inter_cost {
            return Choice {
                mode: DecidedMode::INTRA,
                cost: dc_cost,
                pricing: PricingHint {
                    intra_margin: inter_cost - dc_cost,
                    ..PricingHint::NONE
                },
            };
        }
        // The directional estimate only matters where DC intra loses.
        let directional = (params.intra_directional >= 0.5)
            .then(|| {
                directional_intra_cost(
                    backend,
                    (measurer.dsp, source),
                    block,
                    (lambda, mode_bits),
                    params,
                    (dc_cost, inter_cost),
                )
            })
            .flatten();
        if let Some(cost) = directional.filter(|&cost| cost < inter_cost) {
            // The estimate is open loop (source neighbours): the backend's
            // own intra search may still find inter cheaper.
            return Choice {
                mode: DecidedMode::intra_over(inter.mv),
                cost,
                pricing: PricingHint {
                    intra_margin: inter_cost - cost,
                    ..PricingHint::NONE
                },
            };
        }
        let intra_cost = directional.map_or(dc_cost, |cost| cost.min(dc_cost));
        return Choice {
            mode: DecidedMode::inter(inter.mv),
            cost: inter.cost,
            pricing: PricingHint {
                winner_error: inter.distortion,
                intra_ratio: intra_cost / inter_cost,
                inter_cost,
                ..PricingHint::NONE
            },
        };
    }
    Choice {
        mode: DecidedMode::inter(inter.mv),
        cost: inter.cost,
        pricing: PricingHint::NONE,
    }
}

/// The inter residual variance per luma sample above which DC intra is
/// tried: `decision.intra_gate_lambdas` times λ where set, else the fixed
/// `decision.intra_gate_variance`.
fn intra_gate(params: &DecisionParams, lambda: f32) -> f32 {
    if params.intra_gate_lambdas > 0.0 {
        params.intra_gate_lambdas * lambda
    } else {
        params.intra_gate_variance
    }
}

/// The final stage's intra gate per luma sample:
/// `decision.final_intra_gate_lambdas` times λ where set, else the
/// ranking's [`intra_gate`].
fn final_intra_gate(params: &DecisionParams, lambda: f32) -> f32 {
    if params.final_intra_gate_lambdas > 0.0 {
        params.final_intra_gate_lambdas * lambda
    } else {
        intra_gate(params, lambda)
    }
}

/// The directional intra estimate's cost of square leaf `block`
/// ([`crate::intra_estimate`], from `source`): distortion, psy term and λ times the
/// intra mode's `mode_bits` plus `decision.intra_direction_bits` per
/// sub-block direction. `None` where it cannot or rarely would beat
/// `inter_cost`: where DC intra costs `decision.intra_directional_dc_ratio`
/// times inter or more, and where the signalling alone (a lower bound on
/// the cost) reaches inter's cost.
// Out of line, so `intra_or_inter` keeps its cost with the estimate off.
#[inline(never)]
fn directional_intra_cost<B: CodecBackend>(
    backend: &B,
    (dsp, source): (&Dsp, &Picture),
    block: &Block,
    (lambda, mode_bits): (f32, f32),
    params: &DecisionParams,
    (dc_cost, inter_cost): (f32, f32),
) -> Option<f32> {
    if dc_cost >= params.intra_directional_dc_ratio * inter_cost {
        return None;
    }
    let log2 = backend.intra_estimate_log2();
    let sub_blocks = directional_sub_blocks::<B>(block, log2);
    let signalling = lambda * (mode_bits + params.intra_direction_bits * sub_blocks);
    if signalling >= inter_cost {
        return None;
    }
    let estimate = intra_estimate::estimate(
        dsp,
        source.plane(PlaneIndex::Y),
        (block.x, block.y),
        block.width(),
        log2,
    )?;
    Some(estimate.distortion as f32 + params.psy_strength * estimate.energy_gap as f32 + signalling)
}

/// Sub-blocks of `1 << log2` (at most the leaf) in square leaf `block`: the
/// directions the directional intra estimate signals.
fn directional_sub_blocks<B: CodecBackend>(block: &Block, log2: u8) -> f32 {
    let side = leaf_extent::<B>(block).0;
    let sub = (1usize << log2).min(side);
    ((side / sub) * (side / sub)) as f32
}

/// SVT-AV1's fixed-stage sub-sample search (`md_subpel_search_fixed_stage`):
/// from the starting `(cost, distortion, vector)`, try the four half-sample
/// neighbours, then the four quarter-sample neighbours of the best, each
/// move having to win by `move_bias` (`decision.subpel_move_bias`). `cost` gives (cost,
/// distortion) of a vector. Returns the best `(cost, distortion, vector)`.
///
/// Starts with distortion below `skip_below` are not refined (the backend's
/// [`CodecBackend::subpel_skip_variance`] times the block area). SVT's
/// level-8 exit (`var < 10 · area`) for every codec blocked all refinement
/// on clean, well-predicted content (`bodycam_airport` at 46–50 dB): AV1 coded
/// only whole-sample vectors where SVT uses quarter-sample ones. Removing it
/// (quick clips, PSNR-YUV BD): AV1 bodycam vs svtav1-m12 about +690% → +280%,
/// low-light −5 points; HEVC −1.1 points overall; H.264 unchanged at
/// −8.4% vs openh264 but +6 M instructions/frame on crowd, so H.264 keeps
/// the exit. AV1 +10 M instructions/frame on crowd.
fn refine_subpel(
    (start_cost, start_distortion, start): (f32, f32, Mv),
    skip_below: f32,
    (zero_bias, nonzero_bias, fractional_bias): (f32, f32, f32),
    (quarter, predict): (QuarterStage, QuarterPredict),
    mut cost: impl RefineCost,
) -> (f32, f32, Mv) {
    let mut best = (start_cost, start_distortion, start);
    if start_distortion < skip_below {
        return best;
    }
    for step in SUBPEL_STEPS {
        let centre = best.2;
        if step == 1 && quarter.skips(centre, start) {
            break;
        }
        // SVT biases only moves away from the zero vector. Leaving a
        // full-sample position adds interpolation work to the prediction;
        // moving between sub-sample positions does not, so that margin is
        // separate (`subpel_fractional_bias`).
        let move_bias = if centre == Mv::ZERO {
            zero_bias
        } else if centre.is_full_pel() {
            nonzero_bias
        } else {
            fractional_bias
        };
        // The four vectors of a stage share its centre: one batch. A vector
        // is taken only if `c * move_bias < best`, so one whose distortion
        // alone fails that against the stage's starting best need not be
        // priced.
        let mvs = cross_around(centre, step, CROSS);
        let costs = cost.cost_four_below(mvs, (best.0, move_bias));
        for (mv, (c, d)) in mvs.into_iter().zip(costs) {
            if c * move_bias < best.0 {
                best = (c, d, mv);
            }
        }
        // A half stage that did not move, before a quarter stage that will
        // not run: its costs predict the quarter step
        // (`decision.subpel_quarter_predict`).
        if predict != QuarterPredict::Off
            && step != 1
            && best.2 == centre
            && quarter.skips(centre, start)
        {
            return predicted_quarter(best, (costs, move_bias), predict, &mut cost);
        }
    }
    best
}

/// The half-stage winner `best` (unmoved) after the quarter step its
/// half-stage `costs` predict, under `predict` with the stage's `move_bias`.
fn predicted_quarter(
    best: (f32, f32, Mv),
    (costs, move_bias): ([(f32, f32); 4], f32),
    predict: QuarterPredict,
    cost: &mut impl RefineCost,
) -> (f32, f32, Mv) {
    let Some((index, predicted)) = predicted_step(best.1, costs.map(|(_, d)| d)) else {
        return best;
    };
    let (dx, dy) = CROSS[index];
    let mv = Mv {
        x: best.2.x + dx,
        y: best.2.y + dy,
    };
    match predict {
        QuarterPredict::Off => best,
        QuarterPredict::Probe => {
            let (c, d) = cost.cost_below(mv, (best.0, move_bias));
            if c * move_bias < best.0 {
                (c, d, mv)
            } else {
                best
            }
        }
        // Unmeasured: the cost falls by the predicted distortion gain.
        QuarterPredict::Model { strength } => {
            let gain = best.1 - predicted;
            if gain >= strength * best.1 {
                (best.0 - gain, predicted, mv)
            } else {
                best
            }
        }
    }
}

/// Maximum number of sub-pel finalists re-ranked on the codec's exact filter
/// (`decision.exact_finalists`).
const MAX_EXACT_FINALISTS: usize = 4;

/// Fixed-capacity sorted list of unique motion-vector finalists from
/// [`refine_subpel_finalists`].
struct FinalistList {
    scores: [f32; MAX_EXACT_FINALISTS],
    mvs: [Mv; MAX_EXACT_FINALISTS],
    len: usize,
    cap: usize,
}

impl FinalistList {
    fn new(cap: usize) -> Self {
        Self {
            scores: [f32::INFINITY; MAX_EXACT_FINALISTS],
            mvs: [Mv::ZERO; MAX_EXACT_FINALISTS],
            len: 0,
            cap: cap.clamp(1, MAX_EXACT_FINALISTS),
        }
    }

    /// Adds `mv` at `score` (or lowers its score if listed), keeping the list
    /// sorted by score and at most `cap` long.
    fn push(&mut self, score: f32, mv: Mv) {
        let slot = if let Some(pos) = self.mvs[..self.len].iter().position(|&m| m == mv) {
            if score >= self.scores[pos] {
                return;
            }
            pos
        } else if self.len < self.cap {
            self.len += 1;
            self.len - 1
        } else if score < self.scores[self.len - 1] {
            self.len - 1
        } else {
            return;
        };
        self.scores[slot] = score;
        self.mvs[slot] = mv;
        self.sift_up(slot);
    }

    /// Moves entry `i` towards the front while it scores below its predecessor.
    fn sift_up(&mut self, mut i: usize) {
        while i > 0 && self.scores[i] < self.scores[i - 1] {
            self.scores.swap(i, i - 1);
            self.mvs.swap(i, i - 1);
            i -= 1;
        }
    }

    /// Ensure the bilinear winner `winner` is present (and at index 0 if `cap == 1`).
    fn ensure_winner(&mut self, winner: Mv, winner_cost: f32) {
        if self.cap == 1 {
            self.scores[0] = winner_cost;
            self.mvs[0] = winner;
            self.len = 1;
            return;
        }
        if !self.mvs[..self.len].contains(&winner) {
            if self.len < self.cap {
                self.mvs[self.len] = winner;
                self.scores[self.len] = winner_cost;
                self.len += 1;
            } else {
                self.mvs[self.len - 1] = winner;
                self.scores[self.len - 1] = winner_cost;
            }
        }
    }

    fn entries(&self) -> (&[f32], &[Mv]) {
        (&self.scores[..self.len], &self.mvs[..self.len])
    }
}

/// Bilinear sub-sample refinement that also collects the top `cap` unique
/// finalists for exact-filter re-ranking (`decision.exact_finalists`).
/// Returns `(bilinear_best, finalists, searched)` where `searched` is `false`
/// when `start_distortion < skip_below`.
#[inline(never)]
fn refine_subpel_finalists(
    (start_cost, start_distortion, start): (f32, f32, Mv),
    skip_below: f32,
    (zero_bias, nonzero_bias, fractional_bias): (f32, f32, f32),
    (quarter, predict): (QuarterStage, QuarterPredict),
    cap: usize,
    mut cost: impl RefineCost,
) -> ((f32, f32, Mv), FinalistList, bool) {
    let mut best = (start_cost, start_distortion, start);
    let mut finalists = FinalistList::new(cap);
    if start_distortion < skip_below {
        return (best, finalists, false);
    }
    finalists.push(start_cost, start);
    for step in SUBPEL_STEPS {
        let centre = best.2;
        if step == 1 && quarter.skips(centre, start) {
            break;
        }
        let move_bias = if centre == Mv::ZERO {
            zero_bias
        } else if centre.is_full_pel() {
            nonzero_bias
        } else {
            fractional_bias
        };
        let mvs = cross_around(centre, step, CROSS);
        let costs = cost.cost_four_below(mvs, (best.0, move_bias));
        for (mv, (c, d)) in mvs.into_iter().zip(costs) {
            if c * move_bias < best.0 {
                best = (c, d, mv);
            }
            // Even when bilinear `c` was pruned by `move_bias`, keep the best
            // distortion candidates so exact-filter re-ranking can recover a
            // sharp half/quarter-pel match that bilinear blurred.
            let key = if c.is_finite() { c } else { d * move_bias };
            finalists.push(key, mv);
        }
        if predict != QuarterPredict::Off
            && step != 1
            && best.2 == centre
            && quarter.skips(centre, start)
        {
            best = predicted_quarter(best, (costs, move_bias), predict, &mut cost);
            finalists.push(best.0, best.2);
            finalists.ensure_winner(best.2, best.0);
            return (best, finalists, true);
        }
    }
    finalists.ensure_winner(best.2, best.0);
    (best, finalists, true)
}

/// Maximum bilinear cost ratio (`score / best_bilinear_cost`) for a fractional
/// runner-up finalist to be re-evaluated on the codec's exact luma filter.
const EXACT_FINALIST_MARGIN: f32 = 1.15;

/// Re-rank the top bilinear finalists on the codec's exact luma filter and
/// cache the winning fractional prediction via [`CodecBackend::cache_exact_luma`].
#[inline(never)]
fn rerank_exact_finalists<B: CodecBackend>(
    backend: &mut B,
    measurer: &Measurer<'_>,
    block: &Block,
    filter: Filter,
    lambda: f32,
    bilinear_best_cost: f32,
    (scores, mvs): (&[f32], &[Mv]),
    stats_of: &impl Fn(Mv) -> Variance,
) -> ((f32, f32, Mv), Variance) {
    let cache_luma = B::CACHE_EXACT_LUMA || B::CACHE_UNWEIGHTED_LUMA;
    let margin_limit = bilinear_best_cost * EXACT_FINALIST_MARGIN;
    let mut best: Option<(f32, f32, Mv, Variance)> = None;
    for (idx, (&score, &mv)) in scores.iter().zip(mvs).enumerate() {
        // Always evaluate the first finalist and any full-pel finalist (free);
        // prune fractional runner-ups whose bilinear score is far behind the winner.
        if idx > 0 && !mv.is_full_pel() && score > margin_limit {
            continue;
        }
        let rate = lambda * backend.mode_bits(block, &PredictionMode::Inter(mv));
        if mv.is_full_pel() {
            let stats = stats_of(mv);
            let distortion = n_variance::<B>(stats, block) as f32;
            let cost = distortion + rate;
            if best.is_none_or(|(bc, ..)| cost < bc) {
                best = Some((cost, distortion, mv, stats));
            }
        } else if cache_luma && measurer.allows(mv) {
            measurer.with_exact_prediction(mv, filter, |stats, pixels, _| {
                let distortion = n_variance::<B>(stats, block) as f32;
                let cost = distortion + rate;
                if best.is_none_or(|(bc, ..)| cost < bc) {
                    backend.cache_exact_luma(block, mv, filter, pixels, stats);
                    best = Some((cost, distortion, mv, stats));
                }
            });
        } else {
            let stats = measurer.measure_exact(mv, filter);
            let distortion = n_variance::<B>(stats, block) as f32;
            let cost = distortion + rate;
            if best.is_none_or(|(bc, ..)| cost < bc) {
                best = Some((cost, distortion, mv, stats));
            }
        }
    }
    let (c, d, mv, stats) = best.expect("non-empty finalist list");
    ((c, d, mv), stats)
}

/// Whole-sample `start` (residual `start_stats`) of `block` refined to sub-sample
/// precision by residual variance alone: the half- then quarter-sample
/// cross walk of mode decision ([`refine_subpel`]) without rate, move bias
/// or early exit. Partition planning prices shapes at the vectors mode
/// decision would reach (`planner.rect_rd`).
pub(crate) fn refined_vector<B: CodecBackend>(
    measurer: &Measurer<'_>,
    block: &Block,
    (start, start_stats): (Mv, Variance),
) -> Mv {
    let distortion = |stats: Variance| {
        let distortion = n_variance::<B>(stats, block) as f32;
        (distortion, distortion)
    };
    let cost = BatchedCost {
        one: &|mv| distortion(measurer.measure(mv)),
        four: &|mvs| measurer.measure_four(mvs).map(distortion),
    };
    let (start_distortion, _) = distortion(start_stats);
    let no_bias = (1.0, 1.0, 1.0);
    refine_subpel(
        (start_distortion, start_distortion, start),
        0.0,
        no_bias,
        (QuarterStage::Always, QuarterPredict::Off),
        cost,
    )
    .2
}

/// `n · variance` of residual statistics over `block`: the sum of squared
/// residuals minus the squared residual sum over the area.
pub(crate) fn n_variance<B: CodecBackend>(stats: Variance, block: &Block) -> u64 {
    let log2_width = u32::from(block.log2_width);
    let log2_height = if B::RECTANGULAR_LEAVES {
        u32::from(block.log2_height)
    } else {
        log2_width
    };
    u64::from(stats.n_variance_pow2(log2_width + log2_height))
}

/// Width and height of leaf `block` of backend `B`: for square-only
/// backends ([`CodecBackend::RECTANGULAR_LEAVES`] false) the side twice, so
/// the compiler sees one value (the square path's cost).
#[inline]
pub(crate) fn leaf_extent<B: CodecBackend>(block: &Block) -> (usize, usize) {
    if B::RECTANGULAR_LEAVES {
        (block.width(), block.height())
    } else {
        let side = block.width();
        (side, side)
    }
}

/// Whether leaf `block` of backend `B` is square: always for square-only
/// backends, without comparing the sides ([`leaf_extent`]).
#[inline]
pub(crate) fn is_square_leaf<B: CodecBackend>(block: &Block) -> bool {
    !B::RECTANGULAR_LEAVES || block.is_square()
}

/// log2 of the shorter side of leaf `block` of backend `B`
/// ([`leaf_extent`]).
#[inline]
pub(crate) fn leaf_log2_min_side<B: CodecBackend>(block: &Block) -> u8 {
    if B::RECTANGULAR_LEAVES {
        block.log2_min_side()
    } else {
        block.log2_width
    }
}

/// Area of leaf `block` of backend `B` in luma samples ([`leaf_extent`]).
#[inline]
pub(crate) fn leaf_area<B: CodecBackend>(block: &Block) -> usize {
    let (width, height) = leaf_extent::<B>(block);
    width * height
}

/// Residual statistics of one block at many vectors, with the source view
/// and kernel size set up once: the sum and sum of squares of the luma
/// residual of the block predicted from the reference at a vector;
/// [`n_variance`] of them is the mean-removed distortion mode decision
/// ranks by. Sub-sample positions use libaom's bilinear sub-pixel variance
/// (SVT's `svf`). Reference samples outside the picture repeat its edge:
/// reconstructions carry extended borders, and vectors that reach past them
/// use a clamped copy of the reference window.
///
/// Not generic over the backend (only [`Measurer::new`] is): its methods
/// stay shared, out-of-line functions, as before rectangular leaves. Generic,
/// every method inlined into its single caller per backend and H.264 Fast
/// cost +0.5-0.8% Jetson cycles.
pub(crate) struct Measurer<'a> {
    dsp: &'a Dsp,
    src: PlaneView<'a>,
    plane: &'a Plane,
    /// The whole reference plane, borders included, and the block's
    /// top-left in its coordinates: positions are offsets into this one
    /// view, so a measurement needs no window construction.
    padded: PlaneView<'a>,
    origin: (isize, isize),
    /// The block's variance against windows of `padded`.
    windows: WindowVariance<'a>,
    block: Block,
    size: BlockSize,
    /// Intra refresh: vectors this window forbids measure as
    /// [`FORBIDDEN_STATS`], so no ranking ever picks them.
    refresh: Option<&'a RefreshWindow>,
    /// The decision's shared scratch: the buffer exact predictions reuse
    /// (a local one is used without it) and the search position count.
    scratch: Option<&'a MeasureScratch>,
    /// Exact predictions also report their AC energy ([`crate::psy`]),
    /// measured in the residual variance pass.
    energy: bool,
    /// `decision.distortion_row_step` 2: whole-sample and exact
    /// distortions are measured on the even rows only, at this half-height
    /// size (`None`: every row).
    even_rows: Option<BlockSize>,
    /// The source block's AC energy once known ([`Self::source_energy`]):
    /// both the psy term and the intra gate need it.
    source_energy: Cell<Option<u64>>,
    /// Pre-analysis whose 8x8 moments give that energy without a variance
    /// pass, where they cover the block.
    source_moments: Option<&'a PreAnalysis>,
}

/// Smallest quadrant side [`Measurer::measure_quadrants`] splits a block
/// into: below it four kernel calls cost more than the reuse saves.
const MIN_QUADRANT_SIDE: usize = 16;

/// Statistics of a vector intra refresh forbids: a larger distortion than
/// any real block has (`4096 * 255^2 < u32::MAX`), while the zero vector,
/// always allowed for clean blocks, keeps every ranking finite.
const FORBIDDEN_STATS: Variance = Variance {
    sse: u32::MAX,
    sum: 0,
};

impl<'a> Measurer<'a> {
    /// The measurer of leaf `block` of backend `B` ([`leaf_extent`]).
    pub(crate) fn new<B: CodecBackend>(
        dsp: &'a Dsp,
        source: &'a Picture,
        reference: &'a Picture,
        block: &Block,
    ) -> Self {
        let (w, h) = leaf_extent::<B>(block);
        let plane = reference.plane(PlaneIndex::Y);
        let src = source
            .plane(PlaneIndex::Y)
            .region(block.x as isize, block.y as isize, w, h)
            .expect("blocks lie inside the coded picture");
        let padded = plane.padded_view();
        let size = BlockSize::new(w, h).expect("engine blocks are kernel sizes");
        Self {
            dsp,
            windows: dsp.window_variance(size, &padded, &src),
            src,
            plane,
            padded,
            origin: (
                (block.x + plane.border_x()) as isize,
                (block.y + plane.border_y()) as isize,
            ),
            block: *block,
            size,
            refresh: None,
            scratch: None,
            energy: false,
            even_rows: None,
            source_energy: Cell::new(None),
            source_moments: None,
        }
    }

    /// The block's width and height.
    #[inline]
    fn extent(&self) -> (usize, usize) {
        (self.block.width(), self.block.height())
    }

    /// Whole-sample and exact distortions on every `row_step`th row only
    /// (1 or 2; blocks below 8x8 keep every row).
    pub(crate) fn with_row_step(mut self, row_step: f32) -> Self {
        let (w, h) = self.extent();
        self.even_rows = (row_step >= 1.5)
            .then(|| BlockSize::new(w, h / 2))
            .flatten();
        self
    }

    /// Residual statistics of `pred` against the source: over every row,
    /// or over the even rows scaled to the whole block.
    fn residual_stats(&self, pred: &PlaneView<'_>) -> Variance {
        match self.even_rows {
            None => self.dsp.variance(self.size, &self.src, pred),
            Some(half) => {
                let stats = self
                    .dsp
                    .variance(half, &even_rows(&self.src), &even_rows(pred));
                Variance {
                    sse: stats.sse * 2,
                    sum: stats.sum * 2,
                }
            }
        }
    }

    /// This measurer with the vectors `refresh` forbids for the block made
    /// unattractive ([`FORBIDDEN_STATS`]).
    pub(crate) fn with_refresh(mut self, refresh: Option<&'a RefreshWindow>) -> Self {
        self.refresh = refresh;
        self
    }

    /// Predict exact candidates into `scratch`'s buffer instead of a fresh
    /// one, and count the positions measured in it
    /// ([`DecisionScratch::search_positions`]).
    pub(crate) fn with_scratch(mut self, scratch: &'a MeasureScratch) -> Self {
        self.scratch = Some(scratch);
        self
    }

    /// Note `count` block predictions measured, where they are counted.
    #[inline]
    fn count_positions(&self, count: u64) {
        if let Some(scratch) = self.scratch {
            scratch.count_positions(count);
        }
    }

    /// Whether intra refresh allows predicting the block from `mv`.
    #[inline]
    pub(crate) fn allows(&self, mv: Mv) -> bool {
        self.refresh.is_none_or(|w| w.allows(&self.block, mv))
    }

    /// Exact predictions also measure their AC energy when `on`.
    pub(crate) fn with_prediction_energy(mut self, on: bool) -> Self {
        self.energy = on;
        self
    }

    fn with_prediction_energy_and_rows(
        self,
        params: &DecisionParams,
        source_moments: Option<&'a PreAnalysis>,
    ) -> Self {
        self.with_prediction_energy(params.psy_strength > 0.0)
            .with_row_step(params.distortion_row_step)
            .with_source_moments(source_moments)
    }

    /// Predict the block at `mv` into compound `i16` precision (`AV1 §7.11.3.4`).
    fn predict_compound_into(&self, filter: Filter, mv: Mv, dst: &mut [i16]) {
        let (dx, dy) = mv.full_pel();
        let per_unit = filter.positions() as i32 / MV_UNITS_PER_SAMPLE;
        let (phase_x, phase_y) = mv.fraction();
        let (fx, fy) = ((phase_x * per_unit) as usize, (phase_y * per_unit) as usize);
        let before = filter.before() as isize;
        let extra = filter.taps() - 1;
        let (w, h) = self.extent();
        let (span_w, span_h) = (w + extra, h + extra);
        let (x0, y0) = (
            self.block.x as isize + dx as isize - before,
            self.block.y as isize + dy as isize - before,
        );
        if let Some(window) = self.plane.region(x0, y0, span_w, span_h) {
            self.dsp
                .predict_compound(filter, self.size, &window, (fx, fy), dst, w);
        } else {
            let gathered = clamped_window(self.plane, x0, y0, span_w, span_h);
            let window = PlaneView::new(&gathered, span_w, span_w, span_h)
                .expect("window covers its samples");
            self.dsp
                .predict_compound(filter, self.size, &window, (fx, fy), dst, w);
        }
    }

    /// Residual statistics of the block predicted at `mv` with the codec's
    /// own luma interpolation `filter` (the prediction the codec will code).
    pub(crate) fn measure_exact(&self, mv: Mv, filter: Filter) -> Variance {
        if !self.allows(mv) {
            return FORBIDDEN_STATS;
        }
        if mv.is_full_pel() {
            // Whole-sample vectors are copies under any filter.
            return self.measure(mv);
        }
        self.with_exact_prediction(mv, filter, |stats, _, _| stats)
    }

    /// Borrow the exact samples while they are alive, avoiding a second filter
    /// when a backend preserves the improving candidate for reconstruction.
    fn with_exact_prediction<R>(
        &self,
        mv: Mv,
        filter: Filter,
        consume: impl FnOnce(Variance, &[u8], Option<u64>) -> R,
    ) -> R {
        debug_assert!(!mv.is_full_pel());
        self.count_positions(1);
        let (dx, dy) = mv.full_pel();
        let per_unit = filter.positions() as i32 / MV_UNITS_PER_SAMPLE;
        let (phase_x, phase_y) = mv.fraction();
        let (fx, fy) = ((phase_x * per_unit) as usize, (phase_y * per_unit) as usize);
        let before = filter.before() as isize;
        let extra = filter.taps() - 1;
        let (w, h) = self.extent();
        let span = (w + extra, h + extra);
        let (x0, y0) = (
            self.block.x as isize + dx as isize - before,
            self.block.y as isize + dy as isize - before,
        );
        let area = w * h;
        if let Some(buffer) = self.scratch.map(|s| &s.prediction) {
            let mut buffer = buffer.borrow_mut();
            if buffer.len() < area {
                buffer.resize(area, 0);
            }
            let pred = &mut buffer[..area];
            let (stats, energy) = self.predict_into(filter, x0, y0, span, fx, fy, pred);
            consume(stats, pred, energy)
        } else {
            let mut pred = vec![0; area];
            let (stats, energy) = self.predict_into(filter, x0, y0, span, fx, fy, &mut pred);
            consume(stats, &pred, energy)
        }
    }

    /// [`Self::measure_exact`]'s prediction into `pred` (the block's area):
    /// its residual statistics and, when the measurer measures energy, the
    /// prediction's AC energy from the same pass.
    fn predict_into(
        &self,
        filter: Filter,
        x0: isize,
        y0: isize,
        (span_w, span_h): (usize, usize),
        fx: usize,
        fy: usize,
        pred: &mut [u8],
    ) -> (Variance, Option<u64>) {
        let (w, h) = self.extent();
        let mut dst = PlaneViewMut::new(pred, w, w, h).expect("block fits");
        let predict = |window: &PlaneView<'_>, dst: &mut PlaneViewMut<'_>| {
            // H.264 luma through its phase-specialised kernels (the same samples).
            if filter == Filter::H264Luma {
                self.dsp.h264_predictor(filter, self.size).predict(
                    window.data(),
                    window.stride(),
                    fx,
                    fy,
                    dst.data_mut(),
                    w,
                );
            } else {
                self.dsp.predict(filter, self.size, window, fx, fy, dst);
            }
        };
        if let Some(window) = self.plane.region(x0, y0, span_w, span_h) {
            predict(&window, &mut dst);
        } else {
            let gathered = clamped_window(self.plane, x0, y0, span_w, span_h);
            let window = PlaneView::new(&gathered, span_w, span_w, span_h)
                .expect("window covers its samples");
            predict(&window, &mut dst);
        }
        if self.energy {
            let (stats, energy) = self.stats_and_energy(&dst.as_view());
            (stats, Some(energy))
        } else {
            (self.residual_stats(&dst.as_view()), None)
        }
    }

    /// Residual statistics of `pred` against the source over every row, and
    /// `pred`'s AC energy ([`crate::psy::ac_energy`]), in one pass.
    fn stats_and_energy(&self, pred: &PlaneView<'_>) -> (Variance, u64) {
        let measured = self.dsp.variance_energy(self.size, &self.src, pred);
        let (w, h) = self.extent();
        let area = (w * h) as u32;
        (
            measured.residual,
            u64::from(measured.prediction.n_variance(area)),
        )
    }

    /// [`Self::measure`] of the whole-sample vector `mv` and, when the
    /// measurer measures energy and every row, the reference block's AC
    /// energy from the same pass (`variance_energy`, ~1.2x a `variance`
    /// pass on NEON, where [`Self::prediction_energy`] later is a second
    /// one over the same samples). Even-row distortion and forbidden
    /// vectors have no fused form and report no energy.
    // Out of line: inlined, the gate for measurers without energy grew
    // `code_leaf` and cost H.264 Fast +0.6% Jetson cycles (bodycam) for
    // AV1's 0.03% host instructions.
    fn measure_with_energy(&self, mv: Mv) -> (Variance, Option<u64>) {
        debug_assert!(mv.is_full_pel());
        if !self.energy || self.even_rows.is_some() || !self.allows(mv) {
            return (self.measure(mv), None);
        }
        self.count_positions(1);
        let (w, h) = self.extent();
        let (dx, dy) = mv.full_pel();
        let (x, y) = (
            self.block.x as isize + dx as isize,
            self.block.y as isize + dy as isize,
        );
        let (stats, energy) = if let Some(view) = self.plane.region(x, y, w, h) {
            self.stats_and_energy(&view)
        } else {
            let gathered = clamped_window(self.plane, x, y, w, h);
            self.stats_and_energy(
                &PlaneView::new(&gathered, w, w, h).expect("window covers its samples"),
            )
        };
        (stats, Some(energy))
    }

    /// Take the source block's AC energy from `pre`'s 8x8 moments, which
    /// must describe exactly the source samples the block covers.
    pub(crate) fn with_source_moments(mut self, pre: Option<&'a PreAnalysis>) -> Self {
        self.source_moments = pre;
        self
    }

    /// The source block's AC energy ([`crate::psy::ac_energy`]), found once
    /// and only when asked for: from the pre-analysis moments where they
    /// cover the block (the same value), else with a variance pass.
    fn source_energy(&self) -> u64 {
        if let Some(energy) = self.source_energy.get() {
            return energy;
        }
        let block = &self.block;
        let energy = self
            .source_moments
            .filter(|_| block.is_square())
            .and_then(|pre| pre.luma_energy(block.x, block.y, block.log2_width))
            .unwrap_or_else(|| ac_energy(self.dsp, self.size, &self.src));
        self.source_energy.set(Some(energy));
        energy
    }

    /// AC energy of the prediction at `mv` ([`crate::psy`]): `pixels` when the
    /// caller holds them (exact sub-sample predictions), otherwise the
    /// whole-sample reference block; 0 for vectors intra refresh forbids.
    fn prediction_energy(&self, mv: Mv, pixels: Option<&[u8]>) -> u64 {
        let (w, h) = self.extent();
        let energy = |view: &PlaneView<'_>| ac_energy(self.dsp, self.size, view);
        if let Some(pixels) = pixels {
            return energy(&PlaneView::new(pixels, w, w, h).expect("prediction covers the block"));
        }
        if !self.allows(mv) {
            return 0;
        }
        let (dx, dy) = mv.full_pel();
        let (x, y) = (
            self.block.x as isize + dx as isize,
            self.block.y as isize + dy as isize,
        );
        if let Some(view) = self.plane.region(x, y, w, h) {
            energy(&view)
        } else {
            let gathered = clamped_window(self.plane, x, y, w, h);
            energy(&PlaneView::new(&gathered, w, w, h).expect("window covers its samples"))
        }
    }

    /// Residual statistics of the block at each of `mvs` (the cross of one
    /// refinement stage): [`Self::measure`] of each, measured in one batched
    /// call when all four are ordinary bilinear windows inside the padded
    /// reference.
    pub(crate) fn measure_four(&self, mvs: [Mv; 4]) -> [Variance; 4] {
        if let [Some(a), Some(b), Some(c), Some(d)] = mvs.map(|mv| self.bilinear_window(mv))
            && let Some(stats) = self
                .dsp
                .halfpel_cross_variance(self.size, &self.padded, &self.src, [a, b, c, d])
                .or_else(|| self.windows.at_four([a, b, c, d]))
        {
            self.count_positions(4);
            stats
        } else {
            self.measure_four_fallback(mvs)
        }
    }

    /// A refresh boundary, sampled whole-pel position or clamped reference
    /// needs the individual measurement for each vector.
    #[cold]
    #[inline(never)]
    fn measure_four_fallback(&self, mvs: [Mv; 4]) -> [Variance; 4] {
        mvs.map(|mv| self.measure(mv))
    }

    /// The bilinear window `(x, y, fx, fy)` [`Self::measure`] reads in the
    /// padded reference for `mv`, when it measures `mv` that way and the
    /// window starts inside the plane: allowed, and not a whole-sample
    /// vector it measures on even rows only. (The window may still end
    /// outside the plane.)
    fn bilinear_window(&self, mv: Mv) -> Option<(usize, usize, usize, usize)> {
        let (x, y, fx, fy) = self.window_of(mv);
        let even_rows_copy = fx == 0 && fy == 0 && self.even_rows.is_some();
        if !self.allows(mv) || even_rows_copy {
            return None;
        }
        let (x, y) = padded_position(x, y)?;
        Some((x, y, fx, fy))
    }

    /// Top-left of `mv`'s (w + 1) x (h + 1) bilinear window in the padded
    /// reference and its eighth-sample phases.
    fn window_of(&self, mv: Mv) -> (isize, isize, usize, usize) {
        let (dx, dy) = mv.full_pel();
        // Bilinear kernel offsets are in 1/8 samples.
        let to_eighths = (SUBPEL_VARIANCE_POSITIONS as i32) / MV_UNITS_PER_SAMPLE;
        let (phase_x, phase_y) = mv.fraction();
        (
            self.origin.0 + dx as isize,
            self.origin.1 + dy as isize,
            (phase_x * to_eighths) as usize,
            (phase_y * to_eighths) as usize,
        )
    }

    /// Whether the (w + 1) x (h + 1) window at (`x`, `y`) lies in the
    /// padded reference.
    fn inside(&self, x: isize, y: isize) -> bool {
        x >= 0
            && y >= 0
            && (x as usize) + self.extent().0 < self.padded.width()
            && (y as usize) + self.extent().1 < self.padded.height()
    }

    /// Residual statistics of the four quadrants of the block at `mv`, in
    /// raster order ([`Block::quadrants`]): each equals what the quadrant's
    /// own measurer returns for `mv`, and their sums are this block's
    /// [`Self::measure`]. The planner measures a node this way so that its
    /// children reuse the vectors it shares with them. `None` where that
    /// equality is not plain addition: odd or sub-8 sides, a window outside
    /// the padded reference (clamped copies), intra refresh, row sampling.
    pub(crate) fn measure_quadrants(&self, mv: Mv) -> Option<[Variance; 4]> {
        let (w, h) = self.extent();
        let (half_w, half_h) = (w / 2, h / 2);
        let (x, y, fx, fy) = self.window_of(mv);
        let measurable = self.refresh.is_none()
            && self.even_rows.is_none()
            && half_w >= MIN_QUADRANT_SIDE
            && half_h >= MIN_QUADRANT_SIDE
            && self.inside(x, y);
        if !measurable {
            return None;
        }
        let size = BlockSize::new(half_w, half_h)?;
        let (x, y) = (x as usize, y as usize);
        self.count_positions(1);
        Some(
            [(0, 0), (half_w, 0), (0, half_h), (half_w, half_h)].map(|(dx, dy)| {
                let src = self
                    .src
                    .sub_view(dx, dy, half_w, half_h)
                    .expect("a quadrant lies inside its block");
                self.dsp
                    .prediction_variance(size, &self.padded, x + dx, y + dy, fx, fy, &src)
            }),
        )
    }

    /// Residual statistics of the block at `mv`.
    pub(crate) fn measure(&self, mv: Mv) -> Variance {
        if !self.allows(mv) {
            return FORBIDDEN_STATS;
        }
        self.count_positions(1);
        let (w, h) = self.extent();
        // Every vector reads a (w + 1) x (h + 1) window: within the padded
        // plane it is measured in place, otherwise from a clamped copy.
        let (x, y, fx, fy) = self.window_of(mv);
        if fx == 0 && fy == 0 && self.even_rows.is_some() && self.inside(x, y) {
            let window = self
                .padded
                .sub_view(x as usize, y as usize, w, h)
                .expect("inside the padded plane");
            self.residual_stats(&window)
        } else if let Some(stats) =
            padded_position(x, y).and_then(|(x, y)| self.windows.at(x, y, fx, fy))
        {
            stats
        } else {
            self.measure_clamped(mv, fx, fy)
        }
    }

    /// Copy the edge samples only when the vector exceeds the padded reference.
    #[cold]
    #[inline(never)]
    fn measure_clamped(&self, mv: Mv, fx: usize, fy: usize) -> Variance {
        let (w, h) = self.extent();
        let (dx, dy) = mv.full_pel();
        let x = self.block.x as isize + dx as isize;
        let y = self.block.y as isize + dy as isize;
        let gathered = clamped_window(self.plane, x, y, w + 1, h + 1);
        let window =
            PlaneView::new(&gathered, w + 1, w + 1, h + 1).expect("window covers its samples");
        self.dsp
            .prediction_variance(self.size, &window, 0, 0, fx, fy, &self.src)
    }
}

/// Position (`x`, `y`) of the padded reference as indices, `None` left of or
/// above the plane.
#[inline]
fn padded_position(x: isize, y: isize) -> Option<(usize, usize)> {
    Some((usize::try_from(x).ok()?, usize::try_from(y).ok()?))
}

/// The even rows of `view` (0, 2, …): twice the stride, half the height.
fn even_rows<'v>(view: &PlaneView<'v>) -> PlaneView<'v> {
    PlaneView::new(
        view.data(),
        2 * view.stride(),
        view.width(),
        view.height() / 2,
    )
    .expect("a view's even rows lie inside it")
}

/// A `width` x `height` copy of `plane` at (`x0`, `y0`) with coordinates
/// clamped to the picture (for vectors reaching past the extended border).
fn clamped_window(plane: &Plane, x0: isize, y0: isize, width: usize, height: usize) -> Vec<u8> {
    let view = plane.view();
    let (last_x, last_y) = (view.width() as isize - 1, view.height() as isize - 1);
    let mut out = vec![0u8; width * height];
    for (r, row) in out.chunks_exact_mut(width).enumerate() {
        let src = view.row((y0 + r as isize).clamp(0, last_y) as usize);
        for (c, v) in row.iter_mut().enumerate() {
            *v = src[(x0 + c as isize).clamp(0, last_x) as usize];
        }
    }
    out
}

/// A block of zero samples: the "prediction" whose residual variance is the
/// source's own variance.
static ZEROS: [u8; 64 * 64] = [0; 64 * 64];

/// `n · variance` of the source luma of `block`: the residual variance of a
/// flat (DC) prediction, used as intra's distortion when ranking. The
/// variance kernel against a zero block gives the plain sums of the source.
pub(crate) fn source_variance<B: CodecBackend>(dsp: &Dsp, source: &Picture, block: &Block) -> u64 {
    let (w, h) = leaf_extent::<B>(block);
    let size = BlockSize::new(w, h).expect("partition blocks are kernel sizes");
    let src = source
        .plane(PlaneIndex::Y)
        .region(block.x as isize, block.y as isize, w, h)
        .expect("block inside the picture");
    let zeros = PlaneView::new(&ZEROS, 64, w, h).expect("64x64 covers every block");
    n_variance::<B>(dsp.variance(size, &src, &zeros), block)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn residual_variance(source: &Picture, reference: &Picture, block: &Block, mv: Mv) -> u64 {
        n_variance::<RankingCacheBackend>(
            Measurer::new::<RankingCacheBackend>(&Dsp::new(), source, reference, block).measure(mv),
            block,
        )
    }

    fn picture_with(f: impl Fn(usize, usize) -> u8) -> Picture {
        let mut p = Picture::new(32, 32, 0).unwrap();
        let mut y = p.plane_mut(PlaneIndex::Y).view_mut();
        for r in 0..32 {
            for (c, v) in y.row_mut(r).iter_mut().enumerate() {
                *v = f(r, c);
            }
        }
        p
    }

    #[test]
    fn integer_probe_variance_gate_preserves_disabled_and_strict_boundary() {
        let mut params = DecisionParams::default();
        assert_eq!(params.codec_fullpel_probe_min_lambdas, 0.0);
        assert!(integer_probe_worth_trying(0.0, 2.0, 200.0, &params));
        params.codec_fullpel_probe = 0.0;
        assert!(!integer_probe_worth_trying(1000.0, 2.0, 200.0, &params));
        params.codec_fullpel_probe = 1.0;
        params.codec_fullpel_probe_min_lambdas = 0.25;
        assert!(!integer_probe_worth_trying(99.0, 2.0, 200.0, &params));
        assert!(integer_probe_worth_trying(100.0, 2.0, 200.0, &params));
        assert!(integer_probe_worth_trying(101.0, 2.0, 200.0, &params));
        for (value, valid) in [
            (0.0, true),
            (0.0094, true),
            (64.0, true),
            (-0.1, false),
            (64.1, false),
            (f32::NAN, false),
            (f32::INFINITY, false),
        ] {
            let mut overrides = crate::ParamOverrides::default();
            overrides.set("decision.codec_fullpel_probe_min_lambdas", value);
            assert_eq!(overrides.validate(&[DecisionParams::SPECS]).is_ok(), valid);
        }
    }

    #[test]
    fn integer_probe_skips_overflow_and_keeps_full_pel_coordinates() {
        let block = Block::square(0, 0, 4);
        let max_full_pel = i16::MAX - (MV_UNITS_PER_SAMPLE as i16 - 1);
        for edge in [i16::MIN, max_full_pel] {
            let mut calls = 0;
            let best = best_integer_neighbor::<RankingCacheBackend>(
                Mv { x: edge, y: edge },
                100.0,
                &block,
                |mv| {
                    assert!(mv.is_full_pel());
                    calls += 1;
                    Variance { sum: 0, sse: 100 }
                },
            );
            assert_eq!(calls, 2);
            assert!(best.is_none());
        }
    }

    #[test]
    fn integer_probe_stops_when_zero_variance_cannot_be_improved() {
        let block = Block::square(0, 0, 4);
        assert!(
            best_integer_neighbor::<RankingCacheBackend>(Mv::ZERO, 0.0, &block, |_| {
                panic!("zero-variance seed must not probe")
            })
            .is_none()
        );
        let mut calls = 0;
        let winner = best_integer_neighbor::<RankingCacheBackend>(Mv::ZERO, 100.0, &block, |_| {
            calls += 1;
            Variance { sse: 0, sum: 0 }
        })
        .unwrap();
        assert_eq!(calls, 1);
        assert_eq!(winner.0, Mv::from_full_pel(1, 0));
        assert_eq!(winner.1, Variance { sse: 0, sum: 0 });
    }

    #[test]
    fn integer_probe_keeps_strict_ties_and_returns_measured_statistics() {
        let block = Block::square(0, 0, 4);
        let mut calls = Vec::new();
        let best = best_integer_neighbor::<RankingCacheBackend>(Mv::ZERO, 100.0, &block, |mv| {
            calls.push(mv);
            Variance {
                sum: 0,
                sse: if mv.x == MV_UNITS_PER_SAMPLE as i16 {
                    50
                } else {
                    100
                },
            }
        });
        assert_eq!(calls.len(), 4);
        assert_eq!(
            best,
            Some((
                Mv {
                    x: MV_UNITS_PER_SAMPLE as i16,
                    y: 0
                },
                Variance { sum: 0, sse: 50 }
            ))
        );
        assert!(
            best_integer_neighbor::<RankingCacheBackend>(Mv::ZERO, 100.0, &block, |_| Variance {
                sum: 0,
                sse: 100
            })
            .is_none()
        );
        let defaults = DecisionParams::default();
        assert_eq!(defaults.codec_fullpel_probe, 1.0);
        for (value, valid) in [
            (0.0, true),
            (1.0, true),
            (-0.1, false),
            (1.1, false),
            (f32::NAN, false),
            (f32::INFINITY, false),
        ] {
            let mut params = crate::ParamOverrides::default();
            params.set("decision.codec_fullpel_probe", value);
            assert_eq!(params.validate(&[DecisionParams::SPECS]).is_ok(), valid);
        }
    }

    #[test]
    fn codec_seed_gate_preserves_disabled_and_threshold_boundary_search() {
        let mut params = DecisionParams::from_overrides(&crate::params::ParamOverrides::default());
        let start = (64.0, 64.0, Mv::ZERO);
        let mut calls = 0;
        let mut count = |_| {
            calls += 1;
            (100.0, 100.0)
        };
        refine_codec_seed(start, 1.0, 64.0, &params, &mut count, PresetQuarter);
        assert!(calls > 0);
        params.codec_subpel_seed_lambdas = 2.0;
        let bypassed = |_| panic!("seed should bypass search");
        let kept = refine_codec_seed(start, 1.0, 64.0, &params, bypassed, PresetQuarter);
        assert_eq!(kept, start);
        params.codec_subpel_seed_lambdas = 1.0;
        calls = 0;
        let mut count = |_| {
            calls += 1;
            (100.0, 100.0)
        };
        refine_codec_seed(start, 1.0, 64.0, &params, &mut count, PresetQuarter);
        assert!(calls > 0);
        let flat = |_| (100.0, 100.0);
        refine_codec_seed(start, 0.0, 64.0, &params, flat, PresetQuarter);
    }

    #[test]
    fn quarter_walk_reaches_a_diagonal_from_an_updated_winner() {
        let mut params = DecisionParams::from_overrides(&crate::params::ParamOverrides::default());
        let cost = |mv: Mv| {
            let value = match (mv.x, mv.y) {
                (0, 0) => 100.0,
                (-1, 0) => 70.0,
                (-1, -1) => 20.0,
                _ => 200.0,
            };
            (value, value)
        };
        let start = (100.0, 100.0, Mv::ZERO);
        assert_eq!(
            refine_codec_subpel(start, &params, cost, PresetQuarter).2,
            Mv { x: -1, y: 0 }
        );
        params.subpel_walk = 1.0;
        assert_eq!(
            refine_codec_subpel(start, &params, cost, PresetQuarter).2,
            Mv { x: -1, y: -1 }
        );
    }

    #[test]
    fn svt_bias_is_relative_to_the_initial_me_vector_and_only_the_first_move() {
        let mut params = DecisionParams::from_overrides(&crate::params::ParamOverrides::default());
        params.subpel_walk = 2.0;
        let mut calls = 0;
        let cost = |mv: Mv| {
            calls += 1;
            let value = match (mv.x, mv.y) {
                (6, 0) => 95.0,
                (2, 0) => 80.0,
                (4, 2) => 76.0,
                (5, 2) => 75.0,
                (5, 3) => 74.0,
                _ => 200.0,
            };
            (value, value)
        };
        let start = (100.0, 100.0, Mv { x: 4, y: 0 });
        let result = refine_codec_subpel(start, &params, cost, PresetQuarter);
        assert_eq!(result, (74.0, 74.0, Mv { x: 5, y: 3 }));
        assert_eq!(calls, 6);
    }

    /// A switch that always takes one turn, and keeps what it is told.
    struct FixedTurn {
        turn: Turn,
        recorded: Vec<(f32, f32)>,
    }

    impl QuarterSwitch for &mut FixedTurn {
        fn turn(&mut self) -> Turn {
            self.turn
        }

        fn record(&mut self, before: f32, kept: f32) {
            self.recorded.push((before, kept));
        }
    }

    #[test]
    fn a_switched_quarter_stage_is_measured_or_replaced_by_the_predicted_step() {
        let mut params = DecisionParams::from_overrides(&crate::params::ParamOverrides::default());
        params.subpel_walk = 1.0;
        params.subpel_move_bias = 1.0;
        // The half stage does not move; its costs lean towards -x, where the
        // quarter neighbour is the best vector.
        let value = |mv: Mv| match (mv.x, mv.y) {
            (0, 0) => 100.0,
            (-2, 0) => 120.0,
            (2, 0) => 300.0,
            (-1, 0) => 90.0,
            _ => 400.0,
        };
        let start = (100.0, 100.0, Mv::ZERO);
        let run = |params: &DecisionParams, turn: Turn| {
            let mut calls = 0;
            let mut switch = FixedTurn {
                turn,
                recorded: Vec::new(),
            };
            let cost = |mv: Mv| {
                calls += 1;
                (value(mv), value(mv))
            };
            let best = refine_codec_subpel(start, params, cost, &mut switch);
            (best.2, calls, switch.recorded)
        };
        let found = Mv { x: -1, y: 0 };
        // Measured with no prediction: the cheap alternative loses the gain.
        assert_eq!(run(&params, Turn::Measure), (found, 7, vec![(100.0, 90.0)]));
        // No step predicted: the cheap turn stops at the half stage.
        assert_eq!(run(&params, Turn::Cheap), (Mv::ZERO, 4, vec![]));
        // The parabola predicts the step: one probe finds it, and a
        // measured turn records that the alternative loses nothing.
        params.subpel_quarter_predict = 1.0;
        assert_eq!(run(&params, Turn::Cheap), (found, 5, vec![]));
        assert_eq!(
            run(&params, Turn::Measure),
            (found, 7, vec![(100.0, 100.0)])
        );
        assert_eq!(run(&params, Turn::Preset), (found, 7, vec![]));
    }

    #[test]
    fn exact_shift_has_zero_residual_variance() {
        let reference = picture_with(|r, c| ((r * 7 + c * 13) % 251) as u8);
        // The source is the reference moved left by 3, up by 2.
        let source = picture_with(|r, c| (((r + 2) * 7 + (c + 3) * 13) % 251) as u8);
        let block = Block::square(8, 8, 3);
        assert_eq!(
            residual_variance(&source, &reference, &block, Mv::from_full_pel(3, 2)),
            0
        );
        assert!(residual_variance(&source, &reference, &block, Mv::ZERO) > 0);
    }

    #[test]
    fn out_of_picture_references_are_clamped() {
        let flat = picture_with(|_, _| 77);
        let block = Block::square(0, 0, 3);
        assert_eq!(
            residual_variance(&flat, &flat, &block, Mv::from_full_pel(-40, -40)),
            0
        );
    }
    struct RankingCacheBackend {
        saved: Option<(Mv, Vec<u8>, Variance)>,
    }
    impl CodecBackend for RankingCacheBackend {
        const CACHE_EXACT_LUMA: bool = true;
        fn new(_: &crate::config::EncoderConfig) -> Result<Self, crate::config::ConfigError> {
            Ok(Self { saved: None })
        }
        fn superblock_log2(&self) -> u8 {
            6
        }
        fn min_block_log2(&self) -> u8 {
            3
        }
        fn luma_filter(&self) -> Option<Filter> {
            Some(Filter::Av1)
        }
        fn mv_candidates(&mut self, _: &Block, out: &mut Vec<Mv>) {
            out.push(Mv { x: -1, y: 0 });
        }
        fn mode_bits(&self, _: &Block, mode: &PredictionMode) -> f32 {
            if *mode == PredictionMode::Inter(Mv { x: -1, y: 0 }) {
                0.0
            } else {
                1e9
            }
        }
        fn cache_exact_luma(
            &mut self,
            _: &Block,
            mv: Mv,
            _: Filter,
            pixels: &[u8],
            stats: Variance,
        ) {
            self.saved = Some((mv, pixels.to_vec(), stats));
        }
        fn begin_frame(&mut self, _: &crate::backend::FramePlan) {}
        fn begin_superblock(&mut self, _: usize, _: usize) {}
        fn write_partition(&mut self, _: &Block, _: bool) {}
        fn encode_block(
            &mut self,
            _: &Block,
            _: &crate::backend::BlockDecision,
            _: &Picture,
            _: Option<&Picture>,
            _: &mut Picture,
        ) {
        }
        fn end_frame(&mut self) -> Vec<u8> {
            Vec::new()
        }
    }

    #[test]
    fn cached_prediction_tracks_rate_weighted_winner_not_lowest_distortion() {
        let reference = picture_with(|y, x| ((x * 13 + y * 7 + (x ^ y) * 3) % 251) as u8);
        let source = picture_with(|y, x| (((x + 1) * 13 + y * 7 + ((x + 1) ^ y) * 3) % 251) as u8);
        let dsp = Dsp::new();
        let block = Block::square(8, 8, 3);
        let measurer = Measurer::new::<RankingCacheBackend>(&dsp, &source, &reference, &block);
        assert_eq!(measurer.measure(Mv { x: 4, y: 0 }).sse, 0);
        let mut backend = RankingCacheBackend { saved: None };
        let params = DecisionParams::from_overrides(&crate::params::ParamOverrides::default());
        let choice = choose_inter_mode(
            &dsp,
            &mut backend,
            &block,
            (&source, &reference, None),
            Mv { x: 4, y: 0 },
            None,
            &mut DecisionScratch::default(),
            (None, false),
            &params,
            None,
        );
        assert_eq!(
            choice.mode.prediction(),
            PredictionMode::Inter(Mv { x: -1, y: 0 })
        );
        let (mv, pixels, stats) = backend.saved.unwrap();
        assert_eq!(mv, Mv { x: -1, y: 0 });
        measurer.with_exact_prediction(mv, Filter::Av1, |expected_stats, expected_pixels, _| {
            assert_eq!(stats, expected_stats);
            assert_eq!(pixels, expected_pixels);
        });
    }

    #[test]
    fn subpel_and_skip_shifts_are_exact_at_zero_and_linear_when_active() {
        let mut params = DecisionParams::from_overrides(&crate::params::ParamOverrides::default());
        params.codec_subpel_seed_lambdas = 0.03;
        params.skip_no_refine_lambdas = 0.25;
        assert_eq!(
            effective_subpel_seed_gate(256.0, 33.0, Mv::ZERO, &params),
            0.03
        );
        assert_eq!(effective_skip_no_refine(33.0, &params), 0.25);

        params.subpel_seed_shift_bias = 0.02;
        params.subpel_seed_shift_size = 0.04; // 32x32 (area=1024): (10 - 8)*0.25 = 0.5 -> +0.02
        params.subpel_seed_shift_zero_mv = 0.01;
        let g = effective_subpel_seed_gate(1024.0, 33.0, Mv::ZERO, &params);
        assert!((g - 0.08).abs() < 1e-6, "got {g}");
        let g_mov = effective_subpel_seed_gate(1024.0, 33.0, Mv { x: 4, y: 0 }, &params);
        assert!((g_mov - 0.07).abs() < 1e-6, "got {g_mov}");

        params.skip_no_refine_shift_bias = -0.5;
        assert_eq!(effective_skip_no_refine(33.0, &params), 0.0);
    }
}
