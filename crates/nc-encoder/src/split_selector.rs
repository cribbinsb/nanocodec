//! Zero-trial split selector (`planner.split_selector`): a learned shift of
//! the inter planner's parent bias (`partition_plan.rs`), from evidence the
//! planner already holds when it weighs a node's leaf against its quad
//! split.
//!
//! Why: the RD partition search (`rd_split.rs`) codes a node both ways and
//! overturns about a quarter of the planner's leaf-vs-split choices, and its
//! verdicts are an offline label of the right answer. On those labels the
//! planner's errors are not a global bias (a lower parent bias, fewer
//! splits, lost fixed QP J +3..+7 %), but they follow the node's content:
//! where much residual is left at the node's search vector the planner's
//! per-sample distortion cap (`planner.distortion_cap_per_lambda`) hides
//! what the children would gain, so it keeps leaves the coded cost splits;
//! where the motion search's whole-block SAD is close to its four children's
//! and the residual's mean dominates, it splits nodes the coded cost keeps.
//!
//! How: the planner keeps the leaf when `bias · leaf ≤ 1000 · split`; with the
//! selector it compares `bias · e^h · leaf` instead, where `h` (in units of
//! the log cost ratio) is linear in a few node features:
//! - the residual variance per sample at the node's search vector, over λ;
//! - its mean residual energy per sample, over λ;
//! - the quadrants' summed variance at that vector over the node's (what
//!   separate means gain) and their largest over smallest (imbalance);
//! - the four quadrants' search vectors' distance from the node's;
//! - the motion search's split over whole SAD at the node's size;
//! - whether the node is 64x64.
//!
//! All are known before the node's children are planned (the quadrant
//! statistics come from the node's own measurement, split over its
//! quadrants), so the planner's exact early exit stays exact under the
//! shifted bias. The selector is content-only: it reads no neighbour or
//! co-located depths, so it adds no spatial or temporal feedback and no
//! dependency a wavefront (left/above) order would not have.
//!
//! Fit (docs/handoff/planner-selector.md): the coefficients minimise the
//! recursive coded-cost regret against the RD search's verdicts (Default
//! preset, quad search one level either side, every candidate coded in
//! full) on one half of a 56-clip pool sample split by clip family, HEVC,
//! AV1 and VVC weighted equally; reported on the other half.

use nc_dsp::Variance;

use crate::params::PlannerParams;
use nc_analysis::linear_threshold::LinearThreshold;

/// Coefficients of `h`, in units of the log of the leaf-vs-split cost ratio.
/// Shared by every codec (fit on HEVC, AV1 and VVC together).
mod fit {
    pub const INTERCEPT: f32 = -0.186;
    /// Per unit of `ln(1 + residual variance per sample / λ)`.
    pub const RESIDUAL: f32 = 0.476;
    /// Per unit of `ln(1 + mean residual energy per sample / λ)`.
    pub const MEAN: f32 = -0.294;
    /// Per unit of `ln(quadrant variance sum / node variance)`.
    pub const QUADRANT_GAIN: f32 = 0.843;
    /// Per unit of `ln(largest / smallest quadrant variance)`.
    pub const QUADRANT_IMBALANCE: f32 = 0.040;
    /// Per unit of `ln(1 + quadrant vector distance)` (quarter samples).
    pub const VECTOR_DISTANCE: f32 = -0.032;
    /// Per unit of `ln(split SAD / whole SAD)` of the motion search.
    pub const SEARCH_SPLIT_GAIN: f32 = -0.528;
    /// For a 64x64 node.
    pub const NODE_64: f32 = 0.096;
}

/// Regularises variance ratios of near-flat residuals (per sample), as in
/// the fit.
const VARIANCE_FLOOR: f32 = 1e-3;
/// Regularises the quadrant imbalance ratio (per sample).
const IMBALANCE_FLOOR: f32 = 1.0;
/// Regularises the SAD ratio.
const SAD_FLOOR: f32 = 1.0;
/// The largest node log2 size (64x64).
const LOG2_64: u8 = 6;

/// What the planner knows about a square node when it weighs it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SplitEvidence {
    pub log2: u8,
    /// The backend's λ (mode decision, not the planner's multiple).
    pub lambda: f32,
    /// Luma residual of the node at its search vector.
    pub leaf: Variance,
    /// The quadrants' residuals at the node's search vector, where the node's
    /// own measurement gave them (`None`: their terms are left out).
    pub quadrants: Option<[Variance; 4]>,
    /// Sum over the quadrants of the L1 distance (quarter samples) of their
    /// search vectors from the node's.
    pub vector_distance: u32,
    /// The motion search's (whole, split) SAD at the node's size.
    pub search_sads: Option<(u32, u32)>,
}

/// Corrections to `h` tuned on encodes (`planner.split_selector_d_*`,
/// `linear_threshold.rs`): a delta on each fitted coefficient, plus terms the
/// fit lacks (a 32x32 node, the frame's λ). All 0 is the fitted selector
/// exactly (no correction is built). Per frame role through the
/// `nonbase.planner.split_selector_d_*` overlay (`frame_role.rs`).
pub(crate) type SelectorTerms = LinearThreshold<SELECTOR_FEATURES>;

/// The correction's features, in [`SplitEvidence::features`] order.
pub(crate) const SELECTOR_FEATURES: usize = 9;
/// The node log2 size of the 32x32 term.
const LOG2_32: u8 = 5;

/// The selector's correction from `params`, `None` while all terms are 0.
pub(crate) fn selector_terms(params: &PlannerParams) -> Option<SelectorTerms> {
    LinearThreshold::new(
        params.split_selector_d_bias,
        [
            params.split_selector_d_residual,
            params.split_selector_d_mean,
            params.split_selector_d_quadrant_gain,
            params.split_selector_d_quadrant_imbalance,
            params.split_selector_d_vector_distance,
            params.split_selector_d_search_split_gain,
            params.split_selector_d_node_64,
            params.split_selector_d_node_32,
            params.split_selector_d_lambda,
        ],
    )
}

impl SplitEvidence {
    /// The factor on the parent bias: `e^(scale · (h + terms))`.
    pub fn bias_factor(&self, scale: f32, terms: Option<&SelectorTerms>) -> f32 {
        let h = match terms {
            None => self.shift(),
            Some(terms) => self.shift() + terms.at(&self.features()),
        };
        (scale * h).exp()
    }

    /// The fitted terms' features (0 where the fit leaves a term out), then
    /// the 32x32 flag and `ln λ`.
    fn features(&self) -> [f32; SELECTOR_FEATURES] {
        let log2_area = 2 * u32::from(self.log2);
        let area = (1_u32 << log2_area) as f32;
        let per_lambda = |energy: f32| energy / area / self.lambda.max(f32::MIN_POSITIVE);
        let node_var = self.leaf.n_variance_pow2(log2_area) as f32;
        let mean = (self.leaf.sse as f32 - node_var).max(0.0);
        let (gain, imbalance) = self.quadrants.map_or((0.0, 0.0), |quadrants| {
            let log2_quarter = log2_area - 2;
            let quarter = (1_u32 << log2_quarter) as f32;
            let vars = quadrants.map(|q| q.n_variance_pow2(log2_quarter) as f32 / quarter);
            let sum: f32 = vars.iter().map(|v| v * quarter).sum();
            let (low, high) = vars
                .iter()
                .fold((f32::MAX, 0.0f32), |(lo, hi), &v| (lo.min(v), hi.max(v)));
            (
                ((sum / area + VARIANCE_FLOOR) / (node_var / area + VARIANCE_FLOOR)).ln(),
                ((high + IMBALANCE_FLOOR) / (low + IMBALANCE_FLOOR)).ln(),
            )
        });
        let search = self.search_sads.map_or(0.0, |(whole, split)| {
            ((split as f32 + SAD_FLOOR) / (whole as f32 + SAD_FLOOR)).ln()
        });
        [
            per_lambda(node_var).ln_1p(),
            per_lambda(mean).ln_1p(),
            gain,
            imbalance,
            (self.vector_distance as f32).ln_1p(),
            search,
            f32::from(u8::from(self.log2 >= LOG2_64)),
            f32::from(u8::from(self.log2 == LOG2_32)),
            self.lambda.max(f32::MIN_POSITIVE).ln(),
        ]
    }

    /// `h`: positive favours the split.
    fn shift(&self) -> f32 {
        let log2_area = 2 * u32::from(self.log2);
        let area = (1_u32 << log2_area) as f32;
        let per_lambda = |energy: f32| energy / area / self.lambda.max(f32::MIN_POSITIVE);
        let node_var = self.leaf.n_variance_pow2(log2_area) as f32;
        let mean = (self.leaf.sse as f32 - node_var).max(0.0);
        let mut h = fit::INTERCEPT
            + fit::RESIDUAL * per_lambda(node_var).ln_1p()
            + fit::MEAN * per_lambda(mean).ln_1p()
            + fit::VECTOR_DISTANCE * (self.vector_distance as f32).ln_1p();
        if let Some(quadrants) = self.quadrants {
            let log2_quarter = log2_area - 2;
            let quarter = (1_u32 << log2_quarter) as f32;
            let vars = quadrants.map(|q| q.n_variance_pow2(log2_quarter) as f32 / quarter);
            let sum: f32 = vars.iter().map(|v| v * quarter).sum();
            let (low, high) = vars
                .iter()
                .fold((f32::MAX, 0.0f32), |(lo, hi), &v| (lo.min(v), hi.max(v)));
            h += fit::QUADRANT_GAIN
                * ((sum / area + VARIANCE_FLOOR) / (node_var / area + VARIANCE_FLOOR)).ln()
                + fit::QUADRANT_IMBALANCE
                    * ((high + IMBALANCE_FLOOR) / (low + IMBALANCE_FLOOR)).ln();
        }
        if let Some((whole, split)) = self.search_sads {
            h += fit::SEARCH_SPLIT_GAIN
                * ((split as f32 + SAD_FLOOR) / (whole as f32 + SAD_FLOOR)).ln();
        }
        if self.log2 >= LOG2_64 {
            h += fit::NODE_64;
        }
        h
    }
}
