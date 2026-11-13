//! Inter-frame block partitioning: SVT-AV1's "very light PD0" (VLPD0).
//!
//! `docs/reference/svt-av1-m12-rtc.md` §2. Per superblock, before any block
//! is coded:
//!
//! 1. **Size cap.** The largest block is 32x32 instead of 64x64 when the
//!    spread of the superblock's 8x8 motion-search costs is high
//!    (`get_max_block_size_rtc`: `me_8x8_cost_variance > 50000 · q_weight`).
//!    Past the codec's largest quantizer (`FramePlan::lambda_scale` > 1) the
//!    threshold grows with λ, and the parent bias below keeps falling with
//!    the planned quantizer (see "Rate awareness").
//! 2. **Depth removal.** Splitting below 32x32 is not even considered when
//!    the motion-search costs barely improve from 32x32 to 16x16 to 8x8
//!    (`set_depth_removal_level_controls`, level 7; the relative-improvement
//!    thresholds 25‰ and 50‰ scaled as SVT scales them).
//! 3. **Cost per block**: the variance of the whole-sample motion-compensated
//!    prediction from the *reconstructed* reference (best of the block's
//!    search vector and zero), plus its mean error up to the cost of coding
//!    the mean (`planner.dc_bits` λ), capped per sample at λ/8, plus λ · the
//!    bits of `PARTITION_NONE` (`compute_lpd0_cost_inter`, which counts the
//!    variance alone).
//! 4. **Split decision**, bottom-up: keep the parent if
//!    `bias · parent ≤ 1000 · (children + λ · 2 · bits(PARTITION_SPLIT))`,
//!    `bias = 1100 − 150 · q + {25, 50, 75}` by 8x8 cost spread, clipped to
//!    900..1200 (`test_split_partition_pd0`, `svt_aom_sig_deriv_enc_dec_pd0`).
//!
//! Inter blocks are 64, 32 or 16 square; 8x8 only where the frame edge
//! forces it (SVT disallows 8x8 from M10 up). `planner.inter_min_log2`
//! moves that floor: 8x8 for a quality-first preset, 32x32 for a fast one.
//!
//! ## Rate awareness
//!
//! SVT's variance treats each leaf's mean residual as free. Four children
//! then always "gain" the differences between their means, though at coarse
//! quantizers those means stay uncoded (skipped blocks) or cost a DC
//! coefficient each. `FoodMarket` at 100 kbps (AV1, 300 frames): at 720p the
//! children's variance was 12% below the parent's at the median while their
//! squared error was equal (same vectors), so 78% of the area went to 32x32
//! blocks: four times the blocks 360p codes for the same picture area, and
//! 0.75 dB below 360p upscaled. Charging the mean up to `dc_bits` λ (4: 4 to
//! 64 measured the same), with the λ-aware limits below, took 64x64 from 22%
//! to 64% of the area: PSNR-Y 26.24 → 26.76 dB, MS-SSIM 0.740 → 0.766
//! (360p upscaled: 27.14, 0.788; 360p itself moved ±0.03 dB). At 300 kbps,
//! where the 720p partition already matched 360p's, it is neutral to
//! slightly positive; `ParkJoy` and `CrowdRun` at 100 kbps +0.06 and +0.13 dB.
//! Normal operating range (real-use CBR screen at k 0.1, proxy-6, 150
//! frames): AV1 `J_real` −1.5%, HEVC −0.6%, no guard tripped.
//!
//! SVT's quantizer weighting sees only the coded quantizer, which stops at
//! the codec's largest. Past it the rate control raises λ instead
//! (`FramePlan::lambda_scale`, the squared step ratio): the size-cap
//! threshold is scaled by it, as distortion spreads are compared against
//! λ, and the bias formula's quantizer continues by
//! [`QUANTIZER_PER_STEP_OCTAVE`] per octave of planned step.
//!
//! SVT's units: its `RDCOST(λ, R, D)` weighs a bit at λ/128 squared-error
//! units, so its per-sample distortion cap `λ >> 10` is our λ/8.
//!
//! ## Rectangular shapes
//!
//! For a backend that declares rectangular partitions
//! ([`CodecBackend::partition_shapes`]: VVC binary/ternary splits, AV1
//! `PARTITION_HORZ/VERT`), a node that may be a leaf also prices its shapes
//! with the same leaf-cost model: each rectangular leaf's capped
//! prediction variance at its own best whole-sample vector, plus λ times the
//! shape's partition bits and a leaf's mode bits per leaf. A shape replaces
//! the square plan when it is cheaper than the plan it competes with: a
//! leaf weighted by the parent bias (a shape is a kind of split), or the
//! quad split's cost. Cheap in the real-time way (SVT-AV1 prunes its
//! non-square shapes with similar gates; `VVenC`'s fast MTT modes likewise):
//!
//! - nothing is measured for static nodes, nodes the planner may not code
//!   as leaves, or shapes whose leaves would be smaller than
//!   `planner.rect_min_log2`;
//! - a shape is measured only when the motion search separates its leaves
//!   (`planner.rect_motion_gate`: every search vector of one leaf differs
//!   from every one of the next by that many samples): leaves predicted by
//!   about one vector rarely repay the extra block's bits;
//! - its measurement stops as soon as the summed cost (times
//!   `planner.rect_bias`) reaches the plan it competes with (costs are
//!   nonnegative, so this is exact).
//!
//! The variance model overrates shapes: it misses the transform, which codes
//! smooth residual cheaply, and on whole samples a second vector gains what
//! the square leaf's sub-sample refinement would gain anyway. With
//! `planner.rect_rd` the backend's coded cost
//! ([`CodecBackend::inter_leaf_cost`], at sub-sample refined vectors)
//! confirms each planned shape against the square plan: the variance model
//! proposes, the coded cost decides.

use std::cell::OnceCell;

use nc_analysis::MotionAnalysis;
use nc_analysis::preanalysis::PreAnalysis;
use nc_common::Picture;
use nc_dsp::{Dsp, Variance};

use crate::backend::CodecBackend;
use crate::decision::{Measurer, is_square_leaf, is_static, leaf_area, n_variance, refined_vector};
use crate::encoder::{MAX_TILING_SQUARES, search_vector, search_vectors};
use crate::mv::{MV_UNITS_PER_SAMPLE, Mv};
use crate::params::PlannerParams;
use crate::partition::Block;
use crate::shape::{MAX_SHAPE_PARTS, PartitionShape, ShapeSet};

/// Largest block when the 8x8 cost spread is high (32x32).
const CAPPED_MAX_LOG2: u8 = 5;
// The planner's tunable thresholds and multipliers are `PlannerParams`
// (`params.rs`); the notes below explain their hand-tuned defaults.
//
// `lambda_scale` (default 2): the planner weighs bits and caps distortion
// with the backend's λ times this. With the mode-decision λ itself the
// planner split far more than SVT (low-light: 79% of the area in 32x32
// blocks against SVT's 78% in 64x64), spending coefficient bits on smaller
// transforms. Quick clips, 10 s IP, PSNR-YUV BD vs svtav1-m12 with ×2
// against ×1: AV1 −3.2 points overall (crowd −8, dashcam −7, low-light −4,
// static +3), HEVC −3.2 points; ×4 measured the same as ×2.
// `distortion_cap_per_lambda` (1/8): per-sample distortion cap as a
// fraction of λ (SVT `λ >> 10`). `split_rate_factor` (2): SVT counts the
// split symbol twice (`use_accurate_part_ctx = false`). `bias_*`: the
// parent-bias formula (‰). `high_8x8_cost_variance`: SVT
// `HIGH_8x8_DIST_VAR_TH`. `depth_removal_*`: SVT depth-removal thresholds
// and its `qp_scale_factor`.

/// Parent-bias additions (‰) for an increasing 8x8 cost spread.
const BIAS_SPREAD_STEPS: [(f32, f32); 3] = [(500.0, 25.0), (1000.0, 50.0), (2000.0, 75.0)];
/// SVT quantiser scale (0..63) the formulas above are written in.
const SVT_QP_MAX: f32 = 63.0;
/// Relative quantizer (0..1 over the codec's range) per octave of step near
/// the top of the range: H.264/HEVC 6/51 exactly; AV1 about 36/255 near
/// qindex 255, VVC 6/63. Continues the parent-bias formula past the largest
/// quantizer.
const QUANTIZER_PER_STEP_OCTAVE: f32 = 6.0 / 51.0;

/// The chosen partition of a node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Partition {
    /// Code the node as one block, with the residual statistics measured
    /// for it while planning (`None` for blocks the planner did not cost).
    Leaf(Option<LeafStats>),
    /// Split into four quadrants (raster order).
    Split(Box<[Partition; 4]>),
    /// Coded as the rectangular leaves of a shape, with their measured
    /// statistics (coding order; only the shape's part count are used).
    Shaped(PartitionShape, Box<[Option<LeafStats>; MAX_SHAPE_PARTS]>),
}

/// Residual statistics of a planned leaf at its whole-sample search vector
/// and at the zero vector, reused by mode decision instead of measured
/// again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LeafStats {
    /// The motion-search vector `search` was measured at: the block's
    /// [`search_vector`], or for a rectangle the best of its tiling
    /// squares' vectors ([`search_vectors`]).
    pub search_mv: Mv,
    pub search: Variance,
    pub zero: Variance,
}

/// Everything the planner needs for one frame.
pub(crate) struct Planner<'a, B: CodecBackend> {
    pub dsp: &'a Dsp,
    pub backend: &'a B,
    pub source: &'a Picture,
    pub reference: &'a Picture,
    pub motion: Option<&'a MotionAnalysis>,
    /// Static map (`None` on the first inter frame).
    pub pre: Option<&'a PreAnalysis>,
    /// Coded picture size.
    pub width: usize,
    pub height: usize,
    /// Tunable thresholds.
    pub params: &'a PlannerParams,
    /// Whether rectangular shapes may be planned (not under intra refresh).
    pub shapes: bool,
    /// The frame's [`FramePlan::lambda_scale`]: λ's growth past the codec's
    /// largest quantizer (1 within its range).
    ///
    /// [`FramePlan::lambda_scale`]: crate::backend::FramePlan::lambda_scale
    pub lambda_headroom: f32,
    /// [`Self::leaf_bits`], computed on first use: the backend prices it
    /// from about thirty CDFs, which do not change while one superblock is
    /// planned, and every costed node needs it (re-pricing it per node was
    /// 1.7% of AV1 Faster's P-frame instructions).
    pub leaf_bits: OnceCell<f32>,
}

/// Residual statistics of a node already measured at up to two vectors
/// (its parent's search vector and the zero vector), from the parent's
/// quadrant measurements ([`Measurer::measure_quadrants`]).
#[derive(Debug, Clone, Copy)]
struct Known([Option<(Mv, Variance)>; 2]);

impl Known {
    const NONE: Self = Self([None; 2]);

    /// The statistics at `mv`, if known.
    fn get(&self, mv: Mv) -> Option<Variance> {
        self.0
            .iter()
            .flatten()
            .find(|(at, _)| *at == mv)
            .map(|&(_, stats)| stats)
    }
}

/// The statistics of a block from those of its quadrants.
fn sum_of(parts: [Variance; 4]) -> Variance {
    parts
        .iter()
        .fold(Variance { sse: 0, sum: 0 }, |total, part| Variance {
            sse: total.sse + part.sse,
            sum: total.sum + part.sum,
        })
}

/// Per-superblock limits derived from the motion statistics.
#[derive(Debug, Clone, Copy)]
struct Limits {
    max_log2: u8,
    /// Smallest block allowed inside the frame (`inter_min_log2`, or 32 after
    /// depth removal).
    min_log2: u8,
    bias: f32,
}

impl<B: CodecBackend> Planner<'_, B> {
    /// Plan the superblock `sb`.
    pub fn plan(&self, sb: Block) -> Partition {
        let limits = self.limits(&sb);
        self.plan_node(sb, limits, Known::NONE).1
    }

    /// (cost, partition) of the cheapest way to code `node`.
    /// `known` holds statistics of `node` its parent already measured.
    fn plan_node(&self, node: Block, limits: Limits, known: Known) -> (f32, Partition) {
        let inside = node.is_inside(self.width, self.height);
        let min_block = self.backend.min_block_log2();
        let can_split =
            node.log2_size() > min_block && (!inside || node.log2_size() > limits.min_log2);
        let may_be_leaf = inside && node.log2_size() <= limits.max_log2;
        let lambda = self.backend.lambda() * self.params.lambda_scale;

        // A static node is never split: splitting only spends partition and
        // mode bits on blocks whose residual quantises to zero anyway.
        let mut children_known = [Known::NONE; 4];
        let leaf = may_be_leaf.then(|| {
            let quadrants = can_split.then_some(&mut children_known);
            let (distortion, stats) = self.distortion_reusing(&node, known, quadrants);
            (
                distortion
                    + lambda * (self.backend.partition_bits(&node, false) + self.leaf_bits()),
                stats,
            )
        });
        let is_static = leaf.is_some() && self.pre.is_some_and(|pre| is_static::<B>(pre, &node));
        if is_static {
            return leaf.map_or((0.0, Partition::Leaf(None)), |(cost, stats)| {
                (cost, Partition::Leaf(Some(stats)))
            });
        }
        if !can_split {
            return match leaf {
                Some((cost, stats)) => {
                    self.or_shape(&node, limits, cost, (cost, Partition::Leaf(Some(stats))))
                }
                None => (0.0, Partition::Leaf(None)),
            };
        }
        let leaf_cost = leaf.map(|(cost, _)| cost);
        let mut split_cost =
            lambda * self.params.split_rate_factor * self.backend.partition_bits(&node, true);
        let mut children = std::array::from_fn(|_| Partition::Leaf(None));
        for (i, child) in node.quadrants().into_iter().enumerate() {
            // All remaining distortion/rate costs are nonnegative. Preserve
            // the full search's tie rule while avoiding children that cannot
            // improve its decision (SVT's depth early exit).
            // Isolated legal-rate P probes: HEVC −1.1%, AV1 −1.2%, with
            // byte-identical streams and therefore zero BD-rate change.
            if let Some(cost) = leaf_cost
                && limits.bias * cost <= 1000.0 * split_cost
            {
                let best = (cost, Partition::Leaf(leaf.map(|(_, stats)| stats)));
                return self.or_shape(&node, limits, cost, best);
            }
            if child.x >= self.width || child.y >= self.height {
                // Outside the frame: no syntax, no cost.
                continue;
            }
            let (cost, plan) = self.plan_node(child, limits, children_known[i]);
            split_cost += cost;
            children[i] = plan;
        }
        match leaf_cost {
            Some(cost) if limits.bias * cost <= 1000.0 * split_cost => {
                let best = (cost, Partition::Leaf(leaf.map(|(_, stats)| stats)));
                self.or_shape(&node, limits, cost, best)
            }
            Some(cost) => self.or_shape(
                &node,
                limits,
                cost,
                (split_cost, Partition::Split(Box::new(children))),
            ),
            None => (split_cost, Partition::Split(Box::new(children))),
        }
    }

    /// `best`, the square plan of leaf-capable `node` (whose leaf costs
    /// `leaf_cost`), or the cheapest rectangular shape of the node that
    /// beats it (module docs, "Rectangular shapes").
    fn or_shape(
        &self,
        node: &Block,
        limits: Limits,
        leaf_cost: f32,
        best: (f32, Partition),
    ) -> (f32, Partition) {
        if !B::RECTANGULAR_LEAVES || !self.shapes {
            return best;
        }
        let shapes = self.backend.partition_shapes(node);
        if shapes.is_empty() || node.log2_size() < limits.min_log2 {
            return best;
        }
        // A shape competes like a split: against the leaf with the parent
        // bias, or against the split's own cost.
        let mut bound = match best.1 {
            Partition::Leaf(_) if self.params.rect_against_leaf < 0.5 => return best,
            Partition::Leaf(_) => limits.bias * leaf_cost / 1000.0,
            _ => best.0,
        };
        let lambda = self.backend.lambda() * self.params.lambda_scale;
        let bias = self.params.rect_bias / 1000.0;
        let mut chosen = None;
        for shape in self.worth_measuring(node, shapes) {
            let parts = shape.parts(node);
            let parts = &parts[..shape.part_count()];
            let mut cost = lambda
                * (self.backend.shape_bits(node, shape) + parts.len() as f32 * self.leaf_bits());
            let mut stats = [None; MAX_SHAPE_PARTS];
            for (slot, part) in stats.iter_mut().zip(parts) {
                if bias * cost >= bound {
                    break;
                }
                let (distortion, part_stats) = self.distortion(part);
                cost += distortion;
                *slot = Some(part_stats);
            }
            if bias * cost < bound {
                bound = bias * cost;
                chosen = Some((cost, Partition::Shaped(shape, Box::new(stats))));
            }
        }
        match chosen {
            Some(shaped)
                if self.params.rect_rd < 0.5
                    || self.coded_cost_confirms(node, &best.1, &shaped.1) =>
            {
                shaped
            }
            _ => best,
        }
    }

    /// The shapes of `shapes` worth measuring at `node`: leaves no smaller
    /// than `planner.rect_min_log2` that the motion search separates
    /// (`planner.rect_motion_gate`).
    fn worth_measuring(
        &self,
        node: &Block,
        shapes: ShapeSet,
    ) -> impl Iterator<Item = PartitionShape> + use<'_, B> {
        let min_side = (self.params.rect_min_log2.round() as u8).max(self.backend.min_block_log2());
        let gate = self.params.rect_motion_gate;
        let node = *node;
        shapes.iter().filter(move |shape| {
            let parts = shape.parts(&node);
            shape.log2_min_side(node.log2_size()) >= min_side
                && (gate <= 0.0 || self.motion_separates(&parts[..shape.part_count()], gate))
        })
    }

    /// Whether the backend's rate-distortion cost (`planner.rect_rd`,
    /// [`CodecBackend::inter_leaf_cost`]) confirms the variance model's
    /// choice of `shaped` over the `square` plan of `node`: the shape's
    /// leaves and symbols cost less than the square leaf, or than a quad
    /// split into four leaves (a deeper split is kept). The variance model
    /// proposes, the coded cost decides: it prices a few nodes, and
    /// rejects shapes that save prediction variance the transform would
    /// have coded cheaply anyway. True where the backend has no estimate.
    fn coded_cost_confirms(&self, node: &Block, square: &Partition, shaped: &Partition) -> bool {
        let lambda = self.backend.lambda();
        let leaf = |block: &Block, stats: &LeafStats| {
            self.coded_cost(block, stats)
                .map(|cost| cost + lambda * self.backend.partition_bits(block, false))
        };
        let square_cost = match square {
            Partition::Leaf(Some(stats)) => leaf(node, stats),
            Partition::Split(children) => node.quadrants().iter().zip(children.iter()).try_fold(
                lambda * self.backend.partition_bits(node, true),
                |sum, (child, plan)| match plan {
                    Partition::Leaf(Some(stats)) => leaf(child, stats).map(|cost| sum + cost),
                    _ => None,
                },
            ),
            _ => return false,
        };
        let Partition::Shaped(shape, stats) = shaped else {
            unreachable!("the variance model proposes a shape");
        };
        let shape_cost = shape
            .parts(node)
            .iter()
            .zip(stats.iter())
            .take(shape.part_count())
            .try_fold(
                lambda * self.backend.shape_bits(node, *shape),
                |sum, (part, stats)| {
                    let stats = stats
                        .as_ref()
                        .expect("a proposed shape measured every leaf");
                    self.coded_cost(part, stats).map(|cost| sum + cost)
                },
            );
        // No estimate at all: the variance model decides.
        let Some(shape_cost) = shape_cost else {
            return true;
        };
        square_cost.is_some_and(|square| shape_cost < square)
    }

    /// The backend's rate-distortion cost of leaf `block`
    /// ([`CodecBackend::inter_leaf_cost`]) at its planned search vector
    /// refined to sub-sample precision ([`refined_vector`]): whole-sample
    /// vectors overstate what a second vector gains, since mode decision
    /// refines the square leaf's vector too (pan: a 64x64 leaf
    /// whose motion is fractional splits into halves at neighbouring whole
    /// samples).
    fn coded_cost(&self, block: &Block, stats: &LeafStats) -> Option<f32> {
        let measurer = Measurer::new::<B>(self.dsp, self.source, self.reference, block);
        let mv = refined_vector::<B>(&measurer, block, (stats.search_mv, stats.search));
        self.backend
            .inter_leaf_cost(self.source, self.reference, block, mv)
    }

    /// Whether the motion search separates neighbouring `parts`: every
    /// tiling-square vector of one differs from every one of the next by at
    /// least `samples` whole samples in some component.
    fn motion_separates(&self, parts: &[Block], samples: f32) -> bool {
        let vectors = |part: &Block| {
            let mut out = [Mv::ZERO; MAX_TILING_SQUARES];
            let count = search_vectors(self.motion, part, &mut out);
            (out, count)
        };
        let threshold = (samples * MV_UNITS_PER_SAMPLE as f32).round() as i32;
        parts.windows(2).any(|pair| {
            let ((a, a_count), (b, b_count)) = (vectors(&pair[0]), vectors(&pair[1]));
            a[..a_count].iter().all(|u| {
                b[..b_count].iter().all(|v| {
                    let dx = (i32::from(u.x) - i32::from(v.x)).abs();
                    let dy = (i32::from(u.y) - i32::from(v.y)).abs();
                    dx.max(dy) >= threshold
                })
            })
        })
    }

    /// Mode and vector bits charged to every leaf: `planner.leaf_bits` when
    /// set, else the backend's estimate. Without them the leaf cost is
    /// distortion plus partition bits only, and codecs whose partition
    /// symbols are dear next to split neighbours (VP9, AV1) split where one
    /// block would pay: VP9 `pan_bdd_00cef86b` at 250 kbps coded 2% of its
    /// area in 64x64 blocks against HEVC's 28%, and 2 bits per leaf gained
    /// +0.72 dB P-frame PSNR-Y there (crowd at 800 kbps +0.15 dB). VP9's
    /// estimate (`planner_leaf_bits`): `J_real` TRAIN proxy-6 -4.4% (full
    /// -3.4%, low -5.4%; bodycam -10.9%, pan -2.9%), no guards, +0.5%
    /// instructions; long-GOP (one key, 300 frames, crowd and pan at 400
    /// kbps) without drift.
    fn leaf_bits(&self) -> f32 {
        *self.leaf_bits.get_or_init(|| {
            if self.params.leaf_bits > 0.0 {
                self.params.leaf_bits
            } else {
                self.backend.planner_leaf_bits()
            }
        })
    }

    /// Capped prediction variance of `node` at its best whole-sample vector,
    /// and the residual statistics it was computed from.
    fn distortion(&self, node: &Block) -> (f32, LeafStats) {
        self.distortion_reusing(node, Known::NONE, None)
    }

    /// [`Self::distortion`], taking the statistics in `known` instead of
    /// measuring them again, and with `quadrants` filling in what a square
    /// node's measurements tell each of its quadrants (the planner may
    /// visit them next). Exact: residual sums add up over the quadrants.
    fn distortion_reusing(
        &self,
        node: &Block,
        known: Known,
        quadrants: Option<&mut [Known; 4]>,
    ) -> (f32, LeafStats) {
        let area = leaf_area::<B>(node) as f32;
        let cap = self.backend.lambda()
            * self.params.lambda_scale
            * self.params.distortion_cap_per_lambda
            * area;
        // One measurer for both vectors: its setup costs about as much as
        // a small block's measurement.
        let measurer = Measurer::new::<B>(self.dsp, self.source, self.reference, node);
        let square = is_square_leaf::<B>(node) && node.is_square();
        let mut quadrants = quadrants.filter(|_| square);
        // Known statistics, else the block's own measurement, split over
        // its quadrants for them when it can be.
        let mut measure = |mv: Mv, slot: usize| {
            if let Some(stats) = known.get(mv) {
                return stats;
            }
            if let Some(children) = quadrants.as_deref_mut()
                && let Some(parts) = measurer.measure_quadrants(mv)
            {
                for (child, part) in children.iter_mut().zip(parts) {
                    child.0[slot] = Some((mv, part));
                }
                return sum_of(parts);
            }
            measurer.measure(mv)
        };
        let (search_mv, search) = if is_square_leaf::<B>(node) {
            let mv = search_vector::<B>(self.motion, node);
            (mv, measure(mv, 0))
        } else {
            self.best_search_vector(&measurer, node)
        };
        // A zero search vector (static content) is already measured.
        let zero = if search_mv == Mv::ZERO {
            search
        } else {
            measure(Mv::ZERO, 1)
        };
        let stats = LeafStats {
            search_mv,
            search,
            zero,
        };
        // The mean residual is not free: coding it costs about `dc_bits`,
        // and below that it stays as error (module docs, "Rate awareness").
        let mean_cost = self.backend.lambda() * self.params.lambda_scale * self.params.dc_bits;
        let distortion = [stats.search, stats.zero]
            .iter()
            .map(|&s| {
                let variance = n_variance::<B>(s, node);
                let mean_energy = u64::from(s.sse) - variance;
                variance as f32 + (mean_energy as f32).min(mean_cost)
            })
            .fold(f32::MAX, f32::min)
            .min(cap);
        (distortion, stats)
    }

    /// The best of rectangular `node`'s tiling-square search vectors
    /// ([`search_vectors`]) by residual variance, with its statistics.
    fn best_search_vector(&self, measurer: &Measurer<'_>, node: &Block) -> (Mv, Variance) {
        let mut vectors = [Mv::ZERO; MAX_TILING_SQUARES];
        let count = search_vectors(self.motion, node, &mut vectors);
        vectors[..count]
            .iter()
            .map(|&mv| (mv, measurer.measure(mv)))
            .min_by_key(|&(_, stats)| n_variance::<B>(stats, node))
            .expect("a block has a tiling square")
    }

    fn limits(&self, sb: &Block) -> Limits {
        let coded_q = self.backend.relative_quantizer();
        let qp = coded_q * SVT_QP_MAX;
        // Past the largest quantizer: the planned quantizer, continued by
        // the step octaves λ grew by (λ ∝ step²).
        let headroom = self.lambda_headroom.max(1.0);
        let q = coded_q + QUANTIZER_PER_STEP_OCTAVE * 0.5 * headroom.log2();
        let stats = self.motion.and_then(|m| {
            let s = sb.size();
            m.sb.try_get((sb.x / s) as isize, (sb.y / s) as isize)
                .copied()
        });
        let spread = stats.map_or(0.0, |s| s.cost_variance_8x8);

        let max_log2 = if spread > self.params.high_8x8_cost_variance * qp_weight(qp) * headroom {
            CAPPED_MAX_LOG2
        } else {
            sb.log2_size()
        };

        let floor = self.params.inter_min_log2.round() as u8;
        // Depth removal pays at coarse quantizers, where small blocks rarely
        // repay their bits. At fine quantizers it keeps fine texture in
        // 32x32 blocks: off everywhere, TreesAndGrass 1080p gained 8.9%
        // luma BD at fixed QP, yet J_real TRAIN lost 0.4% at low rates.
        let removable = q >= self.params.depth_removal_min_quantizer;
        let min_log2 = match stats {
            Some(s) if removable && remove_below_32(s.me_dist, spread, qp, self.params) => {
                CAPPED_MAX_LOG2.max(floor)
            }
            _ => floor,
        };

        let mut bias = self.params.bias_base - self.params.bias_q_slope * q;
        for (threshold, add) in BIAS_SPREAD_STEPS.iter().rev() {
            if spread > *threshold {
                bias += add;
                break;
            }
        }
        let max_log2 = max_log2.min(self.backend.max_leaf_log2());
        Limits {
            max_log2,
            min_log2: min_log2.min(max_log2),
            bias: bias.clamp(self.params.bias_min, self.params.bias_max),
        }
    }
}

/// SVT's quantiser weighting of thresholds
/// (`svt_aom_get_qp_based_th_scaling_factors`).
fn qp_weight(qp: f32) -> f32 {
    if qp < 46.0 {
        qp.max(10.0) / SVT_QP_MAX
    } else {
        1.05 - (-(qp - 35.0) / 10.0).exp()
    }
}

/// SVT depth removal: drop the sizes below 32x32 when splitting 32x32 into
/// 16x16, and 16x16 into 8x8, would each improve the summed motion-search
/// cost of the superblock by less than the scaled thresholds.
/// `me_dist` is [8x8, 16x16, 32x32, 64x64].
fn remove_below_32(me_dist: [u32; 4], spread: f32, qp: f32, params: &PlannerParams) -> bool {
    let headroom = (SVT_QP_MAX - qp - 10.0).max(1.0);
    let band = spread / headroom;
    let band_factor = if band < 25_000.0 {
        4.0
    } else if band < 50_000.0 {
        2.0
    } else {
        return false;
    };
    let scale =
        band_factor * ((headroom as u32 >> 4).max(1) as f32) * params.depth_removal_qp_scale;
    let improvement = |coarse: u32, fine: u32| -> f32 {
        if fine == 0 {
            0.0
        } else {
            (coarse.saturating_sub(fine)) as f32 * 1000.0 / fine as f32
        }
    };
    let [c8, c16, c32, _] = me_dist;
    improvement(c32, c16) < params.depth_removal_th_32_16 * scale
        && improvement(c16, c8) < params.depth_removal_th_16_8 * scale
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{BlockDecision, FramePlan};
    use crate::shape::ShapeSet;
    use nc_common::PlaneIndex;

    /// A backend with one bit per partition symbol that declares binary
    /// shapes when `shapes` is set; with `rect_coded_cost`, its
    /// `inter_leaf_cost` prices rectangles at that and squares at nothing.
    struct ShapeBackend {
        shapes: bool,
        rect_coded_cost: Option<f32>,
    }
    impl CodecBackend for ShapeBackend {
        const RECTANGULAR_LEAVES: bool = true;
        fn new(_: &crate::config::EncoderConfig) -> Result<Self, crate::config::ConfigError> {
            Ok(Self {
                shapes: true,
                rect_coded_cost: None,
            })
        }
        fn superblock_log2(&self) -> u8 {
            5
        }
        fn min_block_log2(&self) -> u8 {
            3
        }
        fn lambda(&self) -> f32 {
            1.0
        }
        fn partition_bits(&self, _: &Block, _: bool) -> f32 {
            1.0
        }
        fn partition_shapes(&self, _: &Block) -> ShapeSet {
            if self.shapes {
                ShapeSet::of(&[PartitionShape::Horizontal, PartitionShape::Vertical])
            } else {
                ShapeSet::EMPTY
            }
        }
        fn shape_bits(&self, _: &Block, _: PartitionShape) -> f32 {
            1.0
        }
        fn inter_leaf_cost(&self, _: &Picture, _: &Picture, block: &Block, _: Mv) -> Option<f32> {
            self.rect_coded_cost
                .map(|cost| if block.is_square() { 0.0 } else { cost })
        }
        fn begin_frame(&mut self, _: &FramePlan) {}
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

    /// The plan of a 32x32 block whose residual against the reference is
    /// +40 in its top half and -40 in its bottom half.
    fn plan_of_two_halves(shapes: bool, rect_coded_cost: Option<f32>) -> Partition {
        let source = picture_with(|r, c| (100 + (r * 3 + c * 5) % 17) as u8);
        let reference = picture_with(|r, c| {
            let s = 100 + (r * 3 + c * 5) % 17;
            (if r < 16 { s - 40 } else { s + 40 }) as u8
        });
        let params = PlannerParams {
            rect_motion_gate: 0.0,
            inter_min_log2: 3.0,
            rect_rd: if rect_coded_cost.is_some() { 1.0 } else { 0.0 },
            ..PlannerParams::default()
        };
        let planner = Planner {
            dsp: &Dsp::new(),
            backend: &ShapeBackend {
                shapes,
                rect_coded_cost,
            },
            source: &source,
            reference: &reference,
            motion: None,
            pre: None,
            width: 32,
            height: 32,
            params: &params,
            shapes: true,
            lambda_headroom: 1.0,
            leaf_bits: OnceCell::new(),
        };
        planner.plan(Block::square(0, 0, 5))
    }

    #[test]
    fn halves_with_different_residuals_plan_a_horizontal_shape() {
        assert!(matches!(
            plan_of_two_halves(true, None),
            Partition::Shaped(PartitionShape::Horizontal, _)
        ));
        // Without declared shapes the quadtree splits instead.
        assert!(matches!(
            plan_of_two_halves(false, None),
            Partition::Split(_)
        ));
    }

    #[test]
    fn the_coded_cost_confirms_or_rejects_a_planned_shape() {
        // `planner.rect_rd`: cheap coded rectangles keep the shape, dear
        // ones fall back to the square plan.
        assert!(matches!(
            plan_of_two_halves(true, Some(0.0)),
            Partition::Shaped(PartitionShape::Horizontal, _)
        ));
        assert!(matches!(
            plan_of_two_halves(true, Some(1e9)),
            Partition::Split(_)
        ));
    }

    #[test]
    fn depth_removal_needs_small_improvements() {
        // 32x32 barely worse than 16x16 and 8x8: remove.
        assert!(remove_below_32(
            [1000, 1002, 1004, 1100],
            100.0,
            30.0,
            &PlannerParams::default()
        ));
        // 32x32 much worse than 16x16: keep the smaller sizes.
        assert!(!remove_below_32(
            [500, 600, 1500, 2000],
            100.0,
            30.0,
            &PlannerParams::default()
        ));
    }

    #[test]
    fn qp_weight_is_continuous_enough() {
        assert!((qp_weight(45.0) - qp_weight(46.0)).abs() < 0.1);
        assert!(qp_weight(10.0) < qp_weight(40.0));
    }
}
