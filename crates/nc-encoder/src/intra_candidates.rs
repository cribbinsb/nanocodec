//! Intra mode candidates shared by the backends' intra searches.
//!
//! Every codec costs a short list of intra modes for a block and codes the
//! cheapest: the list is the codec-independent part of that search, chosen
//! by one registered parameter per codec (AV1 `av1.keyframe_intra_modes`,
//! AV2 `av2.intra_modes`). Codecs map these modes onto their own (AV2's
//! vertical and horizontal are its zero-delta directional modes) and may add
//! the modes of already coded neighbours ([`with_neighbours`]), which are
//! likely and cheap to signal in mode-list syntax. [`shortlist_intra`] also
//! provides codec-independent proxy ranking and a bounded full-RD candidate list
//! for directional searches; VVC supplies its prediction and syntax costs.
//!
//! [`directions_by_orientation`] orders the eight directional modes by how
//! much of the block's gradient energy runs along each (a histogram of
//! oriented gradients, as libaom prunes directional modes), so a search can
//! predict only the few likeliest directions.

use nc_common::PlaneView;
use nc_dsp::intra_prediction::IntraMode;

/// The intra modes a block's search costs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntraModeSet {
    /// DC only: no search.
    DcOnly,
    /// DC, the three smooth modes and Paeth.
    NonDirectional,
    /// The non-directional modes plus vertical and horizontal.
    Extended,
    /// DC, smooth, vertical and horizontal: the extended set less the
    /// one-sided smooth modes and Paeth, which add little (AV1 key frames:
    /// 93 % of the extended set's modelled-RD gain over DC, 6 VAL clips).
    Core,
    /// DC and smooth: P-frame intra blocks mostly repair brightness
    /// changes and occlusions, where a smooth gradient does most of the
    /// work (HEVC Fast P frames, VC-1K worst six fixed QP: planar and DC
    /// J -1.8 % against DC alone, planar, DC, H, V and the MPMs -1.6 % at
    /// half again the instructions).
    Smooth,
}

/// The modes of [`IntraModeSet::Core`], in the extended set's order.
const CORE_MODES: [IntraMode; 4] = [
    IntraMode::Dc,
    IntraMode::Vertical,
    IntraMode::Horizontal,
    IntraMode::Smooth,
];

impl IntraModeSet {
    /// The set selected by a registered parameter value: 0 DC only,
    /// 1 non-directional, 2 extended (rounded; values above 2 select the
    /// extended set).
    #[must_use]
    pub fn from_param(value: f32) -> Self {
        match value.round() as i32 {
            i32::MIN..=0 => Self::DcOnly,
            1 => Self::NonDirectional,
            _ => Self::Extended,
        }
    }

    /// Whether the set holds `mode`.
    #[must_use]
    pub fn contains(self, mode: IntraMode) -> bool {
        self.modes().contains(&mode)
    }

    /// The set's modes, in the order searches visit them (ties keep the
    /// earlier mode).
    #[must_use]
    pub fn modes(self) -> &'static [IntraMode] {
        match self {
            Self::DcOnly => &[IntraMode::Dc],
            Self::NonDirectional => &IntraMode::NON_DIRECTIONAL,
            Self::Extended => &IntraMode::EXTENDED,
            Self::Core => &CORE_MODES,
            Self::Smooth => &[IntraMode::Dc, IntraMode::Smooth],
        }
    }
}

/// Line orientations told apart: 0, 22.5, ..., 157.5 degrees.
const ORIENTATIONS: usize = 8;
/// Unit vectors (×1024) of the boundaries between orientation bins, at
/// 11.25 + 22.5k degrees (y up).
const BIN_BOUNDARIES: [(i32, i32); ORIENTATIONS] = [
    (1004, 200),
    (851, 569),
    (569, 851),
    (200, 1004),
    (-200, 1004),
    (-569, 851),
    (-851, 569),
    (-1004, 200),
];
/// The directional mode whose prediction lines run at each orientation
/// (prediction angle modulo 180: H 180, D203 23, D45, D67, V 90, D113, D135,
/// D157).
const MODE_OF_ORIENTATION: [IntraMode; ORIENTATIONS] = [
    IntraMode::Horizontal,
    IntraMode::D203,
    IntraMode::D45,
    IntraMode::D67,
    IntraMode::Vertical,
    IntraMode::D113,
    IntraMode::D135,
    IntraMode::D157,
];

/// Gradient energy (|gx| + |gy|, central differences inside the block) per
/// line orientation, the line running perpendicular to the gradient.
#[must_use]
pub fn orientation_histogram(block: &PlaneView<'_>) -> [u64; ORIENTATIONS] {
    let mut histogram = [0u64; ORIENTATIONS];
    let (w, h) = (block.width(), block.height());
    for y in 1..h.saturating_sub(1) {
        let (above, row, below) = (block.row(y - 1), block.row(y), block.row(y + 1));
        for x in 1..w - 1 {
            let gx = i32::from(row[x + 1]) - i32::from(row[x - 1]);
            // y up, so that angles count as AV1's prediction angles do.
            let gy = i32::from(above[x]) - i32::from(below[x]);
            if gx == 0 && gy == 0 {
                continue;
            }
            // The line direction, turned into the upper half-plane.
            let (mut u, mut v) = (-gy, gx);
            if v < 0 || (v == 0 && u < 0) {
                (u, v) = (-u, -v);
            }
            histogram[orientation_bin(u, v)] += u64::from(gx.unsigned_abs() + gy.unsigned_abs());
        }
    }
    histogram
}

/// The orientation bin of line direction (u, v) in the upper half-plane:
/// the boundaries it has passed, counted round to bin 0 past the last. The
/// boundaries run in angle order, so those passed are a prefix and a binary
/// search (three comparisons) counts them.
fn orientation_bin(u: i32, v: i32) -> usize {
    BIN_BOUNDARIES.partition_point(|&(bx, by)| bx * v - by * u > 0) % ORIENTATIONS
}

/// The eight directional modes, the one most of `block`'s gradient energy
/// runs along first (ties keep the orientation order).
#[must_use]
pub fn directions_by_orientation(block: &PlaneView<'_>) -> [IntraMode; ORIENTATIONS] {
    let histogram = orientation_histogram(block);
    let mut order: [usize; ORIENTATIONS] = std::array::from_fn(|i| i);
    order.sort_by_key(|&i| std::cmp::Reverse(histogram[i]));
    order.map(|i| MODE_OF_ORIENTATION[i])
}

/// `set` followed by the `neighbours` it does not already contain, each once.
#[must_use]
pub fn with_neighbours<T: Copy + PartialEq>(
    set: impl IntoIterator<Item = T>,
    neighbours: impl IntoIterator<Item = T>,
) -> Vec<T> {
    let mut candidates: Vec<T> = set.into_iter().collect();
    for mode in neighbours {
        if !candidates.contains(&mode) {
            candidates.push(mode);
        }
    }
    candidates
}

/// Retain the mandatory intra modes and add the lowest proxy-cost alternatives.
///
/// The codec supplies its prediction/signalling proxy through `rough_cost`;
/// the shared engine owns deduplication, budgeting and deterministic ranking.
/// Every alternative is scored once, ties keep input order, and mandatory
/// modes cannot be displaced. The caller performs its full RD evaluation on
/// the returned list. A budget covering all alternatives avoids proxy work.
#[must_use]
pub fn shortlist_intra<T: Copy + PartialEq>(
    mandatory: impl IntoIterator<Item = T>,
    alternatives: impl IntoIterator<Item = T>,
    extra_budget: usize,
    mut rough_cost: impl FnMut(T) -> f32,
) -> Vec<T> {
    let mut selected = with_neighbours(std::iter::empty(), mandatory);
    if extra_budget == 0 {
        return selected;
    }
    let mut rest = with_neighbours(std::iter::empty(), alternatives);
    rest.retain(|m| !selected.contains(m));
    if extra_budget >= rest.len() {
        selected.extend(rest);
        return selected;
    }
    let mut ranked: Vec<_> = rest.into_iter().map(|m| (m, rough_cost(m))).collect();
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
    selected.extend(ranked.into_iter().take(extra_budget).map(|(m, _)| m));
    selected
}

/// Whether a block's cheapest prediction so far (residual energy `sse` over
/// `pixels` samples) is settled for an intra mode search: below
/// `sse_per_pixel`, the block is flat or already well predicted and the
/// search builds no other prediction. Offline (AV1 key frames, 87 k blocks
/// with every mode RD-costed), ranking blocks by DC's residual energy per
/// sample recovers as much of the mode search's gain in the most active
/// half of the blocks as searching every block; the same energy against λ,
/// or per block, separates them far less.
#[must_use]
pub fn prediction_settles(sse: u64, pixels: usize, sse_per_pixel: f32) -> bool {
    (sse as f32) < sse_per_pixel * pixels as f32
}

/// log2 of the block side at which a [`FlatGate`] applies its base
/// threshold (8x8).
const FLAT_GATE_BASE_LOG2: u8 = 3;

/// [`prediction_settles`] with a threshold that grows by `size_scale` per
/// doubling of the block side above 8x8: mode choice pays on small blocks
/// (AV1 key frames, every mode RD-costed, 6 TRAIN clips x 3 qindex: the
/// fitted per-size thresholds stood about 3x apart per size step). Blocks
/// 8x8 and smaller take the base threshold. A threshold of 0 settles
/// nothing (the gate is off).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlatGate {
    /// Residual energy per sample below which an 8x8 block settles.
    pub sse_per_pixel: f32,
    /// Multiplier per doubling of the block side above 8x8.
    pub size_scale: f32,
}

impl FlatGate {
    /// Whether the gate is on (a positive threshold).
    #[must_use]
    pub fn is_on(&self) -> bool {
        self.sse_per_pixel > 0.0
    }

    /// Whether a `2^log2_side` block whose prediction leaves `sse` over
    /// `pixels` measured samples (a large block may be measured on part of
    /// itself) is settled.
    #[must_use]
    pub fn settles(&self, sse: u64, pixels: usize, log2_side: u8) -> bool {
        let doublings = i32::from(log2_side.saturating_sub(FLAT_GATE_BASE_LOG2));
        prediction_settles(
            sse,
            pixels,
            self.sse_per_pixel * self.size_scale.powi(doublings),
        )
    }
}

/// Level statistics of one quantised transform block, all a
/// [`LevelRateModel`] reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LevelStats {
    /// Sum of the absolute levels.
    pub level_sum: u32,
    /// Nonzero levels.
    pub nonzero: u32,
    /// End of block: one past the last nonzero level in scan order (0: all
    /// zero).
    pub eob: u32,
}

/// Coefficient bits of a quantised transform block estimated from its
/// [`LevelStats`], linear per transform size in the level sum, the nonzero
/// count, `log2(eob + 1)` and a constant for a coded (nonzero) block. It
/// prices intra mode candidates after transform and quantisation without the
/// codec's context-adaptive coefficient walk, as libaom's real-time intra
/// pick (`av1_block_yrd`, `nonrd_opt.c`: 4 bits per unit level plus the EOB's
/// msb) does with fixed weights; codecs supply weights fitted to their own
/// coefficient coding. An all-zero block costs nothing here: its skip
/// signalling is the codec's mode syntax.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LevelRateModel {
    /// Weights of (level sum, nonzero, log2(eob + 1), coded) for square
    /// transforms of side 4, 8, 16, 32 and 64.
    weights: [[f32; 4]; LEVEL_RATE_SIZES],
}

/// Square transform sides a [`LevelRateModel`] covers: 4 to 64.
const LEVEL_RATE_SIZES: usize = 5;
/// log2 of the smallest covered side.
const LEVEL_RATE_MIN_LOG2: u32 = 2;

impl LevelRateModel {
    /// A model from per-size weights of (level sum, nonzero,
    /// log2(eob + 1), coded), sides 4 to 64.
    #[must_use]
    pub const fn new(weights: [[f32; 4]; LEVEL_RATE_SIZES]) -> Self {
        Self { weights }
    }

    /// Estimated bits of a `side_log2` square transform block's levels
    /// (sides outside 4..64 take the nearest covered size); never below one
    /// bit for a coded block.
    #[must_use]
    pub fn bits(&self, side_log2: u32, stats: LevelStats) -> f32 {
        if stats.eob == 0 {
            return 0.0;
        }
        let index = side_log2
            .saturating_sub(LEVEL_RATE_MIN_LOG2)
            .min(LEVEL_RATE_SIZES as u32 - 1) as usize;
        let [sum, nonzero, eob, coded] = self.weights[index];
        let estimate = sum * stats.level_sum as f32
            + nonzero * stats.nonzero as f32
            + eob * ((stats.eob + 1) as f32).log2()
            + coded;
        estimate.max(1.0)
    }
}

/// The `keep` cheapest `candidates` by `cost`, cheapest first, each with its
/// cost: the shared ranking of a cheap intra trial (P-frame intra blocks),
/// where the codec supplies the candidate modes and a prediction SATD plus
/// √λ · mode bits per mode. The bits term matters: ranked by SATD alone,
/// smooth predictors win blocks they cost more to signal (AV1 key frames,
/// +7 % bytes at −1.6 dB on one key). Repeated candidates are costed once;
/// ties keep input order. `keep` 1 is the single mode coded without a
/// further trial; more give a caller's RD stage its short list.
#[must_use]
pub fn rank_intra<T: Copy + PartialEq>(
    candidates: impl IntoIterator<Item = T>,
    keep: usize,
    mut cost: impl FnMut(T) -> f32,
) -> Vec<(T, f32)> {
    let mut ranked: Vec<(T, f32)> = Vec::new();
    for mode in candidates {
        if !ranked.iter().any(|&(m, _)| m == mode) {
            ranked.push((mode, cost(mode)));
        }
    }
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
    ranked.truncate(keep.max(1));
    ranked
}

/// How the angle deltas of the best-ranked direction are searched (AV1,
/// AV2: `angle_delta` around a nominal direction).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AngleDeltaSearch {
    /// Zero deltas only.
    Off,
    /// ±2 first, then ±1 and ±3 beside the better of −2, 0, +2: delta costs
    /// are roughly convex.
    CoarseToFine,
    /// Every nonzero delta.
    Exhaustive,
}

impl AngleDeltaSearch {
    /// The search of a registered parameter: 0 off, 1 coarse to fine, 2 all.
    #[must_use]
    pub fn from_param(value: f32) -> Self {
        match value.round() as i32 {
            ..=0 => Self::Off,
            1 => Self::CoarseToFine,
            _ => Self::Exhaustive,
        }
    }

    /// Cost the deltas this search tries around a direction whose zero delta
    /// costs `nominal`, by `try_delta` (which also records each trial), up
    /// to `max_delta` either way.
    pub fn run<C: PartialOrd + Copy>(
        self,
        nominal: C,
        max_delta: i8,
        mut try_delta: impl FnMut(i8) -> C,
    ) {
        match self {
            Self::Off => {}
            Self::CoarseToFine => {
                let (lower, upper) = (try_delta(-2), try_delta(2));
                let refine: [i8; 2] = if lower < nominal && lower <= upper {
                    [-1, -3]
                } else if upper < nominal {
                    [1, 3]
                } else {
                    [-1, 1]
                };
                for delta in refine {
                    try_delta(delta);
                }
            }
            Self::Exhaustive => {
                for delta in (-max_delta..=max_delta).filter(|&d| d != 0) {
                    try_delta(delta);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_gate_scales_per_size_doubling() {
        let gate = FlatGate {
            sse_per_pixel: 10.0,
            size_scale: 3.0,
        };
        // 8x8: threshold 10 per sample; 16x16: 30; 4x4 takes the 8x8 one.
        assert!(gate.settles(639, 64, 3));
        assert!(!gate.settles(640, 64, 3));
        assert!(gate.settles(29 * 256, 256, 4));
        assert!(!gate.settles(30 * 256, 256, 4));
        assert!(gate.settles(159, 16, 2));
        let off = FlatGate {
            sse_per_pixel: 0.0,
            size_scale: 3.0,
        };
        assert!(!off.is_on() && !off.settles(0, 64, 3));
    }

    #[test]
    fn smooth_set_holds_dc_and_smooth() {
        assert_eq!(
            IntraModeSet::Smooth.modes(),
            &[IntraMode::Dc, IntraMode::Smooth]
        );
        assert!(IntraModeSet::Core.contains(IntraMode::Smooth));
        assert!(!IntraModeSet::Smooth.contains(IntraMode::Vertical));
    }

    #[test]
    fn shortlist_preserves_required_modes_and_stable_ties() {
        let mut evaluated = Vec::new();
        let selected = shortlist_intra([0, 1, 0], [1, 2, 3, 4, 2], 2, |m| {
            evaluated.push(m);
            if m == 3 { 0. } else { 1. }
        });
        assert_eq!(selected, [0, 1, 3, 2]);
        assert_eq!(evaluated, [2, 3, 4]);
        assert_eq!(
            shortlist_intra([0, 1], [2, 3], 0, |_| panic!("no proxy needed")),
            [0, 1]
        );
        assert_eq!(
            shortlist_intra([0, 1], [1, 2, 3], 2, |_| panic!("exhaustive")),
            [0, 1, 2, 3]
        );
    }

    #[test]
    fn orientation_bins_match_counting_every_boundary() {
        for gx in -40..=40 {
            for gy in -40..=40 {
                let (mut u, mut v) = (-gy, gx);
                if v < 0 || (v == 0 && u < 0) {
                    (u, v) = (-u, -v);
                }
                let counted = BIN_BOUNDARIES
                    .iter()
                    .filter(|&&(bx, by)| bx * v - by * u > 0)
                    .count();
                assert_eq!(
                    orientation_bin(u, v),
                    counted % ORIENTATIONS,
                    "({gx}, {gy})"
                );
            }
        }
    }

    #[test]
    fn small_residuals_settle_the_search() {
        assert!(prediction_settles(6399, 64, 100.0));
        assert!(!prediction_settles(6400, 64, 100.0));
        assert!(!prediction_settles(0, 64, 0.0));
    }

    #[test]
    fn level_rate_model_prices_coded_blocks_only() {
        let model = LevelRateModel::new([[1.0, 2.0, 3.0, -4.0]; LEVEL_RATE_SIZES]);
        let zero = LevelStats {
            level_sum: 0,
            nonzero: 0,
            eob: 0,
        };
        assert_eq!(model.bits(3, zero), 0.0);
        let coded = LevelStats {
            level_sum: 5,
            nonzero: 3,
            eob: 7,
        };
        // 5 + 6 + 3 * log2(8) - 4.
        assert_eq!(model.bits(3, coded), 16.0);
        // Sides above 64 take the 64 weights: 1 + 2 + 3 * log2(2) - 4.
        let single = LevelStats {
            level_sum: 1,
            nonzero: 1,
            eob: 1,
        };
        assert_eq!(model.bits(7, single), 2.0);
        let floor = LevelRateModel::new([[0.0, 0.0, 0.0, -9.0]; LEVEL_RATE_SIZES]);
        assert_eq!(floor.bits(9, coded), 1.0);
    }

    #[test]
    fn ranking_keeps_the_cheapest_once_each_in_order() {
        let mut costed = Vec::new();
        let costs = [9., 2., 2., 2., 0.];
        let ranked = rank_intra([3, 1, 3, 2, 4], 3, |m| {
            costed.push(m);
            costs[m]
        });
        // 4 is cheapest; 3, 1 and 2 tie and keep their input order.
        assert_eq!(ranked, [(4, 0.), (3, 2.), (1, 2.)]);
        assert_eq!(costed, [3, 1, 2, 4]);
        assert_eq!(rank_intra([7], 0, |_| 1.).len(), 1);
    }

    #[test]
    fn parameter_values_select_growing_sets() {
        assert_eq!(IntraModeSet::from_param(0.0).modes(), &[IntraMode::Dc]);
        assert_eq!(
            IntraModeSet::from_param(1.0).modes(),
            &IntraMode::NON_DIRECTIONAL
        );
        assert_eq!(IntraModeSet::from_param(2.0).modes(), &IntraMode::EXTENDED);
        assert_eq!(IntraModeSet::from_param(3.0), IntraModeSet::Extended);
    }

    #[test]
    fn stripes_rank_their_own_direction_first() {
        let n = 16;
        let pattern = |f: &dyn Fn(usize, usize) -> usize| {
            let mut samples = vec![0u8; n * n];
            for y in 0..n {
                for x in 0..n {
                    samples[y * n + x] = if f(x, y) % 4 < 2 { 40 } else { 200 };
                }
            }
            samples
        };
        // Vertical stripes vary along x only; "/" stripes are constant along
        // x + y (y down), i.e. lines at 45 degrees; "\\" at 135.
        for (f, mode) in [
            (
                &(|x: usize, _y: usize| x) as &dyn Fn(usize, usize) -> usize,
                IntraMode::Vertical,
            ),
            (&|_x: usize, y: usize| y, IntraMode::Horizontal),
            (&|x: usize, y: usize| x + y, IntraMode::D45),
            (&|x: usize, y: usize| x + n - y, IntraMode::D135),
        ] {
            let samples = pattern(f);
            let view = PlaneView::new(&samples, n, n, n).unwrap();
            assert_eq!(directions_by_orientation(&view)[0], mode);
        }
    }

    #[test]
    fn neighbours_extend_the_set_once_each() {
        assert_eq!(with_neighbours([1, 2], [2, 5, 5, 7]), vec![1, 2, 5, 7]);
    }
}
