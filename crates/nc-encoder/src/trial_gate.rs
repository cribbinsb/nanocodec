//! The fitted key-trial gate (`planner.key_rd_gate`): which planned key
//! nodes the key RD partition search (`rd_split.rs` "Key frames") codes the
//! other way, chosen by a small regression tree instead of the margin
//! thresholds (`planner.key_rd_margin`, `planner.key_rd_merge_margin`).
//!
//! Why: the margin gate trials a node where the key plan nearly tied leaf
//! and split. Offline, on the key RD labels (`NC_KEYRD_DUMP`, branch
//! `intra-selector-diag`: AV1, HEVC, VP9, VVC and AV2 medium, 1530
//! one-frame runs, 20.8 M nodes, every candidate coded), what a trial gains per sample also
//! follows the node's texture and λ, not only the margin: a textured node a
//! few percent from a tie is worth more than a flat one at the tie. Ranked
//! by the tree, the same extra coded area as margin 1.1 recovers 85-95 % of
//! the full search's gain instead of 68-75 % (AV2 66 % against 60 %), held
//! out by source dataset
//! (docs/analysis/learned-decisions-2026-11-22.md). On real one-frame keys
//! the gain is smaller: see that document's "Real encodes".
//!
//! What it reads, all known when the key plan has weighed the node:
//! - the plan's margin (rejected over chosen cost, `NodeMargins`) and its
//!   direction (planned leaf or planned split);
//! - the node's size;
//! - the plan's transform estimate of the leaf's distortion
//!   (`CodecBackend::intra_leaf_cost`) per sample, over λ;
//! - λ.
//!
//! The score is the tree's predicted gain of a trial per sample, in units of
//! 1e-4 λ; a planned node tries its alternative where the score is at least
//! the gate's value. The one value replaces both margin thresholds; its
//! unit, gain per sample of coded area, is the trade a preset makes.
//!
//! Exactness: the tree was fitted on logarithms of the features, but each
//! test is stored in the form the encoder evaluates with one IEEE multiply
//! and compare (thresholds exponentiated offline), so no logarithm or other
//! library function runs and the decision is the same on every target.
//!
//! Fit (registered data; refit when the key search's labels move): branch
//! `learned-probe`, `probe/lgate3.py`, sklearn `DecisionTreeRegressor` depth
//! 6, at least 2000 nodes per leaf, squared error on the gain of a trial per
//! sample over λ (clipped at its 99.5th percentile), on the key RD dump of
//! all five block-tree codecs (AV1, HEVC, VP9, VVC, AV2 medium; 56 clips,
//! frame 0 and mid-clip, three quantizers each: 1530 runs, 20.8 M nodes),
//! codec-free (one table for every codec), 4 M nodes sampled with seed 7.
//! A leaf-mode lever change (HEVC `intra_rd_candidates` 2) left the table's
//! held-out recovery at 96 % against 91 % for a refit.

/// One test of [`GATE_TREE`]: true sends the walk to its first child.
#[derive(Debug, Clone, Copy)]
pub(crate) enum GateTest {
    /// The signed log margin (negative for a planned leaf, positive for a
    /// planned split) is at most a threshold `t`: a planned leaf passes when
    /// its margin is at least `leaf_min` (e^-t), a planned split when its
    /// margin is at most `split_max` (e^t).
    SignedMarginAtMost { leaf_min: f32, split_max: f32 },
    /// The margin is at most this.
    MarginAtMost(f32),
    /// The leaf's estimated distortion is at most this many λ per sample.
    EstimateAtMost(f32),
    /// λ is at most this.
    LambdaAtMost(f32),
    /// The node is not of this log2 size.
    NotLog2(u8),
}

/// A node of [`GATE_TREE`]: a test with its two children's indices, or a
/// leaf's score.
#[derive(Debug, Clone, Copy)]
pub(crate) enum GateNode {
    Test(GateTest, u16, u16),
    Leaf(u16),
}

/// What the key plan knows of a node when the search decides whether to
/// try its alternative.
#[derive(Debug, Clone, Copy)]
pub(crate) struct KeyNodeEvidence {
    /// Rejected over chosen cost, at least 1 (`NodeMargins::get`).
    pub margin: f32,
    /// Whether the plan split the node.
    pub planned_split: bool,
    pub log2: u8,
    /// The plan's distortion estimate of the node as a leaf.
    pub estimate: f32,
    /// The node's area in samples.
    pub area: f32,
    pub lambda: f32,
}

impl GateTest {
    fn passes(self, node: &KeyNodeEvidence) -> bool {
        match self {
            Self::SignedMarginAtMost {
                leaf_min,
                split_max,
            } => {
                if node.planned_split {
                    node.margin <= split_max
                } else {
                    node.margin >= leaf_min
                }
            }
            Self::MarginAtMost(max) => node.margin <= max,
            Self::EstimateAtMost(per_sample) => {
                node.estimate <= per_sample * node.area * node.lambda
            }
            Self::LambdaAtMost(max) => node.lambda <= max,
            Self::NotLog2(log2) => node.log2 != log2,
        }
    }
}

/// The tree's score of `node`: its predicted trial gain per sample, in
/// 1e-4 λ.
pub(crate) fn key_trial_score(node: &KeyNodeEvidence) -> u16 {
    let mut at = 0;
    loop {
        match GATE_TREE[at] {
            GateNode::Leaf(score) => return score,
            GateNode::Test(test, pass, fail) => {
                at = usize::from(if test.passes(node) { pass } else { fail });
            }
        }
    }
}

/// The fitted tree, root first (generated by `probe/lgate3.py`).
const GATE_TREE: &[GateNode] = &[
    GateNode::Test(GateTest::MarginAtMost(1.098_820_8), 1, 64),
    GateNode::Test(GateTest::NotLog2(4), 2, 33),
    GateNode::Test(
        GateTest::SignedMarginAtMost {
            leaf_min: 0.970_760_7,
            split_max: 1.030_12,
        },
        3,
        18,
    ),
    GateNode::Test(GateTest::EstimateAtMost(0.539_153_6), 4, 11),
    GateNode::Test(GateTest::EstimateAtMost(0.250_277_04), 5, 8),
    GateNode::Test(
        GateTest::SignedMarginAtMost {
            leaf_min: 0.999_987_84,
            split_max: 1.000_012_2,
        },
        6,
        7,
    ),
    GateNode::Leaf(33),
    GateNode::Leaf(76),
    GateNode::Test(GateTest::LambdaAtMost(6.539_71), 9, 10),
    GateNode::Leaf(200),
    GateNode::Leaf(104),
    GateNode::Test(GateTest::LambdaAtMost(6.539_71), 12, 15),
    GateNode::Test(
        GateTest::SignedMarginAtMost {
            leaf_min: 0.999_984_26,
            split_max: 1.000_015_7,
        },
        13,
        14,
    ),
    GateNode::Leaf(534),
    GateNode::Leaf(112),
    GateNode::Test(GateTest::EstimateAtMost(1.897_116), 16, 17),
    GateNode::Leaf(188),
    GateNode::Leaf(272),
    GateNode::Test(GateTest::NotLog2(5), 19, 26),
    GateNode::Test(GateTest::MarginAtMost(1.049_313_5), 20, 23),
    GateNode::Test(GateTest::LambdaAtMost(54.909_958), 21, 22),
    GateNode::Leaf(26),
    GateNode::Leaf(16),
    GateNode::Test(GateTest::EstimateAtMost(0.218_705_39), 24, 25),
    GateNode::Leaf(16),
    GateNode::Leaf(5),
    GateNode::Test(GateTest::MarginAtMost(1.050_798_8), 27, 30),
    GateNode::Test(GateTest::LambdaAtMost(54.909_958), 28, 29),
    GateNode::Leaf(97),
    GateNode::Leaf(72),
    GateNode::Test(GateTest::EstimateAtMost(0.532_549_9), 31, 32),
    GateNode::Leaf(69),
    GateNode::Leaf(30),
    GateNode::Test(
        GateTest::SignedMarginAtMost {
            leaf_min: 1.017_161_7,
            split_max: 0.983_127_8,
        },
        34,
        49,
    ),
    GateNode::Test(GateTest::EstimateAtMost(0.910_161_7), 35, 42),
    GateNode::Test(GateTest::EstimateAtMost(0.544_584_04), 36, 39),
    GateNode::Test(GateTest::EstimateAtMost(0.340_029_95), 37, 38),
    GateNode::Leaf(15),
    GateNode::Leaf(56),
    GateNode::Test(GateTest::LambdaAtMost(6.539_71), 40, 41),
    GateNode::Leaf(279),
    GateNode::Leaf(124),
    GateNode::Test(GateTest::LambdaAtMost(6.539_71), 43, 46),
    GateNode::Test(GateTest::EstimateAtMost(2.284_952_6), 44, 45),
    GateNode::Leaf(507),
    GateNode::Leaf(803),
    GateNode::Test(GateTest::EstimateAtMost(2.085_309_5), 47, 48),
    GateNode::Leaf(225),
    GateNode::Leaf(326),
    GateNode::Test(GateTest::LambdaAtMost(13.124_083_5), 50, 57),
    GateNode::Test(
        GateTest::SignedMarginAtMost {
            leaf_min: 0.976_025_2,
            split_max: 1.024_563_7,
        },
        51,
        54,
    ),
    GateNode::Test(GateTest::EstimateAtMost(1.995_729_4), 52, 53),
    GateNode::Leaf(477),
    GateNode::Leaf(605),
    GateNode::Test(GateTest::LambdaAtMost(6.539_71), 55, 56),
    GateNode::Leaf(196),
    GateNode::Leaf(489),
    GateNode::Test(
        GateTest::SignedMarginAtMost {
            leaf_min: 1.000_099_7,
            split_max: 0.999_900_4,
        },
        58,
        61,
    ),
    GateNode::Test(GateTest::EstimateAtMost(0.750_230_85), 59, 60),
    GateNode::Leaf(80),
    GateNode::Leaf(309),
    GateNode::Test(
        GateTest::SignedMarginAtMost {
            leaf_min: 0.970_187_2,
            split_max: 1.030_728_9,
        },
        62,
        63,
    ),
    GateNode::Leaf(442),
    GateNode::Leaf(287),
    GateNode::Test(GateTest::EstimateAtMost(0.363_263_55), 65, 96),
    GateNode::Test(
        GateTest::SignedMarginAtMost {
            leaf_min: 1.0,
            split_max: 1.0,
        },
        66,
        81,
    ),
    GateNode::Test(GateTest::EstimateAtMost(0.195_299_36), 67, 74),
    GateNode::Test(
        GateTest::SignedMarginAtMost {
            leaf_min: 1.369_195_6,
            split_max: 0.730_355_86,
        },
        68,
        71,
    ),
    GateNode::Test(GateTest::LambdaAtMost(6.539_71), 69, 70),
    GateNode::Leaf(7),
    GateNode::Leaf(1),
    GateNode::Test(GateTest::LambdaAtMost(6.539_71), 72, 73),
    GateNode::Leaf(64),
    GateNode::Leaf(7),
    GateNode::Test(GateTest::NotLog2(4), 75, 78),
    GateNode::Test(GateTest::EstimateAtMost(0.261_443_76), 76, 77),
    GateNode::Leaf(43),
    GateNode::Leaf(64),
    GateNode::Test(GateTest::EstimateAtMost(0.276_214_84), 79, 80),
    GateNode::Leaf(8),
    GateNode::Leaf(18),
    GateNode::Test(GateTest::NotLog2(4), 82, 89),
    GateNode::Test(GateTest::NotLog2(6), 83, 86),
    GateNode::Test(GateTest::EstimateAtMost(0.152_649_61), 84, 85),
    GateNode::Leaf(86),
    GateNode::Leaf(39),
    GateNode::Test(GateTest::EstimateAtMost(0.069_885), 87, 88),
    GateNode::Leaf(20),
    GateNode::Leaf(3),
    GateNode::Test(GateTest::LambdaAtMost(43.582_066), 90, 93),
    GateNode::Test(
        GateTest::SignedMarginAtMost {
            leaf_min: 0.867_087_6,
            split_max: 1.153_286,
        },
        91,
        92,
    ),
    GateNode::Leaf(627),
    GateNode::Leaf(559),
    GateNode::Test(GateTest::LambdaAtMost(183.847_72), 94, 95),
    GateNode::Leaf(468),
    GateNode::Leaf(375),
    GateNode::Test(GateTest::NotLog2(4), 97, 110),
    GateNode::Test(
        GateTest::SignedMarginAtMost {
            leaf_min: 1.098_978_3,
            split_max: 0.909_936_13,
        },
        98,
        103,
    ),
    GateNode::Test(GateTest::EstimateAtMost(0.719_479_6), 99, 102),
    GateNode::Test(GateTest::EstimateAtMost(0.572_390_26), 100, 101),
    GateNode::Leaf(110),
    GateNode::Leaf(174),
    GateNode::Leaf(259),
    GateNode::Test(GateTest::NotLog2(6), 104, 107),
    GateNode::Test(GateTest::EstimateAtMost(0.654_813_9), 105, 106),
    GateNode::Leaf(17),
    GateNode::Leaf(7),
    GateNode::Test(GateTest::LambdaAtMost(6.539_71), 108, 109),
    GateNode::Leaf(9),
    GateNode::Leaf(1),
    GateNode::Test(
        GateTest::SignedMarginAtMost {
            leaf_min: 1.112_259,
            split_max: 0.899_071_16,
        },
        111,
        118,
    ),
    GateNode::Test(GateTest::EstimateAtMost(0.844_812_6), 112, 115),
    GateNode::Test(GateTest::EstimateAtMost(0.562_183_5), 113, 114),
    GateNode::Leaf(40),
    GateNode::Leaf(86),
    GateNode::Test(GateTest::LambdaAtMost(6.539_71), 116, 117),
    GateNode::Leaf(337),
    GateNode::Leaf(161),
    GateNode::Test(GateTest::LambdaAtMost(13.124_083_5), 119, 122),
    GateNode::Test(GateTest::LambdaAtMost(6.539_71), 120, 121),
    GateNode::Leaf(161),
    GateNode::Leaf(329),
    GateNode::Test(GateTest::LambdaAtMost(24.133_68), 123, 124),
    GateNode::Leaf(81),
    GateNode::Leaf(182),
];

#[cfg(test)]
mod tests {
    use super::*;

    fn node(margin: f32, planned_split: bool, estimate_per_lambda: f32) -> KeyNodeEvidence {
        KeyNodeEvidence {
            margin,
            planned_split,
            log2: 4,
            estimate: estimate_per_lambda * 256.0 * 100.0,
            area: 256.0,
            lambda: 100.0,
        }
    }

    #[test]
    fn every_walk_ends_at_a_leaf() {
        for (i, n) in GATE_TREE.iter().enumerate() {
            if let GateNode::Test(_, pass, fail) = n {
                assert!(usize::from(*pass) > i && usize::from(*fail) > i);
                assert!(
                    usize::from(*pass) < GATE_TREE.len() && usize::from(*fail) < GATE_TREE.len()
                );
            }
        }
    }

    #[test]
    fn a_textured_near_tie_outscores_a_flat_decisive_node() {
        assert!(
            key_trial_score(&node(1.01, false, 2.0)) > key_trial_score(&node(1.5, false, 0.05))
        );
    }
}
