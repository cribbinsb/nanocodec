//! VP9 frame probability contexts, symbol counts and backward adaptation
//! (VP9 §8.4.2-8.4.4; libvpx `vp9_entropy.c` `vp9_adapt_coef_probs`,
//! `vp9_entropymode.c` `vp9_adapt_mode_probs`, `vp9_entropymv.c`
//! `vp9_adapt_mv_probs` and `vpx_dsp/prob.h`; BSD-3-Clause, see
//! `THIRD_PARTY.md`).
//!
//! Every inter-frame probability the encoder and decoder read comes from a
//! [`FrameContext`]. After each frame both sides merge the frame's symbol
//! [`Counts`] into the context the frame started from; the next frame
//! starts from the result. Key frames reset to the defaults (and code their
//! modes and partitions with the fixed key-frame tables).

use crate::tables::{self, BandProbs};
use crate::tree::{self, Tree};

/// Saturating count and largest update factor of coefficient adaptation
/// (`COEF_COUNT_SAT`, `COEF_MAX_UPDATE_FACTOR`; the frame after a key frame
/// adapts faster: `COEF_MAX_UPDATE_FACTOR_AFTER_KEY`).
const COEF_COUNT_SAT: u32 = 24;
const COEF_MAX_UPDATE_FACTOR: u32 = 112;
const COEF_MAX_UPDATE_FACTOR_AFTER_KEY: u32 = 128;
/// `count_to_update_factor` of mode and vector adaptation
/// (`MODE_MV_MAX_UPDATE_FACTOR` 128 * count / `MODE_MV_COUNT_SAT` 20).
const MODE_MV_UPDATE_FACTOR: [u32; 21] = [
    0, 6, 12, 19, 25, 32, 38, 44, 51, 57, 64, 70, 76, 83, 89, 96, 102, 108, 115, 121, 128,
];
const MODE_MV_COUNT_SAT: u32 = 20;

/// `vp9_mv_joint_tree`: ZERO, column only, row only, both.
pub(crate) const MV_JOINT: &Tree = &Tree::new(&[0, 2, -1, 4, -2, -3]);
/// `vp9_mv_class_tree`: classes 0..10.
pub(crate) const MV_CLASS: &Tree = &Tree::new(&[
    0, 2, -1, 4, 6, 8, -2, -3, 10, 12, -4, -5, -6, 14, 16, 18, -7, -8, -9, -10,
]);
/// `vp9_mv_class0_tree`.
pub(crate) const MV_CLASS0: &Tree = &Tree::new(&[0, -1]);
/// `vp9_mv_fp_tree`: fractional positions 0..3.
pub(crate) const MV_FP: &Tree = &Tree::new(&[0, 2, -1, 4, -2, -3]);

/// Integer-offset bits of the largest vector class (`MV_OFFSET_BITS`).
pub(crate) const MV_OFFSET_BITS: usize = 10;

/// Probabilities of one vector component (`nmv_component`, without the
/// high-precision bits this subset never codes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MvComponentProbs {
    pub sign: u8,
    pub classes: [u8; 10],
    pub class0: [u8; 1],
    pub bits: [u8; MV_OFFSET_BITS],
    pub class0_fp: [[u8; 3]; 2],
    pub fp: [u8; 3],
    /// High-precision (1/8) bit of class 0 and of the other classes.
    pub class0_hp: u8,
    pub hp: u8,
}

/// Every adaptive probability of an inter frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FrameContext {
    pub coef: [[[BandProbs; 2]; 2]; 4],
    pub skip: [u8; 3],
    pub intra_inter: [u8; 4],
    pub single_ref: [[u8; 2]; 5],
    /// `comp_inter_prob`: compound or single prediction, by context.
    pub comp_inter: [u8; 5],
    /// `comp_ref_prob`: the variable reference of a compound block.
    pub comp_ref: [u8; 5],
    pub inter_mode: [[u8; 3]; 7],
    pub y_mode: [[u8; 9]; 4],
    pub uv_mode: [[u8; 9]; 10],
    pub partition: [[u8; 3]; 16],
    /// `tx_probs` by largest transform (8, 16, 32) and context.
    pub tx8: [[u8; 1]; 2],
    pub tx16: [[u8; 2]; 2],
    pub tx32: [[u8; 3]; 2],
    pub mv_joints: [u8; 3],
    pub mv: [MvComponentProbs; 2],
    /// `switchable_interp_prob` by filter context.
    pub switchable_interp: [[u8; 2]; SWITCHABLE_CONTEXTS],
}

/// `default_comp_inter_p` (`vp9_entropymode.c`).
const DEFAULT_COMP_INTER: [u8; 5] = [239, 183, 119, 96, 41];
/// `default_comp_ref_p`.
const DEFAULT_COMP_REF: [u8; 5] = [50, 126, 123, 221, 226];

/// `SWITCHABLE_FILTER_CONTEXTS`: the above/left filters agree on one of
/// the three, or not.
pub(crate) const SWITCHABLE_CONTEXTS: usize = 4;
/// `SWITCHABLE_FILTERS`.
pub(crate) const SWITCHABLE_FILTERS: usize = 3;

impl Default for FrameContext {
    /// The default context (`vp9_setup_past_independence`).
    fn default() -> Self {
        let component = |classes: [u8; 10], class0: u8| MvComponentProbs {
            sign: 128,
            classes,
            class0: [class0],
            bits: [136, 140, 148, 160, 176, 192, 224, 234, 234, 240],
            class0_fp: [[128, 128, 64], [96, 112, 64]],
            fp: [64, 96, 64],
            class0_hp: 160,
            hp: 128,
        };
        Self {
            coef: tables::COEFFICIENTS,
            skip: tables::SKIP,
            intra_inter: tables::INTRA_INTER,
            single_ref: tables::SINGLE_REF,
            comp_inter: DEFAULT_COMP_INTER,
            comp_ref: DEFAULT_COMP_REF,
            inter_mode: tables::INTER_MODE,
            y_mode: tables::Y_MODE,
            uv_mode: tables::UV_MODE,
            partition: tables::PARTITION,
            tx8: [[100], [66]],
            tx16: [[20, 152], [15, 101]],
            tx32: [[3, 136, 37], [5, 52, 13]],
            mv_joints: [32, 64, 96],
            mv: [
                component([224, 144, 192, 168, 192, 176, 192, 198, 198, 245], 216),
                component([216, 128, 176, 160, 176, 176, 192, 198, 198, 208], 208),
            ],
            // `default_switchable_interp_prob`.
            switchable_interp: [[235, 162], [36, 255], [34, 3], [149, 144]],
        }
    }
}

/// Counts of one vector component's symbols.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct MvComponentCounts {
    pub sign: [u32; 2],
    pub classes: [u32; 11],
    pub class0: [u32; 2],
    pub bits: [[u32; 2]; MV_OFFSET_BITS],
    pub class0_fp: [[u32; 4]; 2],
    pub fp: [u32; 4],
    pub class0_hp: [u32; 2],
    pub hp: [u32; 2],
}

/// Coefficient token counts of one band/context: ZERO, ONE, TWO+ and
/// end-of-block (`EOB_MODEL_TOKEN`), plus how often the end-of-block
/// branch was coded.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct TokenCounts {
    pub tokens: [u32; 4],
    pub eob_branch: u32,
}

/// Token counts of one transform size, plane type and reference type:
/// `[band][context]`.
pub(crate) type BandCounts = [[TokenCounts; 6]; 6];

/// Index of the end-of-block count in [`TokenCounts::tokens`].
pub(crate) const EOB_MODEL_TOKEN: usize = 3;

/// The symbols coded in one frame.
#[derive(Debug, Clone)]
pub(crate) struct Counts {
    pub coef: Box<[[[BandCounts; 2]; 2]; 4]>,
    pub skip: [[u32; 2]; 3],
    pub intra_inter: [[u32; 2]; 4],
    pub single_ref: [[[u32; 2]; 2]; 5],
    pub comp_inter: [[u32; 2]; 5],
    pub comp_ref: [[u32; 2]; 5],
    pub inter_mode: [[u32; 4]; 7],
    pub y_mode: [[u32; 10]; 4],
    pub uv_mode: [[u32; 10]; 10],
    pub partition: [[u32; 4]; 16],
    /// Transform sizes chosen, by largest transform and context.
    pub tx8: [[u32; 2]; 2],
    pub tx16: [[u32; 3]; 2],
    pub tx32: [[u32; 4]; 2],
    pub mv_joints: [u32; 4],
    pub mv: [MvComponentCounts; 2],
    pub switchable_interp: [[u32; SWITCHABLE_FILTERS]; SWITCHABLE_CONTEXTS],
    /// Whether the frame allowed high-precision vectors: their bit
    /// probabilities adapt only then (`vp9_adapt_mv_probs(allow_hp)`).
    pub high_precision: bool,
}

impl Counts {
    /// Count a coded transform size (`get_tx_counts`).
    pub(crate) fn count_tx(
        &mut self,
        max: crate::transform::TxSize,
        ctx: usize,
        tx: crate::transform::TxSize,
    ) {
        use crate::transform::TxSize;
        let index = tx as usize;
        match max {
            TxSize::Tx8 => self.tx8[ctx][index] += 1,
            TxSize::Tx16 => self.tx16[ctx][index] += 1,
            TxSize::Tx32 => self.tx32[ctx][index] += 1,
            TxSize::Tx4 => {}
        }
    }
}

impl Default for Counts {
    fn default() -> Self {
        Self {
            coef: Box::new([[[[[TokenCounts::default(); 6]; 6]; 2]; 2]; 4]),
            skip: [[0; 2]; 3],
            intra_inter: [[0; 2]; 4],
            single_ref: [[[0; 2]; 2]; 5],
            comp_inter: [[0; 2]; 5],
            comp_ref: [[0; 2]; 5],
            inter_mode: [[0; 4]; 7],
            y_mode: [[0; 10]; 4],
            uv_mode: [[0; 10]; 10],
            partition: [[0; 4]; 16],
            tx8: [[0; 2]; 2],
            tx16: [[0; 3]; 2],
            tx32: [[0; 4]; 2],
            mv_joints: [0; 4],
            mv: [MvComponentCounts::default(); 2],
            switchable_interp: [[0; SWITCHABLE_FILTERS]; SWITCHABLE_CONTEXTS],
            high_precision: false,
        }
    }
}

/// `get_prob`: probability of zero from counts, clipped to 1..=255.
fn probability(zeros: u32, total: u32) -> u32 {
    let p = (u64::from(zeros) * 256 + u64::from(total >> 1)) / u64::from(total);
    p.clamp(1, 255) as u32
}

/// `weighted_prob`.
fn weighted(pre: u8, prob: u32, factor: u32) -> u8 {
    ((u32::from(pre) * (256 - factor) + prob * factor + 128) >> 8) as u8
}

/// `merge_probs` (coefficient adaptation).
fn merge_coef(pre: u8, [zeros, ones]: [u32; 2], update_factor: u32) -> u8 {
    let total = zeros + ones;
    let prob = if total == 0 {
        128
    } else {
        probability(zeros, total)
    };
    let factor = update_factor * total.min(COEF_COUNT_SAT) / COEF_COUNT_SAT;
    weighted(pre, prob, factor)
}

/// `mode_mv_merge_probs`.
fn merge_mode(pre: u8, [zeros, ones]: [u32; 2]) -> u8 {
    let total = zeros + ones;
    if total == 0 {
        return pre;
    }
    let factor = MODE_MV_UPDATE_FACTOR[total.min(MODE_MV_COUNT_SAT) as usize];
    weighted(pre, probability(zeros, total), factor)
}

/// `vpx_tree_merge_probs`: merge every node of `tree` from leaf counts.
fn merge_tree(tree: &Tree, pre: &[u8], counts: &[u32], probs: &mut [u8]) {
    fn node(tree: &Tree, index: usize, pre: &[u8], counts: &[u32], probs: &mut [u8]) -> u32 {
        let mut side = |entry: i8| {
            if entry <= 0 {
                counts[(-entry) as usize]
            } else {
                node(tree, entry as usize, pre, counts, probs)
            }
        };
        let left = side(tree.entries()[index]);
        let right = side(tree.entries()[index + 1]);
        probs[index >> 1] = merge_mode(pre[index >> 1], [left, right]);
        left + right
    }
    node(tree, 0, pre, counts, probs);
}

impl FrameContext {
    /// `get_tx_probs`: the transform-size probabilities of a block whose
    /// largest transform is `max` (8x8 or larger), in context `ctx`.
    pub(crate) fn tx_probs(&self, max: crate::transform::TxSize, ctx: usize) -> &[u8] {
        use crate::transform::TxSize;
        match max {
            TxSize::Tx8 => &self.tx8[ctx],
            TxSize::Tx16 => &self.tx16[ctx],
            TxSize::Tx32 => &self.tx32[ctx],
            TxSize::Tx4 => &[],
        }
    }

    /// The context after coding a frame that started from `self` with
    /// `counts`: coefficients always, modes and vectors after inter frames
    /// (`vp9_adapt_coef_probs`, `vp9_adapt_mode_probs`, `vp9_adapt_mv_probs`).
    /// `after_key` marks the first frame after a key frame.
    /// `tx_select` says the frame coded transform sizes (`TX_MODE_SELECT`).
    pub(crate) fn adapted(
        &self,
        counts: &Counts,
        inter: bool,
        after_key: bool,
        tx_select: bool,
    ) -> Self {
        self.adapted_from_coded(self, counts, inter, after_key, tx_select)
    }

    /// [`Self::adapted`] for a frame coded with `coded` (`self` plus the
    /// frame's forward updates): a probability that does not adapt keeps
    /// its coded value (the forward-updated transform and skip
    /// probabilities of a key frame, the 1/8-bit ones of a frame without
    /// high-precision vectors), as libvpx's `cm->fc`.
    pub(crate) fn adapted_from_coded(
        &self,
        coded: &Self,
        counts: &Counts,
        inter: bool,
        after_key: bool,
        tx_select: bool,
    ) -> Self {
        let mut next = coded.clone();
        next.adapt(
            self,
            counts,
            Adaptation {
                after_key,
                modes: inter,
                tx_select,
                switchable: true,
                allow_hp: counts.high_precision,
            },
        );
        next
    }

    /// Backward adaptation of a frame's context (`self`: the frame's
    /// probabilities, forward updates included) from `pre`, the context the
    /// frame started from, and the frame's `counts`. Every adapted
    /// probability is merged from `pre`; the others keep the frame's value,
    /// as libvpx's `cm->fc` does.
    pub(crate) fn adapt(&mut self, pre: &Self, counts: &Counts, adaptation: Adaptation) {
        let factor = if adaptation.after_key {
            COEF_MAX_UPDATE_FACTOR_AFTER_KEY
        } else {
            COEF_MAX_UPDATE_FACTOR
        };
        pre.adapt_coefficients(self, counts, factor);
        if adaptation.modes {
            pre.adapt_modes(self, counts, adaptation);
            pre.adapt_vectors(self, counts, adaptation.allow_hp);
        }
    }

    /// `vp9_adapt_coef_probs`.
    fn adapt_coefficients(&self, next: &mut Self, counts: &Counts, factor: u32) {
        for (t, tx) in counts.coef.iter().enumerate() {
            for (p, plane) in tx.iter().enumerate() {
                for (r, reference) in plane.iter().enumerate() {
                    for (b, band) in reference.iter().enumerate() {
                        let contexts = if b == 0 { 3 } else { 6 };
                        for (c, count) in band.iter().enumerate().take(contexts) {
                            let [n0, n1, n2, neob] = count.tokens;
                            let branches =
                                [[neob, count.eob_branch - neob], [n0, n1 + n2], [n1, n2]];
                            let pre = &self.coef[t][p][r][b][c];
                            for (node, branch) in branches.into_iter().enumerate() {
                                next.coef[t][p][r][b][c][node] =
                                    merge_coef(pre[node], branch, factor);
                            }
                        }
                    }
                }
            }
        }
    }

    /// `vp9_adapt_mode_probs`.
    fn adapt_modes(&self, next: &mut Self, counts: &Counts, adaptation: Adaptation) {
        for i in 0..4 {
            next.intra_inter[i] = merge_mode(self.intra_inter[i], counts.intra_inter[i]);
        }
        for i in 0..5 {
            next.comp_inter[i] = merge_mode(self.comp_inter[i], counts.comp_inter[i]);
            next.comp_ref[i] = merge_mode(self.comp_ref[i], counts.comp_ref[i]);
        }
        for i in 0..5 {
            for j in 0..2 {
                next.single_ref[i][j] = merge_mode(self.single_ref[i][j], counts.single_ref[i][j]);
            }
        }
        for i in 0..7 {
            merge_tree(
                tree::INTER_MODE,
                &self.inter_mode[i],
                &counts.inter_mode[i],
                &mut next.inter_mode[i],
            );
        }
        for i in 0..4 {
            merge_tree(
                tree::INTRA_MODE,
                &self.y_mode[i],
                &counts.y_mode[i],
                &mut next.y_mode[i],
            );
        }
        for i in 0..10 {
            merge_tree(
                tree::INTRA_MODE,
                &self.uv_mode[i],
                &counts.uv_mode[i],
                &mut next.uv_mode[i],
            );
        }
        for i in 0..16 {
            merge_tree(
                tree::PARTITION,
                &self.partition[i],
                &counts.partition[i],
                &mut next.partition[i],
            );
        }
        // Only switchable frames adapt the filter probabilities.
        for ctx in (0..SWITCHABLE_CONTEXTS).filter(|_| adaptation.switchable) {
            merge_tree(
                tree::SWITCHABLE_INTERP,
                &self.switchable_interp[ctx],
                &counts.switchable_interp[ctx],
                &mut next.switchable_interp[ctx],
            );
        }
        if adaptation.tx_select {
            // `tx_counts_to_branch_counts_*`: each node splits the sizes
            // at or below it from the larger ones.
            for ctx in 0..2 {
                let [a, b] = counts.tx8[ctx];
                next.tx8[ctx][0] = merge_mode(self.tx8[ctx][0], [a, b]);
                let [a, b, c] = counts.tx16[ctx];
                next.tx16[ctx][0] = merge_mode(self.tx16[ctx][0], [a, b + c]);
                next.tx16[ctx][1] = merge_mode(self.tx16[ctx][1], [b, c]);
                let [a, b, c, d] = counts.tx32[ctx];
                next.tx32[ctx][0] = merge_mode(self.tx32[ctx][0], [a, b + c + d]);
                next.tx32[ctx][1] = merge_mode(self.tx32[ctx][1], [b, c + d]);
                next.tx32[ctx][2] = merge_mode(self.tx32[ctx][2], [c, d]);
            }
        }
        for i in 0..3 {
            next.skip[i] = merge_mode(self.skip[i], counts.skip[i]);
        }
    }

    /// `vp9_adapt_mv_probs`; the 1/8 bits adapt only when the frame
    /// allowed them.
    fn adapt_vectors(&self, next: &mut Self, counts: &Counts, allow_hp: bool) {
        merge_tree(
            MV_JOINT,
            &self.mv_joints,
            &counts.mv_joints,
            &mut next.mv_joints,
        );
        for (i, c) in counts.mv.iter().enumerate() {
            let (pre, comp) = (&self.mv[i], &mut next.mv[i]);
            comp.sign = merge_mode(pre.sign, c.sign);
            merge_tree(MV_CLASS, &pre.classes, &c.classes, &mut comp.classes);
            merge_tree(MV_CLASS0, &pre.class0, &c.class0, &mut comp.class0);
            for j in 0..MV_OFFSET_BITS {
                comp.bits[j] = merge_mode(pre.bits[j], c.bits[j]);
            }
            for j in 0..2 {
                merge_tree(
                    MV_FP,
                    &pre.class0_fp[j],
                    &c.class0_fp[j],
                    &mut comp.class0_fp[j],
                );
            }
            merge_tree(MV_FP, &pre.fp, &c.fp, &mut comp.fp);
            if allow_hp {
                comp.class0_hp = merge_mode(pre.class0_hp, c.class0_hp);
                comp.hp = merge_mode(pre.hp, c.hp);
            }
        }
    }
}

/// Which parts of a context one frame adapts (`vp9_decode_frame`).
#[allow(
    clippy::struct_excessive_bools,
    reason = "independent switches of the reference"
)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Adaptation {
    /// The previous frame was a key frame: coefficients adapt faster.
    pub after_key: bool,
    /// Inter frame: modes and vectors adapt too.
    pub modes: bool,
    /// `TX_MODE_SELECT`: the transform-size probabilities adapt.
    pub tx_select: bool,
    /// Switchable interpolation filter: its probabilities adapt.
    pub switchable: bool,
    /// `allow_high_precision_mv`: the 1/8 bit probabilities adapt.
    pub allow_hp: bool,
}

/// Coefficient probabilities chosen for one frame by forward updates.
#[derive(Debug, Clone)]
pub(crate) struct CoefUpdates {
    /// The frame's coefficient table (the context's, with updates).
    pub coef: [[[BandProbs; 2]; 2]; 4],
    /// Transform sizes whose table is sent (`update_coef_probs_common`).
    pub sent: [bool; 4],
}

impl CoefUpdates {
    /// No updates: the context's table.
    pub(crate) fn none(context: &FrameContext) -> Self {
        Self {
            coef: context.coef,
            sent: [false; 4],
        }
    }

    /// Updates of nodes 0 and 1 that pay for themselves on this frame's
    /// counts (libvpx `vp9_prob_diff_update_savings_search`: from the
    /// frame's own probability towards the old one, keep the best saving);
    /// a size is sent when its savings exceed the cost of its "no update"
    /// flags. The pivot node keeps its probability (it also shapes the
    /// Pareto tail).
    pub(crate) fn choose(context: &FrameContext, counts: &Counts, sizes: usize) -> Self {
        let mut updates = Self::none(context);
        let flag_cost = crate::tree::bit_cost(false, crate::subexp::DIFF_UPDATE_PROB);
        let tables = context.coef.iter().zip(counts.coef.iter()).enumerate();
        for (t, (table, table_counts)) in tables.take(sizes) {
            let mut candidate = *table;
            let (mut savings, mut nodes) = (0.0, 0u32);
            for (planes, plane_counts) in candidate.iter_mut().zip(table_counts.iter()) {
                for (references, reference_counts) in planes.iter_mut().zip(plane_counts) {
                    for (b, (bands, band_counts)) in
                        references.iter_mut().zip(reference_counts).enumerate()
                    {
                        let contexts = if b == 0 { 3 } else { 6 };
                        for (probs, count) in bands.iter_mut().zip(band_counts).take(contexts) {
                            let [n0, n1, n2, neob] = count.tokens;
                            let branches = [[neob, count.eob_branch - neob], [n0, n1 + n2]];
                            for (prob, branch) in probs.iter_mut().zip(branches) {
                                let (new, saving) = best_update(branch, *prob);
                                *prob = new;
                                savings += saving;
                            }
                            nodes += 3;
                        }
                    }
                }
            }
            if savings > nodes as f32 * flag_cost {
                updates.coef[t] = candidate;
                updates.sent[t] = true;
            }
        }
        updates
    }
}

/// Choose forward updates for the [`crate::recorder::MODE_SLOTS`] probabilities
/// against `base` that pay for themselves on `counts` (`VP9 §6.3`).
pub(crate) fn choose_mode_updates(
    base: &FrameContext,
    counts: &[[u32; 2]; crate::recorder::MODE_SLOTS],
) -> [u8; crate::recorder::MODE_SLOTS] {
    let mut probs = crate::recorder::mode_slot_probs(base);
    for (slot, (prob, &branch)) in probs.iter_mut().zip(counts).enumerate() {
        *prob = if slot < crate::recorder::MV_JOINT_SLOT {
            best_update(branch, *prob).0
        } else {
            best_mv_update(branch, *prob)
        };
    }
    probs
}

/// Best forward update for an MV probability (`update_mv_probs`, `VP9 §6.3.14`):
/// signaled with a `252`-probability flag and a 7-bit literal `(prob >> 1)`
/// whose LSB is implicitly `1`.
fn best_mv_update(branch: [u32; 2], old: u8) -> u8 {
    let total = branch[0] + branch[1];
    if total == 0 {
        return old;
    }
    let candidate = (probability(branch[0], total) as u8) | 1;
    let cost = crate::tree::bit_cost(true, crate::subexp::DIFF_UPDATE_PROB)
        - crate::tree::bit_cost(false, crate::subexp::DIFF_UPDATE_PROB)
        + crate::recorder::MV_UPDATE_LITERAL_BITS as f32;
    if branch_bits(branch, old) - branch_bits(branch, candidate) > cost {
        candidate
    } else {
        old
    }
}

/// Bits of `branch` (zeros, ones) coded with probability-of-zero `prob`.
fn branch_bits([zeros, ones]: [u32; 2], prob: u8) -> f32 {
    zeros as f32 * crate::tree::bit_cost(false, prob)
        + ones as f32 * crate::tree::bit_cost(true, prob)
}

/// The update of `old` that saves most on `branch` (after its own cost),
/// or `old` with no saving.
fn best_update(branch: [u32; 2], old: u8) -> (u8, f32) {
    let total = branch[0] + branch[1];
    if total == 0 {
        return (old, 0.0);
    }
    let target = probability(branch[0], total) as u8;
    let before = branch_bits(branch, old);
    // Candidates from the frame's own probability towards the old one, in
    // up to `UPDATE_SEARCH_STEPS` even steps (libvpx walks every value).
    let distance = i32::from(target) - i32::from(old);
    let step = (distance.abs() / UPDATE_SEARCH_STEPS).max(1) * distance.signum();
    let mut best = (old, 0.0);
    let mut prob = i32::from(target);
    while prob != i32::from(old) && (prob - i32::from(old)).signum() == distance.signum() {
        let candidate = prob as u8;
        let saving =
            before - branch_bits(branch, candidate) - crate::subexp::update_cost(old, candidate);
        if saving > best.1 {
            best = (candidate, saving);
        }
        prob -= step;
    }
    best
}

/// Candidate probabilities [`best_update`] evaluates per node.
const UPDATE_SEARCH_STEPS: i32 = 8;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merging_follows_the_reference_arithmetic() {
        // No counts: unchanged. Saturated counts move by the full factor.
        assert_eq!(merge_mode(200, [0, 0]), 200);
        assert_eq!(merge_mode(128, [40, 0]), weighted(128, 255, 128));
        assert_eq!(merge_coef(100, [0, 0], 112), 100);
        assert_eq!(merge_coef(100, [12, 12], 112), weighted(100, 128, 112));
        assert_eq!(merge_coef(100, [6, 6], 112), weighted(100, 128, 56));
    }

    #[test]
    fn tree_merge_counts_every_leaf_once() {
        let mut probs = [0u8; 3];
        merge_tree(tree::PARTITION, &[128; 3], &[20, 0, 0, 20], &mut probs);
        // Node 0 splits NONE (20) from the rest (20): stays near 128.
        assert_eq!(probs[0], weighted(128, 128, 128));
        assert_eq!(probs[2], weighted(128, 1, 128));
    }
}
