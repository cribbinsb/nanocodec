//! The tile's bits, recorded during encoding and written once the frame's
//! forward probability updates are known (libvpx tokenizes first and packs
//! the bitstream afterwards for the same reason).
//!
//! Coefficient "more coefficients" and "nonzero" decisions (nodes 0 and 1
//! of every band/context model) are recorded by model slot and written with
//! the frame's updated probability, and segment ids with the frame's own
//! segment-tree probabilities; every other bit keeps the probability it
//! was coded with.

use crate::bool_coder::{BitSink, BoolWriter};
use crate::probs::FrameContext;
use crate::segmentation::{self, SegmentStats, TREE_PROBS};
use crate::tables::BandProbs;
use crate::transform::TxSize;
use crate::tree::Tree;

/// Updatable coefficient nodes: 4 sizes x 2 plane types x 2 references x
/// 6 bands x 6 contexts x nodes 0 and 1.
pub(crate) const COEF_SLOTS: usize = 4 * 2 * 2 * 6 * 6 * UPDATED_NODES;
/// Nodes of each model the encoder updates (the pivot node 2 shapes the
/// Pareto tail and keeps its probability).
pub(crate) const UPDATED_NODES: usize = 2;
/// Marks a bit with a fixed probability.
const FIXED: u16 = u16::MAX;
/// Slot of segment-tree node 0; the nodes follow the coefficient slots.
const SEGMENT_TREE_SLOT: u16 = COEF_SLOTS as u16;
/// Slot of mode/transform/skip/partition probability 0; follows the segment tree.
const MODE_SLOT_BASE: u16 = SEGMENT_TREE_SLOT + TREE_PROBS as u16;

const TX8_SLOT: usize = 0;
const TX16_SLOT: usize = TX8_SLOT + 2;
const TX32_SLOT: usize = TX16_SLOT + 4;
pub(crate) const SKIP_SLOT: usize = TX32_SLOT + 6;
pub(crate) const INTER_MODE_SLOT: usize = SKIP_SLOT + 3;
pub(crate) const SWITCHABLE_INTERP_SLOT: usize = INTER_MODE_SLOT + 21;
pub(crate) const INTRA_INTER_SLOT: usize = SWITCHABLE_INTERP_SLOT + 8;
const SINGLE_REF_SLOT: usize = INTRA_INTER_SLOT + 4;
const Y_MODE_SLOT: usize = SINGLE_REF_SLOT + 10;
const PARTITION_SLOT: usize = Y_MODE_SLOT + 36;
pub(crate) const MV_JOINT_SLOT: usize = PARTITION_SLOT + 48;
const MV_COMP_SLOT: usize = MV_JOINT_SLOT + 3;
const MV_FP_BASE_SLOT: usize = MV_COMP_SLOT + 2 * 22;
pub(crate) const MV_HP_BASE_SLOT: usize = MV_FP_BASE_SLOT + 2 * 9;
/// Bits of an explicit motion-vector probability update: the new
/// probability's top 7 bits, its lowest bit implied 1 (`update_mv_prob`,
/// VP9 §6.3.17).
pub(crate) const MV_UPDATE_LITERAL_BITS: u32 = 7;
/// Updatable non-coefficient mode/skip/partition/tx/mv probability slots in
/// compressed-header order (`VP9 §6.3`).
pub(crate) const MODE_SLOTS: usize = MV_HP_BASE_SLOT + 2 * 2;

pub(crate) fn mv_sign_slot(axis: usize) -> usize {
    MV_COMP_SLOT + axis * 22
}

pub(crate) fn mv_classes_slot(axis: usize) -> usize {
    MV_COMP_SLOT + axis * 22 + 1
}

pub(crate) fn mv_class0_slot(axis: usize) -> usize {
    MV_COMP_SLOT + axis * 22 + 11
}

pub(crate) fn mv_bits_slot(axis: usize, bit: usize) -> usize {
    MV_COMP_SLOT + axis * 22 + 12 + bit
}

pub(crate) fn mv_class0_fp_slot(axis: usize, d: usize) -> usize {
    MV_FP_BASE_SLOT + axis * 9 + d * 3
}

pub(crate) fn mv_fp_slot(axis: usize) -> usize {
    MV_FP_BASE_SLOT + axis * 9 + 6
}

pub(crate) fn mv_class0_hp_slot(axis: usize) -> usize {
    MV_HP_BASE_SLOT + axis * 2
}

pub(crate) fn mv_hp_slot(axis: usize) -> usize {
    MV_HP_BASE_SLOT + axis * 2 + 1
}

pub(crate) fn tx_slot(max: TxSize, ctx: usize) -> usize {
    match max {
        TxSize::Tx8 => TX8_SLOT + ctx,
        TxSize::Tx16 => TX16_SLOT + ctx * 2,
        TxSize::Tx32 => TX32_SLOT + ctx * 3,
        TxSize::Tx4 => unreachable!("Tx4 codes no transform size"),
    }
}

pub(crate) fn skip_slot(ctx: usize) -> usize {
    SKIP_SLOT + ctx
}

pub(crate) fn inter_mode_slot(ctx: usize) -> usize {
    INTER_MODE_SLOT + ctx * 3
}

pub(crate) fn switchable_interp_slot(ctx: usize) -> usize {
    SWITCHABLE_INTERP_SLOT + ctx * 2
}

pub(crate) fn intra_inter_slot(ctx: usize) -> usize {
    INTRA_INTER_SLOT + ctx
}

pub(crate) fn single_ref_slot(ctx: usize, bit: usize) -> usize {
    SINGLE_REF_SLOT + ctx * 2 + bit
}

pub(crate) fn y_mode_slot(group: usize) -> usize {
    Y_MODE_SLOT + group * 9
}

pub(crate) fn partition_slot(ctx: usize) -> usize {
    PARTITION_SLOT + ctx * 3
}

/// The [`MODE_SLOTS`] probabilities of `fc` in compressed-header order.
pub(crate) fn mode_slot_probs(fc: &FrameContext) -> [u8; MODE_SLOTS] {
    let mut probs = [0u8; MODE_SLOTS];
    // One walk of the slots serves both directions (once per frame).
    visit_mode_slots(&mut fc.clone(), |slot, prob| probs[slot] = *prob);
    probs
}

/// Put the [`MODE_SLOTS`] probabilities `probs` into `fc` (a frame's
/// forward mode updates, VP9 §6.3).
pub(crate) fn set_mode_slot_probs(fc: &mut FrameContext, probs: &[u8; MODE_SLOTS]) {
    visit_mode_slots(fc, |slot, prob| *prob = probs[slot]);
}

/// Call `visit` with each mode slot and its probability in `fc`.
fn visit_mode_slots(fc: &mut FrameContext, mut visit: impl FnMut(usize, &mut u8)) {
    let mut run = |first: usize, probs: &mut [u8]| {
        for (offset, prob) in probs.iter_mut().enumerate() {
            visit(first + offset, prob);
        }
    };
    for ctx in 0..2 {
        run(tx_slot(TxSize::Tx8, ctx), &mut fc.tx8[ctx]);
        run(tx_slot(TxSize::Tx16, ctx), &mut fc.tx16[ctx]);
        run(tx_slot(TxSize::Tx32, ctx), &mut fc.tx32[ctx]);
    }
    run(SKIP_SLOT, &mut fc.skip);
    for ctx in 0..7 {
        run(inter_mode_slot(ctx), &mut fc.inter_mode[ctx]);
    }
    for ctx in 0..4 {
        run(switchable_interp_slot(ctx), &mut fc.switchable_interp[ctx]);
    }
    run(INTRA_INTER_SLOT, &mut fc.intra_inter);
    for ctx in 0..5 {
        run(single_ref_slot(ctx, 0), &mut fc.single_ref[ctx][..1]);
        run(single_ref_slot(ctx, 1), &mut fc.single_ref[ctx][1..]);
    }
    for group in 0..4 {
        run(y_mode_slot(group), &mut fc.y_mode[group]);
    }
    for ctx in 0..16 {
        run(partition_slot(ctx), &mut fc.partition[ctx]);
    }
    run(MV_JOINT_SLOT, &mut fc.mv_joints);
    for (axis, comp) in fc.mv.iter_mut().enumerate() {
        run(mv_sign_slot(axis), std::slice::from_mut(&mut comp.sign));
        run(mv_classes_slot(axis), &mut comp.classes);
        run(mv_class0_slot(axis), &mut comp.class0);
        run(mv_bits_slot(axis, 0), &mut comp.bits);
        for (d, fp) in comp.class0_fp.iter_mut().enumerate() {
            run(mv_class0_fp_slot(axis, d), fp);
        }
        run(mv_fp_slot(axis), &mut comp.fp);
        run(
            mv_class0_hp_slot(axis),
            std::slice::from_mut(&mut comp.class0_hp),
        );
        run(mv_hp_slot(axis), std::slice::from_mut(&mut comp.hp));
    }
}

/// The slot of node `node` (0 or 1) of a coefficient model.
pub(crate) fn coef_slot(
    tx: TxSize,
    plane: usize,
    reference: usize,
    band: usize,
    ctx: usize,
    node: usize,
) -> u16 {
    ((((((tx as usize * 2 + plane) * 2 + reference) * 6 + band) * 6 + ctx) * UPDATED_NODES) + node)
        as u16
}

/// The updatable node probabilities of a coefficient table, by slot.
pub(crate) fn slot_probs(coef: &[[[BandProbs; 2]; 2]; 4]) -> [u8; COEF_SLOTS] {
    let mut probs = [0; COEF_SLOTS];
    for (t, sizes) in coef.iter().enumerate() {
        for (p, planes) in sizes.iter().enumerate() {
            for (r, bands) in planes.iter().enumerate() {
                for (b, contexts) in bands.iter().enumerate() {
                    for (c, nodes) in contexts.iter().enumerate() {
                        for node in 0..UPDATED_NODES {
                            let tx = TxSize::from_log2(t as u8 + 2);
                            probs[usize::from(coef_slot(tx, p, r, b, c, node))] = nodes[node];
                        }
                    }
                }
            }
        }
    }
    probs
}

#[derive(Debug, Clone, Copy)]
struct Recorded {
    bit: bool,
    prob: u8,
    slot: u16,
}

/// Recorded tile bits.
#[derive(Debug, Default)]
pub(crate) struct TileRecorder {
    bits: Vec<Recorded>,
    segment_ids: SegmentStats,
}

impl TileRecorder {
    /// A coefficient node decision coded with `prob`, replayed with the
    /// slot's final probability.
    pub(crate) fn write_coef(&mut self, bit: bool, prob: u8, slot: u16) {
        self.bits.push(Recorded { bit, prob, slot });
    }

    /// A mode/skip/partition/tx node decision coded with `prob` at
    /// `mode_slot` (`0..MODE_SLOTS`), counted and replayed with the slot's
    /// final probability when mode updates are active.
    pub(crate) fn write_mode(&mut self, bit: bool, prob: u8, mode_slot: usize) {
        // Always recorded by slot (no branch per bit): without mode updates
        // the replay keeps the probability the bit was coded with.
        self.bits.push(Recorded {
            bit,
            prob,
            slot: MODE_SLOT_BASE + mode_slot as u16,
        });
    }

    /// Write a tree-coded symbol at `base_mode_slot..base_mode_slot + probs.len()`.
    pub(crate) fn write_mode_tree(
        &mut self,
        tree: &Tree,
        probs: &[u8],
        base_mode_slot: usize,
        value: usize,
    ) {
        for &(node, bit) in tree.path(value) {
            let node = usize::from(node);
            self.write_mode(bit, probs[node], base_mode_slot + node);
        }
    }

    /// Branch `[zeros, ones]` counts of each mode slot, from the recorded
    /// bits (forward mode updates; spatial layers only).
    pub(crate) fn mode_counts(&self) -> Box<[[u32; 2]; MODE_SLOTS]> {
        let mut counts = Box::new([[0; 2]; MODE_SLOTS]);
        for recorded in self
            .bits
            .iter()
            .filter(|r| r.slot != FIXED && r.slot >= MODE_SLOT_BASE)
        {
            counts[usize::from(recorded.slot - MODE_SLOT_BASE)][usize::from(recorded.bit)] += 1;
        }
        counts
    }

    /// An explicitly coded segment id, replayed with the frame's own
    /// tree probabilities (`segment_tree_probs`): a frame that refines a
    /// few blocks then pays almost nothing for the ids of the others
    /// (VP9 TRAIN proxy-4: +1.7% `J_real` with the last segmented frame's
    /// probabilities).
    pub(crate) fn write_segment_id(&mut self, id: u8) {
        self.segment_ids.count(id);
        for (node, bit) in segmentation::tree_path(id) {
            self.bits.push(Recorded {
                bit,
                prob: 0,
                slot: SEGMENT_TREE_SLOT + node as u16,
            });
        }
    }

    /// The segment-tree probabilities of the ids written so far.
    pub(crate) fn segment_tree_probs(&self) -> [u8; TREE_PROBS] {
        self.segment_ids.tree_probs()
    }

    /// Write the tile with the coefficient slots' final probabilities and
    /// the frame's segment-tree probabilities.
    #[cfg(test)]
    pub(crate) fn finish(
        self,
        slot_probs: &[u8; COEF_SLOTS],
        segment_tree_probs: [u8; TREE_PROBS],
    ) -> Vec<u8> {
        self.finish_with_modes(slot_probs, segment_tree_probs, None)
    }

    /// Write the tile with the coefficient slots', segment-tree, and optional
    /// mode slots' final probabilities.
    pub(crate) fn finish_with_modes(
        self,
        slot_probs: &[u8; COEF_SLOTS],
        segment_tree_probs: [u8; TREE_PROBS],
        mode_probs: Option<&[u8; MODE_SLOTS]>,
    ) -> Vec<u8> {
        let mut writer = BoolWriter::new();
        for recorded in self.bits {
            let prob = match recorded.slot {
                FIXED => recorded.prob,
                slot if slot >= MODE_SLOT_BASE => {
                    mode_probs.map_or(recorded.prob, |m| m[usize::from(slot - MODE_SLOT_BASE)])
                }
                slot if slot >= SEGMENT_TREE_SLOT => {
                    segment_tree_probs[usize::from(slot - SEGMENT_TREE_SLOT)]
                }
                slot => slot_probs[usize::from(slot)],
            };
            writer.write(recorded.bit, prob);
        }
        writer.finish()
    }
}

impl BitSink for TileRecorder {
    fn write(&mut self, bit: bool, prob: u8) {
        self.bits.push(Recorded {
            bit,
            prob,
            slot: FIXED,
        });
    }

    fn write_mode(&mut self, bit: bool, prob: u8, slot: usize) {
        TileRecorder::write_mode(self, bit, prob, slot);
    }
}
