//! The distilled flip guide (`planner.flip_tree`): a small decision tree
//! that tells a fast rung's planner where to take the other choice of leaf
//! or quad split, the one the slowest rung's RD partition search would code,
//! without a trial (docs/analysis/distill-fast-2026-11-22.md).
//!
//! Where it fits: the inter planner (`partition_plan.rs`) decides each node
//! from its variance model. On the fast rungs the RD search's partitions are
//! worth AV1 -4..-12 %, HEVC -6..-12 % J and cost almost nothing more to
//! code (the transplant oracle), but the search itself costs +55..150 %
//! cycles. The tree predicts the search's choice at the planner's own cost.
//!
//! Its inputs are what the planner already holds at a node it weighed both
//! ways: the eleven trial bytes (`trial_inputs.rs`), the node's position in
//! its superblock and the frame's mode-decision λ. Thresholds compare those
//! values exactly as training did (λ itself, not its logarithm), so every
//! platform takes the same branches.
//!
//! The trees are generated tables (`flip_trees.rs`, `eval/distill/
//! export_tree.py`) whose headers record their training. An oracle-grade
//! experiment, default off: no preset sets it.

use crate::flip_trees::TREES;
use crate::partition::Block;
use crate::trial_inputs::{TRIAL_INPUT_COUNT, TrialInputs};

/// Inputs of a tree: the trial bytes, the node's x and y offset in its
/// superblock (in superblock sides) and λ.
pub(crate) const FLIP_INPUTS: usize = TRIAL_INPUT_COUNT + 3;

/// One node of a tree.
#[derive(Debug, Clone, Copy)]
pub(crate) enum TreeNode {
    /// Go to `left` when input `input` is at most `threshold`, else `right`.
    Inner {
        input: u8,
        threshold: f32,
        left: u16,
        right: u16,
    },
    /// Whether to flip the planner's choice.
    Leaf(bool),
}

/// A tree, root first.
#[derive(Debug)]
pub(crate) struct FlipTree {
    pub nodes: &'static [TreeNode],
}

impl FlipTree {
    /// Whether the tree flips a node with these inputs.
    fn flips(&self, inputs: &[f32; FLIP_INPUTS]) -> bool {
        let mut at = 0;
        loop {
            match self.nodes[at] {
                TreeNode::Leaf(flip) => return flip,
                TreeNode::Inner {
                    input,
                    threshold,
                    left,
                    right,
                } => {
                    at = usize::from(if inputs[usize::from(input)] <= threshold {
                        left
                    } else {
                        right
                    });
                }
            }
        }
    }
}

/// Which flips the guide may make (`planner.flip_direction`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FlipDirection {
    Both,
    /// Only a planned split coded as its leaf: the flip's cost is the
    /// planner's own leaf, so its label (the coded leaf) is what it gets.
    MergeOnly,
    /// Only a planned leaf split: the quadrants are then the planner's, not
    /// the search's the label priced.
    SplitOnly,
}

impl FlipDirection {
    pub(crate) fn of(value: f32) -> Self {
        match value.round() as i32 {
            1 => Self::MergeOnly,
            2 => Self::SplitOnly,
            _ => Self::Both,
        }
    }

    /// Whether a flip of a node the planner `planned_split` is allowed.
    pub(crate) fn allows(self, planned_split: bool) -> bool {
        match self {
            Self::Both => true,
            Self::MergeOnly => planned_split,
            Self::SplitOnly => !planned_split,
        }
    }
}

/// The tree `planner.flip_tree` selects (1-based; 0 or out of range: none).
pub(crate) fn tree(selector: f32) -> Option<&'static FlipTree> {
    let index = selector.round() as usize;
    index.checked_sub(1).and_then(|i| TREES.get(i))
}

/// Whether `tree` flips the planner's choice at `node` of superblock `sb`.
pub(crate) fn flips(
    tree: &FlipTree,
    inputs: &TrialInputs,
    (node, sb): (&Block, &Block),
    lambda: f32,
) -> bool {
    let mut x = [0.0f32; FLIP_INPUTS];
    for (slot, byte) in x.iter_mut().zip(inputs.0) {
        *slot = f32::from(byte);
    }
    let side = sb.size() as f32;
    x[TRIAL_INPUT_COUNT] = (node.x - sb.x) as f32 / side;
    x[TRIAL_INPUT_COUNT + 1] = (node.y - sb.y) as f32 / side;
    x[TRIAL_INPUT_COUNT + 2] = lambda;
    tree.flips(&x)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// input 0 <= 10 -> leaf(no flip); else input 13 (lambda) <= 5 -> flip, else no.
    const NODES: [TreeNode; 5] = [
        TreeNode::Inner {
            input: 0,
            threshold: 10.0,
            left: 1,
            right: 2,
        },
        TreeNode::Leaf(false),
        TreeNode::Inner {
            input: 13,
            threshold: 5.0,
            left: 3,
            right: 4,
        },
        TreeNode::Leaf(true),
        TreeNode::Leaf(false),
    ];

    #[test]
    fn a_tree_walks_to_its_leaf() {
        let tree = FlipTree { nodes: &NODES };
        let mut inputs = TrialInputs([0; TRIAL_INPUT_COUNT]);
        let (node, sb) = (Block::square(16, 0, 4), Block::square(0, 0, 6));
        assert!(!flips(&tree, &inputs, (&node, &sb), 1.0));
        inputs.0[0] = 20;
        assert!(flips(&tree, &inputs, (&node, &sb), 1.0));
        assert!(!flips(&tree, &inputs, (&node, &sb), 9.0));
    }

    #[test]
    fn no_tree_is_selected_by_zero() {
        assert!(tree(0.0).is_none());
    }
}
