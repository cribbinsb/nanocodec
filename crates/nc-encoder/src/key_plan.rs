//! Key-frame block partitioning, planned per superblock before it is coded.
//!
//! `planner.intra_plan` selects the leaf cost:
//!
//! - **Variance** (1): the source variance of the leaf (the DC prediction's
//!   error, ignoring the neighbours' offset) plus λ · its partition bits.
//!   It knows nothing of coefficient cost, nor that AV1's 64x64 transform
//!   drops every frequency above 32: on `bodycam_airport` it spent +15% key
//!   bytes for −0.3 dB against fixed 16x16 blocks.
//! - **Transform** (2): the backend's [`CodecBackend::intra_leaf_cost`], a
//!   DC prediction from the source neighbours, forward transform,
//!   quantisation, coefficient rate and the remaining distortion. This is
//!   SVT-AV1's I-frame PD0 "light PD0 with a transform" (`PD0_LVL_5`,
//!   `docs/reference/svt-av1-m12-rtc.md` §3). Backends without the estimate
//!   keep the fixed size ([`KeyPlanMode::resolve`]), so the default (2)
//!   changes only AV1.
//!
//! Bottom-up decision: the parent stays unless `intra_bias` · parent >
//! 1000 · (children + split bits).

use nc_common::Picture;
use nc_dsp::Dsp;

use crate::backend::CodecBackend;
use crate::decision::source_variance;
use crate::params::PlannerParams;
use crate::partition::Block;
use crate::partition_plan::Partition;

/// Area (an 8x8 block) `planner.intra_flat_stop` is counted per.
const FLAT_STOP_UNIT_AREA: usize = 64;

/// How key-frame leaves are costed (`planner.intra_plan`, rounded).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KeyPlanMode {
    /// No planning: every leaf has the configured fixed size.
    Fixed,
    /// Source variance plus partition bits.
    Variance,
    /// The backend's transform-based leaf cost.
    Transform,
}

impl KeyPlanMode {
    pub(crate) fn from_param(value: f32) -> Self {
        match value.round() as i32 {
            i32::MIN..=0 => Self::Fixed,
            1 => Self::Variance,
            _ => Self::Transform,
        }
    }

    /// The mode `backend` can run: transform planning needs its leaf cost.
    pub(crate) fn resolve<B: CodecBackend>(value: f32, backend: &B) -> Self {
        match Self::from_param(value) {
            Self::Transform if !backend.supports_intra_leaf_cost() => Self::Fixed,
            mode => mode,
        }
    }
}

/// Everything the key-frame planner needs for one superblock.
pub(crate) struct KeyPlanner<'a, B: CodecBackend> {
    pub dsp: &'a Dsp,
    /// Mutable only for [`CodecBackend::intra_leaf_cost`]'s scratch.
    pub backend: &'a mut B,
    pub source: &'a Picture,
    /// Coded picture size.
    pub width: usize,
    pub height: usize,
    pub params: &'a PlannerParams,
    pub mode: KeyPlanMode,
}

impl<B: CodecBackend> KeyPlanner<'_, B> {
    /// Plan the key-frame superblock `sb`.
    pub fn plan(&mut self, sb: Block) -> Partition {
        self.plan_node(sb).1
    }

    /// λ weighing partition bits against the leaf costs. The transform
    /// cost already uses the backend's own λ; the variance cost keeps the
    /// inter planner's scaled λ it was measured with.
    fn lambda(&self) -> f32 {
        match self.mode {
            KeyPlanMode::Transform => self.backend.lambda(),
            _ => self.backend.lambda() * self.params.lambda_scale,
        }
    }

    fn leaf_cost(&mut self, node: &Block) -> f32 {
        let distortion = match self.mode {
            KeyPlanMode::Transform => self
                .backend
                .intra_leaf_cost(self.source, node)
                .expect("transform planning is resolved only for supporting backends"),
            _ => source_variance::<B>(self.dsp, self.source, node) as f32,
        };
        distortion + self.lambda() * self.backend.partition_bits(node, false)
    }

    /// Speed heuristic (`planner.intra_flat_stop`): a leaf this cheap is
    /// smooth content its quadrants could improve on by little more than
    /// their own partition and mode bits, so they are not planned at all.
    /// Measured in docs/handoff/vvc.md "Key-frame CPU".
    fn is_flat(&self, node: &Block, leaf_cost: f32) -> bool {
        let per_8x8 = (node.area() / FLAT_STOP_UNIT_AREA) as f32;
        self.params.intra_flat_stop > 0.0
            && leaf_cost <= self.params.intra_flat_stop * self.lambda() * per_8x8
    }

    fn plan_node(&mut self, node: Block) -> (f32, Partition) {
        let inside = node.is_inside(self.width, self.height);
        let min_log2 = self.backend.min_block_log2();
        let floor = (self.params.intra_min_log2.round() as u8).max(min_log2);
        let may_be_leaf = inside && node.log2_size() <= self.backend.max_leaf_log2();
        let can_split = node.log2_size() > min_log2 && (!may_be_leaf || node.log2_size() > floor);
        // Uncapped: with the inter cap (λ/8 per sample) every textured
        // superblock looked as cheap as a flat one and stayed 64x64 (AV1
        // low-light key frame: −6.5 dB for −24% bytes).
        let leaf_cost = may_be_leaf.then(|| self.leaf_cost(&node));
        if !can_split || leaf_cost.is_some_and(|cost| self.is_flat(&node, cost)) {
            return (leaf_cost.unwrap_or(0.0), Partition::Leaf(None));
        }
        let mut split_cost = self.lambda()
            * self.params.split_rate_factor
            * self.backend.partition_bits(&node, true);
        let mut children = std::array::from_fn(|_| Partition::Leaf(None));
        for (i, child) in node.quadrants().into_iter().enumerate() {
            // Remaining children have nonnegative costs: once the partial
            // split loses, further planning cannot change the parent choice.
            // Isolated legal-rate P probes: HEVC −1.1%, AV1 −1.2%; parity
            // unchanged, including enabled intra planning (zero BD effect).
            if let Some(cost) = leaf_cost
                && self.params.intra_bias * cost <= 1000.0 * split_cost
            {
                return (cost, Partition::Leaf(None));
            }
            if child.x >= self.width || child.y >= self.height {
                continue;
            }
            let (cost, plan) = self.plan_node(child);
            split_cost += cost;
            children[i] = plan;
        }
        match leaf_cost {
            Some(cost) if self.params.intra_bias * cost <= 1000.0 * split_cost => {
                (cost, Partition::Leaf(None))
            }
            _ => (split_cost, Partition::Split(Box::new(children))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_mode_rounds_the_parameter() {
        assert_eq!(KeyPlanMode::from_param(0.0), KeyPlanMode::Fixed);
        assert_eq!(KeyPlanMode::from_param(0.4), KeyPlanMode::Fixed);
        assert_eq!(KeyPlanMode::from_param(1.0), KeyPlanMode::Variance);
        assert_eq!(KeyPlanMode::from_param(2.0), KeyPlanMode::Transform);
    }
}
