//! Rate-distortion inter transform trees (`hevc.inter_tu_split`): every
//! inter CU's residual quadtree is decided node by node (HEVC §7.3.8.8,
//! `max_transform_hierarchy_depth_inter`).
//!
//! The tree is decided on luma alone, and chroma is then coded once along
//! it. A node's luma is coded whole and, where its residual is unevenly
//! spread (`hevc.inter_tu_split_spread`), also split into four; the cheaper
//! by luma `SSE + λ · bits` (CABAC estimates) is kept. The CU's prediction
//! is fixed, so the alternatives differ only in their residual. Chroma
//! follows (in 4:2:0 its transforms sit at the luma leaves): pricing every
//! trial in three planes cost +10% P-frame instructions for about the same
//! trees. The diffuse-chroma TU16 heuristic
//! (`hevc.tu16_chroma_concentration`) still forces its 32x32 splits without
//! a trial.
//!
//! Inter transforms stay at 8x8 and above: 4x4 inter blocks would need
//! deblocking strengths per 4-sample segment inside a cell, which the
//! 8x8 grid does not keep.

use nc_common::{Picture, PlaneIndex, PlaneView};
use nc_dsp::BlockSize;
use nc_encoder::Block;

use super::rd::SavedSquare;
use super::{HevcBackend, Predicted, Quantisation, plane_rect};
use crate::syntax::{
    TransformUnit, TransformUnits, min_transform_log2, split_transform_flag_present,
};

/// Smallest inter transform the encoder codes (8x8; see the module doc).
const MIN_INTER_TU_LOG2: u8 = 3;
/// The TU16 heuristic's node size.
const TU16_PARENT_LOG2: u8 = 5;

/// What an inter CU's transform tree shares while it is coded.
#[derive(Debug, Clone, Copy)]
struct InterTree {
    cu: Block,
    /// Smallest transform the syntax allows in this CU.
    min_log2: u8,
}

impl HevcBackend {
    /// Code the transform tree of inter `block` from its prediction: the
    /// luma tree first, then chroma along its leaves.
    pub(super) fn code_inter_tree(
        &self,
        block: &Block,
        source: &Picture,
        recon: &mut Picture,
    ) -> TransformUnits {
        let min_log2 = min_transform_log2(
            block.log2_size(),
            self.info.inter_transform_depth,
            self.info.max_transform_log2,
        );
        let mut tus = TransformUnits::default();
        let tree = InterTree {
            cu: *block,
            min_log2,
        };
        self.code_luma_node(&tree, *block, false, source, recon, &mut tus);
        for tu in tus.as_mut_slice() {
            let area = (tu.x, tu.y, tu.log2);
            let mut final_plane = |plane| {
                self.code_inter_plane(
                    block,
                    plane,
                    area,
                    (source, recon),
                    Predicted::Inter,
                    Quantisation::Final,
                )
            };
            tu.cb = final_plane(PlaneIndex::U);
            tu.cr = final_plane(PlaneIndex::V);
        }
        tus
    }

    /// Code the luma of `node` of the tree into `tus` (chroma left
    /// uncoded). Returns its luma cost when `costed` (inside a parent's
    /// trial), else 0.
    fn code_luma_node(
        &self,
        tree: &InterTree,
        node: Block,
        costed: bool,
        source: &Picture,
        recon: &mut Picture,
        tus: &mut TransformUnits,
    ) -> f32 {
        let cu = &tree.cu;
        let max_log2 = self.info.max_transform_log2;
        let forced = node.log2_size() > max_log2
            || (node.log2_size() == TU16_PARENT_LOG2
                && self.tu16_chroma_concentration_q16 > 0
                && self.prefer_tu16(cu, source, node.x, node.y));
        if forced {
            return node
                .quadrants()
                .into_iter()
                .map(|child| self.code_luma_node(tree, child, costed, source, recon, tus))
                .sum();
        }
        let depth = usize::from(cu.log2_size() - node.log2_size());
        let flag_present =
            split_transform_flag_present(node.log2_size(), depth, tree.min_log2, max_log2, false);
        let area = (node.x, node.y, node.log2_size());
        let tu = TransformUnit {
            x: node.x,
            y: node.y,
            log2: node.log2_size(),
            // A node outside any parent's trial is final unless its own
            // split wins (then its leaves are re-coded below).
            luma: self.code_inter_plane(
                cu,
                PlaneIndex::Y,
                area,
                (source, recon),
                Predicted::Inter,
                if costed {
                    Quantisation::Plain
                } else {
                    Quantisation::Final
                },
            ),
            ..TransformUnit::default()
        };
        // A node the unsplit path leaves without luma residual (the zero
        // bounds and quantiser dropped it) is never split: four smaller
        // transforms would code what the whole one judged noise. Split
        // gate: an evenly spread residual gains little from four smaller
        // transforms. The gates are evaluated cheapest first, and the
        // spread only for coded nodes (most are not): P instructions of
        // the split fell from +10.2% / +7.5% to +3.9% / +2.1% (crowd09 /
        // bodycam_bc1, QP 42, spread 2).
        let try_split = flag_present
            && node.log2_size() > MIN_INTER_TU_LOG2
            && tu.luma.coded
            && self.luma_residual_spread(cu, &node, source) >= self.tools.inter_tu_split_spread;
        let flag_bits = |split| self.split_transform_bits(node.log2_size(), flag_present, split);
        let leaf_cost = if costed || try_split {
            self.inter_luma_cost(&tu, depth, source, recon) + self.rd_lambda() * flag_bits(false)
        } else {
            0.0
        };
        // Detail gate: a leaf already cheap per sample (below
        // `inter_tu_split_min_cost` λ) gains little from a split.
        let samples = (node.size() * node.size()) as f32;
        let detailed = leaf_cost >= self.tools.inter_tu_split_min_cost * self.rd_lambda() * samples;
        if !(try_split && detailed) {
            tus.push(tu);
            return leaf_cost;
        }
        // The split must also beat leaving the node's luma uncoded
        // (prediction only), so it never pays to code noise the whole
        // node's cost would rather drop.
        let uncoded_cost = self.inter_luma_uncoded_cost(cu, &node, depth, source)
            + self.rd_lambda() * flag_bits(false);
        let saved = SavedSquare::save(
            recon,
            PlaneIndex::Y,
            plane_rect(PlaneIndex::Y, node.x, node.y, node.log2_size()),
        );
        let start = tus.as_slice().len();
        let split_cost = self.rd_lambda() * flag_bits(true)
            + node
                .quadrants()
                .into_iter()
                .map(|child| self.code_luma_node(tree, child, true, source, recon, tus))
                .sum::<f32>();
        if split_cost < leaf_cost.min(uncoded_cost) {
            if !costed {
                self.finalise_luma(cu, &mut tus.as_mut_slice()[start..], source, recon);
            }
            return split_cost;
        }
        saved.restore(recon);
        tus.truncate(start);
        tus.push(tu);
        leaf_cost
    }

    /// Re-code the luma of `tus`, decided inside a split trial with the
    /// plain quantiser, as final blocks (the trellis, where it is on). Inter
    /// luma transform blocks predict from the CU's motion compensation, not
    /// from each other's reconstruction, so each re-codes on its own.
    fn finalise_luma(
        &self,
        cu: &Block,
        tus: &mut [TransformUnit],
        source: &Picture,
        recon: &mut Picture,
    ) {
        if self.trellis_for(PlaneIndex::Y).is_none() {
            return;
        }
        for tu in tus {
            let area = (tu.x, tu.y, tu.log2);
            tu.luma = self.code_inter_plane(
                cu,
                PlaneIndex::Y,
                area,
                (source, recon),
                Predicted::Inter,
                Quantisation::Final,
            );
        }
    }

    /// Luma `SSE + λ · bits` of `node` coded without residual: the
    /// prediction's error and a zero luma coded-block flag.
    fn inter_luma_uncoded_cost(
        &self,
        cu: &Block,
        node: &Block,
        depth: usize,
        source: &Picture,
    ) -> f32 {
        let n = node.size();
        let start = (node.y - cu.y) * cu.size() + node.x - cu.x;
        let pred = PlaneView::new(&self.inter_pred[0][start..], cu.size(), n, n)
            .expect("prediction covers the node");
        let src = source.plane(PlaneIndex::Y).view();
        let src = src
            .sub_view(node.x, node.y, n, n)
            .expect("node inside the picture");
        let size = BlockSize::new(n, n).expect("transform sizes 8..32");
        let distortion = self.dsp.variance(size, &src, &pred).sse;
        let bits = self.bin_bits(self.contexts.cbf_luma[usize::from(depth == 0)], false);
        distortion as f32 + self.rd_lambda() * bits
    }

    /// Luma `SSE + λ · bits` of a coded inter transform unit at `depth`.
    fn inter_luma_cost(
        &self,
        tu: &TransformUnit,
        depth: usize,
        source: &Picture,
        recon: &Picture,
    ) -> f32 {
        let distortion = self.reconstruction_sse(
            source,
            recon,
            PlaneIndex::Y,
            plane_rect(PlaneIndex::Y, tu.x, tu.y, tu.log2),
        );
        let bits = self.bin_bits(
            self.contexts.cbf_luma[usize::from(depth == 0)],
            tu.luma.coded,
        ) + self.residual_bits(tu.luma, true);
        distortion as f32 + self.rd_lambda() * bits
    }

    /// [`HevcBackend::residual_spread`] of `node`'s luma inter residual.
    fn luma_residual_spread(&self, cu: &Block, node: &Block, source: &Picture) -> f32 {
        let n = node.size();
        let start = (node.y - cu.y) * cu.size() + node.x - cu.x;
        let pred = PlaneView::new(&self.inter_pred[0][start..], cu.size(), n, n)
            .expect("prediction covers the node");
        let src = source.plane(PlaneIndex::Y).view();
        let src = src
            .sub_view(node.x, node.y, n, n)
            .expect("node inside the picture");
        self.residual_spread(&src, &pred, n)
    }
}
