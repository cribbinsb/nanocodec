//! The coded state over one rectangle of the picture, noted so that coding
//! it can be undone and priced: the shared core of the whole-CU trials
//! (`cu_trial.rs`, `CodecBackend::begin_cu_trial`) and the partition
//! search's node trials (`node_trial.rs`, `CodecBackend::begin_node_trial`).
//!
//! A snapshot notes everything coding the rectangle's CUs changes: its
//! reconstruction in all planes, its motion-grid cells with the history
//! list and affine models, the recorded items, the quantization group and
//! the block-QP map's cells (deblocking reads them: a kept coding put back
//! over a lost one's QPs filtered with the wrong QP, 2026-11-21), the
//! per-CU predictors and the P-picture intra count the coarse intra search
//! reads. The ranking's cached luma prediction is dropped, so every coding
//! after an undo predicts afresh.
//!
//! The price of what was coded since a snapshot is the squared error of the
//! three planes over its rectangle (chroma weighted by `2^((QpY − QpC) / 3)`,
//! the inverse of the chroma λ ratio the residual decisions use) plus λ
//! times the bits of the items recorded since, priced by the slice writer's
//! own syntax walk (`coding_tree::code` and `syntax::block` over
//! `coeff::BitCost`) at the slice's initial context states, as every rate
//! estimate of the backend's trials.

use nc_common::{Picture, PlaneIndex};
use nc_encoder::Block;
use nc_encoder::trial_pixels::TrialPixels;

use super::VvcBackend;
use crate::coeff::BitCost;
use crate::contexts::Contexts;
use crate::inter_intra::InterIntraSearch;
use crate::motion::{CuSnapshot, Predictors};
use crate::quant_group::QuantGroup;
use crate::syntax;

/// The coded state of one rectangle (buffers reused snapshot to snapshot).
#[derive(Debug, Default)]
pub(super) struct CodedSnapshot {
    block: Option<Block>,
    pixels: TrialPixels,
    grid: CuSnapshot,
    items: usize,
    /// The items recorded since the snapshot this one follows (a snapshot
    /// taken after a trial, moved out when the kept coding is undone).
    added_items: Vec<syntax::Item>,
    group: Option<QuantGroup>,
    /// The block-QP map's cells over the rectangle (with block QP).
    qp_cells: Vec<u8>,
    current: Option<(Block, Predictors)>,
    inter_intra: Option<InterIntraSearch>,
}

impl CodedSnapshot {
    /// The rectangle noted.
    pub(super) fn block(&self) -> Block {
        self.block.expect("a snapshot was taken")
    }

    /// The number of items recorded when the snapshot was taken.
    pub(super) fn items(&self) -> usize {
        self.items
    }

    /// Move the current winner's syntax aside before undoing it.
    pub(super) fn keep_items(&mut self, items: &mut Vec<syntax::Item>) {
        debug_assert!(self.added_items.is_empty());
        self.added_items.extend(items.drain(self.items..));
    }

    /// Note only `block` and the number of `items` recorded: what a
    /// priced-only trial needs (`CodecBackend::begin_node_pricing`).
    pub(super) fn mark(&mut self, block: &Block, items: usize) {
        self.block = Some(*block);
        self.items = items;
    }
}

/// `(x, y, width, height)` of `block` in `plane` (4:2:0).
pub(super) fn plane_rect(plane: PlaneIndex, block: &Block) -> (usize, usize, usize, usize) {
    let shift = usize::from(plane != PlaneIndex::Y);
    (
        block.x >> shift,
        block.y >> shift,
        block.width() >> shift,
        block.height() >> shift,
    )
}

impl VvcBackend {
    /// Note the block-QP map's cells over `block` (none without block QP).
    fn save_qp_cells(&self, block: &Block, saved: &mut Vec<u8>) {
        match self.block_qp.as_ref() {
            Some(block_qp) => block_qp.map.save(block, saved),
            None => saved.clear(),
        }
    }

    /// Note the coded state of `block` (inside the picture) in `snapshot`.
    pub(super) fn save_coded(
        &mut self,
        block: &Block,
        recon: &Picture,
        snapshot: &mut CodedSnapshot,
    ) {
        snapshot.pixels.save(
            &self.dsp,
            recon,
            (block.x, block.y, block.width(), block.height()),
        );
        self.motion.save_cu(block, &mut snapshot.grid);
        snapshot.block = Some(*block);
        snapshot.items = self.items.len();
        snapshot.added_items.clear();
        snapshot.group = self.quant_group();
        self.save_qp_cells(block, &mut snapshot.qp_cells);
        snapshot.current = self.current;
        snapshot.inter_intra = Some(self.inter_intra_search);
        // A coding after an undo must predict afresh: the cache matched the
        // ranking's winner, which the first coding may already have taken.
        self.cached_luma = None;
    }

    /// Note the coded state of `before`'s rectangle as coded since
    /// `before` was taken in `snapshot`, so that [`Self::restore_coded`]
    /// can put this coding back after others were tried.
    pub(super) fn save_coded_after(
        &self,
        before: &CodedSnapshot,
        recon: &Picture,
        snapshot: &mut CodedSnapshot,
    ) {
        let block = before.block();
        snapshot.pixels.save(
            &self.dsp,
            recon,
            (block.x, block.y, block.width(), block.height()),
        );
        self.motion
            .save_cu_after(&block, &before.grid, &mut snapshot.grid);
        snapshot.block = Some(block);
        snapshot.items = before.items;
        snapshot.added_items.clear();
        snapshot.group = self.quant_group();
        self.save_qp_cells(&block, &mut snapshot.qp_cells);
        snapshot.current = self.current;
        snapshot.inter_intra = Some(self.inter_intra_search);
    }

    /// Put back the coded state `snapshot` noted.
    pub(super) fn restore_coded(&mut self, recon: &mut Picture, snapshot: &mut CodedSnapshot) {
        let block = snapshot.block();
        snapshot.pixels.restore(
            &self.dsp,
            recon,
            (block.x, block.y, block.width(), block.height()),
        );
        self.motion.restore_cu(&block, &snapshot.grid);
        self.items.truncate(snapshot.items);
        self.items.append(&mut snapshot.added_items);
        self.restore_quant_group(snapshot.group);
        if let Some(block_qp) = self.block_qp.as_mut() {
            block_qp.map.restore(&block, &snapshot.qp_cells);
        }
        self.current = snapshot.current;
        if let Some(search) = snapshot.inter_intra {
            self.inter_intra_search = search;
        }
        self.cached_luma = None;
    }

    /// The price of `block` as coded by the items from index `first` on.
    pub(super) fn coded_cost(
        &self,
        block: &Block,
        first: usize,
        source: &Picture,
        recon: &Picture,
    ) -> f32 {
        self.coded_parts(block, first, source, recon).cost()
    }

    /// What the CUs recorded from item `first` on code
    /// (`CodecBackend::node_trial_summary`): read from the recorded levels,
    /// which an undo truncates with the items, so it holds only the coding in
    /// place.
    pub(super) fn coded_summary(&self, first: usize) -> nc_encoder::NodeTrialSummary {
        let mut summary = nc_encoder::NodeTrialSummary::default();
        let max_tb_log2 = self.tools.max_tb_log2();
        let codings = self.items[first..].iter().filter_map(|item| match item {
            syntax::Item::Leaf { coding, .. } => Some(coding),
            syntax::Item::Ctu { .. } | syntax::Item::Split { .. } => None,
        });
        for coding in codings {
            let units = crate::transform_tree::coded_units(
                &coding.cu,
                coding.isp,
                &coding.levels,
                max_tb_log2,
            );
            let luma_samples = |tu: &Block| match coding.isp {
                Some(_) => tu.area(),
                None => crate::transform_tree::plane_samples(tu, 0),
            };
            for (index, (tu, coded)) in units.as_slice().iter().enumerate() {
                for (plane, &flag) in coded.iter().enumerate() {
                    if !flag {
                        continue;
                    }
                    let levels = match (plane, coding.isp) {
                        (0, _) => crate::transform_tree::unit_levels(
                            &coding.levels[0],
                            index,
                            luma_samples(tu),
                        ),
                        // ISP's chroma is one block, coded with the last part.
                        (_, Some(_)) => &coding.levels[plane][..],
                        (_, None) => crate::transform_tree::unit_levels(
                            &coding.levels[plane],
                            index,
                            crate::transform_tree::plane_samples(tu, plane),
                        ),
                    };
                    let nonzero = levels.iter().filter(|&&level| level != 0).count();
                    let luma_log2 = tu.log2_width.max(tu.log2_height);
                    summary.add_block((plane == 0).then_some(luma_log2), nonzero);
                }
            }
        }
        summary
    }

    /// The estimated bits of `items` (CTU markers cost nothing here),
    /// priced with `contexts`, which they adapt.
    pub(super) fn estimated_item_bits(
        &self,
        items: &[syntax::Item],
        contexts: &mut Contexts,
    ) -> f32 {
        let mut bits = BitCost::default();
        for item in items {
            match item {
                &syntax::Item::Split {
                    allowed,
                    contexts: split_contexts,
                    split,
                } => {
                    crate::coding_tree::code(
                        &mut bits,
                        &mut contexts.tree,
                        allowed,
                        split_contexts,
                        split,
                    )
                    .expect("an allowed split");
                }
                syntax::Item::Leaf {
                    contexts: neighbours,
                    coding,
                } => syntax::block(
                    &mut bits,
                    contexts,
                    &self.dsp,
                    (self.inter, self.references.syntax.active, self.tools),
                    neighbours,
                    coding,
                ),
                syntax::Item::Ctu { .. } => {}
            }
        }
        bits.0
    }

    /// [`Self::coded_cost`] in its distortion and bits.
    pub(super) fn coded_parts(
        &self,
        block: &Block,
        first: usize,
        source: &Picture,
        recon: &Picture,
    ) -> nc_encoder::NodeTrialParts {
        let mut contexts = self.contexts.clone();
        let bits = self.estimated_item_bits(&self.items[first..], &mut contexts);
        let [luma, cb, cr] = PlaneIndex::ALL.map(|plane| {
            let (x, y, w, h) = plane_rect(plane, block);
            nc_encoder::block_sse_rect(&self.dsp, (source, recon), plane, (x, y, w, h)) as f32
        });
        let lambda = self.lambda;
        let chroma_weight = lambda / self.quantizers().chroma_lambda(lambda);
        nc_encoder::NodeTrialParts {
            distortion: luma + chroma_weight * (cb + cr),
            bits,
            lambda,
        }
    }
}
