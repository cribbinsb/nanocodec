//! Whole-CU coding trials for the engine's CU pricing
//! (`nc_encoder` `cu_pricing.rs`, `CodecBackend::begin_cu_trial`): a CU is
//! coded by the real coder, priced, and can be undone.
//!
//! A trial notes everything coding one CU changes: the CU's reconstruction
//! in all planes, its grid cells, the recorded items, transform units and
//! levels, the CU and TU counters, the quantization group and the
//! per-CU hints `encode_block` consumes; the ranking's cached luma prediction
//! is dropped, as trials overwrite the buffer it lives in. The price is the squared error of
//! the three planes (chroma weighted by HM's `2^((QpY − QpC) / 3)`, the
//! inverse of the chroma λ ratio SAO uses) plus λ times the CU's bits
//! estimated by the CU writer at the slice's initial context states (as
//! every rate estimate of the backend's own trials).
//!
//! An intra CU coded while a trial is open (the priced alternative, or the
//! ranking's intra choice being priced) takes the full (key-frame) luma
//! mode search instead of the coarse P-frame one: in the shadow log the
//! full search's intra was worth 0.86 points of the worst six's coded cost
//! on the gated leaves against 0.59 for the coarse search's, and the
//! pricing pays for one intra CU per priced leaf at most.

use nc_common::{Picture, PlaneIndex, PlaneView};
use nc_dsp::BlockSize;
use nc_encoder::{Block, CodecBackend, Mv, PredictionMode};

use super::{
    EarlyMergeHint, HevcBackend, InterIntraSearch, Predictors, plane_rect, transform_units,
};
use crate::grid::Cell;
use crate::headers::LOG2_MIN_CB;
use crate::predict;
use crate::quant_group::QuantGroup;
use crate::syntax::{Item, TransformSyntax, estimate_cu_bits};
use crate::transform::MAX_COEFFS;

/// The coded state of one CU before a trial (buffers reused trial to
/// trial).
#[derive(Debug, Default)]
pub(super) struct CuTrial {
    block: Option<Block>,
    planes: [Vec<u8>; 3],
    cells: Vec<Cell>,
    items: usize,
    tus: usize,
    levels: (usize, usize),
    next_cu: u32,
    next_tu: u32,
    group: Option<QuantGroup>,
    current: Option<Predictors>,
    early_merge: Option<EarlyMergeHint>,
    /// The P-frame intra search the trial replaced, while one is open.
    search: Option<InterIntraSearch>,
}

impl HevcBackend {
    /// `CodecBackend::begin_cu_trial`: note `block`'s coded state.
    pub(super) fn save_cu_trial(&mut self, block: &Block, recon: &Picture) -> bool {
        if self.random.is_some() || self.still_skipped.is_some() {
            return false;
        }
        let mut trial = std::mem::take(&mut self.cu_trial);
        for (plane, saved) in PlaneIndex::ALL.into_iter().zip(&mut trial.planes) {
            let (x, y, n) = plane_rect(plane, block.x, block.y, block.log2_size());
            let view = recon.plane(plane).view();
            saved.clear();
            for row in 0..n {
                saved.extend_from_slice(&view.row(y + row)[x..x + n]);
            }
        }
        trial.cells.clear();
        let side = 1usize << LOG2_MIN_CB;
        for y in (block.y..block.y + block.size()).step_by(side) {
            for x in (block.x..block.x + block.size()).step_by(side) {
                trial.cells.push(*self.grid.at(x, y));
            }
        }
        trial.block = Some(*block);
        trial.items = self.items.len();
        trial.tus = self.tu_store.len();
        trial.levels = self.level_store.borrow().mark();
        trial.next_cu = self.next_cu;
        trial.next_tu = self.next_tu;
        trial.group = self.block_qp.as_ref().map(|b| b.group);
        trial.current = self.current;
        // The ranking's cached luma prediction lives in `inter_pred`, which
        // every trial overwrites (the skip vector's prediction, the inter
        // fallback's): a coding after an undo must predict afresh, or it
        // reconstructs from another vector's samples than the decoder.
        self.cached_luma = None;
        trial.early_merge = self.early_merge;
        if let Some(search) = self.trial_intra_search
            && trial.search.is_none()
        {
            trial.search = Some(std::mem::replace(&mut self.inter_intra, search));
        }
        self.cu_trial = trial;
        true
    }

    /// `CodecBackend::end_cu_trial`: the P-frame intra search is back.
    pub(super) fn finish_cu_trial(&mut self) {
        if let Some(search) = self.cu_trial.search.take() {
            self.inter_intra = search;
        }
    }

    /// `CodecBackend::undo_cu_trial`: put the noted state back.
    pub(super) fn restore_cu_trial(&mut self, recon: &mut Picture) {
        let trial = std::mem::take(&mut self.cu_trial);
        let block = trial.block.expect("a trial was begun");
        for (plane, saved) in PlaneIndex::ALL.into_iter().zip(&trial.planes) {
            let (x, y, n) = plane_rect(plane, block.x, block.y, block.log2_size());
            let mut view = recon.plane_mut(plane).view_mut();
            for (row, samples) in saved.chunks_exact(n).enumerate() {
                view.row_mut(y + row)[x..x + n].copy_from_slice(samples);
            }
        }
        let side = 1usize << LOG2_MIN_CB;
        let mut cells = trial.cells.iter();
        for y in (block.y..block.y + block.size()).step_by(side) {
            for x in (block.x..block.x + block.size()).step_by(side) {
                self.grid.fill(x, y, side, *cells.next().expect("noted"));
            }
        }
        self.items.truncate(trial.items);
        self.tu_store.truncate(trial.tus);
        self.level_store.borrow_mut().truncate(trial.levels);
        self.next_cu = trial.next_cu;
        self.next_tu = trial.next_tu;
        if let (Some(block_qp), Some(group)) = (self.block_qp.as_mut(), trial.group) {
            block_qp.group = group;
        }
        self.current = trial.current;
        self.early_merge = trial.early_merge;
        self.cu_trial = trial;
    }

    /// `CodecBackend::cu_trial_cost`: the price of the CU just coded.
    pub(super) fn coded_cu_cost(&self, block: &Block, source: &Picture, recon: &Picture) -> f32 {
        let Some(Item::Cu(cu)) = self.items.last() else {
            unreachable!("encode_block records a CU");
        };
        let mut contexts = self.contexts.clone();
        let bits = estimate_cu_bits(
            &self.bin_costs,
            &mut contexts,
            &self.grid,
            (&self.level_store.borrow(), &self.tu_store),
            TransformSyntax {
                scans: &self.scans,
                max_log2: self.info.max_transform_log2,
                inter_depth: self.info.inter_transform_depth,
                intra_depth: self.info.intra_transform_depth,
            },
            self.info.merge_candidates,
            cu,
        );
        let [luma, cb, cr] = PlaneIndex::ALL.map(|plane| {
            let rect = plane_rect(plane, block.x, block.y, block.log2_size());
            self.reconstruction_sse(source, recon, plane, rect) as f32
        });
        let chroma_weight = (1.0 / self.chroma_lambda_ratio()) as f32;
        luma + chroma_weight * (cb + cr) + self.rd_lambda() * bits
    }

    /// `CodecBackend::cu_trial_skip_vector`: merge candidate 0 where it
    /// predicts from the primary reference (`cu_skip_flag` with
    /// `merge_idx` 0, HEVC §7.3.8.5).
    pub(super) fn first_merge_vector(&mut self, block: &Block) -> Option<Mv> {
        let first = self.predictors(block).merge.get(0);
        (first.ref_idx == 0).then_some(first.mv)
    }

    /// `CodecBackend::final_intra_screen`: the luma squared error of the
    /// best of planar, DC, horizontal and vertical (chosen on the CU's
    /// first transform block, as the coarse P-frame search would), each
    /// transform block predicted from the reconstruction around the CU
    /// (later blocks would see their coded neighbours: pessimistic for
    /// 64x64 CUs), and the intra mode bits the ranking charges.
    pub(super) fn screen_intra_cu(
        &self,
        source: &Picture,
        recon: &Picture,
        block: &Block,
    ) -> (f32, f32) {
        let tus = transform_units(block, self.info.max_transform_log2);
        let tus = tus.as_slice();
        let sse = |mode: u8, &(x, y, log2): &(usize, usize, u8)| -> u64 {
            let n = 1usize << log2;
            let refs = self.intra_refs(recon, PlaneIndex::Y, x, y, n);
            let mut pred = [0u8; MAX_COEFFS];
            predict::predict(&self.dsp, &refs, mode, true, &mut pred);
            let pred = PlaneView::new(&pred, n, n, n).expect("prediction covers the block");
            let src = source.plane(PlaneIndex::Y).view();
            let src = src.sub_view(x, y, n, n).expect("block inside the picture");
            let size = BlockSize::new(n, n).expect("transform sizes 8..32");
            u64::from(self.dsp.variance(size, &src, &pred).sse)
        };
        let mode = [
            predict::PLANAR,
            predict::DC,
            predict::HORIZONTAL,
            predict::VERTICAL,
        ]
        .into_iter()
        .min_by_key(|&mode| sse(mode, &tus[0]))
        .expect("four modes");
        let total: u64 = tus.iter().map(|tu| sse(mode, tu)).sum();
        (
            total as f32,
            self.mode_bits(block, &PredictionMode::IntraDc),
        )
    }
}
