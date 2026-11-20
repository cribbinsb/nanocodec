//! Whole-CU coding trials for the engine's CU pricing (`nc_encoder`
//! `cu_pricing.rs`, `CodecBackend::begin_cu_trial`): a CU is coded by the
//! real coder, priced, and can be undone. HEVC's `backend/cu_trial.rs` does
//! the same over its own coded state.
//!
//! A trial notes everything coding one CU changes: the CU's reconstruction
//! in all planes, its motion-grid cells with the history list and affine
//! models, the recorded items, the quantization group, the per-CU
//! predictors and the P-picture intra count the coarse intra search reads.
//! The ranking's cached luma prediction is dropped, so every coding after
//! an undo predicts afresh. The price is the squared error of the three
//! planes (chroma weighted by `2^((QpY − QpC) / 3)`, the inverse of the
//! chroma λ ratio the residual decisions use) plus λ times the CU's bits,
//! priced by the slice writer's own syntax walk (`syntax::block` over
//! `coeff::BitCost`) at the slice's initial context states, as every rate
//! estimate of the backend's trials.
//!
//! With `vvc.trial_full_intra`, an intra CU coded while a trial is open
//! (the priced alternative, or the ranking's intra choice being priced)
//! takes the key pictures' direction search instead of the P-picture one,
//! as HEVC's trials do (`hevc.trial_full_intra`).
//!
//! Only square CUs are tried (rectangular leaves come only from
//! `vvc.mtt`, off by default).

use nc_common::{Picture, PlaneIndex};
use nc_encoder::Block;

use super::VvcBackend;
use crate::coeff::BitCost;
use crate::inter_intra::InterIntraSearch;
use crate::motion::{CuSnapshot, Predictors};
use crate::quant_group::QuantGroup;
use crate::syntax;

/// The coded state of one CU before a trial (buffers reused trial to
/// trial).
#[derive(Debug, Default)]
pub(super) struct CuTrial {
    block: Option<Block>,
    planes: [Vec<u8>; 3],
    grid: CuSnapshot,
    items: usize,
    group: Option<QuantGroup>,
    current: Option<(Block, Predictors)>,
    inter_intra: Option<InterIntraSearch>,
    /// `vvc.trial_full_intra`.
    full_intra_search: bool,
    /// Whether a trial is open (begun and not yet ended).
    open: bool,
}

impl CuTrial {
    pub(super) fn new(full_intra_search: bool) -> Self {
        Self {
            full_intra_search,
            ..Self::default()
        }
    }

    /// Whether intra CUs take the key pictures' search now.
    pub(super) fn full_intra(&self) -> bool {
        self.open && self.full_intra_search
    }

    /// `CodecBackend::end_cu_trial`: the leaf's pricing is over.
    pub(super) fn end(&mut self) {
        self.open = false;
    }
}

/// `(x, y, side)` of square CU `block` in `plane` (4:2:0).
fn plane_square(plane: PlaneIndex, block: &Block) -> (usize, usize, usize) {
    let shift = usize::from(plane != PlaneIndex::Y);
    (block.x >> shift, block.y >> shift, block.width() >> shift)
}

impl VvcBackend {
    /// `CodecBackend::begin_cu_trial`: note `block`'s coded state.
    pub(super) fn save_cu_trial(&mut self, block: &Block, recon: &Picture) -> bool {
        if block.width() != block.height() {
            return false;
        }
        let mut trial = std::mem::take(&mut self.cu_trial);
        for (plane, saved) in PlaneIndex::ALL.into_iter().zip(&mut trial.planes) {
            let (x, y, n) = plane_square(plane, block);
            let view = recon.plane(plane).view();
            saved.clear();
            for row in 0..n {
                saved.extend_from_slice(&view.row(y + row)[x..x + n]);
            }
        }
        self.motion.save_cu(block, &mut trial.grid);
        trial.block = Some(*block);
        trial.items = self.items.len();
        trial.group = self.quant_group();
        trial.current = self.current;
        trial.inter_intra = Some(self.inter_intra_search);
        trial.open = true;
        // A coding after an undo must predict afresh: the cache matched the
        // ranking's winner, which the first coding may already have taken.
        self.cached_luma = None;
        self.cu_trial = trial;
        true
    }

    /// `CodecBackend::undo_cu_trial`: put the noted state back.
    pub(super) fn restore_cu_trial(&mut self, recon: &mut Picture) {
        let trial = std::mem::take(&mut self.cu_trial);
        let block = trial.block.expect("a trial was begun");
        for (plane, saved) in PlaneIndex::ALL.into_iter().zip(&trial.planes) {
            let (x, y, n) = plane_square(plane, &block);
            let mut view = recon.plane_mut(plane).view_mut();
            for (row, samples) in saved.chunks_exact(n).enumerate() {
                view.row_mut(y + row)[x..x + n].copy_from_slice(samples);
            }
        }
        self.motion.restore_cu(&block, &trial.grid);
        self.items.truncate(trial.items);
        self.restore_quant_group(trial.group);
        self.current = trial.current;
        if let Some(search) = trial.inter_intra {
            self.inter_intra_search = search;
        }
        self.cached_luma = None;
        self.cu_trial = trial;
    }

    /// `CodecBackend::cu_trial_cost`: the price of the CU just coded.
    pub(super) fn coded_cu_cost(&self, block: &Block, source: &Picture, recon: &Picture) -> f32 {
        let Some(syntax::Item::Leaf {
            contexts: neighbours,
            coding,
        }) = self.items.last()
        else {
            unreachable!("encode_block records a leaf");
        };
        let mut contexts = self.contexts.clone();
        let mut bits = BitCost::default();
        syntax::block(
            &mut bits,
            &mut contexts,
            &self.dsp,
            (self.inter, self.references.syntax.active, self.tools),
            neighbours,
            coding,
        );
        let [luma, cb, cr] = PlaneIndex::ALL.map(|plane| {
            nc_encoder::block_sse(
                &self.dsp,
                (source, recon),
                plane,
                plane_square(plane, block),
            ) as f32
        });
        let lambda = self.lambda;
        let chroma_weight = lambda / self.quantizers().chroma_lambda(lambda);
        luma + chroma_weight * (cb + cr) + lambda * bits.0
    }
}
