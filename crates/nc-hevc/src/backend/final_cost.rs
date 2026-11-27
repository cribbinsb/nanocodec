//! HEVC's pricing for the engine's final decision stage
//! (`nc_encoder::CodecBackend::final_intra_screen`, `final_coded_cost` and
//! `skip_vector`; the stage itself is the engine's `final_stage.rs`).
//!
//! Luma only, at the CU's largest transform blocks, without touching the
//! picture or the coded state (H.264's pricing, `nc-h264`
//! `backend/final_cost.rs`, has the same shape):
//! - the skip vector is merge candidate 0 where it predicts from the
//!   reference being decided (`cu_skip_flag` with `merge_idx` 0, §7.3.8.5);
//! - the intra screen is the squared error of the best of planar, DC,
//!   horizontal and vertical (the coarse P-frame intra search's fixed
//!   modes, chosen on the CU's first transform block as that search
//!   chooses), each transform block predicted from the reconstruction
//!   around the CU (later blocks would see their coded neighbours:
//!   pessimistic for 64x64 CUs), plus the intra mode bits the ranking
//!   charges; whole-CU pricing (`cu_pricing.rs`) reads the same screen;
//! - the coded cost is [`HevcBackend::luma_residual_cost`] of each transform
//!   block (plain quantiser, CABAC estimate, distortion measured on the
//!   coefficients) plus λ times the CU's mode bits: a merged vector whose
//!   blocks all quantise to zero is priced as the skip it will be coded as.
//!
//! Chroma, the trellis, the transform tree and the intra mode search are
//! left out (H.264: pricing all three planes cost 2-4 % more instructions
//! for 0.2-0.3 points of J): the coder still runs them on the candidate
//! that wins.

use nc_common::{Picture, PlaneIndex, PlaneView, PlaneViewMut};
use nc_dsp::BlockSize;
use nc_encoder::{Block, CodecBackend, FinalCandidate, Mv, PredictionMode, ReferenceChoice};

use super::{HevcBackend, bi_motion, cost, transform_units};
use crate::inter::{Component, predict_bi_into, predict_luma, predict_samples};
use crate::mvpred::Motion;
use crate::predict::{self, ModeCode};
use crate::transform::MAX_COEFFS;

/// Bits of an inter-slice intra CU's syntax the intra mode code leaves
/// out: `cu_skip_flag`, `pred_mode_flag` and `intra_chroma_pred_mode`
/// (§7.3.8.5), about a bit each.
const INTRA_CU_FLAG_BITS: f32 = 3.0;
/// The modes the intra screen tries: the coarse P-frame intra search's
/// fixed ones.
const SCREENED_MODES: [u8; 4] = [
    predict::PLANAR,
    predict::DC,
    predict::HORIZONTAL,
    predict::VERTICAL,
];

/// The intra mode the last screen chose, with its luma prediction.
#[derive(Debug, Clone, Default)]
pub(super) struct ScreenedIntra {
    block: Option<Block>,
    mode: u8,
    /// The CU's luma prediction, row-major.
    prediction: Vec<u8>,
}

/// Scratch of the stage's pricing (allocated once).
#[derive(Debug, Default)]
pub(super) struct FinalScratch {
    /// One inter candidate's luma prediction, row-major.
    prediction: Vec<u8>,
    screened: ScreenedIntra,
}

impl HevcBackend {
    /// See `nc_encoder::CodecBackend::skip_vector`: merge candidate 0's
    /// vector where it predicts from the reference being decided.
    pub(super) fn merge_skip_vector(&mut self, block: &Block) -> Option<Mv> {
        let first = self.predictors(block).merge.get(0).as_uni()?;
        (first.ref_idx == self.active_ref).then_some(first.mv)
    }

    /// See `nc_encoder::CodecBackend::final_intra_screen`. The prediction is
    /// kept for the pricing.
    pub(super) fn screen_intra_cu(
        &mut self,
        source: &Picture,
        recon: &Picture,
        block: &Block,
    ) -> (f32, f32) {
        let tus = transform_units(block, self.info.max_transform_log2);
        let tus = tus.as_slice();
        let mode = SCREENED_MODES
            .into_iter()
            .min_by_key(|&mode| self.intra_luma_sse((source, recon), block, tus[0], mode, None))
            .expect("four modes");
        let mut screened = std::mem::take(&mut self.final_scratch.screened);
        let side = block.size();
        screened.prediction.resize(side * side, 0);
        let total: u64 = tus
            .iter()
            .map(|&tu| {
                self.intra_luma_sse(
                    (source, recon),
                    block,
                    tu,
                    mode,
                    Some(&mut screened.prediction),
                )
            })
            .sum();
        screened.block = Some(*block);
        screened.mode = mode;
        self.final_scratch.screened = screened;
        (
            total as f32,
            self.mode_bits(block, &PredictionMode::IntraDc),
        )
    }

    /// Squared error of luma transform block `(x, y, log2)` of `block`
    /// predicted with intra `mode` from `recon`'s available neighbours; the
    /// prediction is also written to its place in `keep` (the CU's luma,
    /// row-major).
    fn intra_luma_sse(
        &self,
        (source, recon): (&Picture, &Picture),
        block: &Block,
        (x, y, log2): (usize, usize, u8),
        mode: u8,
        keep: Option<&mut Vec<u8>>,
    ) -> u64 {
        let n = 1usize << log2;
        let refs = self.intra_refs(recon, PlaneIndex::Y, x, y, n);
        let mut pred = [0u8; MAX_COEFFS];
        predict::predict(&self.dsp, &refs, mode, true, &mut pred);
        if let Some(keep) = keep {
            let side = block.size();
            let start = (y - block.y) * side + x - block.x;
            for (row, samples) in pred[..n * n].chunks_exact(n).enumerate() {
                keep[start + row * side..][..n].copy_from_slice(samples);
            }
        }
        let pred = PlaneView::new(&pred, n, n, n).expect("prediction covers the block");
        let src = source.plane(PlaneIndex::Y).view();
        let src = src.sub_view(x, y, n, n).expect("block inside the picture");
        let size = BlockSize::new(n, n).expect("transform sizes 4..32");
        u64::from(self.dsp.variance(size, &src, &pred).sse)
    }

    /// See `nc_encoder::CodecBackend::final_coded_cost`. `None` for intra
    /// without a screen of this CU.
    pub(super) fn price_final(
        &mut self,
        source: &Picture,
        reference: &Picture,
        block: &Block,
        candidate: FinalCandidate,
    ) -> Option<f32> {
        match candidate {
            FinalCandidate::Intra => self.price_intra(source, block),
            FinalCandidate::Inter(mv) => Some(self.price_inter(source, reference, block, mv)),
        }
    }

    /// The coded luma cost of `block` as an intra CU of the screened mode
    /// and prediction.
    fn price_intra(&self, source: &Picture, block: &Block) -> Option<f32> {
        let screened = &self.final_scratch.screened;
        if screened.block != Some(*block) {
            return None;
        }
        let code = ModeCode::of(screened.mode, self.most_probable_modes(block.x, block.y));
        let mut bits = INTRA_CU_FLAG_BITS + self.intra_mode_bits(code);
        let mut distortion = 0u64;
        let side = block.size();
        let src_plane = source.plane(PlaneIndex::Y).view();
        for &(x, y, log2) in transform_units(block, self.info.max_transform_log2).as_slice() {
            let n = 1usize << log2;
            let start = (y - block.y) * side + x - block.x;
            let pred = PlaneView::new(&screened.prediction[start..], side, n, n)
                .expect("the screen predicted the CU");
            let src = src_plane
                .sub_view(x, y, n, n)
                .expect("block inside the picture");
            let cost = self.luma_residual_cost(&src, &pred, Some(screened.mode));
            distortion += cost.distortion;
            bits += cost.bits;
        }
        Some(distortion as f32 + self.rd_lambda() * bits)
    }

    /// The coded luma cost of inter `block` at `mv`.
    fn price_inter(&mut self, source: &Picture, reference: &Picture, block: &Block, mv: Mv) -> f32 {
        let motion = Motion::uni(mv, self.active_ref);
        let side = block.size();
        // The ranking's winner is already predicted (`cache_exact_luma`);
        // any other vector is predicted into the stage's own scratch, so
        // the winner's prediction stays for the coder.
        let cached = self.cached_luma.is_some_and(|c| {
            c.block == *block
                && c.motion == motion
                && c.filter == nc_dsp::Filter::HevcLuma
                && c.frame_index == self.frame_index
        });
        let mut scratch = std::mem::take(&mut self.final_scratch.prediction);
        if !cached {
            scratch.resize(side * side, 0);
            predict_luma(
                &self.dsp,
                reference.plane(PlaneIndex::Y),
                block.x,
                block.y,
                side,
                side,
                mv,
                &mut scratch,
            );
        }
        let prediction: &[u8] = if cached {
            &self.inter_pred[0]
        } else {
            &scratch
        };
        let residual = self.luma_residual(source, block, prediction);
        self.final_scratch.prediction = scratch;
        let merged = self.predictors(block).merge.predicting(motion).is_some();
        let mode_bits = self.mode_bits(block, &PredictionMode::Inter(mv));
        self.coded_inter_cost(residual, merged, mode_bits)
    }

    /// The coded luma cost of `block` bi-predicted by the engine's `pair`
    /// (first half `first_mv` into `references[0]`, the second into
    /// `references[1]`), as [`Self::price_inter`] prices one vector.
    pub(super) fn price_pair(
        &mut self,
        source: &Picture,
        references: [&Picture; 2],
        block: &Block,
        (first_mv, pair): (Mv, ReferenceChoice),
    ) -> Option<f32> {
        let second = match pair {
            ReferenceChoice::PrimaryPair { second_mv } => (second_mv, 0),
            ReferenceChoice::Compound { secondary_mv } => (secondary_mv, 1),
            ReferenceChoice::Primary | ReferenceChoice::Secondary => return None,
        };
        let motion = bi_motion(first_mv, second);
        let side = block.size();
        let mut scratch = std::mem::take(&mut self.final_scratch.prediction);
        scratch.resize(side * side, 0);
        let kernels = self.dsp.at(nc_common::BitDepth::EIGHT);
        predict_samples(
            kernels,
            references[0].plane(PlaneIndex::Y),
            (block.x, block.y),
            (side, side),
            first_mv,
            Component::Luma,
            &mut self.bi_first,
        );
        let mut view = PlaneViewMut::new(&mut scratch, side, side, side).expect("CU prediction");
        predict_bi_into(
            kernels,
            references[1].plane(PlaneIndex::Y),
            (block.x, block.y),
            second.0,
            Component::Luma,
            &self.bi_first[..side * side],
            &mut view,
        );
        let residual = self.luma_residual(source, block, &scratch);
        self.final_scratch.prediction = scratch;
        let merged = self
            .predictors_for(block, 0)
            .merge
            .predicting(motion)
            .is_some();
        let mode_bits = self.bi_mode_bits(block, motion);
        Some(self.coded_inter_cost(residual, merged, mode_bits))
    }

    /// The luma residual of inter `block` against `prediction` (the CU's
    /// samples, row-major): distortion, bits and whether any coefficient is
    /// coded, over its transform units.
    fn luma_residual(&self, source: &Picture, block: &Block, prediction: &[u8]) -> InterResidual {
        let side = block.size();
        let plane = source.plane(PlaneIndex::Y).view();
        let mut residual = InterResidual::default();
        for &(x, y, log2) in transform_units(block, self.info.max_transform_log2).as_slice() {
            let n = 1usize << log2;
            let src = plane
                .sub_view(x, y, n, n)
                .expect("block inside the picture");
            let start = (y - block.y) * side + x - block.x;
            let pred = PlaneView::new(&prediction[start..], side, n, n)
                .expect("prediction covers the block");
            let cost = self.luma_residual_cost(&src, &pred, None);
            residual.distortion += cost.distortion;
            residual.bits += cost.bits;
            residual.coded |= cost.coded;
        }
        residual
    }

    /// Distortion plus λ times the bits of an inter CU with `residual`,
    /// merged or not, whose prediction signals in `mode_bits`.
    fn coded_inter_cost(&self, residual: InterResidual, merged: bool, mode_bits: f32) -> f32 {
        let bits = match (merged, residual.coded) {
            // A skip: no residual syntax at all.
            (true, false) => mode_bits,
            // Merge with residual: the prediction mode, partition and merge
            // flag a skip leaves out.
            (true, true) => mode_bits + cost::MODE_AND_PART + cost::FLAG + residual.bits,
            // `rqt_root_cbf`.
            (false, _) => mode_bits + cost::FLAG + residual.bits,
        };
        residual.distortion as f32 + self.rd_lambda() * bits
    }
}

/// The luma residual of one inter prediction of a CU.
#[derive(Debug, Clone, Copy, Default)]
struct InterResidual {
    distortion: u64,
    bits: f32,
    coded: bool,
}
