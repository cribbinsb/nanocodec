//! VP9's pricing for the engine's final decision stage
//! (`nc_encoder::CodecBackend::final_intra_screen`, `final_coded_cost` and
//! `skip_vector`; the stage itself is the engine's `final_stage.rs`).
//!
//! Luma only, at the block's largest transform, without touching the
//! picture or the coded state (H.264's pricing, `nc-h264`
//! `backend/final_cost.rs`, has the same shape):
//! - the skip vector is `NEARESTMV`, the cheapest vector to signal, which
//!   the winner walk keeps where it wins (`decision.rs`: walking H.264's
//!   `P_Skip` vector cost J +7.2 % on a still camera);
//! - the intra screen is the squared error of the intra modes the coder
//!   would choose ([`Vp9Backend::choose_intra_modes`]), each transform
//!   block predicted from the reconstruction as it stands (blocks after the
//!   first would see their coded neighbours: pessimistic for blocks of
//!   several transforms), plus their mode bits;
//! - the coded cost is the luma residual transformed (DCT for inter, the
//!   mode's type for intra), quantised without the trellis, its squared
//!   error measured on the coefficients and its tokens priced at the
//!   frame's probabilities and the contexts as they stand, plus λ times the
//!   mode and skip-flag bits: a candidate whose levels all quantise to zero
//!   is priced as the skipped block it will be coded as.
//!
//! Chroma, the trellis and the transform-size search are left out (they
//! made H.264's full trials cost +33 to +53 % instructions): the coder
//! still runs them on the candidate that wins.

use nc_common::{Picture, PlaneIndex, PlaneView};
use nc_encoder::{Block, CodecBackend, FinalCandidate, Mv as EngineMv, PredictionMode};

use super::{MI_LOG2, Vp9Backend};
use crate::coeff::{self, Model};
use crate::contexts;
use crate::inter_pred;
use crate::intra_pred;
use crate::recon;
use crate::transform;
use crate::tree;
use crate::types::{IntraMode, max_tx_size};

/// The intra modes the last screen chose, with their luma prediction.
#[derive(Debug, Clone)]
pub(super) struct ScreenedIntra {
    block: Block,
    modes: (IntraMode, IntraMode),
    /// Luma prediction of the block, transform block after transform
    /// block (each row-major at the transform's side).
    prediction: Vec<u8>,
}

/// Scratch of the stage's pricing (allocated once).
#[derive(Debug, Default)]
pub(super) struct FinalScratch {
    /// One candidate's luma prediction, transform block after transform
    /// block.
    prediction: Vec<u8>,
    /// The same, row-major over the block.
    block_prediction: Vec<u8>,
    screened: Option<ScreenedIntra>,
}

impl Vp9Backend {
    /// See `nc_encoder::CodecBackend::skip_vector`: `NEARESTMV`.
    pub(super) fn nearest_vector(&mut self, block: &Block) -> EngineMv {
        let refs = self.refs_for(block);
        self.engine_candidate(refs.nearest)
    }

    /// See `nc_encoder::CodecBackend::final_intra_screen`.
    pub(super) fn screen_intra(
        &mut self,
        source: &Picture,
        recon: &Picture,
        block: &Block,
    ) -> (f32, f32) {
        let tx = max_tx_size(block.log2_size());
        let modes = self.choose_intra_modes(block, tx, source, recon);
        let n = block.size();
        let mut screened = self.final_scratch.screened.take().unwrap_or(ScreenedIntra {
            block: *block,
            modes,
            prediction: Vec::new(),
        });
        screened.block = *block;
        screened.modes = modes;
        screened.prediction.resize(n * n, 0);
        let (row, col) = (block.y >> MI_LOG2, block.x >> MI_LOG2);
        let (top, left) = (row > 0, col > 0);
        let view = recon.plane(PlaneIndex::Y).view();
        let area = tx.area();
        for (tx_block, out) in recon::tx_blocks(PlaneIndex::Y, block.x, block.y, n, tx)
            .zip(screened.prediction.chunks_exact_mut(area))
        {
            intra_pred::predict(
                &self.dsp,
                &view,
                (tx_block.x, tx_block.y),
                tx.side(),
                modes.0,
                tx_block.edges(top, left),
                out,
            );
        }
        let sse = self.tiled_sse(source, block, tx, &screened.prediction);
        let bits = self.inter_frame_intra_bits(block, modes);
        self.final_scratch.screened = Some(screened);
        (sse as f32, bits)
    }

    /// See `nc_encoder::CodecBackend::final_coded_cost`. `None` for intra
    /// without a screen of this block.
    pub(super) fn price_final(
        &mut self,
        source: &Picture,
        reference: &Picture,
        block: &Block,
        candidate: FinalCandidate,
    ) -> Option<f32> {
        let tx = max_tx_size(block.log2_size());
        match candidate {
            FinalCandidate::Inter(mv) => {
                let mut prediction = std::mem::take(&mut self.final_scratch.prediction);
                self.predict_inter_luma(reference, block, mv, tx, &mut prediction);
                let mode_bits = self.mode_bits(block, &PredictionMode::Inter(mv));
                let cost = self.price_luma(source, block, (tx, None), &prediction, mode_bits);
                self.final_scratch.prediction = prediction;
                Some(cost)
            }
            FinalCandidate::Intra => {
                let screened = self
                    .final_scratch
                    .screened
                    .take()
                    .filter(|s| s.block == *block)?;
                let bits = self.inter_frame_intra_bits(block, screened.modes);
                let mode = Some(screened.modes.0);
                let cost = self.price_luma(source, block, (tx, mode), &screened.prediction, bits);
                self.final_scratch.screened = Some(screened);
                Some(cost)
            }
        }
    }

    /// Bits of an intra block in an inter frame: the intra/inter flag and
    /// the luma and chroma modes.
    fn inter_frame_intra_bits(&self, block: &Block, modes: (IntraMode, IntraMode)) -> f32 {
        let (row, col) = (block.y >> MI_LOG2, block.x >> MI_LOG2);
        let intra_inter =
            self.context.intra_inter[contexts::intra_inter(&self.grids.grid, row, col)];
        tree::bit_cost(false, intra_inter) + self.intra_mode_bits(block, modes)
    }

    /// Coded luma cost of `block` predicted by `prediction` (transform block
    /// after transform block at `tx`), inter or (`intra`) intra with that
    /// luma mode: squared error after quantisation plus λ times the token,
    /// `mode_bits` and skip-flag bits (skip set where no level survives).
    fn price_luma(
        &mut self,
        source: &Picture,
        block: &Block,
        (tx, intra): (transform::TxSize, Option<IntraMode>),
        prediction: &[u8],
        mode_bits: f32,
    ) -> f32 {
        let (side, area) = (tx.side(), tx.area());
        let steps = self.block_steps(PlaneIndex::Y);
        // The coder's rounding and dead zone (`quantization_controls`).
        let (plane_quantization, _) = self.quantization_controls(intra.is_some());
        let quantization = plane_quantization[0];
        let ty = recon::tx_type(PlaneIndex::Y, intra, tx, self.lossless);
        let model = Model::of(PlaneIndex::Y, tx, ty, intra.is_none());
        let probs =
            &self.context.coef[tx as usize][model.plane as usize][usize::from(intra.is_none())];
        let src = source.plane(PlaneIndex::Y).view();
        let (mut sse, mut token_bits, mut coded) = (0u64, 0.0, false);
        for (tx_block, pred) in recon::tx_blocks(PlaneIndex::Y, block.x, block.y, block.size(), tx)
            .zip(prediction.chunks_exact(area))
        {
            let scratch = &mut self.scratch;
            let residual = &mut scratch.residual[..area];
            for ((line, pred), dy) in residual
                .chunks_exact_mut(side)
                .zip(pred.chunks_exact(side))
                .zip(0..)
            {
                let row = &src.row(tx_block.y + dy)[tx_block.x..tx_block.x + side];
                for ((r, &s), &p) in line.iter_mut().zip(row).zip(pred) {
                    *r = i16::from(s) - i16::from(p);
                }
            }
            let coefficients = &mut scratch.coefficients[..area];
            transform::forward(&self.dsp, tx, ty, self.lossless, residual, coefficients);
            let levels = &mut scratch.levels[..area];
            let nonzero = transform::quantize(
                &self.dsp,
                tx,
                self.lossless,
                coefficients,
                steps,
                quantization,
                levels,
            );
            let levels = if nonzero {
                &*levels
            } else {
                &transform::ZERO_LEVELS[..area]
            };
            sse += transform::coefficient_sse(tx, coefficients, nonzero.then_some(levels), steps);
            if nonzero {
                coded = true;
                let eob = coeff::end_of_block(&self.dsp, levels, tx, ty);
                let ctx = self.grids.contexts.coefficient(
                    PlaneIndex::Y,
                    tx_block.x >> 2,
                    tx_block.y >> 2,
                    tx,
                );
                token_bits += coeff::cost((levels, eob), model, probs, ctx);
            }
        }
        let (row, col) = (block.y >> MI_LOG2, block.x >> MI_LOG2);
        let skip_prob = self.context.skip[contexts::skip(&self.grids.grid, row, col)];
        let lambda = self.lambda();
        if coded {
            sse as f32 + lambda * (mode_bits + token_bits + tree::bit_cost(false, skip_prob))
        } else {
            sse as f32 + lambda * (mode_bits + tree::bit_cost(true, skip_prob))
        }
    }

    /// Squared luma error of `prediction` (laid out transform block after
    /// transform block at `tx`) against `block` of `source`.
    fn tiled_sse(
        &self,
        source: &Picture,
        block: &Block,
        tx: transform::TxSize,
        prediction: &[u8],
    ) -> u64 {
        let (side, area) = (tx.side(), tx.area());
        let size = nc_dsp::BlockSize::new(side, side).expect("square VP9 transform block");
        recon::tx_blocks(PlaneIndex::Y, block.x, block.y, block.size(), tx)
            .zip(prediction.chunks_exact(area))
            .map(|(tx_block, pred)| {
                let src = source
                    .plane(PlaneIndex::Y)
                    .region(tx_block.x as isize, tx_block.y as isize, side, side)
                    .expect("transform block inside the source");
                let pred = PlaneView::new(pred, side, side, side).expect("prediction covers it");
                u64::from(self.dsp.variance(size, &src, &pred).sse)
            })
            .sum()
    }

    /// `block`'s luma prediction from the active reference at the engine's
    /// `mv` (as the coder converts it, with the frame's filter) into `out`,
    /// laid out transform block after transform block at `tx`.
    fn predict_inter_luma(
        &mut self,
        engine_ref: &Picture,
        block: &Block,
        mv: EngineMv,
        tx: transform::TxSize,
        out: &mut Vec<u8>,
    ) {
        let n = block.size();
        let refs = self.refs_for(block);
        let vp9_mv = self.coded_vector(&refs, mv);
        let (held, ref_size) = self.active_reference();
        let reference = held.as_deref().unwrap_or(engine_ref);
        let filter = self
            .header
            .interp_filter
            .unwrap_or(nc_dsp::Vp9Filter::Regular);
        let mut whole = std::mem::take(&mut self.final_scratch.block_prediction);
        whole.resize(n * n, 0);
        inter_pred::predict_with_ref_size(
            &self.dsp,
            reference,
            ref_size,
            PlaneIndex::Y,
            &self.position(block),
            vp9_mv,
            filter,
            &mut self.inter.window,
            &mut whole,
        );
        out.resize(n * n, 0);
        let (side, area) = (tx.side(), tx.area());
        for (tile, offset) in out.chunks_exact_mut(area).zip(
            recon::tx_blocks(PlaneIndex::Y, block.x, block.y, n, tx)
                .map(|t| (t.y - block.y) * n + t.x - block.x),
        ) {
            for (row, line) in tile.chunks_exact_mut(side).enumerate() {
                line.copy_from_slice(&whole[offset + row * n..][..side]);
            }
        }
        self.final_scratch.block_prediction = whole;
    }
}
