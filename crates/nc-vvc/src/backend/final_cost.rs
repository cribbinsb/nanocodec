//! VVC's pricing for the engine's final decision stage
//! (`nc_encoder::CodecBackend::final_intra_screen` and `final_coded_cost`;
//! the stage itself is the engine's `final_stage.rs`; the skip vector,
//! merge candidate 0, is `CodecBackend::skip_vector` in `backend.rs`).
//!
//! Luma only, at the CU's largest transform units, without touching the
//! picture or the coded state (H.264's pricing, `nc-h264`
//! `backend/final_cost.rs`, has the same shape):
//! - the intra screen is the squared error of the best of the core modes
//!   (planar, DC, horizontal, vertical: the coarse P-picture search's
//!   first candidates), each unit predicted from the reconstruction as it
//!   stands (units after the first would see their coded neighbours:
//!   pessimistic for CUs of several units), plus the intra mode bits the
//!   ranking charges;
//! - the coded cost is `inter_cost::residual_cost` (the pricing the planner's
//!   `inter_leaf_cost` uses: plain quantiser, coefficient bits at a copy of
//!   the contexts) plus λ times the CU's mode bits.
//!
//! Chroma, the trellis, MTS, ISP and the intra direction search are left
//! out (H.264: pricing all three planes cost 2-4 % more instructions for
//! 0.2-0.3 points of J): the coder still runs them on the candidate that
//! wins.

use nc_common::{Picture, PlaneIndex, PlaneView};
use nc_dsp::BlockSize;
use nc_encoder::{Block, CodecBackend, FinalCandidate, PredictionMode};

use super::VvcBackend;
use crate::intra::{self, DC, HORIZONTAL, PLANAR, VERTICAL};
use crate::{inter_cost, transform_tree};

/// The modes the intra screen tries.
const SCREENED_MODES: [u8; 4] = [PLANAR, DC, HORIZONTAL, VERTICAL];

/// The last intra screen: each screened mode's luma prediction of the CU
/// (row-major) and the best one.
#[derive(Debug, Clone, Default)]
pub(super) struct ScreenedIntra {
    block: Option<Block>,
    modes: [Vec<u8>; SCREENED_MODES.len()],
    best: usize,
    /// One unit's predictions (scratch).
    units: [Vec<u8>; SCREENED_MODES.len()],
}

impl VvcBackend {
    /// See `nc_encoder::CodecBackend::final_intra_screen`. The prediction is
    /// kept for the pricing.
    pub(super) fn screen_intra(
        &mut self,
        source: &Picture,
        recon: &Picture,
        block: &Block,
    ) -> (f32, f32) {
        let mut screened = std::mem::take(&mut self.final_screen);
        let side = block.size();
        for prediction in &mut screened.modes {
            prediction.resize(side * side, 0);
        }
        let mut units = std::mem::take(&mut screened.units);
        for tu in transform_tree::units(block, self.tools.max_tb_log2()) {
            intra::predictions_into(
                &self.dsp,
                recon,
                Some(&self.motion),
                (&tu, block),
                &SCREENED_MODES,
                &mut units,
            );
            let (w, dx, dy) = (tu.width(), tu.x - block.x, tu.y - block.y);
            for (unit, prediction) in units.iter().zip(&mut screened.modes) {
                for (row, samples) in unit.chunks_exact(w).enumerate() {
                    prediction[(dy + row) * side + dx..][..w].copy_from_slice(samples);
                }
            }
        }
        screened.units = units;
        let (best, sse) = screened
            .modes
            .iter()
            .map(|prediction| self.luma_sse(source, block, prediction))
            .enumerate()
            .min_by_key(|&(_, sse)| sse)
            .expect("several modes");
        screened.best = best;
        screened.block = Some(*block);
        self.final_screen = screened;
        (sse as f32, self.mode_bits(block, &PredictionMode::IntraDc))
    }

    /// See `nc_encoder::CodecBackend::final_coded_cost`. `None` for intra
    /// without a screen of this CU.
    pub(super) fn price_final(
        &self,
        source: &Picture,
        reference: &Picture,
        block: &Block,
        candidate: FinalCandidate,
    ) -> Option<f32> {
        let quant = self.quantizers().plane(PlaneIndex::Y, false);
        let mut contexts = self.contexts.coeff.clone();
        let mut scratch = self.planning_scratch.borrow_mut();
        let max_tb = self.tools.max_tb_log2();
        let (sse, coefficient_bits, mode_bits) = match candidate {
            FinalCandidate::Inter(mv) => {
                let (sse, bits) = inter_cost::luma_cost(
                    &self.dsp,
                    source,
                    reference,
                    (block, mv),
                    &mut contexts,
                    (quant, max_tb),
                    &mut scratch,
                );
                (sse, bits, self.mode_bits(block, &PredictionMode::Inter(mv)))
            }
            FinalCandidate::Intra => {
                if self.final_screen.block != Some(*block) {
                    return None;
                }
                let quant = self.quantizers().plane(PlaneIndex::Y, true);
                let (sse, bits) = inter_cost::residual_cost(
                    &self.dsp,
                    source,
                    block,
                    &self.final_screen.modes[self.final_screen.best],
                    &mut contexts,
                    (quant, max_tb),
                    &mut scratch,
                );
                (sse, bits, self.mode_bits(block, &PredictionMode::IntraDc))
            }
        };
        Some(sse as f32 + self.lambda * (coefficient_bits + mode_bits))
    }

    /// Squared luma error of `prediction` (row-major) against `block` of
    /// `source`.
    fn luma_sse(&self, source: &Picture, block: &Block, prediction: &[u8]) -> u64 {
        let n = block.size();
        let size = BlockSize::new(n, n).expect("square CUs are kernel sizes");
        let src = source
            .plane(PlaneIndex::Y)
            .region(block.x as isize, block.y as isize, n, n)
            .expect("block inside source");
        let pred = PlaneView::new(prediction, n, n, n).expect("prediction covers the block");
        u64::from(self.dsp.variance(size, &src, &pred).sse)
    }
}
