//! AV1's pricing for the engine's final decision stage
//! (`nc_encoder::CodecBackend::final_intra_screen`, `final_coded_cost` and
//! `skip_vector`; the stage itself is the engine's `final_stage.rs`).
//!
//! Luma only, at the block's largest transform, without touching the
//! picture or the coded state (H.264's pricing, `nc-h264`
//! `backend/final_cost.rs`, has the same shape):
//! - the skip vector is `NEARESTMV` (the block's reference list position 0,
//!   or the global vector, zero, where the list is empty): the cheapest
//!   vector to signal, which the winner walk keeps where it wins
//!   (`decision.rs`: walking H.264's `P_Skip` vector cost J +7.2 % on a
//!   still camera);
//! - the intra screen is the squared error of the intra prediction the
//!   coder would make (DC where `av1.inter_intra_search` is 0, else the best
//!   of the search's smooth and axis-aligned modes) from the
//!   reconstruction, plus its inter-frame mode bits;
//! - the coded cost is [`Av1Backend::residual_rd_cost`] of the luma residual
//!   at the block's transform size (plain quantiser, coefficient bits at the
//!   current CDFs, distortion estimated on the coefficients) plus λ times
//!   the mode and skip-flag bits: a candidate whose levels all quantise to
//!   zero is priced as the skipped block it will be coded as.
//!
//! Chroma, the trellis and the transform-split search are left out (they
//! made H.264's full trials cost +33 to +53 % instructions; pricing chroma
//! bought 0.2-0.3 points of J for 2-4 % instructions there): the coder
//! still runs them on the candidate that wins.

use nc_common::{Picture, PlaneIndex, PlaneView, PlaneViewMut};
use nc_dsp::BlockSize;
use nc_encoder::{Block, CodecBackend, FinalCandidate, Mv, PredictionMode};

use super::p_intra::InterIntraSearch;
use super::{Av1Backend, Prediction, skip_context, to_av1_mv, to_engine_mv};
use crate::inter;
use crate::intra::LumaIntra;
use crate::predict::{IntraMode, dc_value};
use crate::tables::DC_PRED;
use crate::transform::TxSize;

/// The intra modes the screen tries where the coder searches
/// (`av1.inter_intra_search` 1 or 2): the smooth and axis-aligned members of
/// the search's candidates (`p_intra.rs`), whose predictions are cheapest.
const SCREENED_MODES: [IntraMode; 5] = [
    IntraMode::Dc,
    IntraMode::Vertical,
    IntraMode::Horizontal,
    IntraMode::Smooth,
    IntraMode::Paeth,
];

/// The intra prediction the last screen chose, kept for its pricing.
#[derive(Debug, Clone)]
pub(super) struct ScreenedIntra {
    block: Block,
    luma: LumaIntra,
    /// Its luma prediction, row-major.
    prediction: Vec<u8>,
}

/// Scratch of the stage's pricing (allocated once).
#[derive(Debug, Default)]
pub(super) struct FinalScratch {
    /// One candidate's luma prediction, row-major.
    prediction: Vec<u8>,
    screened: Option<ScreenedIntra>,
}

impl Av1Backend {
    /// See `nc_encoder::CodecBackend::skip_vector`: `NEARESTMV`.
    pub(super) fn nearest_vector(&mut self, block: &Block) -> Mv {
        let stack = self.stack_for(block);
        if stack.count > 0 {
            to_engine_mv(stack.mvs[0])
        } else {
            Mv::ZERO
        }
    }

    /// See `nc_encoder::CodecBackend::final_intra_screen`.
    pub(super) fn screen_intra(
        &mut self,
        source: &Picture,
        recon: &Picture,
        block: &Block,
    ) -> (f32, f32) {
        let searched = InterIntraSearch::from_param(self.params.inter_intra_search)
            != InterIntraSearch::DcOnly
            && !self
                .refresh
                .as_ref()
                .is_some_and(|w| w.reads_dirty_above_right(block));
        let modes: &[IntraMode] = if searched {
            &SCREENED_MODES
        } else {
            &SCREENED_MODES[..1]
        };
        let n = block.size();
        let mut screened = self.final_scratch.screened.take().unwrap_or(ScreenedIntra {
            block: *block,
            luma: LumaIntra::DC,
            prediction: Vec::new(),
        });
        let mut trial = std::mem::take(&mut self.final_scratch.prediction);
        trial.resize(n * n, 0);
        screened.prediction.resize(n * n, 0);
        let mut best = u64::MAX;
        for &mode in modes {
            let luma = LumaIntra::plain(mode);
            self.predict_intra_luma(recon, block, luma, &mut trial);
            let sse = self.luma_sse(source, block, &trial);
            if sse < best {
                best = sse;
                screened.luma = luma;
                std::mem::swap(&mut screened.prediction, &mut trial);
            }
        }
        screened.block = *block;
        let bits = self.inter_frame_intra_bits(block, screened.luma);
        self.final_scratch.prediction = trial;
        self.final_scratch.screened = Some(screened);
        (best as f32, bits)
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
        let mut prediction = std::mem::take(&mut self.final_scratch.prediction);
        let n = block.size();
        prediction.resize(n * n, 0);
        let priced = match candidate {
            FinalCandidate::Inter(mv) => {
                self.predict_inter_luma(reference, block, mv, &mut prediction);
                let mode_bits = self.mode_bits(block, &PredictionMode::Inter(mv));
                Some(self.price_luma(source, block, &prediction, (true, DC_PRED), mode_bits))
            }
            FinalCandidate::Intra => {
                let screened = self
                    .final_scratch
                    .screened
                    .take()
                    .filter(|s| s.block == *block);
                let cost = screened.as_ref().map(|s| {
                    let bits = self.inter_frame_intra_bits(block, s.luma);
                    self.price_luma(
                        source,
                        block,
                        &s.prediction,
                        (false, s.luma.intra_dir()),
                        bits,
                    )
                });
                self.final_scratch.screened = screened;
                cost
            }
        };
        self.final_scratch.prediction = prediction;
        priced
    }

    /// Coded luma cost of `block` predicted by `prediction`: the residual's
    /// [`Self::residual_rd_cost`] at the block's transform (`is_inter`, the
    /// intra direction) plus λ times `mode_bits` and the skip flag (set
    /// where no level survives quantisation: the residual cost is then the
    /// prediction's error alone).
    fn price_luma(
        &mut self,
        source: &Picture,
        block: &Block,
        prediction: &[u8],
        (is_inter, intra_dir): (bool, usize),
        mode_bits: f32,
    ) -> f32 {
        let n = block.size();
        let size = BlockSize::new(n, n).expect("AV1 blocks are kernel sizes");
        let src = source
            .plane(PlaneIndex::Y)
            .region(block.x as isize, block.y as isize, n, n)
            .expect("block inside source");
        let pred = PlaneView::new(prediction, n, n, n).expect("prediction covers the block");
        self.dsp
            .residual(size, &src, &pred, &mut self.scratch.residual, n);
        let sse = u64::from(self.dsp.variance(size, &src, &pred).sse);
        let tx = TxSize::from_log2(block.log2_size()).expect("supported transform size");
        let (cost, nonzero) = self.residual_rd_cost(tx, PlaneIndex::Y, is_inter, intra_dir, sse);
        let (above, left) = self.neighbours(block);
        let skip_cdf = &self.cdfs.skip[skip_context(above, left)];
        let lambda = self.lambda();
        if nonzero == 0 {
            sse as f32 + lambda * (mode_bits + skip_cdf.bits(1))
        } else {
            cost + lambda * (mode_bits + skip_cdf.bits(0))
        }
    }

    /// Squared luma error of `prediction` (row-major) against `block` of
    /// `source`.
    fn luma_sse(&self, source: &Picture, block: &Block, prediction: &[u8]) -> u64 {
        let n = block.size();
        let size = BlockSize::new(n, n).expect("AV1 blocks are kernel sizes");
        let src = source
            .plane(PlaneIndex::Y)
            .region(block.x as isize, block.y as isize, n, n)
            .expect("block inside source");
        let pred = PlaneView::new(prediction, n, n, n).expect("prediction covers the block");
        u64::from(self.dsp.variance(size, &src, &pred).sse)
    }

    /// `block`'s luma intra prediction `luma` from `recon` into `out`
    /// (row-major), as `predict_plane` makes it.
    fn predict_intra_luma(&self, recon: &Picture, block: &Block, luma: LumaIntra, out: &mut [u8]) {
        let n = block.size();
        let plane = recon.plane(PlaneIndex::Y);
        if luma == LumaIntra::DC {
            out.fill(dc_value(
                plane,
                block.x,
                block.y,
                n,
                block.y > 0,
                block.x > 0,
            ));
            return;
        }
        let edges = luma.edges(plane, block.x, block.y, n);
        let mut dst = PlaneViewMut::new(out, n, n, n).expect("buffer covers the block");
        luma.predict(&self.dsp, &edges, &mut dst);
    }

    /// `block`'s luma prediction from `reference` at `mv` into `out`
    /// (row-major): the ranking's cached prediction when it is this
    /// vector's, else predicted.
    fn predict_inter_luma(&self, reference: &Picture, block: &Block, mv: Mv, out: &mut [u8]) {
        let n = block.size();
        let mv = to_av1_mv(mv);
        let prediction = Prediction::Inter { reference, mv };
        if self.cached_luma_stats(block, &prediction).is_some() {
            out.copy_from_slice(&self.scratch.luma_prediction[..n * n]);
            return;
        }
        let mut dst = PlaneViewMut::new(out, n, n, n).expect("buffer covers the block");
        inter::predict(
            &self.dsp,
            reference.plane(PlaneIndex::Y),
            block.x,
            block.y,
            n,
            mv,
            false,
            &mut dst,
        );
    }
}
