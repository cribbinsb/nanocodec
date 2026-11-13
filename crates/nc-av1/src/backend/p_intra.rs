//! Intra blocks in inter frames: the luma mode search and the fallback to
//! the inter vector the engine's directional estimate beat.
//!
//! AV1 inter frames used to code every intra block DC. With the engine's
//! directional intra estimate (`decision.intra_directional`,
//! `nc_encoder::intra_estimate`) textured blocks reach intra because a
//! direction predicts them, so DC alone wastes the decision. Blocks the
//! engine marks (`CodecBackend::intra_fallback`), or every inter-frame
//! intra block with `av1.inter_intra_search` 2, get a short search: the
//! estimate's directions (DC, vertical, horizontal, D45, D135) plus smooth
//! and Paeth, ranked by prediction SATD + √λ · mode bits, the best
//! `av1.inter_intra_rd_candidates` then transform-costed as the key-frame
//! search does ([`Av1Backend::rd_best_luma`]). A marked block compares the
//! winner's rate-distortion cost with the inter vector's, both measured on
//! the reconstruction, and codes the cheaper (x264 compares intra with
//! inter the same way in P frames).

use nc_common::{Picture, PlaneIndex};
use nc_dsp::BlockSize;
use nc_encoder::{Block, CodecBackend, Mv, PredictionMode};

use super::{Av1Backend, Prediction, intra_inter_context, size_group, to_av1_mv};
use crate::intra::LumaIntra;
use crate::modeinfo::Av1Mv;
use crate::predict::IntraMode;
use crate::tables::DC_PRED;
use crate::transform::TxSize;

/// The inter-frame intra search's candidates: the directional estimate's
/// directions, then the smooth predictors.
const CANDIDATES: [IntraMode; 7] = [
    IntraMode::Dc,
    IntraMode::Vertical,
    IntraMode::Horizontal,
    IntraMode::D45,
    IntraMode::D135,
    IntraMode::Smooth,
    IntraMode::Paeth,
];

/// `av1.inter_intra_search` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum InterIntraSearch {
    /// Every inter-frame intra block is DC.
    DcOnly,
    /// Blocks the engine's directional estimate sent to intra are searched.
    Marked,
    /// Every inter-frame intra block is searched.
    All,
}

impl InterIntraSearch {
    /// From the parameter's value (rounded).
    pub(super) fn from_param(value: f32) -> Self {
        match value.round() as i32 {
            ..=0 => Self::DcOnly,
            1 => Self::Marked,
            _ => Self::All,
        }
    }
}

/// How an intra block of an inter frame is coded.
pub(super) enum InterFrameIntra<'r> {
    Intra(LumaIntra),
    /// Inter at the engine's vector after all (the fallback).
    Inter(Av1Mv, &'r Picture),
}

impl Av1Backend {
    /// The coding of intra block `block` of an inter frame: its luma mode
    /// ([`InterIntraSearch`]), or the inter vector the engine's estimate beat
    /// where that costs less on the reconstruction. Dirty-edge blocks of an
    /// intra refresh keep DC, which reads no above-right samples.
    pub(super) fn inter_frame_intra<'r>(
        &mut self,
        block: &Block,
        source: &Picture,
        reference: Option<&'r Picture>,
        recon: &mut Picture,
    ) -> InterFrameIntra<'r> {
        let marked = self
            .intra_fallback
            .take()
            .and_then(|(at, mv)| (at == (block.x, block.y, block.log2_size())).then_some(mv));
        let searched = match InterIntraSearch::from_param(self.params.inter_intra_search) {
            InterIntraSearch::DcOnly => false,
            InterIntraSearch::Marked => marked.is_some(),
            InterIntraSearch::All => true,
        } && !self
            .refresh
            .as_ref()
            .is_some_and(|w| w.on_dirty_edge(block));
        if !searched {
            return InterFrameIntra::Intra(LumaIntra::DC);
        }
        let shortlist = self.inter_frame_shortlist(block, source, recon);
        let (intra_cost, luma) = self.rd_best_luma(
            block,
            source,
            recon,
            &shortlist,
            Self::inter_frame_intra_bits,
        );
        match marked.zip(reference) {
            Some((mv, reference))
                if self.inter_rd_cost(block, source, reference, mv, recon) < intra_cost =>
            {
                InterFrameIntra::Inter(to_av1_mv(mv), reference)
            }
            _ => InterFrameIntra::Intra(luma),
        }
    }

    /// The best `av1.inter_intra_rd_candidates` of [`CANDIDATES`] by
    /// prediction SATD + √λ · mode bits.
    fn inter_frame_shortlist(
        &mut self,
        block: &Block,
        source: &Picture,
        recon: &mut Picture,
    ) -> Vec<LumaIntra> {
        let bit_cost = (self.lambda() * self.params.intra_rate_scale).sqrt();
        let mut ranked: Vec<(f32, LumaIntra)> = CANDIDATES
            .iter()
            .map(|&mode| {
                let luma = LumaIntra::plain(mode);
                let satd = self.intra_prediction_satd(block, source, recon, luma);
                (
                    satd + bit_cost * self.inter_frame_intra_bits(block, luma),
                    luma,
                )
            })
            .collect();
        ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
        let keep = self.params.inter_intra_rd_candidates.round().max(1.0) as usize;
        ranked
            .into_iter()
            .take(keep)
            .map(|(_, luma)| luma)
            .collect()
    }

    /// Bits of intra luma `luma` in an inter frame: `is_inter`, `y_mode`
    /// (AV1 §5.11.22 `intra_block_mode_info`) and what follows it
    /// ([`Self::luma_mode_tail_bits`]).
    fn inter_frame_intra_bits(&self, block: &Block, luma: LumaIntra) -> f32 {
        let (above, left) = self.neighbours(block);
        self.cdfs.intra_inter[intra_inter_context(above, left)].bits(0)
            + self.cdfs.y_mode[size_group(block)].bits(luma.mode.symbol())
            + self.luma_mode_tail_bits(block, luma)
    }

    /// Rate-distortion cost of predicting luma of `block` from `reference`
    /// at `mv` (the prediction is written into `recon`, which coding
    /// overwrites): [`Self::residual_rd_cost`] of one transform plus λ ·
    /// the vector's mode bits, the units of [`Self::rd_best_luma`].
    fn inter_rd_cost(
        &mut self,
        block: &Block,
        source: &Picture,
        reference: &Picture,
        mv: Mv,
        recon: &mut Picture,
    ) -> f32 {
        let prediction = Prediction::Inter {
            reference,
            mv: to_av1_mv(mv),
        };
        self.predict_plane(block, PlaneIndex::Y, &prediction, recon);
        let n = block.size();
        let size = BlockSize::new(n, n).expect("blocks are kernel sizes");
        let src = source
            .plane(PlaneIndex::Y)
            .region(block.x as isize, block.y as isize, n, n)
            .expect("block inside source");
        let pred = recon
            .plane(PlaneIndex::Y)
            .region(block.x as isize, block.y as isize, n, n)
            .expect("block inside reconstruction");
        self.dsp
            .residual(size, &src, &pred, &mut self.scratch.residual, n);
        let sse = u64::from(self.dsp.variance(size, &src, &pred).sse);
        let tx = TxSize::from_log2(block.log2_size()).expect("supported transform size");
        let (cost, _) = self.residual_rd_cost(tx, PlaneIndex::Y, true, DC_PRED, sse);
        cost + self.lambda() * self.mode_bits(block, &PredictionMode::Inter(mv))
    }
}
