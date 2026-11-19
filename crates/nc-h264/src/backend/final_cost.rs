//! H.264's pricing for the engine's final decision stage
//! (`nc_encoder::CodecBackend::final_intra_screen` and `final_coded_cost`;
//! the stage itself is the engine's `final_stage.rs`).
//!
//! Luma only, at the macroblock's real syntax: the screen is the best
//! `Intra_16x16` prediction's squared error (no residual) plus λ times its
//! measured header bits; the coded cost is the luma residual transformed
//! (8x8 for inter, as the macroblock coder codes it; `Intra_16x16`'s 4x4 and
//! DC Hadamard for intra), quantised without trellis, reconstructed into a
//! scratch block (the picture is not touched), plus λ times the CABAC bits
//! of the macroblock with chroma uncoded, counted at the slice's contexts as
//! they stand. `P_Skip` is priced as such where the skip vector codes no
//! luma. Chroma, trellis, transform-size RD and residual pruning are left
//! out: together with all three planes they made `h264.rd_refine`'s trials
//! cost +33 % to +53 % instructions (docs/handoff/rd-mode-decision.md).

use nc_common::{Picture, PlaneIndex, PlaneView};
use nc_dsp::BlockSize;
use nc_encoder::{Block, FinalCandidate, Mv};

use super::{H264Backend, MB_SIZE};
use crate::macroblock::{
    IntraModes, Residual, ResidualKind, ResidualSkip, predict_inter_chroma,
    reconstruct_chroma_into, reconstruct_luma_into, transform_luma, transform_macroblock,
};
use crate::mbinfo::MbPartition;
use crate::predict::{
    Availability, ChromaMode, Edges, Intra16x16Mode, chroma_8x8, dc_16x16_sse, luma_16x16,
};
use crate::syntax::CodedMb;

/// Header bits charged to the `Intra_16x16` screen: the mean CABAC bits of
/// an uncoded-residual `I_16x16` macroblock in P slices (skip flag, prefix,
/// `mb_type`, chroma mode, QP delta) on the six worst VC-1K clips, QP 33 and
/// 39 (211,625 macroblocks of the shadow log, 13.49 bits).
const INTRA_16X16_SCREEN_BITS: f32 = 13.5;

/// The `Intra_16x16` prediction the last intra screen chose.
#[derive(Debug, Clone)]
pub(super) struct ScreenedIntra {
    origin: (usize, usize),
    mode: Intra16x16Mode,
    /// The reconstruction's neighbours, to predict the mode when priced.
    edges: Edges<16>,
    /// The chroma neighbours (Cb, Cr), for the chroma DC prediction
    /// `h264.final_chroma` prices.
    chroma_edges: Option<[Edges<8>; 2]>,
}

/// Chroma side of a macroblock (4:2:0).
const MB_CHROMA: usize = MB_SIZE / 2;

impl H264Backend {
    /// See `nc_encoder::CodecBackend::final_intra_screen`: the
    /// `Intra_16x16` mode of least luma squared error from `recon`. DC is
    /// priced in closed form from the source's sums; the other modes are
    /// predicted.
    pub(super) fn screen_intra(
        &mut self,
        source: &Picture,
        recon: &Picture,
        block: &Block,
    ) -> (f32, f32) {
        let (x, y) = (block.x, block.y);
        let edges = Edges::gather(&recon.plane(PlaneIndex::Y).view(), x, y);
        let src = source
            .plane(PlaneIndex::Y)
            .region(x as isize, y as isize, MB_SIZE, MB_SIZE)
            .expect("macroblock inside the picture");
        let size = BlockSize::new(MB_SIZE, MB_SIZE).expect("16x16 is a kernel size");
        let sums = self.dsp.block_sums(size, &src);
        let sum = u32::try_from(sums.sum).expect("sample sums are non-negative");
        let dc = dc_16x16_sse(&edges, sum, u64::from(sums.sse));
        let (mode, sse) = Intra16x16Mode::usable(Availability::at(x, y))
            .filter(|&mode| mode != Intra16x16Mode::Dc)
            .map(|mode| {
                let luma = luma_16x16(&edges, mode);
                let view = PlaneView::new(&luma, MB_SIZE, MB_SIZE, MB_SIZE).expect("16x16 buffer");
                (mode, u64::from(self.dsp.variance(size, &src, &view).sse))
            })
            .fold((Intra16x16Mode::Dc, dc), |best, m| {
                if m.1 < best.1 { m } else { best }
            });
        let chroma_edges = (self.tuning.final_chroma >= 0.5).then(|| {
            [PlaneIndex::U, PlaneIndex::V]
                .map(|plane| Edges::gather(&recon.plane(plane).view(), x / 2, y / 2))
        });
        self.screened_intra = Some(ScreenedIntra {
            origin: (x, y),
            mode,
            edges,
            chroma_edges,
        });
        (sse as f32, INTRA_16X16_SCREEN_BITS)
    }

    /// See `nc_encoder::CodecBackend::final_coded_cost`. `None` for intra
    /// without a screen of this macroblock.
    pub(super) fn price_final(
        &mut self,
        source: &Picture,
        reference: &Picture,
        block: &Block,
        candidate: FinalCandidate,
    ) -> Option<f32> {
        let (x, y) = (block.x, block.y);
        let mut pred = self
            .final_pred
            .take()
            .unwrap_or_else(|| Box::new(crate::macroblock::Prediction::ZERO));
        let chroma = self.tuning.final_chroma >= 0.5;
        let Some(kind) = self.final_prediction(reference, block, candidate, chroma, &mut pred)
        else {
            self.final_pred = Some(pred);
            return None;
        };
        let mut residual = self
            .alt_residual
            .take()
            .unwrap_or_else(|| Box::new(Residual::empty(kind)));
        if chroma {
            transform_macroblock(
                &self.dsp,
                source,
                &pred,
                x,
                y,
                kind,
                &self.quantizers,
                ResidualSkip::NONE,
                &mut residual,
            );
        } else {
            residual.reset(kind);
            transform_luma(
                &self.dsp,
                source,
                &pred.luma,
                (x, y),
                kind,
                &self.quantizers,
                0,
                &mut residual,
            );
        }
        let distortion = self.final_distortion(source, block, &pred, &residual, chroma);
        let bits = self.final_bits(block, candidate, &residual);
        self.alt_residual = Some(residual);
        self.final_pred = Some(pred);
        Some(distortion + self.rd_lambda * bits)
    }

    /// `candidate`'s prediction of `block` into `pred` (chroma too with
    /// `chroma`) and the residual kind it is coded with; `None` for intra
    /// without a screen of this macroblock.
    fn final_prediction(
        &self,
        reference: &Picture,
        block: &Block,
        candidate: FinalCandidate,
        chroma: bool,
        pred: &mut crate::macroblock::Prediction,
    ) -> Option<ResidualKind> {
        let (x, y) = (block.x, block.y);
        match candidate {
            FinalCandidate::Inter(mv) => {
                self.final_inter_luma(reference, block, mv, &mut pred.luma);
                if chroma {
                    predict_inter_chroma(&self.mc, reference, x, y, mv, pred);
                }
                Some(ResidualKind::Inter8x8)
            }
            FinalCandidate::Intra => {
                let screened = self
                    .screened_intra
                    .as_ref()
                    .filter(|s| s.origin == (x, y))?;
                pred.luma = luma_16x16(&screened.edges, screened.mode);
                if let Some(edges) = screened.chroma_edges.filter(|_| chroma) {
                    pred.chroma = edges.map(|e| chroma_8x8(&e, ChromaMode::Dc));
                }
                Some(ResidualKind::Intra16x16)
            }
        }
    }

    /// Squared error of `block` reconstructed from `pred` and `residual`
    /// (scratch only): luma, plus chroma with `chroma`.
    fn final_distortion(
        &self,
        source: &Picture,
        block: &Block,
        pred: &crate::macroblock::Prediction,
        residual: &Residual,
        chroma: bool,
    ) -> f32 {
        let (x, y) = (block.x, block.y);
        let src = source
            .plane(PlaneIndex::Y)
            .region(x as isize, y as isize, MB_SIZE, MB_SIZE)
            .expect("macroblock inside the picture");
        let size = BlockSize::new(MB_SIZE, MB_SIZE).expect("16x16 is a kernel size");
        // Nothing coded: the prediction is the reconstruction.
        let mut distortion = if residual.is_empty() {
            let view = PlaneView::new(&pred.luma, MB_SIZE, MB_SIZE, MB_SIZE).expect("16x16 buffer");
            self.dsp.variance(size, &src, &view).sse as f32
        } else {
            let mut recon = [0u8; MB_SIZE * MB_SIZE];
            reconstruct_luma_into(
                &self.dsp,
                &mut recon,
                &pred.luma,
                residual,
                &self.quantizers,
            );
            let view = PlaneView::new(&recon, MB_SIZE, MB_SIZE, MB_SIZE).expect("16x16 buffer");
            self.dsp.variance(size, &src, &view).sse as f32
        };
        if chroma {
            let mut recon = [[0u8; MB_CHROMA * MB_CHROMA]; 2];
            reconstruct_chroma_into(
                &self.dsp,
                &mut recon,
                &pred.chroma,
                residual,
                &self.quantizers,
            );
            let size = BlockSize::new(MB_CHROMA, MB_CHROMA).expect("8x8 is a kernel size");
            for (block, plane) in recon.iter().zip([PlaneIndex::U, PlaneIndex::V]) {
                let src = source
                    .plane(plane)
                    .region((x / 2) as isize, (y / 2) as isize, MB_CHROMA, MB_CHROMA)
                    .expect("chroma block inside the picture");
                let view =
                    PlaneView::new(block, MB_CHROMA, MB_CHROMA, MB_CHROMA).expect("8x8 buffer");
                distortion += self.dsp.variance(size, &src, &view).sse as f32;
            }
        }
        distortion
    }

    /// CABAC bits of `block` coded as `candidate` with `residual`, at the
    /// slice's contexts as they stand (`P_Skip` where it qualifies).
    fn final_bits(&self, block: &Block, candidate: FinalCandidate, residual: &Residual) -> f32 {
        match candidate {
            FinalCandidate::Inter(mv) => {
                let coded = if Some(mv) == self.skip_mv_of(block) && residual.is_empty() {
                    CodedMb::Skip(mv)
                } else {
                    CodedMb::Inter {
                        partition: MbPartition::P16x16,
                        mvs: [mv; 4],
                        ref_idx: self.coded_ref_idx(),
                        mvp: self.known_mvp(block),
                        residual,
                    }
                };
                self.mb_bits(block, &coded)
            }
            FinalCandidate::Intra => {
                let mode = self
                    .screened_intra
                    .as_ref()
                    .expect("priced intra has a screen")
                    .mode;
                let modes = IntraModes {
                    luma: mode,
                    chroma: ChromaMode::Dc,
                    luma_satd: 0,
                };
                self.mb_bits(block, &CodedMb::Intra16x16 { modes, residual })
            }
        }
    }

    /// The exact 16x16 luma prediction of `block` at `mv` into `luma`: the
    /// ranking's cached prediction when it is this vector's, else predicted.
    fn final_inter_luma(
        &self,
        reference: &Picture,
        block: &Block,
        mv: Mv,
        luma: &mut [u8; MB_SIZE * MB_SIZE],
    ) {
        let cached = self.exact_luma.take();
        match &cached {
            Some(c)
                if c.origin == (block.x, block.y)
                    && c.mv == mv
                    && c.ref_idx == self.selected_ref =>
            {
                *luma = c.pixels;
            }
            _ => self
                .mc
                .luma_16x16(reference.plane(PlaneIndex::Y), block.x, block.y, mv, luma),
        }
        self.exact_luma.set(cached);
    }
}
