//! The encoder's choice of an inter CU's motion beyond the engine's vector.
//!
//! The engine decides inter CUs with quarter-sample vectors. Each inter CU
//! also considers, ranked by luma SATD plus √λ-weighted bits:
//! - regular merge candidates off the quarter-sample grid (merged from an
//!   affine neighbour, or pairwise averages at an eighth sample in lists of
//!   three or more, within `vvc.pairwise_merge_reach` of the vector without
//!   affine motion; the engine cannot price them);
//!
//! and with affine motion enabled (`vvc.affine`):
//! - its sub-block merge candidates (`vvc.affine_merge`);
//! - an affine AMVP model fitted by `affine_search::refine` from the vector
//!   and from the best merged model (`vvc.affine_amvp`).
//!
//! The best is confirmed against the vector by full luma rate-distortion
//! cost: SATD alone mispredicts a quarter of these decisions.
//!
//! Speed heuristics (measured on `bodycam_bc1`, antare, airport and
//! `crowd_mot17_09` at 300 kbps and QP 36; docs/handoff/vvc.md
//! "vvc-quality4"): both searches run only where the vector's luma SATD
//! per sample over √λ exceeds a gate (`vvc.affine_merge_gate` 0.1 keeps
//! ~90% of the merged models' gain; `vvc.affine_amvp_gate` 0.4 keeps ~70%
//! of the fitted models' at ~11% of the area), the fit only where the
//! neighbours' motion differs from the vector (`motion_varies`: half the
//! static CUs skipped at ~1% of the gain), and the search predicts in 8x8
//! blocks (`Frame::predict_coarse`).
use super::{VvcBackend, merge_position};
use crate::affine::Model;
use crate::affine_candidates::{self, AFFINE_AMVP, MAX_SUBBLOCK_MERGE};
use crate::affine_search::{self, Frame};
use crate::motion::{Motion, Mv16, PAIRWISE_LIST_SIZE, Predictors};
use crate::{inter, syntax};
use nc_common::{Picture, PlaneIndex};
use nc_encoder::Block;
use std::borrow::Cow;

/// What the searches of one CU share.
struct Search<'a> {
    frame: Frame<'a>,
    p: Predictors,
    /// The engine's vector, its luma prediction (borrowed from the
    /// engine's cache when it kept one) and SATD, and its SATD-domain cost.
    mv: Mv16,
    vector_prediction: Cow<'a, [u8]>,
    vector_satd: u32,
    vector_cost: f32,
    /// √λ: the SATD-domain price of a bit.
    satd_lambda: f32,
    /// The vector's luma SATD per sample over √λ (the gates' measure).
    leftover: f32,
}

impl VvcBackend {
    /// The motion of inter CU `block`, which the engine chose to code at
    /// vector `mv`: the cheapest alternative of the module's searches
    /// where full luma RD confirms it, else the vector.
    pub(super) fn choose_motion(
        &self,
        block: &Block,
        mv: Mv16,
        source: &Picture,
        reference: &Picture,
    ) -> Motion {
        let vector = Motion::Vector(mv);
        let affine = self.tools.affine.enabled();
        // Without affine motion, only pairwise averages leave the grid, and
        // only those near the vector are tried: a far average rarely beats
        // it, and each costs an exact prediction and two RD checks.
        let reach = 4 * self.pairwise_merge_reach;
        let near = |c: Mv16| affine || ((c.x - mv.x).abs() <= reach && (c.y - mv.y).abs() <= reach);
        if !affine && (reach == 0 || self.tools.merge_candidates < PAIRWISE_LIST_SIZE) {
            return vector;
        }
        let p = self.predictors(block);
        if !affine
            && !p
                .merge
                .as_slice()
                .iter()
                .any(|&m| m.quarter().is_none() && near(m))
        {
            return vector;
        }
        let search = self.search(block, mv, p, source, reference);
        let mut best = self.off_grid_merge(&search, near);
        // The off-grid candidate's prediction is kept in `rd_best_prediction`
        // for its confirmation unless an affine model replaces it.
        let off_grid = best.map(|(candidate, _)| candidate);
        let mut best_affine = None;
        if affine && self.affine_search.merge && search.leftover > self.affine_search.merge_gate {
            best_affine = self.affine_merge(&search);
        }
        if affine
            && self.affine_search.amvp
            && search.leftover > self.affine_search.amvp_gate
            && syntax::affine_signalled(self.tools, block)
            && affine_candidates::motion_varies(&self.motion, block, mv)
            && let Some(fitted) = self.affine_amvp(&search, best_affine)
        {
            keep_cheapest(&mut best_affine, fitted.0, fitted.1);
        }
        if let Some((model, cost)) = best_affine {
            keep_cheapest(&mut best, Motion::Affine(model), cost);
        }
        let Some((candidate, _)) = best.filter(|&(_, cost)| cost < search.vector_cost) else {
            return vector;
        };
        if !self.pairwise_merge_check && Some(candidate) == off_grid {
            return candidate;
        }
        // Both predictions are known already: the vector's from the search,
        // the off-grid candidate's from its ranking.
        let kept = self.rd_best_prediction.borrow();
        let known = (Some(candidate) == off_grid).then_some(&kept[..]);
        let wins = self.luma_rd(&search, &candidate, known)
            < self.luma_rd(&search, &vector, Some(&search.vector_prediction));
        if wins { candidate } else { vector }
    }

    /// The shared state of `block`'s searches (its predictors `p`), with
    /// the vector's cost (from the engine's cached prediction when it kept
    /// one).
    fn search<'a>(
        &'a self,
        block: &Block,
        mv: Mv16,
        p: Predictors,
        source: &'a Picture,
        reference: &'a Picture,
    ) -> Search<'a> {
        let frame = Frame {
            dsp: &self.dsp,
            source,
            reference,
            cu: *block,
        };
        let satd_lambda = self.lambda.sqrt();
        let cached = mv
            .quarter()
            .is_some_and(|q| self.cached_luma == Some((*block, q)));
        let vector_prediction = if cached {
            Cow::Borrowed(&self.cached_luma_samples[..])
        } else {
            let mut prediction = Vec::new();
            vector_prediction(&frame, mv, &mut prediction);
            Cow::Owned(prediction)
        };
        let vector_satd = frame.satd(&vector_prediction);
        let distortion = vector_satd as f32;
        Search {
            vector_cost: distortion + satd_lambda * self.vector_bits(block, &p, mv),
            leftover: distortion / (satd_lambda * block.area() as f32),
            frame,
            p,
            mv,
            vector_prediction,
            vector_satd,
            satd_lambda,
        }
    }

    /// The cheapest regular merge candidate off the quarter-sample grid
    /// among those `near` accepts.
    fn off_grid_merge(&self, s: &Search<'_>, near: impl Fn(Mv16) -> bool) -> Option<(Motion, f32)> {
        let mut best = None;
        let merge = s.p.merge.as_slice();
        // The RD buffers are free until the confirmation (`luma_rd`), which
        // reuses the cheapest candidate's prediction from `kept`.
        let mut prediction = self.rd_prediction.borrow_mut();
        let mut kept = self.rd_best_prediction.borrow_mut();
        for (index, &candidate) in merge.iter().enumerate() {
            if candidate.quarter().is_none()
                && near(candidate)
                && !merge[..index].contains(&candidate)
            {
                let bits = self.vector_bits(&s.frame.cu, &s.p, candidate);
                vector_prediction(&s.frame, candidate, &mut prediction);
                let satd = s.frame.satd(&prediction);
                let cost = satd as f32 + s.satd_lambda * bits;
                if keep_cheapest(&mut best, Motion::Vector(candidate), cost) {
                    std::mem::swap(&mut *prediction, &mut *kept);
                }
            }
        }
        best
    }

    /// The cheapest distinct, non-translational sub-block merge candidate.
    fn affine_merge(&self, s: &Search<'_>) -> Option<(Model, f32)> {
        let cu = &s.frame.cu;
        if !syntax::subblock_merge_signalled(self.tools, cu) {
            return None;
        }
        let list = self.subblock_merge_list(cu);
        let mut prediction = vec![0u8; cu.area()];
        let mut best = None;
        for (index, model) in list.iter().enumerate() {
            if !affine_search::is_affine(model) || list[..index].contains(model) {
                continue;
            }
            let bits = self.subblock_merge_bits(&s.p, index);
            let cost = s.frame.model_satd(model, &mut prediction) as f32 + s.satd_lambda * bits;
            keep_cheapest(&mut best, *model, cost);
        }
        best
    }

    /// The best 4-parameter AMVP model fitted from the vector and, where it
    /// beat the vector, from the best merged model `merged`.
    fn affine_amvp(&self, s: &Search<'_>, merged: Option<(Model, f32)>) -> Option<(Model, f32)> {
        let preds = affine_candidates::amvp_list(&self.motion, &s.frame.cu, false);
        let translation = Model {
            cp: [s.mv, s.mv, Mv16::ZERO],
            six: false,
        };
        let merged = merged
            .filter(|&(model, cost)| !model.six && cost < s.vector_cost)
            .map(|(model, _)| model);
        let mut best = None;
        // The translation's search prediction is the vector's (one vector
        // for every block).
        let starts =
            std::iter::once((translation, Some((&s.vector_prediction[..], s.vector_satd))))
                .chain(merged.map(|model| (model, None)));
        for (start, known) in starts {
            let (model, cost) = affine_search::refine(
                &s.frame,
                start,
                known,
                self.affine_search.iterations,
                |m, satd| satd as f32 + s.satd_lambda * self.affine_amvp_bits(&s.p, &preds, m),
            );
            if affine_search::is_affine(&model) {
                keep_cheapest(&mut best, model, cost);
            }
        }
        best
    }

    /// Full luma rate-distortion cost of coding the CU at `motion`:
    /// quantised residual error plus λ times its coefficient and motion
    /// bits. `known` is its luma prediction when the search already made it.
    fn luma_rd(&self, s: &Search<'_>, motion: &Motion, known: Option<&[u8]>) -> f32 {
        let cu = &s.frame.cu;
        let mut predicted = self.rd_prediction.borrow_mut();
        let prediction = if let Some(prediction) = known {
            prediction
        } else {
            inter::predict_cu_into(
                &self.dsp,
                s.frame.reference,
                PlaneIndex::Y,
                cu,
                motion,
                &mut predicted,
            );
            &predicted[..]
        };
        let quant = self.quantizers().plane(PlaneIndex::Y, false);
        let mut contexts = self.contexts.coeff.clone();
        let (sse, coefficient_bits) = crate::inter_cost::residual_cost(
            &self.dsp,
            s.frame.source,
            cu,
            prediction,
            &mut contexts,
            (quant, self.tools.max_tb_log2()),
            &mut self.planning_scratch.borrow_mut(),
        );
        let motion_bits = match motion {
            Motion::Vector(mv) => self.vector_bits(cu, &s.p, *mv),
            Motion::Affine(model) => {
                if let Some(index) = self.subblock_merge_position(cu, model) {
                    self.subblock_merge_bits(&s.p, index)
                } else {
                    let preds = affine_candidates::amvp_list(&self.motion, cu, model.six);
                    self.affine_amvp_bits(&s.p, &preds, model)
                }
            }
        };
        sse as f32 + self.lambda * (coefficient_bits + motion_bits)
    }

    /// The sub-block merge list of `cu`.
    fn subblock_merge_list(&self, cu: &Block) -> Vec<Model> {
        affine_candidates::merge_list(
            &self.motion,
            cu,
            MAX_SUBBLOCK_MERGE,
            self.tools.affine.six(),
        )
    }

    /// The `merge_subblock_idx` of the first sub-block merge candidate of
    /// `cu` equal to `model`.
    pub(super) fn subblock_merge_position(&self, cu: &Block, model: &Model) -> Option<usize> {
        if !syntax::subblock_merge_signalled(self.tools, cu) {
            return None;
        }
        self.subblock_merge_list(cu)
            .iter()
            .position(|candidate| candidate == model)
    }

    /// Estimated bits of coding one-vector motion `mv` for `cu` (skip or
    /// merge, else AMVP; with affine motion enabled, the flags saying it is
    /// not affine).
    pub(super) fn vector_bits(&self, cu: &Block, p: &Predictors, mv: Mv16) -> f32 {
        let c = &self.contexts;
        if let Some(index) = merge_position(p, mv) {
            let not_subblock = if syntax::subblock_merge_signalled(self.tools, cu) {
                c.subblock_merge[p.affine_ctx].bits(false)
            } else {
                0.
            };
            return c.skip[p.skip_ctx].bits(true)
                + not_subblock
                + syntax::merge_index_bits(c, self.tools.merge_candidates, index);
        }
        let not_affine = if syntax::affine_signalled(self.tools, cu) {
            c.affine[p.affine_ctx].bits(false)
        } else {
            0.
        };
        c.skip[p.skip_ctx].bits(false)
            + c.intra[p.intra_ctx].bits(false)
            + c.merge.bits(false)
            + not_affine
            + syntax::amvp_coding(c, p, mv).2
    }

    /// Estimated bits of merging sub-block candidate `index` (as a skip CU).
    fn subblock_merge_bits(&self, p: &Predictors, index: usize) -> f32 {
        self.contexts.skip[p.skip_ctx].bits(true)
            + syntax::subblock_merge_bits(&self.contexts, p.affine_ctx, index)
    }

    /// Estimated bits of affine AMVP model `model` against its predictors
    /// `preds` (the affine flags and the MVDs).
    fn affine_amvp_bits(&self, p: &Predictors, preds: &[Model; AFFINE_AMVP], model: &Model) -> f32 {
        let c = &self.contexts;
        let model_type = if self.tools.affine.six() {
            c.affine_type.bits(model.six)
        } else {
            0.
        };
        c.skip[p.skip_ctx].bits(false)
            + c.intra[p.intra_ctx].bits(false)
            + c.merge.bits(false)
            + c.affine[p.affine_ctx].bits(true)
            + model_type
            + syntax::affine_amvp_coding(c, preds, model).2
    }
}

/// The CU's luma prediction at vector `mv` into `prediction`.
fn vector_prediction(frame: &Frame<'_>, mv: Mv16, prediction: &mut Vec<u8>) {
    let cu = &frame.cu;
    prediction.resize(cu.area(), 0);
    let luma = frame.reference.plane(PlaneIndex::Y);
    let size = (cu.width(), cu.height());
    let mut out = inter::packed(prediction, size);
    inter::predict_luma(frame.dsp, luma, (cu.x, cu.y), size, mv, &mut out);
}

/// Keep `item` in `best` if it costs less than what `best` holds; whether
/// it was kept.
fn keep_cheapest<T>(best: &mut Option<(T, f32)>, item: T, cost: f32) -> bool {
    let cheaper = best.as_ref().is_none_or(|(_, held)| cost < *held);
    if cheaper {
        *best = Some((item, cost));
    }
    cheaper
}
