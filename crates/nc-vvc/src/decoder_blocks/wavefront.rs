//! Wavefront entropy synchronisation (H.266 §9.3.1, §9.3.2.4): each CTU row
//! of a tile is its own CABAC substream, and starts from the context
//! variables as they were after the first CTU of the row above.
use super::{Context, Parser, filter_syntax};
use crate::contexts::Contexts;

/// Every context variable of a slice, as `TableStateIdxWpp` stores them.
pub(super) struct Saved {
    contexts: Contexts,
    trees: super::tree::Models,
    coefficients: super::coeff::Models,
    skip_coefficients: super::skip_coeff::Models,
    intra: super::intra_syntax::Models,
    inter: super::inter_models::Models,
    transforms: super::transform_syntax::Models,
    sbt: super::sbt::Models,
    ibc: super::ibc::Models,
    palette: super::palette::Models,
    palette_predictor: super::palette::Predictor,
    groups: super::quant_groups::Models,
    filters: filter_syntax::Models,
    cclm: [Context; 2],
    joint: [Context; 3],
    constraint: [Context; 2],
    ref_index: [Context; 2],
}

impl Parser<'_> {
    pub(super) fn save_contexts(&self, filters: &filter_syntax::Models) -> Saved {
        Saved {
            contexts: self.contexts.clone(),
            trees: self.trees.clone(),
            coefficients: self.coefficients.clone(),
            skip_coefficients: self.skip_coefficients.clone(),
            intra: self.intra_models.clone(),
            inter: self.inter_models.clone(),
            transforms: self.transform_models.clone(),
            sbt: self.sbt_models.clone(),
            ibc: self.ibc_models.clone(),
            palette: self.palette_models.clone(),
            palette_predictor: self.palette_predictor.clone(),
            groups: self.groups.models(),
            filters: filters.clone(),
            cclm: self.cclm,
            joint: self.joint,
            constraint: self.constraint,
            ref_index: self.ref_index,
        }
    }

    /// Start a CTU row: the saved context variables, and the slice QP as the
    /// QP predictor (§8.7.1).
    pub(super) fn start_row(&mut self, saved: &Saved, filters: &mut filter_syntax::Models) {
        self.contexts = saved.contexts.clone();
        self.trees = saved.trees.clone();
        self.coefficients = saved.coefficients.clone();
        self.skip_coefficients = saved.skip_coefficients.clone();
        self.intra_models = saved.intra.clone();
        self.inter_models = saved.inter.clone();
        self.transform_models = saved.transforms.clone();
        self.sbt_models = saved.sbt.clone();
        self.ibc_models = saved.ibc.clone();
        self.palette_models = saved.palette.clone();
        self.palette_predictor = saved.palette_predictor.clone();
        self.groups.restart(saved.groups);
        *filters = saved.filters.clone();
        self.cclm = saved.cclm;
        self.joint = saved.joint;
        self.constraint = saved.constraint;
        self.ref_index = saved.ref_index;
        // StatCoeff restarts with the contexts, not synchronised (VTM
        // `riceStatReset` at each tile and entropy-coding-sync row start).
        self.rice.reset_statistics(self.sequence.depth.bits());
    }
}
