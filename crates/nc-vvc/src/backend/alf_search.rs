//! The encoder's luma ALF search settings (`vvc.alf_*` other than the SPS
//! tool flag `vvc.alf`), re-read every picture so that each may take a
//! non-base picture value (`nonbase.vvc.alf_*`, `nc_encoder` `frame_role`),
//! and the filter bank kept for temporal reuse (`vvc.alf_bank_interval`).

use crate::alf::{Analysis, DecideOptions, DecideScratch, Decision};
use crate::alf_aps::SignalledFilters;
use crate::contexts::Contexts;
use nc_common::Picture;
use nc_dsp::Dsp;

nc_encoder::param_group! {
    /// The luma ALF decision's search (per picture). The fields keep the
    /// registered names (`vvc.alf_*`), hence their common prefix.
    #[allow(clippy::struct_field_names)]
    pub struct AlfSearchParams("vvc") {
        alf_aps: 1.0, [0.0, 1.0],
            "with vvc.alf, 1 lets a picture signal its own luma filters in an APS (Wiener filters of its statistics, classes merged by rate and distortion) when their bits pay: sharper, a little more search",
        alf_lambda_scale: 1.0, [0.25, 16.0],
            "with vvc.alf, multiple of the mode-decision λ for ALF decisions: higher filters fewer CTBs, only where the error falls most",
        alf_detail_weight: 0.0, [0.0, 2.0],
            "with vvc.alf_exact_decision, squared-error weight of the AC energy filtering removes from a CTB: higher keeps more detail (VMAF NEG), less PSNR gain",
        alf_exact_decision: 1.0, [0.0, 1.0],
            "with vvc.alf_aps, 1 decides the CTBs of a chosen bank on their exact squared-error change (filtering every candidate CTB) rather than the sampled statistics' estimate: never filters where it hurts, more filtering work",
        alf_fixed_sets: 1.0, [0.0, 1.0],
            "with vvc.alf, 0 never evaluates the 16 fixed filter sets (CTBs choose between off and the APS filters): faster decision, fewer filter choices",
        alf_stats_row_step: crate::alf::STATS_ROW_STEP as f32, [1.0, 8.0],
            "with vvc.alf, the encoder's filter statistics sample every Nth row: higher is faster, a coarser set choice",
        alf_bank_interval: 1.0, [1.0, 16.0],
            "with vvc.alf_aps and vvc.alf_exact_decision (no fixed sets), P pictures re-signal the luma filter bank derived up to N pictures earlier instead of gathering statistics and deriving their own (VVenC's temporal APS reuse; key pictures, and the picture after one that switched the reused bank off, derive; nonbase.vvc.alf_bank_interval above the base value derives on base pictures only): higher is faster, filters fit the picture less",
        alf_reuse_on_only: 0.0, [0.0, 1.0],
            "with vvc.alf_bank_interval above 1, 1 lets a picture that reuses a bank consider only the CTBs the previous picture filtered (the rest stay off without classification or filtering until the next derivation): faster, filters fewer CTBs",
        alf_skip_unchanged: 0.0, [0.0, 2.0],
            "with vvc.alf, 1 leaves ALF off without classification or statistics in P-frame CTUs whose CUs are all skipped (they copy already filtered samples), 2 also in those whose CUs are all inter without luma residual: faster encode and decode",
    }
}

/// The P-frame CTBs whose luma ALF stays off without classification or
/// statistics (`vvc.alf_skip_unchanged`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AlfSkip {
    None,
    /// CTBs whose CUs are all skipped: they copy samples their reference
    /// already filtered.
    Skipped,
    /// CTBs whose CUs are all inter without luma residual.
    LumaPredicted,
}

/// How ALF parameters are searched, the bank kept for reuse and the
/// decision's buffers.
#[derive(Debug)]
pub(super) struct AlfSearch {
    /// `vvc.alf_skip_unchanged`: which P-frame CTBs keep ALF off.
    pub(super) skip_unchanged: AlfSkip,
    /// `vvc.alf_stats_row_step`, `vvc.alf_aps` and the decision's options.
    pub(super) options: DecideOptions,
    /// `vvc.alf_lambda_scale`.
    lambda_scale: f32,
    /// `vvc.alf_bank_interval`: P pictures reuse a bank derived up to this
    /// many pictures earlier.
    bank_interval: usize,
    /// `vvc.alf_reuse_on_only`: pictures that reuse a bank consider only
    /// the CTBs the previous picture filtered.
    reuse_on_only: bool,
    /// The bank later P pictures may reuse, as of the last committed
    /// picture.
    held: Option<HeldBank>,
    /// What `held` becomes when the current picture is committed (recodes
    /// and dropped pictures leave it alone).
    pending: Option<HeldBank>,
    scratch: Box<DecideScratch>,
}

/// A picture's luma ALF bank kept for reuse (`vvc.alf_bank_interval`).
#[derive(Debug)]
struct HeldBank {
    bank: SignalledFilters,
    /// Pictures that have used it, the deriving one included.
    served: usize,
    /// The CTBs (raster order) the last picture using it filtered.
    on: Vec<bool>,
}

impl AlfSearch {
    pub(super) fn new(params: &AlfSearchParams) -> Self {
        let mut search = Self {
            skip_unchanged: AlfSkip::None,
            options: DecideOptions {
                row_step: 1,
                signalled: false,
                fixed: false,
                exact: false,
                detail_weight: 0.,
            },
            lambda_scale: 1.,
            bank_interval: 1,
            reuse_on_only: false,
            held: None,
            pending: None,
            scratch: Box::default(),
        };
        search.configure(params);
        search
    }

    /// Take the settings of `params` for the next picture; the held bank
    /// stays.
    pub(super) fn configure(&mut self, params: &AlfSearchParams) {
        self.skip_unchanged = match params.alf_skip_unchanged.round() as u8 {
            0 => AlfSkip::None,
            1 => AlfSkip::Skipped,
            _ => AlfSkip::LumaPredicted,
        };
        self.options = DecideOptions {
            row_step: params.alf_stats_row_step.round() as usize,
            signalled: params.alf_aps >= 0.5,
            fixed: params.alf_fixed_sets >= 0.5,
            exact: params.alf_exact_decision >= 0.5,
            detail_weight: params.alf_detail_weight,
        };
        self.lambda_scale = params.alf_lambda_scale;
        self.bank_interval = params.alf_bank_interval.round().max(1.) as usize;
        self.reuse_on_only = params.alf_reuse_on_only >= 0.5;
    }

    /// A new picture: forget what an earlier attempt would have committed.
    pub(super) fn begin_picture(&mut self) {
        self.pending = None;
    }

    /// The picture is final: its bank becomes the held one.
    pub(super) fn commit(&mut self) {
        self.held = self.pending.take();
    }

    /// Decide and apply the luma ALF of the deblocked, SAO-filtered
    /// `recon` against `source` (`ctu`-sample CTUs, decision λ `lambda`).
    /// `skipped` names the CTBs that stay off without analysis
    /// (`vvc.alf_skip_unchanged`, P pictures only).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn filter(
        &mut self,
        dsp: &Dsp,
        (source, recon): (&Picture, &mut Picture),
        ctu: usize,
        inter: bool,
        skipped: Option<&[bool]>,
        lambda: f32,
        contexts: &Contexts,
    ) -> Decision {
        // A P picture may reuse the held bank (`vvc.alf_bank_interval`),
        // then (`vvc.alf_reuse_on_only`) only in the CTBs the previous
        // picture filtered.
        let reuses = self.options.exact && self.options.signalled && !self.options.fixed;
        let reuse = self
            .held
            .as_ref()
            .filter(|held| reuses && inter && held.served < self.bank_interval);
        let previous_on = reuse.filter(|_| self.reuse_on_only).map(|held| &held.on);
        let is_skipped =
            |ctb: usize| skipped.is_some_and(|s| s[ctb]) || previous_on.is_some_and(|on| !on[ctb]);
        let analysis = Analysis::new(dsp, recon, ctu, |ctb| !is_skipped(ctb));
        let decision = analysis.decide(
            source,
            lambda * self.lambda_scale,
            contexts,
            is_skipped,
            self.options,
            reuse.map(|held| &held.bank),
            &mut self.scratch,
        );
        analysis.apply_decided(recon, &decision, self.options, &self.scratch);
        let served = match reuse {
            Some(held) => held.served + 1,
            None => 1,
        };
        // A derived bank starts a new run, a reused one continues its run
        // while some CTB still chose it, and a picture without a bank makes
        // the next derive.
        self.pending = decision.signalled.clone().map(|bank| HeldBank {
            bank,
            served,
            on: decision.ctbs.iter().map(Option::is_some).collect(),
        });
        decision
    }
}
