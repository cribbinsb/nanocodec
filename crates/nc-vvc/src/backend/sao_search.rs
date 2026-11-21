//! The encoder's SAO search settings (`vvc.sao_*`), re-read every picture so
//! that each may take a non-base or key picture value (`nonbase.vvc.sao_*`,
//! `key.vvc.sao_*`, `nc_encoder` `frame_role`) as the ALF search's do, and
//! the classes reused between search pictures.

use crate::sao::CtbTypes;

nc_encoder::param_group! {
    /// How SAO parameters are searched (per picture). The fields keep the
    /// registered names (`vvc.sao_*`), hence their common prefix.
    #[allow(clippy::struct_field_names)]
    pub struct SaoSearchParams("vvc") {
        sao_luma_lambda_scale: 1.0, [0.25, 16.0],
            "λ multiple for luma SAO decisions (HEVC's hevc.sao_luma_lambda_scale): higher applies fewer luma offsets (less smoothing)",
        sao_stats_row_step: 4.0, [1.0, 4.0],
            "sample every Nth row for SAO statistics; higher trades decision precision for speed",
        sao_search_interval: 1.0, [1.0, 8.0],
            "SAO types are searched on key pictures and every Nth P picture (by POC) and reused in between, offsets re-derived (HEVC's hevc.sao_search_interval; with vvc.sao_skip_unchanged, all-skip CTUs keep their last types): higher gathers fewer statistics, coarser SAO",
        // Off, as HEVC's hevc.sao_band (no measured gain for its statistics).
        sao_band: 0.0, [0.0, 1.0],
            "1 evaluates SAO band offset (position search, H.266 §8.8.4.3) beside the four edge classes on searched pictures (HEVC's hevc.sao_band)",
        sao_skip_unchanged: 0.0, [0.0, 1.0],
            "1 switches SAO off without statistics in P-frame CTUs whose CUs are all skipped (they copy already filtered samples; HEVC's hevc.sao_skip_unchanged): faster",
    }
}

/// How SAO parameters are searched, and the classes reused between search
/// pictures.
#[derive(Debug)]
pub(super) struct SaoSearch {
    /// `vvc.sao_stats_row_step`: statistics on every Nth row.
    pub row_step: usize,
    /// `vvc.sao_skip_unchanged`: all-skip P-frame CTUs off, no statistics.
    pub skip_unchanged: bool,
    /// `vvc.sao_luma_lambda_scale`.
    pub luma_lambda_scale: f32,
    /// `vvc.sao_search_interval`: types searched every Nth P picture.
    pub interval: u8,
    /// `vvc.sao_band`: band offset is evaluated.
    pub band: bool,
    /// The types of the last searched picture, and those of the picture
    /// being coded once it is a search picture (committed in
    /// `commit_frame`, so a recode never sees them).
    pub classes: Vec<CtbTypes>,
    pub pending: Option<Vec<CtbTypes>>,
}

impl SaoSearch {
    pub(super) fn new(params: &SaoSearchParams) -> Self {
        let mut search = Self {
            row_step: 1,
            skip_unchanged: false,
            luma_lambda_scale: 1.0,
            interval: 1,
            band: false,
            classes: Vec::new(),
            pending: None,
        };
        search.configure(params);
        search
    }

    /// This picture's settings; the reused classes carry over.
    pub(super) fn configure(&mut self, params: &SaoSearchParams) {
        self.row_step = params.sao_stats_row_step.round() as usize;
        self.skip_unchanged = params.sao_skip_unchanged >= 0.5;
        self.luma_lambda_scale = params.sao_luma_lambda_scale;
        self.interval = params.sao_search_interval.round() as u8;
        self.band = params.sao_band >= 0.5;
    }
}
