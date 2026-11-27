//! The staged intra decision's first cut (`decision.intra_stage_keep`,
//! `decision.intra_stage_band`): which of a leaf's intra candidates, ranked
//! by the cheap stage-0 estimate (prediction SATD + √λ · mode bits), go on
//! to the backend's transform-costed estimate (one luma transform,
//! quantised, its coefficients priced by the exact context walk).
//!
//! Why: that transform-costed estimate is where key frames spend their
//! time (AV1 medium prices all ten candidates of every key leaf with it;
//! the walk is 17-39 % of an AV1 key, docs/analysis/intra-cheap-2026-11-21.md
//! §2). SVT-AV1 ranks every candidate by SATD + λ · mode bits (MDS0) and
//! passes only a few within a cost band of the best to its coefficient-cost
//! stage (MDS1: `mds1_cand_base_th`, `nic_scaling`; idea, no code).
//!
//! What it skips: the transform-costed estimate of candidates outside the
//! first `keep` by the stage-0 estimate, or further than `band` per mille
//! above its best. Why safe: the transform-costed estimate still chooses
//! among the survivors, and the stage-0 best always survives. How often the
//! skipped candidates would have won, measured per codec on medium keys and
//! P-frame intra (4 quick clips, 2 quantizers; branch `staged-md-probe`):
//! AV1 keys keep 99.2 % of the transform-costed cost with the top 4 (loss
//! 0.80 %), 99.6 % with a 20 % band (4.7 of 10 candidates kept); HEVC keys
//! 0.53 % (top 4) and 0.35 % (20 % band, 4.1 of 8). Stream effect: in the
//! commit that wires each codec.
//!
//! Key frames and the intra blocks of inter frames take separate cuts
//! (`decision.inter_intra_stage_*` for the latter): their intra blocks are
//! of different kinds (a whole picture against a few repairs), and the
//! codecs already count their candidates separately (AV2
//! `av2.p_intra_rd_candidates`, VVC `vvc.inter_rd_candidates`). Measured:
//! VVC with a 30 % band on both was 1.3 % better on one-frame keys but
//! 0.3 % worse over 150 frames.
//!
//! Every parameter 0 (the default) leaves each codec's own candidate count
//! (`av1.keyframe_intra_rd_candidates`, `hevc.intra_rd_candidates`, ...) in
//! force and code exactly what it coded before.

use crate::params::{DecisionParams, ParamOverrides};

/// Per mille (`decision.intra_stage_band`, `decision.inter_intra_stage_band`).
const BAND_UNIT: f32 = 1000.0;

/// The first cut of the staged intra decision on key frames and on the
/// intra blocks of inter frames (module doc).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct IntraStage {
    key: IntraCut,
    inter: IntraCut,
}

impl IntraStage {
    /// The cuts a codec's encode is configured with (`decision.intra_stage_*`,
    /// `decision.inter_intra_stage_*`).
    #[must_use]
    pub fn from_overrides(overrides: &ParamOverrides) -> Self {
        Self::from_params(&DecisionParams::from_overrides(overrides))
    }

    /// The cuts of `params` (counts rounded).
    #[must_use]
    pub fn from_params(params: &DecisionParams) -> Self {
        let cut =
            |keep: f32, band: f32| IntraCut::new(keep.round().max(0.0) as usize, band.max(0.0));
        Self {
            key: cut(params.intra_stage_keep, params.intra_stage_band),
            inter: cut(params.inter_intra_stage_keep, params.inter_intra_stage_band),
        }
    }

    /// The cut of a key frame's intra leaves (`key`), else of an inter
    /// frame's intra blocks.
    #[must_use]
    pub fn cut(self, key: bool) -> IntraCut {
        if key { self.key } else { self.inter }
    }
}

/// One cut: how many of the ranked candidates, within what band of the
/// best, go on.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct IntraCut {
    /// At most this many candidates survive (0: no count limit).
    keep: usize,
    /// Only candidates within this many per mille above the stage-0 best
    /// survive (0: no band).
    band_permille: f32,
}

impl IntraCut {
    /// A cut keeping at most `keep` candidates (0: all) within
    /// `band_permille` ‰ above the best (0: no band); for codecs that
    /// register their own (VVC `vvc.rd_proxy_margin`).
    #[must_use]
    pub fn new(keep: usize, band_permille: f32) -> Self {
        Self {
            keep,
            band_permille,
        }
    }

    /// The count limit (0: none).
    #[must_use]
    pub fn keep(self) -> usize {
        self.keep
    }

    /// The band, per mille above the best (0: none).
    #[must_use]
    pub fn band_permille(self) -> f32 {
        self.band_permille
    }

    /// Whether either parameter is set: the backend then ranks all its
    /// candidates by the stage-0 estimate and passes [`Self::survivors`] on,
    /// in place of its own candidate count.
    #[must_use]
    pub fn is_active(self) -> bool {
        self.keep > 0 || self.band_permille > 0.0
    }

    /// The survivors of `ranked` (candidate, stage-0 cost), cheapest first:
    /// the first `keep`, and of those the ones within the band of the best.
    /// Ties keep input order. Never empty for a nonempty input.
    #[must_use]
    pub fn survivors<T>(self, mut ranked: Vec<(T, f32)>) -> Vec<(T, f32)> {
        ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
        if self.keep > 0 {
            ranked.truncate(self.keep);
        }
        if self.band_permille > 0.0
            && let Some(&(_, best)) = ranked.first()
        {
            let limit = best * (1.0 + self.band_permille / BAND_UNIT);
            ranked.retain(|&(_, cost)| cost <= limit);
        }
        ranked
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_the_count_then_the_band() {
        let ranked = vec![('a', 10.0), ('b', 13.0), ('c', 11.0), ('d', 30.0)];
        let names = |cut: IntraCut| -> String {
            cut.survivors(ranked.clone())
                .into_iter()
                .map(|(m, _)| m)
                .collect()
        };
        assert_eq!(names(IntraCut::new(0, 0.0)), "acbd");
        assert_eq!(names(IntraCut::new(2, 0.0)), "ac");
        assert_eq!(names(IntraCut::new(0, 200.0)), "ac");
        assert_eq!(names(IntraCut::new(0, 300.0)), "acb");
        assert_eq!(names(IntraCut::new(1, 300.0)), "a");
    }

    #[test]
    fn ties_keep_input_order_and_off_is_inactive() {
        let ranked = vec![('x', 5.0), ('y', 5.0)];
        let kept = IntraCut::new(1, 0.0).survivors(ranked);
        assert_eq!(kept, vec![('x', 5.0)]);
        assert!(!IntraCut::default().is_active());
        assert!(IntraCut::new(0, 1.0).is_active());
    }
}
