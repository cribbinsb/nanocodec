//! Per-superblock effort class: SVT-AV1's light / regular PD1 split.
//!
//! SVT-AV1 real time (M10 to M13, `enc_mode_config.c` light-PD1 levels,
//! `enc_dec_process.c` `lpd1_detector_skip_pd0`; BSD-3-Clause, ideas only)
//! does not spend one effort level on every superblock: most superblocks get
//! the light mode-decision pass, and the ones its detector finds hard get the
//! regular pass. Our presets choose effort per stream or per frame
//! (`nonbase.*`, `frame_role.rs`), so the expensive tools (transform-split
//! trials, trellis, live coefficient-rate walks, small-block planning) run on
//! every superblock or on none.
//!
//! Here effort is continuous: before an inter frame is coded, each of its
//! superblocks gets a **benefit proxy**, the motion search's best 16x16 SAD
//! per sample plus `effort.spread_weight` times the standard deviation of
//! its 8x8 SADs per sample, both in quantizer sample steps. A residual well
//! under the step mostly quantises to zero, so the tools that refine how it
//! is coded have little to win there; measuring against the step makes one
//! rule work at fixed QP and under rate control (SVT's shortcuts are gated
//! the same way, e.g. `100 · dist < th · area · qindex` in
//! `product_coding_loop.c`). A superblock is **regular** when it is
//!
//! - among the `effort.regular_fraction` of the frame's superblocks with the
//!   highest proxy (a work budget: the expensive tools run on a bounded,
//!   predictable share of each frame), or
//! - at or above `effort.regular_proxy` steps (so a frame that is mostly hard
//!   is not starved by the budget);
//!
//! and **light** otherwise. Key frames, spatial layers and, with
//! `effort.light_nonbase_only`, base frames (SVT runs regular PD1 on every
//! base frame) are all regular.
//!
//! A light superblock is coded with the `light.<name> = value` settings on
//! top of the frame's own: the engine's decision and planner groups, and the
//! parameters its backend lists in [`CodecBackend::SUPERBLOCK_PARAMS`] (the
//! ones it reads per block, so they may change between superblocks).
//! With `effort.regular_fraction` 1 (the default) no superblock is light
//! and streams are unchanged.
//!
//! [`CodecBackend::SUPERBLOCK_PARAMS`]: crate::CodecBackend::SUPERBLOCK_PARAMS

use nc_analysis::MotionAnalysis;

use crate::frame_role::FrameRole;
use crate::params::{
    DecisionParams, EffortParams, ParamError, ParamOverrides, ParamSpec, PlannerParams,
};
use crate::partition::Block;

/// Prefix of a light-superblock setting: `light.<name> = value`.
pub const LIGHT_PREFIX: &str = "light.";

/// Index of the 16x16 sum in [`nc_analysis::motion::SbStats::me_dist`]: the
/// smallest inter block the planner codes by default (`planner.inter_min_log2`
/// 4), so the best prediction a light superblock could still get.
const ME_DIST_16X16: usize = 1;
/// Samples per side of an 8x8 motion-search block.
const ME_8X8_SAMPLES: f32 = 64.0;

/// Decision parameters that only make sense per frame: the cadence and the
/// frame-level gates `Encoder::frame_params` reads before any superblock.
const FRAME_ONLY_PARAMS: &[&str] = &[
    "decision.base_period",
    "decision.base_lambda_scale",
    "decision.nonbase_disposable_only",
    "decision.noise_skip",
    "decision.noise_skip_chroma_ratio",
    "decision.intra_frame_skip_me_dist",
    "decision.subpel_walk_max_noise_sigma",
];

/// The engine's parameters of light superblocks for one frame role.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LightParams {
    pub decision: DecisionParams,
    pub planner: PlannerParams,
}

/// The `light.*` settings, parsed once per encoder.
#[derive(Debug, Clone)]
pub(crate) struct LightOverlay {
    /// The classification thresholds (`effort.*`).
    pub effort: EffortParams,
    /// Engine parameters of light superblocks on base and non-base frames.
    base: LightParams,
    nonbase: LightParams,
    /// What the backend re-reads for light superblocks, by role: every
    /// override of the role with the light values on top.
    base_overrides: ParamOverrides,
    nonbase_overrides: ParamOverrides,
}

impl LightOverlay {
    /// The overlay `params` asks for: `None` while `effort.regular_fraction`
    /// is 1. `ordinary` are the stream's settings without prefixed ones,
    /// `nonbase` the non-base frames' overrides when there are any.
    /// `backend_specs` are the backend's parameter groups and
    /// `backend_superblock` the names in them it can switch per superblock.
    ///
    /// # Errors
    ///
    /// A light setting that names an unknown parameter, lies outside its
    /// range, or names one that cannot change between superblocks.
    pub fn from_overrides(
        light: &ParamOverrides,
        ordinary: &ParamOverrides,
        nonbase: Option<&ParamOverrides>,
        (backend_specs, backend_superblock): (&[&[ParamSpec]], &[&str]),
    ) -> Result<Option<Self>, ParamError> {
        let prefixed = |error: ParamError| match error {
            ParamError::Unknown(name) => ParamError::Unknown(format!("{LIGHT_PREFIX}{name}")),
            other => other,
        };
        let every: Vec<&[ParamSpec]> = crate::params::engine_specs()
            .into_iter()
            .chain(backend_specs.iter().copied())
            .collect();
        light.validate(&every).map_err(prefixed)?;
        if let Some(name) = light
            .names()
            .find(|name| !switchable_per_superblock(name, backend_superblock))
        {
            return Err(ParamError::NotPerFrame(format!("{LIGHT_PREFIX}{name}")));
        }
        let effort = EffortParams::from_overrides(ordinary);
        if effort.regular_fraction >= 1.0 {
            return Ok(None);
        }
        let base_overrides = ordinary.overlaid_with(light);
        let nonbase_overrides = nonbase.unwrap_or(ordinary).overlaid_with(light);
        let engine = |overrides: &ParamOverrides| LightParams {
            decision: DecisionParams::from_overrides(overrides),
            planner: PlannerParams::from_overrides(overrides),
        };
        Ok(Some(Self {
            effort,
            base: engine(&base_overrides),
            nonbase: engine(&nonbase_overrides),
            base_overrides,
            nonbase_overrides,
        }))
    }

    /// The engine parameters of light superblocks in a frame of `role`.
    pub fn params(&self, role: FrameRole) -> &LightParams {
        match role {
            FrameRole::Base => &self.base,
            FrameRole::NonBase => &self.nonbase,
        }
    }

    /// The overrides the backend codes light superblocks of a `role` frame
    /// with.
    pub fn backend_overrides(&self, role: FrameRole) -> &ParamOverrides {
        match role {
            FrameRole::Base => &self.base_overrides,
            FrameRole::NonBase => &self.nonbase_overrides,
        }
    }

    /// Whether frames of `role` may have light superblocks.
    pub fn applies_to(&self, role: FrameRole) -> bool {
        role == FrameRole::NonBase || self.effort.light_nonbase_only < 0.5
    }
}

/// Whether parameter `name` may take a light value: the engine's decision
/// and planner groups (bar the frame-level gates) and the backend's
/// per-superblock parameters.
fn switchable_per_superblock(name: &str, backend_superblock: &[&str]) -> bool {
    let engine = [DecisionParams::SPECS, PlannerParams::SPECS]
        .iter()
        .flat_map(|group| group.iter())
        .any(|spec| spec.name == name);
    (engine && !FRAME_ONLY_PARAMS.contains(&name)) || backend_superblock.contains(&name)
}

/// The frame's motion statistics and quantizer, which class its
/// superblocks.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Classifier<'a> {
    pub motion: &'a MotionAnalysis,
    pub effort: &'a EffortParams,
    /// The frame quantizer's sample step.
    pub step: f32,
    /// Coded picture size, luma samples.
    pub width: usize,
    pub height: usize,
}

impl Classifier<'_> {
    /// Whether each superblock of size `1 << sb_log2`, in raster order, is
    /// light (module docs).
    pub fn light_map(&self, sb_log2: u8) -> Vec<bool> {
        let size = 1usize << sb_log2;
        let proxies: Vec<f32> = (0..self.height)
            .step_by(size)
            .flat_map(|y| (0..self.width).step_by(size).map(move |x| (x, y)))
            .map(|(x, y)| self.proxy(&Block::square(x, y, sb_log2)))
            .collect();
        light_classes(&proxies, self.effort)
    }

    /// The benefit proxy of superblock `sb`: the highest of the motion
    /// search's superblocks it covers (or the one that contains it).
    fn proxy(&self, sb: &Block) -> f32 {
        let size = nc_analysis::motion::SB;
        let first = (sb.x / size, sb.y / size);
        let last = (
            (sb.x + sb.width()).min(self.width).saturating_sub(1) / size,
            (sb.y + sb.height()).min(self.height).saturating_sub(1) / size,
        );
        (first.1..=last.1)
            .flat_map(|row| (first.0..=last.0).map(move |col| (col, row)))
            .map(|(col, row)| self.cell_proxy(col, row))
            .fold(0.0, f32::max)
    }

    /// The benefit proxy of the motion search's superblock at (`col`,
    /// `row`); unknown cells count as hard.
    fn cell_proxy(&self, col: usize, row: usize) -> f32 {
        let size = nc_analysis::motion::SB;
        let Some(stats) = self.motion.sb.try_get(col as isize, row as isize) else {
            return f32::INFINITY;
        };
        let (x, y) = (col * size, row * size);
        // The search's sums cover the visible blocks only.
        let samples = (size.min(self.width - x) * size.min(self.height - y)).max(1) as f32;
        let sad = stats.me_dist[ME_DIST_16X16] as f32 / samples;
        let spread = stats.cost_variance_8x8.sqrt() / ME_8X8_SAMPLES;
        (sad + self.effort.spread_weight * spread) / self.step.max(f32::MIN_POSITIVE)
    }
}

/// Whether each superblock with benefit proxy `proxies[i]` is light: it is
/// neither among the `effort.regular_fraction` highest nor at or above
/// `effort.regular_proxy`. Ties at the budget's lowest regular proxy all stay
/// regular, so the class does not depend on raster order.
fn light_classes(proxies: &[f32], effort: &EffortParams) -> Vec<bool> {
    let regular = (effort.regular_fraction * proxies.len() as f32).ceil() as usize;
    let budget_floor = if regular == 0 {
        f32::INFINITY
    } else if regular >= proxies.len() {
        f32::NEG_INFINITY
    } else {
        let mut sorted = proxies.to_vec();
        let (_, floor, _) = sorted.select_nth_unstable_by(proxies.len() - regular, f32::total_cmp);
        *floor
    };
    let absolute = effort.regular_proxy;
    let is_regular = |proxy: f32| proxy >= budget_floor || (absolute > 0.0 && proxy >= absolute);
    proxies.iter().map(|&proxy| !is_regular(proxy)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn overrides(settings: &[(&str, f32)]) -> ParamOverrides {
        let mut o = ParamOverrides::default();
        for &(name, value) in settings {
            o.set(name, value);
        }
        o
    }

    #[test]
    fn no_overlay_while_the_class_test_is_off() {
        let light = overrides(&[("decision.max_cheap_vectors", 1.0)]);
        let overlay =
            LightOverlay::from_overrides(&light, &ParamOverrides::default(), None, (&[], &[]));
        assert!(overlay.unwrap().is_none());
    }

    #[test]
    fn light_values_apply_on_top_of_each_role() {
        let light = overrides(&[("decision.max_cheap_vectors", 1.0)]);
        let ordinary = overrides(&[
            ("effort.regular_fraction", 0.5),
            ("planner.lambda_scale", 3.0),
        ]);
        let nonbase = ordinary.overlaid_with(&overrides(&[("planner.lambda_scale", 4.0)]));
        let overlay = LightOverlay::from_overrides(&light, &ordinary, Some(&nonbase), (&[], &[]))
            .unwrap()
            .unwrap();
        for (role, lambda_scale) in [(FrameRole::Base, 3.0), (FrameRole::NonBase, 4.0)] {
            let params = overlay.params(role);
            assert_eq!(params.decision.max_cheap_vectors, 1.0);
            assert_eq!(params.planner.lambda_scale, lambda_scale);
        }
    }

    #[test]
    fn frame_level_and_unlisted_backend_settings_are_refused() {
        let ordinary = overrides(&[("effort.regular_fraction", 0.5)]);
        for name in ["decision.base_period", "analysis.window_half_width"] {
            let light = overrides(&[(name, 2.0)]);
            let result = LightOverlay::from_overrides(&light, &ordinary, None, (&[], &[]));
            assert!(matches!(result, Err(ParamError::NotPerFrame(_))), "{name}");
        }
        let light = overrides(&[("decision.no_such_setting", 1.0)]);
        assert!(matches!(
            LightOverlay::from_overrides(&light, &ordinary, None, (&[], &[])),
            Err(ParamError::Unknown(name)) if name == "light.decision.no_such_setting"
        ));
    }

    fn effort(regular_fraction: f32, regular_proxy: f32) -> EffortParams {
        EffortParams {
            regular_fraction,
            regular_proxy,
            ..EffortParams::default()
        }
    }

    #[test]
    fn the_budget_keeps_the_highest_proxies_regular() {
        let proxies = [0.1, 3.0, 0.5, 2.0];
        assert_eq!(
            light_classes(&proxies, &effort(0.5, 0.0)),
            [true, false, true, false]
        );
        assert_eq!(light_classes(&proxies, &effort(0.0, 0.0)), [true; 4]);
        assert_eq!(light_classes(&proxies, &effort(1.0, 0.0)), [false; 4]);
        // A quarter of four superblocks rounds up to one.
        assert_eq!(
            light_classes(&proxies, &effort(0.2, 0.0)),
            [true, false, true, true]
        );
    }

    #[test]
    fn the_absolute_threshold_adds_hard_superblocks_and_ties_stay_regular() {
        let proxies = [0.1, 3.0, 0.5, 2.0];
        assert_eq!(
            light_classes(&proxies, &effort(0.25, 0.4)),
            [true, false, false, false]
        );
        assert_eq!(
            light_classes(&[1.0, 1.0, 0.0], &effort(0.3, 0.0)),
            [false, false, true]
        );
    }
}
