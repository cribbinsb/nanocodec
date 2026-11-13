//! Base / non-base effort overlay: cheaper settings on the frames nothing
//! long-lived depends on.
//!
//! SVT-AV1's real-time presets split effort over the flat-IPP cadence: every
//! 4th frame is a "base" frame (the one the QP cadence codes finer and the
//! following frames lean on) and gets the fuller light-PD1 level and the
//! CDEF search; the other three run a lighter level and reuse the CDEF
//! strengths (`enc_mode_config.c`, light-PD1 and CDEF levels; see
//! `docs/handoff/preset-toolsets.md`). A preset reproduces that with
//! `nonbase.<name> = value` settings: on non-base inter frames `<name>`
//! takes `value`, while base and key frames keep the ordinary value.
//!
//! Any parameter that is re-read per frame may take a non-base value: the
//! engine's decision, planner and analysis groups ([`FRAME_PARAM_SPECS`]),
//! and the groups a backend lists in
//! [`CodecBackend::FRAME_PARAM_SPECS`](crate::CodecBackend::FRAME_PARAM_SPECS),
//! which it re-reads in
//! [`CodecBackend::set_frame_params`](crate::CodecBackend::set_frame_params).
//! The parameters that define the cadence itself cannot.
//!
//! Base frames are the engine's existing cadence: the T0 frames with
//! temporal layers, else every `decision.base_period`th frame (the frame
//! `decision.base_lambda_scale` applies to). Key frames are always base.
//! With `decision.nonbase_disposable_only`, only frames no later frame
//! references (the top temporal layer) are non-base: cheap settings there
//! cannot propagate into other frames.
//! Without `nonbase.` settings nothing here runs, so streams are unchanged.

use nc_analysis::motion::SearchConfig;

use crate::params::{
    AnalysisParams, DecisionParams, ParamError, ParamOverrides, ParamSpec, PlannerParams,
    engine_specs,
};

/// Prefix of a non-base frame setting: `nonbase.<name> = value`.
pub const NONBASE_PREFIX: &str = "nonbase.";

/// The engine groups that are read per frame and so may take a non-base
/// value.
pub const FRAME_PARAM_SPECS: &[&[ParamSpec]] = &[
    DecisionParams::SPECS,
    PlannerParams::SPECS,
    AnalysisParams::SPECS,
];

/// Parameters that decide which frames are base: overlaying them per frame
/// would be circular.
const CADENCE_PARAMS: &[&str] = &[
    "decision.base_period",
    "decision.base_lambda_scale",
    "decision.nonbase_disposable_only",
];

/// A frame's place in the effort cadence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameRole {
    /// Key frames and the cadence's base frames: the ordinary settings.
    Base,
    /// The other inter frames: the `nonbase.` settings on top.
    NonBase,
}

/// The engine's per-frame parameters for one role.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RoleParams {
    pub decision: DecisionParams,
    pub planner: PlannerParams,
    pub search: SearchConfig,
}

impl RoleParams {
    fn from_overrides(overrides: &ParamOverrides) -> Self {
        Self {
            decision: DecisionParams::from_overrides(overrides),
            planner: PlannerParams::from_overrides(overrides),
            search: AnalysisParams::from_overrides(overrides).search_config(),
        }
    }
}

/// Both roles' settings, built once per encoder when a preset has
/// `nonbase.` settings.
#[derive(Debug, Clone)]
pub(crate) struct EffortOverlay {
    /// Engine parameters of base frames.
    pub base: RoleParams,
    /// Engine parameters of non-base frames.
    pub nonbase: RoleParams,
    /// Every override with the non-base values on top: what the backend
    /// re-reads on non-base frames.
    pub nonbase_overrides: ParamOverrides,
}

impl EffortOverlay {
    /// The overlay `params` asks for, or `None` without `nonbase.`
    /// settings. `params` holds both kinds of setting;
    /// `backend_frame_specs` are the backend's per-frame groups and
    /// `backend_fixed` the names in them that stay fixed for the stream.
    ///
    /// # Errors
    ///
    /// A non-base setting that names an unknown parameter, lies outside its
    /// range, or names a parameter that is not re-read per frame.
    pub fn from_overrides(
        params: &ParamOverrides,
        backend_frame_specs: &[&[ParamSpec]],
        backend_fixed: &[&str],
    ) -> Result<Option<Self>, ParamError> {
        let (nonbase, ordinary) = params.split_prefix(NONBASE_PREFIX);
        if nonbase.is_empty() {
            return Ok(None);
        }
        let prefixed = |error: ParamError| match error {
            ParamError::Unknown(name) => ParamError::Unknown(format!("{NONBASE_PREFIX}{name}")),
            other => other,
        };
        let every: Vec<&[ParamSpec]> = engine_specs()
            .into_iter()
            .chain(backend_frame_specs.iter().copied())
            .collect();
        nonbase.validate(&every).map_err(prefixed)?;
        let per_frame = |name: &str| {
            FRAME_PARAM_SPECS
                .iter()
                .chain(backend_frame_specs)
                .flat_map(|group| group.iter())
                .any(|spec| spec.name == name)
                && !CADENCE_PARAMS.contains(&name)
                && !backend_fixed.contains(&name)
        };
        if let Some(name) = nonbase.names().find(|name| !per_frame(name)) {
            return Err(ParamError::NotPerFrame(format!("{NONBASE_PREFIX}{name}")));
        }
        let nonbase_overrides = ordinary.overlaid_with(&nonbase);
        Ok(Some(Self {
            base: RoleParams::from_overrides(&ordinary),
            nonbase: RoleParams::from_overrides(&nonbase_overrides),
            nonbase_overrides,
        }))
    }

    /// The engine parameters of `role`.
    pub fn params(&self, role: FrameRole) -> &RoleParams {
        match role {
            FrameRole::Base => &self.base,
            FrameRole::NonBase => &self.nonbase,
        }
    }
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
    fn no_nonbase_settings_means_no_overlay() {
        let o = overrides(&[("decision.max_cheap_vectors", 2.0)]);
        assert!(
            EffortOverlay::from_overrides(&o, &[], &[])
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn nonbase_values_apply_to_nonbase_frames_only() {
        let o = overrides(&[
            ("decision.max_cheap_vectors", 8.0),
            ("nonbase.decision.max_cheap_vectors", 2.0),
            ("nonbase.analysis.window_half_width", 2.0),
        ]);
        let overlay = EffortOverlay::from_overrides(&o, &[], &[])
            .unwrap()
            .unwrap();
        assert_eq!(
            overlay.params(FrameRole::Base).decision.max_cheap_vectors,
            8.0
        );
        assert_eq!(
            overlay
                .params(FrameRole::NonBase)
                .decision
                .max_cheap_vectors,
            2.0
        );
        assert_eq!(overlay.params(FrameRole::Base).search.window_half_width, 4);
        assert_eq!(
            overlay.params(FrameRole::NonBase).search.window_half_width,
            2
        );
    }

    #[test]
    fn rejects_unknown_out_of_range_and_fixed_parameters() {
        let unknown = overrides(&[("nonbase.decision.nope", 1.0)]);
        assert_eq!(
            EffortOverlay::from_overrides(&unknown, &[], &[]).unwrap_err(),
            ParamError::Unknown("nonbase.decision.nope".into())
        );
        let range = overrides(&[("nonbase.decision.max_cheap_vectors", 99.0)]);
        assert!(matches!(
            EffortOverlay::from_overrides(&range, &[], &[]),
            Err(ParamError::OutOfRange { .. })
        ));
        let backend_fixed = overrides(&[("nonbase.decision.lambda_scale", 1.5)]);
        assert!(matches!(
            EffortOverlay::from_overrides(&backend_fixed, &[], &["decision.lambda_scale"]),
            Err(ParamError::NotPerFrame(_))
        ));
        for fixed in ["rate.cadence_base_boost", "decision.base_period"] {
            let o = overrides(&[(&format!("nonbase.{fixed}"), 1.0)]);
            assert!(matches!(
                EffortOverlay::from_overrides(&o, &[], &[]),
                Err(ParamError::NotPerFrame(_))
            ));
        }
    }
}
