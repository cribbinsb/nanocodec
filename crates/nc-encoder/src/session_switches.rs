//! The session switches that stand for heuristic parameters: intra refresh,
//! long-term references and the fixed-QP profile. Every front end (the CLI's
//! `--intra-refresh`, `--ltr-interval` and `--qp` without `--bitrate`; the C
//! API's `nc_encoder_config` fields) maps them through this one place, so the
//! same settings build the same encoder configuration on every path.

use crate::params::ParamOverrides;

/// The refresh sweep's period parameter (`refresh.rs`).
const REFRESH_PERIOD_PARAM: &str = "refresh.period";
/// The long-term reference interval parameter (`ltr.rs`).
const LTR_INTERVAL_PARAM: &str = "ltr.interval";
/// The switch of the codec's fixed-QP profile (`CodecBackend::FIXED_QP_PARAMS`).
const FIXED_QP_PROFILE_PARAM: &str = "rate.fixed_qp_profile";

/// A front end's session switches, applied over its preset and explicit
/// parameter overrides by [`SessionSwitches::apply`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SessionSwitches {
    /// Frames per intra-refresh sweep, when the caller set one (it wins over
    /// any `refresh.period` override).
    pub intra_refresh_period: Option<u32>,
    /// Frames between long-term reference marks, when the caller set one.
    pub ltr_interval: Option<u32>,
    /// The stream is coded at a fixed quantizer (no rate control at start).
    pub fixed_qp: bool,
}

impl SessionSwitches {
    /// Lay the switches over `params`. A fixed-quantizer stream gets the
    /// codec's fixed-QP profile unless the caller or its preset set
    /// `rate.fixed_qp_profile` itself.
    pub fn apply(&self, params: &mut ParamOverrides) {
        if let Some(period) = self.intra_refresh_period {
            params.set(REFRESH_PERIOD_PARAM, period as f32);
        }
        if let Some(interval) = self.ltr_interval {
            params.set(LTR_INTERVAL_PARAM, interval as f32);
        }
        if self.fixed_qp && !params.contains(FIXED_QP_PROFILE_PARAM) {
            params.set(FIXED_QP_PROFILE_PARAM, 1.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_qp_sets_the_profile_unless_pinned() {
        let fixed = SessionSwitches {
            fixed_qp: true,
            ..SessionSwitches::default()
        };
        let mut params = ParamOverrides::default();
        fixed.apply(&mut params);
        assert_eq!(params.get(FIXED_QP_PROFILE_PARAM), Some(1.0));

        let mut pinned = ParamOverrides::default();
        pinned.set(FIXED_QP_PROFILE_PARAM, 0.0);
        fixed.apply(&mut pinned);
        assert_eq!(pinned.get(FIXED_QP_PROFILE_PARAM), Some(0.0));

        let mut rate_controlled = ParamOverrides::default();
        SessionSwitches::default().apply(&mut rate_controlled);
        assert!(!rate_controlled.contains(FIXED_QP_PROFILE_PARAM));
    }

    #[test]
    fn refresh_and_ltr_win_over_overrides() {
        let mut params = ParamOverrides::default();
        params.set(REFRESH_PERIOD_PARAM, 10.0);
        SessionSwitches {
            intra_refresh_period: Some(30),
            ltr_interval: Some(15),
            fixed_qp: false,
        }
        .apply(&mut params);
        assert_eq!(params.get(REFRESH_PERIOD_PARAM), Some(30.0));
        assert_eq!(params.get(LTR_INTERVAL_PARAM), Some(15.0));
    }
}
