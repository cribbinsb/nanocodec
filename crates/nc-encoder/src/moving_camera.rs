//! Fixed-QP frame structure under camera motion: a coarser key and a
//! cadence that follows the scene (`rate.fixed_qp_moving_key`,
//! `rate.fixed_qp_moving_cadence`).
//!
//! A fixed-QP stream codes its key much finer than its inter frames
//! (`key_qp.rs`) and, in a backend's fixed-QP profile
//! (`CodecBackend::FIXED_QP_PARAMS`), the QP cadence at full strength
//! (`qp_cadence.rs`). Both pay where the frames between inherit the finer
//! frame: a fixed camera's background is copied for the whole period. A
//! moving camera (pan, driving, handheld) replaces or resamples the picture
//! within a second, and the extra bits are lost. Rate control already sizes
//! the first key this way from the start-up lookahead
//! (`RateController::observe_startup_motion`) and scales its cadence by the
//! still-block share; this is the same test, by the same threshold, at
//! fixed QP.
//!
//! The first key reads the start-up lookahead (none handed over: the scene
//! counts as still); later frames read the running camera motion of the
//! inter frames before them.
//!
//! The key's offset grows with the motion instead of switching at the
//! threshold: none at 1 sample per frame, all of it from the camera motion
//! at which the cadence's motion gate gives up (`rate.cadence_max_camera_motion`,
//! 2), and none on a fixed camera (`period_ledger::fixed_camera` of the
//! still-block share). Pool fixed-QP draw 2026-11-18 with the plain
//! threshold: a slow pan at 1.02 samples per frame lost 12.4 % with the
//! whole offset, and a still, dark scene (95 % unchanged blocks) whose
//! global-motion fit read 6 samples lost 6.4 %.

use crate::params::RateParams;
use crate::period_ledger::fixed_camera;
use crate::rate_control::STARTUP_CAMERA_MOTION;

/// Weight of the newest inter frame in the running camera motion: handheld
/// shake changes frame by frame, and the structure follows the scene, not
/// one frame.
const CAMERA_MOTION_SMOOTHING: f64 = 0.25;

/// The camera motion of a fixed-QP stream and what its structure does
/// under it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MovingCamera {
    /// `rate.fixed_qp_moving_key`: QP units coarser a key is coded under
    /// camera motion (0 = off).
    key_coarser: f64,
    /// `rate.fixed_qp_moving_cadence`: under camera motion the cadence is
    /// scaled by the still-block share.
    scene_cadence: bool,
    /// `rate.cadence_max_camera_motion`: camera motion from which the key
    /// takes its whole offset.
    full_motion: f64,
    /// Camera motion per frame (luma samples) and share of unchanged
    /// blocks: the start-up lookahead's, then smoothed over the inter
    /// frames; `None` while unmeasured.
    scene: Option<(f64, f64)>,
}

impl MovingCamera {
    /// `explicit_key_offset`: the caller set the key offset itself
    /// (`EncoderConfig::key_qp_offset`), which then holds as given.
    pub(crate) fn new(params: &RateParams, explicit_key_offset: bool) -> Self {
        Self {
            key_coarser: if explicit_key_offset {
                0.
            } else {
                f64::from(params.fixed_qp_moving_key)
            },
            scene_cadence: params.fixed_qp_moving_cadence >= 0.5,
            full_motion: f64::from(params.cadence_max_camera_motion),
            scene: None,
        }
    }

    /// Whether the structure follows camera motion at all (so a fixed-QP
    /// stream wants the start-up lookahead).
    pub(crate) fn enabled(&self) -> bool {
        self.key_coarser > 0. || self.scene_cadence
    }

    /// The start-up lookahead's peak camera displacement and share of
    /// unchanged blocks between the first two source frames.
    pub(crate) fn observe_startup(&mut self, peak_displacement: f64, still_share: f64) {
        self.scene = Some((peak_displacement, still_share));
    }

    /// An inter frame's camera motion and share of unchanged blocks
    /// against the frame before it.
    pub(crate) fn observe(&mut self, camera_motion: f64, still_share: f64) {
        let smooth = |old: f64, new: f64| old + CAMERA_MOTION_SMOOTHING * (new - old);
        self.scene = Some(self.scene.map_or((camera_motion, still_share), |old| {
            (smooth(old.0, camera_motion), smooth(old.1, still_share))
        }));
    }

    /// Whether the running camera motion is a moving camera's.
    pub(crate) fn moving(&self) -> bool {
        self.scene
            .is_some_and(|(motion, _)| motion >= STARTUP_CAMERA_MOTION)
    }

    /// QP units coarser the key being planned is coded.
    pub(crate) fn key_offset(&self) -> f64 {
        let Some((motion, still_share)) = self.scene else {
            return 0.;
        };
        let span = (self.full_motion - STARTUP_CAMERA_MOTION).max(f64::EPSILON);
        let moving = ((motion - STARTUP_CAMERA_MOTION) / span).clamp(0., 1.);
        self.key_coarser * moving * (1. - fixed_camera(still_share))
    }

    /// Whether the frame being planned scales its cadence by the
    /// still-block share, whatever `rate.cadence_static_scaling` says.
    pub(crate) fn cadence_follows_scene(&self) -> bool {
        self.scene_cadence && self.moving()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::ParamOverrides;

    fn camera(settings: &[(&str, f32)], explicit_key_offset: bool) -> MovingCamera {
        let mut overrides = ParamOverrides::default();
        for &(name, value) in settings {
            overrides.set(name, value);
        }
        MovingCamera::new(&RateParams::from_overrides(&overrides), explicit_key_offset)
    }

    /// The cadence gate's default limit: the whole offset from here.
    const FULL: f64 = 2.;

    #[test]
    fn default_is_six() {
        // The engine default (585a09daf): the key 6 QP units coarser under
        // full camera motion; the cadence half stays off.
        let mut c = camera(&[], false);
        assert!(c.enabled());
        c.observe_startup(FULL * 4., 0.);
        assert_eq!(c.key_offset(), 6.);
        assert!(!c.cadence_follows_scene());
    }

    #[test]
    fn off_when_zero() {
        let mut c = camera(&[("rate.fixed_qp_moving_key", 0.)], false);
        assert!(!c.enabled());
        c.observe_startup(FULL * 4., 0.);
        assert_eq!(c.key_offset(), 0.);
        assert!(!c.cadence_follows_scene());
    }

    #[test]
    fn the_structure_changes_only_under_measured_camera_motion() {
        let mut c = camera(
            &[
                ("rate.fixed_qp_moving_key", 6.),
                ("rate.fixed_qp_moving_cadence", 1.),
            ],
            false,
        );
        assert!(c.enabled());
        // Unmeasured (no lookahead handed over): the scene counts as still.
        assert_eq!(c.key_offset(), 0.);
        assert!(!c.cadence_follows_scene());
        c.observe_startup(STARTUP_CAMERA_MOTION / 2., 0.);
        assert_eq!(c.key_offset(), 0.);
        c.observe_startup(FULL, 0.);
        assert_eq!(c.key_offset(), 6.);
        assert!(c.cadence_follows_scene());
        // One still frame of a shaking camera does not end it.
        c.observe(0., 0.);
        assert!(c.cadence_follows_scene());
        // The camera stops: the running motion falls below the threshold.
        for _ in 0..16 {
            c.observe(0., 1.);
        }
        assert_eq!(c.key_offset(), 0.);
        assert!(!c.cadence_follows_scene());
    }

    #[test]
    fn the_key_offset_grows_with_the_motion_and_not_on_a_fixed_camera() {
        let mut c = camera(&[("rate.fixed_qp_moving_key", 6.)], false);
        // At the threshold itself nothing yet; halfway to the gate's limit, half.
        c.observe_startup(STARTUP_CAMERA_MOTION, 0.);
        assert_eq!(c.key_offset(), 0.);
        c.observe_startup(f64::midpoint(STARTUP_CAMERA_MOTION, FULL), 0.);
        assert!((c.key_offset() - 3.).abs() < 1e-9);
        c.observe_startup(FULL * 8., 0.5);
        assert_eq!(c.key_offset(), 6.);
        // A motion estimate over a picture of unchanged blocks is not a
        // moving camera.
        c.observe_startup(FULL * 8., 0.95);
        assert!((c.key_offset() - 6. * 0.25).abs() < 1e-9);
        c.observe_startup(FULL * 8., 1.);
        assert_eq!(c.key_offset(), 0.);
    }

    #[test]
    fn an_explicit_key_offset_holds_as_given() {
        let mut c = camera(&[("rate.fixed_qp_moving_key", 6.)], true);
        assert!(!c.enabled());
        c.observe_startup(FULL * 2., 0.);
        assert_eq!(c.key_offset(), 0.);
    }
}
