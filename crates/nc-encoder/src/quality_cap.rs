//! Quality ceiling of rate control: the finest quantizer step any frame
//! may use, and the temporal-noise diagnostic of the rate log.
//!
//! The target bitrate is a ceiling, not a rate to hit (owner decision, 2026-11-08:
//! "a max quality is appropriate, we don't want to waste bits encoding
//! noise"). Two mechanisms stop spending: this absolute floor,
//! `rate.max_quality_qp` (the H.264 QP of the same pixel-domain step, so one
//! setting serves every codec), and the measured refinement gain
//! (`refinement_gain.rs`). Frames they hold underspend; `RateController`
//! forgives that underspend beyond one buffer so it is never banked for a
//! later burst. A third, `rate.noise_floor_scale` ([`NoiseFloor`]), sets the
//! floor relative to the temporal noise of the still part of the picture,
//! so a noisy still camera stops coding its noise.

use nc_analysis::motion::MotionBlock;
use nc_analysis::noise::MEAN_ABS_DIFF_PER_SIGMA;
use nc_analysis::{Grid, Mv};

use crate::config::Codec;
use crate::params::RateParams;
use crate::qp_cadence::step_ratio;

/// H.264 quantizer step at QP 0 (H.264 §8.5.12.1: Qstep doubles every 6
/// QP from 0.625); HEVC uses the same scale.
const H264_STEP_AT_QP0: f64 = 0.625;
/// Rate-control step units of AV1 and VP9 (the AC quantizer lookup) are
/// eight times the pixel-domain step (AV1 §7.12.2: dequantized
/// coefficients carry three fractional bits).
const AC_LOOKUP_STEPS_PER_PIXEL_STEP: f64 = 8.;
/// `rate.max_quality_qp` value that disables the floor.
const DISABLED: f64 = 0.;
/// Samples in one 8x8 block of the analyzer's fine motion field.
const FINE_BLOCK_SAMPLES: f64 = 64.;
/// Least share of the frame's 8x8 blocks whose best match is the zero
/// vector for `residual_sigma` to estimate the noise from them.
const MIN_STILL_FRACTION: f64 = 0.25;

/// The quality ceiling of rate control: the absolute floor
/// (`rate.max_quality_qp`), the noise-relative one (`rate.noise_floor_scale`)
/// and whether cadence base frames may go below them
/// (`rate.quality_floor_cadence`).
#[derive(Debug, Clone)]
pub(crate) struct QualityFloor {
    /// [`max_quality_step`], in rate-control step units; 0 when off.
    absolute: f64,
    noise: NoiseFloor,
    keeps_cadence: bool,
}

impl QualityFloor {
    pub(crate) fn new(params: &RateParams, codec: Codec) -> Self {
        Self {
            absolute: max_quality_step(params, codec),
            noise: NoiseFloor::new(params, codec),
            keeps_cadence: params.quality_floor_cadence >= 0.5,
        }
    }

    /// Finest step any frame may use, in rate-control step units (0: no
    /// floor): the coarser of the two floors.
    pub(crate) fn step(&self) -> f64 {
        self.absolute.max(self.noise.step())
    }

    /// Whether the noise-relative floor needs every inter frame's sigma.
    pub(crate) fn wants_noise(&self) -> bool {
        self.noise.enabled()
    }

    /// Take the next frame's measured still-block sigma ([`NoiseFloor::observe`]).
    pub(crate) fn observe(&mut self, key: bool, frame_sigma: Option<f64>) {
        self.noise.observe(key, frame_sigma);
    }

    /// Whether cadence base frames keep their offset below the floor (the
    /// floor then bounds the inter level).
    pub(crate) fn keeps_cadence(&self) -> bool {
        self.keeps_cadence
    }
}

/// Finest step any frame may use (`rate.max_quality_qp`), in the
/// backend's rate-control step units; 0 when off.
pub(crate) fn max_quality_step(params: &RateParams, codec: Codec) -> f64 {
    let qp = f64::from(params.max_quality_qp);
    if qp > DISABLED {
        H264_STEP_AT_QP0 * step_ratio(qp) / pixel_step_per_unit(codec)
    } else {
        0.
    }
}

/// The noise-relative quality floor (`rate.noise_floor_scale`): the finest
/// step any frame may use is a multiple of the temporal noise sigma of the
/// still part of the picture.
///
/// What it skips: quantizer steps finer than the noise. On a still scene
/// from a live camera every frame differs from the last by sensor noise
/// alone; a step well below the noise sigma codes that noise into every
/// frame (and the QP cadence's base frames re-code it in a burst), so the
/// rate controller fills any target without the picture getting closer to
/// the scene (docs/analysis/static-noise-2026-11-14.md: AV1 Default at
/// 1 Mbps on a still webcam spends 65 kbit per frame at a flat clean-PSNR).
/// Why safe: the sigma is measured only on blocks whose best match is the
/// zero vector ([`residual_sigma`]), so detail and motion never read as
/// noise, and the floor lifts as soon as too little of the picture is still
/// (an inter frame without the estimate releases it at once). Keys, which
/// have no motion analysis, keep the estimate of the frames before them.
/// Light changes move the estimate within a few frames ([`SIGMA_SMOOTHING`]).
#[derive(Debug, Clone)]
pub(crate) struct NoiseFloor {
    /// Steps per unit of sigma; 0 when off.
    scale: f64,
    /// Pixel-domain step of one rate-control step unit of the codec.
    pixel_step_per_unit: f64,
    /// Running sigma of the still blocks; `None` before the first estimate
    /// and while the picture is not still enough to measure it.
    sigma: Option<f64>,
}

/// Weight of a frame's sigma in the running estimate: about four frames'
/// memory, as the analyzer smooths its sensor-noise estimate.
const SIGMA_SMOOTHING: f64 = 0.25;

impl NoiseFloor {
    pub(crate) fn new(params: &RateParams, codec: Codec) -> Self {
        Self {
            scale: f64::from(params.noise_floor_scale),
            pixel_step_per_unit: pixel_step_per_unit(codec),
            sigma: None,
        }
    }

    /// Whether the floor is on (it then needs every inter frame's sigma).
    pub(crate) fn enabled(&self) -> bool {
        self.scale > 0.
    }

    /// Take the next frame's measured sigma (`None`: too little of it is
    /// still). Keys carry no measurement and keep the estimate.
    pub(crate) fn observe(&mut self, key: bool, frame_sigma: Option<f64>) {
        if key {
            return;
        }
        self.sigma = frame_sigma.map(|s| self.sigma.map_or(s, |p| p + SIGMA_SMOOTHING * (s - p)));
    }

    /// Finest step the floor allows, in rate-control step units; 0 when off
    /// or without an estimate.
    pub(crate) fn step(&self) -> f64 {
        self.sigma
            .filter(|_| self.enabled())
            .map_or(0., |s| self.scale * s / self.pixel_step_per_unit)
    }
}

/// Temporal noise of a frame, for the rate log: the median 8x8 best-match
/// SAD per sample of the blocks whose best match is the zero vector, as
/// the sigma of the noise that would produce it. Unchanged content predicts
/// itself exactly, so what those blocks still differ by is noise (sensor
/// and source compression); moving content is left out. Unlike the
/// analyzer's sensor-noise estimate (flattest blocks only, which already
/// compressed sources leave clean), it also sees the grain and coding
/// flicker of textured areas (`static_otw_homes390` 1.7, `lowlight_pp22_004`
/// 0.9, `static_cam22` 0.1). `None` when too few blocks are still to tell.
pub(crate) fn residual_sigma(fine: &Grid<MotionBlock>) -> Option<f64> {
    let mut sads: Vec<u32> = fine
        .cells()
        .iter()
        .filter(|b| b.mv == Mv::ZERO)
        .map(|b| b.sad)
        .collect();
    if sads.is_empty() || (sads.len() as f64) < MIN_STILL_FRACTION * fine.cells().len() as f64 {
        return None;
    }
    let middle = sads.len() / 2;
    let (_, &mut median, _) = sads.select_nth_unstable(middle);
    Some(f64::from(median) / FINE_BLOCK_SAMPLES / f64::from(MEAN_ABS_DIFF_PER_SIGMA))
}

/// Pixel-domain step of one rate-control step unit of `codec`.
fn pixel_step_per_unit(codec: Codec) -> f64 {
    match codec {
        // `CodecBackend::rate_control_quantizer_step` default: 2^(q/6).
        Codec::H264 | Codec::Hevc | Codec::Vvc => H264_STEP_AT_QP0,
        Codec::Av1 | Codec::Vp9 => 1. / AC_LOOKUP_STEPS_PER_PIXEL_STEP,
        // nc-av2 already divides its Q3 step by eight.
        Codec::Av2 => 1.,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(qp: f32, codec: Codec) -> f64 {
        let mut params = crate::ParamOverrides::default();
        params.set("rate.max_quality_qp", qp);
        max_quality_step(&RateParams::from_overrides(&params), codec)
    }

    #[test]
    fn absolute_floor_is_the_h264_step_in_every_codec() {
        // H.264 QP 24: 2^(24/6) = 16 rate-control units, a 10-sample step.
        assert!((step(24., Codec::H264) - 16.).abs() < 1e-9);
        assert!((step(24., Codec::Av1) - 80.).abs() < 1e-9);
        assert_eq!(step(0., Codec::H264), 0.);
    }

    #[test]
    fn noise_floor_follows_the_still_sigma_and_lifts_without_it() {
        let mut params = crate::ParamOverrides::default();
        params.set("rate.noise_floor_scale", 4.);
        let mut floor = NoiseFloor::new(&RateParams::from_overrides(&params), Codec::H264);
        assert_eq!(floor.step(), 0.);
        floor.observe(false, Some(1.));
        // 4 x sigma 1 = a 4-sample step, in 0.625-sample H.264 units.
        assert!((floor.step() - 6.4).abs() < 1e-9);
        floor.observe(true, None);
        assert!(
            (floor.step() - 6.4).abs() < 1e-9,
            "a key keeps the estimate"
        );
        floor.observe(false, None);
        assert_eq!(floor.step(), 0., "a moving frame lifts the floor");
        let off = NoiseFloor::new(
            &RateParams::from_overrides(&crate::ParamOverrides::default()),
            Codec::H264,
        );
        assert!(!off.enabled());
    }

    #[test]
    fn residual_sigma_reads_only_still_blocks() {
        let still = MotionBlock {
            mv: Mv::ZERO,
            sad: 128,
        };
        let moving = MotionBlock {
            mv: Mv { x: 3, y: 0 },
            sad: 6400,
        };
        let mut cells = vec![still; 30];
        cells.extend(vec![moving; 70]);
        let sigma = residual_sigma(&Grid::from_cells(10, 10, cells)).unwrap();
        assert!((sigma - 2. / f64::from(MEAN_ABS_DIFF_PER_SIGMA)).abs() < 1e-9);
        let few = Grid::from_cells(10, 1, vec![moving; 10]);
        assert!(residual_sigma(&few).is_none());
    }
}
