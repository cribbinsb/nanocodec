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
//! later burst.

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
