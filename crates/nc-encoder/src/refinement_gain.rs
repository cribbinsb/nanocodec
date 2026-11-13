//! Online measure of whether finer inter frames still buy quality, for the
//! quality ceiling of rate control (`quality_cap.rs`).
//!
//! The target bitrate is a ceiling (owner decision, 2026-11-08: "we don't want to
//! waste bits encoding noise"). Whether bits still buy quality is measured,
//! not guessed from a noise proxy: the recent inter frames' luma PSNR is
//! regressed on their quantizer (in QP units of log2 step). The QP cadence
//! gives every cycle frames a few QP apart, so the slope is observable in
//! steady state. A flat slope (HEVC `static_otw_homes390`: inter frames stay
//! at ~33 dB whatever their QP) means refinement has stopped paying; a steep
//! one (`lowlight_pp22_004`: VMAF 70 -> 86 with bits) means it still does.

use std::collections::VecDeque;

use nc_common::{Picture, PlaneIndex};
use nc_dsp::{BlockSize, Dsp};

/// Inter frames in the regression window: eight 4-frame cadence cycles.
const WINDOW_FRAMES: usize = 32;
/// Fewest frames before the slope is trusted.
const MIN_FRAMES: usize = 12;
/// Least quantizer spread (standard deviation, QP units) for a usable
/// slope; without it (a flat cadence, moving content) no cap is applied.
const MIN_QP_SPREAD: f64 = 0.5;
/// Peak 8-bit sample value squared, for PSNR.
const PEAK_SQUARED: f64 = 255. * 255.;
/// Square block over which `luma_mse` sums squared errors.
const MSE_BLOCK: usize = 8;

/// Recent (QP units, luma PSNR) pairs of inter frames and the gain floor.
#[derive(Debug, Clone)]
pub(crate) struct RefinementGain {
    samples: VecDeque<(f64, f64)>,
    /// `rate.min_refinement_gain`: dB per QP below which refinement is
    /// considered not to pay; 0 disables the measure.
    min_gain: f64,
}

impl RefinementGain {
    pub(crate) fn new(min_gain: f64) -> Self {
        Self {
            samples: VecDeque::with_capacity(WINDOW_FRAMES),
            min_gain,
        }
    }

    /// Whether `observe` needs the frames' distortion at all.
    pub(crate) fn enabled(&self) -> bool {
        self.min_gain > 0.
    }

    /// Record an inter frame coded at `qp_units` (6 log2 of its step) with
    /// luma mean squared error `mse`.
    pub(crate) fn observe(&mut self, qp_units: f64, mse: f64) {
        if self.samples.len() == WINDOW_FRAMES {
            self.samples.pop_front();
        }
        self.samples
            .push_back((qp_units, 10. * (PEAK_SQUARED / mse.max(1e-3)).log10()));
    }

    /// Forget the window (a scene change or a new key: the slope belongs to
    /// the old content).
    pub(crate) fn reset(&mut self) {
        self.samples.clear();
    }

    /// Whether finer inter frames have stopped buying quality: the PSNR
    /// gained per QP finer is below `min_gain`, measured over enough frames
    /// and quantizer spread.
    pub(crate) fn exhausted(&self) -> bool {
        if !self.enabled() || self.samples.len() < MIN_FRAMES {
            return false;
        }
        let n = self.samples.len() as f64;
        let (mx, my) = self
            .samples
            .iter()
            .fold((0., 0.), |(a, b), &(x, y)| (a + x / n, b + y / n));
        let (sxx, sxy) = self.samples.iter().fold((0., 0.), |(a, b), &(x, y)| {
            (a + (x - mx) * (x - mx), b + (x - mx) * (y - my))
        });
        if sxx / n < MIN_QP_SPREAD * MIN_QP_SPREAD {
            return false;
        }
        // PSNR falls as QP rises: the gain per finer QP is the negated slope.
        -sxy / sxx < self.min_gain
    }
}

/// Luma mean squared error of `recon` against `source` over the
/// `width` x `height` area (whole 8x8 blocks; the coded area is a multiple
/// of 8).
pub(crate) fn luma_mse(
    dsp: &Dsp,
    source: &Picture,
    recon: &Picture,
    width: usize,
    height: usize,
) -> f64 {
    let size = BlockSize::new(MSE_BLOCK, MSE_BLOCK).expect("8x8 is a block size");
    let src = source.plane(PlaneIndex::Y).view();
    let rec = recon.plane(PlaneIndex::Y).view();
    let (cols, rows) = (width / MSE_BLOCK, height / MSE_BLOCK);
    let mut sse = 0u64;
    for by in 0..rows {
        for bx in 0..cols {
            let (x, y) = (bx * MSE_BLOCK, by * MSE_BLOCK);
            let s = src
                .sub_view(x, y, MSE_BLOCK, MSE_BLOCK)
                .expect("inside the source");
            let r = rec
                .sub_view(x, y, MSE_BLOCK, MSE_BLOCK)
                .expect("inside the recon");
            sse += u64::from(dsp.variance(size, &s, &r).sse);
        }
    }
    sse as f64 / ((cols * rows * MSE_BLOCK * MSE_BLOCK).max(1)) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_flat_slope_is_exhausted_and_a_steep_one_is_not() {
        let mut flat = RefinementGain::new(0.05);
        let mut steep = RefinementGain::new(0.05);
        for i in 0..32 {
            let qp = 30. + f64::from(i % 4);
            flat.observe(qp, 30.);
            // 0.25 dB per QP: MSE x 10^(0.025) per QP.
            steep.observe(qp, 30. * 10f64.powf(0.025 * (qp - 30.)));
        }
        assert!(flat.exhausted());
        assert!(!steep.exhausted());
    }

    #[test]
    fn no_verdict_without_quantizer_spread_or_when_off() {
        let mut same_qp = RefinementGain::new(0.05);
        let mut off = RefinementGain::new(0.);
        for i in 0..32 {
            same_qp.observe(30., 30.);
            off.observe(30. + f64::from(i % 4), 30.);
        }
        assert!(!same_qp.exhausted());
        assert!(!off.exhausted());
    }
}
