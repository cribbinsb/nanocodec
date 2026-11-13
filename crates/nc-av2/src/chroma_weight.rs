//! How the encoder weighs chroma squared error against luma: in inter block
//! decisions (skip or coded, layouts), chroma trimming and the chroma
//! trellis λ.
//!
//! Two sources set the weight. The chroma step: a chroma qindex offset
//! finer than luma is undone if chroma error is priced at the luma λ, so
//! the weight grows by the squared luma/chroma step ratio, as x264's and
//! HM's chroma λ do. Or the content: under PSNR-YUV (6:1:1) the marginal
//! value of chroma SSE relative to luma SSE is about (4/6)·`MSE_Y / MSE_C`, so
//! chroma that is already much cleaner than luma (crowd, pan: 12..18 dB)
//! is cheap to improve, while chroma close to luma's error (bodycam) is
//! not. The adaptive weight follows the previous frame's plane errors.
//!
//! The same error ratio can move the chroma quantizer itself
//! (`nc_encoder::error_gap_offset`, measured by
//! `nc_encoder::plane_error_ratio`): a weight only reshuffles decisions at
//! a fixed chroma step, while the step sets what chroma costs. Pan clips (chroma
//! 17 dB cleaner than luma) gain from a finer chroma step, `bodycam_airport`
//! (4 dB) from a coarser one.

use crate::block_coder::pixel_step;
use crate::transform::PlaneQindex;

/// Bounds of the adaptive weight: the error ratio of a near-flat chroma
/// plane (or a lossless one) must not drive it to extremes.
const ADAPTIVE_MIN: f64 = 0.25;
const ADAPTIVE_MAX: f64 = 16.0;

/// The weight's controls (registered parameters).
#[derive(Debug, Clone, Copy)]
pub(crate) struct ChromaWeight {
    /// Multiplier on the weight (`av2.chroma_distortion_weight`).
    pub scale: f32,
    /// The weight follows the chroma step (`av2.chroma_rd_by_step`).
    pub by_step: bool,
    /// The weight follows the previous frame's luma/chroma error ratio
    /// times this gain, in place of the step (`av2.chroma_weight_adapt`; 0
    /// off), once a ratio has been measured.
    pub adapt_gain: f32,
}

impl ChromaWeight {
    /// The engine chroma policy's weight (`nc_encoder::chroma_policy`):
    /// the squared luma/chroma step ratio alone.
    pub(crate) const BY_STEP: Self = Self {
        scale: 1.0,
        by_step: true,
        adapt_gain: 0.0,
    };

    /// The weight in a frame quantized at `qindex` whose last measured
    /// luma/chroma error ratio is `error_ratio`.
    pub(crate) fn of(self, qindex: PlaneQindex, error_ratio: Option<f64>) -> f64 {
        let scale = f64::from(self.scale);
        match error_ratio.filter(|_| self.adapt_gain > 0.0) {
            Some(ratio) => {
                scale * (f64::from(self.adapt_gain) * ratio).clamp(ADAPTIVE_MIN, ADAPTIVE_MAX)
            }
            None if self.by_step => scale / chroma_step_ratio(qindex).powi(2),
            None => scale,
        }
    }
}

/// The chroma AC step over the luma AC step.
pub(crate) fn chroma_step_ratio(qindex: PlaneQindex) -> f64 {
    pixel_step(qindex.chroma.ac) / pixel_step(qindex.luma.ac)
}
