//! Encoder-only coefficient rate-distortion trimming ("trellis-lite"): AV2's
//! use of the shared trailing rule (`nc_encoder::trim_trailing_ones`, whose
//! static rate model this was), with its exact coefficient-domain distortion.
//!
//! After deadzone quantization, a transform often ends in isolated ±1 levels:
//! each costs its sign, its last-position base symbol, the zero symbols
//! between it and the previous nonzero level and the longer end-of-block
//! code, but removes little distortion. [`Trim::apply`] drops such trailing
//! levels, last first, while the distortion they would add costs less than
//! λ times an estimate of the bits they save. The estimate is a static model
//! (no entropy contexts), so the block-level rate-distortion choices that
//! follow, which price the real syntax, remain the final judge: a transform
//! trimmed to nothing still has to beat the skipped block.
//!
//! Distortion is measured exactly in coefficient units and converted to
//! squared pixel error with [`transform::coefficient_error_scale`].
//!
//! Measured (640×360, 60 frames, q100..210, PSNR-YUV BD vs no trimming):
//! pan −3.18%, crowd −0.48%, at about +3..8% CPU. Chroma trimming at the
//! luma rate scale beats none (crowd, Y BD −1.04% vs −0.17%). `J_real`
//! proxy-6: −3.4% (NEG −4.5%, PSNR −2.1%, key pulse −0.41 dB) but VMAF v1
//! +1.7%, +9.3% on `crowd_pp22` and +5.2% on `pan_jaad_0223` when trimming
//! every frame (`av2.coeff_trim` = 1). The key frame is the culprit: trimming
//! P frames only (the default) screens proxy-6 `J_real` −2.7% with VMAF v1
//! −0.9% (worst clip +1.7%).

use crate::coeff;
use crate::transform::{self, Qindices};
use crate::tx_size::TxSize;

/// The tuning of trimming (encoder parameters).
#[derive(Debug, Clone, Copy)]
pub(crate) struct TrimOptions {
    /// Scale of the luma rate estimate (one is the model above).
    pub luma_rate_scale: f64,
    /// Scale of the chroma rate estimate (zero: chroma is never trimmed).
    pub chroma_rate_scale: f64,
    /// The shared rules (`quant.rd_trim`) in place of the trailing rule
    /// alone, or `None` for that rule.
    pub rules: Option<nc_encoder::RateTrim>,
}

/// The trimming model of one frame.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Trim {
    /// Squared pixel error per bit.
    pub lambda: f64,
    /// Chroma squared error in luma units: divides the chroma λ.
    pub chroma_weight: f64,
    pub options: TrimOptions,
}

impl Trim {
    /// Drop trailing ±1 `levels` of a luma or chroma transform of size `tx`
    /// (quantized at `qindex` from `coefficients`, both raster at the coded
    /// side, end of block `eob`) while that lowers the estimated
    /// rate-distortion cost. Returns the resulting end of block.
    pub(crate) fn apply(
        self,
        luma: bool,
        tx: TxSize,
        qindex: Qindices,
        coefficients: &[i32],
        (levels, eob): (&mut [i32], usize),
    ) -> usize {
        let (rate_scale, lambda) = if luma {
            (self.options.luma_rate_scale, self.lambda)
        } else {
            (
                self.options.chroma_rate_scale,
                self.lambda / self.chroma_weight,
            )
        };
        if rate_scale <= 0.0 {
            return eob;
        }
        let scan = coeff::scan(tx);
        let error_scale = transform::coefficient_error_scale(tx);
        let lambda = lambda * rate_scale;
        if let Some(rules) = &self.options.rules {
            return nc_encoder::trim_levels(rules, levels, scan, eob, lambda, |pos, level| {
                let magnitude = i64::from(coefficients[pos].unsigned_abs());
                let dequantized =
                    i64::from(transform::dequantized_magnitude(tx, qindex.at(pos), level));
                nc_encoder::dequantized_removal_sse(magnitude, dequantized, error_scale)
            });
        }
        nc_encoder::trim_trailing_ones(levels, scan, eob, lambda, |pos| {
            let magnitude = i64::from(coefficients[pos].unsigned_abs());
            let one = i64::from(transform::level_one_magnitude(tx, qindex.at(pos)));
            nc_encoder::dequantized_removal_sse(magnitude, one, error_scale)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const QINDEX: u8 = 120;

    fn trim() -> Trim {
        let step = f64::from(transform::quantizer_step_q3(QINDEX)) / 64.0;
        Trim {
            lambda: 0.135 * step * step,
            chroma_weight: 1.0,
            options: TrimOptions {
                luma_rate_scale: 1.0,
                chroma_rate_scale: 0.0,
                rules: None,
            },
        }
    }

    #[test]
    fn drops_isolated_trailing_ones_and_keeps_strong_levels() {
        let tx = TxSize::Tx8;
        let one = transform::level_one_magnitude(tx, QINDEX);
        let (mut coefficients, mut levels) = ([0; 64], [0; 64]);
        // DC level three, then a weak isolated one at the end of the scan.
        coefficients[0] = 3 * one;
        levels[0] = 3;
        coefficients[63] = one * 3 / 4;
        levels[63] = 1;
        trim().apply(true, tx, QINDEX.into(), &coefficients, (&mut levels, 64));
        assert_eq!((levels[0], levels[63]), (3, 0));
        // Chroma is not trimmed at a zero rate scale.
        levels[63] = 1;
        trim().apply(false, tx, QINDEX.into(), &coefficients, (&mut levels, 64));
        assert_eq!(levels[63], 1);
        // A trailing level above one stops the trimming.
        levels[63] = -2;
        coefficients[63] = -2 * one;
        trim().apply(true, tx, QINDEX.into(), &coefficients, (&mut levels, 64));
        assert_eq!(levels[63], -2);
    }
}
