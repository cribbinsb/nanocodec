//! AV1 quantiser settings. The arithmetic (quantise, and the normative
//! dequantise of AV1 §7.12.3) runs on nc-dsp's kernels; this module chooses
//! their parameters.

use nc_dsp::Av1Quantizer;

use crate::tables::{AC_QLOOKUP, DC_QLOOKUP};
use crate::transform::TxSize;

/// Magnitude at which an 8-bit dequantised coefficient saturates: AV1
/// §7.12.3 clips `Dequant` to `[-(1 << (7 + BitDepth)), (1 << (7 + BitDepth)) - 1]`.
const DEQUANT_SATURATION: i32 = 1 << (7 + 8);

/// Quantiser steps for one frame (no delta-q, no quantisation matrices).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantSteps {
    /// DC (coefficient 0) step.
    pub dc: i32,
    /// AC step.
    pub ac: i32,
}

impl QuantSteps {
    /// Steps for `base_q_idx` at 8-bit depth (AV1 §7.12.2 `dc_q`, `ac_q`).
    #[must_use]
    pub fn for_qindex(base_q_idx: u32) -> Self {
        let q = (base_q_idx as usize).min(DC_QLOOKUP.len() - 1);
        Self {
            dc: i32::from(DC_QLOOKUP[q]),
            ac: i32::from(AC_QLOOKUP[q]),
        }
    }

    /// Kernel parameters for a transform of size `tx` (its `dqDenom`), with
    /// the registered frame-kind rounding offset in units of 1/128 step.
    #[must_use]
    pub fn quantizer(self, tx: TxSize, rounding_q7: i32) -> Av1Quantizer {
        let log_scale = tx.dequant_shift();
        Av1Quantizer {
            dc_step: self.dc,
            ac_step: self.ac,
            rounding_q7,
            max_level: self.max_useful_level(log_scale),
            log_scale,
        }
    }

    /// The smallest level whose dequantised value saturates (AV1 §7.12.3)
    /// at the finer of the two steps: no larger level can change the
    /// reconstruction, so capping there never loses accuracy. A fixed cap
    /// would: at fine steps a full-swing (±255) residual needs levels up to
    /// `2^(15 + dqDenom) / step` (32768 for a 64x64 block at step 8).
    ///
    /// The cap also respects the other bitstream limits: `level * step`
    /// stays below the 24-bit wrap of §7.12.3 (at most `2^17` times the
    /// dc/ac step ratio, which is below 2), and the Golomb remainder stays
    /// within the 20-bit length decoders accept.
    fn max_useful_level(self, log_scale: u32) -> i32 {
        let finer_step = self.dc.min(self.ac);
        let saturating = DEQUANT_SATURATION << log_scale;
        (saturating + finer_step - 1) / finer_step
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_dsp::Dsp;

    #[test]
    fn quantize_then_dequantize_is_close() {
        let q = QuantSteps { dc: 40, ac: 50 }.quantizer(TxSize::Tx4x4, 48);
        let mut levels = [400, -230, 24, -26, 0, 51];
        Dsp::new().av1_quantize(&mut levels, &q);
        assert_eq!(levels, [10, -4, 0, 0, 0, 1]);
        let mut dq = [0; 6];
        Dsp::new().av1_dequantize(&levels, &q, &mut dq);
        assert_eq!(dq, [400, -200, 0, 0, 0, 50]);
    }

    #[test]
    fn steps_grow_with_qindex() {
        let low = QuantSteps::for_qindex(10);
        let high = QuantSteps::for_qindex(200);
        assert!(high.dc > low.dc && high.ac > low.ac);
    }
}
