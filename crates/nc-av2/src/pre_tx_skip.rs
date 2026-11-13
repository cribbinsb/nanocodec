//! Luma pre-transform skip for inter blocks: AV1's `av1.pre_tx_skip` policy
//! (SVT light-PD1 `lpd1_should_perform_tx`, score in
//! [`nc_encoder::pre_tx_skip_score`], decided by [`nc_encoder::pre_tx_skips`]) on AV2's quantizer. A block whose luma
//! prediction error is clearly below what the quantizer would keep codes no
//! luma residual, so its luma transform trials (both layouts) are skipped;
//! chroma is still coded and the block still competes with `skip_txfm`.
//!
//! The signals, as AV1's Default (`pre_tx_skip_model = 1`):
//! - low energy: `100·SSE < energy·area·qindex` (SVT's qindex-linear
//!   gate, here on AV2's qindex: AV2 q100/150/200 have the steps of AV1
//!   q91/170/247, so the gate is AV1's within the operating range);
//! - neighbours: both above and left skipped their transforms. SVT's
//!   "both NEARESTMV" bonus has no AV2 counterpart (AV2 has no NEARESTMV)
//!   and counts as false; at AV1's default score 70 it never decides;
//! - rate–distortion: the residual quantizes to zero
//!   ([`nc_encoder::residual_quantises_to_zero`], AV1 `pre_tx_skip_model = 1`);
//!   AV1's fixed-guess RD signal (`pre_tx_skip_model = 0`, AV1 Fast) is not
//!   ported;
//! - protection band: from the AC step on AV1's scale
//!   ([`nc_encoder::QuantizerBand::from_av1_step`]).

use crate::transform::{self, Qindices};
use nc_common::PlaneView;
use nc_dsp::Dsp;
use nc_encoder::{QuantizerBand, ZeroLimits, pre_tx_skips, residual_quantises_to_zero};

/// AV2 Q3 steps per step on AV1's 8-bit `ac_qlookup` scale.
const Q3_PER_AV1_STEP: f32 = 8.0;
/// AV1-scale steps per orthonormal-DCT residual sample (AV1's
/// `AC_Q_PER_SAMPLE_STEP`).
const AV1_STEP_PER_SAMPLE: f32 = 8.0;

/// The largest orthonormal-DCT coefficients (in residual sample units)
/// that quantize to zero at `qindex` with `rounding`: a coefficient below
/// (1 − rounding) of a step does.
pub(crate) fn zero_limits(qindex: Qindices, rounding: transform::Rounding) -> ZeroLimits {
    let keep = |qindex: u8, q7: i32| {
        let step = transform::quantizer_step_q3(qindex) as f32 / Q3_PER_AV1_STEP;
        (1.0 - q7 as f32 / transform::ROUNDING_Q7_ONE as f32) * step / AV1_STEP_PER_SAMPLE
    };
    ZeroLimits {
        dc: keep(qindex.dc, rounding.dc_q7),
        ac: keep(qindex.ac, rounding.ac_q7),
    }
}

/// The registered controls (`av2.pre_tx_*`, as `av1.pre_tx_*`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PreTxSkip {
    /// Low-energy gate: `100·SSE < energy·area·qindex`.
    pub energy: f32,
    /// Score a block needs to skip (lower skips more).
    pub score: f32,
    /// AC energy allowance of the zero test, in squared zero limits.
    pub texture_scale: f32,
    /// Apply the luma-dominant (stronger) low-quantizer protection.
    pub luma_dominant: bool,
}

/// One inter block's luma, as the skip test sees it.
pub(crate) struct LumaBlock<'a> {
    pub side: usize,
    pub source: PlaneView<'a>,
    pub prediction: PlaneView<'a>,
    /// Squared error of the prediction.
    pub sse: u64,
    /// Whether the above and left blocks both skipped their transforms
    /// (`None` unless both are coded).
    pub both_neighbours_skip: Option<bool>,
}

impl PreTxSkip {
    /// Whether `block`, at the luma qindices `qindex` with the luma
    /// rounding offsets `rounding`, skips its luma transforms.
    pub fn skips_luma(
        &self,
        dsp: &Dsp,
        block: &LumaBlock<'_>,
        qindex: Qindices,
        rounding: transform::Rounding,
    ) -> bool {
        let area = (block.side * block.side) as f32;
        let low_energy = 100.0 * (block.sse as f32) < self.energy * area * f32::from(qindex.ac);
        let step = transform::quantizer_step_q3(qindex.ac) as f32 / Q3_PER_AV1_STEP;
        let quantizes_to_zero = || {
            residual_quantises_to_zero(
                dsp,
                block.side,
                &block.source,
                &block.prediction,
                zero_limits(qindex, rounding),
                self.texture_scale,
            )
        };
        let neighbours = block.both_neighbours_skip.map(|skip| (skip, false));
        pre_tx_skips(
            low_energy,
            neighbours,
            quantizes_to_zero,
            QuantizerBand::from_av1_step(step),
            self.luma_dominant,
            self.score,
        )
    }
}
