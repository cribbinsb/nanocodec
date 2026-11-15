//! Fixed-QP key-frame quantizer, shared by the backends: how much finer
//! than the configured (P-frame) quantizer a key frame is coded when rate
//! control is off. Rate control sizes keys itself (`rate_control.rs`), so
//! this shapes fixed-QP streams only.
//!
//! - [`fixed_qp_key_offset`]: the QP-coded backends' (HEVC, VVC) offset,
//!   derived from the intra period.
//! - [`scaled_quantizer`]: the qindex-coded backends' (AV1, VP9, AV2) key
//!   quantizer, a fraction of the frame's step (`keyframe_step_permille`).
//! - [`offset_key_quantizer`]: the explicit override
//!   (`EncoderConfig::key_qp_offset`, `--key-qp-offset`) every backend
//!   applies the same way.
//!
//! A key frame is referenced, directly or through the chain, by every frame
//! up to the next key, so the longer the intra period the more its quality
//! pays off (SVT-AV1 CQP scales the key quantizer step by 0.24, about
//! −12 QP). The offset grows by 3 QP per factor of 5 in the period and stays
//! within 3 QP of its value at the reference period.

/// Intra period (frames) at which a backend states its offset: 2 s at 30 fps.
pub const KEY_OFFSET_REFERENCE_PERIOD: f64 = 60.0;
/// QP added per factor of [`KEY_OFFSET_PERIOD_RATIO`] in the intra period.
const QP_PER_PERIOD_STEP: f64 = 3.0;
/// Period ratio over which the offset grows by [`QP_PER_PERIOD_STEP`].
const KEY_OFFSET_PERIOD_RATIO: f64 = 5.0;
/// Largest departure from the reference-period offset (the fitted range).
const MAX_PERIOD_ADJUSTMENT: f64 = 3.0;

/// The key-frame QP offset for an intra period of `period` frames, for a
/// backend whose offset at [`KEY_OFFSET_REFERENCE_PERIOD`] is `at_reference`.
#[must_use]
pub fn fixed_qp_key_offset(period: u32, at_reference: f64) -> i32 {
    let steps = (f64::from(period.max(1)) / KEY_OFFSET_REFERENCE_PERIOD).ln()
        / KEY_OFFSET_PERIOD_RATIO.ln();
    let adjustment =
        (QP_PER_PERIOD_STEP * steps).clamp(-MAX_PERIOD_ADJUSTMENT, MAX_PERIOD_ADJUSTMENT);
    (at_reference + adjustment).round() as i32
}

/// A fixed-QP key frame's quantizer: `base` (the planned inter quantizer)
/// made `offset` steps of the codec's own quantizer scale finer (QP for
/// H.264/HEVC/VVC, `base_q_idx` for AV1/VP9/AV2; negative is coarser),
/// within the codec's `min..=max`.
#[must_use]
pub fn offset_key_quantizer(base: u32, offset: i32, min: u32, max: u32) -> u32 {
    (i64::from(base) - i64::from(offset)).clamp(i64::from(min), i64::from(max)) as u32
}

/// The first quantizer in `range` whose step (`step`) is at least `scale`
/// times that of `quantizer` (libvpx `vp9_compute_qdelta`, SVT-AV1 /
/// libaom `compute_qdelta`), or `None` when none is. Callers clamp it to
/// their lossy range.
#[must_use]
pub fn scaled_quantizer(
    range: std::ops::RangeInclusive<u32>,
    step: impl Fn(u32) -> f64,
    quantizer: u32,
    scale: f64,
) -> Option<u32> {
    let target = step(quantizer) * scale;
    range.into_iter().find(|&q| step(q) >= target)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaled_quantizer_is_the_first_step_reaching_the_target() {
        let steps = [4.0, 8.0, 10.0, 16.0, 20.0];
        let step = |q: u32| steps[q as usize];
        assert_eq!(scaled_quantizer(0..=4, step, 4, 0.5), Some(2));
        assert_eq!(scaled_quantizer(0..=4, step, 3, 0.25), Some(0));
        assert_eq!(scaled_quantizer(1..=3, step, 3, 0.25), Some(1));
        assert_eq!(scaled_quantizer(0..=4, step, 4, 2.0), None);
    }

    #[test]
    fn grows_three_qp_per_factor_of_five_within_three_qp() {
        assert_eq!(fixed_qp_key_offset(60, 12.0), 12);
        assert_eq!(fixed_qp_key_offset(300, 12.0), 15);
        assert_eq!(fixed_qp_key_offset(12, 12.0), 9);
        assert_eq!(fixed_qp_key_offset(1, 12.0), 9);
        assert_eq!(fixed_qp_key_offset(100_000, 9.0), 12);
        assert_eq!(fixed_qp_key_offset(0, 9.0), 6);
    }

    #[test]
    fn offset_key_quantizer_stays_in_range() {
        assert_eq!(offset_key_quantizer(30, 8, 0, 51), 22);
        assert_eq!(offset_key_quantizer(5, 8, 1, 51), 1);
        assert_eq!(offset_key_quantizer(48, -6, 0, 51), 51);
        assert_eq!(offset_key_quantizer(120, 0, 1, 255), 120);
    }
}
