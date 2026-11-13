//! Fixed-QP key-frame quantizer offset, shared by the QP-coded backends
//! (HEVC, VVC): how many QP finer than the configured (P-frame) QP a key
//! frame is coded when rate control is off. Rate control sizes keys itself
//! (`rate_control.rs`), so this shapes fixed-QP streams only.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grows_three_qp_per_factor_of_five_within_three_qp() {
        assert_eq!(fixed_qp_key_offset(60, 12.0), 12);
        assert_eq!(fixed_qp_key_offset(300, 12.0), 15);
        assert_eq!(fixed_qp_key_offset(12, 12.0), 9);
        assert_eq!(fixed_qp_key_offset(1, 12.0), 9);
        assert_eq!(fixed_qp_key_offset(100_000, 9.0), 12);
        assert_eq!(fixed_qp_key_offset(0, 9.0), 6);
    }
}
