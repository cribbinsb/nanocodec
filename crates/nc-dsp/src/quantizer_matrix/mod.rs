//! The predefined quantizer matrices of AV1 (§7.12.3 `Quantizer_Matrix`)
//! and AV2, which share them: per matrix level (0..15; 15 is flat) and
//! plane type, a weight in 1/32 units for every coefficient of each
//! transform shape with sides 4..=32 (64-sample sides use the 32 tables),
//! in the transposed (column-major) coefficient order the AV1 and AV2
//! decoders store, and the weighting of a dequantizer step.
//!
//! One table for both codecs: the AV2 set (generated from dav2d, see
//! [`tables`]) holds every AV1 shape, and its 420 AV1-shape tables were
//! checked equal to dav1d 1.5.4 `src/qm.c` (2026-11-27).
mod tables;

use tables::{MATRICES, MAX_SIDE_LOG2, OFFSETS};

/// Weights are in units of 1/32 (32 is flat): `AOM_QM_BITS`.
const WEIGHT_SHIFT: u32 = 5;
const WEIGHT_ROUNDING: u32 = 1 << (WEIGHT_SHIFT - 1);
/// log2 of the 4x4 coefficients of the smallest shape.
const UNIT_AREA_LOG2: u32 = 4;

/// The weights of a `level` matrix of a luma or chroma transform whose
/// sides are `1 << w_log2` and `1 << h_log2` 4-sample units (sides above
/// 32 samples read the 32 tables); `None` for the flat level 15.
#[must_use]
pub fn matrix(level: u8, chroma: bool, (w_log2, h_log2): (u8, u8)) -> Option<&'static [u8]> {
    let tables = MATRICES.get(usize::from(level))?;
    let (w, h) = (w_log2.min(MAX_SIDE_LOG2), h_log2.min(MAX_SIDE_LOG2));
    let offset = usize::from(OFFSETS[usize::from(w)][usize::from(h)]);
    let entries = 1 << (UNIT_AREA_LOG2 + u32::from(w + h));
    Some(&tables[usize::from(chroma)][offset..offset + entries])
}

/// A dequantizer step weighted by a matrix entry (AV1 §7.12.3:
/// `Round2(q * Quantizer_Matrix[..], AOM_QM_BITS)`; dav1d / dav2d
/// `(dq * qm + 16) >> 5`).
#[inline]
#[must_use]
pub fn weighted_step(step: u32, weight: u8) -> u32 {
    (step * u32::from(weight) + WEIGHT_ROUNDING) >> WEIGHT_SHIFT
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_level_has_no_matrix_and_sizes_match_shapes() {
        assert!(matrix(15, false, (0, 0)).is_none());
        assert_eq!(matrix(0, false, (0, 0)).unwrap().len(), 16);
        assert_eq!(matrix(3, true, (3, 1)).unwrap().len(), 32 * 8);
        // 64-sample sides read the 32 tables.
        assert_eq!(matrix(3, true, (4, 4)), matrix(3, true, (3, 3)));
        // AV1 §7.12.3 `Quantizer_Matrix[0][0]` of TX_4X4.
        assert_eq!(
            matrix(0, false, (0, 0)).unwrap(),
            &[
                32, 43, 73, 97, 43, 67, 94, 110, 73, 94, 137, 150, 97, 110, 150, 200
            ]
        );
    }

    #[test]
    fn weighting_rounds_to_nearest() {
        assert_eq!(weighted_step(100, 32), 100);
        assert_eq!(weighted_step(100, 43), 134);
        assert_eq!(weighted_step(1, 16), 1);
    }
}
