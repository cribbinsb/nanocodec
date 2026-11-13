//! Edge thresholds of AV2 deblocking from the quantizer index (AVM v1.0.0
//! `df_quant_from_qindex`, `df_side_from_qindex` and the `side_thresholds`
//! table in `av2/common/av2_loopfilter.c`; BSD-3-Clause-Clear, see
//! `THIRD_PARTY.md`). Eight-bit video only.

use crate::transform::quantizer_step_q3;

/// `QUANT_TABLE_BITS`: the AC quantizer table is in Q3.
const QUANT_TABLE_BITS: u32 = 3;
/// The quantizer threshold is the quantizer step divided by 64.
const Q_THRESHOLD_SHIFT: u32 = 6;
/// Largest native qindex (`MAXQ_8_BITS`).
const MAX_QINDEX: i32 = 255;
/// Eight-bit side thresholds: `(table + (1 << 4)) >> 5`, rounding the
/// table's 1/32 units to samples.
const SIDE_ROUND: i32 = 1 << 4;
const SIDE_SHIFT: u32 = 5;

/// `side_thresholds`: 32 × the fitted side threshold per qindex (AVM's
/// comment: `32 · max(0.0444·q − 2.9936, 0.31·q − 39)`, hand-adjusted at low q).
const SIDE_TABLE: [i16; 296] = [
    -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16,
    -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16,
    -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16,
    -14, -13, -11, -10, -9, -7, -6, -4, -3, -2, 0, 0, 2, 3, 5, 6, 7, 9, 10, 12, 13, 15, 16, 17, 19,
    20, 22, 23, 24, 26, 27, 29, 30, 32, 33, 34, 36, 37, 39, 40, 42, 43, 44, 46, 47, 49, 50, 51, 53,
    54, 56, 57, 59, 60, 61, 63, 64, 66, 67, 69, 70, 71, 73, 74, 76, 77, 78, 80, 81, 83, 84, 86, 87,
    88, 90, 91, 93, 94, 96, 101, 111, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230,
    240, 249, 259, 269, 279, 289, 299, 309, 319, 329, 339, 349, 359, 368, 378, 388, 398, 408, 418,
    428, 438, 448, 458, 468, 478, 488, 497, 507, 517, 527, 537, 547, 557, 567, 577, 587, 597, 607,
    616, 626, 636, 646, 656, 666, 676, 686, 696, 706, 716, 726, 736, 745, 755, 765, 775, 785, 795,
    805, 815, 825, 835, 845, 855, 864, 874, 884, 894, 904, 914, 924, 934, 944, 954, 964, 974, 984,
    993, 1003, 1013, 1023, 1033, 1043, 1053, 1063, 1073, 1083, 1093, 1103, 1112, 1122, 1132, 1142,
    1152, 1162, 1172, 1182, 1192, 1202, 1212, 1222, 1232, 1241, 1251, 1261, 1271, 1281, 1291, 1301,
    1311, 1321, 1331, 1341, 1351, 1360, 1370, 1380, 1390, 1400, 1410, 1420, 1430, 1440, 1450, 1460,
    1470, 1480, 1489, 1499, 1509, 1519, 1529, 1539, 1549, 1559, 1569, 1579, 1589, 1599, 1608, 1618,
    1628, 1638, 1648, 1658, 1668, 1678,
];

/// `df_quant_from_qindex`: the quantizer threshold of edges whose block
/// qindex plus frame offset is `qindex` (any integer; clamped like AVM's
/// `av2_ac_quant_QTX`). Zero below qindex 89: such edges are not
/// filtered.
#[must_use]
pub fn q_threshold(qindex: i32) -> u16 {
    // `get_q`: qindex zero (lossless) keeps its own table entry; any other
    // value is clamped to the lossy range.
    let clamped = if qindex == 0 {
        0
    } else {
        qindex.clamp(1, MAX_QINDEX)
    };
    let step = quantizer_step_q3(clamped as u8);
    let rounded = (step + (1 << (QUANT_TABLE_BITS - 1))) >> QUANT_TABLE_BITS;
    (rounded >> Q_THRESHOLD_SHIFT) as u16
}

/// `df_side_from_qindex`: the side (smoothness) threshold for `qindex`
/// plus offset.
#[must_use]
pub fn side_threshold(qindex: i32) -> u16 {
    let index = qindex.clamp(0, SIDE_TABLE.len() as i32 - 1) as usize;
    ((i32::from(SIDE_TABLE[index]) + SIDE_ROUND).max(0) >> SIDE_SHIFT) as u16
}

/// Both thresholds of every block qindex for one channel's offset,
/// computed once per picture and channel.
#[derive(Debug, Clone)]
pub struct ThresholdTable {
    entries: [(u16, u16); 256],
}

impl ThresholdTable {
    /// The table for a channel whose qindex offset is `offset`.
    #[must_use]
    pub fn new(offset: i32) -> Self {
        Self {
            entries: std::array::from_fn(|qindex| {
                let qindex = qindex as i32 + offset;
                (q_threshold(qindex), side_threshold(qindex))
            }),
        }
    }

    /// `(quantizer, side)` thresholds of a block at `qindex`.
    #[must_use]
    pub fn get(&self, qindex: u8) -> (u16, u16) {
        self.entries[usize::from(qindex)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thresholds_start_where_avm_starts_filtering() {
        // qindex 88: step 62 << 3 gives 62 >> 6 = 0; qindex 89: 64 << 3 → 1.
        assert_eq!(q_threshold(88), 0);
        assert_eq!(q_threshold(89), 1);
        assert_eq!(q_threshold(0), 0);
        assert_eq!(q_threshold(-5), 0);
        assert_eq!(q_threshold(400), q_threshold(255));
        // Side: qindex 79 (table value 16) is the first to round to 1.
        assert_eq!(side_threshold(78), 0);
        assert_eq!(side_threshold(79), 1);
        assert_eq!(side_threshold(-3), 0);
        assert_eq!(side_threshold(295), (1678 + 16) >> 5);
        assert_eq!(side_threshold(1000), side_threshold(295));
    }
}
