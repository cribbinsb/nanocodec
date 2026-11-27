//! Edge thresholds of AV2 deblocking from the quantizer index (AVM v1.0.0
//! `df_quant_from_qindex`, `df_side_from_qindex` and the `side_thresholds`
//! table in `av2/common/av2_loopfilter.c`; BSD-3-Clause-Clear, see
//! `THIRD_PARTY.md`); the deeper forms follow dav2d's `deblock_quant_thr`
//! and `deblock_side_thr` (`db_apply_tmpl.c`).

use crate::transform::dequantizer_step;

/// `QUANT_TABLE_BITS`: the AC quantizer table is in Q3.
const QUANT_TABLE_BITS: u32 = 3;
/// The quantizer threshold is the quantizer step divided by 64.
const Q_THRESHOLD_SHIFT: u32 = 6;
/// Largest native qindex (`MAXQ_8_BITS`), and the indices each step of
/// two bits of depth adds (dav2d `48 * hbd`).
const MAX_QINDEX: i32 = 255;
const QINDEX_PER_DEPTH_STEP: i32 = 48;
/// Eight-bit side thresholds: `(table + (1 << 4)) >> 5`, rounding the
/// table's 1/32 units to samples; each step of depth scales the result up
/// by two bits (dav2d `bitdepth_min_8`).
const SIDE_ROUND: i32 = 1 << 4;
const SIDE_SHIFT: u32 = 5;
const BITS_PER_DEPTH_STEP: u32 = 2;

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
/// `av2_ac_quant_QTX`) in an 8-bit sequence. Zero below qindex 89: such
/// edges are not filtered.
#[must_use]
pub const fn q_threshold(qindex: i32) -> u16 {
    q_threshold_at(qindex, 0)
}

/// [`q_threshold`] in a sequence `high_bit_depth` steps of two bits deeper
/// than 8 (dav2d `hbd`), whose qindex range is that much longer.
#[must_use]
pub const fn q_threshold_at(qindex: i32, high_bit_depth: u8) -> u16 {
    let max = MAX_QINDEX + QINDEX_PER_DEPTH_STEP * high_bit_depth as i32;
    // `get_q`: qindex zero (lossless) keeps its own table entry; any other
    // value is clamped to the lossy range.
    let clamped = if qindex < 0 {
        1
    } else if qindex > max {
        max
    } else {
        qindex
    };
    let step = dequantizer_step(clamped) as i32;
    let rounded = (step + (1 << (QUANT_TABLE_BITS - 1))) >> QUANT_TABLE_BITS;
    (rounded >> Q_THRESHOLD_SHIFT) as u16
}

/// `df_side_from_qindex`: the side (smoothness) threshold for `qindex`
/// plus offset in an 8-bit sequence.
#[must_use]
pub const fn side_threshold(qindex: i32) -> u16 {
    side_threshold_at(qindex, 0)
}

/// [`side_threshold`] in a sequence `high_bit_depth` steps deeper: the
/// table is read at the 8-bit-equivalent qindex and rounded to the deeper
/// samples (dav2d `deblock_side_thr`).
#[must_use]
pub const fn side_threshold_at(qindex: i32, high_bit_depth: u8) -> u16 {
    let extra_bits = BITS_PER_DEPTH_STEP * high_bit_depth as u32;
    let qindex = qindex - QINDEX_PER_DEPTH_STEP * high_bit_depth as i32;
    let last = SIDE_TABLE.len() as i32 - 1;
    let index = if qindex < 0 {
        0
    } else if qindex > last {
        last
    } else {
        qindex
    } as usize;
    let rounded = SIDE_TABLE[index] as i32 + (SIDE_ROUND >> extra_bits);
    ((if rounded < 0 { 0 } else { rounded }) >> (SIDE_SHIFT - extra_bits)) as u16
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
    pub const fn new(offset: i32) -> Self {
        let mut entries = [(0, 0); 256];
        let mut i = 0;
        while i < entries.len() {
            let qindex = i as i32 + offset;
            entries[i] = (q_threshold(qindex), side_threshold(qindex));
            i += 1;
        }
        Self { entries }
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

    #[test]
    fn dequantizer_steps_match_the_eight_bit_table() {
        for qindex in 0..=u8::MAX {
            let step = crate::transform::quantizer_step_q3(qindex);
            assert_eq!(dequantizer_step(i32::from(qindex)) as i32, step, "{qindex}");
        }
    }

    #[test]
    fn ten_bit_thresholds_follow_dav2d() {
        // dav2d `deblock_quant_thr(1, 303)`: (dq_lookup(303) + 4) >> 9, with
        // dq_lookup(303) = 60 << 12.
        assert_eq!(i32::from(q_threshold_at(303, 1)), ((60 << 12) + 4) >> 9);
        assert_eq!(q_threshold_at(400, 1), q_threshold_at(303, 1));
        // `deblock_side_thr(1, q)`: the table at q - 48, (t + 4) >> 3.
        assert_eq!(i32::from(side_threshold_at(48 + 295, 1)), (1678 + 4) >> 3);
        assert_eq!(i32::from(side_threshold_at(48 + 79, 1)), (16 + 4) >> 3);
        assert_eq!(side_threshold_at(10, 1), 0);
    }
}
