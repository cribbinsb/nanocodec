//! Quantization groups (HEVC §7.4.9.14, §8.6.1): how a CU's `QpY` follows
//! from `cu_qp_delta`, tracked identically by the backend (which picks the
//! QP each CU is quantised at) and the syntax writer (which codes the
//! deltas from the recorded `QpY`s).
//!
//! Groups are whole CTBs (`headers::QUANT_GROUP_DEPTH` 0), so a group's
//! left and above neighbours are never inside its CTB and its prediction
//! `qPY_PRED` is `qPY_PREV`: the `QpY` of the last CU of the previous group
//! in decoding order, or the slice QP for the first group (one slice, no
//! tiles, no wavefronts). The delta is coded once, in the group's first
//! transform unit with a coded block; CUs before it take `qPY_PRED`, the
//! one carrying it and every later CU of the group take `qPY_PRED +
//! CuQpDeltaVal`.

/// `QpY` range of 8-bit video: `CuQpDeltaVal` wraps modulo it (§8.6.1).
const QP_RANGE: i32 = 52;
/// Smallest `CuQpDeltaVal` of 8-bit video (§7.4.9.14); the largest is
/// `QP_RANGE + MIN_DELTA - 1` (25).
const MIN_DELTA: i32 = -26;

/// One stream's quantization-group state within a slice.
#[derive(Debug, Clone, Copy)]
pub struct QuantGroup {
    /// `qPY_PRED` of the current group.
    pred: i32,
    /// `QpY` set by the current group's delta, once coded.
    coded: Option<i32>,
    /// `QpY` of the last CU (`qPY_PREV` of the next group).
    last: i32,
}

impl QuantGroup {
    /// The state at the start of a slice coded at `slice_qp`.
    #[must_use]
    pub fn new(slice_qp: i32) -> Self {
        Self {
            pred: slice_qp,
            coded: None,
            last: slice_qp,
        }
    }

    /// Start the next group (at each CTB).
    pub fn begin_group(&mut self) {
        self.pred = self.last;
        self.coded = None;
    }

    /// The QP a CU wanting `wanted` is quantised at: the group's, once its
    /// delta is coded (only one delta per group).
    #[must_use]
    pub fn quantizer(&self, wanted: i32) -> i32 {
        self.coded.unwrap_or(wanted)
    }

    /// Whether the group's delta is still to be coded.
    #[must_use]
    pub fn delta_pending(&self) -> bool {
        self.coded.is_none()
    }

    /// The `CuQpDeltaVal` that gives `qp` from the group's prediction
    /// (wrapped into the 8-bit range), and the group then at `qp`.
    pub fn code_delta(&mut self, qp: i32) -> i32 {
        self.coded = Some(qp);
        (qp - self.pred - MIN_DELTA).rem_euclid(QP_RANGE) + MIN_DELTA
    }

    /// Finish a CU quantised at `quantizer`, with a coded residual or not:
    /// its `QpY`.
    pub fn finish_cu(&mut self, quantizer: i32, residual: bool) -> i32 {
        if residual && self.coded.is_none() {
            self.coded = Some(quantizer);
        }
        self.last = self.coded.unwrap_or(self.pred);
        self.last
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cus_before_the_delta_take_the_prediction() {
        let mut g = QuantGroup::new(30);
        g.begin_group();
        // A skipped CU, then one coded at 24, then one wanting 40.
        assert_eq!(g.finish_cu(24, false), 30);
        assert_eq!(g.quantizer(24), 24);
        assert_eq!(g.finish_cu(24, true), 24);
        assert_eq!(g.quantizer(40), 24);
        assert_eq!(g.finish_cu(24, false), 24);
        // The next group predicts from the last CU.
        g.begin_group();
        assert!(g.delta_pending());
        assert_eq!(g.code_delta(51), 27 - QP_RANGE);
        assert_eq!(g.finish_cu(51, true), 51);
    }

    #[test]
    fn deltas_wrap_into_the_coded_range() {
        for pred in 0..QP_RANGE {
            for qp in 0..QP_RANGE {
                let mut g = QuantGroup::new(pred);
                let delta = g.code_delta(qp);
                assert!((MIN_DELTA..QP_RANGE + MIN_DELTA).contains(&delta));
                assert_eq!((pred + delta + QP_RANGE).rem_euclid(QP_RANGE), qp);
            }
        }
    }
}
