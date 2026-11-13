//! Macroblock QP: prediction and `mb_qp_delta` (H.264 §7.4.5, §8.7.2.2,
//! §9.3.2.7, §9.3.3.1.1.5), shared by the encoder and our decoder.
//!
//! `QP_Y,PRED` is the QP of the previous macroblock in decoding order in the
//! slice (the slice QP for the first). A macroblock carries `mb_qp_delta`
//! only when it has a `macroblock_layer` with residual (or is `I_16x16`);
//! any other macroblock (`P_Skip`, or no coded residual) keeps the
//! predicted QP, which then matters only for deblocking and prediction of
//! the next macroblock.

/// QP values modulo which `mb_qp_delta` wraps (§7.4.5: 52 + `QpBdOffsetY`, 8-bit).
const QP_RANGE: i32 = 52;
/// `mb_qp_delta` range (§7.4.5): `-(26 + QpBdOffsetY / 2) ..= 25 + QpBdOffsetY / 2`.
const DELTA_MIN: i32 = -26;
const DELTA_MAX: i32 = 25;

/// ctxIdxOffset of `mb_qp_delta` (Table 9-34) and the context of its
/// second and later bins (§9.3.3.1.2: ctxIdxInc 2 and 3).
pub const MB_QP_DELTA_CTX: usize = 60;
pub const MB_QP_DELTA_SECOND_BIN_CTX: usize = MB_QP_DELTA_CTX + 2;
pub const MB_QP_DELTA_LATER_BIN_CTX: usize = MB_QP_DELTA_CTX + 3;

/// The running QP state of one slice.
#[derive(Debug, Clone, Copy)]
pub struct QpTracker {
    pred: u8,
    /// The previous macroblock in decoding order coded a nonzero
    /// `mb_qp_delta` (the first bin's ctxIdxInc, §9.3.3.1.1.5).
    prev_delta_nonzero: bool,
}

impl QpTracker {
    /// A slice starting at `slice_qp` (`SliceQP_Y`).
    #[must_use]
    pub const fn new(slice_qp: u8) -> Self {
        Self {
            pred: slice_qp,
            prev_delta_nonzero: false,
        }
    }

    /// The `mb_qp_delta` that takes the next macroblock to `qp`: the
    /// difference, wrapped modulo 52 into the legal range (§7.4.5).
    #[must_use]
    pub fn delta_to(self, qp: u8) -> i32 {
        let delta = i32::from(qp) - i32::from(self.pred);
        if delta > DELTA_MAX {
            delta - QP_RANGE
        } else if delta < DELTA_MIN {
            delta + QP_RANGE
        } else {
            delta
        }
    }

    /// First-bin ctxIdxInc of the next macroblock's `mb_qp_delta`.
    #[must_use]
    pub const fn delta_ctx_inc(self) -> usize {
        self.prev_delta_nonzero as usize
    }

    /// A macroblock that coded `mb_qp_delta = delta`; returns its `QP_Y`
    /// (§7.4.5: `(QP_Y,PRED + mb_qp_delta + 52) % 52`).
    pub fn coded(&mut self, delta: i32) -> u8 {
        let qp = (i32::from(self.pred) + delta + QP_RANGE).rem_euclid(QP_RANGE);
        self.pred = qp as u8;
        self.prev_delta_nonzero = delta != 0;
        self.pred
    }

    /// A macroblock without `mb_qp_delta`; returns its `QP_Y` (the prediction).
    pub fn uncoded(&mut self) -> u8 {
        self.prev_delta_nonzero = false;
        self.pred
    }
}

/// Whether `delta` is a legal `mb_qp_delta` (§7.4.5).
#[cfg(test)]
pub fn delta_in_range(delta: i32) -> bool {
    (DELTA_MIN..=DELTA_MAX).contains(&delta)
}

/// `mb_qp_delta` mapped to the unsigned value whose unary binarization is
/// coded (Table 9-3: k > 0 -> 2k - 1, k <= 0 -> -2k).
#[must_use]
pub fn mapped_delta(delta: i32) -> u32 {
    if delta > 0 {
        (2 * delta - 1) as u32
    } else {
        (-2 * delta) as u32
    }
}

/// Inverse of [`mapped_delta`].
#[cfg(test)]
pub fn unmapped_delta(mapped: u32) -> i32 {
    let half = mapped.div_ceil(2) as i32;
    if mapped % 2 == 1 { half } else { -half }
}

/// Context of unary bin `bin` of `mb_qp_delta` given the first bin's
/// ctxIdxInc (§9.3.3.1.2).
#[must_use]
pub const fn delta_bin_ctx(bin: u32, first_inc: usize) -> usize {
    match bin {
        0 => MB_QP_DELTA_CTX + first_inc,
        1 => MB_QP_DELTA_SECOND_BIN_CTX,
        _ => MB_QP_DELTA_LATER_BIN_CTX,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapping_round_trips_over_the_legal_range() {
        for delta in DELTA_MIN..=DELTA_MAX {
            assert_eq!(unmapped_delta(mapped_delta(delta)), delta);
        }
        assert_eq!(mapped_delta(0), 0);
        assert_eq!(mapped_delta(1), 1);
        assert_eq!(mapped_delta(-1), 2);
    }

    #[test]
    fn prediction_follows_decoding_order() {
        let mut t = QpTracker::new(30);
        assert_eq!(t.coded(t.delta_to(24)), 24);
        assert_eq!(t.delta_ctx_inc(), 1);
        assert_eq!(t.uncoded(), 24); // a skipped macroblock keeps the prediction
        assert_eq!(t.delta_ctx_inc(), 0);
        assert_eq!(t.coded(0), 24);
        assert_eq!(t.coded(-25), 51); // the modulo wrap of §7.4.5
        for (from, to) in [(0u8, 51u8), (51, 0), (10, 45), (45, 10)] {
            let mut t = QpTracker::new(from);
            let delta = t.delta_to(to);
            assert!(delta_in_range(delta), "{from}->{to}: {delta}");
            assert_eq!(t.coded(delta), to);
        }
    }
}
