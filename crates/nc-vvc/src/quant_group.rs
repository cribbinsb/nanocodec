//! Quantization groups (H.266 §7.4.11.10, §8.7.1): how a CU's `QpY`
//! follows from `cu_qp_delta`, tracked identically by the backend (which
//! picks the QP each CU is quantised at) and our decoder; the per-8x8 map
//! of `QpY` deblocking reads (§8.8.3.6); and the delta's syntax.
//!
//! Groups are whole CTBs (`ph_cu_qp_delta_subdiv_*` 0), so a group's left
//! and above neighbours are never inside its CTB and its prediction
//! `qPY_PRED` is `qPY_PREV`: the `QpY` of the last CU of the previous group
//! in decoding order, or the slice QP for the first group (one slice, no
//! tiles) — except the first group of each later CTB row, which takes the
//! `QpY` of the CU above it ([`row_start_prediction`]; unlike HEVC). The
//! delta is coded once, in the group's first
//! transform unit with a coded block; CUs before it take `qPY_PRED`, the
//! one carrying it and every later CU of the group take `qPY_PRED +
//! CuQpDeltaVal`.

use crate::coeff::Bins;
use crate::contexts::Context;
use nc_common::decode::{DecodeError, Result};

/// `QpY` range of 8-bit video (0..=63): `CuQpDeltaVal` wraps modulo it
/// (§8.7.1, `QpBdOffset` 0).
const QP_RANGE: i32 = 64;
/// Smallest `CuQpDeltaVal` of 8-bit video (§7.4.11.10); the largest is
/// `QP_RANGE + MIN_DELTA - 1` (31).
const MIN_DELTA: i32 = -32;
/// `cu_qp_delta_abs` prefix: truncated unary up to this, then EG0 (§9.3.3.10).
const PREFIX_MAX: u32 = 5;
/// Longest EG0 prefix a delta of 8-bit video needs (|delta| - 5 < 2^6).
const MAX_EG_PREFIX: u32 = 6;
/// Side of a [`QpMap`] cell: the smallest CU (8x8 luma).
const CELL_LOG2: u32 = 3;
/// `cu_qp_delta_abs` contexts: initValue 35 for every slice type, shiftIdx 8
/// (§9.3.2.2 Table 70, as `VVenC`'s `DeltaQP` set).
pub(crate) const DELTA_INIT: u8 = 35;
/// See [`DELTA_INIT`].
pub(crate) const DELTA_WINDOW: u8 = 8;

/// One stream's quantization-group state within a slice.
#[derive(Debug, Clone, Copy)]
pub(crate) struct QuantGroup {
    /// `qPY_PRED` of the current group.
    pred: u32,
    /// `QpY` set by the current group's delta, once coded.
    coded: Option<u32>,
    /// `QpY` of the last CU (`qPY_PREV` of the next group).
    last: u32,
}

impl QuantGroup {
    /// The state at the start of a slice coded at `slice_qp`.
    pub(crate) fn new(slice_qp: u32) -> Self {
        Self {
            pred: slice_qp,
            coded: None,
            last: slice_qp,
        }
    }

    /// Start the next group (at each CTB); `above`: the prediction of the
    /// first group of a CTB row below another ([`row_start_prediction`]).
    pub(crate) fn begin_group(&mut self, above: Option<u32>) {
        self.pred = above.unwrap_or(self.last);
        self.coded = None;
    }

    /// The QP a CU wanting `wanted` is quantised at: the group's, once its
    /// delta is coded (only one delta per group).
    pub(crate) fn quantizer(&self, wanted: u32) -> u32 {
        self.coded.unwrap_or(wanted)
    }

    /// Whether the group's delta is still to be coded.
    pub(crate) fn delta_pending(&self) -> bool {
        self.coded.is_none()
    }

    /// Encoder: finish a CU quantised at `quantizer`, with a coded residual
    /// or not. Returns its `QpY` and the `CuQpDeltaVal` it carries, if it is
    /// the group's first CU with a residual.
    pub(crate) fn finish_cu(&mut self, quantizer: u32, residual: bool) -> (u32, Option<i32>) {
        let delta = (residual && self.coded.is_none()).then(|| {
            self.coded = Some(quantizer);
            (quantizer as i32 - self.pred as i32 - MIN_DELTA).rem_euclid(QP_RANGE) + MIN_DELTA
        });
        (self.finish(), delta)
    }

    /// Decoder: finish a CU that carried `delta` (`CuQpDeltaVal`) or none.
    /// Returns its `QpY`.
    pub(crate) fn finish_decoded(&mut self, delta: Option<i32>) -> u32 {
        if let Some(delta) = delta {
            self.coded = Some((self.pred as i32 + delta + QP_RANGE).rem_euclid(QP_RANGE) as u32);
        }
        self.finish()
    }

    fn finish(&mut self) -> u32 {
        self.last = self.coded.unwrap_or(self.pred);
        self.last
    }
}

/// Whether a CU's `cu_qp_delta` is still to be read, as its transform units
/// are parsed (`IsCuQpDeltaCoded`, §7.3.11.10).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeltaRead {
    /// The PPS has no deltas, or the group's was read by an earlier CU.
    Absent,
    /// The CU's first transform unit with a coded block reads it.
    Pending,
    /// Read: `CuQpDeltaVal`.
    Read(i32),
}

impl DeltaRead {
    /// The delta the CU carried, if any.
    pub(crate) fn value(self) -> Option<i32> {
        match self {
            Self::Read(delta) => Some(delta),
            Self::Absent | Self::Pending => None,
        }
    }
}

/// `qPY_PRED` of the group of the CTB at luma (`x`, `y`) when it is the
/// first of a CTB row with one above: the `QpY` of the CU above it
/// (§8.7.1, step 5: the first group of a CTB row predicts from above, whether
/// or not wavefront entry points are used; `VVdeC` and `FFmpeg` agree).
pub(crate) fn row_start_prediction(map: &QpMap, x: usize, y: usize) -> Option<u32> {
    (x == 0 && y > 0).then(|| map.at(0, y - 1))
}

/// `QpY` of every 8x8 luma cell of a picture (deblocking's `QpP`/`QpQ`).
#[derive(Debug, Clone, Default)]
pub(crate) struct QpMap {
    columns: usize,
    cells: Vec<u8>,
}

impl QpMap {
    /// A `width` x `height` picture, every cell at `qp`.
    pub(crate) fn reset(&mut self, width: usize, height: usize, qp: u32) {
        self.columns = width.div_ceil(1 << CELL_LOG2);
        self.cells.clear();
        self.cells
            .resize(self.columns * height.div_ceil(1 << CELL_LOG2), qp as u8);
    }

    /// Record `qp` for CU `cu`.
    pub(crate) fn record(&mut self, cu: &nc_encoder::Block, qp: u32) {
        let (x0, y0) = (cu.x >> CELL_LOG2, cu.y >> CELL_LOG2);
        let (w, h) = (
            (cu.width() >> CELL_LOG2).max(1),
            (cu.height() >> CELL_LOG2).max(1),
        );
        let rows = self.cells.len() / self.columns.max(1);
        for row in y0..(y0 + h).min(rows) {
            let start = row * self.columns + x0;
            let end = row * self.columns + (x0 + w).min(self.columns);
            self.cells[start..end].fill(qp as u8);
        }
    }

    /// `QpY` of the CU covering luma sample (`x`, `y`).
    pub(crate) fn at(&self, x: usize, y: usize) -> u32 {
        u32::from(self.cells[(y >> CELL_LOG2) * self.columns + (x >> CELL_LOG2)])
    }
}

/// §7.3.11.10 `cu_qp_delta_abs` and `cu_qp_delta_sign_flag`: `delta` on
/// write; the decoded `CuQpDeltaVal` either way. `contexts`: the prefix's
/// first bin and its others (§9.3.4.2, Table 132).
pub(crate) fn code_delta(
    io: &mut impl Bins,
    contexts: &mut [Context; 2],
    delta: i32,
) -> Result<i32> {
    let abs = delta.unsigned_abs();
    let mut prefix = 0;
    while prefix < PREFIX_MAX {
        let context = &mut contexts[usize::from(prefix > 0)];
        if !io.bin(context, abs > prefix)? {
            break;
        }
        prefix += 1;
    }
    let abs = if prefix == PREFIX_MAX {
        PREFIX_MAX + exp_golomb0(io, abs.saturating_sub(PREFIX_MAX))?
    } else {
        prefix
    };
    if abs == 0 {
        return Ok(0);
    }
    let negative = io.bypass(u32::from(delta < 0), 1)? == 1;
    let value = abs as i32;
    if value > -MIN_DELTA || (!negative && value == -MIN_DELTA) {
        return Err(DecodeError::Malformed("VVC cu_qp_delta range"));
    }
    Ok(if negative { -value } else { value })
}

/// k-th order Exp-Golomb with k = 0, bypass coded (§9.3.3.6).
fn exp_golomb0(io: &mut impl Bins, value: u32) -> Result<u32> {
    let (mut base, mut k) = (0, 0);
    while io.bypass(u32::from(value >= base + (1 << k)), 1)? == 1 {
        base += 1 << k;
        k += 1;
        if k > MAX_EG_PREFIX {
            return Err(DecodeError::Malformed("VVC cu_qp_delta suffix"));
        }
    }
    let rest = if k > 0 {
        io.bypass(value.wrapping_sub(base), k)?
    } else {
        0
    };
    Ok(base + rest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::CabacWriter;
    use crate::decoder_cabac::Reader;

    #[test]
    fn cus_before_the_delta_take_the_prediction() {
        let mut g = QuantGroup::new(30);
        g.begin_group(None);
        // A skipped CU, then one coded at 24, then one wanting 40.
        assert_eq!(g.finish_cu(24, false), (30, None));
        assert_eq!(g.quantizer(24), 24);
        assert_eq!(g.finish_cu(24, true), (24, Some(-6)));
        assert_eq!(g.quantizer(40), 24);
        assert_eq!(g.finish_cu(24, true), (24, None));
        // The next group predicts from the last CU.
        g.begin_group(None);
        // 24 -> 63 wraps: 24 - 25 + 64.
        assert_eq!(g.finish_cu(63, true), (63, Some(-25)));
        g.begin_group(None);
        assert_eq!(g.finish_cu(0, true), (0, Some(1)));
        // A CTB row's first group predicts from the CU above it.
        g.begin_group(Some(40));
        assert_eq!(g.finish_cu(41, false), (40, None));
    }

    #[test]
    fn decoded_deltas_wrap_like_the_encoder() {
        for (pred, qp) in [(30, 24), (0, 63), (63, 0), (40, 40)] {
            let mut encoder = QuantGroup::new(pred);
            let (_, delta) = encoder.finish_cu(qp, true);
            let delta = delta.unwrap();
            assert!((MIN_DELTA..QP_RANGE + MIN_DELTA).contains(&delta));
            let mut decoder = QuantGroup::new(pred);
            assert_eq!(decoder.finish_decoded(Some(delta)), qp);
        }
    }

    #[test]
    fn deltas_round_trip_through_cabac() {
        let deltas = [0, 1, -1, 4, 5, -5, 6, 12, -20, 31, -32];
        let mut w = CabacWriter::new();
        let mut contexts = [Context::new(DELTA_INIT, DELTA_WINDOW, 30); 2];
        for &d in &deltas {
            assert_eq!(code_delta(&mut w, &mut contexts, d).unwrap(), d);
        }
        w.encode_terminate(true);
        let mut bits = w.finish();
        bits.trailing_bits();
        let bytes = bits.into_bytes();
        let mut r = Reader::new(&bytes).unwrap();
        let mut contexts = [Context::new(DELTA_INIT, DELTA_WINDOW, 30); 2];
        for &d in &deltas {
            assert_eq!(code_delta(&mut r, &mut contexts, 0).unwrap(), d);
        }
    }

    #[test]
    fn the_map_records_cus() {
        let mut map = QpMap::default();
        map.reset(24, 16, 30);
        map.record(&nc_encoder::Block::square(8, 0, 4), 22);
        map.record(&nc_encoder::Block::square(0, 0, 3), 25);
        assert_eq!(row_start_prediction(&map, 0, 8), Some(25));
        assert_eq!(row_start_prediction(&map, 8, 8), None);
        assert_eq!(map.at(0, 8), 30);
        assert_eq!(map.at(8, 8), 22);
        assert_eq!(map.at(23, 15), 22);
    }
}
