//! Cyclic spatial cadence (`rate.cadence_spatial`): the QP cadence's base
//! boost given to a rotating band of still blocks every frame, instead of
//! to the whole picture every period-th frame.
//!
//! The frame-level cadence (`qp_cadence.rs`) makes every period-th frame
//! finer so that the still background is refined once per cycle. On still
//! content such a base frame refines everything its predecessors left and
//! costs 1-9 cycle budgets, while most base frames cost almost nothing: a
//! bimodal size no per-role model predicts (13-22x misses in AV1,
//! `docs/results/rate-control-audit-2026-11-13.md`). The real-time
//! references refine a rotating share of the static blocks every frame
//! instead ("cyclic refresh": libvpx `vp9_aq_cyclicrefresh.c`, libaom
//! `aq_cyclicrefresh.c`, SVT-AV1 `rc_rtc_cbr.c`; the idea, reimplemented):
//! each still block is still offered once per cycle, but every frame does a
//! slice of the work, so frame sizes stay unimodal.
//!
//! Per frame, on the codec's quantizer squares (`CodecBackend::BLOCK_QP_LOG2`):
//!
//! - **Candidates:** squares whose every 8x8 cell has been static for at
//!   least `MIN_STATIC_RUN` frames (`StillMap::static_runs`; the stricter
//!   still map, with its guard ring and longer run, left too few squares)
//!   and whose last coded step is coarser than the band's step (libvpx's
//!   "coded coarser than the boost" rule): a square already that fine has
//!   nothing to gain and stays out.
//! - **When:** after the start-up cadence only; the start-up cadence keeps
//!   its whole finer frames (`QpCadence::flattened`).
//! - **Band:** `1 / period` of the candidates, taken in raster order from
//!   where the previous frame's band stopped, so each candidate comes round
//!   once per cycle (SVT's moving band).
//! - **Offset:** the cadence's base boost (its start-up boost after a key)
//!   times the cadence strength (still share, motion gate), finer.
//!
//! Rate control sees a flat cadence (every frame plays the same role) and
//! budgets the band through the planned block offsets
//! (`RateController::effective_block_offset`), as the references choose
//! their base quantizer from a segment-weighted bits model. Off by default:
//! every preset is byte-identical without it.

use nc_analysis::still::StillMap;

use crate::qp_cadence::{MotionHint, QpCadence, step_ratio};

/// log2 of the side of a still-map cell (8x8 luma).
const CELL_LOG2: u8 = 3;
/// Consecutive static frames every cell of a candidate square needs.
const MIN_STATIC_RUN: u8 = 2;
/// A square counts as coarser than the band when its last step exceeds the
/// band's by more than this ratio (rounding of the codecs' step tables).
const COARSER_MARGIN: f64 = 1.01;

/// Band state of one encoder.
#[derive(Debug)]
pub(crate) struct CyclicCadence {
    /// The configured cadence: period, boosts and strength law.
    cadence: QpCadence,
    /// log2 of the side of one quantizer square, in luma samples.
    unit_log2: u8,
    /// Squares per row and per column.
    squares: (usize, usize),
    /// Next square (raster index) the band starts from.
    cursor: usize,
    /// Sample-domain step each square was last coded at (`INFINITY`:
    /// unknown, refinable).
    last_step: Vec<f64>,
    /// The band's offset of every cell of the frame, QP units (0 outside).
    offsets: Vec<f64>,
    /// Cells per row of `offsets`.
    cols: usize,
    /// Sample step the current frame's band refines to.
    band_step: f64,
}

impl CyclicCadence {
    /// `None` unless `rate.cadence_spatial` is on and the cadence is not
    /// flat.
    pub fn new(enabled: bool, cadence: QpCadence, unit_log2: u8) -> Option<Self> {
        (enabled && !cadence.is_flat()).then(|| Self {
            cadence,
            unit_log2: unit_log2.max(CELL_LOG2),
            squares: (0, 0),
            cursor: 0,
            last_step: Vec::new(),
            offsets: Vec::new(),
            cols: 0,
            band_step: 0.,
        })
    }

    /// Side of a square in cells.
    fn side(&self) -> usize {
        1 << (self.unit_log2 - CELL_LOG2)
    }

    /// Size the maps for a still map of `cols` x `rows` cells; a new size
    /// forgets every square's step.
    fn resize(&mut self, cols: usize, rows: usize) {
        let side = self.side();
        let squares = (cols.div_ceil(side), rows.div_ceil(side));
        if squares != self.squares {
            self.squares = squares;
            self.cursor = 0;
            self.last_step.clear();
            self.last_step.resize(squares.0 * squares.1, f64::INFINITY);
        }
    }

    /// The band's strength (0..1) for a frame with this still share and
    /// motion: the cadence's own law.
    pub fn strength(&self, static_fraction: f64, motion: Option<MotionHint>) -> f64 {
        self.cadence.strength(static_fraction, motion)
    }

    /// A key frame coded at sample step `step`: every square now holds it.
    pub fn key(&mut self, step: f64) {
        self.last_step.fill(step);
        self.cursor = 0;
    }

    /// The band's per-cell offsets for an inter frame `distance` frames
    /// after the last key whose level codes at sample step `level_step`,
    /// with `still` its still map and the cadence `strength`. Empty when
    /// nothing is refined.
    pub fn plan(
        &mut self,
        still: &StillMap,
        distance: u64,
        level_step: f64,
        strength: f64,
    ) -> &[f64] {
        let (cols, rows) = (still.blocks.cols(), still.blocks.rows());
        self.resize(cols, rows);
        self.cols = cols;
        self.offsets.clear();
        if self.cadence.in_startup(distance) {
            return &self.offsets;
        }
        let (period, boost) = self.cadence.band(distance);
        let offset = -boost * strength;
        if period <= 1 || offset >= 0. {
            return &self.offsets;
        }
        let band_step = level_step * step_ratio(offset);
        self.band_step = band_step;
        let candidates: Vec<usize> = (0..self.last_step.len())
            .filter(|&square| {
                self.last_step[square] > band_step * COARSER_MARGIN && self.all_still(still, square)
            })
            .collect();
        if candidates.is_empty() {
            return &self.offsets;
        }
        let share = candidates.len().div_ceil(period as usize);
        let start = candidates.partition_point(|&square| square < self.cursor);
        self.offsets.resize(cols * rows, 0.);
        let mut last = self.cursor;
        for &square in candidates.iter().cycle().skip(start).take(share) {
            self.fill(square, offset, cols, rows);
            last = square;
        }
        self.cursor = (last + 1) % self.last_step.len();
        &self.offsets
    }

    /// Whether every cell of `square` inside the picture has been static
    /// for `MIN_STATIC_RUN` frames.
    fn all_still(&self, still: &StillMap, square: usize) -> bool {
        let side = self.side();
        let (col0, row0) = (
            (square % self.squares.0) * side,
            (square / self.squares.0) * side,
        );
        let (cols, rows) = (still.blocks.cols(), still.blocks.rows());
        (row0..(row0 + side).min(rows)).all(|row| {
            still
                .static_runs
                .row_run(col0, row, (col0 + side).min(cols) - col0)
                .iter()
                .all(|&run| run >= MIN_STATIC_RUN)
        })
    }

    /// Give every cell of `square` the band's `offset`.
    fn fill(&mut self, square: usize, offset: f64, cols: usize, rows: usize) {
        let side = self.side();
        let (col0, row0) = (
            (square % self.squares.0) * side,
            (square / self.squares.0) * side,
        );
        for row in row0..(row0 + side).min(rows) {
            self.offsets[row * cols + col0..row * cols + (col0 + side).min(cols)].fill(offset);
        }
    }

    /// The band square at luma `(x, y)` was skipped as within the camera's
    /// noise (`noise_skip.rs`): it holds what refining would buy, so it is
    /// not offered again until it is coded coarser.
    pub fn settled(&mut self, (x, y): (usize, usize)) {
        let step = self.band_step;
        self.coded((x, y), step);
    }

    /// Whether the cell at luma `(x, y)` is in this frame's band.
    pub fn in_band(&self, (x, y): (usize, usize)) -> bool {
        self.offsets
            .get((y >> CELL_LOG2) * self.cols + (x >> CELL_LOG2))
            .is_some_and(|&offset| offset < 0.)
    }

    /// A block at luma `(x, y)` was coded with a residual at sample step
    /// `step` (a copy or skip keeps its square's step).
    pub fn coded(&mut self, (x, y): (usize, usize), step: f64) {
        let square = (y >> self.unit_log2) * self.squares.0 + (x >> self.unit_log2);
        if let Some(last) = self.last_step.get_mut(square) {
            *last = step;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{ParamOverrides, RateParams};
    use nc_analysis::grid::Grid;

    fn band(period: f32) -> CyclicCadence {
        let mut o = ParamOverrides::default();
        o.set("rate.cadence_period", period);
        o.set("rate.cadence_startup_period", 0.);
        let cadence = QpCadence::from_params(&RateParams::from_overrides(&o));
        CyclicCadence::new(true, cadence, 4).expect("on")
    }

    fn still(cols: usize, rows: usize) -> StillMap {
        StillMap {
            blocks: Grid::filled(cols, rows, true),
            runs: Grid::filled(cols, rows, 10),
            static_runs: Grid::filled(cols, rows, 10),
            fraction: 1.,
        }
    }

    #[test]
    fn off_or_flat_builds_nothing() {
        let cadence =
            QpCadence::from_params(&RateParams::from_overrides(&ParamOverrides::default()));
        assert!(CyclicCadence::new(false, cadence, 4).is_none());
    }

    #[test]
    fn the_band_covers_every_still_square_once_per_cycle() {
        // 16x16 squares (2x2 cells), an 8x4-cell picture: 4x2 = 8 squares.
        let mut b = band(4.);
        let map = still(8, 4);
        let mut refined = vec![0; 8];
        for distance in 1..=4 {
            let offsets = b.plan(&map, distance, 10., 1.).to_vec();
            for (square, count) in refined.iter_mut().enumerate() {
                let (col, row) = ((square % 4) * 2, (square / 4) * 2);
                if offsets[row * 8 + col] < 0. {
                    *count += 1;
                }
            }
        }
        assert_eq!(refined, vec![1; 8]);
    }

    #[test]
    fn squares_already_finer_than_the_band_are_left_out() {
        let mut b = band(4.);
        let map = still(8, 4);
        b.plan(&map, 1, 10., 1.);
        b.key(1.);
        assert_eq!(b.plan(&map, 1, 10., 1.).len(), 0);
    }

    #[test]
    fn moving_cells_keep_their_square_out() {
        let mut b = band(2.);
        let mut map = still(4, 2);
        *map.static_runs.get_mut(0, 0) = 0;
        let offsets = b.plan(&map, 2, 10., 1.).to_vec();
        assert_eq!(offsets[0], 0.);
        assert!(offsets[2] < 0.);
    }
}
