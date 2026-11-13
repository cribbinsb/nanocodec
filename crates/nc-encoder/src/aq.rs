//! Block adaptive quantization: a per-block quantizer offset, in QP units,
//! from two terms that add.
//!
//! - **Variance** (`rate.aq_strength`): detailed (high-variance) blocks mask
//!   quantization noise, flat blocks show it, so moving bits from the first
//!   to the second at equal rate improves perceived quality (x264's and
//!   libvpx's variance AQ; the idea, not their code). The term is
//!   `strength * (log2(variance) - mean log2 variance of the frame)`, finer
//!   where flatter than average.
//! - **Persistence** (`persist.strength`, `persistence.rs`): blocks whose
//!   content later frames predict well get finer, unpredictable ones coarser.
//! - **Reuse** (`persist.reuse`, `reuse.rs`): still blocks whose measured
//!   reconstruction fell below their long-run quality are refined, at the
//!   step their reuse warrants. Its offsets follow from λ and are not
//!   clamped by `rate.aq_max_offset`, which bounds the two heuristic terms.
//!
//! The analysis already has every 8x8 block's sum of squared deviations
//! and motion error, so no pixel pass is added. Cell offsets are pooled per
//! aligned square of the codec's quantizer granularity
//! (`CodecBackend::BLOCK_QP_LOG2`), so every leaf inside one shares its
//! offset and rate control budgets the offsets the codec can code, then
//! clamped to `±rate.aq_max_offset`. Offsets are converted to the codec's
//! native quantizer through its step table, so a QP unit means the same step
//! ratio in every codec. Only backends with block-level QP syntax
//! (`CodecBackend::SUPPORTS_BLOCK_QP`) receive non-zero offsets.

use nc_analysis::MotionAnalysis;
use nc_analysis::preanalysis::{BlockStats, PreAnalysis};

use crate::params::{ParamOverrides, PersistParams, RateParams};
use crate::partition::Block;
use crate::persistence::PersistenceTracker;
use crate::qp_cadence::{nearest_quantizer, step_ratio};

/// Side of a pre-analysis cell (8x8 luma).
const CELL_LOG2: u8 = 3;
/// Samples per cell, the divisor of `sum_sq_dev` for the variance.
const CELL_SAMPLES: f64 = 64.;
/// Variance floor, so flat blocks do not dominate the log mean.
const MIN_VARIANCE: f64 = 1.;
/// Distinct offsets kept free for the reuse term (added after the grid)
/// when the codec codes only a few per frame.
const RESERVED_REUSE_OFFSETS: usize = 1;

/// Per-frame block AQ state of one encoder.
#[derive(Debug)]
pub(crate) struct BlockAq {
    /// Variance term strength (`rate.aq_strength`, 0 = off).
    strength: f64,
    /// Largest offset either way, in QP units (`rate.aq_max_offset`).
    max_offset: f64,
    /// log2 of the side of the square sharing one offset
    /// (`CodecBackend::BLOCK_QP_LOG2`).
    unit_log2: u8,
    /// Grid (QP units) the heuristic terms are rounded to when the codec
    /// codes only a few distinct offsets per frame
    /// (`CodecBackend::BLOCK_QP_OFFSETS`), else `None`.
    offset_step: Option<f64>,
    persistence: Option<PersistenceTracker>,
    cols: usize,
    /// Offset of every cell of the current frame in QP units, pooled per
    /// quantizer square (the heuristic terms clamped).
    offset: Vec<f64>,
    /// The reuse term's share of `offset`, before pooling (scratch).
    reuse_offset: Vec<f64>,
    /// The cells of `offset` that are not zero.
    nonzero: Vec<usize>,
    /// Whether any cell of the planned frame has a non-zero offset.
    active: bool,
    /// Sum of `bit_weight` over the frame.
    total_weight: f64,
    /// Expected share of the frame's bits of every cell (rate control's
    /// block-offset weight): its residual activity, temporal where the
    /// pre-analysis has a previous frame, spatial otherwise.
    bit_weight: Vec<f32>,
}

impl BlockAq {
    /// `None` when every term is off. Offsets are planned per aligned
    /// `1 << unit_log2` square (cells of 8x8 at 3 or below), at most
    /// `max_offsets` distinct non-zero ones per frame.
    pub fn new(overrides: &ParamOverrides, unit_log2: u8, max_offsets: usize) -> Option<Self> {
        let params = RateParams::from_overrides(overrides);
        let persist = PersistParams::from_overrides(overrides);
        let persistence = PersistenceTracker::new(&persist);
        let reuse = persist.reuse >= 0.5;
        (params.aq_strength > 0. || persistence.is_some() || reuse).then(|| Self {
            strength: f64::from(params.aq_strength),
            max_offset: f64::from(params.aq_max_offset),
            unit_log2,
            offset_step: heuristic_offset_step(params.aq_max_offset, max_offsets),
            persistence,
            cols: 0,
            offset: Vec::new(),
            reuse_offset: Vec::new(),
            nonzero: Vec::new(),
            active: false,
            total_weight: 0.,
            bit_weight: Vec::new(),
        })
    }

    /// Measure the frame from its analysis: `motion` is `None` on frames
    /// without a motion search (key frames, the first frame); `reuse` is
    /// the reuse term's offset of every cell (`reuse.rs`), if it is on.
    pub fn plan(
        &mut self,
        pre: &PreAnalysis,
        motion: Option<&MotionAnalysis>,
        reuse: Option<&[f64]>,
    ) {
        self.cols = pre.blocks.cols();
        let reuse = reuse.filter(|r| r.iter().any(|&offset| offset != 0.));
        if self.strength == 0. && self.persistence.is_none() && reuse.is_none() {
            // Nothing to plan (most frames with only the reuse term on):
            // the frame codes as without block AQ.
            self.active = false;
            return;
        }
        if self.strength == 0.
            && self.persistence.is_none()
            && let Some(reuse) = reuse.filter(|r| r.len() == pre.blocks.cells().len())
        {
            self.plan_reuse_only(pre, reuse);
            return;
        }
        self.bit_weight.clear();
        // Gathered first, then the square roots on their own: that loop
        // vectorises (in one pass it cost ~20 instructions per cell). f32
        // is ample for a weight and halves the passes' memory.
        self.bit_weight.extend(
            pre.blocks
                .cells()
                .iter()
                .map(|b| b.sad_prev.unwrap_or(b.sum_sq_dev) as f32),
        );
        for weight in &mut self.bit_weight {
            *weight = 1. + weight.sqrt();
        }
        self.total_weight = self.bit_weight.iter().map(|&w| f64::from(w)).sum();
        self.offset.clear();
        if self.strength > 0. {
            self.offset.extend(pre.blocks.cells().iter().map(|b| {
                (f64::from(b.sum_sq_dev) / CELL_SAMPLES)
                    .max(MIN_VARIANCE)
                    .log2()
            }));
            let count = self.offset.len().max(1) as f64;
            let mean = self.offset.iter().sum::<f64>() / count;
            for offset in &mut self.offset {
                *offset = self.strength * (*offset - mean);
            }
        } else {
            self.offset.resize(pre.blocks.cells().len(), 0.);
        }
        if let Some(persistence) = self.persistence.as_mut() {
            persistence.observe(motion);
            persistence.add_offsets(self.cols, &mut self.offset);
        }
        self.reuse_offset.clear();
        match reuse.filter(|r| r.len() == self.offset.len()) {
            Some(reuse) => self.reuse_offset.extend_from_slice(reuse),
            None => self.reuse_offset.resize(self.offset.len(), 0.),
        }
        // Pooling all-zero offsets (most frames with only the reuse term
        // on) changes nothing.
        let pool = |values: &mut Vec<f64>| {
            if values.iter().any(|&v| v != 0.) {
                pool_units(values, self.cols, self.unit_log2);
            }
        };
        if self.unit_log2 > CELL_LOG2 {
            pool(&mut self.offset);
            pool(&mut self.reuse_offset);
        }
        let max = self.max_offset;
        for (offset, reuse) in self.offset.iter_mut().zip(&self.reuse_offset) {
            let clamped = offset.clamp(-max, max);
            let coded = self
                .offset_step
                .map_or(clamped, |step| (clamped / step).round() * step);
            *offset = coded + reuse;
        }
        self.nonzero.clear();
        self.nonzero
            .extend((0..self.offset.len()).filter(|&cell| self.offset[cell] != 0.));
        self.active = !self.nonzero.is_empty();
    }

    /// Plan a frame whose only offsets are the reuse term's `reuse`: a few
    /// refined cells, so only their quantizer squares are pooled and
    /// written (the general path pooled and scanned every cell, ~2% of
    /// Jetson h264-faster cycles on the frames that refine anything).
    fn plan_reuse_only(&mut self, pre: &PreAnalysis, reuse: &[f64]) {
        if self.bit_weight.len() != reuse.len() {
            self.bit_weight.clear();
            self.bit_weight.resize(reuse.len(), 0.);
        }
        if self.offset.len() == reuse.len() {
            for &cell in &self.nonzero {
                self.offset[cell] = 0.;
            }
        } else {
            self.offset.clear();
            self.offset.resize(reuse.len(), 0.);
        }
        self.nonzero.clear();
        let (cols, rows) = (self.cols, reuse.len() / self.cols.max(1));
        let side = 1usize << self.unit_log2.saturating_sub(CELL_LOG2);
        for cell in (0..reuse.len()).filter(|&cell| reuse[cell] != 0.) {
            if self.offset[cell] != 0. {
                // Its square is pooled already.
                continue;
            }
            let (row0, col0) = ((cell / cols) / side * side, (cell % cols) / side * side);
            let (row1, col1) = ((row0 + side).min(rows), (col0 + side).min(cols));
            let index = |row: usize, col: usize| row * cols + col;
            let sum: f64 = (row0..row1)
                .flat_map(|row| (col0..col1).map(move |col| (row, col)))
                .map(|(row, col)| reuse[index(row, col)])
                .sum();
            let mean = sum / ((row1 - row0) * (col1 - col0)) as f64;
            for row in row0..row1 {
                self.offset[index(row, col0)..index(row, col1)].fill(mean);
                self.nonzero.extend(index(row, col0)..index(row, col1));
            }
        }
        // Only the offset squares' weights are read (the leaves coded off
        // the frame quantizer lie inside them: a square is the codec's
        // largest block); the frame's total comes from rate control's
        // complexity pass ([`Self::set_total_weight`]).
        for &cell in &self.nonzero {
            self.bit_weight[cell] = bit_weight(&pre.blocks.cells()[cell]);
        }
        self.active = !self.nonzero.is_empty();
    }

    /// The frame's total expected bits weight: the sum of every cell's
    /// [`bit_weight`], which rate control's complexity pass computes
    /// anyway (`Complexity`: its temporal sum, or spatial without a
    /// previous frame).
    pub fn set_total_weight(&mut self, total: f64) {
        self.total_weight = total;
    }

    /// Whether the planned frame has any non-zero offset.
    pub fn active(&self) -> bool {
        self.active
    }

    /// `(expected bits weight, offset in QP units)` of every cell with an
    /// offset: the frame's planned block offsets, for rate control's
    /// budgeting with [`Self::total_weight`]. Bits concentrate in active
    /// cells, so an area weight would let the many cheap flat cells that AQ
    /// refines dominate the estimate.
    pub fn offset_cells(&self) -> impl Iterator<Item = (f64, f64)> + '_ {
        self.nonzero
            .iter()
            .map(|&cell| (f64::from(self.bit_weight[cell]), self.offset[cell]))
    }

    /// Whether the reuse term is the only term (its offsets are planned
    /// sparsely, and the frame's total weight is set from outside).
    pub fn reuse_only(&self) -> bool {
        self.strength == 0. && self.persistence.is_none()
    }

    /// Expected bits weight of the whole frame (every cell's).
    pub fn total_weight(&self) -> f64 {
        self.total_weight
    }

    /// Expected bits weight of `block`: the sum over the cells it covers.
    pub fn block_weight(&self, block: &Block) -> f64 {
        if self.cols == 0 {
            return 0.;
        }
        let rows = self.bit_weight.len() / self.cols;
        let (width, height) = cell_extent(block);
        let (col0, row0) = (block.x >> CELL_LOG2, block.y >> CELL_LOG2);
        (row0..(row0 + height).min(rows))
            .flat_map(|row| (col0..(col0 + width).min(self.cols)).map(move |col| (row, col)))
            .map(|(row, col)| f64::from(self.bit_weight[row * self.cols + col]))
            .sum()
    }

    /// Offset of `block` in QP units: the mean offset of the cells it
    /// covers inside the picture.
    #[inline]
    pub fn offset_qp(&self, block: &Block) -> f64 {
        if self.cols == 0 {
            return 0.;
        }
        let (width, height) = cell_extent(block);
        let (col0, row0) = (block.x >> CELL_LOG2, block.y >> CELL_LOG2);
        // A block inside one quantizer square has the square's pooled
        // offset in every cell: read one (the mean over its cells, below,
        // was a few dozen instructions per leaf on every frame with an
        // offset).
        let side = 1usize << self.unit_log2.saturating_sub(CELL_LOG2);
        if width.max(height) <= side && self.unit_log2 > CELL_LOG2 {
            return self
                .offset
                .get(row0 * self.cols + col0)
                .copied()
                .unwrap_or(0.);
        }
        let rows = self.offset.len() / self.cols;
        let (mut sum, mut count) = (0., 0.);
        for row in row0..(row0 + height).min(rows) {
            for col in col0..(col0 + width).min(self.cols) {
                sum += self.offset[row * self.cols + col];
                count += 1.;
            }
        }
        if count == 0. {
            return 0.;
        }
        sum / count
    }
}

/// Give every cell of the `cols`-wide grid `values` the mean of its aligned
/// `1 << unit_log2` quantizer square (the cells of it inside the picture),
/// so each leaf and each cell of rate control's estimate carries the offset
/// the codec can code there.
fn pool_units(values: &mut [f64], cols: usize, unit_log2: u8) {
    let rows = values.len() / cols;
    let side = 1usize << (unit_log2 - CELL_LOG2);
    for row0 in (0..rows).step_by(side) {
        // A band of all-zero offsets pools to itself: the reuse term's few
        // refined cells leave nearly every band so (pooling them all cost
        // ~17 instructions per cell on every frame it refines anything).
        let band = row0 * cols..(row0 + side).min(rows) * cols;
        if values[band].iter().all(|&v| v == 0.) {
            continue;
        }
        for col0 in (0..cols).step_by(side) {
            let (row1, col1) = ((row0 + side).min(rows), (col0 + side).min(cols));
            let index = |row: usize, col: usize| row * cols + col;
            let sum: f64 = (row0..row1)
                .flat_map(|row| (col0..col1).map(move |col| (row, col)))
                .map(|(row, col)| values[index(row, col)])
                .sum();
            let mean = sum / ((row1 - row0) * (col1 - col0)) as f64;
            for row in row0..row1 {
                values[index(row, col0)..index(row, col1)].fill(mean);
            }
        }
    }
}

/// Expected share of the frame's bits of a cell: its residual activity,
/// temporal where the pre-analysis has a previous frame, spatial otherwise.
fn bit_weight(block: &BlockStats) -> f32 {
    1. + (block.sad_prev.unwrap_or(block.sum_sq_dev) as f32).sqrt()
}

/// `(width, height)` of `block` in cells (at least one each).
fn cell_extent(block: &Block) -> (usize, usize) {
    let cells = |log2: u8| (1usize << log2 >> CELL_LOG2).max(1);
    (cells(block.log2_width), cells(block.log2_height))
}

/// A QP-unit offset as a native quantizer offset from `quantizer`, within
/// `bounds`, through the codec's step table.
/// The grid step (QP units) that leaves the heuristic terms
/// `max_offsets - RESERVED_REUSE_OFFSETS` distinct non-zero values, evenly
/// either side of zero over `±max_offset`; `None` when the codec codes any
/// offset. Without it, a codec with a few per-frame offsets (VP9's
/// segments) handed them to the first values in coding order and coded the
/// rest at the nearest: the realised offsets missed rate control's plan
/// (VP9 variance AQ 1: key frames 17% over their budget, `J_real` +23%).
fn heuristic_offset_step(max_offset: f32, max_offsets: usize) -> Option<f64> {
    if max_offsets == usize::MAX {
        return None;
    }
    let per_side = max_offsets.saturating_sub(RESERVED_REUSE_OFFSETS) / 2;
    (per_side > 0).then(|| f64::from(max_offset) / per_side as f64)
}

pub(crate) fn native_offset(steps: &[f64], bounds: (u32, u32), quantizer: u32, offset: f64) -> i16 {
    if offset == 0. {
        return 0;
    }
    let target = steps[quantizer as usize] * step_ratio(offset);
    (i64::from(nearest_quantizer(steps, bounds, target)) - i64::from(quantizer)) as i16
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_analysis::grid::Grid;
    use nc_analysis::preanalysis::BlockStats;

    #[test]
    fn flat_blocks_get_finer_and_busy_blocks_coarser() {
        let mut o = ParamOverrides::default();
        o.set("rate.aq_strength", 1.);
        o.set("persist.reuse", 0.);
        let mut aq = BlockAq::new(&o, 0, usize::MAX).unwrap();
        // Left half flat (variance 4), right half busy (variance 256).
        let cells = (0..4)
            .map(|i| BlockStats {
                sum_sq_dev: if i % 2 == 0 { 4 * 64 } else { 256 * 64 },
                ..BlockStats::default()
            })
            .collect();
        aq.plan(
            &PreAnalysis {
                blocks: Grid::from_cells(2, 2, cells),
                static_fraction: 0.,
                mean_luma: 0.,
            },
            None,
            None,
        );
        let block = |x| Block::square(x, 0, 3);
        assert!((aq.offset_qp(&block(0)) + 3.).abs() < 1e-9);
        assert!((aq.offset_qp(&block(8)) - 3.).abs() < 1e-9);
        let steps: Vec<f64> = (0..52).map(|q| step_ratio(f64::from(q))).collect();
        assert_eq!(native_offset(&steps, (0, 51), 30, -3.), -3);
        assert_eq!(native_offset(&steps, (0, 51), 50, 3.), 1);
        let mut off = ParamOverrides::default();
        off.set("persist.reuse", 0.);
        assert!(BlockAq::new(&off, 0, usize::MAX).is_none());
    }

    #[test]
    fn few_offset_codecs_get_offsets_on_a_grid_that_fits() {
        let mut o = ParamOverrides::default();
        o.set("rate.aq_strength", 1.);
        o.set("persist.reuse", 0.);
        // Seven offsets (VP9's segments): three a side of zero, 8/3 apart.
        let mut aq = BlockAq::new(&o, 0, 7).unwrap();
        let cells = (0..16)
            .map(|i| BlockStats {
                sum_sq_dev: (1u32 << i) * 64,
                ..BlockStats::default()
            })
            .collect();
        aq.plan(
            &PreAnalysis {
                blocks: Grid::from_cells(16, 1, cells),
                static_fraction: 0.,
                mean_luma: 0.,
            },
            None,
            None,
        );
        let step = 8. / 3.;
        let mut levels: Vec<i64> = (0..16)
            .map(|x| aq.offset_qp(&Block::square(x * 8, 0, 3)) / step)
            .inspect(|level| assert!((level - level.round()).abs() < 1e-9))
            .map(|level| level.round() as i64)
            .collect();
        levels.dedup();
        assert_eq!(levels, [-3, -2, -1, 0, 1, 2, 3]);
        assert_eq!(heuristic_offset_step(8., usize::MAX), None);
    }

    #[test]
    fn leaves_share_the_offset_of_their_quantizer_square() {
        let mut o = ParamOverrides::default();
        o.set("rate.aq_strength", 1.);
        o.set("persist.reuse", 0.);
        let mut aq = BlockAq::new(&o, 4, usize::MAX).unwrap();
        // One 16x16 square: a flat and a busy column; a busy square beside it.
        let cells = (0..8)
            .map(|i| BlockStats {
                sum_sq_dev: if i % 4 == 0 { 4 * 64 } else { 256 * 64 },
                ..BlockStats::default()
            })
            .collect();
        aq.plan(
            &PreAnalysis {
                blocks: Grid::from_cells(4, 2, cells),
                static_fraction: 0.,
                mean_luma: 0.,
            },
            None,
            None,
        );
        let leaf = |x, y| aq.offset_qp(&Block::square(x, y, 3));
        let square = aq.offset_qp(&Block::square(0, 0, 4));
        for (x, y) in [(0, 0), (8, 0), (0, 8), (8, 8)] {
            assert!((leaf(x, y) - square).abs() < 1e-9);
        }
        assert!(leaf(0, 0) < leaf(16, 0));
    }
}
