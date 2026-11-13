//! Persistent stillness: which 8x8 blocks are reliably static, for the
//! encoder's static-content levers (zero-vector shortcut, disposable
//! frames).
//!
//! Pre-analysis' `is_static` compares a block with the same block of the
//! reference source only. On its own that is not reliable enough to skip
//! work on (`docs/handoff/static-detect.md`):
//!
//! - slow or low-contrast movers (small people in a crowd, a slow pan over
//!   flat areas) stay under the per-frame threshold while drifting;
//! - the edges of moving objects leave single static blocks next to motion.
//!
//! A block is *still* when
//!
//! 1. it has been static for at least `min_run` consecutive analysed frames
//!    (hysteresis: entering takes several frames, one non-static frame
//!    leaves at once), and
//! 2. every block within `guard` blocks of it is static in this frame (a
//!    moving object's neighbourhood never counts).
//!
//! The tracker only reads pre-analysis results, so it costs a few
//! operations per block. Its state is per instance (one per stream).

use crate::grid::Grid;
use crate::preanalysis::PreAnalysis;

/// How strict the stillness test is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StillConfig {
    /// Consecutive static frames a block needs before it counts as still.
    pub min_run: u8,
    /// Neighbourhood radius, in blocks, that must also be static now.
    pub guard: u8,
}

impl Default for StillConfig {
    fn default() -> Self {
        Self {
            min_run: DEFAULT_MIN_RUN,
            guard: DEFAULT_GUARD,
        }
    }
}

/// Default [`StillConfig::min_run`] (see `docs/handoff/static-detect.md`
/// for the classifier measurements behind it).
pub const DEFAULT_MIN_RUN: u8 = 8;
/// Default [`StillConfig::guard`].
pub const DEFAULT_GUARD: u8 = 1;

/// One frame's stillness, on the pre-analysis 8x8 grid.
#[derive(Debug, Clone, PartialEq)]
pub struct StillMap {
    /// Per-block stillness.
    pub blocks: Grid<bool>,
    /// Consecutive static frames of every still block, saturating at 255;
    /// 0 where the block is not still. How long a block has been reused is
    /// the encoder's estimate of how long it will be (`reuse.rs`).
    pub runs: Grid<u8>,
    /// Consecutive static frames of every block, still or not, saturating
    /// at 255 (0: it changed in this frame).
    pub static_runs: Grid<u8>,
    /// Fraction of blocks that are still.
    pub fraction: f32,
}

/// Per-block static run lengths carried between frames, plus scratch.
#[derive(Debug, Clone)]
pub struct StillTracker {
    cols: usize,
    /// Consecutive static frames per block, raster order.
    runs: Vec<u8>,
    /// 1 where a block (or, once dilated, a neighbour) is not static.
    moving: Vec<u8>,
    /// Dilation scratch, the size of `moving`.
    scratch: Vec<u8>,
}

impl StillTracker {
    /// A tracker for a `cols` x `rows` block grid, with no history.
    #[must_use]
    pub fn new(cols: usize, rows: usize) -> Self {
        let cells = cols * rows;
        Self {
            cols,
            runs: vec![0; cells],
            moving: vec![0; cells],
            scratch: vec![0; cells],
        }
    }

    /// Advance the runs with this frame's pre-analysis and return its
    /// stillness under `config`. Scene cuts need no reset: every block of
    /// a cut is non-static, which restarts its run.
    ///
    /// # Panics
    ///
    /// If `pre` is not on this tracker's grid.
    pub fn update(&mut self, pre: &PreAnalysis, config: StillConfig) -> StillMap {
        let cells = pre.blocks.cells();
        assert!(
            pre.blocks.cols() == self.cols && cells.len() == self.runs.len(),
            "pre-analysis grid does not match the tracker"
        );
        // Plain byte arithmetic over flat slices (branch-free, so it
        // vectorises): the first version, with per-block window slices and
        // grid lookups, cost 2.5% of an HEVC Fast P frame.
        for ((run, moving), block) in self.runs.iter_mut().zip(&mut self.moving).zip(cells) {
            let is_static = u8::from(block.is_static);
            *run = run.saturating_add(1) * is_static;
            *moving = 1 - is_static;
        }
        for _ in 0..config.guard {
            dilate(&mut self.moving, &mut self.scratch, self.cols);
        }
        let still: Vec<bool> = self
            .runs
            .iter()
            .zip(&self.moving)
            .map(|(&run, &moving)| moving == 0 && run >= config.min_run)
            .collect();
        let count = still.iter().filter(|&&s| s).count();
        let runs = self
            .runs
            .iter()
            .zip(&still)
            .map(|(&run, &s)| if s { run } else { 0 })
            .collect();
        let rows = still.len() / self.cols.max(1);
        StillMap {
            fraction: count as f32 / still.len().max(1) as f32,
            blocks: Grid::from_cells(self.cols, rows, still),
            runs: Grid::from_cells(self.cols, rows, runs),
            static_runs: Grid::from_cells(self.cols, rows, self.runs.clone()),
        }
    }
}

/// Grow the set flags of the `cols`-wide raster `cells` by one block in
/// every direction (a 3x3 dilation clipped to the grid), rows first.
fn dilate(cells: &mut [u8], scratch: &mut [u8], cols: usize) {
    let inner = cols - 1;
    for (src, dst) in cells.chunks_exact(cols).zip(scratch.chunks_exact_mut(cols)) {
        dst.copy_from_slice(src);
        for (d, s) in dst[1..].iter_mut().zip(&src[..inner]) {
            *d |= s;
        }
        for (d, s) in dst[..inner].iter_mut().zip(&src[1..]) {
            *d |= s;
        }
    }
    cells.copy_from_slice(scratch);
    let rows: Vec<&[u8]> = scratch.chunks_exact(cols).collect();
    for (row, out) in cells.chunks_exact_mut(cols).enumerate() {
        let neighbours = [row.checked_sub(1), Some(row + 1)];
        for near in neighbours.into_iter().flatten().filter_map(|r| rows.get(r)) {
            for (d, s) in out.iter_mut().zip(*near) {
                *d |= s;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preanalysis::BlockStats;

    fn pre_with(cols: usize, rows: usize, moving: &[(usize, usize)]) -> PreAnalysis {
        let mut blocks = Grid::filled(
            cols,
            rows,
            BlockStats {
                is_static: true,
                ..BlockStats::default()
            },
        );
        for &(c, r) in moving {
            blocks.get_mut(c, r).is_static = false;
        }
        PreAnalysis {
            blocks,
            static_fraction: 0.0,
            mean_luma: 0.0,
        }
    }

    #[test]
    fn stillness_needs_a_run_of_static_frames() {
        let config = StillConfig {
            min_run: 3,
            guard: 0,
        };
        let mut tracker = StillTracker::new(4, 4);
        let pre = pre_with(4, 4, &[]);
        assert_eq!(tracker.update(&pre, config).fraction, 0.0);
        assert_eq!(tracker.update(&pre, config).fraction, 0.0);
        assert_eq!(tracker.update(&pre, config).fraction, 1.0);
        // One non-static frame restarts the block's run.
        let moved = tracker.update(&pre_with(4, 4, &[(1, 1)]), config);
        assert!(!*moved.blocks.get(1, 1));
        assert!(!*tracker.update(&pre, config).blocks.get(1, 1));
    }

    #[test]
    fn the_neighbourhood_of_motion_is_not_still() {
        let config = StillConfig {
            min_run: 1,
            guard: 1,
        };
        let mut tracker = StillTracker::new(6, 6);
        let map = tracker.update(&pre_with(6, 6, &[(2, 2)]), config);
        for (col, row) in map.blocks.positions() {
            let near = col.abs_diff(2) <= 1 && row.abs_diff(2) <= 1;
            assert_eq!(*map.blocks.get(col, row), !near, "block ({col}, {row})");
        }
    }

    #[test]
    fn a_wider_guard_reaches_further_and_clips_at_the_edges() {
        let config = StillConfig {
            min_run: 1,
            guard: 2,
        };
        let mut tracker = StillTracker::new(7, 5);
        let map = tracker.update(&pre_with(7, 5, &[(0, 4)]), config);
        for (col, row) in map.blocks.positions() {
            let near = col <= 2 && row >= 2;
            assert_eq!(*map.blocks.get(col, row), !near, "block ({col}, {row})");
        }
    }
}
