//! Motion-search effort counters.
//!
//! How much searching a frame took, per stage, without any profiler: how
//! many superblocks took each search path, and how many positions the
//! hierarchical (quarter-resolution) and full-pel stages evaluated. They cost
//! a few integer additions per SB, so they are always on.

use crate::motion::SbSearch;

/// Effort of one search stage over a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StageEffort {
    /// Superblocks this stage searched.
    pub blocks: u32,
    /// Positions evaluated: block SADs (HME, quarter resolution, every
    /// other row) or 64x64 8x8-grid evaluations (full-pel).
    pub positions: u32,
}

impl StageEffort {
    /// Mean positions per searched SB.
    #[must_use]
    pub fn mean_positions(&self) -> f32 {
        if self.blocks == 0 {
            0.0
        } else {
            self.positions as f32 / self.blocks as f32
        }
    }

    pub(crate) fn add_block(&mut self, positions: u32) {
        self.blocks += 1;
        self.positions += positions;
    }
}

/// Effort of the whole motion search for one frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MotionEffort {
    /// SBs per search path: [static, one position, window at zero,
    /// hierarchical] (see `SbSearch`).
    pub sb_paths: [u32; 4],
    /// Quarter-resolution window search.
    pub hme: StageEffort,
    /// Seed candidates checked for the HME centre (one SAD each).
    pub hme_seeds: u32,
    /// Half-resolution refinement (positions: 32x32 SADs, every other row).
    pub hme_level1: StageEffort,
    /// Full-pel window search.
    pub full_pel: StageEffort,
    /// SBs with mixed motion that got the extra candidates.
    pub mixed_sbs: u32,
}

impl MotionEffort {
    pub(crate) fn record_sb(&mut self, search: SbSearch) {
        self.sb_paths[search as usize] += 1;
    }

    /// Accumulate another frame's counters.
    pub fn accumulate(&mut self, other: &MotionEffort) {
        for (a, b) in self.sb_paths.iter_mut().zip(other.sb_paths) {
            *a += b;
        }
        for (a, b) in [
            (&mut self.hme, &other.hme),
            (&mut self.hme_level1, &other.hme_level1),
            (&mut self.full_pel, &other.full_pel),
        ] {
            a.blocks += b.blocks;
            a.positions += b.positions;
        }
        self.hme_seeds += other.hme_seeds;
        self.mixed_sbs += other.mixed_sbs;
    }
}
