//! The split override (`planner.split_override`): one shared test that
//! makes every early "do not split" exit split a block whose quadrants the
//! motion search found moving apart and predicted much better on their own.
//!
//! The exits it overrides each judge a block by its whole-block evidence:
//! the inter floor (`planner.inter_min_log2`, regular and `light.`), SAD
//! depth removal (`planner.depth_removal_*`), the plain-variance split
//! decision with the selector off (`planner.split_selector` 0), and
//! H.264's uniform-vector and SATD gates on its 8x8 partition search. Screen
//! text whose row pitch is not a multiple of the block size defeats all of
//! them: rows snap by 0 or 1 sample on different frames, so the block's SAD
//! at one vector is small (sparse glyphs) and the quadrant vectors lie
//! within one sample of each other, while each quadrant at its own vector
//! matches almost exactly. On the demo's scrolling-text scene (854x480, CBR
//! 400k, 120 f) these exits cost 6 to 11 dB of luma on every codec's faster
//! rungs and the floor alone 6 to 9 dB at medium (ledger idea
//! `scrolling-text-split-exits-across-rungs`).
//!
//! The test reads only the motion analysis's own fields: the quadrants'
//! whole-sample search vectors and the aggregated field's `split_sad`
//! (the quadrants' summed best SADs) against `whole_sad` (the block at its
//! one best vector). What it skips nowhere: it only adds splits. Why that is
//! safe on camera content: noise and texture rarely leave quadrants that each
//! match far better at their own whole-sample vector, so the SAD share test
//! stays closed there (the vector test alone opened on camera noise, L1 J
//! +0.25..+1.06 %, ledger `motion-gated-inter-floor`).
//!
//! A second trigger, `owed`, is a plain input: a caller that knows the
//! block holds cells owed a refinement (the reuse term's debts after a key)
//! opens the split with it, whatever the motion says.

use nc_analysis::motion::AggregatedBlock;
use nc_analysis::{MotionAnalysis, Mv};

use crate::params::PlannerParams;

/// log2 of the smallest block the aggregated motion fields describe (16x16);
/// its quadrants are the 8x8 fine field.
const AGGREGATED_MIN_LOG2: u8 = 4;
/// log2 of the largest (64x64).
const AGGREGATED_MAX_LOG2: u8 = 6;
/// log2 of the fine field's blocks (8x8).
const FINE_LOG2: u8 = 3;
/// Largest SAD share a step-scaled share may reach: a split must still
/// predict the quadrants clearly better than the block.
const MAX_SAD_SHARE: f32 = 0.9;

/// log2 of the cells an [`OwedCells`] map holds (8x8, the reuse term's).
const OWED_CELL_LOG2: u8 = 3;

/// The cells (8x8, raster order, `cols` a row) owed a refinement in the
/// frame being planned: the override's second trigger.
#[derive(Debug, Clone, Copy)]
pub(crate) struct OwedCells<'a> {
    pub cells: &'a [bool],
    pub cols: usize,
}

impl OwedCells<'_> {
    /// Whether any cell of the square block at (`x`, `y`) of size
    /// `1 << log2` is owed.
    pub fn any_in(&self, (x, y, log2): (usize, usize, u8)) -> bool {
        let side = 1usize << log2.saturating_sub(OWED_CELL_LOG2);
        let (col0, row0) = (x >> OWED_CELL_LOG2, y >> OWED_CELL_LOG2);
        let cols = col0..(col0 + side).min(self.cols);
        (row0..row0 + side).any(|row| {
            cols.clone().any(|col| {
                self.cells
                    .get(row * self.cols + col)
                    .copied()
                    .unwrap_or(false)
            })
        })
    }
}

/// What the override does at one block ([`SplitOverride::opening`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Opening {
    /// The exits apply as they are.
    Closed,
    /// The block may be split past the floor; the planner's cost decides.
    Allow,
    /// The block is split.
    Force,
}

/// The four quadrants of the square block at (`x`, `y`) of size `1 << log2`.
fn quadrants((x, y, log2): (usize, usize, u8)) -> [(usize, usize, u8); 4] {
    let half = 1usize << (log2 - 1);
    std::array::from_fn(|i| (x + (i & 1) * half, y + (i >> 1) * half, log2 - 1))
}

/// Levels of a [`Survey`]: 16x16, 32x32 and 64x64.
const LEVELS: usize = (AGGREGATED_MAX_LOG2 - AGGREGATED_MIN_LOG2 + 1) as usize;
/// Largest superblock a [`Survey`] covers (128x128: 64 blocks of 16x16 in a
/// `u64`).
const SURVEY_MAX_LOG2: u8 = 7;

/// What the override found in one superblock ([`SplitOverride::survey`]):
/// per level, a bit per block in raster order within the superblock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Survey {
    origin: (usize, usize),
    log2: u8,
    /// Blocks whose own quadrants move apart.
    apart: [u64; LEVELS],
    /// Blocks that are apart or hold a smaller block that is.
    within: [u64; LEVELS],
    /// Whether the ancestors of a block apart are split
    /// (`planner.split_override_force`), else offered to the selector.
    force_ancestors: bool,
}

impl Survey {
    /// What the override does at the square block at (`x`, `y`) of size
    /// `1 << log2` inside the superblock:
    /// - [`Opening::Force`] where the block holds `owed` cells, or (with
    ///   `planner.split_override_force`) a smaller block whose quadrants move
    ///   apart. Forcing those ancestors closed the text scene no better than
    ///   offering them to the selector (default) and cost camera clips
    ///   (pool quick, AV1 medium L1 J +0.96 %);
    /// - [`Opening::Allow`] where the block's own quadrants move apart, or
    ///   (by default) it holds a smaller block whose quadrants do: the exits
    ///   above that block (a large leaf chosen early, depth removal) would
    ///   never let the planner reach it ([`SplitOverride::motion_apart`]).
    ///   The split is offered past the floor and depth removal, decided by
    ///   the split selector's evidence (forcing the block itself lost
    ///   0.6-0.9 dB on rungs whose floor is already 8x8);
    /// - [`Opening::Closed`] otherwise.
    pub fn opening(&self, square: (usize, usize, u8), owed: bool) -> Opening {
        let below =
            square.2 > AGGREGATED_MIN_LOG2 && quadrants(square).iter().any(|&q| self.within_at(q));
        if owed || (below && self.force_ancestors) {
            Opening::Force
        } else if below || self.bit(&self.apart, square) {
            Opening::Allow
        } else {
            Opening::Closed
        }
    }

    /// Whether the block is apart or holds a smaller block that is.
    fn within_at(&self, square: (usize, usize, u8)) -> bool {
        self.bit(&self.within, square)
    }

    fn bit(&self, map: &[u64; LEVELS], (x, y, level): (usize, usize, u8)) -> bool {
        if !(AGGREGATED_MIN_LOG2..=AGGREGATED_MAX_LOG2).contains(&level) || level > self.log2 {
            return false;
        }
        let (Some(dx), Some(dy)) = (x.checked_sub(self.origin.0), y.checked_sub(self.origin.1))
        else {
            return false;
        };
        let side = 1usize << (self.log2 - level);
        let (col, row) = (dx >> level, dy >> level);
        col < side
            && row < side
            && map[usize::from(level - AGGREGATED_MIN_LOG2)] >> (row * side + col) & 1 != 0
    }
}

/// Whether two of the quadrants' search vectors (`None` where none was
/// measured) differ by at least `samples` whole samples in some component.
fn vectors_apart(vectors: [Option<Mv>; 4], samples: i32) -> bool {
    vectors.iter().enumerate().any(|(i, u)| {
        vectors[i + 1..].iter().any(|v| match (u, v) {
            (Some(u), Some(v)) => {
                let dx = (i32::from(u.x) - i32::from(v.x)).abs();
                let dy = (i32::from(u.y) - i32::from(v.y)).abs();
                dx.max(dy) >= samples
            }
            _ => false,
        })
    })
}

/// The override's thresholds, from `planner.split_override_*`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SplitOverride {
    /// Whole samples two quadrant vectors must differ by (in some component).
    samples: i32,
    /// Largest share of the block's SAD its quadrants' summed SADs may be.
    sad_share: f32,
    /// Share added per quantizer step of the block's SAD per sample: a block
    /// whose one-vector residual is large against the step pays for a split
    /// at a smaller relative gain.
    share_per_step: f32,
    /// Smallest SAD per sample the quadrants' own vectors must save (0 by
    /// default: 0.25 and up kept camera streams unchanged but cost the
    /// scrolling-text scene 1.2-1.9 dB, its glyphs being sparse).
    min_gain: f32,
    /// [`Survey::force_ancestors`].
    force_ancestors: bool,
}

impl SplitOverride {
    /// The override `params` configure, `None` while it is off.
    pub fn of(params: &PlannerParams) -> Option<Self> {
        (params.split_override >= 0.5).then(|| Self {
            samples: params.split_override_samples.round().max(1.0) as i32,
            sad_share: params.split_override_sad_share,
            share_per_step: params.split_override_share_per_step,
            min_gain: params.split_override_min_gain,
            force_ancestors: params.split_override_force >= 0.5,
        })
    }

    /// The override's findings over the superblock at (`x`, `y`) of size
    /// `1 << log2`, measured once for every node the planner visits in it.
    /// `step` is the frame quantizer's sample step.
    pub fn survey(
        &self,
        motion: Option<&MotionAnalysis>,
        (x, y, log2): (usize, usize, u8),
        step: f32,
    ) -> Survey {
        let mut survey = Survey {
            origin: (x, y),
            log2,
            apart: [0; LEVELS],
            within: [0; LEVELS],
            force_ancestors: self.force_ancestors,
        };
        if !(AGGREGATED_MIN_LOG2..=SURVEY_MAX_LOG2).contains(&log2) {
            return survey;
        }
        for level in AGGREGATED_MIN_LOG2..=log2.min(AGGREGATED_MAX_LOG2) {
            let side = 1usize << (log2 - level);
            let index = usize::from(level - AGGREGATED_MIN_LOG2);
            for row in 0..side {
                for col in 0..side {
                    let square = (x + (col << level), y + (row << level), level);
                    let bit = 1u64 << (row * side + col);
                    if self.motion_apart(motion, square, step) {
                        survey.apart[index] |= bit;
                    }
                    let below = level > AGGREGATED_MIN_LOG2
                        && quadrants(square).iter().any(|&q| survey.within_at(q));
                    if survey.apart[index] & bit != 0 || below {
                        survey.within[index] |= bit;
                    }
                }
            }
        }
        survey
    }

    /// Whether two of the block's quadrant search vectors differ by at least
    /// `samples` whole samples and the quadrants' summed best SADs are at
    /// most the SAD share of the block's SAD at its one vector.
    fn motion_apart(
        &self,
        motion: Option<&MotionAnalysis>,
        (x, y, log2): (usize, usize, u8),
        step: f32,
    ) -> bool {
        let Some(motion) = motion else {
            return false;
        };
        if !(AGGREGATED_MIN_LOG2..=AGGREGATED_MAX_LOG2).contains(&log2) {
            return false;
        }
        let field = &motion.aggregated[usize::from(log2 - AGGREGATED_MIN_LOG2)];
        let Some(block) = field
            .blocks
            .try_get((x >> log2) as isize, (y >> log2) as isize)
        else {
            return false;
        };
        // The SAD test first: one read, and it closes nearly every camera
        // block before the four vectors are fetched.
        if !self.predicted_apart(*block, log2, step) {
            return false;
        }
        let vector = |(qx, qy, child): (usize, usize, u8)| {
            let (col, row) = ((qx >> child) as isize, (qy >> child) as isize);
            if child == FINE_LOG2 {
                motion.fine.try_get(col, row).map(|b| b.mv)
            } else {
                motion.aggregated[usize::from(child - AGGREGATED_MIN_LOG2)]
                    .blocks
                    .try_get(col, row)
                    .map(|b| b.mv)
            }
        };
        vectors_apart(quadrants((x, y, log2)).map(vector), self.samples)
    }

    /// Whether the quadrants' summed best SADs are at most the SAD share of
    /// the block's SAD at its one vector, saving at least `min_gain` a
    /// sample. A block whose SAD is zero passes: the full-pel search reads
    /// every other row (`nc_analysis::motion`), so thin glyph rows can hide
    /// between them (requiring a nonzero saving cost 1.2-1.9 dB on the
    /// scrolling-text scene).
    fn predicted_apart(&self, block: AggregatedBlock, log2: u8, step: f32) -> bool {
        let share = self.share(block.whole_sad, log2, step);
        let gain = block.whole_sad.saturating_sub(block.split_sad);
        block.split_sad as f32 <= share * block.whole_sad as f32
            && gain as f32 >= self.min_gain * (1u32 << (2 * log2)) as f32
    }

    /// The SAD share for a block whose SAD at its one vector is `whole_sad`.
    fn share(&self, whole_sad: u32, log2: u8, step: f32) -> f32 {
        if self.share_per_step <= 0.0 || step <= 0.0 {
            return self.sad_share;
        }
        let per_sample = whole_sad as f32 / (1u32 << (2 * log2)) as f32;
        (self.sad_share + self.share_per_step * per_sample / step).min(MAX_SAD_SHARE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn override_with(share: f32) -> SplitOverride {
        SplitOverride {
            samples: 1,
            sad_share: share,
            share_per_step: 0.0,
            min_gain: 0.0,
            force_ancestors: true,
        }
    }

    #[test]
    fn owed_cells_are_found_inside_the_block_only() {
        // 4x2 cells, one owed at column 2, row 1.
        let cells = [false, false, false, false, false, false, true, false];
        let owed = OwedCells {
            cells: &cells,
            cols: 4,
        };
        assert!(owed.any_in((16, 0, 4)));
        assert!(!owed.any_in((0, 0, 4)));
        assert!(owed.any_in((0, 0, 5)));
    }

    #[test]
    fn owed_cells_force_whatever_the_motion() {
        let survey = override_with(0.1).survey(None, (0, 0, 6), 8.0);
        assert_eq!(survey.opening((0, 0, 4), true), Opening::Force);
        assert_eq!(survey.opening((0, 0, 5), false), Opening::Closed);
    }

    #[test]
    fn a_block_apart_forces_its_ancestors_and_allows_itself() {
        let mut survey = override_with(0.1).survey(None, (64, 0, 6), 8.0);
        // The 16x16 at (80, 16) inside the superblock at (64, 0).
        survey.apart[0] = 1 << 5;
        survey.within = [1 << 5, 1, 1];
        assert_eq!(survey.opening((80, 16, 4), false), Opening::Allow);
        assert_eq!(survey.opening((64, 0, 5), false), Opening::Force);
        assert_eq!(survey.opening((64, 0, 6), false), Opening::Force);
        assert_eq!(survey.opening((96, 0, 5), false), Opening::Closed);
    }

    #[test]
    fn quadrants_tile_the_block() {
        assert_eq!(
            quadrants((32, 0, 5)),
            [(32, 0, 4), (48, 0, 4), (32, 16, 4), (48, 16, 4)]
        );
    }

    #[test]
    fn the_step_share_grows_with_the_residual_and_is_capped() {
        let o = SplitOverride {
            samples: 1,
            sad_share: 0.1,
            share_per_step: 0.1,
            min_gain: 0.0,
            force_ancestors: true,
        };
        // 16x16 at 4 SAD per sample, step 8: 0.1 + 0.1 * 0.5.
        assert!((o.share(1024, 4, 8.0) - 0.15).abs() < 1e-6);
        assert_eq!(o.share(u32::MAX, 4, 8.0), MAX_SAD_SHARE);
    }

    #[test]
    fn quadrants_apart_and_predicted_better_open() {
        let o = override_with(0.1);
        let mv = |y| Some(Mv { x: 0, y });
        let block = |split_sad| AggregatedBlock {
            mv: Mv { x: 0, y: 0 },
            whole_sad: 400,
            split_sad,
        };
        // The quadrants match far better at their own vectors: open...
        assert!(o.predicted_apart(block(40), 4, 8.0));
        // ...but not at a weak SAD gain (noise, texture).
        assert!(!o.predicted_apart(block(41), 4, 8.0));
        // One quadrant moved a whole sample: apart.
        assert!(vectors_apart([mv(0), mv(1), mv(0), mv(0)], 1));
        // One vector for all: not apart.
        assert!(!vectors_apart([mv(1); 4], 1));
        // A quadrant without a measurement never counts as apart.
        assert!(!vectors_apart([mv(0), None, None, None], 1));
    }
}
