//! Which blocks try the second reference (`refs.second_screen_ratio`,
//! `docs/handoff/tuning.md` "Second reference: where to try it").
//!
//! The second reference (`second_reference.rs`) costs a second mode
//! decision on every block it is offered to, but it wins on few of them
//! (about one tried block in nine, AV1 Default, fixed QP, 35 pool clips),
//! and its gain is concentrated: the best 10% of tries hold over 90% of
//! the rate-distortion saving. This screen predicts those blocks from
//! statistics that cost a few whole-sample SADs per block and a bit per
//! 8x8 cell of the previous frame, and runs the full decision only there:
//!
//! - **SAD ratio.** The better of the second reference's whole-sample SADs
//!   at the zero vector and at the primary search vector scaled to its
//!   distance, over the primary reference's SAD at the primary choice's
//!   vector. Below about 1 the HQ frame already matches the source as well
//!   as the primary's own winner does.
//! - **Room left.** The primary choice's cost per sample in λ: a block the
//!   primary reference predicts poorly has more for a finer frame to win,
//!   so the threshold grows with it (a power law, `second_screen_cost_exponent`).
//! - **Persistence.** Wins persist from frame to frame (45% of tries next to
//!   a previous-frame win win again, 5% elsewhere): near one (the block or
//!   its one-cell ring), the threshold is multiplied by
//!   `second_screen_persist`.
//!
//! The rule is `sad_ratio < ratio · cost^exponent · (persist if near a win)`,
//! fitted on traced per-block decisions (log-linear score, held-out clips
//! and quantizer). Regions re-qualify on their own: the SAD test is
//! independent of the map, so a block whose HQ match improves is tried
//! again without any exploration schedule.
//!
//! Measured (AV1 Default CBR, `--clips adopt` 2026-11-14, against Default;
//! Jetson cycles at 31 f, crowd / bodycam / lowlight 720p): the full arm
//! L2 J -1.95% at +19..+24% cycles; ratio 3.0 L2 -1.60% / L3 -1.64% at
//! +10..+18%;
//! ratio 1.5 L3 -0.63% at +5..+7%. The screen alone costs about +4% Jetson
//! cycles (cache misses on the HQ picture), so it cannot reach the cheap end
//! of the curve (`docs/handoff/tuning.md`, "Second reference: where to try
//! it").

use crate::params::RefParams;

/// Edge of a win-map cell, in luma samples (the pre-analysis grid).
const CELL_LOG2: usize = 3;
/// A block is near a previous win when one lies within this many cells.
const NEAR_CELLS: usize = 1;
/// Cost per sample (in λ) clamped into this range before the power law:
/// a cached planner choice reaches here only as a finite cost, but a
/// zero-cost block must not zero the threshold.
const MIN_COST_PER_SAMPLE: f32 = 1e-3;
const MAX_COST_PER_SAMPLE: f32 = 1e3;

/// The screen's settings and its per-cell win maps.
#[derive(Debug, Clone)]
pub(crate) struct SecondScreen {
    /// `refs.second_screen_ratio`: the SAD ratio a block at one λ per sample,
    /// far from previous wins, must stay under.
    ratio: f32,
    /// `refs.second_screen_cost_exponent`.
    cost_exponent: f32,
    /// `refs.second_screen_persist`.
    persist: f32,
    cols: usize,
    rows: usize,
    /// The frame whose wins `current` holds.
    frame: Option<u64>,
    /// Cells where the previous picture chose the second reference (alone
    /// or in compound).
    previous: Vec<bool>,
    /// Cells where the picture being coded chose it so far.
    current: Vec<bool>,
}

impl SecondScreen {
    /// The screen `params` configure, or `None` when it is off.
    pub(crate) fn new(params: &RefParams) -> Option<Self> {
        (params.second_screen_ratio > 0.0).then(|| Self {
            ratio: params.second_screen_ratio,
            cost_exponent: params.second_screen_cost_exponent,
            persist: params.second_screen_persist,
            cols: 0,
            rows: 0,
            frame: None,
            previous: Vec::new(),
            current: Vec::new(),
        })
    }

    /// Frame `index`, of `width` x `height` luma samples, starts a coding.
    /// The first coding of a new frame makes the last coding's wins the
    /// previous picture's; a re-coding of the same frame forgets its own.
    /// A size change forgets both.
    pub(crate) fn begin_picture(&mut self, index: u64, (width, height): (usize, usize)) {
        let (cols, rows) = (cells(width), cells(height));
        if (cols, rows) == (self.cols, self.rows) {
            if self.frame != Some(index) {
                std::mem::swap(&mut self.previous, &mut self.current);
            }
            self.current.fill(false);
        } else {
            (self.cols, self.rows) = (cols, rows);
            self.previous = vec![false; cols * rows];
            self.current = vec![false; cols * rows];
        }
        self.frame = Some(index);
    }

    /// Block (`x`, `y`, `width` x `height`) chose the second reference.
    pub(crate) fn record_win(&mut self, (x, y): (usize, usize), (width, height): (usize, usize)) {
        let (c0, c1) = (x >> CELL_LOG2, cells(x + width).min(self.cols));
        let (r0, r1) = (y >> CELL_LOG2, cells(y + height).min(self.rows));
        for row in r0..r1 {
            self.current[row * self.cols + c0..row * self.cols + c1].fill(true);
        }
    }

    /// Whether the previous picture chose the second reference within
    /// [`NEAR_CELLS`] of the block.
    fn near_win(&self, (x, y): (usize, usize), (width, height): (usize, usize)) -> bool {
        let c0 = (x >> CELL_LOG2).saturating_sub(NEAR_CELLS);
        let c1 = (cells(x + width) + NEAR_CELLS).min(self.cols);
        let r0 = (y >> CELL_LOG2).saturating_sub(NEAR_CELLS);
        let r1 = (cells(y + height) + NEAR_CELLS).min(self.rows);
        c0 < c1
            && (r0..r1).any(|row| {
                self.previous[row * self.cols + c0..row * self.cols + c1]
                    .iter()
                    .any(|&won| won)
            })
    }

    /// Whether the block tries the second reference: its best second
    /// reference SAD `secondary_sad` against `primary_sad`, the primary
    /// choice's SAD, given that choice's cost per sample in λ.
    pub(crate) fn worth_trying(
        &self,
        block: ((usize, usize), (usize, usize)),
        (secondary_sad, primary_sad): (u32, u32),
        cost_per_sample: f32,
    ) -> bool {
        let room = cost_per_sample
            .clamp(MIN_COST_PER_SAMPLE, MAX_COST_PER_SAMPLE)
            .powf(self.cost_exponent);
        let (secondary_sad, limit) = (
            secondary_sad as f32,
            self.ratio * room * primary_sad.max(1) as f32,
        );
        // The win map is read only when the persistence bonus decides.
        secondary_sad < limit
            || (secondary_sad < self.persist * limit && self.near_win(block.0, block.1))
    }
}

/// Cells covering `samples` luma samples.
fn cells(samples: usize) -> usize {
    samples.div_ceil(1 << CELL_LOG2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::ParamOverrides;

    fn screen() -> SecondScreen {
        let defaults = RefParams::from_overrides(&ParamOverrides::default());
        SecondScreen::new(&RefParams {
            second_screen_ratio: 1.0,
            second_screen_cost_exponent: 0.0,
            second_screen_persist: 2.0,
            ..defaults
        })
        .expect("on")
    }

    #[test]
    fn off_by_default() {
        assert!(
            SecondScreen::new(&RefParams::from_overrides(&ParamOverrides::default())).is_none()
        );
    }

    #[test]
    fn sad_ratio_and_persistence_decide() {
        let mut s = screen();
        s.begin_picture(0, (64, 64));
        let block = ((16, 16), (16, 16));
        assert!(s.worth_trying(block, (90, 100), 1.0));
        assert!(!s.worth_trying(block, (150, 100), 1.0));
        // A win next to the block in the previous picture doubles the limit.
        s.record_win((32, 16), (8, 8));
        s.begin_picture(1, (64, 64));
        assert!(s.worth_trying(block, (150, 100), 1.0));
        // Wins two cells away do not count.
        s.record_win((48, 48), (16, 16));
        s.begin_picture(2, (64, 64));
        assert!(!s.worth_trying(block, (150, 100), 1.0));
    }

    #[test]
    fn recoding_forgets_the_pictures_own_wins() {
        let mut s = screen();
        s.begin_picture(0, (32, 32));
        s.record_win((0, 0), (8, 8));
        s.begin_picture(0, (32, 32));
        s.begin_picture(1, (32, 32));
        assert!(!s.worth_trying(((0, 0), (8, 8)), (150, 100), 1.0));
    }

    #[test]
    fn room_raises_the_limit() {
        let defaults = RefParams::from_overrides(&ParamOverrides::default());
        let mut s = SecondScreen::new(&RefParams {
            second_screen_ratio: 1.0,
            second_screen_cost_exponent: 0.5,
            ..defaults
        })
        .expect("on");
        s.begin_picture(0, (16, 16));
        let block = ((0, 0), (16, 16));
        assert!(!s.worth_trying(block, (150, 100), 1.0));
        assert!(s.worth_trying(block, (150, 100), 4.0));
    }
}
