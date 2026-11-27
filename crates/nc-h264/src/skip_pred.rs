//! Neighbour-predicted screen of the zero-residual luma test (`h264.skip_pred_sad`).
//!
//! The inter zero-residual test (`macroblock::residual_skip_views`) passes a
//! macroblock's luma when each of its sixteen 4x4 SADs is within the
//! quantizer's bound. This screen judges the whole 16x16 luma SAD (the sum of
//! those sixteen, taken from the same SAD grid, so it costs no extra SAD)
//! against the 16x16 SADs of the left, top and top-right macroblocks of the
//! frame: a macroblock much quieter than its neighbours passes as a whole, one
//! much louder fails as a whole, and the 4x4 test decides the rest.
//!
//! Codec reason: the bounds it scales are H.264's per-4x4 zero-block bounds of
//! the `P_Skip` test, and the neighbours are H.264's macroblock raster.
//! The idea follows openh264's neighbour-predicted skip SAD (BSD-2-Clause);
//! no code is taken from it.

/// Share of the neighbour prediction under which a macroblock passes, at
/// strength 0, and its growth per unit of strength (capped at
/// [`ACCEPT_SHARE_MAX`]).
const ACCEPT_SHARE_BASE: f32 = 0.5;
const ACCEPT_SHARE_PER_STRENGTH: f32 = 0.25;
const ACCEPT_SHARE_MAX: f32 = 1.5;
/// A pass also needs the 16x16 SAD within this many 4x4 bounds at strength 0,
/// growing per unit of strength up to [`ACCEPT_BOUNDS_MAX`] (of the sixteen a
/// macroblock holds, so a pass never needs every block at its bound).
const ACCEPT_BOUNDS_BASE: f32 = 6.0;
const ACCEPT_BOUNDS_PER_STRENGTH: f32 = 2.0;
const ACCEPT_BOUNDS_MAX: f32 = 13.0;
/// A fail needs the 16x16 SAD above this many 4x4 bounds at strength 0,
/// falling per unit of strength down to [`REJECT_BOUNDS_MIN`].
const REJECT_BOUNDS_BASE: f32 = 10.0;
const REJECT_BOUNDS_PER_STRENGTH: f32 = 2.0;
const REJECT_BOUNDS_MIN: f32 = 4.0;
/// A fail also needs the 16x16 SAD above the neighbour prediction times
/// `1 + REJECT_MARGIN / strength` (strength floored at
/// [`REJECT_STRENGTH_FLOOR`]), at most [`REJECT_RATIO_MAX`] times it.
const REJECT_MARGIN: f32 = 0.6;
const REJECT_STRENGTH_FLOOR: f32 = 0.25;
const REJECT_RATIO_MAX: f32 = 2.5;
/// 4x4 blocks in a macroblock's luma.
const LUMA_BLOCKS: u32 = 16;

/// The screen's verdict on a macroblock's luma.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LumaScreen {
    /// Every luma block is taken as zero residual.
    Pass,
    /// The luma is coded.
    Fail,
    /// The 4x4 test decides.
    Unsure,
}

/// The frame's 16x16 luma SADs in macroblock raster, 0 where none was
/// recorded (not yet coded, or intra).
#[derive(Debug, Default)]
pub struct NeighbourSads {
    cols: usize,
    sads: Vec<u16>,
}

impl NeighbourSads {
    /// Empties the table for a frame of `cols` x `rows` macroblocks.
    pub fn reset(&mut self, cols: usize, rows: usize) {
        self.cols = cols;
        self.sads.clear();
        self.sads.resize(cols * rows, 0);
    }

    /// Records macroblock `(col, row)`'s 16x16 luma SAD, kept within `1..=u16::MAX`
    /// (0 means "none recorded").
    pub fn record(&mut self, (col, row): (usize, usize), sad: u32) {
        if let Some(slot) = self.sads.get_mut(row * self.cols + col) {
            *slot = sad.clamp(1, u32::from(u16::MAX)) as u16;
        }
    }

    /// The SAD predicted for `(col, row)` from its left, top and top-right
    /// neighbours: the median of three, the mean of two, the one, or `None`.
    fn predict(&self, (col, row): (usize, usize)) -> Option<u32> {
        let at = |c: usize, r: usize| {
            self.sads
                .get(r * self.cols + c)
                .map_or(0, |&sad| u32::from(sad))
        };
        let left = if col > 0 { at(col - 1, row) } else { 0 };
        let top = if row > 0 { at(col, row - 1) } else { 0 };
        let top_right = if row > 0 && col + 1 < self.cols {
            at(col + 1, row - 1)
        } else {
            0
        };
        match (left, top, top_right) {
            (0, 0, 0) => None,
            (a, 0, 0) | (0, a, 0) | (0, 0, a) => Some(a),
            (a, b, 0) | (a, 0, b) | (0, a, b) => Some(a.midpoint(b)),
            (a, b, c) => Some(a.max(b).min(c).max(a.min(b))),
        }
    }

    /// The verdict on `(col, row)`'s luma, whose 16x16 SAD is `sad` and 4x4
    /// zero bound `bound`, at screen `strength` (> 0; higher passes and
    /// fails more macroblocks on their neighbours' evidence).
    pub fn screen(&self, pos: (usize, usize), sad: u32, bound: u32, strength: f32) -> LumaScreen {
        // The 4x4 test's own certainties: above sixteen bounds a block must
        // fail it; within one bound every block passes it.
        if sad > bound.saturating_mul(LUMA_BLOCKS) {
            return LumaScreen::Fail;
        }
        if sad <= bound {
            return LumaScreen::Pass;
        }
        let Some(predicted) = self.predict(pos) else {
            return LumaScreen::Unsure;
        };
        let (predicted, bound) = (predicted as f32, bound as f32);
        let accept_share =
            (ACCEPT_SHARE_BASE + ACCEPT_SHARE_PER_STRENGTH * strength).min(ACCEPT_SHARE_MAX);
        let accept_bounds =
            (ACCEPT_BOUNDS_BASE + ACCEPT_BOUNDS_PER_STRENGTH * strength).min(ACCEPT_BOUNDS_MAX);
        if sad as f32 <= (predicted * accept_share).min(bound * accept_bounds) {
            return LumaScreen::Pass;
        }
        let reject_bounds =
            (REJECT_BOUNDS_BASE - REJECT_BOUNDS_PER_STRENGTH * strength).max(REJECT_BOUNDS_MIN);
        let reject_ratio =
            (1.0 + REJECT_MARGIN / strength.max(REJECT_STRENGTH_FLOOR)).min(REJECT_RATIO_MAX);
        if sad as f32 > (bound * reject_bounds).max(predicted * reject_ratio) {
            return LumaScreen::Fail;
        }
        LumaScreen::Unsure
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(sads: &[(usize, usize, u32)]) -> NeighbourSads {
        let mut table = NeighbourSads::default();
        table.reset(4, 3);
        for &(col, row, sad) in sads {
            table.record((col, row), sad);
        }
        table
    }

    #[test]
    fn the_4x4_tests_certainties_hold_without_neighbours() {
        let empty = table(&[]);
        assert_eq!(empty.screen((1, 1), 17 * 10, 10, 1.0), LumaScreen::Fail);
        assert_eq!(empty.screen((1, 1), 10, 10, 1.0), LumaScreen::Pass);
        assert_eq!(empty.screen((1, 1), 50, 10, 1.0), LumaScreen::Unsure);
    }

    #[test]
    fn prediction_is_the_median_of_left_top_and_top_right() {
        let three = table(&[(0, 1, 100), (1, 0, 300), (2, 0, 200)]);
        assert_eq!(three.predict((1, 1)), Some(200));
        let two = table(&[(0, 1, 100), (1, 0, 300)]);
        assert_eq!(two.predict((1, 1)), Some(200));
        assert_eq!(table(&[]).predict((1, 1)), None);
    }

    #[test]
    fn quiet_blocks_pass_and_loud_blocks_fail_beside_their_neighbours() {
        let busy = table(&[(0, 1, 120), (1, 0, 120), (2, 0, 120)]);
        // Strength 1: pass at or below min(0.75 x 120, 8 bounds) = 80.
        assert_eq!(busy.screen((1, 1), 80, 10, 1.0), LumaScreen::Pass);
        // Fail above max(8 bounds, 1.6 x 120) = 192.
        assert_eq!(busy.screen((1, 1), 193, 10, 1.0), LumaScreen::Fail);
        assert_eq!(busy.screen((1, 1), 150, 10, 1.0), LumaScreen::Unsure);
    }
}
