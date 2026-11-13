//! Per-picture state of already coded blocks, at 8x8 luma granularity (the
//! minimum CU): what later blocks need for contexts, motion vector
//! prediction (HEVC §8.5.3.2) and the deblocking filter (§8.7.2).
//!
//! Every CU is at least 8x8, so one cell never straddles two CUs. Finer
//! detail exists only inside an 8x8 intra CU: its four 4x4 prediction units
//! (`PART_NxN`) keep one luma mode each ([`Cell::intra_modes`]), and its
//! 4x4 luma transform blocks share the cell's transform id. That is all the
//! deblocking filter needs there: an intra CU's edges have boundary strength
//! 2 whatever the coefficients (§8.7.2.4), and 4x4 transform edges off the
//! 8x8 grid are not filtered.

use nc_encoder::Mv;

use crate::headers::{LOG2_CTB, LOG2_MIN_CB};

/// log2 of the minimum transform block (4x4), the unit of the z-scan order.
const LOG2_MIN_TB: usize = 2;

/// `v` with its bits spread to the even bit positions (bit `i` → bit `2i`),
/// for the 4-bit block coordinates inside a 64x64 CTB.
const SPREAD_BITS: [usize; 16] = [0, 1, 4, 5, 16, 17, 20, 21, 64, 65, 68, 69, 80, 81, 84, 85];

/// What is known about one 8x8 cell of the current picture.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Cell {
    /// The CU covering the cell is intra coded.
    pub intra: bool,
    /// Luma intra prediction modes of an intra CU, one per 4x4 quadrant of
    /// the cell in z order (all equal unless the CU is `PART_NxN`), for
    /// `MPM` derivation (§8.4.2).
    pub intra_modes: [u8; 4],
    /// The CU covering the cell is skipped (`cu_skip_flag`).
    pub skip: bool,
    /// Motion vector of the (single) prediction unit, inter CUs only.
    pub mv: Mv,
    /// `CtDepth` of the covering CU.
    pub depth: u8,
    /// The luma transform block covering the cell has coefficients.
    pub luma_coded: bool,
    /// Identifier of the covering CU (unique within the picture).
    pub cu: u32,
    /// Identifier of the covering luma transform block.
    pub tu: u32,
    /// `QpY` of the covering CU (§8.6.1), for deblocking and the syntax
    /// writer's QP prediction.
    pub qp: i8,
}

impl Cell {
    /// The luma intra mode at luma sample `(x, y)` of this cell.
    #[must_use]
    pub fn intra_mode_at(&self, x: usize, y: usize) -> u8 {
        self.intra_modes[quadrant(x, y)]
    }
}

/// [`Grid::available`] for the intra reference units of one aligned square
/// block (HEVC §6.4.1, one slice, no tiles), as luma positions inside the
/// picture.
///
/// An aligned power-of-two square covers one contiguous run of z-scan
/// addresses, so a neighbouring aligned square of the same size is decoded
/// either entirely before the block or entirely after it. The left, above
/// and above-left neighbours always come before; the below-left and
/// above-right squares each need one test. Exact: the unit test compares
/// it with [`Grid::available`] for every aligned block and unit.
#[derive(Debug, Clone, Copy)]
pub struct ReferenceAvailability {
    x: usize,
    y: usize,
    side: usize,
    below_left: bool,
    above_right: bool,
}

impl ReferenceAvailability {
    /// Whether the reference unit at luma `(xn, yn)`, a left-column or
    /// above-row neighbour inside the picture, is available.
    #[must_use]
    pub fn at(&self, xn: usize, yn: usize) -> bool {
        if xn < self.x {
            yn < self.y + self.side || self.below_left
        } else {
            xn < self.x + self.side || self.above_right
        }
    }
}

/// Index of the 4x4 quadrant holding luma sample `(x, y)` within its 8x8
/// cell, in z order.
#[must_use]
pub fn quadrant(x: usize, y: usize) -> usize {
    ((y >> LOG2_MIN_TB) & 1) << 1 | ((x >> LOG2_MIN_TB) & 1)
}

/// The cell grid of one picture.
#[derive(Debug, Clone)]
pub struct Grid {
    cells: Vec<Cell>,
    cols: usize,
    rows: usize,
    width: usize,
    height: usize,
    ctb_cols: usize,
}

impl Grid {
    /// A grid for a `width` x `height` picture.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        let cols = width >> LOG2_MIN_CB;
        let rows = height >> LOG2_MIN_CB;
        Self {
            cells: vec![Cell::default(); cols * rows],
            cols,
            rows,
            width,
            height,
            ctb_cols: width.div_ceil(1 << LOG2_CTB),
        }
    }

    /// Forget the previous picture.
    pub fn reset(&mut self) {
        self.cells.fill(Cell::default());
    }

    /// The cell containing luma sample `(x, y)` (which must be inside the
    /// picture).
    #[must_use]
    pub fn at(&self, x: usize, y: usize) -> &Cell {
        &self.cells[(y >> LOG2_MIN_CB) * self.cols + (x >> LOG2_MIN_CB)]
    }

    /// Cells per row.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Rows of cells.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The cells of row `row`, left to right.
    #[must_use]
    pub fn row(&self, row: usize) -> &[Cell] {
        &self.cells[row * self.cols..(row + 1) * self.cols]
    }

    /// Picture width in luma samples.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Picture height in luma samples.
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Set every cell of the square block at `(x, y)` of side `size`.
    pub fn fill(&mut self, x: usize, y: usize, size: usize, cell: Cell) {
        let side = size >> LOG2_MIN_CB;
        let (col0, row0) = (x >> LOG2_MIN_CB, y >> LOG2_MIN_CB);
        for row in row0..(row0 + side).min(self.rows) {
            let start = row * self.cols + col0;
            let end = start + side.min(self.cols - col0);
            self.cells[start..end].fill(cell);
        }
    }

    /// Update the luma-coded flag and transform id of a transform block.
    /// The 4x4 blocks of a cell are recorded as one: the first sets the id,
    /// and the cell is coded if any of them is.
    #[inline]
    pub fn set_transform(&mut self, x: usize, y: usize, size: usize, tu: u32, coded: bool) {
        if size < 1 << LOG2_MIN_CB {
            self.set_sub_cell_transform(x, y, tu, coded);
            return;
        }
        let side = size >> LOG2_MIN_CB;
        let (col0, row0) = (x >> LOG2_MIN_CB, y >> LOG2_MIN_CB);
        for row in row0..row0 + side {
            for cell in &mut self.cells[row * self.cols + col0..row * self.cols + col0 + side] {
                cell.tu = tu;
                cell.luma_coded = coded;
            }
        }
    }

    /// The neighbouring cell at `(xn, yn)` if it is available to the block
    /// whose top-left sample is `(x, y)`: inside the picture and before the
    /// block in decoding (z-scan) order (HEVC §6.4.1; one slice, no tiles).
    #[must_use]
    pub fn available(&self, x: usize, y: usize, xn: isize, yn: isize) -> Option<&Cell> {
        if xn < 0 || yn < 0 || xn as usize >= self.width || yn as usize >= self.height {
            return None;
        }
        let (xn, yn) = (xn as usize, yn as usize);
        (self.zscan(xn, yn) < self.zscan(x, y)).then(|| self.at(xn, yn))
    }

    /// The cell at `(xn, yn)` when it is the left neighbour `(x - 1, y)`
    /// or the above neighbour `(x, y - 1)` of a block's top-left sample
    /// `(x, y)`: those precede the block in z-scan order whenever they lie
    /// inside the picture (one slice, no tiles; HEVC §6.4.1), so the same
    /// answer as [`Grid::available`] without its two z-scan addresses (the
    /// context derivations call it for every coded CU and split flag).
    #[must_use]
    pub fn left_or_above(&self, xn: isize, yn: isize) -> Option<&Cell> {
        let (xn, yn) = (usize::try_from(xn).ok()?, usize::try_from(yn).ok()?);
        (xn < self.width && yn < self.height).then(|| self.at(xn, yn))
    }

    /// Availability of the intra reference samples of the aligned
    /// `side`x`side` luma block at `(x, y)`, with one z-scan test per
    /// neighbouring region instead of one per 4x4 unit.
    #[must_use]
    pub fn reference_availability(&self, x: usize, y: usize, side: usize) -> ReferenceAvailability {
        let probe = |xn: usize, yn: usize| self.available(x, y, xn as isize, yn as isize).is_some();
        ReferenceAvailability {
            x,
            y,
            side,
            below_left: x > 0 && probe(x - 1, y + side),
            above_right: y > 0 && probe(x + side, y - 1),
        }
    }

    /// The cell at `(xn, yn)`, a neighbour of the block `availability`
    /// describes, if it is inside the picture and available: the same as
    /// [`Grid::available`] for that block.
    #[must_use]
    pub fn reference_cell(
        &self,
        availability: &ReferenceAvailability,
        xn: isize,
        yn: isize,
    ) -> Option<&Cell> {
        if xn < 0 || yn < 0 || xn as usize >= self.width || yn as usize >= self.height {
            return None;
        }
        let (xn, yn) = (xn as usize, yn as usize);
        availability.at(xn, yn).then(|| self.at(xn, yn))
    }

    /// Whether every CU of each CTB is skipped (`cu_skip_flag`: prediction
    /// only, no residual), CTBs in raster order.
    #[must_use]
    pub fn skipped_ctbs(&self) -> Vec<bool> {
        let side = 1usize << (LOG2_CTB - LOG2_MIN_CB);
        let ctb_rows = self.rows.div_ceil(side);
        (0..ctb_rows * self.ctb_cols)
            .map(|index| {
                let (col0, row0) = (
                    (index % self.ctb_cols) * side,
                    (index / self.ctb_cols) * side,
                );
                (row0..(row0 + side).min(self.rows)).all(|row| {
                    let start = row * self.cols + col0;
                    let end = row * self.cols + (col0 + side).min(self.cols);
                    self.cells[start..end].iter().all(|c| c.skip)
                })
            })
            .collect()
    }

    /// [`Grid::set_transform`] of a 4x4 block: the cell's first one sets
    /// the id, any coded one makes the cell coded.
    #[cold]
    fn set_sub_cell_transform(&mut self, x: usize, y: usize, tu: u32, coded: bool) {
        let first = quadrant(x, y) == 0;
        let cell = &mut self.cells[(y >> LOG2_MIN_CB) * self.cols + (x >> LOG2_MIN_CB)];
        if first {
            cell.tu = tu;
            cell.luma_coded = coded;
        } else {
            cell.luma_coded |= coded;
        }
    }

    /// The intra luma modes of the left and above neighbours of the
    /// prediction block at `(x, y)`, as the candidate list derivation sees
    /// them (§8.4.2): only available intra blocks, and the above one only
    /// within the current CTB row.
    #[must_use]
    pub fn neighbour_intra_modes(&self, x: usize, y: usize) -> (Option<u8>, Option<u8>) {
        let mode = |xn: usize, yn: usize| {
            self.left_or_above(xn as isize, yn as isize)
                .filter(|c| c.intra)
                .map(|c| c.intra_mode_at(xn, yn))
        };
        let left = if x == 0 { None } else { mode(x - 1, y) };
        let above = if y.is_multiple_of(1 << LOG2_CTB) {
            None
        } else {
            mode(x, y - 1)
        };
        (left, above)
    }

    /// Set the luma modes of the 8x8 intra cell at `(x, y)` while its
    /// `PART_NxN` prediction units are decided one by one: later units'
    /// candidate lists read the earlier ones (§8.4.2).
    pub fn set_intra_cell(&mut self, x: usize, y: usize, modes: [u8; 4]) {
        let cell = &mut self.cells[(y >> LOG2_MIN_CB) * self.cols + (x >> LOG2_MIN_CB)];
        cell.intra = true;
        cell.intra_modes = modes;
    }

    /// `MinTbAddrZs` of the 4x4 block containing `(x, y)` (HEVC §6.5.2):
    /// CTBs in raster order, 4x4 blocks in z-order within each CTB.
    fn zscan(&self, x: usize, y: usize) -> usize {
        let ctb = (y >> LOG2_CTB) * self.ctb_cols + (x >> LOG2_CTB);
        let mask = (1 << (LOG2_CTB as usize - LOG2_MIN_TB)) - 1;
        let (bx, by) = ((x >> LOG2_MIN_TB) & mask, (y >> LOG2_MIN_TB) & mask);
        let morton = SPREAD_BITS[bx] | (SPREAD_BITS[by] << 1);
        (ctb << (2 * (LOG2_CTB as usize - LOG2_MIN_TB))) | morton
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn availability_follows_z_order() {
        let g = Grid::new(128, 64);
        // Above-right of the second 8x8 block in a CTB is not yet decoded…
        assert!(g.available(8, 8, 16, 7).is_none());
        // …but above-right of the first is.
        assert!(g.available(0, 8, 8, 7).is_some());
        // Below-left: decoded for the last quadrant of a 32x32, not for the
        // second 8x8 of a CTB.
        assert!(g.available(16, 16, 15, 24).is_some());
        assert!(g.available(8, 0, 7, 8).is_none());
        assert!(g.available(8, 8, 7, 16).is_none());
        // Outside the picture.
        assert!(g.available(0, 0, -1, 0).is_none());
        // The previous CTB is fully available, the next one is not.
        assert!(g.available(64, 32, 63, 63).is_some());
        assert!(g.available(0, 32, 64, 31).is_none());
    }

    #[test]
    fn reference_availability_matches_per_unit_tests() {
        const UNIT: usize = 1 << LOG2_MIN_TB;
        let (width, height) = (200, 136);
        let g = Grid::new(width, height);
        for log2 in LOG2_MIN_TB..=LOG2_CTB as usize {
            let side = 1 << log2;
            for y in (0..height).step_by(side) {
                for x in (0..width).step_by(side) {
                    let fast = g.reference_availability(x, y, side);
                    let left = (x > 0).then(|| (0..2 * side).map(|dy| (x - 1, y + dy)));
                    let above = (y > 0).then(|| (0..2 * side).map(|dx| (x + dx, y - 1)));
                    let corner = (x > 0 && y > 0).then(|| (x - 1, y - 1));
                    let units = left.into_iter().flatten().step_by(UNIT);
                    let units = units.chain(above.into_iter().flatten().step_by(UNIT));
                    // Neighbours outside the picture are never available.
                    for (xn, yn) in [(-1, 0), (0, -1), (width as isize, 0), (0, height as isize)] {
                        assert!(g.reference_cell(&fast, xn, yn).is_none());
                    }
                    for (xn, yn) in units
                        .chain(corner)
                        .filter(|&(xn, yn)| xn < width && yn < height)
                    {
                        let slow = g.available(x, y, xn as isize, yn as isize).is_some();
                        assert_eq!(
                            fast.at(xn, yn),
                            slow,
                            "block ({x},{y}) side {side}, unit ({xn},{yn})"
                        );
                        let cell = g.reference_cell(&fast, xn as isize, yn as isize);
                        assert_eq!(cell.is_some(), slow);
                    }
                }
            }
        }
    }

    #[test]
    fn left_or_above_matches_availability() {
        let (width, height) = (200, 136);
        let g = Grid::new(width, height);
        for log2 in LOG2_MIN_TB..=LOG2_CTB as usize {
            let side = 1 << log2;
            for y in (0..height).step_by(side) {
                for x in (0..width).step_by(side) {
                    for (xn, yn) in [(x as isize - 1, y as isize), (x as isize, y as isize - 1)] {
                        assert_eq!(
                            g.left_or_above(xn, yn).is_some(),
                            g.available(x, y, xn, yn).is_some(),
                            "block ({x},{y}) side {side}, neighbour ({xn},{yn})"
                        );
                    }
                }
            }
        }
    }
}
