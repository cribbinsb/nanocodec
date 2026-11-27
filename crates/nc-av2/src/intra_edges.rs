//! AV2 intra prediction of one transform: which neighbouring samples exist,
//! the edge arrays built from them, and the prediction.
//!
//! Follows pinned AVM v1.0.0 (BSD-3-Clause-Clear) `av2_predict_intra_block`,
//! `has_top_right`, `has_bottom_left` and
//! `av2_build_intra_predictors_high_default` (`av2/common/reconintra.c`) for
//! one tile, 64×64 superblocks, square blocks and transforms, and the intra
//! edge filter, intra bi-prediction, MRL, DIP and palette disabled. The
//! pixel kernels are nc-dsp's [`Av2IntraEdges`].
//!
//! Above-right samples exist when the transform's above-right lies in the
//! row above the block, inside the block's own earlier transforms, or in an
//! already coded block of the same superblock (or the superblock row above);
//! below-left ones when they lie in the block to the left, a coded block of
//! the same superblock, or the superblock to the left (down to its bottom).
//! Missing samples repeat the last available one.
use crate::block_state::BlockState;
use crate::intra_modes::Mode;
use crate::partition::SUPERBLOCK;
use nc_common::{Plane, PlaneView};
use nc_dsp::Dsp;
use nc_dsp::intra_prediction::av2::{Av2IntraEdges, Av2Predictor};

/// Mode-information unit: availability is decided per 4×4 luma samples.
const MI: usize = 4;
/// Chroma transforms wider (taller) than this read no above-right
/// (below-left) samples: at most 64 reference samples per edge.
const MAX_CHROMA_EXTENDED_SIDE: usize = 32;
/// Extra left samples read when the below-left is not needed.
const LEFT_TAIL_WITHOUT_BOTTOM: usize = 3;
/// `1 << (bit_depth − 1)`: the value of a missing corner.
const BASE: u8 = 128;
/// Directional angles of pure vertical and horizontal prediction.
const VERTICAL_ANGLE: u16 = 90;
const HORIZONTAL_ANGLE: u16 = 180;

/// The transform to predict: its plane, block and position.
#[derive(Debug, Clone, Copy)]
pub(crate) struct IntraTarget {
    /// Plane index (0 luma, 1 or 2 chroma at half resolution).
    pub plane: usize,
    /// The coding block's luma position.
    pub block: (usize, usize),
    /// The coding block's luma side.
    pub block_size: usize,
    /// The transform's position in plane samples.
    pub tx: (usize, usize),
    /// The transform's side in plane samples.
    pub side: usize,
}

impl IntraTarget {
    fn subsampling(self) -> usize {
        usize::from(self.plane != 0)
    }

    /// The transform's offset inside its block, in plane samples.
    fn offset(self) -> (usize, usize) {
        let ss = self.subsampling();
        (
            self.tx.0 - (self.block.0 >> ss),
            self.tx.1 - (self.block.1 >> ss),
        )
    }
}

/// Whether an edge extension (above-right or below-left) is read, and how
/// many of its samples exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Extension {
    /// The predictor does not read it.
    Unused,
    /// The predictor reads it; this many samples exist (possibly none).
    Samples(usize),
}

/// Which edge extensions a predictor reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Extends {
    /// Neither.
    Neither,
    /// The samples right of the above row.
    AboveRight,
    /// The samples below the left column.
    BelowLeft,
    /// Both.
    Both,
}

/// Which edges a predictor reads (AVM `extend_modes`, or the angle zone).
#[derive(Debug, Clone, Copy)]
struct Needs {
    above: bool,
    left: bool,
    corner: bool,
    extends: Extends,
}

impl Needs {
    fn of(mode: Mode) -> Self {
        let needs = |above, left, corner, extends| Self {
            above,
            left,
            corner,
            extends,
        };
        match mode.angle() {
            Some(angle) if angle < VERTICAL_ANGLE => needs(true, false, true, Extends::AboveRight),
            Some(VERTICAL_ANGLE) => needs(true, false, false, Extends::Neither),
            Some(angle) if angle < HORIZONTAL_ANGLE => needs(true, true, true, Extends::Neither),
            Some(HORIZONTAL_ANGLE) => needs(false, true, false, Extends::Neither),
            Some(_) => needs(false, true, true, Extends::BelowLeft),
            None => match mode {
                Mode::Smooth => needs(true, true, true, Extends::Both),
                Mode::SmoothVertical => needs(true, true, false, Extends::BelowLeft),
                Mode::SmoothHorizontal => needs(true, true, false, Extends::AboveRight),
                _ => needs(true, true, true, Extends::Neither),
            },
        }
    }

    fn above_right(self) -> bool {
        matches!(self.extends, Extends::AboveRight | Extends::Both)
    }

    fn below_left(self) -> bool {
        matches!(self.extends, Extends::BelowLeft | Extends::Both)
    }
}

/// Predict `target` with `mode` from the reconstruction `recon` (the
/// target's plane) into the `side`×`side` raster `output`; `state` says
/// which blocks are already coded; `dsp` runs the pixel kernels.
pub(crate) fn predict(
    recon: &Plane,
    state: &BlockState,
    target: IntraTarget,
    mode: Mode,
    dsp: &Dsp,
    edges: &mut Av2IntraEdges,
    output: &mut [u8],
) {
    let view = recon.view();
    let neighbours = neighbour_counts(target.tx, target.side, (view.width(), view.height()));
    let needs = Needs::of(mode);
    let (x, y) = target.tx;
    if let Some(value) = missing_edge_prediction(
        needs,
        neighbours,
        (|| view.row(y - 1)[x], || view.row(y)[x - 1]),
    ) {
        output.fill(value);
        return;
    }
    *edges = Av2IntraEdges::new();
    prepare_edges(recon, state, target, (&view, neighbours), needs, edges);
    dsp.av2_intra_predict(
        edges,
        predictor(mode, target.plane == 0, neighbours),
        target.side,
        output,
    );
}

#[inline]
fn missing_edge_prediction(
    needs: Needs,
    (n_top, n_left): (usize, usize),
    (above, left): (impl FnOnce() -> u8, impl FnOnce() -> u8),
) -> Option<u8> {
    if !needs.above && n_left == 0 {
        Some(if n_top > 0 { above() } else { BASE + 1 })
    } else if !needs.left && n_top == 0 {
        Some(if n_left > 0 { left() } else { BASE - 1 })
    } else {
        None
    }
}

#[inline]
fn prepare_edges(
    recon: &Plane,
    state: &BlockState,
    target: IntraTarget,
    (view, (n_top, n_left)): (&PlaneView<'_>, (usize, usize)),
    needs: Needs,
    edges: &mut Av2IntraEdges,
) {
    let (x, y) = target.tx;
    let side = target.side;
    let above_ref = |i: usize| view.row(y - 1)[x + i];
    let left_ref = |i: usize| view.row(y + i)[x - 1];
    if needs.left {
        let below_left = if needs.below_left() {
            Extension::Samples(below_left_samples(recon, state, target))
        } else {
            Extension::Unused
        };
        let count = side
            + match below_left {
                Extension::Unused => LEFT_TAIL_WITHOUT_BOTTOM,
                Extension::Samples(_) => side,
            };
        let left = &mut edges.left_mut()[..count];
        if n_left > 0 {
            let available = n_left
                + match below_left {
                    Extension::Samples(n) => n,
                    Extension::Unused => 0,
                };
            copy_left_column(view, (x, y), &mut left[..available]);
            let last = left[available - 1];
            left[available..].fill(last);
        } else if n_top > 0 {
            left.fill(above_ref(0));
        } else {
            left.fill(BASE + 1);
        }
    }
    if needs.above {
        let above_right = if needs.above_right() {
            Extension::Samples(above_right_samples(recon, state, target))
        } else {
            Extension::Unused
        };
        let count = side
            + match above_right {
                Extension::Unused => 0,
                Extension::Samples(_) => side,
            };
        let above = &mut edges.above_mut()[..count];
        if n_top > 0 {
            let available = n_top
                + match above_right {
                    Extension::Samples(n) => n,
                    Extension::Unused => 0,
                };
            nc_common::copy_row(&mut above[..available], &view.row(y - 1)[x..x + available]);
            let last = above[available - 1];
            above[available..].fill(last);
        } else if n_left > 0 {
            above.fill(left_ref(0));
        } else {
            above.fill(BASE - 1);
        }
    }
    if needs.corner {
        edges.set_corner(match (n_top > 0, n_left > 0) {
            (true, true) => view.row(y - 1)[x - 1],
            (true, false) => above_ref(0),
            (false, true) => left_ref(0),
            (false, false) => BASE,
        });
    }
}

#[inline]
fn predictor(mode: Mode, luma: bool, (n_top, n_left): (usize, usize)) -> Av2Predictor {
    match mode {
        Mode::Dc => Av2Predictor::Dc {
            above: n_top > 0,
            left: n_left > 0,
        },
        Mode::Smooth => Av2Predictor::Smooth,
        Mode::SmoothVertical => Av2Predictor::SmoothVertical,
        Mode::SmoothHorizontal => Av2Predictor::SmoothHorizontal,
        Mode::Paeth => Av2Predictor::Paeth,
        Mode::Directional { .. } => Av2Predictor::Directional {
            angle: mode.angle().expect("directional"),
            luma,
        },
    }
}

/// Fully prepared luma neighbours for mode trials that read the same
/// reconstruction. Luma directional prediction can overwrite the first
/// extension sample in each edge when padding its two inner tap ranges.
pub(crate) struct TrialReferences {
    edges: Av2IntraEdges,
    neighbours: (usize, usize),
    side: usize,
}

impl TrialReferences {
    pub(crate) fn luma(
        recon: &Plane,
        state: &BlockState,
        position: (usize, usize),
        side: usize,
    ) -> Self {
        let target = IntraTarget {
            plane: 0,
            block: position,
            block_size: side,
            tx: position,
            side,
        };
        let view = recon.view();
        let neighbours = neighbour_counts(position, side, (view.width(), view.height()));
        let mut edges = Av2IntraEdges::new();
        prepare_edges(
            recon,
            state,
            target,
            (&view, neighbours),
            Needs::of(Mode::Smooth),
            &mut edges,
        );
        Self {
            edges,
            neighbours,
            side,
        }
    }

    pub(crate) fn predict(&mut self, mode: Mode, dsp: &Dsp, output: &mut [u8]) {
        let needs = Needs::of(mode);
        let (n_top, n_left) = self.neighbours;
        if (!needs.above && n_left == 0) || (!needs.left && n_top == 0) {
            // Reading the first sample is only needed on a picture border.
            let above = self.edges.above_mut()[0];
            let left = self.edges.left_mut()[0];
            if let Some(value) =
                missing_edge_prediction(needs, self.neighbours, (|| above, || left))
            {
                output.fill(value);
                return;
            }
        }
        let predictor = predictor(mode, true, self.neighbours);
        let inner_taps = matches!(predictor, Av2Predictor::Directional { angle, .. }
            if angle > VERTICAL_ANGLE && angle < HORIZONTAL_ANGLE);
        let extension = inner_taps.then(|| {
            (
                self.edges.above_mut()[self.side],
                self.edges.left_mut()[self.side],
            )
        });
        dsp.av2_intra_predict(&mut self.edges, predictor, self.side, output);
        if let Some((above, left)) = extension {
            self.edges.above_mut()[self.side] = above;
            self.edges.left_mut()[self.side] = left;
        }
    }
}

/// Copy the column left of `(x, y)` in `view`, from row `y` down, into
/// `left`: one stride walk with one bounds check for the column, not a row
/// lookup per sample.
fn copy_left_column(view: &PlaneView<'_>, (x, y): (usize, usize), left: &mut [u8]) {
    assert!(
        y + left.len() <= view.height(),
        "left column inside the plane"
    );
    let column = view.data()[y * view.stride() + x - 1..]
        .iter()
        .step_by(view.stride());
    for (sample, &value) in left.iter_mut().zip(column) {
        *sample = value;
    }
}

/// Whether a block at (`x`, `y`) predicts from synthetic edges only: with no
/// sample above and none to the left, every mode but DC fills its edges with
/// 127 above and 129 left (`BASE` ∓ 1, AV1 §7.11.2), a constant bias. Anywhere else
/// a missing edge copies the real neighbour on the other side, so in a
/// one-tile frame this is the first block alone.
#[must_use]
pub fn only_synthetic_edges((x, y): (usize, usize)) -> bool {
    (x, y) == (0, 0)
}

/// How many samples above and left of the `side`-wide transform at
/// (`x`, `y`) exist in a plane of `plane_size`: none at the picture's top
/// or left; a transform crossing the right or bottom edge has neighbours
/// only inside the picture and repeats the last one (AVM `n_top_px` /
/// `n_left_px` from `mb_to_right_edge` / `mb_to_bottom_edge`).
fn neighbour_counts(
    (x, y): (usize, usize),
    side: usize,
    plane_size: (usize, usize),
) -> (usize, usize) {
    let n_top = if y > 0 { side.min(plane_size.0 - x) } else { 0 };
    let n_left = if x > 0 { side.min(plane_size.1 - y) } else { 0 };
    (n_top, n_left)
}

/// Available above-right samples of `target` (AVM `has_top_right`), zero
/// when none.
fn above_right_samples(recon: &Plane, state: &BlockState, target: IntraTarget) -> usize {
    let ss = target.subsampling();
    let (x, y) = target.tx;
    let side = target.side;
    let plane_width = recon.view().width();
    let right_available = x + side < plane_width;
    if y == 0 || !right_available {
        return 0;
    }
    let common = side.min(plane_width - (x + side));
    if target.plane != 0 && side > MAX_CHROMA_EXTENDED_SIDE {
        return 0;
    }
    let (col_off, row_off) = target.offset();
    let plane_block = target.block_size >> ss;
    if row_off > 0 {
        // Below the block's first row: only an earlier transform of the block.
        return if col_off + side < plane_block {
            common
        } else {
            0
        };
    }
    if col_off + side < plane_block {
        // In the row above the block.
        return common;
    }
    let (block_x, block_y) = target.block;
    if block_y % SUPERBLOCK == 0 {
        // The superblock row above is complete.
        return common;
    }
    let column = block_x % SUPERBLOCK + target.block_size;
    if column >= SUPERBLOCK {
        return 0;
    }
    let (start_x, row) = (block_x + target.block_size, block_y - MI);
    let units = (side / MI) << ss;
    let coded = (0..units)
        .take_while(|&i| column + i * MI < SUPERBLOCK && state.is_coded(start_x + i * MI, row))
        .count();
    ((coded * MI) >> ss).min(common)
}

/// Available below-left samples of `target` (AVM `has_bottom_left`), zero
/// when none.
fn below_left_samples(recon: &Plane, state: &BlockState, target: IntraTarget) -> usize {
    let ss = target.subsampling();
    let (x, y) = target.tx;
    let side = target.side;
    let plane_height = recon.view().height();
    if x == 0 || y + side >= plane_height {
        return 0;
    }
    let common = side.min(plane_height - (y + side));
    if target.plane != 0 && side > MAX_CHROMA_EXTENDED_SIDE {
        return 0;
    }
    let (col_off, row_off) = target.offset();
    if col_off > 0 {
        return 0;
    }
    let plane_block = target.block_size >> ss;
    if row_off + side < plane_block {
        // In the block to the left.
        return common;
    }
    let (block_x, block_y) = target.block;
    let row = block_y % SUPERBLOCK + target.block_size;
    if block_x % SUPERBLOCK == 0 {
        // The superblock to the left is complete down to its bottom.
        let remaining = (SUPERBLOCK >> ss).saturating_sub(row >> ss);
        return remaining.min(common);
    }
    if row >= SUPERBLOCK {
        return 0;
    }
    let (column, start_y) = (block_x - MI, block_y + target.block_size);
    let units = (side / MI) << ss;
    let coded = (0..units)
        .take_while(|&i| row + i * MI < SUPERBLOCK && state.is_coded(column, start_y + i * MI))
        .count();
    ((coded * MI) >> ss).min(common)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mv_refs::ListConfig;

    #[test]
    fn luma_trials_preserve_extension_samples_between_modes() {
        let mut plane = Plane::new(192, 192, 0).unwrap();
        for y in 0..192 {
            for (x, sample) in plane.view_mut().row_mut(y).iter_mut().enumerate() {
                *sample = u8::try_from((x * 37 + y * 53 + x * y * 7) % 251).unwrap();
            }
        }
        let state = BlockState::new(192, 192, ListConfig::BASIC).unwrap();
        let dsp = Dsp::new();
        for side in [8, 16, 32, 64] {
            for position in [
                (0, 0),
                (64, 0),
                (0, 64),
                (64, 64),
                (64, 72),
                (192 - side, 192 - side),
            ] {
                let target = IntraTarget {
                    plane: 0,
                    block: position,
                    block_size: side,
                    tx: position,
                    side,
                };
                let mut prepared = TrialReferences::luma(&plane, &state, position, side);
                let mut ordinary = Av2IntraEdges::new();
                let mut reference = vec![0; side * side];
                let mut actual = vec![0; side * side];
                let reversed = (0..crate::intra_modes::LUMA_MODES)
                    .rev()
                    .map(Mode::from_joint);
                for mode in Mode::all_luma().chain(reversed) {
                    predict(
                        &plane,
                        &state,
                        target,
                        mode,
                        &dsp,
                        &mut ordinary,
                        &mut reference,
                    );
                    prepared.predict(mode, &dsp, &mut actual);
                    assert_eq!(actual, reference, "{position:?}, {side}, {mode:?}");
                }
            }
        }
    }
}
