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
use nc_common::Plane;
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
    let (x, y) = target.tx;
    let side = target.side;
    let n_top = if y > 0 { side } else { 0 };
    let n_left = if x > 0 { side } else { 0 };
    let needs = Needs::of(mode);
    let above_ref = |i: usize| view.row(y - 1)[x + i];
    let left_ref = |i: usize| view.row(y + i)[x - 1];
    if (!needs.above && n_left == 0) || (!needs.left && n_top == 0) {
        let value = if needs.left {
            if n_top > 0 { above_ref(0) } else { BASE + 1 }
        } else if n_left > 0 {
            left_ref(0)
        } else {
            BASE - 1
        };
        output.fill(value);
        return;
    }
    *edges = Av2IntraEdges::new();
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
            for (i, sample) in left[..available].iter_mut().enumerate() {
                *sample = left_ref(i);
            }
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
    let predictor = match mode {
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
            luma: target.plane == 0,
        },
    };
    dsp.av2_intra_predict(edges, predictor, side, output);
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
