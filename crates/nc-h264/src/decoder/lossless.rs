//! Transform bypass: the residual of a lossless macroblock
//! (`TransformBypassModeFlag`, `qpprime_y_zero_transform_bypass_flag` with
//! `QP′Y` 0) is its levels, untransformed (§8.5.12.1, §8.5.13.1: `rij =
//! cij`), and for intra blocks predicted horizontally or vertically each
//! residual sample is a difference from the one before it in that
//! direction (§8.5.15, the intra residual transform-bypass process).
//!
//! The DPCM runs over the prediction block: a 4x4 or 8x8 block for
//! `Intra_NxN`, the whole 16x16 macroblock for `Intra_16x16`, the whole
//! `MbWidthC` x `MbHeightC` block for 4:2:0 / 4:2:2 chroma.

use nc_common::{BitDepth, PlaneViewMut, Sample};

use super::chroma;
use super::sample::Coefficient;
use super::transform::Residual;

/// `Intra4x4PredMode` / `Intra8x8PredMode` / `Intra16x16PredMode`
/// (Tables 8-2..8-4): 0 vertical, 1 horizontal.
const LUMA_VERTICAL: u8 = 0;
const LUMA_HORIZONTAL: u8 = 1;
/// `intra_chroma_pred_mode` (Table 8-5): 1 horizontal, 2 vertical.
const CHROMA_HORIZONTAL: u8 = 1;
const CHROMA_VERTICAL: u8 = 2;
/// Side of a 4x4, 8x8 and 16x16 block.
const SIDE_4: usize = 4;
const SIDE_8: usize = 8;
const MB_SIDE: usize = 16;
/// Largest residual block (a macroblock's luma, or 4:4:4 / 4:2:2 chroma).
const MAX_SAMPLES: usize = MB_SIDE * MB_SIDE;

/// The direction of the intra residual DPCM (§8.5.15), if any.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Dpcm {
    None,
    /// Vertical prediction: each residual adds the one above it.
    Vertical,
    /// Horizontal prediction: each residual adds the one left of it.
    Horizontal,
}

impl Dpcm {
    /// The DPCM of an intra luma mode (`Intra_NxN` or `Intra_16x16`).
    pub fn of_luma_mode(mode: u8) -> Self {
        match mode {
            LUMA_VERTICAL => Self::Vertical,
            LUMA_HORIZONTAL => Self::Horizontal,
            _ => Self::None,
        }
    }

    /// The DPCM of `intra_chroma_pred_mode` `mode`.
    pub fn of_chroma_mode(mode: u8) -> Self {
        match mode {
            CHROMA_VERTICAL => Self::Vertical,
            CHROMA_HORIZONTAL => Self::Horizontal,
            _ => Self::None,
        }
    }
}

/// The `w` x `h` residual `r` (raster), accumulated along `dpcm`, added to
/// the prediction in `dst` with `Clip1` (§8.5.14).
fn add<T: Sample>(
    r: &mut [i32],
    (w, h): (usize, usize),
    dpcm: Dpcm,
    dst: &mut PlaneViewMut<'_, T>,
    depth: BitDepth,
) {
    match dpcm {
        Dpcm::None => {}
        Dpcm::Vertical => {
            for i in w..w * h {
                r[i] += r[i - w];
            }
        }
        Dpcm::Horizontal => {
            for row in r[..w * h].chunks_exact_mut(w) {
                for x in 1..w {
                    row[x] += row[x - 1];
                }
            }
        }
    }
    for (y, line) in r[..w * h].chunks_exact(w).enumerate() {
        for (sample, &value) in dst.row_mut(y)[..w].iter_mut().zip(line) {
            *sample = depth.clip_to(sample.to_i32() + value);
        }
    }
}

/// Copy the `n` x `n` block `levels` (raster) into the `stride`-wide
/// residual `r` at (`x`, `y`).
fn place<C: Coefficient>(
    r: &mut [i32],
    stride: usize,
    (x, y): (usize, usize),
    n: usize,
    levels: &[C],
) {
    for (row, line) in levels.chunks_exact(n).enumerate() {
        let start = (y + row) * stride + x;
        for (out, &level) in r[start..start + n].iter_mut().zip(line) {
            *out = level.into();
        }
    }
}

/// Add 4x4 block `block` (raster index) of component `plane` of an
/// `Intra_4x4` macroblock to its prediction `dst`, and clear it.
pub(super) fn add_4x4<C: Coefficient, T: Sample>(
    residual: &mut Residual<C>,
    (plane, block): (usize, usize),
    dpcm: Dpcm,
    dst: &mut PlaneViewMut<'_, T>,
    depth: BitDepth,
) {
    let mut r = [0i32; SIDE_4 * SIDE_4];
    place(
        &mut r,
        SIDE_4,
        (0, 0),
        SIDE_4,
        &residual.luma(plane).take_4x4(block),
    );
    add(&mut r, (SIDE_4, SIDE_4), dpcm, dst, depth);
}

/// Add 8x8 block `b8` of component `plane` of an `Intra_8x8` macroblock to
/// its prediction `dst`, and clear it.
pub(super) fn add_8x8<C: Coefficient, T: Sample>(
    residual: &mut Residual<C>,
    (plane, b8): (usize, usize),
    dpcm: Dpcm,
    dst: &mut PlaneViewMut<'_, T>,
    depth: BitDepth,
) {
    let mut r = [0i32; SIDE_8 * SIDE_8];
    place(
        &mut r,
        SIDE_8,
        (0, 0),
        SIDE_8,
        &residual.luma(plane).take_8x8(b8),
    );
    add(&mut r, (SIDE_8, SIDE_8), dpcm, dst, depth);
}

/// Add every block of component `plane` (4x4 blocks, or 8x8 ones under
/// `transform_8x8`) to the macroblock's prediction `dst` (16x16), with
/// the DPCM over the whole macroblock (`Intra_16x16`), and clear them.
pub(super) fn add_macroblock<C: Coefficient, T: Sample>(
    residual: &mut Residual<C>,
    (plane, transform_8x8): (usize, bool),
    dpcm: Dpcm,
    dst: &mut PlaneViewMut<'_, T>,
    depth: BitDepth,
) {
    let mut r = [0i32; MAX_SAMPLES];
    if transform_8x8 {
        for b8 in 0..4 {
            let at = ((b8 % 2) * SIDE_8, (b8 / 2) * SIDE_8);
            place(
                &mut r,
                MB_SIDE,
                at,
                SIDE_8,
                &residual.luma(plane).take_8x8(b8),
            );
        }
    } else {
        for block in 0..16 {
            let at = ((block % 4) * SIDE_4, (block / 4) * SIDE_4);
            place(
                &mut r,
                MB_SIDE,
                at,
                SIDE_4,
                &residual.luma(plane).take_4x4(block),
            );
        }
    }
    add(&mut r, (MB_SIDE, MB_SIDE), dpcm, dst, depth);
}

/// Add the chroma blocks of component `comp` (4:2:0 or 4:2:2: a `w` x
/// `h` block of 4x4 blocks, two per row) to its prediction `dst`, with
/// the DPCM over the whole block, and clear them.
pub(super) fn add_chroma<C: Coefficient, T: Sample>(
    residual: &mut Residual<C>,
    (comp, (w, h)): (usize, (usize, usize)),
    dpcm: Dpcm,
    dst: &mut PlaneViewMut<'_, T>,
    depth: BitDepth,
) {
    let mut r = [0i32; MAX_SAMPLES];
    for b in 0..(w / SIDE_4) * (h / SIDE_4) {
        let at = chroma::chroma_block_position(b);
        place(&mut r, w, at, SIDE_4, &residual.take_chroma_4x4(comp, b));
    }
    add(&mut r, (w, h), dpcm, dst, depth);
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_common::Plane;

    #[test]
    fn dpcm_accumulates_along_its_direction() {
        let mut plane = Plane::<u8>::new(4, 4, 0).unwrap();
        let mut r: [i32; 16] = core::array::from_fn(|i| i as i32 % 3);
        let rows: Vec<i32> = r.to_vec();
        add(
            &mut r,
            (4, 4),
            Dpcm::Vertical,
            &mut plane.view_mut(),
            BitDepth::EIGHT,
        );
        let view = plane.view();
        for x in 0..4 {
            let mut sum = 0;
            for y in 0..4 {
                sum += rows[y * 4 + x];
                assert_eq!(i32::from(view.row(y)[x]), sum);
            }
        }
    }
}
