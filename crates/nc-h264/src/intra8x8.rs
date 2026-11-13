//! The encoder's `Intra_8x8` luma (H.264 §8.3.2): the per-block loop of
//! `I_NxN` macroblocks with `transform_size_8x8_flag`.
//!
//! The four 8x8 blocks are coded in raster (= decoding) order, each
//! predicted from the reconstruction of the blocks before it, so every
//! block is predicted, transformed (8x8, §8.5.13), quantised and
//! reconstructed before the next one is looked at. The prediction kernels
//! (reference sample filter and the nine modes) are in `nc_dsp`. Each
//! block's mode is the one with the lowest 8x8 SATD plus `lambda` times its
//! signalling bits, from the same shortlist as `Intra_4x4`
//! ([`crate::intra_nxn`]) or among every mode.

use nc_common::{Picture, PlaneIndex, PlaneView, PlaneViewMut};
use nc_dsp::{BlockSize, Dsp, H264Intra8x8Edges, H264Intra8x8Refs, H264Quantizer};

use crate::intra_nxn::{Choice, DC, best_of_all_modes, best_of_shortlist, mode_bits};

/// Block side.
const SIDE: usize = 8;
/// Samples per block.
const AREA: usize = SIDE * SIDE;

/// A block's chosen mode, its prediction and its cost.
pub(crate) type ModeChoice = Choice<[u8; AREA]>;

/// The modes and quantised levels of the luma of one `I_NxN` macroblock
/// coded with 8x8 transforms.
#[derive(Debug, Clone)]
pub struct Intra8x8Result {
    /// `Intra8x8PredMode` per 8x8 block, raster.
    pub modes: [u8; 4],
    /// Levels per 8x8 block, raster within each.
    pub levels: [[i32; AREA]; 4],
    /// Sum of the chosen modes' SATD plus `lambda` times their bits.
    pub cost: f32,
}

/// The mode of every 4x4 block (raster, `by * 4 + bx`) of a macroblock
/// whose 8x8 blocks have `modes`: what mode prediction of later blocks
/// reads (§8.3.1.1 and §8.3.2.1 take an `Intra_8x8` neighbour's mode for
/// each of its 4x4 blocks).
pub fn modes_per_4x4(modes: [u8; 4]) -> [u8; 16] {
    core::array::from_fn(|i| modes[(i / 8) * 2 + (i % 4) / 2])
}

/// The neighbours of 8x8 block `b8` (raster) of the macroblock at luma
/// `(x, y)`. `top_right_mb` says whether the macroblock above-right exists.
fn gather(
    recon: &PlaneView<'_>,
    (x, y): (usize, usize),
    b8: usize,
    top_right_mb: bool,
) -> H264Intra8x8Edges {
    let (px, py) = (x + SIDE * (b8 % 2), y + SIDE * (b8 / 2));
    let has_top = py > 0;
    let has_left = px > 0;
    // The samples above-right (§6.4.11.2): above the macroblock they are
    // there unless they lie in a missing above-right macroblock; inside it,
    // block 1 is decoded before block 2 and block 3's lie in the next
    // macroblock.
    let has_top_right = match b8 {
        0 => has_top,
        1 => has_top && top_right_mb,
        2 => true,
        _ => false,
    };
    let mut edges = H264Intra8x8Edges {
        top: [0; 2 * SIDE],
        left: [0; SIDE],
        corner: 0,
        has_top,
        has_top_right,
        has_left,
        has_corner: has_top && has_left,
    };
    if has_top {
        let row = recon.row(py - 1);
        let width = if has_top_right { 2 * SIDE } else { SIDE };
        edges.top[..width].copy_from_slice(&row[px..px + width]);
    }
    if has_left {
        for (j, l) in edges.left.iter_mut().enumerate() {
            *l = recon.row(py + j)[px - 1];
        }
    }
    if edges.has_corner {
        edges.corner = recon.row(py - 1)[px - 1];
    }
    edges
}

/// Transform and quantise `src - pred` of one 8x8 block; returns the levels
/// and the reconstruction `pred` + residual (raster).
fn code_block(
    dsp: &Dsp,
    src: &PlaneView<'_>,
    pred: &[u8; AREA],
    quantizer: &H264Quantizer,
) -> ([i32; AREA], [u8; AREA]) {
    let size = BlockSize::new(SIDE, SIDE).expect("8x8 is a kernel size");
    let pred_view = PlaneView::new(pred, SIDE, SIDE, SIDE).expect("prediction covers the block");
    let mut residual = [0i16; AREA];
    dsp.residual(size, src, &pred_view, &mut residual, SIDE);
    let mut levels = [0i32; AREA];
    dsp.h264_forward_8x8(&residual, SIDE, &mut levels);
    let mut recon = *pred;
    if dsp.h264_quantize_8x8(&mut levels, quantizer) == 0 {
        return (levels, recon);
    }
    let mut dst = PlaneViewMut::new(&mut recon, SIDE, SIDE, SIDE).expect("8x8 buffer");
    dsp.h264_add_inverse_8x8(&levels, quantizer, &mut dst);
    (levels, recon)
}

/// The prediction of one block in mode `m`, and its SATD plus `lambda`
/// times its signalling bits.
fn evaluate_mode(
    dsp: &Dsp,
    src: &PlaneView<'_>,
    refs: &H264Intra8x8Refs,
    (predicted, lambda): (u8, f32),
    m: u8,
) -> ModeChoice {
    let size = BlockSize::new(SIDE, SIDE).expect("8x8 is a kernel size");
    let mut pred = [0u8; AREA];
    dsp.h264_intra_8x8(refs, m, &mut pred);
    let view = PlaneView::new(&pred, SIDE, SIDE, SIDE).expect("prediction covers the block");
    let cost = dsp.satd(size, src, &view) as f32 + lambda * mode_bits(m, predicted);
    (m, pred, cost)
}

/// Code the luma of the macroblock at `(x, y)` as `Intra_8x8`, writing its
/// reconstruction into `recon`. `predicted_mode(bx, by, modes)` returns the
/// predicted mode of the block whose top-left 4x4 block is `(bx, by)`,
/// given the modes chosen so far per 4x4 block (raster; §8.3.2.1 reads the
/// same neighbouring 4x4 blocks as §8.3.1.1). `lambda` weighs mode bits
/// against SATD; `all_modes` evaluates every mode instead of the shortlist.
///
/// Returns `None`, with the reconstruction partly written, as soon as the
/// cost reaches `abort_above`: a caller that keeps the cheapest choice
/// loses nothing by stopping there, since the cost only grows.
#[allow(clippy::too_many_arguments)]
pub fn code_intra_8x8(
    dsp: &Dsp,
    source: &Picture,
    recon: &mut Picture,
    (x, y): (usize, usize),
    top_right_mb: bool,
    quantizer: &H264Quantizer,
    (lambda, all_modes): (f32, bool),
    predicted_mode: impl Fn(usize, usize, &[u8; 16]) -> u8,
    abort_above: f32,
) -> Option<Intra8x8Result> {
    code_intra_8x8_with(
        dsp,
        source,
        recon,
        (x, y),
        top_right_mb,
        quantizer,
        predicted_mode,
        |src, refs, predicted| {
            let evaluate = |m| evaluate_mode(dsp, src, refs, (predicted, lambda), m);
            let allows = |m| refs.allows(m);
            if all_modes {
                best_of_all_modes(allows, evaluate)
            } else {
                best_of_shortlist(predicted, allows, evaluate)
            }
        },
        abort_above,
    )
}

/// [`code_intra_8x8`] with each block's mode chosen by `choose_mode(source
/// block, references, predicted mode)`, which must return a mode the
/// references allow.
#[allow(clippy::too_many_arguments)]
pub(crate) fn code_intra_8x8_with(
    dsp: &Dsp,
    source: &Picture,
    recon: &mut Picture,
    (x, y): (usize, usize),
    top_right_mb: bool,
    quantizer: &H264Quantizer,
    predicted_mode: impl Fn(usize, usize, &[u8; 16]) -> u8,
    mut choose_mode: impl FnMut(&PlaneView<'_>, &H264Intra8x8Refs, u8) -> ModeChoice,
    abort_above: f32,
) -> Option<Intra8x8Result> {
    let mut result = Intra8x8Result {
        modes: [DC; 4],
        levels: [[0; AREA]; 4],
        cost: 0.0,
    };
    for b8 in 0..4 {
        let (px, py) = (x + SIDE * (b8 % 2), y + SIDE * (b8 / 2));
        let src = source
            .plane(PlaneIndex::Y)
            .region(px as isize, py as isize, SIDE, SIDE)
            .expect("block inside the picture");
        let refs = gather(&recon.plane(PlaneIndex::Y).view(), (x, y), b8, top_right_mb).filtered();
        let own = modes_per_4x4(result.modes);
        let predicted = predicted_mode(2 * (b8 % 2), 2 * (b8 / 2), &own);
        let (mode, pred, cost) = choose_mode(&src, &refs, predicted);
        result.modes[b8] = mode;
        result.cost += cost;
        if result.cost >= abort_above {
            return None;
        }

        // Transform, quantise and reconstruct this block before the next one
        // predicts from it.
        let (levels, block_recon) = code_block(dsp, &src, &pred, quantizer);
        result.levels[b8] = levels;
        let mut dst = recon
            .plane_mut(PlaneIndex::Y)
            .region_mut(px as isize, py as isize, SIDE, SIDE)
            .expect("block inside the picture");
        for (r, row) in block_recon.as_chunks::<SIDE>().0.iter().enumerate() {
            dst.row_mut(r).copy_from_slice(row);
        }
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes_cover_their_quadrants() {
        let per_4x4 = modes_per_4x4([0, 1, 2, 3]);
        assert_eq!(
            per_4x4,
            [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3],
            "raster 4x4 blocks take their quadrant's mode"
        );
    }
}
