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
//! ([`crate::intra_nxn`]) or among every mode; or, with [`Intra8x8Rdo`]'s
//! mode choice, the one with the lowest squared error plus λ times its CABAC
//! bits among the modes with the lowest SATD cost (as `Intra_4x4` chooses
//! with `h264.intra_rd`). Its levels may be trellis-quantised.

use nc_common::{Picture, PlaneIndex, PlaneView, PlaneViewMut};
use nc_dsp::{BlockSize, Dsp, H264Intra8x8Edges, H264Intra8x8Refs, H264Quantizer};

use crate::cabac::BitCounter;
use crate::intra_nxn::{Choice, DC, MODE_COUNT, best_of_all_modes, best_of_shortlist, mode_bits};
use crate::macroblock::{all_zero, scale_8x8};
use crate::residual::trellis::{RateModel, TrellisBlock, TrellisQuantizer};
use crate::residual::write_luma_8x8_block;
use crate::syntax::write_intra_4x4_mode;
use crate::tables::ZIGZAG_8X8;

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
    pub levels: [[i16; AREA]; 4],
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

/// One coded 8x8 block: its levels and its reconstruction (raster).
pub(crate) type CodedBlock = ([i16; AREA], [u8; AREA]);

/// Transform and quantise `src - pred` of one 8x8 block, letting
/// `requantize(coefficients, levels)` revise the levels; returns the levels
/// and the reconstruction `pred` + residual (raster).
pub(crate) fn code_block(
    dsp: &Dsp,
    src: &PlaneView<'_>,
    pred: &[u8; AREA],
    quantizer: &H264Quantizer,
    requantize: impl FnOnce(&[i16; AREA], &mut [i16; AREA]),
) -> CodedBlock {
    let size = BlockSize::new(SIDE, SIDE).expect("8x8 is a kernel size");
    let pred_view = PlaneView::new(pred, SIDE, SIDE, SIDE).expect("prediction covers the block");
    let mut residual = [0i16; AREA];
    dsp.residual(size, src, &pred_view, &mut residual, SIDE);
    let mut levels = [0i16; AREA];
    dsp.h264_forward_8x8(&residual, SIDE, &mut levels);
    let transformed = levels;
    let mut recon = *pred;
    if dsp.h264_quantize_8x8(&mut levels, quantizer) == 0 {
        return (levels, recon);
    }
    requantize(&transformed, &mut levels);
    if all_zero(&levels) {
        return (levels, recon);
    }
    let mut dst = PlaneViewMut::new(&mut recon, SIDE, SIDE, SIDE).expect("8x8 buffer");
    dsp.h264_add_inverse_8x8(&levels, quantizer, &mut dst);
    (levels, recon)
}

/// Rate-distortion coding of each `Intra_8x8` block: trellis quantisation
/// of its levels (`h264.trellis_i8x8`) and its mode chosen by coded cost
/// (`h264.intra8x8_rd_candidates`), priced with the slice's CABAC contexts
/// as they stand (`bins`).
pub struct Intra8x8Rdo<'a, 'b> {
    pub bins: &'a BitCounter<'b>,
    pub search: &'a mut TrellisQuantizer,
    /// The trellis's λ (sample squared error per bit) and survivors per
    /// position, if the levels are trellis-quantised.
    pub trellis: Option<(f32, usize)>,
    /// The mode choice's λ (squared error per bit) and how many of the modes
    /// with the lowest SATD cost are coded and priced, if modes are chosen
    /// by coded cost.
    pub mode_rd: Option<(f32, usize)>,
}

impl Intra8x8Rdo<'_, '_> {
    /// Trellis-quantise the plain `levels` of a block from its transform
    /// coefficients `coeffs` (no `coded_block_flag`: 4:2:0 8x8 luma blocks
    /// signal coding in `coded_block_pattern`, §7.3.5.3.3).
    fn requantize(
        &mut self,
        coeffs: &[i16; AREA],
        quantizer: &H264Quantizer,
        levels: &mut [i16; AREA],
    ) {
        if let Some((lambda, max_states)) = self.trellis {
            let mut model = RateModel::new(self.bins, TrellisBlock::Luma8x8);
            self.search.quantize(
                &mut model,
                coeffs,
                &ZIGZAG_8X8,
                (scale_8x8(quantizer), lambda, max_states),
                levels,
            );
        }
    }

    /// Code one block predicted by `pred` (mode `m`), with the trellis if
    /// set.
    fn code(
        &mut self,
        dsp: &Dsp,
        src: &PlaneView<'_>,
        pred: &[u8; AREA],
        quantizer: &H264Quantizer,
    ) -> CodedBlock {
        code_block(dsp, src, pred, quantizer, |coeffs, levels| {
            self.requantize(coeffs, quantizer, levels);
        })
    }

    /// The mode of one block with the lowest squared error plus `lambda`
    /// times its CABAC bits (mode, levels) among the `candidates` allowed
    /// modes with the lowest SATD cost (`satd_lambda` per mode bit; ties go
    /// to the lower SATD cost, then the lower mode), each coded as
    /// [`Self::code`] codes it. Returns the mode, its cost and its coded
    /// block, which the caller keeps rather than coding the mode again.
    #[allow(clippy::too_many_arguments)]
    fn best_mode(
        &mut self,
        dsp: &Dsp,
        src: &PlaneView<'_>,
        refs: &H264Intra8x8Refs,
        (predicted, satd_lambda): (u8, f32),
        quantizer: &H264Quantizer,
        (lambda, candidates): (f32, usize),
    ) -> (u8, f32, CodedBlock) {
        let mut ranked = [(DC, [0; AREA], f32::INFINITY); MODE_COUNT as usize];
        let mut count = 0;
        for m in (0..MODE_COUNT).filter(|&m| refs.allows(m)) {
            ranked[count] = evaluate_mode(dsp, src, refs, (predicted, satd_lambda), m);
            count += 1;
        }
        let ranked = &mut ranked[..count];
        ranked.sort_by(|a, b| a.2.total_cmp(&b.2));
        let mut best: Option<(u8, f32, CodedBlock)> = None;
        for (m, pred, _) in ranked.iter().take(candidates.max(1)) {
            let coded = self.code(dsp, src, pred, quantizer);
            let cost = block_sse(src, &coded.1) + lambda * self.bits(dsp, *m, predicted, &coded.0);
            if best.as_ref().is_none_or(|&(_, c, _)| cost < c) {
                best = Some((*m, cost, coded));
            }
        }
        best.expect("DC is always allowed")
    }

    /// CABAC bits of a block's mode and levels.
    fn bits(&self, dsp: &Dsp, mode: u8, predicted: u8, levels: &[i16; AREA]) -> f32 {
        let mut counter = self.bins.fresh();
        // Intra8x8PredMode has Intra4x4PredMode's syntax (§7.3.5.1).
        write_intra_4x4_mode(&mut counter, mode, predicted);
        if !all_zero(levels) {
            write_luma_8x8_block(&mut counter, dsp, levels);
        }
        counter.bits()
    }
}

/// Squared error between one source block and its reconstruction (raster).
fn block_sse(src: &PlaneView<'_>, recon: &[u8; AREA]) -> f32 {
    let sse: u32 = recon
        .as_chunks::<SIDE>()
        .0
        .iter()
        .enumerate()
        .flat_map(|(r, row)| src.row(r)[..SIDE].iter().zip(row))
        .map(|(&s, &p)| {
            let d = i32::from(s) - i32::from(p);
            (d * d) as u32
        })
        .sum();
    sse as f32
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
/// With `rdo`, each block's levels may be trellis-quantised and its mode
/// chosen by rate-distortion cost (the result's `cost` is then that cost).
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
    mut rdo: Option<Intra8x8Rdo<'_, '_>>,
    abort_above: f32,
) -> Option<Intra8x8Result> {
    code_intra_8x8_with(
        source,
        recon,
        (x, y),
        top_right_mb,
        predicted_mode,
        |src, refs, predicted| {
            if let Some(r) = rdo.as_mut()
                && let Some(rd) = r.mode_rd
            {
                return r.best_mode(dsp, src, refs, (predicted, lambda), quantizer, rd);
            }
            let evaluate = |m| evaluate_mode(dsp, src, refs, (predicted, lambda), m);
            let allows = |m| refs.allows(m);
            let (mode, pred, cost) = if all_modes {
                best_of_all_modes(allows, evaluate)
            } else {
                best_of_shortlist(predicted, allows, evaluate)
            };
            let coded = match rdo.as_mut() {
                Some(r) => r.code(dsp, src, &pred, quantizer),
                None => code_block(dsp, src, &pred, quantizer, |_, _| {}),
            };
            (mode, cost, coded)
        },
        abort_above,
    )
}

/// [`code_intra_8x8`] with each block's mode chosen and coded by
/// `choose_mode(source block, references, predicted mode)`, which returns a
/// mode the references allow, its cost and the coded block.
pub(crate) fn code_intra_8x8_with(
    source: &Picture,
    recon: &mut Picture,
    (x, y): (usize, usize),
    top_right_mb: bool,
    predicted_mode: impl Fn(usize, usize, &[u8; 16]) -> u8,
    mut choose_mode: impl FnMut(&PlaneView<'_>, &H264Intra8x8Refs, u8) -> (u8, f32, CodedBlock),
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
        // The block is transformed, quantised and reconstructed before the
        // next one predicts from it.
        let (mode, cost, (levels, block_recon)) = choose_mode(&src, &refs, predicted);
        result.modes[b8] = mode;
        result.cost += cost;
        if result.cost >= abort_above {
            return None;
        }
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
