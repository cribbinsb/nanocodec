//! `Intra_4x4` luma prediction (H.264 §8.3.1) and the encoder's per-block
//! loop for `I_NxN` macroblocks with 4x4 transforms.
//!
//! The sixteen 4x4 blocks are coded in decoding order (`luma4x4BlkIdx`,
//! §6.4.3: 8x8 quadrants in raster order, 4x4 blocks in raster order within
//! each). Each block is predicted from the reconstruction of the blocks
//! before it, so every block is predicted, transformed, quantised and
//! reconstructed before the next one is looked at. The mode of each block
//! is the one with the lowest SATD plus `lambda` times its signalling bits,
//! among the shortlist of [`crate::intra_nxn`] (or every mode). The search
//! is this module's own, not `intra_nxn`'s generic one: generic, the 4x4
//! predictor was no longer inlined per mode and key frames cost 4-20% more
//! instructions.

use nc_common::{Picture, PlaneIndex, PlaneView, PlaneViewMut};
use nc_dsp::{BlockSize, Dsp, H264Quantizer};

use crate::macroblock::{LumaTrellis, all_zero, scale_4x4};
use crate::residual::trellis::{RateModel, TrellisBlock, TrellisQuantizer};
use crate::residual::{BlockCat, SCAN_4X4, write_residual_block};
use crate::syntax::write_intra_4x4_mode;
use crate::tables::ZIGZAG_4X4;

use crate::intra_nxn::mode_bits;
pub use crate::intra_nxn::{
    DC, DIAGONAL_DOWN_LEFT, DIAGONAL_DOWN_RIGHT, HORIZONTAL, HORIZONTAL_DOWN, HORIZONTAL_UP,
    MODE_COUNT, VERTICAL, VERTICAL_LEFT, VERTICAL_RIGHT,
};

/// `(x, y)` in 4x4 units within the macroblock of block `luma4x4BlkIdx`.
pub const fn block_position(blk_idx: usize) -> (usize, usize) {
    let (qx, qy) = (2 * ((blk_idx / 4) % 2), 2 * (blk_idx / 8));
    (qx + blk_idx % 2, qy + (blk_idx % 4) / 2)
}

/// `luma4x4BlkIdx` of the block at `(bx, by)` (4x4 units).
pub const fn block_index(bx: usize, by: usize) -> usize {
    8 * (by / 2) + 4 * (bx / 2) + 2 * (by % 2) + bx % 2
}

/// The neighbouring samples of one 4x4 block: `top[i] = p[i, -1]` for
/// `i = 0..8` (positions 4..8 substituted when the top-right is missing),
/// `left[j] = p[-1, j]`, `corner = p[-1, -1]`, each with its availability.
pub(crate) struct Neighbours {
    top: [i32; 8],
    left: [i32; 4],
    corner: i32,
    has_top: bool,
    has_left: bool,
    has_corner: bool,
}

impl Neighbours {
    /// Neighbours of the 4x4 block at luma sample `(px, py)`, which is block
    /// `(bx, by)` of its macroblock. `top_right_mb` says whether the
    /// macroblock above-right exists (for the blocks on the top row).
    pub(crate) fn gather(
        recon: &PlaneView<'_>,
        px: usize,
        py: usize,
        bx: usize,
        by: usize,
        top_right_mb: bool,
    ) -> Self {
        let has_top = py > 0;
        let has_left = px > 0;
        // The top-right 4x4 block exists if it is inside the picture and
        // decoded already (§6.4.11.4): in the row above the macroblock it is
        // there unless it lies in an unavailable above-right macroblock;
        // inside the macroblock it must come earlier in decoding order.
        let has_top_right = has_top
            && if by == 0 {
                bx < 3 || top_right_mb
            } else {
                bx < 3 && block_index(bx + 1, by - 1) < block_index(bx, by)
            };
        let mut top = [0i32; 8];
        if has_top {
            let row = recon.row(py - 1);
            for (i, t) in top.iter_mut().enumerate().take(4) {
                *t = i32::from(row[px + i]);
            }
            for i in 4..8 {
                top[i] = if has_top_right {
                    i32::from(row[px + i])
                } else {
                    top[3]
                };
            }
        }
        let mut left = [0i32; 4];
        if has_left {
            for (j, l) in left.iter_mut().enumerate() {
                *l = i32::from(recon.row(py + j)[px - 1]);
            }
        }
        let has_corner = has_top && has_left;
        let corner = if has_corner {
            i32::from(recon.row(py - 1)[px - 1])
        } else {
            0
        };
        Self {
            top,
            left,
            corner,
            has_top,
            has_left,
            has_corner,
        }
    }

    /// Whether `mode` can be used (§8.3.1.2: each mode needs the samples it
    /// reads; DC works with none).
    pub(crate) fn allows(&self, mode: u8) -> bool {
        match mode {
            VERTICAL | DIAGONAL_DOWN_LEFT | VERTICAL_LEFT => self.has_top,
            HORIZONTAL | HORIZONTAL_UP => self.has_left,
            DC => true,
            _ => self.has_top && self.has_left && self.has_corner,
        }
    }

    /// `p[i, -1]` for `i >= -1`.
    fn t(&self, i: i32) -> i32 {
        if i < 0 {
            self.corner
        } else {
            self.top[i as usize]
        }
    }

    /// `p[-1, j]` for `j >= -1`.
    fn l(&self, j: i32) -> i32 {
        if j < 0 {
            self.corner
        } else {
            self.left[j as usize]
        }
    }

    /// The prediction of the block in `mode` (§8.3.1.2.1–9), raster order.
    ///
    /// The mode is matched once and each mode fills the block with its own
    /// loop; the per-sample formulas are those of the standard.
    pub(crate) fn predict(&self, mode: u8) -> [u8; 16] {
        let avg3 = |a: i32, b: i32, c: i32| ((a + 2 * b + c + 2) >> 2) as u8;
        let avg2 = |a: i32, b: i32| ((a + b + 1) >> 1) as u8;
        let mut out = [0u8; 16];
        let mut fill = |f: &dyn Fn(i32, i32) -> u8| {
            for (i, v) in out.iter_mut().enumerate() {
                *v = f((i % 4) as i32, (i / 4) as i32);
            }
        };
        match mode {
            VERTICAL => {
                for (i, v) in out.iter_mut().enumerate() {
                    *v = self.top[i % 4] as u8;
                }
            }
            HORIZONTAL => {
                for (i, v) in out.iter_mut().enumerate() {
                    *v = self.left[i / 4] as u8;
                }
            }
            DC => out = [self.dc() as u8; 16],
            DIAGONAL_DOWN_LEFT => fill(&|x, y| {
                if x == 3 && y == 3 {
                    ((self.t(6) + 3 * self.t(7) + 2) >> 2) as u8
                } else {
                    avg3(self.t(x + y), self.t(x + y + 1), self.t(x + y + 2))
                }
            }),
            DIAGONAL_DOWN_RIGHT => fill(&|x, y| match x.cmp(&y) {
                core::cmp::Ordering::Greater => {
                    avg3(self.t(x - y - 2), self.t(x - y - 1), self.t(x - y))
                }
                core::cmp::Ordering::Less => {
                    avg3(self.l(y - x - 2), self.l(y - x - 1), self.l(y - x))
                }
                core::cmp::Ordering::Equal => avg3(self.t(0), self.corner, self.l(0)),
            }),
            VERTICAL_RIGHT => fill(&|x, y| {
                let i = x - (y >> 1);
                match 2 * x - y {
                    0 | 2 | 4 | 6 => avg2(self.t(i - 1), self.t(i)),
                    1 | 3 | 5 => avg3(self.t(i - 2), self.t(i - 1), self.t(i)),
                    -1 => avg3(self.l(0), self.corner, self.t(0)),
                    _ => avg3(self.l(y - 1), self.l(y - 2), self.l(y - 3)),
                }
            }),
            HORIZONTAL_DOWN => fill(&|x, y| {
                let j = y - (x >> 1);
                match 2 * y - x {
                    0 | 2 | 4 | 6 => avg2(self.l(j - 1), self.l(j)),
                    1 | 3 | 5 => avg3(self.l(j - 2), self.l(j - 1), self.l(j)),
                    -1 => avg3(self.l(0), self.corner, self.t(0)),
                    _ => avg3(self.t(x - 1), self.t(x - 2), self.t(x - 3)),
                }
            }),
            VERTICAL_LEFT => fill(&|x, y| {
                let i = x + (y >> 1);
                if y % 2 == 0 {
                    avg2(self.t(i), self.t(i + 1))
                } else {
                    avg3(self.t(i), self.t(i + 1), self.t(i + 2))
                }
            }),
            HORIZONTAL_UP => fill(&|x, y| {
                let j = y + (x >> 1);
                match x + 2 * y {
                    0 | 2 | 4 => avg2(self.l(j), self.l(j + 1)),
                    1 | 3 => avg3(self.l(j), self.l(j + 1), self.l(j + 2)),
                    5 => ((self.l(2) + 3 * self.l(3) + 2) >> 2) as u8,
                    _ => self.l(3) as u8,
                }
            }),
            _ => unreachable!("Intra4x4PredMode is 0..=8"),
        }
        out
    }

    /// DC prediction (§8.3.1.2.3).
    fn dc(&self) -> i32 {
        let sum_top: i32 = self.top[..4].iter().sum();
        let sum_left: i32 = self.left.iter().sum();
        match (self.has_top, self.has_left) {
            (true, true) => (sum_top + sum_left + 4) >> 3,
            (true, false) => (sum_top + 2) >> 2,
            (false, true) => (sum_left + 2) >> 2,
            (false, false) => 128,
        }
    }
}

/// The modes and quantised levels of one `I_NxN` macroblock coded with 4x4
/// transforms.
#[derive(Debug, Clone)]
pub struct Intra4x4Result {
    /// `Intra4x4PredMode` per block, raster order (`by * 4 + bx`).
    pub modes: [u8; 16],
    /// Levels per block, raster order of blocks, raster within each block.
    pub levels: [[i32; 16]; 16],
    /// Sum of the chosen modes' SATD plus `lambda` times their bits.
    pub cost: f32,
}

/// A block's chosen mode, its prediction and its cost.
pub(crate) type ModeChoice = (u8, [u8; 16], f32);

/// Rate-distortion coding of each block (`h264.intra_rd`) and its trellis
/// quantisation (`quant.trellis` with `h264.trellis_i4x4`): the CABAC
/// prices and the `coded_block_flag` neighbours outside the macroblock
/// (`luma`, whose λ is the trellis's), and the search's buffers.
pub struct Intra4x4Rdo<'a, 'b> {
    pub luma: &'a LumaTrellis<'b>,
    pub search: &'a mut TrellisQuantizer,
    /// Trellis-quantise each block's levels.
    pub trellis: bool,
    /// If set, choose each block's mode by rate-distortion cost instead of
    /// by SATD.
    pub mode_rd: Option<ModeRd>,
}

/// Rate-distortion mode choice of `Intra_4x4` blocks.
#[derive(Debug, Clone, Copy)]
pub struct ModeRd {
    /// Squared error per bit.
    pub lambda: f32,
    /// SATD per mode bit, for the pre-ranking.
    pub satd_lambda: f32,
    /// The modes with the lowest SATD cost that are coded and priced.
    pub candidates: usize,
}

impl Intra4x4Rdo<'_, '_> {
    /// `ctxIdxInc` of the `coded_block_flag` of block `(bx, by)`; `coded`
    /// holds the levels of the macroblock's blocks decided so far (raster).
    fn cbf_ctx_inc(&self, (bx, by): (usize, usize), coded: &[[i32; 16]; 16]) -> usize {
        let coded = |bx: usize, by: usize| !all_zero(&coded[by * 4 + bx]);
        let a = if bx > 0 {
            coded(bx - 1, by)
        } else {
            self.luma.left_coded[by]
        };
        let b = if by > 0 {
            coded(bx, by - 1)
        } else {
            self.luma.above_coded[bx]
        };
        usize::from(a) + 2 * usize::from(b)
    }

    /// Trellis-quantise the plain `levels` of a block with `coded_block_flag`
    /// context `cbf_ctx_inc` from its transform coefficients `coeffs`.
    fn quantize(
        &mut self,
        cbf_ctx_inc: usize,
        coeffs: &[i32; 16],
        quantizer: &H264Quantizer,
        levels: &mut [i32; 16],
    ) {
        let settings = self.luma.settings;
        let mut model = RateModel::new(
            settings.bins,
            TrellisBlock::Small {
                cat: BlockCat::Luma4x4,
                cbf_ctx_inc,
            },
        );
        self.search.quantize(
            &mut model,
            coeffs,
            &ZIGZAG_4X4,
            (scale_4x4(quantizer), settings.lambda, settings.max_states),
            levels,
        );
    }

    /// The mode of one block with the lowest squared error plus λ times its
    /// CABAC bits (mode, `coded_block_flag` with context `cbf_ctx_inc`,
    /// levels), each mode coded as [`code_block`] codes it, among the
    /// `rd.candidates` modes with the lowest SATD cost.
    #[allow(clippy::too_many_arguments)]
    fn best_mode(
        &mut self,
        dsp: &Dsp,
        src: &PlaneView<'_>,
        neighbours: &Neighbours,
        predicted: u8,
        quantizer: &H264Quantizer,
        cbf_ctx_inc: usize,
        rd: ModeRd,
    ) -> ModeChoice {
        let bins = self.luma.settings.bins;
        let mut ranked = [(DC, [0; 16], f32::INFINITY); MODE_COUNT as usize];
        let mut allowed = 0;
        for m in (0..MODE_COUNT).filter(|&m| neighbours.allows(m)) {
            ranked[allowed] = evaluate_mode(dsp, src, neighbours, (predicted, rd.satd_lambda), m);
            allowed += 1;
        }
        let ranked = &mut ranked[..allowed];
        ranked.sort_by(|a, b| a.2.total_cmp(&b.2));
        let mut best: Option<ModeChoice> = None;
        for &(m, pred, _) in ranked.iter().take(rd.candidates.max(1)) {
            let (levels, recon) = code_block(dsp, src, &pred, quantizer, |coeffs, levels| {
                if self.trellis {
                    self.quantize(cbf_ctx_inc, coeffs, quantizer, levels);
                }
            });
            let distortion: u32 = (0..16)
                .map(|i| {
                    let d = i32::from(src.row(i / 4)[i % 4]) - i32::from(recon[i]);
                    (d * d) as u32
                })
                .sum();
            let mut counter = bins.fresh();
            write_intra_4x4_mode(&mut counter, m, predicted);
            write_residual_block(
                &mut counter,
                dsp,
                BlockCat::Luma4x4,
                &levels,
                &SCAN_4X4,
                cbf_ctx_inc,
            );
            let cost = distortion as f32 + rd.lambda * counter.bits();
            if best.is_none_or(|(_, _, c)| cost < c) {
                best = Some((m, pred, cost));
            }
        }
        best.expect("DC is always allowed")
    }
}

/// Transform and quantise `src - pred` of one 4x4 block, letting
/// `requantize(coefficients, levels)` revise the levels; returns the levels
/// and the reconstruction `pred` + residual (raster).
fn code_block(
    dsp: &Dsp,
    src: &PlaneView<'_>,
    pred: &[u8; 16],
    quantizer: &H264Quantizer,
    requantize: impl FnOnce(&[i32; 16], &mut [i32; 16]),
) -> ([i32; 16], [u8; 16]) {
    let block4 = BlockSize::new(4, 4).expect("4x4 is a kernel size");
    let pred_view = PlaneView::new(pred, 4, 4, 4).expect("prediction covers the block");
    let mut residual = [0i16; 16];
    dsp.residual(block4, src, &pred_view, &mut residual, 4);
    let mut coeffs = [[0i32; 16]; 1];
    dsp.h264_forward_4x4(&residual, 4, 1, &mut coeffs);
    let transformed = coeffs[0];
    dsp.h264_quantize_4x4(&mut coeffs, quantizer);
    requantize(&transformed, &mut coeffs[0]);
    let mut recon = *pred;
    let mut dst = PlaneViewMut::new(&mut recon, 4, 4, 4).expect("4x4 buffer");
    dsp.h264_add_inverse_4x4(&coeffs, None, quantizer, 1, &mut dst);
    (coeffs[0], recon)
}

/// Code the luma of the macroblock at `(x, y)` as `Intra_4x4`, writing its
/// reconstruction into `recon`. `predicted_mode(bx, by, modes)` returns
/// `predIntra4x4PredMode` of block `(bx, by)` given the modes chosen so far
/// in this macroblock (raster order). `lambda` weighs mode bits against SATD;
/// `all_modes` evaluates every mode instead of the shortlist. With `rdo`,
/// each block's levels may be trellis-quantised and its mode chosen by
/// rate-distortion cost (the result's `cost` is then that cost).
///
/// Returns `None`, with the reconstruction partly written, as soon as the
/// cost reaches `abort_above`: a caller that keeps the cheapest choice
/// loses nothing by stopping there, since the cost only grows.
#[allow(clippy::too_many_arguments)]
pub fn code_intra_4x4(
    dsp: &Dsp,
    source: &Picture,
    recon: &mut Picture,
    (x, y): (usize, usize),
    top_right_mb: bool,
    quantizer: &H264Quantizer,
    (lambda, all_modes): (f32, bool),
    predicted_mode: impl Fn(usize, usize, &[u8; 16]) -> u8,
    rdo: Option<Intra4x4Rdo<'_, '_>>,
    abort_above: f32,
) -> Option<Intra4x4Result> {
    code_intra_4x4_with(
        dsp,
        source,
        recon,
        (x, y),
        top_right_mb,
        quantizer,
        predicted_mode,
        |src, neighbours, predicted| {
            if all_modes {
                best_of_all_modes(dsp, src, neighbours, predicted, lambda)
            } else {
                best_mode(dsp, src, neighbours, predicted, lambda)
            }
        },
        rdo,
        abort_above,
    )
}

/// [`code_intra_4x4`] with each block's mode chosen by `choose_mode(source
/// block, neighbours, predicted mode)`, which must return a mode the
/// neighbours allow (unless `rdo` chooses by rate-distortion cost).
#[allow(clippy::too_many_arguments)]
pub(crate) fn code_intra_4x4_with(
    dsp: &Dsp,
    source: &Picture,
    recon: &mut Picture,
    (x, y): (usize, usize),
    top_right_mb: bool,
    quantizer: &H264Quantizer,
    predicted_mode: impl Fn(usize, usize, &[u8; 16]) -> u8,
    mut choose_mode: impl FnMut(&PlaneView<'_>, &Neighbours, u8) -> ModeChoice,
    mut rdo: Option<Intra4x4Rdo<'_, '_>>,
    abort_above: f32,
) -> Option<Intra4x4Result> {
    let mut modes = [DC; 16];
    let mut levels = [[0i32; 16]; 16];
    let mut cost = 0.0;
    for blk_idx in 0..16 {
        let (bx, by) = block_position(blk_idx);
        let (px, py) = (x + 4 * bx, y + 4 * by);
        let src = source
            .plane(PlaneIndex::Y)
            .region(px as isize, py as isize, 4, 4)
            .expect("block inside the picture");
        let neighbours = Neighbours::gather(
            &recon.plane(PlaneIndex::Y).view(),
            px,
            py,
            bx,
            by,
            top_right_mb,
        );
        let predicted = predicted_mode(bx, by, &modes);
        let cbf_ctx_inc = rdo.as_ref().map(|r| r.cbf_ctx_inc((bx, by), &levels));
        let mode_rd = rdo.as_ref().and_then(|r| r.mode_rd);
        let (mode, pred, mode_cost) = match (rdo.as_mut(), cbf_ctx_inc, mode_rd) {
            (Some(r), Some(inc), Some(rd)) => {
                r.best_mode(dsp, &src, &neighbours, predicted, quantizer, inc, rd)
            }
            _ => choose_mode(&src, &neighbours, predicted),
        };
        modes[by * 4 + bx] = mode;
        cost += mode_cost;
        if cost >= abort_above {
            return None;
        }

        // Transform, quantise and reconstruct this block before the next one
        // predicts from it.
        let (block_levels, block_recon) =
            code_block(dsp, &src, &pred, quantizer, |coeffs, block_levels| {
                if let (Some(r), Some(inc)) = (rdo.as_mut(), cbf_ctx_inc)
                    && r.trellis
                {
                    r.quantize(inc, coeffs, quantizer, block_levels);
                }
            });
        levels[by * 4 + bx] = block_levels;
        let mut dst = recon
            .plane_mut(PlaneIndex::Y)
            .region_mut(px as isize, py as isize, 4, 4)
            .expect("block inside the picture");
        for r in 0..4 {
            dst.row_mut(r)
                .copy_from_slice(&block_recon[r * 4..r * 4 + 4]);
        }
    }
    Some(Intra4x4Result {
        modes,
        levels,
        cost,
    })
}

/// The prediction of one block in mode `m`, and its SATD plus `lambda`
/// times its signalling bits.
fn evaluate_mode(
    dsp: &Dsp,
    src: &PlaneView<'_>,
    neighbours: &Neighbours,
    (predicted, lambda): (u8, f32),
    m: u8,
) -> ModeChoice {
    let block4 = BlockSize::new(4, 4).expect("4x4 is a kernel size");
    let pred = neighbours.predict(m);
    let view = PlaneView::new(&pred, 4, 4, 4).expect("prediction covers the block");
    let bits = mode_bits(m, predicted);
    (m, pred, dsp.satd(block4, src, &view) as f32 + lambda * bits)
}

/// The cheaper of two mode choices (the first on a tie).
fn better(a: ModeChoice, b: ModeChoice) -> ModeChoice {
    if b.2 < a.2 { b } else { a }
}

/// The mode of one block with the lowest SATD plus `lambda` times its bits,
/// among every mode the neighbours allow (`h264.intra4x4_all_modes`).
fn best_of_all_modes(
    dsp: &Dsp,
    src: &PlaneView<'_>,
    neighbours: &Neighbours,
    predicted: u8,
    lambda: f32,
) -> ModeChoice {
    let evaluate = |m| evaluate_mode(dsp, src, neighbours, (predicted, lambda), m);
    (0..MODE_COUNT)
        .filter(|&m| m != DC && neighbours.allows(m))
        .fold(evaluate(DC), |best, m| better(best, evaluate(m)))
}

/// The mode of one block with the lowest SATD plus `lambda` times its bits,
/// from the shortlist in the module comment.
fn best_mode(
    dsp: &Dsp,
    src: &PlaneView<'_>,
    neighbours: &Neighbours,
    predicted: u8,
    lambda: f32,
) -> ModeChoice {
    let evaluate = |m| evaluate_mode(dsp, src, neighbours, (predicted, lambda), m);
    // Shortlist: the three basic modes and the predicted one, then only
    // the diagonal modes next to the best basic direction.
    let mut best = evaluate(DC);
    for m in [VERTICAL, HORIZONTAL, predicted] {
        if m != DC && neighbours.allows(m) {
            best = better(best, evaluate(m));
        }
    }
    let refinements: &[u8] = match best.0 {
        VERTICAL => &[VERTICAL_LEFT, VERTICAL_RIGHT],
        HORIZONTAL => &[HORIZONTAL_UP, HORIZONTAL_DOWN],
        DC => &[DIAGONAL_DOWN_LEFT, DIAGONAL_DOWN_RIGHT],
        _ => &[],
    };
    for &m in refinements {
        if m != predicted && neighbours.allows(m) {
            best = better(best, evaluate(m));
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_order_round_trips() {
        for idx in 0..16 {
            let (bx, by) = block_position(idx);
            assert_eq!(block_index(bx, by), idx);
        }
        assert_eq!(block_position(5), (3, 0));
        assert_eq!(block_position(10), (0, 3));
    }

    fn neighbours(top: [i32; 8], left: [i32; 4], corner: i32) -> Neighbours {
        Neighbours {
            top,
            left,
            corner,
            has_top: true,
            has_left: true,
            has_corner: true,
        }
    }

    #[test]
    fn flat_neighbours_predict_flat_blocks() {
        let n = neighbours([90; 8], [90; 4], 90);
        for mode in 0..9 {
            assert_eq!(n.predict(mode), [90; 16], "mode {mode}");
        }
    }

    #[test]
    fn directional_modes_follow_their_edges() {
        let n = neighbours([10, 20, 30, 40, 50, 60, 70, 80], [11, 21, 31, 41], 5);
        let v = n.predict(VERTICAL);
        let h = n.predict(HORIZONTAL);
        for i in 0..16 {
            assert_eq!(i32::from(v[i]), n.top[i % 4]);
            assert_eq!(i32::from(h[i]), n.left[i / 4]);
        }
        // Diagonal down-left at (3, 3): (p[6,-1] + 3 * p[7,-1] + 2) >> 2.
        assert_eq!(
            n.predict(DIAGONAL_DOWN_LEFT)[15],
            ((70 + 3 * 80 + 2) >> 2) as u8
        );
        // Horizontal-up beyond zHU = 5 repeats p[-1, 3].
        assert_eq!(n.predict(HORIZONTAL_UP)[15], 41);
    }
}
