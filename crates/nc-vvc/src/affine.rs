//! H.266 affine motion (§8.5.5): a CU's 4- or 6-parameter model from its
//! control-point vectors, the 4x4 sub-block vectors derived from it
//! (§8.5.5.9, with the memory-bandwidth fallback), and the sub-block luma
//! and chroma prediction (§8.5.6.3 with the 6-tap affine luma filter and
//! averaged chroma vectors). No prediction refinement with optical flow
//! (`sps_affine_prof_enabled_flag` 0).
//!
//! Derived from `VVdeC` `UnitTools::setAllAffineMv` and
//! `InterPrediction::xPredAffineBlk` (BSD-3-Clause-Clear; `THIRD_PARTY.md`).
use crate::motion::{Mv16, SUB_BLOCK};
use nc_common::{Plane, PlaneIndex, PlaneViewMut};
use nc_dsp::{Dsp, Filter};
use nc_encoder::Block;

/// §8.5.5.9: control-point differences are scaled to `1 << MODEL_SHIFT`
/// per luma sample (`MAX_CU_DEPTH`).
const MODEL_SHIFT: u32 = 7;
/// §8.5.5.9 `fallbackModeTriggered`: the reference area one sub-block row
/// or column of a uni-predicted CU may span, in samples of a 6-tap window
/// (`VVdeC` `isSubblockVectorSpreadOverLimit`).
const SPREAD_TAPS: i64 = 6;
const SPREAD_LONG: i64 = SPREAD_TAPS + 9;
const SPREAD_SHORT: i64 = SPREAD_TAPS + 5;

/// An affine CU's motion: its control-point vectors (top-left, top-right
/// and, for the 6-parameter model, bottom-left; §8.5.5.2 `cpMvLX`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Model {
    pub cp: [Mv16; 3],
    /// `MotionModelIdc` 2 (6 parameters); else 1 (4 parameters, `cp[2]`
    /// unused).
    pub six: bool,
}

impl Model {
    /// The control points this model codes (2 or 3).
    pub(crate) fn points(&self) -> usize {
        if self.six { 3 } else { 2 }
    }

    /// The sub-block vector field of this model on a CU of `width` x
    /// `height` luma samples (§8.5.5.9).
    pub(crate) fn field(&self, width: usize, height: usize) -> Field {
        self.field_for_prediction(width, height, false)
    }

    /// Decoder fields also apply the bidirectional reference-area limit.
    pub(crate) fn field_for_prediction(
        &self,
        width: usize,
        height: usize,
        bidirectional: bool,
    ) -> Field {
        let [lt, rt, lb] = self.cp.map(|v| (i64::from(v.x), i64::from(v.y)));
        let scale = |d: i64, log2: u32| d << (MODEL_SHIFT - log2);
        let (log2_w, log2_h) = (width.ilog2(), height.ilog2());
        let hor = (scale(rt.0 - lt.0, log2_w), scale(rt.1 - lt.1, log2_w));
        let ver = if self.six {
            (scale(lb.0 - lt.0, log2_h), scale(lb.1 - lt.1, log2_h))
        } else {
            (-hor.1, hor.0)
        };
        let base = (lt.0 << MODEL_SHIFT, lt.1 << MODEL_SHIFT);
        let over_limit = if bidirectional {
            let step = 4 << (MODEL_SHIFT + 4);
            let span = |a: i64, b: i64, step: i64| {
                let values = [0, 4 * a + step, 4 * b, 4 * a + 4 * b + step];
                ((values.iter().max().unwrap() - values.iter().min().unwrap()) >> (MODEL_SHIFT + 4))
                    + SPREAD_TAPS
                    + 3
            };
            span(hor.0, ver.0, step) * span(ver.1, hor.1, step) > SPREAD_LONG * SPREAD_LONG
        } else {
            spread_over_limit(hor, ver)
        };
        let fallback = over_limit.then(|| {
            let (half_w, half_h) = (width as i64 / 2, height as i64 / 2);
            Mv16 {
                x: round_shift(base.0 + hor.0 * half_w + ver.0 * half_h, MODEL_SHIFT),
                y: round_shift(base.1 + hor.1 * half_w + ver.1 * half_h, MODEL_SHIFT),
            }
            .clip()
        });
        Field {
            base,
            hor,
            ver,
            fallback,
        }
    }
}

/// The vectors of an affine CU's 4x4 sub-blocks.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Field {
    /// The top-left control point, and the vector change per luma sample
    /// along x and along y, at `1 << MODEL_SHIFT`.
    base: (i64, i64),
    hor: (i64, i64),
    ver: (i64, i64),
    /// The one vector of every sub-block when the model spreads too far.
    fallback: Option<Mv16>,
}

impl Field {
    /// Per-sample PROF corrections, in 1/32-sample units, relative to a 4×4 centre.
    /// Constant fields and the bandwidth fallback have no optical-flow refinement.
    pub(crate) fn refinement(&self) -> Option<[[i32; 2]; 16]> {
        if self.fallback.is_some() || (self.hor == (0, 0) && self.ver == (0, 0)) {
            return None;
        }
        Some(std::array::from_fn(|i| {
            let x = 4 * (i % 4) as i64 - 6;
            let y = 4 * (i / 4) as i64 - 6;
            [
                round_shift(self.hor.0 * x + self.ver.0 * y, 8).clamp(-31, 31),
                round_shift(self.hor.1 * x + self.ver.1 * y, 8).clamp(-31, 31),
            ]
        }))
    }

    /// The vector of sub-block (`col`, `row`): the model at its centre.
    pub(crate) fn at(&self, col: usize, row: usize) -> Mv16 {
        if let Some(mv) = self.fallback {
            return mv;
        }
        let centre = |i: usize| (SUB_BLOCK * i + SUB_BLOCK / 2) as i64;
        let (cx, cy) = (centre(col), centre(row));
        Mv16 {
            x: round_shift(self.base.0 + self.hor.0 * cx + self.ver.0 * cy, MODEL_SHIFT),
            y: round_shift(self.base.1 + self.hor.1 * cx + self.ver.1 * cy, MODEL_SHIFT),
        }
        .clip()
    }
}

impl Field {
    /// The model's vector at the centre of the `size`-sample block
    /// (`col`, `row`) of the CU (the encoder's coarser search grid).
    pub(crate) fn at_block(&self, size: usize, col: usize, row: usize) -> Mv16 {
        if let Some(mv) = self.fallback {
            return mv;
        }
        let centre = |i: usize| (size * i + size / 2) as i64;
        let (cx, cy) = (centre(col), centre(row));
        Mv16 {
            x: round_shift(self.base.0 + self.hor.0 * cx + self.ver.0 * cy, MODEL_SHIFT),
            y: round_shift(self.base.1 + self.hor.1 * cx + self.ver.1 * cy, MODEL_SHIFT),
        }
        .clip()
    }
}

/// `roundAffineMv`: `v / 2^shift` rounded to nearest, ties towards zero.
pub(crate) fn round_shift(v: i64, shift: u32) -> i32 {
    ((v + (1 << (shift - 1)) - i64::from(v >= 0)) >> shift) as i32
}

/// §8.5.5.9: whether a uni-predicted model's sub-block row or column reads
/// more reference samples than the bandwidth limit allows (then every
/// sub-block takes the centre vector).
fn spread_over_limit(hor: (i64, i64), ver: (i64, i64)) -> bool {
    let step = (SUB_BLOCK as i64) << (MODEL_SHIFT + 4);
    let span = |d: i64| (d.abs() >> (MODEL_SHIFT + 4)) + SPREAD_TAPS + 3;
    let sub = SUB_BLOCK as i64;
    span(sub * hor.0 + step) * span(sub * hor.1) > SPREAD_LONG * SPREAD_SHORT
        || span(sub * ver.0) * span(sub * ver.1 + step) > SPREAD_SHORT * SPREAD_LONG
}

/// The affine prediction of `plane` of CU `cu` from `reference` into `out`
/// (the CU's samples of that plane): luma per 4x4 sub-block at
/// its vector with the 6-tap filter; chroma per 4x4 chroma block at the
/// rounded mean of the vectors of its top-left and bottom-right luma
/// sub-blocks (§8.5.5.9 `mvCLX`), 1/32-sample.
pub(crate) fn predict(
    dsp: &Dsp,
    reference: &Plane,
    plane: PlaneIndex,
    cu: &Block,
    field: &Field,
    out: &mut PlaneViewMut<'_>,
) {
    let chroma = plane != PlaneIndex::Y;
    let shift = usize::from(chroma);
    let (w, h) = (cu.width() >> shift, cu.height() >> shift);
    let per_row = w / SUB_BLOCK;
    let mut vectors = [Mv16::ZERO; crate::inter::MAX_SUB_BLOCKS];
    let vectors = &mut vectors[..per_row * (h / SUB_BLOCK)];
    for (i, mv) in vectors.iter_mut().enumerate() {
        let (col, row) = (i % per_row, i / per_row);
        *mv = if chroma {
            let (a, b) = (
                field.at(2 * col, 2 * row),
                field.at(2 * col + 1, 2 * row + 1),
            );
            let mean = |a: i32, b: i32| round_shift(i64::from(a) + i64::from(b), 1);
            Mv16 {
                x: mean(a.x, b.x),
                y: mean(a.y, b.y),
            }
        } else {
            field.at(col, row)
        };
    }
    let filter = if chroma {
        Filter::VvcChroma
    } else {
        Filter::VvcAffineLuma
    };
    crate::inter::predict_sub_blocks(
        dsp,
        filter,
        reference,
        (cu.x >> shift, cu.y >> shift),
        w,
        SUB_BLOCK,
        vectors,
        out,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mv(x: i32, y: i32) -> Mv16 {
        Mv16 { x, y }
    }

    #[test]
    fn a_translation_moves_every_sub_block_alike() {
        let model = Model {
            cp: [mv(-37, 21); 3],
            six: false,
        };
        let field = model.field(32, 16);
        for (col, row) in [(0, 0), (7, 3), (3, 1)] {
            assert_eq!(field.at(col, row), mv(-37, 21));
        }
    }

    #[test]
    fn a_zoom_grows_outwards_from_the_top_left() {
        // Top-right control point 16 sixteenths right of the top-left one
        // on a 16-wide CU: one sixteenth more per sample, both axes.
        let model = Model {
            cp: [mv(0, 0), mv(16, 0), mv(0, 0)],
            six: false,
        };
        let field = model.field(16, 16);
        assert_eq!(field.at(0, 0), mv(2, 2));
        assert_eq!(field.at(3, 0), mv(14, 2));
        assert_eq!(field.at(1, 2), mv(6, 10));
    }

    #[test]
    fn a_wide_spread_falls_back_to_the_centre_vector() {
        let model = Model {
            cp: [mv(0, 0), mv(16 * 64, 0), mv(0, 0)],
            six: false,
        };
        let field = model.field(16, 16);
        assert_eq!(field.at(0, 0), field.at(3, 3));
        assert_eq!(field.at(0, 0), mv(512, 512));
    }

    #[test]
    fn rounding_ties_go_towards_zero() {
        assert_eq!(round_shift(64, 7), 0);
        assert_eq!(round_shift(65, 7), 1);
        assert_eq!(round_shift(-64, 7), 0);
        assert_eq!(round_shift(-65, 7), -1);
        assert_eq!(round_shift(3, 1), 1);
        assert_eq!(round_shift(-3, 1), -1);
    }
}
