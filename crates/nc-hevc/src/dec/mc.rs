//! Inter sample prediction (H.265 §8.5.3.3): fractional sample
//! interpolation at 14-bit intermediate precision (§8.5.3.3.3) and the
//! default and explicit weighted sample prediction (§8.5.3.3.4).
//!
//! Uni-directional prediction with default weights is the encoder's case;
//! it goes through the nc-dsp rounding kernels (`crate::inter`) whenever
//! the block splits into power-of-two pieces, which gives the same samples.
//! Bi-prediction, explicit weights and 2- or 6-sample chroma sides take the
//! 14-bit samples of each list (`crate::inter::predict_samples`) and
//! combine them with the weighted prediction kernel.
//!
//! Written once over the sample type (`docs/handoff/high-bit-depth.md`):
//! each plane's kernels run at that plane's depth, which sets the
//! weighted-prediction shifts (`shift1 = 14 − BitDepth`, `shift2 = 15 −
//! BitDepth`); the explicit offsets arrive scaled to it (`slice`).
//!
//! Chroma planes follow the picture's chroma format (none for 4:0:0): a
//! block's chroma is its luma rectangle subsampled per axis, and the luma
//! vector addresses it in eighth chroma samples scaled per axis
//! (`crate::inter::Component`, §8.5.3.2.10).

#![allow(
    clippy::many_single_char_names,
    reason = "H.265 sample and block variables keep their specification names (x, y, w, h, n)"
)]

use nc_common::{BitDepth, BitDepths, Picture, Plane, PlaneIndex};
use nc_dsp::{DepthKernels, Dsp, HevcWeights, Pixel};
use nc_encoder::Mv;

use crate::inter::Component;

use super::slice::Weight;

/// Precision of the prediction samples weighted prediction combines
/// (`predSamplesLX`, 14 bits at every depth): `shift1 = 14 − BitDepth`.
const PREDICTION_BITS: u32 = 14;

/// One list's contribution to a prediction block.
#[derive(Debug, Clone, Copy)]
pub(super) struct Source<'a, T: Pixel> {
    pub picture: &'a Picture<T>,
    pub mv: Mv,
    /// Explicit weights, `None` for default weighted prediction.
    pub weight: Option<Weight>,
}

/// Explicit weighting denominators (`luma_log2_weight_denom`,
/// `ChromaLog2WeightDenom`).
pub(super) type Denominators = [u32; 2];

/// Working memory reused across prediction blocks, so that no block pays
/// for clearing buffers it overwrites anyway.
#[derive(Debug, Default)]
pub(super) struct Buffers {
    preds: [Vec<i16>; 2],
}

/// Predict the `w`x`h` luma block at `(x, y)` and its chroma (`chroma`: how
/// vectors address the chroma planes, `None` for 4:0:0) into `dst` from one
/// or two sources.
#[allow(clippy::too_many_arguments)]
pub(super) fn predict<T: Pixel>(
    dsp: &Dsp,
    depths: BitDepths,
    chroma: Option<Component>,
    buffers: &mut Buffers,
    dst: &mut Picture<T>,
    (x, y): (usize, usize),
    (w, h): (usize, usize),
    sources: &[Source<'_, T>],
    denominators: Denominators,
) {
    let mut target = Target {
        dsp,
        depths,
        buffers,
        dst,
        sources,
        denominators,
    };
    target.plane(0, Component::Luma, (x, y), (w, h));
    if let Some(chroma) = chroma {
        let (sx, sy) = chroma.subsampling();
        for c in 1..3 {
            target.plane(c, chroma, (x >> sx, y >> sy), (w >> sx, h >> sy));
        }
    }
}

/// Where [`predict`] writes and what it predicts from.
struct Target<'p, 's, T: Pixel> {
    dsp: &'p Dsp,
    depths: BitDepths,
    buffers: &'p mut Buffers,
    dst: &'p mut Picture<T>,
    sources: &'p [Source<'s, T>],
    denominators: Denominators,
}

impl<T: Pixel> Target<'_, '_, T> {
    /// The `w`x`h` block at `(x, y)` of plane `c`, addressed as `component`.
    #[inline(always)]
    #[allow(
        clippy::inline_always,
        reason = "one copy per plane kind: luma's component is constant"
    )]
    fn plane(
        &mut self,
        c: usize,
        component: Component,
        (x, y): (usize, usize),
        (w, h): (usize, usize),
    ) {
        let sources = self.sources;
        let plane = PlaneIndex::ALL[c];
        let kernels = self.dsp.at(self.depths.plane(c));
        let explicit = sources.iter().any(|s| s.weight.is_some());
        if sources.len() == 1
            && !explicit
            && fast_uni(
                kernels,
                sources[0].picture.plane(plane),
                self.dst.plane_mut(plane),
                (x, y),
                (w, h),
                sources[0].mv,
                component,
            )
        {
            return;
        }
        if sources.len() == 2
            && !explicit
            && fast_bi(
                kernels,
                [
                    sources[0].picture.plane(plane),
                    sources[1].picture.plane(plane),
                ],
                [sources[0].mv, sources[1].mv],
                &mut self.buffers.preds[0],
                self.dst.plane_mut(plane),
                (x, y),
                (w, h),
                component,
            )
        {
            return;
        }
        for (pred, source) in self.buffers.preds.iter_mut().zip(sources) {
            crate::inter::predict_samples(
                kernels,
                source.picture.plane(plane),
                (x, y),
                (w, h),
                source.mv,
                component,
                pred,
            );
        }
        let [first, second] = &self.buffers.preds;
        let second = if sources.len() == 2 { second } else { first };
        let weights = combination(
            sources,
            c,
            self.denominators[usize::from(c > 0)],
            kernels.depth(),
        );
        let mut view = self
            .dst
            .plane_mut(plane)
            .region_mut(x as isize, y as isize, w, h)
            .expect("prediction block inside the picture");
        kernels.hevc_weighted_prediction_centered(
            weights,
            &first[..w * h],
            &second[..w * h],
            &mut view,
        );
    }
}

/// The weighted sample prediction of component `c` (§8.5.3.3.4) as one
/// `clamp((a·w0 + b·w1 + round) >> shift)` (the `hevc_weight` kernel's
/// form): default or explicit (equations 8-252, 8-253), one or two lists,
/// at `depth`.
fn combination<T: Pixel>(
    sources: &[Source<'_, T>],
    c: usize,
    denominator: u32,
    depth: BitDepth,
) -> HevcWeights {
    let bi = sources.len() == 2;
    let explicit = sources.iter().any(|s| s.weight.is_some());
    // `shift1 = 14 − bitDepth` (uni), `shift2 = 15 − bitDepth` (bi).
    let uni_shift = PREDICTION_BITS - depth.bits();
    if !explicit {
        let shift = uni_shift + u32::from(bi);
        return HevcWeights {
            w0: 1,
            w1: i16::from(bi),
            round: 1 << (shift - 1),
            shift,
        };
    }
    let log2wd = denominator + uni_shift;
    let [w0, w1] = [0, 1].map(|i| sources.get(i).and_then(|s| s.weight).unwrap_or_default());
    // Weights lie in −128..=255 (§7.4.7.3).
    let weight = |w: &Weight| w.weight[c] as i16;
    let offset = |w: &Weight| w.offset[c];
    if bi {
        HevcWeights {
            w0: weight(&w0),
            w1: weight(&w1),
            round: (offset(&w0) + offset(&w1) + 1) << log2wd,
            shift: log2wd + 1,
        }
    } else {
        // The offset added after the shift is a multiple of 2^log2WD before it.
        HevcWeights {
            w0: weight(&w0),
            w1: 0,
            round: (1 << (log2wd - 1)) + (offset(&w0) << log2wd),
            shift: log2wd,
        }
    }
}

/// Split a block dimension into one or two power-of-two kernel sides (`>= 4`),
/// covering square, rectangular, and AMP (`12, 24, 48`) partitions.
fn block_pieces(n: usize) -> Option<[usize; 2]> {
    match n {
        4 | 8 | 16 | 32 | 64 => Some([n, 0]),
        12 | 24 | 48 => Some([n * 2 / 3, n / 3]),
        _ => None,
    }
}

/// Default uni-prediction through the nc-dsp kernels, when both sides
/// split into power-of-two pieces of at least 4 samples (interpolation is
/// separable per sample, so pieces predict exactly as the whole block).
/// Returns whether it ran.
#[inline(always)]
#[allow(
    clippy::inline_always,
    reason = "inlined into each plane kind of `Target::plane`, as before it existed"
)]
fn fast_uni<T: Pixel>(
    kernels: DepthKernels<'_>,
    reference: &Plane<T>,
    dst: &mut Plane<T>,
    (x, y): (usize, usize),
    (w, h): (usize, usize),
    mv: Mv,
    component: Component,
) -> bool {
    let (Some(ws), Some(hs)) = (block_pieces(w), block_pieces(h)) else {
        return false;
    };
    if ws[1] == 0 && hs[1] == 0 {
        // One kernel-sized piece: the common case, without the piece loop.
        let mut view = dst
            .region_mut(x as isize, y as isize, w, h)
            .expect("prediction block inside the picture");
        crate::inter::predict_into(kernels, reference, (x, y), mv, component, &mut view);
        return true;
    }
    let mut oy = 0;
    for ph in hs.into_iter().filter(|&n| n > 0) {
        let mut ox = 0;
        for pw in ws.into_iter().filter(|&n| n > 0) {
            let (bx, by) = (x + ox, y + oy);
            let mut view = dst
                .region_mut(bx as isize, by as isize, pw, ph)
                .expect("prediction block inside the picture");
            crate::inter::predict_into(kernels, reference, (bx, by), mv, component, &mut view);
            ox += pw;
        }
        oy += ph;
    }
    true
}

/// Default bi-prediction through [`DepthKernels::predict_compound`] +
/// [`DepthKernels::hevc_predict_bi`] for power-of-two and AMP partitions.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
#[allow(
    clippy::inline_always,
    reason = "inlined into each plane kind of `Target::plane`, as before it existed"
)]
fn fast_bi<T: Pixel>(
    kernels: DepthKernels<'_>,
    refs: [&Plane<T>; 2],
    mvs: [Mv; 2],
    first: &mut Vec<i16>,
    dst: &mut Plane<T>,
    (x, y): (usize, usize),
    (w, h): (usize, usize),
    component: Component,
) -> bool {
    let (Some(ws), Some(hs)) = (block_pieces(w), block_pieces(h)) else {
        return false;
    };
    if ws[1] == 0 && hs[1] == 0 {
        crate::inter::predict_samples(kernels, refs[0], (x, y), (w, h), mvs[0], component, first);
        let mut view = dst
            .region_mut(x as isize, y as isize, w, h)
            .expect("prediction block inside the picture");
        crate::inter::predict_bi_into(
            kernels,
            refs[1],
            (x, y),
            mvs[1],
            component,
            &first[..w * h],
            &mut view,
        );
        return true;
    }
    let mut oy = 0;
    for ph in hs.into_iter().filter(|&n| n > 0) {
        let mut ox = 0;
        for pw in ws.into_iter().filter(|&n| n > 0) {
            let (bx, by) = (x + ox, y + oy);
            crate::inter::predict_samples(
                kernels,
                refs[0],
                (bx, by),
                (pw, ph),
                mvs[0],
                component,
                first,
            );
            let mut view = dst
                .region_mut(bx as isize, by as isize, pw, ph)
                .expect("prediction block inside the picture");
            crate::inter::predict_bi_into(
                kernels,
                refs[1],
                (bx, by),
                mvs[1],
                component,
                &first[..pw * ph],
                &mut view,
            );
            ox += pw;
        }
        oy += ph;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_common::Picture;

    /// Luma filter tap signs at the half-sample position (`fL[2]`).
    const HALF_TAP_POSITIVE: [bool; 8] = [false, true, false, true, true, false, true, false];

    /// Bi-prediction of a block whose top-left sample sees the largest
    /// 2-D half/half filter output 8-bit samples can give (above `i16::MAX`
    /// before the >> 6, about 33150 after it): reference samples 255 where
    /// the horizontal and vertical tap signs agree, 0 where they differ.
    /// Both lists predict it, so the weighted average saturates to 255; a
    /// 14-bit intermediate that wraps in 16 bits would give 0 instead.
    #[test]
    fn extreme_half_sample_bi_prediction_saturates() {
        let (x, y) = (16usize, 16usize);
        let mut reference = Picture::<u8>::new(64, 64, 32).unwrap();
        {
            let mut view = reference.plane_mut(PlaneIndex::Y).view_mut();
            for r in 0..64 {
                for (c, s) in view.row_mut(r).iter_mut().enumerate().take(64) {
                    // The window of output (0, 0) starts 3 samples up and
                    // left of the block, plus the half-sample vector's 0.
                    let (tr, tc) = (r as isize - (y as isize - 3), c as isize - (x as isize - 3));
                    *s = if (0..8).contains(&tr) && (0..8).contains(&tc) {
                        let agree =
                            HALF_TAP_POSITIVE[tr as usize] == HALF_TAP_POSITIVE[tc as usize];
                        if agree { 255 } else { 0 }
                    } else {
                        128
                    };
                }
            }
        }
        reference.extend_borders();
        let half = Mv { x: 2, y: 2 };
        let source = Source {
            picture: &reference,
            mv: half,
            weight: None,
        };
        let mut dst = Picture::new(64, 64, 32).unwrap();
        let dsp = Dsp::new();
        predict(
            &dsp,
            BitDepths::EIGHT,
            Some(Component::CHROMA_420),
            &mut Buffers::default(),
            &mut dst,
            (x, y),
            (8, 8),
            &[source, source],
            [0, 0],
        );
        assert_eq!(dst.plane(PlaneIndex::Y).view().row(y)[x], 255);
    }
}
