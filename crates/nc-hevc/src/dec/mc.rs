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

#![allow(
    clippy::many_single_char_names,
    reason = "H.265 sample and block variables keep their specification names (x, y, w, h, n)"
)]

use nc_common::{Picture, Plane, PlaneIndex};
use nc_dsp::{BlockSize, Dsp, HevcWeights};
use nc_encoder::Mv;

use super::slice::Weight;

/// `shift1 = 14 − bitDepth` of weighted prediction, `shift2 = 15 − bitDepth`.
const UNI_SHIFT: u32 = 6;
const BI_SHIFT: u32 = 7;

/// One list's contribution to a prediction block.
#[derive(Debug, Clone, Copy)]
pub(super) struct Source<'a> {
    pub picture: &'a Picture,
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

/// Predict the `w`x`h` luma block at `(x, y)` and its chroma into `dst`
/// from one or two sources.
pub(super) fn predict(
    dsp: &Dsp,
    buffers: &mut Buffers,
    dst: &mut Picture,
    (x, y): (usize, usize),
    (w, h): (usize, usize),
    sources: &[Source<'_>],
    denominators: Denominators,
) {
    let explicit = sources.iter().any(|s| s.weight.is_some());
    for (c, plane) in PlaneIndex::ALL.into_iter().enumerate() {
        let shift = usize::from(c > 0);
        let (px, py, pw, ph) = (x >> shift, y >> shift, w >> shift, h >> shift);
        if sources.len() == 1
            && !explicit
            && fast_uni(
                dsp,
                sources[0].picture.plane(plane),
                dst.plane_mut(plane),
                (px, py),
                (pw, ph),
                sources[0].mv,
                c > 0,
            )
        {
            continue;
        }
        if sources.len() == 2 && !explicit && BlockSize::new(pw, ph).is_some() {
            // Default bi-prediction of a kernel-sized block: the second
            // list's prediction goes straight into the average.
            let first = &mut buffers.preds[0];
            crate::inter::predict_samples(
                dsp,
                sources[0].picture.plane(plane),
                (px, py),
                (pw, ph),
                sources[0].mv,
                c > 0,
                first,
            );
            let mut view = dst
                .plane_mut(plane)
                .region_mut(px as isize, py as isize, pw, ph)
                .expect("prediction block inside the picture");
            crate::inter::predict_bi_into(
                dsp,
                sources[1].picture.plane(plane),
                (px, py),
                sources[1].mv,
                c > 0,
                &first[..pw * ph],
                &mut view,
            );
            continue;
        }
        for (pred, source) in buffers.preds.iter_mut().zip(sources) {
            crate::inter::predict_samples(
                dsp,
                source.picture.plane(plane),
                (px, py),
                (pw, ph),
                source.mv,
                c > 0,
                pred,
            );
        }
        let [first, second] = &buffers.preds;
        let second = if sources.len() == 2 { second } else { first };
        let weights = combination(sources, c, denominators[usize::from(c > 0)]);
        let mut view = dst
            .plane_mut(plane)
            .region_mut(px as isize, py as isize, pw, ph)
            .expect("prediction block inside the picture");
        dsp.hevc_weighted_prediction_centered(
            weights,
            &first[..pw * ph],
            &second[..pw * ph],
            &mut view,
        );
    }
}

/// The weighted sample prediction of component `c` (§8.5.3.3.4) as one
/// `clamp((a·w0 + b·w1 + round) >> shift)` (the `hevc_weight` kernel's
/// form): default or explicit (equations 8-252, 8-253), one or two lists.
fn combination(sources: &[Source<'_>], c: usize, denominator: u32) -> HevcWeights {
    let bi = sources.len() == 2;
    let explicit = sources.iter().any(|s| s.weight.is_some());
    if !explicit {
        return if bi {
            HevcWeights {
                w0: 1,
                w1: 1,
                round: 1 << (BI_SHIFT - 1),
                shift: BI_SHIFT,
            }
        } else {
            HevcWeights {
                w0: 1,
                w1: 0,
                round: 1 << (UNI_SHIFT - 1),
                shift: UNI_SHIFT,
            }
        };
    }
    let log2wd = denominator + UNI_SHIFT;
    let [w0, w1] = [0, 1].map(|i| sources.get(i).and_then(|s| s.weight).unwrap_or_default());
    // Weights lie in −128..=255 (§7.4.7.3).
    let weight = |w: &Weight| w.weight[c] as i16;
    if bi {
        HevcWeights {
            w0: weight(&w0),
            w1: weight(&w1),
            round: (w0.offset[c] + w1.offset[c] + 1) << log2wd,
            shift: log2wd + 1,
        }
    } else {
        // The offset added after the shift is a multiple of 2^log2WD before it.
        HevcWeights {
            w0: weight(&w0),
            w1: 0,
            round: (1 << (log2wd - 1)) + (w0.offset[c] << log2wd),
            shift: log2wd,
        }
    }
}

/// Default uni-prediction through the nc-dsp kernels, when both sides
/// split into power-of-two pieces of at least 4 samples (interpolation is
/// separable per sample, so pieces predict exactly as the whole block).
/// Returns whether it ran.
fn fast_uni(
    dsp: &Dsp,
    reference: &Plane,
    dst: &mut Plane,
    (x, y): (usize, usize),
    (w, h): (usize, usize),
    mv: Mv,
    chroma: bool,
) -> bool {
    let pieces = |n: usize| -> Option<[usize; 2]> {
        match n {
            4 | 8 | 16 | 32 | 64 => Some([n, 0]),
            12 | 24 | 48 => Some([n * 2 / 3, n / 3]),
            _ => None,
        }
    };
    let (Some(ws), Some(hs)) = (pieces(w), pieces(h)) else {
        return false;
    };
    if ws[1] == 0 && hs[1] == 0 {
        // One kernel-sized piece: the common case, without the piece loop.
        let mut view = dst
            .region_mut(x as isize, y as isize, w, h)
            .expect("prediction block inside the picture");
        crate::inter::predict_into(dsp, reference, (x, y), mv, chroma, &mut view);
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
            crate::inter::predict_into(dsp, reference, (bx, by), mv, chroma, &mut view);
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
        let mut reference = Picture::new(64, 64, 32).unwrap();
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
