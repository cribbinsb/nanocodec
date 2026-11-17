//! Unrounded HEVC/VVC interpolation for weighted and bidirectional prediction.
//! Uses the ordinary shared tap families; retains six fractional sample bits.
use super::scalar::{SECOND_PASS_SHIFT, interpolation_shifts};
use super::{
    HevcChromaTaps, HevcLumaTaps, SeparableTaps, VvcAffineLumaTaps, VvcChromaTaps, VvcLumaTaps,
};
use crate::{Dsp, Filter};
use nc_common::{BitDepth, PlaneView, Sample};

/// Smallest and largest block side of a high-precision prediction.
const SIDES: std::ops::RangeInclusive<usize> = 2..=64;

impl Dsp {
    /// Interpolate a rectangle into signed samples with six fractional bits.
    /// The source starts at the filter window's upper-left, as for `predict`
    /// ([`crate::DepthKernels::predict_high_precision`] for deeper samples).
    ///
    /// # Panics
    /// If the filter is not HEVC/VVC, dimensions are outside 2..=64, phases
    /// are invalid, the alternate half filter is selected outside VVC luma,
    /// or either buffer is too small.
    pub fn predict_high_precision(
        &self,
        filter: Filter,
        size: (usize, usize),
        window: &PlaneView<'_>,
        phases: (usize, usize),
        alternate_half: bool,
        out: &mut [i16],
    ) {
        check_high_precision(filter, size, window, phases, alternate_half);
        high_precision_at(
            filter,
            size,
            window,
            phases,
            alternate_half,
            out,
            BitDepth::EIGHT,
        );
    }
}

/// The checks of [`Dsp::predict_high_precision`], for samples of any type.
pub(crate) fn check_high_precision<S: Sample>(
    filter: Filter,
    (w, h): (usize, usize),
    window: &PlaneView<'_, S>,
    phases: (usize, usize),
    alternate_half: bool,
) {
    assert!(!alternate_half || filter == Filter::VvcLuma);
    assert!(SIDES.contains(&w) && SIDES.contains(&h));
    assert!(window.width() >= w + filter.taps() - 1 && window.height() >= h + filter.taps() - 1);
    assert!(phases.0 < filter.positions() && phases.1 < filter.positions());
}

/// [`Dsp::predict_high_precision`] at `depth`, after
/// [`check_high_precision`]: `predSamplesLX` of H.266 §8.5.6.3.2 (HEVC
/// §8.5.3.3.3.1), 14-bit at every depth up to 12.
///
/// # Panics
/// If `out` is shorter than the block, or the filter is not HEVC/VVC.
pub(crate) fn high_precision_at<S: Sample>(
    filter: Filter,
    size: (usize, usize),
    window: &PlaneView<'_, S>,
    phases: (usize, usize),
    alternate_half: bool,
    out: &mut [i16],
    depth: BitDepth,
) {
    // Folds to the constant 8 in the 8-bit instance.
    let depth = depth.of::<S>();
    let out = &mut out[..size.0 * size.1];
    // Direct calls, not a chosen function pointer: each instance then
    // inlines into this one, as the 8-bit kernel did before it was generic
    // (the pointer kept six out-of-line copies: up to +1 % 8-bit VVC decode
    // instructions, callgrind on the foreign corpus).
    match filter {
        Filter::HevcLuma => interpolate::<S, HevcLumaTaps, 8>(size, window, phases, out, depth),
        Filter::HevcChroma => interpolate::<S, HevcChromaTaps, 4>(size, window, phases, out, depth),
        Filter::VvcLuma if alternate_half => {
            interpolate::<S, HalfLumaTaps, 8>(size, window, phases, out, depth);
        }
        Filter::VvcLuma => interpolate::<S, VvcLumaTaps, 8>(size, window, phases, out, depth),
        Filter::VvcAffineLuma => {
            interpolate::<S, VvcAffineLumaTaps, 8>(size, window, phases, out, depth);
        }
        Filter::VvcChroma => interpolate::<S, VvcChromaTaps, 4>(size, window, phases, out, depth),
        _ => panic!("filter has no six-bit prediction precision"),
    }
}

/// One block: whole samples `<< shift3`, a first pass `>> shift1`, a second
/// pass `>> shift2` (H.266 §8.5.6.3.2). A vertical-only position runs as a
/// whole-sample first pass and a second pass: `(Σ c·(s << shift3)) >> 6`
/// equals `(Σ c·s) >> shift1` because `shift1 + shift3 = 6` at every depth.
fn interpolate<S: Sample, T: SeparableTaps<N>, const N: usize>(
    (w, h): (usize, usize),
    source: &PlaneView<'_, S>,
    (fx, fy): (usize, usize),
    out: &mut [i16],
    depth: BitDepth,
) {
    let (shift1, shift3) = interpolation_shifts(depth.of::<S>());
    let before = N / 2 - 1;
    let cx = T::TABLE[fx];
    let cy = T::TABLE[fy];
    let rows = if fy == 0 { h } else { h + N - 1 };
    let mut small = [0i32; 23 * 16];
    let mut medium;
    let mut heap;
    let horizontal: &mut [i32] = if rows * w <= small.len() {
        &mut small[..rows * w]
    } else if rows * w <= 4096 {
        medium = [0i32; 4096];
        &mut medium[..rows * w]
    } else {
        heap = vec![0i32; rows * w];
        &mut heap
    };
    for (y, row) in horizontal.chunks_exact_mut(w).enumerate() {
        let input = source.row(y + if fy == 0 { before } else { 0 });
        for (x, out) in row.iter_mut().enumerate() {
            *out = if fx == 0 {
                input[x + before].to_i32() << shift3
            } else {
                cx.iter()
                    .enumerate()
                    .map(|(k, &c)| i32::from(c) * input[x + k].to_i32())
                    .sum::<i32>()
                    >> shift1
            };
        }
    }
    for (y, row) in out.chunks_exact_mut(w).enumerate() {
        for (x, out) in row.iter_mut().enumerate() {
            let value = if fy == 0 {
                horizontal[y * w + x]
            } else {
                cy.iter()
                    .enumerate()
                    .map(|(k, &c)| i32::from(c) * horizontal[(y + k) * w + x])
                    .sum::<i32>()
                    >> SECOND_PASS_SHIFT
            };
            *out = value as i16;
        }
    }
}

/// H.266 alternate half-sample luma filter selected by AMVR half precision.
pub(super) struct HalfLumaTaps;
impl SeparableTaps<8> for HalfLumaTaps {
    const TABLE: &'static [[i16; 8]] = &{
        let mut taps = super::VVC_LUMA;
        taps[8] = [0, 3, 9, 20, 20, 9, 3, 0];
        taps
    };
}

#[cfg(test)]
#[path = "precision_tests.rs"]
mod tests;
