//! Unrounded HEVC/VVC interpolation for weighted and bidirectional prediction.
//! Uses the ordinary shared tap families; retains six fractional sample bits.
use super::{
    HevcChromaTaps, HevcLumaTaps, SeparableTaps, VvcAffineLumaTaps, VvcChromaTaps, VvcLumaTaps,
};
use crate::{Dsp, Filter};
use nc_common::PlaneView;
impl Dsp {
    /// Interpolate a rectangle into signed samples with six fractional bits.
    /// The source starts at the filter window's upper-left, as for `predict`.
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
        assert!(!alternate_half || filter == Filter::VvcLuma);
        let (w, h) = size;
        assert!((2..=64).contains(&w) && (2..=64).contains(&h));
        assert!(
            window.width() >= w + filter.taps() - 1 && window.height() >= h + filter.taps() - 1
        );
        assert!(phases.0 < filter.positions() && phases.1 < filter.positions());
        let out = &mut out[..w * h];
        match filter {
            Filter::HevcLuma => interpolate::<HevcLumaTaps, 8>(size, window, phases, out),
            Filter::HevcChroma => interpolate::<HevcChromaTaps, 4>(size, window, phases, out),
            Filter::VvcLuma if alternate_half => {
                interpolate::<HalfLumaTaps, 8>(size, window, phases, out);
            }
            Filter::VvcLuma => interpolate::<VvcLumaTaps, 8>(size, window, phases, out),
            Filter::VvcAffineLuma => interpolate::<VvcAffineLumaTaps, 8>(size, window, phases, out),
            Filter::VvcChroma => interpolate::<VvcChromaTaps, 4>(size, window, phases, out),
            _ => panic!("filter has no six-bit prediction precision"),
        }
    }
}
fn interpolate<T: SeparableTaps<N>, const N: usize>(
    (w, h): (usize, usize),
    source: &PlaneView<'_>,
    (fx, fy): (usize, usize),
    out: &mut [i16],
) {
    let before = N / 2 - 1;
    let cx = T::TABLE[fx];
    let cy = T::TABLE[fy];
    let rows = if fy == 0 { h } else { h + N - 1 };
    let mut horizontal = vec![0i32; rows * w];
    for (y, row) in horizontal.chunks_exact_mut(w).enumerate() {
        let input = source.row(y + if fy == 0 { before } else { 0 });
        for (x, out) in row.iter_mut().enumerate() {
            *out = if fx == 0 {
                i32::from(input[x + before]) << 6
            } else {
                cx.iter()
                    .enumerate()
                    .map(|(k, &c)| i32::from(c) * i32::from(input[x + k]))
                    .sum()
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
                    >> 6
            };
            *out = value as i16;
        }
    }
}

/// H.266 alternate half-sample luma filter selected by AMVR half precision.
struct HalfLumaTaps;
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
