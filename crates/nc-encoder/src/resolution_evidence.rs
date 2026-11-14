//! The evidence adaptive coded resolution (`resolution.rs`) switches on:
//! the luma error a viewer sees at the stream size, of the coded picture and
//! of the best the half size could show.
//!
//! A half-size picture is displayed upscaled by the player
//! (`nc_dsp::display`), so it can never be closer to the source than the
//! source itself halved (as the encoder halves it, `source_scale.rs`) and
//! upscaled again: that error is the half size's *floor*. Coded at the same
//! rate, the half size spends about four times the bits per sample, so its
//! coding error is smaller; it wins only where the full size's coding error
//! is large next to the floor (soft or blurred content loses little to the
//! halving; detailed content loses much). Both are measured on the stream's
//! own frames, so the decision follows the content and does not depend on
//! rate control's level, which any coding tool that spends a few more bits
//! per level moves (`docs/handoff/adaptive-resolution.md`, "Evidence").
//!
//! Luma only: the score's metrics (PSNR-YUV 6:1:1, VMAF, MS-SSIM) are
//! dominated by luma, and chroma halves the same way.

use nc_common::{Plane, PlaneView};
use nc_dsp::Dsp;
use nc_dsp::display::DisplayScaler;

use crate::source_scale::scale_source_plane;

/// Mean squared luma error per sample at the stream size, as displayed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Distortions {
    /// The source halved and shown upscaled: the best the half size can do.
    pub(crate) floor: f64,
    /// The coded picture as shown.
    pub(crate) coded: f64,
}

/// Scratch for measuring [`Distortions`] between a stream size and its half
/// size; allocated once per encoder.
#[derive(Debug)]
pub(crate) struct HalfSizeProbe {
    scaler: DisplayScaler,
    /// A half-size luma plane (the halved source of a full-size frame).
    half: Plane,
    /// A stream-size luma plane (a half-size plane as displayed).
    shown: Plane,
}

impl HalfSizeProbe {
    /// A probe for a stream of `stream` size whose half size is `half`.
    pub(crate) fn new(stream: (usize, usize), half: (usize, usize)) -> nc_common::Result<Self> {
        Ok(Self {
            scaler: DisplayScaler::default(),
            half: Plane::new(half.0, half.1, 0)?,
            shown: Plane::new(stream.0, stream.1, 0)?,
        })
    }

    /// A frame coded at the stream size: its `source` and reconstruction
    /// `recon` (stream-size luma views).
    pub(crate) fn full_size(
        &mut self,
        dsp: &Dsp,
        source: &PlaneView<'_>,
        recon: &PlaneView<'_>,
    ) -> Distortions {
        let Self {
            scaler,
            half,
            shown,
        } = self;
        scale_source_plane(dsp, source, &mut half.view_mut());
        let floor = shown_error(dsp, (scaler, shown), source, &half.view());
        Distortions {
            floor,
            coded: mean_squared_error(dsp, source, recon),
        }
    }

    /// A frame coded at the half size: the stream-size `source`, the halved
    /// source it was coded from (`coded_source`) and its reconstruction
    /// `recon` (half-size luma views).
    pub(crate) fn half_size(
        &mut self,
        dsp: &Dsp,
        source: &PlaneView<'_>,
        coded_source: &PlaneView<'_>,
        recon: &PlaneView<'_>,
    ) -> Distortions {
        let target = (&mut self.scaler, &mut self.shown);
        let floor = shown_error(dsp, target, source, coded_source);
        let target = (&mut self.scaler, &mut self.shown);
        Distortions {
            floor,
            coded: shown_error(dsp, target, source, recon),
        }
    }
}

/// The error against `source` of the half-size `picture` as displayed
/// (scaled by `scaler` into `shown`).
fn shown_error(
    dsp: &Dsp,
    (scaler, shown): (&mut DisplayScaler, &mut Plane),
    source: &PlaneView<'_>,
    picture: &PlaneView<'_>,
) -> f64 {
    scaler.scale(picture, &mut shown.view_mut());
    mean_squared_error(dsp, source, &shown.view())
}

/// Mean squared error per sample between two equally sized planes.
fn mean_squared_error(dsp: &Dsp, a: &PlaneView<'_>, b: &PlaneView<'_>) -> f64 {
    let samples = (a.width() * a.height()).max(1);
    dsp.sse(a, b) as f64 / samples as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(width: usize, height: usize, value: impl Fn(usize, usize) -> u8) -> Plane {
        let mut p = Plane::new(width, height, 0).expect("valid size");
        let mut view = p.view_mut();
        for y in 0..height {
            for (x, s) in view.row_mut(y).iter_mut().enumerate() {
                *s = value(x, y);
            }
        }
        p
    }

    #[test]
    fn flat_content_has_no_floor_and_detail_has_one() {
        let dsp = Dsp::new();
        let mut probe = HalfSizeProbe::new((64, 32), (32, 16)).expect("valid sizes");
        let flat = plane(64, 32, |_, _| 120);
        let d = probe.full_size(&dsp, &flat.view(), &flat.view());
        assert_eq!(
            d,
            Distortions {
                floor: 0.0,
                coded: 0.0
            }
        );
        // A one-sample checkerboard vanishes when halved.
        let fine = plane(64, 32, |x, y| if (x + y) % 2 == 0 { 0 } else { 200 });
        let d = probe.full_size(&dsp, &fine.view(), &fine.view());
        assert!(d.floor > 9_000.0, "floor {}", d.floor);
        assert_eq!(d.coded, 0.0);
    }

    #[test]
    fn a_half_size_frame_coded_exactly_shows_its_floor() {
        let dsp = Dsp::new();
        let mut probe = HalfSizeProbe::new((64, 32), (32, 16)).expect("valid sizes");
        let source = plane(64, 32, |x, y| ((x * 7 + y * 3) % 256) as u8);
        let mut halved = Plane::new(32, 16, 0).expect("valid size");
        scale_source_plane(&dsp, &source.view(), &mut halved.view_mut());
        let d = probe.half_size(&dsp, &source.view(), &halved.view(), &halved.view());
        assert_eq!(d.floor, d.coded);
        let full = probe.full_size(&dsp, &source.view(), &source.view());
        assert_eq!(full.floor, d.floor);
    }
}
