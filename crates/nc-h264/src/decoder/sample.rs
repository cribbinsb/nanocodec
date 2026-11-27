//! The decoder's sample types (`docs/handoff/high-bit-depth.md`): the
//! decoding process is written once over `T: H264Sample` and compiled for
//! `u8` (8-bit streams, with the 8-bit nc-dsp kernels it always had) and
//! `u16` (High 10, through [`nc_dsp::Dsp::at`]).
//!
//! What differs between the two beyond [`nc_dsp::Pixel`]: the type of the
//! scaled transform coefficients (16 bits suffice at 8 bits, §8.5.12.1
//! bounds them by `2^(7 + BitDepth)`, so 10-bit ones need 32), and the
//! kernels whose 8-bit forms the decoder calls directly for speed
//! (interpolation handles, the macroblock residual).

use nc_common::{BitDepth, PlaneView, PlaneViewMut};
use nc_dsp::{BlockSize, Dsp, Filter, H264Predictor, Pixel};

use super::Result;
use super::intra::{self, Avail};

/// A scaled transform coefficient.
pub(super) trait Coefficient:
    Copy + Default + PartialEq + core::fmt::Debug + Into<i32>
{
    const ZERO: Self;
    /// `value` clamped to the type: a conforming stream stays inside it;
    /// hostile input is clamped rather than wrapped.
    fn clamped(value: i64) -> Self;
}

impl Coefficient for i16 {
    const ZERO: Self = 0;
    fn clamped(value: i64) -> Self {
        value.clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16
    }
}

impl Coefficient for i32 {
    const ZERO: Self = 0;
    fn clamped(value: i64) -> Self {
        value.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
    }
}

/// One interpolation: the 8-bit kernel handle of the block shape, and the
/// filter and size the other depths are predicted with.
#[derive(Clone, Copy)]
pub(super) struct McKernel<'k> {
    pub predictor: &'k H264Predictor,
    pub filter: Filter,
    pub size: BlockSize,
}

/// A sample type of the decoder.
pub(super) trait H264Sample: Pixel {
    type Coeff: Coefficient;

    /// Predict `kernel`'s block at sub-sample offsets `frac` from `window`
    /// (its first tap's sample, `stride`) into `out` (`out_stride`).
    #[allow(clippy::too_many_arguments, reason = "the kernel call's views")]
    fn interpolate(
        dsp: &Dsp,
        depth: BitDepth,
        kernel: &McKernel<'_>,
        window: &[Self],
        stride: usize,
        frac: (usize, usize),
        out: &mut [Self],
        out_stride: usize,
    );

    /// Add the scaled 4x4 luma blocks `coded` (bit per raster block, `ac`
    /// those with AC coefficients) of a macroblock to its 16x16 samples.
    fn add_4x4_mb(
        dsp: &Dsp,
        depth: BitDepth,
        coeffs: &[[Self::Coeff; 16]; 16],
        masks: (u16, u16),
        dst: &mut PlaneViewMut<'_, Self>,
    );

    /// Add one scaled 4x4 block.
    fn add_4x4(
        dsp: &Dsp,
        depth: BitDepth,
        coeffs: &[Self::Coeff; 16],
        dst: &mut PlaneViewMut<'_, Self>,
    );

    /// Add one scaled 8x8 block.
    fn add_8x8(
        dsp: &Dsp,
        depth: BitDepth,
        coeffs: &[Self::Coeff; 64],
        dst: &mut PlaneViewMut<'_, Self>,
    );

    /// `Intra_4x4` prediction of the block at `at` (§8.3.1).
    fn intra_4x4(
        _dsp: &Dsp,
        plane: &mut PlaneViewMut<'_, Self>,
        at: (usize, usize),
        mode: u8,
        avail: (Avail, BitDepth),
    ) -> Result<()> {
        intra::predict_nxn(plane, at, 4, mode, avail)
    }

    /// `Intra_8x8` prediction of the block at `at` (§8.3.2).
    fn intra_8x8(
        dsp: &Dsp,
        plane: &mut PlaneViewMut<'_, Self>,
        at: (usize, usize),
        mode: u8,
        avail: (Avail, BitDepth),
    ) -> Result<()>;
}

impl H264Sample for u8 {
    type Coeff = i16;

    #[inline]
    fn interpolate(
        _dsp: &Dsp,
        _depth: BitDepth,
        kernel: &McKernel<'_>,
        window: &[u8],
        stride: usize,
        (fx, fy): (usize, usize),
        out: &mut [u8],
        out_stride: usize,
    ) {
        kernel
            .predictor
            .predict(window, stride, fx, fy, out, out_stride);
    }

    #[inline]
    fn add_4x4_mb(
        dsp: &Dsp,
        _depth: BitDepth,
        coeffs: &[[i16; 16]; 16],
        (coded, ac): (u16, u16),
        dst: &mut PlaneViewMut<'_, u8>,
    ) {
        dsp.h264_add_scaled_mb(coeffs, coded, ac, dst);
    }

    #[inline]
    fn add_4x4(dsp: &Dsp, _depth: BitDepth, coeffs: &[i16; 16], dst: &mut PlaneViewMut<'_, u8>) {
        dsp.h264_add_scaled_4x4(core::slice::from_ref(coeffs), 1, dst);
    }

    #[inline]
    fn add_8x8(dsp: &Dsp, _depth: BitDepth, coeffs: &[i16; 64], dst: &mut PlaneViewMut<'_, u8>) {
        dsp.h264_add_scaled_8x8(coeffs, dst);
    }

    #[inline]
    fn intra_4x4(
        dsp: &Dsp,
        plane: &mut PlaneViewMut<'_, Self>,
        at: (usize, usize),
        mode: u8,
        (avail, _depth): (Avail, BitDepth),
    ) -> Result<()> {
        intra::predict_4x4(dsp, plane, at, mode, avail)
    }

    #[inline]
    fn intra_8x8(
        dsp: &Dsp,
        plane: &mut PlaneViewMut<'_, u8>,
        (x, y): (usize, usize),
        mode: u8,
        (avail, _depth): (Avail, BitDepth),
    ) -> Result<()> {
        intra::predict_8x8(dsp, plane, x, y, mode, avail)
    }
}

/// Side of a luma 4x4 block and of a macroblock.
const SIDE_4: usize = 4;
const BLOCKS_WIDE: usize = 4;

impl H264Sample for u16 {
    type Coeff = i32;

    fn interpolate(
        dsp: &Dsp,
        depth: BitDepth,
        kernel: &McKernel<'_>,
        window: &[u16],
        stride: usize,
        frac: (usize, usize),
        out: &mut [u16],
        out_stride: usize,
    ) {
        let (w, h) = (kernel.size.width(), kernel.size.height());
        let extra = kernel.filter.taps() - 1;
        let window = PlaneView::new(window, stride, w + extra, h + extra)
            .expect("interpolation window inside the reference");
        let mut dst =
            PlaneViewMut::new(out, out_stride, w, h).expect("prediction block inside its buffer");
        dsp.at(depth)
            .predict(kernel.filter, kernel.size, &window, frac, &mut dst);
    }

    fn add_4x4_mb(
        dsp: &Dsp,
        depth: BitDepth,
        coeffs: &[[i32; 16]; 16],
        (coded, _ac): (u16, u16),
        dst: &mut PlaneViewMut<'_, u16>,
    ) {
        for (b, block) in coeffs.iter().enumerate() {
            if (coded >> b) & 1 == 0 {
                continue;
            }
            let (x, y) = ((b % BLOCKS_WIDE) * SIDE_4, (b / BLOCKS_WIDE) * SIDE_4);
            let mut view = dst
                .sub_view_mut(x, y, SIDE_4, SIDE_4)
                .expect("4x4 block inside the macroblock");
            dsp.at(depth).h264_inverse_add_4x4(block, &mut view);
        }
    }

    fn add_4x4(dsp: &Dsp, depth: BitDepth, coeffs: &[i32; 16], dst: &mut PlaneViewMut<'_, u16>) {
        dsp.at(depth).h264_inverse_add_4x4(coeffs, dst);
    }

    fn add_8x8(dsp: &Dsp, depth: BitDepth, coeffs: &[i32; 64], dst: &mut PlaneViewMut<'_, u16>) {
        dsp.at(depth).h264_inverse_add_8x8(coeffs, dst);
    }

    /// The decoder's own transcription of §8.3.2.2 (nc-dsp's `Intra_8x8`
    /// kernels are 8-bit only).
    fn intra_8x8(
        _dsp: &Dsp,
        plane: &mut PlaneViewMut<'_, u16>,
        at: (usize, usize),
        mode: u8,
        avail: (Avail, BitDepth),
    ) -> Result<()> {
        intra::predict_nxn(plane, at, 8, mode, avail)
    }
}
