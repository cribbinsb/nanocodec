//! The VP9 kernels at any bit depth: the routing behind [`DepthKernels`]'
//! `vp9_*` methods (`docs/handoff/high-bit-depth.md`).
//!
//! For `u8` each method is the 8-bit [`crate::Dsp`] method, with its SIMD
//! tiers; for `u16` it is the family's scalar reference instantiated at
//! `u16` with the depth (`vp9_intra::predict_at`,
//! `vp9_mc::{vp9_convolve_into_at, vp9_convolve_scaled_at}`,
//! `vp9_txfm::vp9_inverse_add_high`, `vp9_deblock::vp9_filter_edge_at`).
//! libvpx's high-bit-depth VP9 (`vpx_highbd_*`) is the 8-bit arithmetic
//! with clips, thresholds and the no-edge intra values at the depth, and
//! 32-bit transform stages instead of 16-bit ones.
//!
//! None of AV1's deeper kernels fits VP9's arithmetic: AV1 rounds its
//! two-dimensional prediction at intermediate precision where VP9 clips
//! each pass to samples, its transforms use 12-bit cosines with other stage
//! ranges, and its 13-tap deblocking differs from VP9's 15-tap one.

use super::DepthKernels;
use crate::high_depth::av1::Av1Coefficient;
use crate::vp9_mc::{VP9_TAPS, VP9_TAPS_BEFORE};
use crate::{
    Vp9Filter, Vp9FilterLimits, Vp9FilterWidth, Vp9IntraEdges, Vp9IntraMode, Vp9Pass,
    Vp9ScaledScratch,
};
use nc_common::Sample;

/// A VP9 inverse transform's passes (columns, rows).
pub type Vp9Passes = (Vp9Pass, Vp9Pass);

/// The VP9 kernels of a sample type (see the module docs); generic decoder
/// code calls them through [`DepthKernels`].
#[allow(clippy::too_many_arguments)]
pub trait Vp9Pixel: Sample {
    /// Dequantised coefficients of this depth: §8.6.2 bounds them to
    /// `8 + BitDepth` bits, so `i16` at 8 bits and `i32` above (libvpx's
    /// high-bit-depth `tran_low_t`).
    type Vp9Coefficient: Av1Coefficient;

    #[doc(hidden)]
    fn vp9_intra_predict(
        kernels: DepthKernels<'_>,
        mode: Vp9IntraMode,
        side: usize,
        edges: &Vp9IntraEdges<Self>,
        out: &mut [Self],
    );
    #[doc(hidden)]
    fn vp9_convolve(
        kernels: DepthKernels<'_>,
        kind: Vp9Filter,
        window: (&[Self], usize),
        phase: (usize, usize),
        size: (usize, usize),
        dst: (&mut [Self], usize),
        average: bool,
    );
    #[doc(hidden)]
    fn vp9_convolve_scaled(
        kernels: DepthKernels<'_>,
        kind: Vp9Filter,
        window: (&[Self], usize),
        steps: ((usize, usize), (usize, usize)),
        size: (usize, usize),
        dst: (&mut [Self], usize),
        average: bool,
        scratch: &mut Vp9ScaledScratch,
    );
    #[doc(hidden)]
    fn vp9_inverse_add(
        kernels: DepthKernels<'_>,
        log2: u8,
        passes: Vp9Passes,
        coefficients: &[Self::Vp9Coefficient],
        nonzero: (usize, usize),
        dst: (&mut [Self], usize),
    );
    #[doc(hidden)]
    fn vp9_filter_edge(
        kernels: DepthKernels<'_>,
        data: &mut [Self],
        at: (usize, usize, usize),
        lines: usize,
        width: Vp9FilterWidth,
        limits: Vp9FilterLimits,
    );
}

impl Vp9Pixel for u8 {
    type Vp9Coefficient = i16;

    #[inline]
    fn vp9_intra_predict(
        kernels: DepthKernels<'_>,
        mode: Vp9IntraMode,
        side: usize,
        edges: &Vp9IntraEdges,
        out: &mut [u8],
    ) {
        kernels.dsp.vp9_intra_predict(mode, side, edges, out);
    }

    #[inline]
    fn vp9_convolve(
        kernels: DepthKernels<'_>,
        kind: Vp9Filter,
        (window, stride): (&[u8], usize),
        phase: (usize, usize),
        size: (usize, usize),
        (dst, dst_stride): (&mut [u8], usize),
        average: bool,
    ) {
        if average {
            kernels
                .dsp
                .vp9_convolve_avg_into(kind, window, stride, phase, size, dst, dst_stride);
        } else {
            kernels
                .dsp
                .vp9_convolve_into(kind, window, stride, phase, size, dst, dst_stride);
        }
    }

    #[inline]
    fn vp9_convolve_scaled(
        kernels: DepthKernels<'_>,
        kind: Vp9Filter,
        window: (&[u8], usize),
        (x, y): ((usize, usize), (usize, usize)),
        size: (usize, usize),
        dst: (&mut [u8], usize),
        average: bool,
        scratch: &mut Vp9ScaledScratch,
    ) {
        kernels
            .dsp
            .vp9_convolve_scaled_into(kind, window, x, y, size, dst, average, scratch);
    }

    #[inline]
    fn vp9_inverse_add(
        kernels: DepthKernels<'_>,
        log2: u8,
        passes: Vp9Passes,
        coefficients: &[i16],
        nonzero: (usize, usize),
        dst: (&mut [u8], usize),
    ) {
        kernels
            .dsp
            .vp9_inverse_add(log2, passes, coefficients, nonzero, dst);
    }

    #[inline]
    fn vp9_filter_edge(
        kernels: DepthKernels<'_>,
        data: &mut [u8],
        (edge, step, along): (usize, usize, usize),
        lines: usize,
        width: Vp9FilterWidth,
        limits: Vp9FilterLimits,
    ) {
        kernels
            .dsp
            .vp9_filter_edge(data, edge, step, along, lines, width, limits);
    }
}

impl Vp9Pixel for u16 {
    type Vp9Coefficient = i32;

    fn vp9_intra_predict(
        kernels: DepthKernels<'_>,
        mode: Vp9IntraMode,
        side: usize,
        edges: &Vp9IntraEdges<u16>,
        out: &mut [u16],
    ) {
        assert!(matches!(side, 4 | 8 | 16 | 32), "VP9 transform side {side}");
        crate::vp9_intra::predict_at(mode, side, edges, out, kernels.depth);
    }

    fn vp9_convolve(
        kernels: DepthKernels<'_>,
        kind: Vp9Filter,
        window: (&[u16], usize),
        phase: (usize, usize),
        (width, height): (usize, usize),
        dst: (&mut [u16], usize),
        average: bool,
    ) {
        if width == 0 || height == 0 {
            return;
        }
        crate::mc::generic::check_plane(
            window.0,
            window.1,
            width + VP9_TAPS - 1,
            height + VP9_TAPS - 1,
        );
        crate::mc::generic::check_plane(dst.0, dst.1, width, height);
        crate::vp9_mc::vp9_convolve_into_at(
            kind,
            window,
            phase,
            (width, height),
            dst,
            average,
            kernels.depth,
        );
    }

    fn vp9_convolve_scaled(
        kernels: DepthKernels<'_>,
        kind: Vp9Filter,
        (window, stride): (&[u16], usize),
        steps: ((usize, usize), (usize, usize)),
        size: (usize, usize),
        dst: (&mut [u16], usize),
        average: bool,
        _scratch: &mut Vp9ScaledScratch,
    ) {
        // The 8-bit driver's window starts `VP9_TAPS_BEFORE` rows above the
        // first row the scalar reference reads.
        crate::vp9_mc::vp9_convolve_scaled_at(
            kind,
            (&window[VP9_TAPS_BEFORE * stride..], stride),
            steps,
            size,
            dst,
            average,
            kernels.depth,
        );
    }

    fn vp9_inverse_add(
        kernels: DepthKernels<'_>,
        log2: u8,
        passes: Vp9Passes,
        coefficients: &[i32],
        _nonzero: (usize, usize),
        (dst, dst_stride): (&mut [u16], usize),
    ) {
        assert!((2..=5).contains(&log2), "VP9 transform size outside 4..=32");
        assert!(
            log2 != 5 || passes == (Vp9Pass::Dct, Vp9Pass::Dct),
            "VP9 32-point ADST is unsupported"
        );
        let side = 1_usize << log2;
        assert!(dst_stride >= side && dst.len() > (side - 1) * dst_stride + side - 1);
        crate::vp9_txfm::vp9_inverse_add_high(
            log2,
            passes,
            &coefficients[..side * side],
            (dst, dst_stride),
            kernels.depth,
        );
    }

    fn vp9_filter_edge(
        kernels: DepthKernels<'_>,
        data: &mut [u16],
        at: (usize, usize, usize),
        lines: usize,
        width: Vp9FilterWidth,
        limits: Vp9FilterLimits,
    ) {
        crate::vp9_deblock::vp9_filter_edge_at(data, at, lines, width, limits, kernels.depth);
    }
}

#[allow(clippy::too_many_arguments)]
impl DepthKernels<'_> {
    /// [`crate::Dsp::vp9_intra_predict`] at this depth (the edges' fixed
    /// values already at the depth: `mid ∓ 1`).
    #[inline]
    pub fn vp9_intra_predict<T: Vp9Pixel>(
        self,
        mode: Vp9IntraMode,
        side: usize,
        edges: &Vp9IntraEdges<T>,
        out: &mut [T],
    ) {
        T::vp9_intra_predict(self, mode, side, edges, out);
    }

    /// [`crate::Dsp::vp9_convolve_into`] (or, `average`,
    /// [`crate::Dsp::vp9_convolve_avg_into`]) at this depth: each pass
    /// clipped to the depth.
    #[inline]
    pub fn vp9_convolve<T: Vp9Pixel>(
        self,
        kind: Vp9Filter,
        window: (&[T], usize),
        phase: (usize, usize),
        size: (usize, usize),
        dst: (&mut [T], usize),
        average: bool,
    ) {
        T::vp9_convolve(self, kind, window, phase, size, dst, average);
    }

    /// [`crate::Dsp::vp9_convolve_scaled_into`] at this depth, with the same
    /// window contract; `steps` are `((x0, x_step), (y0, y_step))`.
    #[inline]
    pub fn vp9_convolve_scaled<T: Vp9Pixel>(
        self,
        kind: Vp9Filter,
        window: (&[T], usize),
        steps: ((usize, usize), (usize, usize)),
        size: (usize, usize),
        dst: (&mut [T], usize),
        average: bool,
        scratch: &mut Vp9ScaledScratch,
    ) {
        T::vp9_convolve_scaled(self, kind, window, steps, size, dst, average, scratch);
    }

    /// [`crate::Dsp::vp9_inverse_add`] at this depth: coefficients of
    /// [`Vp9Pixel::Vp9Coefficient`], the result clipped to the depth.
    #[inline]
    pub fn vp9_inverse_add<T: Vp9Pixel>(
        self,
        log2: u8,
        passes: Vp9Passes,
        coefficients: &[T::Vp9Coefficient],
        nonzero: (usize, usize),
        dst: (&mut [T], usize),
    ) {
        T::vp9_inverse_add(self, log2, passes, coefficients, nonzero, dst);
    }

    /// [`crate::Dsp::vp9_filter_edge`] at this depth (`at` is `(edge, step,
    /// along)`; `limits` as at 8 bits, scaled by the kernel).
    #[inline]
    pub fn vp9_filter_edge<T: Vp9Pixel>(
        self,
        data: &mut [T],
        at: (usize, usize, usize),
        lines: usize,
        width: Vp9FilterWidth,
        limits: Vp9FilterLimits,
    ) {
        T::vp9_filter_edge(self, data, at, lines, width, limits);
    }
}
