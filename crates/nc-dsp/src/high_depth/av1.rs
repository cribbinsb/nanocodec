//! The AV1 kernels at any bit depth: the routing behind [`DepthKernels`]'
//! `av1_*` methods (`docs/handoff/high-bit-depth.md`).
//!
//! For `u8` each method is the 8-bit [`Dsp`] method, with its SIMD tiers;
//! for `u16` it is the family's scalar reference instantiated at `u16`
//! with the depth (the `*_at` functions: `av1_inverse::high`,
//! `intra_prediction::av1_block`, `av1_mc`, `deblock::av1`, `cdef::band`,
//! `av1_restoration`), until a vector tier pays for a hot path.
//!
//! The codec-side depth arithmetic (dequantiser tables, palette colours,
//! film grain parameters) stays in nc-av1; blends that have no 8-bit SIMD
//! kernel are called there directly at the depth (`av1_mc::blend::*_at`).

use core::fmt::Debug;
use core::ops::Range;

use nc_common::{Picture, PlaneView, PlaneViewMut, Sample};

use super::{Av1Taps, DepthKernels};
use crate::av1_film_grain::Av1FilmGrainParams;
use crate::av1_mc::blend::{Av1CompoundPred, Av1Mask, av1_blend_average_at, av1_blend_mask_at};
use crate::av1_mc::convolve::{
    Av1McAxis, KERNEL_EDGE, check_prediction, predict_compound_scalar, predict_scalar,
};
use crate::av1_mc::filter::{AV1_FILTER_TAPS, Av1Filters};
use crate::av1_mc::warp::{
    Av1Shear, av1_warp, av1_warp_at, av1_warp_compound, av1_warp_compound_at,
};
use crate::av1_restoration::{Av1LoopRestoration, Av1LrScratch, Av1StripeBoundaries};
use crate::cdef::band::{CdefLines, CdefSets, CdefUnit};
use crate::deblock::runs::{EdgeRun, Plane as RunPlane, av1_scalar_at};
use crate::intra_prediction::av1_block::{BlockEdges, IntraKind};
use crate::intra_prediction::filter::FilterIntraMode;
use crate::{Av1EdgeFilter, Av1TxSize, Av1TxType, BlockSize};

/// A dequantised AV1 coefficient as a decoder stores it: §7.12.3 clamps it
/// to `BitDepth + 8` bits, so `i16` holds every 8-bit one and `i32` every
/// deeper one.
pub trait Av1Coefficient: Copy + Default + Debug + Send + Sync + 'static {
    /// `value` stored (the caller has clamped it to the depth's range).
    fn from_i32(value: i32) -> Self;
    /// The coefficient as `i32`.
    fn to_i32(self) -> i32;
}

impl Av1Coefficient for i16 {
    #[inline]
    fn from_i32(value: i32) -> Self {
        value as i16
    }
    #[inline]
    fn to_i32(self) -> i32 {
        i32::from(self)
    }
}

impl Av1Coefficient for i32 {
    #[inline]
    fn from_i32(value: i32) -> Self {
        value
    }
    #[inline]
    fn to_i32(self) -> i32 {
        self
    }
}

/// The AV1 kernels of a sample type (see the module docs); generic decoder
/// code calls them through [`DepthKernels`].
#[allow(clippy::too_many_arguments)]
pub trait Av1Pixel: Sample {
    /// The coefficient storage of this depth.
    type Coefficient: Av1Coefficient;

    #[doc(hidden)]
    fn av1_inverse_add(
        kernels: DepthKernels<'_>,
        tx: Av1TxSize,
        kind: Av1TxType,
        coeffs: &mut [Self::Coefficient],
        eob: usize,
        dst: &mut PlaneViewMut<'_, Self>,
    );
    #[doc(hidden)]
    fn av1_inverse_wht_add(
        kernels: DepthKernels<'_>,
        coeffs: &[i32; 16],
        dst: &mut PlaneViewMut<'_, Self>,
    );
    #[doc(hidden)]
    fn av1_intra_predict_block(
        kernels: DepthKernels<'_>,
        edges: &BlockEdges<Self>,
        kind: IntraKind,
        dst: &mut PlaneViewMut<'_, Self>,
    );
    #[doc(hidden)]
    fn av1_intra_predict_filter_block(
        kernels: DepthKernels<'_>,
        edges: &BlockEdges<Self>,
        mode: FilterIntraMode,
        dst: &mut PlaneViewMut<'_, Self>,
    );
    #[doc(hidden)]
    fn av1_put_samples(
        kernels: DepthKernels<'_>,
        filters: Av1Filters,
        src: (&[Self], usize),
        size: (usize, usize),
        fracs: (usize, usize),
        dst: (&mut [Self], usize),
    );
    #[doc(hidden)]
    fn av1_prep_samples(
        kernels: DepthKernels<'_>,
        filters: Av1Filters,
        src: (&[Self], usize),
        size: (usize, usize),
        fracs: (usize, usize),
        dst: (&mut [i16], usize),
    );
    #[doc(hidden)]
    fn av1_predict(
        kernels: DepthKernels<'_>,
        filters: Av1Filters,
        window: &PlaneView<'_, Self>,
        size: (usize, usize),
        axes: (Av1McAxis, Av1McAxis),
        dst: &mut PlaneViewMut<'_, Self>,
    );
    #[doc(hidden)]
    fn av1_predict_compound(
        kernels: DepthKernels<'_>,
        filters: Av1Filters,
        window: &PlaneView<'_, Self>,
        size: (usize, usize),
        axes: (Av1McAxis, Av1McAxis),
        dst: (&mut [i16], usize),
    );
    #[doc(hidden)]
    fn av1_average_compound(
        kernels: DepthKernels<'_>,
        predictions: (&[i16], &[i16]),
        size: (usize, usize),
        dst: &mut PlaneViewMut<'_, Self>,
    );
    #[doc(hidden)]
    fn av1_blend_mask(
        kernels: DepthKernels<'_>,
        predictions: (Av1CompoundPred<'_>, Av1CompoundPred<'_>),
        mask: Av1Mask<'_>,
        size: (usize, usize),
        dst: &mut PlaneViewMut<'_, Self>,
    );
    #[doc(hidden)]
    fn av1_warp(
        kernels: DepthKernels<'_>,
        reference: &PlaneView<'_, Self>,
        model: (&[i32; 6], Av1Shear),
        position: (usize, usize),
        size: (usize, usize),
        sub: (bool, bool),
        dst: &mut PlaneViewMut<'_, Self>,
    );
    #[doc(hidden)]
    fn av1_warp_compound(
        kernels: DepthKernels<'_>,
        reference: &PlaneView<'_, Self>,
        model: (&[i32; 6], Av1Shear),
        position: (usize, usize),
        size: (usize, usize),
        sub: (bool, bool),
        dst: (&mut [i16], usize),
    );
    #[doc(hidden)]
    fn av1_loop_filter_runs(
        kernels: DepthKernels<'_>,
        dst: &mut PlaneViewMut<'_, Self>,
        vertical: bool,
        runs: &[EdgeRun<Av1EdgeFilter>],
    );
    #[doc(hidden)]
    fn cdef_filter_band(
        kernels: DepthKernels<'_>,
        lines: &mut CdefLines<Self>,
        picture: &mut Picture<Self>,
        rows: Range<usize>,
        units: &[CdefUnit],
        sets: &CdefSets,
    );
    #[doc(hidden)]
    fn av1_loop_restoration(
        kernels: DepthKernels<'_>,
        restoration: &Av1LoopRestoration<'_>,
        plane: &mut PlaneViewMut<'_, Self>,
        boundaries: (&Av1StripeBoundaries<Self>, bool),
        scratch: &mut Av1LrScratch<Self>,
    );
    #[doc(hidden)]
    fn av1_add_film_grain(
        kernels: DepthKernels<'_>,
        params: &Av1FilmGrainParams,
        input: [&PlaneView<'_, Self>; 3],
        size: (usize, usize),
        output: [&mut PlaneViewMut<'_, Self>; 3],
    );
}

/// An unscaled prediction the `u16` kernel table covers: its taps, the
/// width class of its tiles and their size (blocks of 128 tile into 64;
/// sides of 4 to 128 samples, powers of two).
struct Unscaled {
    taps: Av1Taps<'static>,
    class: usize,
    tile: (usize, usize),
}

impl Unscaled {
    /// The case of a `w`x`h` block with `filters` at `axes`, if both
    /// directions are unscaled and its sides are kernel sides.
    fn of(
        filters: Av1Filters,
        (w, h): (usize, usize),
        axes: (Av1McAxis, Av1McAxis),
    ) -> Option<Self> {
        let (phase_x, phase_y) = (axes.0.unscaled_phase()?, axes.1.unscaled_phase()?);
        let side = |n: usize| {
            (n.is_power_of_two() && (KERNEL_MIN_SIDE..=2 * KERNEL_EDGE).contains(&n))
                .then(|| n.min(KERNEL_EDGE))
        };
        let tile = (side(w)?, side(h)?);
        Some(Self {
            taps: (&filters.x.kernel(w)[phase_x], &filters.y.kernel(h)[phase_y]),
            class: (tile.0.trailing_zeros() - KERNEL_MIN_SIDE.trailing_zeros()) as usize,
            tile,
        })
    }

    /// The top-left corners of the tiles covering a `w`x`h` block.
    fn corners(&self, (w, h): (usize, usize)) -> impl Iterator<Item = (usize, usize)> + use<> {
        let (tw, th) = self.tile;
        (0..h)
            .step_by(th)
            .flat_map(move |top| (0..w).step_by(tw).map(move |left| (left, top)))
    }
}

/// The smallest kernel side.
const KERNEL_MIN_SIDE: usize = 4;
/// Side of a luma CDEF block (chroma: 4).
const CDEF_LUMA_SIZE: usize = 8;

impl Av1Pixel for u8 {
    type Coefficient = i16;

    fn av1_inverse_add(
        kernels: DepthKernels<'_>,
        tx: Av1TxSize,
        kind: Av1TxType,
        coeffs: &mut [i16],
        eob: usize,
        dst: &mut PlaneViewMut<'_, u8>,
    ) {
        kernels.dsp.av1_inverse_add_i16(tx, kind, coeffs, eob, dst);
    }

    fn av1_inverse_wht_add(
        kernels: DepthKernels<'_>,
        coeffs: &[i32; 16],
        dst: &mut PlaneViewMut<'_, u8>,
    ) {
        kernels.dsp.av1_inverse_wht_add(coeffs, dst);
    }

    fn av1_intra_predict_block(
        kernels: DepthKernels<'_>,
        edges: &BlockEdges<u8>,
        kind: IntraKind,
        dst: &mut PlaneViewMut<'_, u8>,
    ) {
        kernels.dsp.av1_intra_predict_block(edges, kind, dst);
    }

    fn av1_intra_predict_filter_block(
        kernels: DepthKernels<'_>,
        edges: &BlockEdges<u8>,
        mode: FilterIntraMode,
        dst: &mut PlaneViewMut<'_, u8>,
    ) {
        kernels.dsp.av1_intra_predict_filter_block(edges, mode, dst);
    }

    #[inline]
    fn av1_put_samples(
        kernels: DepthKernels<'_>,
        filters: Av1Filters,
        src: (&[u8], usize),
        size: (usize, usize),
        fracs: (usize, usize),
        dst: (&mut [u8], usize),
    ) {
        kernels.dsp.av1_put_samples(filters, src, size, fracs, dst);
    }

    #[inline]
    fn av1_prep_samples(
        kernels: DepthKernels<'_>,
        filters: Av1Filters,
        src: (&[u8], usize),
        size: (usize, usize),
        fracs: (usize, usize),
        dst: (&mut [i16], usize),
    ) {
        kernels.dsp.av1_prep_samples(filters, src, size, fracs, dst);
    }

    #[inline]
    fn av1_predict(
        kernels: DepthKernels<'_>,
        filters: Av1Filters,
        window: &PlaneView<'_, u8>,
        size: (usize, usize),
        axes: (Av1McAxis, Av1McAxis),
        dst: &mut PlaneViewMut<'_, u8>,
    ) {
        kernels.dsp.av1_predict(filters, window, size, axes, dst);
    }

    fn av1_predict_compound(
        kernels: DepthKernels<'_>,
        filters: Av1Filters,
        window: &PlaneView<'_, u8>,
        size: (usize, usize),
        axes: (Av1McAxis, Av1McAxis),
        (dst, stride): (&mut [i16], usize),
    ) {
        kernels
            .dsp
            .av1_predict_compound(filters, window, size, axes, dst, stride);
    }

    fn av1_average_compound(
        kernels: DepthKernels<'_>,
        (p0, p1): (&[i16], &[i16]),
        size: (usize, usize),
        dst: &mut PlaneViewMut<'_, u8>,
    ) {
        match BlockSize::new(size.0, size.1) {
            Some(kernel_size) => kernels
                .dsp
                .average_compound(kernel_size, (p0, p1), size.0, dst),
            None => crate::av1_mc::blend::av1_blend_average(
                Av1CompoundPred {
                    data: p0,
                    stride: size.0,
                },
                Av1CompoundPred {
                    data: p1,
                    stride: size.0,
                },
                size,
                dst,
            ),
        }
    }

    fn av1_blend_mask(
        kernels: DepthKernels<'_>,
        (p0, p1): (Av1CompoundPred<'_>, Av1CompoundPred<'_>),
        mask: Av1Mask<'_>,
        size: (usize, usize),
        dst: &mut PlaneViewMut<'_, u8>,
    ) {
        crate::av1_mc::blend::av1_blend_mask(kernels.dsp, p0, p1, mask, size, dst);
    }

    fn av1_warp(
        kernels: DepthKernels<'_>,
        reference: &PlaneView<'_, u8>,
        (model, shear): (&[i32; 6], Av1Shear),
        position: (usize, usize),
        size: (usize, usize),
        sub: (bool, bool),
        dst: &mut PlaneViewMut<'_, u8>,
    ) {
        av1_warp(
            kernels.dsp,
            reference,
            model,
            shear,
            position,
            size,
            sub,
            dst,
        );
    }

    fn av1_warp_compound(
        kernels: DepthKernels<'_>,
        reference: &PlaneView<'_, u8>,
        (model, shear): (&[i32; 6], Av1Shear),
        position: (usize, usize),
        size: (usize, usize),
        sub: (bool, bool),
        (dst, stride): (&mut [i16], usize),
    ) {
        av1_warp_compound(
            kernels.dsp,
            reference,
            model,
            shear,
            position,
            size,
            sub,
            dst,
            stride,
        );
    }

    fn av1_loop_filter_runs(
        kernels: DepthKernels<'_>,
        dst: &mut PlaneViewMut<'_, u8>,
        vertical: bool,
        runs: &[EdgeRun<Av1EdgeFilter>],
    ) {
        kernels.dsp.av1_loop_filter_runs(dst, vertical, runs);
    }

    fn cdef_filter_band(
        kernels: DepthKernels<'_>,
        lines: &mut CdefLines<u8>,
        picture: &mut Picture<u8>,
        rows: Range<usize>,
        units: &[CdefUnit],
        sets: &CdefSets,
    ) {
        kernels
            .dsp
            .cdef_filter_band(lines, picture, rows, units, sets);
    }

    fn av1_loop_restoration(
        kernels: DepthKernels<'_>,
        restoration: &Av1LoopRestoration<'_>,
        plane: &mut PlaneViewMut<'_, u8>,
        (boundaries, sub_y): (&Av1StripeBoundaries<u8>, bool),
        scratch: &mut Av1LrScratch<u8>,
    ) {
        restoration.apply(kernels.dsp, plane, boundaries, sub_y, scratch);
    }

    fn av1_add_film_grain(
        kernels: DepthKernels<'_>,
        params: &Av1FilmGrainParams,
        input: [&PlaneView<'_, u8>; 3],
        size: (usize, usize),
        output: [&mut PlaneViewMut<'_, u8>; 3],
    ) {
        kernels.dsp.av1_add_film_grain(params, input, size, output);
    }
}

impl Av1Pixel for u16 {
    type Coefficient = i32;

    fn av1_inverse_add(
        kernels: DepthKernels<'_>,
        tx: Av1TxSize,
        kind: Av1TxType,
        coeffs: &mut [i32],
        eob: usize,
        dst: &mut PlaneViewMut<'_, u16>,
    ) {
        assert!(
            dst.width() >= tx.width() && dst.height() >= tx.height(),
            "inverse destination smaller than {tx:?}"
        );
        if eob == 0 {
            return;
        }
        let stride = dst.stride();
        crate::av1_inverse::high::inverse_add_at(
            tx,
            kind,
            coeffs,
            dst.data_mut(),
            stride,
            kernels.depth,
        );
    }

    fn av1_inverse_wht_add(
        kernels: DepthKernels<'_>,
        coeffs: &[i32; 16],
        dst: &mut PlaneViewMut<'_, u16>,
    ) {
        assert!(
            dst.width() >= 4 && dst.height() >= 4,
            "WHT destination below 4x4"
        );
        crate::av1_inverse::wht::inverse_wht_add(coeffs, dst, kernels.depth);
    }

    fn av1_intra_predict_block(
        _kernels: DepthKernels<'_>,
        edges: &BlockEdges<u16>,
        kind: IntraKind,
        dst: &mut PlaneViewMut<'_, u16>,
    ) {
        edges.predict(kind, dst);
    }

    fn av1_intra_predict_filter_block(
        _kernels: DepthKernels<'_>,
        edges: &BlockEdges<u16>,
        mode: FilterIntraMode,
        dst: &mut PlaneViewMut<'_, u16>,
    ) {
        edges.predict_filter(mode, dst);
    }

    fn av1_put_samples(
        kernels: DepthKernels<'_>,
        filters: Av1Filters,
        (src, src_stride): (&[u16], usize),
        (w, h): (usize, usize),
        fracs: (usize, usize),
        (dst, dst_stride): (&mut [u16], usize),
    ) {
        let span = (w + AV1_FILTER_TAPS - 1, h + AV1_FILTER_TAPS - 1);
        let window =
            PlaneView::new(src, src_stride, span.0, span.1).expect("window covers its samples");
        let mut dst =
            PlaneViewMut::new(dst, dst_stride, w, h).expect("destination covers the block");
        let axes = (Av1McAxis::unscaled(fracs.0), Av1McAxis::unscaled(fracs.1));
        Self::av1_predict(kernels, filters, &window, (w, h), axes, &mut dst);
    }

    fn av1_prep_samples(
        kernels: DepthKernels<'_>,
        filters: Av1Filters,
        (src, src_stride): (&[u16], usize),
        (w, h): (usize, usize),
        fracs: (usize, usize),
        dst: (&mut [i16], usize),
    ) {
        let span = (w + AV1_FILTER_TAPS - 1, h + AV1_FILTER_TAPS - 1);
        let window =
            PlaneView::new(src, src_stride, span.0, span.1).expect("window covers its samples");
        let axes = (Av1McAxis::unscaled(fracs.0), Av1McAxis::unscaled(fracs.1));
        Self::av1_predict_compound(kernels, filters, &window, (w, h), axes, dst);
    }

    fn av1_predict(
        kernels: DepthKernels<'_>,
        filters: Av1Filters,
        window: &PlaneView<'_, u16>,
        size: (usize, usize),
        axes: (Av1McAxis, Av1McAxis),
        dst: &mut PlaneViewMut<'_, u16>,
    ) {
        let Some(case) = Unscaled::of(filters, size, axes) else {
            return predict_scalar(filters, window, size, axes, dst, kernels.depth);
        };
        check_prediction(window, size, axes);
        assert!(
            dst.width() >= size.0 && dst.height() >= size.1,
            "AV1 prediction destination too small"
        );
        let (ws, ds) = (window.stride(), dst.stride());
        let kernel = kernels.dsp.high.av1_put[case.class];
        for (left, top) in case.corners(size) {
            let src = &window.data()[top * ws + left..];
            let out = &mut dst.data_mut()[top * ds + left..];
            // SAFETY: the table matches a supported ISA; the window covers
            // every tile's taps and the destination every tile (checked
            // above; the kernel indexes its slices).
            unsafe { kernel((src, ws), (out, ds), case.tile.1, case.taps, kernels.depth) };
        }
    }

    fn av1_predict_compound(
        kernels: DepthKernels<'_>,
        filters: Av1Filters,
        window: &PlaneView<'_, u16>,
        (w, h): (usize, usize),
        axes: (Av1McAxis, Av1McAxis),
        (dst, stride): (&mut [i16], usize),
    ) {
        assert!(
            stride >= w && dst.len() >= (h - 1) * stride + w,
            "AV1 compound prediction buffer too small for {w}x{h}"
        );
        let Some(case) = Unscaled::of(filters, (w, h), axes) else {
            return predict_compound_scalar(
                filters,
                window,
                (w, h),
                axes,
                dst,
                stride,
                kernels.depth,
            );
        };
        check_prediction(window, (w, h), axes);
        let ws = window.stride();
        let kernel = kernels.dsp.high.av1_prep[case.class];
        for (left, top) in case.corners((w, h)) {
            let src = &window.data()[top * ws + left..];
            // SAFETY: as for `av1_predict`.
            unsafe {
                kernel(
                    (src, ws),
                    (&mut dst[top * stride + left..], stride),
                    case.tile.1,
                    case.taps,
                    kernels.depth,
                );
            }
        }
    }

    fn av1_average_compound(
        kernels: DepthKernels<'_>,
        (p0, p1): (&[i16], &[i16]),
        size: (usize, usize),
        dst: &mut PlaneViewMut<'_, u16>,
    ) {
        let prediction = |data| Av1CompoundPred {
            data,
            stride: size.0,
        };
        av1_blend_average_at(prediction(p0), prediction(p1), size, dst, kernels.depth);
    }

    fn av1_blend_mask(
        kernels: DepthKernels<'_>,
        (p0, p1): (Av1CompoundPred<'_>, Av1CompoundPred<'_>),
        mask: Av1Mask<'_>,
        size: (usize, usize),
        dst: &mut PlaneViewMut<'_, u16>,
    ) {
        av1_blend_mask_at(p0, p1, mask, size, dst, kernels.depth);
    }

    fn av1_warp(
        kernels: DepthKernels<'_>,
        reference: &PlaneView<'_, u16>,
        (model, shear): (&[i32; 6], Av1Shear),
        position: (usize, usize),
        size: (usize, usize),
        sub: (bool, bool),
        dst: &mut PlaneViewMut<'_, u16>,
    ) {
        av1_warp_at(
            reference,
            model,
            shear,
            position,
            size,
            sub,
            dst,
            kernels.depth,
        );
    }

    fn av1_warp_compound(
        kernels: DepthKernels<'_>,
        reference: &PlaneView<'_, u16>,
        (model, shear): (&[i32; 6], Av1Shear),
        position: (usize, usize),
        size: (usize, usize),
        sub: (bool, bool),
        dst: (&mut [i16], usize),
    ) {
        av1_warp_compound_at(
            reference,
            model,
            shear,
            position,
            size,
            sub,
            dst,
            kernels.depth,
        );
    }

    fn av1_loop_filter_runs(
        kernels: DepthKernels<'_>,
        dst: &mut PlaneViewMut<'_, u16>,
        vertical: bool,
        runs: &[EdgeRun<Av1EdgeFilter>],
    ) {
        let plane = RunPlane {
            width: dst.width(),
            height: dst.height(),
            stride: dst.stride(),
            vertical,
        };
        av1_scalar_at(dst.data_mut(), plane, runs, kernels.depth);
    }

    fn cdef_filter_band(
        kernels: DepthKernels<'_>,
        lines: &mut CdefLines<u16>,
        picture: &mut Picture<u16>,
        rows: Range<usize>,
        units: &[CdefUnit],
        sets: &CdefSets,
    ) {
        let interior = |size: usize,
                        (src, stride, origin): (&[u16], usize, usize),
                        args: &crate::cdef::CdefArgs,
                        (dst, dst_stride): (&mut [u16], usize)| {
            let kernel = kernels.dsp.high.cdef_interior[usize::from(size == CDEF_LUMA_SIZE)];
            // SAFETY: the table matches a supported ISA; the band driver
            // passes interior blocks only, whose taps lie in `src`, and the
            // kernel indexes its slices.
            unsafe { kernel(src, stride, origin, args, dst, dst_stride) }
        };
        crate::cdef::band::cdef_filter_band_at(
            (lines, picture),
            rows,
            units,
            (sets, kernels.depth),
            &interior,
        );
    }

    fn av1_loop_restoration(
        kernels: DepthKernels<'_>,
        restoration: &Av1LoopRestoration<'_>,
        plane: &mut PlaneViewMut<'_, u16>,
        boundaries: (&Av1StripeBoundaries<u16>, bool),
        scratch: &mut Av1LrScratch<u16>,
    ) {
        restoration.apply_at(plane, boundaries, scratch, kernels.depth);
    }

    fn av1_add_film_grain(
        kernels: DepthKernels<'_>,
        params: &Av1FilmGrainParams,
        input: [&PlaneView<'_, u16>; 3],
        size: (usize, usize),
        output: [&mut PlaneViewMut<'_, u16>; 3],
    ) {
        kernels
            .dsp
            .av1_add_film_grain_at(params, input, size, output, kernels.depth);
    }
}

#[allow(clippy::too_many_arguments)]
impl DepthKernels<'_> {
    /// [`Dsp::av1_inverse_add_i16`](crate::Dsp::av1_inverse_add_i16) at
    /// this depth: coefficients of [`Av1Pixel::Coefficient`], the result
    /// clipped to the depth (§7.13.3 with its `BitDepth` ranges).
    pub fn av1_inverse_add<T: Av1Pixel>(
        self,
        tx: Av1TxSize,
        kind: Av1TxType,
        coeffs: &mut [T::Coefficient],
        eob: usize,
        dst: &mut PlaneViewMut<'_, T>,
    ) {
        T::av1_inverse_add(self, tx, kind, coeffs, eob, dst);
    }

    /// [`Dsp::av1_inverse_wht_add`](crate::Dsp::av1_inverse_wht_add) at this
    /// depth.
    pub fn av1_inverse_wht_add<T: Av1Pixel>(
        self,
        coeffs: &[i32; 16],
        dst: &mut PlaneViewMut<'_, T>,
    ) {
        T::av1_inverse_wht_add(self, coeffs, dst);
    }

    /// [`Dsp::av1_intra_predict_block`](crate::Dsp::av1_intra_predict_block)
    /// for any sample type (the edges carry the depth).
    pub fn av1_intra_predict_block<T: Av1Pixel>(
        self,
        edges: &BlockEdges<T>,
        kind: IntraKind,
        dst: &mut PlaneViewMut<'_, T>,
    ) {
        T::av1_intra_predict_block(self, edges, kind, dst);
    }

    /// [`Dsp::av1_intra_predict_filter_block`](crate::Dsp::av1_intra_predict_filter_block)
    /// for any sample type.
    pub fn av1_intra_predict_filter_block<T: Av1Pixel>(
        self,
        edges: &BlockEdges<T>,
        mode: FilterIntraMode,
        dst: &mut PlaneViewMut<'_, T>,
    ) {
        T::av1_intra_predict_filter_block(self, edges, mode, dst);
    }

    /// [`Dsp::av1_put_samples`](crate::Dsp::av1_put_samples) at this depth.
    #[inline]
    pub fn av1_put_samples<T: Av1Pixel>(
        self,
        filters: Av1Filters,
        src: (&[T], usize),
        size: (usize, usize),
        fracs: (usize, usize),
        dst: (&mut [T], usize),
    ) {
        T::av1_put_samples(self, filters, src, size, fracs, dst);
    }

    /// [`Dsp::av1_prep_samples`](crate::Dsp::av1_prep_samples) at this
    /// depth (compound values less
    /// [`av1_compound_bias`](crate::av1_mc::av1_compound_bias)).
    #[inline]
    pub fn av1_prep_samples<T: Av1Pixel>(
        self,
        filters: Av1Filters,
        src: (&[T], usize),
        size: (usize, usize),
        fracs: (usize, usize),
        dst: (&mut [i16], usize),
    ) {
        T::av1_prep_samples(self, filters, src, size, fracs, dst);
    }

    /// [`Dsp::av1_predict`](crate::Dsp::av1_predict) at this depth.
    #[inline]
    pub fn av1_predict<T: Av1Pixel>(
        self,
        filters: Av1Filters,
        window: &PlaneView<'_, T>,
        size: (usize, usize),
        axes: (Av1McAxis, Av1McAxis),
        dst: &mut PlaneViewMut<'_, T>,
    ) {
        T::av1_predict(self, filters, window, size, axes, dst);
    }

    /// [`Dsp::av1_predict_compound`](crate::Dsp::av1_predict_compound) at
    /// this depth, into `dst` (`h` rows at its stride).
    pub fn av1_predict_compound<T: Av1Pixel>(
        self,
        filters: Av1Filters,
        window: &PlaneView<'_, T>,
        size: (usize, usize),
        axes: (Av1McAxis, Av1McAxis),
        dst: (&mut [i16], usize),
    ) {
        T::av1_predict_compound(self, filters, window, size, axes, dst);
    }

    /// The compound average (§7.11.3.1) of two `size` predictions (rows
    /// `size.0` apart) at this depth.
    pub fn av1_average_compound<T: Av1Pixel>(
        self,
        predictions: (&[i16], &[i16]),
        size: (usize, usize),
        dst: &mut PlaneViewMut<'_, T>,
    ) {
        T::av1_average_compound(self, predictions, size, dst);
    }

    /// [`av1_blend_mask`](crate::av1_mc::blend::av1_blend_mask) at this
    /// depth.
    pub fn av1_blend_mask<T: Av1Pixel>(
        self,
        predictions: (Av1CompoundPred<'_>, Av1CompoundPred<'_>),
        mask: Av1Mask<'_>,
        size: (usize, usize),
        dst: &mut PlaneViewMut<'_, T>,
    ) {
        T::av1_blend_mask(self, predictions, mask, size, dst);
    }

    /// [`av1_warp`] at this depth.
    pub fn av1_warp<T: Av1Pixel>(
        self,
        reference: &PlaneView<'_, T>,
        model: (&[i32; 6], Av1Shear),
        position: (usize, usize),
        size: (usize, usize),
        sub: (bool, bool),
        dst: &mut PlaneViewMut<'_, T>,
    ) {
        T::av1_warp(self, reference, model, position, size, sub, dst);
    }

    /// [`av1_warp_compound`] at this depth.
    pub fn av1_warp_compound<T: Av1Pixel>(
        self,
        reference: &PlaneView<'_, T>,
        model: (&[i32; 6], Av1Shear),
        position: (usize, usize),
        size: (usize, usize),
        sub: (bool, bool),
        dst: (&mut [i16], usize),
    ) {
        T::av1_warp_compound(self, reference, model, position, size, sub, dst);
    }

    /// [`Dsp::av1_loop_filter_runs`](crate::Dsp::av1_loop_filter_runs) at
    /// this depth (the 8-bit thresholds scaled inside).
    pub fn av1_loop_filter_runs<T: Av1Pixel>(
        self,
        dst: &mut PlaneViewMut<'_, T>,
        vertical: bool,
        runs: &[EdgeRun<Av1EdgeFilter>],
    ) {
        T::av1_loop_filter_runs(self, dst, vertical, runs);
    }

    /// [`Dsp::cdef_filter_band`](crate::Dsp::cdef_filter_band) at this
    /// depth.
    pub fn cdef_filter_band<T: Av1Pixel>(
        self,
        lines: &mut CdefLines<T>,
        picture: &mut Picture<T>,
        rows: Range<usize>,
        units: &[CdefUnit],
        sets: &CdefSets,
    ) {
        T::cdef_filter_band(self, lines, picture, rows, units, sets);
    }

    /// [`Dsp::av1_add_film_grain`](crate::Dsp::av1_add_film_grain) at this
    /// depth.
    pub fn av1_add_film_grain<T: Av1Pixel>(
        self,
        params: &Av1FilmGrainParams,
        input: [&PlaneView<'_, T>; 3],
        size: (usize, usize),
        output: [&mut PlaneViewMut<'_, T>; 3],
    ) {
        T::av1_add_film_grain(self, params, input, size, output);
    }

    /// [`Av1LoopRestoration::apply`] at this depth.
    pub fn av1_loop_restoration<T: Av1Pixel>(
        self,
        restoration: &Av1LoopRestoration<'_>,
        plane: &mut PlaneViewMut<'_, T>,
        boundaries: (&Av1StripeBoundaries<T>, bool),
        scratch: &mut Av1LrScratch<T>,
    ) {
        T::av1_loop_restoration(self, restoration, plane, boundaries, scratch);
    }
}
