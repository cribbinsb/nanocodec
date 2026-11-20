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
use crate::av1_mc::blend::{Av1CompoundPred, Av1Mask, Av1ObmcEdge};
use crate::av1_mc::convolve::{Av1McAxis, KERNEL_EDGE, check_prediction};
use crate::av1_mc::filter::{AV1_FILTER_TAPS, AV1_SUBPEL_POSITIONS, Av1Filters};
use crate::av1_mc::inter_round0;
use crate::av1_mc::warp::{Av1Shear, av1_warp, av1_warp_compound, warp_compound, warp_final};
use crate::av1_mc::warp_kernels::{WarpOut, WarpPhases};
use crate::av1_restoration::{Av1LoopRestoration, Av1LrScratch, Av1StripeBoundaries};
use crate::cdef::band::{CdefLines, CdefSets, CdefUnit};
use crate::deblock::runs::{EdgeRun, Plane as RunPlane};
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
    fn av1_cfl_predict(
        kernels: DepthKernels<'_>,
        luma: &PlaneView<'_, Self>,
        luma_extent: (usize, usize),
        dc: Self,
        alpha_q3: i32,
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
    fn av1_blend_distance(
        kernels: DepthKernels<'_>,
        predictions: (Av1CompoundPred<'_>, Av1CompoundPred<'_>),
        weights: [u8; 2],
        size: (usize, usize),
        dst: &mut PlaneViewMut<'_, Self>,
    );
    #[doc(hidden)]
    fn av1_blend_interintra(
        kernels: DepthKernels<'_>,
        intra: &PlaneView<'_, Self>,
        mask: Av1Mask<'_>,
        size: (usize, usize),
        dst: &mut PlaneViewMut<'_, Self>,
    );
    #[doc(hidden)]
    fn av1_obmc_blend(
        kernels: DepthKernels<'_>,
        obmc: &PlaneView<'_, Self>,
        edge: Av1ObmcEdge,
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
        lanes: &mut crate::LoopFilterLanes,
    );
    #[doc(hidden)]
    fn av1_lane_pass(
        kernels: DepthKernels<'_>,
        dst: &PlaneViewMut<'_, Self>,
        vertical: bool,
        chroma: bool,
        lanes: &mut crate::LoopFilterLanes,
    ) -> bool;
    #[doc(hidden)]
    fn av1_loop_filter_lanes(
        kernels: DepthKernels<'_>,
        dst: &mut PlaneViewMut<'_, Self>,
        vertical: bool,
        lanes: &mut crate::LoopFilterLanes,
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
#[derive(Clone, Copy)]
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
const NARROW_PADDED_MAX: usize = 16;
const NARROW_WINDOW: usize = (NARROW_PADDED_MAX + AV1_FILTER_TAPS - 1).pow(2);
const NARROW_BLOCK: usize = NARROW_PADDED_MAX * NARROW_PADDED_MAX;

#[cold]
#[inline(never)]
fn put_samples_u16_general(
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
    let mut dst = PlaneViewMut::new(dst, dst_stride, w, h).expect("destination covers the block");
    let axes = (Av1McAxis::unscaled(fracs.0), Av1McAxis::unscaled(fracs.1));
    u16::av1_predict(kernels, filters, &window, (w, h), axes, &mut dst);
}

#[cold]
#[inline(never)]
fn prep_samples_u16_general(
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
    u16::av1_predict_compound(kernels, filters, &window, (w, h), axes, dst);
}

fn narrow_unscaled_u16(
    filters: Av1Filters,
    (w, h): (usize, usize),
    axes: (Av1McAxis, Av1McAxis),
) -> Option<Unscaled> {
    if w.min(h) != 2 || !w.is_power_of_two() || !h.is_power_of_two() {
        return None;
    }
    let padded = (w.max(KERNEL_MIN_SIDE), h.max(KERNEL_MIN_SIDE));
    if padded.0.max(padded.1) > NARROW_PADDED_MAX {
        return None;
    }
    let (phase_x, phase_y) = (axes.0.unscaled_phase()?, axes.1.unscaled_phase()?);
    Some(Unscaled {
        taps: (&filters.x.kernel(w)[phase_x], &filters.y.kernel(h)[phase_y]),
        class: (padded.0.trailing_zeros() - KERNEL_MIN_SIDE.trailing_zeros()) as usize,
        tile: padded,
    })
}

fn put_narrow_u16(
    kernels: DepthKernels<'_>,
    case: Unscaled,
    window: &PlaneView<'_, u16>,
    (w, h): (usize, usize),
    dst: &mut PlaneViewMut<'_, u16>,
) {
    let (pw, ph) = case.tile;
    let span_w = pw + AV1_FILTER_TAPS - 1;
    let mut samples = [0u16; NARROW_WINDOW];
    for (r, row) in samples
        .chunks_exact_mut(span_w)
        .take(h + AV1_FILTER_TAPS - 1)
        .enumerate()
    {
        row[..w + AV1_FILTER_TAPS - 1].copy_from_slice(&window.row(r)[..w + AV1_FILTER_TAPS - 1]);
    }
    let mut block = [0u16; NARROW_BLOCK];
    let kernel = kernels.dsp.high.av1_mc.put[case.class];
    // SAFETY: `samples` covers `(ph + 7) * span_w` and `block` covers `ph * pw`.
    unsafe {
        kernel(
            (&samples, span_w),
            (&mut block, pw),
            ph,
            case.taps,
            kernels.depth,
        );
    }
    for r in 0..h {
        dst.row_mut(r)[..w].copy_from_slice(&block[r * pw..r * pw + w]);
    }
}

fn prep_narrow_u16(
    kernels: DepthKernels<'_>,
    case: Unscaled,
    window: &PlaneView<'_, u16>,
    (w, h): (usize, usize),
    (dst, stride): (&mut [i16], usize),
) {
    let (pw, ph) = case.tile;
    let span_w = pw + AV1_FILTER_TAPS - 1;
    let mut samples = [0u16; NARROW_WINDOW];
    for (r, row) in samples
        .chunks_exact_mut(span_w)
        .take(h + AV1_FILTER_TAPS - 1)
        .enumerate()
    {
        row[..w + AV1_FILTER_TAPS - 1].copy_from_slice(&window.row(r)[..w + AV1_FILTER_TAPS - 1]);
    }
    let mut block = [0i16; NARROW_BLOCK];
    let kernel = kernels.dsp.high.av1_mc.prep[case.class];
    // SAFETY: `samples` covers `(ph + 7) * span_w` and `block` covers `ph * pw`.
    unsafe {
        kernel(
            (&samples, span_w),
            (&mut block, pw),
            ph,
            case.taps,
            kernels.depth,
        );
    }
    for r in 0..h {
        dst[r * stride..r * stride + w].copy_from_slice(&block[r * pw..r * pw + w]);
    }
}

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

    #[inline]
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

    fn av1_cfl_predict(
        kernels: DepthKernels<'_>,
        luma: &PlaneView<'_, u8>,
        luma_extent: (usize, usize),
        dc: u8,
        alpha_q3: i32,
        dst: &mut PlaneViewMut<'_, u8>,
    ) {
        // SAFETY: `high` matches this CPU; `cfl_u8` validates views and extents.
        unsafe {
            (kernels.dsp.high.av1_intra.cfl_u8)(
                luma,
                luma_extent,
                dc,
                alpha_q3,
                dst,
                kernels.depth,
            );
        }
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
        let (w, h) = size;
        // The average kernels take edges up to 64: a 128-sample edge
        // (128x128, 128x64 and 64x128 blocks) is averaged as 64-sample
        // tiles read at the block's stride. Scalar, those blocks were 4 % of
        // a libaom "good" 720p decode (2026-11-19).
        match BlockSize::new(w.min(KERNEL_EDGE), h.min(KERNEL_EDGE)) {
            Some(tile) => {
                for y in (0..h).step_by(tile.height()) {
                    for x in (0..w).step_by(tile.width()) {
                        let at = y * w + x;
                        let mut window = dst
                            .sub_view_mut(x, y, tile.width(), tile.height())
                            .expect("AV1 compound destination covers the block");
                        kernels
                            .dsp
                            .average_compound(tile, (&p0[at..], &p1[at..]), w, &mut window);
                    }
                }
            }
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
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        // SAFETY: `high` matches this CPU; kernel validates `p0`, `p1`, `mask`, and `dst`.
        unsafe {
            (kernels.dsp.high.av1_mc.blend.mask_u8)(p0, p1, mask, size, dst, kernels.depth);
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        crate::av1_mc::blend::av1_blend_mask(kernels.dsp, p0, p1, mask, size, dst);
    }

    fn av1_blend_distance(
        kernels: DepthKernels<'_>,
        (p0, p1): (Av1CompoundPred<'_>, Av1CompoundPred<'_>),
        weights: [u8; 2],
        size: (usize, usize),
        dst: &mut PlaneViewMut<'_, u8>,
    ) {
        // SAFETY: `high` matches this CPU; kernel validates `p0`, `p1`, and `dst`.
        unsafe {
            (kernels.dsp.high.av1_mc.blend.distance_u8)(p0, p1, weights, size, dst, kernels.depth);
        }
    }

    fn av1_blend_interintra(
        kernels: DepthKernels<'_>,
        intra: &PlaneView<'_, u8>,
        mask: Av1Mask<'_>,
        size: (usize, usize),
        dst: &mut PlaneViewMut<'_, u8>,
    ) {
        // SAFETY: `high` matches this CPU; kernel validates `intra`, `mask`, and `dst`.
        unsafe { (kernels.dsp.high.av1_mc.blend.interintra_u8)(intra, mask, size, dst) };
    }

    fn av1_obmc_blend(
        kernels: DepthKernels<'_>,
        obmc: &PlaneView<'_, u8>,
        edge: Av1ObmcEdge,
        size: (usize, usize),
        dst: &mut PlaneViewMut<'_, u8>,
    ) {
        // SAFETY: `high` matches this CPU; kernel validates `obmc` and `dst`.
        unsafe { (kernels.dsp.high.av1_mc.blend.obmc_u8)(obmc, edge, size, dst) };
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
        lanes: &mut crate::LoopFilterLanes,
    ) {
        kernels.dsp.av1_loop_filter_runs(dst, vertical, runs, lanes);
    }

    fn av1_lane_pass(
        kernels: DepthKernels<'_>,
        dst: &PlaneViewMut<'_, u8>,
        vertical: bool,
        chroma: bool,
        lanes: &mut crate::LoopFilterLanes,
    ) -> bool {
        kernels.dsp.av1_lane_pass(dst, vertical, chroma, lanes)
    }

    fn av1_loop_filter_lanes(
        kernels: DepthKernels<'_>,
        dst: &mut PlaneViewMut<'_, u8>,
        vertical: bool,
        lanes: &mut crate::LoopFilterLanes,
    ) {
        kernels.dsp.av1_loop_filter_lanes(dst, vertical, lanes);
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
        assert!(kind.fits(tx), "{kind:?} is not defined for {tx:?}");
        let (width, height) = (tx.width(), tx.height());
        assert!(
            coeffs.len() >= tx.coded_area(),
            "{} coefficients for a {width}x{height} transform",
            coeffs.len()
        );
        assert!(
            dst.width() >= width && dst.height() >= height,
            "inverse destination smaller than {tx:?}"
        );
        if eob == 0 {
            return;
        }
        let depth = kernels.depth;
        let stride = dst.stride();
        if eob == 1 {
            if kind == Av1TxType::DctDct {
                let res = crate::av1_inverse::generic::dc_residual_high(tx, coeffs[0], depth);
                // SAFETY: `high` matches this CPU; destination validated above.
                unsafe {
                    (kernels.dsp.high.av1_inverse.dc)(
                        res,
                        width,
                        height,
                        dst.data_mut(),
                        stride,
                        depth,
                    );
                }
                return;
            }
            if kind == Av1TxType::Idtx {
                let res = crate::av1_inverse::generic::idtx_dc_residual(tx, coeffs[0], depth);
                let p0 = &mut dst.data_mut()[0];
                *p0 = depth.clip_to(i32::from(*p0) + res);
                return;
            }
        }
        let nonzero_rows = if eob >= tx.coded_area() {
            tx.coded_height()
        } else if depth.bits() == 10 && kind == Av1TxType::DctDct && width == height && width >= 8 {
            crate::av1_inverse::av1_dct_nonzero_side(eob).min(tx.coded_height())
        } else {
            crate::av1_inverse::av1_inverse_nonzero_rows(tx, kind, eob)
        };
        let index = tx.index().expect("valid size (checked by fits)");
        let (vertical, horizontal) = kind.kernels();
        match depth.bits() {
            10 => {
                // SAFETY: `high` matches this CPU; lengths validated above.
                unsafe {
                    (kernels.dsp.high.av1_inverse.add_10[index])(
                        vertical,
                        horizontal,
                        coeffs,
                        nonzero_rows,
                        dst.data_mut(),
                        stride,
                        depth,
                    );
                }
            }
            12 => {
                // SAFETY: `high` matches this CPU; lengths validated above.
                unsafe {
                    (kernels.dsp.high.av1_inverse.add_12[index])(
                        vertical,
                        horizontal,
                        coeffs,
                        nonzero_rows,
                        dst.data_mut(),
                        stride,
                        depth,
                    );
                }
            }
            _ => crate::av1_inverse::high::inverse_add_at(
                tx,
                kind,
                coeffs,
                dst.data_mut(),
                stride,
                depth,
            ),
        }
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

    #[inline]
    fn av1_intra_predict_block(
        kernels: DepthKernels<'_>,
        edges: &BlockEdges<u16>,
        kind: IntraKind,
        dst: &mut PlaneViewMut<'_, u16>,
    ) {
        use crate::intra_prediction::IntraMode;
        use crate::intra_prediction::av1_block::Square;
        let intra = &kernels.dsp.high.av1_intra;
        edges.predict_full(
            |square, prediction, dst| match prediction {
                Square::Angle(angle) => {
                    // SAFETY: `high` matches this CPU; `directional` validates angle and `dst`.
                    unsafe { (intra.directional)(square, angle, dst) };
                }
                Square::Mode(IntraMode::Paeth) => {
                    // SAFETY: `high` matches this CPU; `paeth_u16` validates `dst`.
                    unsafe { (intra.paeth_u16)(edges, dst) };
                }
                Square::Mode(
                    IntraMode::Smooth | IntraMode::SmoothVertical | IntraMode::SmoothHorizontal,
                ) => {
                    // SAFETY: `high` matches this CPU; `smooth_u16` validates `dst`.
                    unsafe { (intra.smooth_u16)(edges, kind, dst) };
                }
                Square::Mode(mode) => square.predict(mode, dst),
            },
            // SAFETY: `high` matches this CPU.
            |e, dst| unsafe { (intra.paeth_u16)(e, dst) },
            // SAFETY: `high` matches this CPU.
            |e, k, dst| unsafe { (intra.smooth_u16)(e, k, dst) },
            // SAFETY: `high` matches this CPU.
            |win, weight, out| unsafe { (intra.run_u16)(win, weight, out) },
            None,
            kind,
            dst,
        );
    }

    fn av1_intra_predict_filter_block(
        kernels: DepthKernels<'_>,
        edges: &BlockEdges<u16>,
        mode: FilterIntraMode,
        dst: &mut PlaneViewMut<'_, u16>,
    ) {
        let patches = kernels.dsp.high.av1_intra.filter_patches;
        let depth = kernels.depth;
        edges.predict_filter_with(
            // SAFETY: `high` matches this CPU; `predict_filter_with` checks block sides.
            |grid, width, height, mode| unsafe { patches(grid, width, height, mode, depth) },
            mode,
            dst,
        );
    }

    fn av1_cfl_predict(
        kernels: DepthKernels<'_>,
        luma: &PlaneView<'_, u16>,
        luma_extent: (usize, usize),
        dc: u16,
        alpha_q3: i32,
        dst: &mut PlaneViewMut<'_, u16>,
    ) {
        // SAFETY: `high` matches this CPU; `cfl_u16` validates views and extents.
        unsafe {
            (kernels.dsp.high.av1_intra.cfl_u16)(
                luma,
                luma_extent,
                dc,
                alpha_q3,
                dst,
                kernels.depth,
            );
        }
    }

    #[allow(
        clippy::inline_always,
        reason = "hot sample-buffer entry; out of line its setup added ~27M instructions on 1080p"
    )]
    #[inline(always)]
    fn av1_put_samples(
        kernels: DepthKernels<'_>,
        filters: Av1Filters,
        (src, src_stride): (&[u16], usize),
        (w, h): (usize, usize),
        (fx, fy): (usize, usize),
        (dst, dst_stride): (&mut [u16], usize),
    ) {
        const KERNEL_SIDE_MASK: usize = 4 | 8 | 16 | 32 | 64;
        if (w & KERNEL_SIDE_MASK) != 0
            && w.is_power_of_two()
            && (h & KERNEL_SIDE_MASK) != 0
            && h.is_power_of_two()
            && (fx | fy) < AV1_SUBPEL_POSITIONS
            && src_stride >= w + AV1_FILTER_TAPS - 1
            && dst_stride >= w
        {
            let class = (w.trailing_zeros() - KERNEL_MIN_SIDE.trailing_zeros()) as usize;
            let taps = (&filters.x.kernel(w)[fx], &filters.y.kernel(h)[fy]);
            let kernel = kernels.dsp.high.av1_mc.put[class];
            // SAFETY: `high` matches this CPU; `kernel` validates `src` and `dst` lengths.
            unsafe { kernel((src, src_stride), (dst, dst_stride), h, taps, kernels.depth) };
            return;
        }
        put_samples_u16_general(
            kernels,
            filters,
            (src, src_stride),
            (w, h),
            (fx, fy),
            (dst, dst_stride),
        );
    }

    #[allow(
        clippy::inline_always,
        reason = "called twice per plane on every compound block; out of line cost 26.6M self Ir"
    )]
    #[inline(always)]
    fn av1_prep_samples(
        kernels: DepthKernels<'_>,
        filters: Av1Filters,
        (src, src_stride): (&[u16], usize),
        (w, h): (usize, usize),
        (fx, fy): (usize, usize),
        (dst, dst_stride): (&mut [i16], usize),
    ) {
        const KERNEL_SIDE_MASK: usize = 4 | 8 | 16 | 32 | 64;
        if (w & KERNEL_SIDE_MASK) != 0
            && w.is_power_of_two()
            && (h & KERNEL_SIDE_MASK) != 0
            && h.is_power_of_two()
            && (fx | fy) < AV1_SUBPEL_POSITIONS
            && src_stride >= w + AV1_FILTER_TAPS - 1
            && dst_stride >= w
        {
            let class = (w.trailing_zeros() - KERNEL_MIN_SIDE.trailing_zeros()) as usize;
            let taps = (&filters.x.kernel(w)[fx], &filters.y.kernel(h)[fy]);
            let kernel = kernels.dsp.high.av1_mc.prep[class];
            // SAFETY: `high` matches this CPU; `kernel` validates `src` and `dst` lengths.
            unsafe { kernel((src, src_stride), (dst, dst_stride), h, taps, kernels.depth) };
            return;
        }
        prep_samples_u16_general(
            kernels,
            filters,
            (src, src_stride),
            (w, h),
            (fx, fy),
            (dst, dst_stride),
        );
    }

    #[inline]
    fn av1_predict(
        kernels: DepthKernels<'_>,
        filters: Av1Filters,
        window: &PlaneView<'_, u16>,
        size: (usize, usize),
        axes: (Av1McAxis, Av1McAxis),
        dst: &mut PlaneViewMut<'_, u16>,
    ) {
        if let Some(case) = Unscaled::of(filters, size, axes) {
            check_prediction(window, size, axes);
            assert!(
                dst.width() >= size.0 && dst.height() >= size.1,
                "AV1 prediction destination too small"
            );
            let (ws, ds) = (window.stride(), dst.stride());
            let kernel = kernels.dsp.high.av1_mc.put[case.class];
            if case.tile == size {
                // SAFETY: `high` matches this CPU; window and destination cover the block.
                unsafe {
                    kernel(
                        (window.data(), ws),
                        (dst.data_mut(), ds),
                        size.1,
                        case.taps,
                        kernels.depth,
                    );
                }
                return;
            }
            for (left, top) in case.corners(size) {
                let src = &window.data()[top * ws + left..];
                let out = &mut dst.data_mut()[top * ds + left..];
                // SAFETY: `high` matches this CPU; window and destination cover every tile.
                unsafe { kernel((src, ws), (out, ds), case.tile.1, case.taps, kernels.depth) };
            }
            return;
        }
        if let Some(padded) = narrow_unscaled_u16(filters, size, axes) {
            check_prediction(window, size, axes);
            assert!(
                dst.width() >= size.0 && dst.height() >= size.1,
                "AV1 prediction destination too small"
            );
            put_narrow_u16(kernels, padded, window, size, dst);
            return;
        }
        // SAFETY: `high` matches this CPU; `scaled_put` validates window and `dst`.
        unsafe {
            (kernels.dsp.high.av1_mc.scaled_put)(filters, window, size, axes, dst, kernels.depth);
        }
    }

    #[inline]
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
        if let Some(case) = Unscaled::of(filters, (w, h), axes) {
            check_prediction(window, (w, h), axes);
            let ws = window.stride();
            let kernel = kernels.dsp.high.av1_mc.prep[case.class];
            if case.tile == (w, h) {
                // SAFETY: as for `av1_predict`.
                unsafe {
                    kernel(
                        (window.data(), ws),
                        (dst, stride),
                        h,
                        case.taps,
                        kernels.depth,
                    );
                }
                return;
            }
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
            return;
        }
        if let Some(padded) = narrow_unscaled_u16(filters, (w, h), axes) {
            check_prediction(window, (w, h), axes);
            prep_narrow_u16(kernels, padded, window, (w, h), (dst, stride));
            return;
        }
        // SAFETY: `high` matches this CPU; `scaled_prep` validates window and `dst`.
        unsafe {
            (kernels.dsp.high.av1_mc.scaled_prep)(
                filters,
                window,
                (w, h),
                axes,
                (dst, stride),
                kernels.depth,
            );
        }
    }

    #[inline]
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
        // SAFETY: `high` matches this CPU; kernel validates `p0`, `p1`, and `dst`.
        unsafe {
            (kernels.dsp.high.av1_mc.blend.average)(
                prediction(p0),
                prediction(p1),
                size,
                dst,
                kernels.depth,
            );
        }
    }

    fn av1_blend_mask(
        kernels: DepthKernels<'_>,
        (p0, p1): (Av1CompoundPred<'_>, Av1CompoundPred<'_>),
        mask: Av1Mask<'_>,
        size: (usize, usize),
        dst: &mut PlaneViewMut<'_, u16>,
    ) {
        // SAFETY: `high` matches this CPU; kernel validates `p0`, `p1`, `mask`, and `dst`.
        unsafe {
            (kernels.dsp.high.av1_mc.blend.mask)(p0, p1, mask, size, dst, kernels.depth);
        }
    }

    fn av1_blend_distance(
        kernels: DepthKernels<'_>,
        (p0, p1): (Av1CompoundPred<'_>, Av1CompoundPred<'_>),
        weights: [u8; 2],
        size: (usize, usize),
        dst: &mut PlaneViewMut<'_, u16>,
    ) {
        // SAFETY: `high` matches this CPU; kernel validates `p0`, `p1`, and `dst`.
        unsafe {
            (kernels.dsp.high.av1_mc.blend.distance)(p0, p1, weights, size, dst, kernels.depth);
        }
    }

    fn av1_blend_interintra(
        kernels: DepthKernels<'_>,
        intra: &PlaneView<'_, u16>,
        mask: Av1Mask<'_>,
        size: (usize, usize),
        dst: &mut PlaneViewMut<'_, u16>,
    ) {
        // SAFETY: `high` matches this CPU; kernel validates `intra`, `mask`, and `dst`.
        unsafe { (kernels.dsp.high.av1_mc.blend.interintra)(intra, mask, size, dst) };
    }

    fn av1_obmc_blend(
        kernels: DepthKernels<'_>,
        obmc: &PlaneView<'_, u16>,
        edge: Av1ObmcEdge,
        size: (usize, usize),
        dst: &mut PlaneViewMut<'_, u16>,
    ) {
        // SAFETY: `high` matches this CPU; kernel validates `obmc` and `dst`.
        unsafe { (kernels.dsp.high.av1_mc.blend.obmc)(obmc, edge, size, dst) };
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
        let r0 = inter_round0(kernels.depth);
        let warp_fn = kernels.dsp.high.av1_mc.warp8x8;
        let kernel = |w: (&[u16], usize), phases: &WarpPhases, r1: u32, out: &mut WarpOut| {
            // SAFETY: `high` matches this CPU; `warp` passes a validated 15x16 window.
            unsafe { warp_fn(w, phases, (r0, r1), out) };
        };
        warp_final(
            &kernel,
            reference,
            (model, shear),
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
        let r0 = inter_round0(kernels.depth);
        let warp_fn = kernels.dsp.high.av1_mc.warp8x8;
        let kernel = |w: (&[u16], usize), phases: &WarpPhases, r1: u32, out: &mut WarpOut| {
            // SAFETY: `high` matches this CPU; `warp` passes a validated 15x16 window.
            unsafe { warp_fn(w, phases, (r0, r1), out) };
        };
        warp_compound(
            &kernel,
            reference,
            (model, shear),
            position,
            size,
            sub,
            dst,
            kernels.depth,
        );
    }

    /// High depth has no lane kernels: its passes take run lists.
    fn av1_lane_pass(
        _kernels: DepthKernels<'_>,
        _dst: &PlaneViewMut<'_, u16>,
        _vertical: bool,
        _chroma: bool,
        _lanes: &mut crate::LoopFilterLanes,
    ) -> bool {
        false
    }

    fn av1_loop_filter_lanes(
        _kernels: DepthKernels<'_>,
        _dst: &mut PlaneViewMut<'_, u16>,
        _vertical: bool,
        _lanes: &mut crate::LoopFilterLanes,
    ) {
        unreachable!("high-depth passes never begin a lane pass");
    }

    fn av1_loop_filter_runs(
        kernels: DepthKernels<'_>,
        dst: &mut PlaneViewMut<'_, u16>,
        vertical: bool,
        runs: &[EdgeRun<Av1EdgeFilter>],
        _lanes: &mut crate::LoopFilterLanes,
    ) {
        let plane = RunPlane {
            width: dst.width(),
            height: dst.height(),
            stride: dst.stride(),
            vertical,
        };
        // SAFETY: `high` matches this CPU; `ValidatedRuns` validates every run's footprint.
        unsafe {
            (kernels.dsp.high.av1_deblock_runs)(dst.data_mut(), plane, runs, kernels.depth);
        }
    }

    fn cdef_filter_band(
        kernels: DepthKernels<'_>,
        lines: &mut CdefLines<u16>,
        picture: &mut Picture<u16>,
        rows: Range<usize>,
        units: &[CdefUnit],
        sets: &CdefSets,
    ) {
        kernels
            .dsp
            .cdef_filter_band_u16(lines, picture, rows, units, (sets, kernels.depth));
    }

    fn av1_loop_restoration(
        kernels: DepthKernels<'_>,
        restoration: &Av1LoopRestoration<'_>,
        plane: &mut PlaneViewMut<'_, u16>,
        boundaries: (&Av1StripeBoundaries<u16>, bool),
        scratch: &mut Av1LrScratch<u16>,
    ) {
        restoration.apply_u16(kernels.dsp, plane, boundaries, scratch, kernels.depth);
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
            .av1_add_film_grain_u16(params, input, size, output, kernels.depth);
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

    /// One-pass AV1 Chroma-from-Luma (`CfL`, §7.11.5) prediction at this depth.
    pub fn av1_cfl_predict<T: Av1Pixel>(
        self,
        luma: &PlaneView<'_, T>,
        luma_extent: (usize, usize),
        dc: T,
        alpha_q3: i32,
        dst: &mut PlaneViewMut<'_, T>,
    ) {
        T::av1_cfl_predict(self, luma, luma_extent, dc, alpha_q3, dst);
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

    /// [`av1_blend_distance`] at this depth.
    pub fn av1_blend_distance<T: Av1Pixel>(
        self,
        predictions: (Av1CompoundPred<'_>, Av1CompoundPred<'_>),
        weights: [u8; 2],
        size: (usize, usize),
        dst: &mut PlaneViewMut<'_, T>,
    ) {
        T::av1_blend_distance(self, predictions, weights, size, dst);
    }

    /// [`av1_difference_weight_mask`](crate::av1_mc::mask::av1_difference_weight_mask) at this depth.
    pub fn av1_difference_weight_mask(
        self,
        (p0, p1): (Av1CompoundPred<'_>, Av1CompoundPred<'_>),
        inverse: bool,
        size: (usize, usize),
        mask: &mut [u8],
    ) {
        // SAFETY: `high` matches this CPU; kernel validates `p0`, `p1`, and `mask`.
        unsafe {
            (self.dsp.high.av1_mc.blend.diff_mask)(p0, p1, inverse, size, mask, self.depth);
        }
    }

    /// [`av1_blend_interintra`] at this depth.
    pub fn av1_blend_interintra<T: Av1Pixel>(
        self,
        intra: &PlaneView<'_, T>,
        mask: Av1Mask<'_>,
        size: (usize, usize),
        dst: &mut PlaneViewMut<'_, T>,
    ) {
        T::av1_blend_interintra(self, intra, mask, size, dst);
    }

    /// [`av1_obmc_blend`] at this depth.
    pub fn av1_obmc_blend<T: Av1Pixel>(
        self,
        obmc: &PlaneView<'_, T>,
        edge: Av1ObmcEdge,
        size: (usize, usize),
        dst: &mut PlaneViewMut<'_, T>,
    ) {
        T::av1_obmc_blend(self, obmc, edge, size, dst);
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
        lanes: &mut crate::LoopFilterLanes,
    ) {
        T::av1_loop_filter_runs(self, dst, vertical, runs, lanes);
    }

    /// [`Dsp::av1_lane_pass`](crate::Dsp::av1_lane_pass) at this depth
    /// (`false` above 8 bits: no lane kernels).
    #[must_use]
    pub fn av1_lane_pass<T: Av1Pixel>(
        self,
        dst: &PlaneViewMut<'_, T>,
        vertical: bool,
        chroma: bool,
        lanes: &mut crate::LoopFilterLanes,
    ) -> bool {
        T::av1_lane_pass(self, dst, vertical, chroma, lanes)
    }

    /// [`Dsp::av1_loop_filter_lanes`](crate::Dsp::av1_loop_filter_lanes)
    /// at this depth, after [`DepthKernels::av1_lane_pass`].
    pub fn av1_loop_filter_lanes<T: Av1Pixel>(
        self,
        dst: &mut PlaneViewMut<'_, T>,
        vertical: bool,
        lanes: &mut crate::LoopFilterLanes,
    ) {
        T::av1_loop_filter_lanes(self, dst, vertical, lanes);
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
