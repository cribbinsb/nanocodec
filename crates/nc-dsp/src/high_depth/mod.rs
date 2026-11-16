//! Kernels at any bit depth: one generic entry point, [`Dsp::at`], for the
//! decoders' sample-type-generic code (`docs/handoff/high-bit-depth.md`).
//!
//! A decoder written once over `T: Pixel` (`u8` or `u16`, monomorphised
//! twice, as dav1d builds its `bitdepth` templates) calls
//! `dsp.at(depth).kernel(...)`:
//!
//! - for `u8` the call is the 8-bit [`Dsp`] method itself, with its SIMD
//!   tiers: the 8-bit instance compiles to exactly the code it had before
//!   high bit depth existed (`depth` is always 8 there and unused);
//! - for `u16` it is this module's table, [`HighFns`], whose entries are the
//!   same scalar references the 8-bit tables use, instantiated at `u16`
//!   (the `*_at` functions of each kernel family, generic over
//!   [`nc_common::Sample`] with the depth as a runtime [`BitDepth`]), until
//!   an ISA installs a vector kernel for a measured hot path.
//!
//! Every entry here validates its views exactly as the 8-bit method does,
//! so a vector kernel installed later inherits a sound contract.
//!
//! Adding a codec: give its kernel family `*_at` scalar references generic
//! over the sample type (keep the 8-bit SIMD kernels as they are), add the
//! `u16` entry to [`HighFns`], and the routing method to [`Pixel`] and
//! [`DepthKernels`].

#![allow(unsafe_code)]

pub(crate) mod av1;
mod inverse;
// Vector u16 kernels exist for x86 and aarch64 only.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod vector;
pub(crate) mod vp9;
mod vvc_filter;
mod vvc_recon;

pub(crate) use inverse::whole_block as inverse_whole_block;
pub use vvc_recon::VvcReconPixel;

use nc_common::{BitDepth, PlaneView, PlaneViewMut, Sample};

use crate::block::BlockSize;
use crate::codec_txfm::HevcTransform;
use crate::deblock::{HEVC_LINES, HevcEdge};
use crate::dispatch::{
    check_covers, check_covers_i16, check_sub_blocks, hevc_edge_at, sao_offsets, sao_rect,
};
use crate::hevc_intra::PlanarEdges;
use crate::hevc_weight::HevcWeights;
use crate::mc::{Filter, SubBlockMotion};
use crate::sao::Rect;
use crate::{Dsp, Isa, VvcGeometric};

/// Uni-prediction of a block of one width class from a window of `u16`
/// samples (`crate::mc::ConvolveFn` with the depth).
pub(crate) type Convolve16Fn =
    unsafe fn(&[u16], usize, &mut [u16], usize, usize, (usize, usize), BitDepth);
/// Centred 14-bit prediction (`crate::mc::PrepFn` with the depth).
pub(crate) type Prep16Fn =
    unsafe fn(&[u16], usize, &mut [i16], usize, usize, (usize, usize), BitDepth);
/// Default bi-prediction (`crate::mc::BiFn` with the depth).
pub(crate) type Bi16Fn =
    unsafe fn(&[u16], usize, (&[i16], usize), &mut [u16], usize, usize, (usize, usize), BitDepth);
/// HEVC weighted sample prediction (`crate::hevc_weight::WeightFn` with the depth).
pub(crate) type Weight16Fn =
    unsafe fn(&HevcWeights, &[i16], &[i16], usize, usize, &mut [u16], usize, BitDepth);
/// HEVC angular intra lines (`crate::hevc_intra::AngularFn` at `u16`).
pub(crate) type Angular16Fn = unsafe fn(&[u16], usize, i32, &mut [u16]);
/// HEVC planar intra (`crate::hevc_intra::PlanarFn` at `u16`).
pub(crate) type Planar16Fn = unsafe fn(&PlanarEdges<'_, u16>, usize, &mut [u16]);
/// HEVC inverse transform to the residual at a depth (`20 − BitDepth`);
/// the extent (rows, columns) covers every nonzero coefficient.
pub(crate) type Inverse16Fn = unsafe fn(&[i16], &mut [i16], (usize, usize), BitDepth);
/// Add an `n`x`n` residual to the samples at a stride, clipping to the depth.
pub(crate) type AddResidual16Fn = unsafe fn(&[i16], usize, &mut [u16], usize, BitDepth);
/// HEVC deblocking of one edge (`crate::deblock::HevcEdgeFn` with the depth).
pub(crate) type Edge16Fn = unsafe fn(&mut [u16], &HevcEdge, BitDepth);
/// SAO edge-offset application (`crate::sao::ApplyFn` with the depth).
pub(crate) type SaoApply16Fn =
    unsafe fn(&[u16], usize, &mut [u16], usize, (isize, isize), [i16; 4], Rect, BitDepth);
/// SAO band-offset application (`crate::sao::BandFn` with the depth).
pub(crate) type SaoBand16Fn = unsafe fn(&mut [u16], usize, Rect, u8, [i16; 4], &[bool], BitDepth);
/// The fixed taps of an unscaled AV1 prediction: horizontal, vertical.
pub(crate) type Av1Taps<'t> = (&'t [i16; 8], &'t [i16; 8]);
/// Unscaled AV1 prediction of a block of one width class at final
/// precision (`av1_mc::convolve::unscaled_put_at`): window and stride,
/// destination and stride, height (at most 64), taps, depth.
pub(crate) type Av1Put16Fn =
    for<'t> unsafe fn((&[u16], usize), (&mut [u16], usize), usize, Av1Taps<'t>, BitDepth);
/// CDEF of a block whose taps lie inside the plane
/// (`cdef::scalar::interior_rows`): source, stride, origin, arguments,
/// destination and stride.
pub(crate) type CdefInterior16Fn =
    unsafe fn(&[u16], usize, usize, &crate::cdef::CdefArgs, &mut [u16], usize);
/// [`Av1Put16Fn`] at compound precision (`unscaled_prep_at`).
pub(crate) type Av1Prep16Fn =
    for<'t> unsafe fn((&[u16], usize), (&mut [i16], usize), usize, Av1Taps<'t>, BitDepth);

/// HEVC/VVC prediction kept at six fractional bits
/// (`Dsp::predict_high_precision` with the depth).
pub(crate) type HighPrecision16Fn = unsafe fn(
    Filter,
    (usize, usize),
    &PlaneView<'_, u16>,
    (usize, usize),
    bool,
    &mut [i16],
    BitDepth,
);
/// VVC geometric-partition blend (`Dsp::vvc_geometric_blend` with the depth).
pub(crate) type GeometricBlend16Fn = unsafe fn(
    VvcGeometric,
    (usize, usize),
    (usize, usize),
    bool,
    (&[i16], &[i16]),
    &mut [u16],
    BitDepth,
);
/// VVC DMVR bilinear prediction (`Dsp::vvc_dmvr_bilinear` with the depth).
pub(crate) type DmvrBilinear16Fn =
    unsafe fn((usize, usize), &PlaneView<'_, u16>, (usize, usize), &mut [i16], BitDepth);
/// VVC BDOF (`Dsp::vvc_bdof` with the depth).
pub(crate) type Bdof16Fn = unsafe fn((usize, usize), [&[i16]; 2], &mut [u16], BitDepth);

/// Kernels per width class (4, 8, 16, 32, 64).
const WIDTH_CLASSES: usize = 5;
/// The separable 14-bit interpolation families (`SeparableTaps` tables:
/// HEVC luma and chroma, VVC luma, affine luma and chroma), which share
/// HEVC's and VVC's intermediate precision (`shift1 = Min(4, BitDepth −
/// 8)`, `shift2 = 6`, `shift3 = Max(2, 14 − BitDepth)`).
const FAMILIES: usize = 5;

/// The `u16` kernels of one ISA.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HighFns {
    /// Uni-prediction by family ([`family`]) and width class.
    pub predict: [[Convolve16Fn; WIDTH_CLASSES]; FAMILIES],
    /// Centred prediction by family and width class.
    pub prep: [[Prep16Fn; WIDTH_CLASSES]; FAMILIES],
    /// Default bi-prediction by family and width class.
    pub bi: [[Bi16Fn; WIDTH_CLASSES]; FAMILIES],
    pub hevc_weight: Weight16Fn,
    pub hevc_angular: Angular16Fn,
    pub hevc_planar: Planar16Fn,
    /// By [`HevcTransform`].
    pub hevc_inverse: [Inverse16Fn; 5],
    pub add_residual: AddResidual16Fn,
    pub hevc_deblock_luma: Edge16Fn,
    pub hevc_deblock_chroma: Edge16Fn,
    pub sao_apply: SaoApply16Fn,
    pub sao_band: SaoBand16Fn,
    pub high_precision: HighPrecision16Fn,
    pub vvc_geometric_blend: GeometricBlend16Fn,
    pub vvc_dmvr_bilinear: DmvrBilinear16Fn,
    pub vvc_bdof: Bdof16Fn,
    /// VVC deblocking and ALF (`vvc_filter`).
    pub vvc_filter: vvc_filter::VvcFilterFns,
    pub vvc_recon: vvc_recon::VvcReconFns,
    /// Unscaled AV1 prediction by width class.
    pub av1_put: [Av1Put16Fn; WIDTH_CLASSES],
    /// Unscaled AV1 compound prediction by width class.
    pub av1_prep: [Av1Prep16Fn; WIDTH_CLASSES],
    /// CDEF of interior 4x4 and 8x8 blocks.
    pub cdef_interior: [CdefInterior16Fn; 2],
}

/// One kernel per width class (4, 8, 16, 32, 64) of `$kernel` generic over
/// the sample type and the width.
macro_rules! av1_widths {
    ($($kernel:ident)::+) => {
        [
            $($kernel)::+::<4>,
            $($kernel)::+::<8>,
            $($kernel)::+::<16>,
            $($kernel)::+::<32>,
            $($kernel)::+::<64>,
        ]
    };
}
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) use av1_widths;

/// The kernels of one filter family by width class (`$kernel` generic
/// over the taps, the tap count and the width).
macro_rules! by_width {
    ($($kernel:ident)::+, $taps:ty, $n:literal) => {
        [
            $($kernel)::+::<$taps, $n, 4>,
            $($kernel)::+::<$taps, $n, 8>,
            $($kernel)::+::<$taps, $n, 16>,
            $($kernel)::+::<$taps, $n, 32>,
            $($kernel)::+::<$taps, $n, 64>,
        ]
    };
}

/// One kernel for every family, in [`family`] order.
macro_rules! families {
    ($($kernel:ident)::+) => {
        [
            $crate::high_depth::by_width!($($kernel)::+, $crate::mc::HevcLumaTaps, 8),
            $crate::high_depth::by_width!($($kernel)::+, $crate::mc::HevcChromaTaps, 4),
            $crate::high_depth::by_width!($($kernel)::+, $crate::mc::VvcLumaTaps, 8),
            $crate::high_depth::by_width!($($kernel)::+, $crate::mc::VvcAffineLumaTaps, 8),
            $crate::high_depth::by_width!($($kernel)::+, $crate::mc::VvcChromaTaps, 4),
        ]
    };
}
pub(crate) use by_width;
// The ISA tables (`vector`, x86 and aarch64 only) name this by path.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) use families;

/// The scalar `u16` instances of the MC references, by taps and width.
mod scalar {
    use nc_common::BitDepth;

    use super::Av1Taps;
    use crate::av1_mc::convolve::{unscaled_prep_at, unscaled_put_at};
    use crate::mc::SeparableTaps;
    use crate::mc::scalar::{hevc_bi_at, hevc_prep_at, separable_at};

    pub(super) fn av1_put<const W: usize>(
        src: (&[u16], usize),
        dst: (&mut [u16], usize),
        h: usize,
        taps: Av1Taps<'_>,
        depth: BitDepth,
    ) {
        unscaled_put_at::<u16, W>(src, dst, h, taps, depth);
    }

    pub(super) fn av1_prep<const W: usize>(
        src: (&[u16], usize),
        dst: (&mut [i16], usize),
        h: usize,
        taps: Av1Taps<'_>,
        depth: BitDepth,
    ) {
        unscaled_prep_at::<u16, W>(src, dst, h, taps, depth);
    }

    pub(super) fn predict<T: SeparableTaps<N>, const N: usize, const W: usize>(
        src: &[u16],
        ss: usize,
        dst: &mut [u16],
        ds: usize,
        h: usize,
        frac: (usize, usize),
        depth: BitDepth,
    ) {
        separable_at::<u16, T, N, W>(src, ss, dst, ds, h, frac, depth);
    }

    pub(super) fn prep<T: SeparableTaps<N>, const N: usize, const W: usize>(
        src: &[u16],
        ss: usize,
        dst: &mut [i16],
        ds: usize,
        h: usize,
        frac: (usize, usize),
        depth: BitDepth,
    ) {
        hevc_prep_at::<u16, T, N, W>(src, ss, dst, ds, h, frac, depth);
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn bi<T: SeparableTaps<N>, const N: usize, const W: usize>(
        src: &[u16],
        ss: usize,
        first: (&[i16], usize),
        dst: &mut [u16],
        ds: usize,
        h: usize,
        frac: (usize, usize),
        depth: BitDepth,
    ) {
        hevc_bi_at::<u16, T, N, W>(src, ss, first, dst, ds, h, frac, depth);
    }
}

impl HighFns {
    /// The table for `isa`: the scalar references, and the ISA's vector
    /// kernels where one exists (see the module docs for adding one).
    pub(crate) fn for_isa(isa: Isa) -> Self {
        #[cfg_attr(
            not(any(target_arch = "x86_64", target_arch = "aarch64")),
            allow(
                unused_mut,
                reason = "only the x86 and aarch64 tables install vector kernels"
            )
        )]
        let mut table = Self {
            predict: families!(scalar::predict),
            prep: families!(scalar::prep),
            bi: families!(scalar::bi),
            hevc_weight: crate::hevc_weight::reference::weight_at::<u16>,
            hevc_angular: crate::hevc_intra::reference::angular::<u16>,
            hevc_planar: crate::hevc_intra::reference::planar::<u16>,
            hevc_inverse: inverse::kernels(isa),
            add_residual: add_residual::<u16>,
            hevc_deblock_luma: crate::deblock::scalar::hevc_luma_edge_at::<u16>,
            hevc_deblock_chroma: crate::deblock::scalar::hevc_chroma_edge_at::<u16>,
            sao_apply: crate::sao::scalar::apply_at::<u16>,
            sao_band: crate::sao::scalar::band_at::<u16>,
            high_precision: crate::mc::precision::high_precision_at::<u16>,
            vvc_geometric_blend: crate::vvc_geo::blend_at::<u16>,
            vvc_dmvr_bilinear: crate::vvc_dmvr::bilinear_at::<u16>,
            vvc_bdof: crate::vvc_bdof::scalar_at::<u16>,
            vvc_filter: vvc_filter::VvcFilterFns::SCALAR,
            vvc_recon: vvc_recon::VvcReconFns::SCALAR,
            av1_put: av1_widths!(scalar::av1_put),
            av1_prep: av1_widths!(scalar::av1_prep),
            cdef_interior: [
                crate::cdef::scalar::interior_rows::<u16, 4>,
                crate::cdef::scalar::interior_rows::<u16, 8>,
            ],
        };
        match isa {
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 | Isa::Avx512 => {
                table.predict = vector::avx2::PREDICT;
                table.prep = vector::avx2::PREP;
                table.bi = vector::avx2::BI;
                table.hevc_weight = vector::avx2::WEIGHT;
                table.av1_put = vector::avx2::AV1_PUT;
                table.av1_prep = vector::avx2::AV1_PREP;
                table.cdef_interior = vector::avx2::CDEF_INTERIOR;
            }
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => {
                table.predict = vector::neon::PREDICT;
                table.prep = vector::neon::PREP;
                table.bi = vector::neon::BI;
                table.hevc_weight = vector::neon::WEIGHT;
                table.av1_put = vector::neon::AV1_PUT;
                table.av1_prep = vector::neon::AV1_PREP;
                table.cdef_interior = vector::neon::CDEF_INTERIOR;
            }
            _ => {}
        }
        table
    }
}

/// Add the `n`x`n` residual (row-major) to `dst` at `stride` with `Clip1`
/// to `depth`: the reconstruction of every block-based codec
/// (`recSamples = Clip1(predSamples + resSamples)`, HEVC eq. 8-262, VVC
/// eq. 8-1012, AV1 §7.13.3 with its own `Round2`s done beforehand).
pub(crate) fn add_residual<T: Sample>(
    residual: &[i16],
    n: usize,
    dst: &mut [T],
    stride: usize,
    depth: BitDepth,
) {
    // Folds to the constant 8 in the 8-bit instance.
    let depth = depth.of::<T>();
    for (row, line) in residual[..n * n].chunks_exact(n).enumerate() {
        for (s, &r) in dst[row * stride..row * stride + n].iter_mut().zip(line) {
            *s = depth.clip_to(s.to_i32() + i32::from(r));
        }
    }
}

/// A sample type with kernels: `u8` (the 8-bit [`Dsp`] methods) or `u16`
/// ([`HighFns`]). The decoders' generic code is written over `T: Pixel`
/// and calls the kernels through [`DepthKernels`]; these methods are the
/// routing behind it, one per kernel (a kernel family with many methods
/// keeps its routing in a supertrait beside it: `vvc_filter::VvcFilterPixel`,
/// [`VvcReconPixel`]).
#[allow(clippy::too_many_arguments)]
pub trait Pixel:
    Sample + vvc_filter::VvcFilterPixel + VvcReconPixel + av1::Av1Pixel + vp9::Vp9Pixel
{
    #[doc(hidden)]
    fn predict(
        kernels: DepthKernels<'_>,
        filter: Filter,
        size: BlockSize,
        window: &PlaneView<'_, Self>,
        frac: (usize, usize),
        dst: &mut PlaneViewMut<'_, Self>,
    );
    #[doc(hidden)]
    fn predict_compound(
        kernels: DepthKernels<'_>,
        filter: Filter,
        size: BlockSize,
        window: &PlaneView<'_, Self>,
        frac: (usize, usize),
        dst: &mut [i16],
        dst_stride: usize,
    );
    #[doc(hidden)]
    fn hevc_predict_bi(
        kernels: DepthKernels<'_>,
        filter: Filter,
        size: BlockSize,
        window: &PlaneView<'_, Self>,
        frac: (usize, usize),
        first: (&[i16], usize),
        dst: &mut PlaneViewMut<'_, Self>,
    );
    #[doc(hidden)]
    fn predict_sub_blocks(
        kernels: DepthKernels<'_>,
        filter: Filter,
        size: BlockSize,
        window: &PlaneView<'_, Self>,
        motion: &[SubBlockMotion],
        per_row: usize,
        dst: &mut PlaneViewMut<'_, Self>,
    );
    #[doc(hidden)]
    fn predict_high_precision(
        kernels: DepthKernels<'_>,
        filter: Filter,
        size: (usize, usize),
        window: &PlaneView<'_, Self>,
        phases: (usize, usize),
        alternate_half: bool,
        out: &mut [i16],
    );
    #[doc(hidden)]
    fn hevc_weighted_prediction(
        kernels: DepthKernels<'_>,
        weights: HevcWeights,
        a: &[i16],
        b: &[i16],
        dst: &mut PlaneViewMut<'_, Self>,
    );
    #[doc(hidden)]
    fn vvc_geometric_blend(
        kernels: DepthKernels<'_>,
        mask: VvcGeometric,
        origin: (usize, usize),
        size: (usize, usize),
        chroma: bool,
        sources: (&[i16], &[i16]),
        out: &mut [Self],
    );
    #[doc(hidden)]
    fn vvc_dmvr_bilinear(
        kernels: DepthKernels<'_>,
        size: (usize, usize),
        input: &PlaneView<'_, Self>,
        phases: (usize, usize),
        out: &mut [i16],
    );
    #[doc(hidden)]
    fn vvc_bdof(
        kernels: DepthKernels<'_>,
        size: (usize, usize),
        inputs: [&[i16]; 2],
        out: &mut [Self],
    );
    #[doc(hidden)]
    fn hevc_weighted_prediction_centered(
        kernels: DepthKernels<'_>,
        weights: HevcWeights,
        a: &[i16],
        b: &[i16],
        dst: &mut PlaneViewMut<'_, Self>,
    );
    #[doc(hidden)]
    fn hevc_intra_angular(
        kernels: DepthKernels<'_>,
        refs: &[Self],
        n: usize,
        angle: i32,
        out: &mut [Self],
    );
    #[doc(hidden)]
    fn hevc_intra_planar(
        kernels: DepthKernels<'_>,
        edges: &PlanarEdges<'_, Self>,
        n: usize,
        out: &mut [Self],
    );
    #[doc(hidden)]
    fn hevc_inverse_add(
        kernels: DepthKernels<'_>,
        transform: HevcTransform,
        coeffs: &[i16],
        extent: (usize, usize),
        dst: &mut PlaneViewMut<'_, Self>,
    );
    #[doc(hidden)]
    fn hevc_deblock_luma_edge(
        kernels: DepthKernels<'_>,
        dst: &mut PlaneViewMut<'_, Self>,
        at: (usize, usize),
        vertical: bool,
        tc: [i16; 2],
        beta: i16,
    );
    #[doc(hidden)]
    fn hevc_deblock_chroma_edge(
        kernels: DepthKernels<'_>,
        dst: &mut PlaneViewMut<'_, Self>,
        at: (usize, usize),
        vertical: bool,
        lines: usize,
        tc: [i16; 4],
    );
    #[doc(hidden)]
    fn sao_band_apply_masked(
        kernels: DepthKernels<'_>,
        position: u8,
        offsets: [i16; 4],
        plane: &mut PlaneViewMut<'_, Self>,
        rect: (usize, usize, usize, usize),
        exempt: &[bool],
    );
    #[doc(hidden)]
    fn sao_edge_apply(
        kernels: DepthKernels<'_>,
        class: u8,
        offsets: [i16; 4],
        input: &PlaneView<'_, Self>,
        out: &mut PlaneViewMut<'_, Self>,
        rect: (usize, usize, usize, usize),
    );
}

/// The kernels of a [`Dsp`] at one bit depth ([`Dsp::at`]).
#[derive(Debug, Clone, Copy)]
pub struct DepthKernels<'a> {
    dsp: &'a Dsp,
    depth: BitDepth,
}

impl Dsp {
    /// This table's kernels at `depth`, for code generic over the sample
    /// type (module docs of `high_depth`). With `u8` samples `depth` must
    /// be 8.
    #[must_use]
    #[inline]
    pub fn at(&self, depth: BitDepth) -> DepthKernels<'_> {
        DepthKernels { dsp: self, depth }
    }
}

/// SAO offsets at 8 bits (`sao_offset_abs` <= 7, HEVC §7.4.9.3.2).
fn eight_offsets(offsets: [i16; 4]) -> [i8; 4] {
    offsets.map(|v| i8::try_from(v).expect("8-bit SAO offset"))
}

/// The kernels' width class (4, 8, 16, 32, 64 → 0..5).
fn width_class(size: BlockSize) -> usize {
    size.width_class()
}

/// The [`HighFns`] family index of a separable 14-bit filter.
///
/// # Panics
/// For the other filters (AV1, VP9, H.264), which have no high-bit-depth
/// kernels yet.
fn family(filter: Filter) -> usize {
    match filter {
        Filter::HevcLuma => 0,
        Filter::HevcChroma => 1,
        Filter::VvcLuma => 2,
        Filter::VvcAffineLuma => 3,
        Filter::VvcChroma => 4,
        other => panic!("{other:?} has no high-bit-depth kernel"),
    }
}

/// Check a prediction window and sub-sample offsets, as the 8-bit methods do.
fn check_window<T: Sample>(
    filter: Filter,
    size: BlockSize,
    window: &PlaneView<'_, T>,
    (frac_x, frac_y): (usize, usize),
) {
    let extra = filter.taps() - 1;
    assert!(
        window.width() >= size.width() + extra && window.height() >= size.height() + extra,
        "{filter:?} prediction window {}x{} is smaller than {}x{}",
        window.width(),
        window.height(),
        size.width() + extra,
        size.height() + extra
    );
    assert!(
        frac_x < filter.positions() && frac_y < filter.positions(),
        "{filter:?} sub-sample offsets ({frac_x}, {frac_y}) out of range"
    );
}

impl Pixel for u8 {
    fn predict(
        kernels: DepthKernels<'_>,
        filter: Filter,
        size: BlockSize,
        window: &PlaneView<'_, u8>,
        (frac_x, frac_y): (usize, usize),
        dst: &mut PlaneViewMut<'_, u8>,
    ) {
        kernels
            .dsp
            .predict(filter, size, window, frac_x, frac_y, dst);
    }

    fn predict_compound(
        kernels: DepthKernels<'_>,
        filter: Filter,
        size: BlockSize,
        window: &PlaneView<'_, u8>,
        frac: (usize, usize),
        dst: &mut [i16],
        dst_stride: usize,
    ) {
        kernels
            .dsp
            .predict_compound(filter, size, window, frac, dst, dst_stride);
    }

    fn hevc_predict_bi(
        kernels: DepthKernels<'_>,
        filter: Filter,
        size: BlockSize,
        window: &PlaneView<'_, u8>,
        frac: (usize, usize),
        first: (&[i16], usize),
        dst: &mut PlaneViewMut<'_, u8>,
    ) {
        kernels
            .dsp
            .hevc_predict_bi(filter, size, window, frac, first, dst);
    }

    fn predict_sub_blocks(
        kernels: DepthKernels<'_>,
        filter: Filter,
        size: BlockSize,
        window: &PlaneView<'_, u8>,
        motion: &[SubBlockMotion],
        per_row: usize,
        dst: &mut PlaneViewMut<'_, u8>,
    ) {
        kernels
            .dsp
            .predict_sub_blocks(filter, size, window, motion, per_row, dst);
    }

    fn predict_high_precision(
        kernels: DepthKernels<'_>,
        filter: Filter,
        size: (usize, usize),
        window: &PlaneView<'_, u8>,
        phases: (usize, usize),
        alternate_half: bool,
        out: &mut [i16],
    ) {
        kernels
            .dsp
            .predict_high_precision(filter, size, window, phases, alternate_half, out);
    }

    fn hevc_weighted_prediction(
        kernels: DepthKernels<'_>,
        weights: HevcWeights,
        a: &[i16],
        b: &[i16],
        dst: &mut PlaneViewMut<'_, u8>,
    ) {
        kernels.dsp.hevc_weighted_prediction(weights, a, b, dst);
    }

    fn vvc_geometric_blend(
        kernels: DepthKernels<'_>,
        mask: VvcGeometric,
        origin: (usize, usize),
        size: (usize, usize),
        chroma: bool,
        sources: (&[i16], &[i16]),
        out: &mut [u8],
    ) {
        kernels
            .dsp
            .vvc_geometric_blend(mask, origin, size, chroma, sources, out);
    }

    fn vvc_dmvr_bilinear(
        kernels: DepthKernels<'_>,
        size: (usize, usize),
        input: &PlaneView<'_, u8>,
        phases: (usize, usize),
        out: &mut [i16],
    ) {
        kernels.dsp.vvc_dmvr_bilinear(size, input, phases, out);
    }

    fn vvc_bdof(
        kernels: DepthKernels<'_>,
        size: (usize, usize),
        inputs: [&[i16]; 2],
        out: &mut [u8],
    ) {
        kernels.dsp.vvc_bdof(size, inputs, out);
    }

    fn hevc_weighted_prediction_centered(
        kernels: DepthKernels<'_>,
        weights: HevcWeights,
        a: &[i16],
        b: &[i16],
        dst: &mut PlaneViewMut<'_, u8>,
    ) {
        kernels
            .dsp
            .hevc_weighted_prediction_centered(weights, a, b, dst);
    }

    fn hevc_intra_angular(
        kernels: DepthKernels<'_>,
        refs: &[u8],
        n: usize,
        angle: i32,
        out: &mut [u8],
    ) {
        kernels.dsp.hevc_intra_angular(refs, n, angle, out);
    }

    fn hevc_intra_planar(
        kernels: DepthKernels<'_>,
        edges: &PlanarEdges<'_, u8>,
        n: usize,
        out: &mut [u8],
    ) {
        kernels.dsp.hevc_intra_planar(edges, n, out);
    }

    fn hevc_inverse_add(
        kernels: DepthKernels<'_>,
        transform: HevcTransform,
        coeffs: &[i16],
        extent: (usize, usize),
        dst: &mut PlaneViewMut<'_, u8>,
    ) {
        kernels.dsp.hevc_inverse_add(transform, coeffs, extent, dst);
    }

    fn hevc_deblock_luma_edge(
        kernels: DepthKernels<'_>,
        dst: &mut PlaneViewMut<'_, u8>,
        at: (usize, usize),
        vertical: bool,
        tc: [i16; 2],
        beta: i16,
    ) {
        kernels
            .dsp
            .hevc_deblock_luma_edge_i16(dst, at, vertical, tc, beta);
    }

    fn hevc_deblock_chroma_edge(
        kernels: DepthKernels<'_>,
        dst: &mut PlaneViewMut<'_, u8>,
        at: (usize, usize),
        vertical: bool,
        lines: usize,
        tc: [i16; 4],
    ) {
        kernels
            .dsp
            .hevc_deblock_chroma_edge_i16(dst, at, vertical, lines, tc);
    }

    fn sao_band_apply_masked(
        kernels: DepthKernels<'_>,
        position: u8,
        offsets: [i16; 4],
        plane: &mut PlaneViewMut<'_, u8>,
        rect: (usize, usize, usize, usize),
        exempt: &[bool],
    ) {
        kernels
            .dsp
            .sao_band_apply_masked(position, eight_offsets(offsets), plane, rect, exempt);
    }

    fn sao_edge_apply(
        kernels: DepthKernels<'_>,
        class: u8,
        offsets: [i16; 4],
        input: &PlaneView<'_, u8>,
        out: &mut PlaneViewMut<'_, u8>,
        rect: (usize, usize, usize, usize),
    ) {
        kernels
            .dsp
            .sao_edge_apply(class, eight_offsets(offsets), input, out, rect);
    }
}

impl Pixel for u16 {
    fn predict(
        kernels: DepthKernels<'_>,
        filter: Filter,
        size: BlockSize,
        window: &PlaneView<'_, u16>,
        frac: (usize, usize),
        dst: &mut PlaneViewMut<'_, u16>,
    ) {
        check_window(filter, size, window, frac);
        check_covers(size, &dst.as_view(), "destination");
        let stride = dst.stride();
        let kernel = kernels.dsp.high.predict[family(filter)][width_class(size)];
        // SAFETY: the views cover the window and the block (checked above);
        // the table matches a supported ISA (checked in `with_isa`).
        unsafe {
            kernel(
                window.data(),
                window.stride(),
                dst.data_mut(),
                stride,
                size.height(),
                frac,
                kernels.depth,
            );
        }
    }

    fn predict_compound(
        kernels: DepthKernels<'_>,
        filter: Filter,
        size: BlockSize,
        window: &PlaneView<'_, u16>,
        frac: (usize, usize),
        dst: &mut [i16],
        dst_stride: usize,
    ) {
        check_window(filter, size, window, frac);
        check_covers_i16(size, dst.len(), dst_stride, "compound prediction");
        let kernel = kernels.dsp.high.prep[family(filter)][width_class(size)];
        // SAFETY: as for `predict`.
        unsafe {
            kernel(
                window.data(),
                window.stride(),
                dst,
                dst_stride,
                size.height(),
                frac,
                kernels.depth,
            );
        }
    }

    fn hevc_predict_bi(
        kernels: DepthKernels<'_>,
        filter: Filter,
        size: BlockSize,
        window: &PlaneView<'_, u16>,
        frac: (usize, usize),
        (first, first_stride): (&[i16], usize),
        dst: &mut PlaneViewMut<'_, u16>,
    ) {
        check_window(filter, size, window, frac);
        check_covers_i16(size, first.len(), first_stride, "first prediction");
        check_covers(size, &dst.as_view(), "destination");
        let stride = dst.stride();
        let kernel = kernels.dsp.high.bi[family(filter)][width_class(size)];
        // SAFETY: as for `predict`.
        unsafe {
            kernel(
                window.data(),
                window.stride(),
                (first, first_stride),
                dst.data_mut(),
                stride,
                size.height(),
                frac,
                kernels.depth,
            );
        }
    }

    fn predict_sub_blocks(
        kernels: DepthKernels<'_>,
        filter: Filter,
        size: BlockSize,
        window: &PlaneView<'_, u16>,
        motion: &[SubBlockMotion],
        per_row: usize,
        dst: &mut PlaneViewMut<'_, u16>,
    ) {
        check_sub_blocks(filter, size, window, motion, per_row, &dst.as_view());
        let (w, h) = (size.width(), size.height());
        let kernel = kernels.dsp.high.predict[family(filter)][width_class(size)];
        let (src, src_stride) = (window.data(), window.stride());
        let dst_stride = dst.stride();
        let out = dst.data_mut();
        for (i, m) in motion.iter().enumerate() {
            let from = usize::from(m.y) * src_stride + usize::from(m.x);
            let to = (i / per_row) * h * dst_stride + (i % per_row) * w;
            // SAFETY: the sub-block's window lies inside `window` and its
            // place inside `dst` (checked above); the table matches a
            // supported ISA.
            unsafe {
                kernel(
                    &src[from..],
                    src_stride,
                    &mut out[to..],
                    dst_stride,
                    h,
                    (usize::from(m.frac_x), usize::from(m.frac_y)),
                    kernels.depth,
                );
            }
        }
    }

    fn predict_high_precision(
        kernels: DepthKernels<'_>,
        filter: Filter,
        size: (usize, usize),
        window: &PlaneView<'_, u16>,
        phases: (usize, usize),
        alternate_half: bool,
        out: &mut [i16],
    ) {
        crate::mc::precision::check_high_precision(filter, size, window, phases, alternate_half);
        assert!(
            out.len() >= size.0 * size.1,
            "prediction shorter than the block"
        );
        // SAFETY: the window and output cover the block (checked above);
        // the table matches a supported ISA.
        unsafe {
            (kernels.dsp.high.high_precision)(
                filter,
                size,
                window,
                phases,
                alternate_half,
                out,
                kernels.depth,
            );
        }
    }

    fn hevc_weighted_prediction(
        kernels: DepthKernels<'_>,
        weights: HevcWeights,
        a: &[i16],
        b: &[i16],
        dst: &mut PlaneViewMut<'_, u16>,
    ) {
        let (w, h) = (dst.width(), dst.height());
        assert!(
            a.len() >= w * h && b.len() >= w * h,
            "prediction shorter than the block"
        );
        assert!(
            (1..=crate::hevc_weight::MAX_SHIFT).contains(&weights.shift),
            "weighted prediction shift out of range"
        );
        let stride = dst.stride();
        // SAFETY: the slices cover the block (checked above, and by the view).
        unsafe {
            (kernels.dsp.high.hevc_weight)(
                &weights,
                a,
                b,
                w,
                h,
                dst.data_mut(),
                stride,
                kernels.depth,
            );
        }
    }

    fn vvc_geometric_blend(
        kernels: DepthKernels<'_>,
        mask: VvcGeometric,
        origin: (usize, usize),
        size: (usize, usize),
        chroma: bool,
        sources: (&[i16], &[i16]),
        out: &mut [u16],
    ) {
        let count = size.0 * size.1;
        assert!(
            sources.0.len() >= count && sources.1.len() >= count && out.len() >= count,
            "geometric blend buffers shorter than the block"
        );
        // SAFETY: every buffer covers the block (checked above); the table
        // matches a supported ISA.
        unsafe {
            (kernels.dsp.high.vvc_geometric_blend)(
                mask,
                origin,
                size,
                chroma,
                sources,
                out,
                kernels.depth,
            );
        }
    }

    fn vvc_dmvr_bilinear(
        kernels: DepthKernels<'_>,
        size: (usize, usize),
        input: &PlaneView<'_, u16>,
        phases: (usize, usize),
        out: &mut [i16],
    ) {
        crate::vvc_dmvr::check_bilinear(size, input, phases, out);
        // SAFETY: window, phases and output checked; the table matches a
        // supported ISA.
        unsafe { (kernels.dsp.high.vvc_dmvr_bilinear)(size, input, phases, out, kernels.depth) }
    }

    fn vvc_bdof(
        kernels: DepthKernels<'_>,
        size: (usize, usize),
        inputs: [&[i16]; 2],
        out: &mut [u16],
    ) {
        crate::vvc_bdof::check(size, inputs, out.len());
        // SAFETY: dimensions and buffers checked; the table matches a
        // supported ISA.
        unsafe { (kernels.dsp.high.vvc_bdof)(size, inputs, out, kernels.depth) }
    }

    fn hevc_weighted_prediction_centered(
        kernels: DepthKernels<'_>,
        mut weights: HevcWeights,
        a: &[i16],
        b: &[i16],
        dst: &mut PlaneViewMut<'_, u16>,
    ) {
        weights.round += i32::from(crate::mc::HEVC_PREP_OFFSET)
            * (i32::from(weights.w0) + i32::from(weights.w1));
        Self::hevc_weighted_prediction(kernels, weights, a, b, dst);
    }

    fn hevc_intra_angular(
        kernels: DepthKernels<'_>,
        refs: &[u16],
        n: usize,
        angle: i32,
        out: &mut [u16],
    ) {
        use crate::hevc_intra::MAX_ANGLE;
        assert!(matches!(n, 4 | 8 | 16 | 32), "HEVC intra side {n}");
        assert!(
            (-MAX_ANGLE..=MAX_ANGLE).contains(&angle),
            "HEVC angle {angle}"
        );
        assert_eq!(refs.len(), 3 * n + 1, "HEVC main reference length");
        let out = &mut out[..n * n];
        // SAFETY: checked sizes; the table matches a supported ISA.
        unsafe { (kernels.dsp.high.hevc_angular)(refs, n, angle, out) }
    }

    fn hevc_intra_planar(
        kernels: DepthKernels<'_>,
        edges: &PlanarEdges<'_, u16>,
        n: usize,
        out: &mut [u16],
    ) {
        assert!(matches!(n, 4 | 8 | 16 | 32), "HEVC intra side {n}");
        let edges = PlanarEdges {
            above: &edges.above[..n],
            left: &edges.left[..n],
            ..*edges
        };
        let out = &mut out[..n * n];
        // SAFETY: as for `hevc_intra_angular`.
        unsafe { (kernels.dsp.high.hevc_planar)(&edges, n, out) }
    }

    fn hevc_inverse_add(
        kernels: DepthKernels<'_>,
        transform: HevcTransform,
        coeffs: &[i16],
        (rows, columns): (usize, usize),
        dst: &mut PlaneViewMut<'_, u16>,
    ) {
        let n = transform.side();
        assert!(coeffs.len() >= n * n, "coefficients do not cover the block");
        assert!(
            rows <= n && columns <= n,
            "extent ({rows}, {columns}) beyond the {n}x{n} transform"
        );
        assert!(
            dst.width() >= n && dst.height() >= n,
            "destination smaller than HEVC transform"
        );
        if rows == 0 || columns == 0 {
            return;
        }
        let mut residual = [0i16; inverse::MAX_AREA];
        let residual = &mut residual[..n * n];
        let stride = dst.stride();
        // SAFETY: the coefficients and residual cover the block, the
        // destination view covers `n` rows of `n` (checked above); the table
        // matches a supported ISA.
        unsafe {
            (kernels.dsp.high.hevc_inverse[transform as usize])(
                coeffs,
                residual,
                (rows, columns),
                kernels.depth,
            );
            (kernels.dsp.high.add_residual)(residual, n, dst.data_mut(), stride, kernels.depth);
        }
    }

    fn hevc_deblock_luma_edge(
        kernels: DepthKernels<'_>,
        dst: &mut PlaneViewMut<'_, u16>,
        at: (usize, usize),
        vertical: bool,
        tc: [i16; 2],
        beta: i16,
    ) {
        let edge = hevc_edge_at(dst, at, vertical, HEVC_LINES, [tc[0], tc[1], 0, 0], beta);
        // SAFETY: `hevc_edge_at` checked that every sample the lines read
        // lies inside `dst`; the table matches a supported ISA.
        unsafe { (kernels.dsp.high.hevc_deblock_luma)(dst.data_mut(), &edge, kernels.depth) }
    }

    fn hevc_deblock_chroma_edge(
        kernels: DepthKernels<'_>,
        dst: &mut PlaneViewMut<'_, u16>,
        at: (usize, usize),
        vertical: bool,
        lines: usize,
        tc: [i16; 4],
    ) {
        assert!(
            lines == HEVC_LINES || lines == HEVC_LINES / 2,
            "HEVC chroma edges are 4 or 8 lines"
        );
        let edge = hevc_edge_at(dst, at, vertical, lines, tc, 0);
        // SAFETY: as for `hevc_deblock_luma_edge`.
        unsafe { (kernels.dsp.high.hevc_deblock_chroma)(dst.data_mut(), &edge, kernels.depth) }
    }

    fn sao_band_apply_masked(
        kernels: DepthKernels<'_>,
        position: u8,
        offsets: [i16; 4],
        plane: &mut PlaneViewMut<'_, u16>,
        (x0, y0, x1, y1): (usize, usize, usize, usize),
        exempt: &[bool],
    ) {
        assert!(
            position < 32 && x0 <= x1 && x1 <= plane.width() && y0 <= y1 && y1 <= plane.height()
        );
        assert!(exempt.is_empty() || exempt.len() == (x1 - x0) * (y1 - y0));
        let stride = plane.stride();
        let rect = Rect {
            x0,
            y0,
            x1,
            y1,
            row_step: 1,
        };
        // SAFETY: the view validates its storage; the rectangle and mask are
        // checked above; the table matches a supported ISA.
        unsafe {
            (kernels.dsp.high.sao_band)(
                plane.data_mut(),
                stride,
                rect,
                position,
                offsets,
                exempt,
                kernels.depth,
            );
        }
    }

    fn sao_edge_apply(
        kernels: DepthKernels<'_>,
        class: u8,
        offsets: [i16; 4],
        input: &PlaneView<'_, u16>,
        out: &mut PlaneViewMut<'_, u16>,
        rect: (usize, usize, usize, usize),
    ) {
        assert!(
            input.width() == out.width() && input.height() == out.height(),
            "SAO planes differ in size"
        );
        let rect = sao_rect(class, input.width(), input.height(), rect);
        let offs = sao_offsets(class, input.stride());
        let out_stride = out.stride();
        // SAFETY: `sao_rect` checked the rectangle and its neighbours lie
        // inside both planes; the table matches a supported ISA.
        unsafe {
            (kernels.dsp.high.sao_apply)(
                input.data(),
                input.stride(),
                out.data_mut(),
                out_stride,
                offs,
                offsets,
                rect,
                kernels.depth,
            );
        }
    }
}

impl DepthKernels<'_> {
    /// The depth these kernels clip to.
    #[must_use]
    pub fn depth(&self) -> BitDepth {
        self.depth
    }

    /// [`Dsp::predict`] at this depth (HEVC and VVC filters only beyond 8
    /// bits).
    ///
    /// # Panics
    /// As [`Dsp::predict`].
    pub fn predict<T: Pixel>(
        self,
        filter: Filter,
        size: BlockSize,
        window: &PlaneView<'_, T>,
        frac: (usize, usize),
        dst: &mut PlaneViewMut<'_, T>,
    ) {
        T::predict(self, filter, size, window, frac, dst);
    }

    /// [`Dsp::predict_compound`] at this depth: `predSamplesLX − 8192`,
    /// 14-bit at every depth (HEVC and VVC filters only beyond 8 bits).
    ///
    /// # Panics
    /// As [`Dsp::predict_compound`].
    pub fn predict_compound<T: Pixel>(
        self,
        filter: Filter,
        size: BlockSize,
        window: &PlaneView<'_, T>,
        frac: (usize, usize),
        dst: &mut [i16],
        dst_stride: usize,
    ) {
        T::predict_compound(self, filter, size, window, frac, dst, dst_stride);
    }

    /// [`Dsp::hevc_predict_bi`] at this depth.
    ///
    /// # Panics
    /// As [`Dsp::hevc_predict_bi`].
    pub fn hevc_predict_bi<T: Pixel>(
        self,
        filter: Filter,
        size: BlockSize,
        window: &PlaneView<'_, T>,
        frac: (usize, usize),
        first: (&[i16], usize),
        dst: &mut PlaneViewMut<'_, T>,
    ) {
        T::hevc_predict_bi(self, filter, size, window, frac, first, dst);
    }

    /// [`Dsp::predict_sub_blocks`] at this depth (HEVC and VVC filters
    /// only beyond 8 bits): each sub-block rounded to samples as
    /// [`Self::predict`].
    ///
    /// # Panics
    /// As [`Dsp::predict_sub_blocks`].
    pub fn predict_sub_blocks<T: Pixel>(
        self,
        filter: Filter,
        size: BlockSize,
        window: &PlaneView<'_, T>,
        motion: &[SubBlockMotion],
        per_row: usize,
        dst: &mut PlaneViewMut<'_, T>,
    ) {
        T::predict_sub_blocks(self, filter, size, window, motion, per_row, dst);
    }

    /// [`Dsp::predict_high_precision`] at this depth: `predSamplesLX` of
    /// H.266 §8.5.6.3.2, uncentred, with `shift1 = Min(4, BitDepth − 8)`,
    /// `shift2 = 6`, `shift3 = Max(2, 14 − BitDepth)` (14-bit for depths up
    /// to 12, so it fits `i16`).
    ///
    /// # Panics
    /// As [`Dsp::predict_high_precision`].
    pub fn predict_high_precision<T: Pixel>(
        self,
        filter: Filter,
        size: (usize, usize),
        window: &PlaneView<'_, T>,
        phases: (usize, usize),
        alternate_half: bool,
        out: &mut [i16],
    ) {
        T::predict_high_precision(self, filter, size, window, phases, alternate_half, out);
    }

    /// [`Dsp::hevc_weighted_prediction`] (uncentred predictions, the form
    /// VVC uses) at this depth: the codec folds the depth into `weights`
    /// (H.266 §8.5.6.6.2 default and BCW weighting, `shift2 = 15 −
    /// BitDepth`; §8.5.6.6.3 explicit, `shift1 = 14 − BitDepth`, offsets `<<
    /// (BitDepth − 8)`); the clip is this depth's.
    ///
    /// # Panics
    /// As [`Dsp::hevc_weighted_prediction`].
    pub fn hevc_weighted_prediction<T: Pixel>(
        self,
        weights: HevcWeights,
        a: &[i16],
        b: &[i16],
        dst: &mut PlaneViewMut<'_, T>,
    ) {
        T::hevc_weighted_prediction(self, weights, a, b, dst);
    }

    /// [`Dsp::vvc_geometric_blend`] at this depth (H.266 §8.5.7.3: `shift1 =
    /// Max(5, 17 − BitDepth)`, clipped to the depth).
    ///
    /// # Panics
    /// As [`Dsp::vvc_geometric_blend`].
    pub fn vvc_geometric_blend<T: Pixel>(
        self,
        mask: VvcGeometric,
        origin: (usize, usize),
        size: (usize, usize),
        chroma: bool,
        sources: (&[i16], &[i16]),
        out: &mut [T],
    ) {
        T::vvc_geometric_blend(self, mask, origin, size, chroma, sources, out);
    }

    /// [`Dsp::vvc_dmvr_bilinear`] at this depth (H.266 §8.5.3.2.2: `shift1 =
    /// BitDepth − 6`): 10-bit output at every depth, so [`Dsp::vvc_dmvr`]
    /// searches it unchanged.
    ///
    /// # Panics
    /// As [`Dsp::vvc_dmvr_bilinear`].
    pub fn vvc_dmvr_bilinear<T: Pixel>(
        self,
        size: (usize, usize),
        input: &PlaneView<'_, T>,
        phases: (usize, usize),
        out: &mut [i16],
    ) {
        T::vvc_dmvr_bilinear(self, size, input, phases, out);
    }

    /// [`Dsp::vvc_bdof`] at this depth (H.266 §8.5.6.5: `shift4 = Max(3, 15
    /// − BitDepth)`, clipped to the depth; the flow arithmetic on the 14-bit
    /// inputs is the same at every depth).
    ///
    /// # Panics
    /// As [`Dsp::vvc_bdof`].
    pub fn vvc_bdof<T: Pixel>(self, size: (usize, usize), inputs: [&[i16]; 2], out: &mut [T]) {
        T::vvc_bdof(self, size, inputs, out);
    }

    /// [`Dsp::vvc_dmvr`]: the search reads the 10-bit bilinear output of
    /// [`Self::vvc_dmvr_bilinear`], the same at every depth.
    ///
    /// # Panics
    /// As [`Dsp::vvc_dmvr`].
    #[must_use]
    pub fn vvc_dmvr(self, size: (usize, usize), inputs: [&[i16]; 2]) -> ([i32; 2], bool) {
        self.dsp.vvc_dmvr(size, inputs)
    }

    /// [`Dsp::vvc_prof_refine`]: 14-bit samples in and out at every depth up
    /// to 12 (`dILimit = 1 << Max(13, BitDepth + 1)` is 8192, H.266
    /// §8.5.6.4).
    ///
    /// # Panics
    /// As [`Dsp::vvc_prof_refine`].
    pub fn vvc_prof_refine(
        self,
        samples: &[i16; 36],
        corrections: &[[i8; 2]; 16],
        out: &mut [i16; 16],
    ) {
        self.dsp.vvc_prof_refine(samples, corrections, out);
    }

    /// [`Dsp::hevc_weighted_prediction_centered`] at this depth (the codec
    /// folds `14 − BitDepth` into `weights.shift`; the clip is this depth's).
    ///
    /// # Panics
    /// As [`Dsp::hevc_weighted_prediction`].
    pub fn hevc_weighted_prediction_centered<T: Pixel>(
        self,
        weights: HevcWeights,
        a: &[i16],
        b: &[i16],
        dst: &mut PlaneViewMut<'_, T>,
    ) {
        T::hevc_weighted_prediction_centered(self, weights, a, b, dst);
    }

    /// [`Dsp::hevc_intra_angular`] for any sample type.
    ///
    /// # Panics
    /// As [`Dsp::hevc_intra_angular`].
    pub fn hevc_intra_angular<T: Pixel>(self, refs: &[T], n: usize, angle: i32, out: &mut [T]) {
        T::hevc_intra_angular(self, refs, n, angle, out);
    }

    /// [`Dsp::hevc_intra_planar`] for any sample type.
    ///
    /// # Panics
    /// As [`Dsp::hevc_intra_planar`].
    pub fn hevc_intra_planar<T: Pixel>(self, edges: &PlanarEdges<'_, T>, n: usize, out: &mut [T]) {
        T::hevc_intra_planar(self, edges, n, out);
    }

    /// [`Dsp::hevc_inverse_add`] at this depth: the second inverse pass
    /// shifts by `20 − BitDepth` (HEVC §8.6.4.2) and the sum is clipped to
    /// the depth.
    ///
    /// # Panics
    /// As [`Dsp::hevc_inverse_add`].
    pub fn hevc_inverse_add<T: Pixel>(
        self,
        transform: HevcTransform,
        coeffs: &[i16],
        extent: (usize, usize),
        dst: &mut PlaneViewMut<'_, T>,
    ) {
        T::hevc_inverse_add(self, transform, coeffs, extent, dst);
    }

    /// [`Dsp::hevc_deblock_luma_edge`] at this depth, `tc` and `beta`
    /// already scaled to it (at 8 bits they fit `u8`).
    ///
    /// # Panics
    /// As [`Dsp::hevc_deblock_luma_edge`].
    pub fn hevc_deblock_luma_edge<T: Pixel>(
        self,
        dst: &mut PlaneViewMut<'_, T>,
        at: (usize, usize),
        vertical: bool,
        tc: [i16; 2],
        beta: i16,
    ) {
        T::hevc_deblock_luma_edge(self, dst, at, vertical, tc, beta);
    }

    /// [`Dsp::hevc_deblock_chroma_edge`] at this depth, `tc` scaled to it.
    ///
    /// # Panics
    /// As [`Dsp::hevc_deblock_chroma_edge`].
    pub fn hevc_deblock_chroma_edge<T: Pixel>(
        self,
        dst: &mut PlaneViewMut<'_, T>,
        at: (usize, usize),
        vertical: bool,
        lines: usize,
        tc: [i16; 4],
    ) {
        T::hevc_deblock_chroma_edge(self, dst, at, vertical, lines, tc);
    }

    /// [`Dsp::sao_band_apply_masked`] at this depth: bands of `2^(BitDepth
    /// − 5)` values, `SaoOffsetVal` offsets (scaled to the depth).
    ///
    /// # Panics
    /// As [`Dsp::sao_band_apply_masked`].
    pub fn sao_band_apply_masked<T: Pixel>(
        self,
        position: u8,
        offsets: [i16; 4],
        plane: &mut PlaneViewMut<'_, T>,
        rect: (usize, usize, usize, usize),
        exempt: &[bool],
    ) {
        T::sao_band_apply_masked(self, position, offsets, plane, rect, exempt);
    }

    /// [`Dsp::sao_edge_apply`] at this depth (`SaoOffsetVal` offsets).
    ///
    /// # Panics
    /// As [`Dsp::sao_edge_apply`].
    pub fn sao_edge_apply<T: Pixel>(
        self,
        class: u8,
        offsets: [i16; 4],
        input: &PlaneView<'_, T>,
        out: &mut PlaneViewMut<'_, T>,
        rect: (usize, usize, usize, usize),
    ) {
        T::sao_edge_apply(self, class, offsets, input, out, rect);
    }
}

#[cfg(test)]
mod av1_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod vp9_tests;
#[cfg(test)]
mod vvc_tests;
