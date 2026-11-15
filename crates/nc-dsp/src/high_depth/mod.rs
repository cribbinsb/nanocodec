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

mod inverse;
mod vector;

use nc_common::{BitDepth, PlaneView, PlaneViewMut, Sample};

use crate::block::BlockSize;
use crate::codec_txfm::HevcTransform;
use crate::deblock::{HEVC_LINES, HevcEdge};
use crate::dispatch::{check_covers, check_covers_i16, hevc_edge_at, sao_offsets, sao_rect};
use crate::hevc_intra::PlanarEdges;
use crate::hevc_weight::HevcWeights;
use crate::mc::Filter;
use crate::sao::Rect;
use crate::{Dsp, Isa};

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
}

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
pub(crate) use {by_width, families};

/// The scalar `u16` instances of the MC references, by taps and width.
mod scalar {
    use nc_common::BitDepth;

    use crate::mc::SeparableTaps;
    use crate::mc::scalar::{hevc_bi_at, hevc_prep_at, separable_at};

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
        };
        match isa {
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 | Isa::Avx512 => {
                table.predict = vector::avx2::PREDICT;
                table.prep = vector::avx2::PREP;
                table.bi = vector::avx2::BI;
                table.hevc_weight = vector::avx2::WEIGHT;
            }
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => {
                table.predict = vector::neon::PREDICT;
                table.prep = vector::neon::PREP;
                table.bi = vector::neon::BI;
                table.hevc_weight = vector::neon::WEIGHT;
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
    for (row, line) in residual[..n * n].chunks_exact(n).enumerate() {
        for (s, &r) in dst[row * stride..row * stride + n].iter_mut().zip(line) {
            *s = depth.clip_to(s.to_i32() + i32::from(r));
        }
    }
}

/// A sample type with kernels: `u8` (the 8-bit [`Dsp`] methods) or `u16`
/// ([`HighFns`]). The decoders' generic code is written over `T: Pixel`
/// and calls the kernels through [`DepthKernels`]; these methods are the
/// routing behind it, one per kernel.
#[allow(clippy::too_many_arguments)]
pub trait Pixel: Sample {
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

    fn hevc_weighted_prediction_centered(
        kernels: DepthKernels<'_>,
        mut weights: HevcWeights,
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
        weights.round += i32::from(crate::mc::HEVC_PREP_OFFSET)
            * (i32::from(weights.w0) + i32::from(weights.w1));
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
mod tests;
