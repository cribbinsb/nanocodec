//! NEON prediction kernels: the generic kernels on 8 lanes (4 for 4-wide
//! blocks). AV1 regular and HEVC luma use optional dot-product kernels after runtime
//! feature detection; the remaining codecs keep the generic NEON kernels.
//!
//! Kernel contract: see the `mc` module docs.

use super::generic;
use super::{
    Av1Family, BiFn, CompoundFns, ConvolveFn, HevcChromaTaps, HevcLumaTaps, McFns, PrepFn, Regular,
    SeparableTaps, Sharp, Smooth, VvcAffineLumaTaps, VvcChromaTaps, VvcLumaTaps,
};
use crate::simd::neon::{I16x4, I16x8};

/// The NEON prediction table (NEON is always present on `AArch64`).
pub(crate) fn mc_fns() -> McFns {
    let mut functions = McFns {
        compound: CompoundFns {
            prep: [
                prep_family::<Regular>(),
                prep_family::<Smooth>(),
                prep_family::<Sharp>(),
            ],
            average: [
                average_w4,
                average_w8,
                average_w16,
                average_w32,
                average_w64,
            ],
            hevc_prep: [
                hevc_prep_family::<HevcLumaTaps, 8>(),
                hevc_prep_family::<HevcChromaTaps, 4>(),
            ],
            hevc_bi: [
                hevc_bi_family::<HevcLumaTaps, 8>(),
                hevc_bi_family::<HevcChromaTaps, 4>(),
            ],
        },
        av1: av1_family::<Regular>(),
        av1_smooth: av1_family::<Smooth>(),
        av1_sharp: av1_family::<Sharp>(),
        hevc_luma: separable_family::<HevcLumaTaps, 8>(),
        hevc_chroma: separable_family::<HevcChromaTaps, 4>(),
        h264_luma: [
            h264_luma_w4,
            h264_luma_w8,
            h264_luma_w16,
            h264_luma_w32,
            h264_luma_w64,
        ],
        h264_chroma: [
            h264_chroma_w4,
            h264_chroma_w8,
            h264_chroma_w16,
            h264_chroma_w32,
            h264_chroma_w64,
        ],
        vvc_luma: separable_family::<VvcLumaTaps, 8>(),
        vvc_affine_luma: separable_family::<VvcAffineLumaTaps, 8>(),
        vvc_chroma: separable_family::<VvcChromaTaps, 4>(),
        h264_phases: None,
        av1_mixed: None,
    };
    // H.264 luma at widths 16..64: byte-vector six-tap (`neon_h264`).
    functions.h264_luma[2..].copy_from_slice(&[
        super::neon_h264::h264_luma::<16>,
        super::neon_h264::h264_luma::<32>,
        super::neon_h264::h264_luma::<64>,
    ]);
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        functions.av1 = super::av1_neon::put_fns::<Regular>();
        functions.av1_smooth = super::av1_neon::put_fns::<Smooth>();
        functions.av1_sharp = super::av1_neon::put_fns::<Sharp>();
        functions.compound.prep = [
            super::av1_neon::prep_fns::<Regular>(),
            super::av1_neon::prep_fns::<Smooth>(),
            super::av1_neon::prep_fns::<Sharp>(),
        ];
        functions.hevc_luma = super::neon_dotprod::hevc_luma_fns();
        functions.av1_mixed = Some(super::av1_neon::mixed_fns());
    }
    functions
}

/// The kernels of separable table `T` by width class.
fn separable_family<T: SeparableTaps<N>, const N: usize>() -> [ConvolveFn; 5] {
    [
        separable_w4::<T, N>,
        separable_w8::<T, N>,
        separable_w16::<T, N>,
        separable_w32::<T, N>,
        separable_w64::<T, N>,
    ]
}

/// The AV1 kernels of filter family `F` by width class.
fn av1_family<F: Av1Family>() -> [ConvolveFn; 5] {
    [
        av1_w4::<F>,
        av1_w8::<F>,
        av1_w16::<F>,
        av1_w32::<F>,
        av1_w64::<F>,
    ]
}

/// The AV1 compound-precision kernels of filter family `F` by width class.
fn prep_family<F: Av1Family>() -> [PrepFn; 5] {
    [
        prep_w4::<F>,
        prep_w8::<F>,
        prep_w16::<F>,
        prep_w32::<F>,
        prep_w64::<F>,
    ]
}

/// HEVC compound-precision kernels by width class.
fn hevc_prep_family<T: SeparableTaps<N>, const N: usize>() -> [PrepFn; 5] {
    [
        hevc_prep_w4::<T, N>,
        hevc_prep_w8::<T, N>,
        hevc_prep_w16::<T, N>,
        hevc_prep_w32::<T, N>,
        hevc_prep_w64::<T, N>,
    ]
}

/// Defines the compound kernels of one width on one lane type.
macro_rules! compound_kernels {
    ($lanes:ty, $width:literal, $prep:ident, $average:ident, $hevc_prep:ident) => {
        #[target_feature(enable = "neon")]
        unsafe fn $prep<F: Av1Family>(
            src: &[u8],
            ss: usize,
            dst: &mut [i16],
            ds: usize,
            h: usize,
            fx: usize,
            fy: usize,
        ) {
            // SAFETY: NEON is enabled for this function.
            unsafe { generic::av1_prep::<$lanes, F>(src, ss, dst, ds, $width, h, fx, fy) }
        }

        #[target_feature(enable = "neon")]
        unsafe fn $hevc_prep<T: SeparableTaps<N>, const N: usize>(
            src: &[u8],
            ss: usize,
            dst: &mut [i16],
            ds: usize,
            h: usize,
            fx: usize,
            fy: usize,
        ) {
            let (tx, ty) = (&T::TABLE[fx], &T::TABLE[fy]);
            // SAFETY: NEON is enabled for this function.
            unsafe {
                generic::hevc_prep::<$lanes, N>(src, ss, dst, ds, $width, h, tx, ty, (fx, fy))
            }
        }

        #[target_feature(enable = "neon")]
        unsafe fn $average(p0: &[i16], p1: &[i16], ss: usize, dst: &mut [u8], ds: usize, h: usize) {
            // SAFETY: NEON is enabled for this function.
            unsafe { generic::av1_average::<$lanes>(p0, p1, ss, dst, ds, $width, h) }
        }
    };
}

compound_kernels!(I16x4, 4, prep_w4, average_w4, hevc_prep_w4);
compound_kernels!(I16x8, 8, prep_w8, average_w8, hevc_prep_w8);
compound_kernels!(I16x8, 16, prep_w16, average_w16, hevc_prep_w16);
compound_kernels!(I16x8, 32, prep_w32, average_w32, hevc_prep_w32);
compound_kernels!(I16x8, 64, prep_w64, average_w64, hevc_prep_w64);

/// HEVC default bi-prediction kernels by width class.
fn hevc_bi_family<T: SeparableTaps<N>, const N: usize>() -> [BiFn; 5] {
    [
        hevc_bi_w4::<T, N>,
        hevc_bi_w8::<T, N>,
        hevc_bi_w16::<T, N>,
        hevc_bi_w32::<T, N>,
        hevc_bi_w64::<T, N>,
    ]
}

/// Defines the HEVC default bi-prediction kernel of one width on one lane
/// type.
macro_rules! hevc_bi_kernel {
    ($lanes:ty, $width:literal, $name:ident) => {
        #[target_feature(enable = "neon")]
        unsafe fn $name<T: SeparableTaps<N>, const N: usize>(
            src: &[u8],
            ss: usize,
            first: &[i16],
            fs: usize,
            dst: &mut [u8],
            ds: usize,
            h: usize,
            fx: usize,
            fy: usize,
        ) {
            let taps = (&T::TABLE[fx], &T::TABLE[fy]);
            // SAFETY: NEON is enabled for this function.
            unsafe {
                generic::hevc_bi::<$lanes, N>(
                    src,
                    ss,
                    (first, fs),
                    dst,
                    ds,
                    ($width, h),
                    taps,
                    (fx, fy),
                )
            }
        }
    };
}

hevc_bi_kernel!(I16x4, 4, hevc_bi_w4);
hevc_bi_kernel!(I16x8, 8, hevc_bi_w8);
hevc_bi_kernel!(I16x8, 16, hevc_bi_w16);
hevc_bi_kernel!(I16x8, 32, hevc_bi_w32);
hevc_bi_kernel!(I16x8, 64, hevc_bi_w64);

/// Defines the five prediction kernels of one width on one lane type.
macro_rules! width_kernels {
    ($lanes:ty, $width:literal, $av1:ident, $separable:ident,
     $h264_luma:ident, $h264_chroma:ident) => {
        #[target_feature(enable = "neon")]
        unsafe fn $av1<F: Av1Family>(
            src: &[u8],
            ss: usize,
            dst: &mut [u8],
            ds: usize,
            h: usize,
            fx: usize,
            fy: usize,
        ) {
            // SAFETY: NEON is enabled for this function.
            unsafe { generic::av1_convolve::<$lanes, F>(src, ss, dst, ds, $width, h, fx, fy) }
        }

        #[target_feature(enable = "neon")]
        unsafe fn $separable<T: SeparableTaps<N>, const N: usize>(
            src: &[u8],
            ss: usize,
            dst: &mut [u8],
            ds: usize,
            h: usize,
            fx: usize,
            fy: usize,
        ) {
            let (tx, ty) = (&T::TABLE[fx], &T::TABLE[fy]);
            // SAFETY: NEON is enabled for this function.
            unsafe { generic::hevc::<$lanes, N>(src, ss, dst, ds, $width, h, tx, ty, (fx, fy)) }
        }

        #[target_feature(enable = "neon")]
        unsafe fn $h264_luma(
            src: &[u8],
            ss: usize,
            dst: &mut [u8],
            ds: usize,
            h: usize,
            fx: usize,
            fy: usize,
        ) {
            // SAFETY: NEON is enabled for this function.
            unsafe { generic::h264_luma::<$lanes>(src, ss, dst, ds, $width, h, fx, fy) }
        }

        #[target_feature(enable = "neon")]
        unsafe fn $h264_chroma(
            src: &[u8],
            ss: usize,
            dst: &mut [u8],
            ds: usize,
            h: usize,
            fx: usize,
            fy: usize,
        ) {
            // SAFETY: NEON is enabled for this function.
            unsafe { generic::h264_chroma::<$lanes>(src, ss, dst, ds, $width, h, fx, fy) }
        }
    };
}

width_kernels!(I16x4, 4, av1_w4, separable_w4, h264_luma_w4, h264_chroma_w4);
width_kernels!(I16x8, 8, av1_w8, separable_w8, h264_luma_w8, h264_chroma_w8);
width_kernels!(
    I16x8,
    16,
    av1_w16,
    separable_w16,
    h264_luma_w16,
    h264_chroma_w16
);
width_kernels!(
    I16x8,
    32,
    av1_w32,
    separable_w32,
    h264_luma_w32,
    h264_chroma_w32
);
width_kernels!(
    I16x8,
    64,
    av1_w64,
    separable_w64,
    h264_luma_w64,
    h264_chroma_w64
);
