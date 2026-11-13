//! WebAssembly SIMD128 prediction kernels: the generic kernels on 8 lanes
//! (4 for 4-wide blocks), with AV1 in `av1_simd128` and the HEVC / H.266
//! separable filters in `separable_simd128`. SIMD128 is enabled for the
//! whole build.
//!
//! Kernel contract: see the `mc` module docs.

use super::generic::{self, h264_chroma_classes, h264_luma_phases};
use super::{
    BiFn, CompoundFns, H264_CHROMA_BOTH, H264_CHROMA_COPY, H264_CHROMA_HORIZONTAL,
    H264_CHROMA_SHAPES, H264_CHROMA_VERTICAL, H264_LUMA_SHAPES, H264PhaseFns, HevcChromaTaps,
    HevcLumaTaps, McFns, PrepFn, Regular, SeparableTaps, Sharp, Smooth, VvcAffineLumaTaps,
    VvcChromaTaps, VvcLumaTaps,
};
use crate::simd::I16Lanes;
use crate::simd::simd128::{I16x4, I16x8};

/// The SIMD128 prediction table.
pub(crate) fn mc_fns() -> McFns {
    McFns {
        compound: CompoundFns {
            prep: [
                super::av1_simd128::prep_family::<Regular>(),
                super::av1_simd128::prep_family::<Smooth>(),
                super::av1_simd128::prep_family::<Sharp>(),
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
        av1: super::av1_simd128::family::<Regular>(),
        av1_smooth: super::av1_simd128::family::<Smooth>(),
        av1_sharp: super::av1_simd128::family::<Sharp>(),
        hevc_luma: super::separable_simd128::family::<HevcLumaTaps, 8, 4>(),
        hevc_chroma: super::separable_simd128::family::<HevcChromaTaps, 4, 2>(),
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
        vvc_luma: super::separable_simd128::family::<VvcLumaTaps, 8, 4>(),
        vvc_affine_luma: super::separable_simd128::family::<VvcAffineLumaTaps, 8, 4>(),
        vvc_chroma: super::separable_simd128::family::<VvcChromaTaps, 4, 2>(),
        h264_phases: Some(h264_phase_fns()),
        av1_mixed: Some(super::av1_simd128::mixed_fns()),
    }
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
    ($lanes:ty, $width:literal, $average:ident, $hevc_prep:ident) => {
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
            // SAFETY: SIMD128 is enabled for the whole build.
            unsafe {
                generic::hevc_prep::<$lanes, N>(src, ss, dst, ds, $width, h, tx, ty, (fx, fy))
            }
        }

        unsafe fn $average(p0: &[i16], p1: &[i16], ss: usize, dst: &mut [u8], ds: usize, h: usize) {
            // SAFETY: SIMD128 is enabled for the whole build.
            unsafe { generic::av1_average::<$lanes>(p0, p1, ss, dst, ds, $width, h) }
        }
    };
}

compound_kernels!(I16x4, 4, average_w4, hevc_prep_w4);
compound_kernels!(I16x8, 8, average_w8, hevc_prep_w8);
compound_kernels!(I16x8, 16, average_w16, hevc_prep_w16);
compound_kernels!(I16x8, 32, average_w32, hevc_prep_w32);
compound_kernels!(I16x8, 64, average_w64, hevc_prep_w64);

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
            // SAFETY: SIMD128 is enabled for the whole build.
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
    ($lanes:ty, $width:literal, $h264_luma:ident, $h264_chroma:ident) => {
        unsafe fn $h264_luma(
            src: &[u8],
            ss: usize,
            dst: &mut [u8],
            ds: usize,
            h: usize,
            fx: usize,
            fy: usize,
        ) {
            // SAFETY: SIMD128 is enabled for the whole build.
            unsafe { generic::h264_luma::<$lanes>(src, ss, dst, ds, $width, h, fx, fy) }
        }

        unsafe fn $h264_chroma(
            src: &[u8],
            ss: usize,
            dst: &mut [u8],
            ds: usize,
            h: usize,
            fx: usize,
            fy: usize,
        ) {
            // SAFETY: SIMD128 is enabled for the whole build.
            unsafe { generic::h264_chroma::<$lanes>(src, ss, dst, ds, $width, h, fx, fy) }
        }
    };
}

width_kernels!(I16x4, 4, h264_luma_w4, h264_chroma_w4);
width_kernels!(I16x8, 8, h264_luma_w8, h264_chroma_w8);
width_kernels!(I16x8, 16, h264_luma_w16, h264_chroma_w16);
width_kernels!(I16x8, 32, h264_luma_w32, h264_chroma_w32);
width_kernels!(I16x8, 64, h264_luma_w64, h264_chroma_w64);

/// The SIMD128 H.264 kernels per shape and phase: shapes 16 wide on
/// `I16x8`, 8 wide on `I16x8`, 4 wide on `I16x4`, as the width kernels.
fn h264_phase_fns() -> H264PhaseFns {
    H264PhaseFns {
        luma: [
            h264_luma_phases!(h264_luma_phase, I16x8, 0),
            h264_luma_phases!(h264_luma_phase, I16x8, 1),
            h264_luma_phases!(h264_luma_phase, I16x8, 2),
            h264_luma_phases!(h264_luma_phase, I16x8, 3),
            h264_luma_phases!(h264_luma_phase, I16x4, 4),
        ],
        chroma: [
            h264_chroma_classes!(h264_chroma_class, I16x8, 0),
            h264_chroma_classes!(h264_chroma_class, I16x8, 1),
            h264_chroma_classes!(h264_chroma_class, I16x4, 2),
            h264_chroma_classes!(h264_chroma_class, I16x4, 3),
        ],
    }
}

/// H.264 luma of a `W`x`H` block at quarter-sample phase (`FX`, `FY`):
/// the generic kernel with every branch on size and phase resolved at
/// compile time (the `h`, `fx` and `fy` arguments equal the constants).
unsafe fn h264_luma_phase<
    V: I16Lanes,
    const W: usize,
    const H: usize,
    const FX: usize,
    const FY: usize,
>(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    h: usize,
    fx: usize,
    fy: usize,
) {
    debug_assert_eq!((h, fx, fy), (H, FX, FY));
    // SAFETY: SIMD128 is enabled for the whole build.
    unsafe { generic::h264_luma::<V>(src, ss, dst, ds, W, H, FX, FY) }
}

/// H.264 chroma of a `W`x`H` block whose offsets are in class `CLASS`
/// (`H264_CHROMA_*`); the offsets themselves stay arguments (the weights).
unsafe fn h264_chroma_class<V: I16Lanes, const W: usize, const H: usize, const CLASS: usize>(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    h: usize,
    fx: usize,
    fy: usize,
) {
    debug_assert_eq!(h, H);
    // SAFETY: SIMD128 is enabled for the whole build.
    unsafe { generic::h264_chroma_of_class::<V>(src, ss, dst, ds, W, H, fx, fy, CLASS) }
}
