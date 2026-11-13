//! WebAssembly SIMD128 residual and variance kernels: the generic kernels on
//! 8 lanes (4 for 4-wide blocks). SIMD128 is enabled for the whole build.
//!
//! Kernel contract: see the `diff` module docs.

use core::arch::wasm32::{
    i32x4_add, i32x4_dot_i16x8, i32x4_extract_lane, i32x4_splat, u8x16_max, u8x16_min, u8x16_sub,
    u16x8_extend_high_u8x16, u16x8_extend_low_u8x16, u64x2,
};

use super::generic;
use super::{DiffFns, Variance, VarianceEnergy};
use crate::simd::simd128::{I16x4, I16x8};

/// The SIMD128 residual table.
pub(crate) fn diff_fns() -> DiffFns {
    DiffFns {
        residual: [
            residual_w4::<false>,
            residual_w8::<false>,
            residual_w16::<false>,
            residual_w32::<false>,
            residual_w64::<false>,
        ],
        residual_sad: [
            residual_w4::<true>,
            residual_w8::<true>,
            residual_w16::<true>,
            residual_w32::<true>,
            residual_w64::<true>,
        ],
        add_residual: [
            add_residual_w4,
            add_residual_w8,
            add_residual_w16,
            add_residual_w32,
            add_residual_w64,
        ],
        variance: [
            variance_w4,
            variance_w8,
            variance_w16,
            variance_w32,
            variance_w64,
        ],
        variance_energy: [
            variance_energy_w4,
            variance_energy_w8,
            variance_energy_w16,
            variance_energy_w32,
            variance_energy_w64,
        ],
        block_sums: [
            block_sums_w4,
            block_sums_w8,
            block_sums_w16,
            block_sums_w32,
            block_sums_w64,
        ],
        sub_pixel_variance: [
            sub_pixel_variance_w4,
            sub_pixel_variance_w8,
            sub_pixel_variance_w16,
            sub_pixel_variance_w32,
            sub_pixel_variance_w64,
        ],
        sse_8x8_row,
    }
}

/// Defines the four kernels of one width on one lane type.
macro_rules! width_kernels {
    ($lanes:ty, $width:literal, $residual:ident, $add:ident, $variance:ident, $energy:ident, $subpel:ident) => {
        unsafe fn $residual<const SUM_ABS: bool>(
            src: &[u8],
            src_stride: usize,
            pred: &[u8],
            pred_stride: usize,
            out: &mut [i16],
            out_stride: usize,
            h: usize,
        ) -> u32 {
            // SAFETY: SIMD128 is enabled for the whole build.
            unsafe {
                generic::residual::<$lanes, $width, SUM_ABS>(
                    src,
                    src_stride,
                    pred,
                    pred_stride,
                    out,
                    out_stride,
                    h,
                )
            }
        }

        unsafe fn $add(
            dst: &mut [u8],
            dst_stride: usize,
            residual: &[i16],
            residual_stride: usize,
            h: usize,
        ) {
            // SAFETY: SIMD128 is enabled for the whole build.
            unsafe {
                generic::add_residual::<$lanes, $width>(
                    dst,
                    dst_stride,
                    residual,
                    residual_stride,
                    h,
                );
            }
        }

        unsafe fn $variance(
            src: &[u8],
            src_stride: usize,
            pred: &[u8],
            pred_stride: usize,
            h: usize,
        ) -> Variance {
            // SAFETY: SIMD128 is enabled for the whole build.
            unsafe { generic::variance::<$lanes, $width>(src, src_stride, pred, pred_stride, h) }
        }

        unsafe fn $energy(
            src: &[u8],
            src_stride: usize,
            pred: &[u8],
            pred_stride: usize,
            h: usize,
        ) -> VarianceEnergy {
            // SAFETY: SIMD128 is enabled for the whole build.
            unsafe {
                generic::variance_energy::<$lanes, $width>(src, src_stride, pred, pred_stride, h)
            }
        }

        unsafe fn $subpel(
            rf: &[u8],
            ref_stride: usize,
            x_offset: usize,
            y_offset: usize,
            src: &[u8],
            src_stride: usize,
            h: usize,
        ) -> Variance {
            // SAFETY: SIMD128 is enabled for the whole build.
            unsafe {
                generic::sub_pixel_variance::<$lanes, $width>(
                    rf, ref_stride, x_offset, y_offset, src, src_stride, h,
                )
            }
        }
    };
}

width_kernels!(
    I16x4,
    4,
    residual_w4,
    add_residual_w4,
    variance_w4,
    variance_energy_w4,
    sub_pixel_variance_w4
);
width_kernels!(
    I16x8,
    8,
    residual_w8,
    add_residual_w8,
    variance_w8,
    variance_energy_w8,
    sub_pixel_variance_w8
);
width_kernels!(
    I16x8,
    16,
    residual_w16,
    add_residual_w16,
    variance_w16,
    variance_energy_w16,
    sub_pixel_variance_w16
);
width_kernels!(
    I16x8,
    32,
    residual_w32,
    add_residual_w32,
    variance_w32,
    variance_energy_w32,
    sub_pixel_variance_w32
);
width_kernels!(
    I16x8,
    64,
    residual_w64,
    add_residual_w64,
    variance_w64,
    variance_energy_w64,
    sub_pixel_variance_w64
);

/// Defines one width's block-sums kernel on a lane type.
macro_rules! block_sums_kernel {
    ($lanes:ty, $width:literal, $name:ident) => {
        unsafe fn $name(src: &[u8], src_stride: usize, h: usize) -> Variance {
            // SAFETY: SIMD128 is enabled for the whole build.
            unsafe { generic::block_sums::<$lanes, $width>(src, src_stride, h) }
        }
    };
}

block_sums_kernel!(I16x4, 4, block_sums_w4);
block_sums_kernel!(I16x8, 8, block_sums_w8);
block_sums_kernel!(I16x8, 16, block_sums_w16);
block_sums_kernel!(I16x8, 32, block_sums_w32);
block_sums_kernel!(I16x8, 64, block_sums_w64);

/// SSE of each 8x8 block along a row, two rows of a block per 16-byte
/// vector: `|a - b|` from max - min, widened to 16 bits and squared and
/// pair-summed by the i16 dot product (at most 2 * 255^2 per i32 lane).
#[target_feature(enable = "simd128")]
unsafe fn sse_8x8_row(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    out: &mut [u32],
) {
    let rows = |data: &[u8], stride: usize, y: usize| {
        // SAFETY: rows y and y + 1 (8 bytes each) are inside the block
        // (kernel contract).
        unsafe {
            let p = data.as_ptr();
            u64x2(
                p.add(y * stride).cast::<u64>().read_unaligned(),
                p.add((y + 1) * stride).cast::<u64>().read_unaligned(),
            )
        }
    };
    for (i, sse) in out.iter_mut().enumerate() {
        let (src, pred) = (&src[8 * i..], &pred[8 * i..]);
        let mut acc = i32x4_splat(0);
        for y in (0..8).step_by(2) {
            let (a, b) = (rows(src, src_stride, y), rows(pred, pred_stride, y));
            let d = u8x16_sub(u8x16_max(a, b), u8x16_min(a, b));
            for half in [u16x8_extend_low_u8x16(d), u16x8_extend_high_u8x16(d)] {
                acc = i32x4_add(acc, i32x4_dot_i16x8(half, half));
            }
        }
        let lanes = [
            i32x4_extract_lane::<0>(acc),
            i32x4_extract_lane::<1>(acc),
            i32x4_extract_lane::<2>(acc),
            i32x4_extract_lane::<3>(acc),
        ];
        *sse = lanes.iter().sum::<i32>() as u32;
    }
}
