//! NEON residual and variance kernels: the generic kernels on 8 lanes (4 for
//! 4-wide blocks).
//!
//! Kernel contract: see the `diff` module docs.

use core::arch::aarch64::{
    vabd_u8, vabdq_u8, vaddvq_u32, vdupq_n_u32, vget_high_u8, vget_low_u8, vld1_u8, vld1q_u8,
    vmull_u8, vpadalq_u16,
};

use super::generic;
use super::{DiffFns, Variance, VarianceEnergy};
use crate::simd::neon::{I16x4, I16x8};

/// The NEON residual table (NEON is always present on `AArch64`).
pub(crate) fn diff_fns() -> DiffFns {
    let mut functions = DiffFns {
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
    };
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        functions.variance[2..].copy_from_slice(&[
            super::neon_dotprod::variance::<16>,
            super::neon_dotprod::variance::<32>,
            super::neon_dotprod::variance::<64>,
        ]);
        functions.variance_energy[2..].copy_from_slice(&[
            super::neon_dotprod::variance_energy::<16>,
            super::neon_dotprod::variance_energy::<32>,
            super::neon_dotprod::variance_energy::<64>,
        ]);
        functions.block_sums[2..].copy_from_slice(&[
            super::neon_dotprod::block_sums::<16>,
            super::neon_dotprod::block_sums::<32>,
            super::neon_dotprod::block_sums::<64>,
        ]);
        functions.sub_pixel_variance[2..].copy_from_slice(&[
            super::neon_dotprod::subpel::<16>,
            super::neon_dotprod::subpel::<32>,
            super::neon_dotprod::subpel::<64>,
        ]);
    }
    functions
}

/// Defines the four kernels of one width on one lane type.
macro_rules! width_kernels {
    ($lanes:ty, $width:literal, $residual:ident, $add:ident, $variance:ident, $energy:ident, $subpel:ident) => {
        #[target_feature(enable = "neon")]
        unsafe fn $residual<const SUM_ABS: bool>(
            src: &[u8],
            src_stride: usize,
            pred: &[u8],
            pred_stride: usize,
            out: &mut [i16],
            out_stride: usize,
            h: usize,
        ) -> u32 {
            // SAFETY: NEON is enabled for this function.
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

        #[target_feature(enable = "neon")]
        unsafe fn $add(
            dst: &mut [u8],
            dst_stride: usize,
            residual: &[i16],
            residual_stride: usize,
            h: usize,
        ) {
            // SAFETY: NEON is enabled for this function.
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

        #[target_feature(enable = "neon")]
        unsafe fn $variance(
            src: &[u8],
            src_stride: usize,
            pred: &[u8],
            pred_stride: usize,
            h: usize,
        ) -> Variance {
            // SAFETY: NEON is enabled for this function.
            unsafe { generic::variance::<$lanes, $width>(src, src_stride, pred, pred_stride, h) }
        }

        #[target_feature(enable = "neon")]
        unsafe fn $energy(
            src: &[u8],
            src_stride: usize,
            pred: &[u8],
            pred_stride: usize,
            h: usize,
        ) -> VarianceEnergy {
            // SAFETY: NEON is enabled for this function.
            unsafe {
                generic::variance_energy::<$lanes, $width>(src, src_stride, pred, pred_stride, h)
            }
        }

        #[target_feature(enable = "neon")]
        unsafe fn $subpel(
            rf: &[u8],
            ref_stride: usize,
            x_offset: usize,
            y_offset: usize,
            src: &[u8],
            src_stride: usize,
            h: usize,
        ) -> Variance {
            // SAFETY: NEON is enabled for this function.
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
        #[target_feature(enable = "neon")]
        unsafe fn $name(src: &[u8], src_stride: usize, h: usize) -> Variance {
            // SAFETY: NEON is enabled for this function.
            unsafe { generic::block_sums::<$lanes, $width>(src, src_stride, h) }
        }
    };
}

block_sums_kernel!(I16x4, 4, block_sums_w4);
block_sums_kernel!(I16x8, 8, block_sums_w8);
block_sums_kernel!(I16x8, 16, block_sums_w16);
block_sums_kernel!(I16x8, 32, block_sums_w32);
block_sums_kernel!(I16x8, 64, block_sums_w64);

/// SSE of each 8x8 block along a row: two blocks per 16-byte row load, a
/// single 8-byte one for an odd tail. `vabd` gives `|a - b|`, `vmull` its
/// exact square in u16 (at most 255^2), pairwise accumulated into u32.
#[target_feature(enable = "neon")]
unsafe fn sse_8x8_row(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    out: &mut [u32],
) {
    let (pairs, tail) = out.as_chunks_mut::<2>();
    for (p, pair) in pairs.iter_mut().enumerate() {
        let x = 16 * p;
        let mut acc = [vdupq_n_u32(0); 2];
        for y in 0..8 {
            // SAFETY: row y (16 bytes) of blocks 2p, 2p + 1 is inside both
            // (kernel contract).
            let d = unsafe {
                vabdq_u8(
                    vld1q_u8(src.as_ptr().add(y * src_stride + x)),
                    vld1q_u8(pred.as_ptr().add(y * pred_stride + x)),
                )
            };
            acc[0] = vpadalq_u16(acc[0], vmull_u8(vget_low_u8(d), vget_low_u8(d)));
            acc[1] = vpadalq_u16(acc[1], vmull_u8(vget_high_u8(d), vget_high_u8(d)));
        }
        *pair = [vaddvq_u32(acc[0]), vaddvq_u32(acc[1])];
    }
    if let [last] = tail {
        let x = 16 * pairs.len();
        let mut acc = vdupq_n_u32(0);
        for y in 0..8 {
            // SAFETY: row y (8 bytes) of the last block is inside both.
            let d = unsafe {
                vabd_u8(
                    vld1_u8(src.as_ptr().add(y * src_stride + x)),
                    vld1_u8(pred.as_ptr().add(y * pred_stride + x)),
                )
            };
            acc = vpadalq_u16(acc, vmull_u8(d, d));
        }
        *last = vaddvq_u32(acc);
    }
}
