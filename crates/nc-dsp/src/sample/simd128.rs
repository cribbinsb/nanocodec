//! WebAssembly SIMD128 sample kernels.
//!
//! Same structure as the NEON versions: pairwise widening adds for sums and
//! box sums, widening multiplies for squares, 16 downscale outputs per
//! iteration with the scalar row function for the tail.
//!
//! Kernel contract: see the `sample` module docs.

use core::arch::wasm32::{
    u8x16_narrow_i16x8, u16x8_add, u16x8_extadd_pairwise_u8x16, u16x8_extmul_high_u8x16,
    u16x8_extmul_low_u8x16, u16x8_shr, u16x8_splat, u32x4_add, u32x4_extadd_pairwise_u16x8,
    u32x4_extract_lane, u32x4_splat, u64x2, v128, v128_load, v128_store,
};

use super::{Moments, SampleFns, scalar};

/// Outputs produced per iteration of the downscale loop.
const DOWNSCALE_STEP: usize = 16;

/// The SIMD128 kernel table (SIMD128 is a compile-time feature).
pub(crate) fn sample_fns() -> SampleFns {
    // The pair kernels stay scalar here: only the encoder's speed presets
    // use them, and the browser module decodes.
    SampleFns {
        moments_8x8_row: moments_row::<8>,
        moments_8x4_row: moments_row::<4>,
        downscale_2x,
        pair_mean_2x: scalar::pairs_2x::<false>,
        pair_pick_2x: scalar::pairs_2x::<true>,
    }
}

/// [`moments_block`] for each 8-wide, `ROWS`-tall block along a row.
#[target_feature(enable = "simd128")]
unsafe fn moments_row<const ROWS: usize>(src: &[u8], src_stride: usize, out: &mut [Moments]) {
    for (i, m) in out.iter_mut().enumerate() {
        // SAFETY: block i lies inside the row (kernel contract).
        *m = unsafe { moments_block::<ROWS>(&src[8 * i..], src_stride) };
    }
}

/// # Safety
///
/// Kernel contract (`sample` module docs).
#[target_feature(enable = "simd128")]
unsafe fn moments_block<const ROWS: usize>(src: &[u8], src_stride: usize) -> Moments {
    let mut sum = u16x8_splat(0);
    let mut sum_sq = u32x4_splat(0);
    for y in (0..ROWS).step_by(2) {
        // SAFETY: rows y and y + 1 (8 bytes each) are inside the block.
        let v = unsafe {
            let p = src.as_ptr();
            u64x2(
                p.add(y * src_stride).cast::<u64>().read_unaligned(),
                p.add((y + 1) * src_stride).cast::<u64>().read_unaligned(),
            )
        };
        sum = u16x8_add(sum, u16x8_extadd_pairwise_u8x16(v));
        // Squares are at most 255^2 = 65025, exact in u16.
        sum_sq = u32x4_add(
            sum_sq,
            u32x4_extadd_pairwise_u16x8(u16x8_extmul_low_u8x16(v, v)),
        );
        sum_sq = u32x4_add(
            sum_sq,
            u32x4_extadd_pairwise_u16x8(u16x8_extmul_high_u8x16(v, v)),
        );
    }
    Moments {
        sum: sum_lanes(u32x4_extadd_pairwise_u16x8(sum)),
        sum_sq: sum_lanes(sum_sq),
    }
}

/// # Safety
///
/// Kernel contract (`sample` module docs).
#[target_feature(enable = "simd128")]
unsafe fn downscale_2x(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    w: usize,
    h: usize,
) {
    let simd_width = w - w % DOWNSCALE_STEP;
    for y in 0..h {
        let top = &src[2 * y * src_stride..];
        let bottom = &src[(2 * y + 1) * src_stride..];
        let out = &mut dst[y * dst_stride..y * dst_stride + w];
        for x in (0..simd_width).step_by(DOWNSCALE_STEP) {
            // SAFETY: input bytes 2x .. 2x + 32 lie within the 2w-wide source
            // rows and outputs x .. x + 16 within the output row, since
            // x + 16 <= w (kernel contract).
            unsafe {
                let t = top.as_ptr().add(2 * x);
                let b = bottom.as_ptr().add(2 * x);
                let left = box_means(load(t), load(b));
                let right = box_means(load(t.add(16)), load(b.add(16)));
                v128_store(
                    out.as_mut_ptr().add(x).cast(),
                    u8x16_narrow_i16x8(left, right),
                );
            }
        }
        scalar::downscale_2x_row(
            &top[2 * simd_width..],
            &bottom[2 * simd_width..],
            &mut out[simd_width..],
        );
    }
}

/// # Safety
///
/// 16 bytes must be readable at `p`.
#[target_feature(enable = "simd128")]
#[inline]
unsafe fn load(p: *const u8) -> v128 {
    // SAFETY: guaranteed by the caller.
    unsafe { v128_load(p.cast()) }
}

/// Rounded means of eight 2x2 boxes from 16 bytes of each input row.
#[target_feature(enable = "simd128")]
#[inline]
fn box_means(top: v128, bottom: v128) -> v128 {
    let sum = u16x8_add(
        u16x8_extadd_pairwise_u8x16(top),
        u16x8_extadd_pairwise_u8x16(bottom),
    );
    u16x8_shr(u16x8_add(sum, u16x8_splat(2)), 2)
}

#[target_feature(enable = "simd128")]
#[inline]
fn sum_lanes(v: v128) -> u32 {
    u32x4_extract_lane::<0>(v)
        + u32x4_extract_lane::<1>(v)
        + u32x4_extract_lane::<2>(v)
        + u32x4_extract_lane::<3>(v)
}
