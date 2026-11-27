//! Independent 8x8-cell SSE with NEON byte dot products.
//!
//! Byte squared differences follow libaom 3.15.3
//! `aom_dsp/arm/sse_neon_dotprod.c` (BSD-2-Clause and AOM patent licence;
//! `THIRD_PARTY.md`). Two neighboring cells share each 16-byte row load;
//! a final eight-byte cell packs two rows instead, without reading padding.

use core::arch::aarch64::{
    uint32x2_t, uint32x2x2_t, vabdq_u8, vaddq_u32, vaddvq_u32, vcombine_u8, vdupq_n_u16,
    vdupq_n_u32, vget_low_s32, vget_low_u32, vld1_u8, vld1q_u8, vld1q_u8_x4, vpadalq_u8,
    vpaddlq_s16, vpaddq_s32, vpaddq_u32, vreinterpret_u32_s32, vreinterpretq_s16_u16, vst1_u32,
    vst2_u32, vsubq_u16,
};

use super::Variance;
use crate::simd::neon_dot::udot;

/// Total SSE in raster order. Retain the four accumulators across as
/// many rows as can be reduced exactly in u32.
#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn plane_sse(a: &nc_common::PlaneView<'_>, b: &nc_common::PlaneView<'_>) -> u64 {
    const MAX_SAMPLES: usize = (u32::MAX / (u8::MAX as u32).pow(2)) as usize;
    let (width, height) = (a.width(), a.height());
    if width == 0 || height == 0 {
        return 0;
    }
    if width > MAX_SAMPLES {
        // SAFETY: the dispatch checked the ISA and equal plane dimensions.
        return unsafe { super::generic::plane_sse::<crate::simd::neon::I16x8>(a, b) };
    }
    let rows_per_sum = MAX_SAMPLES / width;
    let blocks = width / 64 * 64;
    let vectors = width / 16 * 16;
    let mut total = 0;
    for top in (0..height).step_by(rows_per_sum) {
        let mut s0 = vdupq_n_u32(0);
        let mut s1 = s0;
        let mut s2 = s0;
        let mut s3 = s0;
        // SAFETY: views of the same dimensions cover these rows and their
        // full width. Vector loads end at the last full vector of each row.
        unsafe {
            for y in top..(top + rows_per_sum).min(height) {
                let a = a.data().as_ptr().add(y * a.stride());
                let b = b.data().as_ptr().add(y * b.stride());
                for x in (0..blocks).step_by(64) {
                    let (a, b) = (vld1q_u8_x4(a.add(x)), vld1q_u8_x4(b.add(x)));
                    let d0 = vabdq_u8(a.0, b.0);
                    let d1 = vabdq_u8(a.1, b.1);
                    let d2 = vabdq_u8(a.2, b.2);
                    let d3 = vabdq_u8(a.3, b.3);
                    s0 = udot(s0, d0, d0);
                    s1 = udot(s1, d1, d1);
                    s2 = udot(s2, d2, d2);
                    s3 = udot(s3, d3, d3);
                }
                for x in (blocks..vectors).step_by(16) {
                    let difference = vabdq_u8(vld1q_u8(a.add(x)), vld1q_u8(b.add(x)));
                    s0 = udot(s0, difference, difference);
                }
                for x in vectors..width {
                    total += u64::from(a.add(x).read().abs_diff(b.add(x).read())).pow(2);
                }
            }
        }
        let sum = vaddq_u32(vaddq_u32(s0, s1), vaddq_u32(s2, s3));
        total += u64::from(vaddvq_u32(sum));
    }
    total
}

/// Dsp validates eight rows of `out.len()*8` bytes in both inputs, and
/// enables dotprod. Each output is a separate 8x8 cell's exact SSE.
#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn sse_8x8_row(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    out: &mut [u32],
) {
    let (pairs, tail) = out.as_chunks_mut::<2>();
    for (p, pair) in pairs.iter_mut().enumerate() {
        let x = 16 * p;
        // SAFETY: both adjacent 8x8 cells are in the validated rows.
        let (cells, _) = unsafe { two_cells::<false>(src, src_stride, pred, pred_stride, x) };
        // SAFETY: pair is exactly two writable, naturally aligned u32 values.
        unsafe { vst1_u32(pair.as_mut_ptr(), cells) };
    }
    if let [last] = tail {
        let x = 16 * pairs.len();
        let mut sum = vdupq_n_u32(0);
        for y in (0..8).step_by(2) {
            // SAFETY: each load reads exactly eight bytes of the final cell;
            // y is 0,2,4,6, so the paired following row is also validated.
            let (a, b) = unsafe {
                (
                    vcombine_u8(
                        vld1_u8(src.as_ptr().add(y * src_stride + x)),
                        vld1_u8(src.as_ptr().add((y + 1) * src_stride + x)),
                    ),
                    vcombine_u8(
                        vld1_u8(pred.as_ptr().add(y * pred_stride + x)),
                        vld1_u8(pred.as_ptr().add((y + 1) * pred_stride + x)),
                    ),
                )
            };
            let difference = vabdq_u8(a, b);
            sum = udot(sum, difference, difference);
        }
        *last = vaddvq_u32(sum);
    }
}

/// Variance of each cell using the same two-cell loads as the SSE-only entry.
#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn variance_8x8_row(
    src: &[u8],
    ss: usize,
    pred: &[u8],
    ps: usize,
    out: &mut [Variance],
) {
    let (pairs, tail) = out.as_chunks_mut::<2>();
    for (i, pair) in pairs.iter_mut().enumerate() {
        // SAFETY: both adjacent 8x8 cells are in the validated rows.
        let (squares, sums) = unsafe { two_cells::<true>(src, ss, pred, ps, 16 * i) };
        // SAFETY: the pair is C-layout [u32, i32, u32, i32].
        unsafe { vst2_u32(pair.as_mut_ptr().cast(), uint32x2x2_t(squares, sums)) };
    }
    if let [last] = tail {
        let x = 16 * pairs.len();
        // SAFETY: the last cell covers exactly eight bytes per row.
        *last = unsafe { super::neon_dotprod::variance_w8(&src[x..], ss, &pred[x..], ps, 8) };
    }
}

/// Each dot-product accumulator owns four columns. The word accumulators
/// own two columns and never exceed 2 * 8 * 255 before the signed subtract.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn two_cells<const WITH_SUM: bool>(
    src: &[u8],
    ss: usize,
    pred: &[u8],
    ps: usize,
    x: usize,
) -> (uint32x2_t, uint32x2_t) {
    let mut squares = vdupq_n_u32(0);
    let mut sum_src = vdupq_n_u16(0);
    let mut sum_pred = sum_src;
    for y in 0..8 {
        // SAFETY: the caller validates both complete cells on every row.
        let (a, b) = unsafe {
            (
                vld1q_u8(src.as_ptr().add(y * ss + x)),
                vld1q_u8(pred.as_ptr().add(y * ps + x)),
            )
        };
        let difference = vabdq_u8(a, b);
        squares = udot(squares, difference, difference);
        if WITH_SUM {
            sum_src = vpadalq_u8(sum_src, a);
            sum_pred = vpadalq_u8(sum_pred, b);
        }
    }
    let words = vreinterpretq_s16_u16(vsubq_u16(sum_src, sum_pred));
    let pairs = vpaddlq_s16(words);
    (
        vget_low_u32(vpaddq_u32(squares, squares)),
        vreinterpret_u32_s32(vget_low_s32(vpaddq_s32(pairs, pairs))),
    )
}
