//! Independent 8x8-cell SSE with NEON byte dot products.
//!
//! Byte squared differences follow libaom 3.15.3
//! `aom_dsp/arm/sse_neon_dotprod.c` (BSD-2-Clause and AOM patent licence;
//! `THIRD_PARTY.md`). Two neighboring cells share each 16-byte row load;
//! a final eight-byte cell packs two rows instead, without reading padding.

use core::arch::aarch64::{
    vabdq_u8, vaddvq_u32, vcombine_u8, vdupq_n_u32, vget_low_u32, vld1_u8, vld1q_u8, vpaddq_u32,
    vst1_u32,
};

use crate::simd::neon_dot::udot;

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
        let mut sum = vdupq_n_u32(0);
        for y in 0..8 {
            // SAFETY: the caller validates both complete cells on every row.
            let difference = unsafe {
                vabdq_u8(
                    vld1q_u8(src.as_ptr().add(y * src_stride + x)),
                    vld1q_u8(pred.as_ptr().add(y * pred_stride + x)),
                )
            };
            sum = udot(sum, difference, difference);
        }
        // UDOT lanes 0/1 belong to the first cell, 2/3 to the second.
        // Each cell is bounded by 64*255^2 < 2^22, requiring u32.
        let cells = vget_low_u32(vpaddq_u32(sum, sum));
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
