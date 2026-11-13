//! NEON forward colour conversion with bounded unsigned 16-bit arithmetic.
//! `ld4` deinterleaves one sixteen-pixel row. Luma accumulates widening
//! `u8` products onto its rounding and offset (`vmlal_u8`), and one
//! narrowing shift gives the bytes. Chroma sums each 2x2 group with a
//! pairwise widening add and a pairwise widening accumulate, a rounding
//! shift forms the exact `(sum + 2) / 4` mean, and the matrix accumulates
//! onto its rounding and offset in wrapping 16-bit lanes (numerators stay
//! inside 0..65536, `convert` module docs) before one narrowing shift.
//! See the convert module for matrix bounds and contracts.
use super::{bt601_forward as matrix, scalar};
use core::arch::aarch64::{
    uint8x8_t, uint8x16_t, uint8x16x4_t, uint16x8_t, uint32x4_t, vcombine_u8, vdup_n_u8,
    vdupq_n_u16, vdupq_n_u32, vget_high_u8, vget_low_u8, vld1q_u8_x4, vld4q_u8, vmlal_u8,
    vmlaq_n_u16, vpadalq_u8, vpaddlq_u8, vreinterpretq_u8_u16, vreinterpretq_u8_u32,
    vreinterpretq_u16_u32, vrshrq_n_u16, vshrn_n_u16, vst1_u8, vst1q_u8, vuzp1q_u16, vuzp2q_u8,
};

/// `ROUND + (offset << SHIFT)` as the starting value of a 16-bit sum.
fn bias(offset: i32) -> u16 {
    (matrix::ROUND + (offset << matrix::SHIFT)) as u16
}

/// Luma bytes of eight pixels from their R, G and B bytes.
#[inline]
#[target_feature(enable = "neon")]
fn luma8(r: uint8x8_t, g: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    let [wr, wg, wb] = matrix::Y.map(|w| vdup_n_u8(w as u8));
    let sum = vdupq_n_u16(bias(matrix::LUMA_OFFSET));
    let sum = vmlal_u8(vmlal_u8(vmlal_u8(sum, r, wr), g, wg), b, wb);
    vshrn_n_u16::<8>(sum)
}

#[inline]
#[target_feature(enable = "neon")]
fn luma16(p: uint8x16x4_t) -> uint8x16_t {
    vcombine_u8(
        luma8(vget_low_u8(p.0), vget_low_u8(p.1), vget_low_u8(p.2)),
        luma8(vget_high_u8(p.0), vget_high_u8(p.1), vget_high_u8(p.2)),
    )
}

/// Rounded means of the eight 2x2 groups of one channel's two rows.
#[inline]
#[target_feature(enable = "neon")]
fn mean(a: uint8x16_t, b: uint8x16_t) -> uint16x8_t {
    vrshrq_n_u16::<2>(vpadalq_u8(vpaddlq_u8(a), b))
}

/// Cb or Cr bytes of eight groups: weights may be negative (wrapping
/// 16-bit products, exact because every numerator fits `u16`).
#[inline]
#[target_feature(enable = "neon")]
fn chroma8([r, g, b]: [uint16x8_t; 3], weights: [i32; 3]) -> uint8x8_t {
    let sum = vdupq_n_u16(bias(matrix::CHROMA_OFFSET));
    let sum = vmlaq_n_u16(sum, r, weights[0] as u16);
    let sum = vmlaq_n_u16(sum, g, weights[1] as u16);
    vshrn_n_u16::<8>(vmlaq_n_u16(sum, b, weights[2] as u16))
}

/// # Safety
///
/// Kernel contract (`convert` module docs).
#[allow(clippy::too_many_arguments, reason = "matches RgbaToI420Fn")]
#[target_feature(enable = "neon")]
pub(super) unsafe fn rgba_to_i420(
    rgba: &[u8],
    luma: &mut [u8],
    luma_stride: usize,
    cb: &mut [u8],
    cr: &mut [u8],
    chroma_stride: usize,
    width: usize,
    height: usize,
) {
    let planes = Planes {
        luma,
        luma_stride,
        cb,
        cr,
        chroma_stride,
    };
    // SAFETY: forwarded kernel contract.
    unsafe { convert(rgba, planes, width, height, |_, pixels| luma16(pixels)) };
}

/// [`rgba_to_i420`] with luma from unsigned dot products: `udot` sums
/// `66 R + 129 G + 25 B + 0 A` of each packed pixel (all weights fit a
/// byte) onto its rounding and offset in one 32-bit lane, whose byte 1 is
/// then the luma sample (the sum stays below 65536), gathered by two
/// unzips. This replaces the twelve widening multiplies per 16 pixels.
///
/// # Safety
///
/// Kernel contract (`convert` module docs); the CPU must support the
/// dot-product extension.
#[allow(clippy::too_many_arguments, reason = "matches RgbaToI420Fn")]
#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn rgba_to_i420_dotprod(
    rgba: &[u8],
    luma: &mut [u8],
    luma_stride: usize,
    cb: &mut [u8],
    cr: &mut [u8],
    chroma_stride: usize,
    width: usize,
    height: usize,
) {
    let planes = Planes {
        luma,
        luma_stride,
        cb,
        cr,
        chroma_stride,
    };
    let [wr, wg, wb] = matrix::Y.map(|w| w as u32);
    let weights = vreinterpretq_u8_u32(vdupq_n_u32(wr | (wg << 8) | (wb << 16)));
    let start = vdupq_n_u32(u32::from(bias(matrix::LUMA_OFFSET)));
    // SAFETY: forwarded kernel contract; `udot` needs the dot-product
    // extension, which the caller guarantees.
    unsafe {
        convert(rgba, planes, width, height, |src, _| {
            let raw = vld1q_u8_x4(src);
            let sums = [raw.0, raw.1, raw.2, raw.3].map(|quad| udot(start, quad, weights));
            let low = vuzp1q_u16(
                vreinterpretq_u16_u32(sums[0]),
                vreinterpretq_u16_u32(sums[1]),
            );
            let high = vuzp1q_u16(
                vreinterpretq_u16_u32(sums[2]),
                vreinterpretq_u16_u32(sums[3]),
            );
            vuzp2q_u8(vreinterpretq_u8_u16(low), vreinterpretq_u8_u16(high))
        });
    }
}

/// `sum + a . b` per 32-bit lane over unsigned bytes. Rust's unsigned-dot
/// intrinsic postdates the crate's minimum Rust version (as for the signed
/// one in `mc::neon_dotprod`).
#[inline]
#[target_feature(enable = "neon,dotprod")]
fn udot(mut sum: uint32x4_t, a: uint8x16_t, b: uint8x16_t) -> uint32x4_t {
    // SAFETY: dotprod is enabled; register-only unsigned byte products
    // accumulate into four u32 lanes. No memory, stack or flags are touched.
    unsafe {
        core::arch::asm!(
            "udot {sum:v}.4s, {a:v}.16b, {b:v}.16b",
            sum = inout(vreg) sum,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    sum
}

/// The output planes of one conversion.
struct Planes<'a> {
    luma: &'a mut [u8],
    luma_stride: usize,
    cb: &'a mut [u8],
    cr: &'a mut [u8],
    chroma_stride: usize,
}

/// The row-pair loop shared by both kernels; `luma16(src, pixels)` gives
/// the luma of the 16 packed pixels at `src` (also deinterleaved in
/// `pixels`). Inlined into each caller, so it takes that caller's target
/// features.
///
/// # Safety
///
/// Kernel contract (`convert` module docs); `luma16` must be safe to call
/// for 16 readable packed pixels.
#[allow(
    clippy::inline_always,
    reason = "each kernel needs its own copy compiled with its target features"
)]
#[inline(always)]
unsafe fn convert(
    rgba: &[u8],
    planes: Planes<'_>,
    width: usize,
    height: usize,
    luma16: impl Fn(*const u8, uint8x16x4_t) -> uint8x16_t,
) {
    let Planes {
        luma,
        luma_stride,
        cb,
        cr,
        chroma_stride,
    } = planes;
    let vector_width = width / 16 * 16;
    for row in 0..height / 2 {
        let rows = &rgba[8 * width * row..][..8 * width];
        for x in (0..vector_width).step_by(16) {
            // SAFETY: this full sixteen-pixel group lies in both input rows and
            // both luma rows; its eight chroma outputs lie inside the chroma row.
            unsafe {
                let (top, bottom) = (
                    rows.as_ptr().add(4 * x),
                    rows.as_ptr().add(4 * width + 4 * x),
                );
                let upper = vld4q_u8(top);
                let lower = vld4q_u8(bottom);
                vst1q_u8(
                    luma.as_mut_ptr().add(2 * row * luma_stride + x),
                    luma16(top, upper),
                );
                vst1q_u8(
                    luma.as_mut_ptr().add((2 * row + 1) * luma_stride + x),
                    luma16(bottom, lower),
                );
                let means = [
                    mean(upper.0, lower.0),
                    mean(upper.1, lower.1),
                    mean(upper.2, lower.2),
                ];
                vst1_u8(
                    cb.as_mut_ptr().add(row * chroma_stride + x / 2),
                    chroma8(means, matrix::CB),
                );
                vst1_u8(
                    cr.as_mut_ptr().add(row * chroma_stride + x / 2),
                    chroma8(means, matrix::CR),
                );
            }
        }
        scalar::rgba_to_luma_row(
            &rows[4 * vector_width..4 * width],
            &mut luma[2 * row * luma_stride + vector_width..2 * row * luma_stride + width],
        );
        scalar::rgba_to_luma_row(
            &rows[4 * width + 4 * vector_width..],
            &mut luma
                [(2 * row + 1) * luma_stride + vector_width..(2 * row + 1) * luma_stride + width],
        );
        scalar::rgba_to_chroma_row(
            rows,
            vector_width / 2,
            &mut cb[row * chroma_stride + vector_width / 2..row * chroma_stride + width / 2],
            &mut cr[row * chroma_stride + vector_width / 2..row * chroma_stride + width / 2],
        );
    }
    if height % 2 == 1 {
        scalar::rgba_to_luma_row(
            &rgba[4 * width * (height - 1)..],
            &mut luma[(height - 1) * luma_stride..(height - 1) * luma_stride + width],
        );
    }
}
