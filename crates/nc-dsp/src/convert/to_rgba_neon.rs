//! NEON I420 to RGBA (`AArch64`).
//!
//! The same exact 16-bit split form of the display matrix as the other
//! kernels ([`RgbSplit`]), 16 pixels of each row per iteration. The 8
//! chroma samples of a row pair become offset-free 16-bit lanes in one
//! widening subtract, their terms are computed once and repeated for each
//! pixel pair by `zip`. Per pixel, one widening multiply-accumulate forms
//! `42 Y + bias` (the 16-bit result wraps, but the true value fits `i16`),
//! and each channel is an add plus one shift-right-and-accumulate. A
//! saturating narrow is the scalar clamp, and `vst4q_u8` interleaves RGBA
//! as it stores. The scalar row function takes the tail.
//!
//! The row-function structure follows libyuv's `I422ToARGBRow_NEON` (BSD
//! 3-clause, Copyright 2011 The `LibYuv` Project Authors; see
//! `THIRD_PARTY.md`); the arithmetic is our exact 8.8 form, not libyuv's
//! 6-bit one.
//!
//! Kernel contract: see the `convert` module docs.

use core::arch::aarch64::{
    int16x8_t, uint8x16x4_t, vaddq_s16, vcombine_u8, vdup_n_u8, vdupq_n_u8, vdupq_n_u16,
    vget_high_u8, vget_low_u8, vld1_u8, vld1q_u8, vmlal_u8, vmlaq_n_s16, vmovl_u8, vmulq_n_s16,
    vqmovun_s16, vreinterpretq_s16_u16, vsraq_n_s16, vst4q_u8, vsubl_u8, vsubq_s16, vzip1q_s16,
    vzip2q_s16,
};

use super::{ALPHA_OPAQUE, CHROMA_OFFSET, DisplayMatrix, RgbSplit, SHIFT, scalar};

/// Pixels of each row converted per iteration.
const STEP: usize = 16;

/// Chroma terms for 16 pixels of a row (8 chroma samples, each repeated for
/// its pixel pair), as the low and high 8 pixels.
struct ChromaTerms {
    /// Shifted remainders of R, G and B ([`RgbSplit`]).
    remainder: [[int16x8_t; 2]; 3],
    /// `Cr'` (added to R, subtracted from G).
    cr: [int16x8_t; 2],
    /// `2 Cb'` (added to B).
    cb2: [int16x8_t; 2],
}

/// # Safety
///
/// Kernel contract (`convert` module docs).
#[target_feature(enable = "neon")]
pub(crate) unsafe fn i420_to_rgba_rows(
    matrix: DisplayMatrix,
    luma: [&[u8]; 2],
    cb: &[u8],
    cr: &[u8],
    rgba: [&mut [u8]; 2],
) {
    let [top, bottom] = rgba;
    // SAFETY: forwarded kernel contract; a lone row has an empty partner.
    unsafe {
        if luma[1].is_empty() {
            rows(matrix, [luma[0]], cb, cr, [top]);
        } else {
            rows(matrix, luma, cb, cr, [top, bottom]);
        }
    }
}

/// `N` rows sharing one chroma row.
///
/// # Safety
///
/// Kernel contract (`convert` module docs).
#[inline]
#[target_feature(enable = "neon")]
unsafe fn rows<const N: usize>(
    matrix: DisplayMatrix,
    luma: [&[u8]; N],
    cb: &[u8],
    cr: &[u8],
    mut rgba: [&mut [u8]; N],
) {
    let split = matrix.split();
    let w = luma[0].len();
    let simd_width = w - w % STEP;
    for x in (0..simd_width).step_by(STEP) {
        // SAFETY: x + 16 <= w, so chroma bytes x / 2 .. x / 2 + 8 lie inside
        // w.div_ceil(2), luma bytes x .. x + 16 inside each w-sample row and
        // RGBA bytes 4x .. 4x + 64 inside each 4w-byte row (kernel contract).
        unsafe {
            let terms = chroma_terms(
                &split,
                vld1_u8(cb.as_ptr().add(x / 2)),
                vld1_u8(cr.as_ptr().add(x / 2)),
            );
            for (row, out) in luma.iter().zip(rgba.iter_mut()) {
                let samples = vld1q_u8(row.as_ptr().add(x));
                let low = rgb8(&split, vget_low_u8(samples), &terms, 0);
                let high = rgb8(&split, vget_high_u8(samples), &terms, 1);
                let channel = |c: usize| vcombine_u8(vqmovun_s16(low[c]), vqmovun_s16(high[c]));
                vst4q_u8(
                    out.as_mut_ptr().add(4 * x),
                    uint8x16x4_t(channel(0), channel(1), channel(2), vdupq_n_u8(ALPHA_OPAQUE)),
                );
            }
        }
    }
    let chroma = simd_width / 2;
    for (row, out) in luma.into_iter().zip(rgba) {
        scalar::i420_to_rgba_row(
            matrix,
            &row[simd_width..],
            &cb[chroma..],
            &cr[chroma..],
            &mut out[4 * simd_width..],
        );
    }
}

/// The chroma terms of 8 Cb and Cr samples for their 16 pixels.
#[inline]
#[target_feature(enable = "neon")]
fn chroma_terms(
    split: &RgbSplit,
    cb: core::arch::aarch64::uint8x8_t,
    cr: core::arch::aarch64::uint8x8_t,
) -> ChromaTerms {
    let offset = vdup_n_u8(CHROMA_OFFSET as u8);
    let u = vreinterpretq_s16_u16(vsubl_u8(cb, offset));
    let v = vreinterpretq_s16_u16(vsubl_u8(cr, offset));
    let red = vmulq_n_s16(v, split.r_cr);
    let green = vmlaq_n_s16(vmulq_n_s16(u, split.g_cb), v, split.g_cr);
    let blue = vmulq_n_s16(u, split.b_cb);
    let cb2 = vaddq_s16(u, u);
    ChromaTerms {
        remainder: [pair_up(red), pair_up(green), pair_up(blue)],
        cr: pair_up(v),
        cb2: pair_up(cb2),
    }
}

/// Each of 8 lanes repeated for its two pixels: lanes 0-3 and 4-7.
#[inline]
#[target_feature(enable = "neon")]
fn pair_up(c: int16x8_t) -> [int16x8_t; 2] {
    [vzip1q_s16(c, c), vzip2q_s16(c, c)]
}

/// R, G and B of 8 pixels in 16-bit lanes (unclamped) from their luma and
/// half `half` (0: low 8 pixels, 1: high) of the chroma terms.
#[inline]
#[target_feature(enable = "neon")]
fn rgb8(
    split: &RgbSplit,
    luma: core::arch::aarch64::uint8x8_t,
    terms: &ChromaTerms,
    half: usize,
) -> [int16x8_t; 3] {
    // y Y + bias: the 16-bit lanes wrap, but the true value fits i16; the
    // luma weight is non-negative (`RgbSplit::y`), as the unsigned multiply needs.
    let scaled = vreinterpretq_s16_u16(vmlal_u8(
        vdupq_n_u16(split.bias as u16),
        luma,
        vdup_n_u8(split.y as u8),
    ));
    let luma = vreinterpretq_s16_u16(vmovl_u8(luma));
    let linear = [
        vaddq_s16(luma, terms.cr[half]),
        vsubq_s16(luma, terms.cr[half]),
        vaddq_s16(luma, terms.cb2[half]),
    ];
    [0, 1, 2].map(|c| {
        vsraq_n_s16::<{ SHIFT as i32 }>(linear[c], vaddq_s16(scaled, terms.remainder[c][half]))
    })
}
