//! WebAssembly SIMD128 I420 to RGBA: the browser demo's CPU display path.
//!
//! Works in 16-bit lanes with the exact split form of the BT.601 inverse
//! ([`bt601_split`]): per pixel a widening load and one multiply (`42 Y`),
//! and per channel an add, an arithmetic shift and two adds. The chroma
//! terms of a 2x2 group (with `BIAS` folded in) are computed once, at chroma
//! resolution, for both rows of a row pair, and widened to pixel pairs by
//! lane shuffles. A saturating
//! narrow to `u8` is the scalar clamp; byte and 16-bit interleaves (each a
//! single `punpck` in V8 on x86, `zip` on Arm) form RGBA. 16 pixels of each
//! row per iteration; the scalar row function takes the tail.
//!
//! The row-function structure (chroma shared by a row pair, channels kept
//! in 16-bit lanes until one saturating pack) follows libyuv's
//! `I422ToARGBRow` family (BSD 3-clause, Copyright 2011 The `LibYuv` Project
//! Authors; see `THIRD_PARTY.md`); the arithmetic is our exact 8.8 form,
//! not libyuv's 6-bit one.
//!
//! Kernel contract: see the `convert` module docs.

use core::arch::wasm32::{
    i8x16_shuffle, i16x8_add, i16x8_mul, i16x8_shl, i16x8_shr, i16x8_shuffle, i16x8_splat,
    i16x8_sub, u8x16_narrow_i16x8, u8x16_splat, u16x8_load_extend_u8x8, v128, v128_store,
};

use super::{ALPHA_OPAQUE, bt601, bt601_split, scalar};

/// Pixels of each row converted per iteration.
const STEP: usize = 16;

/// Chroma terms for 16 pixels of a row (8 chroma samples, each repeated for
/// its pixel pair), as the low and high 8 pixels.
struct ChromaTerms {
    /// Remainders of R, G and B without the luma term, `BIAS` included
    /// (`bt601_split`).
    remainder: [[v128; 2]; 3],
    /// `Cr'` (added to R, subtracted from G).
    cr: [v128; 2],
    /// `2 Cb'` (added to B).
    cb2: [v128; 2],
}

/// # Safety
///
/// Kernel contract (`convert` module docs).
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn i420_to_rgba_rows(
    luma: [&[u8]; 2],
    cb: &[u8],
    cr: &[u8],
    rgba: [&mut [u8]; 2],
) {
    let [top, bottom] = rgba;
    // SAFETY: forwarded kernel contract; a lone row has an empty partner.
    unsafe {
        if luma[1].is_empty() {
            rows([luma[0]], cb, cr, [top]);
        } else {
            rows(luma, cb, cr, [top, bottom]);
        }
    }
}

/// `N` rows sharing one chroma row.
///
/// # Safety
///
/// Kernel contract (`convert` module docs).
#[inline]
#[target_feature(enable = "simd128")]
unsafe fn rows<const N: usize>(luma: [&[u8]; N], cb: &[u8], cr: &[u8], mut rgba: [&mut [u8]; N]) {
    let w = luma[0].len();
    let simd_width = w - w % STEP;
    for x in (0..simd_width).step_by(STEP) {
        // SAFETY: x + 16 <= w, so chroma bytes x / 2 .. x / 2 + 8 lie inside
        // w.div_ceil(2), luma bytes x .. x + 16 inside each w-sample row and
        // RGBA bytes 4x .. 4x + 64 inside each 4w-byte row (kernel contract).
        unsafe {
            let terms = chroma_terms(
                u16x8_load_extend_u8x8(cb.as_ptr().add(x / 2)),
                u16x8_load_extend_u8x8(cr.as_ptr().add(x / 2)),
            );
            for (row, out) in luma.iter().zip(rgba.iter_mut()) {
                let samples = row.as_ptr().add(x);
                let low = rgb8(u16x8_load_extend_u8x8(samples), &terms, 0);
                let high = rgb8(u16x8_load_extend_u8x8(samples.add(8)), &terms, 1);
                store_rgba(out.as_mut_ptr().add(4 * x), low, high);
            }
        }
    }
    let chroma = simd_width / 2;
    for (row, out) in luma.into_iter().zip(rgba) {
        scalar::i420_to_rgba_row(
            &row[simd_width..],
            &cb[chroma..],
            &cr[chroma..],
            &mut out[4 * simd_width..],
        );
    }
}

/// The chroma terms of 8 Cb and Cr samples (16-bit lanes) for their 16
/// pixels. `BIAS` is folded into the remainders here, once per 2x2 group.
#[inline]
#[target_feature(enable = "simd128")]
fn chroma_terms(cb: v128, cr: v128) -> ChromaTerms {
    let offset = i16x8_splat(bt601::CHROMA_OFFSET as i16);
    let bias = i16x8_splat(bt601_split::BIAS);
    let u = i16x8_sub(cb, offset);
    let v = i16x8_sub(cr, offset);
    let red = i16x8_add(i16x8_mul(v, i16x8_splat(bt601_split::R_CR)), bias);
    let green = i16x8_add(
        i16x8_add(
            i16x8_mul(u, i16x8_splat(bt601_split::G_CB)),
            i16x8_mul(v, i16x8_splat(bt601_split::G_CR)),
        ),
        bias,
    );
    let blue = i16x8_add(i16x8_shl(u, bt601_split::B_CB_SHIFT), bias);
    let cb2 = i16x8_add(u, u);
    ChromaTerms {
        remainder: [pair_up(red), pair_up(green), pair_up(blue)],
        cr: pair_up(v),
        cb2: pair_up(cb2),
    }
}

/// Each of 8 lanes repeated for its two pixels: lanes 0-3 and 4-7.
#[inline]
#[target_feature(enable = "simd128")]
fn pair_up(c: v128) -> [v128; 2] {
    [
        i16x8_shuffle::<0, 0, 1, 1, 2, 2, 3, 3>(c, c),
        i16x8_shuffle::<4, 4, 5, 5, 6, 6, 7, 7>(c, c),
    ]
}

/// R, G and B of 8 pixels in 16-bit lanes (unclamped) from their luma and
/// half `half` (0: low 8 pixels, 1: high) of the chroma terms.
#[inline]
#[target_feature(enable = "simd128")]
fn rgb8(luma: v128, terms: &ChromaTerms, half: usize) -> [v128; 3] {
    let scaled = i16x8_mul(luma, i16x8_splat(bt601_split::Y));
    let linear = [
        i16x8_add(luma, terms.cr[half]),
        i16x8_sub(luma, terms.cr[half]),
        i16x8_add(luma, terms.cb2[half]),
    ];
    [0, 1, 2].map(|c| {
        let remainder = i16x8_add(scaled, terms.remainder[c][half]);
        i16x8_add(linear[c], i16x8_shr(remainder, bt601::SHIFT))
    })
}

/// Clamp 16 pixels' channels to bytes and store them as 64 RGBA bytes.
///
/// # Safety
///
/// `out` must be valid for 64 bytes of writes.
#[inline]
#[target_feature(enable = "simd128")]
unsafe fn store_rgba(out: *mut u8, low: [v128; 3], high: [v128; 3]) {
    let [red, green, blue] = [0, 1, 2].map(|c| u8x16_narrow_i16x8(low[c], high[c]));
    let alpha = u8x16_splat(ALPHA_OPAQUE);
    let rg_low =
        i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(red, green);
    let rg_high =
        i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(red, green);
    let ba_low =
        i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(blue, alpha);
    let ba_high =
        i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(blue, alpha);
    let pixels = [
        i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(rg_low, ba_low),
        i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(rg_low, ba_low),
        i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(rg_high, ba_high),
        i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(rg_high, ba_high),
    ];
    for (i, quad) in pixels.into_iter().enumerate() {
        // SAFETY: 16 * i + 16 <= 64 bytes (caller contract).
        unsafe { v128_store(out.add(16 * i).cast(), quad) };
    }
}
