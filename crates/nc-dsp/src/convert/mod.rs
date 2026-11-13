//! Pixel-format conversion kernels: camera layouts into the encoder's planar
//! 4:2:0 source buffer, and decoded pictures into display RGBA; plus the
//! browser demo's frame-quality sum.
//!
//! - **Deinterleave:** split a semi-planar (NV12) chroma plane of Cb, Cr
//!   byte pairs into separate Cb and Cr planes. Planar layouts (I420, YV12)
//!   need only row copies, which `copy_from_slice` already does at memory
//!   speed, so they have no kernel here.
//! - **I420 to RGBA:** limited-range BT.601 in 8.8 fixed point (the browser
//!   demo's display conversion, `nc-web`), a row pair sharing one chroma
//!   row per call, alpha opaque. Every result is clamped to 0..=255, so SIMD
//!   versions may narrow with saturation; they compute in 16-bit lanes with
//!   the exact rewrite in [`bt601_split`].
//! - **RGBA to I420:** the inverse (the browser demo's encoder input):
//!   limited-range BT.601 luma per pixel, chroma from the rounded mean of
//!   each whole 2x2 group (a trailing odd row or column gets no chroma).
//!   Every intermediate fits 16 bits: luma sums are at most 56228 (unsigned),
//!   chroma sums lie in -28432..=28688, and every result is already inside
//!   16..=240, so the scalar clamps never bind.
//! - **Row SSE:** the sum of squared differences of two sample rows (the
//!   browser demo's per-frame PSNR, `nc-web`). A row of at most
//!   [`MAX_SSE_ROW`] samples sums to at most `65536 * 255^2 < 2^32`.
//!
//! The native forward kernels fold rounding and output offsets into the
//! weighted sum. The final numerators are 4224..=60324 for Y and
//! 4336..=61456 for Cb/Cr, fitting unsigned 16 bits. Wrapping 16-bit
//! multiply/add (including negative coefficients modulo 65536) followed by
//! an unsigned shift is therefore exactly the scalar signed arithmetic.
//!
//! # Kernel contract
//!
//! Every kernel is an `unsafe fn`, called only from [`crate::Dsp`], which
//! guarantees
//!
//! 1. `deinterleave_uv`: `src` covers `h` rows of `2 * w` bytes at
//!    `src_stride`; `dst_u` and `dst_v` cover `h` rows of `w` samples at
//!    their strides, with `w, h >= 1`;
//! 2. `i420_to_rgba_rows`: `luma[0]` holds `w >= 1` samples, `luma[1]`
//!    either `w` (a row pair sharing the chroma row) or none (a lone last
//!    row), `cb` and `cr` at least `w.div_ceil(2)`, and each `rgba[i]`
//!    exactly `4 * luma[i].len()` bytes;
//! 3. `rgba_to_i420`: `rgba` holds `h` rows of `4 * w` bytes (packed),
//!    `luma` covers `h` rows of `w` samples at `luma_stride`, and `cb`, `cr`
//!    cover `h / 2` rows of `w / 2` samples at `chroma_stride`, `w, h >= 1`;
//! 4. `sse_row`: `a` and `b` have the same length, at most [`MAX_SSE_ROW`];
//! 5. the CPU supports the kernel's ISA.

pub(crate) mod scalar;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub(crate) mod avx2;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub(crate) mod avx512;

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
pub(crate) mod neon;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
pub(crate) mod simd128;

/// Split `h` rows of `w` interleaved (Cb, Cr) pairs into a Cb and a Cr plane.
pub(crate) type DeinterleaveUvFn = unsafe fn(
    src: &[u8],
    src_stride: usize,
    dst_u: &mut [u8],
    u_stride: usize,
    dst_v: &mut [u8],
    v_stride: usize,
    w: usize,
    h: usize,
);

/// Convert one or two rows of `luma[0].len()` pixels that share one chroma
/// row (each `cb`/`cr` sample covers a 2x2 group) into opaque RGBA. Two rows
/// per call let SIMD kernels compute each chroma term once for four pixels.
pub(crate) type I420ToRgbaRowsFn =
    unsafe fn(luma: [&[u8]; 2], cb: &[u8], cr: &[u8], rgba: [&mut [u8]; 2]);

/// Convert `h` packed RGBA rows of `w` pixels into I420 planes.
pub(crate) type RgbaToI420Fn = unsafe fn(
    rgba: &[u8],
    luma: &mut [u8],
    luma_stride: usize,
    cb: &mut [u8],
    cr: &mut [u8],
    chroma_stride: usize,
    w: usize,
    h: usize,
);

/// Sum of `(a[i] - b[i])^2` over two equally long rows.
pub(crate) type SseRowFn = unsafe fn(a: &[u8], b: &[u8]) -> u32;

/// Longest row `sse_row` accepts: its sum then fits `u32`.
pub(crate) const MAX_SSE_ROW: usize = 1 << 16;

/// Limited-range BT.601 RGB to YCbCr in 8.8 fixed point:
/// `Y = 16 + ((66 R + 129 G + 25 B + 128) >> 8)`,
/// `Cb = 128 + ((-38 R - 74 G + 112 B + 128) >> 8)`,
/// `Cr = 128 + ((112 R - 94 G - 18 B + 128) >> 8)`.
pub(crate) mod bt601_forward {
    /// Luma weights of R, G, B.
    pub(crate) const Y: [i32; 3] = [66, 129, 25];
    /// Cb weights of R, G, B.
    pub(crate) const CB: [i32; 3] = [-38, -74, 112];
    /// Cr weights of R, G, B.
    pub(crate) const CR: [i32; 3] = [112, -94, -18];
    /// Rounding of the 8-bit shift.
    pub(crate) const ROUND: i32 = 128;
    /// Fraction bits of the weights.
    pub(crate) const SHIFT: u32 = 8;
    /// Luma black level.
    pub(crate) const LUMA_OFFSET: i32 = 16;
    /// Chroma zero level.
    pub(crate) const CHROMA_OFFSET: i32 = 128;
}

/// Limited-range BT.601 YCbCr to RGB in 8.8 fixed point:
/// `R = (Y' * 298 + Cr' * 409 + 128) >> 8` and so on, with `Y' = Y - 16`,
/// `Cb' = Cb - 128`, `Cr' = Cr - 128` (Rec. ITU-R BT.601 matrix scaled by
/// 255/219 and 255/224).
pub(crate) mod bt601 {
    /// Luma black level.
    pub(crate) const LUMA_OFFSET: i32 = 16;
    /// Chroma zero level.
    pub(crate) const CHROMA_OFFSET: i32 = 128;
    /// Luma weight, shared by all three outputs.
    pub(crate) const Y: i32 = 298;
    /// Cr weight in R.
    pub(crate) const R_CR: i32 = 409;
    /// Cb weight in G (subtracted).
    pub(crate) const G_CB: i32 = 100;
    /// Cr weight in G (subtracted).
    pub(crate) const G_CR: i32 = 208;
    /// Cb weight in B.
    pub(crate) const B_CB: i32 = 516;
    /// Rounding of the final shift.
    pub(crate) const ROUND: i32 = 128;
    /// Fraction bits of the weights.
    pub(crate) const SHIFT: u32 = 8;
}

/// The [`bt601`] inverse rewritten so that every intermediate fits `i16`,
/// which lets SIMD kernels work on 16-bit lanes (twice the pixels of the
/// 32-bit products) and still match the scalar `i32` results exactly.
///
/// Each 8.8 weight splits into a multiple of 256 and a remainder:
/// `298 = 256 + 42`, `409 = 256 + 153`, `-208 = -256 + 48`, `516 = 512 + 4`.
/// With `Cb' = Cb - 128`, `Cr' = Cr - 128`, and since
/// `(256 a + b) >> 8 == a + (b >> 8)` for an arithmetic shift,
///
/// - `R = Y + Cr'     + ((42 Y + 153 Cr'           + BIAS) >> 8)`,
/// - `G = Y - Cr'     + ((42 Y - 100 Cb' + 48 Cr'  + BIAS) >> 8)`,
/// - `B = Y + 2 Cb'   + ((42 Y + 4 Cb'             + BIAS) >> 8)`,
///
/// where `BIAS = 128 - 298 * 16` folds the rounding and the whole luma
/// offset (its 256 part included) into one constant, so no per-pixel
/// offset is needed. The shifted sums lie in -24224..=25501 (R),
/// -23484..=24966 (G) and -5152..=6578 (B), and the final values in
/// -223..=481, all inside `i16`; a saturating narrow to `u8` is the scalar
/// clamp. The exhaustive test below checks every (Y, Cb, Cr).
pub(crate) mod bt601_split {
    use super::bt601;

    /// Luma weight after removing 256.
    pub(crate) const Y: i16 = (bt601::Y - 256) as i16;
    /// Cr weight in R after removing 256.
    pub(crate) const R_CR: i16 = (bt601::R_CR - 256) as i16;
    /// Cb weight in G (already below 256).
    pub(crate) const G_CB: i16 = -(bt601::G_CB as i16);
    /// Cr weight in G after adding back 256.
    pub(crate) const G_CR: i16 = (256 - bt601::G_CR) as i16;
    /// Cb weight in B after removing 512.
    pub(crate) const B_CB: i16 = (bt601::B_CB - 512) as i16;
    /// `B_CB` as a left shift (it is a power of two).
    pub(crate) const B_CB_SHIFT: u32 = B_CB.trailing_zeros();
    const _: () = assert!(1 << B_CB_SHIFT == B_CB);
    /// Rounding plus the whole luma offset (`-298 * 16`).
    pub(crate) const BIAS: i16 = (bt601::ROUND - bt601::Y * bt601::LUMA_OFFSET) as i16;
}

/// Opaque alpha.
pub(crate) const ALPHA_OPAQUE: u8 = 255;

/// One ISA's implementations of every conversion kernel.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ConvertFns {
    pub deinterleave_uv: DeinterleaveUvFn,
    pub i420_to_rgba_rows: I420ToRgbaRowsFn,
    pub rgba_to_i420: RgbaToI420Fn,
    pub sse_row: SseRowFn,
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod rgba_avx2;
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod rgba_neon;
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod to_rgba_avx2;
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod to_rgba_neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
mod to_rgba_simd128;

#[cfg(test)]
mod tests {
    use super::{bt601, bt601_split, scalar};

    /// The split form, evaluated in checked `i16` arithmetic (any overflow
    /// panics), for one pixel.
    fn split_rgb(y: u8, cb: u8, cr: u8) -> [u8; 3] {
        let (y, u, v) = (
            i16::from(y),
            i16::from(cb) - bt601::CHROMA_OFFSET as i16,
            i16::from(cr) - bt601::CHROMA_OFFSET as i16,
        );
        let m = y * bt601_split::Y + bt601_split::BIAS;
        let channel = |linear: i16, remainder: i16| {
            let value = linear
                .checked_add((m.checked_add(remainder).unwrap()) >> 8)
                .unwrap();
            value.clamp(0, 255) as u8
        };
        [
            channel(y + v, bt601_split::R_CR * v),
            channel(y - v, bt601_split::G_CB * u + bt601_split::G_CR * v),
            channel(y + 2 * u, bt601_split::B_CB * u),
        ]
    }

    #[test]
    fn split_form_matches_scalar_for_every_sample() {
        let mut rgba = [0u8; 4];
        for y in 0..=255u8 {
            for cb in 0..=255u8 {
                for cr in 0..=255u8 {
                    scalar::i420_to_rgba_row(&[y], &[cb], &[cr], &mut rgba);
                    assert_eq!(split_rgb(y, cb, cr), rgba[..3], "Y={y} Cb={cb} Cr={cr}");
                }
            }
        }
    }
}
