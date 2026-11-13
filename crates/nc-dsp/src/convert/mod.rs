//! Pixel-format conversion kernels: camera layouts into the encoder's planar
//! 4:2:0 source buffer, and decoded pictures into display RGBA; plus the
//! browser demo's frame-quality sum.
//!
//! - **Deinterleave:** split a semi-planar (NV12) chroma plane of Cb, Cr
//!   byte pairs into separate Cb and Cr planes. Planar layouts (I420, YV12)
//!   need only row copies, which `copy_from_slice` already does at memory
//!   speed, so they have no kernel here.
//! - **I420 to RGBA:** a [`DisplayMatrix`] (BT.601 or BT.709, limited or
//!   full range) in 8.8 fixed point (the browser demo's display conversion,
//!   `nc-web`), a row pair sharing one chroma row per call, alpha opaque.
//!   Every result is clamped to 0..=255, so SIMD versions may narrow with
//!   saturation; they compute in 16-bit lanes with the exact rewrite in
//!   [`RgbSplit`].
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
    unsafe fn(matrix: DisplayMatrix, luma: [&[u8]; 2], cb: &[u8], cr: &[u8], rgba: [&mut [u8]; 2]);

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

/// The YCbCr to RGB conversion a display applies: the colour matrix the
/// samples were made with and their range (`nc_common::ColourInfo`; the
/// browser demo's display, `nc-web`). Every form is 8.8 fixed point,
/// `R = (Yw (Y - Yo) + Rv Cr' + 128) >> 8`,
/// `G = (Yw (Y - Yo) - Gu Cb' - Gv Cr' + 128) >> 8`,
/// `B = (Yw (Y - Yo) + Bu Cb' + 128) >> 8`, clamped to 0..=255, with
/// `Cb' = Cb - 128` and `Cr' = Cr - 128`; limited range scales luma by
/// 255/219 (`Yw = 298`, `Yo = 16`) and the chroma weights by 255/224, full
/// range has `Yw = 256`, `Yo = 0`. The weights are the matrix's
/// (Rec. ITU-R BT.601 / BT.709 Kr, Kb) inverse rounded to 1/256.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DisplayMatrix {
    /// Limited-range BT.601 (SMPTE 170M, BT.470 B/G): standard-definition
    /// video, and what the demo's RGBA input conversion produces.
    Bt601Limited,
    /// Limited-range BT.709: HD video, as the sample clips' decoded planes.
    Bt709Limited,
    /// Full-range BT.601 (JPEG, many webcams).
    Bt601Full,
    /// Full-range BT.709.
    Bt709Full,
}

impl DisplayMatrix {
    /// Every matrix (checks and tests).
    pub const ALL: [Self; 4] = [
        Self::Bt601Limited,
        Self::Bt709Limited,
        Self::Bt601Full,
        Self::Bt709Full,
    ];

    /// This conversion's 8.8 weights.
    pub(crate) const fn weights(self) -> YuvToRgb {
        match self {
            Self::Bt601Limited => YuvToRgb::limited(409, 100, 208, 516),
            Self::Bt709Limited => YuvToRgb::limited(459, 55, 136, 541),
            Self::Bt601Full => YuvToRgb::full(359, 88, 183, 454),
            Self::Bt709Full => YuvToRgb::full(403, 48, 120, 475),
        }
    }

    /// This conversion in the 16-bit split form the SIMD kernels use.
    pub(crate) const fn split(self) -> RgbSplit {
        RgbSplit::of(self.weights())
    }
}

/// The 8.8 weights of one [`DisplayMatrix`] (formula in its docs).
#[derive(Debug, Clone, Copy)]
pub(crate) struct YuvToRgb {
    /// Luma weight `Yw`.
    pub y: i32,
    /// Luma black level `Yo`.
    pub luma_offset: i32,
    /// Cr weight in R.
    pub r_cr: i32,
    /// Cb weight in G (subtracted).
    pub g_cb: i32,
    /// Cr weight in G (subtracted).
    pub g_cr: i32,
    /// Cb weight in B.
    pub b_cb: i32,
}

impl YuvToRgb {
    /// Limited-range luma: weight 298 (255/219 in 8.8), black at 16.
    const fn limited(r_cr: i32, g_cb: i32, g_cr: i32, b_cb: i32) -> Self {
        Self {
            y: 298,
            luma_offset: 16,
            r_cr,
            g_cb,
            g_cr,
            b_cb,
        }
    }

    /// Full-range luma: weight 256 (one), black at 0.
    const fn full(r_cr: i32, g_cb: i32, g_cr: i32, b_cb: i32) -> Self {
        Self {
            y: 256,
            luma_offset: 0,
            r_cr,
            g_cb,
            g_cr,
            b_cb,
        }
    }
}

/// Chroma zero level.
pub(crate) const CHROMA_OFFSET: i32 = 128;
/// Rounding of the final shift.
pub(crate) const ROUND: i32 = 128;
/// Fraction bits of the weights.
pub(crate) const SHIFT: u32 = 8;

/// A [`YuvToRgb`] rewritten so that every intermediate fits `i16`, which
/// lets SIMD kernels work on 16-bit lanes (twice the pixels of the 32-bit
/// products) and still match the scalar `i32` results exactly.
///
/// Each 8.8 weight splits into a multiple of 256 and a remainder; R takes
/// `256 Cr'`, G `-256 Cr'` and B `512 Cb'` whole. With `Cb' = Cb - 128`,
/// `Cr' = Cr - 128`, and since `(256 a + b) >> 8 == a + (b >> 8)` for an
/// arithmetic shift,
///
/// - `R = Y + Cr'   + ((y Y + r Cr'        + bias) >> 8)`,
/// - `G = Y - Cr'   + ((y Y + gu Cb' + gv Cr' + bias) >> 8)`,
/// - `B = Y + 2 Cb' + ((y Y + b Cb'        + bias) >> 8)`,
///
/// where `y = Yw - 256` and `bias = 128 - Yw Yo` folds the rounding and the
/// whole luma offset (its 256 part included) into one constant, so no
/// per-pixel offset is needed. For BT.601 limited range the shifted sums
/// lie in -24224..=25501 (R), -23484..=24966 (G) and -5152..=6578 (B); for
/// BT.709 limited range, the widest, R reaches -30624..=31851. The final
/// values stay inside `i16`, and a saturating narrow to `u8` is the scalar
/// clamp. The exhaustive test below checks every (Y, Cb, Cr) of every
/// matrix.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RgbSplit {
    /// Luma weight after removing 256 (0 for full range, which NEON's
    /// unsigned widening multiply requires to be non-negative).
    pub y: i16,
    /// Cr weight in R after removing 256.
    pub r_cr: i16,
    /// Cb weight in G (whole: no part is taken out).
    pub g_cb: i16,
    /// Cr weight in G after adding back 256.
    pub g_cr: i16,
    /// Cb weight in B after removing 512.
    pub b_cb: i16,
    /// Rounding plus the whole luma offset (`-Yw Yo`).
    pub bias: i16,
}

impl RgbSplit {
    const fn of(w: YuvToRgb) -> Self {
        Self {
            y: (w.y - 256) as i16,
            r_cr: (w.r_cr - 256) as i16,
            g_cb: -w.g_cb as i16,
            g_cr: (256 - w.g_cr) as i16,
            b_cb: (w.b_cb - 512) as i16,
            bias: (ROUND - w.y * w.luma_offset) as i16,
        }
    }
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
    use super::{CHROMA_OFFSET, DisplayMatrix, SHIFT, scalar};

    /// The split form, evaluated in checked `i16` arithmetic (any overflow
    /// panics), for one pixel.
    fn split_rgb(matrix: DisplayMatrix, y: u8, cb: u8, cr: u8) -> [u8; 3] {
        let split = matrix.split();
        let (y, u, v) = (
            i16::from(y),
            i16::from(cb) - CHROMA_OFFSET as i16,
            i16::from(cr) - CHROMA_OFFSET as i16,
        );
        let scaled = y
            .checked_mul(split.y)
            .unwrap()
            .checked_add(split.bias)
            .unwrap();
        let channel = |linear: i16, remainder: i16| {
            let value = linear
                .checked_add((scaled.checked_add(remainder).unwrap()) >> SHIFT)
                .unwrap();
            value.clamp(0, 255) as u8
        };
        [
            channel(y + v, split.r_cr * v),
            channel(y - v, split.g_cb * u + split.g_cr * v),
            channel(y + 2 * u, split.b_cb * u),
        ]
    }

    #[test]
    fn split_form_matches_scalar_for_every_sample() {
        let mut rgba = [0u8; 4];
        for matrix in DisplayMatrix::ALL {
            for y in 0..=255u8 {
                for cb in 0..=255u8 {
                    for cr in 0..=255u8 {
                        scalar::i420_to_rgba_row(matrix, &[y], &[cb], &[cr], &mut rgba);
                        assert_eq!(
                            split_rgb(matrix, y, cb, cr),
                            rgba[..3],
                            "{matrix:?} Y={y} Cb={cb} Cr={cr}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn matrices_map_black_white_and_grey() {
        let mut rgba = [0u8; 4];
        for (matrix, black, white) in [
            (DisplayMatrix::Bt601Limited, 16, 235),
            (DisplayMatrix::Bt709Limited, 16, 235),
            (DisplayMatrix::Bt601Full, 0, 255),
            (DisplayMatrix::Bt709Full, 0, 255),
        ] {
            scalar::i420_to_rgba_row(matrix, &[black], &[128], &[128], &mut rgba);
            assert_eq!(rgba, [0, 0, 0, 255], "{matrix:?}");
            scalar::i420_to_rgba_row(matrix, &[white], &[128], &[128], &mut rgba);
            assert_eq!(rgba, [255; 4], "{matrix:?}");
        }
    }
}
