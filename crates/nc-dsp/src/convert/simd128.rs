//! WebAssembly SIMD128 conversion kernels.
//!
//! Deinterleave: one `i8x16_shuffle` of two 16-byte loads gathers the even
//! (Cb) bytes, another the odd (Cr) bytes; 16 pairs per iteration, with the
//! scalar row function for the tail.
//!
//! I420 to RGBA: `to_rgba_simd128`.
//!
//! RGBA to I420: eight pixels of two rows per iteration, each pixel loaded
//! once for its luma and its 2x2 group's chroma. A mask and a 16-bit shift
//! split each pixel's 32 bits into 16-bit lanes `[R, B]` and `[G, alpha]`;
//! two `dot_i16x8_s` with weight pairs (alpha weighs zero) give the exact
//! `i32` weighted sum, which is rounded, offset and shifted as the scalar
//! code does in `i32`. The chroma groups add both rows' lanes (at most
//! 510), then adjacent pixels by even/odd 32-bit lane shuffles (at most
//! 1020), and round the means as the scalar `(sum + 2) / 4`. Results lie
//! inside 16..=240 (`convert` module docs): the saturating narrows never
//! clamp. Eight-pixel batches reduce live channel vectors compared with
//! sixteen: paired V8 SIMD128 timings at 720p improve about 18.5%.
//!
//! Row SSE: 16 samples per iteration. `|a - b|` is `max - min` in unsigned
//! bytes; widened to 16-bit lanes, `dot_i16x8_s` squares and pairs them
//! (at most `2 * 255^2` per lane). Each 32-bit accumulator lane gains at
//! most `4 * 255^2` per iteration, so a row of `MAX_SSE_ROW` samples
//! (4096 iterations) stays below `2^31`; the scalar function takes the tail.
//!
//! Kernel contract: see the `convert` module docs.

use core::arch::wasm32::{
    i8x16_shuffle, i16x8_narrow_i32x4, i32x4_add, i32x4_dot_i16x8, i32x4_extract_lane, i32x4_shr,
    i32x4_shuffle, i32x4_splat, u8x16_max, u8x16_min, u8x16_narrow_i16x8, u8x16_sub, u16x8_add,
    u16x8_extend_high_u8x16, u16x8_extend_low_u8x16, u16x8_shr, u16x8_splat, v128, v128_and,
    v128_load, v128_store, v128_store32_lane, v128_store64_lane,
};

use super::{ConvertFns, bt601_forward, scalar};

/// Pairs split per iteration of the deinterleave loop.
const DEINTERLEAVE_STEP: usize = 16;

/// The SIMD128 kernel table (SIMD128 is a compile-time feature).
pub(crate) fn convert_fns() -> ConvertFns {
    ConvertFns {
        deinterleave_uv,
        i420_to_rgba_rows: super::to_rgba_simd128::i420_to_rgba_rows,
        rgba_to_i420,
        sse_row,
    }
}

/// Samples summed per iteration of the row SSE loop.
const SSE_STEP: usize = 16;

/// # Safety
///
/// Kernel contract (`convert` module docs).
#[target_feature(enable = "simd128")]
unsafe fn sse_row(a: &[u8], b: &[u8]) -> u32 {
    let n = a.len();
    let simd_width = n - n % SSE_STEP;
    let mut sums = i32x4_splat(0);
    for x in (0..simd_width).step_by(SSE_STEP) {
        // SAFETY: x + 16 <= n = a.len() = b.len() (kernel contract).
        let (va, vb) = unsafe {
            (
                v128_load(a.as_ptr().add(x).cast()),
                v128_load(b.as_ptr().add(x).cast()),
            )
        };
        let diff = u8x16_sub(u8x16_max(va, vb), u8x16_min(va, vb));
        let (low, high) = (u16x8_extend_low_u8x16(diff), u16x8_extend_high_u8x16(diff));
        sums = i32x4_add(sums, i32x4_dot_i16x8(low, low));
        sums = i32x4_add(sums, i32x4_dot_i16x8(high, high));
    }
    let lanes = [
        i32x4_extract_lane::<0>(sums),
        i32x4_extract_lane::<1>(sums),
        i32x4_extract_lane::<2>(sums),
        i32x4_extract_lane::<3>(sums),
    ];
    // Each lane is below 2^31 (module docs), so the casts are exact.
    let simd_sum: u32 = lanes.iter().map(|&lane| lane as u32).sum();
    simd_sum + scalar::sse_row(&a[simd_width..], &b[simd_width..])
}

/// Pixels converted per iteration of the RGBA to I420 loop.
const I420_STEP: usize = 8;

/// # Safety
///
/// Kernel contract (`convert` module docs).
#[allow(clippy::too_many_arguments, reason = "matches DeinterleaveUvFn")]
#[target_feature(enable = "simd128")]
unsafe fn deinterleave_uv(
    src: &[u8],
    src_stride: usize,
    dst_u: &mut [u8],
    u_stride: usize,
    dst_v: &mut [u8],
    v_stride: usize,
    w: usize,
    h: usize,
) {
    let simd_width = w - w % DEINTERLEAVE_STEP;
    for y in 0..h {
        let pairs = &src[y * src_stride..y * src_stride + 2 * w];
        let cb_row = &mut dst_u[y * u_stride..y * u_stride + w];
        let cr_row = &mut dst_v[y * v_stride..y * v_stride + w];
        for x in (0..simd_width).step_by(DEINTERLEAVE_STEP) {
            // SAFETY: source bytes 2x .. 2x + 32 lie inside the 2w-byte row
            // and outputs x .. x + 16 inside the w-sample rows, since
            // x + 16 <= w (kernel contract).
            unsafe {
                let start = pairs.as_ptr().add(2 * x);
                let (low, high) = (v128_load(start.cast()), v128_load(start.add(16).cast()));
                let cb = i8x16_shuffle::<0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30>(
                    low, high,
                );
                let cr = i8x16_shuffle::<1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31>(
                    low, high,
                );
                v128_store(cb_row.as_mut_ptr().add(x).cast(), cb);
                v128_store(cr_row.as_mut_ptr().add(x).cast(), cr);
            }
        }
        scalar::deinterleave_uv_row(
            &pairs[2 * simd_width..],
            &mut cb_row[simd_width..],
            &mut cr_row[simd_width..],
        );
    }
}

/// Two 16-bit lanes `[low, high]` in each 32-bit lane: one `dot_i16x8_s`
/// weight pair.
const fn weight_pair(low: i32, high: i32) -> i32 {
    (low as u16 as i32) | ((high as u16 as i32) << 16)
}

/// Bytes of the R, B pair and of the G, alpha pair in each pixel's 32 bits.
const RED_BLUE_MASK: i32 = 0x00ff_00ff;

/// Four RGBA pixels as 16-bit lanes: `[R, B]` and `[G, alpha]` per pixel.
#[derive(Clone, Copy)]
struct Channels {
    red_blue: v128,
    green_alpha: v128,
}

/// Weighted sums `w_r R + w_g G + w_b B` of the four pixels in `c`, in
/// 32-bit lanes (alpha weighs zero).
#[inline]
#[target_feature(enable = "simd128")]
fn weighted4(c: Channels, [w_red, w_green, w_blue]: [i32; 3]) -> v128 {
    i32x4_add(
        i32x4_dot_i16x8(c.red_blue, i32x4_splat(weight_pair(w_red, w_blue))),
        i32x4_dot_i16x8(c.green_alpha, i32x4_splat(weight_pair(w_green, 0))),
    )
}

/// The rounded, offset result `offset + ((sum + ROUND) >> SHIFT)` of four
/// weighted sums: the offset is folded into the rounding term as
/// `offset << SHIFT`, exact under the arithmetic shift.
#[inline]
#[target_feature(enable = "simd128")]
fn scaled4(sum: v128, offset: i32) -> v128 {
    let bias = i32x4_splat(bt601_forward::ROUND + (offset << bt601_forward::SHIFT));
    i32x4_shr(i32x4_add(sum, bias), bt601_forward::SHIFT)
}

/// Eight packed RGBA pixels at `src` as two [`Channels`].
///
/// # Safety
///
/// `src` must be valid for reading 32 bytes.
#[inline]
#[target_feature(enable = "simd128")]
unsafe fn load_channels(src: *const u8) -> [Channels; 2] {
    let mask = i32x4_splat(RED_BLUE_MASK);
    [0, 16].map(|offset| {
        // SAFETY: the caller guarantees 32 readable bytes.
        let pixels = unsafe { v128_load(src.add(offset).cast()) };
        Channels {
            red_blue: v128_and(pixels, mask),
            green_alpha: u16x8_shr(pixels, 8),
        }
    })
}

/// Luma bytes of eight pixels, in the low half of the vector.
#[inline]
#[target_feature(enable = "simd128")]
fn luma8(pixels: [Channels; 2]) -> v128 {
    let [a, b] =
        pixels.map(|p| scaled4(weighted4(p, bt601_forward::Y), bt601_forward::LUMA_OFFSET));
    let words = i16x8_narrow_i32x4(a, b);
    u8x16_narrow_i16x8(words, words)
}

/// Mean R, B and G (alpha) of the four 2x2 groups covered by 8 pixels of
/// two rows (`upper`, `lower`): the vertical sums of each pixel, then
/// adjacent pixels (adjacent 32-bit lanes) paired by even/odd lane
/// shuffles, rounded as the scalar `(sum + 2) / 4`.
#[inline]
#[target_feature(enable = "simd128")]
fn group_means(upper: [Channels; 2], lower: [Channels; 2]) -> Channels {
    let mean = |pick: fn(Channels) -> v128| {
        let a = u16x8_add(pick(upper[0]), pick(lower[0]));
        let b = u16x8_add(pick(upper[1]), pick(lower[1]));
        let sums = u16x8_add(
            i32x4_shuffle::<0, 2, 4, 6>(a, b),
            i32x4_shuffle::<1, 3, 5, 7>(a, b),
        );
        u16x8_shr(u16x8_add(sums, u16x8_splat(2)), 2)
    };
    Channels {
        red_blue: mean(|c| c.red_blue),
        green_alpha: mean(|c| c.green_alpha),
    }
}

/// Cb and Cr of the four 2x2 groups of `means`, in 32-bit lanes.
#[inline]
#[target_feature(enable = "simd128")]
fn chroma4(means: Channels) -> [v128; 2] {
    let offset = bt601_forward::CHROMA_OFFSET;
    [
        scaled4(weighted4(means, bt601_forward::CB), offset),
        scaled4(weighted4(means, bt601_forward::CR), offset),
    ]
}

/// Luma of one packed RGBA row: eight pixels per iteration, the scalar row
/// function for the tail.
#[target_feature(enable = "simd128")]
fn luma_row(row: &[u8], out: &mut [u8]) {
    let simd_width = out.len() - out.len() % I420_STEP;
    for x in (0..simd_width).step_by(I420_STEP) {
        // SAFETY: x + 8 <= out.len(): 32 RGBA bytes and eight luma
        // samples lie inside the row.
        unsafe {
            let pixels = load_channels(row.as_ptr().add(4 * x));
            v128_store64_lane::<0>(luma8(pixels), out.as_mut_ptr().add(x).cast());
        }
    }
    scalar::rgba_to_luma_row(&row[4 * simd_width..], &mut out[simd_width..]);
}

/// # Safety
///
/// Kernel contract (`convert` module docs).
#[allow(clippy::too_many_arguments, reason = "matches RgbaToI420Fn")]
#[target_feature(enable = "simd128")]
unsafe fn rgba_to_i420(
    rgba: &[u8],
    luma: &mut [u8],
    luma_stride: usize,
    cb: &mut [u8],
    cr: &mut [u8],
    chroma_stride: usize,
    w: usize,
    h: usize,
) {
    let simd_width = w - w % I420_STEP;
    let (groups, simd_groups) = (w / 2, simd_width / 2);
    for y in 0..h / 2 {
        let rows = &rgba[8 * w * y..][..8 * w];
        let (top, bottom) = rows.split_at(4 * w);
        let (luma_top, luma_bottom) = luma[2 * y * luma_stride..].split_at_mut(luma_stride);
        let (luma_top, luma_bottom) = (&mut luma_top[..w], &mut luma_bottom[..w]);
        let cb_row = &mut cb[chroma_stride * y..][..groups];
        let cr_row = &mut cr[chroma_stride * y..][..groups];
        for x in (0..simd_width).step_by(I420_STEP) {
            // SAFETY: x + 8 <= w: 32 RGBA bytes, eight luma samples
            // and four chroma samples of each row are in bounds.
            unsafe {
                let upper = load_channels(top.as_ptr().add(4 * x));
                let lower = load_channels(bottom.as_ptr().add(4 * x));
                v128_store64_lane::<0>(luma8(upper), luma_top.as_mut_ptr().add(x).cast());
                v128_store64_lane::<0>(luma8(lower), luma_bottom.as_mut_ptr().add(x).cast());
                let [cb_values, cr_values] = chroma4(group_means(upper, lower));
                let words = i16x8_narrow_i32x4(cb_values, cr_values);
                let bytes = u8x16_narrow_i16x8(words, words);
                v128_store32_lane::<0>(bytes, cb_row.as_mut_ptr().add(x / 2).cast());
                v128_store32_lane::<1>(bytes, cr_row.as_mut_ptr().add(x / 2).cast());
            }
        }
        scalar::rgba_to_luma_row(&top[4 * simd_width..], &mut luma_top[simd_width..]);
        scalar::rgba_to_luma_row(&bottom[4 * simd_width..], &mut luma_bottom[simd_width..]);
        scalar::rgba_to_chroma_row(
            rows,
            simd_groups,
            &mut cb_row[simd_groups..],
            &mut cr_row[simd_groups..],
        );
    }
    if h % 2 == 1 {
        let y = h - 1;
        luma_row(
            &rgba[4 * w * y..][..4 * w],
            &mut luma[luma_stride * y..][..w],
        );
    }
}
