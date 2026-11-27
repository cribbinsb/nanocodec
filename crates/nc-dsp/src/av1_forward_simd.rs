//! SIMD 2-D drivers of [`crate::av1_forward`] across NEON, AVX2, AVX-512 and
//! wasm32 `simd128`, generic over [`I32Lanes`].
//!
//! Each pass transforms `V::LANES` columns (or rows) in parallel with a
//! register-tile transpose between passes, matching `txfm::generic`:
//! - Vertical `FLIPADST` reverses the residual row index on load (zero cost).
//! - Horizontal `FLIPADST` reverses the transposed column index on load in
//!   the row pass (zero cost, no lane shuffles).
//! - 64-point dimensions run [`dct::fdct64_low32`] and only round-shift,
//!   transpose and store the kept 32 frequencies.
//! - Blocks up to 16x16 (all ADST / FLIPADST transforms) use a 1 KiB
//!   uninitialised stack buffer; larger blocks (32- and 64-point DCT / IDTX)
//!   use an 8 KiB uninitialised stack buffer.
//!
//! Arithmetic follows libaom `fwd_txfm2d_c` (`av1/encoder/av1_fwd_txfm2d.c`,
//! BSD-2-Clause, Copyright (c) 2016, Alliance for Open Media; see
//! `THIRD_PARTY.md`). The lane-parallel column-then-row `I32Lanes` +
//! register-tile transpose pattern is shared with `txfm::generic`, `vp9_txfm`
//! and `av2_txfm`; VP9's ADST (14-bit constants, no intermediate rounding
//! shifts) and AV2's ADST (matrix transform) keep their own 1-D arithmetic
//! while sharing the same 2-D SIMD layout and transpose technique.

#![allow(clippy::inline_always)]

use core::mem::MaybeUninit;

use crate::av1_forward::{
    COS_BIT_COL, COS_BIT_ROW, SHIFTS, SQRT2, SQRT2_BITS, fadst4, fadst8, fadst16, fidentity,
};
use crate::av1_inverse::{Av1TxSize, Av1TxType, Kernel1d};
use crate::simd::I32Lanes;
use crate::txfm::{cospi, dct};

/// `(0..n).step_by(V::LANES)` as a counted loop so the compiler drops slice
/// bounds checks inside the tile loops.
#[inline(always)]
fn lane_groups<V: I32Lanes>(n: usize) -> impl Iterator<Item = usize> {
    (0..n / V::LANES).map(|group| group * V::LANES)
}

#[inline(always)]
fn fixed<V, const N: usize>(slice: &[V]) -> &[V; N] {
    slice.try_into().expect("transform length")
}

#[inline(always)]
fn fixed_mut<V, const N: usize>(slice: &mut [V]) -> &mut [V; N] {
    slice.try_into().expect("transform length")
}

/// One 1-D forward transform on `V::LANES` parallel lines (4 to 64 points).
#[inline(always)]
fn forward_1d_lanes<V: I32Lanes>(kernel: Kernel1d, cos_bit: u32, input: &[V], output: &mut [V]) {
    let cos = cospi(cos_bit);
    match (kernel, input.len()) {
        (Kernel1d::Dct, 4) => dct::fdct4(fixed(input), fixed_mut(output), cos, cos_bit),
        (Kernel1d::Dct, 8) => dct::fdct8(fixed(input), fixed_mut(output), cos, cos_bit),
        (Kernel1d::Dct, 16) => dct::fdct16(fixed(input), fixed_mut(output), cos, cos_bit),
        (Kernel1d::Dct, 32) => dct::fdct32(fixed(input), fixed_mut(output), cos, cos_bit),
        (Kernel1d::Dct, 64) => dct::fdct64_low32(fixed(input), fixed_mut(output), cos, cos_bit),
        (Kernel1d::Adst | Kernel1d::FlipAdst, 4) => {
            fadst4(fixed(input), fixed_mut(output), cos_bit);
        }
        (Kernel1d::Adst | Kernel1d::FlipAdst, 8) => {
            fadst8(fixed(input), fixed_mut(output), cos_bit);
        }
        (Kernel1d::Adst | Kernel1d::FlipAdst, 16) => {
            fadst16(fixed(input), fixed_mut(output), cos_bit);
        }
        (Kernel1d::Identity, _) => fidentity(input, output),
        (k, n) => unreachable!("no {n}-point {k:?} (checked by Av1TxType::fits)"),
    }
}

/// Apply one of libaom's middle rounding shifts (`0`, `1`, `2`, `4`) with a
/// compile-time constant shift count.
#[inline(always)]
fn apply_mid_shift<V: I32Lanes>(y: &mut [V], shift: u32) {
    match shift {
        0 => {}
        1 => {
            for v in y {
                *v = v.round_shift(1);
            }
        }
        2 => {
            for v in y {
                *v = v.round_shift(2);
            }
        }
        4 => {
            for v in y {
                *v = v.round_shift(4);
            }
        }
        s => {
            for v in y {
                *v = v.round_shift(s);
            }
        }
    }
}

/// Column pass of `forward_2d_lanes`: writes `width * coded_height` elements
/// into `buf` transposed (`buf[c * coded_height + r]`).
///
/// # Safety
///
/// `V`'s ISA must be available, and `V::LANES` must divide both `tx.width()`
/// and `tx.coded_height()`.
#[inline(always)]
unsafe fn column_pass<'buf, V: I32Lanes, const MAX_N: usize>(
    tx: Av1TxSize,
    vertical: Kernel1d,
    input_shift: u32,
    mid_shift: u32,
    cos_bit_col: u32,
    residual: &[i16],
    stride: usize,
    buf: &'buf mut [MaybeUninit<i32>],
) -> &'buf [i32] {
    let (width, height) = (tx.width(), tx.height());
    let coded_height = tx.coded_height();
    let flip = vertical == Kernel1d::FlipAdst;
    let mut x = [V::splat(0); MAX_N];
    let mut y = [V::splat(0); MAX_N];

    for c0 in lane_groups::<V>(width) {
        for (r, slot) in x[..height].iter_mut().enumerate() {
            let src_r = if flip { height - 1 - r } else { r };
            let v = V::load_i16(&residual[src_r * stride + c0..]);
            *slot = match input_shift {
                0 => v,
                2 => v.shl(2),
                s => v.shl(s),
            };
        }
        forward_1d_lanes(vertical, cos_bit_col, &x[..height], &mut y[..height]);
        apply_mid_shift(&mut y[..coded_height], mid_shift);
        for r0 in lane_groups::<V>(coded_height) {
            V::store_transposed(
                &y[r0..r0 + V::LANES],
                &mut buf[c0 * coded_height + r0..],
                coded_height,
            );
        }
    }

    // SAFETY: `c0` covers `0..width` and `r0` covers `0..coded_height` in
    // complete `V::LANES x V::LANES` tiles, so every element in
    // `buf[..width * coded_height]` is initialised; `MaybeUninit<i32>` has
    // the layout of `i32`.
    unsafe { core::slice::from_raw_parts(buf.as_ptr().cast::<i32>(), width * coded_height) }
}

/// Row pass of `forward_2d_lanes`: transforms `coded_height` rows of `width`
/// elements from `transposed` (`transposed[c * coded_height + r]`) into
/// column-major `out[k * coded_height + r]`.
#[inline(always)]
fn row_pass<V: I32Lanes, const MAX_N: usize>(
    tx: Av1TxSize,
    horizontal: Kernel1d,
    output_shift: u32,
    cos_bit_row: u32,
    transposed: &[i32],
    out: &mut [i16],
) {
    let (width, height) = (tx.width(), tx.height());
    let (coded_width, coded_height) = (tx.coded_width(), tx.coded_height());
    let flip = horizontal == Kernel1d::FlipAdst;
    let two_to_one = width == 2 * height || height == 2 * width;
    let mut x = [V::splat(0); MAX_N];
    let mut y = [V::splat(0); MAX_N];

    for r0 in lane_groups::<V>(coded_height) {
        for (c, slot) in x[..width].iter_mut().enumerate() {
            let src_c = if flip { width - 1 - c } else { c };
            *slot = V::load(&transposed[src_c * coded_height + r0..]);
        }
        forward_1d_lanes(horizontal, cos_bit_row, &x[..width], &mut y[..width]);
        for (k, &lane) in y[..coded_width].iter().enumerate() {
            let mut v = match output_shift {
                0 => lane,
                2 => lane.round_shift(2),
                s => lane.round_shift(s),
            };
            if two_to_one {
                v = v.mul_scalar(SQRT2).round_shift(SQRT2_BITS);
            }
            v.store_sat16(&mut out[k * coded_height + r0..]);
        }
    }
}

/// Generic 2-D AV1 forward transform on `V::LANES` parallel lines.
///
/// # Safety
///
/// `V`'s ISA must be available, and `V::LANES` must divide both `tx.width()`
/// and `tx.coded_height()`.
#[inline(always)]
unsafe fn forward_2d<V: I32Lanes, const MAX_N: usize, const BUF_LEN: usize>(
    tx: Av1TxSize,
    kind: Av1TxType,
    residual: &[i16],
    stride: usize,
    out: &mut [i16],
) {
    let index = tx.index().expect("a valid transform size");
    let (input_shift, mid_shift, output_shift) = SHIFTS[index];
    let w_idx = usize::from(tx.width_log2 - 2);
    let h_idx = usize::from(tx.height_log2 - 2);
    let (cos_bit_col, cos_bit_row) = (COS_BIT_COL[w_idx][h_idx], COS_BIT_ROW[w_idx][h_idx]);
    let (vertical, horizontal) = kind.kernels();

    let mut buf = [MaybeUninit::<i32>::uninit(); BUF_LEN];
    // SAFETY: forwarded contract.
    let transposed = unsafe {
        column_pass::<V, MAX_N>(
            tx,
            vertical,
            input_shift,
            mid_shift,
            cos_bit_col,
            residual,
            stride,
            &mut buf,
        )
    };
    row_pass::<V, MAX_N>(tx, horizontal, output_shift, cos_bit_row, transposed, out);
}

/// Dispatch between the 1 KiB small-block buffer (`<= 16x16`) and the 8 KiB
/// large-block buffer (`32` or `64` side).
///
/// # Safety
///
/// As [`forward_2d`].
#[inline(always)]
unsafe fn forward_sized<V: I32Lanes>(
    tx: Av1TxSize,
    kind: Av1TxType,
    residual: &[i16],
    stride: usize,
    out: &mut [i16],
) {
    if tx.width() <= 16 && tx.height() <= 16 {
        // SAFETY: forwarded contract; `width <= 16 && height <= 16` fits
        // `MAX_N = 16` and `BUF_LEN = 256`.
        unsafe { forward_2d::<V, 16, 256>(tx, kind, residual, stride, out) };
    } else {
        // SAFETY: forwarded contract; `width <= 64 && coded_height <= 32`
        // fits `MAX_N = 64` and `BUF_LEN = 64 * 32`.
        unsafe { forward_2d::<V, 64, 2048>(tx, kind, residual, stride, out) };
    }
}

/// AVX2 2-D AV1 forward transform (8 lanes when both dimensions >= 8, 4 lanes
/// when either side is 4).
///
/// # Safety
///
/// Caller must guarantee `avx2` is available on the running CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn avx2(
    tx: Av1TxSize,
    kind: Av1TxType,
    residual: &[i16],
    stride: usize,
    out: &mut [i16],
) {
    if tx.width() >= 8 && tx.coded_height() >= 8 {
        // SAFETY: `avx2` is enabled and 8 divides both `width` and `coded_height`.
        unsafe { forward_sized::<crate::simd::avx2::I32x8>(tx, kind, residual, stride, out) };
    } else {
        // SAFETY: `avx2` is enabled and 4 divides both `width` and `coded_height`.
        unsafe { forward_sized::<crate::simd::avx2::I32x4>(tx, kind, residual, stride, out) };
    }
}

/// AVX-512 2-D AV1 forward transform (16 lanes when both dimensions >= 16,
/// 8 lanes when >= 8, 4 lanes when either side is 4).
///
/// # Safety
///
/// Caller must guarantee AVX-512F/BW/DQ/VL and AVX2 are available on the CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2")]
pub(crate) unsafe fn avx512(
    tx: Av1TxSize,
    kind: Av1TxType,
    residual: &[i16],
    stride: usize,
    out: &mut [i16],
) {
    if tx.width() >= 16 && tx.coded_height() >= 16 {
        // SAFETY: AVX-512 is enabled and 16 divides both `width` and `coded_height`.
        unsafe { forward_sized::<crate::simd::avx512::I32x16>(tx, kind, residual, stride, out) };
    } else if tx.width() >= 8 && tx.coded_height() >= 8 {
        // SAFETY: AVX2 is enabled and 8 divides both `width` and `coded_height`.
        unsafe { forward_sized::<crate::simd::avx2::I32x8>(tx, kind, residual, stride, out) };
    } else {
        // SAFETY: AVX2 is enabled and 4 divides both `width` and `coded_height`.
        unsafe { forward_sized::<crate::simd::avx2::I32x4>(tx, kind, residual, stride, out) };
    }
}

/// Granite Rapids keeps the AVX2 code generation for measured narrow
/// shapes while leaving mixed and full-width transforms on the ICL path.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2")]
pub(crate) unsafe fn avx512_granite(
    tx: Av1TxSize,
    kind: Av1TxType,
    residual: &[i16],
    stride: usize,
    out: &mut [i16],
) {
    // One bit per pair of log2 dimensions (2..=6), so dispatch needs
    // only a mask test after selecting the transform type.
    const fn shape_bit(width_log2: u8, height_log2: u8) -> u32 {
        1 << ((width_log2 - 2) * 5 + (height_log2 - 2))
    }
    const IDTX_AVX2: u32 =
        shape_bit(3, 5) | shape_bit(2, 4) | shape_bit(2, 3) | shape_bit(4, 2) | shape_bit(5, 3);
    const ADST_AVX2: u32 = shape_bit(3, 2) | shape_bit(2, 3);
    let shapes = match kind {
        Av1TxType::Idtx => IDTX_AVX2,
        Av1TxType::AdstAdst | Av1TxType::FlipadstAdst => ADST_AVX2,
        _ => 0,
    };
    let prefer_avx2 = shapes & shape_bit(tx.width_log2, tx.height_log2) != 0;
    // SAFETY: both entries take the same checked buffers. This function
    // enables all the required ISA features.
    unsafe {
        if prefer_avx2 {
            avx2(tx, kind, residual, stride, out);
        } else {
            avx512(tx, kind, residual, stride, out);
        }
    }
}

/// NEON 2-D AV1 forward transform (4 lanes across every AV1 size and type).
///
/// # Safety
///
/// Caller must guarantee `neon` is available on the running CPU.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn neon(
    tx: Av1TxSize,
    kind: Av1TxType,
    residual: &[i16],
    stride: usize,
    out: &mut [i16],
) {
    // SAFETY: `neon` is enabled and 4 divides both `width` and `coded_height`.
    unsafe { forward_sized::<crate::simd::neon::I32x4>(tx, kind, residual, stride, out) };
}

/// WebAssembly `simd128` 2-D AV1 forward transform (4 lanes across every AV1
/// size and type).
///
/// # Safety
///
/// Caller must guarantee `simd128` is enabled in the target WebAssembly build.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn simd128(
    tx: Av1TxSize,
    kind: Av1TxType,
    residual: &[i16],
    stride: usize,
    out: &mut [i16],
) {
    // SAFETY: `simd128` is enabled and 4 divides both `width` and `coded_height`.
    unsafe { forward_sized::<crate::simd::simd128::I32x4>(tx, kind, residual, stride, out) };
}
