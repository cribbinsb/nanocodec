//! WebAssembly SIMD128 transform kernels: the generic 2-D DCT drivers on 4
//! lanes, the guarded 16-bit inverses (`simd128_inverse`) for 8x8 and
//! larger, and quantisation by exact reciprocal multiplication.
//!
//! Kernel contract: see the `txfm` module docs. This module only exists in
//! builds with SIMD128 enabled, so its kernels need no run-time check.

use core::arch::wasm32::{
    i32x4_abs, i32x4_add, i32x4_bitmask, i32x4_eq, i32x4_shl, i32x4_shr, i32x4_shuffle,
    i32x4_splat, i32x4_sub, i64x2_add, i64x2_extmul_high_i32x4, i64x2_extmul_low_i32x4,
    i64x2_extract_lane, i64x2_splat, u32x4_min, u32x4_splat, u64x2_extmul_high_u32x4,
    u64x2_extmul_low_u32x4, u64x2_shr, v128_load, v128_store, v128_xor,
};

use super::{
    Av1Quantizer, Reciprocal, TxfmFns, generic, quant_terms, quantize_one, simd128_inverse,
};
use crate::simd::simd128::I32x4;

/// The SIMD128 transform table.
pub(crate) fn txfm_fns() -> TxfmFns {
    TxfmFns {
        forward_dct: [forward_4, forward_8, forward_16, forward_32],
        inverse_dct_add: [
            inverse_4,
            simd128_inverse::inverse_8,
            simd128_inverse::inverse_16,
            simd128_inverse::inverse_32,
        ],
        forward_dct_64: forward_64,
        inverse_dct_add_64: simd128_inverse::inverse_64,
        inverse_dct_add_sparse: [
            simd128_inverse::inverse_32_sparse,
            simd128_inverse::inverse_64_sparse,
        ],
        coefficient_gain,
        quantize,
        quantize_from,
        dequantize,
        // A single 4x4 tile is slower in vectors than scalar (measured).
        transpose: [
            super::scalar::transpose::<4>,
            transpose::<8>,
            transpose::<16>,
            transpose::<32>,
        ],
        forward_dct_raster: [
            None,
            Some(forward_8_raster),
            Some(forward_16_raster),
            Some(forward_32_raster),
        ],
        forward_dct_low: [
            Some(forward_16_low),
            Some(forward_32_low),
            Some(forward_64_low),
        ],
        forward_dct_raster_low: [Some(forward_16_low_raster), Some(forward_32_low_raster)],
        forward_dct_raster_bounded: [forward_16_bounded, forward_32_bounded],
    }
}

/// As the NEON kernel: `|c| < 2^22` and `d` is signed 16-bit, so
/// `|2c - d| < 2^24`, each product `(2c - d) · d` is below `2^40` and a
/// 1024-term sum below `2^50`; the products widen into two 64-bit lanes per
/// half (`i64x2.extmul`), exactly as the scalar oracle sums in `i64`.
///
/// # Safety
///
/// Kernel contract (`txfm` module docs).
#[target_feature(enable = "simd128")]
unsafe fn coefficient_gain(original: &[i32], dequantized: &[i32]) -> i64 {
    let n = original.len().min(dequantized.len());
    let (original, dequantized) = (&original[..n], &dequantized[..n]);
    let end = n / 4 * 4;
    let mut sum = i64x2_splat(0);
    for i in (0..end).step_by(4) {
        // SAFETY: i + 4 <= end <= n, so both 4-lane loads are in bounds.
        let (c, d) = unsafe {
            (
                v128_load(original[i..].as_ptr().cast()),
                v128_load(dequantized[i..].as_ptr().cast()),
            )
        };
        let t = i32x4_sub(i32x4_shl(c, 1), d);
        sum = i64x2_add(
            sum,
            i64x2_add(i64x2_extmul_low_i32x4(t, d), i64x2_extmul_high_i32x4(t, d)),
        );
    }
    // SAFETY: the scalar oracle has no ISA requirement.
    let tail = unsafe { super::scalar::coefficient_gain(&original[end..], &dequantized[end..]) };
    i64x2_extract_lane::<0>(sum) + i64x2_extract_lane::<1>(sum) + tail
}

unsafe fn transpose<const N: usize>(src: &[i32], dst: &mut [i32]) {
    generic::transpose::<I32x4, N>(src, dst);
}

macro_rules! simd128_sizes {
    ($($forward:ident, $inverse:ident);*) => {$(
        unsafe fn $forward(residual: &[i16], stride: usize, out: &mut [i32]) {
            // SAFETY: SIMD128 is enabled for the whole build.
            unsafe { generic::$forward::<I32x4>(residual, stride, out) }
        }

        unsafe fn $inverse(coeffs: &[i32], dst: &mut [u8], dst_stride: usize) {
            // SAFETY: SIMD128 is enabled for the whole build.
            unsafe { generic::$inverse::<I32x4>(coeffs, dst, dst_stride) }
        }
    )*};
}

/// Raster-storing forward DCTs (VP9's layout, no scratch or transpose).
macro_rules! simd128_raster {
    ($($name:ident, $layout:ident);*) => {$(
        unsafe fn $name(residual: &[i16], stride: usize, out: &mut [i32]) {
            // SAFETY: SIMD128 is enabled for the whole build.
            unsafe { generic::$layout::<I32x4, true>(residual, stride, out) }
        }
    )*};
}

simd128_raster!(forward_8_raster, forward_8_layout; forward_16_raster, forward_16_layout;
                forward_32_raster, forward_32_layout);

simd128_sizes!(forward_4, inverse_4);

/// Forward DCTs of the sizes whose inverses run on 16-bit lanes
/// (`simd128_inverse`).
macro_rules! simd128_forward {
    ($($forward:ident),*) => {$(
        unsafe fn $forward(residual: &[i16], stride: usize, out: &mut [i32]) {
            // SAFETY: SIMD128 is enabled for the whole build.
            unsafe { generic::$forward::<I32x4>(residual, stride, out) }
        }
    )*};
}

simd128_forward!(forward_8, forward_16, forward_32);

unsafe fn forward_64(residual: &[i16], stride: usize, out: &mut [i32]) {
    // SAFETY: SIMD128 is enabled for the whole build.
    unsafe { generic::forward_64_pruned::<I32x4>(residual, stride, out) }
}

/// Quantise: the reference division on coefficient 0 and on ragged tails,
/// and an exact multiply-and-shift division (`Reciprocal`) on 4 AC
/// coefficients at a time, from `src` into `dst`.
///
/// # Safety
///
/// `src` and `dst` hold `len` values; they are either disjoint or equal
/// (in place): each vector is loaded before its levels are stored.
unsafe fn quantize_levels(src: *const i32, dst: *mut i32, len: usize, q: &Av1Quantizer) -> u32 {
    if len == 0 {
        return 0;
    }
    // SAFETY: `len >= 1` (caller contract for both pointers).
    let dc = unsafe { quantize_one(*src, 0, q) };
    // SAFETY: as above.
    unsafe { *dst = dc };
    let mut nonzero = u32::from(dc != 0);
    let (offset, divisor) = quant_terms(q, q.ac_step);
    let Some(reciprocal) = Reciprocal::new(divisor) else {
        // SAFETY: forwarded contract.
        return nonzero + unsafe { quantize_tail(src, dst, 1, len, q) };
    };
    let input_shift = 7 + q.log_scale;
    let multiplier = u32x4_splat(reciprocal.multiplier);
    let offset = u32x4_splat(offset);
    let max_level = u32x4_splat(q.max_level as u32);
    let done = 1 + (len - 1) / 4 * 4;
    for i in (1..done).step_by(4) {
        // SAFETY: `i + 4 <= len` (16 bytes); loaded before the store below.
        let c = unsafe { v128_load(src.add(i).cast()) };
        let numerator = i32x4_add(i32x4_shl(i32x4_abs(c), input_shift), offset);
        let low = u64x2_shr(
            u64x2_extmul_low_u32x4(numerator, multiplier),
            reciprocal.shift,
        );
        let high = u64x2_shr(
            u64x2_extmul_high_u32x4(numerator, multiplier),
            reciprocal.shift,
        );
        let level = u32x4_min(i32x4_shuffle::<0, 2, 4, 6>(low, high), max_level);
        // Negate where c < 0: (level ^ m) - m with m = c >> 31.
        let sign = i32x4_shr(c, 31);
        let signed = i32x4_sub(v128_xor(level, sign), sign);
        // SAFETY: `i + 4 <= len`.
        unsafe { v128_store(dst.add(i).cast(), signed) };
        nonzero += 4 - u32::from(i32x4_bitmask(i32x4_eq(signed, i32x4_splat(0)))).count_ones();
    }
    // SAFETY: forwarded contract.
    nonzero + unsafe { quantize_tail(src, dst, done, len, q) }
}

/// [`quantize_levels`] in place.
unsafe fn quantize(coeffs: &mut [i32], q: &Av1Quantizer) -> u32 {
    let (len, coeffs) = (coeffs.len(), coeffs.as_mut_ptr());
    // SAFETY: source and destination are the same slice; every value is
    // read before its level is written (`quantize_levels`).
    unsafe { quantize_levels(coeffs.cast_const(), coeffs, len, q) }
}

/// [`quantize_levels`] from `coeffs` into `out`, which keeps `coeffs`.
unsafe fn quantize_from(coeffs: &[i32], q: &Av1Quantizer, out: &mut [i32]) -> u32 {
    let out = &mut out[..coeffs.len()];
    // SAFETY: both slices hold `coeffs.len()` values.
    unsafe { quantize_levels(coeffs.as_ptr(), out.as_mut_ptr(), coeffs.len(), q) }
}

/// The reference quantiser on `src[i]` into `dst[i]` for `i` in
/// `first..len`; returns the non-zero count.
///
/// # Safety
///
/// As [`quantize_levels`].
unsafe fn quantize_tail(
    src: *const i32,
    dst: *mut i32,
    first: usize,
    len: usize,
    q: &Av1Quantizer,
) -> u32 {
    let mut nonzero = 0;
    for i in first..len {
        // SAFETY: `i < len` (caller contract).
        unsafe {
            let level = quantize_one(*src.add(i), i, q);
            *dst.add(i) = level;
            nonzero += u32::from(level != 0);
        }
    }
    nonzero
}

#[target_feature(enable = "simd128")]
unsafe fn dequantize(levels: &[i32], q: &Av1Quantizer, out: &mut [i32]) {
    // SAFETY: SIMD128 is enabled for this function.
    unsafe { generic::dequantize::<I32x4>(levels, q, out) }
}

/// The low-frequency 16-point forward (`generic::forward_16_low`).
unsafe fn forward_16_low(residual: &[i16], stride: usize, out: &mut [i32]) {
    // SAFETY: the ISA is enabled for this function; forwarded contract.
    unsafe {
        generic::forward_16_low::<I32x4, { super::low_keep(16) }, false>(residual, stride, out);
    }
}

/// The low-frequency 16-point forward (raster) (`generic::forward_16_low`).
unsafe fn forward_16_low_raster(residual: &[i16], stride: usize, out: &mut [i32]) {
    // SAFETY: the ISA is enabled for this function; forwarded contract.
    unsafe {
        generic::forward_16_low::<I32x4, { super::low_keep(16) }, true>(residual, stride, out);
    }
}

/// The low-frequency 32-point forward (`generic::forward_32_low`).
unsafe fn forward_32_low(residual: &[i16], stride: usize, out: &mut [i32]) {
    // SAFETY: the ISA is enabled for this function; forwarded contract.
    unsafe {
        generic::forward_32_low::<I32x4, { super::low_keep(32) }, false>(residual, stride, out);
    }
}

/// The low-frequency 64x64 forward (`generic::forward_64_low`).
unsafe fn forward_64_low(residual: &[i16], stride: usize, out: &mut [i32]) {
    // SAFETY: the ISA is enabled for this function; forwarded contract.
    unsafe { generic::forward_64_low::<I32x4, { super::low_keep(64) }>(residual, stride, out) }
}

/// The low-frequency 32-point forward (raster) (`generic::forward_32_low`).
unsafe fn forward_32_low_raster(residual: &[i16], stride: usize, out: &mut [i32]) {
    // SAFETY: the ISA is enabled for this function; forwarded contract.
    unsafe {
        generic::forward_32_low::<I32x4, { super::low_keep(32) }, true>(residual, stride, out);
    }
}

/// The bounded raster 16-point forward (`generic::forward_16_bounded`).
unsafe fn forward_16_bounded(
    residual: &[i16],
    stride: usize,
    zero_limit: u32,
    out: &mut [i32],
) -> bool {
    // SAFETY: the ISA is enabled for this function; forwarded contract.
    unsafe { generic::forward_16_bounded::<I32x4>(residual, stride, zero_limit, out) }
}

/// The bounded raster 32-point forward (`generic::forward_32_bounded`).
unsafe fn forward_32_bounded(
    residual: &[i16],
    stride: usize,
    zero_limit: u32,
    out: &mut [i32],
) -> bool {
    // SAFETY: the ISA is enabled for this function; forwarded contract.
    unsafe { generic::forward_32_bounded::<I32x4>(residual, stride, zero_limit, out) }
}
