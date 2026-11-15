//! WebAssembly SIMD128 transform kernels: the generic 2-D DCT drivers on 4
//! lanes, the guarded 16-bit inverses (`simd128_inverse`) for 8x8 and
//! larger, and quantisation by exact reciprocal multiplication.
//!
//! Kernel contract: see the `txfm` module docs. This module only exists in
//! builds with SIMD128 enabled, so its kernels need no run-time check.

#![allow(
    clippy::inline_always,
    reason = "the storage loads and stores must inline into the ISA kernels"
)]

use core::arch::wasm32::{
    i16x8_narrow_i32x4, i32x4_abs, i32x4_add, i32x4_bitmask, i32x4_eq, i32x4_extend_high_i16x8,
    i32x4_extend_low_i16x8, i32x4_shl, i32x4_shr, i32x4_shuffle, i32x4_splat, i32x4_sub, i64x2_add,
    i64x2_extmul_high_i32x4, i64x2_extmul_low_i32x4, i64x2_extract_lane, i64x2_splat, u32x4_min,
    u32x4_splat, u64x2_extmul_high_u32x4, u64x2_extmul_low_u32x4, u64x2_shr, v128, v128_load,
    v128_load64_zero, v128_store, v128_store64_lane, v128_xor,
};

use super::scalar::{fix_dc, quantize_tail};
use super::{
    Av1Quantizer, QuantStorage, Reciprocal, TxfmFns, generic, quant_terms, simd128_inverse,
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
            simd128_inverse::inverse_16_sparse,
            simd128_inverse::inverse_32_sparse,
            simd128_inverse::inverse_64_sparse,
        ],
        coefficient_gain,
        quantize,
        quantize_from: quantize_from::<i16>,
        quantize_from_wide: quantize_from::<i32>,
        dequantize,
        forward_dct_raster: [
            forward_4_raster,
            forward_8_raster,
            forward_16_raster,
            forward_32_raster,
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
unsafe fn coefficient_gain(original: &[i16], dequantized: &[i16]) -> i64 {
    let n = original.len().min(dequantized.len());
    let (original, dequantized) = (&original[..n], &dequantized[..n]);
    let end = n / 4 * 4;
    let mut sum = i64x2_splat(0);
    for i in (0..end).step_by(4) {
        // SAFETY: i + 4 <= end <= n, so both 4-lane loads are in bounds.
        let (c, d) = unsafe {
            (
                i32x4_extend_low_i16x8(v128_load64_zero(original[i..].as_ptr().cast())),
                i32x4_extend_low_i16x8(v128_load64_zero(dequantized[i..].as_ptr().cast())),
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

macro_rules! simd128_sizes {
    ($($forward:ident, $inverse:ident);*) => {$(
        unsafe fn $forward(residual: &[i16], stride: usize, out: &mut [i16]) {
            // SAFETY: SIMD128 is enabled for the whole build.
            unsafe { generic::$forward::<I32x4>(residual, stride, out) }
        }

        unsafe fn $inverse(coeffs: &[i16], dst: &mut [u8], dst_stride: usize) {
            // SAFETY: SIMD128 is enabled for the whole build.
            unsafe { generic::$inverse::<I32x4>(coeffs, dst, dst_stride) }
        }
    )*};
}

/// Raster-storing forward DCTs (VP9's layout, no scratch or transpose).
macro_rules! simd128_raster {
    ($($name:ident, $layout:ident);*) => {$(
        unsafe fn $name(residual: &[i16], stride: usize, out: &mut [i16]) {
            // SAFETY: SIMD128 is enabled for the whole build.
            unsafe { generic::$layout::<I32x4, true>(residual, stride, out) }
        }
    )*};
}

simd128_raster!(forward_4_raster, forward_4_layout; forward_8_raster, forward_8_layout; forward_16_raster, forward_16_layout;
                forward_32_raster, forward_32_layout);

simd128_sizes!(forward_4, inverse_4);

/// Forward DCTs of the sizes whose inverses run on 16-bit lanes
/// (`simd128_inverse`).
macro_rules! simd128_forward {
    ($($forward:ident),*) => {$(
        unsafe fn $forward(residual: &[i16], stride: usize, out: &mut [i16]) {
            // SAFETY: SIMD128 is enabled for the whole build.
            unsafe { generic::$forward::<I32x4>(residual, stride, out) }
        }
    )*};
}

simd128_forward!(forward_8, forward_16, forward_32);

unsafe fn forward_64(residual: &[i16], stride: usize, out: &mut [i16]) {
    // SAFETY: SIMD128 is enabled for the whole build.
    unsafe { generic::forward_64_pruned::<I32x4>(residual, stride, out) }
}

/// Quantiser storage loaded and stored as two four-lane 32-bit halves (then
/// one half for a short tail).
trait Lanes: QuantStorage {
    /// Eight values at `src`, sign-extended to 32 bits.
    ///
    /// # Safety
    ///
    /// `src` holds eight values.
    unsafe fn load8(src: *const Self) -> (v128, v128);
    /// Store two four-lane halves (each lane fitting `Self`) at `dst`.
    ///
    /// # Safety
    ///
    /// `dst` holds eight values.
    unsafe fn store8(dst: *mut Self, lanes: (v128, v128));
    /// Four values at `src`, sign-extended.
    ///
    /// # Safety
    ///
    /// `src` holds four values.
    unsafe fn load4(src: *const Self) -> v128;
    /// Store four lanes (each fitting `Self`) at `dst`.
    ///
    /// # Safety
    ///
    /// `dst` holds four values.
    unsafe fn store4(dst: *mut Self, lanes: v128);
}

impl Lanes for i32 {
    #[inline(always)]
    unsafe fn load8(src: *const Self) -> (v128, v128) {
        // SAFETY: caller contract.
        unsafe { (v128_load(src.cast()), v128_load(src.add(4).cast())) }
    }
    #[inline(always)]
    unsafe fn store8(dst: *mut Self, (low, high): (v128, v128)) {
        // SAFETY: caller contract.
        unsafe {
            v128_store(dst.cast(), low);
            v128_store(dst.add(4).cast(), high);
        }
    }
    #[inline(always)]
    unsafe fn load4(src: *const Self) -> v128 {
        // SAFETY: caller contract.
        unsafe { v128_load(src.cast()) }
    }
    #[inline(always)]
    unsafe fn store4(dst: *mut Self, lanes: v128) {
        // SAFETY: caller contract.
        unsafe { v128_store(dst.cast(), lanes) }
    }
}

impl Lanes for i16 {
    #[inline(always)]
    unsafe fn load8(src: *const Self) -> (v128, v128) {
        // SAFETY: caller contract: one 128-bit load.
        let values = unsafe { v128_load(src.cast()) };
        (
            i32x4_extend_low_i16x8(values),
            i32x4_extend_high_i16x8(values),
        )
    }
    #[inline(always)]
    unsafe fn store8(dst: *mut Self, (low, high): (v128, v128)) {
        // One narrow and one 128-bit store. Exact: every lane fits 16 bits.
        // SAFETY: caller contract.
        unsafe { v128_store(dst.cast(), i16x8_narrow_i32x4(low, high)) }
    }
    #[inline(always)]
    unsafe fn load4(src: *const Self) -> v128 {
        // SAFETY: caller contract (8 bytes).
        unsafe { i32x4_extend_low_i16x8(v128_load64_zero(src.cast())) }
    }
    #[inline(always)]
    unsafe fn store4(dst: *mut Self, lanes: v128) {
        // SAFETY: caller contract (8 bytes); exact as in `store8`.
        unsafe { v128_store64_lane::<0>(i16x8_narrow_i32x4(lanes, lanes), dst.cast()) }
    }
}

/// The AC quantiser as lane constants: `level = min(numerator / divisor,
/// max_level)`, signed like the coefficient, with the division done by the
/// exact reciprocal.
struct AcQuantizer {
    input_shift: u32,
    offset: v128,
    multiplier: v128,
    shift: u32,
    max_level: v128,
}

impl AcQuantizer {
    /// The signed levels of four coefficients.
    #[inline(always)]
    fn level(&self, c: v128) -> v128 {
        let numerator = i32x4_add(i32x4_shl(i32x4_abs(c), self.input_shift), self.offset);
        let low = u64x2_shr(
            u64x2_extmul_low_u32x4(numerator, self.multiplier),
            self.shift,
        );
        let high = u64x2_shr(
            u64x2_extmul_high_u32x4(numerator, self.multiplier),
            self.shift,
        );
        let level = u32x4_min(i32x4_shuffle::<0, 2, 4, 6>(low, high), self.max_level);
        // Negate where c < 0: (level ^ m) - m with m = c >> 31.
        let sign = i32x4_shr(c, 31);
        i32x4_sub(v128_xor(level, sign), sign)
    }
}

/// Nonzero lanes of four levels.
#[inline(always)]
fn count_nonzero(levels: v128) -> u32 {
    4 - u32::from(i32x4_bitmask(i32x4_eq(levels, i32x4_splat(0)))).count_ones()
}

/// Quantise: an exact multiply-and-shift division (`Reciprocal`) on 8
/// coefficients at a time (then 4) from coefficient 0 at the AC step, the
/// reference division on the ragged tail, then the DC fix-up (`fix_dc`),
/// from `src` into `dst`.
///
/// # Safety
///
/// `src` and `dst` hold `len` values; they are either disjoint or equal
/// (in place): each vector is loaded before its levels are stored.
unsafe fn quantize_levels<T: Lanes>(
    src: *const T,
    dst: *mut T,
    len: usize,
    q: &Av1Quantizer,
) -> u32 {
    let (offset, divisor) = quant_terms(q, q.ac_step);
    let Some(reciprocal) = Reciprocal::new(divisor) else {
        // SAFETY: forwarded contract.
        return unsafe { quantize_tail(src, dst, 0, len, q) };
    };
    if len == 0 {
        return 0;
    }
    // SAFETY: `len >= 1`; read before the vector pass overwrites it.
    let dc = unsafe { (*src).widen() };
    let mut nonzero = 0;
    let ac = AcQuantizer {
        input_shift: 7 + q.log_scale,
        offset: u32x4_splat(offset),
        multiplier: u32x4_splat(reciprocal.multiplier),
        shift: reciprocal.shift,
        max_level: u32x4_splat(q.max_level as u32),
    };
    let octets_end = len / 8 * 8;
    for i in (0..octets_end).step_by(8) {
        // SAFETY: `i + 8 <= len`; loaded before the store below.
        let (c0, c1) = unsafe { T::load8(src.add(i)) };
        let levels = (ac.level(c0), ac.level(c1));
        // SAFETY: as above.
        unsafe { T::store8(dst.add(i), levels) };
        nonzero += count_nonzero(levels.0) + count_nonzero(levels.1);
    }
    let mut done = octets_end;
    if done + 4 <= len {
        // SAFETY: `done + 4 <= len`; loaded before the store below.
        let level = ac.level(unsafe { T::load4(src.add(done)) });
        // SAFETY: as above.
        unsafe { T::store4(dst.add(done), level) };
        nonzero += count_nonzero(level);
        done += 4;
    }
    // SAFETY: forwarded contract.
    nonzero += unsafe { quantize_tail(src, dst, done, len, q) };
    if done == 0 {
        // The tail quantised coefficient 0 at the DC step.
        nonzero
    } else {
        // SAFETY: the vector pass wrote `dst[0]`.
        unsafe { fix_dc(dst, dc, nonzero, q) }
    }
}

/// [`quantize_levels`] in place.
unsafe fn quantize(coeffs: &mut [i16], q: &Av1Quantizer) -> u32 {
    let (len, coeffs) = (coeffs.len(), coeffs.as_mut_ptr());
    // SAFETY: source and destination are the same slice; every value is
    // read before its level is written (`quantize_levels`).
    unsafe { quantize_levels(coeffs.cast_const(), coeffs, len, q) }
}

/// [`quantize_levels`] from `coeffs` into `out`, which keeps `coeffs`.
unsafe fn quantize_from<T: Lanes>(coeffs: &[T], q: &Av1Quantizer, out: &mut [T]) -> u32 {
    let out = &mut out[..coeffs.len()];
    // SAFETY: both slices hold `coeffs.len()` values.
    unsafe { quantize_levels(coeffs.as_ptr(), out.as_mut_ptr(), coeffs.len(), q) }
}

#[target_feature(enable = "simd128")]
unsafe fn dequantize(levels: &[i16], q: &Av1Quantizer, out: &mut [i16]) {
    // SAFETY: SIMD128 is enabled for this function.
    unsafe { generic::dequantize::<I32x4>(levels, q, out) }
}

/// The low-frequency 16-point forward (`generic::forward_16_low`).
unsafe fn forward_16_low(residual: &[i16], stride: usize, out: &mut [i16]) {
    // SAFETY: the ISA is enabled for this function; forwarded contract.
    unsafe {
        generic::forward_16_low::<I32x4, { super::low_keep(16) }, false>(residual, stride, out);
    }
}

/// The low-frequency 16-point forward (raster) (`generic::forward_16_low`).
unsafe fn forward_16_low_raster(residual: &[i16], stride: usize, out: &mut [i16]) {
    // SAFETY: the ISA is enabled for this function; forwarded contract.
    unsafe {
        generic::forward_16_low::<I32x4, { super::low_keep(16) }, true>(residual, stride, out);
    }
}

/// The low-frequency 32-point forward (`generic::forward_32_low`).
unsafe fn forward_32_low(residual: &[i16], stride: usize, out: &mut [i16]) {
    // SAFETY: the ISA is enabled for this function; forwarded contract.
    unsafe {
        generic::forward_32_low::<I32x4, { super::low_keep(32) }, false>(residual, stride, out);
    }
}

/// The low-frequency 64x64 forward (`generic::forward_64_low`).
unsafe fn forward_64_low(residual: &[i16], stride: usize, out: &mut [i16]) {
    // SAFETY: the ISA is enabled for this function; forwarded contract.
    unsafe { generic::forward_64_low::<I32x4, { super::low_keep(64) }>(residual, stride, out) }
}

/// The low-frequency 32-point forward (raster) (`generic::forward_32_low`).
unsafe fn forward_32_low_raster(residual: &[i16], stride: usize, out: &mut [i16]) {
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
    out: &mut [i16],
) -> bool {
    // SAFETY: the ISA is enabled for this function; forwarded contract.
    unsafe { generic::forward_16_bounded::<I32x4>(residual, stride, zero_limit, out) }
}

/// The bounded raster 32-point forward (`generic::forward_32_bounded`).
unsafe fn forward_32_bounded(
    residual: &[i16],
    stride: usize,
    zero_limit: u32,
    out: &mut [i16],
) -> bool {
    // SAFETY: the ISA is enabled for this function; forwarded contract.
    unsafe { generic::forward_32_bounded::<I32x4>(residual, stride, zero_limit, out) }
}
