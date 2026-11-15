//! Scalar reference transform kernels: the generic 2-D drivers on one `i32`
//! lane, and the per-coefficient quantiser definitions.

use super::generic;
use super::{Av1Quantizer, QuantStorage, TxfmFns, dequantize_one, quantize_one};

pub(crate) fn txfm_fns() -> TxfmFns {
    TxfmFns {
        forward_dct: [forward_4, forward_8, forward_16, forward_32],
        inverse_dct_add: [inverse_4, inverse_8, inverse_16, inverse_32],
        forward_dct_64: forward_64,
        inverse_dct_add_64: inverse_64,
        inverse_dct_add_sparse: [inverse_16_sparse, inverse_32_sparse, inverse_64_sparse],
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
        forward_dct_low: [None; 3],
        forward_dct_raster_low: [None; 2],
        forward_dct_raster_bounded: [forward_16_bounded, forward_32_bounded],
    }
}

macro_rules! scalar_sizes {
    ($($forward:ident, $inverse:ident);*) => {$(
        unsafe fn $forward(residual: &[i16], stride: usize, out: &mut [i16]) {
            // SAFETY: the scalar lane type needs no special CPU support.
            unsafe { generic::$forward::<i32>(residual, stride, out) }
        }
        unsafe fn $inverse(coeffs: &[i16], dst: &mut [u8], dst_stride: usize) {
            // SAFETY: as above.
            unsafe { generic::$inverse::<i32>(coeffs, dst, dst_stride) }
        }
    )*};
}

scalar_sizes!(forward_4, inverse_4; forward_8, inverse_8; forward_16, inverse_16;
              forward_32, inverse_32; forward_64, inverse_64);

/// Raster-storing forward DCTs (VP9's layout).
macro_rules! scalar_raster {
    ($($name:ident, $layout:ident);*) => {$(
        unsafe fn $name(residual: &[i16], stride: usize, out: &mut [i16]) {
            // SAFETY: the scalar lane type needs no special CPU support.
            unsafe { generic::$layout::<i32, true>(residual, stride, out) }
        }
    )*};
}

scalar_raster!(forward_4_raster, forward_4_layout; forward_8_raster, forward_8_layout;
               forward_16_raster, forward_16_layout; forward_32_raster, forward_32_layout);

unsafe fn quantize(coeffs: &mut [i16], q: &Av1Quantizer) -> u32 {
    let mut nonzero = 0;
    for (i, c) in coeffs.iter_mut().enumerate() {
        *c = i16::narrow(quantize_one(i32::from(*c), i, q));
        nonzero += u32::from(*c != 0);
    }
    nonzero
}

unsafe fn quantize_from<T: QuantStorage>(coeffs: &[T], q: &Av1Quantizer, out: &mut [T]) -> u32 {
    let mut nonzero = 0;
    for (i, (&c, level)) in coeffs.iter().zip(out.iter_mut()).enumerate() {
        let value = quantize_one(c.widen(), i, q);
        *level = T::narrow(value);
        nonzero += u32::from(value != 0);
    }
    nonzero
}

unsafe fn dequantize(levels: &[i16], q: &Av1Quantizer, out: &mut [i16]) {
    for (i, (&level, dq)) in levels.iter().zip(out.iter_mut()).enumerate() {
        // Clamped to 16 bits (`dequantize_one`): the narrowing is exact.
        *dq = dequantize_one(i32::from(level), i, q) as i16;
    }
}

/// The full reference 16x16 inverse: coefficients outside the low square are
/// zero, so `limit` changes nothing.
unsafe fn inverse_16_sparse(coeffs: &[i16], dst: &mut [u8], stride: usize, _limit: usize) {
    // SAFETY: this table is installed only when its ISA is supported.
    unsafe { inverse_16(coeffs, dst, stride) }
}

unsafe fn inverse_32_sparse(coeffs: &[i16], dst: &mut [u8], stride: usize, limit: usize) {
    // SAFETY: this table is installed only when its ISA is supported.
    unsafe { generic::inverse_32_sparse::<i32>(coeffs, dst, stride, limit) }
}

unsafe fn inverse_64_sparse(coeffs: &[i16], dst: &mut [u8], stride: usize, limit: usize) {
    // SAFETY: this table is installed only when its ISA is supported.
    unsafe { generic::inverse_64_sparse::<i32>(coeffs, dst, stride, limit) }
}

/// Independent scalar oracle: retain the original two-product formula.
pub(crate) unsafe fn coefficient_gain(original: &[i16], dequantized: &[i16]) -> i64 {
    original
        .iter()
        .zip(dequantized)
        .map(|(&c, &d)| {
            let (c, d) = (i64::from(c), i64::from(d));
            2 * c * d - d * d
        })
        .sum()
}

/// The bounded raster 16-point forward (`generic::forward_16_bounded`).
unsafe fn forward_16_bounded(
    residual: &[i16],
    stride: usize,
    zero_limit: u32,
    out: &mut [i16],
) -> bool {
    // SAFETY: the ISA is enabled for this function; forwarded contract.
    unsafe { generic::forward_16_bounded::<i32>(residual, stride, zero_limit, out) }
}

/// The bounded raster 32-point forward (`generic::forward_32_bounded`).
unsafe fn forward_32_bounded(
    residual: &[i16],
    stride: usize,
    zero_limit: u32,
    out: &mut [i16],
) -> bool {
    // SAFETY: the ISA is enabled for this function; forwarded contract.
    unsafe { generic::forward_32_bounded::<i32>(residual, stride, zero_limit, out) }
}

/// The reference quantiser on `src[i]` into `dst[i]` for `i` in
/// `first..len` (the SIMD kernels' ragged tails); returns the number of
/// non-zero levels.
///
/// # Safety
///
/// `src` and `dst` hold `len` values; they are either disjoint or equal
/// (in place).
#[inline]
pub(crate) unsafe fn quantize_tail<T: QuantStorage>(
    src: *const T,
    dst: *mut T,
    first: usize,
    len: usize,
    q: &Av1Quantizer,
) -> u32 {
    let mut nonzero = 0;
    for i in first..len {
        // SAFETY: `i < len` (caller contract); read before written.
        unsafe {
            let level = quantize_one((*src.add(i)).widen(), i, q);
            *dst.add(i) = T::narrow(level);
            nonzero += u32::from(level != 0);
        }
    }
    nonzero
}

/// The SIMD kernels quantise whole vectors from coefficient 0, at the AC
/// step: their loads then match the forward transform's aligned vector
/// stores (store-to-load forwarding) and no AC vector straddles a ragged
/// start. This replaces level 0 by the DC step's level of `dc` (coefficient
/// 0, read before the vector pass, which may run in place) and returns the
/// vector pass's non-zero count `nonzero` corrected for it.
///
/// # Safety
///
/// `dst` holds at least one value, written by the vector pass.
#[inline]
pub(crate) unsafe fn fix_dc<T: QuantStorage>(
    dst: *mut T,
    dc: i32,
    nonzero: u32,
    q: &Av1Quantizer,
) -> u32 {
    // SAFETY: caller contract.
    let at_ac_step = unsafe { (*dst).widen() };
    let level = quantize_one(dc, 0, q);
    // SAFETY: caller contract.
    unsafe { *dst = T::narrow(level) };
    nonzero - u32::from(at_ac_step != 0) + u32::from(level != 0)
}
