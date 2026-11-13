//! Scalar reference transform kernels: the generic 2-D drivers on one `i32`
//! lane, and the per-coefficient quantiser definitions.

use super::generic;
use super::{Av1Quantizer, TxfmFns, dequantize_one, quantize_one};

pub(crate) fn txfm_fns() -> TxfmFns {
    TxfmFns {
        forward_dct: [forward_4, forward_8, forward_16, forward_32],
        inverse_dct_add: [inverse_4, inverse_8, inverse_16, inverse_32],
        forward_dct_64: forward_64,
        inverse_dct_add_64: inverse_64,
        inverse_dct_add_sparse: [inverse_32_sparse, inverse_64_sparse],
        coefficient_gain,
        quantize,
        quantize_from,
        dequantize,
        transpose: [
            transpose::<4>,
            transpose::<8>,
            transpose::<16>,
            transpose::<32>,
        ],
        forward_dct_raster: [None; 4],
        forward_dct_low: [None; 3],
        forward_dct_raster_low: [None; 2],
        forward_dct_raster_bounded: [forward_16_bounded, forward_32_bounded],
    }
}

pub(crate) unsafe fn transpose<const N: usize>(src: &[i32], dst: &mut [i32]) {
    generic::transpose::<i32, N>(src, dst);
}

macro_rules! scalar_sizes {
    ($($forward:ident, $inverse:ident);*) => {$(
        unsafe fn $forward(residual: &[i16], stride: usize, out: &mut [i32]) {
            // SAFETY: the scalar lane type needs no special CPU support.
            unsafe { generic::$forward::<i32>(residual, stride, out) }
        }
        unsafe fn $inverse(coeffs: &[i32], dst: &mut [u8], dst_stride: usize) {
            // SAFETY: as above.
            unsafe { generic::$inverse::<i32>(coeffs, dst, dst_stride) }
        }
    )*};
}

scalar_sizes!(forward_4, inverse_4; forward_8, inverse_8; forward_16, inverse_16;
              forward_32, inverse_32; forward_64, inverse_64);

unsafe fn quantize(coeffs: &mut [i32], q: &Av1Quantizer) -> u32 {
    let mut nonzero = 0;
    for (i, c) in coeffs.iter_mut().enumerate() {
        *c = quantize_one(*c, i, q);
        nonzero += u32::from(*c != 0);
    }
    nonzero
}

unsafe fn quantize_from(coeffs: &[i32], q: &Av1Quantizer, out: &mut [i32]) -> u32 {
    let mut nonzero = 0;
    for (i, (&c, level)) in coeffs.iter().zip(out.iter_mut()).enumerate() {
        *level = quantize_one(c, i, q);
        nonzero += u32::from(*level != 0);
    }
    nonzero
}

unsafe fn dequantize(levels: &[i32], q: &Av1Quantizer, out: &mut [i32]) {
    for (i, (&level, dq)) in levels.iter().zip(out.iter_mut()).enumerate() {
        *dq = dequantize_one(level, i, q);
    }
}

unsafe fn inverse_32_sparse(coeffs: &[i32], dst: &mut [u8], stride: usize, limit: usize) {
    // SAFETY: this table is installed only when its ISA is supported.
    unsafe { generic::inverse_32_sparse::<i32>(coeffs, dst, stride, limit) }
}

unsafe fn inverse_64_sparse(coeffs: &[i32], dst: &mut [u8], stride: usize, limit: usize) {
    // SAFETY: this table is installed only when its ISA is supported.
    unsafe { generic::inverse_64_sparse::<i32>(coeffs, dst, stride, limit) }
}

/// Independent scalar oracle: retain the original two-product formula.
pub(crate) unsafe fn coefficient_gain(original: &[i32], dequantized: &[i32]) -> i64 {
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
    out: &mut [i32],
) -> bool {
    // SAFETY: the ISA is enabled for this function; forwarded contract.
    unsafe { generic::forward_16_bounded::<i32>(residual, stride, zero_limit, out) }
}

/// The bounded raster 32-point forward (`generic::forward_32_bounded`).
unsafe fn forward_32_bounded(
    residual: &[i16],
    stride: usize,
    zero_limit: u32,
    out: &mut [i32],
) -> bool {
    // SAFETY: the ISA is enabled for this function; forwarded contract.
    unsafe { generic::forward_32_bounded::<i32>(residual, stride, zero_limit, out) }
}
