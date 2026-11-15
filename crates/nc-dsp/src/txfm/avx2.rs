//! AVX2 transform kernels: the generic 2-D DCT drivers on 8 lanes (4 for
//! 4x4), the guarded 16-bit inverses (`avx2_inverse`) for 8x8 and larger,
//! and quantisation by exact reciprocal multiplication.
//!
//! Kernel contract: see the `txfm` module docs.

#![allow(
    clippy::inline_always,
    reason = "the storage loads and stores must inline into the ISA kernels"
)]

use core::arch::x86_64::{
    __m256i, _mm_cvtsi32_si128, _mm_loadu_si128, _mm_storeu_si128, _mm256_abs_epi32,
    _mm256_add_epi32, _mm256_add_epi64, _mm256_blend_epi32, _mm256_castsi256_si128,
    _mm256_cmpeq_epi32, _mm256_cvtepi16_epi32, _mm256_loadu_si256, _mm256_min_epi32,
    _mm256_mul_epi32, _mm256_mul_epu32, _mm256_packs_epi32, _mm256_permute4x64_epi64,
    _mm256_set1_epi32, _mm256_set1_epi64x, _mm256_setzero_si256, _mm256_sign_epi32,
    _mm256_sll_epi32, _mm256_slli_epi32, _mm256_slli_epi64, _mm256_srl_epi64, _mm256_srli_epi64,
    _mm256_storeu_si256, _mm256_sub_epi32,
};

use super::scalar::{fix_dc, quantize_tail};
use super::{Av1Quantizer, QuantStorage, Reciprocal, TxfmFns, avx2_inverse, generic, quant_terms};
use crate::simd::avx2::{I32x4, I32x8};

/// The AVX2 transform table.
///
/// # Safety
///
/// The CPU must support AVX2 (`Isa::Avx2.is_supported()`).
pub(crate) unsafe fn txfm_fns() -> TxfmFns {
    TxfmFns {
        forward_dct: [forward_4, forward_8, forward_16, forward_32],
        inverse_dct_add: [
            inverse_4,
            avx2_inverse::inverse_8,
            avx2_inverse::inverse_16,
            avx2_inverse::inverse_32,
        ],
        forward_dct_64: forward_64,
        inverse_dct_add_64: avx2_inverse::inverse_64,
        inverse_dct_add_sparse: [
            avx2_inverse::inverse_16_sparse,
            avx2_inverse::inverse_32_sparse,
            avx2_inverse::inverse_64_sparse,
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

macro_rules! avx2_forward {
    ($($forward:ident, $lanes:ty);*) => {$(
        #[target_feature(enable = "avx2")]
        unsafe fn $forward(residual: &[i16], stride: usize, out: &mut [i16]) {
            // SAFETY: AVX2 is enabled for this function.
            unsafe { generic::$forward::<$lanes>(residual, stride, out) }
        }
    )*};
}

/// The 32-bit 4x4 inverse (below the 16-bit drivers' eight lanes).
macro_rules! avx2_inverse {
    ($($inverse:ident, $lanes:ty);*) => {$(
        #[target_feature(enable = "avx2")]
        unsafe fn $inverse(coeffs: &[i32], dst: &mut [u8], dst_stride: usize) {
            // SAFETY: AVX2 is enabled for this function.
            unsafe { generic::$inverse::<$lanes>(coeffs, dst, dst_stride) }
        }
    )*};
}

/// Raster-storing forward DCTs (VP9's layout, no scratch or transpose).
macro_rules! avx2_raster {
    ($($name:ident, $layout:ident, $lanes:ty);*) => {$(
        #[target_feature(enable = "avx2")]
        unsafe fn $name(residual: &[i16], stride: usize, out: &mut [i16]) {
            // SAFETY: AVX2 is enabled for this function.
            unsafe { generic::$layout::<$lanes, true>(residual, stride, out) }
        }
    )*};
}

avx2_raster!(forward_4_raster, forward_4_layout, I32x4; forward_8_raster, forward_8_layout, I32x8;
             forward_16_raster, forward_16_layout, I32x8;
             forward_32_raster, forward_32_layout, I32x8);

avx2_forward!(forward_4, I32x4; forward_8, I32x8; forward_16, I32x8; forward_32, I32x8);
avx2_inverse!(inverse_4, I32x4);

#[target_feature(enable = "avx2")]
unsafe fn forward_64(residual: &[i16], stride: usize, out: &mut [i16]) {
    // SAFETY: AVX2 is enabled for this function.
    unsafe { generic::forward_64_pruned::<I32x8>(residual, stride, out) }
}

/// Quantiser storage loaded and stored eight 32-bit lanes at a time.
pub(super) trait Lanes8: QuantStorage {
    /// Eight values at `src`, sign-extended to 32 bits.
    ///
    /// # Safety
    ///
    /// AVX2 is enabled; `src` holds eight values.
    unsafe fn load8(src: *const Self) -> __m256i;
    /// Store eight 32-bit lanes (each fitting `Self`) at `dst`.
    ///
    /// # Safety
    ///
    /// AVX2 is enabled; `dst` holds eight values.
    unsafe fn store8(dst: *mut Self, lanes: __m256i);
}

impl Lanes8 for i32 {
    #[inline(always)]
    unsafe fn load8(src: *const Self) -> __m256i {
        // SAFETY: caller contract.
        unsafe { _mm256_loadu_si256(src.cast()) }
    }
    #[inline(always)]
    unsafe fn store8(dst: *mut Self, lanes: __m256i) {
        // SAFETY: caller contract.
        unsafe { _mm256_storeu_si256(dst.cast(), lanes) }
    }
}

impl Lanes8 for i16 {
    #[inline(always)]
    unsafe fn load8(src: *const Self) -> __m256i {
        // SAFETY: caller contract (16 bytes).
        unsafe { _mm256_cvtepi16_epi32(_mm_loadu_si128(src.cast())) }
    }
    #[inline(always)]
    unsafe fn store8(dst: *mut Self, lanes: __m256i) {
        // `packs` narrows within each 128-bit half; the permute gathers
        // the two low quadwords. Exact: every lane fits 16 bits.
        // SAFETY: caller contract (16 bytes).
        unsafe {
            let packed = _mm256_permute4x64_epi64::<0b1000>(_mm256_packs_epi32(lanes, lanes));
            _mm_storeu_si128(dst.cast(), _mm256_castsi256_si128(packed));
        }
    }
}

/// Quantise: an exact multiply-and-shift division (`Reciprocal`) on 8
/// coefficients at a time from coefficient 0 at the AC step, the reference
/// division on the ragged tail, then the DC fix-up (`fix_dc`), from `src`
/// into `dst`.
///
/// # Safety
///
/// `src` and `dst` hold `len` values; they are either disjoint or equal
/// (in place): each vector is loaded before its levels are stored.
#[target_feature(enable = "avx2")]
unsafe fn quantize_levels<T: Lanes8>(
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
    let input_shift = _mm_cvtsi32_si128(7 + q.log_scale as i32);
    let result_shift = _mm_cvtsi32_si128(reciprocal.shift as i32);
    let multiplier = _mm256_set1_epi64x(i64::from(reciprocal.multiplier));
    let offset = _mm256_set1_epi32(offset as i32);
    let max_level = _mm256_set1_epi32(q.max_level);
    let done = len / 8 * 8;
    // Zero levels counted per lane (each all-ones compare mask subtracts
    // -1) and summed once at the end: the generic x86-64 baseline has no
    // POPCNT, so a per-vector `count_ones` of the movemask cost a software
    // popcount (~0.5 % of x86 AV1 Faster instructions, 2026-11-11).
    let mut zeros = _mm256_setzero_si256();
    for i in (0..done).step_by(8) {
        // SAFETY: `i + 8 <= len`; loaded before the store below.
        let c = unsafe { T::load8(src.add(i)) };
        let numerator =
            _mm256_add_epi32(_mm256_sll_epi32(_mm256_abs_epi32(c), input_shift), offset);
        // 32x32 -> 64-bit products of the even and odd lanes, then the shift.
        let even = _mm256_srl_epi64(_mm256_mul_epu32(numerator, multiplier), result_shift);
        let odd = _mm256_srl_epi64(
            _mm256_mul_epu32(_mm256_srli_epi64::<32>(numerator), multiplier),
            result_shift,
        );
        let quotient = _mm256_blend_epi32::<0b1010_1010>(even, _mm256_slli_epi64::<32>(odd));
        let signed = _mm256_sign_epi32(_mm256_min_epi32(quotient, max_level), c);
        // SAFETY: `i + 8 <= len`.
        unsafe { T::store8(dst.add(i), signed) };
        zeros = _mm256_sub_epi32(zeros, _mm256_cmpeq_epi32(signed, _mm256_setzero_si256()));
    }
    let mut lanes = [0_i32; 8];
    // SAFETY: `lanes` holds eight `i32`.
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), zeros) };
    let nonzero = done as u32 - lanes.iter().sum::<i32>() as u32;
    // SAFETY: forwarded contract.
    let nonzero = nonzero + unsafe { quantize_tail(src, dst, done, len, q) };
    if done == 0 {
        // The tail quantised coefficient 0 at the DC step.
        nonzero
    } else {
        // SAFETY: the vector pass wrote `dst[0]`.
        unsafe { fix_dc(dst, dc, nonzero, q) }
    }
}

/// [`quantize_levels`] in place.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn quantize(coeffs: &mut [i16], q: &Av1Quantizer) -> u32 {
    let (len, coeffs) = (coeffs.len(), coeffs.as_mut_ptr());
    // SAFETY: source and destination are the same slice; every value is
    // read before its level is written (`quantize_levels`).
    unsafe { quantize_levels(coeffs.cast_const(), coeffs, len, q) }
}

/// [`quantize_levels`] from `coeffs` into `out`, which keeps `coeffs`.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn quantize_from<T: Lanes8>(
    coeffs: &[T],
    q: &Av1Quantizer,
    out: &mut [T],
) -> u32 {
    let out = &mut out[..coeffs.len()];
    // SAFETY: both slices hold `coeffs.len()` values.
    unsafe { quantize_levels(coeffs.as_ptr(), out.as_mut_ptr(), coeffs.len(), q) }
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn dequantize(levels: &[i16], q: &Av1Quantizer, out: &mut [i32]) {
    // SAFETY: AVX2 is enabled for this function.
    unsafe { generic::dequantize::<I32x8>(levels, q, out) }
}

/// Same bounded factoring as NEON, with signed widening products for even
/// and odd i32 lanes;64-bit accumulators preserve negative gains exactly.
#[target_feature(enable = "avx2")]
unsafe fn coefficient_gain(original: &[i16], dequantized: &[i32]) -> i64 {
    // SAFETY: AVX2 table installed only with ISA support; complete groups
    // are checked before loads. Remaining coefficients use the scalar oracle.
    unsafe {
        let n = original.len().min(dequantized.len());
        let end = n / 8 * 8;
        let mut even = _mm256_setzero_si256();
        let mut odd = _mm256_setzero_si256();
        for i in (0..end).step_by(8) {
            let c = _mm256_cvtepi16_epi32(_mm_loadu_si128(original.as_ptr().add(i).cast()));
            let d = _mm256_loadu_si256(dequantized.as_ptr().add(i).cast());
            let t = _mm256_sub_epi32(_mm256_slli_epi32::<1>(c), d);
            even = _mm256_add_epi64(even, _mm256_mul_epi32(t, d));
            odd = _mm256_add_epi64(
                odd,
                _mm256_mul_epi32(_mm256_srli_epi64::<32>(t), _mm256_srli_epi64::<32>(d)),
            );
        }
        let mut lanes = [0_i64; 4];
        _mm256_storeu_si256(lanes.as_mut_ptr().cast(), _mm256_add_epi64(even, odd));
        lanes.iter().sum::<i64>()
            + super::scalar::coefficient_gain(&original[end..], &dequantized[end..])
    }
}

/// The low-frequency 16-point forward (`generic::forward_16_low`).
#[target_feature(enable = "avx2")]
unsafe fn forward_16_low(residual: &[i16], stride: usize, out: &mut [i16]) {
    // SAFETY: the ISA is enabled for this function; forwarded contract.
    unsafe {
        generic::forward_16_low::<I32x8, { super::low_keep(16) }, false>(residual, stride, out);
    }
}

/// The low-frequency 16-point forward (raster) (`generic::forward_16_low`).
#[target_feature(enable = "avx2")]
unsafe fn forward_16_low_raster(residual: &[i16], stride: usize, out: &mut [i16]) {
    // SAFETY: the ISA is enabled for this function; forwarded contract.
    unsafe {
        generic::forward_16_low::<I32x8, { super::low_keep(16) }, true>(residual, stride, out);
    }
}

/// The low-frequency 32-point forward (`generic::forward_32_low`).
#[target_feature(enable = "avx2")]
unsafe fn forward_32_low(residual: &[i16], stride: usize, out: &mut [i16]) {
    // SAFETY: the ISA is enabled for this function; forwarded contract.
    unsafe {
        generic::forward_32_low::<I32x8, { super::low_keep(32) }, false>(residual, stride, out);
    }
}

/// The low-frequency 64x64 forward (`generic::forward_64_low`).
#[target_feature(enable = "avx2")]
unsafe fn forward_64_low(residual: &[i16], stride: usize, out: &mut [i16]) {
    // SAFETY: the ISA is enabled for this function; forwarded contract.
    unsafe { generic::forward_64_low::<I32x8, { super::low_keep(64) }>(residual, stride, out) }
}

/// The low-frequency 32-point forward (raster) (`generic::forward_32_low`).
#[target_feature(enable = "avx2")]
unsafe fn forward_32_low_raster(residual: &[i16], stride: usize, out: &mut [i16]) {
    // SAFETY: the ISA is enabled for this function; forwarded contract.
    unsafe {
        generic::forward_32_low::<I32x8, { super::low_keep(32) }, true>(residual, stride, out);
    }
}

/// The bounded raster 16-point forward (`generic::forward_16_bounded`).
#[target_feature(enable = "avx2")]
unsafe fn forward_16_bounded(
    residual: &[i16],
    stride: usize,
    zero_limit: u32,
    out: &mut [i16],
) -> bool {
    // SAFETY: the ISA is enabled for this function; forwarded contract.
    unsafe { generic::forward_16_bounded::<I32x8>(residual, stride, zero_limit, out) }
}

/// The bounded raster 32-point forward (`generic::forward_32_bounded`).
#[target_feature(enable = "avx2")]
unsafe fn forward_32_bounded(
    residual: &[i16],
    stride: usize,
    zero_limit: u32,
    out: &mut [i16],
) -> bool {
    // SAFETY: the ISA is enabled for this function; forwarded contract.
    unsafe { generic::forward_32_bounded::<I32x8>(residual, stride, zero_limit, out) }
}
