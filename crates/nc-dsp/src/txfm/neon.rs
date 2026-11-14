//! NEON transform kernels: the generic 2-D DCT drivers on 4 lanes for 4x4,
//! guarded 16-bit forward (`neon_forward`) and inverse (`neon_inverse`)
//! drivers for 8x8 and larger, and quantisation by exact reciprocal
//! multiplication.
//!
//! Kernel contract: see the `txfm` module docs.

#![allow(clippy::inline_always)]

use core::arch::aarch64::{
    int32x4_t, uint32x4_t, vabsq_s32, vaddq_s64, vaddq_u32, vaddvq_s64, vaddvq_u32, vdupq_n_s32,
    vdupq_n_s64, vdupq_n_u32, veorq_s32, vget_low_s32, vget_low_u32, vld1q_s32, vmaxq_u32,
    vmaxvq_u32, vminq_u32, vmlal_high_s32, vmlal_s32, vmull_high_u32, vmull_u32,
    vreinterpretq_s32_u32, vreinterpretq_u32_s32, vreinterpretq_u32_u64, vshlq_n_s32, vshlq_u32,
    vshrq_n_s32, vst1q_s32, vsubq_s32, vsubq_u32, vtstq_s32, vuzp2q_u32,
};

use super::{Av1Quantizer, Reciprocal, TxfmFns, generic, quant_terms, quantize_one};
use crate::simd::neon::I32x4;

/// The NEON transform table (NEON is always present on `AArch64`).
pub(crate) fn txfm_fns() -> TxfmFns {
    TxfmFns {
        forward_dct: [
            forward_4,
            super::neon_forward::forward_8,
            super::neon_forward::forward_16,
            super::neon_forward::forward_32,
        ],
        inverse_dct_add: [
            inverse_4,
            super::neon_inverse::inverse_8,
            super::neon_inverse::inverse_16,
            super::neon_inverse::inverse_32,
        ],
        forward_dct_64: super::neon_forward::forward_64,
        inverse_dct_add_64: super::neon_inverse::inverse_64,
        inverse_dct_add_sparse: [
            super::neon_inverse::inverse_32_sparse,
            super::neon_inverse::inverse_64_sparse,
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
            Some(super::neon_forward::forward_8_raster),
            Some(super::neon_forward::forward_16_raster),
            Some(super::neon_forward::forward_32_raster),
        ],
        forward_dct_low: [
            Some(super::neon_forward::forward_16_low),
            Some(super::neon_forward::forward_32_low),
            Some(super::neon_forward::forward_64_low),
        ],
        forward_dct_raster_low: [
            Some(super::neon_forward::forward_16_low_raster),
            Some(super::neon_forward::forward_32_low_raster),
        ],
        forward_dct_raster_bounded: [
            super::neon_forward::forward_16_bounded,
            super::neon_forward::forward_32_bounded,
        ],
    }
}

#[target_feature(enable = "neon")]
unsafe fn transpose<const N: usize>(src: &[i32], dst: &mut [i32]) {
    generic::transpose::<I32x4, N>(src, dst);
}

macro_rules! neon_sizes {
    ($($forward:ident, $inverse:ident);*) => {$(
        #[target_feature(enable = "neon")]
        unsafe fn $forward(residual: &[i16], stride: usize, out: &mut [i32]) {
            // SAFETY: NEON is enabled for this function.
            unsafe { generic::$forward::<I32x4>(residual, stride, out) }
        }

        #[target_feature(enable = "neon")]
        unsafe fn $inverse(coeffs: &[i32], dst: &mut [u8], dst_stride: usize) {
            // SAFETY: NEON is enabled for this function.
            unsafe { generic::$inverse::<I32x4>(coeffs, dst, dst_stride) }
        }
    )*};
}

neon_sizes!(forward_4, inverse_4);

/// Quantise: the reference division on coefficient 0 and on ragged tails,
/// and an exact multiply-and-shift division (`Reciprocal`) on 8 AC
/// coefficients at a time (then 4), from `src` into `dst`.
///
/// # Safety
///
/// `src` and `dst` hold `len` values; they are either disjoint or equal
/// (in place): each vector is loaded before its levels are stored.
#[target_feature(enable = "neon")]
unsafe fn quantize_levels(src: *const i32, dst: *mut i32, len: usize, q: &Av1Quantizer) -> u32 {
    if len == 0 {
        return 0;
    }
    // SAFETY: `len >= 1` (caller contract for both pointers).
    let dc = unsafe { quantize_one(*src, 0, q) };
    let dc_nonzero = u32::from(dc != 0);
    let (offset, divisor) = quant_terms(q, q.ac_step);
    // AV2 shares this entry with larger scales; retain its scalar fallback.
    if divisor.is_power_of_two() && q.log_scale <= 2 {
        // SAFETY: len is positive; this helper processes only AC coefficients.
        unsafe { *dst = dc };
        // SAFETY: forwarded source/destination extent; DC was handled above.
        return dc_nonzero + unsafe { quantize_power_of_two(src, dst, len, q, offset, divisor) };
    }
    let Some(reciprocal) = Reciprocal::new(divisor) else {
        // SAFETY: len is positive; the scalar fallback starts at index one.
        unsafe { *dst = dc };
        // SAFETY: forwarded contract.
        return dc_nonzero + unsafe { quantize_tail(src, dst, 1, len, q) };
    };
    let ac = AcQuantizer::new(q, offset, divisor, reciprocal);
    let mut nonzero_lanes = vdupq_n_u32(0);
    // Quantize aligned vectors with AC parameters, then replace lane zero
    // with its saved DC result. This avoids three scalar AC divisions on
    // normal power-of-two block lengths. The original DC was read above;
    // no store precedes the corresponding load, including in-place calls.
    let octets_end = len / 8 * 8;
    for i in (0..octets_end).step_by(8) {
        // SAFETY: `i + 8 <= len`; loaded before the stores below.
        let (c0, c1) = unsafe { (vld1q_s32(src.add(i)), vld1q_s32(src.add(i + 4))) };
        let (n0, n1) = (ac.numerator(c0), ac.numerator(c1));
        // Coefficients below one quantizer divisor all produce zero levels:
        // skip the reciprocal products and the nonzero count. Exact (the
        // scalar reference defines the threshold). Isolated native AV1 CPU
        // fell 2.14% bodycam / 2.92% crowd with identical streams when this
        // skip was introduced on 4 lanes.
        let (l0, l1) = if vmaxvq_u32(vmaxq_u32(n0, n1)) < divisor {
            (vdupq_n_s32(0), vdupq_n_s32(0))
        } else {
            let levels = (ac.level(n0, c0), ac.level(n1, c1));
            nonzero_lanes = count_nonzero(count_nonzero(nonzero_lanes, levels.0), levels.1);
            levels
        };
        // SAFETY: `i + 8 <= len`.
        unsafe {
            vst1q_s32(dst.add(i), l0);
            vst1q_s32(dst.add(i + 4), l1);
        }
    }
    let mut done = octets_end;
    if done + 4 <= len {
        // SAFETY: `done + 4 <= len`.
        let c = unsafe { vld1q_s32(src.add(done)) };
        let n = ac.numerator(c);
        let level = if vmaxvq_u32(n) < divisor {
            vdupq_n_s32(0)
        } else {
            ac.level(n, c)
        };
        // SAFETY: as above.
        unsafe { vst1q_s32(dst.add(done), level) };
        nonzero_lanes = count_nonzero(nonzero_lanes, level);
        done += 4;
    }
    // SAFETY: forwarded contract.
    let nonzero = vaddvq_u32(nonzero_lanes) + unsafe { quantize_tail(src, dst, done, len, q) };
    // SAFETY: len is positive, and the vector/tail pass wrote destination
    // zero. Remove that lane's provisional count before installing DC.
    unsafe {
        let provisional_dc_nonzero = u32::from(*dst != 0);
        *dst = dc;
        nonzero - provisional_dc_nonzero + dc_nonzero
    }
}

/// Exact power-of-two division needs only a lane shift, not scalar division.
/// The numerator is unchanged from the general quantizer; no approximate
/// reciprocal or narrower coefficient contract is introduced.
///
/// # Safety
/// Same buffers and coefficient bounds as `quantize_levels`; `divisor` is
/// a power of two. Coefficient zero has already been processed separately.
#[target_feature(enable = "neon")]
unsafe fn quantize_power_of_two(
    src: *const i32,
    dst: *mut i32,
    len: usize,
    q: &Av1Quantizer,
    offset: u32,
    divisor: u32,
) -> u32 {
    let input_shift = vdupq_n_s32(7 + q.log_scale as i32);
    let divide_shift = vdupq_n_s32(-(divisor.trailing_zeros() as i32));
    let offset = vdupq_n_u32(offset);
    let limit = vdupq_n_u32(q.max_level as u32);
    let mut counts = vdupq_n_u32(0);
    let end = 1 + (len - 1) / 4 * 4;
    for i in (1..end).step_by(4) {
        // SAFETY: i + 4 <= len. Load precedes the store for in-place calls.
        let c = unsafe { vld1q_s32(src.add(i)) };
        let numerator = vaddq_u32(
            vshlq_u32(vreinterpretq_u32_s32(vabsq_s32(c)), input_shift),
            offset,
        );
        let magnitude = vreinterpretq_s32_u32(vminq_u32(vshlq_u32(numerator, divide_shift), limit));
        let sign = vshrq_n_s32::<31>(c);
        let level = vsubq_s32(veorq_s32(magnitude, sign), sign);
        counts = count_nonzero(counts, level);
        // SAFETY: same four valid destination lanes as above.
        unsafe { vst1q_s32(dst.add(i), level) };
    }
    // SAFETY: the remaining indices are within the forwarded extents.
    vaddvq_u32(counts) + unsafe { quantize_tail(src, dst, end, len, q) }
}

/// [`quantize_levels`] in place.
#[target_feature(enable = "neon")]
unsafe fn quantize(coeffs: &mut [i32], q: &Av1Quantizer) -> u32 {
    let (len, coeffs) = (coeffs.len(), coeffs.as_mut_ptr());
    // SAFETY: source and destination are the same slice; every value is
    // read before its level is written (`quantize_levels`).
    unsafe { quantize_levels(coeffs.cast_const(), coeffs, len, q) }
}

/// [`quantize_levels`] from `coeffs` into `out`, which keeps `coeffs`.
#[target_feature(enable = "neon")]
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

/// Add 1 to each lane of `counts` where `levels` is nonzero.
#[inline(always)]
fn count_nonzero(counts: uint32x4_t, levels: int32x4_t) -> uint32x4_t {
    // SAFETY: NEON kernels only. `vtst` gives all ones (-1) for a nonzero
    // lane, so subtracting it adds 1.
    unsafe { vsubq_u32(counts, vtstq_s32(levels, levels)) }
}

/// The AC quantiser as lane constants: `level = min(numerator / divisor,
/// max_level)`, signed like the coefficient, with the division done by the
/// exact reciprocal.
struct AcQuantizer {
    input_shift: int32x4_t,
    offset: uint32x4_t,
    multiplier: uint32x4_t,
    /// Right shift that remains after taking the high 32 bits of the
    /// 64-bit product (negative: `vshl` shifts right).
    high_shift: int32x4_t,
    max_level: uint32x4_t,
}

impl AcQuantizer {
    /// Bits dropped by taking the high half of a 64-bit product.
    const HALF_BITS: u32 = 32;

    #[inline(always)]
    fn new(q: &Av1Quantizer, offset: u32, divisor: u32, reciprocal: Reciprocal) -> Self {
        debug_assert!(reciprocal.shift >= Self::HALF_BITS);
        debug_assert!(divisor > 0);
        // SAFETY: NEON kernels only.
        unsafe {
            Self {
                input_shift: vdupq_n_s32(7 + q.log_scale as i32),
                offset: vdupq_n_u32(offset),
                multiplier: vdupq_n_u32(reciprocal.multiplier),
                high_shift: vdupq_n_s32(-((reciprocal.shift - Self::HALF_BITS) as i32)),
                max_level: vdupq_n_u32(q.max_level as u32),
            }
        }
    }

    /// `|c| * 2^log_scale * 128 + step * rounding_q7` (below 2^31 under
    /// the coefficient contract).
    #[inline(always)]
    fn numerator(&self, c: int32x4_t) -> uint32x4_t {
        // SAFETY: NEON kernels only.
        unsafe {
            vaddq_u32(
                vshlq_u32(vreinterpretq_u32_s32(vabsq_s32(c)), self.input_shift),
                self.offset,
            )
        }
    }

    /// The signed level of `c` from its numerator. `(n * m) >> shift`
    /// equals `((n * m) >> 32) >> (shift - 32)`: the high halves of both
    /// products come out of one unzip.
    #[inline(always)]
    fn level(&self, numerator: uint32x4_t, c: int32x4_t) -> int32x4_t {
        // SAFETY: NEON kernels only.
        unsafe {
            let low = vreinterpretq_u32_u64(vmull_u32(
                vget_low_u32(numerator),
                vget_low_u32(self.multiplier),
            ));
            let high = vreinterpretq_u32_u64(vmull_high_u32(numerator, self.multiplier));
            let quotient = vshlq_u32(vuzp2q_u32(low, high), self.high_shift);
            let level = vreinterpretq_s32_u32(vminq_u32(quotient, self.max_level));
            // Negate where c < 0: (level ^ m) - m with m = c >> 31.
            let sign = vshrq_n_s32::<31>(c);
            vsubq_s32(veorq_s32(level, sign), sign)
        }
    }
}

#[target_feature(enable = "neon")]
unsafe fn dequantize(levels: &[i32], q: &Av1Quantizer, out: &mut [i32]) {
    // SAFETY: NEON is enabled for this function.
    unsafe { generic::dequantize::<I32x4>(levels, q, out) }
}

/// `|c| < 2^22`, `d` is signed16-bit: `|2c-d| < 2^24`; the1024-term
/// sum magnitude is below `2^49`. Widen only the multiply, not its inputs.
#[target_feature(enable = "neon")]
unsafe fn coefficient_gain(original: &[i32], dequantized: &[i32]) -> i64 {
    // SAFETY: NEON table installed only with ISA support; complete groups
    // are checked before loads. Remaining coefficients use the scalar oracle.
    unsafe {
        let n = original.len().min(dequantized.len());
        let end = n / 4 * 4;
        let mut low = vdupq_n_s64(0);
        let mut high = vdupq_n_s64(0);
        for i in (0..end).step_by(4) {
            let c = vld1q_s32(original.as_ptr().add(i));
            let d = vld1q_s32(dequantized.as_ptr().add(i));
            let t = vsubq_s32(vshlq_n_s32::<1>(c), d);
            low = vmlal_s32(low, vget_low_s32(t), vget_low_s32(d));
            high = vmlal_high_s32(high, t, d);
        }
        vaddvq_s64(vaddq_s64(low, high))
            + super::scalar::coefficient_gain(&original[end..], &dequantized[end..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_ac_vectors_restore_dc_and_count_exactly() {
        for ac_step in [5, 73, 1828] {
            for dc_step in [4, 129, 1828] {
                for log_scale in 0..=5 {
                    for len in (0..=37).chain([64, 256, 1024]) {
                        let q = Av1Quantizer {
                            dc_step,
                            ac_step,
                            rounding_q7: 48,
                            max_level: 8191,
                            log_scale,
                        };
                        let input: Vec<_> = (0..len)
                            .map(|i| {
                                if i == 0 {
                                    75
                                } else {
                                    ((i * 139 + 17) % 8192) as i32 - 4096
                                }
                            })
                            .collect();
                        let expected: Vec<_> = input
                            .iter()
                            .enumerate()
                            .map(|(i, &v)| quantize_one(v, i, &q))
                            .collect();
                        let mut out = vec![12345; len + 1];
                        // SAFETY: NEON is mandatory; output covers every input.
                        let count = unsafe { quantize_from(&input, &q, &mut out) };
                        assert_eq!(&out[..len], &expected, "{q:?}, len={len}");
                        assert_eq!(out[len], 12345);
                        assert_eq!(count as usize, expected.iter().filter(|&&v| v != 0).count());
                        let mut inplace = input;
                        // SAFETY: NEON is mandatory and the input slice is valid.
                        assert_eq!(unsafe { quantize(&mut inplace, &q) }, count);
                        assert_eq!(inplace, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn power_of_two_division_matches_scalar_and_preserves_tail() {
        let magnitudes = [
            0,
            1,
            3,
            4,
            7,
            8,
            15,
            16,
            31,
            32,
            63,
            64,
            127,
            128,
            32767,
            (1 << 22) - 1,
        ];
        for ac_step in [4, 8, 16, 32, 64, 128] {
            for rounding_q7 in [0, 41, 48, 64, 127] {
                for log_scale in [0, 1, 2, 3, 4, 5] {
                    for max_level in [1, 8191, 32768] {
                        let q = Av1Quantizer {
                            dc_step: 73,
                            ac_step,
                            rounding_q7,
                            max_level,
                            log_scale,
                        };
                        for len in 0..=37 {
                            let source: Vec<_> = (0..len)
                                .map(|i| {
                                    let magnitude = magnitudes[i % magnitudes.len()];
                                    let magnitude = if log_scale > 2 {
                                        magnitude.min((1 << 16) - 1)
                                    } else {
                                        magnitude
                                    };
                                    if i % 2 == 0 { magnitude } else { -magnitude }
                                })
                                .collect();
                            let expected: Vec<_> = source
                                .iter()
                                .enumerate()
                                .map(|(i, &v)| quantize_one(v, i, &q))
                                .collect();
                            let mut output = vec![12345; len + 1];
                            // SAFETY: NEON is mandatory; output covers the input.
                            let count = unsafe { quantize_from(&source, &q, &mut output) };
                            assert_eq!(&output[..len], &expected, "{q:?}, len={len}");
                            assert_eq!(output[len], 12345);
                            assert_eq!(
                                count as usize,
                                expected.iter().filter(|&&v| v != 0).count()
                            );
                            let mut inplace = source;
                            // SAFETY: NEON is mandatory and the slice is valid.
                            assert_eq!(unsafe { quantize(&mut inplace, &q) }, count);
                            assert_eq!(inplace, expected);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn zero_vector_threshold_matches_scalar_on_both_sides() {
        for ac_step in [5, 7, 11, 129, 255, 1828] {
            for rounding_q7 in [0, 64, 127] {
                for log_scale in [0, 1, 2] {
                    let q = Av1Quantizer {
                        dc_step: 4,
                        ac_step,
                        rounding_q7,
                        log_scale,
                        max_level: 8191,
                    };
                    let (offset, divisor) = quant_terms(&q, ac_step);
                    let bound = ((divisor - offset - 1) >> (7 + log_scale)) as i32;
                    let input = [9, bound, -bound, 0, bound, bound + 1, -bound - 1, 1, 0];
                    let expected: Vec<_> = input
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| quantize_one(v, i, &q))
                        .collect();
                    let mut actual = input;
                    // SAFETY: NEON is mandatory on AArch64; all buffers are valid.
                    let nonzero = unsafe { quantize(&mut actual, &q) };
                    assert_eq!(actual.as_slice(), expected, "{q:?}");
                    assert_eq!(
                        nonzero as usize,
                        expected.iter().filter(|&&v| v != 0).count()
                    );
                }
            }
        }
    }
}
