//! NEON transform kernels: the generic 2-D DCT drivers on 4 lanes for 4x4,
//! guarded 16-bit forward (`neon_forward`) and inverse (`neon_inverse`)
//! drivers for 8x8 and larger, and quantisation by exact reciprocal
//! multiplication.
//!
//! Kernel contract: see the `txfm` module docs.

#![allow(clippy::inline_always)]

use core::arch::aarch64::{
    int32x4_t, uint32x4_t, vabsq_s32, vaddq_u32, vaddvq_s64, vaddvq_u32, vdupq_n_s32, vdupq_n_s64,
    vdupq_n_u32, veorq_s32, vget_low_s16, vget_low_u32, vld1_s16, vld1q_s16, vld1q_s32, vmaxq_u32,
    vmaxvq_u32, vminq_u32, vmovl_high_s16, vmovl_s16, vmovn_s32, vmull_high_s16, vmull_high_u32,
    vmull_s16, vmull_u32, vpadalq_s32, vreinterpretq_s16_s32, vreinterpretq_s32_u32,
    vreinterpretq_u32_s32, vreinterpretq_u32_u64, vshlq_u32, vshrq_n_s32, vst1_s16, vst1q_s16,
    vst1q_s32, vsubq_s32, vsubq_u32, vtstq_s32, vuzp1q_s16, vuzp2q_u32,
};

use super::scalar::{fix_dc, quantize_tail};
use super::{Av1Quantizer, QuantStorage, Reciprocal, TxfmFns, generic, quant_terms};
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
            super::neon_inverse::inverse_16_sparse,
            super::neon_inverse::inverse_32_sparse,
            super::neon_inverse::inverse_64_sparse,
        ],
        coefficient_gain,
        quantize,
        quantize_from: quantize_from::<i16>,
        quantize_from_wide: quantize_from::<i32>,
        dequantize,
        forward_dct_raster: [
            forward_4_raster,
            super::neon_forward::forward_8_raster,
            super::neon_forward::forward_16_raster,
            super::neon_forward::forward_32_raster,
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

macro_rules! neon_sizes {
    ($($forward:ident, $inverse:ident);*) => {$(
        #[target_feature(enable = "neon")]
        unsafe fn $forward(residual: &[i16], stride: usize, out: &mut [i16]) {
            // SAFETY: NEON is enabled for this function.
            unsafe { generic::$forward::<I32x4>(residual, stride, out) }
        }

        #[target_feature(enable = "neon")]
        unsafe fn $inverse(coeffs: &[i16], dst: &mut [u8], dst_stride: usize) {
            // SAFETY: NEON is enabled for this function.
            unsafe { generic::$inverse::<I32x4>(coeffs, dst, dst_stride) }
        }
    )*};
}

neon_sizes!(forward_4, inverse_4);

/// [`forward_4`] with raster coefficients (VP9's layout).
#[target_feature(enable = "neon")]
unsafe fn forward_4_raster(residual: &[i16], stride: usize, out: &mut [i16]) {
    // SAFETY: NEON is enabled for this function.
    unsafe { generic::forward_4_layout::<I32x4, true>(residual, stride, out) }
}

/// Quantiser storage loaded and stored as two four-lane 32-bit halves.
trait Lanes: QuantStorage {
    /// Eight values at `src`, sign-extended to 32 bits.
    ///
    /// # Safety
    ///
    /// `src` holds eight values.
    unsafe fn load8(src: *const Self) -> (int32x4_t, int32x4_t);
    /// Store two four-lane halves (each lane fitting `Self`) at `dst`.
    ///
    /// # Safety
    ///
    /// `dst` holds eight values.
    unsafe fn store8(dst: *mut Self, lanes: (int32x4_t, int32x4_t));
    /// Four values at `src`, sign-extended.
    ///
    /// # Safety
    ///
    /// `src` holds four values.
    unsafe fn load4(src: *const Self) -> int32x4_t;
    /// Store four lanes (each fitting `Self`) at `dst`.
    ///
    /// # Safety
    ///
    /// `dst` holds four values.
    unsafe fn store4(dst: *mut Self, lanes: int32x4_t);
}

impl Lanes for i32 {
    #[inline(always)]
    unsafe fn load8(src: *const Self) -> (int32x4_t, int32x4_t) {
        // SAFETY: caller contract.
        unsafe { (vld1q_s32(src), vld1q_s32(src.add(4))) }
    }
    #[inline(always)]
    unsafe fn store8(dst: *mut Self, (low, high): (int32x4_t, int32x4_t)) {
        // SAFETY: caller contract.
        unsafe {
            vst1q_s32(dst, low);
            vst1q_s32(dst.add(4), high);
        }
    }
    #[inline(always)]
    unsafe fn load4(src: *const Self) -> int32x4_t {
        // SAFETY: caller contract.
        unsafe { vld1q_s32(src) }
    }
    #[inline(always)]
    unsafe fn store4(dst: *mut Self, lanes: int32x4_t) {
        // SAFETY: caller contract.
        unsafe { vst1q_s32(dst, lanes) }
    }
}

impl Lanes for i16 {
    #[inline(always)]
    unsafe fn load8(src: *const Self) -> (int32x4_t, int32x4_t) {
        // SAFETY: caller contract: one 128-bit load.
        unsafe {
            let values = vld1q_s16(src);
            (vmovl_s16(vget_low_s16(values)), vmovl_high_s16(values))
        }
    }
    #[inline(always)]
    unsafe fn store8(dst: *mut Self, (low, high): (int32x4_t, int32x4_t)) {
        // The low halves of the 32-bit lanes, in one unzip and one 128-bit
        // store. Exact: every lane fits 16 bits.
        // SAFETY: caller contract.
        unsafe {
            vst1q_s16(
                dst,
                vuzp1q_s16(vreinterpretq_s16_s32(low), vreinterpretq_s16_s32(high)),
            );
        }
    }
    #[inline(always)]
    unsafe fn load4(src: *const Self) -> int32x4_t {
        // SAFETY: caller contract.
        unsafe { vmovl_s16(vld1_s16(src)) }
    }
    #[inline(always)]
    unsafe fn store4(dst: *mut Self, lanes: int32x4_t) {
        // SAFETY: caller contract; exact as in `store8`.
        unsafe { vst1_s16(dst, vmovn_s32(lanes)) }
    }
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
#[target_feature(enable = "neon")]
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
    let ac = AcQuantizer::new(q, offset, divisor, reciprocal);
    let mut nonzero_lanes = vdupq_n_u32(0);
    let octets_end = len / 8 * 8;
    for i in (0..octets_end).step_by(8) {
        // SAFETY: `i + 8 <= len`; loaded before the stores below.
        let (c0, c1) = unsafe { T::load8(src.add(i)) };
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
        unsafe { T::store8(dst.add(i), (l0, l1)) };
    }
    let mut done = octets_end;
    if done + 4 <= len {
        // SAFETY: `done + 4 <= len`.
        let c = unsafe { T::load4(src.add(done)) };
        let n = ac.numerator(c);
        let level = if vmaxvq_u32(n) < divisor {
            vdupq_n_s32(0)
        } else {
            ac.level(n, c)
        };
        // SAFETY: as above.
        unsafe { T::store4(dst.add(done), level) };
        nonzero_lanes = count_nonzero(nonzero_lanes, level);
        done += 4;
    }
    // SAFETY: forwarded contract.
    let nonzero = vaddvq_u32(nonzero_lanes) + unsafe { quantize_tail(src, dst, done, len, q) };
    if done == 0 {
        // The tail quantised coefficient 0 at the DC step.
        nonzero
    } else {
        // SAFETY: the vector pass wrote `dst[0]`.
        unsafe { fix_dc(dst, dc, nonzero, q) }
    }
}

/// [`quantize_levels`] in place.
#[target_feature(enable = "neon")]
unsafe fn quantize(coeffs: &mut [i16], q: &Av1Quantizer) -> u32 {
    let (len, coeffs) = (coeffs.len(), coeffs.as_mut_ptr());
    // SAFETY: source and destination are the same slice; every value is
    // read before its level is written (`quantize_levels`).
    unsafe { quantize_levels(coeffs.cast_const(), coeffs, len, q) }
}

/// [`quantize_levels`] from `coeffs` into `out`, which keeps `coeffs`.
#[target_feature(enable = "neon")]
unsafe fn quantize_from<T: Lanes>(coeffs: &[T], q: &Av1Quantizer, out: &mut [T]) -> u32 {
    let out = &mut out[..coeffs.len()];
    // SAFETY: both slices hold `coeffs.len()` values.
    unsafe { quantize_levels(coeffs.as_ptr(), out.as_mut_ptr(), coeffs.len(), q) }
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
unsafe fn dequantize(levels: &[i16], q: &Av1Quantizer, out: &mut [i16]) {
    // SAFETY: NEON is enabled for this function.
    unsafe { generic::dequantize::<I32x4>(levels, q, out) }
}

/// `2 * sum(c * d) - sum(d * d)` on eight 16-bit lanes: every product of
/// two signed 16-bit values fits 32 bits exactly (`smull`, at most `2^30`),
/// and the products accumulate pairwise into 64-bit lanes (`sadalp`), so
/// no sum can overflow.
#[target_feature(enable = "neon")]
unsafe fn coefficient_gain(original: &[i16], dequantized: &[i16]) -> i64 {
    // SAFETY: NEON table installed only with ISA support; complete groups
    // are checked before loads. Remaining coefficients use the scalar oracle.
    unsafe {
        let n = original.len().min(dequantized.len());
        let end = n / 8 * 8;
        let mut cross = vdupq_n_s64(0);
        let mut square = vdupq_n_s64(0);
        for i in (0..end).step_by(8) {
            let c = vld1q_s16(original.as_ptr().add(i));
            let d = vld1q_s16(dequantized.as_ptr().add(i));
            cross = vpadalq_s32(cross, vmull_s16(vget_low_s16(c), vget_low_s16(d)));
            cross = vpadalq_s32(cross, vmull_high_s16(c, d));
            square = vpadalq_s32(square, vmull_s16(vget_low_s16(d), vget_low_s16(d)));
            square = vpadalq_s32(square, vmull_high_s16(d, d));
        }
        2 * vaddvq_s64(cross) - vaddvq_s64(square)
            + super::scalar::coefficient_gain(&original[end..], &dequantized[end..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::txfm::quantize_one;

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
                    let input = [9, bound, -bound, 0, bound, bound + 1, -bound - 1, 1, 0]
                        .map(|v: i32| v as i16);
                    let expected: Vec<_> = input
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| quantize_one(i32::from(v), i, &q) as i16)
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
