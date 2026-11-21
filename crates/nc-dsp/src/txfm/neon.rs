//! NEON transform kernels: the generic 2-D DCT drivers on 4 lanes for 4x4,
//! guarded 16-bit forward (`neon_forward`) and inverse (`neon_inverse`)
//! drivers for 8x8 and larger, and quantisation by exact reciprocal
//! multiplication.
//!
//! Kernel contract: see the `txfm` module docs.

#![allow(clippy::inline_always)]

use core::arch::aarch64::{
    int16x8_t, int32x4_t, uint32x4_t, vabsq_s16, vabsq_s32, vaddlvq_u16, vaddq_u32, vaddvq_s64,
    vaddvq_u32, vandq_u32, vdupq_n_s16, vdupq_n_s32, vdupq_n_s64, vdupq_n_u16, vdupq_n_u32,
    veorq_s16, veorq_s32, vget_low_s16, vget_low_u16, vget_low_u32, vld1_s16, vld1q_s16, vld1q_s32,
    vmaxq_u32, vmaxvq_u16, vmaxvq_u32, vminq_u16, vminq_u32, vmovl_high_s16, vmovl_high_u16,
    vmovl_s16, vmovl_u16, vmovn_s32, vmull_high_n_u16, vmull_high_s16, vmull_high_u32, vmull_n_u16,
    vmull_s16, vmull_u32, vpadalq_s32, vqmovn_high_s32, vqmovn_high_u32, vqmovn_s32, vqmovn_u32,
    vreinterpretq_s16_s32, vreinterpretq_s16_u16, vreinterpretq_s32_u32, vreinterpretq_u16_s16,
    vreinterpretq_u32_s32, vreinterpretq_u32_u64, vshlq_u32, vshrq_n_s16, vshrq_n_s32, vst1_s16,
    vst1q_s16, vst1q_s32, vsubq_s16, vsubq_s32, vsubq_u16, vsubq_u32, vtstq_s16, vtstq_s32,
    vuzp1q_s16, vuzp2q_u32,
};

use super::scalar::{fix_dc, quantize_tail};
use super::{Av1Quantizer, QuantStorage, QuantTerms, Reciprocal, TxfmFns, generic};
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
        quantize_from: quantize_from_i16::<i16>,
        quantize_from_wide: quantize_from::<i32, i32>,
        quantize_from_widening: quantize_from_i16::<i32>,
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

/// Quantiser storage loaded and stored as two four-lane 32-bit halves
/// (shared with the HEVC dequantiser, `codec_txfm::neon`).
pub(crate) trait Lanes: QuantStorage {
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
    /// Store eight 16-bit lanes (each fitting `Self`) at `dst`.
    ///
    /// # Safety
    ///
    /// `dst` holds eight values.
    unsafe fn store8_i16(dst: *mut Self, lanes: int16x8_t);
    /// Store eight zero values at `dst`.
    ///
    /// # Safety
    ///
    /// `dst` holds eight values.
    unsafe fn store8_zero(dst: *mut Self);
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
    #[inline(always)]
    unsafe fn store8_i16(dst: *mut Self, lanes: int16x8_t) {
        // SAFETY: caller contract.
        unsafe {
            vst1q_s32(dst, vmovl_s16(vget_low_s16(lanes)));
            vst1q_s32(dst.add(4), vmovl_high_s16(lanes));
        }
    }
    #[inline(always)]
    unsafe fn store8_zero(dst: *mut Self) {
        // SAFETY: caller contract.
        unsafe {
            let z = vdupq_n_s32(0);
            vst1q_s32(dst, z);
            vst1q_s32(dst.add(4), z);
        }
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
    #[inline(always)]
    unsafe fn store8_i16(dst: *mut Self, lanes: int16x8_t) {
        // SAFETY: caller contract.
        unsafe { vst1q_s16(dst, lanes) }
    }
    #[inline(always)]
    unsafe fn store8_zero(dst: *mut Self) {
        // SAFETY: caller contract.
        unsafe { vst1q_s16(dst, vdupq_n_s16(0)) }
    }
}

/// Largest unsigned coefficient magnitude that quantises to zero at the AC
/// step, or `None` when even zero produces a nonzero level.
#[inline(always)]
fn ac_zero_limit(q: &QuantTerms) -> Option<u32> {
    q.ac.divisor
        .checked_sub(q.ac.offset)
        .and_then(|d| d.checked_sub(1))
        .map(|d| d >> q.input_shift)
}

/// Quantise 16-bit coefficients in eight-lane 16-bit groups: the all-zero
/// check runs on `|c|` in 16 bits before any widening, and nonzero groups
/// narrow, clamp, sign and count in 16 bits. Falls back to [`quantize_levels`]
/// when the level cap exceeds `i16::MAX`, no coefficient quantises to zero,
/// or the slice could overflow the 16-bit per-lane nonzero counters.
///
/// # Safety
///
/// `src` and `dst` hold `len` values; they are either disjoint or equal
/// (in place): each vector is loaded before its levels are stored.
#[target_feature(enable = "neon")]
unsafe fn quantize_levels_i16<D: Lanes>(
    src: *const i16,
    dst: *mut D,
    len: usize,
    q: &QuantTerms,
) -> u32 {
    let divisor = q.ac.divisor;
    let Some(reciprocal) = Reciprocal::new(divisor) else {
        // SAFETY: forwarded contract.
        return unsafe { quantize_tail(src, dst, 0, len, q) };
    };
    if len == 0 {
        return 0;
    }
    let Some(limit) = ac_zero_limit(q) else {
        // SAFETY: forwarded contract.
        return unsafe { quantize_levels(src, dst, len, q) };
    };
    if q.ac.cap > i32::from(i16::MAX) || len > usize::from(u16::MAX) * 8 {
        // SAFETY: forwarded contract.
        return unsafe { quantize_levels(src, dst, len, q) };
    }
    // SAFETY: `len >= 1`; read before the vector pass overwrites it.
    let dc = unsafe { i32::from(*src) };
    let ac = AcQuantizer::new(q, reciprocal);
    let zero_limit = u16::try_from(limit).unwrap_or(u16::MAX);
    let max_level = vdupq_n_u16(q.ac.cap as u16);
    let mut nonzero_lanes = vdupq_n_u16(0);
    let octets_end = len / 8 * 8;
    for i in (0..octets_end).step_by(8) {
        // SAFETY: `i + 8 <= len`; loaded before the stores below.
        let raw = unsafe { vld1q_s16(src.add(i)) };
        let mag = vreinterpretq_u16_s16(vabsq_s16(raw));
        if vmaxvq_u16(mag) <= zero_limit {
            // SAFETY: `i + 8 <= len`.
            unsafe { D::store8_zero(dst.add(i)) };
            continue;
        }
        let n0 = ac.numerator_abs(vmovl_u16(vget_low_u16(mag)));
        let n1 = ac.numerator_abs(vmovl_high_u16(mag));
        let q0 = ac.quotient(n0);
        let q1 = ac.quotient(n1);
        let narrowed = vqmovn_high_u32(vqmovn_u32(q0), q1);
        let level = vreinterpretq_s16_u16(vminq_u16(narrowed, max_level));
        let sign = vshrq_n_s16::<15>(raw);
        let signed = vsubq_s16(veorq_s16(level, sign), sign);
        // SAFETY: `i + 8 <= len`.
        unsafe { D::store8_i16(dst.add(i), signed) };
        nonzero_lanes = vsubq_u16(nonzero_lanes, vtstq_s16(signed, signed));
    }
    let mut nonzero = vaddlvq_u16(nonzero_lanes);
    let mut done = octets_end;
    if done + 4 <= len {
        // SAFETY: `done + 4 <= len`.
        let coeffs = unsafe { i16::load4(src.add(done)) };
        let mags = vreinterpretq_u32_s32(vabsq_s32(coeffs));
        let level = if vmaxvq_u32(mags) <= limit {
            vdupq_n_s32(0)
        } else {
            let numerators = ac.numerator_abs(mags);
            let levels = ac.level(numerators, coeffs);
            nonzero += vaddvq_u32(count_nonzero(vdupq_n_u32(0), levels));
            levels
        };
        // SAFETY: as above.
        unsafe { D::store4(dst.add(done), level) };
        done += 4;
    }
    // SAFETY: forwarded contract.
    nonzero += unsafe { quantize_tail(src, dst, done, len, q) };
    if done == 0 {
        nonzero
    } else {
        // SAFETY: the vector pass wrote `dst[0]`.
        unsafe { fix_dc(dst, dc, nonzero, q) }
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
unsafe fn quantize_levels<S: Lanes, D: Lanes>(
    src: *const S,
    dst: *mut D,
    len: usize,
    q: &QuantTerms,
) -> u32 {
    let divisor = q.ac.divisor;
    let Some(reciprocal) = Reciprocal::new(divisor) else {
        // SAFETY: forwarded contract.
        return unsafe { quantize_tail(src, dst, 0, len, q) };
    };
    if len == 0 {
        return 0;
    }
    // SAFETY: `len >= 1`; read before the vector pass overwrites it.
    let dc = unsafe { (*src).widen() };
    let ac = AcQuantizer::new(q, reciprocal);
    let zero_limit = ac_zero_limit(q);
    let mut nonzero_lanes = vdupq_n_u32(0);
    let octets_end = len / 8 * 8;
    for i in (0..octets_end).step_by(8) {
        // SAFETY: `i + 8 <= len`; loaded before the stores below.
        let (c0, c1) = unsafe { S::load8(src.add(i)) };
        let a0 = vreinterpretq_u32_s32(vabsq_s32(c0));
        let a1 = vreinterpretq_u32_s32(vabsq_s32(c1));
        if let Some(limit) = zero_limit
            && vmaxvq_u32(vmaxq_u32(a0, a1)) <= limit
        {
            // SAFETY: `i + 8 <= len`.
            unsafe { D::store8_zero(dst.add(i)) };
            continue;
        }
        let (n0, n1) = (ac.numerator_abs(a0), ac.numerator_abs(a1));
        let levels = (ac.level(n0, c0), ac.level(n1, c1));
        nonzero_lanes = count_nonzero(count_nonzero(nonzero_lanes, levels.0), levels.1);
        // SAFETY: `i + 8 <= len`.
        unsafe { D::store8(dst.add(i), levels) };
    }
    let mut done = octets_end;
    if done + 4 <= len {
        // SAFETY: `done + 4 <= len`.
        let c = unsafe { S::load4(src.add(done)) };
        let a = vreinterpretq_u32_s32(vabsq_s32(c));
        let level = if zero_limit.is_some_and(|limit| vmaxvq_u32(a) <= limit) {
            vdupq_n_s32(0)
        } else {
            let l = ac.level(ac.numerator_abs(a), c);
            nonzero_lanes = count_nonzero(nonzero_lanes, l);
            l
        };
        // SAFETY: as above.
        unsafe { D::store4(dst.add(done), level) };
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

/// [`quantize_levels_i16`] in place.
#[target_feature(enable = "neon")]
unsafe fn quantize(coeffs: &mut [i16], q: &QuantTerms) -> u32 {
    let (len, coeffs) = (coeffs.len(), coeffs.as_mut_ptr());
    // SAFETY: source and destination are the same slice; every value is
    // read before its level is written (`quantize_levels_i16`).
    unsafe { quantize_levels_i16(coeffs.cast_const(), coeffs, len, q) }
}

/// [`quantize_levels_i16`] from `coeffs` into `out`, which keeps `coeffs`.
#[target_feature(enable = "neon")]
unsafe fn quantize_from_i16<D: Lanes>(coeffs: &[i16], q: &QuantTerms, out: &mut [D]) -> u32 {
    let out = &mut out[..coeffs.len()];
    // SAFETY: both slices hold `coeffs.len()` values.
    unsafe { quantize_levels_i16(coeffs.as_ptr(), out.as_mut_ptr(), coeffs.len(), q) }
}

/// [`quantize_levels`] from `coeffs` into `out`, which keeps `coeffs`.
#[target_feature(enable = "neon")]
unsafe fn quantize_from<S: Lanes, D: Lanes>(coeffs: &[S], q: &QuantTerms, out: &mut [D]) -> u32 {
    let out = &mut out[..coeffs.len()];
    // SAFETY: both slices hold `coeffs.len()` values.
    unsafe { quantize_levels(coeffs.as_ptr(), out.as_mut_ptr(), coeffs.len(), q) }
}

/// Fold the last nonzero position into quantization. The vector loop uses
/// the AC terms for coefficient zero, so its tentative end of one must
/// use the actual DC level at the end.
///
/// # Safety
///
/// NEON must be available. `out` and `scan` cover `coeffs`; `scan` is
/// a raster-to-scan-position-plus-one table with DC first.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn quantize_wide_eob(
    coeffs: &[i32],
    q: &QuantTerms,
    out: &mut [i32],
    scan: &[u16],
) -> usize {
    let (chunks, tail) = coeffs.as_chunks::<8>();
    let Some(reciprocal) = Reciprocal::new(q.ac.divisor) else {
        return super::scalar::quantize_wide_eob(coeffs, q, out, scan);
    };
    let Some(&dc) = coeffs.first() else {
        return 0;
    };
    let (dst, tail_out) = out[..coeffs.len()].as_chunks_mut::<8>();
    let (scans, tail_scan) = scan.as_chunks::<8>();
    let ac = AcQuantizer::new(q, reciprocal);
    let zero_limit = ac_zero_limit(q);
    let mut largest = vdupq_n_u16(0);
    for ((coeff, dst), end) in chunks.iter().zip(dst).zip(scans) {
        // SAFETY: all three arguments cover one eight-lane group.
        unsafe {
            let (c0, c1) = i32::load8(coeff.as_ptr());
            let a0 = vreinterpretq_u32_s32(vabsq_s32(c0));
            let a1 = vreinterpretq_u32_s32(vabsq_s32(c1));
            if let Some(limit) = zero_limit
                && vmaxvq_u32(vmaxq_u32(a0, a1)) <= limit
            {
                i32::store8_zero(dst.as_mut_ptr());
                continue;
            }
            let l0 = ac.level(ac.numerator_abs(a0), c0);
            let l1 = ac.level(ac.numerator_abs(a1), c1);
            i32::store8(dst.as_mut_ptr(), (l0, l1));
            let mask = core::arch::aarch64::vcombine_u16(
                core::arch::aarch64::vmovn_u32(vtstq_s32(l0, l0)),
                core::arch::aarch64::vmovn_u32(vtstq_s32(l1, l1)),
            );
            largest = core::arch::aarch64::vmaxq_u16(
                largest,
                core::arch::aarch64::vandq_u16(mask, core::arch::aarch64::vld1q_u16(end.as_ptr())),
            );
        }
    }
    let mut end = usize::from(vmaxvq_u16(largest));
    let done = chunks.len() * 8;
    for (i, ((&coeff, dst), &position)) in tail.iter().zip(tail_out).zip(tail_scan).enumerate() {
        *dst = q.level(coeff, done + i);
        if *dst != 0 {
            end = end.max(usize::from(position));
        }
    }
    out[0] = q.level(dc, 0);
    if end <= 1 {
        usize::from(out[0] != 0)
    } else {
        end
    }
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
    fn new(q: &QuantTerms, reciprocal: Reciprocal) -> Self {
        debug_assert!(reciprocal.shift >= Self::HALF_BITS);
        // SAFETY: NEON kernels only.
        unsafe {
            Self {
                input_shift: vdupq_n_s32(q.input_shift as i32),
                offset: vdupq_n_u32(q.ac.offset),
                multiplier: vdupq_n_u32(reciprocal.multiplier),
                high_shift: vdupq_n_s32(-((reciprocal.shift - Self::HALF_BITS) as i32)),
                max_level: vdupq_n_u32(q.ac.cap as u32),
            }
        }
    }

    /// `(abs_c << input_shift) + offset` (below 2^31 under the coefficient
    /// contract).
    #[inline(always)]
    fn numerator_abs(&self, abs_c: uint32x4_t) -> uint32x4_t {
        // SAFETY: NEON kernels only.
        unsafe { vaddq_u32(vshlq_u32(abs_c, self.input_shift), self.offset) }
    }

    /// Unsigned quotient `(n * m) >> shift` = `((n * m) >> 32) >> (shift - 32)`.
    #[inline(always)]
    fn quotient(&self, numerator: uint32x4_t) -> uint32x4_t {
        // SAFETY: NEON kernels only.
        unsafe {
            let low = vreinterpretq_u32_u64(vmull_u32(
                vget_low_u32(numerator),
                vget_low_u32(self.multiplier),
            ));
            let high = vreinterpretq_u32_u64(vmull_high_u32(numerator, self.multiplier));
            vshlq_u32(vuzp2q_u32(low, high), self.high_shift)
        }
    }

    /// The signed level of `c` from its numerator.
    #[inline(always)]
    fn level(&self, numerator: uint32x4_t, c: int32x4_t) -> int32x4_t {
        // SAFETY: NEON kernels only.
        unsafe {
            let quotient = self.quotient(numerator);
            let level = vreinterpretq_s32_u32(vminq_u32(quotient, self.max_level));
            // Negate where c < 0: (level ^ m) - m with m = c >> 31.
            let sign = vshrq_n_s32::<31>(c);
            vsubq_s32(veorq_s32(level, sign), sign)
        }
    }
}

/// AV1 dequantisation (AV1 §7.12.3) on eight 16-bit lanes with all-zero group
/// skip and unsigned widening multiplies (`umull`/`umull2`).
#[target_feature(enable = "neon")]
unsafe fn dequantize(levels: &[i16], q: &Av1Quantizer, out: &mut [i16]) {
    let Some(&dc) = levels.first() else {
        return;
    };
    let Ok(step) = u16::try_from(q.ac_step) else {
        // SAFETY: NEON is enabled for this function.
        return unsafe { generic::dequantize::<I32x4>(levels, q, out) };
    };
    let n = levels.len();
    let out = &mut out[..n];
    let (src_chunks, src_tail) = levels.as_chunks::<8>();
    let (dst_chunks, dst_tail) = out.as_chunks_mut::<8>();
    let mask = vdupq_n_u32(super::DEQUANT_MASK as u32);
    let shift = vdupq_n_s32(-(q.log_scale as i32));
    let zero = vdupq_n_s16(0);
    for (src, dst) in src_chunks.iter().zip(dst_chunks) {
        // SAFETY: NEON is enabled; each chunk holds eight values.
        unsafe {
            let raw = vld1q_s16(src.as_ptr());
            let mag = vreinterpretq_u16_s16(vabsq_s16(raw));
            if vmaxvq_u16(mag) == 0 {
                vst1q_s16(dst.as_mut_ptr(), zero);
                continue;
            }
            let low = vshlq_u32(vandq_u32(vmull_n_u16(vget_low_u16(mag), step), mask), shift);
            let high = vshlq_u32(vandq_u32(vmull_high_n_u16(mag, step), mask), shift);
            let sign = vshrq_n_s16::<15>(raw);
            let s_low = vmovl_s16(vget_low_s16(sign));
            let s_high = vmovl_high_s16(sign);
            let signed_low = vsubq_s32(veorq_s32(vreinterpretq_s32_u32(low), s_low), s_low);
            let signed_high = vsubq_s32(veorq_s32(vreinterpretq_s32_u32(high), s_high), s_high);
            vst1q_s16(
                dst.as_mut_ptr(),
                vqmovn_high_s32(vqmovn_s32(signed_low), signed_high),
            );
        }
    }
    let done = src_chunks.len() * 8;
    for (i, (&level, dq)) in src_tail.iter().zip(dst_tail).enumerate() {
        *dq = super::dequantize_one(i32::from(level), done + i, q) as i16;
    }
    out[0] = super::dequantize_one(i32::from(dc), 0, q) as i16;
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
                    let terms = QuantTerms::from(&q);
                    let bound =
                        ((terms.ac.divisor - terms.ac.offset - 1) >> terms.input_shift) as i32;
                    let input = [9, bound, -bound, 0, bound, bound + 1, -bound - 1, 1, 0]
                        .map(|v: i32| v as i16);
                    let expected: Vec<_> = input
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| terms.level(i32::from(v), i) as i16)
                        .collect();
                    let mut actual = input;
                    // SAFETY: NEON is mandatory on AArch64; all buffers are valid.
                    let nonzero = unsafe { quantize(&mut actual, &terms) };
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
