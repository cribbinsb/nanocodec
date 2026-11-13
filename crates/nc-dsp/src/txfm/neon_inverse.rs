//! NEON lanes of the guarded 16-bit AV1 inverse DCTs (`narrow_inverse`):
//! wrapping 16-bit adds, widening multiplies with a rounding narrow, and
//! rounding doubling-high multiplies (`vqrdmulh`) for single and equal
//! weights, exact inside the L1 guard.
//!
//! Kernel contract: see the `txfm` module docs.

#![allow(clippy::inline_always)]

use core::arch::aarch64::{
    int16x8_t, uint16x8_t, vabsq_s16, vaddq_s16, vaddw_u8, vdupq_n_s16, vdupq_n_u16, vget_low_s16,
    vld1_u8, vld1q_s16, vld1q_s32, vmaxvq_u16, vmlal_high_n_s16, vmlal_n_s16, vmull_high_n_s16,
    vmull_n_s16, vqaddq_u16, vqmovn_high_s32, vqmovn_s32, vqmovun_s16, vqrdmulhq_n_s16,
    vreinterpretq_s16_u16, vreinterpretq_u16_s16, vrshrn_high_n_s32, vrshrn_n_s32, vrshrq_n_s16,
    vst1_u8, vst1q_s16, vsubq_s16,
};

use super::generic::INVERSE_COS_BIT;
use super::inverse_lanes::InverseLanes;
use super::narrow_inverse::{self, COLUMN_SHIFT, GROUP, NarrowLanes, ROW_SHIFT_1, ROW_SHIFT_2};
use super::neon_forward::transpose;
use crate::av1_inverse::Kernel1d;
use crate::simd::neon::I32x4;

const COS_BIT: i32 = INVERSE_COS_BIT as i32;
/// `vqrdmulh` computes `(2 * a * b + 2^15) >> 16`; a weight scaled by
/// `2^(15 - COS_BIT)` turns that into libaom's `round_shift(a * w, COS_BIT)`.
const Q15_WEIGHT_SHIFT: i32 = 15 - COS_BIT;

/// Eight 16-bit lanes of one inverse pass, valid only inside the L1 limit.
#[derive(Clone, Copy)]
pub(super) struct Narrow(int16x8_t);

impl Narrow {
    /// `round_shift(a * w, COS_BIT)` by widening, for any 16-bit weight.
    #[inline(always)]
    fn widening_mul_round(w: i16, a: Self) -> Self {
        // SAFETY: NEON kernels only. The products are exact in 32 bits; the
        // guard keeps the rounded result in 16 bits.
        unsafe {
            let low = vmull_n_s16(vget_low_s16(a.0), w);
            let high = vmull_high_n_s16(a.0, w);
            Self(vrshrn_high_n_s32::<COS_BIT>(
                vrshrn_n_s32::<COS_BIT>(low),
                high,
            ))
        }
    }
}

impl InverseLanes for Narrow {
    #[inline(always)]
    fn splat(value: i32) -> Self {
        // SAFETY: NEON kernels only; the graphs splat only zero.
        unsafe { Self(vdupq_n_s16(value as i16)) }
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // SAFETY: NEON kernels only; the guard keeps the sum in 16 bits.
        unsafe { Self(vaddq_s16(self.0, other.0)) }
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        // SAFETY: NEON kernels only; the guard keeps the difference in 16 bits.
        unsafe { Self(vsubq_s16(self.0, other.0)) }
    }

    #[inline(always)]
    fn half_btf(w0: i32, a: Self, w1: i32, b: Self, bit: u32) -> Self {
        debug_assert_eq!(bit, INVERSE_COS_BIT);
        // Equal-magnitude weights (the cospi[32] rotations) add or subtract
        // first: the guard keeps that pre-add in 16 bits too.
        if w0.abs() < 1 << COS_BIT {
            if w0 == w1 {
                return Self::mul_round(w0, a.add(b), bit);
            }
            if w0 == -w1 {
                return Self::mul_round(w0, a.sub(b), bit);
            }
        }
        // SAFETY: NEON kernels only. Q12 weights fit 16 bits; the widened
        // product sums are exact, and the guard keeps the rounded result in
        // 16 bits.
        unsafe {
            let low = vmlal_n_s16(
                vmull_n_s16(vget_low_s16(a.0), w0 as i16),
                vget_low_s16(b.0),
                w1 as i16,
            );
            let high = vmlal_high_n_s16(vmull_high_n_s16(a.0, w0 as i16), b.0, w1 as i16);
            Self(vrshrn_high_n_s32::<COS_BIT>(
                vrshrn_n_s32::<COS_BIT>(low),
                high,
            ))
        }
    }

    #[inline(always)]
    fn mul_round(w: i32, a: Self, bit: u32) -> Self {
        debug_assert_eq!(bit, INVERSE_COS_BIT);
        if w.abs() < 1 << COS_BIT {
            // SAFETY: NEON kernels only. The scaled weight is not -2^15, so
            // the doubling multiply never saturates and rounds exactly as
            // libaom's round_shift.
            unsafe { Self(vqrdmulhq_n_s16(a.0, (w << Q15_WEIGHT_SHIFT) as i16)) }
        } else {
            Self::widening_mul_round(w as i16, a)
        }
    }

    /// Inside the guard every node already fits 16 bits.
    #[inline(always)]
    fn clamp16(self) -> Self {
        self
    }
}

impl NarrowLanes for Narrow {
    const LANES: usize = GROUP;
    type Norm = uint16x8_t;
    type Wide = I32x4;

    #[inline(always)]
    fn norm_zero() -> uint16x8_t {
        // SAFETY: NEON kernels only.
        unsafe { vdupq_n_u16(0) }
    }

    #[inline(always)]
    fn accumulate_l1(self, norm: uint16x8_t) -> uint16x8_t {
        // SAFETY: NEON kernels only.
        unsafe { vqaddq_u16(norm, vreinterpretq_u16_s16(vabsq_s16(self.0))) }
    }

    #[inline(always)]
    fn within_limit(norm: uint16x8_t, limit: u16) -> bool {
        // SAFETY: NEON kernels only.
        unsafe { vmaxvq_u16(norm) <= limit }
    }

    #[inline(always)]
    fn load_i32_saturating(src: &[i32]) -> Self {
        let src = &src[..GROUP];
        // SAFETY: NEON kernels only; two four-element loads of an
        // eight-element slice. Saturating narrowing is the clamp.
        unsafe {
            Self(vqmovn_high_s32(
                vqmovn_s32(vld1q_s32(src.as_ptr())),
                vld1q_s32(src.as_ptr().add(4)),
            ))
        }
    }

    #[inline(always)]
    fn load_i16(src: &[i16]) -> Self {
        let src = &src[..GROUP];
        // SAFETY: NEON kernels only; an eight-element slice.
        unsafe { Self(vld1q_s16(src.as_ptr())) }
    }

    #[inline(always)]
    unsafe fn store_groups(self, dst: *mut i16, _group_stride: usize) {
        // SAFETY: NEON kernels only; the caller's eight writable values.
        unsafe { vst1q_s16(dst, self.0) }
    }

    #[inline(always)]
    fn round_shift<const SHIFT: u32>(self) -> Self {
        // SAFETY: NEON kernels only; `vrshr` rounds without overflow.
        unsafe {
            Self(match SHIFT {
                ROW_SHIFT_2 => vrshrq_n_s16::<{ ROW_SHIFT_2 as i32 }>(self.0),
                ROW_SHIFT_1 => vrshrq_n_s16::<{ ROW_SHIFT_1 as i32 }>(self.0),
                COLUMN_SHIFT => vrshrq_n_s16::<{ COLUMN_SHIFT as i32 }>(self.0),
                _ => unreachable!("inverse shifts are the row and column shifts"),
            })
        }
    }

    #[inline(always)]
    fn transpose(rows: [Self; GROUP]) -> [Self; GROUP] {
        // SAFETY: NEON kernels only; register-only transpose.
        unsafe { transpose(rows.map(|row| row.0)) }.map(Self)
    }

    #[inline(always)]
    fn add_to_pixels(self, pixels: &mut [u8]) {
        let pixels = &mut pixels[..GROUP];
        // SAFETY: NEON kernels only; an eight-byte array. The residual and
        // pixel sum is within i16, so the wrapping unsigned add is exact; the
        // unsigned saturating narrow is the 8-bit clip.
        unsafe {
            let sum = vaddw_u8(vreinterpretq_u16_s16(self.0), vld1_u8(pixels.as_ptr()));
            vst1_u8(pixels.as_mut_ptr(), vqmovun_s16(vreinterpretq_s16_u16(sum)));
        }
    }
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn inverse_8(coeffs: &[i32], dst: &mut [u8], stride: usize) {
    narrow_inverse::inverse_dct::<Narrow, 8>(coeffs, dst, stride, 8);
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn inverse_16(coeffs: &[i32], dst: &mut [u8], stride: usize) {
    narrow_inverse::inverse_dct::<Narrow, 16>(coeffs, dst, stride, 16);
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn inverse_32(coeffs: &[i32], dst: &mut [u8], stride: usize) {
    narrow_inverse::inverse_dct::<Narrow, 32>(coeffs, dst, stride, 32);
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn inverse_64(coeffs: &[i32], dst: &mut [u8], stride: usize) {
    narrow_inverse::inverse_dct::<Narrow, 64>(coeffs, dst, stride, 32);
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn inverse_32_sparse(
    coeffs: &[i32],
    dst: &mut [u8],
    stride: usize,
    limit: usize,
) {
    narrow_inverse::inverse_dct::<Narrow, 32>(coeffs, dst, stride, limit);
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn inverse_64_sparse(
    coeffs: &[i32],
    dst: &mut [u8],
    stride: usize,
    limit: usize,
) {
    narrow_inverse::inverse_dct::<Narrow, 64>(coeffs, dst, stride, limit);
}

/// The AV1 inverse of a `W`x`H` transform with both sides at least 8 and
/// the 1-D kernels `(vertical, horizontal)`, column-major `coeffs` whose rows
/// from `nonzero_rows` on are zero, added to `dst` (the `av1_inverse` table
/// entry for those sizes).
///
/// # Safety
///
/// None beyond the slice contract (NEON is always present on `AArch64`).
#[target_feature(enable = "neon")]
pub(crate) unsafe fn av1_inverse_add<const W: usize, const H: usize>(
    kernels: (Kernel1d, Kernel1d),
    coeffs: &[i32],
    nonzero_rows: usize,
    dst: &mut [u8],
    stride: usize,
) {
    let extent = narrow_inverse::Inverse::of::<W, H>(kernels, nonzero_rows);
    narrow_inverse::inverse::<Narrow, W, H>(extent, coeffs, dst, stride);
}

#[cfg(test)]
mod tests {
    #[test]
    fn guard_boundary_matches_scalar_on_both_sides() {
        // SAFETY: NEON is mandatory on AArch64.
        unsafe {
            super::super::narrow_inverse::tests::assert_guard_boundary(
                super::inverse_32_sparse,
                super::inverse_64_sparse,
            );
        }
    }
}
