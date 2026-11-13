//! SIMD128 lanes of the guarded 16-bit AV1 inverse DCTs (`narrow_inverse`):
//! wrapping 16-bit adds, `i32x4.dot_i16x8_s` on interleaved pairs for
//! two-weight rotations, and the rounding Q15 multiply for single and equal
//! weights, exact inside the L1 guard.
//!
//! Kernel contract: see the `txfm` module docs. This module only exists in
//! builds with SIMD128 enabled, so its kernels need no run-time check.

#![allow(clippy::inline_always)]

use core::arch::wasm32::{
    i16x8_abs, i16x8_add, i16x8_narrow_i32x4, i16x8_shr, i16x8_shuffle, i16x8_splat, i16x8_sub,
    i32x4_add, i32x4_dot_i16x8, i32x4_extmul_high_i16x8, i32x4_extmul_low_i16x8, i32x4_shr,
    i32x4_shuffle, i32x4_splat, i64x2_shuffle, u8x16_narrow_i16x8, u16x8_add_sat,
    u16x8_extend_low_u8x16, u16x8_gt, u16x8_splat, v128, v128_and, v128_any_true, v128_load,
    v128_load64_zero, v128_store, v128_store64_lane,
};

use super::generic::INVERSE_COS_BIT;
use super::inverse_lanes::InverseLanes;
use super::narrow_inverse::{self, GROUP, NarrowLanes};
use crate::av1_inverse::Kernel1d;
use crate::simd::simd128::{I32x4, relaxed::q15_mul_round};

/// `q15_mul_round` computes `(a * b + 2^14) >> 15`; a weight scaled by
/// `2^(15 - COS_BIT)` turns that into libaom's `round_shift(a * w, COS_BIT)`.
const Q15_WEIGHT_SHIFT: u32 = 15 - INVERSE_COS_BIT;
/// libaom's rounding offset of a `COS_BIT` shift.
const COS_ROUND: i32 = 1 << (INVERSE_COS_BIT - 1);

/// Eight 16-bit lanes of one inverse pass, valid only inside the L1 limit.
#[derive(Clone, Copy)]
pub(super) struct Narrow(v128);

/// `round_shift(x, COS_BIT)` of two `i32x4` halves, narrowed (the guard
/// keeps the results in 16 bits, so the saturating narrow is exact).
#[inline(always)]
fn round_narrow(low: v128, high: v128) -> Narrow {
    let round = i32x4_splat(COS_ROUND);
    Narrow(i16x8_narrow_i32x4(
        i32x4_shr(i32x4_add(low, round), INVERSE_COS_BIT),
        i32x4_shr(i32x4_add(high, round), INVERSE_COS_BIT),
    ))
}

impl InverseLanes for Narrow {
    #[inline(always)]
    fn splat(value: i32) -> Self {
        // The graphs splat only zero.
        Self(i16x8_splat(value as i16))
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // The guard keeps the sum in 16 bits.
        Self(i16x8_add(self.0, other.0))
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        // The guard keeps the difference in 16 bits.
        Self(i16x8_sub(self.0, other.0))
    }

    #[inline(always)]
    fn half_btf(w0: i32, a: Self, w1: i32, b: Self, bit: u32) -> Self {
        debug_assert_eq!(bit, INVERSE_COS_BIT);
        // Equal-magnitude weights (the cospi[32] rotations) add or subtract
        // first: the guard keeps that pre-add in 16 bits too.
        if w0.abs() < 1 << INVERSE_COS_BIT {
            if w0 == w1 {
                return Self::mul_round(w0, a.add(b), bit);
            }
            if w0 == -w1 {
                return Self::mul_round(w0, a.sub(b), bit);
            }
        }
        // Lanes (a_i, b_i) dotted with (w0, w1): exact 32-bit sums of Q12
        // products (weights fit 16 bits).
        let weights = i32x4_splat((w1 << 16) | (w0 & 0xffff));
        let low = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(a.0, b.0);
        let high = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(a.0, b.0);
        round_narrow(
            i32x4_dot_i16x8(low, weights),
            i32x4_dot_i16x8(high, weights),
        )
    }

    #[inline(always)]
    fn mul_round(w: i32, a: Self, bit: u32) -> Self {
        debug_assert_eq!(bit, INVERSE_COS_BIT);
        if w.abs() < 1 << INVERSE_COS_BIT {
            // The scaled weight is not -2^15, so the Q15 multiply never
            // saturates and rounds exactly as libaom's round_shift.
            Self(q15_mul_round(
                a.0,
                i16x8_splat((w << Q15_WEIGHT_SHIFT) as i16),
            ))
        } else {
            let weight = i16x8_splat(w as i16);
            round_narrow(
                i32x4_extmul_low_i16x8(a.0, weight),
                i32x4_extmul_high_i16x8(a.0, weight),
            )
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
    type Norm = v128;
    type Wide = I32x4;

    #[inline(always)]
    fn norm_zero() -> v128 {
        u16x8_splat(0)
    }

    #[inline(always)]
    fn accumulate_l1(self, norm: v128) -> v128 {
        // `|i16::MIN|` wraps to 0x8000, which reads correctly as unsigned.
        u16x8_add_sat(norm, i16x8_abs(self.0))
    }

    #[inline(always)]
    fn within_limit(norm: v128, limit: u16) -> bool {
        !v128_any_true(u16x8_gt(norm, u16x8_splat(limit)))
    }

    #[inline(always)]
    fn load_i32_saturating(src: &[i32]) -> Self {
        let src = &src[..GROUP];
        // SAFETY: two 16-byte loads of an eight-element slice.
        let (low, high) = unsafe {
            (
                v128_load(src.as_ptr().cast()),
                v128_load(src.as_ptr().add(4).cast()),
            )
        };
        // The signed saturating narrow is libaom's 16-bit input clamp.
        Self(i16x8_narrow_i32x4(low, high))
    }

    #[inline(always)]
    fn load_i16(src: &[i16]) -> Self {
        let src = &src[..GROUP];
        // SAFETY: a 16-byte load of an eight-element slice.
        Self(unsafe { v128_load(src.as_ptr().cast()) })
    }

    #[inline(always)]
    unsafe fn store_groups(self, dst: *mut i16, _group_stride: usize) {
        // SAFETY: a 16-byte store to the caller's eight writable values.
        unsafe { v128_store(dst.cast(), self.0) }
    }

    #[inline(always)]
    fn round_shift<const SHIFT: u32>(self) -> Self {
        // `(x >> s) + ((x >> (s - 1)) & 1)`: the rounding shift without the
        // overflow of adding the offset first.
        let carry = v128_and(i16x8_shr(self.0, SHIFT - 1), i16x8_splat(1));
        Self(i16x8_add(i16x8_shr(self.0, SHIFT), carry))
    }

    #[inline(always)]
    fn transpose(rows: [Self; GROUP]) -> [Self; GROUP] {
        transpose8(rows.map(|row| row.0)).map(Self)
    }

    #[inline(always)]
    fn add_to_pixels(self, pixels: &mut [u8]) {
        let pixels = &mut pixels[..GROUP];
        // SAFETY: an eight-byte load and store of an eight-byte array.
        unsafe {
            let prediction = u16x8_extend_low_u8x16(v128_load64_zero(pixels.as_ptr().cast()));
            // The residual and pixel sum is within i16; the unsigned
            // saturating narrow is the 8-bit clip.
            let sum = i16x8_add(self.0, prediction);
            v128_store64_lane::<0>(u8x16_narrow_i16x8(sum, sum), pixels.as_mut_ptr().cast());
        }
    }
}

pub(super) unsafe fn inverse_8(coeffs: &[i32], dst: &mut [u8], stride: usize) {
    narrow_inverse::inverse_dct::<Narrow, 8>(coeffs, dst, stride, 8);
}

pub(super) unsafe fn inverse_16(coeffs: &[i32], dst: &mut [u8], stride: usize) {
    narrow_inverse::inverse_dct::<Narrow, 16>(coeffs, dst, stride, 16);
}

pub(super) unsafe fn inverse_32(coeffs: &[i32], dst: &mut [u8], stride: usize) {
    narrow_inverse::inverse_dct::<Narrow, 32>(coeffs, dst, stride, 32);
}

pub(super) unsafe fn inverse_64(coeffs: &[i32], dst: &mut [u8], stride: usize) {
    narrow_inverse::inverse_dct::<Narrow, 64>(coeffs, dst, stride, 32);
}

pub(super) unsafe fn inverse_32_sparse(
    coeffs: &[i32],
    dst: &mut [u8],
    stride: usize,
    limit: usize,
) {
    narrow_inverse::inverse_dct::<Narrow, 32>(coeffs, dst, stride, limit);
}

pub(super) unsafe fn inverse_64_sparse(
    coeffs: &[i32],
    dst: &mut [u8],
    stride: usize,
    limit: usize,
) {
    narrow_inverse::inverse_dct::<Narrow, 64>(coeffs, dst, stride, limit);
}

/// Transpose eight rows of eight 16-bit lanes: pairs of rows interleaved by
/// 16-bit lane, then by 32-bit pairs, then by 64-bit halves (standard
/// unpacks, one instruction each under V8).
#[inline(always)]
pub(super) fn transpose8(rows: [v128; GROUP]) -> [v128; GROUP] {
    let lanes: [v128; GROUP] = core::array::from_fn(|k| {
        let (x, y) = (rows[k & !1], rows[k | 1]);
        if k % 2 == 0 {
            i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(x, y)
        } else {
            i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(x, y)
        }
    });
    // `quads[4 * h + q]`: columns 2q and 2q + 1 of rows 4h..4h + 4.
    let quads: [v128; GROUP] = core::array::from_fn(|k| {
        let (block, pair) = (k / 4 * 4, k % 4);
        let (x, y) = (lanes[block + pair / 2], lanes[block + 2 + pair / 2]);
        if pair % 2 == 0 {
            i32x4_shuffle::<0, 4, 1, 5>(x, y)
        } else {
            i32x4_shuffle::<2, 6, 3, 7>(x, y)
        }
    });
    core::array::from_fn(|column| {
        let (x, y) = (quads[column / 2], quads[4 + column / 2]);
        if column % 2 == 0 {
            i64x2_shuffle::<0, 2>(x, y)
        } else {
            i64x2_shuffle::<1, 3>(x, y)
        }
    })
}

/// The AV1 inverse of a `W`x`H` transform with both sides at least 8 and
/// the 1-D kernels `(vertical, horizontal)`, column-major `coeffs` whose rows
/// from `nonzero_rows` on are zero, added to `dst` (the `av1_inverse` table
/// entry for those sizes).
///
/// # Safety
///
/// None beyond the slice contract (the build enables SIMD128).
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
        // SAFETY: the build enables SIMD128.
        unsafe {
            super::super::narrow_inverse::tests::assert_guard_boundary(
                super::inverse_32_sparse,
                super::inverse_64_sparse,
            );
        }
    }
}
