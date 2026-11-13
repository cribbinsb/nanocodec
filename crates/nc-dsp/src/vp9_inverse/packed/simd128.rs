//! SIMD128 VP9 inverse lanes with paired signed dot products and wrapping
//! low-word narrows. Both separable passes keep eight transforms in vectors.

use super::{COS_BITS, Lanes, Products};
use crate::Vp9Pass;
use crate::txfm::inverse_lanes::InverseLanes;
use core::arch::wasm32::{
    i16x8_add, i16x8_q15mulr_sat, i16x8_shuffle, i16x8_splat, i16x8_sub, i32x4_add,
    i32x4_dot_i16x8, i32x4_extend_high_i16x8, i32x4_extend_low_i16x8, i32x4_extmul_high_i16x8,
    i32x4_extmul_low_i16x8, i32x4_shr, i32x4_splat, i32x4_sub, u8x16_narrow_i16x8,
    u16x8_extend_low_u8x16, u16x8_narrow_i32x4, v128, v128_and, v128_load, v128_load64_zero,
    v128_store, v128_store64_lane,
};

const HALF: usize = 4;
const LOW_WORD: i32 = 0xffff;

#[derive(Clone, Copy)]
struct Narrow(v128);
#[derive(Clone, Copy)]
struct Wide([v128; 2]);

impl Products for Wide {
    type Narrow = Narrow;
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        Self([
            i32x4_add(self.0[0], other.0[0]),
            i32x4_add(self.0[1], other.0[1]),
        ])
    }
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        Self([
            i32x4_sub(self.0[0], other.0[0]),
            i32x4_sub(self.0[1], other.0[1]),
        ])
    }
    #[inline(always)]
    fn narrow(self) -> Narrow {
        let mask = i32x4_splat(LOW_WORD);
        Narrow(u16x8_narrow_i32x4(
            v128_and(self.0[0], mask),
            v128_and(self.0[1], mask),
        ))
    }
    #[inline(always)]
    fn round14(self) -> Narrow {
        let round = i32x4_splat(1 << (COS_BITS - 1));
        Self([
            i32x4_shr(i32x4_add(self.0[0], round), COS_BITS),
            i32x4_shr(i32x4_add(self.0[1], round), COS_BITS),
        ])
        .narrow()
    }
}

impl Narrow {
    /// `round_shift(w * self, 14)` from `doubled = 2 * w`
    /// ([`super::doubled_weight`]).
    #[inline(always)]
    fn mul_doubled(self, doubled: i16) -> Self {
        // `doubled` is never -32768, so the multiply cannot saturate.
        Self(i16x8_q15mulr_sat(self.0, i16x8_splat(doubled)))
    }
}

impl InverseLanes for Narrow {
    #[inline(always)]
    fn splat(value: i32) -> Self {
        Self(i16x8_splat(value as i16))
    }
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        Self(i16x8_add(self.0, other.0))
    }
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        Self(i16x8_sub(self.0, other.0))
    }
    #[inline(always)]
    fn half_btf(w0: i32, a: Self, w1: i32, b: Self, bit: u32) -> Self {
        debug_assert_eq!(bit, COS_BITS);
        Self::products(w0, a, w1, b).round14()
    }
    #[inline(always)]
    fn mul_round(w: i32, a: Self, bit: u32) -> Self {
        debug_assert_eq!(bit, COS_BITS);
        match super::doubled_weight(w) {
            Some(doubled) => a.mul_doubled(doubled),
            None => a.wide(w).round14(),
        }
    }
    #[inline(always)]
    fn clamp16(self) -> Self {
        self
    }
}

impl Lanes for Narrow {
    type Wide = Wide;
    #[inline(always)]
    fn wide(self, weight: i32) -> Wide {
        let weight = i16x8_splat(weight as i16);
        Wide([
            i32x4_extmul_low_i16x8(self.0, weight),
            i32x4_extmul_high_i16x8(self.0, weight),
        ])
    }
    #[inline(always)]
    fn products(w0: i32, a: Self, w1: i32, b: Self) -> Wide {
        let packed = i32::from(w0 as u16) | (i32::from(w1 as u16) << 16);
        let weights = i32x4_splat(packed);
        let lo = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(a.0, b.0);
        let hi = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(a.0, b.0);
        Wide([i32x4_dot_i16x8(lo, weights), i32x4_dot_i16x8(hi, weights)])
    }
    #[inline(always)]
    fn load_i32_wrapping(src: &[i32]) -> Self {
        let src = &src[..super::LANES];
        // SAFETY: the checked slice holds both unaligned four-element loads.
        unsafe {
            Wide([
                v128_load(src.as_ptr().cast()),
                v128_load(src.as_ptr().add(HALF).cast()),
            ])
            .narrow()
        }
    }
    #[inline(always)]
    fn load_i16(src: &[i16]) -> Self {
        let src = &src[..super::LANES];
        // SAFETY: the checked slice holds all sixteen bytes of the vector.
        unsafe { Self(v128_load(src.as_ptr().cast())) }
    }
    #[inline(always)]
    fn store_i16(self, dst: &mut [i16]) {
        let dst = &mut dst[..super::LANES];
        // SAFETY: the checked destination holds the full unaligned store.
        unsafe {
            v128_store(dst.as_mut_ptr().cast(), self.0);
        }
    }
    #[inline(always)]
    fn store_shifted(self, dst: &mut [i32], shift: u32) {
        let dst = &mut dst[..super::LANES];
        let round = i32x4_splat(1 << (shift - 1));
        let lo = i32x4_shr(i32x4_add(i32x4_extend_low_i16x8(self.0), round), shift);
        let hi = i32x4_shr(i32x4_add(i32x4_extend_high_i16x8(self.0), round), shift);
        // SAFETY: the checked destination holds both four-element stores.
        unsafe {
            v128_store(dst.as_mut_ptr().cast(), lo);
            v128_store(dst.as_mut_ptr().add(HALF).cast(), hi);
        }
    }
    #[inline(always)]
    unsafe fn add_shifted(self, dst: *mut u8, shift: u32) {
        // `q15mulr` by 2^(15 - shift) is exactly `(x + 2^(shift - 1)) >>
        // shift` (it saturates only for -32768 * -32768); the residual plus
        // a pixel fits i16 and the unsigned saturating narrow clips.
        let residual = i16x8_q15mulr_sat(self.0, i16x8_splat(1 << (15 - shift)));
        // SAFETY: the caller's destination holds the eight samples.
        unsafe {
            let pixels = u16x8_extend_low_u8x16(v128_load64_zero(dst.cast_const().cast()));
            let bytes = u8x16_narrow_i16x8(i16x8_add(residual, pixels), residual);
            v128_store64_lane::<0>(bytes, dst.cast());
        }
    }
    #[inline(always)]
    fn zip_lo(self, other: Self) -> Self {
        Self(i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(self.0, other.0))
    }
    #[inline(always)]
    fn zip_hi(self, other: Self) -> Self {
        Self(i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(self.0, other.0))
    }
}

#[target_feature(enable = "simd128")]
pub(crate) unsafe fn inverse(
    log2: u8,
    columns: Vp9Pass,
    rows: Vp9Pass,
    input: &[i32],
    output: &mut [i32],
) {
    super::inverse::<Narrow>(log2, columns, rows, input, output);
}

#[target_feature(enable = "simd128")]
pub(crate) unsafe fn inverse_add(
    log2: u8,
    columns: Vp9Pass,
    rows: Vp9Pass,
    input: &[i32],
    nonzero: (usize, usize),
    dst: &mut [u8],
    dst_stride: usize,
) {
    super::inverse_add::<Narrow>(log2, (columns, rows), input, nonzero, (dst, dst_stride));
}
