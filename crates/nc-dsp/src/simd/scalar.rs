//! One-lane `i32` implementation of [`I32Lanes`]: the transform reference.
//!
//! `half_btf` and `round_shift` compute in 64 bits exactly as libaom does;
//! SIMD implementations compute in 32 bits, which gives identical results
//! within the documented input ranges.

use core::mem::MaybeUninit;

use super::I32Lanes;

impl I32Lanes for i32 {
    const LANES: usize = 1;

    fn splat(value: i32) -> Self {
        value
    }

    fn add(self, other: Self) -> Self {
        self.wrapping_add(other)
    }

    fn sub(self, other: Self) -> Self {
        self.wrapping_sub(other)
    }

    fn half_btf(w0: i32, a: Self, w1: i32, b: Self, bit: u32) -> Self {
        let sum = i64::from(w0) * i64::from(a) + i64::from(w1) * i64::from(b);
        ((sum + (1i64 << (bit - 1))) >> bit) as i32
    }

    fn clamp16(self) -> Self {
        self.clamp(i32::from(i16::MIN), i32::from(i16::MAX))
    }

    fn round_shift(self, bits: u32) -> Self {
        if bits == 0 {
            self
        } else {
            ((i64::from(self) + (1i64 << (bits - 1))) >> bits) as i32
        }
    }

    fn shl(self, bits: u32) -> Self {
        self << bits
    }

    fn mul_scalar(self, c: i32) -> Self {
        self.wrapping_mul(c)
    }

    fn mul(self, other: Self) -> Self {
        self.wrapping_mul(other)
    }

    fn abs(self) -> Self {
        self.wrapping_abs()
    }

    fn shr_logical(self, bits: u32) -> Self {
        ((self as u32) >> bits) as i32
    }

    fn sra(self, bits: u32) -> Self {
        self >> bits
    }

    fn min(self, other: Self) -> Self {
        Ord::min(self, other)
    }

    fn max(self, other: Self) -> Self {
        Ord::max(self, other)
    }

    fn with_sign_of(self, like: Self) -> Self {
        if like < 0 { self.wrapping_neg() } else { self }
    }

    fn count_nonzero(self) -> u32 {
        u32::from(self != 0)
    }

    fn load(src: &[i32]) -> Self {
        src[0]
    }

    fn store(self, dst: &mut [i32]) {
        dst[0] = self;
    }

    fn load_i16(src: &[i16]) -> Self {
        i32::from(src[0])
    }

    fn add_to_u8(self, dst: &mut [u8]) {
        dst[0] = (i32::from(dst[0]) + self).clamp(0, 255) as u8;
    }

    fn store_transposed(rows: &[Self], dst: &mut [MaybeUninit<i32>], _dst_stride: usize) {
        dst[0] = MaybeUninit::new(rows[0]);
    }

    fn gt(self, other: Self) -> Self {
        -i32::from(self > other)
    }

    fn eq(self, other: Self) -> Self {
        -i32::from(self == other)
    }

    fn or(self, other: Self) -> Self {
        self | other
    }

    fn select(self, if_set: Self, if_clear: Self) -> Self {
        if self != 0 { if_set } else { if_clear }
    }
}
