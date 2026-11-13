//! Sixteen AV1 filter lines in one byte vector, with two widened word halves.
//! The shared filter graph supplies masks and smoothing; paired 8x8 transposes
//! keep both vertical groups in registers. Eight/four-line tails use the
//! existing narrow adapter, so this module never accesses padded edge lines.
//!
//! The byte/word/dword transpose layout follows the unpack stages reviewed in
//! dav1d 1.5.4 `src/x86/loopfilter_sse.asm` (`TRANSPOSE_16X16B`, VideoLAN/dav1d
//! authors and Two Orioles, BSD-2-Clause). It is adapted to paired 8x8 groups
//! in Rust SIMD128; shared normative filtering remains in `av1_packed`.

#![allow(clippy::inline_always)]

use super::{Bytes, LANES, PairedBytes, Words};
use core::arch::wasm32::{
    i8x16_add_sat, i8x16_bitmask, i8x16_eq, i8x16_shr, i8x16_shuffle, i8x16_sub_sat, i16x8_add,
    i16x8_extend_high_u8x16, i16x8_extend_low_u8x16, i16x8_shr, i16x8_shuffle, i16x8_splat,
    i16x8_sub, i32x4_shuffle, i64x2_shuffle, u8x16_add_sat, u8x16_max, u8x16_narrow_i16x8,
    u8x16_shr, u8x16_splat, u8x16_sub_sat, v128, v128_and, v128_load32_zero, v128_load64_zero,
    v128_store16_lane, v128_store32_lane, v128_store64_lane, v128_xor,
};

#[derive(Clone, Copy)]
pub(crate) struct B(v128);

#[derive(Clone, Copy)]
pub(crate) struct W([v128; 2]);

impl Bytes for B {
    type W = W;

    #[inline(always)]
    fn splat(v: u8) -> Self {
        Self(u8x16_splat(v))
    }

    #[inline(always)]
    unsafe fn load(p: *const u8) -> Self {
        // SAFETY: the base trait promises eight bytes, not sixteen.
        unsafe { Self(v128_load64_zero(p.cast())) }
    }

    #[inline(always)]
    unsafe fn store(self, p: *mut u8) {
        // SAFETY: the base trait promises eight writable bytes.
        unsafe {
            v128_store64_lane::<0>(self.0, p.cast());
        }
    }

    #[inline(always)]
    unsafe fn load4(p: *const u8) -> Self {
        // SAFETY: the caller provides four readable bytes.
        unsafe { Self(v128_load32_zero(p.cast())) }
    }

    #[inline(always)]
    unsafe fn store4(self, p: *mut u8) {
        // SAFETY: the caller provides four writable bytes.
        unsafe {
            v128_store32_lane::<0>(self.0, p.cast());
        }
    }

    #[inline(always)]
    unsafe fn store2(self, p: *mut u8) {
        // SAFETY: the caller provides two writable bytes.
        unsafe {
            v128_store16_lane::<0>(self.0, p.cast());
        }
    }

    #[inline(always)]
    fn max(self, b: Self) -> Self {
        Self(u8x16_max(self.0, b.0))
    }
    #[inline(always)]
    fn sub_sat(self, b: Self) -> Self {
        Self(u8x16_sub_sat(self.0, b.0))
    }
    #[inline(always)]
    fn add_sat(self, b: Self) -> Self {
        Self(u8x16_add_sat(self.0, b.0))
    }
    #[inline(always)]
    fn signed_add(self, b: Self) -> Self {
        Self(i8x16_add_sat(self.0, b.0))
    }
    #[inline(always)]
    fn signed_sub(self, b: Self) -> Self {
        Self(i8x16_sub_sat(self.0, b.0))
    }
    #[inline(always)]
    fn and(self, b: Self) -> Self {
        Self(v128_and(self.0, b.0))
    }
    #[inline(always)]
    fn xor(self, b: Self) -> Self {
        Self(v128_xor(self.0, b.0))
    }
    #[inline(always)]
    fn eq(self, b: Self) -> Self {
        Self(i8x16_eq(self.0, b.0))
    }
    #[inline(always)]
    fn shr<const N: i32>(self) -> Self {
        Self(u8x16_shr(self.0, N as u32))
    }
    #[inline(always)]
    fn signed_shr<const N: i32>(self) -> Self {
        Self(i8x16_shr(self.0, N as u32))
    }

    #[inline(always)]
    fn zip_lo(self, b: Self) -> Self {
        Self(i8x16_shuffle::<
            0,
            16,
            1,
            17,
            2,
            18,
            3,
            19,
            8,
            24,
            9,
            25,
            10,
            26,
            11,
            27,
        >(self.0, b.0))
    }

    #[inline(always)]
    fn zip_hi(self, b: Self) -> Self {
        Self(i8x16_shuffle::<
            4,
            20,
            5,
            21,
            6,
            22,
            7,
            23,
            12,
            28,
            13,
            29,
            14,
            30,
            15,
            31,
        >(self.0, b.0))
    }

    #[inline(always)]
    fn transpose(rows: [Self; LANES]) -> [Self; LANES] {
        // Full-width interleaves map to native unpack instructions in V8.
        // Keeping each eight-line group inside its own half at every stage
        // instead needs arbitrary byte shuffles, which are more expensive.
        let r = rows.map(|row| row.0);
        let a = [
            i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(r[0], r[1]),
            i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(
                r[0], r[1],
            ),
            i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(r[2], r[3]),
            i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(
                r[2], r[3],
            ),
            i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(r[4], r[5]),
            i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(
                r[4], r[5],
            ),
            i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(r[6], r[7]),
            i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(
                r[6], r[7],
            ),
        ];
        let b = [
            i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(a[0], a[2]),
            i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(a[0], a[2]),
            i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(a[4], a[6]),
            i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(a[4], a[6]),
            i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(a[1], a[3]),
            i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(a[1], a[3]),
            i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(a[5], a[7]),
            i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(a[5], a[7]),
        ];
        let c = [
            i32x4_shuffle::<0, 4, 1, 5>(b[0], b[2]),
            i32x4_shuffle::<2, 6, 3, 7>(b[0], b[2]),
            i32x4_shuffle::<0, 4, 1, 5>(b[1], b[3]),
            i32x4_shuffle::<2, 6, 3, 7>(b[1], b[3]),
            i32x4_shuffle::<0, 4, 1, 5>(b[4], b[6]),
            i32x4_shuffle::<2, 6, 3, 7>(b[4], b[6]),
            i32x4_shuffle::<0, 4, 1, 5>(b[5], b[7]),
            i32x4_shuffle::<2, 6, 3, 7>(b[5], b[7]),
        ];
        [
            Self(i64x2_shuffle::<0, 2>(c[0], c[4])),
            Self(i64x2_shuffle::<1, 3>(c[0], c[4])),
            Self(i64x2_shuffle::<0, 2>(c[1], c[5])),
            Self(i64x2_shuffle::<1, 3>(c[1], c[5])),
            Self(i64x2_shuffle::<0, 2>(c[2], c[6])),
            Self(i64x2_shuffle::<1, 3>(c[2], c[6])),
            Self(i64x2_shuffle::<0, 2>(c[3], c[7])),
            Self(i64x2_shuffle::<1, 3>(c[3], c[7])),
        ]
    }

    #[inline(always)]
    fn any(self) -> bool {
        i8x16_bitmask(self.0) != 0
    }

    #[inline(always)]
    fn widen(self) -> W {
        W([
            i16x8_extend_low_u8x16(self.0),
            i16x8_extend_high_u8x16(self.0),
        ])
    }

    #[inline(always)]
    fn narrow(v: W) -> Self {
        Self(u8x16_narrow_i16x8(v.0[0], v.0[1]))
    }
}

impl Words for W {
    #[inline(always)]
    fn splat(v: i16) -> Self {
        Self([i16x8_splat(v); 2])
    }
    #[inline(always)]
    fn add(self, b: Self) -> Self {
        Self([i16x8_add(self.0[0], b.0[0]), i16x8_add(self.0[1], b.0[1])])
    }
    #[inline(always)]
    fn sub(self, b: Self) -> Self {
        Self([i16x8_sub(self.0[0], b.0[0]), i16x8_sub(self.0[1], b.0[1])])
    }
    #[inline(always)]
    fn shr<const N: i32>(self) -> Self {
        Self([
            i16x8_shr(self.0[0], N as u32),
            i16x8_shr(self.0[1], N as u32),
        ])
    }
}

impl PairedBytes for B {
    #[inline(always)]
    unsafe fn load_pair(first: *const u8, second: *const u8) -> Self {
        // SAFETY: caller provides two eight-byte spans and enables SIMD128.
        unsafe {
            Self(i64x2_shuffle::<0, 2>(
                v128_load64_zero(first.cast()),
                v128_load64_zero(second.cast()),
            ))
        }
    }

    #[inline(always)]
    unsafe fn store_pair(self, first: *mut u8, second: *mut u8) {
        // SAFETY: caller provides two writable eight-byte spans.
        unsafe {
            v128_store64_lane::<0>(self.0, first.cast());
            v128_store64_lane::<1>(self.0, second.cast());
        }
    }
}
