//! SIMD128 byte and word lanes of the packed loop filters (`lpf_bytes`).

use super::{Bytes, LANES, MaskWords, QuadBytes, WordTable, Words};
use crate::simd::simd128::relaxed::{q15_mul_round, select_i16, swizzle_in_bounds};
use core::arch::wasm32::{
    i8x16_add_sat, i8x16_bitmask, i8x16_eq, i8x16_shr, i8x16_shuffle, i8x16_sub_sat, i16x8_add,
    i16x8_extend_low_u8x16, i16x8_shr, i16x8_shuffle, i16x8_splat, i16x8_sub, i32x4_shuffle,
    i64x2_shuffle, u8x16_add_sat, u8x16_max, u8x16_narrow_i16x8, u8x16_shr, u8x16_splat,
    u8x16_sub_sat, v128, v128_and, v128_load64_zero, v128_store64_lane, v128_xor,
};
use core::arch::wasm32::{
    i16x8_abs, i16x8_eq, i16x8_gt, i16x8_max, i16x8_min, i16x8_mul, u32x4, v128_andnot,
    v128_any_true, v128_load, v128_or, v128_store,
};
#[derive(Clone, Copy)]
pub(crate) struct B(v128);
#[derive(Clone, Copy)]
pub(crate) struct W(v128);
impl Bytes for B {
    type W = W;
    #[inline(always)]
    fn splat(v: u8) -> Self {
        Self(u8x16_splat(v))
    }
    #[inline(always)]
    unsafe fn load(p: *const u8) -> Self {
        // SAFETY: the caller provides eight readable bytes and enables the ISA.
        unsafe { Self(v128_load64_zero(p.cast())) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut u8) {
        // SAFETY: the caller provides eight writable bytes and enables the ISA.
        unsafe {
            v128_store64_lane::<0>(self.0, p.cast());
        }
    }
    #[inline(always)]
    unsafe fn store2(self, p: *mut u8) {
        // SAFETY: caller supplies two writable bytes and enables this ISA.
        unsafe {
            core::arch::wasm32::v128_store16_lane::<0>(self.0, p.cast());
        }
    }
    #[inline(always)]
    unsafe fn load4(p: *const u8) -> Self {
        // SAFETY: caller supplies four readable bytes and enables this ISA.
        unsafe { Self(core::arch::wasm32::v128_load32_zero(p.cast())) }
    }
    #[inline(always)]
    unsafe fn store4(self, p: *mut u8) {
        // SAFETY: caller supplies four writable bytes and enables this ISA.
        unsafe {
            core::arch::wasm32::v128_store32_lane::<0>(self.0, p.cast());
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
            8,
            8,
            8,
            8,
            8,
            8,
            8,
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
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
        >(self.0, b.0))
    }
    #[inline(always)]
    fn any(self) -> bool {
        i8x16_bitmask(self.0) & 255 != 0
    }
    #[inline(always)]
    fn widen(self) -> W {
        W(i16x8_extend_low_u8x16(self.0))
    }
    #[inline(always)]
    fn narrow(v: W) -> Self {
        Self(u8x16_narrow_i16x8(v.0, v.0))
    }
    /// Byte, word and dword interleaves of row pairs (each a single
    /// unpack on x86), producing two columns per vector, and one shuffle to
    /// bring each odd column to the low half: 16 shuffles instead of the
    /// default's 24 irregular ones.
    #[inline(always)]
    fn transpose(rows: [Self; super::LANES]) -> [Self; super::LANES] {
        let r = rows.map(|row| row.0);
        // `bytes[k]`: rows 2k and 2k + 1 interleaved, (r[2k][j], r[2k+1][j]).
        let bytes: [v128; 4] = core::array::from_fn(|k| {
            i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(
                r[2 * k],
                r[2 * k + 1],
            )
        });
        // Four rows' bytes of columns 0..3 (low) and 4..7 (high) per dword.
        let words = [
            i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(bytes[0], bytes[1]),
            i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(bytes[0], bytes[1]),
            i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(bytes[2], bytes[3]),
            i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(bytes[2], bytes[3]),
        ];
        // Columns 2m (low half) and 2m + 1 (high half), all eight rows.
        let pairs = [
            i32x4_shuffle::<0, 4, 1, 5>(words[0], words[2]),
            i32x4_shuffle::<2, 6, 3, 7>(words[0], words[2]),
            i32x4_shuffle::<0, 4, 1, 5>(words[1], words[3]),
            i32x4_shuffle::<2, 6, 3, 7>(words[1], words[3]),
        ];
        core::array::from_fn(|column| {
            let pair = pairs[column / 2];
            Self(if column % 2 == 0 {
                pair
            } else {
                i64x2_shuffle::<1, 1>(pair, pair)
            })
        })
    }
}
impl Words for W {
    #[inline(always)]
    fn splat(v: i16) -> Self {
        Self(i16x8_splat(v))
    }
    #[inline(always)]
    fn add(self, b: Self) -> Self {
        Self(i16x8_add(self.0, b.0))
    }
    #[inline(always)]
    fn sub(self, b: Self) -> Self {
        Self(i16x8_sub(self.0, b.0))
    }
    #[inline(always)]
    fn shr<const N: i32>(self) -> Self {
        Self(i16x8_shr(self.0, N as u32))
    }
}
impl MaskWords for W {
    #[inline(always)]
    fn lookup(self, table: &WordTable) -> Self {
        // SAFETY: both byte tables contain sixteen bytes; caller enables the ISA.
        unsafe {
            // The lookup contract restricts every word index to 0..=15.
            let index = u8x16_narrow_i16x8(self.0, self.0);
            let low = swizzle_in_bounds(v128_load(table.low.as_ptr().cast()), index);
            let high = swizzle_in_bounds(v128_load(table.high.as_ptr().cast()), index);
            Self(i8x16_shuffle::<
                0,
                16,
                1,
                17,
                2,
                18,
                3,
                19,
                4,
                20,
                5,
                21,
                6,
                22,
                7,
                23,
            >(low, high))
        }
    }
    #[inline(always)]
    fn load(src: &[i16; LANES]) -> Self {
        // SAFETY: `src` holds 16 bytes.
        unsafe { Self(v128_load(src.as_ptr().cast())) }
    }
    #[inline(always)]
    fn store(self, dst: &mut [i16; LANES]) {
        // SAFETY: as `load`.
        unsafe { v128_store(dst.as_mut_ptr().cast(), self.0) }
    }
    #[inline(always)]
    fn abs(self) -> Self {
        Self(i16x8_abs(self.0))
    }
    #[inline(always)]
    fn min(self, b: Self) -> Self {
        Self(i16x8_min(self.0, b.0))
    }
    #[inline(always)]
    fn max(self, b: Self) -> Self {
        Self(i16x8_max(self.0, b.0))
    }
    #[inline(always)]
    fn mul(self, b: Self) -> Self {
        Self(i16x8_mul(self.0, b.0))
    }
    #[inline(always)]
    fn mul_q15_round(self, b: Self) -> Self {
        Self(q15_mul_round(self.0, b.0))
    }
    #[inline(always)]
    fn gt(self, b: Self) -> Self {
        Self(i16x8_gt(self.0, b.0))
    }
    #[inline(always)]
    fn eq(self, b: Self) -> Self {
        Self(i16x8_eq(self.0, b.0))
    }
    #[inline(always)]
    fn and(self, b: Self) -> Self {
        Self(v128_and(self.0, b.0))
    }
    #[inline(always)]
    fn or(self, b: Self) -> Self {
        Self(v128_or(self.0, b.0))
    }
    #[inline(always)]
    fn and_not(self, b: Self) -> Self {
        Self(v128_andnot(self.0, b.0))
    }
    #[inline(always)]
    fn select(self, if_set: Self, if_clear: Self) -> Self {
        Self(select_i16(self.0, if_set.0, if_clear.0))
    }
    #[inline(always)]
    fn any(self) -> bool {
        v128_any_true(self.0)
    }
}
impl QuadBytes for B {
    #[inline(always)]
    unsafe fn load_lines4(src: *const u8) -> [Self; 4] {
        // SAFETY: the caller provides 32 readable bytes.
        let (low, high) = unsafe { (v128_load(src.cast()), v128_load(src.add(16).cast())) };
        // Line l is byte 4k + l of the 32: lanes 0..3 from `low`, 4..7 from
        // `high` (the upper eight bytes are unused).
        [
            Self(i8x16_shuffle::<
                0,
                4,
                8,
                12,
                16,
                20,
                24,
                28,
                0,
                4,
                8,
                12,
                16,
                20,
                24,
                28,
            >(low, high)),
            Self(i8x16_shuffle::<
                1,
                5,
                9,
                13,
                17,
                21,
                25,
                29,
                1,
                5,
                9,
                13,
                17,
                21,
                25,
                29,
            >(low, high)),
            Self(i8x16_shuffle::<
                2,
                6,
                10,
                14,
                18,
                22,
                26,
                30,
                2,
                6,
                10,
                14,
                18,
                22,
                26,
                30,
            >(low, high)),
            Self(i8x16_shuffle::<
                3,
                7,
                11,
                15,
                19,
                23,
                27,
                31,
                3,
                7,
                11,
                15,
                19,
                23,
                27,
                31,
            >(low, high)),
        ]
    }
    #[inline(always)]
    unsafe fn store_lines4(lines: [Self; 4], dst: *mut u8) {
        let [l0, l1, l2, l3] = lines.map(|line| line.0);
        // Lines 0 and 1 (2 and 3) interleaved by byte: lane k's pair is
        // word k; then the pairs interleaved by word give byte 4k + l.
        let pairs01 =
            i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(l0, l1);
        let pairs23 =
            i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(l2, l3);
        let low = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(pairs01, pairs23);
        let high = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(pairs01, pairs23);
        // SAFETY: the caller provides 32 writable bytes.
        unsafe {
            v128_store(dst.cast(), low);
            v128_store(dst.add(16).cast(), high);
        }
    }
    #[inline(always)]
    unsafe fn load_halves(low: *const u8, high: *const u8) -> Self {
        // SAFETY: the caller provides four readable bytes at each pointer.
        unsafe {
            let half = |bytes: *const u8| bytes.cast::<u32>().read_unaligned();
            Self(u32x4(half(low), half(high), 0, 0))
        }
    }
}
