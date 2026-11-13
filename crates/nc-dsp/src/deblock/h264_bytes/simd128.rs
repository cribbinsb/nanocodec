//! SIMD128 lanes of the byte-lane H.264 deblocking filter (`h264_bytes`).
//!
//! As the x86 lanes: the floor average is the rounding average minus the
//! dropped bit, `p + delta` adds and subtracts the two signed parts of
//! `delta` with unsigned saturation, and rows transpose with an unpack
//! ladder (here constant shuffles).
#![allow(clippy::inline_always)] // lane operations inline into the kernels

use super::H264Bytes;
use core::arch::wasm32::{
    i8x16_max, i8x16_min, i8x16_narrow_i16x8, i8x16_neg, i8x16_shuffle, i16x8_add,
    i16x8_extend_high_u8x16, i16x8_extend_low_u8x16, i16x8_shl, i16x8_shr, i16x8_shuffle,
    i16x8_splat, i16x8_sub, i32x4_shuffle, i64x2_shuffle, u8x16_add_sat, u8x16_avgr, u8x16_lt,
    u8x16_max, u8x16_min, u8x16_narrow_i16x8, u8x16_splat, u8x16_sub, u8x16_sub_sat, u16x8_shr,
    u64x2, v128, v128_and, v128_andnot, v128_any_true, v128_bitselect, v128_load, v128_load32_lane,
    v128_load64_lane, v128_load64_zero, v128_or, v128_store, v128_store32_lane, v128_store64_lane,
    v128_xor,
};

/// Sixteen byte lanes in one SIMD128 register.
#[derive(Clone, Copy)]
pub(crate) struct B(v128);

impl H264Bytes for B {
    #[inline(always)]
    fn splat(v: u8) -> Self {
        Self(u8x16_splat(v))
    }
    #[inline(always)]
    fn from_halves(lo: u64, hi: u64) -> Self {
        Self(u64x2(lo, hi))
    }
    #[inline(always)]
    unsafe fn load(p: *const u8) -> Self {
        // SAFETY: the caller provides 16 readable bytes (wasm loads need no
        // alignment).
        unsafe { Self(v128_load(p.cast())) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut u8) {
        // SAFETY: the caller provides 16 writable bytes.
        unsafe { v128_store(p.cast(), self.0) }
    }
    #[inline(always)]
    unsafe fn load_halves(lo: *const u8, hi: *const u8) -> Self {
        // SAFETY: the caller provides 8 readable bytes at each pointer.
        unsafe {
            Self(v128_load64_lane::<1>(
                v128_load64_zero(lo.cast()),
                hi.cast(),
            ))
        }
    }
    #[inline(always)]
    unsafe fn store_halves(self, lo: *mut u8, hi: *mut u8) {
        // SAFETY: the caller provides 8 writable bytes at each pointer.
        unsafe {
            v128_store64_lane::<0>(self.0, lo.cast());
            v128_store64_lane::<1>(self.0, hi.cast());
        }
    }
    #[inline(always)]
    unsafe fn load_rows_8(rows: [*const u8; 16]) -> [Self; 8] {
        // Interleave row pairs' bytes, then words, doublewords and
        // quadwords: after each stage a vector holds twice as many rows of
        // half as many columns (the x86 unpack ladder).
        let mut bytes = [u8x16_splat(0); 8];
        for (k, v) in bytes.iter_mut().enumerate() {
            // SAFETY: the caller provides 8 readable bytes at each row.
            *v = unsafe {
                unpack_lo8(
                    v128_load64_zero(rows[2 * k].cast()),
                    v128_load64_zero(rows[2 * k + 1].cast()),
                )
            };
        }
        let mut words = [u8x16_splat(0); 8];
        for j in 0..4 {
            words[2 * j] = unpack_lo16(bytes[2 * j], bytes[2 * j + 1]);
            words[2 * j + 1] = unpack_hi16(bytes[2 * j], bytes[2 * j + 1]);
        }
        let mut dwords = [u8x16_splat(0); 8];
        for h in 0..2 {
            for c in 0..2 {
                let (top, bottom) = (words[4 * h + c], words[4 * h + 2 + c]);
                dwords[4 * h + 2 * c] = unpack_lo32(top, bottom);
                dwords[4 * h + 2 * c + 1] = unpack_hi32(top, bottom);
            }
        }
        let mut out = [Self::splat(0); 8];
        for pair in 0..4 {
            let (low, high) = (dwords[pair], dwords[4 + pair]);
            out[2 * pair] = Self(unpack_lo64(low, high));
            out[2 * pair + 1] = Self(unpack_hi64(low, high));
        }
        out
    }
    #[inline(always)]
    unsafe fn store_rows_8(columns: [Self; 8], rows: [*mut u8; 16]) {
        // The ladder backwards: column pairs' bytes (lines 0..8 low, 8..16
        // high), then words, then doublewords: two whole rows per vector.
        let mut bytes = [u8x16_splat(0); 8];
        for k in 0..4 {
            let (a, b) = (columns[2 * k].0, columns[2 * k + 1].0);
            bytes[k] = unpack_lo8(a, b);
            bytes[4 + k] = unpack_hi8(a, b);
        }
        let mut words = [u8x16_splat(0); 8];
        for half in 0..2 {
            let base = 4 * half;
            words[2 * half] = unpack_lo16(bytes[base], bytes[base + 1]);
            words[2 * half + 1] = unpack_hi16(bytes[base], bytes[base + 1]);
            words[4 + 2 * half] = unpack_lo16(bytes[base + 2], bytes[base + 3]);
            words[4 + 2 * half + 1] = unpack_hi16(bytes[base + 2], bytes[base + 3]);
        }
        for j in 0..4 {
            let pairs = [
                unpack_lo32(words[j], words[4 + j]),
                unpack_hi32(words[j], words[4 + j]),
            ];
            for (k, pair) in pairs.into_iter().enumerate() {
                let line = 4 * j + 2 * k;
                // SAFETY: the caller provides 8 writable bytes at each row.
                unsafe { Self(pair).store_halves(rows[line], rows[line + 1]) };
            }
        }
    }
    #[inline(always)]
    unsafe fn load_rows_4(rows: [*const u8; 16]) -> [Self; 4] {
        let mut quads = [u8x16_splat(0); 4];
        for (k, quad) in quads.iter_mut().enumerate() {
            // SAFETY: the caller provides 4 readable bytes at each row.
            *quad = unsafe {
                let mut v = u8x16_splat(0);
                v = v128_load32_lane::<0>(v, rows[4 * k].cast());
                v = v128_load32_lane::<1>(v, rows[4 * k + 1].cast());
                v = v128_load32_lane::<2>(v, rows[4 * k + 2].cast());
                v128_load32_lane::<3>(v, rows[4 * k + 3].cast())
            };
        }
        let [a, b, c, d] = transpose_4x4_dwords([
            transpose_4x4_bytes(quads[0]),
            transpose_4x4_bytes(quads[1]),
            transpose_4x4_bytes(quads[2]),
            transpose_4x4_bytes(quads[3]),
        ]);
        [Self(a), Self(b), Self(c), Self(d)]
    }
    #[inline(always)]
    unsafe fn store_rows_4(columns: [Self; 4], rows: [*mut u8; 16]) {
        // Both 4x4 transposes are their own inverses.
        let dwords = transpose_4x4_dwords([columns[0].0, columns[1].0, columns[2].0, columns[3].0]);
        for (k, &dword) in dwords.iter().enumerate() {
            let quad = transpose_4x4_bytes(dword);
            // SAFETY: the caller provides 4 writable bytes at each row.
            unsafe {
                v128_store32_lane::<0>(quad, rows[4 * k].cast());
                v128_store32_lane::<1>(quad, rows[4 * k + 1].cast());
                v128_store32_lane::<2>(quad, rows[4 * k + 2].cast());
                v128_store32_lane::<3>(quad, rows[4 * k + 3].cast());
            }
        }
    }
    #[inline(always)]
    fn abs_diff(self, b: Self) -> Self {
        Self(v128_or(
            u8x16_sub_sat(self.0, b.0),
            u8x16_sub_sat(b.0, self.0),
        ))
    }
    #[inline(always)]
    fn less(self, b: Self) -> Self {
        Self(u8x16_lt(self.0, b.0))
    }
    #[inline(always)]
    fn and(self, b: Self) -> Self {
        Self(v128_and(self.0, b.0))
    }
    #[inline(always)]
    fn and_not(self, b: Self) -> Self {
        Self(v128_andnot(self.0, b.0))
    }
    #[inline(always)]
    fn select(self, a: Self, b: Self) -> Self {
        Self(v128_bitselect(a.0, b.0, self.0))
    }
    #[inline(always)]
    fn sub(self, b: Self) -> Self {
        Self(u8x16_sub(self.0, b.0))
    }
    #[inline(always)]
    fn add_sat(self, b: Self) -> Self {
        Self(u8x16_add_sat(self.0, b.0))
    }
    #[inline(always)]
    fn sub_sat(self, b: Self) -> Self {
        Self(u8x16_sub_sat(self.0, b.0))
    }
    #[inline(always)]
    fn min(self, b: Self) -> Self {
        Self(u8x16_min(self.0, b.0))
    }
    #[inline(always)]
    fn max(self, b: Self) -> Self {
        Self(u8x16_max(self.0, b.0))
    }
    #[inline(always)]
    fn avg_round(self, b: Self) -> Self {
        Self(u8x16_avgr(self.0, b.0))
    }
    #[inline(always)]
    fn avg_floor(self, b: Self) -> Self {
        // The rounding average exceeds the floor by one where the sum is odd.
        let odd = v128_and(v128_xor(self.0, b.0), u8x16_splat(1));
        Self(u8x16_sub(u8x16_avgr(self.0, b.0), odd))
    }
    #[inline(always)]
    fn any(self) -> bool {
        v128_any_true(self.0)
    }
    #[inline(always)]
    fn normal_delta(p1: Self, p0: Self, q0: Self, q1: Self) -> Self {
        // Words hold the exact differences (|x| ≤ 255; the sum fits i16);
        // the narrow saturates to i8.
        let half = |w: fn(v128) -> v128| {
            let d = i16x8_shl(i16x8_sub(w(q0.0), w(p0.0)), 2);
            let e = i16x8_sub(w(p1.0), w(q1.0));
            i16x8_shr(i16x8_add(i16x8_add(d, e), i16x8_splat(4)), 3)
        };
        Self(i8x16_narrow_i16x8(
            half(i16x8_extend_low_u8x16),
            half(i16x8_extend_high_u8x16),
        ))
    }
    #[inline(always)]
    fn clip_signed(self, t: Self) -> Self {
        Self(i8x16_min(i8x16_max(self.0, i8x16_neg(t.0)), t.0))
    }
    #[inline(always)]
    fn add_signed(self, delta: Self) -> Self {
        let zero = u8x16_splat(0);
        let up = i8x16_max(delta.0, zero);
        let down = i8x16_max(i8x16_neg(delta.0), zero);
        Self(u8x16_sub_sat(u8x16_add_sat(self.0, up), down))
    }
    #[inline(always)]
    fn sub_signed(self, delta: Self) -> Self {
        // `delta > −128`, so its negation is exact.
        self.add_signed(Self(i8x16_neg(delta.0)))
    }
    #[inline(always)]
    fn strong_side(x3: Self, x2: Self, x1: Self, x0: Self, y0: Self, y1: Self) -> [Self; 3] {
        // Sums are at most 8 · 255, inside u16; the narrow never saturates.
        let half = |w: fn(v128) -> v128| {
            let (x3, x2, x1, x0, y0, y1) = (w(x3.0), w(x2.0), w(x1.0), w(x0.0), w(y0.0), w(y1.0));
            let s = i16x8_add(i16x8_add(x1, x0), y0);
            let s2 = i16x8_add(s, x2);
            [
                // (2·x3 + 3·x2 + x1 + x0 + y0 + 4) >> 3
                u16x8_shr(
                    i16x8_add(
                        i16x8_add(i16x8_shl(i16x8_add(x3, x2), 1), s2),
                        i16x8_splat(4),
                    ),
                    3,
                ),
                // (x2 + x1 + x0 + y0 + 2) >> 2
                u16x8_shr(i16x8_add(s2, i16x8_splat(2)), 2),
                // (x2 + 2·x1 + 2·x0 + 2·y0 + y1 + 4) >> 3
                u16x8_shr(
                    i16x8_add(i16x8_add(i16x8_add(s, s2), y1), i16x8_splat(4)),
                    3,
                ),
            ]
        };
        let (low, high) = (half(i16x8_extend_low_u8x16), half(i16x8_extend_high_u8x16));
        [
            Self(u8x16_narrow_i16x8(low[0], high[0])),
            Self(u8x16_narrow_i16x8(low[1], high[1])),
            Self(u8x16_narrow_i16x8(low[2], high[2])),
        ]
    }
}

#[inline(always)]
fn unpack_lo8(a: v128, b: v128) -> v128 {
    i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(a, b)
}

#[inline(always)]
fn unpack_hi8(a: v128, b: v128) -> v128 {
    i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(a, b)
}

#[inline(always)]
fn unpack_lo16(a: v128, b: v128) -> v128 {
    i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(a, b)
}

#[inline(always)]
fn unpack_hi16(a: v128, b: v128) -> v128 {
    i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(a, b)
}

#[inline(always)]
fn unpack_lo32(a: v128, b: v128) -> v128 {
    i32x4_shuffle::<0, 4, 1, 5>(a, b)
}

#[inline(always)]
fn unpack_hi32(a: v128, b: v128) -> v128 {
    i32x4_shuffle::<2, 6, 3, 7>(a, b)
}

#[inline(always)]
fn unpack_lo64(a: v128, b: v128) -> v128 {
    i64x2_shuffle::<0, 2>(a, b)
}

#[inline(always)]
fn unpack_hi64(a: v128, b: v128) -> v128 {
    i64x2_shuffle::<1, 3>(a, b)
}

/// Transpose of each 4x4 byte matrix (four rows of four bytes) in place.
#[inline(always)]
fn transpose_4x4_bytes(v: v128) -> v128 {
    i8x16_shuffle::<0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15>(v, v)
}

/// Transpose of the 4x4 matrix of 32-bit elements (vector `i`, element `j`).
#[inline(always)]
fn transpose_4x4_dwords(v: [v128; 4]) -> [v128; 4] {
    let (lo01, hi01) = (unpack_lo32(v[0], v[1]), unpack_hi32(v[0], v[1]));
    let (lo23, hi23) = (unpack_lo32(v[2], v[3]), unpack_hi32(v[2], v[3]));
    [
        unpack_lo64(lo01, lo23),
        unpack_hi64(lo01, lo23),
        unpack_lo64(hi01, hi23),
        unpack_hi64(hi01, hi23),
    ]
}
