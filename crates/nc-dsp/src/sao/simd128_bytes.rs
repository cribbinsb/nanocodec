//! Byte-wide SAO apply adapter for SIMD128 (`Lanes` for `sao::bytes::edge`,
//! plus an `i8x16.swizzle` LUT `band` kernel that replaces the 4-iteration
//! compare/mask/add loop and avoids V8's widened byte-shift lowering).
#![allow(clippy::inline_always)]

use super::Rect;
use super::bytes::Bytes;
use core::arch::wasm32::{
    i8x16_swizzle, u8x16, u8x16_add, u8x16_add_sat, u8x16_eq, u8x16_gt, u8x16_splat, u8x16_sub,
    u8x16_sub_sat, u16x8_shr, v128, v128_and, v128_load, v128_store,
};

const LANES: usize = 16;

#[derive(Clone, Copy)]
pub(super) struct Lanes(v128);
impl Bytes for Lanes {
    const LANES: usize = LANES;
    #[inline(always)]
    unsafe fn load(p: *const u8) -> Self {
        // SAFETY: called only within the ISA entry; pointers cover LANES bytes.
        unsafe { Self(v128_load(p.cast())) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut u8) {
        // SAFETY: called only within the ISA entry; pointers cover LANES bytes.
        unsafe { v128_store(p.cast(), self.0) }
    }
    #[inline(always)]
    fn splat(v: u8) -> Self {
        Self(u8x16_splat(v))
    }
    #[inline(always)]
    fn eq(self, b: Self) -> Self {
        Self(u8x16_eq(self.0, b.0))
    }
    #[inline(always)]
    fn gt(self, b: Self) -> Self {
        Self(u8x16_gt(self.0, b.0))
    }
    #[inline(always)]
    fn table(values: [u8; 16]) -> Self {
        // SAFETY: the SIMD128 entry is active, and the table is a full 16 bytes.
        unsafe { Self(v128_load(values.as_ptr().cast())) }
    }

    #[inline(always)]
    fn lookup(self, table: Self) -> Self {
        Self(i8x16_swizzle(table.0, self.0))
    }

    #[inline(always)]
    fn and(self, b: Self) -> Self {
        Self(v128_and(self.0, b.0))
    }
    #[inline(always)]
    fn add(self, b: Self) -> Self {
        Self(u8x16_add(self.0, b.0))
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
}

#[target_feature(enable = "simd128")]
pub(super) unsafe fn band(
    plane: &mut [u8],
    stride: usize,
    rect: Rect,
    position: u8,
    off: [i8; 4],
    exempt: &[bool],
) {
    let pos = [
        off[0].max(0) as u8,
        off[1].max(0) as u8,
        off[2].max(0) as u8,
        off[3].max(0) as u8,
    ];
    let neg = [
        off[0].min(0).unsigned_abs(),
        off[1].min(0).unsigned_abs(),
        off[2].min(0).unsigned_abs(),
        off[3].min(0).unsigned_abs(),
    ];
    let pos_lut = u8x16(
        pos[0], pos[1], pos[2], pos[3], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    );
    let neg_lut = u8x16(
        neg[0], neg[1], neg[2], neg[3], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    );
    let v_pos = u8x16_splat(position);
    let mask31 = u8x16_splat(31);
    let zero = u8x16_splat(0);
    let width = rect.x1 - rect.x0;
    let has_exempt = !exempt.is_empty();
    // SAFETY: dispatch checked the rectangle and packed mask footprint. Each
    // vector stays inside a row; bool has a one-byte 0/1 representation.
    unsafe {
        for y in rect.y0..rect.y1 {
            let row = plane.as_mut_ptr().add(y * stride + rect.x0);
            let mask_start = (y - rect.y0) * width;
            let mut col = 0;
            while col + LANES <= width {
                let cur: v128 = v128_load(row.add(col).cast());
                // Logical right shift by 3 via 16-bit shift + mask avoids V8's
                // widened byte-shift lowering; result is in 0..=31.
                let b = v128_and(u16x8_shr(cur, 3), mask31);
                let idx = v128_and(u8x16_sub(b, v_pos), mask31);
                // Plain i8x16_swizzle is spec-guaranteed to return 0 for idx >= 16.
                let mut plus = i8x16_swizzle(pos_lut, idx);
                let mut minus = i8x16_swizzle(neg_lut, idx);
                if has_exempt {
                    let enabled = u8x16_eq(
                        v128_load(exempt.as_ptr().cast::<u8>().add(mask_start + col).cast()),
                        zero,
                    );
                    plus = v128_and(plus, enabled);
                    minus = v128_and(minus, enabled);
                }
                let res = u8x16_sub_sat(u8x16_add_sat(cur, plus), minus);
                v128_store(row.add(col).cast(), res);
                col += LANES;
            }
            for x in col..width {
                if !has_exempt || !exempt[mask_start + x] {
                    let p = row.add(x);
                    *p = super::scalar::band_sample(*p, position, off);
                }
            }
        }
    }
}
