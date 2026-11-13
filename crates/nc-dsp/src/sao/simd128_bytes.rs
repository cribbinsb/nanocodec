//! Byte-wide SAO apply adapter; scheduling and scalar tails are shared.
#![allow(clippy::inline_always)]
use super::bytes::Bytes;
use core::arch::wasm32::{
    u8x16_add, u8x16_add_sat, u8x16_eq, u8x16_gt, u8x16_shr, u8x16_splat, u8x16_sub, u8x16_sub_sat,
    v128, v128_and, v128_load, v128_store,
};
#[derive(Clone, Copy)]
pub(super) struct Lanes(v128);
impl Bytes for Lanes {
    const LANES: usize = 16;
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
    #[inline(always)]
    fn band(self) -> Self {
        Self(u8x16_shr(self.0, 3))
    }
}
