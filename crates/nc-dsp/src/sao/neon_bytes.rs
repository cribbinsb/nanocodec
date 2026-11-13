//! Byte-wide SAO apply adapter; scheduling and scalar tails are shared.
#![allow(clippy::inline_always)]
use super::bytes::Bytes;
use core::arch::aarch64::{
    uint8x16_t, vaddq_u8, vandq_u8, vceqq_u8, vcgtq_u8, vdupq_n_u8, vld1q_u8, vqaddq_u8, vqsubq_u8,
    vshrq_n_u8, vst1q_u8, vsubq_u8,
};
#[derive(Clone, Copy)]
pub(super) struct Lanes(uint8x16_t);
impl Bytes for Lanes {
    const LANES: usize = 16;
    #[inline(always)]
    unsafe fn load(p: *const u8) -> Self {
        // SAFETY: called only within the ISA entry; pointers cover LANES bytes.
        unsafe { Self(vld1q_u8(p)) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut u8) {
        // SAFETY: called only within the ISA entry; pointers cover LANES bytes.
        unsafe { vst1q_u8(p, self.0) }
    }
    #[inline(always)]
    fn splat(v: u8) -> Self {
        // SAFETY: inlined only into the corresponding target-feature entry.
        unsafe { Self(vdupq_n_u8(v)) }
    }
    #[inline(always)]
    fn eq(self, b: Self) -> Self {
        // SAFETY: inlined only into the corresponding target-feature entry.
        unsafe { Self(vceqq_u8(self.0, b.0)) }
    }
    #[inline(always)]
    fn gt(self, b: Self) -> Self {
        // SAFETY: inlined only into the corresponding target-feature entry.
        unsafe { Self(vcgtq_u8(self.0, b.0)) }
    }
    #[inline(always)]
    fn and(self, b: Self) -> Self {
        // SAFETY: inlined only into the corresponding target-feature entry.
        unsafe { Self(vandq_u8(self.0, b.0)) }
    }
    #[inline(always)]
    fn add(self, b: Self) -> Self {
        // SAFETY: inlined only into the corresponding target-feature entry.
        unsafe { Self(vaddq_u8(self.0, b.0)) }
    }
    #[inline(always)]
    fn sub(self, b: Self) -> Self {
        // SAFETY: inlined only into the corresponding target-feature entry.
        unsafe { Self(vsubq_u8(self.0, b.0)) }
    }
    #[inline(always)]
    fn add_sat(self, b: Self) -> Self {
        // SAFETY: inlined only into the corresponding target-feature entry.
        unsafe { Self(vqaddq_u8(self.0, b.0)) }
    }
    #[inline(always)]
    fn sub_sat(self, b: Self) -> Self {
        // SAFETY: inlined only into the corresponding target-feature entry.
        unsafe { Self(vqsubq_u8(self.0, b.0)) }
    }
    #[inline(always)]
    fn band(self) -> Self {
        // SAFETY: inlined only into the corresponding target-feature entry.
        unsafe { Self(vshrq_n_u8::<3>(self.0)) }
    }
}
