//! Byte-wide SAO apply adapter; scheduling and scalar tails are shared.
#![allow(clippy::inline_always)]
use super::bytes::Bytes;
use core::arch::x86_64::{
    __m256i, _mm256_add_epi8, _mm256_adds_epu8, _mm256_and_si256, _mm256_cmpeq_epi8,
    _mm256_cmpgt_epi8, _mm256_loadu_si256, _mm256_set1_epi8, _mm256_srli_epi16,
    _mm256_storeu_si256, _mm256_sub_epi8, _mm256_subs_epu8, _mm256_xor_si256,
};
#[derive(Clone, Copy)]
pub(super) struct Lanes(__m256i);
impl Bytes for Lanes {
    const LANES: usize = 32;
    #[inline(always)]
    unsafe fn load(p: *const u8) -> Self {
        // SAFETY: called only within the ISA entry; pointers cover LANES bytes.
        unsafe { Self(_mm256_loadu_si256(p.cast())) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut u8) {
        // SAFETY: called only within the ISA entry; pointers cover LANES bytes.
        unsafe { _mm256_storeu_si256(p.cast(), self.0) }
    }
    #[inline(always)]
    fn splat(v: u8) -> Self {
        // SAFETY: inlined only into the corresponding target-feature entry.
        unsafe { Self(_mm256_set1_epi8(v as i8)) }
    }
    #[inline(always)]
    fn eq(self, b: Self) -> Self {
        // SAFETY: inlined only into the corresponding target-feature entry.
        unsafe { Self(_mm256_cmpeq_epi8(self.0, b.0)) }
    }
    #[inline(always)]
    fn gt(self, b: Self) -> Self {
        // SAFETY: inlined only into the corresponding target-feature entry.
        unsafe {
            Self(_mm256_cmpgt_epi8(
                _mm256_xor_si256(self.0, _mm256_set1_epi8(-128)),
                _mm256_xor_si256(b.0, _mm256_set1_epi8(-128)),
            ))
        }
    }
    #[inline(always)]
    fn and(self, b: Self) -> Self {
        // SAFETY: inlined only into the corresponding target-feature entry.
        unsafe { Self(_mm256_and_si256(self.0, b.0)) }
    }
    #[inline(always)]
    fn add(self, b: Self) -> Self {
        // SAFETY: inlined only into the corresponding target-feature entry.
        unsafe { Self(_mm256_add_epi8(self.0, b.0)) }
    }
    #[inline(always)]
    fn sub(self, b: Self) -> Self {
        // SAFETY: inlined only into the corresponding target-feature entry.
        unsafe { Self(_mm256_sub_epi8(self.0, b.0)) }
    }
    #[inline(always)]
    fn add_sat(self, b: Self) -> Self {
        // SAFETY: inlined only into the corresponding target-feature entry.
        unsafe { Self(_mm256_adds_epu8(self.0, b.0)) }
    }
    #[inline(always)]
    fn sub_sat(self, b: Self) -> Self {
        // SAFETY: inlined only into the corresponding target-feature entry.
        unsafe { Self(_mm256_subs_epu8(self.0, b.0)) }
    }
    #[inline(always)]
    fn band(self) -> Self {
        // SAFETY: inlined only into the corresponding target-feature entry.
        unsafe {
            Self(_mm256_and_si256(
                _mm256_srli_epi16::<3>(self.0),
                _mm256_set1_epi8(31),
            ))
        }
    }
}
