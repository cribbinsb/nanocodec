//! AVX2 paired eight-byte groups for sixteen independent AV1 edge lines.
//! Each 128-bit half holds eight live bytes; byte unpacks widen both groups
//! directly into sixteen i16 lanes. Tile IO loads/transposes both groups together.

#![allow(clippy::inline_always)]
use super::{Bytes, PairedBytes, Words};
use core::arch::x86_64::{
    __m256i, _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_loadl_epi64, _mm_storel_epi64,
    _mm256_add_epi16, _mm256_adds_epi8, _mm256_adds_epu8, _mm256_and_si256, _mm256_castsi256_si128,
    _mm256_cmpeq_epi8, _mm256_extracti128_si256, _mm256_max_epu8, _mm256_movemask_epi8,
    _mm256_packs_epi16, _mm256_packus_epi16, _mm256_set_m128i, _mm256_set1_epi8, _mm256_set1_epi16,
    _mm256_setzero_si256, _mm256_srai_epi16, _mm256_srli_epi16, _mm256_srli_si256,
    _mm256_sub_epi16, _mm256_subs_epi8, _mm256_subs_epu8, _mm256_unpacklo_epi8, _mm256_xor_si256,
};
#[derive(Clone, Copy)]
pub(crate) struct B(__m256i);
#[derive(Clone, Copy)]
pub(crate) struct W(__m256i);
impl Bytes for B {
    type W = W;
    #[inline(always)]
    fn splat(v: u8) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm256_set1_epi8(v as i8)) }
    }
    #[inline(always)]
    unsafe fn load(p: *const u8) -> Self {
        // SAFETY: caller supplies eight readable bytes and enables AVX2.
        unsafe { Self::load_pair(p, p) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut u8) {
        // SAFETY: caller supplies eight writable bytes and enables AVX2.
        unsafe {
            _mm_storel_epi64(p.cast(), _mm256_castsi256_si128(self.0));
        }
    }
    #[inline(always)]
    unsafe fn load4(p: *const u8) -> Self {
        // SAFETY: caller supplies four readable bytes and enables AVX2.
        unsafe {
            let v = _mm_cvtsi32_si128(p.cast::<i32>().read_unaligned());
            Self(_mm256_set_m128i(v, v))
        }
    }
    #[inline(always)]
    unsafe fn store4(self, p: *mut u8) {
        // SAFETY: caller supplies four writable bytes and enables AVX2.
        unsafe {
            p.cast::<i32>()
                .write_unaligned(_mm_cvtsi128_si32(_mm256_castsi256_si128(self.0)));
        }
    }
    #[inline(always)]
    unsafe fn store2(self, p: *mut u8) {
        // SAFETY: caller supplies two writable bytes and enables AVX2.
        unsafe {
            p.cast::<u16>()
                .write_unaligned(_mm_cvtsi128_si32(_mm256_castsi256_si128(self.0)) as u16);
        }
    }
    #[inline(always)]
    fn max(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm256_max_epu8(self.0, b.0)) }
    }
    #[inline(always)]
    fn sub_sat(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm256_subs_epu8(self.0, b.0)) }
    }
    #[inline(always)]
    fn add_sat(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm256_adds_epu8(self.0, b.0)) }
    }
    #[inline(always)]
    fn signed_add(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm256_adds_epi8(self.0, b.0)) }
    }
    #[inline(always)]
    fn signed_sub(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm256_subs_epi8(self.0, b.0)) }
    }
    #[inline(always)]
    fn and(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm256_and_si256(self.0, b.0)) }
    }
    #[inline(always)]
    fn xor(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm256_xor_si256(self.0, b.0)) }
    }
    #[inline(always)]
    fn eq(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm256_cmpeq_epi8(self.0, b.0)) }
    }
    #[inline(always)]
    fn shr<const N: i32>(self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe {
            Self(_mm256_and_si256(
                _mm256_srli_epi16::<N>(self.0),
                _mm256_set1_epi8((255_u8 >> N) as i8),
            ))
        }
    }
    #[inline(always)]
    fn signed_shr<const N: i32>(self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe {
            Self(_mm256_packs_epi16(
                _mm256_srai_epi16::<N>(_mm256_srai_epi16::<8>(_mm256_unpacklo_epi8(
                    self.0, self.0,
                ))),
                _mm256_setzero_si256(),
            ))
        }
    }
    #[inline(always)]
    fn zip_lo(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm256_unpacklo_epi8(self.0, b.0)) }
    }
    #[inline(always)]
    fn zip_hi(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm256_srli_si256::<8>(_mm256_unpacklo_epi8(self.0, b.0))) }
    }
    #[inline(always)]
    fn any(self) -> bool {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { _mm256_movemask_epi8(self.0) & 0x00ff_00ff != 0 }
    }
    #[inline(always)]
    fn widen(self) -> W {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { W(_mm256_unpacklo_epi8(self.0, _mm256_setzero_si256())) }
    }
    #[inline(always)]
    fn narrow(v: W) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm256_packus_epi16(v.0, v.0)) }
    }
}
impl Words for W {
    #[inline(always)]
    fn splat(v: i16) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(_mm256_set1_epi16(v)) }
    }
    #[inline(always)]
    fn add(self, b: Self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(_mm256_add_epi16(self.0, b.0)) }
    }
    #[inline(always)]
    fn sub(self, b: Self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(_mm256_sub_epi16(self.0, b.0)) }
    }
    #[inline(always)]
    fn shr<const N: i32>(self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(_mm256_srai_epi16::<N>(self.0)) }
    }
}

impl PairedBytes for B {
    #[inline(always)]
    unsafe fn load_pair(first: *const u8, second: *const u8) -> Self {
        // SAFETY: caller supplies two readable eight-byte spans and AVX2.
        unsafe {
            Self(_mm256_set_m128i(
                _mm_loadl_epi64(second.cast()),
                _mm_loadl_epi64(first.cast()),
            ))
        }
    }
    #[inline(always)]
    unsafe fn store_pair(self, first: *mut u8, second: *mut u8) {
        // SAFETY: caller supplies two writable eight-byte spans and AVX2.
        unsafe {
            _mm_storel_epi64(first.cast(), _mm256_castsi256_si128(self.0));
            _mm_storel_epi64(second.cast(), _mm256_extracti128_si256::<1>(self.0));
        }
    }
}
