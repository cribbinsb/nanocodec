//! AVX2 byte and word lanes of the packed loop filters (`lpf_bytes`).

use super::{Bytes, LANES, MaskWords, QuadBytes, WordTable, Words};
use core::arch::x86_64::{
    __m128i, _mm_add_epi16, _mm_adds_epi8, _mm_adds_epu8, _mm_and_si128, _mm_cmpeq_epi8,
    _mm_cvtepi8_epi16, _mm_cvtepu8_epi16, _mm_loadl_epi64, _mm_max_epu8, _mm_movemask_epi8,
    _mm_packs_epi16, _mm_packus_epi16, _mm_set1_epi8, _mm_set1_epi16, _mm_setzero_si128,
    _mm_srai_epi16, _mm_srli_epi16, _mm_srli_si128, _mm_storel_epi64, _mm_sub_epi16, _mm_subs_epi8,
    _mm_subs_epu8, _mm_unpacklo_epi8, _mm_xor_si128,
};
use core::arch::x86_64::{
    _mm_abs_epi16, _mm_andnot_si128, _mm_blendv_epi8, _mm_cmpeq_epi16, _mm_cmpgt_epi16,
    _mm_cvtsi32_si128, _mm_insert_epi32, _mm_loadu_si128, _mm_max_epi16, _mm_min_epi16,
    _mm_mulhrs_epi16, _mm_mullo_epi16, _mm_or_si128, _mm_shuffle_epi8, _mm_shuffle_epi32,
    _mm_storeu_si128, _mm_testz_si128, _mm_unpackhi_epi32, _mm_unpackhi_epi64, _mm_unpacklo_epi32,
    _mm_unpacklo_epi64,
};
#[derive(Clone, Copy)]
pub(crate) struct B(__m128i);
#[derive(Clone, Copy)]
pub(crate) struct W(__m128i);
impl Bytes for B {
    type W = W;
    #[inline(always)]
    fn splat(v: u8) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm_set1_epi8(v as i8)) }
    }
    #[inline(always)]
    unsafe fn load(p: *const u8) -> Self {
        // SAFETY: the caller provides eight readable bytes and enables the ISA.
        unsafe { Self(_mm_loadl_epi64(p.cast())) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut u8) {
        // SAFETY: the caller provides eight writable bytes and enables the ISA.
        unsafe {
            _mm_storel_epi64(p.cast(), self.0);
        }
    }
    #[inline(always)]
    unsafe fn store2(self, p: *mut u8) {
        // SAFETY: caller supplies two writable bytes and enables this ISA.
        unsafe {
            p.cast::<u16>()
                .write_unaligned(core::arch::x86_64::_mm_extract_epi16::<0>(self.0) as u16);
        }
    }
    #[inline(always)]
    unsafe fn load4(p: *const u8) -> Self {
        // SAFETY: caller supplies four readable bytes and enables this ISA.
        unsafe { Self(_mm_cvtsi32_si128(p.cast::<i32>().read_unaligned())) }
    }
    #[inline(always)]
    unsafe fn store4(self, p: *mut u8) {
        // SAFETY: caller supplies four writable bytes and enables this ISA.
        unsafe {
            p.cast::<i32>()
                .write_unaligned(core::arch::x86_64::_mm_cvtsi128_si32(self.0));
        }
    }
    #[inline(always)]
    fn max(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm_max_epu8(self.0, b.0)) }
    }
    #[inline(always)]
    fn sub_sat(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm_subs_epu8(self.0, b.0)) }
    }
    #[inline(always)]
    fn add_sat(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm_adds_epu8(self.0, b.0)) }
    }
    #[inline(always)]
    fn signed_add(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm_adds_epi8(self.0, b.0)) }
    }
    #[inline(always)]
    fn signed_sub(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm_subs_epi8(self.0, b.0)) }
    }
    #[inline(always)]
    fn and(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm_and_si128(self.0, b.0)) }
    }
    #[inline(always)]
    fn xor(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm_xor_si128(self.0, b.0)) }
    }
    #[inline(always)]
    fn eq(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm_cmpeq_epi8(self.0, b.0)) }
    }
    #[inline(always)]
    fn shr<const N: i32>(self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe {
            Self(_mm_and_si128(
                _mm_srli_epi16::<N>(self.0),
                _mm_set1_epi8((255_u8 >> N) as i8),
            ))
        }
    }
    #[inline(always)]
    fn signed_shr<const N: i32>(self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe {
            Self(_mm_packs_epi16(
                _mm_srai_epi16::<N>(_mm_cvtepi8_epi16(self.0)),
                _mm_setzero_si128(),
            ))
        }
    }
    #[inline(always)]
    fn zip_lo(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm_unpacklo_epi8(self.0, b.0)) }
    }
    #[inline(always)]
    fn zip_hi(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm_srli_si128::<8>(_mm_unpacklo_epi8(self.0, b.0))) }
    }
    #[inline(always)]
    fn any(self) -> bool {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { _mm_movemask_epi8(self.0) & 255 != 0 }
    }
    #[inline(always)]
    fn widen(self) -> W {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { W(_mm_cvtepu8_epi16(self.0)) }
    }
    #[inline(always)]
    fn narrow(v: W) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm_packus_epi16(v.0, v.0)) }
    }
}
impl Words for W {
    #[inline(always)]
    fn splat(v: i16) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(_mm_set1_epi16(v)) }
    }
    #[inline(always)]
    fn add(self, b: Self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(_mm_add_epi16(self.0, b.0)) }
    }
    #[inline(always)]
    fn sub(self, b: Self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(_mm_sub_epi16(self.0, b.0)) }
    }
    #[inline(always)]
    fn shr<const N: i32>(self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(_mm_srai_epi16::<N>(self.0)) }
    }
}
impl MaskWords for W {
    #[inline(always)]
    fn lookup(self, table: &WordTable) -> Self {
        // SAFETY: both byte tables contain sixteen bytes; caller enables the ISA.
        unsafe {
            let index = _mm_packus_epi16(self.0, self.0);
            let low = _mm_shuffle_epi8(_mm_loadu_si128(table.low.as_ptr().cast()), index);
            let high = _mm_shuffle_epi8(_mm_loadu_si128(table.high.as_ptr().cast()), index);
            Self(_mm_unpacklo_epi8(low, high))
        }
    }
    #[inline(always)]
    fn load(src: &[i16; LANES]) -> Self {
        // SAFETY: `src` holds 16 bytes; only reached through the ISA entry
        // point.
        unsafe { Self(_mm_loadu_si128(src.as_ptr().cast())) }
    }
    #[inline(always)]
    fn store(self, dst: &mut [i16; LANES]) {
        // SAFETY: as `load`.
        unsafe { _mm_storeu_si128(dst.as_mut_ptr().cast(), self.0) }
    }
    #[inline(always)]
    fn abs(self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(_mm_abs_epi16(self.0)) }
    }
    #[inline(always)]
    fn min(self, b: Self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(_mm_min_epi16(self.0, b.0)) }
    }
    #[inline(always)]
    fn max(self, b: Self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(_mm_max_epi16(self.0, b.0)) }
    }
    #[inline(always)]
    fn mul(self, b: Self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(_mm_mullo_epi16(self.0, b.0)) }
    }
    #[inline(always)]
    fn mul_q15_round(self, b: Self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(_mm_mulhrs_epi16(self.0, b.0)) }
    }
    #[inline(always)]
    fn gt(self, b: Self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(_mm_cmpgt_epi16(self.0, b.0)) }
    }
    #[inline(always)]
    fn eq(self, b: Self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(_mm_cmpeq_epi16(self.0, b.0)) }
    }
    #[inline(always)]
    fn and(self, b: Self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(_mm_and_si128(self.0, b.0)) }
    }
    #[inline(always)]
    fn or(self, b: Self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(_mm_or_si128(self.0, b.0)) }
    }
    #[inline(always)]
    fn and_not(self, b: Self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(_mm_andnot_si128(b.0, self.0)) }
    }
    #[inline(always)]
    fn select(self, if_set: Self, if_clear: Self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(_mm_blendv_epi8(if_clear.0, if_set.0, self.0)) }
    }
    #[inline(always)]
    fn any(self) -> bool {
        // SAFETY: only reached through the ISA entry point.
        unsafe { _mm_testz_si128(self.0, self.0) == 0 }
    }
}
/// `_mm_shuffle_epi32` order of dwords 0, 2, 1, 3: the low halves of two
/// line pairs first.
const LOW_HALVES_FIRST: i32 = 0b11_01_10_00;
/// Byte order that turns 16 bytes of four interleaved lines (byte `4k + l`)
/// into four 4-byte groups by line (byte `4l + k`); it is its own inverse.
const LINES4_ORDER: [i8; 16] = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15];
impl QuadBytes for B {
    #[inline(always)]
    unsafe fn load_lines4(src: *const u8) -> [Self; 4] {
        // SAFETY: the caller provides 32 readable bytes and the ISA.
        unsafe {
            let order = _mm_loadu_si128(LINES4_ORDER.as_ptr().cast());
            // Lanes 0..3 and 4..7, each grouped by line.
            let low = _mm_shuffle_epi8(_mm_loadu_si128(src.cast()), order);
            let high = _mm_shuffle_epi8(_mm_loadu_si128(src.add(16).cast()), order);
            // Lines 0 and 1 (low and high eight bytes), 2 and 3.
            let lines01 = _mm_unpacklo_epi32(low, high);
            let lines23 = _mm_unpackhi_epi32(low, high);
            [
                Self(lines01),
                Self(_mm_srli_si128::<8>(lines01)),
                Self(lines23),
                Self(_mm_srli_si128::<8>(lines23)),
            ]
        }
    }
    #[inline(always)]
    unsafe fn store_lines4(lines: [Self; 4], dst: *mut u8) {
        // SAFETY: the caller provides 32 writable bytes and the ISA.
        unsafe {
            let order = _mm_loadu_si128(LINES4_ORDER.as_ptr().cast());
            let lines01 = _mm_unpacklo_epi64(lines[0].0, lines[1].0);
            let lines23 = _mm_unpacklo_epi64(lines[2].0, lines[3].0);
            // Dwords: line 0 lanes 0..3, line 1 lanes 0..3, line 0 lanes
            // 4..7, line 1 lanes 4..7 (and lines 2, 3).
            let lines01 = _mm_shuffle_epi32::<LOW_HALVES_FIRST>(lines01);
            let lines23 = _mm_shuffle_epi32::<LOW_HALVES_FIRST>(lines23);
            let low = _mm_shuffle_epi8(_mm_unpacklo_epi64(lines01, lines23), order);
            let high = _mm_shuffle_epi8(_mm_unpackhi_epi64(lines01, lines23), order);
            _mm_storeu_si128(dst.cast(), low);
            _mm_storeu_si128(dst.add(16).cast(), high);
        }
    }
    #[inline(always)]
    unsafe fn load_halves(low: *const u8, high: *const u8) -> Self {
        // SAFETY: the caller provides four readable bytes at each pointer
        // and the ISA.
        unsafe {
            let half = |bytes: *const u8| bytes.cast::<i32>().read_unaligned();
            Self(_mm_insert_epi32::<1>(
                _mm_cvtsi32_si128(half(low)),
                half(high),
            ))
        }
    }
}
