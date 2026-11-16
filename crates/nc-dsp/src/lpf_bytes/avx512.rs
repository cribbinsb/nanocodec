//! Four independent eight-line byte groups for the shared loop-filter family.
//! Each 128-bit quarter holds eight live bytes and widens to eight i16 words,
//! exactly the AVX2 paired layout. AV1/VP9 use groups for independent lines;
//! AV2 vertical ramps use one group per row, with eight tap lanes per group.
//! AV2 decisions keep eight narrow segment lanes; only the ramp rows widen.
//!
//! dav1d 1.5.4 `loopfilter_avx512.asm` (8-bit) and `loopfilter16_avx512.asm`
//! (high depth) informed the grouped transposes and widened filtering; idiomatic
//! Rust intrinsics retain our shared scalar equations. GFNI replaces byte
//! shifts without widening; older AVX-512 CPUs keep AVX2. VBMI is not required.
//! Copyright 2018, 2022 VideoLAN/dav1d authors and Two Orioles, LLC, BSD-2-Clause;
//! see `THIRD_PARTY.md` and `docs/licenses/dav1d/COPYING`.
#![allow(clippy::inline_always)]
use super::{Bytes, GroupedBytes, LANES, MaskWords, QuadBytes, WordTable, Words};
use core::arch::x86_64::{
    __m512i, _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_loadl_epi64, _mm_loadu_si128,
    _mm_storel_epi64, _mm_storeu_si128, _mm512_abs_epi16, _mm512_add_epi16, _mm512_adds_epi8,
    _mm512_adds_epu8, _mm512_and_si512, _mm512_andnot_si512, _mm512_broadcast_i32x4,
    _mm512_broadcastq_epi64, _mm512_bsrli_epi128, _mm512_castsi128_si512, _mm512_castsi512_si128,
    _mm512_cmpeq_epi8_mask, _mm512_cmpeq_epi16_mask, _mm512_cmpgt_epi16_mask,
    _mm512_extracti32x4_epi32, _mm512_gf2p8affine_epi64_epi8, _mm512_inserti32x4, _mm512_max_epi16,
    _mm512_max_epu8, _mm512_min_epi16, _mm512_movepi8_mask, _mm512_movm_epi8, _mm512_movm_epi16,
    _mm512_mulhrs_epi16, _mm512_mullo_epi16, _mm512_or_si512, _mm512_packus_epi16,
    _mm512_set_epi64, _mm512_set1_epi8, _mm512_set1_epi16, _mm512_set1_epi64, _mm512_setzero_si512,
    _mm512_shuffle_epi8, _mm512_sra_epi16, _mm512_sub_epi16, _mm512_subs_epi8, _mm512_subs_epu8,
    _mm512_ternarylogic_epi32, _mm512_test_epi64_mask, _mm512_unpacklo_epi8, _mm512_xor_si512,
};
// Low/high byte indices of each row's signed i16 step, repeated per qword.
const ROW_STEP_BYTES: [i64; 4] = [
    0x0100_0100_0100_0100,
    0x0302_0302_0302_0302,
    0x0504_0504_0504_0504,
    0x0706_0706_0706_0706,
];

#[derive(Clone, Copy)]
pub(crate) struct B(__m512i);
#[derive(Clone, Copy)]
pub(crate) struct W(__m512i);
impl Bytes for B {
    type W = W;
    #[inline(always)]
    fn splat(v: u8) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm512_set1_epi8(v as i8)) }
    }
    #[inline(always)]
    unsafe fn load(p: *const u8) -> Self {
        // SAFETY: caller supplies eight bytes and enables the ISA.
        unsafe { Self::load_groups(p, 0) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut u8) {
        // SAFETY: caller supplies eight writable bytes and enables the ISA.
        unsafe {
            _mm_storel_epi64(p.cast(), _mm512_castsi512_si128(self.0));
        }
    }
    #[inline(always)]
    unsafe fn load4(p: *const u8) -> Self {
        // SAFETY: caller supplies four bytes and enables the ISA.
        unsafe {
            Self(_mm512_broadcast_i32x4(_mm_cvtsi32_si128(
                p.cast::<i32>().read_unaligned(),
            )))
        }
    }
    #[inline(always)]
    unsafe fn store4(self, p: *mut u8) {
        // SAFETY: caller supplies four writable bytes and enables AVX-512F/BW/VL.
        unsafe {
            p.cast::<i32>()
                .write_unaligned(_mm_cvtsi128_si32(_mm512_castsi512_si128(self.0)));
        }
    }
    #[inline(always)]
    unsafe fn store2(self, p: *mut u8) {
        // SAFETY: caller supplies two writable bytes and enables AVX-512F/BW/VL.
        unsafe {
            p.cast::<u16>()
                .write_unaligned(_mm_cvtsi128_si32(_mm512_castsi512_si128(self.0)) as u16);
        }
    }
    #[inline(always)]
    fn max(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm512_max_epu8(self.0, b.0)) }
    }
    #[inline(always)]
    fn sub_sat(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm512_subs_epu8(self.0, b.0)) }
    }
    #[inline(always)]
    fn add_sat(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm512_adds_epu8(self.0, b.0)) }
    }
    #[inline(always)]
    fn signed_add(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm512_adds_epi8(self.0, b.0)) }
    }
    #[inline(always)]
    fn signed_sub(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm512_subs_epi8(self.0, b.0)) }
    }
    #[inline(always)]
    fn and(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm512_and_si512(self.0, b.0)) }
    }
    #[inline(always)]
    fn xor(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm512_xor_si512(self.0, b.0)) }
    }
    #[inline(always)]
    fn eq(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm512_movm_epi8(_mm512_cmpeq_epi8_mask(self.0, b.0))) }
    }
    #[inline(always)]
    fn shr<const N: i32>(self) -> Self {
        // SAFETY: only the AVX-512 + GFNI table uses these byte lanes.
        unsafe {
            Self(_mm512_gf2p8affine_epi64_epi8::<0>(
                self.0,
                _mm512_set1_epi64(const { super::gfni::shift_matrix(N, false) }),
            ))
        }
    }
    #[inline(always)]
    fn signed_shr<const N: i32>(self) -> Self {
        // SAFETY: GFNI is enabled by the checked ISA entry; the matrix sign-extends.
        unsafe {
            Self(_mm512_gf2p8affine_epi64_epi8::<0>(
                self.0,
                _mm512_set1_epi64(const { super::gfni::shift_matrix(N, true) }),
            ))
        }
    }
    #[inline(always)]
    fn zip_lo(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm512_unpacklo_epi8(self.0, b.0)) }
    }
    #[inline(always)]
    fn zip_hi(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm512_bsrli_epi128::<8>(_mm512_unpacklo_epi8(self.0, b.0))) }
    }
    #[inline(always)]
    fn any(self) -> bool {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { _mm512_movepi8_mask(self.0) & 0x00ff_00ff_00ff_00ff != 0 }
    }
    #[inline(always)]
    fn widen(self) -> W {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { W(_mm512_unpacklo_epi8(self.0, _mm512_setzero_si512())) }
    }
    #[inline(always)]
    fn narrow(v: W) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm512_packus_epi16(v.0, v.0)) }
    }
}
impl Words for W {
    #[inline(always)]
    fn splat(v: i16) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(_mm512_set1_epi16(v)) }
    }
    #[inline(always)]
    fn add(self, b: Self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(_mm512_add_epi16(self.0, b.0)) }
    }
    #[inline(always)]
    fn sub(self, b: Self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(_mm512_sub_epi16(self.0, b.0)) }
    }
    #[inline(always)]
    fn shr<const N: i32>(self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(_mm512_sra_epi16(self.0, _mm_cvtsi32_si128(N))) }
    }
}

impl GroupedBytes for B {
    const GROUPS: usize = 4;
    #[inline(always)]
    unsafe fn load_groups(first: *const u8, distance: usize) -> Self {
        // SAFETY: caller validates all four eight-byte spans and enables the ISA.
        unsafe {
            let a = _mm512_castsi128_si512(_mm_loadl_epi64(first.cast()));
            let a = _mm512_inserti32x4::<1>(a, _mm_loadl_epi64(first.add(distance).cast()));
            let a = _mm512_inserti32x4::<2>(a, _mm_loadl_epi64(first.add(2 * distance).cast()));
            Self(_mm512_inserti32x4::<3>(
                a,
                _mm_loadl_epi64(first.add(3 * distance).cast()),
            ))
        }
    }
    #[inline(always)]
    unsafe fn store_groups(self, first: *mut u8, distance: usize) {
        // SAFETY: caller validates all four writable spans and enables the ISA.
        unsafe {
            _mm_storel_epi64(first.cast(), _mm512_castsi512_si128(self.0));
            _mm_storel_epi64(
                first.add(distance).cast(),
                _mm512_extracti32x4_epi32::<1>(self.0),
            );
            _mm_storel_epi64(
                first.add(2 * distance).cast(),
                _mm512_extracti32x4_epi32::<2>(self.0),
            );
            _mm_storel_epi64(
                first.add(3 * distance).cast(),
                _mm512_extracti32x4_epi32::<3>(self.0),
            );
        }
    }
}
impl MaskWords for W {
    #[inline(always)]
    fn lookup(self, table: &WordTable) -> Self {
        // SAFETY: tables hold sixteen bytes, indices are 0..15; ISA enabled.
        unsafe {
            let index = _mm512_packus_epi16(self.0, self.0);
            let low = _mm512_shuffle_epi8(
                _mm512_broadcast_i32x4(_mm_loadu_si128(table.low.as_ptr().cast())),
                index,
            );
            let high = _mm512_shuffle_epi8(
                _mm512_broadcast_i32x4(_mm_loadu_si128(table.high.as_ptr().cast())),
                index,
            );
            Self(_mm512_unpacklo_epi8(low, high))
        }
    }
    #[inline(always)]
    fn load(src: &[i16; LANES]) -> Self {
        // SAFETY: eight readable words; ISA enabled. Decision lanes are
        // replicated, ramp factors address eight taps in each independent row.
        unsafe { Self(_mm512_broadcast_i32x4(_mm_loadu_si128(src.as_ptr().cast()))) }
    }
    #[inline(always)]
    fn store(self, dst: &mut [i16; LANES]) {
        // SAFETY: eight writable words; decision results agree in all groups.
        unsafe {
            _mm_storeu_si128(dst.as_mut_ptr().cast(), _mm512_castsi512_si128(self.0));
        }
    }
    #[inline(always)]
    fn abs(self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm512_abs_epi16(self.0)) }
    }
    #[inline(always)]
    fn min(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm512_min_epi16(self.0, b.0)) }
    }
    #[inline(always)]
    fn max(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm512_max_epi16(self.0, b.0)) }
    }
    #[inline(always)]
    fn mul(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm512_mullo_epi16(self.0, b.0)) }
    }
    #[inline(always)]
    fn mul_q15_round(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm512_mulhrs_epi16(self.0, b.0)) }
    }
    #[inline(always)]
    fn gt(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm512_movm_epi16(_mm512_cmpgt_epi16_mask(self.0, b.0))) }
    }
    #[inline(always)]
    fn eq(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm512_movm_epi16(_mm512_cmpeq_epi16_mask(self.0, b.0))) }
    }
    #[inline(always)]
    fn and(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm512_and_si512(self.0, b.0)) }
    }
    #[inline(always)]
    fn or(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm512_or_si512(self.0, b.0)) }
    }
    #[inline(always)]
    fn and_not(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm512_andnot_si512(b.0, self.0)) }
    }
    #[inline(always)]
    fn select(self, if_set: Self, if_clear: Self) -> Self {
        // SAFETY: mask bits select if_set when set (truth table 0xca).
        unsafe {
            Self(_mm512_ternarylogic_epi32::<0xca>(
                self.0, if_set.0, if_clear.0,
            ))
        }
    }
    #[inline(always)]
    fn any(self) -> bool {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { _mm512_test_epi64_mask(self.0, self.0) != 0 }
    }
}
impl QuadBytes for B {
    #[inline(always)]
    unsafe fn load_lines4(src: *const u8) -> [Self; 4] {
        // SAFETY: caller supplies 32 bytes and enables AVX2 plus AVX-512.
        unsafe {
            let narrow = super::avx2::B::load_lines4(src);
            let mut result = [Self::splat(0); 4];
            for i in 0..4 {
                result[i] = Self(_mm512_broadcast_i32x4(narrow[i].0));
            }
            result
        }
    }
    #[inline(always)]
    unsafe fn store_lines4(lines: [Self; 4], dst: *mut u8) {
        // SAFETY: caller supplies 32 writable bytes and enables the ISA.
        unsafe {
            let mut narrow = [super::avx2::B::splat(0); 4];
            for i in 0..4 {
                narrow[i] = super::avx2::B(_mm512_castsi512_si128(lines[i].0));
            }
            super::avx2::B::store_lines4(narrow, dst);
        }
    }
    #[inline(always)]
    unsafe fn load_halves(low: *const u8, high: *const u8) -> Self {
        // SAFETY: caller supplies four bytes per pointer and enables the ISA.
        unsafe {
            Self(_mm512_broadcast_i32x4(
                super::avx2::B::load_halves(low, high).0,
            ))
        }
    }
    #[inline(always)]
    unsafe fn ramp_lines4(
        negative: *mut u8,
        positive: *mut u8,
        along: usize,
        steps: [i16; 4],
        factors_negative: &[i16; LANES],
        factors_positive: &[i16; LANES],
    ) {
        // Each 128-bit group repeats the step of its own row. A byte
        // shuffle broadcasts four signed i16 values without scalar products.
        // SAFETY: `steps` holds eight readable bytes, and both four-row windows
        // are validated by the caller, which also enables AVX-512F/BW/VL.
        unsafe {
            let row_steps = _mm512_broadcastq_epi64(_mm_loadl_epi64(steps.as_ptr().cast()));
            let order = _mm512_set_epi64(
                ROW_STEP_BYTES[3],
                ROW_STEP_BYTES[3],
                ROW_STEP_BYTES[2],
                ROW_STEP_BYTES[2],
                ROW_STEP_BYTES[1],
                ROW_STEP_BYTES[1],
                ROW_STEP_BYTES[0],
                ROW_STEP_BYTES[0],
            );
            let steps = W(_mm512_shuffle_epi8(row_steps, order));
            let p = Self::load_groups(negative, along).widen();
            let q = Self::load_groups(positive, along).widen();
            Self::narrow(p.add(steps.mul_q15_round(W::load(factors_negative))))
                .store_groups(negative, along);
            Self::narrow(q.sub(steps.mul_q15_round(W::load(factors_positive))))
                .store_groups(positive, along);
        }
    }
}
