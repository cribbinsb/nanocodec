//! AVX2 byte lanes of the dav1d-layout loop filters (`deblock::lanes`):
//! all 32 bytes of a vector are lines, words are a pair of vectors (the
//! low and high eight bytes of each 128-bit half), and vertical edges
//! transpose 4, 8 or 16 columns of 32 rows with the byte perfect shuffle
//! (rows `i` and `i + 16` share a vector, one per 128-bit half), as dav1d
//! 1.5.4 `src/x86/loopfilter_avx2.asm` (BSD-2-Clause, see `THIRD_PARTY.md`).

use super::{Bytes, Words};
use crate::deblock::lanes::kernel::LaneBytes;
use crate::deblock::lanes::{LaneEdge, UNIT_LINES};
use core::arch::x86_64::{
    __m128i, __m256i, _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_extract_epi32, _mm_insert_epi32,
    _mm_loadl_epi64, _mm_loadu_si128, _mm_storel_epi64, _mm_storeu_si128, _mm_unpackhi_epi64,
    _mm256_add_epi16, _mm256_adds_epi8, _mm256_adds_epu8, _mm256_and_si256, _mm256_blendv_epi8,
    _mm256_castsi256_si128, _mm256_cmpeq_epi8, _mm256_cmpgt_epi32, _mm256_extracti128_si256,
    _mm256_loadu_si256, _mm256_maddubs_epi16, _mm256_max_epu8, _mm256_movemask_epi8,
    _mm256_mulhrs_epi16, _mm256_packus_epi16, _mm256_set_m128i, _mm256_set1_epi8,
    _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setr_epi8, _mm256_setzero_si256,
    _mm256_shuffle_epi8, _mm256_srai_epi16, _mm256_srli_epi16, _mm256_srli_epi32,
    _mm256_storeu_si256, _mm256_sub_epi8, _mm256_sub_epi16, _mm256_subs_epi8, _mm256_subs_epu8,
    _mm256_testz_si256, _mm256_unpackhi_epi8, _mm256_unpackhi_epi64, _mm256_unpacklo_epi8,
    _mm256_unpacklo_epi64, _mm256_xor_si256,
};

/// 32 lines, one per byte.
#[derive(Clone, Copy)]
pub(crate) struct B(__m256i);
/// The 32 lines widened: the low and high eight bytes of each 128-bit half.
#[derive(Clone, Copy)]
pub(crate) struct W(__m256i, __m256i);

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
        unsafe {
            Self(_mm256_set_m128i(
                _mm_cvtsi32_si128(0),
                _mm_loadl_epi64(p.cast()),
            ))
        }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut u8) {
        // SAFETY: caller supplies eight writable bytes and enables AVX2.
        unsafe { _mm_storel_epi64(p.cast(), _mm256_castsi256_si128(self.0)) }
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
    unsafe fn load4(p: *const u8) -> Self {
        // SAFETY: caller supplies four readable bytes and enables AVX2.
        unsafe {
            let v = _mm_cvtsi32_si128(p.cast::<i32>().read_unaligned());
            Self(_mm256_set_m128i(_mm_cvtsi32_si128(0), v))
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
    /// dav1d's byte shift: clear the low bits, shift the words, and
    /// restore the sign with `(v ^ s) − s`.
    #[inline(always)]
    fn signed_shr<const N: i32>(self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe {
            let sign = _mm256_set1_epi8((0x80_u8 >> N) as i8);
            let high = _mm256_and_si256(self.0, _mm256_set1_epi8((0xff_u8 << N) as i8));
            let shifted = _mm256_srli_epi16::<N>(high);
            Self(_mm256_sub_epi8(_mm256_xor_si256(shifted, sign), sign))
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
        unsafe { Self(_mm256_unpackhi_epi8(self.0, b.0)) }
    }
    #[inline(always)]
    fn any(self) -> bool {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { _mm256_testz_si256(self.0, self.0) == 0 }
    }
    #[inline(always)]
    fn widen(self) -> W {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe {
            let zero = _mm256_setzero_si256();
            W(
                _mm256_unpacklo_epi8(self.0, zero),
                _mm256_unpackhi_epi8(self.0, zero),
            )
        }
    }
    #[inline(always)]
    fn narrow(v: W) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm256_packus_epi16(v.0, v.1)) }
    }
    fn transpose(_rows: [Self; super::LANES]) -> [Self; super::LANES] {
        unreachable!("32-line groups transpose through LaneBytes")
    }
}

impl Words for W {
    #[inline(always)]
    fn splat(v: i16) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(_mm256_set1_epi16(v), _mm256_set1_epi16(v)) }
    }
    #[inline(always)]
    fn add(self, b: Self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(_mm256_add_epi16(self.0, b.0), _mm256_add_epi16(self.1, b.1)) }
    }
    #[inline(always)]
    fn sub(self, b: Self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(_mm256_sub_epi16(self.0, b.0), _mm256_sub_epi16(self.1, b.1)) }
    }
    #[inline(always)]
    fn shr<const N: i32>(self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe {
            Self(
                _mm256_srai_epi16::<N>(self.0),
                _mm256_srai_epi16::<N>(self.1),
            )
        }
    }
}

/// One byte round of the perfect shuffle on `N` vectors: vector `i` and
/// `i + N/2` interleave into outputs `2i` (low bytes) and `2i + 1` (high).
#[inline(always)]
fn shuffle_round<const N: usize>(v: &[__m256i; N]) -> [__m256i; N] {
    let mut out = *v;
    for i in 0..N / 2 {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe {
            out[2 * i] = _mm256_unpacklo_epi8(v[i], v[i + N / 2]);
            out[2 * i + 1] = _mm256_unpackhi_epi8(v[i], v[i + N / 2]);
        }
    }
    out
}

/// 128-bit `low` in the low half and `high` in the high half.
#[inline(always)]
fn halves(low: __m128i, high: __m128i) -> __m256i {
    // SAFETY: the ISA entry point enables these instructions.
    unsafe { _mm256_set_m128i(high, low) }
}

/// Rows of the 32-row group: row `r` and `r + 16` share a vector.
const HALF_ROWS: isize = 16;

impl LaneBytes for B {
    const LINES: usize = 32;
    #[inline(always)]
    unsafe fn load_units(p: *const LaneEdge) -> Self {
        // SAFETY: caller supplies the group's eight entries (32 bytes).
        unsafe { Self(_mm256_loadu_si256(p.cast())) }
    }
    #[inline(always)]
    unsafe fn clear_units(p: *mut LaneEdge) {
        // SAFETY: caller supplies the group's eight entries (32 bytes).
        unsafe { _mm256_storeu_si256(p.cast(), _mm256_setzero_si256()) }
    }
    #[inline(always)]
    fn group_width(units: Self) -> u8 {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe {
            let widths = _mm256_srli_epi32::<24>(units.0);
            if _mm256_testz_si256(widths, widths) != 0 {
                return 0;
            }
            let above = |w: i32| {
                _mm256_movemask_epi8(_mm256_cmpgt_epi32(widths, _mm256_set1_epi32(w))) != 0
            };
            1 + u8::from(above(1)) + u8::from(above(2))
        }
    }
    #[inline(always)]
    fn unit_byte<const K: i32>(units: Self) -> Self {
        let k = K as i8;
        // SAFETY: the ISA entry point enables these instructions.
        unsafe {
            let index = _mm256_setr_epi8(
                k,
                k,
                k,
                k,
                k + 4,
                k + 4,
                k + 4,
                k + 4,
                k + 8,
                k + 8,
                k + 8,
                k + 8,
                k + 12,
                k + 12,
                k + 12,
                k + 12,
                k,
                k,
                k,
                k,
                k + 4,
                k + 4,
                k + 4,
                k + 4,
                k + 8,
                k + 8,
                k + 8,
                k + 8,
                k + 12,
                k + 12,
                k + 12,
                k + 12,
            );
            Self(_mm256_shuffle_epi8(units.0, index))
        }
    }
    #[inline(always)]
    fn width_at_least<const WIDTH: i32>(units: Self) -> Self {
        // A unit's entry is the dword of its four lanes.
        // SAFETY: the ISA entry point enables these instructions.
        unsafe {
            Self(_mm256_cmpgt_epi32(
                _mm256_srli_epi32::<24>(units.0),
                _mm256_set1_epi32(WIDTH - 1),
            ))
        }
    }
    #[inline(always)]
    fn blend(self, new: Self, mask: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(_mm256_blendv_epi8(self.0, new.0, mask.0)) }
    }
    #[inline(always)]
    fn madd<const WA: i8, const WB: i8>(a: Self, b: Self) -> W {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe {
            let weights = _mm256_set1_epi16(i16::from_le_bytes([WA as u8, WB as u8]));
            W(
                _mm256_maddubs_epi16(_mm256_unpacklo_epi8(a.0, b.0), weights),
                _mm256_maddubs_epi16(_mm256_unpackhi_epi8(a.0, b.0), weights),
            )
        }
    }
    #[inline(always)]
    fn round_shr<const SHIFT: i32>(w: W) -> W {
        // `mulhrs` by 2^(15 − SHIFT) is `(w + 2^(SHIFT−1)) >> SHIFT`.
        // SAFETY: the ISA entry point enables these instructions.
        unsafe {
            let scale = _mm256_set1_epi16(1 << (15 - SHIFT));
            W(
                _mm256_mulhrs_epi16(w.0, scale),
                _mm256_mulhrs_epi16(w.1, scale),
            )
        }
    }
    #[inline(always)]
    unsafe fn load_rows<const N: usize>(first: *const u8, stride: isize) -> [Self; N] {
        let mut rows = [Self::splat(0); N];
        for (i, row) in rows.iter_mut().enumerate() {
            // SAFETY: caller supplies `N` readable rows of 32 bytes.
            *row = unsafe { Self(_mm256_loadu_si256(first.offset(i as isize * stride).cast())) };
        }
        rows
    }
    #[inline(always)]
    unsafe fn store_row(self, p: *mut u8) {
        // SAFETY: caller supplies 32 writable bytes.
        unsafe { _mm256_storeu_si256(p.cast(), self.0) }
    }
    #[inline(always)]
    unsafe fn load_columns<const N: usize>(first: *const u8, stride: isize) -> [Self; N] {
        let row = |r: isize| first.wrapping_offset(r * stride);
        let mut out = [Self::splat(0); N];
        // SAFETY: caller supplies `N` readable bytes on each of 32 rows.
        unsafe {
            match N {
                16 => {
                    let mut v = [_mm256_setzero_si256(); 16];
                    for (i, v) in v.iter_mut().enumerate() {
                        let i = i as isize;
                        *v = halves(
                            _mm_loadu_si128(row(i).cast()),
                            _mm_loadu_si128(row(i + HALF_ROWS).cast()),
                        );
                    }
                    for _ in 0..4 {
                        v = shuffle_round(&v);
                    }
                    for (o, v) in out.iter_mut().zip(v) {
                        *o = Self(v);
                    }
                }
                8 => {
                    // Vector i: rows i, i + 8 (low half), i + 16, i + 24.
                    let pair = |r: isize| {
                        let low = _mm_loadl_epi64(row(r).cast());
                        let high = _mm_loadl_epi64(row(r + 8).cast());
                        core::arch::x86_64::_mm_unpacklo_epi64(low, high)
                    };
                    let mut v = [_mm256_setzero_si256(); 8];
                    for (i, v) in v.iter_mut().enumerate() {
                        let i = i as isize;
                        *v = halves(pair(i), pair(i + HALF_ROWS));
                    }
                    for _ in 0..3 {
                        v = shuffle_round(&v);
                    }
                    // v[k]: columns 2k, 2k + 1 of rows 0..7 | v[k + 4]: rows 8..15.
                    for k in 0..4 {
                        out[2 * k] = Self(_mm256_unpacklo_epi64(v[k], v[k + 4]));
                        out[2 * k + 1] = Self(_mm256_unpackhi_epi64(v[k], v[k + 4]));
                    }
                }
                4 => {
                    // Vector k: rows 4k..4k + 3 as dwords, and 16 rows on.
                    let quad = |r: isize| {
                        let mut v = _mm_cvtsi32_si128(row(r).cast::<i32>().read_unaligned());
                        v = _mm_insert_epi32::<1>(v, row(r + 1).cast::<i32>().read_unaligned());
                        v = _mm_insert_epi32::<2>(v, row(r + 2).cast::<i32>().read_unaligned());
                        _mm_insert_epi32::<3>(v, row(r + 3).cast::<i32>().read_unaligned())
                    };
                    let mut v = [_mm256_setzero_si256(); 4];
                    for (k, v) in v.iter_mut().enumerate() {
                        let r = 4 * k as isize;
                        *v = halves(quad(r), quad(r + HALF_ROWS));
                    }
                    for _ in 0..4 {
                        v = shuffle_round(&v);
                    }
                    for (o, v) in out.iter_mut().zip(v) {
                        *o = Self(v);
                    }
                }
                _ => unreachable!("4, 8 or 16 columns"),
            }
        }
        out
    }
    #[inline(always)]
    unsafe fn store_columns<const N: usize>(columns: &[Self; N], first: *mut u8, stride: isize) {
        let row = |r: isize| first.wrapping_offset(r * stride);
        // A loop, not `array::map`: its closure would be compiled outside
        // this function's target features.
        let mut v = [Self::splat(0).0; N];
        for (v, c) in v.iter_mut().zip(columns) {
            *v = c.0;
        }
        // SAFETY: caller supplies `N` writable bytes on each of 32 rows.
        unsafe {
            match N {
                16 => {
                    for _ in 0..4 {
                        v = shuffle_round(&v);
                    }
                    for (i, v) in v.iter().enumerate() {
                        let i = i as isize;
                        _mm_storeu_si128(row(i).cast(), _mm256_castsi256_si128(*v));
                        _mm_storeu_si128(
                            row(i + HALF_ROWS).cast(),
                            _mm256_extracti128_si256::<1>(*v),
                        );
                    }
                }
                8 => {
                    for _ in 0..3 {
                        v = shuffle_round(&v);
                    }
                    // v[k]: rows 2k, 2k + 1 (and 16 rows on) as qwords.
                    for (k, v) in v.iter().enumerate() {
                        let r = 2 * k as isize;
                        for (half, base) in [
                            (_mm256_castsi256_si128(*v), r),
                            (_mm256_extracti128_si256::<1>(*v), r + HALF_ROWS),
                        ] {
                            _mm_storel_epi64(row(base).cast(), half);
                            _mm_storel_epi64(row(base + 1).cast(), _mm_unpackhi_epi64(half, half));
                        }
                    }
                }
                4 => {
                    for _ in 0..2 {
                        v = shuffle_round(&v);
                    }
                    // v[k]: rows 4k..4k + 3 (and 16 rows on) as dwords.
                    for (k, v) in v.iter().enumerate() {
                        let r = 4 * k as isize;
                        for (half, base) in [
                            (_mm256_castsi256_si128(*v), r),
                            (_mm256_extracti128_si256::<1>(*v), r + HALF_ROWS),
                        ] {
                            row(base)
                                .cast::<i32>()
                                .write_unaligned(_mm_cvtsi128_si32(half));
                            row(base + 1)
                                .cast::<i32>()
                                .write_unaligned(_mm_extract_epi32::<1>(half));
                            row(base + 2)
                                .cast::<i32>()
                                .write_unaligned(_mm_extract_epi32::<2>(half));
                            row(base + 3)
                                .cast::<i32>()
                                .write_unaligned(_mm_extract_epi32::<3>(half));
                        }
                    }
                }
                _ => unreachable!("4, 8 or 16 columns"),
            }
        }
    }
}

/// Lines of one entry, as the layout assumes.
const _: () = assert!(UNIT_LINES == 4);
