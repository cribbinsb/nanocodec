//! SSE4.1 lanes (128-bit, used inside the AVX2 kernels) of the byte-lane
//! H.264 deblocking filter (`h264_bytes`).
#![allow(clippy::inline_always)] // lane operations inline into the kernels
//!
//! x86 has no unsigned byte compare, no floor average and no signed
//! accumulate into unsigned bytes: `a < b` is `max(a, b) != a`, the floor
//! average is the rounding `pavgb` minus the dropped bit, and `p + delta`
//! adds the positive and subtracts the negative part of `delta` with
//! unsigned saturation (one of them is zero). Rows transpose with the
//! unpack ladder (bytes, words, doublewords, quadwords).

use super::H264Bytes;
use core::arch::x86_64::{
    __m128i, _mm_add_epi16, _mm_adds_epu8, _mm_and_si128, _mm_andnot_si128, _mm_avg_epu8,
    _mm_blendv_epi8, _mm_castpd_si128, _mm_castsi128_pd, _mm_cmpeq_epi8, _mm_cvtsi32_si128,
    _mm_cvtsi128_si32, _mm_extract_epi32, _mm_insert_epi32, _mm_loadh_pd, _mm_loadl_epi64,
    _mm_loadu_si128, _mm_max_epi8, _mm_max_epu8, _mm_min_epi8, _mm_min_epu8, _mm_movemask_epi8,
    _mm_or_si128, _mm_packs_epi16, _mm_packus_epi16, _mm_set_epi64x, _mm_set1_epi8, _mm_set1_epi16,
    _mm_setr_epi8, _mm_setzero_si128, _mm_shuffle_epi8, _mm_slli_epi16, _mm_srai_epi16,
    _mm_srli_epi16, _mm_storeh_pd, _mm_storel_epi64, _mm_storeu_si128, _mm_sub_epi8, _mm_sub_epi16,
    _mm_subs_epu8, _mm_unpackhi_epi8, _mm_unpackhi_epi16, _mm_unpackhi_epi32, _mm_unpackhi_epi64,
    _mm_unpacklo_epi8, _mm_unpacklo_epi16, _mm_unpacklo_epi32, _mm_unpacklo_epi64, _mm_xor_si128,
};

/// Sixteen byte lanes in one SSE register.
#[derive(Clone, Copy)]
pub(crate) struct B(__m128i);

// Every intrinsic below is SSE4.1 or older, which the AVX2 kernels using
// these lanes enable; register-only intrinsics have no other precondition.
impl H264Bytes for B {
    #[inline(always)]
    fn splat(v: u8) -> Self {
        // SAFETY: register-only (see above).
        unsafe { Self(_mm_set1_epi8(v as i8)) }
    }
    #[inline(always)]
    fn from_halves(lo: u64, hi: u64) -> Self {
        // SAFETY: register-only.
        unsafe { Self(_mm_set_epi64x(hi as i64, lo as i64)) }
    }
    #[inline(always)]
    unsafe fn load(p: *const u8) -> Self {
        // SAFETY: the caller provides 16 readable bytes.
        unsafe { Self(_mm_loadu_si128(p.cast())) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut u8) {
        // SAFETY: the caller provides 16 writable bytes.
        unsafe { _mm_storeu_si128(p.cast(), self.0) }
    }
    #[inline(always)]
    unsafe fn load_halves(lo: *const u8, hi: *const u8) -> Self {
        // SAFETY: the caller provides 8 readable bytes at each pointer.
        unsafe {
            let low = _mm_castsi128_pd(_mm_loadl_epi64(lo.cast()));
            Self(_mm_castpd_si128(_mm_loadh_pd(low, hi.cast())))
        }
    }
    #[inline(always)]
    unsafe fn store_halves(self, lo: *mut u8, hi: *mut u8) {
        // SAFETY: the caller provides 8 writable bytes at each pointer.
        unsafe {
            _mm_storel_epi64(lo.cast(), self.0);
            _mm_storeh_pd(hi.cast(), _mm_castsi128_pd(self.0));
        }
    }
    #[inline(always)]
    unsafe fn load_rows_8(rows: [*const u8; 16]) -> [Self; 8] {
        // Interleave row pairs' bytes, then words, doublewords and
        // quadwords: after each stage a vector holds twice as many rows of
        // half as many columns.
        let mut bytes = [zero(); 8];
        for (k, v) in bytes.iter_mut().enumerate() {
            // SAFETY: the caller provides 8 readable bytes at each row.
            *v = unsafe {
                _mm_unpacklo_epi8(
                    _mm_loadl_epi64(rows[2 * k].cast()),
                    _mm_loadl_epi64(rows[2 * k + 1].cast()),
                )
            };
        }
        // SAFETY: register-only.
        unsafe {
            // Rows 4j..4j + 4, columns 0..4 (even k) or 4..8 (odd k).
            let mut words = [zero(); 8];
            for j in 0..4 {
                words[2 * j] = _mm_unpacklo_epi16(bytes[2 * j], bytes[2 * j + 1]);
                words[2 * j + 1] = _mm_unpackhi_epi16(bytes[2 * j], bytes[2 * j + 1]);
            }
            // Rows 8h..8h + 8, two columns each: {0, 1}, {2, 3}, {4, 5},
            // {6, 7} for h = 0 (vectors 0..4) and h = 1 (4..8).
            let mut dwords = [zero(); 8];
            for h in 0..2 {
                for c in 0..2 {
                    let (top, bottom) = (words[4 * h + c], words[4 * h + 2 + c]);
                    dwords[4 * h + 2 * c] = _mm_unpacklo_epi32(top, bottom);
                    dwords[4 * h + 2 * c + 1] = _mm_unpackhi_epi32(top, bottom);
                }
            }
            let mut out = [Self::splat(0); 8];
            for pair in 0..4 {
                let (low, high) = (dwords[pair], dwords[4 + pair]);
                out[2 * pair] = Self(_mm_unpacklo_epi64(low, high));
                out[2 * pair + 1] = Self(_mm_unpackhi_epi64(low, high));
            }
            out
        }
    }
    #[inline(always)]
    unsafe fn store_rows_8(columns: [Self; 8], rows: [*mut u8; 16]) {
        // The ladder backwards: interleave column pairs' bytes (lines 0..8
        // in the low unpack, 8..16 in the high), then words, then
        // doublewords, which leaves two whole rows per vector.
        // SAFETY: register-only.
        let rows_pairs = unsafe {
            let mut bytes = [zero(); 8];
            for k in 0..4 {
                let (a, b) = (columns[2 * k].0, columns[2 * k + 1].0);
                bytes[k] = _mm_unpacklo_epi8(a, b);
                bytes[4 + k] = _mm_unpackhi_epi8(a, b);
            }
            // Lines 4j..4j + 4 of columns 0..4 (`words[j]`) and 4..8
            // (`words[4 + j]`), j over lines 0..16 in fours.
            let mut words = [zero(); 8];
            for half in 0..2 {
                let base = 4 * half;
                words[2 * half] = _mm_unpacklo_epi16(bytes[base], bytes[base + 1]);
                words[2 * half + 1] = _mm_unpackhi_epi16(bytes[base], bytes[base + 1]);
                words[4 + 2 * half] = _mm_unpacklo_epi16(bytes[base + 2], bytes[base + 3]);
                words[4 + 2 * half + 1] = _mm_unpackhi_epi16(bytes[base + 2], bytes[base + 3]);
            }
            let mut pairs = [zero(); 8];
            for j in 0..4 {
                pairs[2 * j] = _mm_unpacklo_epi32(words[j], words[4 + j]);
                pairs[2 * j + 1] = _mm_unpackhi_epi32(words[j], words[4 + j]);
            }
            pairs
        };
        for (k, pair) in rows_pairs.into_iter().enumerate() {
            // SAFETY: the caller provides 8 writable bytes at each row.
            unsafe { Self(pair).store_halves(rows[2 * k], rows[2 * k + 1]) };
        }
    }
    #[inline(always)]
    unsafe fn load_rows_4(rows: [*const u8; 16]) -> [Self; 4] {
        let mut quads = [zero(); 4];
        for (k, quad) in quads.iter_mut().enumerate() {
            // SAFETY: the caller provides 4 readable bytes at each row.
            *quad = unsafe {
                let mut v = _mm_cvtsi32_si128(rows[4 * k].cast::<i32>().read_unaligned());
                v = _mm_insert_epi32::<1>(v, rows[4 * k + 1].cast::<i32>().read_unaligned());
                v = _mm_insert_epi32::<2>(v, rows[4 * k + 2].cast::<i32>().read_unaligned());
                _mm_insert_epi32::<3>(v, rows[4 * k + 3].cast::<i32>().read_unaligned())
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
        let quads = [
            transpose_4x4_bytes(dwords[0]),
            transpose_4x4_bytes(dwords[1]),
            transpose_4x4_bytes(dwords[2]),
            transpose_4x4_bytes(dwords[3]),
        ];
        for (k, quad) in quads.into_iter().enumerate() {
            // SAFETY: the caller provides 4 writable bytes at each row.
            unsafe {
                rows[4 * k]
                    .cast::<i32>()
                    .write_unaligned(_mm_cvtsi128_si32(quad));
                rows[4 * k + 1]
                    .cast::<i32>()
                    .write_unaligned(_mm_extract_epi32::<1>(quad));
                rows[4 * k + 2]
                    .cast::<i32>()
                    .write_unaligned(_mm_extract_epi32::<2>(quad));
                rows[4 * k + 3]
                    .cast::<i32>()
                    .write_unaligned(_mm_extract_epi32::<3>(quad));
            }
        }
    }
    #[inline(always)]
    fn abs_diff(self, b: Self) -> Self {
        // SAFETY: register-only.
        unsafe {
            Self(_mm_or_si128(
                _mm_subs_epu8(self.0, b.0),
                _mm_subs_epu8(b.0, self.0),
            ))
        }
    }
    #[inline(always)]
    fn less(self, b: Self) -> Self {
        // SAFETY: register-only. `a < b` exactly when `max(a, b) != a`.
        unsafe {
            let not_less = _mm_cmpeq_epi8(_mm_max_epu8(self.0, b.0), self.0);
            Self(_mm_xor_si128(not_less, _mm_set1_epi8(-1)))
        }
    }
    #[inline(always)]
    fn and(self, b: Self) -> Self {
        // SAFETY: register-only.
        unsafe { Self(_mm_and_si128(self.0, b.0)) }
    }
    #[inline(always)]
    fn and_not(self, b: Self) -> Self {
        // SAFETY: register-only.
        unsafe { Self(_mm_andnot_si128(b.0, self.0)) }
    }
    #[inline(always)]
    fn select(self, a: Self, b: Self) -> Self {
        // SAFETY: register-only.
        unsafe { Self(_mm_blendv_epi8(b.0, a.0, self.0)) }
    }
    #[inline(always)]
    fn sub(self, b: Self) -> Self {
        // SAFETY: register-only.
        unsafe { Self(_mm_sub_epi8(self.0, b.0)) }
    }
    #[inline(always)]
    fn add_sat(self, b: Self) -> Self {
        // SAFETY: register-only.
        unsafe { Self(_mm_adds_epu8(self.0, b.0)) }
    }
    #[inline(always)]
    fn sub_sat(self, b: Self) -> Self {
        // SAFETY: register-only.
        unsafe { Self(_mm_subs_epu8(self.0, b.0)) }
    }
    #[inline(always)]
    fn min(self, b: Self) -> Self {
        // SAFETY: register-only.
        unsafe { Self(_mm_min_epu8(self.0, b.0)) }
    }
    #[inline(always)]
    fn max(self, b: Self) -> Self {
        // SAFETY: register-only.
        unsafe { Self(_mm_max_epu8(self.0, b.0)) }
    }
    #[inline(always)]
    fn avg_round(self, b: Self) -> Self {
        // SAFETY: register-only.
        unsafe { Self(_mm_avg_epu8(self.0, b.0)) }
    }
    #[inline(always)]
    fn avg_floor(self, b: Self) -> Self {
        // SAFETY: register-only. The rounding average exceeds the floor by
        // one exactly where the sum is odd.
        unsafe {
            let odd = _mm_and_si128(_mm_xor_si128(self.0, b.0), _mm_set1_epi8(1));
            Self(_mm_sub_epi8(_mm_avg_epu8(self.0, b.0), odd))
        }
    }
    #[inline(always)]
    fn any(self) -> bool {
        // SAFETY: register-only.
        unsafe { _mm_movemask_epi8(self.0) != 0 }
    }
    #[inline(always)]
    fn normal_delta(p1: Self, p0: Self, q0: Self, q1: Self) -> Self {
        // SAFETY: register-only; `packs` saturates to i8.
        unsafe {
            Self(_mm_packs_epi16(
                delta_half::<true>(p1.0, p0.0, q0.0, q1.0),
                delta_half::<false>(p1.0, p0.0, q0.0, q1.0),
            ))
        }
    }
    #[inline(always)]
    fn clip_signed(self, t: Self) -> Self {
        // SAFETY: register-only.
        unsafe {
            let negative = _mm_sub_epi8(_mm_setzero_si128(), t.0);
            Self(_mm_min_epi8(_mm_max_epi8(self.0, negative), t.0))
        }
    }
    #[inline(always)]
    fn add_signed(self, delta: Self) -> Self {
        // SAFETY: register-only.
        unsafe {
            let zero = _mm_setzero_si128();
            let up = _mm_max_epi8(delta.0, zero);
            let down = _mm_max_epi8(_mm_sub_epi8(zero, delta.0), zero);
            Self(_mm_subs_epu8(_mm_adds_epu8(self.0, up), down))
        }
    }
    #[inline(always)]
    fn sub_signed(self, delta: Self) -> Self {
        // SAFETY: register-only; `delta > −128`, so its negation is exact.
        let negated = unsafe { Self(_mm_sub_epi8(_mm_setzero_si128(), delta.0)) };
        self.add_signed(negated)
    }
    #[inline(always)]
    fn strong_side(x3: Self, x2: Self, x1: Self, x0: Self, y0: Self, y1: Self) -> [Self; 3] {
        let x = [x3.0, x2.0, x1.0, x0.0, y0.0, y1.0];
        let (low, high) = (strong_half::<true>(x), strong_half::<false>(x));
        // SAFETY: register-only; the sums are at most 2040, so `packus`
        // never saturates.
        unsafe {
            [
                Self(_mm_packus_epi16(low[0], high[0])),
                Self(_mm_packus_epi16(low[1], high[1])),
                Self(_mm_packus_epi16(low[2], high[2])),
            ]
        }
    }
}

/// An all-zero register.
#[inline(always)]
fn zero() -> __m128i {
    // SAFETY: register-only.
    unsafe { _mm_setzero_si128() }
}

/// Bytes `0..8` (`LOW`) or `8..16` of `v` zero-extended to words.
#[inline(always)]
fn widen<const LOW: bool>(v: __m128i) -> __m128i {
    // SAFETY: register-only.
    unsafe {
        if LOW {
            _mm_unpacklo_epi8(v, _mm_setzero_si128())
        } else {
            _mm_unpackhi_epi8(v, _mm_setzero_si128())
        }
    }
}

/// `((q0 − p0)·4 + (p1 − q1) + 4) >> 3` of eight lines as words (exact:
/// |x| ≤ 255 for each difference, so the sum fits i16).
#[inline(always)]
fn delta_half<const LOW: bool>(p1: __m128i, p0: __m128i, q0: __m128i, q1: __m128i) -> __m128i {
    // SAFETY: register-only.
    unsafe {
        let d = _mm_slli_epi16::<2>(_mm_sub_epi16(widen::<LOW>(q0), widen::<LOW>(p0)));
        let e = _mm_sub_epi16(widen::<LOW>(p1), widen::<LOW>(q1));
        _mm_srai_epi16::<3>(_mm_add_epi16(_mm_add_epi16(d, e), _mm_set1_epi16(4)))
    }
}

/// The strong outputs `[x2', x1', x0']` of eight lines as words, from
/// `[x3, x2, x1, x0, y0, y1]` (sums at most 8 · 255).
#[inline(always)]
fn strong_half<const LOW: bool>(v: [__m128i; 6]) -> [__m128i; 3] {
    let [x3, x2, x1, x0, y0, y1] = [
        widen::<LOW>(v[0]),
        widen::<LOW>(v[1]),
        widen::<LOW>(v[2]),
        widen::<LOW>(v[3]),
        widen::<LOW>(v[4]),
        widen::<LOW>(v[5]),
    ];
    // SAFETY: register-only.
    unsafe {
        let s = _mm_add_epi16(_mm_add_epi16(x1, x0), y0);
        let s2 = _mm_add_epi16(s, x2);
        let (two, four) = (_mm_set1_epi16(2), _mm_set1_epi16(4));
        [
            // (2·x3 + 3·x2 + x1 + x0 + y0 + 4) >> 3
            _mm_srli_epi16::<3>(_mm_add_epi16(
                _mm_add_epi16(_mm_slli_epi16::<1>(_mm_add_epi16(x3, x2)), s2),
                four,
            )),
            // (x2 + x1 + x0 + y0 + 2) >> 2
            _mm_srli_epi16::<2>(_mm_add_epi16(s2, two)),
            // (x2 + 2·x1 + 2·x0 + 2·y0 + y1 + 4) >> 3
            _mm_srli_epi16::<3>(_mm_add_epi16(_mm_add_epi16(_mm_add_epi16(s, s2), y1), four)),
        ]
    }
}

/// Transpose of each 4x4 byte matrix (four rows of four bytes) in place.
#[inline(always)]
fn transpose_4x4_bytes(v: __m128i) -> __m128i {
    // SAFETY: register-only (SSSE3).
    unsafe {
        _mm_shuffle_epi8(
            v,
            _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15),
        )
    }
}

/// Transpose of the 4x4 matrix of 32-bit elements (vector `i`, element `j`).
#[inline(always)]
fn transpose_4x4_dwords(v: [__m128i; 4]) -> [__m128i; 4] {
    // SAFETY: register-only.
    unsafe {
        let (lo01, hi01) = (
            _mm_unpacklo_epi32(v[0], v[1]),
            _mm_unpackhi_epi32(v[0], v[1]),
        );
        let (lo23, hi23) = (
            _mm_unpacklo_epi32(v[2], v[3]),
            _mm_unpackhi_epi32(v[2], v[3]),
        );
        [
            _mm_unpacklo_epi64(lo01, lo23),
            _mm_unpackhi_epi64(lo01, lo23),
            _mm_unpacklo_epi64(hi01, hi23),
            _mm_unpackhi_epi64(hi01, hi23),
        ]
    }
}
