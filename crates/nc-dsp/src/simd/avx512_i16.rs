//! Thirty-two word lanes for the shared prediction graph. Accumulators use
//! the AVX2 lane implementation's paired-tap layout, widened to four 128-bit
//! groups: no filter equations or reference-window rules are duplicated.
//! See `simd`'s private-lane contract: methods inline only into wrappers with
//! AVX-512F/BW/VL enabled. No DQ or VNNI instructions are required.
//! HEVC horizontal sums remain below 112 * 255 = 28,560; vertical
//! accumulation stays in i32, with rounding before signed-word narrowing.

#![allow(clippy::inline_always)]

use super::avx512::I32x16;
use super::{HIGH_LOOKUP_TABLE, I16FilterLanes, I16Lanes, I16Pairs, packed_pair};
use core::arch::x86_64::{
    __m512i, _mm_cvtsi32_si128, _mm_loadl_epi64, _mm_loadu_si128, _mm_setr_epi8, _mm_storel_epi64,
    _mm_storeu_si128, _mm_unpackhi_epi64, _mm_unpacklo_epi64, _mm256_castsi256_si128,
    _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_set_m128i, _mm256_setr_epi32,
    _mm256_storeu_si256, _mm512_abs_epi16, _mm512_add_epi16, _mm512_add_epi32, _mm512_adds_epi16,
    _mm512_alignr_epi64, _mm512_and_si512, _mm512_broadcast_i32x4, _mm512_castsi256_si512,
    _mm512_castsi512_si128, _mm512_castsi512_si256, _mm512_cmpeq_epi16_mask,
    _mm512_cmpgt_epi16_mask, _mm512_cmplt_epi16_mask, _mm512_cvtepi32_epi16, _mm512_cvtepu8_epi16,
    _mm512_cvtepu16_epi32, _mm512_cvtepu32_epi64, _mm512_cvtusepi16_epi8,
    _mm512_extracti32x4_epi32, _mm512_extracti64x4_epi64, _mm512_i32gather_epi32,
    _mm512_inserti64x4, _mm512_loadu_si512, _mm512_madd_epi16, _mm512_mask_blend_epi16,
    _mm512_mask_sub_epi16, _mm512_maskz_set1_epi16, _mm512_max_epi16, _mm512_min_epi16,
    _mm512_min_epu16, _mm512_mulhrs_epi16, _mm512_mullo_epi16, _mm512_packs_epi32,
    _mm512_permutex2var_epi16, _mm512_permutexvar_epi64, _mm512_set_epi16, _mm512_set_epi64,
    _mm512_set1_epi16, _mm512_set1_epi32, _mm512_setzero_si512, _mm512_shuffle_epi8,
    _mm512_shuffle_epi32, _mm512_shuffle_i64x2, _mm512_slli_epi32, _mm512_sra_epi16,
    _mm512_sra_epi32, _mm512_srai_epi32, _mm512_srl_epi16, _mm512_srli_epi32, _mm512_storeu_si512,
    _mm512_sub_epi16, _mm512_subs_epu16, _mm512_test_epi16_mask, _mm512_unpackhi_epi16,
    _mm512_unpackhi_epi32, _mm512_unpackhi_epi64, _mm512_unpacklo_epi16, _mm512_unpacklo_epi32,
    _mm512_unpacklo_epi64,
};

/// Thirty-two signed words, private to AVX-512 kernels.
#[derive(Clone, Copy)]
pub(crate) struct I16x32(__m512i);

impl I16x32 {
    /// Load two 4x4 word blocks (`u` in lanes `0..16`, `v` in lanes `16..32`).
    ///
    /// # Safety
    /// Caller must run inside an AVX-512F/BW/VL function.
    #[inline(always)]
    pub(crate) unsafe fn load_pair4(u: [&[i16]; 4], v: [&[i16]; 4]) -> Self {
        let pack4 = |r: [&[i16]; 4]| {
            let (r0, r1, r2, r3) = (&r[0][..4], &r[1][..4], &r[2][..4], &r[3][..4]);
            // SAFETY: caller runs inside an AVX-512F/BW/VL kernel; each row slice holds 4 words.
            unsafe {
                let lo = _mm_unpacklo_epi64(
                    _mm_loadl_epi64(r0.as_ptr().cast()),
                    _mm_loadl_epi64(r1.as_ptr().cast()),
                );
                let hi = _mm_unpacklo_epi64(
                    _mm_loadl_epi64(r2.as_ptr().cast()),
                    _mm_loadl_epi64(r3.as_ptr().cast()),
                );
                _mm256_set_m128i(hi, lo)
            }
        };
        // SAFETY: caller runs inside an AVX-512F/BW/VL kernel.
        unsafe {
            Self(_mm512_inserti64x4::<1>(
                _mm512_castsi256_si512(pack4(u)),
                pack4(v),
            ))
        }
    }

    /// Store two 4x4 word blocks (`u` from lanes `0..16`, `v` from lanes `16..32`).
    ///
    /// # Safety
    /// Caller must run inside an AVX-512F/BW/VL function.
    #[inline(always)]
    pub(crate) unsafe fn store_pair4(self, u: [&mut [i16]; 4], v: [&mut [i16]; 4]) {
        let [u0, u1, u2, u3] = u;
        let [v0, v1, v2, v3] = v;
        let (u0, u1, u2, u3) = (&mut u0[..4], &mut u1[..4], &mut u2[..4], &mut u3[..4]);
        let (v0, v1, v2, v3) = (&mut v0[..4], &mut v1[..4], &mut v2[..4], &mut v3[..4]);
        // SAFETY: caller runs inside an AVX-512F/BW/VL kernel; each row slice holds 4 words.
        unsafe {
            let q0 = _mm512_castsi512_si128(self.0);
            let q1 = _mm512_extracti32x4_epi32::<1>(self.0);
            let q2 = _mm512_extracti32x4_epi32::<2>(self.0);
            let q3 = _mm512_extracti32x4_epi32::<3>(self.0);
            _mm_storel_epi64(u0.as_mut_ptr().cast(), q0);
            _mm_storel_epi64(u1.as_mut_ptr().cast(), _mm_unpackhi_epi64(q0, q0));
            _mm_storel_epi64(u2.as_mut_ptr().cast(), q1);
            _mm_storel_epi64(u3.as_mut_ptr().cast(), _mm_unpackhi_epi64(q1, q1));
            _mm_storel_epi64(v0.as_mut_ptr().cast(), q2);
            _mm_storel_epi64(v1.as_mut_ptr().cast(), _mm_unpackhi_epi64(q2, q2));
            _mm_storel_epi64(v2.as_mut_ptr().cast(), q3);
            _mm_storel_epi64(v3.as_mut_ptr().cast(), _mm_unpackhi_epi64(q3, q3));
        }
    }
}

impl I16Lanes for I16x32 {
    const LANES: usize = 32;
    type Acc = [__m512i; 2];

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs).
        unsafe { Self(_mm512_min_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs).
        unsafe { Self(_mm512_max_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs).
        unsafe { Self(_mm512_abs_epi16(self.0)) }
    }

    #[inline(always)]
    fn with_sign_of(self, like: Self) -> Self {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs).
        unsafe {
            Self(_mm512_mask_sub_epi16(
                self.0,
                _mm512_cmplt_epi16_mask(like.0, _mm512_setzero_si512()),
                _mm512_setzero_si512(),
                self.0,
            ))
        }
    }

    #[inline(always)]
    fn splat(value: i16) -> Self {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { Self(_mm512_set1_epi16(value)) }
    }

    #[inline(always)]
    fn load_u8(src: &[u8]) -> Self {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let src = &src[..32];
            Self(_mm512_cvtepu8_epi16(_mm256_loadu_si256(
                src.as_ptr().cast(),
            )))
        }
    }

    #[inline(always)]
    fn pair_sum_u8(src: &[u8]) -> Self {
        use core::arch::x86_64::{_mm512_loadu_si512, _mm512_maddubs_epi16, _mm512_set1_epi8};
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs);
        // slices are indexed before every load.
        unsafe {
            let src = &src[..64];
            Self(_mm512_maddubs_epi16(
                _mm512_loadu_si512(src.as_ptr().cast()),
                _mm512_set1_epi8(1),
            ))
        }
    }

    #[inline(always)]
    fn load(src: &[i16]) -> Self {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let src = &src[..32];
            Self(_mm512_loadu_si512(src.as_ptr().cast()))
        }
    }

    #[inline(always)]
    fn store(self, dst: &mut [i16]) {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let dst = &mut dst[..32];
            _mm512_storeu_si512(dst.as_mut_ptr().cast(), self.0);
        }
    }

    #[inline(always)]
    fn lookup_u12(self, table: &[i16; HIGH_LOOKUP_TABLE]) -> Self {
        // SAFETY: AVX-512F/BW/VL enabled by caller; `clamped` keeps indices in 0..=4095.
        unsafe {
            let clamped = _mm512_min_epi16(
                _mm512_max_epi16(self.0, _mm512_setzero_si512()),
                _mm512_set1_epi16(4095),
            );
            #[allow(clippy::cast_ptr_alignment, reason = "vpgatherdd is unaligned")]
            let base = table.as_ptr().cast();
            let lo_idx = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(clamped));
            let hi_idx = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64::<1>(clamped));
            let low = _mm512_i32gather_epi32::<2>(lo_idx, base);
            let high = _mm512_i32gather_epi32::<2>(hi_idx, base);
            let low = _mm512_srai_epi32::<16>(_mm512_slli_epi32::<16>(low));
            let high = _mm512_srai_epi32::<16>(_mm512_slli_epi32::<16>(high));
            Self(_mm512_inserti64x4::<1>(
                _mm512_castsi256_si512(_mm512_cvtepi32_epi16(low)),
                _mm512_cvtepi32_epi16(high),
            ))
        }
    }

    #[inline(always)]
    fn pair_avg_u16(pairs: &[u16]) -> Self {
        let pairs = &pairs[..64];
        // SAFETY: AVX-512F/BW/VL enabled by caller; `pairs` holds 64 `u16` elements.
        unsafe {
            let p0 = _mm512_loadu_si512(pairs.as_ptr().cast());
            let p1 = _mm512_loadu_si512(pairs.as_ptr().add(32).cast());
            let mask = _mm512_set1_epi32(0xFFFF);
            let one = _mm512_set1_epi32(1);
            let avg16 = |p: __m512i| {
                let sum = _mm512_add_epi32(_mm512_and_si512(p, mask), _mm512_srli_epi32::<16>(p));
                _mm512_srli_epi32::<1>(_mm512_add_epi32(sum, one))
            };
            Self(_mm512_inserti64x4::<1>(
                _mm512_castsi256_si512(_mm512_cvtepi32_epi16(avg16(p0))),
                _mm512_cvtepi32_epi16(avg16(p1)),
            ))
        }
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { Self(_mm512_add_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn add_sat(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module
        // docs); register operations only.
        unsafe { Self(_mm512_adds_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { Self(_mm512_sub_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { Self(_mm512_mullo_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn mul_q15_round(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs).
        unsafe { Self(_mm512_mulhrs_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn sra(self, bits: u32) -> Self {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { Self(_mm512_sra_epi16(self.0, _mm_cvtsi32_si128(bits as i32))) }
    }

    #[inline(always)]
    fn srl(self, bits: u32) -> Self {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs).
        unsafe { Self(_mm512_srl_epi16(self.0, _mm_cvtsi32_si128(bits as i32))) }
    }

    #[inline(always)]
    fn clamp_u8(self) -> Self {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs).
        unsafe {
            Self(_mm512_max_epi16(
                _mm512_min_epi16(self.0, _mm512_set1_epi16(255)),
                _mm512_setzero_si512(),
            ))
        }
    }

    #[inline(always)]
    fn store_u8_sat(self, dst: &mut [u8]) {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let dst = &mut dst[..32];
            // vpmovuswb interprets its input as unsigned: clamp negatives
            // first, or negative filter lobes would become white pixels.
            let nonnegative = _mm512_max_epi16(self.0, _mm512_setzero_si512());
            _mm256_storeu_si256(dst.as_mut_ptr().cast(), _mm512_cvtusepi16_epi8(nonnegative));
        }
    }

    #[inline(always)]
    fn acc_zero() -> Self::Acc {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { [_mm512_setzero_si512(); 2] }
    }

    #[inline(always)]
    fn acc_mul_add(acc: Self::Acc, a: Self, coeff: i16) -> Self::Acc {
        Self::acc_mul_add2(acc, a, coeff, Self::splat(0), 0)
    }

    /// `pmaddwd` on `a` and `b` interleaved. The accumulators hold lanes in
    /// unpack order: `acc[0]` contains 0-3, 8-11, 16-19, 24-27 and
    /// `acc[1]` the other four-lane groups. Lane-local `packssdw` restores
    /// the original contiguous word order.
    #[inline(always)]
    fn acc_mul_add2(acc: Self::Acc, a: Self, ca: i16, b: Self, cb: i16) -> Self::Acc {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs).
        unsafe {
            let taps = _mm512_set1_epi32(i32::from(ca as u16) | (i32::from(cb) << 16));
            [
                _mm512_add_epi32(
                    acc[0],
                    _mm512_madd_epi16(_mm512_unpacklo_epi16(a.0, b.0), taps),
                ),
                _mm512_add_epi32(
                    acc[1],
                    _mm512_madd_epi16(_mm512_unpackhi_epi16(a.0, b.0), taps),
                ),
            ]
        }
    }

    #[inline(always)]
    fn acc_vec_mul_add2(a: Self, wa: Self, b: Self, wb: Self) -> Self::Acc {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs).
        unsafe {
            [
                _mm512_madd_epi16(
                    _mm512_unpacklo_epi16(a.0, b.0),
                    _mm512_unpacklo_epi16(wa.0, wb.0),
                ),
                _mm512_madd_epi16(
                    _mm512_unpackhi_epi16(a.0, b.0),
                    _mm512_unpackhi_epi16(wa.0, wb.0),
                ),
            ]
        }
    }

    #[inline(always)]
    fn acc_vec_fma2(acc: Self::Acc, a: Self, wa: Self, b: Self, wb: Self) -> Self::Acc {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs).
        unsafe {
            let prod = Self::acc_vec_mul_add2(a, wa, b, wb);
            [
                _mm512_add_epi32(acc[0], prod[0]),
                _mm512_add_epi32(acc[1], prod[1]),
            ]
        }
    }

    #[inline(always)]
    fn acc_add(acc: Self::Acc, a: Self) -> Self::Acc {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            [
                _mm512_add_epi32(acc[0], _mm512_madd_epi16(a.0, _mm512_set1_epi16(1))),
                acc[1],
            ]
        }
    }

    #[inline(always)]
    fn acc_add_sq(acc: Self::Acc, a: Self) -> Self::Acc {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            [
                _mm512_add_epi32(acc[0], _mm512_madd_epi16(a.0, a.0)),
                acc[1],
            ]
        }
    }

    #[inline(always)]
    fn acc_round_shift_store_u8(acc: Self::Acc, round: i32, shift: u32, dst: &mut [u8]) {
        let words = Self::acc_round_shift_narrow(acc, round, shift);
        words.store_u8_sat(dst);
    }

    #[inline(always)]
    fn acc_round_shift_narrow(acc: Self::Acc, round: i32, shift: u32) -> Self {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs).
        unsafe {
            let (r, n) = (_mm512_set1_epi32(round), _mm_cvtsi32_si128(shift as i32));
            let lo = _mm512_sra_epi32(_mm512_add_epi32(acc[0], r), n);
            let hi = _mm512_sra_epi32(_mm512_add_epi32(acc[1], r), n);
            // Unpack order in, lane order out (see `acc_mul_add2`).
            Self(_mm512_packs_epi32(lo, hi))
        }
    }

    #[inline(always)]
    fn acc_sum(acc: Self::Acc) -> i64 {
        let mut lanes = [0_i32; 32];
        // SAFETY: each half of `lanes` holds sixteen i32 slots (64 bytes).
        unsafe {
            _mm512_storeu_si512(lanes.as_mut_ptr().cast(), acc[0]);
            _mm512_storeu_si512(lanes.as_mut_ptr().add(16).cast(), acc[1]);
        }
        lanes.into_iter().map(i64::from).sum()
    }
}

// The SAO statistics graph uses eq_mask and and. The remaining operations
// complete the shared filter-lane contract without adding a separate graph.
impl I16FilterLanes for I16x32 {
    #[inline(always)]
    fn load_halves(lo: &[i16], hi: &[i16]) -> Self {
        let (lo, hi) = (&lo[..16], &hi[..16]);
        // SAFETY: each slice contains 32 bytes; the wrapper enables F/BW/VL.
        unsafe {
            Self(_mm512_inserti64x4::<1>(
                _mm512_castsi256_si512(_mm256_loadu_si256(lo.as_ptr().cast())),
                _mm256_loadu_si256(hi.as_ptr().cast()),
            ))
        }
    }

    #[inline(always)]
    fn load_u8_halves(lo: &[u8], hi: &[u8]) -> Self {
        let (lo, hi) = (&lo[..16], &hi[..16]);
        // SAFETY: each slice contains 16 bytes; the wrapper enables F/BW/VL.
        unsafe {
            Self(_mm512_cvtepu8_epi16(_mm256_set_m128i(
                _mm_loadu_si128(hi.as_ptr().cast()),
                _mm_loadu_si128(lo.as_ptr().cast()),
            )))
        }
    }

    #[inline(always)]
    fn store_u8_sat_halves(self, lo: &mut [u8], hi: &mut [u8]) {
        let (lo, hi) = (&mut lo[..16], &mut hi[..16]);
        // SAFETY: each output has exactly 16 writable bytes; F/BW/VL and
        // inherited AVX2 are enabled. Clamp signed negatives before vpmovuswb.
        unsafe {
            let bytes = _mm512_cvtusepi16_epi8(_mm512_max_epi16(self.0, _mm512_setzero_si512()));
            _mm_storeu_si128(lo.as_mut_ptr().cast(), _mm256_castsi256_si128(bytes));
            _mm_storeu_si128(hi.as_mut_ptr().cast(), _mm256_extracti128_si256::<1>(bytes));
        }
    }

    #[inline(always)]
    fn store_halves(self, lo: &mut [i16], hi: &mut [i16]) {
        let (lo, hi) = (&mut lo[..16], &mut hi[..16]);
        // SAFETY: each output has 16 writable words; AVX-512F/BW/VL enabled.
        unsafe {
            _mm256_storeu_si256(lo.as_mut_ptr().cast(), _mm512_castsi512_si256(self.0));
            _mm256_storeu_si256(
                hi.as_mut_ptr().cast(),
                _mm512_extracti64x4_epi64::<1>(self.0),
            );
        }
    }

    #[inline(always)]
    fn combine_hi_lo(self, next: Self) -> Self {
        // SAFETY: wrapper enables AVX-512F/BW/VL.
        unsafe { Self(_mm512_shuffle_i64x2::<0b01_00_11_10>(self.0, next.0)) }
    }

    #[inline(always)]
    fn transpose_groups8(rows: [Self; 8]) -> [Self; 8] {
        // SAFETY: wrapper enables AVX-512F/BW/VL; in-lane unpacks transpose all four 8x8 tiles.
        unsafe {
            let [r0, r1, r2, r3, r4, r5, r6, r7] = rows;
            let (p0, p1) = (
                _mm512_unpacklo_epi16(r0.0, r1.0),
                _mm512_unpackhi_epi16(r0.0, r1.0),
            );
            let (p2, p3) = (
                _mm512_unpacklo_epi16(r2.0, r3.0),
                _mm512_unpackhi_epi16(r2.0, r3.0),
            );
            let (p4, p5) = (
                _mm512_unpacklo_epi16(r4.0, r5.0),
                _mm512_unpackhi_epi16(r4.0, r5.0),
            );
            let (p6, p7) = (
                _mm512_unpacklo_epi16(r6.0, r7.0),
                _mm512_unpackhi_epi16(r6.0, r7.0),
            );
            let (q0, q1) = (_mm512_unpacklo_epi32(p0, p2), _mm512_unpackhi_epi32(p0, p2));
            let (q2, q3) = (_mm512_unpacklo_epi32(p1, p3), _mm512_unpackhi_epi32(p1, p3));
            let (q4, q5) = (_mm512_unpacklo_epi32(p4, p6), _mm512_unpackhi_epi32(p4, p6));
            let (q6, q7) = (_mm512_unpacklo_epi32(p5, p7), _mm512_unpackhi_epi32(p5, p7));
            [
                Self(_mm512_unpacklo_epi64(q0, q4)),
                Self(_mm512_unpackhi_epi64(q0, q4)),
                Self(_mm512_unpacklo_epi64(q1, q5)),
                Self(_mm512_unpackhi_epi64(q1, q5)),
                Self(_mm512_unpacklo_epi64(q2, q6)),
                Self(_mm512_unpackhi_epi64(q2, q6)),
                Self(_mm512_unpacklo_epi64(q3, q7)),
                Self(_mm512_unpackhi_epi64(q3, q7)),
            ]
        }
    }

    #[inline(always)]
    fn load_quarters(rows: [&[i16]; 4]) -> Self {
        let (r0, r1, r2, r3) = (&rows[0][..8], &rows[1][..8], &rows[2][..8], &rows[3][..8]);
        // SAFETY: each slice holds 8 words (16 bytes); AVX-512F/BW/VL enabled.
        unsafe {
            let lo = _mm256_set_m128i(
                _mm_loadu_si128(r1.as_ptr().cast()),
                _mm_loadu_si128(r0.as_ptr().cast()),
            );
            let hi = _mm256_set_m128i(
                _mm_loadu_si128(r3.as_ptr().cast()),
                _mm_loadu_si128(r2.as_ptr().cast()),
            );
            Self(_mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi))
        }
    }

    #[inline(always)]
    fn load_u8_quarters(rows: [&[u8]; 4]) -> Self {
        let (r0, r1, r2, r3) = (&rows[0][..8], &rows[1][..8], &rows[2][..8], &rows[3][..8]);
        // SAFETY: each slice holds 8 bytes; AVX-512F/BW/VL enabled.
        unsafe {
            let lo = _mm_unpacklo_epi64(
                _mm_loadl_epi64(r0.as_ptr().cast()),
                _mm_loadl_epi64(r1.as_ptr().cast()),
            );
            let hi = _mm_unpacklo_epi64(
                _mm_loadl_epi64(r2.as_ptr().cast()),
                _mm_loadl_epi64(r3.as_ptr().cast()),
            );
            Self(_mm512_cvtepu8_epi16(_mm256_set_m128i(hi, lo)))
        }
    }

    #[inline(always)]
    fn store_quarters(self, rows: [&mut [i16]; 4]) {
        let [r0, r1, r2, r3] = rows;
        let (r0, r1) = (&mut r0[..8], &mut r1[..8]);
        let (r2, r3) = (&mut r2[..8], &mut r3[..8]);
        // SAFETY: each slice holds 8 writable words; AVX-512F/BW/VL enabled.
        unsafe {
            _mm_storeu_si128(r0.as_mut_ptr().cast(), _mm512_castsi512_si128(self.0));
            _mm_storeu_si128(
                r1.as_mut_ptr().cast(),
                _mm512_extracti32x4_epi32::<1>(self.0),
            );
            _mm_storeu_si128(
                r2.as_mut_ptr().cast(),
                _mm512_extracti32x4_epi32::<2>(self.0),
            );
            _mm_storeu_si128(
                r3.as_mut_ptr().cast(),
                _mm512_extracti32x4_epi32::<3>(self.0),
            );
        }
    }

    #[inline(always)]
    fn min_u16(self, other: Self) -> Self {
        // SAFETY: the wrapper enables AVX-512F/BW/VL.
        unsafe { Self(_mm512_min_epu16(self.0, other.0)) }
    }

    #[inline(always)]
    fn sub_sat_u16(self, other: Self) -> Self {
        // SAFETY: the wrapper enables AVX-512F/BW/VL.
        unsafe { Self(_mm512_subs_epu16(self.0, other.0)) }
    }

    #[inline(always)]
    fn eq_mask(self, other: Self) -> Self {
        // SAFETY: the wrapper enables AVX-512F/BW/VL.
        unsafe {
            Self(_mm512_maskz_set1_epi16(
                _mm512_cmpeq_epi16_mask(self.0, other.0),
                -1,
            ))
        }
    }

    #[inline(always)]
    fn and(self, other: Self) -> Self {
        // SAFETY: the wrapper enables AVX-512F/BW/VL.
        unsafe { Self(_mm512_and_si512(self.0, other.0)) }
    }

    #[inline(always)]
    fn gt_mask(self, other: Self) -> Self {
        // SAFETY: the wrapper enables AVX-512F/BW/VL.
        unsafe {
            Self(_mm512_maskz_set1_epi16(
                _mm512_cmpgt_epi16_mask(self.0, other.0),
                -1,
            ))
        }
    }

    #[inline(always)]
    fn blend(self, if_set: Self, if_clear: Self) -> Self {
        // SAFETY: masks have only zero/-1 lanes under the trait contract;
        // the wrapper enables AVX-512F/BW/VL.
        unsafe {
            Self(_mm512_mask_blend_epi16(
                _mm512_cmplt_epi16_mask(self.0, _mm512_setzero_si512()),
                if_clear.0,
                if_set.0,
            ))
        }
    }

    #[inline(always)]
    fn any_nonzero(self) -> bool {
        // SAFETY: wrapper enables AVX-512F/BW/VL.
        unsafe { _mm512_test_epi16_mask(self.0, self.0) != 0 }
    }

    #[inline(always)]
    fn load_u8_deinterleaved(src: &[u8]) -> (Self, Self) {
        let src = &src[..64];
        // SAFETY: wrapper enables AVX-512F/BW/VL; `src` holds 64 bytes and
        // `_mm512_loadu_si512` accepts 1-byte alignment.
        unsafe {
            let raw = Self(_mm512_loadu_si512(src.as_ptr().cast()));
            (raw.and(Self::splat(0x00ff)), raw.srl(8))
        }
    }

    #[inline(always)]
    fn load_u16_deinterleaved(src: &[u16]) -> (Self, Self) {
        let src = &src[..64];
        // SAFETY: wrapper enables AVX-512F/BW/VL; `src` holds 64 `u16` elements.
        unsafe {
            let v0 = _mm512_loadu_si512(src.as_ptr().cast());
            let v1 = _mm512_loadu_si512(src.as_ptr().add(32).cast());
            let even_idx = _mm512_set_epi16(
                62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22,
                20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0,
            );
            let odd_idx = _mm512_add_epi16(even_idx, _mm512_set1_epi16(1));
            (
                Self(_mm512_permutex2var_epi16(v0, even_idx, v1)),
                Self(_mm512_permutex2var_epi16(v0, odd_idx, v1)),
            )
        }
    }

    #[inline(always)]
    fn combine_mid_halves(self, next: Self) -> Self {
        // SAFETY: wrapper enables AVX-512F/BW/VL.
        unsafe { Self(_mm512_alignr_epi64::<4>(next.0, self.0)) }
    }

    #[inline(always)]
    fn shift_quarter(self, next: Self) -> Self {
        // SAFETY: wrapper enables AVX-512F/BW/VL.
        unsafe { Self(_mm512_alignr_epi64::<2>(next.0, self.0)) }
    }

    #[inline(always)]
    fn expand_alf_params(blocks: &[[i16; 12]]) -> [[Self; 2]; 6] {
        let b = &blocks[..8];
        // `_mm512_unpacklo_epi16` places blocks 0, 2, 4, 6 in the four 128-bit lanes;
        // `_mm512_unpackhi_epi16` places blocks 1, 3, 5, 7 in the four 128-bit lanes.
        // SAFETY: wrapper enables AVX-512F/BW/VL; each block has 12 `i16`s (24 bytes).
        unsafe {
            let lo0 = _mm512_inserti64x4::<1>(
                _mm512_castsi256_si512(_mm256_set_m128i(
                    _mm_loadu_si128(b[2].as_ptr().cast()),
                    _mm_loadu_si128(b[0].as_ptr().cast()),
                )),
                _mm256_set_m128i(
                    _mm_loadu_si128(b[6].as_ptr().cast()),
                    _mm_loadu_si128(b[4].as_ptr().cast()),
                ),
            );
            let hi0 = _mm512_inserti64x4::<1>(
                _mm512_castsi256_si512(_mm256_set_m128i(
                    _mm_loadl_epi64(b[2].as_ptr().add(8).cast()),
                    _mm_loadl_epi64(b[0].as_ptr().add(8).cast()),
                )),
                _mm256_set_m128i(
                    _mm_loadl_epi64(b[6].as_ptr().add(8).cast()),
                    _mm_loadl_epi64(b[4].as_ptr().add(8).cast()),
                ),
            );
            let lo1 = _mm512_inserti64x4::<1>(
                _mm512_castsi256_si512(_mm256_set_m128i(
                    _mm_loadu_si128(b[3].as_ptr().cast()),
                    _mm_loadu_si128(b[1].as_ptr().cast()),
                )),
                _mm256_set_m128i(
                    _mm_loadu_si128(b[7].as_ptr().cast()),
                    _mm_loadu_si128(b[5].as_ptr().cast()),
                ),
            );
            let hi1 = _mm512_inserti64x4::<1>(
                _mm512_castsi256_si512(_mm256_set_m128i(
                    _mm_loadl_epi64(b[3].as_ptr().add(8).cast()),
                    _mm_loadl_epi64(b[1].as_ptr().add(8).cast()),
                )),
                _mm256_set_m128i(
                    _mm_loadl_epi64(b[7].as_ptr().add(8).cast()),
                    _mm_loadl_epi64(b[5].as_ptr().add(8).cast()),
                ),
            );
            [
                [
                    Self(_mm512_shuffle_epi32::<0b00_00_00_00>(lo0)),
                    Self(_mm512_shuffle_epi32::<0b00_00_00_00>(lo1)),
                ],
                [
                    Self(_mm512_shuffle_epi32::<0b01_01_01_01>(lo0)),
                    Self(_mm512_shuffle_epi32::<0b01_01_01_01>(lo1)),
                ],
                [
                    Self(_mm512_shuffle_epi32::<0b10_10_10_10>(lo0)),
                    Self(_mm512_shuffle_epi32::<0b10_10_10_10>(lo1)),
                ],
                [
                    Self(_mm512_shuffle_epi32::<0b11_11_11_11>(lo0)),
                    Self(_mm512_shuffle_epi32::<0b11_11_11_11>(lo1)),
                ],
                [
                    Self(_mm512_shuffle_epi32::<0b00_00_00_00>(hi0)),
                    Self(_mm512_shuffle_epi32::<0b00_00_00_00>(hi1)),
                ],
                [
                    Self(_mm512_shuffle_epi32::<0b01_01_01_01>(hi0)),
                    Self(_mm512_shuffle_epi32::<0b01_01_01_01>(hi1)),
                ],
            ]
        }
    }

    #[inline(always)]
    fn splat_alf_pair(t0: i16, t1: i16) -> [Self; 2] {
        // SAFETY: wrapper enables AVX-512F/BW/VL.
        let v = unsafe { Self(_mm512_set1_epi32(packed_pair(t0, t1))) };
        [v, v]
    }

    #[inline(always)]
    fn acc_alf_clipped_pair(
        acc: Self::Acc,
        (a0, b0): (Self, Self),
        (a1, b1): (Self, Self),
        centre: Self,
        lim: [Self; 2],
        coeff: [Self; 2],
    ) -> Self::Acc {
        // SAFETY: wrapper enables AVX-512F/BW/VL.
        unsafe {
            let v00 = _mm512_sub_epi16(a0.0, centre.0);
            let v10 = _mm512_sub_epi16(a1.0, centre.0);
            let v01 = _mm512_sub_epi16(b0.0, centre.0);
            let v11 = _mm512_sub_epi16(b1.0, centre.0);
            let (mut u0, mut u1) = (
                _mm512_unpacklo_epi16(v00, v10),
                _mm512_unpackhi_epi16(v00, v10),
            );
            let (mut u2, mut u3) = (
                _mm512_unpacklo_epi16(v01, v11),
                _mm512_unpackhi_epi16(v01, v11),
            );
            let zero = _mm512_setzero_si512();
            let (neg0, neg1) = (
                _mm512_sub_epi16(zero, lim[0].0),
                _mm512_sub_epi16(zero, lim[1].0),
            );
            u0 = _mm512_max_epi16(_mm512_min_epi16(u0, lim[0].0), neg0);
            u1 = _mm512_max_epi16(_mm512_min_epi16(u1, lim[1].0), neg1);
            u2 = _mm512_max_epi16(_mm512_min_epi16(u2, lim[0].0), neg0);
            u3 = _mm512_max_epi16(_mm512_min_epi16(u3, lim[1].0), neg1);
            let sum_lo = _mm512_add_epi16(u0, u2);
            let sum_hi = _mm512_add_epi16(u1, u3);
            [
                _mm512_add_epi32(acc[0], _mm512_madd_epi16(sum_lo, coeff[0].0)),
                _mm512_add_epi32(acc[1], _mm512_madd_epi16(sum_hi, coeff[1].0)),
            ]
        }
    }

    #[inline(always)]
    fn acc_alf_linear_pair(acc: Self::Acc, d0: Self, d1: Self, coeff: [Self; 2]) -> Self::Acc {
        // SAFETY: wrapper enables AVX-512F/BW/VL.
        unsafe {
            let lo = _mm512_unpacklo_epi16(d0.0, d1.0);
            let hi = _mm512_unpackhi_epi16(d0.0, d1.0);
            [
                _mm512_add_epi32(acc[0], _mm512_madd_epi16(lo, coeff[0].0)),
                _mm512_add_epi32(acc[1], _mm512_madd_epi16(hi, coeff[1].0)),
            ]
        }
    }

    #[inline(always)]
    fn expand_alf_pair(blocks: &[[i16; 12]], p: usize) -> [Self; 2] {
        let b = &blocks[..8];
        let t = 2 * p;
        assert!(t + 1 < 12);
        // SAFETY: wrapper enables AVX-512F/BW/VL; `t + 1 < 12` checked above.
        unsafe {
            let pairs = _mm256_setr_epi32(
                b[0].as_ptr().add(t).cast::<i32>().read_unaligned(),
                b[1].as_ptr().add(t).cast::<i32>().read_unaligned(),
                b[2].as_ptr().add(t).cast::<i32>().read_unaligned(),
                b[3].as_ptr().add(t).cast::<i32>().read_unaligned(),
                b[4].as_ptr().add(t).cast::<i32>().read_unaligned(),
                b[5].as_ptr().add(t).cast::<i32>().read_unaligned(),
                b[6].as_ptr().add(t).cast::<i32>().read_unaligned(),
                b[7].as_ptr().add(t).cast::<i32>().read_unaligned(),
            );
            let wide = _mm512_cvtepu32_epi64(pairs);
            let lane0 = _mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 8, 9, 8, 9, 8, 9, 8, 9);
            let lane1 = _mm_setr_epi8(2, 3, 2, 3, 2, 3, 2, 3, 10, 11, 10, 11, 10, 11, 10, 11);
            let shuf0 = _mm512_broadcast_i32x4(lane0);
            let shuf1 = _mm512_broadcast_i32x4(lane1);
            [
                Self(_mm512_shuffle_epi8(wide, shuf0)),
                Self(_mm512_shuffle_epi8(wide, shuf1)),
            ]
        }
    }
}

impl I16Pairs for I16x32 {
    type Pairs = I32x16;

    #[inline(always)]
    fn splat_pair(first: i16, second: i16) -> Self {
        // SAFETY: wrapper enables AVX-512F/BW/VL.
        unsafe { Self(_mm512_set1_epi32(packed_pair(first, second))) }
    }

    #[inline(always)]
    fn pair_dot(self, other: Self) -> I32x16 {
        // SAFETY: wrapper enables AVX-512F/BW/VL.
        unsafe { I32x16(_mm512_madd_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn narrow_pairs(low: I32x16, high: I32x16) -> Self {
        // `_mm512_packs_epi32` packs 128-bit lanes `[L0, H0, L1, H1, L2, H2, L3, H3]`;
        // permuting qwords `[0, 2, 4, 6, 1, 3, 5, 7]` restores contiguous lane order.
        // SAFETY: wrapper enables AVX-512F/BW/VL.
        unsafe {
            let idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
            Self(_mm512_permutexvar_epi64(
                idx,
                _mm512_packs_epi32(low.0, high.0),
            ))
        }
    }

    #[inline(always)]
    fn interleave_pairs(even: I32x16, odd: I32x16) -> Self {
        // SAFETY: wrapper enables AVX-512F/BW/VL.
        unsafe {
            Self(_mm512_mask_blend_epi16(
                0xAAAA_AAAA,
                even.0,
                _mm512_slli_epi32::<16>(odd.0),
            ))
        }
    }
}

impl super::I16x8Tiles for I16x32 {
    #[inline(always)]
    fn splat_segments(values: &[i16; 8]) -> Self {
        use core::arch::x86_64::{_mm512_broadcast_i32x4, _mm512_shuffle_epi8};
        const SELECT: [u8; 64] = [
            0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6,
            7, 6, 7, 8, 9, 8, 9, 8, 9, 8, 9, 10, 11, 10, 11, 10, 11, 10, 11, 12, 13, 12, 13, 12,
            13, 12, 13, 14, 15, 14, 15, 14, 15, 14, 15,
        ];
        // SAFETY: enabled ISA; eight words and one complete shuffle vector.
        unsafe {
            Self(_mm512_shuffle_epi8(
                _mm512_broadcast_i32x4(_mm_loadu_si128(values.as_ptr().cast())),
                _mm512_loadu_si512(SELECT.as_ptr().cast()),
            ))
        }
    }
    #[inline(always)]
    fn zip_tiles_lo(self, other: Self) -> Self {
        // SAFETY: private lanes run only inside AVX-512F/BW entry points.
        unsafe { Self(_mm512_unpacklo_epi16(self.0, other.0)) }
    }
    #[inline(always)]
    fn zip_tiles_hi(self, other: Self) -> Self {
        // SAFETY: as above; unpacks stay inside each 128-bit tile.
        unsafe { Self(_mm512_unpackhi_epi16(self.0, other.0)) }
    }
    #[inline(always)]
    fn shuffle_tiles(self, pattern: &[u8; 16]) -> Self {
        use core::arch::x86_64::{_mm512_broadcast_i32x4, _mm512_shuffle_epi8};
        // SAFETY: enabled ISA; the pattern is exactly sixteen readable bytes.
        unsafe {
            Self(_mm512_shuffle_epi8(
                self.0,
                _mm512_broadcast_i32x4(_mm_loadu_si128(pattern.as_ptr().cast())),
            ))
        }
    }
    #[inline(always)]
    fn load_tiles8(rows: [&[u8]; 4]) -> Self {
        let (a, b, c, d) = (&rows[0][..8], &rows[1][..8], &rows[2][..8], &rows[3][..8]);
        // SAFETY: enabled ISA; each load reads eight checked bytes.
        unsafe {
            let lo = _mm_unpacklo_epi64(
                _mm_loadl_epi64(a.as_ptr().cast()),
                _mm_loadl_epi64(b.as_ptr().cast()),
            );
            let hi = _mm_unpacklo_epi64(
                _mm_loadl_epi64(c.as_ptr().cast()),
                _mm_loadl_epi64(d.as_ptr().cast()),
            );
            Self(_mm512_cvtepu8_epi16(_mm256_set_m128i(hi, lo)))
        }
    }
    #[inline(always)]
    fn store_tiles8(self, rows: [&mut [u8]; 4]) {
        use core::arch::x86_64::_mm_srli_si128;
        let [a, b, c, d] = rows;
        let (a, b, c, d) = (&mut a[..8], &mut b[..8], &mut c[..8], &mut d[..8]);
        // SAFETY: enabled ISA; each output has eight writable bytes.
        unsafe {
            let bytes = _mm512_cvtusepi16_epi8(_mm512_max_epi16(self.0, _mm512_setzero_si512()));
            let lo = _mm256_castsi256_si128(bytes);
            let hi = _mm256_extracti128_si256::<1>(bytes);
            _mm_storel_epi64(a.as_mut_ptr().cast(), lo);
            _mm_storel_epi64(b.as_mut_ptr().cast(), _mm_srli_si128::<8>(lo));
            _mm_storel_epi64(c.as_mut_ptr().cast(), hi);
            _mm_storel_epi64(d.as_mut_ptr().cast(), _mm_srli_si128::<8>(hi));
        }
    }
    #[inline(always)]
    fn load_u16_tiles8(rows: [&[u16]; 4]) -> Self {
        let (a, b, c, d) = (&rows[0][..8], &rows[1][..8], &rows[2][..8], &rows[3][..8]);
        // SAFETY: enabled ISA; each load reads eight checked `u16` samples.
        unsafe {
            let lo = _mm256_set_m128i(
                _mm_loadu_si128(b.as_ptr().cast()),
                _mm_loadu_si128(a.as_ptr().cast()),
            );
            let hi = _mm256_set_m128i(
                _mm_loadu_si128(d.as_ptr().cast()),
                _mm_loadu_si128(c.as_ptr().cast()),
            );
            Self(_mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi))
        }
    }
    #[inline(always)]
    fn store_u16_tiles8(self, rows: [&mut [u16]; 4], max_val: i16) {
        let [a, b, c, d] = rows;
        let (a, b, c, d) = (&mut a[..8], &mut b[..8], &mut c[..8], &mut d[..8]);
        let clamped = self.max(Self::splat(0)).min(Self::splat(max_val)).0;
        // SAFETY: enabled ISA; each destination has eight writable `u16` samples.
        unsafe {
            let lo = _mm512_castsi512_si256(clamped);
            let hi = _mm512_extracti64x4_epi64::<1>(clamped);
            _mm_storeu_si128(a.as_mut_ptr().cast(), _mm256_castsi256_si128(lo));
            _mm_storeu_si128(b.as_mut_ptr().cast(), _mm256_extracti128_si256::<1>(lo));
            _mm_storeu_si128(c.as_mut_ptr().cast(), _mm256_castsi256_si128(hi));
            _mm_storeu_si128(d.as_mut_ptr().cast(), _mm256_extracti128_si256::<1>(hi));
        }
    }
}
