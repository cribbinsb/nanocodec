//! Thirty-two word lanes for the shared prediction graph. Accumulators use
//! the AVX2 lane implementation's paired-tap layout, widened to four 128-bit
//! groups: no filter equations or reference-window rules are duplicated.
//! See `simd`'s private-lane contract: methods inline only into wrappers with
//! AVX-512F/BW/VL enabled. No DQ or VNNI instructions are required.
//! HEVC horizontal sums remain below 112 * 255 = 28,560; vertical
//! accumulation stays in i32, with rounding before signed-word narrowing.

#![allow(clippy::inline_always)]

use super::{HIGH_LOOKUP_TABLE, I16FilterLanes, I16Lanes};
use core::arch::x86_64::{
    __m512i, _mm_cvtsi32_si128, _mm_loadl_epi64, _mm_loadu_si128, _mm_storel_epi64,
    _mm_storeu_si128, _mm_unpackhi_epi64, _mm_unpacklo_epi64, _mm256_castsi256_si128,
    _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_set_m128i, _mm256_storeu_si256,
    _mm512_abs_epi16, _mm512_add_epi16, _mm512_add_epi32, _mm512_adds_epi16, _mm512_and_si512,
    _mm512_castsi256_si512, _mm512_castsi512_si128, _mm512_castsi512_si256,
    _mm512_cmpeq_epi16_mask, _mm512_cmpgt_epi16_mask, _mm512_cmplt_epi16_mask,
    _mm512_cvtepi32_epi16, _mm512_cvtepu8_epi16, _mm512_cvtepu16_epi32, _mm512_cvtusepi16_epi8,
    _mm512_extracti32x4_epi32, _mm512_extracti64x4_epi64, _mm512_i32gather_epi32,
    _mm512_inserti64x4, _mm512_loadu_si512, _mm512_madd_epi16, _mm512_mask_blend_epi16,
    _mm512_mask_sub_epi16, _mm512_maskz_set1_epi16, _mm512_max_epi16, _mm512_min_epi16,
    _mm512_min_epu16, _mm512_mulhrs_epi16, _mm512_mullo_epi16, _mm512_packs_epi32,
    _mm512_set1_epi16, _mm512_set1_epi32, _mm512_setzero_si512, _mm512_slli_epi32,
    _mm512_sra_epi16, _mm512_sra_epi32, _mm512_srai_epi32, _mm512_srl_epi16, _mm512_srli_epi32,
    _mm512_storeu_si512, _mm512_sub_epi16, _mm512_subs_epu16, _mm512_test_epi16_mask,
    _mm512_unpackhi_epi16, _mm512_unpacklo_epi16,
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
            let extend = |v: __m512i| _mm512_srai_epi32::<16>(_mm512_slli_epi32::<16>(v));
            Self(_mm512_inserti64x4::<1>(
                _mm512_castsi256_si512(_mm512_cvtepi32_epi16(extend(low))),
                _mm512_cvtepi32_epi16(extend(high)),
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
}
