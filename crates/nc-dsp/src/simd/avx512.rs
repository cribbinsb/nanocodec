//! Sixteen i32 lanes for the shared transform graphs, with a 16x16
//! transpose and masked comparisons. Values stay private and methods inline
//! only into AVX-512F/BW/VL wrappers, as required by `simd`'s contract.
//! Uses the existing AVX2 arithmetic and range proofs; no transform graph is
//! duplicated. Narrow packed transforms in dav1d's BSD-2 `itx_avx512.asm`
//! informed the review, but are not substituted without matching range proofs.

#![allow(clippy::inline_always)]
use super::I32Lanes;
use core::arch::x86_64::{
    __m512i, _mm_cvtsi32_si128, _mm_loadu_si128, _mm_storeu_si128, _mm256_loadu_si256,
    _mm512_abs_epi32, _mm512_add_epi32, _mm512_cmpeq_epi32_mask, _mm512_cmpgt_epi32_mask,
    _mm512_cmplt_epi32_mask, _mm512_cmpneq_epi32_mask, _mm512_cvtepi16_epi32, _mm512_cvtepu8_epi32,
    _mm512_cvtusepi32_epi8, _mm512_loadu_si512, _mm512_mask_blend_epi32, _mm512_mask_sub_epi32,
    _mm512_maskz_set1_epi32, _mm512_max_epi32, _mm512_min_epi32, _mm512_mullo_epi32,
    _mm512_or_si512, _mm512_set1_epi32, _mm512_setzero_si512, _mm512_shuffle_i32x4,
    _mm512_sll_epi32, _mm512_sra_epi32, _mm512_srl_epi32, _mm512_storeu_si512, _mm512_sub_epi32,
    _mm512_unpackhi_epi32, _mm512_unpackhi_epi64, _mm512_unpacklo_epi32, _mm512_unpacklo_epi64,
};
use core::mem::MaybeUninit;

/// Sixteen signed 32-bit transform lanes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct I32x16(__m512i);

impl I32Lanes for I32x16 {
    const LANES: usize = 16;

    #[inline(always)]
    fn splat(value: i32) -> Self {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { Self(_mm512_set1_epi32(value)) }
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { Self(_mm512_add_epi32(self.0, other.0)) }
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { Self(_mm512_sub_epi32(self.0, other.0)) }
    }

    #[inline(always)]
    fn half_btf(w0: i32, a: Self, w1: i32, b: Self, bit: u32) -> Self {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let sum = _mm512_add_epi32(
                _mm512_mullo_epi32(a.0, _mm512_set1_epi32(w0)),
                _mm512_mullo_epi32(b.0, _mm512_set1_epi32(w1)),
            );
            let rounded = _mm512_add_epi32(sum, _mm512_set1_epi32(1 << (bit - 1)));
            Self(_mm512_sra_epi32(rounded, _mm_cvtsi32_si128(bit as i32)))
        }
    }

    #[inline(always)]
    fn clamp16(self) -> Self {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            Self(_mm512_max_epi32(
                _mm512_min_epi32(self.0, _mm512_set1_epi32(i32::from(i16::MAX))),
                _mm512_set1_epi32(i32::from(i16::MIN)),
            ))
        }
    }

    #[inline(always)]
    fn round_shift(self, bits: u32) -> Self {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            if bits == 0 {
                return self;
            }
            let rounded = _mm512_add_epi32(self.0, _mm512_set1_epi32(1 << (bits - 1)));
            Self(_mm512_sra_epi32(rounded, _mm_cvtsi32_si128(bits as i32)))
        }
    }

    #[inline(always)]
    fn shl(self, bits: u32) -> Self {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { Self(_mm512_sll_epi32(self.0, _mm_cvtsi32_si128(bits as i32))) }
    }

    #[inline(always)]
    fn mul_scalar(self, c: i32) -> Self {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs).
        unsafe { Self(_mm512_mullo_epi32(self.0, _mm512_set1_epi32(c))) }
    }

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs).
        unsafe { Self(_mm512_mullo_epi32(self.0, other.0)) }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs).
        unsafe { Self(_mm512_abs_epi32(self.0)) }
    }

    #[inline(always)]
    fn shr_logical(self, bits: u32) -> Self {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs).
        unsafe { Self(_mm512_srl_epi32(self.0, _mm_cvtsi32_si128(bits as i32))) }
    }

    #[inline(always)]
    fn sra(self, bits: u32) -> Self {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs).
        unsafe { Self(_mm512_sra_epi32(self.0, _mm_cvtsi32_si128(bits as i32))) }
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs).
        unsafe { Self(_mm512_min_epi32(self.0, other.0)) }
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs).
        unsafe { Self(_mm512_max_epi32(self.0, other.0)) }
    }

    #[inline(always)]
    fn with_sign_of(self, like: Self) -> Self {
        // SAFETY: these private lane values exist only inside AVX-512 kernels.
        unsafe {
            Self(_mm512_mask_sub_epi32(
                self.0,
                _mm512_cmplt_epi32_mask(like.0, _mm512_setzero_si512()),
                _mm512_setzero_si512(),
                self.0,
            ))
        }
    }

    #[inline(always)]
    fn count_nonzero(self) -> u32 {
        // SAFETY: these private lane values exist only inside AVX-512 kernels.
        unsafe { _mm512_cmpneq_epi32_mask(self.0, _mm512_setzero_si512()).count_ones() }
    }

    #[inline(always)]
    fn load(src: &[i32]) -> Self {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let src = &src[..16];
            Self(_mm512_loadu_si512(src.as_ptr().cast()))
        }
    }

    #[inline(always)]
    fn store(self, dst: &mut [i32]) {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let dst = &mut dst[..16];
            _mm512_storeu_si512(dst.as_mut_ptr().cast(), self.0);
        }
    }

    #[inline(always)]
    fn load_i16(src: &[i16]) -> Self {
        let src = &src[..16];
        // SAFETY: sixteen readable i16 values; this ISA is available.
        unsafe {
            Self(_mm512_cvtepi16_epi32(_mm256_loadu_si256(
                src.as_ptr().cast(),
            )))
        }
    }

    #[inline(always)]
    fn add_to_u8(self, dst: &mut [u8]) {
        let dst = &mut dst[..16];
        // SAFETY: sixteen readable/writable bytes; this ISA is available.
        unsafe {
            let pred = _mm512_cvtepu8_epi32(_mm_loadu_si128(dst.as_ptr().cast()));
            let sum = _mm512_max_epi32(_mm512_setzero_si512(), _mm512_add_epi32(pred, self.0));
            _mm_storeu_si128(dst.as_mut_ptr().cast(), _mm512_cvtusepi32_epi8(sum));
        }
    }

    #[inline(always)]
    fn store_transposed(rows: &[Self], dst: &mut [MaybeUninit<i32>], stride: usize) {
        assert!(rows.len() >= 16);
        // SAFETY: these private lane values exist only inside AVX-512 kernels;
        // each destination row is bounds-checked before storing.
        unsafe {
            let mut pairs = [_mm512_setzero_si512(); 16];
            for i in 0..8 {
                pairs[2 * i] = _mm512_unpacklo_epi32(rows[2 * i].0, rows[2 * i + 1].0);
                pairs[2 * i + 1] = _mm512_unpackhi_epi32(rows[2 * i].0, rows[2 * i + 1].0);
            }
            let mut quads = pairs;
            for i in (0..16).step_by(4) {
                quads[i] = _mm512_unpacklo_epi64(pairs[i], pairs[i + 2]);
                quads[i + 1] = _mm512_unpackhi_epi64(pairs[i], pairs[i + 2]);
                quads[i + 2] = _mm512_unpacklo_epi64(pairs[i + 1], pairs[i + 3]);
                quads[i + 3] = _mm512_unpackhi_epi64(pairs[i + 1], pairs[i + 3]);
            }
            let mut out = quads;
            for i in 0..4 {
                let a = _mm512_shuffle_i32x4::<0x88>(quads[i], quads[i + 4]);
                let b = _mm512_shuffle_i32x4::<0xdd>(quads[i], quads[i + 4]);
                let c = _mm512_shuffle_i32x4::<0x88>(quads[i + 8], quads[i + 12]);
                let d = _mm512_shuffle_i32x4::<0xdd>(quads[i + 8], quads[i + 12]);
                out[i] = _mm512_shuffle_i32x4::<0x88>(a, c);
                out[i + 4] = _mm512_shuffle_i32x4::<0x88>(b, d);
                out[i + 8] = _mm512_shuffle_i32x4::<0xdd>(a, c);
                out[i + 12] = _mm512_shuffle_i32x4::<0xdd>(b, d);
            }
            for (i, value) in out.into_iter().enumerate() {
                let row = &mut dst[i * stride..][..16];
                _mm512_storeu_si512(row.as_mut_ptr().cast(), value);
            }
        }
    }

    #[inline(always)]
    fn gt(self, other: Self) -> Self {
        // SAFETY: these private lane values exist only inside AVX-512 kernels.
        unsafe {
            Self(_mm512_maskz_set1_epi32(
                _mm512_cmpgt_epi32_mask(self.0, other.0),
                -1,
            ))
        }
    }

    #[inline(always)]
    fn eq(self, other: Self) -> Self {
        // SAFETY: these private lane values exist only inside AVX-512 kernels.
        unsafe {
            Self(_mm512_maskz_set1_epi32(
                _mm512_cmpeq_epi32_mask(self.0, other.0),
                -1,
            ))
        }
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX-512F/BW/VL kernels (module docs).
        unsafe { Self(_mm512_or_si512(self.0, other.0)) }
    }

    #[inline(always)]
    fn select(self, if_set: Self, if_clear: Self) -> Self {
        // SAFETY: these private lane values exist only inside AVX-512 kernels.
        unsafe {
            Self(_mm512_mask_blend_epi32(
                _mm512_cmpneq_epi32_mask(self.0, _mm512_setzero_si512()),
                if_clear.0,
                if_set.0,
            ))
        }
    }
}

/// Sixteen adjacent pairs of signed 16-bit transform values.
#[derive(Debug, Clone, Copy)]
pub(crate) struct WordPairs16(__m512i);

impl I32x16 {
    /// Pair lane values that the HEVC forward range proof keeps in i16.
    #[inline(always)]
    pub(crate) fn word_pairs(self, high: Self) -> WordPairs16 {
        // SAFETY: lane values exist only inside AVX-512F/BW/VL kernels.
        unsafe {
            WordPairs16(core::arch::x86_64::_mm512_mask_blend_epi16(
                0xaaaa_aaaa,
                self.0,
                core::arch::x86_64::_mm512_slli_epi32::<16>(high.0),
            ))
        }
    }
}
impl WordPairs16 {
    /// Exact signed paired dot product, with two packed i16 weights.
    #[inline(always)]
    pub(crate) fn dot(self, weights: i32) -> I32x16 {
        // SAFETY: lane values exist only inside AVX-512F/BW/VL kernels.
        unsafe {
            I32x16(core::arch::x86_64::_mm512_madd_epi16(
                self.0,
                _mm512_set1_epi32(weights),
            ))
        }
    }
}

impl I32x16 {
    /// Transpose four independent 4x4 tiles, one per 128-bit sublane.
    #[inline(always)]
    pub(crate) fn transpose4(rows: [Self; 4]) -> [Self; 4] {
        // SAFETY: private values exist only inside supported ISA kernels.
        unsafe {
            let a = _mm512_unpacklo_epi32(rows[0].0, rows[1].0);
            let b = _mm512_unpackhi_epi32(rows[0].0, rows[1].0);
            let c = _mm512_unpacklo_epi32(rows[2].0, rows[3].0);
            let d = _mm512_unpackhi_epi32(rows[2].0, rows[3].0);
            [
                Self(_mm512_unpacklo_epi64(a, c)),
                Self(_mm512_unpackhi_epi64(a, c)),
                Self(_mm512_unpacklo_epi64(b, d)),
                Self(_mm512_unpackhi_epi64(b, d)),
            ]
        }
    }

    /// Transpose a 4x4 matrix of 128-bit chunks: four block registers to
    /// four row registers (or the inverse, since a transpose is involutive).
    #[inline(always)]
    pub(crate) fn transpose_chunks(rows: [Self; 4]) -> [Self; 4] {
        // SAFETY: private values exist only inside supported ISA kernels.
        unsafe {
            let a = _mm512_shuffle_i32x4::<0x44>(rows[0].0, rows[1].0);
            let b = _mm512_shuffle_i32x4::<0xee>(rows[0].0, rows[1].0);
            let c = _mm512_shuffle_i32x4::<0x44>(rows[2].0, rows[3].0);
            let d = _mm512_shuffle_i32x4::<0xee>(rows[2].0, rows[3].0);
            [
                Self(_mm512_shuffle_i32x4::<0x88>(a, c)),
                Self(_mm512_shuffle_i32x4::<0xdd>(a, c)),
                Self(_mm512_shuffle_i32x4::<0x88>(b, d)),
                Self(_mm512_shuffle_i32x4::<0xdd>(b, d)),
            ]
        }
    }
}
