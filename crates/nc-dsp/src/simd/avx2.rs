//! x86-64 vector types: 256-bit AVX2 for wide rows, 128-bit SSE4.1 (implied
//! by AVX2) for narrow ones.
//!
//! See the `simd` module docs for why the methods are safe functions: they
//! are only ever called from kernels running under
//! `#[target_feature(enable = "avx2")]`.

#![allow(clippy::inline_always)]

use core::arch::x86_64::{
    __m128i, __m256i, _mm_abs_epi16, _mm_abs_epi32, _mm_add_epi16, _mm_add_epi32, _mm_adds_epi16,
    _mm_alignr_epi8, _mm_and_si128, _mm_blend_epi16, _mm_blendv_epi8, _mm_castsi128_ps,
    _mm_cmpeq_epi16, _mm_cmpeq_epi32, _mm_cmpgt_epi16, _mm_cmpgt_epi32, _mm_cvtepi16_epi32,
    _mm_cvtepu8_epi16, _mm_cvtepu8_epi32, _mm_cvtepu16_epi32, _mm_cvtsi32_si128, _mm_cvtsi128_si32,
    _mm_extract_epi32, _mm_hadd_epi32, _mm_loadl_epi64, _mm_loadu_si128, _mm_madd_epi16,
    _mm_maddubs_epi16, _mm_max_epi16, _mm_max_epi32, _mm_min_epi16, _mm_min_epi32, _mm_min_epu16,
    _mm_movemask_ps, _mm_mulhrs_epi16, _mm_mullo_epi16, _mm_mullo_epi32, _mm_or_si128,
    _mm_packs_epi32, _mm_packus_epi16, _mm_packus_epi32, _mm_set1_epi16, _mm_set1_epi32,
    _mm_setr_epi32, _mm_setzero_si128, _mm_shuffle_epi8, _mm_shuffle_epi32, _mm_sign_epi16,
    _mm_sign_epi32, _mm_sll_epi32, _mm_slli_epi32, _mm_sra_epi16, _mm_sra_epi32, _mm_srl_epi16,
    _mm_srl_epi32, _mm_srli_si128, _mm_storel_epi64, _mm_storeu_si128, _mm_sub_epi16,
    _mm_sub_epi32, _mm_subs_epu16, _mm_testz_si128, _mm_unpackhi_epi8, _mm_unpackhi_epi16,
    _mm_unpackhi_epi32, _mm_unpackhi_epi64, _mm_unpacklo_epi8, _mm_unpacklo_epi16,
    _mm_unpacklo_epi32, _mm_unpacklo_epi64, _mm256_abs_epi16, _mm256_abs_epi32, _mm256_add_epi16,
    _mm256_add_epi32, _mm256_adds_epi16, _mm256_alignr_epi8, _mm256_and_si256, _mm256_blend_epi16,
    _mm256_blendv_epi8, _mm256_castsi256_ps, _mm256_castsi256_si128, _mm256_cmpeq_epi16,
    _mm256_cmpeq_epi32, _mm256_cmpgt_epi16, _mm256_cmpgt_epi32, _mm256_cvtepi16_epi32,
    _mm256_cvtepu8_epi16, _mm256_cvtepu8_epi32, _mm256_cvtepu16_epi32, _mm256_cvtepu32_epi64,
    _mm256_extracti128_si256, _mm256_i32gather_epi32, _mm256_loadu_si256, _mm256_madd_epi16,
    _mm256_maddubs_epi16, _mm256_max_epi16, _mm256_max_epi32, _mm256_min_epi16, _mm256_min_epi32,
    _mm256_min_epu16, _mm256_movemask_ps, _mm256_mulhrs_epi16, _mm256_mullo_epi16,
    _mm256_mullo_epi32, _mm256_or_si256, _mm256_packs_epi32, _mm256_permute2x128_si256,
    _mm256_permute4x64_epi64, _mm256_set_m128i, _mm256_set1_epi16, _mm256_set1_epi32,
    _mm256_setr_epi8, _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_shuffle_epi32,
    _mm256_sign_epi16, _mm256_sign_epi32, _mm256_sll_epi32, _mm256_slli_epi32, _mm256_sra_epi16,
    _mm256_sra_epi32, _mm256_srai_epi32, _mm256_srl_epi16, _mm256_srl_epi32, _mm256_srli_epi32,
    _mm256_storeu_si256, _mm256_sub_epi16, _mm256_sub_epi32, _mm256_subs_epu16, _mm256_testz_si256,
    _mm256_unpackhi_epi16, _mm256_unpackhi_epi32, _mm256_unpackhi_epi64, _mm256_unpacklo_epi16,
    _mm256_unpacklo_epi32, _mm256_unpacklo_epi64,
};

use core::mem::MaybeUninit;

use super::{
    HIGH_LOOKUP_TABLE, I16FilterLanes, I16Lanes, I16Pairs, I16x8Shuffles, I32Lanes, LOOKUP_TABLE,
    WordPairLanes, packed_pair,
};

/// Eight `i32` lanes (AVX2).
#[derive(Debug, Clone, Copy)]
pub(crate) struct I32x8(pub(crate) __m256i);

/// Four `i32` lanes (SSE4.1).
#[derive(Debug, Clone, Copy)]
pub(crate) struct I32x4(pub(crate) __m128i);

/// Eight lanes of two `i16` each: the low 16 bits of two [`I32x8`] vectors,
/// interleaved for `vpmaddwd` dot products.
#[derive(Debug, Clone, Copy)]
pub(crate) struct WordPairs8(__m256i);

impl I32x8 {
    /// Lane `i` holds (`self[i]`, `high[i]`) truncated to 16 bits each:
    /// exact when every value fits `i16`.
    #[inline(always)]
    pub(crate) fn word_pairs(self, high: Self) -> WordPairs8 {
        /// Word blend mask taking the odd (high) word of each lane from `high`.
        const ODD_WORDS: i32 = 0xAA;
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe {
            WordPairs8(_mm256_blend_epi16::<ODD_WORDS>(
                self.0,
                _mm256_slli_epi32::<16>(high.0),
            ))
        }
    }
}

impl WordPairs8 {
    /// `low * w0 + high * w1` per lane, exact in `i32` (`vpmaddwd`), with
    /// the weights packed as `weights = (w1 << 16) | (w0 as u16)`.
    #[inline(always)]
    pub(crate) fn dot(self, weights: i32) -> I32x8 {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { I32x8(_mm256_madd_epi16(self.0, _mm256_set1_epi32(weights))) }
    }
}

/// Sixteen `i16` lanes (AVX2).
#[derive(Debug, Clone, Copy)]
pub(crate) struct I16x16(pub(crate) __m256i);

/// Eight `i16` lanes (SSE4.1).
#[derive(Debug, Clone, Copy)]
pub(crate) struct I16x8(pub(crate) __m128i);

/// Four `i16` lanes in the low half of an SSE register; the high half
/// stays zero.
#[derive(Debug, Clone, Copy)]
pub(crate) struct I16x4(__m128i);

/// Shift counts for the variable-shift intrinsics.
#[inline(always)]
fn count(bits: u32) -> __m128i {
    // SAFETY: these types run only inside AVX2 kernels (module docs);
    // slices are indexed before every load and store.
    unsafe { _mm_cvtsi32_si128(bits as i32) }
}

// ---------------------------------------------------------------- I32x8 ----

impl WordPairLanes for I32x8 {
    type Pairs = WordPairs8;

    #[inline(always)]
    fn transpose_registers(rows: &[Self], columns: &mut [Self]) {
        columns.copy_from_slice(&transpose_i32x8(rows));
    }

    #[inline(always)]
    fn word_pairs(self, high: Self) -> WordPairs8 {
        I32x8::word_pairs(self, high)
    }

    #[inline(always)]
    fn pair_dot(pairs: WordPairs8, weights: i32) -> Self {
        pairs.dot(weights)
    }
}

impl I32Lanes for I32x8 {
    const LANES: usize = 8;

    #[inline(always)]
    fn splat(value: i32) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { Self(_mm256_set1_epi32(value)) }
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { Self(_mm256_add_epi32(self.0, other.0)) }
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { Self(_mm256_sub_epi32(self.0, other.0)) }
    }

    #[inline(always)]
    fn half_btf(w0: i32, a: Self, w1: i32, b: Self, bit: u32) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let sum = _mm256_add_epi32(
                _mm256_mullo_epi32(a.0, _mm256_set1_epi32(w0)),
                _mm256_mullo_epi32(b.0, _mm256_set1_epi32(w1)),
            );
            let rounded = _mm256_add_epi32(sum, _mm256_set1_epi32(1 << (bit - 1)));
            Self(_mm256_sra_epi32(rounded, count(bit)))
        }
    }

    #[inline(always)]
    fn clamp16(self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            Self(_mm256_max_epi32(
                _mm256_min_epi32(self.0, _mm256_set1_epi32(i32::from(i16::MAX))),
                _mm256_set1_epi32(i32::from(i16::MIN)),
            ))
        }
    }

    #[inline(always)]
    fn round_shift(self, bits: u32) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            if bits == 0 {
                return self;
            }
            let rounded = _mm256_add_epi32(self.0, _mm256_set1_epi32(1 << (bits - 1)));
            Self(_mm256_sra_epi32(rounded, count(bits)))
        }
    }

    #[inline(always)]
    fn shl(self, bits: u32) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { Self(_mm256_sll_epi32(self.0, count(bits))) }
    }

    #[inline(always)]
    fn mul_scalar(self, c: i32) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm256_mullo_epi32(self.0, _mm256_set1_epi32(c))) }
    }

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm256_mullo_epi32(self.0, other.0)) }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm256_abs_epi32(self.0)) }
    }

    #[inline(always)]
    fn shr_logical(self, bits: u32) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm256_srl_epi32(self.0, count(bits))) }
    }

    #[inline(always)]
    fn sra(self, bits: u32) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm256_sra_epi32(self.0, count(bits))) }
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm256_min_epi32(self.0, other.0)) }
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm256_max_epi32(self.0, other.0)) }
    }

    #[inline(always)]
    fn with_sign_of(self, like: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe {
            Self(_mm256_sign_epi32(
                self.0,
                _mm256_or_si256(like.0, _mm256_set1_epi32(1)),
            ))
        }
    }

    #[inline(always)]
    fn count_nonzero(self) -> u32 {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe {
            let zero = _mm256_cmpeq_epi32(self.0, _mm256_setzero_si256());
            8 - (_mm256_movemask_ps(_mm256_castsi256_ps(zero)) as u32).count_ones()
        }
    }

    #[inline(always)]
    fn load(src: &[i32]) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let src = &src[..8];
            Self(_mm256_loadu_si256(src.as_ptr().cast()))
        }
    }

    #[inline(always)]
    fn store(self, dst: &mut [i32]) {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let dst = &mut dst[..8];
            _mm256_storeu_si256(dst.as_mut_ptr().cast(), self.0);
        }
    }

    #[inline(always)]
    fn load_i16(src: &[i16]) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let src = &src[..8];
            Self(_mm256_cvtepi16_epi32(_mm_loadu_si128(src.as_ptr().cast())))
        }
    }

    #[inline(always)]
    fn add_to_u8(self, dst: &mut [u8]) {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let dst = &mut dst[..8];
            let residual = _mm_packs_epi32(
                _mm256_castsi256_si128(self.0),
                _mm256_extracti128_si256::<1>(self.0),
            );
            let pred = _mm_cvtepu8_epi16(_mm_loadl_epi64(dst.as_ptr().cast()));
            // Saturating the residual to 16 bits cannot change the clipped sum.
            let sum = _mm_packus_epi16(_mm_add_epi16(residual, pred), _mm_setzero_si128());
            _mm_storel_epi64(dst.as_mut_ptr().cast(), sum);
        }
    }

    #[inline(always)]
    fn add_constant_u8(residual: i32, n: usize, dst: &mut [u8], stride: usize) {
        // SAFETY: AVX2 kernels only; caller's slice covers `n` rows at `stride`.
        unsafe {
            crate::constant_add::add::<crate::constant_add::avx2::Bytes>(
                residual, n, n, dst, stride,
            );
        }
    }

    #[inline(always)]
    fn add_to_u16(self, dst: &mut [u16], max_val: i32) {
        let dst = &mut dst[..8];
        // SAFETY: AVX2 kernels only; `dst` holds 8 `u16` samples.
        unsafe {
            let pred = _mm256_cvtepu16_epi32(_mm_loadu_si128(dst.as_ptr().cast()));
            let sum = _mm256_min_epi32(
                _mm256_max_epi32(_mm256_add_epi32(pred, self.0), _mm256_setzero_si256()),
                _mm256_set1_epi32(max_val),
            );
            let packed = _mm_packus_epi32(
                _mm256_castsi256_si128(sum),
                _mm256_extracti128_si256::<1>(sum),
            );
            _mm_storeu_si128(dst.as_mut_ptr().cast(), packed);
        }
    }

    #[inline(always)]
    fn store_transposed(rows: &[Self], dst: &mut [MaybeUninit<i32>], dst_stride: usize) {
        for (r, value) in transpose_i32x8(rows).into_iter().enumerate() {
            let dst = &mut dst[r * dst_stride..][..Self::LANES];
            // SAFETY: ISA enabled; the slice covers one complete vector.
            unsafe {
                _mm256_storeu_si256(dst.as_mut_ptr().cast(), value.0);
            }
        }
    }

    #[inline(always)]
    fn store_sat16(self, dst: &mut [i16]) {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let dst = &mut dst[..8];
            let words = _mm_packs_epi32(
                _mm256_castsi256_si128(self.0),
                _mm256_extracti128_si256::<1>(self.0),
            );
            _mm_storeu_si128(dst.as_mut_ptr().cast(), words);
        }
    }

    #[inline(always)]
    fn store_sat16_pair(low: Self, high: Self, dst: &mut [i16]) {
        /// `vpermq` order putting the pack's `low` quads before its `high`
        /// quads.
        const LOW_THEN_HIGH: i32 = 0b11_01_10_00;
        let dst = &mut dst[..16];
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // `dst` holds 16 values.
        unsafe {
            let words =
                _mm256_permute4x64_epi64::<LOW_THEN_HIGH>(_mm256_packs_epi32(low.0, high.0));
            _mm256_storeu_si256(dst.as_mut_ptr().cast(), words);
        }
    }

    #[inline(always)]
    fn store_transposed_sat16(rows: &[Self], dst: &mut [MaybeUninit<i16>], dst_stride: usize) {
        /// `vpermq` order putting each pack's first-row quads before its
        /// second-row quads.
        const ROWS_IN_HALVES: i32 = 0b11_01_10_00;
        let out = transpose_i32x8(rows);
        for (r, pair) in out.as_chunks::<2>().0.iter().enumerate() {
            // SAFETY: ISA enabled; each slice covers one complete 8-lane row.
            unsafe {
                let words = _mm256_permute4x64_epi64::<ROWS_IN_HALVES>(_mm256_packs_epi32(
                    pair[0].0, pair[1].0,
                ));
                let first = &mut dst[2 * r * dst_stride..][..Self::LANES];
                _mm_storeu_si128(first.as_mut_ptr().cast(), _mm256_castsi256_si128(words));
                let second = &mut dst[(2 * r + 1) * dst_stride..][..Self::LANES];
                _mm_storeu_si128(
                    second.as_mut_ptr().cast(),
                    _mm256_extracti128_si256::<1>(words),
                );
            }
        }
    }

    #[inline(always)]
    fn add_transposed_u8(rows: &[Self], dst: &mut [u8], dst_stride: usize) {
        let needed = dst_stride
            .checked_mul(Self::LANES - 1)
            .and_then(|n| n.checked_add(Self::LANES))
            .expect("transpose tile extent");
        let dst = &mut dst[..needed];
        for (r, value) in transpose_i32x8(rows).into_iter().enumerate() {
            // SAFETY: r < LANES and the checked span covers the last row's
            // full vector; each temporary row borrow ends in this iteration.
            let row = unsafe {
                core::slice::from_raw_parts_mut(dst.as_mut_ptr().add(r * dst_stride), Self::LANES)
            };
            value.add_to_u8(row);
        }
    }

    #[inline(always)]
    fn add_transposed_u16(rows: &[Self], dst: &mut [u16], dst_stride: usize, max_val: i32) {
        let needed = dst_stride
            .checked_mul(Self::LANES - 1)
            .and_then(|n| n.checked_add(Self::LANES))
            .expect("transpose tile extent");
        let dst = &mut dst[..needed];
        for (r, value) in transpose_i32x8(rows).into_iter().enumerate() {
            // SAFETY: r < LANES and the checked span covers the last row's
            // full vector; each temporary row borrow ends in this iteration.
            let row = unsafe {
                core::slice::from_raw_parts_mut(dst.as_mut_ptr().add(r * dst_stride), Self::LANES)
            };
            value.add_to_u16(row, max_val);
        }
    }

    #[inline(always)]
    fn gt(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm256_cmpgt_epi32(self.0, other.0)) }
    }

    #[inline(always)]
    fn eq(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm256_cmpeq_epi32(self.0, other.0)) }
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm256_or_si256(self.0, other.0)) }
    }

    #[inline(always)]
    fn select(self, if_set: Self, if_clear: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm256_blendv_epi8(if_clear.0, if_set.0, self.0)) }
    }
}

// ---------------------------------------------------------------- I32x4 ----

impl I32Lanes for I32x4 {
    const LANES: usize = 4;

    #[inline(always)]
    fn splat(value: i32) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { Self(_mm_set1_epi32(value)) }
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { Self(_mm_add_epi32(self.0, other.0)) }
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { Self(_mm_sub_epi32(self.0, other.0)) }
    }

    #[inline(always)]
    fn half_btf(w0: i32, a: Self, w1: i32, b: Self, bit: u32) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let sum = _mm_add_epi32(
                _mm_mullo_epi32(a.0, _mm_set1_epi32(w0)),
                _mm_mullo_epi32(b.0, _mm_set1_epi32(w1)),
            );
            let rounded = _mm_add_epi32(sum, _mm_set1_epi32(1 << (bit - 1)));
            Self(_mm_sra_epi32(rounded, count(bit)))
        }
    }

    #[inline(always)]
    fn clamp16(self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            Self(_mm_max_epi32(
                _mm_min_epi32(self.0, _mm_set1_epi32(i32::from(i16::MAX))),
                _mm_set1_epi32(i32::from(i16::MIN)),
            ))
        }
    }

    #[inline(always)]
    fn round_shift(self, bits: u32) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            if bits == 0 {
                return self;
            }
            let rounded = _mm_add_epi32(self.0, _mm_set1_epi32(1 << (bits - 1)));
            Self(_mm_sra_epi32(rounded, count(bits)))
        }
    }

    #[inline(always)]
    fn shl(self, bits: u32) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { Self(_mm_sll_epi32(self.0, count(bits))) }
    }

    #[inline(always)]
    fn mul_scalar(self, c: i32) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm_mullo_epi32(self.0, _mm_set1_epi32(c))) }
    }

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm_mullo_epi32(self.0, other.0)) }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm_abs_epi32(self.0)) }
    }

    #[inline(always)]
    fn shr_logical(self, bits: u32) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm_srl_epi32(self.0, count(bits))) }
    }

    #[inline(always)]
    fn sra(self, bits: u32) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm_sra_epi32(self.0, count(bits))) }
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm_min_epi32(self.0, other.0)) }
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm_max_epi32(self.0, other.0)) }
    }

    #[inline(always)]
    fn with_sign_of(self, like: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe {
            Self(_mm_sign_epi32(
                self.0,
                _mm_or_si128(like.0, _mm_set1_epi32(1)),
            ))
        }
    }

    #[inline(always)]
    fn count_nonzero(self) -> u32 {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe {
            let zero = _mm_cmpeq_epi32(self.0, _mm_setzero_si128());
            4 - (_mm_movemask_ps(_mm_castsi128_ps(zero)) as u32).count_ones()
        }
    }

    #[inline(always)]
    fn load(src: &[i32]) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let src = &src[..4];
            Self(_mm_loadu_si128(src.as_ptr().cast()))
        }
    }

    #[inline(always)]
    fn store(self, dst: &mut [i32]) {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let dst = &mut dst[..4];
            _mm_storeu_si128(dst.as_mut_ptr().cast(), self.0);
        }
    }

    #[inline(always)]
    fn load_i16(src: &[i16]) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let src = &src[..4];
            Self(_mm_cvtepi16_epi32(_mm_loadl_epi64(src.as_ptr().cast())))
        }
    }

    #[inline(always)]
    fn add_to_u8(self, dst: &mut [u8]) {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let dst = &mut dst[..4];
            let pred = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(i32::from_le_bytes(
                (&*dst).try_into().expect("4 bytes"),
            )));
            let sum = _mm_add_epi32(self.0, pred);
            let packed = _mm_packus_epi16(_mm_packs_epi32(sum, sum), _mm_setzero_si128());
            dst.copy_from_slice(&_mm_cvtsi128_si32(packed).to_le_bytes());
        }
    }

    #[inline(always)]
    fn add_constant_u8(residual: i32, n: usize, dst: &mut [u8], stride: usize) {
        // SAFETY: AVX2 kernels only; caller's slice covers `n` rows at `stride`.
        unsafe {
            crate::constant_add::add::<crate::constant_add::avx2::Bytes>(
                residual, n, n, dst, stride,
            );
        }
    }

    #[inline(always)]
    fn add_to_u16(self, dst: &mut [u16], max_val: i32) {
        let dst = &mut dst[..4];
        // SAFETY: AVX2 kernels only; `dst` holds 4 `u16` samples.
        unsafe {
            let pred = _mm_cvtepu16_epi32(_mm_loadl_epi64(dst.as_ptr().cast()));
            let sum = _mm_min_epi32(
                _mm_max_epi32(_mm_add_epi32(pred, self.0), _mm_setzero_si128()),
                _mm_set1_epi32(max_val),
            );
            _mm_storel_epi64(dst.as_mut_ptr().cast(), _mm_packus_epi32(sum, sum));
        }
    }

    #[inline(always)]
    fn store_transposed(rows: &[Self], dst: &mut [MaybeUninit<i32>], dst_stride: usize) {
        for (r, value) in transpose_i32x4(rows).into_iter().enumerate() {
            let dst = &mut dst[r * dst_stride..][..Self::LANES];
            // SAFETY: ISA enabled; the slice covers one complete vector.
            unsafe {
                _mm_storeu_si128(dst.as_mut_ptr().cast(), value.0);
            }
        }
    }

    #[inline(always)]
    fn store_sat16(self, dst: &mut [i16]) {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let dst = &mut dst[..4];
            _mm_storel_epi64(dst.as_mut_ptr().cast(), _mm_packs_epi32(self.0, self.0));
        }
    }

    #[inline(always)]
    fn store_sat16_pair(low: Self, high: Self, dst: &mut [i16]) {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // `dst` holds eight values.
        unsafe {
            let dst = &mut dst[..8];
            _mm_storeu_si128(dst.as_mut_ptr().cast(), _mm_packs_epi32(low.0, high.0));
        }
    }

    #[inline(always)]
    fn store_transposed_sat16(rows: &[Self], dst: &mut [MaybeUninit<i16>], dst_stride: usize) {
        /// Bytes of one 4-lane row of words: the second row's offset.
        const ROW_BYTES: i32 = 8;
        let out = transpose_i32x4(rows);
        for (r, pair) in out.as_chunks::<2>().0.iter().enumerate() {
            // SAFETY: ISA enabled; each slice covers one complete 4-lane row.
            unsafe {
                let words = _mm_packs_epi32(pair[0].0, pair[1].0);
                let first = &mut dst[2 * r * dst_stride..][..Self::LANES];
                _mm_storel_epi64(first.as_mut_ptr().cast(), words);
                let second = &mut dst[(2 * r + 1) * dst_stride..][..Self::LANES];
                _mm_storel_epi64(
                    second.as_mut_ptr().cast(),
                    _mm_srli_si128::<ROW_BYTES>(words),
                );
            }
        }
    }

    #[inline(always)]
    fn add_transposed_u8(rows: &[Self], dst: &mut [u8], dst_stride: usize) {
        let needed = dst_stride
            .checked_mul(Self::LANES - 1)
            .and_then(|n| n.checked_add(Self::LANES))
            .expect("transpose tile extent");
        let dst = &mut dst[..needed];
        for (r, value) in transpose_i32x4(rows).into_iter().enumerate() {
            // SAFETY: r < LANES and the checked span covers the last row's
            // full vector; each temporary row borrow ends in this iteration.
            let row = unsafe {
                core::slice::from_raw_parts_mut(dst.as_mut_ptr().add(r * dst_stride), Self::LANES)
            };
            value.add_to_u8(row);
        }
    }

    #[inline(always)]
    fn add_transposed_u16(rows: &[Self], dst: &mut [u16], dst_stride: usize, max_val: i32) {
        let needed = dst_stride
            .checked_mul(Self::LANES - 1)
            .and_then(|n| n.checked_add(Self::LANES))
            .expect("transpose tile extent");
        let dst = &mut dst[..needed];
        for (r, value) in transpose_i32x4(rows).into_iter().enumerate() {
            // SAFETY: r < LANES and the checked span covers the last row's
            // full vector; each temporary row borrow ends in this iteration.
            let row = unsafe {
                core::slice::from_raw_parts_mut(dst.as_mut_ptr().add(r * dst_stride), Self::LANES)
            };
            value.add_to_u16(row, max_val);
        }
    }

    #[inline(always)]
    fn gt(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm_cmpgt_epi32(self.0, other.0)) }
    }

    #[inline(always)]
    fn eq(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm_cmpeq_epi32(self.0, other.0)) }
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm_or_si128(self.0, other.0)) }
    }

    #[inline(always)]
    fn select(self, if_set: Self, if_clear: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm_blendv_epi8(if_clear.0, if_set.0, self.0)) }
    }
}

// --------------------------------------------------------------- I16x16 ----

/// Sum the eight `i32` lanes of `v` into an `i64`.
#[inline(always)]
pub(crate) fn sum_i32x8(v: __m256i) -> i64 {
    let mut lanes = [0i32; 8];
    I32x8(v).store(&mut lanes);
    lanes.iter().map(|&x| i64::from(x)).sum()
}

/// Sum the four `i32` lanes of `v` into an `i64`.
#[inline(always)]
fn sum_i32x4(v: __m128i) -> i64 {
    let mut lanes = [0i32; 4];
    I32x4(v).store(&mut lanes);
    lanes.iter().map(|&x| i64::from(x)).sum()
}

/// `clamp((v + round) >> shift)` on four `i32` lanes, as `i32`.
#[inline(always)]
fn round_shift_x4(v: __m128i, round: i32, shift: u32) -> __m128i {
    // SAFETY: these types run only inside AVX2 kernels (module docs);
    // slices are indexed before every load and store.
    unsafe { _mm_sra_epi32(_mm_add_epi32(v, _mm_set1_epi32(round)), count(shift)) }
}

impl I16Lanes for I16x16 {
    const LANES: usize = 16;
    type Acc = [__m256i; 2];

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm256_min_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm256_max_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm256_abs_epi16(self.0)) }
    }

    #[inline(always)]
    fn with_sign_of(self, like: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm256_sign_epi16(self.0, like.0)) }
    }

    #[inline(always)]
    fn splat(value: i16) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { Self(_mm256_set1_epi16(value)) }
    }

    #[inline(always)]
    fn load_u8(src: &[u8]) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let src = &src[..16];
            Self(_mm256_cvtepu8_epi16(_mm_loadu_si128(src.as_ptr().cast())))
        }
    }

    #[inline(always)]
    fn pair_sum_u8(src: &[u8]) -> Self {
        use core::arch::x86_64::{_mm256_loadu_si256, _mm256_maddubs_epi16, _mm256_set1_epi8};
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load.
        unsafe {
            let src = &src[..32];
            Self(_mm256_maddubs_epi16(
                _mm256_loadu_si256(src.as_ptr().cast()),
                _mm256_set1_epi8(1),
            ))
        }
    }

    #[inline(always)]
    fn load_i8(src: &[i8]) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let src = &src[..16];
            Self(core::arch::x86_64::_mm256_cvtepi8_epi16(_mm_loadu_si128(
                src.as_ptr().cast(),
            )))
        }
    }

    #[inline(always)]
    fn store_i8(self, dst: &mut [i8]) {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let dst = &mut dst[..16];
            let bytes = core::arch::x86_64::_mm_packs_epi16(
                _mm256_castsi256_si128(self.0),
                _mm256_extracti128_si256::<1>(self.0),
            );
            _mm_storeu_si128(dst.as_mut_ptr().cast(), bytes);
        }
    }

    #[inline(always)]
    fn load(src: &[i16]) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let src = &src[..16];
            Self(_mm256_loadu_si256(src.as_ptr().cast()))
        }
    }

    #[inline(always)]
    fn lookup(table: &[i16; LOOKUP_TABLE], indices: &[u8]) -> Self {
        let indices: &[u8; 16] = indices[..16].try_into().expect("sixteen indices");
        // Two 32-bit gathers at `table + 2 * index` (the padding entry keeps
        // the last one inside), the low halves sign-extended and packed.
        // SAFETY: AVX2 kernels only; every 4-byte read starts at an entry
        // below 256 of the 257-entry table.
        unsafe {
            let bytes = _mm_loadu_si128(indices.as_ptr().cast());
            // The gather reads unaligned 32-bit words; it needs no alignment.
            #[allow(
                clippy::cast_ptr_alignment,
                reason = "vpgatherdd has no alignment requirement"
            )]
            let base = table.as_ptr().cast::<i32>();
            let low = _mm256_i32gather_epi32::<2>(base, _mm256_cvtepu8_epi32(bytes));
            let high =
                _mm256_i32gather_epi32::<2>(base, _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(bytes)));
            let extend = |v: __m256i| _mm256_srai_epi32::<16>(_mm256_slli_epi32::<16>(v));
            // packs interleaves the 128-bit halves; the permute restores order.
            Self(_mm256_permute4x64_epi64::<0b11_01_10_00>(
                _mm256_packs_epi32(extend(low), extend(high)),
            ))
        }
    }

    #[inline(always)]
    fn lookup_u12(self, table: &[i16; HIGH_LOOKUP_TABLE]) -> Self {
        // SAFETY: AVX2 kernels only; `clamped` keeps every index in 0..=4095
        // inside the 4097-entry table.
        unsafe {
            let clamped = _mm256_min_epi16(
                _mm256_max_epi16(self.0, _mm256_setzero_si256()),
                _mm256_set1_epi16(4095),
            );
            #[allow(clippy::cast_ptr_alignment, reason = "vpgatherdd is unaligned")]
            let base = table.as_ptr().cast::<i32>();
            let lo_idx = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(clamped));
            let hi_idx = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(clamped));
            let low = _mm256_i32gather_epi32::<2>(base, lo_idx);
            let high = _mm256_i32gather_epi32::<2>(base, hi_idx);
            let low = _mm256_srai_epi32::<16>(_mm256_slli_epi32::<16>(low));
            let high = _mm256_srai_epi32::<16>(_mm256_slli_epi32::<16>(high));
            Self(_mm256_permute4x64_epi64::<0b11_01_10_00>(
                _mm256_packs_epi32(low, high),
            ))
        }
    }

    #[inline(always)]
    fn pair_avg_u16(pairs: &[u16]) -> Self {
        let pairs = &pairs[..32];
        // SAFETY: AVX2 kernels only; `pairs` holds 32 `u16` elements.
        unsafe {
            let p0 = _mm256_loadu_si256(pairs.as_ptr().cast());
            let p1 = _mm256_loadu_si256(pairs.as_ptr().add(16).cast());
            let mask = _mm256_set1_epi32(0xFFFF);
            let one = _mm256_set1_epi32(1);
            let avg8 = |p: __m256i| {
                let sum = _mm256_add_epi32(_mm256_and_si256(p, mask), _mm256_srli_epi32::<16>(p));
                _mm256_srli_epi32::<1>(_mm256_add_epi32(sum, one))
            };
            Self(_mm256_permute4x64_epi64::<0b11_01_10_00>(
                _mm256_packs_epi32(avg8(p0), avg8(p1)),
            ))
        }
    }

    #[inline(always)]
    fn store(self, dst: &mut [i16]) {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let dst = &mut dst[..16];
            _mm256_storeu_si256(dst.as_mut_ptr().cast(), self.0);
        }
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { Self(_mm256_add_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn add_sat(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs); register
        // operations only.
        unsafe { Self(_mm256_adds_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { Self(_mm256_sub_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { Self(_mm256_mullo_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn mul_q15_round(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm256_mulhrs_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn sra(self, bits: u32) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { Self(_mm256_sra_epi16(self.0, count(bits))) }
    }

    #[inline(always)]
    fn srl(self, bits: u32) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm256_srl_epi16(self.0, count(bits))) }
    }

    #[inline(always)]
    fn clamp_u8(self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe {
            Self(_mm256_max_epi16(
                _mm256_min_epi16(self.0, _mm256_set1_epi16(255)),
                _mm256_setzero_si256(),
            ))
        }
    }

    #[inline(always)]
    fn mul_add_u8_pair(a: &[u8], b: &[u8], wa: u8, wb: u8) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load.
        unsafe {
            let (a, b) = (&a[..16], &b[..16]);
            let (a, b) = (
                _mm_loadu_si128(a.as_ptr().cast()),
                _mm_loadu_si128(b.as_ptr().cast()),
            );
            let pairs = _mm256_set_m128i(_mm_unpackhi_epi8(a, b), _mm_unpacklo_epi8(a, b));
            let weights = _mm256_set1_epi16(i16::from_le_bytes([wa, wb]));
            Self(_mm256_maddubs_epi16(pairs, weights))
        }
    }

    #[inline(always)]
    fn store_u8_sat(self, dst: &mut [u8]) {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let dst = &mut dst[..16];
            let packed = _mm_packus_epi16(
                _mm256_castsi256_si128(self.0),
                _mm256_extracti128_si256::<1>(self.0),
            );
            _mm_storeu_si128(dst.as_mut_ptr().cast(), packed);
        }
    }

    #[inline(always)]
    fn acc_zero() -> Self::Acc {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { [_mm256_setzero_si256(); 2] }
    }

    #[inline(always)]
    fn acc_mul_add(acc: Self::Acc, a: Self, coeff: i16) -> Self::Acc {
        Self::acc_mul_add2(acc, a, coeff, Self::splat(0), 0)
    }

    /// `pmaddwd` on `a` and `b` interleaved. The accumulators hold lanes in
    /// unpack order (`acc[0]`: 0-3 and 8-11, `acc[1]`: 4-7 and 12-15), which
    /// `packssdw` turns back into lane order.
    #[inline(always)]
    fn acc_mul_add2(acc: Self::Acc, a: Self, ca: i16, b: Self, cb: i16) -> Self::Acc {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe {
            let taps = _mm256_set1_epi32(i32::from(ca as u16) | (i32::from(cb) << 16));
            [
                _mm256_add_epi32(
                    acc[0],
                    _mm256_madd_epi16(_mm256_unpacklo_epi16(a.0, b.0), taps),
                ),
                _mm256_add_epi32(
                    acc[1],
                    _mm256_madd_epi16(_mm256_unpackhi_epi16(a.0, b.0), taps),
                ),
            ]
        }
    }

    #[inline(always)]
    fn acc_vec_mul_add2(a: Self, wa: Self, b: Self, wb: Self) -> Self::Acc {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe {
            [
                _mm256_madd_epi16(
                    _mm256_unpacklo_epi16(a.0, b.0),
                    _mm256_unpacklo_epi16(wa.0, wb.0),
                ),
                _mm256_madd_epi16(
                    _mm256_unpackhi_epi16(a.0, b.0),
                    _mm256_unpackhi_epi16(wa.0, wb.0),
                ),
            ]
        }
    }

    #[inline(always)]
    fn acc_vec_fma2(acc: Self::Acc, a: Self, wa: Self, b: Self, wb: Self) -> Self::Acc {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe {
            let prod = Self::acc_vec_mul_add2(a, wa, b, wb);
            [
                _mm256_add_epi32(acc[0], prod[0]),
                _mm256_add_epi32(acc[1], prod[1]),
            ]
        }
    }

    #[inline(always)]
    fn acc_add(acc: Self::Acc, a: Self) -> Self::Acc {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            [
                _mm256_add_epi32(acc[0], _mm256_madd_epi16(a.0, _mm256_set1_epi16(1))),
                acc[1],
            ]
        }
    }

    #[inline(always)]
    fn acc_add_sq(acc: Self::Acc, a: Self) -> Self::Acc {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            [
                _mm256_add_epi32(acc[0], _mm256_madd_epi16(a.0, a.0)),
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
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe {
            let (r, n) = (_mm256_set1_epi32(round), count(shift));
            let lo = _mm256_sra_epi32(_mm256_add_epi32(acc[0], r), n);
            let hi = _mm256_sra_epi32(_mm256_add_epi32(acc[1], r), n);
            // Unpack order in, lane order out (see `acc_mul_add2`).
            Self(_mm256_packs_epi32(lo, hi))
        }
    }

    #[inline(always)]
    fn acc_sum(acc: Self::Acc) -> i64 {
        sum_i32x8(acc[0]) + sum_i32x8(acc[1])
    }
}

// ---------------------------------------------------------------- I16x8 ----

impl I16Lanes for I16x8 {
    const LANES: usize = 8;
    type Acc = [__m128i; 2];

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm_min_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm_max_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm_abs_epi16(self.0)) }
    }

    #[inline(always)]
    fn with_sign_of(self, like: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm_sign_epi16(self.0, like.0)) }
    }

    #[inline(always)]
    fn splat(value: i16) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { Self(_mm_set1_epi16(value)) }
    }

    #[inline(always)]
    fn load_u8(src: &[u8]) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let src = &src[..8];
            Self(_mm_cvtepu8_epi16(_mm_loadl_epi64(src.as_ptr().cast())))
        }
    }

    #[inline(always)]
    fn pair_sum_u8(src: &[u8]) -> Self {
        use core::arch::x86_64::{_mm_loadu_si128, _mm_maddubs_epi16, _mm_set1_epi8};
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load.
        unsafe {
            let src = &src[..16];
            Self(_mm_maddubs_epi16(
                _mm_loadu_si128(src.as_ptr().cast()),
                _mm_set1_epi8(1),
            ))
        }
    }

    #[inline(always)]
    fn load(src: &[i16]) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let src = &src[..8];
            Self(_mm_loadu_si128(src.as_ptr().cast()))
        }
    }

    #[inline(always)]
    fn store(self, dst: &mut [i16]) {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let dst = &mut dst[..8];
            _mm_storeu_si128(dst.as_mut_ptr().cast(), self.0);
        }
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { Self(_mm_add_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn add_sat(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs); register
        // operations only.
        unsafe { Self(_mm_adds_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { Self(_mm_sub_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { Self(_mm_mullo_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn mul_q15_round(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm_mulhrs_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn sra(self, bits: u32) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { Self(_mm_sra_epi16(self.0, count(bits))) }
    }

    #[inline(always)]
    fn srl(self, bits: u32) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm_srl_epi16(self.0, count(bits))) }
    }

    #[inline(always)]
    fn clamp_u8(self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe {
            Self(_mm_max_epi16(
                _mm_min_epi16(self.0, _mm_set1_epi16(255)),
                _mm_setzero_si128(),
            ))
        }
    }

    #[inline(always)]
    fn mul_add_u8_pair(a: &[u8], b: &[u8], wa: u8, wb: u8) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load.
        unsafe {
            let (a, b) = (&a[..8], &b[..8]);
            let pairs = _mm_unpacklo_epi8(
                _mm_loadl_epi64(a.as_ptr().cast()),
                _mm_loadl_epi64(b.as_ptr().cast()),
            );
            let weights = _mm_set1_epi16(i16::from_le_bytes([wa, wb]));
            Self(_mm_maddubs_epi16(pairs, weights))
        }
    }

    #[inline(always)]
    fn store_u8_sat(self, dst: &mut [u8]) {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let dst = &mut dst[..8];
            let packed = _mm_packus_epi16(self.0, _mm_setzero_si128());
            _mm_storel_epi64(dst.as_mut_ptr().cast(), packed);
        }
    }

    #[inline(always)]
    fn acc_zero() -> Self::Acc {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { [_mm_setzero_si128(); 2] }
    }

    #[inline(always)]
    fn acc_mul_add(acc: Self::Acc, a: Self, coeff: i16) -> Self::Acc {
        Self::acc_mul_add2(acc, a, coeff, Self::splat(0), 0)
    }

    /// `pmaddwd` on `a` and `b` interleaved (`acc[0]`: lanes 0-3, `acc[1]`:
    /// lanes 4-7).
    #[inline(always)]
    fn acc_mul_add2(acc: Self::Acc, a: Self, ca: i16, b: Self, cb: i16) -> Self::Acc {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe {
            let taps = _mm_set1_epi32(i32::from(ca as u16) | (i32::from(cb) << 16));
            [
                _mm_add_epi32(acc[0], _mm_madd_epi16(_mm_unpacklo_epi16(a.0, b.0), taps)),
                _mm_add_epi32(acc[1], _mm_madd_epi16(_mm_unpackhi_epi16(a.0, b.0), taps)),
            ]
        }
    }

    #[inline(always)]
    fn acc_add(acc: Self::Acc, a: Self) -> Self::Acc {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            [
                _mm_add_epi32(acc[0], _mm_madd_epi16(a.0, _mm_set1_epi16(1))),
                acc[1],
            ]
        }
    }

    #[inline(always)]
    fn acc_vec_mul_add2(a: Self, wa: Self, b: Self, wb: Self) -> Self::Acc {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe {
            [
                _mm_madd_epi16(_mm_unpacklo_epi16(a.0, b.0), _mm_unpacklo_epi16(wa.0, wb.0)),
                _mm_madd_epi16(_mm_unpackhi_epi16(a.0, b.0), _mm_unpackhi_epi16(wa.0, wb.0)),
            ]
        }
    }

    #[inline(always)]
    fn acc_vec_fma2(acc: Self::Acc, a: Self, wa: Self, b: Self, wb: Self) -> Self::Acc {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe {
            let prod = Self::acc_vec_mul_add2(a, wa, b, wb);
            [
                _mm_add_epi32(acc[0], prod[0]),
                _mm_add_epi32(acc[1], prod[1]),
            ]
        }
    }

    #[inline(always)]
    fn acc_add_sq(acc: Self::Acc, a: Self) -> Self::Acc {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { [_mm_add_epi32(acc[0], _mm_madd_epi16(a.0, a.0)), acc[1]] }
    }

    #[inline(always)]
    fn acc_round_shift_store_u8(acc: Self::Acc, round: i32, shift: u32, dst: &mut [u8]) {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let dst = &mut dst[..8];
            let words = _mm_packs_epi32(
                round_shift_x4(acc[0], round, shift),
                round_shift_x4(acc[1], round, shift),
            );
            let packed = _mm_packus_epi16(words, _mm_setzero_si128());
            _mm_storel_epi64(dst.as_mut_ptr().cast(), packed);
        }
    }

    #[inline(always)]
    fn acc_round_shift_narrow(acc: Self::Acc, round: i32, shift: u32) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe {
            Self(_mm_packs_epi32(
                round_shift_x4(acc[0], round, shift),
                round_shift_x4(acc[1], round, shift),
            ))
        }
    }

    #[inline(always)]
    fn acc_sum(acc: Self::Acc) -> i64 {
        sum_i32x4(acc[0]) + sum_i32x4(acc[1])
    }
}

// ---------------------------------------------------------------- I16x4 ----

impl I16Lanes for I16x4 {
    const LANES: usize = 4;
    type Acc = __m128i;

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm_min_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm_max_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm_abs_epi16(self.0)) }
    }

    #[inline(always)]
    fn with_sign_of(self, like: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm_sign_epi16(self.0, like.0)) }
    }

    #[inline(always)]
    fn splat(value: i16) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            Self(_mm_unpacklo_epi64(
                _mm_set1_epi16(value),
                _mm_setzero_si128(),
            ))
        }
    }

    #[inline(always)]
    fn load_u8(src: &[u8]) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let bytes = i32::from_le_bytes(src[..4].try_into().expect("4 bytes"));
            Self(_mm_cvtepu8_epi16(_mm_cvtsi32_si128(bytes)))
        }
    }

    #[inline(always)]
    fn load(src: &[i16]) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let src = &src[..4];
            Self(_mm_loadl_epi64(src.as_ptr().cast()))
        }
    }

    #[inline(always)]
    fn store(self, dst: &mut [i16]) {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let dst = &mut dst[..4];
            _mm_storel_epi64(dst.as_mut_ptr().cast(), self.0);
        }
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { Self(_mm_add_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn add_sat(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs); register
        // operations only.
        unsafe { Self(_mm_adds_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { Self(_mm_sub_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { Self(_mm_mullo_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn mul_q15_round(self, other: Self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm_mulhrs_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn sra(self, bits: u32) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { Self(_mm_sra_epi16(self.0, count(bits))) }
    }

    #[inline(always)]
    fn srl(self, bits: u32) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { Self(_mm_srl_epi16(self.0, count(bits))) }
    }

    #[inline(always)]
    fn clamp_u8(self) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe {
            Self(_mm_max_epi16(
                _mm_min_epi16(self.0, _mm_set1_epi16(255)),
                _mm_setzero_si128(),
            ))
        }
    }

    #[inline(always)]
    fn store_u8_sat(self, dst: &mut [u8]) {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let packed = _mm_packus_epi16(self.0, _mm_setzero_si128());
            dst[..4].copy_from_slice(&_mm_cvtsi128_si32(packed).to_le_bytes());
        }
    }

    #[inline(always)]
    fn acc_zero() -> Self::Acc {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { _mm_setzero_si128() }
    }

    #[inline(always)]
    fn acc_mul_add(acc: Self::Acc, a: Self, coeff: i16) -> Self::Acc {
        Self::acc_mul_add2(acc, a, coeff, Self::splat(0), 0)
    }

    /// `pmaddwd` on the low four lanes of `a` and `b` interleaved.
    #[inline(always)]
    fn acc_mul_add2(acc: Self::Acc, a: Self, ca: i16, b: Self, cb: i16) -> Self::Acc {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe {
            let taps = _mm_set1_epi32(i32::from(ca as u16) | (i32::from(cb) << 16));
            _mm_add_epi32(acc, _mm_madd_epi16(_mm_unpacklo_epi16(a.0, b.0), taps))
        }
    }

    #[inline(always)]
    fn acc_vec_mul_add2(a: Self, wa: Self, b: Self, wb: Self) -> Self::Acc {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { _mm_madd_epi16(_mm_unpacklo_epi16(a.0, b.0), _mm_unpacklo_epi16(wa.0, wb.0)) }
    }

    #[inline(always)]
    fn acc_vec_fma2(acc: Self::Acc, a: Self, wa: Self, b: Self, wb: Self) -> Self::Acc {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe { _mm_add_epi32(acc, Self::acc_vec_mul_add2(a, wa, b, wb)) }
    }

    #[inline(always)]
    fn acc_add(acc: Self::Acc, a: Self) -> Self::Acc {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { _mm_add_epi32(acc, _mm_madd_epi16(a.0, _mm_set1_epi16(1))) }
    }

    #[inline(always)]
    fn acc_add_sq(acc: Self::Acc, a: Self) -> Self::Acc {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe { _mm_add_epi32(acc, _mm_madd_epi16(a.0, a.0)) }
    }

    #[inline(always)]
    fn acc_round_shift_store_u8(acc: Self::Acc, round: i32, shift: u32, dst: &mut [u8]) {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let v = round_shift_x4(acc, round, shift);
            let packed = _mm_packus_epi16(_mm_packs_epi32(v, v), _mm_setzero_si128());
            dst[..4].copy_from_slice(&_mm_cvtsi128_si32(packed).to_le_bytes());
        }
    }

    #[inline(always)]
    fn acc_round_shift_narrow(acc: Self::Acc, round: i32, shift: u32) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs).
        unsafe {
            Self(_mm_packs_epi32(
                round_shift_x4(acc, round, shift),
                _mm_setzero_si128(),
            ))
        }
    }

    #[inline(always)]
    fn acc_sum(acc: Self::Acc) -> i64 {
        sum_i32x4(acc)
    }
}

// ------------------------------------------------------- filter lanes ----

impl I16FilterLanes for I16x16 {
    #[inline(always)]
    fn load_halves(lo: &[i16], hi: &[i16]) -> Self {
        let (lo, hi) = (&lo[..8], &hi[..8]);
        // SAFETY: AVX2 kernels only (module docs); `lo` and `hi` hold 8
        // values each.
        unsafe {
            Self(_mm256_set_m128i(
                _mm_loadu_si128(hi.as_ptr().cast()),
                _mm_loadu_si128(lo.as_ptr().cast()),
            ))
        }
    }

    #[inline(always)]
    fn load_u8_halves(lo: &[u8], hi: &[u8]) -> Self {
        let (lo, hi) = (&lo[..8], &hi[..8]);
        // SAFETY: AVX2 kernels only (module docs); `lo` and `hi` hold 8 bytes each.
        unsafe {
            Self(_mm256_cvtepu8_epi16(_mm_unpacklo_epi64(
                _mm_loadl_epi64(lo.as_ptr().cast()),
                _mm_loadl_epi64(hi.as_ptr().cast()),
            )))
        }
    }

    #[inline(always)]
    fn store_u8_sat_halves(self, lo: &mut [u8], hi: &mut [u8]) {
        let (lo, hi) = (&mut lo[..8], &mut hi[..8]);
        // SAFETY: AVX2 kernels only (module docs); `lo` and `hi` hold 8
        // bytes each.
        unsafe {
            let packed = _mm_packus_epi16(
                _mm256_castsi256_si128(self.0),
                _mm256_extracti128_si256::<1>(self.0),
            );
            _mm_storel_epi64(lo.as_mut_ptr().cast(), packed);
            _mm_storel_epi64(hi.as_mut_ptr().cast(), _mm_unpackhi_epi64(packed, packed));
        }
    }

    #[inline(always)]
    fn store_halves(self, lo: &mut [i16], hi: &mut [i16]) {
        let (lo, hi) = (&mut lo[..8], &mut hi[..8]);
        // SAFETY: AVX2 kernels only; `lo` and `hi` hold 8 words each.
        unsafe {
            _mm_storeu_si128(lo.as_mut_ptr().cast(), _mm256_castsi256_si128(self.0));
            _mm_storeu_si128(
                hi.as_mut_ptr().cast(),
                _mm256_extracti128_si256::<1>(self.0),
            );
        }
    }

    #[inline(always)]
    fn combine_hi_lo(self, next: Self) -> Self {
        // SAFETY: AVX2 kernels only (module docs).
        unsafe { Self(_mm256_permute2x128_si256::<0x21>(self.0, next.0)) }
    }

    #[inline(always)]
    fn transpose_groups8(rows: [Self; 8]) -> [Self; 8] {
        // SAFETY: AVX2 kernels only; in-lane unpacks transpose both 128-bit 8x8 tiles.
        unsafe {
            let [r0, r1, r2, r3, r4, r5, r6, r7] = rows;
            let (p0, p1) = (
                _mm256_unpacklo_epi16(r0.0, r1.0),
                _mm256_unpackhi_epi16(r0.0, r1.0),
            );
            let (p2, p3) = (
                _mm256_unpacklo_epi16(r2.0, r3.0),
                _mm256_unpackhi_epi16(r2.0, r3.0),
            );
            let (p4, p5) = (
                _mm256_unpacklo_epi16(r4.0, r5.0),
                _mm256_unpackhi_epi16(r4.0, r5.0),
            );
            let (p6, p7) = (
                _mm256_unpacklo_epi16(r6.0, r7.0),
                _mm256_unpackhi_epi16(r6.0, r7.0),
            );
            let (q0, q1) = (_mm256_unpacklo_epi32(p0, p2), _mm256_unpackhi_epi32(p0, p2));
            let (q2, q3) = (_mm256_unpacklo_epi32(p1, p3), _mm256_unpackhi_epi32(p1, p3));
            let (q4, q5) = (_mm256_unpacklo_epi32(p4, p6), _mm256_unpackhi_epi32(p4, p6));
            let (q6, q7) = (_mm256_unpacklo_epi32(p5, p7), _mm256_unpackhi_epi32(p5, p7));
            [
                Self(_mm256_unpacklo_epi64(q0, q4)),
                Self(_mm256_unpackhi_epi64(q0, q4)),
                Self(_mm256_unpacklo_epi64(q1, q5)),
                Self(_mm256_unpackhi_epi64(q1, q5)),
                Self(_mm256_unpacklo_epi64(q2, q6)),
                Self(_mm256_unpackhi_epi64(q2, q6)),
                Self(_mm256_unpacklo_epi64(q3, q7)),
                Self(_mm256_unpackhi_epi64(q3, q7)),
            ]
        }
    }

    #[inline(always)]
    fn load_quarters(rows: [&[i16]; 4]) -> Self {
        let r0 = &rows[0][..4];
        let r1 = &rows[1][..4];
        let r2 = &rows[2][..4];
        let r3 = &rows[3][..4];
        // SAFETY: AVX2 kernels only; each row slice holds 4 words (8 bytes).
        unsafe {
            let lo = _mm_unpacklo_epi64(
                _mm_loadl_epi64(r0.as_ptr().cast()),
                _mm_loadl_epi64(r1.as_ptr().cast()),
            );
            let hi = _mm_unpacklo_epi64(
                _mm_loadl_epi64(r2.as_ptr().cast()),
                _mm_loadl_epi64(r3.as_ptr().cast()),
            );
            Self(_mm256_set_m128i(hi, lo))
        }
    }

    #[inline(always)]
    fn load_u8_quarters(rows: [&[u8]; 4]) -> Self {
        let r0 = i32::from_le_bytes(rows[0][..4].try_into().unwrap());
        let r1 = i32::from_le_bytes(rows[1][..4].try_into().unwrap());
        let r2 = i32::from_le_bytes(rows[2][..4].try_into().unwrap());
        let r3 = i32::from_le_bytes(rows[3][..4].try_into().unwrap());
        // SAFETY: AVX2 kernels only (module docs).
        unsafe { Self(_mm256_cvtepu8_epi16(_mm_setr_epi32(r0, r1, r2, r3))) }
    }

    #[inline(always)]
    fn store_quarters(self, rows: [&mut [i16]; 4]) {
        let [r0, r1, r2, r3] = rows;
        let (r0, r1) = (&mut r0[..4], &mut r1[..4]);
        let (r2, r3) = (&mut r2[..4], &mut r3[..4]);
        // SAFETY: AVX2 kernels only; each row slice holds 4 words (8 bytes).
        unsafe {
            let lo = _mm256_castsi256_si128(self.0);
            let hi = _mm256_extracti128_si256::<1>(self.0);
            _mm_storel_epi64(r0.as_mut_ptr().cast(), lo);
            _mm_storel_epi64(r1.as_mut_ptr().cast(), _mm_unpackhi_epi64(lo, lo));
            _mm_storel_epi64(r2.as_mut_ptr().cast(), hi);
            _mm_storel_epi64(r3.as_mut_ptr().cast(), _mm_unpackhi_epi64(hi, hi));
        }
    }

    #[inline(always)]
    fn min_u16(self, other: Self) -> Self {
        // SAFETY: AVX2 kernels only (module docs).
        unsafe { Self(_mm256_min_epu16(self.0, other.0)) }
    }

    #[inline(always)]
    fn sub_sat_u16(self, other: Self) -> Self {
        // SAFETY: AVX2 kernels only (module docs).
        unsafe { Self(_mm256_subs_epu16(self.0, other.0)) }
    }

    #[inline(always)]
    fn eq_mask(self, other: Self) -> Self {
        // SAFETY: AVX2 kernels only (module docs).
        unsafe { Self(_mm256_cmpeq_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn and(self, other: Self) -> Self {
        // SAFETY: AVX2 kernels only (module docs).
        unsafe { Self(_mm256_and_si256(self.0, other.0)) }
    }

    #[inline(always)]
    fn gt_mask(self, other: Self) -> Self {
        // SAFETY: AVX2 kernels only (module docs).
        unsafe { Self(_mm256_cmpgt_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn blend(self, if_set: Self, if_clear: Self) -> Self {
        // SAFETY: AVX2 kernels only (module docs). Mask lanes are all ones
        // or all zeros, so a byte blend is a lane blend.
        unsafe { Self(_mm256_blendv_epi8(if_clear.0, if_set.0, self.0)) }
    }

    #[inline(always)]
    fn any_nonzero(self) -> bool {
        // SAFETY: AVX2 kernels only (module docs).
        unsafe { _mm256_testz_si256(self.0, self.0) == 0 }
    }

    #[inline(always)]
    fn load_u8_deinterleaved(src: &[u8]) -> (Self, Self) {
        let src = &src[..32];
        // SAFETY: AVX2 kernels only; `src` holds 32 bytes and `_mm256_loadu_si256`
        // accepts 1-byte alignment.
        unsafe {
            let raw = Self(_mm256_loadu_si256(src.as_ptr().cast()));
            (raw.and(Self::splat(0x00ff)), raw.srl(8))
        }
    }

    #[inline(always)]
    fn load_u16_deinterleaved(src: &[u16]) -> (Self, Self) {
        let src = &src[..32];
        // SAFETY: AVX2 kernels only; `src` holds 32 `u16` elements (64 bytes).
        unsafe {
            let v0 = _mm256_loadu_si256(src.as_ptr().cast());
            let v1 = _mm256_loadu_si256(src.as_ptr().add(16).cast());
            let shuf = _mm256_setr_epi8(
                0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, 0, 1, 4, 5, 8, 9, 12, 13, 2,
                3, 6, 7, 10, 11, 14, 15,
            );
            let s0 = _mm256_shuffle_epi8(v0, shuf);
            let s1 = _mm256_shuffle_epi8(v1, shuf);
            let even = _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_unpacklo_epi64(s0, s1));
            let odd = _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_unpackhi_epi64(s0, s1));
            (Self(even), Self(odd))
        }
    }

    #[inline(always)]
    fn combine_mid_halves(self, next: Self) -> Self {
        // SAFETY: AVX2 kernels only (module docs).
        unsafe { Self(_mm256_permute2x128_si256::<0x21>(self.0, next.0)) }
    }

    #[inline(always)]
    fn shift_quarter(self, next: Self) -> Self {
        // SAFETY: AVX2 kernels only (module docs).
        unsafe {
            let mid = _mm256_permute2x128_si256::<0x21>(self.0, next.0);
            Self(_mm256_alignr_epi8::<8>(mid, self.0))
        }
    }

    #[inline(always)]
    fn expand_alf_params(blocks: &[[i16; 12]]) -> [[Self; 2]; 6] {
        let b = &blocks[..4];
        // `_mm256_unpacklo_epi16` places block 0 in lane 0 and block 2 in lane 1;
        // `_mm256_unpackhi_epi16` places block 1 in lane 0 and block 3 in lane 1
        // (matching VVdeC `simdFilter7x7Blk<AVX2>`).
        // SAFETY: AVX2 kernels only; each block has 12 `i16` elements (24 bytes).
        unsafe {
            let lo0 = _mm256_set_m128i(
                _mm_loadu_si128(b[2].as_ptr().cast()),
                _mm_loadu_si128(b[0].as_ptr().cast()),
            );
            let hi0 = _mm256_set_m128i(
                _mm_loadl_epi64(b[2].as_ptr().add(8).cast()),
                _mm_loadl_epi64(b[0].as_ptr().add(8).cast()),
            );
            let lo1 = _mm256_set_m128i(
                _mm_loadu_si128(b[3].as_ptr().cast()),
                _mm_loadu_si128(b[1].as_ptr().cast()),
            );
            let hi1 = _mm256_set_m128i(
                _mm_loadl_epi64(b[3].as_ptr().add(8).cast()),
                _mm_loadl_epi64(b[1].as_ptr().add(8).cast()),
            );
            [
                [
                    Self(_mm256_shuffle_epi32::<0x00>(lo0)),
                    Self(_mm256_shuffle_epi32::<0x00>(lo1)),
                ],
                [
                    Self(_mm256_shuffle_epi32::<0x55>(lo0)),
                    Self(_mm256_shuffle_epi32::<0x55>(lo1)),
                ],
                [
                    Self(_mm256_shuffle_epi32::<0xAA>(lo0)),
                    Self(_mm256_shuffle_epi32::<0xAA>(lo1)),
                ],
                [
                    Self(_mm256_shuffle_epi32::<0xFF>(lo0)),
                    Self(_mm256_shuffle_epi32::<0xFF>(lo1)),
                ],
                [
                    Self(_mm256_shuffle_epi32::<0x00>(hi0)),
                    Self(_mm256_shuffle_epi32::<0x00>(hi1)),
                ],
                [
                    Self(_mm256_shuffle_epi32::<0x55>(hi0)),
                    Self(_mm256_shuffle_epi32::<0x55>(hi1)),
                ],
            ]
        }
    }

    #[inline(always)]
    fn splat_alf_pair(t0: i16, t1: i16) -> [Self; 2] {
        // SAFETY: AVX2 kernels only (module docs).
        let v = unsafe { Self(_mm256_set1_epi32(packed_pair(t0, t1))) };
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
        // SAFETY: AVX2 kernels only (module docs).
        unsafe {
            let v00 = _mm256_sub_epi16(a0.0, centre.0);
            let v10 = _mm256_sub_epi16(a1.0, centre.0);
            let v01 = _mm256_sub_epi16(b0.0, centre.0);
            let v11 = _mm256_sub_epi16(b1.0, centre.0);
            let (mut u0, mut u1) = (
                _mm256_unpacklo_epi16(v00, v10),
                _mm256_unpackhi_epi16(v00, v10),
            );
            let (mut u2, mut u3) = (
                _mm256_unpacklo_epi16(v01, v11),
                _mm256_unpackhi_epi16(v01, v11),
            );
            let zero = _mm256_setzero_si256();
            let (neg0, neg1) = (
                _mm256_sub_epi16(zero, lim[0].0),
                _mm256_sub_epi16(zero, lim[1].0),
            );
            u0 = _mm256_max_epi16(_mm256_min_epi16(u0, lim[0].0), neg0);
            u1 = _mm256_max_epi16(_mm256_min_epi16(u1, lim[1].0), neg1);
            u2 = _mm256_max_epi16(_mm256_min_epi16(u2, lim[0].0), neg0);
            u3 = _mm256_max_epi16(_mm256_min_epi16(u3, lim[1].0), neg1);
            let sum_lo = _mm256_add_epi16(u0, u2);
            let sum_hi = _mm256_add_epi16(u1, u3);
            [
                _mm256_add_epi32(acc[0], _mm256_madd_epi16(sum_lo, coeff[0].0)),
                _mm256_add_epi32(acc[1], _mm256_madd_epi16(sum_hi, coeff[1].0)),
            ]
        }
    }

    #[inline(always)]
    fn acc_alf_linear_pair(acc: Self::Acc, d0: Self, d1: Self, coeff: [Self; 2]) -> Self::Acc {
        // SAFETY: AVX2 kernels only (module docs).
        unsafe {
            let lo = _mm256_unpacklo_epi16(d0.0, d1.0);
            let hi = _mm256_unpackhi_epi16(d0.0, d1.0);
            [
                _mm256_add_epi32(acc[0], _mm256_madd_epi16(lo, coeff[0].0)),
                _mm256_add_epi32(acc[1], _mm256_madd_epi16(hi, coeff[1].0)),
            ]
        }
    }

    #[inline(always)]
    fn expand_alf_pair(blocks: &[[i16; 12]], p: usize) -> [Self; 2] {
        let b = &blocks[..4];
        let t = 2 * p;
        assert!(t + 1 < 12);
        // SAFETY: AVX2 kernels only; `t + 1 < 12` checked above so reading 4 bytes at `t` is in-bounds.
        unsafe {
            let p0 = b[0].as_ptr().add(t).cast::<i32>().read_unaligned();
            let p1 = b[1].as_ptr().add(t).cast::<i32>().read_unaligned();
            let p2 = b[2].as_ptr().add(t).cast::<i32>().read_unaligned();
            let p3 = b[3].as_ptr().add(t).cast::<i32>().read_unaligned();
            let wide = _mm256_cvtepu32_epi64(_mm_setr_epi32(p0, p1, p2, p3));
            let shuf0 = _mm256_setr_epi8(
                0, 1, 0, 1, 0, 1, 0, 1, 8, 9, 8, 9, 8, 9, 8, 9, 0, 1, 0, 1, 0, 1, 0, 1, 8, 9, 8, 9,
                8, 9, 8, 9,
            );
            let shuf1 = _mm256_setr_epi8(
                2, 3, 2, 3, 2, 3, 2, 3, 10, 11, 10, 11, 10, 11, 10, 11, 2, 3, 2, 3, 2, 3, 2, 3, 10,
                11, 10, 11, 10, 11, 10, 11,
            );
            [
                Self(_mm256_shuffle_epi8(wide, shuf0)),
                Self(_mm256_shuffle_epi8(wide, shuf1)),
            ]
        }
    }
}

impl I16FilterLanes for I16x8 {
    #[inline(always)]
    fn load_halves(lo: &[i16], hi: &[i16]) -> Self {
        let (lo, hi) = (&lo[..4], &hi[..4]);
        // SAFETY: AVX2 kernels only (module docs); `lo` and `hi` hold 4
        // values (8 bytes) each.
        unsafe {
            Self(_mm_unpacklo_epi64(
                _mm_loadl_epi64(lo.as_ptr().cast()),
                _mm_loadl_epi64(hi.as_ptr().cast()),
            ))
        }
    }

    #[inline(always)]
    fn load_u8_halves(lo: &[u8], hi: &[u8]) -> Self {
        let lo = i32::from_le_bytes(lo[..4].try_into().unwrap());
        let hi = i32::from_le_bytes(hi[..4].try_into().unwrap());
        // SAFETY: AVX2 kernels only (module docs).
        unsafe { Self(_mm_cvtepu8_epi16(_mm_setr_epi32(lo, hi, 0, 0))) }
    }

    #[inline(always)]
    fn store_u8_sat_halves(self, lo: &mut [u8], hi: &mut [u8]) {
        // SAFETY: AVX2 kernels only (module docs).
        let (lo_bytes, hi_bytes) = unsafe {
            let packed = _mm_packus_epi16(self.0, self.0);
            (_mm_cvtsi128_si32(packed), _mm_extract_epi32::<1>(packed))
        };
        lo[..4].copy_from_slice(&lo_bytes.to_le_bytes());
        hi[..4].copy_from_slice(&hi_bytes.to_le_bytes());
    }

    #[inline(always)]
    fn store_halves(self, lo: &mut [i16], hi: &mut [i16]) {
        let (lo, hi) = (&mut lo[..4], &mut hi[..4]);
        // SAFETY: AVX2 kernels only; `lo` and `hi` hold 4 words (8 bytes) each.
        unsafe {
            _mm_storel_epi64(lo.as_mut_ptr().cast(), self.0);
            _mm_storel_epi64(hi.as_mut_ptr().cast(), _mm_unpackhi_epi64(self.0, self.0));
        }
    }

    #[inline(always)]
    fn combine_hi_lo(self, next: Self) -> Self {
        // SAFETY: AVX2 kernels only (module docs).
        unsafe { Self(_mm_alignr_epi8::<8>(next.0, self.0)) }
    }

    #[inline(always)]
    fn min_u16(self, other: Self) -> Self {
        // SAFETY: AVX2 kernels only (module docs).
        unsafe { Self(_mm_min_epu16(self.0, other.0)) }
    }

    #[inline(always)]
    fn sub_sat_u16(self, other: Self) -> Self {
        // SAFETY: AVX2 kernels only (module docs).
        unsafe { Self(_mm_subs_epu16(self.0, other.0)) }
    }

    #[inline(always)]
    fn eq_mask(self, other: Self) -> Self {
        // SAFETY: AVX2 kernels only (module docs).
        unsafe { Self(_mm_cmpeq_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn and(self, other: Self) -> Self {
        // SAFETY: AVX2 kernels only (module docs).
        unsafe { Self(_mm_and_si128(self.0, other.0)) }
    }

    #[inline(always)]
    fn gt_mask(self, other: Self) -> Self {
        // SAFETY: AVX2 kernels only (module docs).
        unsafe { Self(_mm_cmpgt_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn blend(self, if_set: Self, if_clear: Self) -> Self {
        // SAFETY: AVX2 kernels only (module docs). Mask lanes are all ones
        // or all zeros, so a byte blend is a lane blend.
        unsafe { Self(_mm_blendv_epi8(if_clear.0, if_set.0, self.0)) }
    }

    #[inline(always)]
    fn combine_mid_halves(self, next: Self) -> Self {
        // SAFETY: AVX2 kernels only (module docs).
        unsafe { Self(_mm_alignr_epi8::<8>(next.0, self.0)) }
    }

    #[inline(always)]
    fn shift_quarter(self, next: Self) -> Self {
        // SAFETY: AVX2 kernels only (module docs).
        unsafe { Self(_mm_alignr_epi8::<4>(next.0, self.0)) }
    }

    #[inline(always)]
    fn expand_alf_params(blocks: &[[i16; 12]]) -> [[Self; 2]; 6] {
        let b = &blocks[..2];
        // SAFETY: AVX2 kernels only; each block has 12 `i16` elements (24 bytes).
        unsafe {
            let lo0 = _mm_loadu_si128(b[0].as_ptr().cast());
            let hi0 = _mm_loadl_epi64(b[0].as_ptr().add(8).cast());
            let lo1 = _mm_loadu_si128(b[1].as_ptr().cast());
            let hi1 = _mm_loadl_epi64(b[1].as_ptr().add(8).cast());
            [
                [
                    Self(_mm_shuffle_epi32::<0x00>(lo0)),
                    Self(_mm_shuffle_epi32::<0x00>(lo1)),
                ],
                [
                    Self(_mm_shuffle_epi32::<0x55>(lo0)),
                    Self(_mm_shuffle_epi32::<0x55>(lo1)),
                ],
                [
                    Self(_mm_shuffle_epi32::<0xAA>(lo0)),
                    Self(_mm_shuffle_epi32::<0xAA>(lo1)),
                ],
                [
                    Self(_mm_shuffle_epi32::<0xFF>(lo0)),
                    Self(_mm_shuffle_epi32::<0xFF>(lo1)),
                ],
                [
                    Self(_mm_shuffle_epi32::<0x00>(hi0)),
                    Self(_mm_shuffle_epi32::<0x00>(hi1)),
                ],
                [
                    Self(_mm_shuffle_epi32::<0x55>(hi0)),
                    Self(_mm_shuffle_epi32::<0x55>(hi1)),
                ],
            ]
        }
    }

    #[inline(always)]
    fn splat_alf_pair(t0: i16, t1: i16) -> [Self; 2] {
        // SAFETY: AVX2 kernels only (module docs).
        let v = unsafe { Self(_mm_set1_epi32(packed_pair(t0, t1))) };
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
        // SAFETY: AVX2 kernels only (module docs).
        unsafe {
            let v00 = _mm_sub_epi16(a0.0, centre.0);
            let v10 = _mm_sub_epi16(a1.0, centre.0);
            let v01 = _mm_sub_epi16(b0.0, centre.0);
            let v11 = _mm_sub_epi16(b1.0, centre.0);
            let (mut u0, mut u1) = (_mm_unpacklo_epi16(v00, v10), _mm_unpackhi_epi16(v00, v10));
            let (mut u2, mut u3) = (_mm_unpacklo_epi16(v01, v11), _mm_unpackhi_epi16(v01, v11));
            let zero = _mm_setzero_si128();
            let (neg0, neg1) = (_mm_sub_epi16(zero, lim[0].0), _mm_sub_epi16(zero, lim[1].0));
            u0 = _mm_max_epi16(_mm_min_epi16(u0, lim[0].0), neg0);
            u1 = _mm_max_epi16(_mm_min_epi16(u1, lim[1].0), neg1);
            u2 = _mm_max_epi16(_mm_min_epi16(u2, lim[0].0), neg0);
            u3 = _mm_max_epi16(_mm_min_epi16(u3, lim[1].0), neg1);
            let sum_lo = _mm_add_epi16(u0, u2);
            let sum_hi = _mm_add_epi16(u1, u3);
            [
                _mm_add_epi32(acc[0], _mm_madd_epi16(sum_lo, coeff[0].0)),
                _mm_add_epi32(acc[1], _mm_madd_epi16(sum_hi, coeff[1].0)),
            ]
        }
    }

    #[inline(always)]
    fn acc_alf_linear_pair(acc: Self::Acc, d0: Self, d1: Self, coeff: [Self; 2]) -> Self::Acc {
        // SAFETY: AVX2 kernels only (module docs).
        unsafe {
            let lo = _mm_unpacklo_epi16(d0.0, d1.0);
            let hi = _mm_unpackhi_epi16(d0.0, d1.0);
            [
                _mm_add_epi32(acc[0], _mm_madd_epi16(lo, coeff[0].0)),
                _mm_add_epi32(acc[1], _mm_madd_epi16(hi, coeff[1].0)),
            ]
        }
    }

    #[inline(always)]
    fn any_nonzero(self) -> bool {
        // SAFETY: AVX2 kernels only (module docs).
        unsafe { _mm_testz_si128(self.0, self.0) == 0 }
    }

    #[inline(always)]
    fn transpose_groups8(rows: [Self; 8]) -> [Self; 8] {
        // SAFETY: AVX2 kernels only; in-lane unpacks transpose the 8x8 tile.
        unsafe {
            let [r0, r1, r2, r3, r4, r5, r6, r7] = rows;
            let (p0, p1) = (
                _mm_unpacklo_epi16(r0.0, r1.0),
                _mm_unpackhi_epi16(r0.0, r1.0),
            );
            let (p2, p3) = (
                _mm_unpacklo_epi16(r2.0, r3.0),
                _mm_unpackhi_epi16(r2.0, r3.0),
            );
            let (p4, p5) = (
                _mm_unpacklo_epi16(r4.0, r5.0),
                _mm_unpackhi_epi16(r4.0, r5.0),
            );
            let (p6, p7) = (
                _mm_unpacklo_epi16(r6.0, r7.0),
                _mm_unpackhi_epi16(r6.0, r7.0),
            );
            let (q0, q1) = (_mm_unpacklo_epi32(p0, p2), _mm_unpackhi_epi32(p0, p2));
            let (q2, q3) = (_mm_unpacklo_epi32(p1, p3), _mm_unpackhi_epi32(p1, p3));
            let (q4, q5) = (_mm_unpacklo_epi32(p4, p6), _mm_unpackhi_epi32(p4, p6));
            let (q6, q7) = (_mm_unpacklo_epi32(p5, p7), _mm_unpackhi_epi32(p5, p7));
            [
                Self(_mm_unpacklo_epi64(q0, q4)),
                Self(_mm_unpackhi_epi64(q0, q4)),
                Self(_mm_unpacklo_epi64(q1, q5)),
                Self(_mm_unpackhi_epi64(q1, q5)),
                Self(_mm_unpacklo_epi64(q2, q6)),
                Self(_mm_unpackhi_epi64(q2, q6)),
                Self(_mm_unpacklo_epi64(q3, q7)),
                Self(_mm_unpackhi_epi64(q3, q7)),
            ]
        }
    }
}

impl I16Pairs for I16x16 {
    type Pairs = I32x8;

    #[inline(always)]
    fn splat_pair(first: i16, second: i16) -> Self {
        // SAFETY: AVX2 kernels only (module docs).
        unsafe { Self(_mm256_set1_epi32(packed_pair(first, second))) }
    }

    #[inline(always)]
    fn pair_dot(self, other: Self) -> I32x8 {
        // SAFETY: AVX2 kernels only (module docs).
        unsafe { I32x8(_mm256_madd_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn narrow_pairs(low: I32x8, high: I32x8) -> Self {
        // `packs` interleaves the 128-bit halves (low 0-3, high 0-3, low
        // 4-7, high 4-7); the permute restores lane order.
        // SAFETY: AVX2 kernels only (module docs).
        unsafe {
            Self(_mm256_permute4x64_epi64::<0b11_01_10_00>(
                _mm256_packs_epi32(low.0, high.0),
            ))
        }
    }

    #[inline(always)]
    fn interleave_pairs(even: I32x8, odd: I32x8) -> Self {
        // SAFETY: AVX2 kernels only (module docs).
        unsafe {
            Self(_mm256_blend_epi16::<0xAA>(
                even.0,
                _mm256_slli_epi32::<16>(odd.0),
            ))
        }
    }
}

impl I16Pairs for I16x8 {
    type Pairs = I32x4;

    #[inline(always)]
    fn splat_pair(first: i16, second: i16) -> Self {
        // SAFETY: AVX2 kernels only (module docs).
        unsafe { Self(_mm_set1_epi32(packed_pair(first, second))) }
    }

    #[inline(always)]
    fn pair_dot(self, other: Self) -> I32x4 {
        // SAFETY: AVX2 kernels only (module docs).
        unsafe { I32x4(_mm_madd_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn narrow_pairs(low: I32x4, high: I32x4) -> Self {
        // SAFETY: AVX2 kernels only (module docs).
        unsafe { Self(_mm_packs_epi32(low.0, high.0)) }
    }

    #[inline(always)]
    fn interleave_pairs(even: I32x4, odd: I32x4) -> Self {
        // SAFETY: AVX2 kernels only (module docs).
        unsafe { Self(_mm_blend_epi16::<0xAA>(even.0, _mm_slli_epi32::<16>(odd.0))) }
    }
}

impl I16x8Shuffles for I16x8 {
    type Wide = I32x4;

    #[inline(always)]
    fn shuffle_bytes(self, pattern: &[u8; 16]) -> Self {
        // SAFETY: AVX2 kernels only (module docs); `pattern` holds 16 bytes.
        // `pshufb` zeroes the bytes whose pattern has bit 7 set.
        unsafe {
            Self(_mm_shuffle_epi8(
                self.0,
                _mm_loadu_si128(pattern.as_ptr().cast()),
            ))
        }
    }

    #[inline(always)]
    fn zip_lo(self, other: Self) -> Self {
        // SAFETY: AVX2 kernels only (module docs).
        unsafe { Self(_mm_unpacklo_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn zip_hi(self, other: Self) -> Self {
        // SAFETY: AVX2 kernels only (module docs).
        unsafe { Self(_mm_unpackhi_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn madd(self, other: Self) -> I32x4 {
        // SAFETY: AVX2 kernels only (module docs).
        unsafe { I32x4(_mm_madd_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    fn sum_lanes4(x: [I32x4; 4]) -> I32x4 {
        // SAFETY: AVX2 kernels only (module docs).
        unsafe {
            I32x4(_mm_hadd_epi32(
                _mm_hadd_epi32(x[0].0, x[1].0),
                _mm_hadd_epi32(x[2].0, x[3].0),
            ))
        }
    }
}

/// Register transpose shared by residual stores and fused reconstruction.
#[inline(always)]
pub(crate) fn transpose_i32x8(rows: &[I32x8]) -> [I32x8; 8] {
    // SAFETY: called only by this module's ISA-enabled kernels.
    unsafe {
        let r = [
            rows[0].0, rows[1].0, rows[2].0, rows[3].0, rows[4].0, rows[5].0, rows[6].0, rows[7].0,
        ];
        // Stage 1: interleave pairs of rows within 128-bit lanes.
        let a0 = _mm256_unpacklo_epi32(r[0], r[1]);
        let a1 = _mm256_unpackhi_epi32(r[0], r[1]);
        let a2 = _mm256_unpacklo_epi32(r[2], r[3]);
        let a3 = _mm256_unpackhi_epi32(r[2], r[3]);
        let a4 = _mm256_unpacklo_epi32(r[4], r[5]);
        let a5 = _mm256_unpackhi_epi32(r[4], r[5]);
        let a6 = _mm256_unpacklo_epi32(r[6], r[7]);
        let a7 = _mm256_unpackhi_epi32(r[6], r[7]);
        // Stage 2: 64-bit interleave gives 4x4 transposes within each lane.
        let b0 = _mm256_unpacklo_epi64(a0, a2);
        let b1 = _mm256_unpackhi_epi64(a0, a2);
        let b2 = _mm256_unpacklo_epi64(a1, a3);
        let b3 = _mm256_unpackhi_epi64(a1, a3);
        let b4 = _mm256_unpacklo_epi64(a4, a6);
        let b5 = _mm256_unpackhi_epi64(a4, a6);
        let b6 = _mm256_unpacklo_epi64(a5, a7);
        let b7 = _mm256_unpackhi_epi64(a5, a7);
        // Stage 3: combine the 128-bit halves of rows 0-3 and 4-7.
        let out = [
            _mm256_permute2x128_si256::<0x20>(b0, b4),
            _mm256_permute2x128_si256::<0x20>(b1, b5),
            _mm256_permute2x128_si256::<0x20>(b2, b6),
            _mm256_permute2x128_si256::<0x20>(b3, b7),
            _mm256_permute2x128_si256::<0x31>(b0, b4),
            _mm256_permute2x128_si256::<0x31>(b1, b5),
            _mm256_permute2x128_si256::<0x31>(b2, b6),
            _mm256_permute2x128_si256::<0x31>(b3, b7),
        ];
        out.map(I32x8)
    }
}

/// Register transpose shared by residual stores and fused reconstruction.
#[inline(always)]
pub(crate) fn transpose_i32x4(rows: &[I32x4]) -> [I32x4; 4] {
    // SAFETY: called only by this module's ISA-enabled kernels.
    unsafe {
        let r = [rows[0].0, rows[1].0, rows[2].0, rows[3].0];
        let a0 = _mm_unpacklo_epi32(r[0], r[1]);
        let a1 = _mm_unpackhi_epi32(r[0], r[1]);
        let a2 = _mm_unpacklo_epi32(r[2], r[3]);
        let a3 = _mm_unpackhi_epi32(r[2], r[3]);
        let out = [
            _mm_unpacklo_epi64(a0, a2),
            _mm_unpackhi_epi64(a0, a2),
            _mm_unpacklo_epi64(a1, a3),
            _mm_unpackhi_epi64(a1, a3),
        ];
        out.map(I32x4)
    }
}

impl super::I16x8Tiles for I16x16 {
    #[inline(always)]
    fn splat_segments(values: &[i16; 8]) -> Self {
        const SELECT: [u8; 32] = [
            0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6,
            7, 6, 7,
        ];
        // SAFETY: enabled ISA; eight words and one complete shuffle vector.
        unsafe {
            let data = _mm_loadu_si128(values.as_ptr().cast());
            Self(_mm256_shuffle_epi8(
                _mm256_set_m128i(data, data),
                _mm256_loadu_si256(SELECT.as_ptr().cast()),
            ))
        }
    }
    #[inline(always)]
    fn zip_tiles_lo(self, other: Self) -> Self {
        // SAFETY: private lanes run only inside an AVX2 entry point.
        unsafe { Self(_mm256_unpacklo_epi16(self.0, other.0)) }
    }
    #[inline(always)]
    fn zip_tiles_hi(self, other: Self) -> Self {
        // SAFETY: as above; unpacks stay inside each 128-bit tile.
        unsafe { Self(_mm256_unpackhi_epi16(self.0, other.0)) }
    }
    #[inline(always)]
    fn shuffle_tiles(self, pattern: &[u8; 16]) -> Self {
        // SAFETY: enabled ISA; the pattern is exactly sixteen readable bytes.
        unsafe {
            let p = _mm_loadu_si128(pattern.as_ptr().cast());
            Self(_mm256_shuffle_epi8(self.0, _mm256_set_m128i(p, p)))
        }
    }
    #[inline(always)]
    fn load_tiles8(rows: [&[u8]; 4]) -> Self {
        let (a, b) = (&rows[0][..8], &rows[1][..8]);
        // SAFETY: enabled ISA; each load reads eight checked bytes.
        unsafe {
            Self(_mm256_cvtepu8_epi16(_mm_unpacklo_epi64(
                _mm_loadl_epi64(a.as_ptr().cast()),
                _mm_loadl_epi64(b.as_ptr().cast()),
            )))
        }
    }
    #[inline(always)]
    fn store_tiles8(self, rows: [&mut [u8]; 4]) {
        let [a, b, _, _] = rows;
        self.store_u8_sat_halves(a, b);
    }
    #[inline(always)]
    fn load_u16_tiles8(rows: [&[u16]; 4]) -> Self {
        let (a, b) = (&rows[0][..8], &rows[1][..8]);
        // SAFETY: enabled ISA; each load reads eight checked `u16` samples.
        unsafe {
            Self(_mm256_set_m128i(
                _mm_loadu_si128(b.as_ptr().cast()),
                _mm_loadu_si128(a.as_ptr().cast()),
            ))
        }
    }
    #[inline(always)]
    fn store_u16_tiles8(self, rows: [&mut [u16]; 4], max_val: i16) {
        let [a, b, _, _] = rows;
        let (a, b) = (&mut a[..8], &mut b[..8]);
        let clamped = self.max(Self::splat(0)).min(Self::splat(max_val)).0;
        // SAFETY: enabled ISA; each destination has eight writable `u16` samples.
        unsafe {
            _mm_storeu_si128(a.as_mut_ptr().cast(), _mm256_castsi256_si128(clamped));
            _mm_storeu_si128(
                b.as_mut_ptr().cast(),
                _mm256_extracti128_si256::<1>(clamped),
            );
        }
    }
}

impl crate::sao::high_simd::SaoWordLanes for I16x16 {
    #[inline(always)]
    fn offset_table(offsets: [i16; 4]) -> Self {
        use core::arch::x86_64::{_mm_loadu_si128, _mm256_broadcastsi128_si256};
        let table = crate::sao::high_simd::offset_words(offsets);
        // SAFETY: the table is eight readable words, in the AVX2 SAO entry.
        unsafe {
            Self(_mm256_broadcastsi128_si256(_mm_loadu_si128(
                table.as_ptr().cast(),
            )))
        }
    }

    #[inline(always)]
    fn select_offset(self, table: Self) -> Self {
        use core::arch::x86_64::{
            _mm256_add_epi16, _mm256_mullo_epi16, _mm256_set1_epi16, _mm256_shuffle_epi8,
        };
        // SAFETY: called from AVX2 SAO. The two byte indices within each
        // word select the full signed offset, even outside the i8 range.
        unsafe {
            let index = _mm256_add_epi16(
                _mm256_mullo_epi16(self.0, _mm256_set1_epi16(0x0202)),
                _mm256_set1_epi16(0x0504),
            );
            Self(_mm256_shuffle_epi8(table.0, index))
        }
    }
}
