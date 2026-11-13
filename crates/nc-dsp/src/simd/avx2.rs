//! x86-64 vector types: 256-bit AVX2 for wide rows, 128-bit SSE4.1 (implied
//! by AVX2) for narrow ones.
//!
//! See the `simd` module docs for why the methods are safe functions: they
//! are only ever called from kernels running under
//! `#[target_feature(enable = "avx2")]`.

#![allow(clippy::inline_always)]

use core::arch::x86_64::{
    __m128i, __m256i, _mm_abs_epi16, _mm_abs_epi32, _mm_add_epi16, _mm_add_epi32, _mm_and_si128,
    _mm_blendv_epi8, _mm_castsi128_ps, _mm_cmpeq_epi16, _mm_cmpeq_epi32, _mm_cmpgt_epi16,
    _mm_cmpgt_epi32, _mm_cvtepi16_epi32, _mm_cvtepu8_epi16, _mm_cvtepu8_epi32, _mm_cvtsi32_si128,
    _mm_cvtsi128_si32, _mm_extract_epi32, _mm_hadd_epi32, _mm_loadl_epi64, _mm_loadu_si128,
    _mm_madd_epi16, _mm_maddubs_epi16, _mm_max_epi16, _mm_max_epi32, _mm_min_epi16, _mm_min_epi32,
    _mm_min_epu16, _mm_movemask_ps, _mm_mulhrs_epi16, _mm_mullo_epi16, _mm_mullo_epi32,
    _mm_or_si128, _mm_packs_epi32, _mm_packus_epi16, _mm_set1_epi16, _mm_set1_epi32,
    _mm_setzero_si128, _mm_shuffle_epi8, _mm_sign_epi16, _mm_sign_epi32, _mm_sll_epi32,
    _mm_sra_epi16, _mm_sra_epi32, _mm_srl_epi32, _mm_storel_epi64, _mm_storeu_si128, _mm_sub_epi16,
    _mm_sub_epi32, _mm_subs_epu16, _mm_unpackhi_epi8, _mm_unpackhi_epi16, _mm_unpackhi_epi32,
    _mm_unpackhi_epi64, _mm_unpacklo_epi8, _mm_unpacklo_epi16, _mm_unpacklo_epi32,
    _mm_unpacklo_epi64, _mm256_abs_epi16, _mm256_abs_epi32, _mm256_add_epi16, _mm256_add_epi32,
    _mm256_and_si256, _mm256_blend_epi16, _mm256_blendv_epi8, _mm256_castsi256_ps,
    _mm256_castsi256_si128, _mm256_cmpeq_epi16, _mm256_cmpeq_epi32, _mm256_cmpgt_epi16,
    _mm256_cmpgt_epi32, _mm256_cvtepi16_epi32, _mm256_cvtepu8_epi16, _mm256_extracti128_si256,
    _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_max_epi16,
    _mm256_max_epi32, _mm256_min_epi16, _mm256_min_epi32, _mm256_min_epu16, _mm256_movemask_ps,
    _mm256_mulhrs_epi16, _mm256_mullo_epi16, _mm256_mullo_epi32, _mm256_or_si256,
    _mm256_packs_epi32, _mm256_permute2x128_si256, _mm256_permute4x64_epi64, _mm256_set_m128i,
    _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setzero_si256, _mm256_sign_epi16,
    _mm256_sign_epi32, _mm256_sll_epi32, _mm256_slli_epi32, _mm256_sra_epi16, _mm256_sra_epi32,
    _mm256_srl_epi32, _mm256_storeu_si256, _mm256_sub_epi16, _mm256_sub_epi32, _mm256_subs_epu16,
    _mm256_unpackhi_epi16, _mm256_unpackhi_epi32, _mm256_unpackhi_epi64, _mm256_unpacklo_epi16,
    _mm256_unpacklo_epi32, _mm256_unpacklo_epi64,
};

use core::mem::MaybeUninit;

use super::{I16FilterLanes, I16Lanes, I16Pairs, I16x8Shuffles, I32Lanes, packed_pair};

/// Eight `i32` lanes (AVX2).
#[derive(Debug, Clone, Copy)]
pub(crate) struct I32x8(__m256i);

/// Four `i32` lanes (SSE4.1).
#[derive(Debug, Clone, Copy)]
pub(crate) struct I32x4(__m128i);

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
pub(crate) struct I16x16(__m256i);

/// Eight `i16` lanes (SSE4.1).
#[derive(Debug, Clone, Copy)]
pub(crate) struct I16x8(__m128i);

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
fn sum_i32x8(v: __m256i) -> i64 {
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
    fn load(src: &[i16]) -> Self {
        // SAFETY: these types run only inside AVX2 kernels (module docs);
        // slices are indexed before every load and store.
        unsafe {
            let src = &src[..16];
            Self(_mm256_loadu_si256(src.as_ptr().cast()))
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
fn transpose_i32x8(rows: &[I32x8]) -> [I32x8; 8] {
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
fn transpose_i32x4(rows: &[I32x4]) -> [I32x4; 4] {
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
