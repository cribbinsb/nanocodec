//! Packed-byte 4/8-wide statistics shared by all six codecs, for the AVX2
//! and AVX-512 tiers alike (an eight-row 512-bit form with VNNI squares was
//! slower on Zen 5; see `diff::avx512`). The row layout, interleaved
//! `pmaddubsw` difference and combined sum/SSE reduction follow SVT-AV1
//! v4.3.0 `ASM_AVX2/variance_avx2.c` (`variance8_kernel_avx2`),
//! BSD-2-Clause with AOM patent licence; Copyright 2016 AOM. See
//! `THIRD_PARTY.md`.
//!
//! The low eight bytes of each 128-bit half hold one 8-pixel row or two
//! 4-pixel rows. Interleaving source/prediction gives unsigned byte pairs;
//! multiply-add by (1,-1) produces exact signed-word differences. Thus each
//! step handles two 8-wide or four 4-wide rows. Exact-size loads preserve
//! the original plane contract.
//!
//! At most 32 rows visit an AVX2 word lane: |sum| <= 32*255 = 8160.
//! The square sum of an entire 8x64 block is <= 512*255^2 = 33,292,800,
//! so only the paired square products and their totals need i32 lanes.
#![allow(clippy::inline_always)]

use super::generic::covers;
use super::{Variance, VarianceEnergy};
use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi32, _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_extract_epi32,
    _mm_insert_epi32, _mm_loadl_epi64, _mm_srli_si128, _mm256_add_epi16, _mm256_add_epi32,
    _mm256_blend_epi32, _mm256_broadcastq_epi64, _mm256_castsi128_si256, _mm256_castsi256_si128,
    _mm256_extracti128_si256, _mm256_inserti128_si256, _mm256_madd_epi16, _mm256_maddubs_epi16,
    _mm256_set1_epi16, _mm256_setzero_si256, _mm256_unpackhi_epi32, _mm256_unpacklo_epi8,
    _mm256_unpacklo_epi32,
};

/// Unsigned source byte times 1, unsigned prediction byte times -1.
const DIFFERENCE_WEIGHTS: i16 = i16::from_le_bytes([1, u8::MAX]);
const NARROW_WIDTHS: [usize; 2] = [4, 8];
const MAX_HEIGHT: usize = 64;

/// Register operations for the same packed-row graph. Values are bytes on
/// load, words for differences/sums, and dwords for squares. Only construct
/// them inside the ISA-enabled wrappers; every method inlines there.
pub(super) trait PackedRows: Copy {
    const PIXELS: usize;
    fn zero() -> Self;
    /// W is 4 or 8; `rows` is a full payload. Those complete rows of W
    /// bytes are readable; the ISA is enabled.
    unsafe fn load<const W: usize>(base: *const u8, stride: usize, rows: usize) -> Self;
    fn difference(self, pred: Self) -> Self;
    fn widen(self) -> Self;
    fn add_words(self, other: Self) -> Self;
    fn add_squares(self, words: Self) -> Self;
    fn reduce(sum: Self, squared: Self) -> Variance;
}

/// SVT's two-row 8-wide layout; four rows for width four.
#[derive(Clone, Copy)]
pub(super) struct Avx2Rows(__m256i);

/// The first two four-byte rows in the low eight bytes of an SSE register.
#[inline(always)]
unsafe fn load_pair4(base: *const u8, stride: usize) -> __m128i {
    // SAFETY: the caller checked exactly four bytes in both rows. These
    // scalar operands fold into vmovd/vpinsrd memory operands, not GPR gathers.
    unsafe {
        _mm_insert_epi32::<1>(
            _mm_cvtsi32_si128(base.cast::<i32>().read_unaligned()),
            base.add(stride).cast::<i32>().read_unaligned(),
        )
    }
}

impl PackedRows for Avx2Rows {
    const PIXELS: usize = 16;
    #[inline(always)]
    fn zero() -> Self {
        // SAFETY: AVX2 is enabled by the statistics wrapper.
        unsafe { Self(_mm256_setzero_si256()) }
    }
    #[inline(always)]
    unsafe fn load<const W: usize>(base: *const u8, stride: usize, rows: usize) -> Self {
        debug_assert_eq!(rows, Self::PIXELS / W);
        // SAFETY: every loaded row is inside the validated block. Only its
        // first four/eight bytes are accessed, including the final row.
        unsafe {
            Self(if W == 8 {
                let first = _mm_loadl_epi64(base.cast());
                let second = _mm256_broadcastq_epi64(_mm_loadl_epi64(base.add(stride).cast()));
                _mm256_blend_epi32::<0xf0>(_mm256_castsi128_si256(first), second)
            } else {
                _mm256_inserti128_si256::<1>(
                    _mm256_castsi128_si256(load_pair4(base, stride)),
                    load_pair4(base.add(2 * stride), stride),
                )
            })
        }
    }
    #[inline(always)]
    fn difference(self, pred: Self) -> Self {
        // SAFETY: register-only AVX2; each (src,pred) product lies in [-255,255].
        unsafe {
            Self(_mm256_maddubs_epi16(
                _mm256_unpacklo_epi8(self.0, pred.0),
                _mm256_set1_epi16(DIFFERENCE_WEIGHTS),
            ))
        }
    }
    #[inline(always)]
    fn widen(self) -> Self {
        // SAFETY: register-only AVX2, low byte payload widened with zero high bytes.
        unsafe { Self(_mm256_unpacklo_epi8(self.0, _mm256_setzero_si256())) }
    }
    #[inline(always)]
    fn add_words(self, other: Self) -> Self {
        // SAFETY: register-only AVX2; the module bound keeps the sum in i16.
        unsafe { Self(_mm256_add_epi16(self.0, other.0)) }
    }
    #[inline(always)]
    fn add_squares(self, words: Self) -> Self {
        // SAFETY: paired products and the full block total fit i32 (module docs).
        unsafe {
            Self(_mm256_add_epi32(
                self.0,
                _mm256_madd_epi16(words.0, words.0),
            ))
        }
    }
    #[inline(always)]
    fn reduce(sum: Self, squared: Self) -> Variance {
        // SAFETY: register-only AVX2; paired word sums are exact in i32.
        unsafe {
            let sum = _mm256_madd_epi16(sum.0, _mm256_set1_epi16(1));
            finish(_mm256_add_epi32(
                _mm256_unpacklo_epi32(squared.0, sum),
                _mm256_unpackhi_epi32(squared.0, sum),
            ))
        }
    }
}

/// Alternating squared/signed sums, already paired within each 128-bit lane.
#[inline(always)]
unsafe fn finish(both: __m256i) -> Variance {
    // SAFETY: AVX2 is enabled by the wrappers. All block sums fit i32/u32.
    unsafe {
        let folded = _mm_add_epi32(
            _mm256_castsi256_si128(both),
            _mm256_extracti128_si256::<1>(both),
        );
        let result = _mm_add_epi32(folded, _mm_srli_si128::<8>(folded));
        Variance {
            sse: _mm_cvtsi128_si32(result) as u32,
            sum: _mm_extract_epi32::<1>(result),
        }
    }
}

#[inline(always)]
unsafe fn accumulate<V: PackedRows, const W: usize, const ENERGY: bool, const SOURCE_ONLY: bool>(
    src: &[u8],
    ss: usize,
    pred: &[u8],
    ps: usize,
    h: usize,
) -> VarianceEnergy {
    debug_assert!(NARROW_WIDTHS.contains(&W) && h <= MAX_HEIGHT && h.is_multiple_of(4));
    debug_assert!(covers(src.len(), ss, h, W) && (SOURCE_ONLY || covers(pred.len(), ps, h, W)));
    let (mut sum, mut squared, mut psum, mut psquared) =
        (V::zero(), V::zero(), V::zero(), V::zero());
    let rows_per_vector = V::PIXELS / W;
    for y in (0..h).step_by(rows_per_vector) {
        let rows = rows_per_vector.min(h - y);
        // SAFETY: the validated block covers exactly these complete rows.
        let source = unsafe { V::load::<W>(src.as_ptr().add(y * ss), ss, rows) };
        let difference = if SOURCE_ONLY {
            source.widen()
        } else {
            // SAFETY: same validated rows of the prediction, at its own stride.
            let prediction = unsafe { V::load::<W>(pred.as_ptr().add(y * ps), ps, rows) };
            if ENERGY {
                let values = prediction.widen();
                psum = psum.add_words(values);
                psquared = psquared.add_squares(values);
            }
            source.difference(prediction)
        };
        sum = sum.add_words(difference);
        squared = squared.add_squares(difference);
    }
    VarianceEnergy {
        residual: V::reduce(sum, squared),
        prediction: if ENERGY {
            V::reduce(psum, psquared)
        } else {
            Variance::default()
        },
    }
}

/// Ordinary variance on the same packed graph as prediction energy.
#[inline(always)]
pub(super) unsafe fn variance<V: PackedRows, const W: usize>(
    src: &[u8],
    ss: usize,
    pred: &[u8],
    ps: usize,
    h: usize,
) -> Variance {
    // SAFETY: the ISA-enabled wrapper forwards the validated block contract.
    unsafe { accumulate::<V, W, false, false>(src, ss, pred, ps, h).residual }
}

/// Residual statistics and prediction energy in one pass over the same rows.
#[inline(always)]
pub(super) unsafe fn variance_energy<V: PackedRows, const W: usize>(
    src: &[u8],
    ss: usize,
    pred: &[u8],
    ps: usize,
    h: usize,
) -> VarianceEnergy {
    // SAFETY: the ISA-enabled wrapper forwards the validated block contract.
    unsafe { accumulate::<V, W, true, false>(src, ss, pred, ps, h) }
}

/// Source statistics, with no prediction loads or subtraction.
#[inline(always)]
pub(super) unsafe fn block_sums<V: PackedRows, const W: usize>(
    src: &[u8],
    ss: usize,
    h: usize,
) -> Variance {
    // SAFETY: the ISA-enabled wrapper forwards the validated source contract.
    unsafe { accumulate::<V, W, false, true>(src, ss, &[], 0, h).residual }
}
