//! AVX2/ICL contiguous loads and row-statistic deinterleaving. Each dword
//! lane is a different block: no pixel row gathering or block-wise call.

#![allow(
    clippy::inline_always,
    reason = "ISA primitives and the shared graph must inline into their target-feature wrapper"
)]

use super::generic::{self, Lanes, MAX_LANES};
use super::{AnalysisBlock, AnalysisChanges, AnalysisSummary, Moments};
use core::arch::x86_64::{
    __m256i, __m512i, _mm256_add_epi32, _mm256_and_si256, _mm256_castsi256_si128,
    _mm256_cvtepu16_epi32, _mm256_hadd_epi32, _mm256_loadu_si256, _mm256_madd_epi16,
    _mm256_max_epi32, _mm256_max_epu16, _mm256_mullo_epi32, _mm256_permute2x128_si256,
    _mm256_permute4x64_epi64, _mm256_permutevar8x32_epi32, _mm256_set1_epi16, _mm256_set1_epi32,
    _mm256_setr_epi8, _mm256_setr_epi32, _mm256_shuffle_epi8, _mm256_srli_epi32, _mm256_srli_epi64,
    _mm256_storeu_si256, _mm256_sub_epi32, _mm512_add_epi32, _mm512_and_si512,
    _mm512_castsi512_si256, _mm512_cvtepu16_epi32, _mm512_loadu_si512, _mm512_madd_epi16,
    _mm512_max_epi32, _mm512_max_epu16, _mm512_mullo_epi32, _mm512_permutex2var_epi32,
    _mm512_permutexvar_epi32, _mm512_set1_epi16, _mm512_set1_epi32, _mm512_setr_epi32,
    _mm512_srli_epi32, _mm512_srli_epi64, _mm512_storeu_si512, _mm512_sub_epi32,
};

#[derive(Clone, Copy)]
struct Avx2(__m256i);

impl Lanes for Avx2 {
    const LANES: usize = 8;
    #[inline(always)]
    fn splat(value: i32) -> Self {
        // SAFETY: only instantiated inside the matching feature wrapper.
        Self(unsafe { _mm256_set1_epi32(value) })
    }
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // SAFETY: only instantiated inside the matching feature wrapper.
        Self(unsafe { _mm256_add_epi32(self.0, other.0) })
    }
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        // SAFETY: only instantiated inside the matching feature wrapper.
        Self(unsafe { _mm256_sub_epi32(self.0, other.0) })
    }
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        // SAFETY: only instantiated inside the matching feature wrapper.
        Self(unsafe { _mm256_mullo_epi32(self.0, other.0) })
    }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        // SAFETY: only instantiated inside the matching feature wrapper.
        Self(unsafe { _mm256_max_epi32(self.0, other.0) })
    }
    #[inline(always)]
    fn shr6(self) -> Self {
        // SAFETY: only instantiated inside the matching feature wrapper.
        Self(unsafe { _mm256_srli_epi32::<6>(self.0) })
    }
    #[inline(always)]
    fn store(self, out: &mut [i32; MAX_LANES]) {
        // SAFETY: the fixed array holds all 8 lanes; the ISA is supported.
        unsafe { _mm256_storeu_si256(out.as_mut_ptr().cast(), self.0) }
    }
    #[inline(always)]
    unsafe fn moments(src: *const Moments) -> (Self, Self) {
        // SAFETY: a complete batch of repr(C) pairs, inside the ISA wrapper.
        unsafe {
            let index = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
            let low = _mm256_permutevar8x32_epi32(_mm256_loadu_si256(src.cast()), index);
            let high = _mm256_permutevar8x32_epi32(_mm256_loadu_si256(src.add(4).cast()), index);
            (
                Self(_mm256_permute2x128_si256::<0x20>(low, high)),
                Self(_mm256_permute2x128_si256::<0x31>(low, high)),
            )
        }
    }
    #[inline(always)]
    unsafe fn changes(changes: AnalysisChanges<'_>, start: usize) -> (Self, Self) {
        // SAFETY: complete luma/chroma batches and the matching ISA.
        unsafe {
            let low = _mm256_loadu_si256(changes.luma.as_ptr().add(start).cast());
            let high = _mm256_loadu_si256(changes.luma.as_ptr().add(start + 4).cast());
            let ones = _mm256_set1_epi16(1);
            let pairs_a = _mm256_madd_epi16(low, ones);
            let pairs_b = _mm256_madd_epi16(high, ones);
            let sum = _mm256_permute4x64_epi64::<0xd8>(_mm256_hadd_epi32(pairs_a, pairs_b));
            let qa = _mm256_max_epu16(low, _mm256_srli_epi64::<32>(low));
            let qb = _mm256_max_epu16(high, _mm256_srli_epi64::<32>(high));
            let qa = _mm256_max_epu16(qa, _mm256_srli_epi64::<16>(qa));
            let qb = _mm256_max_epu16(qb, _mm256_srli_epi64::<16>(qb));
            let even = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);
            let qa = _mm256_permutevar8x32_epi32(qa, even);
            let qb = _mm256_permutevar8x32_epi32(qb, even);
            let busiest = _mm256_and_si256(
                _mm256_permute2x128_si256::<0x20>(qa, qb),
                _mm256_set1_epi32(i32::from(u16::MAX)),
            );
            let chroma_u = _mm256_loadu_si256(changes.chroma[0].as_ptr().add(start / 2).cast());
            let chroma_v = _mm256_loadu_si256(changes.chroma[1].as_ptr().add(start / 2).cast());
            let index = if changes.bottom {
                _mm256_setr_epi8(
                    4, 5, 6, 7, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 6, 7, 12, 13,
                    14, 15, -1, -1, -1, -1, -1, -1, -1, -1,
                )
            } else {
                _mm256_setr_epi8(
                    0, 1, 2, 3, 8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 8, 9, 10,
                    11, -1, -1, -1, -1, -1, -1, -1, -1,
                )
            };
            let chroma = _mm256_shuffle_epi8(_mm256_max_epu16(chroma_u, chroma_v), index);
            let chroma = _mm256_permute4x64_epi64::<0xd8>(chroma);
            let chroma = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(chroma));
            (Self(sum), Self(_mm256_max_epi32(busiest, chroma)))
        }
    }
}

#[derive(Clone, Copy)]
struct Icl(__m512i);

impl Lanes for Icl {
    const LANES: usize = 16;
    #[inline(always)]
    fn splat(value: i32) -> Self {
        // SAFETY: only instantiated inside the matching feature wrapper.
        Self(unsafe { _mm512_set1_epi32(value) })
    }
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // SAFETY: only instantiated inside the matching feature wrapper.
        Self(unsafe { _mm512_add_epi32(self.0, other.0) })
    }
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        // SAFETY: only instantiated inside the matching feature wrapper.
        Self(unsafe { _mm512_sub_epi32(self.0, other.0) })
    }
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        // SAFETY: only instantiated inside the matching feature wrapper.
        Self(unsafe { _mm512_mullo_epi32(self.0, other.0) })
    }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        // SAFETY: only instantiated inside the matching feature wrapper.
        Self(unsafe { _mm512_max_epi32(self.0, other.0) })
    }
    #[inline(always)]
    fn shr6(self) -> Self {
        // SAFETY: only instantiated inside the matching feature wrapper.
        Self(unsafe { _mm512_srli_epi32::<6>(self.0) })
    }
    #[inline(always)]
    fn store(self, out: &mut [i32; MAX_LANES]) {
        // SAFETY: the fixed array holds all 16 lanes; the ISA is supported.
        unsafe { _mm512_storeu_si512(out.as_mut_ptr().cast(), self.0) }
    }
    #[inline(always)]
    unsafe fn moments(src: *const Moments) -> (Self, Self) {
        // SAFETY: a complete batch of repr(C) pairs, inside the ISA wrapper.
        unsafe {
            let low = _mm512_loadu_si512(src.cast());
            let high = _mm512_loadu_si512(src.add(8).cast());
            let even = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
            (
                Self(_mm512_permutex2var_epi32(low, even, high)),
                Self(_mm512_permutex2var_epi32(
                    low,
                    _mm512_add_epi32(even, _mm512_set1_epi32(1)),
                    high,
                )),
            )
        }
    }
    #[inline(always)]
    unsafe fn changes(changes: AnalysisChanges<'_>, start: usize) -> (Self, Self) {
        // SAFETY: complete luma/chroma batches and the matching ISA.
        unsafe {
            let low = _mm512_loadu_si512(changes.luma.as_ptr().add(start).cast());
            let high = _mm512_loadu_si512(changes.luma.as_ptr().add(start + 8).cast());
            let even = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
            let odd = _mm512_add_epi32(even, _mm512_set1_epi32(1));
            let ones = _mm512_set1_epi16(1);
            let pa = _mm512_madd_epi16(low, ones);
            let pb = _mm512_madd_epi16(high, ones);
            let sum = _mm512_add_epi32(
                _mm512_permutex2var_epi32(pa, even, pb),
                _mm512_permutex2var_epi32(pa, odd, pb),
            );
            let qa = _mm512_max_epu16(low, _mm512_srli_epi64::<32>(low));
            let qb = _mm512_max_epu16(high, _mm512_srli_epi64::<32>(high));
            let qa = _mm512_max_epu16(qa, _mm512_srli_epi64::<16>(qa));
            let qb = _mm512_max_epu16(qb, _mm512_srli_epi64::<16>(qb));
            let busiest = _mm512_and_si512(
                _mm512_permutex2var_epi32(qa, even, qb),
                _mm512_set1_epi32(i32::from(u16::MAX)),
            );
            let chroma_u = _mm512_loadu_si512(changes.chroma[0].as_ptr().add(start / 2).cast());
            let chroma_v = _mm512_loadu_si512(changes.chroma[1].as_ptr().add(start / 2).cast());
            let index = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14);
            let index = _mm512_add_epi32(index, _mm512_set1_epi32(i32::from(changes.bottom)));
            let chroma = _mm512_permutexvar_epi32(index, _mm512_max_epu16(chroma_u, chroma_v));
            let chroma = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(chroma));
            (Self(sum), Self(_mm512_max_epi32(busiest, chroma)))
        }
    }
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn avx2(
    m: &[Moments],
    changes: Option<AnalysisChanges<'_>>,
    out: &mut [AnalysisBlock],
) -> AnalysisSummary {
    // SAFETY: the table validates AVX2 support and the caller checks slices.
    unsafe { generic::run::<Avx2>(m, changes, out) }
}

#[target_feature(
    enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl,avx512cd,avx512vbmi,avx512vbmi2,avx512vnni,avx512bitalg,avx512vpopcntdq,gfni"
)]
pub(super) unsafe fn icl(
    m: &[Moments],
    changes: Option<AnalysisChanges<'_>>,
    out: &mut [AnalysisBlock],
) -> AnalysisSummary {
    // SAFETY: the table validates the full ICL tier and the caller checks slices.
    unsafe { generic::run::<Icl>(m, changes, out) }
}
