//! Four independent blocks per ordinary SIMD128 vector, with contiguous
//! loads and byte/dword shuffles matching the native statistics graph.

#![allow(
    clippy::inline_always,
    reason = "ISA primitives and the shared graph must inline into their target-feature wrapper"
)]

use super::generic::{self, Lanes, MAX_LANES};
use super::{AnalysisBlock, AnalysisChanges, AnalysisSummary, Moments};
use core::arch::wasm32::{
    i8x16_shuffle, i16x8_splat, i32x4_add, i32x4_dot_i16x8, i32x4_max, i32x4_mul, i32x4_shuffle,
    i32x4_splat, i32x4_sub, u16x8_max, u32x4_shr, u64x2_shr, v128, v128_and, v128_load, v128_store,
};

#[derive(Clone, Copy)]
struct Wasm(v128);

impl Lanes for Wasm {
    const LANES: usize = 4;
    #[inline(always)]
    fn splat(value: i32) -> Self {
        Self(i32x4_splat(value))
    }
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        Self(i32x4_add(self.0, other.0))
    }
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        Self(i32x4_sub(self.0, other.0))
    }
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        Self(i32x4_mul(self.0, other.0))
    }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        Self(i32x4_max(self.0, other.0))
    }
    #[inline(always)]
    fn shr6(self) -> Self {
        Self(u32x4_shr(self.0, 6))
    }
    #[inline(always)]
    fn store(self, out: &mut [i32; MAX_LANES]) {
        // SAFETY: a full four-lane output fits the fixed array.
        unsafe { v128_store(out.as_mut_ptr().cast(), self.0) }
    }
    #[inline(always)]
    unsafe fn moments(src: *const Moments) -> (Self, Self) {
        // SAFETY: a complete batch of four repr(C) moment pairs.
        unsafe {
            let low = v128_load(src.cast());
            let high = v128_load(src.add(2).cast());
            (
                Self(i32x4_shuffle::<0, 2, 4, 6>(low, high)),
                Self(i32x4_shuffle::<1, 3, 5, 7>(low, high)),
            )
        }
    }
    #[inline(always)]
    unsafe fn changes(changes: AnalysisChanges<'_>, start: usize) -> (Self, Self) {
        // SAFETY: four luma tuples and two chroma tuples per plane fit.
        unsafe {
            let low = v128_load(changes.luma.as_ptr().add(start).cast());
            let high = v128_load(changes.luma.as_ptr().add(start + 2).cast());
            let pa = i32x4_dot_i16x8(low, i16x8_splat(1));
            let pb = i32x4_dot_i16x8(high, i16x8_splat(1));
            let sum = i32x4_add(
                i32x4_shuffle::<0, 2, 4, 6>(pa, pb),
                i32x4_shuffle::<1, 3, 5, 7>(pa, pb),
            );
            let qa = u16x8_max(low, u64x2_shr(low, 32));
            let qb = u16x8_max(high, u64x2_shr(high, 32));
            let qa = u16x8_max(qa, u64x2_shr(qa, 16));
            let qb = u16x8_max(qb, u64x2_shr(qb, 16));
            let busiest = v128_and(
                i32x4_shuffle::<0, 2, 4, 6>(qa, qb),
                i32x4_splat(i32::from(u16::MAX)),
            );
            let chroma_u = v128_load(changes.chroma[0].as_ptr().add(start / 2).cast());
            let chroma_v = v128_load(changes.chroma[1].as_ptr().add(start / 2).cast());
            let uv = u16x8_max(chroma_u, chroma_v);
            let zero = i32x4_splat(0);
            let chroma = if changes.bottom {
                i8x16_shuffle::<4, 5, 16, 16, 6, 7, 16, 16, 12, 13, 16, 16, 14, 15, 16, 16>(
                    uv, zero,
                )
            } else {
                i8x16_shuffle::<0, 1, 16, 16, 2, 3, 16, 16, 8, 9, 16, 16, 10, 11, 16, 16>(uv, zero)
            };
            (Self(sum), Self(i32x4_max(busiest, chroma)))
        }
    }
}

#[target_feature(enable = "simd128")]
pub(super) unsafe fn stats(
    m: &[Moments],
    changes: Option<AnalysisChanges<'_>>,
    out: &mut [AnalysisBlock],
) -> AnalysisSummary {
    // SAFETY: only compiled with SIMD128 and the caller checks row extents.
    unsafe { generic::run::<Wasm>(m, changes, out) }
}
