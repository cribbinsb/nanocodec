//! SIMD128 VP9 interpolation with widening byte products of tap magnitudes.
//! Positive and negative products accumulate separately in u16 (each sum plus
//! rounding fits; `positive_and_negative_sums_fit_u16`). Their saturating
//! difference is the filter sum clipped below at zero; rounding, a logical
//! shift and a saturating narrow then give the reference result exactly.
//! Rows are eight outputs: sixteen (both halves of each load) measured
//! slower under V8, whose 16 vector registers then spill (2026-11-12).

use super::driver::{ROUND, RowKernel, SHIFT, layout};
use crate::Vp9Filter;
use crate::vp9_mc::VP9_TAPS;
use core::arch::wasm32::{
    i16x8_add, u8x16_avgr, u8x16_narrow_i16x8, u8x16_splat, u16x8_extmul_low_u8x16, u16x8_shr,
    u16x8_splat, u16x8_sub_sat, v128, v128_load32_zero, v128_load64_zero, v128_store32_lane,
    v128_store64_lane,
};
struct Simd128;

/// The rounded, shifted 8-bit results of one half (eight outputs) of the
/// filter, from the products of each tap (`product(k)`).
#[inline(always)]
fn half<const LAYOUT: u8>(product: impl Fn(usize) -> v128) -> v128 {
    let clipped = if LAYOUT == layout::BILINEAR {
        // Both centre taps are positive and sum to 128: no u16 overflow.
        i16x8_add(product(3), product(4))
    } else {
        let (mut positive, mut negative) = (u16x8_splat(0), u16x8_splat(0));
        for k in 0..VP9_TAPS {
            if layout::negative(LAYOUT, k) {
                negative = i16x8_add(negative, product(k));
            } else {
                positive = i16x8_add(positive, product(k));
            }
        }
        u16x8_sub_sat(positive, negative)
    };
    u16x8_shr(i16x8_add(clipped, u16x8_splat(ROUND as u16)), SHIFT)
}

/// Up to eight outputs from the eight tap inputs `pixels(k)`.
#[inline(always)]
fn filter<const LAYOUT: u8>(pixels: impl Fn(usize) -> v128, f: &[v128; 8]) -> v128 {
    let sums = half::<LAYOUT>(|k| u16x8_extmul_low_u8x16(pixels(k), f[k]));
    u8x16_narrow_i16x8(sums, u16x8_splat(0))
}

impl RowKernel for Simd128 {
    type Filters = [v128; 8];
    type Row = v128;
    const LANES: usize = 8;
    #[inline(always)]
    unsafe fn prepare(taps: &[i16; 8]) -> Self::Filters {
        // Magnitudes are <=127 for nonidentity phases.
        taps.map(|tap| u8x16_splat(tap.unsigned_abs() as u8))
    }
    #[inline(always)]
    unsafe fn load<const N: usize>(src: *const u8) -> v128 {
        // SAFETY: the caller provides N readable bytes; each load reads N.
        unsafe {
            if N == 4 {
                v128_load32_zero(src.cast())
            } else {
                v128_load64_zero(src.cast())
            }
        }
    }
    #[inline(always)]
    unsafe fn store<const N: usize, const AVG: bool>(dst: *mut u8, row: v128) {
        // SAFETY: the caller provides N writable bytes, also readable when AVG; each store writes N.
        unsafe {
            let row = if AVG {
                u8x16_avgr(row, Self::load::<N>(dst))
            } else {
                row
            };
            if N == 4 {
                v128_store32_lane::<0>(row, dst.cast());
            } else {
                v128_store64_lane::<0>(row, dst.cast());
            }
        }
    }
    #[inline(always)]
    unsafe fn horizontal<const N: usize, const LAYOUT: u8>(src: *const u8, f: &[v128; 8]) -> v128 {
        // SAFETY: the caller provides N + 7 readable bytes: tap k reads k..k + N.
        unsafe { filter::<LAYOUT>(|k| Self::load::<N>(src.add(k)), f) }
    }
    #[inline(always)]
    unsafe fn vertical<const N: usize, const LAYOUT: u8>(rows: &[v128; 8], f: &[v128; 8]) -> v128 {
        filter::<LAYOUT>(|k| rows[k], f)
    }
}
#[target_feature(enable = "simd128")]
pub(super) unsafe fn convolve(
    kind: Vp9Filter,
    src: &[u8],
    stride: usize,
    phase: (usize, usize),
    size: (usize, usize),
    dst: &mut [u8],
    dst_stride: usize,
    average: bool,
) {
    // SAFETY: dispatch validated the input and this entry enables SIMD128.
    unsafe {
        if average {
            super::driver::entry::<Simd128, true>(kind, src, stride, phase, size, dst, dst_stride);
        } else {
            super::driver::entry::<Simd128, false>(kind, src, stride, phase, size, dst, dst_stride);
        }
    }
}
