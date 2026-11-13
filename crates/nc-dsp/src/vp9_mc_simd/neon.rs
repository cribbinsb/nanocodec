//! NEON VP9 interpolation with widening byte products of tap magnitudes.
//! Positive and negative products accumulate separately in u16 (each sum fits;
//! `positive_and_negative_sums_fit_u16`). Their saturating difference is the
//! filter sum clipped below at zero, so a rounding saturating narrow gives the
//! reference's round-shift-clip exactly, for every table.

use super::driver::{RowKernel, SHIFT, layout};
use crate::Vp9Filter;
use core::arch::aarch64::{
    uint8x8_t, vdup_n_u8, vdup_n_u32, vdupq_n_u16, vld1_lane_u32, vld1_u8, vmlal_u8, vmull_u8,
    vqrshrn_n_u16, vqsubq_u16, vreinterpret_u8_u32, vreinterpret_u32_u8, vrhadd_u8, vst1_lane_u32,
    vst1_u8,
};
struct Neon;

/// N (at most 8) outputs from the eight tap inputs `pixels(k)`.
#[inline(always)]
unsafe fn filter<const LAYOUT: u8>(
    pixels: impl Fn(usize) -> uint8x8_t,
    f: &[uint8x8_t; 8],
) -> uint8x8_t {
    // SAFETY: the ISA entry enables NEON.
    unsafe {
        if LAYOUT == layout::BILINEAR {
            // Both centre taps are positive and sum to 128: no u16 overflow.
            vqrshrn_n_u16::<{ SHIFT as i32 }>(vmlal_u8(vmull_u8(pixels(3), f[3]), pixels(4), f[4]))
        } else {
            // Two independent chains; the zero starts fold into multiplies.
            let (mut positive, mut negative) = (vdupq_n_u16(0), vdupq_n_u16(0));
            for (k, &magnitude) in f.iter().enumerate() {
                if layout::negative(LAYOUT, k) {
                    negative = vmlal_u8(negative, pixels(k), magnitude);
                } else {
                    positive = vmlal_u8(positive, pixels(k), magnitude);
                }
            }
            vqrshrn_n_u16::<{ SHIFT as i32 }>(vqsubq_u16(positive, negative))
        }
    }
}

impl RowKernel for Neon {
    type Filters = [uint8x8_t; 8];
    type Row = uint8x8_t;
    const LANES: usize = 8;
    #[inline(always)]
    unsafe fn prepare<const LAYOUT: u8>(taps: &[i16; 8]) -> Self::Filters {
        // SAFETY: NEON is enabled by the entry point; magnitudes are <=127.
        unsafe { taps.map(|tap| vdup_n_u8(tap.unsigned_abs() as u8)) }
    }
    #[inline(always)]
    unsafe fn load<const N: usize>(src: *const u8) -> uint8x8_t {
        // SAFETY: the caller provides N readable bytes; N=4 uses a four-byte
        // lane load rather than overreading eight bytes.
        unsafe {
            if N == 4 {
                vreinterpret_u8_u32(vld1_lane_u32::<0>(src.cast(), vdup_n_u32(0)))
            } else {
                vld1_u8(src)
            }
        }
    }
    #[inline(always)]
    unsafe fn store<const N: usize, const AVG: bool>(dst: *mut u8, row: uint8x8_t) {
        // SAFETY: the caller provides N writable bytes, also readable when AVG; each store writes N.
        unsafe {
            let row = if AVG {
                vrhadd_u8(row, Self::load::<N>(dst))
            } else {
                row
            };
            if N == 4 {
                vst1_lane_u32::<0>(dst.cast(), vreinterpret_u32_u8(row));
            } else {
                vst1_u8(dst, row);
            }
        }
    }
    #[inline(always)]
    unsafe fn horizontal<const N: usize, const LAYOUT: u8>(
        src: *const u8,
        f: &Self::Filters,
    ) -> uint8x8_t {
        // SAFETY: the caller provides N + 7 readable bytes: tap k reads k..k + N.
        unsafe { filter::<LAYOUT>(|k| Self::load::<N>(src.add(k)), f) }
    }
    #[inline(always)]
    unsafe fn vertical<const N: usize, const LAYOUT: u8>(
        rows: &[uint8x8_t; 8],
        f: &Self::Filters,
    ) -> uint8x8_t {
        // SAFETY: the ISA entry enables NEON.
        unsafe { filter::<LAYOUT>(|k| rows[k], f) }
    }
}
#[target_feature(enable = "neon")]
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
    // SAFETY: dispatch validated the input and this entry enables NEON.
    unsafe {
        if average {
            super::driver::entry::<Neon, true>(kind, src, stride, phase, size, dst, dst_stride);
        } else {
            super::driver::entry::<Neon, false>(kind, src, stride, phase, size, dst, dst_stride);
        }
    }
}
