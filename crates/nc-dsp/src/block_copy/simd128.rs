//! SIMD128 byte moves/averages and two eight-word halves per input vector.

use super::rows::Ops;
use core::arch::wasm32::{
    i16x8_shl, i16x8_splat, i16x8_sub, u8x16_avgr, u16x8_extend_high_u8x16, u16x8_extend_low_u8x16,
    v128_load, v128_load32_zero, v128_load64_zero, v128_store, v128_store32_lane,
    v128_store64_lane,
};

pub(super) struct Simd128;
impl Ops for Simd128 {
    const COPY_BYTES: usize = 16;
    const PREP_BYTES: usize = 16;
    const ROWS: usize = 1;
    #[inline(never)]
    #[target_feature(enable = "simd128")]
    unsafe fn copy_width<const W: usize, const AVG: bool>(
        src: &[u8],
        ss: usize,
        dst: &mut [u8],
        ds: usize,
        w: usize,
        h: usize,
    ) {
        // SAFETY: forwarded rectangle and ISA contract. Only this width's
        // loop can contribute register saves to its frame.
        unsafe { super::rows::copy_width::<Self, W, AVG>(src, ss, dst, ds, w, h) }
    }
    #[inline(never)]
    #[target_feature(enable = "simd128")]
    unsafe fn prep_width<const W: usize, const SHIFT: i32>(
        src: &[u8],
        ss: usize,
        dst: &mut [i16],
        ds: usize,
        w: usize,
        h: usize,
        bias: i16,
    ) {
        // SAFETY: validated footprint and bounded word precision.
        unsafe { super::rows::prep_width::<Self, W, SHIFT>(src, ss, dst, ds, w, h, bias) }
    }
    #[inline(never)]
    #[target_feature(enable = "simd128")]
    unsafe fn prep_unbiased_width<const W: usize>(
        src: &[u8],
        ss: usize,
        dst: &mut [i16],
        ds: usize,
        w: usize,
        h: usize,
    ) {
        // SAFETY: validated footprint; byte samples <<4 fit signed words.
        unsafe { super::rows::prep_width::<Self, W, 4>(src, ss, dst, ds, w, h, 0) }
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn copy<const N: usize, const AVG: bool>(src: *const u8, dst: *mut u8) {
        // SAFETY: exact-width loads and stores cover N bytes, never padding.
        unsafe {
            if N == 16 {
                let a = v128_load(src.cast());
                let a = if AVG {
                    u8x16_avgr(a, v128_load(dst.cast()))
                } else {
                    a
                };
                v128_store(dst.cast(), a);
            } else if N == 8 {
                let a = v128_load64_zero(src.cast());
                let a = if AVG {
                    u8x16_avgr(a, v128_load64_zero(dst.cast()))
                } else {
                    a
                };
                v128_store64_lane::<0>(a, dst.cast());
            } else if N == 4 {
                let a = v128_load32_zero(src.cast());
                let a = if AVG {
                    u8x16_avgr(a, v128_load32_zero(dst.cast()))
                } else {
                    a
                };
                v128_store32_lane::<0>(a, dst.cast());
            } else {
                super::scalar::Scalar::copy::<N, AVG>(src, dst);
            }
        }
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn prep<const N: usize, const SHIFT: i32>(src: *const u8, dst: *mut i16, bias: i16) {
        // SAFETY: the caller bounds both rectangles and the scaled words.
        unsafe {
            if N == 16 {
                let a = v128_load(src.cast());
                let b = i16x8_splat(bias);
                let lo = i16x8_sub(i16x8_shl(u16x8_extend_low_u8x16(a), SHIFT as u32), b);
                let hi = i16x8_sub(i16x8_shl(u16x8_extend_high_u8x16(a), SHIFT as u32), b);
                v128_store(dst.cast(), lo);
                v128_store(dst.add(8).cast(), hi);
            } else if N == 8 || N == 4 {
                let a = if N == 8 {
                    v128_load64_zero(src.cast())
                } else {
                    v128_load32_zero(src.cast())
                };
                let a = i16x8_sub(
                    i16x8_shl(u16x8_extend_low_u8x16(a), SHIFT as u32),
                    i16x8_splat(bias),
                );
                if N == 8 {
                    v128_store(dst.cast(), a);
                } else {
                    v128_store64_lane::<0>(a, dst.cast());
                }
            } else {
                super::scalar::Scalar::prep::<N, SHIFT>(src, dst, bias);
            }
        }
    }
}

#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn copy<const AVG: bool>(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) {
    // SAFETY: validated rectangle; this build enables SIMD128.
    unsafe { super::rows::copy::<Simd128, AVG>(src, ss, dst, ds, w, h) }
}

#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn prep(
    src: &[u8],
    ss: usize,
    dst: &mut [i16],
    ds: usize,
    size: (usize, usize),
    scale: (u32, i16),
) {
    let (w, h) = size;
    let (shift, bias) = scale;
    // SAFETY: validated rectangle and precision; SIMD128 is enabled.
    unsafe { super::rows::prep_entry!(Simd128, src, ss, dst, ds, w, h, shift, bias) }
}

/// Common eight-bit AV1/AV2 compound precision: unbiased sample << 4.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn prep_unbiased(
    src: &[u8],
    ss: usize,
    dst: &mut [i16],
    ds: usize,
    (w, h): (usize, usize),
) {
    // SAFETY: validated rectangle, byte << 4 fits i16, and this entry's ISA is enabled.
    unsafe { super::rows::prep_unbiased::<Simd128>(src, ss, dst, ds, w, h) }
}
