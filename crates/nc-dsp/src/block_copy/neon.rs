//! NEON rows: sixteen bytes per move, and sixteen widened pixels per load.

use super::rows::Ops;
use core::arch::aarch64::{
    vcreate_u8, vdupq_n_s16, vget_low_s16, vget_low_u8, vld1_u8, vld1q_u8, vreinterpret_u32_u8,
    vreinterpretq_s16_u16, vrhadd_u8, vrhaddq_u8, vshll_high_n_u8, vshll_n_u8, vst1_lane_u32,
    vst1_s16, vst1_u8, vst1q_s16, vst1q_u8, vsubq_s16,
};

pub(super) struct Neon;
impl Ops for Neon {
    const COPY_BYTES: usize = 16;
    const PREP_BYTES: usize = 16;
    #[inline(never)]
    #[target_feature(enable = "neon")]
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
    #[target_feature(enable = "neon")]
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
    #[target_feature(enable = "neon")]
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
    #[target_feature(enable = "neon")]
    unsafe fn copy<const N: usize, const AVG: bool>(src: *const u8, dst: *mut u8) {
        // SAFETY: all loads/stores access exactly N initialized samples.
        unsafe {
            if N == 16 {
                let a = vld1q_u8(src);
                let a = if AVG { vrhaddq_u8(a, vld1q_u8(dst)) } else { a };
                vst1q_u8(dst, a);
            } else if N == 8 {
                let a = vld1_u8(src);
                let a = if AVG { vrhadd_u8(a, vld1_u8(dst)) } else { a };
                vst1_u8(dst, a);
            } else if N == 4 {
                let a = vcreate_u8(u64::from(src.cast::<u32>().read_unaligned()));
                let a = if AVG {
                    vrhadd_u8(a, vcreate_u8(u64::from(dst.cast::<u32>().read_unaligned())))
                } else {
                    a
                };
                vst1_lane_u32::<0>(dst.cast(), vreinterpret_u32_u8(a));
            } else {
                super::scalar::Scalar::copy::<N, AVG>(src, dst);
            }
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn prep<const N: usize, const SHIFT: i32>(src: *const u8, dst: *mut i16, bias: i16) {
        // SAFETY: N bytes and N words are covered; narrow loads never read
        // unused row padding. The caller guarantees scaled values fit i16.
        unsafe {
            if N == 16 {
                let a = vld1q_u8(src);
                let b = vdupq_n_s16(bias);
                let lo = vreinterpretq_s16_u16(vshll_n_u8::<SHIFT>(vget_low_u8(a)));
                let hi = vreinterpretq_s16_u16(vshll_high_n_u8::<SHIFT>(a));
                vst1q_s16(dst, vsubq_s16(lo, b));
                vst1q_s16(dst.add(8), vsubq_s16(hi, b));
            } else if N == 8 || N == 4 {
                let a = if N == 8 {
                    vld1_u8(src)
                } else {
                    vcreate_u8(u64::from(src.cast::<u32>().read_unaligned()))
                };
                let a = vreinterpretq_s16_u16(vshll_n_u8::<SHIFT>(a));
                let a = vsubq_s16(a, vdupq_n_s16(bias));
                if N == 8 {
                    vst1q_s16(dst, a);
                } else {
                    vst1_s16(dst, vget_low_s16(a));
                }
            } else {
                super::scalar::Scalar::prep::<N, SHIFT>(src, dst, bias);
            }
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn copy<const AVG: bool>(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) {
    // SAFETY: forwarded validated rectangle; AArch64 provides NEON.
    unsafe { super::rows::copy::<Neon, AVG>(src, ss, dst, ds, w, h) }
}

#[inline]
#[target_feature(enable = "neon")]
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
    // SAFETY: validated rectangle; shift <= 7 and scaled samples fit i16.
    unsafe { super::rows::prep_entry!(Neon, src, ss, dst, ds, w, h, shift, bias) }
}

/// Common eight-bit AV1/AV2 compound precision: unbiased sample << 4.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn prep_unbiased(
    src: &[u8],
    ss: usize,
    dst: &mut [i16],
    ds: usize,
    (w, h): (usize, usize),
) {
    // SAFETY: validated rectangle, byte << 4 fits i16, and this entry's ISA is enabled.
    unsafe { super::rows::prep_unbiased::<Neon>(src, ss, dst, ds, w, h) }
}
