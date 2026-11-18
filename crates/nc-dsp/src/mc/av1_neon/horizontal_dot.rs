//! dav1d 1.5.4 `mc_dotprod.S` horizontal 16-output schedule: five shared
//! shuffles feed eight by-element dot products. BSD-2-Clause, VideoLAN/dav1d
//! authors; see `THIRD_PARTY.md`. Rounding is folded into the dot bias.
//!
//! H-only accesses the middle h rows of the validated h+7-row allocation.
//! Unused vector lanes may cross a row's right edge, but remain within that
//! allocation: even the final filtered row has four source rows after it.

use super::{Output, TAP_CENTRE};
use crate::mc::neon_dotprod::{GATHER, dot_lane};
use core::arch::aarch64::{
    int8x8_t, int8x16_t, int16x8_t, int32x4_t, vdupq_n_s32, vdupq_n_u8, veorq_u8, vld1q_s16,
    vld1q_u8, vmovn_s16, vqtbl1q_s8, vreinterpretq_s8_u8, vreinterpretq_s16_s32, vuzp1q_s16,
};

#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn sum(a: int8x16_t, b: int8x16_t, coefficients: int8x8_t, bias: int32x4_t) -> int32x4_t {
    // SAFETY: dotprod is enabled and all operands are initialized registers.
    unsafe { dot_lane::<1>(dot_lane::<0>(bias, a, coefficients), b, coefficients) }
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn words(a: int32x4_t, b: int32x4_t) -> int16x8_t {
    vuzp1q_s16(vreinterpretq_s16_s32(a), vreinterpretq_s16_s32(b))
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn filter<const W: usize, O: Output>(
    src: *const u8,
    ss: usize,
    h: usize,
    taps: &[i16; 8],
    out: &mut O,
) {
    // SAFETY: the parent checked source/output footprints. The latest load
    // ends at (h+2)*ss+W+12, below (h+6)*ss+W+7 because ss>=W+7>=11.
    unsafe {
        let coefficients = vmovn_s16(vld1q_s16(taps.as_ptr()));
        let bias = vdupq_n_s32(O::HORIZONTAL_BIAS);
        let sign = vdupq_n_u8(128);
        let g0 = vld1q_u8(GATHER[0].as_ptr());
        let g4 = vld1q_u8(GATHER[1].as_ptr());
        let g8 = vld1q_u8(GATHER[2].as_ptr());
        for row in 0..h {
            let origin = src.add((row + TAP_CENTRE) * ss);
            for col in (0..W).step_by(W.min(16)) {
                let a = vreinterpretq_s8_u8(veorq_u8(vld1q_u8(origin.add(col)), sign));
                let first = vqtbl1q_s8(a, g0);
                let middle = vqtbl1q_s8(a, g4);
                let s0 = sum(first, middle, coefficients, bias);
                if W == 4 {
                    out.horizontal(row, col, words(s0, s0));
                } else {
                    let last = vqtbl1q_s8(a, g8);
                    let s1 = sum(middle, last, coefficients, bias);
                    out.horizontal(row, col, words(s0, s1));
                    if W >= 16 {
                        let b = vreinterpretq_s8_u8(veorq_u8(vld1q_u8(origin.add(col + 12)), sign));
                        let next = vqtbl1q_s8(b, g0);
                        let end = vqtbl1q_s8(b, g4);
                        let s2 = sum(last, next, coefficients, bias);
                        let s3 = sum(next, end, coefficients, bias);
                        out.horizontal(row, col + 8, words(s2, s3));
                    }
                }
            }
        }
    }
}
