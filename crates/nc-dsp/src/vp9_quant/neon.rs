//! NEON VP9 quantization: eight 16-bit coefficients per load, quantized as
//! two four-lane 32-bit halves, the exact quotients by one doubling high
//! multiply (`sqdmulh`) with the step's 31-bit multiplier (module docs).

use super::{Vp9QuantStep, split};
use core::arch::aarch64::{
    int16x4_t, uint32x4_t, vabsq_s16, vabsq_s32, vaddq_u32, vdupq_n_s32, vdupq_n_u16, vdupq_n_u32,
    veorq_s32, vget_high_s16, vget_low_s16, vld1q_s16, vmaxq_u16, vmaxvq_u16, vmaxvq_u32,
    vminq_u32, vmovl_s16, vorrq_u32, vqdmulhq_s32, vreinterpretq_s32_u32, vreinterpretq_u16_s16,
    vreinterpretq_u32_s32, vshlq_u32, vshrq_n_s32, vst1q_s32, vsubq_s32,
};

/// Arithmetic shift giving each lane's sign mask (all ones if negative).
const SIGN_SHIFT: i32 = 31;
/// Coefficients per 128-bit load.
const LANES: usize = 8;
/// 32-bit lanes per half.
const HALF: usize = LANES / 2;

#[target_feature(enable = "neon")]
pub(super) unsafe fn quantize(
    coefficients: &[i16],
    dc: &Vp9QuantStep,
    ac: &Vp9QuantStep,
    levels: &mut [i32],
) -> bool {
    split::<LANES>(coefficients, dc, ac, levels, |input, output| {
        // SAFETY: NEON is enabled; every load and store is one whole
        // chunk (or half chunk) of the checked slices.
        unsafe {
            let multiplier = vdupq_n_s32(ac.high_multiplier);
            let offset = vdupq_n_u32(ac.offset);
            let cap = vdupq_n_u32(ac.cap);
            let shift = vdupq_n_s32(ac.shift as i32);
            let mut any = vdupq_n_u32(0);
            // The signed level of four coefficients; `any` collects the
            // magnitudes.
            let level = |c: int16x4_t, any: &mut uint32x4_t| {
                let c = vmovl_s16(c);
                let magnitude = vreinterpretq_u32_s32(vabsq_s32(c));
                let numerator = vaddq_u32(vshlq_u32(magnitude, shift), offset);
                // Numerators are below 2^18: non-negative as i32, and the
                // doubling product never saturates.
                let quotient = vreinterpretq_u32_s32(vqdmulhq_s32(
                    vreinterpretq_s32_u32(numerator),
                    multiplier,
                ));
                let capped = vminq_u32(quotient, cap);
                *any = vorrq_u32(*any, capped);
                let sign = vshrq_n_s32::<SIGN_SHIFT>(c);
                vsubq_s32(veorq_s32(vreinterpretq_s32_u32(capped), sign), sign)
            };
            for (c, out) in input.iter().zip(output) {
                let c = vld1q_s16(c.as_ptr());
                vst1q_s32(out.as_mut_ptr(), level(vget_low_s16(c), &mut any));
                vst1q_s32(
                    out.as_mut_ptr().add(HALF),
                    level(vget_high_s16(c), &mut any),
                );
            }
            vmaxvq_u32(any) != 0
        }
    })
}

/// NEON [`super::MagnitudeFn`]: eight absolute values per vector, an
/// unsigned 16-bit running maximum (`|i16::MIN|` reads as 32768), then
/// `umaxv`.
#[target_feature(enable = "neon")]
pub(super) unsafe fn largest_magnitude(coefficients: &[i16]) -> u32 {
    let (whole, tail) = coefficients.as_chunks::<LANES>();
    // SAFETY: NEON is enabled; every load is one whole eight-lane chunk.
    let vector = unsafe {
        let mut largest = vdupq_n_u16(0);
        for chunk in whole {
            let magnitude = vreinterpretq_u16_s16(vabsq_s16(vld1q_s16(chunk.as_ptr())));
            largest = vmaxq_u16(largest, magnitude);
        }
        u32::from(vmaxvq_u16(largest))
    };
    // SAFETY: the scalar reference has no preconditions.
    vector.max(unsafe { super::largest_magnitude(tail) })
}
