//! NEON VP9 quantization: four coefficients per vector, the exact quotients
//! by one doubling high multiply (`sqdmulh`) with the step's 31-bit
//! multiplier (module docs).

use super::{Vp9QuantStep, split};
use core::arch::aarch64::{
    vabsq_s32, vaddq_u32, vdupq_n_s32, vdupq_n_u32, veorq_s32, vld1q_s32, vmaxq_u32, vmaxvq_u32,
    vminq_u32, vorrq_u32, vqdmulhq_s32, vreinterpretq_s32_u32, vreinterpretq_u32_s32, vshlq_u32,
    vshrq_n_s32, vst1q_s32, vsubq_s32,
};

/// Arithmetic shift giving each lane's sign mask (all ones if negative).
const SIGN_SHIFT: i32 = 31;

#[target_feature(enable = "neon")]
pub(super) unsafe fn quantize(
    coefficients: &[i32],
    dc: &Vp9QuantStep,
    ac: &Vp9QuantStep,
    levels: &mut [i32],
) -> bool {
    split::<4>(coefficients, dc, ac, levels, |input, output| {
        // SAFETY: NEON is enabled; every load and store is one whole
        // four-lane chunk of the checked slices.
        unsafe {
            let multiplier = vdupq_n_s32(ac.high_multiplier);
            let offset = vdupq_n_u32(ac.offset);
            let cap = vdupq_n_u32(ac.cap);
            let shift = vdupq_n_s32(ac.shift as i32);
            let mut any = vdupq_n_u32(0);
            for (c, level) in input.iter().zip(output) {
                let c = vld1q_s32(c.as_ptr());
                let magnitude = vreinterpretq_u32_s32(vabsq_s32(c));
                let numerator = vaddq_u32(vshlq_u32(magnitude, shift), offset);
                // Numerators are below 2^18: non-negative as i32, and the
                // doubling product never saturates.
                let quotient = vreinterpretq_u32_s32(vqdmulhq_s32(
                    vreinterpretq_s32_u32(numerator),
                    multiplier,
                ));
                let capped = vminq_u32(quotient, cap);
                any = vorrq_u32(any, capped);
                let sign = vshrq_n_s32::<SIGN_SHIFT>(c);
                let signed = vsubq_s32(veorq_s32(vreinterpretq_s32_u32(capped), sign), sign);
                vst1q_s32(level.as_mut_ptr(), signed);
            }
            vmaxvq_u32(any) != 0
        }
    })
}

/// NEON [`super::MagnitudeFn`]: four absolute values per vector, an
/// unsigned running maximum, then `umaxv`.
#[target_feature(enable = "neon")]
pub(super) unsafe fn largest_magnitude(coefficients: &[i32]) -> u32 {
    let (whole, tail) = coefficients.as_chunks::<4>();
    // SAFETY: NEON is enabled; every load is one whole four-lane chunk.
    let vector = unsafe {
        let mut largest = vdupq_n_u32(0);
        for chunk in whole {
            let magnitude = vreinterpretq_u32_s32(vabsq_s32(vld1q_s32(chunk.as_ptr())));
            largest = vmaxq_u32(largest, magnitude);
        }
        vmaxvq_u32(largest)
    };
    // SAFETY: the scalar reference has no preconditions.
    vector.max(unsafe { super::largest_magnitude(tail) })
}
