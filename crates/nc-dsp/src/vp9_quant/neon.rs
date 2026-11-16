//! NEON largest coefficient magnitude (VP9's all-zero test).

use core::arch::aarch64::{
    vabsq_s16, vdupq_n_u16, vld1q_s16, vmaxq_u16, vmaxvq_u16, vreinterpretq_u16_s16,
};

/// Coefficients per 128-bit load.
const LANES: usize = 8;

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
