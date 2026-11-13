//! SIMD128 VP9 quantization: four coefficients per vector, the exact
//! reciprocal products by widening `extmul`, quotients by a lane shuffle.

use super::{Vp9QuantStep, split};
use core::arch::wasm32::{
    i32x4_abs, i32x4_add, i32x4_shl, i32x4_shr, i32x4_shuffle, i32x4_sub, u32x4_extract_lane,
    u32x4_max, u32x4_min, u32x4_splat, u64x2_extmul_high_u32x4, u64x2_extmul_low_u32x4,
    v128_any_true, v128_load, v128_or, v128_store, v128_xor,
};

/// Arithmetic shift giving each lane's sign mask (all ones if negative).
const SIGN_SHIFT: u32 = 31;

pub(super) unsafe fn quantize(
    coefficients: &[i32],
    dc: &Vp9QuantStep,
    ac: &Vp9QuantStep,
    levels: &mut [i32],
) -> bool {
    split::<4>(coefficients, dc, ac, levels, |input, output| {
        let reciprocal = u32x4_splat(ac.reciprocal);
        let offset = u32x4_splat(ac.offset);
        let cap = u32x4_splat(ac.cap);
        let mut any = u32x4_splat(0);
        for (c, level) in input.iter().zip(output) {
            // SAFETY: SIMD128 is enabled for the whole build; the load and
            // store are one whole four-lane chunk of the checked slices.
            unsafe {
                let c = v128_load(c.as_ptr().cast());
                let numerator = i32x4_add(i32x4_shl(i32x4_abs(c), ac.shift), offset);
                let low = u64x2_extmul_low_u32x4(numerator, reciprocal);
                let high = u64x2_extmul_high_u32x4(numerator, reciprocal);
                // Each product's high half is its quotient.
                let quotient = i32x4_shuffle::<1, 3, 5, 7>(low, high);
                let capped = u32x4_min(quotient, cap);
                any = v128_or(any, capped);
                let sign = i32x4_shr(c, SIGN_SHIFT);
                v128_store(
                    level.as_mut_ptr().cast(),
                    i32x4_sub(v128_xor(capped, sign), sign),
                );
            }
        }
        v128_any_true(any)
    })
}

/// SIMD128 [`super::MagnitudeFn`]: four absolute values per vector and an
/// unsigned running maximum, reduced lane by lane.
pub(super) unsafe fn largest_magnitude(coefficients: &[i32]) -> u32 {
    let (whole, tail) = coefficients.as_chunks::<4>();
    // SAFETY: simd128 is enabled at compile time; every load is one whole
    // four-lane chunk.
    let vector = unsafe {
        let mut largest = u32x4_splat(0);
        for chunk in whole {
            largest = u32x4_max(largest, i32x4_abs(v128_load(chunk.as_ptr().cast())));
        }
        u32x4_extract_lane::<0>(largest)
            .max(u32x4_extract_lane::<1>(largest))
            .max(u32x4_extract_lane::<2>(largest))
            .max(u32x4_extract_lane::<3>(largest))
    };
    // SAFETY: the scalar reference has no preconditions.
    vector.max(unsafe { super::largest_magnitude(tail) })
}
