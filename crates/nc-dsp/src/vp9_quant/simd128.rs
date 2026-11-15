//! SIMD128 VP9 quantization: eight 16-bit coefficients per load, quantized
//! as two four-lane 32-bit halves, the exact reciprocal products by
//! widening `extmul`, quotients by a lane shuffle.

use super::{Vp9QuantStep, split};
use core::arch::wasm32::{
    i16x8_abs, i32x4_abs, i32x4_add, i32x4_extend_high_i16x8, i32x4_extend_low_i16x8, i32x4_shl,
    i32x4_shr, i32x4_shuffle, i32x4_sub, u16x8_extract_lane, u16x8_max, u16x8_splat, u32x4_min,
    u32x4_splat, u64x2_extmul_high_u32x4, u64x2_extmul_low_u32x4, v128, v128_any_true, v128_load,
    v128_or, v128_store, v128_xor,
};

/// Arithmetic shift giving each lane's sign mask (all ones if negative).
const SIGN_SHIFT: u32 = 31;
/// Coefficients per 128-bit load.
const LANES: usize = 8;
/// 32-bit lanes per half.
const HALF: usize = LANES / 2;

pub(super) unsafe fn quantize(
    coefficients: &[i16],
    dc: &Vp9QuantStep,
    ac: &Vp9QuantStep,
    levels: &mut [i32],
) -> bool {
    split::<LANES>(coefficients, dc, ac, levels, |input, output| {
        let reciprocal = u32x4_splat(ac.reciprocal);
        let offset = u32x4_splat(ac.offset);
        let cap = u32x4_splat(ac.cap);
        let mut any = u32x4_splat(0);
        // The signed level of four 32-bit coefficients; `any` collects the
        // magnitudes.
        let level = |c: v128, any: &mut v128| {
            let numerator = i32x4_add(i32x4_shl(i32x4_abs(c), ac.shift), offset);
            let low = u64x2_extmul_low_u32x4(numerator, reciprocal);
            let high = u64x2_extmul_high_u32x4(numerator, reciprocal);
            // Each product's high half is its quotient.
            let quotient = i32x4_shuffle::<1, 3, 5, 7>(low, high);
            let capped = u32x4_min(quotient, cap);
            *any = v128_or(*any, capped);
            let sign = i32x4_shr(c, SIGN_SHIFT);
            i32x4_sub(v128_xor(capped, sign), sign)
        };
        for (c, out) in input.iter().zip(output) {
            // SAFETY: SIMD128 is enabled for the whole build; the load is one
            // whole chunk and each store one half of the checked slices.
            unsafe {
                let c = v128_load(c.as_ptr().cast());
                v128_store(
                    out.as_mut_ptr().cast(),
                    level(i32x4_extend_low_i16x8(c), &mut any),
                );
                v128_store(
                    out.as_mut_ptr().add(HALF).cast(),
                    level(i32x4_extend_high_i16x8(c), &mut any),
                );
            }
        }
        v128_any_true(any)
    })
}

/// SIMD128 [`super::MagnitudeFn`]: eight absolute values per vector and an
/// unsigned 16-bit running maximum (`|i16::MIN|` reads as 32768), reduced
/// lane by lane.
pub(super) unsafe fn largest_magnitude(coefficients: &[i16]) -> u32 {
    let (whole, tail) = coefficients.as_chunks::<LANES>();
    // SAFETY: simd128 is enabled at compile time; every load is one whole
    // eight-lane chunk.
    let vector = unsafe {
        let mut largest = u16x8_splat(0);
        for chunk in whole {
            largest = u16x8_max(largest, i16x8_abs(v128_load(chunk.as_ptr().cast())));
        }
        [
            u16x8_extract_lane::<0>(largest),
            u16x8_extract_lane::<1>(largest),
            u16x8_extract_lane::<2>(largest),
            u16x8_extract_lane::<3>(largest),
            u16x8_extract_lane::<4>(largest),
            u16x8_extract_lane::<5>(largest),
            u16x8_extract_lane::<6>(largest),
            u16x8_extract_lane::<7>(largest),
        ]
        .into_iter()
        .max()
        .map_or(0, u32::from)
    };
    // SAFETY: the scalar reference has no preconditions.
    vector.max(unsafe { super::largest_magnitude(tail) })
}
