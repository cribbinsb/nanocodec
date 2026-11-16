//! SIMD128 largest coefficient magnitude (VP9's all-zero test).

use core::arch::wasm32::{i16x8_abs, u16x8_extract_lane, u16x8_max, u16x8_splat, v128_load};

/// Coefficients per 128-bit load.
const LANES: usize = 8;

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
