//! SIMD128 Gram rows: eight values per vector, `i32x4.dot_i16x8_s`
//! products summed in four 32-bit lanes per pair, reduced in 64 bits once
//! per pair.

use super::GRAM_ROW_ALIGN;
use core::arch::wasm32::{
    i32x4_add, i32x4_dot_i16x8, i32x4_extract_lane, i32x4_splat, v128, v128_load,
};

/// Values per vector.
const LANES: usize = 8;

/// Eight values of a row.
#[target_feature(enable = "simd128")]
fn load(values: &[i16; LANES]) -> v128 {
    // SAFETY: SIMD128 is enabled; the load reads exactly the 16 bytes of `values`.
    unsafe { v128_load(values.as_ptr().cast()) }
}

/// The sum of the four 32-bit lanes of `sum`, in 64 bits.
#[target_feature(enable = "simd128")]
fn reduce(sum: v128) -> i64 {
    i64::from(i32x4_extract_lane::<0>(sum))
        + i64::from(i32x4_extract_lane::<1>(sum))
        + i64::from(i32x4_extract_lane::<2>(sum))
        + i64::from(i32x4_extract_lane::<3>(sum))
}

/// SIMD128 [`super::GramRowsFn`]: each 32-bit lane takes two products of
/// every eighth column, so it sums at most 2^10 products of at most 2^20.
#[target_feature(enable = "simd128")]
pub(super) unsafe fn gram_rows(rows: &[&[i16]], out: &mut [i64]) {
    debug_assert!(GRAM_ROW_ALIGN.is_multiple_of(LANES));
    let mut entry = 0;
    for (i, first) in rows.iter().enumerate() {
        let (a, _) = first.as_chunks::<LANES>();
        for second in &rows[i..] {
            let (b, _) = second.as_chunks::<LANES>();
            let mut sum = i32x4_splat(0);
            for (a, b) in a.iter().zip(b) {
                sum = i32x4_add(sum, i32x4_dot_i16x8(load(a), load(b)));
            }
            out[entry] += reduce(sum);
            entry += 1;
        }
    }
}
