//! SIMD128 end of block: 8 levels per step, the table masked by
//! `level != 0`, a running unsigned maximum, then the lanes' maximum.

use core::arch::wasm32::{
    i16x8_narrow_i32x4, i16x8_ne, i16x8_splat, i32x4_abs, i32x4_extract_lane, i32x4_ne,
    i32x4_splat, u16x8_extract_lane, u16x8_max, u16x8_splat, v128, v128_and, v128_load, v128_or,
};

/// Levels per step.
const LANES: usize = 8;

/// The SIMD128 [`super::EobFn`].
///
/// # Safety
///
/// None beyond the build's SIMD128.
pub(super) unsafe fn end_of_block(levels: &[i16], table: &[u16]) -> usize {
    let n = levels.len().min(table.len());
    let (whole, _) = levels[..n].as_chunks::<LANES>();
    let (ends, _) = table[..n].as_chunks::<LANES>();
    let mut largest = u16x8_splat(0);
    for (chunk, end) in whole.iter().zip(ends) {
        // SAFETY: each load is one whole 8-element chunk.
        let (level, end) = unsafe {
            (
                v128_load(chunk.as_ptr().cast()),
                v128_load(end.as_ptr().cast()),
            )
        };
        let coded = i16x8_ne(level, i16x8_splat(0));
        largest = u16x8_max(largest, v128_and(coded, end));
    }
    let done = whole.len() * LANES;
    let vector = maximum(largest);
    vector.max(super::scalar(&levels[done..n], &table[done..n]))
}

/// The largest unsigned 16-bit lane.
fn maximum(lanes: v128) -> usize {
    [
        u16x8_extract_lane::<0>(lanes),
        u16x8_extract_lane::<1>(lanes),
        u16x8_extract_lane::<2>(lanes),
        u16x8_extract_lane::<3>(lanes),
        u16x8_extract_lane::<4>(lanes),
        u16x8_extract_lane::<5>(lanes),
        u16x8_extract_lane::<6>(lanes),
        u16x8_extract_lane::<7>(lanes),
    ]
    .into_iter()
    .max()
    .map_or(0, usize::from)
}

/// The SIMD128 [`super::WideEobFn`]: 8 levels per step as two 4-lane
/// loads, whose nonzero masks narrow onto the 8 table entries.
///
/// # Safety
///
/// None beyond the build's SIMD128.
pub(super) unsafe fn end_of_block_wide(levels: &[i32], table: &[u16]) -> (usize, u32) {
    let n = levels.len().min(table.len());
    let (whole, _) = levels[..n].as_chunks::<LANES>();
    let (ends, _) = table[..n].as_chunks::<LANES>();
    let zero = i32x4_splat(0);
    let (mut largest, mut bits) = (u16x8_splat(0), zero);
    for (chunk, end) in whole.iter().zip(ends) {
        // SAFETY: each load is one whole chunk or one half of it.
        let (low, high, end) = unsafe {
            (
                v128_load(chunk.as_ptr().cast()),
                v128_load(chunk[LANES / 2..].as_ptr().cast()),
                v128_load(end.as_ptr().cast()),
            )
        };
        let coded = i16x8_narrow_i32x4(i32x4_ne(low, zero), i32x4_ne(high, zero));
        largest = u16x8_max(largest, v128_and(coded, end));
        bits = v128_or(bits, v128_or(i32x4_abs(low), i32x4_abs(high)));
    }
    let done = whole.len() * LANES;
    let (end, tail_bits) = super::scalar_wide(&levels[done..n], &table[done..n]);
    let or = (i32x4_extract_lane::<0>(bits)
        | i32x4_extract_lane::<1>(bits)
        | i32x4_extract_lane::<2>(bits)
        | i32x4_extract_lane::<3>(bits)) as u32;
    (maximum(largest).max(end), or | tail_bits)
}
