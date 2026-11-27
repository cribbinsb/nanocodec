//! SIMD128 network inference. Accumulator: eight vectors of eight sums held
//! across the row list. Dense: signed extended byte multiplies of the low
//! and high halves added in 16 bits (exact, see the module doc) and widened
//! by `i32x4.extadd_pairwise_i16x8_s`, four outputs to each 16-byte input
//! load and a shuffled transpose to reduce them. Activations: an arithmetic
//! shift, signed-saturating narrows, then a maximum with zero.

use super::layout::{COLUMN_BLOCK, DENSE_GROUP, column};
use core::arch::wasm32::{
    i8x16_max, i8x16_narrow_i16x8, i8x16_splat, i16x8_add, i16x8_extmul_high_i8x16,
    i16x8_extmul_low_i8x16, i16x8_narrow_i32x4, i16x8_shr, i16x8_splat, i16x8_sub, i32x4_add,
    i32x4_extadd_pairwise_i16x8, i32x4_extract_lane, i32x4_shr, i32x4_shuffle, i32x4_splat,
    i64x2_extract_lane, v128, v128_load, v128_store,
};

/// Accumulator sums per vector.
const LANES: usize = 8;
/// Vectors of an accumulator block.
const BLOCK_VECTORS: usize = COLUMN_BLOCK / LANES;
/// Accumulator sums per activation step (two vectors narrow to one).
const CLIP_STEP: usize = 2 * LANES;
/// Inputs (bytes) per vector.
const BYTES: usize = 16;
/// 32-bit sums per vector.
const SUMS: usize = 4;
/// 32-bit sums per activation step (two vectors narrow to eight bytes).
const SUM_STEP: usize = 2 * SUMS;

/// One vector of `N` values of type `T` (16 bytes).
#[inline]
#[target_feature(enable = "simd128")]
fn load<T, const N: usize>(values: &[T; N]) -> v128 {
    const { assert!(size_of::<[T; N]>() == size_of::<v128>()) };
    // SAFETY: the load reads exactly the 16 bytes of `values`.
    unsafe { v128_load(values.as_ptr().cast()) }
}

/// Write one vector of `N` values of type `T` (16 bytes).
#[inline]
#[target_feature(enable = "simd128")]
fn store<T, const N: usize>(values: &mut [T; N], vector: v128) {
    const { assert!(size_of::<[T; N]>() == size_of::<v128>()) };
    // SAFETY: the store writes exactly the 16 bytes of `values`.
    unsafe { v128_store(values.as_mut_ptr().cast(), vector) }
}

/// `sums ± column`, 16-bit wrapping.
#[inline]
#[target_feature(enable = "simd128")]
fn combine<const SUBTRACT: bool>(sums: v128, column: v128) -> v128 {
    if SUBTRACT {
        i16x8_sub(sums, column)
    } else {
        i16x8_add(sums, column)
    }
}

/// The SIMD128 [`super::ColumnsFn`], adding or subtracting.
///
/// # Safety
///
/// None beyond the build's SIMD128.
#[target_feature(enable = "simd128")]
pub(super) unsafe fn update_columns<const SUBTRACT: bool>(
    acc: &mut [i16],
    columns: &[i16],
    rows: &[u16],
) {
    let width = acc.len();
    let (blocks, rest) = acc.as_chunks_mut::<COLUMN_BLOCK>();
    let blocks_end = blocks.len() * COLUMN_BLOCK;
    for (index, block) in blocks.iter_mut().enumerate() {
        let start = index * COLUMN_BLOCK;
        let (vectors, _) = block.as_chunks_mut::<LANES>();
        let mut sums = [i16x8_splat(0); BLOCK_VECTORS];
        for (sums, lanes) in sums.iter_mut().zip(vectors.iter()) {
            *sums = load(lanes);
        }
        for &row in rows {
            let values: &[i16; COLUMN_BLOCK] = column(columns, width, row, start);
            let (values, _) = values.as_chunks::<LANES>();
            for (sums, lanes) in sums.iter_mut().zip(values) {
                *sums = combine::<SUBTRACT>(*sums, load(lanes));
            }
        }
        for (sums, lanes) in sums.iter().zip(vectors) {
            store(lanes, *sums);
        }
    }
    let (vectors, _) = rest.as_chunks_mut::<LANES>();
    for (index, lanes) in vectors.iter_mut().enumerate() {
        let start = blocks_end + index * LANES;
        let mut sums = load(lanes);
        for &row in rows {
            let values: &[i16; LANES] = column(columns, width, row, start);
            sums = combine::<SUBTRACT>(sums, load(values));
        }
        store(lanes, sums);
    }
}

/// The SIMD128 [`super::ClipI16Fn`].
///
/// # Safety
///
/// None beyond the build's SIMD128.
#[target_feature(enable = "simd128")]
pub(super) unsafe fn clip_i16(acc: &[i16], shift: u32, out: &mut [u8]) {
    let (values, _) = acc.as_chunks::<CLIP_STEP>();
    let (bytes, _) = out.as_chunks_mut::<CLIP_STEP>();
    for (values, bytes) in values.iter().zip(bytes) {
        let (halves, _) = values.as_chunks::<LANES>();
        let low = i16x8_shr(load(&halves[0]), shift);
        let high = i16x8_shr(load(&halves[1]), shift);
        // The narrow saturates at 127, the largest activation.
        let clipped = i8x16_max(i8x16_narrow_i16x8(low, high), i8x16_splat(0));
        store(bytes, clipped);
    }
}

/// The SIMD128 [`super::ClipI32Fn`]: eight sums a step, the rest scalar.
///
/// # Safety
///
/// None beyond the build's SIMD128.
#[target_feature(enable = "simd128")]
pub(super) unsafe fn clip_i32(sums: &[i32], shift: u32, out: &mut [u8]) {
    let (values, rest) = sums.as_chunks::<SUM_STEP>();
    let (bytes, rest_bytes) = out.as_chunks_mut::<SUM_STEP>();
    for (values, bytes) in values.iter().zip(bytes) {
        let (halves, _) = values.as_chunks::<SUMS>();
        let low = i32x4_shr(load(&halves[0]), shift);
        let high = i32x4_shr(load(&halves[1]), shift);
        let words = i16x8_narrow_i32x4(low, high);
        // The second narrow saturates at 127, the largest activation.
        let clipped = i8x16_max(i8x16_narrow_i16x8(words, words), i8x16_splat(0));
        *bytes = i64x2_extract_lane::<0>(clipped).to_le_bytes();
    }
    super::clip_i32(rest, shift, rest_bytes);
}

/// `sums` plus the byte products of `x` and one vector of weights (four
/// products to a sum).
#[inline]
#[target_feature(enable = "simd128")]
fn accumulate(sums: v128, x: v128, weights: &[i8; BYTES]) -> v128 {
    let weights = load(weights);
    // The inputs are at most 127, so they are the same value as signed
    // bytes, and a low and a high product sum to within ±32512
    // (`super::MAX_PAIR_SUM`): no wrap.
    let pairs = i16x8_add(
        i16x8_extmul_low_i8x16(weights, x),
        i16x8_extmul_high_i8x16(weights, x),
    );
    i32x4_add(sums, i32x4_extadd_pairwise_i16x8(pairs))
}

/// Lanes 0 and 2 of `p` and `q` interleaved, plus lanes 1 and 3: the two
/// half totals of each, `(p01, q01, p23, q23)`.
#[inline]
#[target_feature(enable = "simd128")]
fn half_totals(p: v128, q: v128) -> v128 {
    i32x4_add(
        i32x4_shuffle::<0, 4, 2, 6>(p, q),
        i32x4_shuffle::<1, 5, 3, 7>(p, q),
    )
}

/// The totals of four vectors of 32-bit sums, one per lane.
#[inline]
#[target_feature(enable = "simd128")]
fn reduce_group(sums: [v128; DENSE_GROUP]) -> v128 {
    let [a, b, c, d] = sums;
    let (ab, cd) = (half_totals(a, b), half_totals(c, d));
    i32x4_add(
        i32x4_shuffle::<0, 1, 4, 5>(ab, cd),
        i32x4_shuffle::<2, 3, 6, 7>(ab, cd),
    )
}

/// The dot products of `x` with the four weight rows of `rows`.
#[inline]
#[target_feature(enable = "simd128")]
fn dot_group(x: &[u8], rows: &[i8]) -> v128 {
    let (x, _) = x.as_chunks::<BYTES>();
    let (weights, _) = rows.as_chunks::<BYTES>();
    let per_row = x.len();
    let mut sums = [i32x4_splat(0); DENSE_GROUP];
    for (index, x) in x.iter().enumerate() {
        let x = load(x);
        for (row, sums) in sums.iter_mut().enumerate() {
            *sums = accumulate(*sums, x, &weights[row * per_row + index]);
        }
    }
    reduce_group(sums)
}

/// The dot product of `x` with one weight row.
#[inline]
#[target_feature(enable = "simd128")]
fn dot(x: &[u8], row: &[i8]) -> i32 {
    let (x, _) = x.as_chunks::<BYTES>();
    let (weights, _) = row.as_chunks::<BYTES>();
    let mut sums = i32x4_splat(0);
    for (x, weights) in x.iter().zip(weights) {
        sums = accumulate(sums, load(x), weights);
    }
    i32x4_extract_lane::<0>(sums)
        .wrapping_add(i32x4_extract_lane::<1>(sums))
        .wrapping_add(i32x4_extract_lane::<2>(sums))
        .wrapping_add(i32x4_extract_lane::<3>(sums))
}

/// The SIMD128 [`super::DenseFn`].
///
/// # Safety
///
/// None beyond the build's SIMD128.
#[target_feature(enable = "simd128")]
pub(super) unsafe fn dense(x: &[u8], w: &[i8], bias: &[i32], out: &mut [i32]) {
    let (groups, rest) = out.as_chunks_mut::<DENSE_GROUP>();
    let (group_bias, rest_bias) = bias.as_chunks::<DENSE_GROUP>();
    let (group_rows, rest_rows) = w.split_at(groups.len() * DENSE_GROUP * x.len());
    let group_rows = group_rows.chunks_exact(DENSE_GROUP * x.len());
    for ((out, bias), rows) in groups.iter_mut().zip(group_bias).zip(group_rows) {
        store(out, i32x4_add(dot_group(x, rows), load(bias)));
    }
    let rest_rows = rest_rows.chunks_exact(x.len());
    for ((out, &bias), row) in rest.iter_mut().zip(rest_bias).zip(rest_rows) {
        *out = bias.wrapping_add(dot(x, row));
    }
}
