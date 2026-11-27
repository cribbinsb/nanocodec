//! NEON network inference. Accumulator: eight vectors of eight sums held in
//! registers across the row list. Dense: signed widening byte multiplies
//! (`smull`, then `smlal2` onto the same 16-bit lanes: exact, see the module
//! doc) widened by a pairwise add-accumulate (`sadalp`), four outputs to
//! each 16-byte input load and three `addp` to reduce them. Activations: an
//! arithmetic shift, saturating narrows (`sqxtn`), then a maximum with zero.

use super::layout::{COLUMN_BLOCK, DENSE_GROUP, column};
use core::arch::aarch64::{
    int8x16_t, int16x8_t, int32x4_t, vaddq_s16, vaddq_s32, vaddvq_s32, vcombine_s16, vdup_n_s8,
    vdupq_n_s8, vdupq_n_s16, vdupq_n_s32, vget_low_s8, vld1q_s8, vld1q_s16, vld1q_s32, vld1q_u8,
    vmax_s8, vmaxq_s8, vmlal_high_s8, vmull_s8, vpadalq_s16, vpaddq_s32, vqmovn_high_s16,
    vqmovn_s16, vqmovn_s32, vreinterpretq_s8_u8, vshlq_s16, vshlq_s32, vst1_s8, vst1q_s8,
    vst1q_s16, vst1q_s32, vsubq_s16,
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

/// One vector of sums.
#[inline]
#[target_feature(enable = "neon")]
fn load(values: &[i16; LANES]) -> int16x8_t {
    // SAFETY: the load reads exactly the 16 bytes of `values`.
    unsafe { vld1q_s16(values.as_ptr()) }
}

/// Write one vector of sums.
#[inline]
#[target_feature(enable = "neon")]
fn store(values: &mut [i16; LANES], sums: int16x8_t) {
    // SAFETY: the store writes exactly the 16 bytes of `values`.
    unsafe { vst1q_s16(values.as_mut_ptr(), sums) }
}

/// `sums ± column`, 16-bit wrapping.
#[inline]
#[target_feature(enable = "neon")]
fn combine<const SUBTRACT: bool>(sums: int16x8_t, column: int16x8_t) -> int16x8_t {
    if SUBTRACT {
        vsubq_s16(sums, column)
    } else {
        vaddq_s16(sums, column)
    }
}

/// The NEON [`super::ColumnsFn`], adding or subtracting.
///
/// # Safety
///
/// None beyond NEON, which `AArch64` always has.
#[target_feature(enable = "neon")]
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
        let mut sums = [vdupq_n_s16(0); BLOCK_VECTORS];
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
            sums = combine::<SUBTRACT>(sums, load(column(columns, width, row, start)));
        }
        store(lanes, sums);
    }
}

/// The NEON [`super::ClipI16Fn`].
///
/// # Safety
///
/// None beyond NEON, which `AArch64` always has.
#[target_feature(enable = "neon")]
pub(super) unsafe fn clip_i16(acc: &[i16], shift: u32, out: &mut [u8]) {
    // A left shift by a negative count is the arithmetic right shift.
    let count = vdupq_n_s16(-(shift as i16));
    let (values, _) = acc.as_chunks::<CLIP_STEP>();
    let (bytes, _) = out.as_chunks_mut::<CLIP_STEP>();
    for (values, bytes) in values.iter().zip(bytes) {
        let (low, high) = values.split_at(LANES);
        // SAFETY: each load reads one 8-value half of `values`.
        let (low, high) = unsafe { (vld1q_s16(low.as_ptr()), vld1q_s16(high.as_ptr())) };
        // The narrows saturate at 127, the largest activation.
        let narrow = vqmovn_high_s16(vqmovn_s16(vshlq_s16(low, count)), vshlq_s16(high, count));
        let clipped = vmaxq_s8(narrow, vdupq_n_s8(0));
        // SAFETY: the store writes exactly the 16 bytes of `bytes`.
        unsafe { vst1q_s8(bytes.as_mut_ptr().cast(), clipped) };
    }
}

/// The NEON [`super::ClipI32Fn`]: eight sums a step, the rest scalar.
///
/// # Safety
///
/// None beyond NEON, which `AArch64` always has.
#[target_feature(enable = "neon")]
pub(super) unsafe fn clip_i32(sums: &[i32], shift: u32, out: &mut [u8]) {
    // A left shift by a negative count is the arithmetic right shift.
    let count = vdupq_n_s32(-(shift as i32));
    let (values, rest) = sums.as_chunks::<SUM_STEP>();
    let (bytes, rest_bytes) = out.as_chunks_mut::<SUM_STEP>();
    for (values, bytes) in values.iter().zip(bytes) {
        let (low, high) = values.split_at(SUMS);
        // SAFETY: each load reads one 4-value half of `values`.
        let (low, high) = unsafe { (vld1q_s32(low.as_ptr()), vld1q_s32(high.as_ptr())) };
        let words = vcombine_s16(
            vqmovn_s32(vshlq_s32(low, count)),
            vqmovn_s32(vshlq_s32(high, count)),
        );
        // The second narrow saturates at 127, the largest activation.
        let clipped = vmax_s8(vqmovn_s16(words), vdup_n_s8(0));
        // SAFETY: the store writes exactly the 8 bytes of `bytes`.
        unsafe { vst1_s8(bytes.as_mut_ptr().cast(), clipped) };
    }
    super::clip_i32(rest, shift, rest_bytes);
}

/// One vector of inputs, as signed bytes (they are at most 127).
#[inline]
#[target_feature(enable = "neon")]
fn inputs(x: &[u8; BYTES]) -> int8x16_t {
    // SAFETY: the load reads exactly the 16 bytes of `x`.
    vreinterpretq_s8_u8(unsafe { vld1q_u8(x.as_ptr()) })
}

/// `sums` plus the byte products of `x` and one vector of weights (four
/// products to a sum).
#[inline]
#[target_feature(enable = "neon")]
fn accumulate(sums: int32x4_t, x: int8x16_t, weights: &[i8; BYTES]) -> int32x4_t {
    // SAFETY: the load reads exactly the 16 bytes of `weights`.
    let weights = unsafe { vld1q_s8(weights.as_ptr()) };
    // No wrap: `x` is at most 127, so a low and a high product sum to
    // within ±32512 (`super::MAX_PAIR_SUM`).
    let pairs = vmlal_high_s8(vmull_s8(vget_low_s8(weights), vget_low_s8(x)), weights, x);
    vpadalq_s16(sums, pairs)
}

/// The totals of four vectors of 32-bit sums, one per lane.
#[inline]
#[target_feature(enable = "neon")]
fn reduce_group(sums: [int32x4_t; DENSE_GROUP]) -> int32x4_t {
    let [a, b, c, d] = sums;
    vpaddq_s32(vpaddq_s32(a, b), vpaddq_s32(c, d))
}

/// The dot products of `x` with the four weight rows of `rows`.
#[inline]
#[target_feature(enable = "neon")]
fn dot_group(x: &[u8], rows: &[i8]) -> int32x4_t {
    let (x, _) = x.as_chunks::<BYTES>();
    let (weights, _) = rows.as_chunks::<BYTES>();
    let per_row = x.len();
    let mut sums = [vdupq_n_s32(0); DENSE_GROUP];
    for (index, x) in x.iter().enumerate() {
        let x = inputs(x);
        for (row, sums) in sums.iter_mut().enumerate() {
            *sums = accumulate(*sums, x, &weights[row * per_row + index]);
        }
    }
    reduce_group(sums)
}

/// The dot product of `x` with one weight row.
#[inline]
#[target_feature(enable = "neon")]
fn dot(x: &[u8], row: &[i8]) -> i32 {
    let (x, _) = x.as_chunks::<BYTES>();
    let (weights, _) = row.as_chunks::<BYTES>();
    let mut sums = vdupq_n_s32(0);
    for (x, weights) in x.iter().zip(weights) {
        sums = accumulate(sums, inputs(x), weights);
    }
    vaddvq_s32(sums)
}

/// The NEON [`super::DenseFn`].
///
/// # Safety
///
/// None beyond NEON, which `AArch64` always has.
#[target_feature(enable = "neon")]
pub(super) unsafe fn dense(x: &[u8], w: &[i8], bias: &[i32], out: &mut [i32]) {
    let (groups, rest) = out.as_chunks_mut::<DENSE_GROUP>();
    let (group_bias, rest_bias) = bias.as_chunks::<DENSE_GROUP>();
    let (group_rows, rest_rows) = w.split_at(groups.len() * DENSE_GROUP * x.len());
    let group_rows = group_rows.chunks_exact(DENSE_GROUP * x.len());
    for ((out, bias), rows) in groups.iter_mut().zip(group_bias).zip(group_rows) {
        // SAFETY: the load and the store cover exactly the 16 bytes of
        // `bias` and of `out`.
        unsafe {
            let sums = vaddq_s32(dot_group(x, rows), vld1q_s32(bias.as_ptr()));
            vst1q_s32(out.as_mut_ptr(), sums);
        }
    }
    let rest_rows = rest_rows.chunks_exact(x.len());
    for ((out, &bias), row) in rest.iter_mut().zip(rest_bias).zip(rest_rows) {
        *out = bias.wrapping_add(dot(x, row));
    }
}
