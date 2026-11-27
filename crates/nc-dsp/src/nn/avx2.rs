//! AVX2 network inference. Accumulator: four vectors of sixteen sums held
//! in registers across the row list. Dense: `vpmaddubsw` pair sums (exact,
//! see the module doc) widened by `vpmaddwd` with ones, four outputs to
//! each 32-byte input load and three `vphaddd` to reduce them. Activations:
//! an arithmetic shift, signed-saturating packs to bytes, then a maximum
//! with zero.

use super::layout::{COLUMN_BLOCK, DENSE_GROUP, column};
use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi32, _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_hadd_epi32,
    _mm_loadu_si128, _mm_max_epi8, _mm_packs_epi16, _mm_packs_epi32, _mm_setzero_si128,
    _mm_storel_epi64, _mm_storeu_si128, _mm256_add_epi16, _mm256_add_epi32, _mm256_castsi256_si128,
    _mm256_extracti128_si256, _mm256_hadd_epi32, _mm256_loadu_si256, _mm256_madd_epi16,
    _mm256_maddubs_epi16, _mm256_set1_epi16, _mm256_setzero_si256, _mm256_sra_epi16,
    _mm256_sra_epi32, _mm256_storeu_si256, _mm256_sub_epi16,
};

/// Accumulator sums per vector.
const LANES: usize = 16;
/// Vectors of an accumulator block.
const BLOCK_VECTORS: usize = COLUMN_BLOCK / LANES;
/// Inputs (bytes) per vector.
const BYTES: usize = 32;
/// 32-bit sums per vector.
const SUMS: usize = 8;

/// One vector of sums.
#[inline]
#[target_feature(enable = "avx2")]
fn load(values: &[i16; LANES]) -> __m256i {
    // SAFETY: the load reads exactly the 32 bytes of `values`.
    unsafe { _mm256_loadu_si256(values.as_ptr().cast()) }
}

/// Write one vector of sums.
#[inline]
#[target_feature(enable = "avx2")]
fn store(values: &mut [i16; LANES], sums: __m256i) {
    // SAFETY: the store writes exactly the 32 bytes of `values`.
    unsafe { _mm256_storeu_si256(values.as_mut_ptr().cast(), sums) }
}

/// `sums ± column`, 16-bit wrapping.
#[inline]
#[target_feature(enable = "avx2")]
fn combine<const SUBTRACT: bool>(sums: __m256i, column: __m256i) -> __m256i {
    if SUBTRACT {
        _mm256_sub_epi16(sums, column)
    } else {
        _mm256_add_epi16(sums, column)
    }
}

/// The accumulator update of `acc`, the sums from index `first` of an
/// accumulator `width` wide, a vector at a time (what a block leaves over;
/// the AVX-512 kernel's remainder too).
#[inline]
#[target_feature(enable = "avx2")]
pub(super) fn update_vectors<const SUBTRACT: bool>(
    acc: &mut [i16],
    first: usize,
    columns: &[i16],
    width: usize,
    rows: &[u16],
) {
    let (vectors, _) = acc.as_chunks_mut::<LANES>();
    for (index, lanes) in vectors.iter_mut().enumerate() {
        let start = first + index * LANES;
        let mut sums = load(lanes);
        for &row in rows {
            sums = combine::<SUBTRACT>(sums, load(column(columns, width, row, start)));
        }
        store(lanes, sums);
    }
}

/// The AVX2 [`super::ColumnsFn`], adding or subtracting.
///
/// # Safety
///
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
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
        let mut sums = [_mm256_setzero_si256(); BLOCK_VECTORS];
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
    update_vectors::<SUBTRACT>(rest, blocks_end, columns, width, rows);
}

/// The AVX2 [`super::ClipI16Fn`].
///
/// # Safety
///
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn clip_i16(acc: &[i16], shift: u32, out: &mut [u8]) {
    let count = _mm_cvtsi32_si128(shift as i32);
    let (values, _) = acc.as_chunks::<LANES>();
    let (bytes, _) = out.as_chunks_mut::<LANES>();
    for (values, bytes) in values.iter().zip(bytes) {
        let shifted = _mm256_sra_epi16(load(values), count);
        // The pack saturates at 127, the largest activation.
        let packed = _mm_packs_epi16(
            _mm256_castsi256_si128(shifted),
            _mm256_extracti128_si256::<1>(shifted),
        );
        let clipped = _mm_max_epi8(packed, _mm_setzero_si128());
        // SAFETY: the store writes exactly the 16 bytes of `bytes`.
        unsafe { _mm_storeu_si128(bytes.as_mut_ptr().cast(), clipped) };
    }
}

/// The AVX2 [`super::ClipI32Fn`]: eight sums a step, the rest scalar.
///
/// # Safety
///
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn clip_i32(sums: &[i32], shift: u32, out: &mut [u8]) {
    let count = _mm_cvtsi32_si128(shift as i32);
    let (values, rest) = sums.as_chunks::<SUMS>();
    let (bytes, rest_bytes) = out.as_chunks_mut::<SUMS>();
    for (values, bytes) in values.iter().zip(bytes) {
        // SAFETY: the load reads exactly the 32 bytes of `values`.
        let values = unsafe { _mm256_loadu_si256(values.as_ptr().cast()) };
        let shifted = _mm256_sra_epi32(values, count);
        let words = _mm_packs_epi32(
            _mm256_castsi256_si128(shifted),
            _mm256_extracti128_si256::<1>(shifted),
        );
        // The second pack saturates at 127, the largest activation.
        let clipped = _mm_max_epi8(_mm_packs_epi16(words, words), _mm_setzero_si128());
        // SAFETY: the store writes the low 8 bytes, exactly `bytes`.
        unsafe { _mm_storel_epi64(bytes.as_mut_ptr().cast(), clipped) };
    }
    super::clip_i32(rest, shift, rest_bytes);
}

/// The eight 32-bit sums of the byte products of `x` and one vector of
/// weights (four products each).
#[inline]
#[target_feature(enable = "avx2")]
fn products(x: __m256i, weights: &[i8; BYTES]) -> __m256i {
    // SAFETY: the load reads exactly the 32 bytes of `weights`.
    let weights = unsafe { _mm256_loadu_si256(weights.as_ptr().cast()) };
    // No saturation: `x` is at most 127, so a pair sum stays within
    // ±32512 (`super::MAX_PAIR_SUM`).
    let pairs = _mm256_maddubs_epi16(x, weights);
    _mm256_madd_epi16(pairs, _mm256_set1_epi16(1))
}

/// One vector of inputs.
#[inline]
#[target_feature(enable = "avx2")]
fn inputs(x: &[u8; BYTES]) -> __m256i {
    // SAFETY: the load reads exactly the 32 bytes of `x`.
    unsafe { _mm256_loadu_si256(x.as_ptr().cast()) }
}

/// The totals of four vectors of 32-bit sums, one per lane.
#[inline]
#[target_feature(enable = "avx2")]
pub(super) fn reduce_group(sums: [__m256i; DENSE_GROUP]) -> __m128i {
    let [a, b, c, d] = sums;
    // Each 128-bit half then holds its part of a, b, c and d in turn.
    let halves = _mm256_hadd_epi32(_mm256_hadd_epi32(a, b), _mm256_hadd_epi32(c, d));
    _mm_add_epi32(
        _mm256_castsi256_si128(halves),
        _mm256_extracti128_si256::<1>(halves),
    )
}

/// The total of one vector of 32-bit sums.
#[inline]
#[target_feature(enable = "avx2")]
pub(super) fn reduce(sums: __m256i) -> i32 {
    let half = _mm_add_epi32(
        _mm256_castsi256_si128(sums),
        _mm256_extracti128_si256::<1>(sums),
    );
    let pairs = _mm_hadd_epi32(half, half);
    _mm_cvtsi128_si32(_mm_hadd_epi32(pairs, pairs))
}

/// The dot products of `x` with the four weight rows of `rows`.
#[inline]
#[target_feature(enable = "avx2")]
fn dot_group(x: &[u8], rows: &[i8]) -> __m128i {
    let (x, _) = x.as_chunks::<BYTES>();
    let (weights, _) = rows.as_chunks::<BYTES>();
    let per_row = x.len();
    let mut sums = [_mm256_setzero_si256(); DENSE_GROUP];
    for (index, x) in x.iter().enumerate() {
        let x = inputs(x);
        for (row, sums) in sums.iter_mut().enumerate() {
            *sums = _mm256_add_epi32(*sums, products(x, &weights[row * per_row + index]));
        }
    }
    reduce_group(sums)
}

/// The dot product of `x` with one weight row.
#[inline]
#[target_feature(enable = "avx2")]
fn dot(x: &[u8], row: &[i8]) -> i32 {
    let (x, _) = x.as_chunks::<BYTES>();
    let (weights, _) = row.as_chunks::<BYTES>();
    let mut sums = _mm256_setzero_si256();
    for (x, weights) in x.iter().zip(weights) {
        sums = _mm256_add_epi32(sums, products(inputs(x), weights));
    }
    reduce(sums)
}

/// Add the four biases of a group to its dot products and store them.
#[inline]
#[target_feature(enable = "avx2")]
pub(super) fn finish_group(dots: __m128i, bias: &[i32; DENSE_GROUP], out: &mut [i32; DENSE_GROUP]) {
    // SAFETY: the load and the store cover exactly the 16 bytes of `bias`
    // and of `out`.
    unsafe {
        let sums = _mm_add_epi32(dots, _mm_loadu_si128(bias.as_ptr().cast()));
        _mm_storeu_si128(out.as_mut_ptr().cast(), sums);
    }
}

/// The AVX2 [`super::DenseFn`].
///
/// # Safety
///
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn dense(x: &[u8], w: &[i8], bias: &[i32], out: &mut [i32]) {
    let (groups, rest) = out.as_chunks_mut::<DENSE_GROUP>();
    let (group_bias, rest_bias) = bias.as_chunks::<DENSE_GROUP>();
    let (group_rows, rest_rows) = w.split_at(groups.len() * DENSE_GROUP * x.len());
    let group_rows = group_rows.chunks_exact(DENSE_GROUP * x.len());
    for ((out, bias), rows) in groups.iter_mut().zip(group_bias).zip(group_rows) {
        finish_group(dot_group(x, rows), bias, out);
    }
    let rest_rows = rest_rows.chunks_exact(x.len());
    for ((out, &bias), row) in rest.iter_mut().zip(rest_bias).zip(rest_rows) {
        *out = bias.wrapping_add(dot(x, row));
    }
}
