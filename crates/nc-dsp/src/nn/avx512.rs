//! AVX-512 network inference. Accumulator: two vectors of 32 sums held in
//! registers across the row list. Dense: VNNI `vpdpbusd` (four byte
//! products straight into each 32-bit sum, no 16-bit stage), 64 inputs a
//! step and a 256-bit step for an odd 32, reduced as the AVX2 kernel
//! reduces. Activations: an arithmetic shift, a signed-saturating narrowing
//! move (`vpmovswb`, `vpmovsdb`), then a maximum with zero. What is left
//! over below a 512-bit step runs the AVX2 kernels.

use super::avx2;
use super::layout::{COLUMN_BLOCK, DENSE_GROUP, column};
use core::arch::x86_64::{
    __m128i, __m256i, __m512i, _mm_cvtsi32_si128, _mm_max_epi8, _mm_setzero_si128,
    _mm_storeu_si128, _mm256_add_epi32, _mm256_dpbusd_epi32, _mm256_loadu_si256, _mm256_max_epi8,
    _mm256_setzero_si256, _mm256_storeu_si256, _mm512_add_epi16, _mm512_castsi512_si256,
    _mm512_cvtsepi16_epi8, _mm512_cvtsepi32_epi8, _mm512_dpbusd_epi32, _mm512_extracti64x4_epi64,
    _mm512_loadu_si512, _mm512_setzero_si512, _mm512_sra_epi16, _mm512_sra_epi32,
    _mm512_storeu_si512, _mm512_sub_epi16,
};

/// Accumulator sums per vector.
const LANES: usize = 32;
/// Vectors of an accumulator block.
const BLOCK_VECTORS: usize = COLUMN_BLOCK / LANES;
/// Inputs (bytes) per vector.
const BYTES: usize = 64;
/// Inputs of the 256-bit step that takes an odd half vector.
const HALF_BYTES: usize = BYTES / 2;
/// 32-bit sums per vector.
const SUMS: usize = 16;

/// One vector of sums.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn load(values: &[i16; LANES]) -> __m512i {
    // SAFETY: the load reads exactly the 64 bytes of `values`.
    unsafe { _mm512_loadu_si512(values.as_ptr().cast()) }
}

/// The AVX-512 [`super::ColumnsFn`], adding or subtracting.
///
/// # Safety
///
/// The CPU must support the AVX-512 tier (`Isa::Avx512`).
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
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
        let mut sums = [_mm512_setzero_si512(); BLOCK_VECTORS];
        for (sums, lanes) in sums.iter_mut().zip(vectors.iter()) {
            *sums = load(lanes);
        }
        for &row in rows {
            let values: &[i16; COLUMN_BLOCK] = column(columns, width, row, start);
            let (values, _) = values.as_chunks::<LANES>();
            for (sums, lanes) in sums.iter_mut().zip(values) {
                *sums = if SUBTRACT {
                    _mm512_sub_epi16(*sums, load(lanes))
                } else {
                    _mm512_add_epi16(*sums, load(lanes))
                };
            }
        }
        for (sums, lanes) in sums.iter().zip(vectors) {
            // SAFETY: the store writes exactly the 64 bytes of `lanes`.
            unsafe { _mm512_storeu_si512(lanes.as_mut_ptr().cast(), *sums) };
        }
    }
    avx2::update_vectors::<SUBTRACT>(rest, blocks_end, columns, width, rows);
}

/// The AVX-512 [`super::ClipI16Fn`]: 32 sums a step, then the AVX2 kernel
/// for an odd 16.
///
/// # Safety
///
/// The CPU must support the AVX-512 tier (`Isa::Avx512`).
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub(super) unsafe fn clip_i16(acc: &[i16], shift: u32, out: &mut [u8]) {
    let count = _mm_cvtsi32_si128(shift as i32);
    let (values, rest) = acc.as_chunks::<LANES>();
    let (bytes, rest_bytes) = out.as_chunks_mut::<LANES>();
    for (values, bytes) in values.iter().zip(bytes) {
        // The narrowing saturates at 127, the largest activation.
        let narrow = _mm512_cvtsepi16_epi8(_mm512_sra_epi16(load(values), count));
        let clipped = _mm256_max_epi8(narrow, _mm256_setzero_si256());
        // SAFETY: the store writes exactly the 32 bytes of `bytes`.
        unsafe { _mm256_storeu_si256(bytes.as_mut_ptr().cast(), clipped) };
    }
    // SAFETY: the AVX-512 tier includes AVX2.
    unsafe { avx2::clip_i16(rest, shift, rest_bytes) };
}

/// The AVX-512 [`super::ClipI32Fn`]: 16 sums a step, then the AVX2 kernel.
///
/// # Safety
///
/// The CPU must support the AVX-512 tier (`Isa::Avx512`).
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub(super) unsafe fn clip_i32(sums: &[i32], shift: u32, out: &mut [u8]) {
    let count = _mm_cvtsi32_si128(shift as i32);
    let (values, rest) = sums.as_chunks::<SUMS>();
    let (bytes, rest_bytes) = out.as_chunks_mut::<SUMS>();
    for (values, bytes) in values.iter().zip(bytes) {
        // SAFETY: the load reads exactly the 64 bytes of `values`.
        let values = unsafe { _mm512_loadu_si512(values.as_ptr().cast()) };
        // The narrowing saturates at 127, the largest activation.
        let narrow: __m128i = _mm512_cvtsepi32_epi8(_mm512_sra_epi32(values, count));
        let clipped = _mm_max_epi8(narrow, _mm_setzero_si128());
        // SAFETY: the store writes exactly the 16 bytes of `bytes`.
        unsafe { _mm_storeu_si128(bytes.as_mut_ptr().cast(), clipped) };
    }
    // SAFETY: the AVX-512 tier includes AVX2.
    unsafe { avx2::clip_i32(rest, shift, rest_bytes) };
}

/// The inputs as whole vectors and, if their count is odd in 32s, one half
/// vector.
#[inline]
fn split_inputs(x: &[u8]) -> (&[[u8; BYTES]], Option<&[u8; HALF_BYTES]>) {
    let (whole, half) = x.as_chunks::<BYTES>();
    (whole, half.first_chunk())
}

/// `sums` plus the products of `x` and the 64 weights from `at` of `weights`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
fn accumulate(sums: __m512i, x: __m512i, weights: &[i8], at: usize) -> __m512i {
    let weights: &[i8; BYTES] = weights[at..].first_chunk().expect("a whole weight row");
    // SAFETY: the load reads exactly the 64 bytes of `weights`.
    let weights = unsafe { _mm512_loadu_si512(weights.as_ptr().cast()) };
    _mm512_dpbusd_epi32(sums, x, weights)
}

/// `sums` folded to 256 bits, plus the products of the half vector `x`
/// (if any) and the 32 weights from `at` of `weights`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn finish(sums: __m512i, x: Option<&[u8; HALF_BYTES]>, weights: &[i8], at: usize) -> __m256i {
    let folded = _mm256_add_epi32(
        _mm512_castsi512_si256(sums),
        _mm512_extracti64x4_epi64::<1>(sums),
    );
    let Some(x) = x else {
        return folded;
    };
    let weights: &[i8; HALF_BYTES] = weights[at..].first_chunk().expect("a whole weight row");
    // SAFETY: each load reads exactly the 32 bytes of its array.
    let (x, weights) = unsafe {
        (
            _mm256_loadu_si256(x.as_ptr().cast()),
            _mm256_loadu_si256(weights.as_ptr().cast()),
        )
    };
    _mm256_dpbusd_epi32(folded, x, weights)
}

/// One vector of inputs.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn inputs(x: &[u8; BYTES]) -> __m512i {
    // SAFETY: the load reads exactly the 64 bytes of `x`.
    unsafe { _mm512_loadu_si512(x.as_ptr().cast()) }
}

/// The dot products of `x` with the four weight rows of `rows`, as 256-bit
/// sums to reduce.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn dot_group(x: &[u8], rows: &[i8]) -> [__m256i; DENSE_GROUP] {
    let per_row = x.len();
    let (whole, half) = split_inputs(x);
    let mut sums = [_mm512_setzero_si512(); DENSE_GROUP];
    for (index, x) in whole.iter().enumerate() {
        let x = inputs(x);
        for (row, sums) in sums.iter_mut().enumerate() {
            *sums = accumulate(*sums, x, rows, row * per_row + index * BYTES);
        }
    }
    let done = whole.len() * BYTES;
    let mut folded = [_mm256_setzero_si256(); DENSE_GROUP];
    for (row, (folded, sums)) in folded.iter_mut().zip(sums).enumerate() {
        *folded = finish(sums, half, rows, row * per_row + done);
    }
    folded
}

/// The dot product of `x` with one weight row.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn dot(x: &[u8], row: &[i8]) -> i32 {
    let (whole, half) = split_inputs(x);
    let mut sums = _mm512_setzero_si512();
    for (index, x) in whole.iter().enumerate() {
        sums = accumulate(sums, inputs(x), row, index * BYTES);
    }
    avx2::reduce(finish(sums, half, row, whole.len() * BYTES))
}

/// The AVX-512 [`super::DenseFn`].
///
/// # Safety
///
/// The CPU must support the AVX-512 tier (`Isa::Avx512`).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
pub(super) unsafe fn dense(x: &[u8], w: &[i8], bias: &[i32], out: &mut [i32]) {
    let (groups, rest) = out.as_chunks_mut::<DENSE_GROUP>();
    let (group_bias, rest_bias) = bias.as_chunks::<DENSE_GROUP>();
    let (group_rows, rest_rows) = w.split_at(groups.len() * DENSE_GROUP * x.len());
    let group_rows = group_rows.chunks_exact(DENSE_GROUP * x.len());
    for ((out, bias), rows) in groups.iter_mut().zip(group_bias).zip(group_rows) {
        avx2::finish_group(avx2::reduce_group(dot_group(x, rows)), bias, out);
    }
    let rest_rows = rest_rows.chunks_exact(x.len());
    for ((out, &bias), row) in rest.iter_mut().zip(rest_bias).zip(rest_rows) {
        *out = bias.wrapping_add(dot(x, row));
    }
}
