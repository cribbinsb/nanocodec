//! AVX2 sample kernels (x86-64).
//!
//! Moments: `vpsadbw` against zero gives row sums; squares come from widening
//! to 16 bits and `vpmaddwd` with itself. Downscale: `vpmaddubsw` with ones
//! sums horizontal pairs, the two rows are added, rounded and packed; 32
//! outputs per iteration, with the scalar row function for the tail.
//!
//! Kernel contract: see the `sample` module docs.

use core::arch::x86_64::{
    __m256i, _mm_add_epi32, _mm_cvtsi128_si32, _mm_shuffle_epi32, _mm256_add_epi16,
    _mm256_add_epi32, _mm256_add_epi64, _mm256_and_si256, _mm256_castsi256_si128,
    _mm256_cvtepu8_epi16, _mm256_extracti128_si256, _mm256_hadd_epi32, _mm256_loadu_si256,
    _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_packus_epi16, _mm256_permute4x64_epi64,
    _mm256_sad_epu8, _mm256_set1_epi8, _mm256_set1_epi16, _mm256_setr_epi64x, _mm256_setzero_si256,
    _mm256_srli_epi16, _mm256_storeu_si256,
};

use super::{Moments, SampleFns, scalar};

/// Outputs produced per iteration of the downscale loop.
const DOWNSCALE_STEP: usize = 32;

/// The AVX2 kernel table.
///
/// # Safety
///
/// The CPU must support AVX2.
pub(crate) unsafe fn sample_fns() -> SampleFns {
    SampleFns {
        moments_8x8_row: moments_row::<8>,
        moments_8x4_row: moments_row::<4>,
        downscale_2x,
        pair_mean_2x: pairs_2x::<false>,
        pair_pick_2x: pairs_2x::<true>,
    }
}

/// Blocks measured per iteration of [`moments_8x8_row`]: one 32-byte load
/// per row covers four 8x8 blocks.
const MOMENTS_GROUP: usize = 4;

/// Moments of each 8-wide, `ROWS`-tall block along a row, four blocks per
/// 32-byte row load ([`moments_quad`]); the tail per block ([`moments_8x8`],
/// or the scalar kernel for 4-row blocks).
#[target_feature(enable = "avx2")]
pub(super) unsafe fn moments_row<const ROWS: usize>(
    src: &[u8],
    src_stride: usize,
    out: &mut [Moments],
) {
    let groups = out.len() / MOMENTS_GROUP;
    let (quads, tail) = out.as_chunks_mut::<MOMENTS_GROUP>();
    for (g, quad) in quads.iter_mut().enumerate() {
        // SAFETY: blocks 4g..4g + 4 lie inside the row (kernel contract).
        *quad = unsafe { moments_quad::<ROWS>(&src[32 * g..], src_stride) };
    }
    let first = MOMENTS_GROUP * groups;
    if ROWS != 8 {
        scalar::moments_row::<ROWS>(&src[8 * first..], src_stride, tail);
        return;
    }
    for (i, m) in tail.iter_mut().enumerate() {
        // SAFETY: the block lies inside the row (kernel contract).
        *m = unsafe { moments_8x8(&src[8 * (first + i)..], src_stride) };
    }
}

/// Moments of four horizontally adjacent 8-wide, `ROWS`-tall blocks. `vpsadbw` against zero
/// sums each 8-byte quarter of a row (one block's row) into its own u64
/// lane; squares of each 16-byte half, widened, `vpmaddwd` into i32 lanes
/// whose 128-bit halves hold one block each (at most 8 rows x 2 x 255^2).
///
/// # Safety
///
/// Kernel contract for a 32-sample-wide block.
#[target_feature(enable = "avx2")]
unsafe fn moments_quad<const ROWS: usize>(
    src: &[u8],
    src_stride: usize,
) -> [Moments; MOMENTS_GROUP] {
    let zero = _mm256_setzero_si256();
    let (mut sums, mut squares_01, mut squares_23) = (zero, zero, zero);
    for y in 0..ROWS {
        // SAFETY: row y (32 bytes) is inside the four blocks.
        let v = unsafe { _mm256_loadu_si256(src.as_ptr().add(y * src_stride).cast()) };
        sums = _mm256_add_epi64(sums, _mm256_sad_epu8(v, zero));
        let (lo, hi) = (
            _mm256_cvtepu8_epi16(_mm256_castsi256_si128(v)),
            _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(v)),
        );
        squares_01 = _mm256_add_epi32(squares_01, _mm256_madd_epi16(lo, lo));
        squares_23 = _mm256_add_epi32(squares_23, _mm256_madd_epi16(hi, hi));
    }
    // Two horizontal adds leave block 0's total in 32-bit lane 0, block 2's
    // in lane 1, block 1's in lane 4 and block 3's in lane 5.
    let pairs = _mm256_hadd_epi32(squares_01, squares_23);
    let totals = _mm256_hadd_epi32(pairs, pairs);
    let mut sum = [0u64; MOMENTS_GROUP];
    let mut square = [0u32; 8];
    // SAFETY: both arrays are 32 bytes.
    unsafe {
        _mm256_storeu_si256(sum.as_mut_ptr().cast(), sums);
        _mm256_storeu_si256(square.as_mut_ptr().cast(), totals);
    }
    let lane_of_block = [0, 4, 1, 5];
    core::array::from_fn(|b| Moments {
        sum: sum[b] as u32,
        sum_sq: square[lane_of_block[b]],
    })
}

/// # Safety
///
/// Kernel contract (`sample` module docs).
#[target_feature(enable = "avx2")]
unsafe fn moments_8x8(src: &[u8], src_stride: usize) -> Moments {
    let row = |y: usize| -> i64 {
        // SAFETY: row y (8 bytes) is inside the block (kernel contract).
        unsafe {
            src.as_ptr()
                .add(y * src_stride)
                .cast::<i64>()
                .read_unaligned()
        }
    };
    let lo = _mm256_setr_epi64x(row(0), row(1), row(2), row(3));
    let hi = _mm256_setr_epi64x(row(4), row(5), row(6), row(7));
    let zero = _mm256_setzero_si256();
    let sums = _mm256_add_epi64(_mm256_sad_epu8(lo, zero), _mm256_sad_epu8(hi, zero));

    let mut squares = _mm256_setzero_si256();
    for v in [lo, hi] {
        for half in [_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v)] {
            let wide = _mm256_cvtepu8_epi16(half);
            // Pairs of 255^2 sum to 130050, well inside i32.
            squares = _mm256_add_epi32(squares, _mm256_madd_epi16(wide, wide));
        }
    }
    Moments {
        sum: sum_u64_lanes(sums),
        sum_sq: sum_i32_lanes(squares),
    }
}

/// # Safety
///
/// Kernel contract (`sample` module docs).
#[target_feature(enable = "avx2")]
unsafe fn downscale_2x(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    w: usize,
    h: usize,
) {
    let simd_width = w - w % DOWNSCALE_STEP;
    for y in 0..h {
        let top = &src[2 * y * src_stride..];
        let bottom = &src[(2 * y + 1) * src_stride..];
        let out = &mut dst[y * dst_stride..y * dst_stride + w];
        for x in (0..simd_width).step_by(DOWNSCALE_STEP) {
            // SAFETY: input bytes 2x .. 2x + 64 lie within the 2w-wide
            // source rows, since x + 32 <= w (kernel contract).
            let (left, right) = unsafe {
                (
                    box_means_16(top.as_ptr().add(2 * x), bottom.as_ptr().add(2 * x)),
                    box_means_16(
                        top.as_ptr().add(2 * x + 32),
                        bottom.as_ptr().add(2 * x + 32),
                    ),
                )
            };
            // packus interleaves 128-bit lanes: [l0-7, r0-7, l8-15, r8-15];
            // the permute restores [l0-15, r0-15].
            let packed =
                _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packus_epi16(left, right));
            // SAFETY: outputs x..x + 32 are inside the w-wide output row.
            unsafe { _mm256_storeu_si256(out.as_mut_ptr().add(x).cast(), packed) };
        }
        scalar::downscale_2x_row(
            &top[2 * simd_width..],
            &bottom[2 * simd_width..],
            &mut out[simd_width..],
        );
    }
}

/// [`scalar::pairs_2x`], 32 outputs per iteration: `vpmaddubsw` with ones
/// sums each pair (the mean) or a 0x00FF mask keeps each pair's left sample
/// (the pick), then the 16-bit lanes pack back to bytes.
///
/// # Safety
///
/// Kernel contract (`sample` module docs).
#[target_feature(enable = "avx2")]
unsafe fn pairs_2x<const PICK: bool>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    w: usize,
    h: usize,
) {
    let simd_width = w - w % DOWNSCALE_STEP;
    let pair = |p: *const u8| -> __m256i {
        // SAFETY: the caller reads 32 bytes inside the source row.
        let v = unsafe { _mm256_loadu_si256(p.cast()) };
        if PICK {
            _mm256_and_si256(v, _mm256_set1_epi16(0x00FF))
        } else {
            let sum = _mm256_maddubs_epi16(v, _mm256_set1_epi8(1));
            _mm256_srli_epi16::<1>(_mm256_add_epi16(sum, _mm256_set1_epi16(1)))
        }
    };
    for y in 0..h {
        let row = &src[y * src_stride..];
        let out = &mut dst[y * dst_stride..y * dst_stride + w];
        for x in (0..simd_width).step_by(DOWNSCALE_STEP) {
            // Input bytes 2x .. 2x + 64 lie within the 2w-wide source row,
            // since x + 32 <= w (kernel contract).
            let (left, right) = (
                pair(row[2 * x..].as_ptr()),
                pair(row[2 * x + 32..].as_ptr()),
            );
            // As in `downscale_2x`: the permute undoes packus' lane order.
            let packed =
                _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packus_epi16(left, right));
            // SAFETY: outputs x..x + 32 are inside the w-wide output row.
            unsafe { _mm256_storeu_si256(out.as_mut_ptr().add(x).cast(), packed) };
        }
        scalar::pairs_2x_row::<PICK>(&row[2 * simd_width..], &mut out[simd_width..]);
    }
}

/// Rounded 2x2 box means for 16 outputs, as 16-bit lanes, from 32 bytes of
/// each input row.
///
/// # Safety
///
/// 32 bytes must be readable at `top` and at `bottom`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn box_means_16(top: *const u8, bottom: *const u8) -> __m256i {
    // SAFETY: guaranteed by the caller.
    let (t, b) = unsafe {
        (
            _mm256_loadu_si256(top.cast()),
            _mm256_loadu_si256(bottom.cast()),
        )
    };
    let ones = _mm256_set1_epi8(1);
    let sum = _mm256_add_epi16(_mm256_maddubs_epi16(t, ones), _mm256_maddubs_epi16(b, ones));
    _mm256_srli_epi16::<2>(_mm256_add_epi16(sum, _mm256_set1_epi16(2)))
}

#[target_feature(enable = "avx2")]
#[inline]
fn sum_u64_lanes(v: __m256i) -> u32 {
    let sum = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    let high = _mm_shuffle_epi32::<0b10_10_10_10>(sum);
    _mm_cvtsi128_si32(_mm_add_epi32(sum, high)) as u32
}

#[target_feature(enable = "avx2")]
#[inline]
fn sum_i32_lanes(v: __m256i) -> u32 {
    let v = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    let v = _mm_add_epi32(v, _mm_shuffle_epi32::<0b01_00_11_10>(v));
    let v = _mm_add_epi32(v, _mm_shuffle_epi32::<0b10_11_00_01>(v));
    _mm_cvtsi128_si32(v) as u32
}
