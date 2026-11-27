//! AVX2 residual and variance kernels: the generic kernels on 16 lanes (8
//! for 8-wide blocks, 4 for 4-wide ones). Narrow variance/energy/block sums
//! instead share SVT-style packed-byte row statistics in `x86_narrow`.
//!
//! Kernel contract: see the `diff` module docs.

use core::arch::x86_64::{
    __m128i, __m256i, _mm_avg_epu8, _mm_loadu_si128, _mm256_add_epi16, _mm256_add_epi32,
    _mm256_cvtepu8_epi16, _mm256_hadd_epi32, _mm256_loadu_si256, _mm256_madd_epi16,
    _mm256_max_epu8, _mm256_min_epu8, _mm256_set1_epi16, _mm256_setr_epi64x, _mm256_setzero_si256,
    _mm256_storeu_si256, _mm256_sub_epi8, _mm256_sub_epi16, _mm256_unpackhi_epi8,
    _mm256_unpacklo_epi8,
};

use super::x86_narrow::Avx2Rows;
use super::{DiffFns, Prediction, ResidualRows, Variance, VarianceEnergy};
use super::{generic, x86_narrow};
use crate::simd::avx2::{I16x4, I16x8, I16x16, sum_i32x8};

/// The residual table.
///
/// # Safety
///
/// The CPU must support AVX2 (`Isa::Avx2.is_supported()`).
pub(crate) unsafe fn diff_fns() -> DiffFns {
    DiffFns {
        residual: [
            residual_w4::<false>,
            residual_w8::<false>,
            residual_w16::<false>,
            residual_w32::<false>,
            residual_w64::<false>,
        ],
        residual_sad: [
            residual_w4::<true>,
            residual_w8::<true>,
            residual_w16::<true>,
            residual_w32::<true>,
            residual_w64::<true>,
        ],
        reconstruct: [
            reconstruct_w4,
            reconstruct_w8,
            reconstruct_w16,
            reconstruct_w32,
            reconstruct_w64,
        ],
        variance: [
            variance_w4,
            variance_w8,
            variance_w16,
            variance_w32,
            variance_w64,
        ],
        variance_energy: [
            variance_energy_w4,
            variance_energy_w8,
            variance_energy_w16,
            variance_energy_w32,
            variance_energy_w64,
        ],
        block_sums: [
            block_sums_w4,
            block_sums_w8,
            block_sums_w16,
            block_sums_w32,
            block_sums_w64,
        ],
        halfpel_cross: Some([
            halfpel_cross::<16>,
            halfpel_cross::<32>,
            halfpel_cross::<64>,
        ]),
        sub_pixel_variance: [
            sub_pixel_variance_w4,
            sub_pixel_variance_w8,
            sub_pixel_variance_w16,
            sub_pixel_variance_w32,
            sub_pixel_variance_w64,
        ],
        sse_8x8_row,
        plane_sse,
        variance_8x8_row: Some(super::x86_variance_row::avx2),
    }
}

/// Measure the four half-pel candidates with the same rolling byte
/// loads and rounded averages as the Ice Lake tier.
#[target_feature(enable = "avx2")]
unsafe fn halfpel_cross<const W: usize>(
    src: &[u8],
    ss: usize,
    rf: &[u8],
    rs: usize,
    h: usize,
) -> [Variance; 4] {
    let zero = _mm256_setzero_si256();
    let mut cross = HalfpelCross::<W> {
        sum: [zero; 4],
        squares: [zero; 4],
    };
    // SAFETY: the caller validated both the block and its half-pel window.
    unsafe { super::halfpel_cross::measure::<_, W>(&mut cross, src, ss, rf, rs, h) };
    [
        reduce_cross::<W>(cross.sum[0], cross.squares[0]),
        reduce_cross::<W>(cross.sum[1], cross.squares[1]),
        reduce_cross::<W>(cross.sum[2], cross.squares[2]),
        reduce_cross::<W>(cross.sum[3], cross.squares[3]),
    ]
}

#[target_feature(enable = "avx2")]
fn reduce_cross<const W: usize>(sum: __m256i, squares: __m256i) -> Variance {
    // Widths 16 and 32 add at most 2 * 64 signed byte differences per lane.
    let sum = if W < 64 {
        _mm256_madd_epi16(sum, _mm256_set1_epi16(1))
    } else {
        sum
    };
    Variance {
        sum: sum_i32x8(sum) as i32,
        sse: sum_i32x8(squares) as u32,
    }
}

struct HalfpelCross<const W: usize> {
    sum: [__m256i; 4],
    squares: [__m256i; 4],
}

impl<const W: usize> super::halfpel_cross::Cross for HalfpelCross<W> {
    type Pixels = __m128i;
    const LANES: usize = 16;

    #[allow(clippy::inline_always)]
    #[inline(always)]
    unsafe fn load(ptr: *const u8) -> Self::Pixels {
        // SAFETY: the AVX2 wrapper validates the window for 16-byte loads.
        unsafe { _mm_loadu_si128(ptr.cast()) }
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn update(&mut self, source: Self::Pixels, middle: Self::Pixels, taps: [Self::Pixels; 4]) {
        // SAFETY: only the AVX2 wrapper uses this. The 64-wide block
        // widens each paired sum before accumulating its four stripes.
        unsafe {
            let source = _mm256_cvtepu8_epi16(source);
            for (i, tap) in taps.into_iter().enumerate() {
                let pred = _mm256_cvtepu8_epi16(_mm_avg_epu8(middle, tap));
                let difference = _mm256_sub_epi16(source, pred);
                self.sum[i] = if W < 64 {
                    _mm256_add_epi16(self.sum[i], difference)
                } else {
                    let pairs = _mm256_madd_epi16(difference, _mm256_set1_epi16(1));
                    _mm256_add_epi32(self.sum[i], pairs)
                };
                self.squares[i] =
                    _mm256_add_epi32(self.squares[i], _mm256_madd_epi16(difference, difference));
            }
        }
    }
}

/// Defines the four kernels of one width on one lane type.
macro_rules! width_kernels {
    ($lanes:ty, $stats:ident, $stats_lanes:ty, $width:literal, $residual:ident, $add:ident, $variance:ident, $energy:ident, $subpel:ident) => {
        #[target_feature(enable = "avx2")]
        unsafe fn $residual<const SUM_ABS: bool>(
            src: &[u8],
            src_stride: usize,
            pred: &[u8],
            pred_stride: usize,
            out: &mut [i16],
            out_stride: usize,
            h: usize,
        ) -> u32 {
            // SAFETY: AVX2 is enabled for this function.
            unsafe {
                generic::residual::<$lanes, $width, SUM_ABS>(
                    src,
                    src_stride,
                    pred,
                    pred_stride,
                    out,
                    out_stride,
                    h,
                )
            }
        }

        #[target_feature(enable = "avx2")]
        unsafe fn $add(
            prediction: Prediction<'_>,
            residual: ResidualRows<'_>,
            dst: &mut [u8],
            dst_stride: usize,
            h: usize,
        ) {
            // SAFETY: AVX2 is enabled for this function.
            unsafe {
                generic::reconstruct::<$lanes, $width>(prediction, residual, dst, dst_stride, h);
            }
        }

        #[target_feature(enable = "avx2")]
        pub(super) unsafe fn $variance(
            src: &[u8],
            src_stride: usize,
            pred: &[u8],
            pred_stride: usize,
            h: usize,
        ) -> Variance {
            // SAFETY: AVX2 is enabled and dispatch validates the block.
            unsafe {
                if $width == 16 {
                    x86_narrow::variance_w16(src, src_stride, pred, pred_stride, h)
                } else {
                    $stats::variance::<$stats_lanes, $width>(src, src_stride, pred, pred_stride, h)
                }
            }
        }

        #[target_feature(enable = "avx2")]
        unsafe fn $energy(
            src: &[u8],
            src_stride: usize,
            pred: &[u8],
            pred_stride: usize,
            h: usize,
        ) -> VarianceEnergy {
            // SAFETY: AVX2 is enabled for this function.
            unsafe {
                $stats::variance_energy::<$stats_lanes, $width>(
                    src,
                    src_stride,
                    pred,
                    pred_stride,
                    h,
                )
            }
        }

        #[target_feature(enable = "avx2")]
        unsafe fn $subpel(
            rf: &[u8],
            ref_stride: usize,
            x_offset: usize,
            y_offset: usize,
            src: &[u8],
            src_stride: usize,
            h: usize,
        ) -> Variance {
            // SAFETY: AVX2 is enabled for this function.
            unsafe {
                generic::sub_pixel_variance::<$lanes, $width>(
                    rf, ref_stride, x_offset, y_offset, src, src_stride, h,
                )
            }
        }
    };
}

width_kernels!(
    I16x4,
    x86_narrow,
    Avx2Rows,
    4,
    residual_w4,
    reconstruct_w4,
    variance_w4,
    variance_energy_w4,
    sub_pixel_variance_w4
);
width_kernels!(
    I16x8,
    x86_narrow,
    Avx2Rows,
    8,
    residual_w8,
    reconstruct_w8,
    variance_w8,
    variance_energy_w8,
    sub_pixel_variance_w8
);
width_kernels!(
    I16x16,
    generic,
    I16x16,
    16,
    residual_w16,
    reconstruct_w16,
    variance_w16,
    variance_energy_w16,
    sub_pixel_variance_w16
);
width_kernels!(
    I16x16,
    generic,
    I16x16,
    32,
    residual_w32,
    reconstruct_w32,
    variance_w32,
    variance_energy_w32,
    sub_pixel_variance_w32
);
width_kernels!(
    I16x16,
    generic,
    I16x16,
    64,
    residual_w64,
    reconstruct_w64,
    variance_w64,
    variance_energy_w64,
    sub_pixel_variance_w64
);

/// Defines one width's block-sums kernel on a lane type.
macro_rules! block_sums_kernel {
    ($stats:ident, $lanes:ty, $width:literal, $name:ident) => {
        #[target_feature(enable = "avx2")]
        unsafe fn $name(src: &[u8], src_stride: usize, h: usize) -> Variance {
            // SAFETY: AVX2 is enabled for this function.
            unsafe { $stats::block_sums::<$lanes, $width>(src, src_stride, h) }
        }
    };
}

block_sums_kernel!(x86_narrow, Avx2Rows, 4, block_sums_w4);
block_sums_kernel!(x86_narrow, Avx2Rows, 8, block_sums_w8);
block_sums_kernel!(generic, I16x16, 16, block_sums_w16);
block_sums_kernel!(generic, I16x16, 32, block_sums_w32);
block_sums_kernel!(generic, I16x16, 64, block_sums_w64);

/// Blocks measured per iteration of [`sse_8x8_row`]: one 32-byte row load
/// covers four 8x8 blocks.
const SSE_GROUP: usize = 4;

/// SSE of each 8x8 block along a row: four blocks per 32-byte row load
/// ([`sse_8x8_quad`]), single blocks for the tail ([`sse_8x8`]).
#[target_feature(enable = "avx2")]
unsafe fn sse_8x8_row(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    out: &mut [u32],
) {
    let groups = out.len() / SSE_GROUP;
    let (quads, tail) = out.as_chunks_mut::<SSE_GROUP>();
    for (g, quad) in quads.iter_mut().enumerate() {
        let x = 32 * g;
        // SAFETY: blocks 4g..4g + 4 lie inside the row (kernel contract).
        *quad = unsafe { sse_8x8_quad(&src[x..], src_stride, &pred[x..], pred_stride) };
    }
    for (i, sse) in tail.iter_mut().enumerate() {
        let x = 8 * (SSE_GROUP * groups + i);
        // SAFETY: the block lies inside the row (kernel contract).
        *sse = unsafe { sse_8x8(&src[x..], src_stride, &pred[x..], pred_stride) };
    }
}

/// `|a - b|` per byte, widened into its two 16-bit halves (per 128-bit lane:
/// bytes 0..8 and 8..16), each squared and summed in pairs into i32 lanes
/// (at most 2 * 255^2).
#[target_feature(enable = "avx2")]
fn squared_differences(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
    let zero = _mm256_setzero_si256();
    let d = _mm256_sub_epi8(_mm256_max_epu8(a, b), _mm256_min_epu8(a, b));
    let (lo, hi) = (_mm256_unpacklo_epi8(d, zero), _mm256_unpackhi_epi8(d, zero));
    (_mm256_madd_epi16(lo, lo), _mm256_madd_epi16(hi, hi))
}

/// SSE of four horizontally adjacent 8x8 blocks. Per 128-bit lane the low
/// unpack holds blocks 0 and 2, the high one blocks 1 and 3 (at most
/// 64 * 255^2 per block, well inside i32).
///
/// # Safety
///
/// Kernel contract for a 32-sample-wide block.
#[target_feature(enable = "avx2")]
unsafe fn sse_8x8_quad(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
) -> [u32; SSE_GROUP] {
    let (mut even, mut odd) = (_mm256_setzero_si256(), _mm256_setzero_si256());
    for y in 0..8 {
        // SAFETY: row y (32 bytes) is inside the four blocks of both.
        let (a, b) = unsafe {
            (
                _mm256_loadu_si256(src.as_ptr().add(y * src_stride).cast()),
                _mm256_loadu_si256(pred.as_ptr().add(y * pred_stride).cast()),
            )
        };
        let (lo, hi) = squared_differences(a, b);
        even = _mm256_add_epi32(even, lo);
        odd = _mm256_add_epi32(odd, hi);
    }
    // Two horizontal adds leave block 0 in lane 0, block 1 in lane 1,
    // block 2 in lane 4 and block 3 in lane 5.
    let pairs = _mm256_hadd_epi32(even, odd);
    let totals = _mm256_hadd_epi32(pairs, pairs);
    let mut lanes = [0u32; 8];
    // SAFETY: `lanes` is 32 bytes.
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), totals) };
    [lanes[0], lanes[1], lanes[4], lanes[5]]
}

/// SSE of one 8x8 block: its eight rows gathered four to a register.
///
/// # Safety
///
/// Kernel contract for an 8-sample-wide block.
#[target_feature(enable = "avx2")]
unsafe fn sse_8x8(src: &[u8], src_stride: usize, pred: &[u8], pred_stride: usize) -> u32 {
    let rows = |data: &[u8], stride: usize, y0: usize| {
        // SAFETY: rows y0..y0 + 4 (8 bytes each) are inside the block.
        let row =
            |y: usize| unsafe { data.as_ptr().add(y * stride).cast::<i64>().read_unaligned() };
        _mm256_setr_epi64x(row(y0), row(y0 + 1), row(y0 + 2), row(y0 + 3))
    };
    let mut total = _mm256_setzero_si256();
    for y0 in [0, 4] {
        let (lo, hi) = squared_differences(rows(src, src_stride, y0), rows(pred, pred_stride, y0));
        total = _mm256_add_epi32(total, _mm256_add_epi32(lo, hi));
    }
    let mut lanes = [0u32; 8];
    // SAFETY: `lanes` is 32 bytes.
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), total) };
    lanes.iter().sum()
}

#[target_feature(enable = "avx2")]
unsafe fn plane_sse(a: &nc_common::PlaneView<'_>, b: &nc_common::PlaneView<'_>) -> u64 {
    // SAFETY: the dispatch verifies the ISA and that both planes match.
    unsafe { generic::plane_sse::<I16x16>(a, b) }
}
