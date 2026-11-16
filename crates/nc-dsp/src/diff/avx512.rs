//! AVX-512 variance with 32 signed-16-bit differences per vector. As in
//! SVT-AV1 v4.3.0 `variance_avx512.c` (Intel 2021, BSD-2-Clause and AOM
//! patent licence), sums stay narrow until the final reduction; squared
//! differences use paired widening multiplies. Our block contract bounds
//! each narrow lane by 2*64*255=32640, including 64-wide blocks.
//!
//! Sixteen-wide blocks pack two rows into a vector: a half-empty 512-bit
//! vector was slower than AVX2 on Zen 5 and Sapphire Rapids (0.65–0.84x, docs/handoff/avx512.md). Every
//! block height is a multiple of four, so packed rows never leave the block.
//!
//! Widths 4 and 8 use the shared packed-byte graph in `x86_narrow` (SVT
//! byte-pair differences and ICL VNNI squares), including four-row tails.
//!
//! Sub-pixel variance at widths 32 and 64 runs the shared bilinear graph
//! (`generic::sub_pixel_variance`) on 32 word lanes; width 16 packs two rows
//! per vector and shares each first-pass row between the two output rows
//! that use it.

use super::generic::{self, covers};
use super::x86_narrow::{self, IclRows};
use super::{DiffFns, Variance, VarianceEnergy, bilinear_taps};
use crate::simd::avx512_i16::I16x32;
use core::arch::x86_64::{
    __m512i, _mm_loadu_si128, _mm256_castsi128_si256, _mm256_inserti128_si256,
    _mm256_maskz_loadu_epi8, _mm512_add_epi16, _mm512_add_epi32, _mm512_cvtepu8_epi16,
    _mm512_madd_epi16, _mm512_mulhrs_epi16, _mm512_reduce_add_epi32, _mm512_set1_epi16,
    _mm512_setzero_si512, _mm512_shuffle_i64x2, _mm512_sub_epi16,
};

/// Word lanes per vector.
const LANES: usize = 32;

pub(crate) fn install(table: &mut DiffFns) {
    table.variance[..2].copy_from_slice(&[narrow_variance::<4>, narrow_variance::<8>]);
    table.variance[2..].copy_from_slice(&[variance::<16>, variance::<32>, variance::<64>]);
    table.variance_energy[..2].copy_from_slice(&[narrow_energy::<4>, narrow_energy::<8>]);
    table.variance_energy[2..].copy_from_slice(&[energy::<16>, energy::<32>, energy::<64>]);
    table.block_sums[..2].copy_from_slice(&[block_sums::<4>, block_sums::<8>]);
    table.sub_pixel_variance[2..].copy_from_slice(&[
        sub_pixel_variance_w16,
        sub_pixel_variance_wide::<32>,
        sub_pixel_variance_wide::<64>,
    ]);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx512vnni,avx2")]
unsafe fn narrow_variance<const W: usize>(
    src: &[u8],
    ss: usize,
    pred: &[u8],
    ps: usize,
    h: usize,
) -> Variance {
    // SAFETY: the dispatcher forwards the validated block and ICL contracts.
    unsafe { x86_narrow::variance::<IclRows, W>(src, ss, pred, ps, h) }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx512vnni,avx2")]
unsafe fn narrow_energy<const W: usize>(
    src: &[u8],
    ss: usize,
    pred: &[u8],
    ps: usize,
    h: usize,
) -> VarianceEnergy {
    // SAFETY: the dispatcher forwards the validated block and ICL contracts.
    unsafe { x86_narrow::variance_energy::<IclRows, W>(src, ss, pred, ps, h) }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn variance<const W: usize>(
    src: &[u8],
    ss: usize,
    pred: &[u8],
    ps: usize,
    h: usize,
) -> Variance {
    // SAFETY: the dispatcher's validated block contract is forwarded.
    unsafe { accumulate::<W, false>(src, ss, pred, ps, h).residual }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn energy<const W: usize>(
    src: &[u8],
    ss: usize,
    pred: &[u8],
    ps: usize,
    h: usize,
) -> VarianceEnergy {
    // SAFETY: the dispatcher's validated block contract is forwarded.
    unsafe { accumulate::<W, true>(src, ss, pred, ps, h) }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx512vnni,avx2")]
unsafe fn block_sums<const W: usize>(src: &[u8], ss: usize, h: usize) -> Variance {
    // SAFETY: the dispatcher forwards the validated source and ICL contracts.
    unsafe { x86_narrow::block_sums::<IclRows, W>(src, ss, h) }
}

/// Rows of a `W`-wide block packed into one vector (1 from width 32 up).
const fn rows_per_vector(w: usize) -> usize {
    if w >= LANES { 1 } else { LANES / w }
}

/// Samples `x..x + 32` of row `y` (`W` >= 32, masked to `W`), or rows
/// `y..y + 32 / W` side by side (`W` 16), widened to words.
///
/// # Safety
///
/// AVX-512F/BW/VL available; the rows read lie inside `data` at `stride`.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn load_rows<const W: usize>(data: &[u8], stride: usize, y: usize, x: usize) -> __m512i {
    let at = |r: usize| data.as_ptr().wrapping_add(r * stride + x);
    // SAFETY: the caller guarantees every row read is inside `data`.
    let bytes = unsafe {
        if W == 16 {
            return load_pair(data, stride, y, y + 1, x);
        }
        _mm256_maskz_loadu_epi8(u32::MAX >> (LANES - W.min(LANES)), at(y).cast())
    };
    _mm512_cvtepu8_epi16(bytes)
}

/// Samples `x..x + 16` of rows `y0` and `y1` side by side, widened to words.
///
/// # Safety
///
/// AVX-512F/BW/VL available; both rows' 16 samples lie inside `data`.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn load_pair(data: &[u8], stride: usize, y0: usize, y1: usize, x: usize) -> __m512i {
    let at = |r: usize| data.as_ptr().wrapping_add(r * stride + x);
    // SAFETY: the caller guarantees both rows are inside `data`.
    let bytes = unsafe {
        _mm256_inserti128_si256::<1>(
            _mm256_castsi128_si256(_mm_loadu_si128(at(y0).cast())),
            _mm_loadu_si128(at(y1).cast()),
        )
    };
    _mm512_cvtepu8_epi16(bytes)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn accumulate<const W: usize, const ENERGY: bool>(
    src: &[u8],
    ss: usize,
    pred: &[u8],
    ps: usize,
    h: usize,
) -> VarianceEnergy {
    let rows = rows_per_vector(W);
    debug_assert!(
        h.is_multiple_of(rows) && covers(src.len(), ss, h, W) && covers(pred.len(), ps, h, W)
    );
    let zero = _mm512_setzero_si512();
    let (mut sum, mut squared, mut psum, mut psquared) = (zero, zero, zero, zero);
    for y in (0..h).step_by(rows) {
        for x in (0..W).step_by(LANES) {
            // SAFETY: rows y..y + rows of the block, inside both (contract).
            let (a, b) = unsafe {
                (
                    load_rows::<W>(src, ss, y, x),
                    load_rows::<W>(pred, ps, y, x),
                )
            };
            let d = _mm512_sub_epi16(a, b);
            sum = _mm512_add_epi16(sum, d);
            squared = _mm512_add_epi32(squared, _mm512_madd_epi16(d, d));
            if ENERGY {
                psum = _mm512_add_epi16(psum, b);
                psquared = _mm512_add_epi32(psquared, _mm512_madd_epi16(b, b));
            }
        }
    }
    VarianceEnergy {
        residual: reduce(sum, squared),
        prediction: reduce(psum, psquared),
    }
}

/// The [`Variance`] of narrow word sums and i32 squared sums.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
fn reduce(sum: __m512i, squared: __m512i) -> Variance {
    Variance {
        sum: _mm512_reduce_add_epi32(_mm512_madd_epi16(sum, _mm512_set1_epi16(1))),
        sse: _mm512_reduce_add_epi32(squared) as u32,
    }
}

/// Sub-pixel variance of a 32- or 64-wide block: the shared graph on 32
/// word lanes, one row (or half a row) per vector.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn sub_pixel_variance_wide<const W: usize>(
    rf: &[u8],
    ref_stride: usize,
    x_offset: usize,
    y_offset: usize,
    src: &[u8],
    src_stride: usize,
    h: usize,
) -> Variance {
    // SAFETY: AVX-512F/BW/VL are enabled for this function.
    unsafe {
        generic::sub_pixel_variance::<I16x32, W>(
            rf, ref_stride, x_offset, y_offset, src, src_stride, h,
        )
    }
}

/// Sub-pixel variance of a 16-wide block, two rows per vector. The
/// arithmetic is the shared graph's (`generic::sub_pixel_variance`): a
/// zero offset skips its pass, and each bilinear tap is the exact
/// `a + mulhrs(b - a, f1 << 8)`.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn sub_pixel_variance_w16(
    rf: &[u8],
    ref_stride: usize,
    x_offset: usize,
    y_offset: usize,
    src: &[u8],
    src_stride: usize,
    h: usize,
) -> Variance {
    const W: usize = 16;
    // Checked once, so the row loop's raw loads need no bounds checks.
    assert!(
        h.is_multiple_of(2)
            && covers(rf.len(), ref_stride, h + 1, W + 1)
            && covers(src.len(), src_stride, h, W)
    );
    let block = PairBlock {
        rf,
        ref_stride,
        src,
        src_stride,
        h,
        fx: q15_weight(x_offset),
        fy: q15_weight(y_offset),
    };
    // SAFETY: the assertion above covers every row the passes read.
    unsafe {
        match (x_offset, y_offset) {
            (_, 0) => block.variance::<true, false>(),
            (0, _) => block.variance::<false, true>(),
            _ => block.variance::<true, true>(),
        }
    }
}

/// The second bilinear tap at `offset` as a Q15 weight (see
/// `generic::bilinear_blend`: `f1 <= 112`, so `f1 << 8` fits an i16).
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
fn q15_weight(offset: usize) -> __m512i {
    _mm512_set1_epi16(bilinear_taps(offset).1 << 8)
}

/// `(a * (128 - f1) + b * f1 + 64) >> 7` as `a + mulhrs(b - a, f1 << 8)`,
/// exact for `|b - a| <= 255` (`generic::bilinear_blend`).
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
fn blend(a: __m512i, b: __m512i, w: __m512i) -> __m512i {
    _mm512_add_epi16(a, _mm512_mulhrs_epi16(_mm512_sub_epi16(b, a), w))
}

/// The reference window, source block and Q15 tap weights of one 16-wide
/// sub-pixel variance.
struct PairBlock<'a> {
    rf: &'a [u8],
    ref_stride: usize,
    src: &'a [u8],
    src_stride: usize,
    h: usize,
    fx: __m512i,
    fy: __m512i,
}

impl PairBlock<'_> {
    /// First-pass rows `y` and `second` side by side (`second` is `y + 1`,
    /// or `y` again for the window's last row, which has no successor).
    ///
    /// # Safety
    ///
    /// AVX-512F/BW/VL available; rows `y` and `second` (17 samples each)
    /// inside the reference window.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn first_pass<const H: bool>(&self, y: usize, second: usize) -> __m512i {
        let (rf, stride) = (self.rf, self.ref_stride);
        // SAFETY: the caller guarantees both rows, 17 samples each.
        unsafe {
            let left = load_pair(rf, stride, y, second, 0);
            if H {
                blend(left, load_pair(rf, stride, y, second, 1), self.fx)
            } else {
                left
            }
        }
    }

    /// The variance with the horizontal and vertical passes enabled or
    /// skipped (`H`, `V`); a skipped pass has offset 0.
    ///
    /// # Safety
    ///
    /// AVX-512F/BW/VL available; `rf` holds `h + 1` rows of 17 samples and
    /// `src` `h` rows of 16 (`h` even).
    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn variance<const H: bool, const V: bool>(&self) -> Variance {
        let h = self.h;
        let zero = _mm512_setzero_si512();
        let (mut sum, mut squared) = (zero, zero);
        // First-pass rows (y, y + 1) of the pair being predicted.
        // SAFETY: rows 0 and 1 are inside the window (h >= 2).
        let mut upper = unsafe { self.first_pass::<H>(0, 1) };
        for y in (0..h).step_by(2) {
            let pred = if V {
                // Rows y + 2 and y + 3; the window ends at row h.
                // SAFETY: both rows are at most h, inside the window.
                let next = unsafe { self.first_pass::<H>(y + 2, (y + 3).min(h)) };
                // Rows (y + 1, y + 2): the high half of `upper`, the low of
                // `next` (128-bit groups 2, 3 of the first, 0, 1 of the second).
                let lower = _mm512_shuffle_i64x2::<0b01_00_11_10>(upper, next);
                let pred = blend(upper, lower, self.fy);
                upper = next;
                pred
            } else {
                let pred = upper;
                if y + 2 < h {
                    // SAFETY: rows y + 2 and y + 3 are inside the block.
                    upper = unsafe { self.first_pass::<H>(y + 2, y + 3) };
                }
                pred
            };
            // SAFETY: source rows y and y + 1 are inside the block.
            let source = unsafe { load_rows::<16>(self.src, self.src_stride, y, 0) };
            // A prediction of 8-bit samples, so |diff| <= 255 and each word
            // lane's sum over h / 2 <= 32 rows stays within i16.
            let diff = _mm512_sub_epi16(source, pred);
            sum = _mm512_add_epi16(sum, diff);
            squared = _mm512_add_epi32(squared, _mm512_madd_epi16(diff, diff));
        }
        reduce(sum, squared)
    }
}
