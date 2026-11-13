//! AVX2 AV1 intra kernels: filter intra is the generic code on 8 lanes;
//! directional prediction runs its interpolation with byte multiply-adds
//! (`_mm*_maddubs_epi16` on interleaved sample pairs and weights; every
//! `a · (32 − w) + b · w` fits i16) and transposes its column tiles 8x8 in
//! registers. Smooth blends in unsigned 16-bit lanes (`smooth`); Paeth is
//! the generic code.

#![allow(clippy::inline_always)]

use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi16, _mm_and_si128, _mm_avg_epu16, _mm_cvtepu8_epi16,
    _mm_loadl_epi64, _mm_loadu_si128, _mm_maddubs_epi16, _mm_mulhrs_epi16, _mm_mullo_epi16,
    _mm_packus_epi16, _mm_set1_epi16, _mm_srli_epi16, _mm_storel_epi64, _mm_storeu_si128,
    _mm_sub_epi16, _mm_unpackhi_epi8, _mm_unpacklo_epi8, _mm_xor_si128, _mm256_add_epi16,
    _mm256_and_si256, _mm256_avg_epu16, _mm256_castsi256_si128, _mm256_cvtepu8_epi16,
    _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_maddubs_epi16, _mm256_mulhrs_epi16,
    _mm256_mullo_epi16, _mm256_packus_epi16, _mm256_set1_epi16, _mm256_srli_epi16,
    _mm256_storeu_si256, _mm256_sub_epi16, _mm256_unpackhi_epi8, _mm256_unpacklo_epi8,
    _mm256_xor_si256,
};

use nc_common::PlaneViewMut;

use crate::intra_prediction::directional::{self, DirectionalKernels};
use crate::intra_prediction::{IntraEdges, IntraMode, SMOOTH_SHIFT, smooth_weights};

super::generic::av1_intra_kernels!(#[target_feature(enable = "avx2")];
    crate::simd::avx2::I16x8,
    crate::simd::avx2::I16x16
);

/// Outputs per step: a 256-bit, a 128-bit and a 64-bit run.
const WIDE_STEP: usize = 32;
const STEP: usize = 16;
const HALF_STEP: usize = 8;
/// The interpolation weights sum to this.
const WEIGHT_SUM: i16 = 32;
/// `_mm_mulhrs_epi16` by this is `Round2(x, 5)`: `(x · 2^10 + 2^14) >> 15`.
const ROUND2_5: i16 = 1 << 10;

/// AVX2 directional kernels.
struct Avx2Directional;

impl DirectionalKernels for Avx2Directional {
    #[inline(always)]
    fn run(window: &[u8], weight: usize, out: &mut [u8]) {
        assert_eq!(window.len(), out.len() + 1);
        let len = out.len();
        if len < HALF_STEP {
            directional::interpolate_run(window, weight, out);
            return;
        }
        let far = weight as i16;
        // Byte pairs (near weight, far weight) for the (a, b) pairs.
        let pair = (far << 8) | (WEIGHT_SUM - far);
        // SAFETY: AVX2 is enabled (the caller's target feature); every load
        // and store covers checked subslices.
        unsafe {
            let (weights, round) = (_mm256_set1_epi16(pair), _mm256_set1_epi16(ROUND2_5));
            let mut done = 0;
            while done + WIDE_STEP <= len {
                let a = _mm256_loadu_si256(window[done..][..WIDE_STEP].as_ptr().cast());
                let b = _mm256_loadu_si256(window[done + 1..][..WIDE_STEP].as_ptr().cast());
                // Per 128-bit lane: unpack, multiply-add and pack keep the
                // sample order.
                let low = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(a, b), weights);
                let high = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(a, b), weights);
                let packed = _mm256_packus_epi16(
                    _mm256_mulhrs_epi16(low, round),
                    _mm256_mulhrs_epi16(high, round),
                );
                _mm256_storeu_si256(out[done..][..WIDE_STEP].as_mut_ptr().cast(), packed);
                done += WIDE_STEP;
            }
            let (weights, round) = (_mm_set1_epi16(pair), _mm_set1_epi16(ROUND2_5));
            if done + STEP <= len {
                let a = _mm_loadu_si128(window[done..][..STEP].as_ptr().cast());
                let b = _mm_loadu_si128(window[done + 1..][..STEP].as_ptr().cast());
                let low =
                    _mm_mulhrs_epi16(_mm_maddubs_epi16(_mm_unpacklo_epi8(a, b), weights), round);
                let high =
                    _mm_mulhrs_epi16(_mm_maddubs_epi16(_mm_unpackhi_epi8(a, b), weights), round);
                _mm_storeu_si128(
                    out[done..][..STEP].as_mut_ptr().cast(),
                    _mm_packus_epi16(low, high),
                );
                done += STEP;
            }
            // The rest in half steps, the last one ending at `len` (it may
            // recompute outputs already stored, with the same values).
            while done < len {
                let at = done.min(len - HALF_STEP);
                let a = _mm_loadl_epi64(window[at..][..HALF_STEP].as_ptr().cast());
                let b = _mm_loadl_epi64(window[at + 1..][..HALF_STEP].as_ptr().cast());
                let sum =
                    _mm_mulhrs_epi16(_mm_maddubs_epi16(_mm_unpacklo_epi8(a, b), weights), round);
                _mm_storel_epi64(
                    out[at..][..HALF_STEP].as_mut_ptr().cast(),
                    _mm_packus_epi16(sum, sum),
                );
                done = at + HALF_STEP;
            }
        }
    }

    #[inline(always)]
    fn store_transposed(columns: &[u8], side: usize, dst: &mut PlaneViewMut<'_>) {
        // SAFETY: AVX2 is enabled (the caller's target feature).
        unsafe {
            super::generic::store_transposed::<crate::lpf_bytes::avx2::B>(columns, side, dst);
        }
    }
}

/// `IntraEdges::predict_angle` with the AVX2 kernels.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn predict_angle(edges: &IntraEdges, angle: u16, dst: &mut PlaneViewMut<'_>) {
    edges.predict_angle_with::<Avx2Directional>(angle, dst);
}

/// One row's terms of a smooth prediction in u16 lanes (every value is at
/// most `255 · 256`, so wrapping 16-bit arithmetic is exact):
/// `v = wy · top + (256 − wy) · bottom` and `h = 256 · right + wx · (left
/// − right)`.
#[derive(Clone, Copy)]
struct SmoothRow {
    /// `wy`, and `(256 − wy) · bottom`.
    wy: i16,
    vertical: i16,
    /// `left − right` (wrapping), and `256 · right`.
    slope: i16,
    horizontal: i16,
}

impl SmoothRow {
    fn new(wy: u8, left: u8, bottom: u8, right: u8) -> Self {
        let scale = 1u16 << SMOOTH_SHIFT;
        Self {
            wy: i16::from(wy),
            vertical: ((scale - u16::from(wy)) * u16::from(bottom)) as i16,
            slope: (u16::from(left).wrapping_sub(u16::from(right))) as i16,
            horizontal: (scale * u16::from(right)) as i16,
        }
    }
}

/// `Round2(·, 8)`'s rounding term.
const SMOOTH_ROUND: i16 = 1 << (SMOOTH_SHIFT - 1);

/// Smooth outputs (as u16 lanes) from `top` and column weights `wx`
/// widened to u16. Smooth's `(v + h + 256) >> 9` is `(⌊(v + h) / 2⌋ +
/// 128) >> 8`, and `⌊(v + h) / 2⌋` is the rounding average minus the
/// carried low bit.
#[inline(always)]
fn smooth16(mode: IntraMode, row: SmoothRow, top: __m256i, wx: __m256i) -> __m256i {
    // SAFETY: AVX2 is enabled (the caller's target feature); register
    // operations only.
    unsafe {
        let vertical = _mm256_add_epi16(
            _mm256_mullo_epi16(top, _mm256_set1_epi16(row.wy)),
            _mm256_set1_epi16(row.vertical),
        );
        let horizontal = _mm256_add_epi16(
            _mm256_mullo_epi16(wx, _mm256_set1_epi16(row.slope)),
            _mm256_set1_epi16(row.horizontal),
        );
        let sum = match mode {
            IntraMode::Smooth => {
                let carry =
                    _mm256_and_si256(_mm256_xor_si256(vertical, horizontal), _mm256_set1_epi16(1));
                _mm256_sub_epi16(_mm256_avg_epu16(vertical, horizontal), carry)
            }
            IntraMode::SmoothVertical => vertical,
            _ => horizontal,
        };
        _mm256_srli_epi16::<8>(_mm256_add_epi16(sum, _mm256_set1_epi16(SMOOTH_ROUND)))
    }
}

/// [`smooth16`] on eight lanes.
#[inline(always)]
fn smooth8(mode: IntraMode, row: SmoothRow, top: __m128i, wx: __m128i) -> __m128i {
    // SAFETY: AVX2 is enabled (the caller's target feature); register
    // operations only.
    unsafe {
        let vertical = _mm_add_epi16(
            _mm_mullo_epi16(top, _mm_set1_epi16(row.wy)),
            _mm_set1_epi16(row.vertical),
        );
        let horizontal = _mm_add_epi16(
            _mm_mullo_epi16(wx, _mm_set1_epi16(row.slope)),
            _mm_set1_epi16(row.horizontal),
        );
        let sum = match mode {
            IntraMode::Smooth => {
                let carry = _mm_and_si128(_mm_xor_si128(vertical, horizontal), _mm_set1_epi16(1));
                _mm_sub_epi16(_mm_avg_epu16(vertical, horizontal), carry)
            }
            IntraMode::SmoothVertical => vertical,
            _ => horizontal,
        };
        _mm_srli_epi16::<8>(_mm_add_epi16(sum, _mm_set1_epi16(SMOOTH_ROUND)))
    }
}

/// AV1 smooth prediction (§7.11.2.6) of `mode`, one of the three smooth
/// modes, on a side of 8 or more: 16 outputs per step (8 on 8x8).
#[target_feature(enable = "avx2")]
unsafe fn smooth(edges: &IntraEdges, mode: IntraMode, dst: &mut PlaneViewMut<'_>) {
    let n = edges.size();
    assert!(n >= HALF_STEP && dst.width() == n && dst.height() == n);
    let (above, left) = (edges.above_row(), edges.left_column());
    let weights = smooth_weights(n);
    let (bottom, right) = (left[n - 1], above[n - 1]);
    // SAFETY: AVX2 is enabled; every load and store covers checked
    // subslices.
    unsafe {
        for (y, (&wy, &l)) in weights.iter().zip(left).enumerate() {
            let row = SmoothRow::new(wy, l, bottom, right);
            let out = &mut dst.row_mut(y)[..n];
            if n == HALF_STEP {
                let top = _mm_cvtepu8_epi16(_mm_loadl_epi64(above[..HALF_STEP].as_ptr().cast()));
                let wx = _mm_cvtepu8_epi16(_mm_loadl_epi64(weights[..HALF_STEP].as_ptr().cast()));
                let value = smooth8(mode, row, top, wx);
                _mm_storel_epi64(
                    out[..HALF_STEP].as_mut_ptr().cast(),
                    _mm_packus_epi16(value, value),
                );
                continue;
            }
            for x0 in (0..n).step_by(STEP) {
                let top =
                    _mm256_cvtepu8_epi16(_mm_loadu_si128(above[x0..][..STEP].as_ptr().cast()));
                let wx =
                    _mm256_cvtepu8_epi16(_mm_loadu_si128(weights[x0..][..STEP].as_ptr().cast()));
                let value = smooth16(mode, row, top, wx);
                let packed = _mm_packus_epi16(
                    _mm256_castsi256_si128(value),
                    _mm256_extracti128_si256::<1>(value),
                );
                _mm_storeu_si128(out[x0..][..STEP].as_mut_ptr().cast(), packed);
            }
        }
    }
}
