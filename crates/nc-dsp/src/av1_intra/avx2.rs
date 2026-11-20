//! AVX2 AV1 intra kernels: filter intra patches by byte-pair
//! multiply-adds, after dav1d 1.5.4 `src/x86/ipred_avx2.asm`
//! (`ipred_filter_8bpc`'s `FILTER_XMM`, BSD-2-Clause, see `THIRD_PARTY.md`):
//! a patch's seven inputs paired by one `pshufb`, each pair broadcast to the
//! eight outputs and multiplied by their two taps with `pmaddubsw`, four of
//! those per patch (the other ISAs' generic code broadcasts each input
//! alone: seven multiplies and scalar gathers). Directional prediction runs
//! its interpolation with the shared directional run (`crate::intra_runs`,
//! byte multiply-adds) and transposes its column tiles 8x8 in registers.
//! Smooth blends in unsigned 16-bit lanes (`smooth`); Paeth is the generic
//! code.

#![allow(clippy::inline_always)]

use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi16, _mm_and_si128, _mm_avg_epu16, _mm_cvtepu8_epi16,
    _mm_cvtsi128_si32, _mm_extract_epi32, _mm_insert_epi8, _mm_loadl_epi64, _mm_loadu_si128,
    _mm_maddubs_epi16, _mm_mullo_epi16, _mm_packus_epi16, _mm_set1_epi16, _mm_setr_epi8,
    _mm_shuffle_epi8, _mm_shuffle_epi32, _mm_srai_epi16, _mm_srli_epi16, _mm_storel_epi64,
    _mm_storeu_si128, _mm_sub_epi16, _mm_xor_si128, _mm256_add_epi16, _mm256_and_si256,
    _mm256_avg_epu16, _mm256_castsi256_si128, _mm256_cvtepu8_epi16, _mm256_extracti128_si256,
    _mm256_mullo_epi16, _mm256_set1_epi16, _mm256_srli_epi16, _mm256_sub_epi16, _mm256_xor_si256,
};

use nc_common::PlaneViewMut;

use crate::intra_prediction::directional::DirectionalKernels;
use crate::intra_prediction::filter::{
    FILTER_INTRA_MAX_SIZE, FILTER_TAP_PAIRS, FilterGrid, FilterIntraMode, SCALE_BITS,
};
use crate::intra_prediction::{IntraEdges, IntraMode, SMOOTH_SHIFT, smooth_weights};

super::generic::av1_intra_kernels!(#[target_feature(enable = "avx2")];
    crate::simd::avx2::I16x8,
    crate::simd::avx2::I16x16
);

/// Patch width and height (`filter`'s 4x2 patches).
const PATCH_W: usize = 4;
const PATCH_H: usize = 2;

/// Filter intra patches of a `width`x`height` block
/// (`filter::predict_patches`), one patch per 128-bit register.
///
/// Each `pmaddubsw` pair sum is at most `255 · 32` in magnitude (|tap| ≤
/// 16), the patch sum at most `255 · 52`: exact in 16 bits. The arithmetic
/// shift of `sum + 8` differs from `Round2Signed` only below zero, where
/// the saturating pack clips both to 0.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn filter_patches(
    grid: &mut FilterGrid,
    width: usize,
    height: usize,
    mode: FilterIntraMode,
) {
    /// Samples per grid row: the left edge column and the block.
    const ROW: usize = FILTER_INTRA_MAX_SIZE + 1;
    assert!(width <= FILTER_INTRA_MAX_SIZE && height <= FILTER_INTRA_MAX_SIZE);
    let taps = &FILTER_TAP_PAIRS[mode.symbol()];
    // SAFETY: AVX2 is enabled; each pair row is 16 bytes.
    let (t01, t23, t45, t6) = unsafe {
        (
            _mm_loadu_si128(taps[0].as_ptr().cast()),
            _mm_loadu_si128(taps[1].as_ptr().cast()),
            _mm_loadu_si128(taps[2].as_ptr().cast()),
            _mm_loadu_si128(taps[3].as_ptr().cast()),
        )
    };
    // Bytes: corner, above 0..3, left 0, left 1 (from the loads below), as
    // the pairs (0 1) (2 3) (4 5) (6 −) twice per 32-bit lane.
    let pairs = _mm_setr_epi8(0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, -1, 6, -1);
    let round = _mm_set1_epi16(1 << (SCALE_BITS - 1));
    for r in (1..=height).step_by(PATCH_H) {
        for c in (1..=width).step_by(PATCH_W) {
            let flat = grid.as_flattened();
            let start = (r - 1) * ROW + c - 1;
            // The corner and four above (eight bytes read; `r + 1 ≤ 32`, so
            // the row after is in the grid), then the two left inputs.
            let above = &flat[start..start + 8];
            // SAFETY: AVX2 is enabled; `above` holds the 8 bytes read.
            let v = unsafe { _mm_loadl_epi64(above.as_ptr().cast()) };
            let v = _mm_insert_epi8::<5>(v, i32::from(grid[r][c - 1]));
            let v = _mm_shuffle_epi8(
                _mm_insert_epi8::<6>(v, i32::from(grid[r + 1][c - 1])),
                pairs,
            );
            let sum = _mm_add_epi16(
                _mm_add_epi16(
                    _mm_maddubs_epi16(_mm_shuffle_epi32::<0x00>(v), t01),
                    _mm_maddubs_epi16(_mm_shuffle_epi32::<0x55>(v), t23),
                ),
                _mm_add_epi16(
                    _mm_add_epi16(_mm_maddubs_epi16(_mm_shuffle_epi32::<0xaa>(v), t45), round),
                    _mm_maddubs_epi16(_mm_shuffle_epi32::<0xff>(v), t6),
                ),
            );
            let out = _mm_packus_epi16(_mm_srai_epi16::<{ SCALE_BITS as i32 }>(sum), sum);
            let rows = [_mm_cvtsi128_si32(out), _mm_extract_epi32::<1>(out)];
            for (dr, value) in rows.into_iter().enumerate() {
                grid[r + dr][c..c + PATCH_W].copy_from_slice(&value.to_ne_bytes());
            }
        }
    }
}

/// Outputs per step of the smooth kernels: a 128-bit and a 64-bit row.
const STEP: usize = 16;
const HALF_STEP: usize = 8;

/// AVX2 directional kernels.
struct Avx2Directional;

impl DirectionalKernels for Avx2Directional {
    #[inline(always)]
    fn run(window: &[u8], weight: usize, out: &mut [u8]) {
        // SAFETY: AVX2 is enabled (the caller's target feature).
        unsafe { crate::intra_runs::avx2::run2(window, weight, out) }
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

/// AVX2 8-bit directional interpolation run (for rectangular blocks too).
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn run_u8(window: &[u8], weight: usize, out: &mut [u8]) {
    <Avx2Directional as DirectionalKernels>::run(window, weight, out);
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
    // One inlined copy per mode, so that no row tests the mode again.
    // SAFETY: AVX2 is enabled (this function's target feature).
    unsafe {
        match mode {
            IntraMode::Smooth => smooth_rows(edges, IntraMode::Smooth, dst),
            IntraMode::SmoothVertical => smooth_rows(edges, IntraMode::SmoothVertical, dst),
            _ => smooth_rows(edges, IntraMode::SmoothHorizontal, dst),
        }
    }
}

/// The rows of [`smooth`] for one `mode`.
///
/// # Safety
///
/// AVX2 must be enabled in the caller.
#[inline(always)]
unsafe fn smooth_rows(edges: &IntraEdges, mode: IntraMode, dst: &mut PlaneViewMut<'_>) {
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
