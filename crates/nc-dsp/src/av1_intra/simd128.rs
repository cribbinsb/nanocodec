//! SIMD128 AV1 intra kernels: filter intra is the generic code on 8 lanes;
//! directional prediction runs its interpolation with the shared
//! directional run (`crate::intra_runs`, widening multiply-adds) and
//! transposes its column tiles 8x8 in registers. Smooth
//! blends in unsigned 16-bit lanes (`smooth`); Paeth is the generic code.

#![allow(clippy::inline_always)]

use core::arch::wasm32::{
    i16x8_mul, u8x16_narrow_i16x8, u16x8_add, u16x8_avgr, u16x8_extend_low_u8x16, u16x8_shr,
    u16x8_splat, u16x8_sub, v128, v128_and, v128_load64_zero, v128_store64_lane, v128_xor,
};

use nc_common::PlaneViewMut;

use crate::intra_prediction::directional::DirectionalKernels;
use crate::intra_prediction::{IntraEdges, IntraMode, SMOOTH_SHIFT, smooth_weights};

super::generic::filter_patch_kernel!(; crate::simd::simd128::I16x8);
super::generic::av1_intra_kernels!(;
    crate::simd::simd128::I16x8,
    crate::simd::simd128::I16x8
);

/// Outputs per half step of the smooth kernels.
const HALF_STEP: usize = 8;

/// SIMD128 directional kernels.
struct Simd128Directional;

impl DirectionalKernels for Simd128Directional {
    #[inline(always)]
    fn run(window: &[u8], weight: usize, out: &mut [u8]) {
        // SAFETY: SIMD128 is enabled for the whole build.
        unsafe { crate::intra_runs::simd128::run2(window, weight, out) }
    }

    #[inline(always)]
    fn store_transposed(columns: &[u8], side: usize, dst: &mut PlaneViewMut<'_>) {
        // SAFETY: SIMD128 is enabled for the whole build.
        unsafe {
            super::generic::store_transposed::<crate::lpf_bytes::simd128::B>(columns, side, dst);
        }
    }
}

/// `IntraEdges::predict_angle` with the SIMD128 kernels.
pub(crate) unsafe fn predict_angle(edges: &IntraEdges, angle: u16, dst: &mut PlaneViewMut<'_>) {
    edges.predict_angle_with::<Simd128Directional>(angle, dst);
}

/// One row's terms of a smooth prediction in u16 lanes (every value is at
/// most `255 · 256`, so wrapping 16-bit arithmetic is exact):
/// `v = wy · top + (256 − wy) · bottom` and `h = 256 · right + wx · (left
/// − right)`.
#[derive(Clone, Copy)]
struct SmoothRow {
    /// `wy`, and `(256 − wy) · bottom`.
    wy: v128,
    vertical: v128,
    /// `left − right` (wrapping), and `256 · right`.
    slope: v128,
    horizontal: v128,
}

impl SmoothRow {
    fn new(wy: u8, left: u8, bottom: u8, right: u8) -> Self {
        let scale = 1u16 << SMOOTH_SHIFT;
        Self {
            wy: u16x8_splat(u16::from(wy)),
            vertical: u16x8_splat((scale - u16::from(wy)) * u16::from(bottom)),
            slope: u16x8_splat(u16::from(left).wrapping_sub(u16::from(right))),
            horizontal: u16x8_splat(scale * u16::from(right)),
        }
    }
}

/// Eight smooth outputs (u16 lanes) from `top` and column weights `wx`
/// widened to u16. Smooth's `(v + h + 256) >> 9` is `(⌊(v + h) / 2⌋ +
/// 128) >> 8`, and `⌊(v + h) / 2⌋` is the rounding average minus the
/// carried low bit.
#[inline(always)]
fn smooth8(mode: IntraMode, row: SmoothRow, top: v128, wx: v128) -> v128 {
    let vertical = u16x8_add(i16x8_mul(top, row.wy), row.vertical);
    let horizontal = u16x8_add(i16x8_mul(wx, row.slope), row.horizontal);
    let sum = match mode {
        IntraMode::Smooth => {
            let carry = v128_and(v128_xor(vertical, horizontal), u16x8_splat(1));
            u16x8_sub(u16x8_avgr(vertical, horizontal), carry)
        }
        IntraMode::SmoothVertical => vertical,
        _ => horizontal,
    };
    u16x8_shr(
        u16x8_add(sum, u16x8_splat(1 << (SMOOTH_SHIFT - 1))),
        SMOOTH_SHIFT,
    )
}

/// AV1 smooth prediction (§7.11.2.6) of `mode`, one of the three smooth
/// modes, on a side of 8 or more: 8 outputs per step.
unsafe fn smooth(edges: &IntraEdges, mode: IntraMode, dst: &mut PlaneViewMut<'_>) {
    let n = edges.size();
    assert!(n >= HALF_STEP && dst.width() == n && dst.height() == n);
    let (above, left) = (edges.above_row(), edges.left_column());
    let weights = smooth_weights(n);
    let (bottom, right) = (left[n - 1], above[n - 1]);
    for (y, (&wy, &l)) in weights.iter().zip(left).enumerate() {
        let row = SmoothRow::new(wy, l, bottom, right);
        let out = &mut dst.row_mut(y)[..n];
        for x0 in (0..n).step_by(HALF_STEP) {
            // SAFETY: SIMD128 is enabled for the whole build; every load
            // and store covers checked subslices.
            unsafe {
                let top = u16x8_extend_low_u8x16(v128_load64_zero(
                    above[x0..][..HALF_STEP].as_ptr().cast(),
                ));
                let wx = u16x8_extend_low_u8x16(v128_load64_zero(
                    weights[x0..][..HALF_STEP].as_ptr().cast(),
                ));
                let value = smooth8(mode, row, top, wx);
                v128_store64_lane::<0>(
                    u8x16_narrow_i16x8(value, value),
                    out[x0..][..HALF_STEP].as_mut_ptr().cast(),
                );
            }
        }
    }
}
