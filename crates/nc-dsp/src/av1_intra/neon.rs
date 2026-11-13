//! NEON AV1 intra kernels: filter intra is the generic code on 8 lanes;
//! directional prediction runs its interpolation as byte kernels (16 or 8
//! outputs per step; every `a · (32 − w) + b · w` fits u16) and transposes
//! its column tiles 8x8 in registers. Smooth blends in unsigned 16-bit
//! lanes (`smooth`); Paeth is the generic code.

#![allow(clippy::inline_always)]

use core::arch::aarch64::{
    uint8x8_t, vcombine_u8, vdup_n_u8, vdupq_n_u8, vget_high_u8, vget_low_u8, vhaddq_u16, vld1_u8,
    vld1q_u8, vmlal_high_u8, vmlal_u8, vmull_high_u8, vmull_u8, vrshrn_high_n_u16, vrshrn_n_u16,
    vst1_u8, vst1q_u8, vsub_u8,
};

use nc_common::PlaneViewMut;

use crate::intra_prediction::directional::{self, DirectionalKernels};
use crate::intra_prediction::{IntraEdges, IntraMode, smooth_weights};

super::generic::av1_intra_kernels!(#[target_feature(enable = "neon")];
    crate::simd::neon::I16x8,
    crate::simd::neon::I16x8
);

/// Outputs per full and per half step of a run.
const STEP: usize = 16;
const HALF_STEP: usize = 8;
/// The interpolation weights sum to this.
const WEIGHT_SUM: u8 = 32;

/// NEON directional kernels.
struct NeonDirectional;

impl DirectionalKernels for NeonDirectional {
    #[inline(always)]
    fn run(window: &[u8], weight: usize, out: &mut [u8]) {
        assert_eq!(window.len(), out.len() + 1);
        let len = out.len();
        if len < HALF_STEP {
            directional::interpolate_run(window, weight, out);
            return;
        }
        let far = weight as u8;
        // SAFETY: NEON is enabled (the caller's target feature); every load
        // and store covers checked subslices.
        unsafe {
            let (near_q, far_q) = (vdupq_n_u8(WEIGHT_SUM - far), vdupq_n_u8(far));
            let mut done = 0;
            while done + STEP <= len {
                let a = vld1q_u8(window[done..][..STEP].as_ptr());
                let b = vld1q_u8(window[done + 1..][..STEP].as_ptr());
                let low = vmlal_u8(
                    vmull_u8(vget_low_u8(a), vget_low_u8(near_q)),
                    vget_low_u8(b),
                    vget_low_u8(far_q),
                );
                let high = vmlal_high_u8(vmull_high_u8(a, near_q), b, far_q);
                let packed = vrshrn_high_n_u16::<5>(vrshrn_n_u16::<5>(low), high);
                vst1q_u8(out[done..][..STEP].as_mut_ptr(), packed);
                done += STEP;
            }
            // The rest in half steps, the last one ending at `len` (it may
            // recompute outputs already stored, with the same values).
            let (near_d, far_d) = (vdup_n_u8(WEIGHT_SUM - far), vdup_n_u8(far));
            while done < len {
                let at = done.min(len - HALF_STEP);
                let a = vld1_u8(window[at..][..HALF_STEP].as_ptr());
                let b = vld1_u8(window[at + 1..][..HALF_STEP].as_ptr());
                let sum = vmlal_u8(vmull_u8(a, near_d), b, far_d);
                vst1_u8(out[at..][..HALF_STEP].as_mut_ptr(), vrshrn_n_u16::<5>(sum));
                done = at + HALF_STEP;
            }
        }
    }

    #[inline(always)]
    fn store_transposed(columns: &[u8], side: usize, dst: &mut PlaneViewMut<'_>) {
        // SAFETY: NEON is enabled (the caller's target feature).
        unsafe {
            super::generic::store_transposed::<crate::lpf_bytes::neon::B>(columns, side, dst);
        }
    }
}

/// `IntraEdges::predict_angle` with the NEON kernels.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn predict_angle(edges: &IntraEdges, angle: u16, dst: &mut PlaneViewMut<'_>) {
    edges.predict_angle_with::<NeonDirectional>(angle, dst);
}

/// One row's constants of a smooth prediction: the row weight `wy` and
/// `256 − wy`, and the left, bottom-left and top-right samples.
#[derive(Clone, Copy)]
struct SmoothRow {
    wy: uint8x8_t,
    wy_rest: uint8x8_t,
    left: uint8x8_t,
    bottom: uint8x8_t,
    right: uint8x8_t,
}

/// Eight smooth outputs from their `top` samples and column weights `wx`.
/// `v = wy · top + (256 − wy) · bottom` and `h = wx · left + (256 − wx) ·
/// right` are each at most `255 · 256`, so they fit u16; smooth's
/// `(v + h + 256) >> 9` is `(⌊(v + h) / 2⌋ + 128) >> 8`, a halving add
/// and a rounding narrow.
#[inline(always)]
fn smooth8(mode: IntraMode, row: SmoothRow, top: uint8x8_t, wx: uint8x8_t) -> uint8x8_t {
    // SAFETY: NEON is enabled (the caller's target feature); register
    // operations only.
    unsafe {
        // No closures: they would not inherit the caller's target feature.
        let vertical = vmlal_u8(vmull_u8(top, row.wy), row.bottom, row.wy_rest);
        let horizontal = vmlal_u8(vmull_u8(row.left, wx), row.right, vsub_u8(vdup_n_u8(0), wx));
        match mode {
            IntraMode::Smooth => vrshrn_n_u16::<8>(vhaddq_u16(vertical, horizontal)),
            IntraMode::SmoothVertical => vrshrn_n_u16::<8>(vertical),
            _ => vrshrn_n_u16::<8>(horizontal),
        }
    }
}

/// AV1 smooth prediction (§7.11.2.6) of `mode`, one of the three smooth
/// modes, on a side of 8 or more: 16 outputs per step (8 on 8x8).
#[target_feature(enable = "neon")]
unsafe fn smooth(edges: &IntraEdges, mode: IntraMode, dst: &mut PlaneViewMut<'_>) {
    let n = edges.size();
    assert!(n >= HALF_STEP && dst.width() == n && dst.height() == n);
    let (above, left) = (edges.above_row(), edges.left_column());
    let weights = smooth_weights(n);
    // SAFETY: NEON is enabled; every load and store covers checked
    // subslices.
    unsafe {
        let (bottom, right) = (vdup_n_u8(left[n - 1]), vdup_n_u8(above[n - 1]));
        for (y, (&wy, &l)) in weights.iter().zip(left).enumerate() {
            let row = SmoothRow {
                wy: vdup_n_u8(wy),
                wy_rest: vdup_n_u8(wy.wrapping_neg()),
                left: vdup_n_u8(l),
                bottom,
                right,
            };
            let out = &mut dst.row_mut(y)[..n];
            if n == HALF_STEP {
                let top = vld1_u8(above[..HALF_STEP].as_ptr());
                let value = smooth8(mode, row, top, vld1_u8(weights[..HALF_STEP].as_ptr()));
                vst1_u8(out[..HALF_STEP].as_mut_ptr(), value);
                continue;
            }
            for x0 in (0..n).step_by(STEP) {
                let top = vld1q_u8(above[x0..][..STEP].as_ptr());
                let wx = vld1q_u8(weights[x0..][..STEP].as_ptr());
                let low = smooth8(mode, row, vget_low_u8(top), vget_low_u8(wx));
                let high = smooth8(mode, row, vget_high_u8(top), vget_high_u8(wx));
                vst1q_u8(out[x0..][..STEP].as_mut_ptr(), vcombine_u8(low, high));
            }
        }
    }
}
