//! Four-row nonlinear luma ALF using the coefficient lane scheduling of
//! Clear BSD VVdeC `AdaptiveLoopFilter_neon.cpp`. See `THIRD_PARTY.md`.

use core::arch::aarch64::*;

use nc_common::BitDepth;

use crate::simd::neon::{I16x8, I16x8Kernel};
use crate::simd::{I16Lanes, SimdSample};

use super::{VVC_ALF_BLOCK, VVC_ALF_REACH, VVC_ALF_TAPS, load_luma_tap_ptr};

type Coefficients = [[int16x4_t; 2]; VVC_ALF_TAPS / 4];
type ClipLimits = [[(int16x8_t, int16x8_t); 4]; VVC_ALF_TAPS / 4];

#[inline(always)]
#[allow(clippy::inline_always)]
fn prepare_limits(limits: Coefficients) -> ClipLimits {
    // SAFETY: only called inside the NEON filter.
    unsafe {
        limits.map(|[left, right]| {
            macro_rules! lane {
                ($index:literal) => {{
                    let limit = vcombine_s16(
                        vdup_lane_s16::<$index>(left),
                        vdup_lane_s16::<$index>(right),
                    );
                    (limit, vnegq_s16(limit))
                }};
            }
            [lane!(0), lane!(1), lane!(2), lane!(3)]
        })
    }
}

#[inline(always)]
#[allow(clippy::inline_always)]
unsafe fn filter_row<S: SimdSample>(
    sources: &[*const S; 7],
    x: usize,
    coeffs: &Coefficients,
    clip_limits: &ClipLimits,
    shift: u32,
    max_value: I16x8,
    output: *mut S,
) {
    // SAFETY: the caller ensures the halo and output cover eight samples.
    unsafe {
        let centre = S::load_lanes::<I16x8>(sources[3].add(x + VVC_ALF_REACH)).0;
        let clipped_sum = |tap, (hi, lo)| {
            let (left, right) = load_luma_tap_ptr::<S, I16x8>(sources, x, tap);
            let left = vmaxq_s16(vminq_s16(vsubq_s16(left.0, centre), hi), lo);
            let right = vmaxq_s16(vminq_s16(vsubq_s16(right.0, centre), hi), lo);
            vaddq_s16(left, right)
        };
        let first = clipped_sum(0, clip_limits[0][0]);
        let mut lo = vmull_lane_s16::<0>(vget_low_s16(first), coeffs[0][0]);
        let mut hi = vmull_high_lane_s16::<0>(first, coeffs[0][1]);
        macro_rules! accumulate {
            ($group:literal, $lane:literal) => {{
                let sum = clipped_sum($group * 4 + $lane, clip_limits[$group][$lane]);
                lo = vmlal_lane_s16::<$lane>(lo, vget_low_s16(sum), coeffs[$group][0]);
                hi = vmlal_high_lane_s16::<$lane>(hi, sum, coeffs[$group][1]);
            }};
        }
        accumulate!(0, 1);
        accumulate!(0, 2);
        accumulate!(0, 3);
        accumulate!(1, 0);
        accumulate!(1, 1);
        accumulate!(1, 2);
        accumulate!(1, 3);
        accumulate!(2, 0);
        accumulate!(2, 1);
        accumulate!(2, 2);
        accumulate!(2, 3);
        let delta = if shift == 7 {
            vrshrn_high_n_s32::<7>(vrshrn_n_s32::<7>(lo), hi)
        } else {
            vrshrn_high_n_s32::<10>(vrshrn_n_s32::<10>(lo), hi)
        };
        // For -128..127 coefficients and samples up to 10 bits, the rounded
        // correction fits an i16. Saturation only saves the final output clip.
        S::store_clamped(
            I16x8Kernel(vqaddq_s16(centre, delta)),
            I16x8Kernel(vdupq_n_s16(0)),
            max_value,
            output.add(x),
        );
    }
}

#[inline(always)]
#[allow(clippy::inline_always)]
pub(super) unsafe fn filter_vectors<S: SimdSample>(
    sources: &[[*const S; 7]; VVC_ALF_BLOCK],
    outputs: &[*mut S; VVC_ALF_BLOCK],
    coefficients: &[[i16; VVC_ALF_TAPS]],
    limits: &[[i16; VVC_ALF_TAPS]],
    shifts: &[u32; VVC_ALF_BLOCK],
    depth: BitDepth,
    start: usize,
    end: usize,
) {
    let max_value = I16x8::splat(depth.max() as i16);
    let blocks = start / VVC_ALF_BLOCK..end / VVC_ALF_BLOCK;
    let coefficients = coefficients[blocks.clone()].chunks_exact(2);
    let limits = limits[blocks].chunks_exact(2);
    for (step, (coefficients, limits)) in coefficients.zip(limits).enumerate() {
        let x = start + 8 * step;
        let pair = |rows: &[[i16; VVC_ALF_TAPS]]| {
            // SAFETY: each group has two blocks of twelve words.
            let [left, right] = [rows[0].as_ptr(), rows[1].as_ptr()];
            unsafe {
                let a = vld1q_s16(left);
                let b = vld1q_s16(right);
                [
                    [vget_low_s16(a), vget_low_s16(b)],
                    [vget_high_s16(a), vget_high_s16(b)],
                    [vld1_s16(left.add(8)), vld1_s16(right.add(8))],
                ]
            }
        };
        let coeffs = pair(coefficients);
        let clip_limits = prepare_limits(pair(limits));
        for row in 0..VVC_ALF_BLOCK {
            // SAFETY: inherited from the validated block-row wrapper.
            unsafe {
                filter_row(
                    &sources[row],
                    x,
                    &coeffs,
                    &clip_limits,
                    shifts[row],
                    max_value,
                    outputs[row],
                );
            }
        }
    }
}
