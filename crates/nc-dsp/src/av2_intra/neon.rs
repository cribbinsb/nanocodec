//! NEON AV2 intra kernels. Smooth is the generic code on 8 lanes (4 for
//! 4×4 blocks). The interpolation runs are byte kernels, 16 outputs per
//! step: every 2-tap product sum fits u16 (`255 · 32`), and the 4-tap
//! filter's taps have fixed signs (`−, +, +, −`, `FOUR_TAP`), so its
//! positive and negative halves are separate u16 sums (`255 · 152` at
//! most); their saturating difference clips at zero and the saturating
//! rounding narrow clips at 255, which is `Clip1(Round2(sum, 7))` exactly.
//! The generic 32-bit-accumulator runs measured 0.3–0.9× of the
//! auto-vectorised scalar runs on the Jetson.

#![allow(clippy::inline_always)]

use core::arch::aarch64::{
    uint8x16_t, vdupq_n_u8, vget_low_u8, vld1q_u8, vmlal_high_u8, vmlal_u8, vmull_high_u8,
    vmull_u8, vqrshrn_high_n_u16, vqrshrn_n_u16, vqsubq_u16, vrshrn_high_n_u16, vrshrn_n_u16,
    vst1q_u8,
};

use crate::av2_intra::Av2IntraFns;
use crate::intra_prediction::av2::{self as reference, FOUR_TAP, TWO_TAP_BITS};
use crate::simd::neon::{I16x4, I16x8};

/// Outputs per step.
const STEP: usize = 16;
/// The 2-tap weights sum to this.
const TWO_TAP_SUM: u8 = 1 << TWO_TAP_BITS;

/// This ISA's AV2 intra table.
pub(crate) fn av2_intra_fns() -> Av2IntraFns {
    Av2IntraFns { run4, run2, smooth }
}

/// Sixteen samples from `samples[0..16]`.
#[inline(always)]
fn load(samples: &[u8]) -> uint8x16_t {
    let samples = &samples[..STEP];
    // SAFETY: NEON is enabled (callers' target feature); 16 checked bytes.
    unsafe { vld1q_u8(samples.as_ptr()) }
}

#[target_feature(enable = "neon")]
unsafe fn run2(window: &[u8], shift: usize, out: &mut [u8]) {
    let second = shift as u8;
    let (a_weight, b_weight) = (vdupq_n_u8(TWO_TAP_SUM - second), vdupq_n_u8(second));
    let mut done = 0;
    while done + STEP <= out.len() {
        let a = load(&window[done..]);
        let b = load(&window[done + 1..]);
        let low = vmlal_u8(
            vmull_u8(vget_low_u8(a), vget_low_u8(a_weight)),
            vget_low_u8(b),
            vget_low_u8(b_weight),
        );
        let high = vmlal_high_u8(vmull_high_u8(a, a_weight), b, b_weight);
        let packed = vrshrn_high_n_u16::<5>(vrshrn_n_u16::<5>(low), high);
        let dst = &mut out[done..done + STEP];
        // SAFETY: 16 checked bytes.
        unsafe { vst1q_u8(dst.as_mut_ptr(), packed) };
        done += STEP;
    }
    reference::run2(&window[done..], shift, &mut out[done..]);
}

#[target_feature(enable = "neon")]
unsafe fn run4(window: &[u8], shift: usize, out: &mut [u8]) {
    let taps = FOUR_TAP[shift];
    debug_assert!(taps[0] <= 0 && taps[1] >= 0 && taps[2] >= 0 && taps[3] <= 0);
    let magnitude = |tap: i32| vdupq_n_u8(tap.unsigned_abs() as u8);
    let (outer0, inner1, inner2, outer3) = (
        magnitude(taps[0]),
        magnitude(taps[1]),
        magnitude(taps[2]),
        magnitude(taps[3]),
    );
    let mut done = 0;
    while done + STEP <= out.len() {
        let s = [0, 1, 2, 3].map(|k| load(&window[done + k..]));
        let positive_low = vmlal_u8(
            vmull_u8(vget_low_u8(s[1]), vget_low_u8(inner1)),
            vget_low_u8(s[2]),
            vget_low_u8(inner2),
        );
        let negative_low = vmlal_u8(
            vmull_u8(vget_low_u8(s[0]), vget_low_u8(outer0)),
            vget_low_u8(s[3]),
            vget_low_u8(outer3),
        );
        let positive_high = vmlal_high_u8(vmull_high_u8(s[1], inner1), s[2], inner2);
        let negative_high = vmlal_high_u8(vmull_high_u8(s[0], outer0), s[3], outer3);
        let packed = vqrshrn_high_n_u16::<7>(
            vqrshrn_n_u16::<7>(vqsubq_u16(positive_low, negative_low)),
            vqsubq_u16(positive_high, negative_high),
        );
        let dst = &mut out[done..done + STEP];
        // SAFETY: 16 checked bytes.
        unsafe { vst1q_u8(dst.as_mut_ptr(), packed) };
        done += STEP;
    }
    reference::run4(&window[done..], shift, &mut out[done..]);
}

super::generic::av2_smooth_kernel!(#[target_feature(enable = "neon")]; I16x4, I16x4, I16x8);
