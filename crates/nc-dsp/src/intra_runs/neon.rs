//! NEON [`super::run2`] and [`super::run4`]: byte kernels (`umull`/`umlal`
//! by the weights, a rounding narrow), 16, 8 or 4 outputs per step (`run2`
//! moved here from the AV1 directional kernels, which it serves with HEVC
//! and AV2). `run4`'s taps have fixed signs per row (inner two
//! non-negative, outer two of one sign), so it sums tap magnitudes in u16:
//! the inner pair, plus or (saturating) minus the outer pair; the
//! saturating rounding narrow then clips at 255, which is
//! `Clip1((sum + 32) >> 6)` exactly (the same technique as AV2's runs).

#![allow(clippy::inline_always)]

use core::arch::aarch64::{
    uint8x8_t, uint8x16_t, uint16x8_t, vaddq_u16, vdup_n_u8, vdup_n_u32, vdupq_n_u8, vget_lane_u32,
    vget_low_u8, vld1_u8, vld1q_u8, vmlal_high_u8, vmlal_u8, vmull_high_u8, vmull_u8,
    vqrshrn_high_n_u16, vqrshrn_n_u16, vqsubq_u16, vreinterpret_u8_u32, vreinterpret_u32_u8,
    vrshrn_high_n_u16, vrshrn_n_u16, vst1_u8, vst1q_u8,
};

use super::{FOUR_TAP_BITS, QUARTER_STEP, Runs};
use crate::intra_prediction::directional::interpolate_run;

/// Outputs per full and per half step of a run.
const STEP: usize = 16;
const HALF_STEP: usize = 8;
/// The interpolation weights sum to this.
const WEIGHT_SUM: u8 = 32;

/// [`super::run2`] for the dispatch tables.
#[target_feature(enable = "neon")]
pub(super) unsafe fn run2_entry(window: &[u8], weight: usize, out: &mut [u8]) {
    // SAFETY: NEON is enabled for this function.
    unsafe { run2(window, weight, out) }
}

/// 16 outputs from `window[..17]` at weight `far` (one 128-bit step).
#[inline(always)]
fn step16(window: &[u8], far: u8, out: &mut [u8]) {
    let (window, out) = (&window[..=STEP], &mut out[..STEP]);
    // SAFETY: NEON is enabled (callers' target feature); the loads and the
    // store cover the checked subslices.
    unsafe {
        let (near_q, far_q) = (vdupq_n_u8(WEIGHT_SUM - far), vdupq_n_u8(far));
        let a = vld1q_u8(window.as_ptr());
        let b = vld1q_u8(window[1..].as_ptr());
        let low = vmlal_u8(
            vmull_u8(vget_low_u8(a), vget_low_u8(near_q)),
            vget_low_u8(b),
            vget_low_u8(far_q),
        );
        let high = vmlal_high_u8(vmull_high_u8(a, near_q), b, far_q);
        vst1q_u8(
            out.as_mut_ptr(),
            vrshrn_high_n_u16::<5>(vrshrn_n_u16::<5>(low), high),
        );
    }
}

/// 8 outputs from `window[..9]` (a 64-bit step).
#[inline(always)]
fn step8(window: &[u8], far: u8, out: &mut [u8]) {
    let (window, out) = (&window[..=HALF_STEP], &mut out[..HALF_STEP]);
    // SAFETY: as for `step16`.
    unsafe {
        let (near_d, far_d) = (vdup_n_u8(WEIGHT_SUM - far), vdup_n_u8(far));
        let a = vld1_u8(window.as_ptr());
        let b = vld1_u8(window[1..].as_ptr());
        let sum = vrshrn_n_u16::<5>(vmlal_u8(vmull_u8(a, near_d), b, far_d));
        vst1_u8(out.as_mut_ptr(), sum);
    }
}

/// 4 outputs from `window[..5]` (a 32-bit step).
#[inline(always)]
fn step4(window: &[u8], far: u8, out: &mut [u8]) {
    let (window, out) = (&window[..=QUARTER_STEP], &mut out[..QUARTER_STEP]);
    // SAFETY: as for `step16`.
    unsafe {
        let (near_d, far_d) = (vdup_n_u8(WEIGHT_SUM - far), vdup_n_u8(far));
        let (a, b) = (quarter(window), quarter(&window[1..]));
        let sum = vrshrn_n_u16::<5>(vmlal_u8(vmull_u8(a, near_d), b, far_d));
        let lane = vget_lane_u32::<0>(vreinterpret_u32_u8(sum));
        out.copy_from_slice(&lane.to_le_bytes());
    }
}

/// [`super::run2`], inlined into the caller's NEON code.
///
/// # Safety
///
/// NEON must be enabled in the caller. Every load and store indexes its
/// slice, so wrong lengths panic.
#[inline(always)]
pub(crate) unsafe fn run2(window: &[u8], weight: usize, out: &mut [u8]) {
    assert_eq!(window.len(), out.len() + 1);
    let len = out.len();
    if len < QUARTER_STEP {
        interpolate_run(window, weight, out);
        return;
    }
    let far = weight as u8;
    let mut done = 0;
    while done + STEP <= len {
        step16(&window[done..], far, &mut out[done..]);
        done += STEP;
    }
    // The rest in half (then quarter) steps, the last one ending at `len`
    // (it may recompute outputs already stored, with the same values).
    while done < len {
        if len >= HALF_STEP {
            let at = done.min(len - HALF_STEP);
            step8(&window[at..], far, &mut out[at..]);
            done = at + HALF_STEP;
        } else {
            let at = done.min(len - QUARTER_STEP);
            step4(&window[at..], far, &mut out[at..]);
            done = at + QUARTER_STEP;
        }
    }
}

/// [`super::run2`] of a run of `N` outputs (4, 8, or a multiple of 16),
/// with no length dispatch.
///
/// # Safety
///
/// NEON must be enabled in the caller; `window` holds `N + 1` samples and
/// `out` `N`.
#[inline(always)]
pub(crate) unsafe fn run2_fixed<const N: usize>(window: &[u8], weight: usize, out: &mut [u8]) {
    let far = weight as u8;
    match N {
        QUARTER_STEP => step4(window, far, out),
        HALF_STEP => step8(window, far, out),
        _ => {
            for at in (0..N).step_by(STEP) {
                step16(&window[at..], far, &mut out[at..]);
            }
        }
    }
}

/// The first four bytes of `samples` in the low lanes (the rest repeat
/// them; only the low four outputs are stored).
#[inline(always)]
fn quarter(samples: &[u8]) -> uint8x8_t {
    let lane = u32::from_le_bytes([samples[0], samples[1], samples[2], samples[3]]);
    // SAFETY: NEON is enabled (callers' target feature); register only.
    unsafe { vreinterpret_u8_u32(vdup_n_u32(lane)) }
}

/// NEON [`Runs`].
pub(crate) struct NeonRuns;

impl Runs for NeonRuns {
    #[inline(always)]
    unsafe fn run2(window: &[u8], weight: usize, out: &mut [u8]) {
        // SAFETY: forwarded contract.
        unsafe { run2(window, weight, out) }
    }

    #[inline(always)]
    unsafe fn run2_fixed<const N: usize>(window: &[u8], weight: usize, out: &mut [u8]) {
        // SAFETY: forwarded contract.
        unsafe { run2_fixed::<N>(window, weight, out) }
    }

    #[inline(always)]
    unsafe fn run4(window: &[u8], taps: [i8; 4], out: &mut [u8]) {
        // SAFETY: forwarded contract.
        unsafe { run4(window, taps, out) }
    }
}

/// [`super::run4`], inlined into the caller's NEON code (module docs).
///
/// # Safety
///
/// NEON must be enabled in the caller. Every load and store indexes its
/// slice, so wrong lengths panic.
#[inline(always)]
pub(crate) unsafe fn run4(window: &[u8], taps: [i8; 4], out: &mut [u8]) {
    assert_eq!(window.len(), out.len() + 3);
    if out.len() < QUARTER_STEP {
        super::run4(window, taps, out);
        return;
    }
    debug_assert!(taps[1] >= 0 && taps[2] >= 0);
    let magnitudes = taps.map(i8::unsigned_abs);
    // SAFETY: forwarded contract.
    unsafe {
        if taps[0] < 0 || taps[3] < 0 {
            debug_assert!(taps[0] <= 0 && taps[3] <= 0);
            run4_signed::<true>(window, magnitudes, out);
        } else {
            run4_signed::<false>(window, magnitudes, out);
        }
    }
}

/// The inner pair's products plus (or saturating minus, with
/// `OUTER_NEGATIVE`) the outer pair's, for eight lanes.
#[inline(always)]
fn combine<const OUTER_NEGATIVE: bool>(inner: uint16x8_t, outer: uint16x8_t) -> uint16x8_t {
    // SAFETY: NEON is enabled (callers' target feature); register only.
    unsafe {
        if OUTER_NEGATIVE {
            vqsubq_u16(inner, outer)
        } else {
            vaddq_u16(inner, outer)
        }
    }
}

/// Eight `run4` sums from the four tap vectors `s` and tap magnitudes `t`.
#[inline(always)]
fn sum8<const OUTER_NEGATIVE: bool>(s: [uint8x8_t; 4], t: [uint8x8_t; 4]) -> uint16x8_t {
    // SAFETY: NEON is enabled (callers' target feature); register only.
    unsafe {
        let inner = vmlal_u8(vmull_u8(s[1], t[1]), s[2], t[2]);
        let outer = vmlal_u8(vmull_u8(s[0], t[0]), s[3], t[3]);
        combine::<OUTER_NEGATIVE>(inner, outer)
    }
}

/// [`run4`] with the outer taps' sign fixed.
///
/// # Safety
///
/// As [`run4`]; `out.len() >= QUARTER_STEP`.
#[inline(always)]
unsafe fn run4_signed<const OUTER_NEGATIVE: bool>(
    window: &[u8],
    magnitudes: [u8; 4],
    out: &mut [u8],
) {
    let len = out.len();
    // SAFETY: NEON is enabled (caller contract); every load and store
    // covers checked subslices.
    unsafe {
        let t_q: [uint8x16_t; 4] = magnitudes.map(|m| vdupq_n_u8(m));
        let t_d: [uint8x8_t; 4] = magnitudes.map(|m| vdup_n_u8(m));
        let mut done = 0;
        while done + STEP <= len {
            let w = &window[done..][..STEP + 3];
            let s = [
                vld1q_u8(w[..STEP].as_ptr()),
                vld1q_u8(w[1..][..STEP].as_ptr()),
                vld1q_u8(w[2..][..STEP].as_ptr()),
                vld1q_u8(w[3..][..STEP].as_ptr()),
            ];
            let inner_high = vmlal_high_u8(vmull_high_u8(s[1], t_q[1]), s[2], t_q[2]);
            let outer_high = vmlal_high_u8(vmull_high_u8(s[0], t_q[0]), s[3], t_q[3]);
            let low = sum8::<OUTER_NEGATIVE>(s.map(|v| vget_low_u8(v)), t_d);
            let high = combine::<OUTER_NEGATIVE>(inner_high, outer_high);
            let packed = vqrshrn_high_n_u16::<{ FOUR_TAP_BITS as i32 }>(
                vqrshrn_n_u16::<{ FOUR_TAP_BITS as i32 }>(low),
                high,
            );
            vst1q_u8(out[done..][..STEP].as_mut_ptr(), packed);
            done += STEP;
        }
        // The rest in half (then quarter) steps, the last one ending at
        // `len` (it may recompute outputs already stored, with the same
        // values).
        let step = if len < HALF_STEP {
            QUARTER_STEP
        } else {
            HALF_STEP
        };
        while done < len {
            let at = done.min(len - step);
            let w = &window[at..][..step + 3];
            let s = if step == HALF_STEP {
                [
                    vld1_u8(w[..HALF_STEP].as_ptr()),
                    vld1_u8(w[1..][..HALF_STEP].as_ptr()),
                    vld1_u8(w[2..][..HALF_STEP].as_ptr()),
                    vld1_u8(w[3..][..HALF_STEP].as_ptr()),
                ]
            } else {
                [
                    quarter(w),
                    quarter(&w[1..]),
                    quarter(&w[2..]),
                    quarter(&w[3..]),
                ]
            };
            let sum = vqrshrn_n_u16::<{ FOUR_TAP_BITS as i32 }>(sum8::<OUTER_NEGATIVE>(s, t_d));
            if step == HALF_STEP {
                vst1_u8(out[at..][..HALF_STEP].as_mut_ptr(), sum);
            } else {
                let lane = vget_lane_u32::<0>(vreinterpret_u32_u8(sum));
                out[at..at + QUARTER_STEP].copy_from_slice(&lane.to_le_bytes());
            }
            done = at + step;
        }
    }
}
