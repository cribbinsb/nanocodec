//! SIMD128 [`super::run2`] and [`super::run4`]: 16-bit widening
//! multiply-adds (`extmul`) of the sample vectors by their weights, 16, 8
//! or 4 outputs per step (`run2` moved here from the AV1 directional
//! kernels, which it serves with HEVC and AV2; `run4` on signed 16-bit
//! lanes, exact by the module docs' bounds, clipped by the saturating
//! narrow). This module only exists in builds with SIMD128 enabled.

#![allow(clippy::inline_always)]

use core::arch::wasm32::{
    i16x8_add, i16x8_mul, i16x8_shr, i16x8_splat, u8x16_narrow_i16x8, u8x16_splat, u16x8_add,
    u16x8_extend_high_u8x16, u16x8_extend_low_u8x16, u16x8_extmul_high_u8x16,
    u16x8_extmul_low_u8x16, u16x8_shr, u16x8_splat, v128, v128_load, v128_load32_zero,
    v128_load64_zero, v128_store, v128_store32_lane, v128_store64_lane,
};

use super::{FOUR_TAP_BITS, QUARTER_STEP, Runs};
use crate::intra_prediction::directional::interpolate_run;

/// Outputs per full and per half step of a run.
const STEP: usize = 16;
const HALF_STEP: usize = 8;
/// The interpolation weights sum to `1 << WEIGHT_BITS`.
const WEIGHT_BITS: u32 = 5;
const WEIGHT_SUM: u8 = 1 << WEIGHT_BITS;

/// `Round2(a · near + b · far, 5)` for the low and high eight bytes.
#[inline(always)]
fn interpolate(a: v128, b: v128, near: v128, far: v128) -> v128 {
    let round = u16x8_splat(1 << (WEIGHT_BITS - 1));
    let sum = |low: bool| {
        let (x, y) = if low {
            (
                u16x8_extmul_low_u8x16(a, near),
                u16x8_extmul_low_u8x16(b, far),
            )
        } else {
            (
                u16x8_extmul_high_u8x16(a, near),
                u16x8_extmul_high_u8x16(b, far),
            )
        };
        u16x8_shr(u16x8_add(u16x8_add(x, y), round), WEIGHT_BITS)
    };
    u8x16_narrow_i16x8(sum(true), sum(false))
}

/// [`super::run2`] for the dispatch tables.
pub(super) unsafe fn run2_entry(window: &[u8], weight: usize, out: &mut [u8]) {
    // SAFETY: SIMD128 is enabled for the whole build.
    unsafe { run2(window, weight, out) }
}

/// [`super::run2`].
///
/// # Safety
///
/// None beyond the build's SIMD128 (every load and store indexes its
/// slice, so wrong lengths panic); `unsafe` to match the other ISAs.
#[inline(always)]
pub(crate) unsafe fn run2(window: &[u8], weight: usize, out: &mut [u8]) {
    assert_eq!(window.len(), out.len() + 1);
    let len = out.len();
    if len < QUARTER_STEP {
        interpolate_run(window, weight, out);
        return;
    }
    let far = weight as u8;
    let (near, far) = (u8x16_splat(WEIGHT_SUM - far), u8x16_splat(far));
    // SAFETY: SIMD128 is enabled for the whole build; every load and store
    // covers checked subslices.
    unsafe {
        let mut done = 0;
        while done + STEP <= len {
            let a = v128_load(window[done..][..STEP].as_ptr().cast());
            let b = v128_load(window[done + 1..][..STEP].as_ptr().cast());
            v128_store(
                out[done..][..STEP].as_mut_ptr().cast(),
                interpolate(a, b, near, far),
            );
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
            if step == HALF_STEP {
                let a = v128_load64_zero(window[at..][..HALF_STEP].as_ptr().cast());
                let b = v128_load64_zero(window[at + 1..][..HALF_STEP].as_ptr().cast());
                v128_store64_lane::<0>(
                    interpolate(a, b, near, far),
                    out[at..][..HALF_STEP].as_mut_ptr().cast(),
                );
            } else {
                let a = v128_load32_zero(window[at..][..QUARTER_STEP].as_ptr().cast());
                let b = v128_load32_zero(window[at + 1..][..QUARTER_STEP].as_ptr().cast());
                v128_store32_lane::<0>(
                    interpolate(a, b, near, far),
                    out[at..][..QUARTER_STEP].as_mut_ptr().cast(),
                );
            }
            done = at + step;
        }
    }
}

/// SIMD128 [`Runs`].
pub(crate) struct Simd128Runs;

impl Runs for Simd128Runs {
    #[inline(always)]
    unsafe fn run2(window: &[u8], weight: usize, out: &mut [u8]) {
        // SAFETY: forwarded contract.
        unsafe { run2(window, weight, out) }
    }

    #[inline(always)]
    unsafe fn run4(window: &[u8], taps: [i8; 4], out: &mut [u8]) {
        // SAFETY: forwarded contract.
        unsafe { run4(window, taps, out) }
    }
}

/// Sixteen `run4` outputs from the four tap vectors `s`: the zero-extended
/// samples times the taps in signed 16-bit lanes (exact: every partial sum
/// fits, module docs), rounded, shifted and narrowed with saturation.
#[inline(always)]
fn sum16(s: [v128; 4], taps: [i8; 4]) -> v128 {
    let round = i16x8_splat(1 << (FOUR_TAP_BITS - 1));
    let half = |low: bool| {
        let mut sum = round;
        for (&samples, &tap) in s.iter().zip(&taps) {
            let widened = if low {
                u16x8_extend_low_u8x16(samples)
            } else {
                u16x8_extend_high_u8x16(samples)
            };
            sum = i16x8_add(sum, i16x8_mul(widened, i16x8_splat(i16::from(tap))));
        }
        i16x8_shr(sum, FOUR_TAP_BITS)
    };
    u8x16_narrow_i16x8(half(true), half(false))
}

/// [`super::run4`] (module docs).
///
/// # Safety
///
/// None beyond the build's SIMD128 (every load and store indexes its
/// slice, so wrong lengths panic); `unsafe` to match the other ISAs.
#[inline(always)]
pub(crate) unsafe fn run4(window: &[u8], taps: [i8; 4], out: &mut [u8]) {
    assert_eq!(window.len(), out.len() + 3);
    let len = out.len();
    if len < QUARTER_STEP {
        super::run4(window, taps, out);
        return;
    }
    // SAFETY: SIMD128 is enabled for the whole build; every load and store
    // covers checked subslices.
    unsafe {
        let mut done = 0;
        while done + STEP <= len {
            let w = &window[done..][..STEP + 3];
            let s = [
                v128_load(w[..STEP].as_ptr().cast()),
                v128_load(w[1..][..STEP].as_ptr().cast()),
                v128_load(w[2..][..STEP].as_ptr().cast()),
                v128_load(w[3..][..STEP].as_ptr().cast()),
            ];
            v128_store(out[done..][..STEP].as_mut_ptr().cast(), sum16(s, taps));
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
            if step == HALF_STEP {
                let s = [
                    v128_load64_zero(w[..HALF_STEP].as_ptr().cast()),
                    v128_load64_zero(w[1..][..HALF_STEP].as_ptr().cast()),
                    v128_load64_zero(w[2..][..HALF_STEP].as_ptr().cast()),
                    v128_load64_zero(w[3..][..HALF_STEP].as_ptr().cast()),
                ];
                v128_store64_lane::<0>(sum16(s, taps), out[at..][..HALF_STEP].as_mut_ptr().cast());
            } else {
                let s = [
                    v128_load32_zero(w[..QUARTER_STEP].as_ptr().cast()),
                    v128_load32_zero(w[1..][..QUARTER_STEP].as_ptr().cast()),
                    v128_load32_zero(w[2..][..QUARTER_STEP].as_ptr().cast()),
                    v128_load32_zero(w[3..][..QUARTER_STEP].as_ptr().cast()),
                ];
                v128_store32_lane::<0>(
                    sum16(s, taps),
                    out[at..][..QUARTER_STEP].as_mut_ptr().cast(),
                );
            }
            done = at + step;
        }
    }
}
