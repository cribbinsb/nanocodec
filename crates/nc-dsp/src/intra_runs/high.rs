//! Shared `u16` directional interpolation runs over [`I16Lanes`] for HEVC
//! Main 10 and VVC Main 10 (square and rectangular angular prediction).
//!
//! For all bit depths `8..=12`, samples lie in `0..=4095`, so both 2-tap
//! linear (`sum <= 32 * 4095`) and 4-tap cubic/Gaussian (`|sum| <= 68 * 4095`)
//! sums accumulate in exact 32-bit lanes (`acc_mul_add2`) before rounding and
//! narrowing back to `u16`.

#![allow(unsafe_code, clippy::inline_always)]

use crate::simd::I16Lanes;

/// 2-tap blend: `out[k] = (window[k] * (32 - w) + window[k + 1] * w + 16) >> 5`
/// for `w <= 32` and `out.len()` a multiple of `V::LANES`.
#[inline(always)]
pub(crate) unsafe fn run2_u16<V: I16Lanes>(window: &[u16], weight: usize, out: &mut [u16]) {
    let n = out.len();
    if weight == 32 {
        out.copy_from_slice(&window[1..=n]);
        return;
    }
    if weight == 0 {
        out.copy_from_slice(&window[..n]);
        return;
    }
    let w1 = weight as i16;
    let w0 = 32 - w1;
    for k in (0..n).step_by(V::LANES) {
        let acc = V::acc_mul_add2(
            V::acc_zero(),
            V::load_u16(&window[k..]),
            w0,
            V::load_u16(&window[k + 1..]),
            w1,
        );
        V::acc_round_shift_narrow(acc, 16, 5).store_u16(&mut out[k..]);
    }
}

/// 4-tap cubic/Gaussian blend:
/// `out[k] = clamp((Σ taps[i] * window[k + i] + 32) >> 6, 0, max_val)`
/// for `out.len()` a multiple of `V::LANES`.
#[inline(always)]
pub(crate) unsafe fn run4_u16<V: I16Lanes>(
    window: &[u16],
    taps: [i16; 4],
    out: &mut [u16],
    max_val: i16,
) {
    let (zero, hi) = (V::splat(0), V::splat(max_val));
    for k in (0..out.len()).step_by(V::LANES) {
        let mut acc = V::acc_mul_add2(
            V::acc_zero(),
            V::load_u16(&window[k..]),
            taps[0],
            V::load_u16(&window[k + 1..]),
            taps[1],
        );
        acc = V::acc_mul_add2(
            acc,
            V::load_u16(&window[k + 2..]),
            taps[2],
            V::load_u16(&window[k + 3..]),
            taps[3],
        );
        V::acc_round_shift_narrow(acc, 32, 6)
            .max(zero)
            .min(hi)
            .store_u16(&mut out[k..]);
    }
}
