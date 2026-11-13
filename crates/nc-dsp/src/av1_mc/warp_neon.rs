//! Baseline NEON AV1 warp, following dav1d 1.5.4's
//! `warp_filter_horz_neon` / `warp_affine_8x8_8bpc_neon` in
//! `src/arm/64/mc.S` and `transpose_8x8b_xtl` in `util.S` (BSD-2-Clause;
//! Copyright 2018 VideoLAN/dav1d authors, Janne Grunau and Martin Storsjo;
//! see `THIRD_PARTY.md`). This is ordinary NEON, not optional dotprod.
//!
//! Signed centered bytes make each horizontal 8-tap sum fit a word:
//! abs(sum) <= 128 * 222 = 28416. Rounding by three and adding 2048
//! restores our uncentered intermediate exactly, since every filter sums
//! to 128. Eight horizontal rows feed a vertical widening multiply-add
//! across columns. Unlike dav1d's final output kernel we retain i32 sums,
//! variable round1 and no clipping: callers also consume compound values.

#![allow(unsafe_code, clippy::inline_always)]

use core::arch::aarch64::{
    int8x8_t, int16x8_t, vaddq_s16, vcombine_s8, vdup_n_s8, vdupq_n_s16, vdupq_n_s32, vdupq_n_u8,
    veorq_u8, vext_s8, vget_high_s8, vget_high_s16, vget_low_s8, vget_low_s16, vld1_s8, vld1q_u8,
    vmlal_s16, vmovl_s8, vmull_s8, vmull_s16, vpaddq_s16, vreinterpretq_s8_s32,
    vreinterpretq_s8_u8, vreinterpretq_s16_s8, vreinterpretq_s32_s16, vrshlq_s32, vrshrq_n_s16,
    vst1q_s32, vtrn1q_s16, vtrn1q_s32, vtrn2q_s16, vtrn2q_s32, vzip1q_s8,
};

use super::warp::warp_taps_i8;
use super::warp_kernels::{WINDOW_SIDE, WINDOW_STRIDE, WarpOut, WarpPhases, window_span};
use super::{FILTER_BITS, INTER_ROUND0};

/// Convert unsigned source bytes to signed bytes without changing their order.
const SAMPLE_CENTER: u8 = 128;
/// Restore the unsigned horizontal sum after its first rounding stage.
const CENTER_RESTORE: i16 = ((SAMPLE_CENTER as i32 * (1 << FILTER_BITS)) >> INTER_ROUND0) as i16;

#[inline(always)]
unsafe fn taps(phase: i32) -> int8x8_t {
    // SAFETY: the checked shared table accessor returns eight readable bytes;
    // every caller is inside the NEON wrapper.
    unsafe { vld1_s8(warp_taps_i8(phase).as_ptr()) }
}

#[inline(always)]
unsafe fn horizontal(row: &[u8], start: i32, alpha: i32) -> int16x8_t {
    let row = &row[..WINDOW_STRIDE];
    // SAFETY: exactly sixteen source bytes are readable. Register extracts
    // use only source samples 0..14, and all arithmetic runs under NEON.
    unsafe {
        let signed =
            vreinterpretq_s8_u8(veorq_u8(vld1q_u8(row.as_ptr()), vdupq_n_u8(SAMPLE_CENTER)));
        let (lo, hi) = (vget_low_s8(signed), vget_high_s8(signed));
        let p0 = vmull_s8(lo, taps(start));
        let p1 = vmull_s8(vext_s8::<1>(lo, hi), taps(start + alpha));
        let p2 = vmull_s8(vext_s8::<2>(lo, hi), taps(start + 2 * alpha));
        let p3 = vmull_s8(vext_s8::<3>(lo, hi), taps(start + 3 * alpha));
        let p4 = vmull_s8(vext_s8::<4>(lo, hi), taps(start + 4 * alpha));
        let p5 = vmull_s8(vext_s8::<5>(lo, hi), taps(start + 5 * alpha));
        let p6 = vmull_s8(vext_s8::<6>(lo, hi), taps(start + 6 * alpha));
        let p7 = vmull_s8(vext_s8::<7>(lo, hi), taps(start + 7 * alpha));
        let a = vpaddq_s16(vpaddq_s16(p0, p1), vpaddq_s16(p2, p3));
        let b = vpaddq_s16(vpaddq_s16(p4, p5), vpaddq_s16(p6, p7));
        vaddq_s16(
            vrshrq_n_s16::<{ INTER_ROUND0 as i32 }>(vpaddq_s16(a, b)),
            vdupq_n_s16(CENTER_RESTORE),
        )
    }
}

/// Transpose eight byte coefficient rows and sign-extend each resulting tap
/// column, using dav1d's zip-byte / transpose-word / transpose-dword order.
#[inline(always)]
unsafe fn vertical_taps(start: i32, gamma: i32) -> [int16x8_t; 8] {
    // SAFETY: only register operations and the checked eight-byte table loads;
    // the outer wrapper enables NEON.
    unsafe {
        let zero = vdup_n_s8(0);
        let pair = |pair01, pair23| {
            vreinterpretq_s16_s8(vzip1q_s8(
                vcombine_s8(pair01, zero),
                vcombine_s8(pair23, zero),
            ))
        };
        let pair01 = pair(taps(start), taps(start + gamma));
        let pair23 = pair(taps(start + 2 * gamma), taps(start + 3 * gamma));
        let pair45 = pair(taps(start + 4 * gamma), taps(start + 5 * gamma));
        let pair67 = pair(taps(start + 6 * gamma), taps(start + 7 * gamma));
        let even03 = vreinterpretq_s32_s16(vtrn1q_s16(pair01, pair23));
        let odd03 = vreinterpretq_s32_s16(vtrn2q_s16(pair01, pair23));
        let even47 = vreinterpretq_s32_s16(vtrn1q_s16(pair45, pair67));
        let odd47 = vreinterpretq_s32_s16(vtrn2q_s16(pair45, pair67));
        let t0 = vreinterpretq_s8_s32(vtrn1q_s32(even03, even47));
        let t2 = vreinterpretq_s8_s32(vtrn2q_s32(even03, even47));
        let t1 = vreinterpretq_s8_s32(vtrn1q_s32(odd03, odd47));
        let t3 = vreinterpretq_s8_s32(vtrn2q_s32(odd03, odd47));
        [
            vmovl_s8(vget_low_s8(t0)),
            vmovl_s8(vget_low_s8(t1)),
            vmovl_s8(vget_low_s8(t2)),
            vmovl_s8(vget_low_s8(t3)),
            vmovl_s8(vget_high_s8(t0)),
            vmovl_s8(vget_high_s8(t1)),
            vmovl_s8(vget_high_s8(t2)),
            vmovl_s8(vget_high_s8(t3)),
        ]
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn warp8x8(
    (window, stride): (&[u8], usize),
    phases: &WarpPhases,
    round1: u32,
    out: &mut WarpOut,
) {
    assert!(stride >= WINDOW_STRIDE && window.len() >= window_span(stride));
    assert!((1..=11).contains(&round1));
    // SAFETY: window extents are checked above; each horizontal call reads
    // exactly sixteen bytes, through row 14. Table accesses are checked and
    // the two output stores write exactly eight i32 samples per row.
    unsafe {
        let (sx, alpha, beta) = phases.x;
        let (sy, gamma, delta) = phases.y;
        let mut rows = [vdupq_n_s16(0); 8];
        for (i, row) in rows.iter_mut().take(7).enumerate() {
            *row = horizontal(&window[i * stride..], sx + beta * (i as i32 - 3), alpha);
        }
        let shift = vdupq_n_s32(-(round1 as i32));
        for r in 0..8 {
            let i = r + 7;
            debug_assert!(i < WINDOW_SIDE);
            rows[7] = horizontal(&window[i * stride..], sx + beta * (i as i32 - 3), alpha);
            let coeffs = vertical_taps(sy + delta * r as i32, gamma);
            let mut lo = vmull_s16(vget_low_s16(rows[0]), vget_low_s16(coeffs[0]));
            let mut hi = vmull_s16(vget_high_s16(rows[0]), vget_high_s16(coeffs[0]));
            for k in 1..8 {
                lo = vmlal_s16(lo, vget_low_s16(rows[k]), vget_low_s16(coeffs[k]));
                hi = vmlal_s16(hi, vget_high_s16(rows[k]), vget_high_s16(coeffs[k]));
            }
            vst1q_s32(out.as_mut_ptr().add(r * 8), vrshlq_s32(lo, shift));
            vst1q_s32(out.as_mut_ptr().add(r * 8 + 4), vrshlq_s32(hi, shift));
            rows[0] = rows[1];
            rows[1] = rows[2];
            rows[2] = rows[3];
            rows[3] = rows[4];
            rows[4] = rows[5];
            rows[5] = rows[6];
            rows[6] = rows[7];
        }
    }
}
