//! Four-output-row vertical 8-tap groups, following libvpx's
//! `highbd_vpx_convolve8_sve2.c` (BSD-3-Clause, see `THIRD_PARTY.md`).

#![allow(
    clippy::inline_always,
    reason = "inline the rolling packed row helpers"
)]

use core::arch::aarch64::{
    int16x8_t, uint16x4_t, vcombine_s16, vdup_n_s16, vdup_n_u16, vdupq_n_s64, vld1_s16, vld1_u16,
    vld1q_s16, vmin_u16, vqrshrun_n_s32, vreinterpretq_s16_s32, vreinterpretq_s32_s16,
    vreinterpretq_s32_s64, vrhadd_u16, vst1_u16, vuzp1q_s32, vzip1q_s16, vzip1q_s32, vzip2q_s32,
};

use nc_common::BitDepth;

use crate::simd::sve2::dot_halfwords;

#[inline(always)]
unsafe fn transpose([a, b, c, d]: [int16x8_t; 4]) -> [int16x8_t; 2] {
    // SAFETY: the caller enables NEON. Each output lane group holds one
    // column of the four input rows.
    unsafe {
        let ab = vreinterpretq_s32_s16(vzip1q_s16(a, b));
        let cd = vreinterpretq_s32_s16(vzip1q_s16(c, d));
        [
            vreinterpretq_s16_s32(vzip1q_s32(ab, cd)),
            vreinterpretq_s16_s32(vzip2q_s32(ab, cd)),
        ]
    }
}

#[inline(always)]
unsafe fn load(src: *const u16) -> int16x8_t {
    // SAFETY: the vertical driver provides four readable words.
    unsafe { vcombine_s16(vld1_s16(src.cast()), vdup_n_s16(0)) }
}

#[inline(always)]
unsafe fn filtered(
    first: [int16x8_t; 2],
    second: [int16x8_t; 2],
    taps: int16x8_t,
    max: uint16x4_t,
) -> uint16x4_t {
    // SAFETY: the SVE2 entry checked the ISA, layout and sample range.
    unsafe {
        let zero = vdupq_n_s64(0);
        let left = dot_halfwords::<1>(dot_halfwords::<0>(zero, first[0], taps), second[0], taps);
        let right = dot_halfwords::<1>(dot_halfwords::<0>(zero, first[1], taps), second[1], taps);
        let sums = vuzp1q_s32(vreinterpretq_s32_s64(left), vreinterpretq_s32_s64(right));
        vmin_u16(vqrshrun_n_s32::<7>(sums), max)
    }
}

/// Full eight-tap vertical filtering, in four-column strips.
///
/// # Safety
/// SVE2 is available, h is a multiple of four, and the buffers have
/// respectively h+7 and h rows of W samples. Samples are at most 12 bits.
#[inline(never)]
#[target_feature(enable = "neon")]
pub(super) unsafe fn vertical<const W: usize, const AVG: bool>(
    (src, ss): (*const u16, usize),
    (dst, ds): (*mut u16, usize),
    h: usize,
    taps: &[i16; 8],
    depth: BitDepth,
) {
    // SAFETY: the caller checked both footprints. Each iteration reads four
    // new rows, so the last load is on row h+6.
    unsafe {
        let taps = vld1q_s16(taps.as_ptr());
        let max = vdup_n_u16(depth.max() as u16);
        for x in (0..W).step_by(4) {
            let start = src.add(x);
            let row0 = load(start);
            let row1 = load(start.add(ss));
            let row2 = load(start.add(2 * ss));
            let row3 = load(start.add(3 * ss));
            let mut row4 = load(start.add(4 * ss));
            let mut row5 = load(start.add(5 * ss));
            let mut row6 = load(start.add(6 * ss));
            let mut packed0 = transpose([row0, row1, row2, row3]);
            let mut packed1 = transpose([row1, row2, row3, row4]);
            let mut packed2 = transpose([row2, row3, row4, row5]);
            let mut packed3 = transpose([row3, row4, row5, row6]);
            for y in (0..h).step_by(4) {
                let next = start.add((y + 7) * ss);
                let row7 = load(next);
                let row8 = load(next.add(ss));
                let row9 = load(next.add(2 * ss));
                let row10 = load(next.add(3 * ss));
                let next0 = transpose([row4, row5, row6, row7]);
                let next1 = transpose([row5, row6, row7, row8]);
                let next2 = transpose([row6, row7, row8, row9]);
                let next3 = transpose([row7, row8, row9, row10]);
                let out = dst.add(y * ds + x);
                let predictions = [
                    filtered(packed0, next0, taps, max),
                    filtered(packed1, next1, taps, max),
                    filtered(packed2, next2, taps, max),
                    filtered(packed3, next3, taps, max),
                ];
                for (row, pred) in predictions.into_iter().enumerate() {
                    let out = out.add(row * ds);
                    let pred = if AVG {
                        vrhadd_u16(pred, vld1_u16(out))
                    } else {
                        pred
                    };
                    vst1_u16(out, pred);
                }
                (packed0, packed1, packed2, packed3) = (next0, next1, next2, next3);
                (row4, row5, row6) = (row8, row9, row10);
            }
        }
    }
}
