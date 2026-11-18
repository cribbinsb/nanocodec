//! dav1d 1.5.4 `mc_dotprod.S` vertical schedule: two transposed four-row
//! groups per four columns, SDOT, then TBL2 to retain three rows and append
//! the next. BSD-2-Clause, VideoLAN/dav1d authors; see `THIRD_PARTY.md`.
//! Exact four/eight/sixteen-byte reads preserve nc-dsp's unpadded contract.

#![allow(
    clippy::inline_always,
    reason = "keep register shuffles in the ISA loop"
)]

use super::{Output, TAP_CENTRE};
use crate::mc::neon_dotprod::{DOT_PIXEL_BIAS, dot_lane};
use core::arch::aarch64::{
    int8x16_t, int8x16x2_t, uint8x16_t, vcombine_u8, vcreate_u8, vdupq_n_s32, vdupq_n_u8, veorq_u8,
    vld1_u8, vld1q_s16, vld1q_u8, vmovn_s16, vqtbl2q_s8, vreinterpretq_s8_u8,
    vreinterpretq_s16_s32, vreinterpretq_u8_u16, vreinterpretq_u16_u8, vuzp1q_s16, vzip1q_u8,
    vzip1q_u16, vzip2q_u8, vzip2q_u16,
};

/// Shift one row within each transposed four-column group.
const SHIFT: [u8; 16] = [1, 2, 3, 16, 5, 6, 7, 20, 9, 10, 11, 24, 13, 14, 15, 28];
/// The new row is untransposed; each group selects its four columns.
const APPEND: [[u8; 16]; 4] = [
    [1, 2, 3, 16, 5, 6, 7, 17, 9, 10, 11, 18, 13, 14, 15, 19],
    [1, 2, 3, 20, 5, 6, 7, 21, 9, 10, 11, 22, 13, 14, 15, 23],
    [1, 2, 3, 24, 5, 6, 7, 25, 9, 10, 11, 26, 13, 14, 15, 27],
    [1, 2, 3, 28, 5, 6, 7, 29, 9, 10, 11, 30, 13, 14, 15, 31],
];

#[inline(always)]
unsafe fn load<const W: usize>(src: *const u8) -> uint8x16_t {
    // SAFETY: the parent validated W readable pixels at this row/column.
    unsafe {
        if W >= 16 {
            vld1q_u8(src)
        } else {
            let row = if W == 4 {
                vcreate_u8(u64::from(src.cast::<u32>().read_unaligned()))
            } else {
                vld1_u8(src)
            };
            vcombine_u8(row, row)
        }
    }
}

#[inline(always)]
unsafe fn transpose(rows: [uint8x16_t; 4]) -> [int8x16_t; 4] {
    // SAFETY: register-only interleaving packs four rows per four columns.
    unsafe {
        let a = vreinterpretq_u16_u8(vzip1q_u8(rows[0], rows[1]));
        let b = vreinterpretq_u16_u8(vzip1q_u8(rows[2], rows[3]));
        let c = vreinterpretq_u16_u8(vzip2q_u8(rows[0], rows[1]));
        let d = vreinterpretq_u16_u8(vzip2q_u8(rows[2], rows[3]));
        let bias = vdupq_n_u8(128);
        [
            vreinterpretq_s8_u8(veorq_u8(vreinterpretq_u8_u16(vzip1q_u16(a, b)), bias)),
            vreinterpretq_s8_u8(veorq_u8(vreinterpretq_u8_u16(vzip2q_u16(a, b)), bias)),
            vreinterpretq_s8_u8(veorq_u8(vreinterpretq_u8_u16(vzip1q_u16(c, d)), bias)),
            vreinterpretq_s8_u8(veorq_u8(vreinterpretq_u8_u16(vzip2q_u16(c, d)), bias)),
        ]
    }
}

/// Eight signed-byte coefficients sum to 64. Pixel high-bit flipping adds
/// a constant -8192 to their dot product; the initial accumulator restores it.
#[inline]
#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn filter<const W: usize, O: Output>(
    src: *const u8,
    ss: usize,
    h: usize,
    taps: &[i16; 8],
    out: &mut O,
) {
    if h == 0 {
        return;
    }
    let groups = W.min(16) / 4;
    // SAFETY: the parent validated h+7 source rows and every output row.
    // The coefficient table contains eight signed-byte values.
    unsafe {
        let coefficients = vmovn_s16(vld1q_s16(taps.as_ptr()));
        let shift = vld1q_u8(SHIFT.as_ptr());
        let mut append = [shift; 4];
        for g in 0..groups {
            append[g] = vld1q_u8(APPEND[g].as_ptr());
        }
        for col in (0..W).step_by(W.min(16)) {
            let origin = src.add(TAP_CENTRE + col);
            let mut first = transpose([
                load::<W>(origin),
                load::<W>(origin.add(ss)),
                load::<W>(origin.add(2 * ss)),
                load::<W>(origin.add(3 * ss)),
            ]);
            let mut last = transpose([
                load::<W>(origin.add(4 * ss)),
                load::<W>(origin.add(5 * ss)),
                load::<W>(origin.add(6 * ss)),
                load::<W>(origin.add(7 * ss)),
            ]);
            for row in 0..h {
                let mut sums = [vdupq_n_s32(DOT_PIXEL_BIAS); 4];
                for g in 0..groups {
                    sums[g] = dot_lane::<0>(sums[g], first[g], coefficients);
                    sums[g] = dot_lane::<1>(sums[g], last[g], coefficients);
                }
                let low = vuzp1q_s16(
                    vreinterpretq_s16_s32(sums[0]),
                    vreinterpretq_s16_s32(sums[1]),
                );
                out.vertical(row, col, low);
                if W >= 16 {
                    let high = vuzp1q_s16(
                        vreinterpretq_s16_s32(sums[2]),
                        vreinterpretq_s16_s32(sums[3]),
                    );
                    out.vertical(row, col + 8, high);
                }
                // Avoid dav1d's padded-read assumption at the final row.
                if row + 1 < h {
                    let next = vreinterpretq_s8_u8(veorq_u8(
                        load::<W>(origin.add((row + 8) * ss)),
                        vdupq_n_u8(128),
                    ));
                    for g in 0..groups {
                        first[g] = vqtbl2q_s8(int8x16x2_t(first[g], last[g]), shift);
                        last[g] = vqtbl2q_s8(int8x16x2_t(last[g], next), append[g]);
                    }
                }
            }
        }
    }
}
