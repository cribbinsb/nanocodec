//! VP9 vertical byte interpolation, in four-row batches. The rolling
//! transposed windows follow libvpx's `vpx_convolve8_neon_i8mm.c`
//! (BSD-3-Clause, see `THIRD_PARTY.md`).

#![allow(clippy::inline_always, reason = "keep the rolling groups in registers")]

use crate::mc::neon_dotprod::{transpose_four_rows, unsigned_dot_lane};
use core::arch::aarch64::{
    int8x8_t, uint8x8_t, uint8x16_t, uint8x16x2_t, vcombine_u8, vcombine_u16, vdupq_n_s32, vld1_u8,
    vld1q_s16, vld1q_u8, vmovn_s16, vqmovn_u16, vqrshrun_n_s32, vqtbl2q_u8, vrhadd_u8, vst1_u8,
};

type Columns = [uint8x16_t; 2];

/// The transposed four-row groups overlap by one, two, or three rows.
const MERGE: [[u8; 16]; 3] = {
    let mut masks = [[0; 16]; 3];
    let mut shift = 1;
    while shift <= 3 {
        let mut i = 0;
        while i < 16 {
            let row = i % 4 + shift;
            masks[shift - 1][i] = (i / 4 * 4 + row + if row >= 4 { 12 } else { 0 }) as u8;
            i += 1;
        }
        shift += 1;
    }
    masks
};

#[inline(always)]
unsafe fn pack(rows: [uint8x16_t; 4]) -> Columns {
    // SAFETY: the caller's NEON entry enables the register-only transpose.
    let [left, right, _, _] = unsafe { transpose_four_rows::<false>(rows) };
    [left, right]
}

#[inline(always)]
unsafe fn merge(first: Columns, last: Columns, mask: uint8x16_t) -> Columns {
    // SAFETY: the caller enabled NEON. The table shuffles use registers only.
    unsafe {
        [
            vqtbl2q_u8(uint8x16x2_t(first[0], last[0]), mask),
            vqtbl2q_u8(uint8x16x2_t(first[1], last[1]), mask),
        ]
    }
}

#[inline(always)]
unsafe fn load(src: *const u8) -> uint8x16_t {
    // SAFETY: the caller validated eight readable bytes at this row's offset.
    unsafe {
        let row = vld1_u8(src);
        vcombine_u8(row, row)
    }
}

#[inline(always)]
unsafe fn output(first: Columns, last: Columns, taps: int8x8_t) -> uint8x8_t {
    // SAFETY: I8MM is enabled by dispatch. Noninteger taps fit signed bytes.
    // The full sum may exceed signed i16, so round and clip from i32.
    unsafe {
        let sum = |i: usize| {
            let low = unsigned_dot_lane::<0>(vdupq_n_s32(0), first[i], taps);
            unsigned_dot_lane::<1>(low, last[i], taps)
        };
        vqmovn_u16(vcombine_u16(
            vqrshrun_n_s32::<7>(sum(0)),
            vqrshrun_n_s32::<7>(sum(1)),
        ))
    }
}

/// Filter one eight-column strip. The source starts three rows above it.
///
/// # Safety
///
/// I8MM must be available. Height is a positive multiple of four. Source
/// and destination cover h+7 and h rows respectively, each eight bytes.
#[target_feature(enable = "neon")]
pub(super) unsafe fn strip<const AVG: bool>(
    mut src: *const u8,
    ss: usize,
    mut dst: *mut u8,
    ds: usize,
    h: usize,
    taps: int8x8_t,
) {
    // SAFETY: load the first seven rows and then exactly four new rows per
    // iteration. Each output writes the corresponding eight-byte row.
    unsafe {
        let masks = [
            vld1q_u8(MERGE[0].as_ptr()),
            vld1q_u8(MERGE[1].as_ptr()),
            vld1q_u8(MERGE[2].as_ptr()),
        ];
        let (row0, row1, row2, row3, row4, row5, row6) = (
            load(src),
            load(src.add(ss)),
            load(src.add(2 * ss)),
            load(src.add(3 * ss)),
            load(src.add(4 * ss)),
            load(src.add(5 * ss)),
            load(src.add(6 * ss)),
        );
        let mut first = [
            pack([row0, row1, row2, row3]),
            pack([row1, row2, row3, row4]),
            pack([row2, row3, row4, row5]),
            pack([row3, row4, row5, row6]),
        ];
        src = src.add(7 * ss);
        for row in 0..h / 4 {
            let end = pack([
                load(src),
                load(src.add(ss)),
                load(src.add(2 * ss)),
                load(src.add(3 * ss)),
            ]);
            let last = [
                merge(first[3], end, masks[0]),
                merge(first[3], end, masks[1]),
                merge(first[3], end, masks[2]),
                end,
            ];
            for i in 0..4 {
                let out = dst.add(i * ds);
                let filtered = output(first[i], last[i], taps);
                let value = if AVG {
                    vrhadd_u8(filtered, vld1_u8(out))
                } else {
                    filtered
                };
                vst1_u8(out, value);
            }
            first = last;
            if row + 1 < h / 4 {
                src = src.add(4 * ss);
                dst = dst.add(4 * ds);
            }
        }
    }
}

/// Dispatch the strips of an eligible VP9 vertical-only filter.
///
/// # Safety
///
/// I8MM must be available. Both rectangles must satisfy the VP9 filter
/// contract, width is divisible by eight and height is divisible by four.
#[inline(never)]
#[target_feature(enable = "neon")]
pub(super) unsafe fn filter<const AVG: bool>(
    src: &[u8],
    ss: usize,
    (w, h): (usize, usize),
    dst: &mut [u8],
    ds: usize,
    taps: &[i16; 8],
) {
    // SAFETY: the caller validates the full footprints and the selected ISA.
    unsafe {
        let taps = vmovn_s16(vld1q_s16(taps.as_ptr()));
        for col in (0..w).step_by(8) {
            strip::<AVG>(
                src.as_ptr().add(col + crate::vp9_mc::VP9_TAPS_BEFORE),
                ss,
                dst.as_mut_ptr().add(col),
                ds,
                h,
                taps,
            );
        }
    }
}
