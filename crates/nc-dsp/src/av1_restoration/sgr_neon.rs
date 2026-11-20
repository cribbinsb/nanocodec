//! The NEON arithmetic of the self-guided stripe kernel ([`super::sgr_stripe`],
//! which holds the schedule and the layout, shared with
//! [`super::sgr_avx2`]), after dav1d 1.5.4 `src/arm/64/looprestoration.S`
//! and `looprestoration_common.S` (`sgr_box35_row_h_8bpc_neon`,
//! `sgr_box5_vert_neon`; BSD-2-Clause, see `THIRD_PARTY.md`).
//!
//! A step's 16 box columns are four 32-bit registers of squares and two
//! 16-bit registers of sums, in column order. The taps of a step are `ext`
//! windows of its 24 bytes; sums widen with `uaddl`/`uaddw`, squares are
//! `umull` (exact in 16 bits) summed in pairs by `uaddl`. `z` saturates to
//! a byte (`uqxtn`) and `x = 256 − A` comes from a 256-byte table by four
//! `tbl`/`tbx` lookups, as dav1d avoids a division. The neighbourhoods work
//! on the shared packed `x | B << 12` words, four columns a register; the
//! projections narrow the differences to 16 bits for the single pass
//! (`sqrdmulh` by `16 · w`, AVX2's `pmulhrsw`) and stay in 32 bits for the
//! mixed one.

#![allow(
    clippy::many_single_char_names,
    reason = "the spec's and dav1d's sample, tap and statistic names"
)]
#![allow(clippy::inline_always)]
#![allow(unsafe_code)]

use core::arch::aarch64::{
    int32x4_t, uint8x16_t, uint16x8_t, uint32x4_t, vaddl_high_u8, vaddl_high_u16, vaddl_u8,
    vaddl_u16, vaddq_s16, vaddq_s32, vaddq_u16, vaddq_u32, vaddw_high_u8, vaddw_high_u16, vaddw_u8,
    vaddw_u16, vandq_u32, vbicq_s32, vcombine_s16, vcombine_u8, vcombine_u16, vdupq_n_s16,
    vdupq_n_s32, vdupq_n_u8, vdupq_n_u32, vextq_u8, vget_high_u16, vget_low_u8, vget_low_u16,
    vld1_u8, vld1q_s32, vld1q_u8, vld1q_u8_x4, vld1q_u16, vld1q_u32, vmlaq_n_s32, vmlsq_s32,
    vmovl_high_u8, vmovl_high_u16, vmovl_u8, vmovl_u16, vmull_high_u8, vmull_n_u16, vmull_u8,
    vmull_u16, vmulq_n_u32, vmulq_u32, vorrq_u32, vqmovn_s32, vqmovn_u16, vqmovn_u32, vqmovun_s16,
    vqrdmulhq_s16, vqsubq_u32, vqtbl4q_u8, vqtbx4q_u8, vreinterpretq_s16_u16,
    vreinterpretq_s32_u32, vreinterpretq_u32_s32, vrshrq_n_u32, vshlq_n_s32, vshrq_n_s32,
    vshrq_n_u32, vst1q_s32, vst1q_u8, vst1q_u16, vst1q_u32, vsubq_u8,
};

use nc_common::PlaneViewMut;

use super::self_guided::{
    MTABLE_BITS, SGR_BITS, SgrStripe, SgrStripeScratch, one_by_x, x_by_xplus1,
};
use super::sgr_stripe::{
    B_ROUND, BLEND_SHIFT, Kept, ONE_ROW_SHIFT, STEP, Single, StripeLanes, SumRow, TWO_ROW_SHIFT,
    X_BITS,
};

/// Columns (32-bit lanes) per register.
const QUAD: usize = 4;
/// Columns (16-bit lanes) per register.
const HALF: usize = 8;
/// Entries of one `tbl` over four registers.
const TABLE_PART: usize = 64;
/// `x = 256 − A` for each `z` (AV1 §7.17.3; `z ≥ 255` gives 0).
const X_TABLE: [u8; 256] = {
    let mut table = [0; 256];
    let mut z = 0;
    while z < table.len() {
        table[z] = ((1 << SGR_BITS) - x_by_xplus1(z as u32)) as u8;
        z += 1;
    }
    table
};

/// One step's box sums: 16 sums in 16 bits, their squares in 32, both in
/// column order.
#[derive(Clone, Copy)]
struct Sums {
    sum: [uint16x8_t; 2],
    squares: [uint32x4_t; 4],
}

/// The taps `0 ..= 4` of a step's 16 box columns: column `c` reads bytes
/// `c ..= c + 4` of the step's 24.
///
/// # Safety
///
/// NEON must be enabled in the caller; `src` holds 24 readable bytes.
#[inline(always)]
unsafe fn taps(src: *const u8) -> [uint8x16_t; 5] {
    // SAFETY: as the contract.
    unsafe {
        let low = vld1q_u8(src);
        let high = vcombine_u8(vld1_u8(src.add(STEP)), vld1_u8(src.add(STEP)));
        [
            low,
            vextq_u8::<1>(low, high),
            vextq_u8::<2>(low, high),
            vextq_u8::<3>(low, high),
            vextq_u8::<4>(low, high),
        ]
    }
}

/// The sum of two taps per column (16 bits).
#[inline(always)]
unsafe fn pair_sum(a: uint8x16_t, b: uint8x16_t) -> [uint16x8_t; 2] {
    // SAFETY: NEON is enabled in the calling kernel.
    unsafe {
        [
            vaddl_u8(vget_low_u8(a), vget_low_u8(b)),
            vaddl_high_u8(a, b),
        ]
    }
}

/// `sum + t` per column.
#[inline(always)]
unsafe fn add_tap(sum: [uint16x8_t; 2], t: uint8x16_t) -> [uint16x8_t; 2] {
    // SAFETY: NEON is enabled in the calling kernel.
    unsafe { [vaddw_u8(sum[0], vget_low_u8(t)), vaddw_high_u8(sum[1], t)] }
}

/// Each column's square (at most `255²`: exact in 16 bits).
#[inline(always)]
unsafe fn square(t: uint8x16_t) -> [uint16x8_t; 2] {
    // SAFETY: NEON is enabled in the calling kernel.
    unsafe {
        [
            vmull_u8(vget_low_u8(t), vget_low_u8(t)),
            vmull_high_u8(t, t),
        ]
    }
}

/// `a² + b²` per column, widened to 32 bits.
#[inline(always)]
unsafe fn pair_squares(a: uint8x16_t, b: uint8x16_t) -> [uint32x4_t; 4] {
    // SAFETY: NEON is enabled in the calling kernel.
    unsafe {
        let (a, b) = (square(a), square(b));
        [
            vaddl_u16(vget_low_u16(a[0]), vget_low_u16(b[0])),
            vaddl_high_u16(a[0], b[0]),
            vaddl_u16(vget_low_u16(a[1]), vget_low_u16(b[1])),
            vaddl_high_u16(a[1], b[1]),
        ]
    }
}

/// `squares + t²` per column.
#[inline(always)]
unsafe fn add_square(squares: [uint32x4_t; 4], t: uint8x16_t) -> [uint32x4_t; 4] {
    // SAFETY: NEON is enabled in the calling kernel.
    unsafe {
        let t = square(t);
        [
            vaddw_u16(squares[0], vget_low_u16(t[0])),
            vaddw_high_u16(squares[1], t[0]),
            vaddw_u16(squares[2], vget_low_u16(t[1])),
            vaddw_high_u16(squares[3], t[1]),
        ]
    }
}

#[inline(always)]
unsafe fn add_squares(a: [uint32x4_t; 4], b: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
    // SAFETY: NEON is enabled in the calling kernel.
    unsafe {
        [
            vaddq_u32(a[0], b[0]),
            vaddq_u32(a[1], b[1]),
            vaddq_u32(a[2], b[2]),
            vaddq_u32(a[3], b[3]),
        ]
    }
}

#[inline(always)]
unsafe fn add_sums(a: [uint16x8_t; 2], b: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
    // SAFETY: NEON is enabled in the calling kernel.
    unsafe { [vaddq_u16(a[0], b[0]), vaddq_u16(a[1], b[1])] }
}

/// Radius-1 sums of taps 1 to 3.
#[inline(always)]
unsafe fn sums3([_, t1, t2, t3, _]: [uint8x16_t; 5]) -> Sums {
    // SAFETY: NEON is enabled in the calling kernel.
    unsafe {
        Sums {
            sum: add_tap(pair_sum(t1, t3), t2),
            squares: add_square(pair_squares(t1, t3), t2),
        }
    }
}

/// The constants of one radius's statistics.
#[derive(Clone, Copy)]
struct Statistics {
    /// Box size `n` (9 or 25).
    count: u32,
    /// `s`.
    scale: u32,
    /// `one_by_x(n)`.
    reciprocal: u16,
}

impl Statistics {
    /// The packed `x | B << 12` words of one step's box sums, in column
    /// order (AV1 §7.17.3 `p`, `z`, `a2`, `b2`).
    #[inline(always)]
    unsafe fn packed(self, box_sums: Sums) -> [uint32x4_t; 4] {
        // SAFETY: NEON is enabled in the calling kernel.
        unsafe {
            let sums = [
                vget_low_u16(box_sums.sum[0]),
                vget_high_u16(box_sums.sum[0]),
                vget_low_u16(box_sums.sum[1]),
                vget_high_u16(box_sums.sum[1]),
            ];
            // z = (p · s + 2^19) >> 20, saturated to a byte; p = n · Σx² −
            // (Σx)² (the scalar code's saturating difference), `p · s`
            // below 2^32 (as in AVX2's kernel).
            let z = [0, 1, 2, 3].map(|q| {
                vrshrq_n_u32::<{ MTABLE_BITS as i32 }>(vmulq_n_u32(
                    vqsubq_u32(
                        vmulq_n_u32(box_sums.squares[q], self.count),
                        vmull_u16(sums[q], sums[q]),
                    ),
                    self.scale,
                ))
            });
            let z = vcombine_u8(
                vqmovn_u16(vcombine_u16(vqmovn_u32(z[0]), vqmovn_u32(z[1]))),
                vqmovn_u16(vcombine_u16(vqmovn_u32(z[2]), vqmovn_u32(z[3]))),
            );
            let x = x_of(z);
            let x = [vmovl_u8(vget_low_u8(x)), vmovl_high_u8(x)];
            let x = [
                vmovl_u16(vget_low_u16(x[0])),
                vmovl_high_u16(x[0]),
                vmovl_u16(vget_low_u16(x[1])),
                vmovl_high_u16(x[1]),
            ];
            let b_mask = vdupq_n_u32(u32::MAX << X_BITS);
            let rounding = vdupq_n_u32(B_ROUND as u32);
            [0, 1, 2, 3].map(|q| {
                // Σx · one_by_x(n) · x, at most 6375 · 164 · 255 (radius 2)
                // or 2295 · 455 · 255 (radius 1): below 2^32.
                let b = vmulq_u32(vmull_n_u16(sums[q], self.reciprocal), x[q]);
                vorrq_u32(vandq_u32(vaddq_u32(b, rounding), b_mask), x[q])
            })
        }
    }
}

/// `X_TABLE[z]` per byte: four `tbl`/`tbx` lookups of 64 entries (the
/// table loads are left to the compiler to keep in registers or reload).
#[inline(always)]
unsafe fn x_of(z: uint8x16_t) -> uint8x16_t {
    // SAFETY: NEON is enabled in the calling kernel; each part is 64 bytes
    // of the table.
    unsafe {
        let part = vdupq_n_u8(TABLE_PART as u8);
        let mut index = z;
        let mut x = vqtbl4q_u8(vld1q_u8_x4(X_TABLE.as_ptr()), index);
        for k in 1..X_TABLE.len() / TABLE_PART {
            index = vsubq_u8(index, part);
            x = vqtbx4q_u8(x, vld1q_u8_x4(X_TABLE[k * TABLE_PART..].as_ptr()), index);
        }
        x
    }
}

// ------------------------------------------------- neighbourhoods ----

/// Split packed words into `x` sums and `B` sums.
#[inline(always)]
unsafe fn unpack(v: int32x4_t) -> (int32x4_t, int32x4_t) {
    // SAFETY: NEON is enabled in the calling kernel.
    unsafe {
        (
            vbicq_s32(v, vdupq_n_s32(-1 << X_BITS)),
            vreinterpretq_s32_u32(vshrq_n_u32::<X_BITS>(vreinterpretq_u32_s32(v))),
        )
    }
}

/// Packed words of box columns `j − 1`, `j`, `j + 1` for 4 output columns
/// from `j` (`packed` starts at box column −1).
#[inline(always)]
unsafe fn neighbours(packed: *const i32, j: usize) -> [int32x4_t; 3] {
    // SAFETY: the caller's columns are inside the row (`sgr_stripe` layout).
    unsafe {
        [
            vld1q_s32(packed.add(j)),
            vld1q_s32(packed.add(j + 1)),
            vld1q_s32(packed.add(j + 2)),
        ]
    }
}

/// A radius-2 box row's weights 5, 6, 5 (one packed add: at most
/// `16 · 255` in the `x` field).
#[inline(always)]
unsafe fn weigh565([l, c, r]: [int32x4_t; 3]) -> int32x4_t {
    // SAFETY: NEON is enabled in the calling kernel.
    unsafe {
        let s = vaddq_s32(vaddq_s32(l, c), r);
        vaddq_s32(vaddq_s32(vshlq_n_s32::<2>(s), s), c)
    }
}

/// A radius-1 box row's weights 2, 2, 2 and 3, 4, 3 (together 16 a side:
/// the centre row's 4, 4, 4 is split between the two halves).
#[inline(always)]
unsafe fn weigh222_343([l, c, r]: [int32x4_t; 3]) -> (int32x4_t, int32x4_t) {
    // SAFETY: NEON is enabled in the calling kernel.
    unsafe {
        let outer = vaddq_s32(l, r);
        let s = vaddq_s32(outer, c);
        let w222 = vaddq_s32(s, s);
        (
            w222,
            core::arch::aarch64::vsubq_s32(vaddq_s32(w222, w222), outer),
        )
    }
}

/// `(B − x · src) >> SHIFT` of one radius (its rounding is in `B`).
#[inline(always)]
unsafe fn difference<const SHIFT: i32>(x: int32x4_t, b: int32x4_t, src: int32x4_t) -> int32x4_t {
    // SAFETY: NEON is enabled in the calling kernel. `x · src` is at most
    // `32 · 255 · 255`.
    unsafe { vshrq_n_s32::<SHIFT>(vmlsq_s32(b, x, src)) }
}

/// The radius-1 `x` and `B` sums of output row `i` from the new box row
/// `i + 1`: the 222 row of box row `i` with the 343 rows of `i − 1` and
/// `i + 1`. Leaves box row `i + 1`'s 222 and 343 rows for row `i + 1`.
#[inline(always)]
unsafe fn radius1_sums(
    packed: *const i32,
    (ab222, ab343): Kept,
    j: usize,
) -> (int32x4_t, int32x4_t) {
    // SAFETY: NEON is enabled in the calling kernel; columns inside rows.
    unsafe {
        let (new222, new343) = weigh222_343(neighbours(packed, j));
        let old222 = vld1q_s32(ab222.add(j));
        let above = vaddq_s32(old222, vld1q_s32(ab343.add(j)));
        let below = vaddq_s32(old222, new343);
        vst1q_s32(ab222.add(j), new222);
        vst1q_s32(ab343.add(j), new343);
        let (above, below) = (unpack(above), unpack(below));
        (vaddq_s32(above.0, below.0), vaddq_s32(above.1, below.1))
    }
}

/// The radius-2 `x` and `B` sums of an output row: between two box rows
/// (the new one's 565 sums plus those kept, which it replaces), or on the
/// kept one.
#[inline(always)]
unsafe fn radius2_sums<const BETWEEN: bool>(
    packed: *const i32,
    (x565, b565): Kept,
    j: usize,
) -> (int32x4_t, int32x4_t) {
    // SAFETY: NEON is enabled in the calling kernel; columns inside rows.
    unsafe {
        let kept = (vld1q_s32(x565.add(j)), vld1q_s32(b565.add(j)));
        if !BETWEEN {
            return kept;
        }
        let (x, b) = unpack(weigh565(neighbours(packed, j)));
        vst1q_s32(x565.add(j), x);
        vst1q_s32(b565.add(j), b);
        (vaddq_s32(x, kept.0), vaddq_s32(b, kept.1))
    }
}

/// The difference of `kind`'s neighbourhood for 4 columns from `j`.
#[inline(always)]
unsafe fn single_difference(
    kind: Single,
    packed: *const i32,
    kept: Kept,
    j: usize,
    src: int32x4_t,
) -> int32x4_t {
    // SAFETY: as the sums' functions.
    unsafe {
        match kind {
            Single::FiveBetween => {
                let (x, b) = radius2_sums::<true>(packed, kept, j);
                difference::<TWO_ROW_SHIFT>(x, b, src)
            }
            Single::FiveOn => {
                let (x, b) = radius2_sums::<false>(packed, kept, j);
                difference::<ONE_ROW_SHIFT>(x, b, src)
            }
            Single::Three => {
                let (x, b) = radius1_sums(packed, kept, j);
                difference::<TWO_ROW_SHIFT>(x, b, src)
            }
        }
    }
}

/// Widen 16 source bytes to four registers of 32-bit lanes.
#[inline(always)]
unsafe fn widen(bytes: uint8x16_t) -> [int32x4_t; 4] {
    // SAFETY: NEON is enabled in the calling kernel.
    unsafe {
        let words = [vmovl_u8(vget_low_u8(bytes)), vmovl_high_u8(bytes)];
        [
            vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(words[0]))),
            vreinterpretq_s32_u32(vmovl_high_u16(words[0])),
            vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(words[1]))),
            vreinterpretq_s32_u32(vmovl_high_u16(words[1])),
        ]
    }
}

/// Write the 16 `bytes` to `dst` from column `j`, or those left at the
/// row's end.
#[inline(always)]
unsafe fn store_step(dst: &mut [u8], j: usize, bytes: uint8x16_t) {
    // SAFETY: NEON is enabled in the calling kernel; the full store is
    // checked, the tail goes through a local.
    unsafe {
        if j + STEP <= dst.len() {
            vst1q_u8(dst.as_mut_ptr().add(j), bytes);
        } else {
            let mut tail = [0_u8; STEP];
            vst1q_u8(tail.as_mut_ptr(), bytes);
            let n = dst.len() - j;
            dst[j..].copy_from_slice(&tail[..n]);
        }
    }
}

/// The NEON lanes of [`super::sgr_stripe`].
struct Neon;

impl StripeLanes for Neon {
    type Sums = Sums;
    type Stats = Statistics;

    #[inline(always)]
    unsafe fn statistics(count: i32, scale: u32) -> Statistics {
        Statistics {
            count: count as u32,
            scale,
            // At most 455 (`one_by_x(9)`).
            reciprocal: one_by_x(count as u32) as u16,
        }
    }

    #[inline(always)]
    unsafe fn load(row: SumRow, k: usize) -> Sums {
        // SAFETY: the caller's step is inside the row; NEON is enabled.
        unsafe {
            let (sum, squares) = (
                row.sum.add(k * STEP).cast::<u16>(),
                row.squares.add(k * STEP),
            );
            Sums {
                sum: [vld1q_u16(sum), vld1q_u16(sum.add(HALF))],
                squares: [0, 1, 2, 3].map(|q| vld1q_u32(squares.add(q * QUAD).cast())),
            }
        }
    }

    #[inline(always)]
    unsafe fn store(row: SumRow, k: usize, v: Sums) {
        // SAFETY: as `load`.
        unsafe {
            let (sum, squares) = (
                row.sum.add(k * STEP).cast::<u16>(),
                row.squares.add(k * STEP),
            );
            vst1q_u16(sum, v.sum[0]);
            vst1q_u16(sum.add(HALF), v.sum[1]);
            for (q, value) in v.squares.into_iter().enumerate() {
                vst1q_u32(squares.add(q * QUAD).cast(), value);
            }
        }
    }

    #[inline(always)]
    unsafe fn add(a: Sums, b: Sums) -> Sums {
        // SAFETY: NEON is enabled in the calling kernel.
        unsafe {
            Sums {
                sum: add_sums(a.sum, b.sum),
                squares: add_squares(a.squares, b.squares),
            }
        }
    }

    #[inline(always)]
    unsafe fn sums3(src: *const u8) -> Sums {
        // SAFETY: NEON is enabled; the caller's row holds 24 bytes.
        unsafe { sums3(taps(src)) }
    }

    #[inline(always)]
    unsafe fn sums5(src: *const u8) -> Sums {
        // SAFETY: as `sums3`.
        unsafe {
            let [t0, t1, t2, t3, t4] = taps(src);
            Sums {
                sum: add_tap(add_sums(pair_sum(t0, t4), pair_sum(t1, t3)), t2),
                squares: add_square(add_squares(pair_squares(t0, t4), pair_squares(t1, t3)), t2),
            }
        }
    }

    #[inline(always)]
    unsafe fn sums35(src: *const u8) -> (Sums, Sums) {
        // SAFETY: as `sums3`.
        unsafe {
            let taps = taps(src);
            let three = sums3(taps);
            let five = Sums {
                sum: add_sums(three.sum, pair_sum(taps[0], taps[4])),
                squares: add_squares(three.squares, pair_squares(taps[0], taps[4])),
            };
            (three, five)
        }
    }

    #[inline(always)]
    unsafe fn store_packed(stats: Statistics, sums: Sums, dst: *mut i32, k: usize) {
        // SAFETY: NEON is enabled; the step is inside the row.
        unsafe {
            let dst = dst.add(k * STEP);
            for (q, value) in stats.packed(sums).into_iter().enumerate() {
                vst1q_u32(dst.add(q * QUAD).cast(), value);
            }
        }
    }

    #[inline(always)]
    unsafe fn radius1_start(packed: *const i32, (ab222, ab343): Kept, w: usize) {
        // SAFETY: NEON is enabled in the calling kernel; columns inside rows.
        unsafe {
            for j in (0..w).step_by(QUAD) {
                let (w222, w343) = weigh222_343(neighbours(packed, j));
                vst1q_s32(ab222.add(j), w222);
                vst1q_s32(ab343.add(j), w343);
            }
        }
    }

    #[inline(always)]
    unsafe fn radius2_start(packed: *const i32, (x565, b565): Kept, w: usize) {
        // SAFETY: NEON is enabled in the calling kernel; columns inside rows.
        unsafe {
            for j in (0..w).step_by(QUAD) {
                let (x, b) = unpack(weigh565(neighbours(packed, j)));
                vst1q_s32(x565.add(j), x);
                vst1q_s32(b565.add(j), b);
            }
        }
    }

    #[inline(always)]
    unsafe fn mix_row<const BETWEEN: bool>(
        src: &[u8],
        dst: &mut [u8],
        (packed5, five): (*const i32, Kept),
        (packed3, three): (*const i32, Kept),
        [w0, w1]: [i32; 2],
    ) {
        let w = dst.len();
        assert!(src.len() >= w.next_multiple_of(STEP));
        // SAFETY: NEON is enabled in the calling kernel; `src` covers each
        // 16-byte load (checked), the rows each step's columns (layout).
        unsafe {
            let rounding = vdupq_n_s32(1 << (BLEND_SHIFT - 1));
            for j in (0..w).step_by(STEP) {
                let s = widen(vld1q_u8(src.as_ptr().add(j)));
                let mut words = [vdupq_n_s16(0); 2];
                for (half, word) in words.iter_mut().enumerate() {
                    let mut out = [vdupq_n_s32(0); 2];
                    for (q, value) in out.iter_mut().enumerate() {
                        let c = j + half * HALF + q * QUAD;
                        let s = s[half * 2 + q];
                        let (x5, b5) = radius2_sums::<BETWEEN>(packed5, five, c);
                        let d5 = if BETWEEN {
                            difference::<TWO_ROW_SHIFT>(x5, b5, s)
                        } else {
                            difference::<ONE_ROW_SHIFT>(x5, b5, s)
                        };
                        let (x3, b3) = radius1_sums(packed3, three, c);
                        let d3 = difference::<TWO_ROW_SHIFT>(x3, b3, s);
                        let v = vmlaq_n_s32(vmlaq_n_s32(rounding, d5, w0), d3, w1);
                        *value = vaddq_s32(vshrq_n_s32::<BLEND_SHIFT>(v), s);
                    }
                    *word = vcombine_s16(vqmovn_s32(out[0]), vqmovn_s32(out[1]));
                }
                let bytes = vcombine_u8(vqmovun_s16(words[0]), vqmovun_s16(words[1]));
                store_step(dst, j, bytes);
            }
        }
    }

    #[inline(always)]
    unsafe fn single_row(
        src: &[u8],
        dst: &mut [u8],
        weight: i32,
        (kind, packed, kept): (Single, *const i32, Kept),
    ) {
        let w = dst.len();
        assert!(src.len() >= w.next_multiple_of(STEP));
        // SAFETY: NEON is enabled in the calling kernel; `src` covers each
        // 16-byte load (checked).
        unsafe {
            // `sqrdmulh` by 16 · w is `(w · d + 2^10) >> 11`; |16 · w| <
            // 2^15 for every legal weight.
            let weight = vdupq_n_s16((weight << (15 - BLEND_SHIFT)) as i16);
            for j in (0..w).step_by(STEP) {
                let bytes = vld1q_u8(src.as_ptr().add(j));
                let s = widen(bytes);
                let d = [0, 1, 2, 3]
                    .map(|q| vqmovn_s32(single_difference(kind, packed, kept, j + q * QUAD, s[q])));
                let words = [vmovl_u8(vget_low_u8(bytes)), vmovl_high_u8(bytes)];
                let v = [0, 1].map(|half| {
                    vaddq_s16(
                        vqrdmulhq_s16(vcombine_s16(d[2 * half], d[2 * half + 1]), weight),
                        vreinterpretq_s16_u16(words[half]),
                    )
                });
                let bytes = vcombine_u8(vqmovun_s16(v[0]), vqmovun_s16(v[1]));
                store_step(dst, j, bytes);
            }
        }
    }
}

/// Self-guided-filter a stripe region ([`super::sgr_stripe::run`]).
///
/// # Safety
///
/// NEON must be available. Extents are checked by `run`.
#[target_feature(enable = "neon")]
pub(super) unsafe fn sgr_stripe(
    params: SgrStripe,
    window: (&[u8], usize),
    size: (usize, usize),
    scratch: &mut SgrStripeScratch,
    dst: (&mut PlaneViewMut<'_>, usize, usize),
) {
    // SAFETY: NEON is enabled (this function's target feature).
    unsafe { super::sgr_stripe::run::<Neon>(params, window, size, scratch, dst) }
}
