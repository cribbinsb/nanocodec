//! Multi-tier `u8` and `u16` SIMD kernels for [`crate::Dsp::predict_high_precision`]
//! (H.266 §8.5.6.3.2, HEVC §8.5.3.3.3.1).
//!
//! At every bit depth `8..=12`, `shift1 = min(4, depth - 8)` scales horizontal
//! sums `(Σ cx[k] * s[x + k]) >> shift1` into `-6143..=22522`, which fits
//! signed `i16` lanes without saturation. In 2-D interpolation, vertical sums
//! `(Σ cy[k] * h[y + k]) >> 6` lie in `-12000..=33247`; subtracting `8192 << 6`
//! before narrowing keeps `-20192..=25055` strictly inside `i16`, and a
//! wrapping `+ 8192` reproduces `(sum >> 6) as i16` bit-for-bit.
//!
//! Blocks of width `>= 8` decompose into `W`/`M`/`V`-lane vertical strips that
//! slide the active 4, 6, or 8 horizontal rows in SIMD registers with pre-splatted
//! tap-pair vectors and zero stack scratch; `4`- and `8`-wide blocks pack 2 or 4
//! rows per vector (`VVdeC` `InterpolationFilterX86` strip/register layout,
//! BSD-3-Clause-Clear; see `THIRD_PARTY.md`).

#![allow(unsafe_code, clippy::inline_always)]

use core::mem::MaybeUninit;

use nc_common::{BitDepth, PlaneView};

use crate::simd::{I16FilterLanes, I16Lanes, I16Pairs, I32Lanes, SimdSample};

use super::precision::HalfLumaTaps;
use super::scalar::{SECOND_PASS_SHIFT, interpolation_shifts};
use super::{
    HevcChromaTaps, HevcLumaTaps, SeparableTaps, VvcAffineLumaTaps, VvcChromaTaps, VvcLumaTaps,
};

const BIAS: i16 = 8192;
const BIAS_PRE_SHIFT: i32 = -(8192 << SECOND_PASS_SHIFT);
pub(super) type OutScale = (i32, u32, i16);
const PRECISION_SCALE: OutScale = (BIAS_PRE_SHIFT, SECOND_PASS_SHIFT, BIAS);
pub(super) const PREP_SCALE: OutScale = (BIAS_PRE_SHIFT, SECOND_PASS_SHIFT, 0);

#[inline(always)]
fn finish_acc<L: I16Lanes>(acc: L::Acc, (round, shift, bias): OutScale) -> L {
    let v = L::acc_round_shift_narrow(acc, round, shift);
    if bias == 0 { v } else { v.add(L::splat(bias)) }
}

#[inline(always)]
fn splat_horiz_pairs<const N: usize, L: I16Pairs>(cx: &[i16; N]) -> [L; 4] {
    if N == 4 {
        let (t01, t23) = (L::splat_pair(cx[0], cx[1]), L::splat_pair(cx[2], cx[3]));
        [t01, t23, t01, t23]
    } else if cx[0] == 0 && cx[7] == 0 {
        let t12 = L::splat_pair(cx[1], cx[2]);
        let t34 = L::splat_pair(cx[3], cx[4]);
        let t56 = L::splat_pair(cx[5], cx[6]);
        [t12, t34, t56, t56]
    } else {
        [
            L::splat_pair(cx[0], cx[1]),
            L::splat_pair(cx[2], cx[3]),
            L::splat_pair(cx[4], cx[5]),
            L::splat_pair(cx[6], cx[7]),
        ]
    }
}

#[inline(always)]
fn splat_vert_pairs<const N: usize, L: I16FilterLanes>(cy: &[i16; N]) -> [[L; 2]; 4] {
    if N == 4 {
        let (t01, t23) = (
            L::splat_alf_pair(cy[0], cy[1]),
            L::splat_alf_pair(cy[2], cy[3]),
        );
        [t01, t23, t01, t23]
    } else if cy[0] == 0 && cy[7] == 0 {
        let t12 = L::splat_alf_pair(cy[1], cy[2]);
        let t34 = L::splat_alf_pair(cy[3], cy[4]);
        let t56 = L::splat_alf_pair(cy[5], cy[6]);
        [t12, t34, t56, t56]
    } else {
        [
            L::splat_alf_pair(cy[0], cy[1]),
            L::splat_alf_pair(cy[2], cy[3]),
            L::splat_alf_pair(cy[4], cy[5]),
            L::splat_alf_pair(cy[6], cy[7]),
        ]
    }
}

#[inline(always)]
unsafe fn horiz_ptr_acc<const N: usize, S: SimdSample, L: I16Lanes>(
    p: *const S,
    six_x: bool,
    cx: &[i16; N],
    round: i32,
    shift: u32,
) -> L {
    if L::FAST_HORIZ_EXT && (N == 4 || N == 8) {
        // SAFETY: caller validated `p` covers `L::LANES + N - 1` readable samples,
        // so `p[0..L::LANES]` and `p[N - 1..L::LANES + N - 1]` are both in bounds.
        let (s0, s_last) = unsafe { (S::load_lanes::<L>(p), S::load_lanes::<L>(p.add(N - 1))) };
        let acc = L::horiz_taps_acc::<N>(s0, s_last, six_x, cx);
        return L::acc_round_shift_narrow(acc, round, shift);
    }
    let mut acc = L::acc_zero();
    // SAFETY: caller validated `p` covers `L::LANES + N - 1` readable samples.
    unsafe {
        if N == 4 {
            acc = L::acc_mul_add2(
                acc,
                S::load_lanes::<L>(p),
                cx[0],
                S::load_lanes::<L>(p.add(1)),
                cx[1],
            );
            acc = L::acc_mul_add2(
                acc,
                S::load_lanes::<L>(p.add(2)),
                cx[2],
                S::load_lanes::<L>(p.add(3)),
                cx[3],
            );
        } else if six_x {
            acc = L::acc_mul_add2(
                acc,
                S::load_lanes::<L>(p.add(1)),
                cx[1],
                S::load_lanes::<L>(p.add(2)),
                cx[2],
            );
            acc = L::acc_mul_add2(
                acc,
                S::load_lanes::<L>(p.add(3)),
                cx[3],
                S::load_lanes::<L>(p.add(4)),
                cx[4],
            );
            acc = L::acc_mul_add2(
                acc,
                S::load_lanes::<L>(p.add(5)),
                cx[5],
                S::load_lanes::<L>(p.add(6)),
                cx[6],
            );
        } else {
            acc = L::acc_mul_add2(
                acc,
                S::load_lanes::<L>(p),
                cx[0],
                S::load_lanes::<L>(p.add(1)),
                cx[1],
            );
            acc = L::acc_mul_add2(
                acc,
                S::load_lanes::<L>(p.add(2)),
                cx[2],
                S::load_lanes::<L>(p.add(3)),
                cx[3],
            );
            acc = L::acc_mul_add2(
                acc,
                S::load_lanes::<L>(p.add(4)),
                cx[4],
                S::load_lanes::<L>(p.add(5)),
                cx[5],
            );
            acc = L::acc_mul_add2(
                acc,
                S::load_lanes::<L>(p.add(6)),
                cx[6],
                S::load_lanes::<L>(p.add(7)),
                cx[7],
            );
        }
    }
    L::acc_round_shift_narrow(acc, round, shift)
}

#[inline(always)]
pub(super) fn horiz_vec_round<const N: usize, L: I16Lanes>(
    src: &[u16],
    x: usize,
    cx: &[i16; N],
    round: i32,
    shift: u32,
) -> L {
    let p = src[x..x + L::LANES + N - 1].as_ptr();
    let six_x = N == 8 && cx[0] == 0 && cx[7] == 0;
    // SAFETY: `src[x..x + L::LANES + N - 1]` bounds-checked once above.
    unsafe { horiz_ptr_acc::<N, u16, L>(p, six_x, cx, round, shift) }
}

#[inline(always)]
unsafe fn horiz_ptr_pair_round<const N: usize, S: SimdSample, L: I16Pairs>(
    p: *const S,
    six_x: bool,
    tx: &[L; 4],
    cx: &[i16; N],
    round: i32,
    shift: u32,
) -> L {
    if !L::FAST_PAIR_DOT {
        // SAFETY: caller validated `p` covers `L::LANES + N - 1` readable samples.
        return unsafe { horiz_ptr_acc::<N, S, L>(p, six_x, cx, round, shift) };
    }
    // SAFETY: caller validated `p` covers `L::LANES + N - 1` readable samples.
    let (mut even, mut odd) = unsafe {
        if N == 4 {
            let e = S::load_lanes::<L>(p)
                .pair_dot(tx[0])
                .add(S::load_lanes::<L>(p.add(2)).pair_dot(tx[1]));
            let o = S::load_lanes::<L>(p.add(1))
                .pair_dot(tx[0])
                .add(S::load_lanes::<L>(p.add(3)).pair_dot(tx[1]));
            (e, o)
        } else if six_x {
            let e = S::load_lanes::<L>(p.add(1))
                .pair_dot(tx[0])
                .add(S::load_lanes::<L>(p.add(3)).pair_dot(tx[1]))
                .add(S::load_lanes::<L>(p.add(5)).pair_dot(tx[2]));
            let o = S::load_lanes::<L>(p.add(2))
                .pair_dot(tx[0])
                .add(S::load_lanes::<L>(p.add(4)).pair_dot(tx[1]))
                .add(S::load_lanes::<L>(p.add(6)).pair_dot(tx[2]));
            (e, o)
        } else {
            let e = S::load_lanes::<L>(p)
                .pair_dot(tx[0])
                .add(S::load_lanes::<L>(p.add(2)).pair_dot(tx[1]))
                .add(S::load_lanes::<L>(p.add(4)).pair_dot(tx[2]))
                .add(S::load_lanes::<L>(p.add(6)).pair_dot(tx[3]));
            let o = S::load_lanes::<L>(p.add(1))
                .pair_dot(tx[0])
                .add(S::load_lanes::<L>(p.add(3)).pair_dot(tx[1]))
                .add(S::load_lanes::<L>(p.add(5)).pair_dot(tx[2]))
                .add(S::load_lanes::<L>(p.add(7)).pair_dot(tx[3]));
            (e, o)
        }
    };
    if round != 0 {
        let r_vec = L::Pairs::splat(round);
        even = even.add(r_vec);
        odd = odd.add(r_vec);
    }
    if shift == 0 {
        L::interleave_pairs(even, odd)
    } else {
        L::interleave_pairs(even.sra(shift), odd.sra(shift))
    }
}

#[inline(always)]
pub(super) fn horiz_pair_vec_round<const N: usize, L: I16Pairs>(
    src: &[u16],
    x: usize,
    cx: &[i16; N],
    round: i32,
    shift: u32,
) -> L {
    let p = src[x..x + L::LANES + N - 1].as_ptr();
    let six_x = N == 8 && cx[0] == 0 && cx[7] == 0;
    let tx = splat_horiz_pairs::<N, L>(cx);
    // SAFETY: `src[x..x + L::LANES + N - 1]` bounds-checked once above.
    unsafe { horiz_ptr_pair_round::<N, u16, L>(p, six_x, &tx, cx, round, shift) }
}

#[inline(always)]
unsafe fn horiz_2rows_acc<const N: usize, S: SimdSample, L: I16FilterLanes>(
    p0: *const S,
    p1: *const S,
    six_x: bool,
    cx: &[i16; N],
    shift1: u32,
) -> L {
    let mut acc = L::acc_zero();
    // SAFETY: caller validated `p0` and `p1` cover `L::LANES / 2 + N - 1` samples.
    unsafe {
        if N == 4 {
            acc = L::acc_mul_add2(
                acc,
                S::load_2rows(p0, p1, 0),
                cx[0],
                S::load_2rows(p0, p1, 1),
                cx[1],
            );
            acc = L::acc_mul_add2(
                acc,
                S::load_2rows(p0, p1, 2),
                cx[2],
                S::load_2rows(p0, p1, 3),
                cx[3],
            );
        } else if six_x {
            acc = L::acc_mul_add2(
                acc,
                S::load_2rows(p0, p1, 1),
                cx[1],
                S::load_2rows(p0, p1, 2),
                cx[2],
            );
            acc = L::acc_mul_add2(
                acc,
                S::load_2rows(p0, p1, 3),
                cx[3],
                S::load_2rows(p0, p1, 4),
                cx[4],
            );
            acc = L::acc_mul_add2(
                acc,
                S::load_2rows(p0, p1, 5),
                cx[5],
                S::load_2rows(p0, p1, 6),
                cx[6],
            );
        } else {
            acc = L::acc_mul_add2(
                acc,
                S::load_2rows(p0, p1, 0),
                cx[0],
                S::load_2rows(p0, p1, 1),
                cx[1],
            );
            acc = L::acc_mul_add2(
                acc,
                S::load_2rows(p0, p1, 2),
                cx[2],
                S::load_2rows(p0, p1, 3),
                cx[3],
            );
            acc = L::acc_mul_add2(
                acc,
                S::load_2rows(p0, p1, 4),
                cx[4],
                S::load_2rows(p0, p1, 5),
                cx[5],
            );
            acc = L::acc_mul_add2(
                acc,
                S::load_2rows(p0, p1, 6),
                cx[6],
                S::load_2rows(p0, p1, 7),
                cx[7],
            );
        }
    }
    L::acc_round_shift_narrow(acc, 0, shift1)
}

#[inline(always)]
unsafe fn horiz_ptr_2rows<const N: usize, S: SimdSample, L: I16FilterLanes + I16Pairs>(
    p0: *const S,
    p1: *const S,
    six_x: bool,
    tx: &[L; 4],
    cx: &[i16; N],
    shift1: u32,
) -> L {
    if !L::FAST_PAIR_DOT {
        // SAFETY: caller validated `p0` and `p1` cover `L::LANES / 2 + N - 1` samples.
        return unsafe { horiz_2rows_acc::<N, S, L>(p0, p1, six_x, cx, shift1) };
    }
    // SAFETY: caller validated `p0` and `p1` cover `L::LANES / 2 + N - 1` samples.
    let (even, odd) = unsafe {
        if N == 4 {
            let e = S::load_2rows::<L>(p0, p1, 0)
                .pair_dot(tx[0])
                .add(S::load_2rows::<L>(p0, p1, 2).pair_dot(tx[1]));
            let o = S::load_2rows::<L>(p0, p1, 1)
                .pair_dot(tx[0])
                .add(S::load_2rows::<L>(p0, p1, 3).pair_dot(tx[1]));
            (e, o)
        } else if six_x {
            let e = S::load_2rows::<L>(p0, p1, 1)
                .pair_dot(tx[0])
                .add(S::load_2rows::<L>(p0, p1, 3).pair_dot(tx[1]))
                .add(S::load_2rows::<L>(p0, p1, 5).pair_dot(tx[2]));
            let o = S::load_2rows::<L>(p0, p1, 2)
                .pair_dot(tx[0])
                .add(S::load_2rows::<L>(p0, p1, 4).pair_dot(tx[1]))
                .add(S::load_2rows::<L>(p0, p1, 6).pair_dot(tx[2]));
            (e, o)
        } else {
            let e = S::load_2rows::<L>(p0, p1, 0)
                .pair_dot(tx[0])
                .add(S::load_2rows::<L>(p0, p1, 2).pair_dot(tx[1]))
                .add(S::load_2rows::<L>(p0, p1, 4).pair_dot(tx[2]))
                .add(S::load_2rows::<L>(p0, p1, 6).pair_dot(tx[3]));
            let o = S::load_2rows::<L>(p0, p1, 1)
                .pair_dot(tx[0])
                .add(S::load_2rows::<L>(p0, p1, 3).pair_dot(tx[1]))
                .add(S::load_2rows::<L>(p0, p1, 5).pair_dot(tx[2]))
                .add(S::load_2rows::<L>(p0, p1, 7).pair_dot(tx[3]));
            (e, o)
        }
    };
    if shift1 == 0 {
        L::interleave_pairs(even, odd)
    } else {
        L::interleave_pairs(even.sra(shift1), odd.sra(shift1))
    }
}

#[inline(always)]
unsafe fn horiz_4rows_acc<const N: usize, S: SimdSample, L: I16FilterLanes>(
    r: [*const S; 4],
    six_x: bool,
    cx: &[i16; N],
    shift1: u32,
) -> L {
    let mut acc = L::acc_zero();
    // SAFETY: caller validated each `r[k]` covers `L::LANES / 4 + N - 1` samples.
    unsafe {
        if N == 4 {
            acc = L::acc_mul_add2(acc, S::load_4rows(r, 0), cx[0], S::load_4rows(r, 1), cx[1]);
            acc = L::acc_mul_add2(acc, S::load_4rows(r, 2), cx[2], S::load_4rows(r, 3), cx[3]);
        } else if six_x {
            acc = L::acc_mul_add2(acc, S::load_4rows(r, 1), cx[1], S::load_4rows(r, 2), cx[2]);
            acc = L::acc_mul_add2(acc, S::load_4rows(r, 3), cx[3], S::load_4rows(r, 4), cx[4]);
            acc = L::acc_mul_add2(acc, S::load_4rows(r, 5), cx[5], S::load_4rows(r, 6), cx[6]);
        } else {
            acc = L::acc_mul_add2(acc, S::load_4rows(r, 0), cx[0], S::load_4rows(r, 1), cx[1]);
            acc = L::acc_mul_add2(acc, S::load_4rows(r, 2), cx[2], S::load_4rows(r, 3), cx[3]);
            acc = L::acc_mul_add2(acc, S::load_4rows(r, 4), cx[4], S::load_4rows(r, 5), cx[5]);
            acc = L::acc_mul_add2(acc, S::load_4rows(r, 6), cx[6], S::load_4rows(r, 7), cx[7]);
        }
    }
    L::acc_round_shift_narrow(acc, 0, shift1)
}

#[inline(always)]
unsafe fn horiz_ptr_4rows<const N: usize, S: SimdSample, L: I16FilterLanes + I16Pairs>(
    r: [*const S; 4],
    six_x: bool,
    tx: &[L; 4],
    cx: &[i16; N],
    shift1: u32,
) -> L {
    if !L::FAST_PAIR_DOT {
        // SAFETY: caller validated each `r[k]` covers `L::LANES / 4 + N - 1` samples.
        return unsafe { horiz_4rows_acc::<N, S, L>(r, six_x, cx, shift1) };
    }
    // SAFETY: caller validated each `r[k]` covers `L::LANES / 4 + N - 1` samples.
    let (even, odd) = unsafe {
        if N == 4 {
            let e = S::load_4rows::<L>(r, 0)
                .pair_dot(tx[0])
                .add(S::load_4rows::<L>(r, 2).pair_dot(tx[1]));
            let o = S::load_4rows::<L>(r, 1)
                .pair_dot(tx[0])
                .add(S::load_4rows::<L>(r, 3).pair_dot(tx[1]));
            (e, o)
        } else if six_x {
            let e = S::load_4rows::<L>(r, 1)
                .pair_dot(tx[0])
                .add(S::load_4rows::<L>(r, 3).pair_dot(tx[1]))
                .add(S::load_4rows::<L>(r, 5).pair_dot(tx[2]));
            let o = S::load_4rows::<L>(r, 2)
                .pair_dot(tx[0])
                .add(S::load_4rows::<L>(r, 4).pair_dot(tx[1]))
                .add(S::load_4rows::<L>(r, 6).pair_dot(tx[2]));
            (e, o)
        } else {
            let e = S::load_4rows::<L>(r, 0)
                .pair_dot(tx[0])
                .add(S::load_4rows::<L>(r, 2).pair_dot(tx[1]))
                .add(S::load_4rows::<L>(r, 4).pair_dot(tx[2]))
                .add(S::load_4rows::<L>(r, 6).pair_dot(tx[3]));
            let o = S::load_4rows::<L>(r, 1)
                .pair_dot(tx[0])
                .add(S::load_4rows::<L>(r, 3).pair_dot(tx[1]))
                .add(S::load_4rows::<L>(r, 5).pair_dot(tx[2]))
                .add(S::load_4rows::<L>(r, 7).pair_dot(tx[3]));
            (e, o)
        }
    };
    if shift1 == 0 {
        L::interleave_pairs(even, odd)
    } else {
        L::interleave_pairs(even.sra(shift1), odd.sra(shift1))
    }
}

#[inline(always)]
pub(super) fn vert_u16_vec_round<const N: usize, L: I16Lanes>(
    rows: &[&[u16]; N],
    x: usize,
    cy: &[i16; N],
    round: i32,
    shift: u32,
) -> L {
    let mut acc = L::acc_zero();
    if N == 4 {
        acc = L::acc_mul_add2(
            acc,
            L::load_u16(&rows[0][x..]),
            cy[0],
            L::load_u16(&rows[1][x..]),
            cy[1],
        );
        acc = L::acc_mul_add2(
            acc,
            L::load_u16(&rows[2][x..]),
            cy[2],
            L::load_u16(&rows[3][x..]),
            cy[3],
        );
    } else if cy[0] == 0 && cy[7] == 0 {
        acc = L::acc_mul_add2(
            acc,
            L::load_u16(&rows[1][x..]),
            cy[1],
            L::load_u16(&rows[2][x..]),
            cy[2],
        );
        acc = L::acc_mul_add2(
            acc,
            L::load_u16(&rows[3][x..]),
            cy[3],
            L::load_u16(&rows[4][x..]),
            cy[4],
        );
        acc = L::acc_mul_add2(
            acc,
            L::load_u16(&rows[5][x..]),
            cy[5],
            L::load_u16(&rows[6][x..]),
            cy[6],
        );
    } else {
        acc = L::acc_mul_add2(
            acc,
            L::load_u16(&rows[0][x..]),
            cy[0],
            L::load_u16(&rows[1][x..]),
            cy[1],
        );
        acc = L::acc_mul_add2(
            acc,
            L::load_u16(&rows[2][x..]),
            cy[2],
            L::load_u16(&rows[3][x..]),
            cy[3],
        );
        acc = L::acc_mul_add2(
            acc,
            L::load_u16(&rows[4][x..]),
            cy[4],
            L::load_u16(&rows[5][x..]),
            cy[5],
        );
        acc = L::acc_mul_add2(
            acc,
            L::load_u16(&rows[6][x..]),
            cy[6],
            L::load_u16(&rows[7][x..]),
            cy[7],
        );
    }
    L::acc_round_shift_narrow(acc, round, shift)
}

#[inline(always)]
pub(super) fn vert_i16_vec<const N: usize, L: I16Lanes>(
    rows: &[&[i16]; N],
    x: usize,
    cy: &[i16; N],
) -> L {
    let mut acc = L::acc_zero();
    if N == 4 {
        acc = L::acc_mul_add2(
            acc,
            L::load(&rows[0][x..]),
            cy[0],
            L::load(&rows[1][x..]),
            cy[1],
        );
        acc = L::acc_mul_add2(
            acc,
            L::load(&rows[2][x..]),
            cy[2],
            L::load(&rows[3][x..]),
            cy[3],
        );
    } else if cy[0] == 0 && cy[7] == 0 {
        acc = L::acc_mul_add2(
            acc,
            L::load(&rows[1][x..]),
            cy[1],
            L::load(&rows[2][x..]),
            cy[2],
        );
        acc = L::acc_mul_add2(
            acc,
            L::load(&rows[3][x..]),
            cy[3],
            L::load(&rows[4][x..]),
            cy[4],
        );
        acc = L::acc_mul_add2(
            acc,
            L::load(&rows[5][x..]),
            cy[5],
            L::load(&rows[6][x..]),
            cy[6],
        );
    } else {
        acc = L::acc_mul_add2(
            acc,
            L::load(&rows[0][x..]),
            cy[0],
            L::load(&rows[1][x..]),
            cy[1],
        );
        acc = L::acc_mul_add2(
            acc,
            L::load(&rows[2][x..]),
            cy[2],
            L::load(&rows[3][x..]),
            cy[3],
        );
        acc = L::acc_mul_add2(
            acc,
            L::load(&rows[4][x..]),
            cy[4],
            L::load(&rows[5][x..]),
            cy[5],
        );
        acc = L::acc_mul_add2(
            acc,
            L::load(&rows[6][x..]),
            cy[6],
            L::load(&rows[7][x..]),
            cy[7],
        );
    }
    L::acc_round_shift_narrow(acc, BIAS_PRE_SHIFT, SECOND_PASS_SHIFT).add(L::splat(BIAS))
}

#[inline(always)]
unsafe fn vert_ptr_vec<const N: usize, S: SimdSample, L: I16Lanes>(
    base: *const S,
    w: usize,
    six_tap: bool,
    cy: &[i16; N],
    scale: OutScale,
) -> L {
    let mut acc = L::acc_zero();
    // SAFETY: caller validated `base` across the active rows of `w` stride.
    unsafe {
        if N == 4 {
            acc = L::acc_mul_add2(
                acc,
                S::load_lanes::<L>(base),
                cy[0],
                S::load_lanes::<L>(base.add(w)),
                cy[1],
            );
            acc = L::acc_mul_add2(
                acc,
                S::load_lanes::<L>(base.add(2 * w)),
                cy[2],
                S::load_lanes::<L>(base.add(3 * w)),
                cy[3],
            );
        } else if six_tap {
            acc = L::acc_mul_add2(
                acc,
                S::load_lanes::<L>(base),
                cy[1],
                S::load_lanes::<L>(base.add(w)),
                cy[2],
            );
            acc = L::acc_mul_add2(
                acc,
                S::load_lanes::<L>(base.add(2 * w)),
                cy[3],
                S::load_lanes::<L>(base.add(3 * w)),
                cy[4],
            );
            acc = L::acc_mul_add2(
                acc,
                S::load_lanes::<L>(base.add(4 * w)),
                cy[5],
                S::load_lanes::<L>(base.add(5 * w)),
                cy[6],
            );
        } else {
            acc = L::acc_mul_add2(
                acc,
                S::load_lanes::<L>(base),
                cy[0],
                S::load_lanes::<L>(base.add(w)),
                cy[1],
            );
            acc = L::acc_mul_add2(
                acc,
                S::load_lanes::<L>(base.add(2 * w)),
                cy[2],
                S::load_lanes::<L>(base.add(3 * w)),
                cy[3],
            );
            acc = L::acc_mul_add2(
                acc,
                S::load_lanes::<L>(base.add(4 * w)),
                cy[4],
                S::load_lanes::<L>(base.add(5 * w)),
                cy[5],
            );
            acc = L::acc_mul_add2(
                acc,
                S::load_lanes::<L>(base.add(6 * w)),
                cy[6],
                S::load_lanes::<L>(base.add(7 * w)),
                cy[7],
            );
        }
    }
    finish_acc::<L>(acc, scale)
}

#[inline(always)]
fn vert_regs_4_pre<L: I16FilterLanes>(r: [L; 4], ty: &[[L; 2]; 4], scale: OutScale) -> L {
    let mut acc = L::acc_zero();
    acc = L::acc_alf_linear_pair(acc, r[0], r[1], ty[0]);
    acc = L::acc_alf_linear_pair(acc, r[2], r[3], ty[1]);
    finish_acc::<L>(acc, scale)
}

#[inline(always)]
fn vert_regs_6_pre<L: I16FilterLanes>(r: [L; 6], ty: &[[L; 2]; 4], scale: OutScale) -> L {
    let mut acc = L::acc_zero();
    acc = L::acc_alf_linear_pair(acc, r[0], r[1], ty[0]);
    acc = L::acc_alf_linear_pair(acc, r[2], r[3], ty[1]);
    acc = L::acc_alf_linear_pair(acc, r[4], r[5], ty[2]);
    finish_acc::<L>(acc, scale)
}

#[inline(always)]
fn vert_regs_8_pre<L: I16FilterLanes>(r: [L; 8], ty: &[[L; 2]; 4], scale: OutScale) -> L {
    let mut acc = L::acc_zero();
    acc = L::acc_alf_linear_pair(acc, r[0], r[1], ty[0]);
    acc = L::acc_alf_linear_pair(acc, r[2], r[3], ty[1]);
    acc = L::acc_alf_linear_pair(acc, r[4], r[5], ty[2]);
    acc = L::acc_alf_linear_pair(acc, r[6], r[7], ty[3]);
    finish_acc::<L>(acc, scale)
}

#[inline(always)]
unsafe fn strip_n4<const N: usize, S: SimdSample, L: I16FilterLanes + I16Pairs>(
    (w, h): (usize, usize),
    (mut src_p, stride): (*const S, usize),
    six_x: bool,
    (tx, cx, ty): (&[L; 4], &[i16; N], &[[L; 2]; 4]),
    shift1: u32,
    scale: OutScale,
    mut dst_p: *mut u16,
) {
    // SAFETY: caller validated `src_p` across `(h + 3) * stride` and `dst_p` across `h * w`.
    unsafe {
        let mut r0 = horiz_ptr_pair_round::<N, S, L>(src_p, six_x, tx, cx, 0, shift1);
        src_p = src_p.add(stride);
        let mut r1 = horiz_ptr_pair_round::<N, S, L>(src_p, six_x, tx, cx, 0, shift1);
        src_p = src_p.add(stride);
        let mut r2 = horiz_ptr_pair_round::<N, S, L>(src_p, six_x, tx, cx, 0, shift1);
        for _ in 0..h {
            src_p = src_p.add(stride);
            let r3 = horiz_ptr_pair_round::<N, S, L>(src_p, six_x, tx, cx, 0, shift1);
            vert_regs_4_pre([r0, r1, r2, r3], ty, scale).store_u16_at(dst_p);
            dst_p = dst_p.add(w);
            (r0, r1, r2) = (r1, r2, r3);
        }
    }
}

#[inline(always)]
unsafe fn strip_n6<const N: usize, S: SimdSample, L: I16FilterLanes + I16Pairs>(
    (w, h): (usize, usize),
    (mut src_p, stride): (*const S, usize),
    six_x: bool,
    (tx, cx, ty): (&[L; 4], &[i16; N], &[[L; 2]; 4]),
    shift1: u32,
    scale: OutScale,
    mut dst_p: *mut u16,
) {
    // SAFETY: caller validated `src_p` across `(h + 5) * stride` and `dst_p` across `h * w`.
    unsafe {
        let mut r1 = horiz_ptr_pair_round::<N, S, L>(src_p, six_x, tx, cx, 0, shift1);
        src_p = src_p.add(stride);
        let mut r2 = horiz_ptr_pair_round::<N, S, L>(src_p, six_x, tx, cx, 0, shift1);
        src_p = src_p.add(stride);
        let mut r3 = horiz_ptr_pair_round::<N, S, L>(src_p, six_x, tx, cx, 0, shift1);
        src_p = src_p.add(stride);
        let mut r4 = horiz_ptr_pair_round::<N, S, L>(src_p, six_x, tx, cx, 0, shift1);
        src_p = src_p.add(stride);
        let mut r5 = horiz_ptr_pair_round::<N, S, L>(src_p, six_x, tx, cx, 0, shift1);
        for _ in 0..h {
            src_p = src_p.add(stride);
            let r6 = horiz_ptr_pair_round::<N, S, L>(src_p, six_x, tx, cx, 0, shift1);
            vert_regs_6_pre([r1, r2, r3, r4, r5, r6], ty, scale).store_u16_at(dst_p);
            dst_p = dst_p.add(w);
            (r1, r2, r3, r4, r5) = (r2, r3, r4, r5, r6);
        }
    }
}

#[inline(always)]
unsafe fn strip_n8<const N: usize, S: SimdSample, L: I16FilterLanes + I16Pairs>(
    (w, h): (usize, usize),
    (mut src_p, stride): (*const S, usize),
    six_x: bool,
    (tx, cx, ty): (&[L; 4], &[i16; N], &[[L; 2]; 4]),
    shift1: u32,
    scale: OutScale,
    mut dst_p: *mut u16,
) {
    // SAFETY: caller validated `src_p` across `(h + 7) * stride` and `dst_p` across `h * w`.
    unsafe {
        let mut r0 = horiz_ptr_pair_round::<N, S, L>(src_p, six_x, tx, cx, 0, shift1);
        src_p = src_p.add(stride);
        let mut r1 = horiz_ptr_pair_round::<N, S, L>(src_p, six_x, tx, cx, 0, shift1);
        src_p = src_p.add(stride);
        let mut r2 = horiz_ptr_pair_round::<N, S, L>(src_p, six_x, tx, cx, 0, shift1);
        src_p = src_p.add(stride);
        let mut r3 = horiz_ptr_pair_round::<N, S, L>(src_p, six_x, tx, cx, 0, shift1);
        src_p = src_p.add(stride);
        let mut r4 = horiz_ptr_pair_round::<N, S, L>(src_p, six_x, tx, cx, 0, shift1);
        src_p = src_p.add(stride);
        let mut r5 = horiz_ptr_pair_round::<N, S, L>(src_p, six_x, tx, cx, 0, shift1);
        src_p = src_p.add(stride);
        let mut r6 = horiz_ptr_pair_round::<N, S, L>(src_p, six_x, tx, cx, 0, shift1);
        for _ in 0..h {
            src_p = src_p.add(stride);
            let r7 = horiz_ptr_pair_round::<N, S, L>(src_p, six_x, tx, cx, 0, shift1);
            vert_regs_8_pre([r0, r1, r2, r3, r4, r5, r6, r7], ty, scale).store_u16_at(dst_p);
            dst_p = dst_p.add(w);
            (r0, r1, r2, r3, r4, r5, r6) = (r1, r2, r3, r4, r5, r6, r7);
        }
    }
}

/// Zero-stack register-sliding 2-D interpolation of one `L::LANES`-wide strip at `x_off`.
#[inline(always)]
fn two_pass_strip_1vec<const N: usize, S: SimdSample, L: I16FilterLanes + I16Pairs>(
    (w, h, x_off): (usize, usize, usize),
    source: &PlaneView<'_, S>,
    (cx, cy): (&[i16; N], &[i16; N]),
    (tx, ty): (&[L; 4], &[[L; 2]; 4]),
    shift1: u32,
    scale: OutScale,
    out: &mut [i16],
) {
    if h == 0 {
        return;
    }
    let six_x = N == 8 && cx[0] == 0 && cx[7] == 0;
    let six_y = N == 8 && cy[0] == 0 && cy[7] == 0;
    let y_off = usize::from(six_y);
    let active = if N == 4 {
        4
    } else if six_y {
        6
    } else {
        8
    };
    let stride = source.stride();
    let req_src = (y_off + h + active - 2) * stride + x_off + L::LANES + N - 1;
    let src_data = &source.data()[..req_src];
    let dst_data = &mut out[..w * h];
    debug_assert!(x_off + L::LANES <= w);
    // SAFETY: `src_data` and `dst_data` lengths checked once above.
    unsafe {
        let src_p = src_data.as_ptr().add(y_off * stride + x_off);
        let dst_p = dst_data.as_mut_ptr().add(x_off).cast::<u16>();
        if N == 4 {
            strip_n4::<N, S, L>(
                (w, h),
                (src_p, stride),
                six_x,
                (tx, cx, ty),
                shift1,
                scale,
                dst_p,
            );
        } else if six_y {
            strip_n6::<N, S, L>(
                (w, h),
                (src_p, stride),
                six_x,
                (tx, cx, ty),
                shift1,
                scale,
                dst_p,
            );
        } else {
            strip_n8::<N, S, L>(
                (w, h),
                (src_p, stride),
                six_x,
                (tx, cx, ty),
                shift1,
                scale,
                dst_p,
            );
        }
    }
}

#[inline(always)]
fn horiz_row<const N: usize, S: SimdSample, W: I16Pairs, M: I16Pairs, V: I16Pairs, H: I16Lanes>(
    src: &[S],
    cx: &[i16; N],
    shift1: u32,
    dst: &mut [i16],
) {
    let w = dst.len();
    let six_x = N == 8 && cx[0] == 0 && cx[7] == 0;
    let p = src[..w + N - 1].as_ptr();
    let (tx_w, tx_m, tx_v) = (
        splat_horiz_pairs::<N, W>(cx),
        splat_horiz_pairs::<N, M>(cx),
        splat_horiz_pairs::<N, V>(cx),
    );
    let mut x = 0;
    // SAFETY: `src[..w + N - 1]` validated once above.
    unsafe {
        while x + W::LANES <= w {
            horiz_ptr_pair_round::<N, S, W>(p.add(x), six_x, &tx_w, cx, 0, shift1)
                .store(&mut dst[x..]);
            x += W::LANES;
        }
        if M::LANES < W::LANES && x + M::LANES <= w {
            horiz_ptr_pair_round::<N, S, M>(p.add(x), six_x, &tx_m, cx, 0, shift1)
                .store(&mut dst[x..]);
            x += M::LANES;
        }
        if V::LANES < M::LANES && x + V::LANES <= w {
            horiz_ptr_pair_round::<N, S, V>(p.add(x), six_x, &tx_v, cx, 0, shift1)
                .store(&mut dst[x..]);
            x += V::LANES;
        }
        if x + H::LANES <= w {
            horiz_ptr_acc::<N, S, H>(p.add(x), six_x, cx, 0, shift1).store(&mut dst[x..]);
            x += H::LANES;
        }
        if x < w && w >= H::LANES {
            let tail = w - H::LANES;
            horiz_ptr_acc::<N, S, H>(p.add(tail), six_x, cx, 0, shift1).store(&mut dst[tail..]);
            return;
        }
    }
    while x < w {
        let mut sum = 0i32;
        for k in 0..N {
            sum += i32::from(cx[k]) * src[x + k].to_i32();
        }
        dst[x] = (sum >> shift1) as i16;
        x += 1;
    }
}

#[inline(always)]
fn horiz_w2x4<S: SimdSample, V: I16Lanes>(rows: [&[S]; 4], cx: &[i16; 4], shift1: u32) -> V {
    let mut acc = V::acc_zero();
    acc = V::acc_mul_add2(
        acc,
        S::load_w2x4::<V>(rows, 0),
        cx[0],
        S::load_w2x4::<V>(rows, 1),
        cx[1],
    );
    acc = V::acc_mul_add2(
        acc,
        S::load_w2x4::<V>(rows, 2),
        cx[2],
        S::load_w2x4::<V>(rows, 3),
        cx[3],
    );
    V::acc_round_shift_narrow(acc, 0, shift1)
}

#[inline(always)]
fn two_pass_w2<S: SimdSample, V: I16FilterLanes>(
    h: usize,
    source: &PlaneView<'_, S>,
    (cx, cy): (&[i16; 4], &[i16; 4]),
    shift1: u32,
    scale: OutScale,
    out: &mut [i16],
) {
    let ty = splat_vert_pairs::<4, V>(cy);
    let last = h + 2;
    let mut q0 = horiz_w2x4::<S, V>(
        [source.row(0), source.row(1), source.row(2), source.row(3)],
        cx,
        shift1,
    );
    let mut oy = 0;
    while oy + 4 <= h {
        let q1 = horiz_w2x4::<S, V>(
            [
                source.row(oy + 4),
                source.row((oy + 5).min(last)),
                source.row((oy + 6).min(last)),
                source.row((oy + 7).min(last)),
            ],
            cx,
            shift1,
        );
        let r1234 = q0.shift_quarter(q1);
        let r2345 = q0.combine_mid_halves(q1);
        let r3456 = r2345.shift_quarter(q1.combine_mid_halves(q1));
        vert_regs_4_pre([q0, r1234, r2345, r3456], &ty, scale).store(&mut out[oy * 2..]);
        q0 = q1;
        oy += 4;
    }
    if oy < h {
        let q1 = horiz_w2x4::<S, V>(
            [
                source.row((oy + 4).min(last)),
                source.row((oy + 5).min(last)),
                source.row((oy + 6).min(last)),
                source.row((oy + 7).min(last)),
            ],
            cx,
            shift1,
        );
        let r1234 = q0.shift_quarter(q1);
        let r2345 = q0.combine_mid_halves(q1);
        let r3456 = r2345.shift_quarter(q1.combine_mid_halves(q1));
        let mut tmp = [0i16; 8];
        vert_regs_4_pre([q0, r1234, r2345, r3456], &ty, scale).store(&mut tmp);
        let rem = (h - oy) * 2;
        out[oy * 2..oy * 2 + rem].copy_from_slice(&tmp[..rem]);
    }
}

#[inline(always)]
fn two_pass_w4_n4_regs<const N: usize, S: SimdSample, M: I16FilterLanes + I16Pairs>(
    h: usize,
    source: &PlaneView<'_, S>,
    (cx, cy): (&[i16; N], &[i16; N]),
    shift1: u32,
    scale: OutScale,
    out: &mut [i16],
) {
    let stride = source.stride();
    let last = h + 2;
    let src = source.data()[..last * stride + 4 + N - 1].as_ptr();
    let tx = splat_horiz_pairs::<N, M>(cx);
    let ty = splat_vert_pairs::<N, M>(cy);
    // SAFETY: `src` bounds-checked up to row `last` (`4 + N - 1` cols) once above.
    unsafe {
        let mut q0 = horiz_ptr_4rows::<N, S, M>(
            [
                src,
                src.add(stride),
                src.add(2 * stride),
                src.add(3 * stride),
            ],
            false,
            &tx,
            cx,
            shift1,
        );
        let mut oy = 0;
        while oy + 4 <= h {
            let q1 = horiz_ptr_4rows::<N, S, M>(
                [
                    src.add((oy + 4) * stride),
                    src.add((oy + 5).min(last) * stride),
                    src.add((oy + 6).min(last) * stride),
                    src.add((oy + 7).min(last) * stride),
                ],
                false,
                &tx,
                cx,
                shift1,
            );
            let r1234 = q0.shift_quarter(q1);
            let r2345 = q0.combine_mid_halves(q1);
            let r3456 = r2345.shift_quarter(q1.combine_mid_halves(q1));
            vert_regs_4_pre([q0, r1234, r2345, r3456], &ty, scale).store(&mut out[oy * 4..]);
            q0 = q1;
            oy += 4;
        }
        if oy < h {
            let q1 = horiz_ptr_4rows::<N, S, M>(
                [
                    src.add((oy + 4).min(last) * stride),
                    src.add((oy + 5).min(last) * stride),
                    src.add((oy + 6).min(last) * stride),
                    src.add((oy + 7).min(last) * stride),
                ],
                false,
                &tx,
                cx,
                shift1,
            );
            let r1234 = q0.shift_quarter(q1);
            let r2345 = q0.combine_mid_halves(q1);
            let r3456 = r2345.shift_quarter(q1.combine_mid_halves(q1));
            let mut tmp = [0i16; 16];
            vert_regs_4_pre([q0, r1234, r2345, r3456], &ty, scale).store(&mut tmp);
            let rem = (h - oy) * 4;
            out[oy * 4..oy * 4 + rem].copy_from_slice(&tmp[..rem]);
        }
    }
}

#[inline(always)]
fn two_pass_w4<
    const N: usize,
    S: SimdSample,
    M: I16FilterLanes + I16Pairs,
    V: I16FilterLanes + I16Pairs,
    H: I16Lanes,
>(
    h: usize,
    source: &PlaneView<'_, S>,
    (cx, cy): (&[i16; N], &[i16; N]),
    shift1: u32,
    scale: OutScale,
    out: &mut [i16],
) {
    if N == 4 && M::LANES == 16 && h >= 2 {
        two_pass_w4_n4_regs::<N, S, M>(h, source, (cx, cy), shift1, scale, out);
        return;
    }
    let six_x = N == 8 && cx[0] == 0 && cx[7] == 0;
    let six_tap = N == 8 && cy[0] == 0 && cy[7] == 0;
    let y_off = usize::from(six_tap);
    let rows = h + N - 1 - 2 * y_off;
    let stride = source.stride();
    let src_base = source.data()[..(y_off + rows - 1) * stride + 4 + N - 1]
        .as_ptr()
        .wrapping_add(y_off * stride);
    let (tx_m, tx_v) = (splat_horiz_pairs::<N, M>(cx), splat_horiz_pairs::<N, V>(cx));
    let mut scratch = [MaybeUninit::<i16>::uninit(); 39 * 4];
    // SAFETY: `h <= 32` so `rows <= 39`; every element in `0..rows * 4` is written
    // in the horizontal loop below before the vertical pass reads it.
    let horiz =
        unsafe { core::slice::from_raw_parts_mut(scratch.as_mut_ptr().cast::<i16>(), rows * 4) };
    let mut y = 0;
    // SAFETY: `src_base` covers `rows` rows of `4 + N - 1` samples; `horiz` holds `rows * 4`.
    unsafe {
        while M::LANES == 16 && y + 4 <= rows {
            let p = src_base.add(y * stride);
            let r = [p, p.add(stride), p.add(2 * stride), p.add(3 * stride)];
            horiz_ptr_4rows::<N, S, M>(r, six_x, &tx_m, cx, shift1).store(&mut horiz[y * 4..]);
            y += 4;
        }
        while V::LANES == 8 && y + 2 <= rows {
            let p = src_base.add(y * stride);
            horiz_ptr_2rows::<N, S, V>(p, p.add(stride), six_x, &tx_v, cx, shift1)
                .store(&mut horiz[y * 4..]);
            y += 2;
        }
        while y < rows {
            let p = src_base.add(y * stride);
            horiz_ptr_acc::<N, S, H>(p, six_x, cx, 0, shift1).store(&mut horiz[y * 4..]);
            y += 1;
        }
        let hp = horiz.as_ptr().cast::<u16>();
        let mut oy = 0;
        while M::LANES == 16 && oy + 4 <= h {
            vert_ptr_vec::<N, u16, M>(hp.add(oy * 4), 4, six_tap, cy, scale)
                .store(&mut out[oy * 4..]);
            oy += 4;
        }
        while V::LANES == 8 && oy + 2 <= h {
            vert_ptr_vec::<N, u16, V>(hp.add(oy * 4), 4, six_tap, cy, scale)
                .store(&mut out[oy * 4..]);
            oy += 2;
        }
        while oy < h {
            vert_ptr_vec::<N, u16, H>(hp.add(oy * 4), 4, six_tap, cy, scale)
                .store(&mut out[oy * 4..]);
            oy += 1;
        }
    }
}

#[inline(always)]
unsafe fn w8_regs_n4<const N: usize, S: SimdSample, M: I16FilterLanes + I16Pairs>(
    h: usize,
    (src, stride, last): (*const S, usize, usize),
    six_x: bool,
    (tx, cx, ty): (&[M; 4], &[i16; N], &[[M; 2]; 4]),
    shift1: u32,
    scale: OutScale,
    out: &mut [i16],
) {
    // SAFETY: caller validated `src` up to row `last` (`8 + N - 1` cols).
    unsafe {
        let mut p01 = horiz_ptr_2rows::<N, S, M>(src, src.add(stride), six_x, tx, cx, shift1);
        let mut p23 = horiz_ptr_2rows::<N, S, M>(
            src.add(2 * stride),
            src.add(3 * stride),
            six_x,
            tx,
            cx,
            shift1,
        );
        let mut oy = 0;
        while oy + 2 <= h {
            let p45 = horiz_ptr_2rows::<N, S, M>(
                src.add((oy + 4) * stride),
                src.add((oy + 5).min(last) * stride),
                six_x,
                tx,
                cx,
                shift1,
            );
            vert_regs_4_pre(
                [
                    p01,
                    p01.combine_mid_halves(p23),
                    p23,
                    p23.combine_mid_halves(p45),
                ],
                ty,
                scale,
            )
            .store(&mut out[oy * 8..]);
            (p01, p23) = (p23, p45);
            oy += 2;
        }
        if oy < h {
            let mut tmp = [0i16; 16];
            vert_regs_4_pre(
                [
                    p01,
                    p01.combine_mid_halves(p23),
                    p23,
                    p23.combine_mid_halves(p23),
                ],
                ty,
                scale,
            )
            .store(&mut tmp);
            out[oy * 8..oy * 8 + 8].copy_from_slice(&tmp[..8]);
        }
    }
}

#[inline(always)]
unsafe fn w8_regs_n6<const N: usize, S: SimdSample, M: I16FilterLanes + I16Pairs>(
    h: usize,
    (src, stride, last): (*const S, usize, usize),
    six_x: bool,
    (tx, cx, ty): (&[M; 4], &[i16; N], &[[M; 2]; 4]),
    shift1: u32,
    scale: OutScale,
    out: &mut [i16],
) {
    // SAFETY: caller validated `src` up to row `last` (`8 + N - 1` cols).
    unsafe {
        let mut p01 =
            horiz_ptr_2rows::<N, S, M>(src.add(stride), src.add(2 * stride), six_x, tx, cx, shift1);
        let mut p23 = horiz_ptr_2rows::<N, S, M>(
            src.add(3 * stride),
            src.add(4 * stride),
            six_x,
            tx,
            cx,
            shift1,
        );
        let mut p45 = horiz_ptr_2rows::<N, S, M>(
            src.add(5 * stride),
            src.add(6 * stride),
            six_x,
            tx,
            cx,
            shift1,
        );
        let mut oy = 0;
        while oy + 2 <= h {
            let p67 = horiz_ptr_2rows::<N, S, M>(
                src.add((oy + 7) * stride),
                src.add((oy + 8).min(last) * stride),
                six_x,
                tx,
                cx,
                shift1,
            );
            vert_regs_6_pre(
                [
                    p01,
                    p01.combine_mid_halves(p23),
                    p23,
                    p23.combine_mid_halves(p45),
                    p45,
                    p45.combine_mid_halves(p67),
                ],
                ty,
                scale,
            )
            .store(&mut out[oy * 8..]);
            (p01, p23, p45) = (p23, p45, p67);
            oy += 2;
        }
        if oy < h {
            let mut tmp = [0i16; 16];
            vert_regs_6_pre(
                [
                    p01,
                    p01.combine_mid_halves(p23),
                    p23,
                    p23.combine_mid_halves(p45),
                    p45,
                    p45.combine_mid_halves(p45),
                ],
                ty,
                scale,
            )
            .store(&mut tmp);
            out[oy * 8..oy * 8 + 8].copy_from_slice(&tmp[..8]);
        }
    }
}

#[inline(always)]
unsafe fn w8_regs_n8<const N: usize, S: SimdSample, M: I16FilterLanes + I16Pairs>(
    h: usize,
    (src, stride, last): (*const S, usize, usize),
    six_x: bool,
    (tx, cx, ty): (&[M; 4], &[i16; N], &[[M; 2]; 4]),
    shift1: u32,
    scale: OutScale,
    out: &mut [i16],
) {
    // SAFETY: caller validated `src` up to row `last` (`8 + N - 1` cols).
    unsafe {
        let mut p01 = horiz_ptr_2rows::<N, S, M>(src, src.add(stride), six_x, tx, cx, shift1);
        let mut p23 = horiz_ptr_2rows::<N, S, M>(
            src.add(2 * stride),
            src.add(3 * stride),
            six_x,
            tx,
            cx,
            shift1,
        );
        let mut p45 = horiz_ptr_2rows::<N, S, M>(
            src.add(4 * stride),
            src.add(5 * stride),
            six_x,
            tx,
            cx,
            shift1,
        );
        let mut p67 = horiz_ptr_2rows::<N, S, M>(
            src.add(6 * stride),
            src.add(7 * stride),
            six_x,
            tx,
            cx,
            shift1,
        );
        let mut oy = 0;
        while oy + 2 <= h {
            let p89 = horiz_ptr_2rows::<N, S, M>(
                src.add((oy + 8) * stride),
                src.add((oy + 9).min(last) * stride),
                six_x,
                tx,
                cx,
                shift1,
            );
            vert_regs_8_pre(
                [
                    p01,
                    p01.combine_mid_halves(p23),
                    p23,
                    p23.combine_mid_halves(p45),
                    p45,
                    p45.combine_mid_halves(p67),
                    p67,
                    p67.combine_mid_halves(p89),
                ],
                ty,
                scale,
            )
            .store(&mut out[oy * 8..]);
            (p01, p23, p45, p67) = (p23, p45, p67, p89);
            oy += 2;
        }
        if oy < h {
            let mut tmp = [0i16; 16];
            vert_regs_8_pre(
                [
                    p01,
                    p01.combine_mid_halves(p23),
                    p23,
                    p23.combine_mid_halves(p45),
                    p45,
                    p45.combine_mid_halves(p67),
                    p67,
                    p67.combine_mid_halves(p67),
                ],
                ty,
                scale,
            )
            .store(&mut tmp);
            out[oy * 8..oy * 8 + 8].copy_from_slice(&tmp[..8]);
        }
    }
}

#[inline(always)]
fn two_pass_w8_regs<const N: usize, S: SimdSample, M: I16FilterLanes + I16Pairs>(
    h: usize,
    source: &PlaneView<'_, S>,
    (cx, cy): (&[i16; N], &[i16; N]),
    shift1: u32,
    scale: OutScale,
    out: &mut [i16],
) {
    let six_x = N == 8 && cx[0] == 0 && cx[7] == 0;
    let six_tap = N == 8 && cy[0] == 0 && cy[7] == 0;
    let last = if N == 4 {
        h + 2
    } else if six_tap {
        h + 5
    } else {
        h + 6
    };
    let stride = source.stride();
    let src = source.data()[..last * stride + 8 + N - 1].as_ptr();
    let tx = splat_horiz_pairs::<N, M>(cx);
    let ty = splat_vert_pairs::<N, M>(cy);
    // SAFETY: `src` bounds-checked up to row `last` (`8 + N - 1` cols) once above.
    unsafe {
        if N == 4 {
            w8_regs_n4::<N, S, M>(
                h,
                (src, stride, last),
                six_x,
                (&tx, cx, &ty),
                shift1,
                scale,
                out,
            );
        } else if six_tap {
            w8_regs_n6::<N, S, M>(
                h,
                (src, stride, last),
                six_x,
                (&tx, cx, &ty),
                shift1,
                scale,
                out,
            );
        } else {
            w8_regs_n8::<N, S, M>(
                h,
                (src, stride, last),
                six_x,
                (&tx, cx, &ty),
                shift1,
                scale,
                out,
            );
        }
    }
}

#[inline(always)]
fn two_pass_w8<
    const N: usize,
    S: SimdSample,
    W: I16FilterLanes + I16Pairs,
    M: I16FilterLanes + I16Pairs,
    V: I16FilterLanes + I16Pairs,
>(
    h: usize,
    source: &PlaneView<'_, S>,
    (cx, cy): (&[i16; N], &[i16; N]),
    shift1: u32,
    scale: OutScale,
    out: &mut [i16],
) {
    if M::LANES == 16 && W::LANES == 16 && h >= 2 {
        two_pass_w8_regs::<N, S, M>(h, source, (cx, cy), shift1, scale, out);
        return;
    }
    let six_x = N == 8 && cx[0] == 0 && cx[7] == 0;
    let six_tap = N == 8 && cy[0] == 0 && cy[7] == 0;
    let y_off = usize::from(six_tap);
    let rows = h + N - 1 - 2 * y_off;
    let stride = source.stride();
    let src_base = source.data()[..(y_off + rows - 1) * stride + 8 + N - 1]
        .as_ptr()
        .wrapping_add(y_off * stride);
    let (tx_w, tx_m, tx_v) = (
        splat_horiz_pairs::<N, W>(cx),
        splat_horiz_pairs::<N, M>(cx),
        splat_horiz_pairs::<N, V>(cx),
    );
    let mut scratch = [MaybeUninit::<i16>::uninit(); 39 * 8];
    // SAFETY: `h <= 32` so `rows <= 39`; every element in `0..rows * 8` is written
    // in the horizontal loop below before the vertical pass reads it.
    let horiz =
        unsafe { core::slice::from_raw_parts_mut(scratch.as_mut_ptr().cast::<i16>(), rows * 8) };
    let mut y = 0;
    // SAFETY: `src_base` covers `rows` rows of `8 + N - 1` samples; `horiz` holds `rows * 8`.
    unsafe {
        while W::LANES == 32 && y + 4 <= rows {
            let p = src_base.add(y * stride);
            let r = [p, p.add(stride), p.add(2 * stride), p.add(3 * stride)];
            horiz_ptr_4rows::<N, S, W>(r, six_x, &tx_w, cx, shift1).store(&mut horiz[y * 8..]);
            y += 4;
        }
        while M::LANES == 16 && y + 2 <= rows {
            let p = src_base.add(y * stride);
            horiz_ptr_2rows::<N, S, M>(p, p.add(stride), six_x, &tx_m, cx, shift1)
                .store(&mut horiz[y * 8..]);
            y += 2;
        }
        while y < rows {
            let p = src_base.add(y * stride);
            horiz_ptr_pair_round::<N, S, V>(p, six_x, &tx_v, cx, 0, shift1)
                .store(&mut horiz[y * 8..]);
            y += 1;
        }
        let hp = horiz.as_ptr().cast::<u16>();
        let mut oy = 0;
        while W::LANES == 32 && oy + 4 <= h {
            vert_ptr_vec::<N, u16, W>(hp.add(oy * 8), 8, six_tap, cy, scale)
                .store(&mut out[oy * 8..]);
            oy += 4;
        }
        while M::LANES == 16 && oy + 2 <= h {
            vert_ptr_vec::<N, u16, M>(hp.add(oy * 8), 8, six_tap, cy, scale)
                .store(&mut out[oy * 8..]);
            oy += 2;
        }
        while oy < h {
            vert_ptr_vec::<N, u16, V>(hp.add(oy * 8), 8, six_tap, cy, scale)
                .store(&mut out[oy * 8..]);
            oy += 1;
        }
    }
}

#[inline(always)]
fn two_pass_small_fallback<
    const N: usize,
    S: SimdSample,
    W: I16Pairs,
    M: I16Pairs,
    V: I16Pairs,
    H: I16Lanes,
>(
    (w, h): (usize, usize),
    source: &PlaneView<'_, S>,
    (cx, cy): (&[i16; N], &[i16; N]),
    shift1: u32,
    scale: OutScale,
    out: &mut [i16],
) {
    let six_tap = N == 8 && cy[0] == 0 && cy[7] == 0;
    let y_off = usize::from(six_tap);
    let active = if N == 4 {
        4
    } else if six_tap {
        6
    } else {
        8
    };
    let mut ring = [[0i16; 8]; 8];
    for (k, slot) in ring.iter_mut().enumerate().take(active - 1) {
        horiz_row::<N, S, W, M, V, H>(source.row(y_off + k), cx, shift1, &mut slot[..w]);
    }
    for (y, dst) in out.chunks_exact_mut(w).take(h).enumerate() {
        let new_idx = (y + active - 1) & 7;
        horiz_row::<N, S, W, M, V, H>(
            source.row(y_off + y + active - 1),
            cx,
            shift1,
            &mut ring[new_idx][..w],
        );
        for (x, out_x) in dst.iter_mut().enumerate() {
            let mut sum = 0i32;
            for k in 0..active {
                sum += i32::from(cy[y_off + k]) * i32::from(ring[(y + k) & 7][x]);
            }
            *out_x = (((sum + scale.0) >> scale.1) as i16).wrapping_add(scale.2);
        }
    }
}

#[inline(always)]
fn two_pass_lanes_of<
    T: SeparableTaps<N>,
    const N: usize,
    S: SimdSample,
    W: I16FilterLanes + I16Pairs,
    M: I16FilterLanes + I16Pairs,
    V: I16FilterLanes + I16Pairs,
    H: I16Lanes,
>(
    (w, h): (usize, usize),
    source: &PlaneView<'_, S>,
    (fx, fy): (usize, usize),
    shift1: u32,
    scale: OutScale,
    out: &mut [i16],
) {
    let cx = &T::TABLE[fx];
    let cy = &T::TABLE[fy];
    if w == 2 && N == 4 && V::LANES == 8 && h >= 2 {
        let cx4: &[i16; 4] = cx.as_slice().try_into().unwrap();
        let cy4: &[i16; 4] = cy.as_slice().try_into().unwrap();
        two_pass_w2::<S, V>(h, source, (cx4, cy4), shift1, scale, out);
        return;
    }
    if w == 4 && h <= 32 {
        two_pass_w4::<N, S, M, V, H>(h, source, (cx, cy), shift1, scale, out);
        return;
    }
    if w == 8 && V::LANES == 8 && W::LANES > 8 && h <= 32 {
        two_pass_w8::<N, S, W, M, V>(h, source, (cx, cy), shift1, scale, out);
        return;
    }
    if w >= V::LANES {
        let mut sx = 0;
        if w >= W::LANES {
            let tx_w = splat_horiz_pairs::<N, W>(cx);
            let ty_w = splat_vert_pairs::<N, W>(cy);
            while sx + W::LANES <= w {
                two_pass_strip_1vec::<N, S, W>(
                    (w, h, sx),
                    source,
                    (cx, cy),
                    (&tx_w, &ty_w),
                    shift1,
                    scale,
                    out,
                );
                sx += W::LANES;
            }
        }
        if M::LANES < W::LANES && sx + M::LANES <= w {
            let tx_m = splat_horiz_pairs::<N, M>(cx);
            let ty_m = splat_vert_pairs::<N, M>(cy);
            two_pass_strip_1vec::<N, S, M>(
                (w, h, sx),
                source,
                (cx, cy),
                (&tx_m, &ty_m),
                shift1,
                scale,
                out,
            );
            sx += M::LANES;
        }
        if sx < w {
            let tx_v = splat_horiz_pairs::<N, V>(cx);
            let ty_v = splat_vert_pairs::<N, V>(cy);
            if sx + V::LANES <= w {
                two_pass_strip_1vec::<N, S, V>(
                    (w, h, sx),
                    source,
                    (cx, cy),
                    (&tx_v, &ty_v),
                    shift1,
                    scale,
                    out,
                );
                sx += V::LANES;
            }
            if sx < w {
                two_pass_strip_1vec::<N, S, V>(
                    (w, h, w - V::LANES),
                    source,
                    (cx, cy),
                    (&tx_v, &ty_v),
                    shift1,
                    scale,
                    out,
                );
            }
        }
        return;
    }
    two_pass_small_fallback::<N, S, W, M, V, H>((w, h), source, (cx, cy), shift1, scale, out);
}

#[inline(always)]
pub(super) fn two_pass_lanes<
    T: SeparableTaps<N>,
    const N: usize,
    W: I16FilterLanes + I16Pairs,
    M: I16FilterLanes + I16Pairs,
    V: I16FilterLanes + I16Pairs,
    H: I16Lanes,
>(
    size: (usize, usize),
    source: &PlaneView<'_, u16>,
    phases: (usize, usize),
    shift1: u32,
    scale: OutScale,
    out: &mut [i16],
) {
    two_pass_lanes_of::<T, N, u16, W, M, V, H>(size, source, phases, shift1, scale, out);
}

#[inline(always)]
unsafe fn vert_strip_1vec<const N: usize, S: SimdSample, L: I16FilterLanes>(
    (w, h, x_off): (usize, usize, usize),
    source: &PlaneView<'_, S>,
    before: usize,
    cy: &[i16; N],
    ty: &[[L; 2]; 4],
    scale: OutScale,
    out: &mut [i16],
) {
    if h == 0 {
        return;
    }
    let six_y = N == 8 && cy[0] == 0 && cy[7] == 0;
    let y_off = usize::from(six_y);
    let active = if N == 4 {
        4
    } else if six_y {
        6
    } else {
        8
    };
    let stride = source.stride();
    let req_src = (y_off + h + active - 2) * stride + before + x_off + L::LANES;
    let src_data = &source.data()[..req_src];
    let dst_data = &mut out[..w * h];
    // SAFETY: `src_data` and `dst_data` validated once above.
    unsafe {
        let mut src_p = src_data.as_ptr().add(y_off * stride + before + x_off);
        let mut dst_p = dst_data.as_mut_ptr().add(x_off).cast::<u16>();
        if N == 4 {
            let mut r0 = S::load_lanes::<L>(src_p);
            src_p = src_p.add(stride);
            let mut r1 = S::load_lanes::<L>(src_p);
            src_p = src_p.add(stride);
            let mut r2 = S::load_lanes::<L>(src_p);
            for _ in 0..h {
                src_p = src_p.add(stride);
                let r3 = S::load_lanes::<L>(src_p);
                vert_regs_4_pre([r0, r1, r2, r3], ty, scale).store_u16_at(dst_p);
                dst_p = dst_p.add(w);
                (r0, r1, r2) = (r1, r2, r3);
            }
        } else if six_y {
            let mut r1 = S::load_lanes::<L>(src_p);
            src_p = src_p.add(stride);
            let mut r2 = S::load_lanes::<L>(src_p);
            src_p = src_p.add(stride);
            let mut r3 = S::load_lanes::<L>(src_p);
            src_p = src_p.add(stride);
            let mut r4 = S::load_lanes::<L>(src_p);
            src_p = src_p.add(stride);
            let mut r5 = S::load_lanes::<L>(src_p);
            for _ in 0..h {
                src_p = src_p.add(stride);
                let r6 = S::load_lanes::<L>(src_p);
                vert_regs_6_pre([r1, r2, r3, r4, r5, r6], ty, scale).store_u16_at(dst_p);
                dst_p = dst_p.add(w);
                (r1, r2, r3, r4, r5) = (r2, r3, r4, r5, r6);
            }
        } else {
            let mut r0 = S::load_lanes::<L>(src_p);
            src_p = src_p.add(stride);
            let mut r1 = S::load_lanes::<L>(src_p);
            src_p = src_p.add(stride);
            let mut r2 = S::load_lanes::<L>(src_p);
            src_p = src_p.add(stride);
            let mut r3 = S::load_lanes::<L>(src_p);
            src_p = src_p.add(stride);
            let mut r4 = S::load_lanes::<L>(src_p);
            src_p = src_p.add(stride);
            let mut r5 = S::load_lanes::<L>(src_p);
            src_p = src_p.add(stride);
            let mut r6 = S::load_lanes::<L>(src_p);
            for _ in 0..h {
                src_p = src_p.add(stride);
                let r7 = S::load_lanes::<L>(src_p);
                vert_regs_8_pre([r0, r1, r2, r3, r4, r5, r6, r7], ty, scale).store_u16_at(dst_p);
                dst_p = dst_p.add(w);
                (r0, r1, r2, r3, r4, r5, r6) = (r1, r2, r3, r4, r5, r6, r7);
            }
        }
    }
}

#[inline(always)]
fn vert_only_lanes_of<
    const N: usize,
    S: SimdSample,
    W: I16FilterLanes,
    M: I16FilterLanes,
    V: I16FilterLanes,
    H: I16Lanes,
>(
    (w, h): (usize, usize),
    source: &PlaneView<'_, S>,
    before: usize,
    cy: &[i16; N],
    scale: OutScale,
    out: &mut [i16],
) {
    if h == 0 || w == 0 {
        return;
    }
    if w >= V::LANES {
        let mut sx = 0;
        // SAFETY: `vert_strip_1vec` checks `source` and `out` bounds before raw loads/stores.
        unsafe {
            if w >= W::LANES {
                let ty_w = splat_vert_pairs::<N, W>(cy);
                while sx + W::LANES <= w {
                    vert_strip_1vec::<N, S, W>((w, h, sx), source, before, cy, &ty_w, scale, out);
                    sx += W::LANES;
                }
            }
            if M::LANES < W::LANES && sx + M::LANES <= w {
                let ty_m = splat_vert_pairs::<N, M>(cy);
                vert_strip_1vec::<N, S, M>((w, h, sx), source, before, cy, &ty_m, scale, out);
                sx += M::LANES;
            }
            if sx < w {
                let ty_v = splat_vert_pairs::<N, V>(cy);
                if sx + V::LANES <= w {
                    vert_strip_1vec::<N, S, V>((w, h, sx), source, before, cy, &ty_v, scale, out);
                    sx += V::LANES;
                }
                if sx < w {
                    vert_strip_1vec::<N, S, V>(
                        (w, h, w - V::LANES),
                        source,
                        before,
                        cy,
                        &ty_v,
                        scale,
                        out,
                    );
                }
            }
        }
        return;
    }
    let six_y = N == 8 && cy[0] == 0 && cy[7] == 0;
    let y_off = usize::from(six_y);
    let stride = source.stride();
    let active = if N == 4 {
        4
    } else if six_y {
        6
    } else {
        8
    };
    let req_src = (y_off + h + active - 2) * stride + before + w;
    let src_data = &source.data()[..req_src];
    if w == 4 {
        // SAFETY: `src_data` validated across `(y_off + h + active - 1)` rows of `before + 4`.
        unsafe {
            let mut src_p = src_data.as_ptr().add(y_off * stride + before);
            for dst in out.as_chunks_mut::<4>().0.iter_mut().take(h) {
                vert_ptr_vec::<N, S, H>(src_p, stride, six_y, cy, scale).store(dst);
                src_p = src_p.add(stride);
            }
        }
        return;
    }
    for (y, dst) in out.chunks_exact_mut(w).take(h).enumerate() {
        for (x, out_x) in dst.iter_mut().enumerate() {
            let mut sum = 0i32;
            for k in 0..active {
                sum += i32::from(cy[y_off + k])
                    * src_data[(y_off + y + k) * stride + before + x].to_i32();
            }
            *out_x = (((sum + scale.0) >> scale.1) as i16).wrapping_add(scale.2);
        }
    }
}

#[inline(always)]
pub(super) fn vert_only_lanes<
    const N: usize,
    W: I16FilterLanes,
    M: I16FilterLanes,
    V: I16FilterLanes,
    H: I16Lanes,
>(
    size: (usize, usize),
    source: &PlaneView<'_, u16>,
    before: usize,
    cy: &[i16; N],
    scale: OutScale,
    out: &mut [i16],
) {
    vert_only_lanes_of::<N, u16, W, M, V, H>(size, source, before, cy, scale, out);
}

#[inline(always)]
fn copy_scaled<S: SimdSample, W: I16Lanes, M: I16Lanes, V: I16Lanes, H: I16Lanes>(
    (w, h): (usize, usize),
    source: &PlaneView<'_, S>,
    before: usize,
    shift3: u32,
    out: &mut [i16],
) {
    if w == 0 || h == 0 {
        return;
    }
    let stride = source.stride();
    let src_data = &source.data()[..(before + h - 1) * stride + before + w];
    let scale = 1i16 << shift3;
    let (sw, sm, sv, sh) = (
        W::splat(scale),
        M::splat(scale),
        V::splat(scale),
        H::splat(scale),
    );
    for (y, dst) in out.chunks_exact_mut(w).take(h).enumerate() {
        let src = &src_data[(y + before) * stride + before..][..w];
        let mut x = 0;
        while x + W::LANES <= w {
            S::load_slice::<W>(&src[x..]).mul(sw).store(&mut dst[x..]);
            x += W::LANES;
        }
        if M::LANES < W::LANES && x + M::LANES <= w {
            S::load_slice::<M>(&src[x..]).mul(sm).store(&mut dst[x..]);
            x += M::LANES;
        }
        if V::LANES < M::LANES && x + V::LANES <= w {
            S::load_slice::<V>(&src[x..]).mul(sv).store(&mut dst[x..]);
            x += V::LANES;
        }
        if x + H::LANES <= w {
            S::load_slice::<H>(&src[x..]).mul(sh).store(&mut dst[x..]);
            x += H::LANES;
        }
        if x < w && w >= H::LANES {
            let tail = w - H::LANES;
            S::load_slice::<H>(&src[tail..])
                .mul(sh)
                .store(&mut dst[tail..]);
            continue;
        }
        while x < w {
            dst[x] = (src[x].to_i32() << shift3) as i16;
            x += 1;
        }
    }
}

#[inline(always)]
fn horiz_only_lanes<
    const N: usize,
    S: SimdSample,
    W: I16Pairs,
    M: I16Pairs,
    V: I16Pairs,
    H: I16Lanes,
>(
    (w, h): (usize, usize),
    source: &PlaneView<'_, S>,
    before: usize,
    cx: &[i16; N],
    shift1: u32,
    out: &mut [i16],
) {
    if w == 0 || h == 0 {
        return;
    }
    if w == 2 && N == 4 && V::LANES == 8 && h >= 4 {
        let cx4: &[i16; 4] = cx.as_slice().try_into().unwrap();
        let mut y = 0;
        while y + 4 <= h {
            horiz_w2x4::<S, V>(
                [
                    source.row(y + before),
                    source.row(y + before + 1),
                    source.row(y + before + 2),
                    source.row(y + before + 3),
                ],
                cx4,
                shift1,
            )
            .store(&mut out[y * 2..]);
            y += 4;
        }
        for (dy, dst) in out[y * 2..].as_chunks_mut::<2>().0.iter_mut().enumerate() {
            horiz_row::<N, S, W, M, V, H>(source.row(y + dy + before), cx, shift1, dst);
        }
        return;
    }
    let six_x = N == 8 && cx[0] == 0 && cx[7] == 0;
    let stride = source.stride();
    let req_src = (before + h - 1) * stride + w + N - 1;
    let src_data = &source.data()[..req_src];
    let (tx_w, tx_m, tx_v) = (
        splat_horiz_pairs::<N, W>(cx),
        splat_horiz_pairs::<N, M>(cx),
        splat_horiz_pairs::<N, V>(cx),
    );
    // SAFETY: `src_data` validated across `(before + h)` rows of `w + N - 1` samples.
    unsafe {
        let mut row_p = src_data.as_ptr().add(before * stride);
        for dst in out.chunks_exact_mut(w).take(h) {
            let mut x = 0;
            while x + W::LANES <= w {
                horiz_ptr_pair_round::<N, S, W>(row_p.add(x), six_x, &tx_w, cx, 0, shift1)
                    .store(&mut dst[x..]);
                x += W::LANES;
            }
            if M::LANES < W::LANES && x + M::LANES <= w {
                horiz_ptr_pair_round::<N, S, M>(row_p.add(x), six_x, &tx_m, cx, 0, shift1)
                    .store(&mut dst[x..]);
                x += M::LANES;
            }
            if V::LANES < M::LANES && x + V::LANES <= w {
                horiz_ptr_pair_round::<N, S, V>(row_p.add(x), six_x, &tx_v, cx, 0, shift1)
                    .store(&mut dst[x..]);
                x += V::LANES;
            }
            if x + H::LANES <= w {
                horiz_ptr_acc::<N, S, H>(row_p.add(x), six_x, cx, 0, shift1).store(&mut dst[x..]);
                x += H::LANES;
            }
            if x < w && w >= H::LANES {
                let tail = w - H::LANES;
                horiz_ptr_acc::<N, S, H>(row_p.add(tail), six_x, cx, 0, shift1)
                    .store(&mut dst[tail..]);
            } else {
                while x < w {
                    let mut sum = 0i32;
                    for (k, &tap) in cx.iter().enumerate() {
                        sum += i32::from(tap) * (*row_p.add(x + k)).to_i32();
                    }
                    dst[x] = (sum >> shift1) as i16;
                    x += 1;
                }
            }
            row_p = row_p.add(stride);
        }
    }
}

#[inline(always)]
pub(super) fn interpolate_lanes<
    T: SeparableTaps<N>,
    const N: usize,
    S: SimdSample,
    W: I16FilterLanes + I16Pairs,
    M: I16FilterLanes + I16Pairs,
    V: I16FilterLanes + I16Pairs,
    H: I16Lanes,
>(
    (w, h): (usize, usize),
    source: &PlaneView<'_, S>,
    (fx, fy): (usize, usize),
    out: &mut [i16],
    depth: BitDepth,
) {
    let depth = depth.of::<S>();
    let out = &mut out[..w * h];
    let (shift1, shift3) = interpolation_shifts(depth);
    debug_assert_eq!(shift1 + shift3, SECOND_PASS_SHIFT);
    let before = N / 2 - 1;
    if fx == 0 && fy == 0 {
        copy_scaled::<S, W, M, V, H>((w, h), source, before, shift3, out);
        return;
    }
    if fy == 0 {
        horiz_only_lanes::<N, S, W, M, V, H>((w, h), source, before, &T::TABLE[fx], shift1, out);
        return;
    }
    if fx == 0 {
        vert_only_lanes_of::<N, S, W, M, V, H>(
            (w, h),
            source,
            before,
            &T::TABLE[fy],
            (0, shift1, 0),
            out,
        );
        return;
    }
    two_pass_lanes_of::<T, N, S, W, M, V, H>(
        (w, h),
        source,
        (fx, fy),
        shift1,
        PRECISION_SCALE,
        out,
    );
}

macro_rules! isa_high_precision {
    ($mod_name:ident, $w:ty, $m:ty, $v:ty, $h:ty, $feature:literal) => {
        pub(crate) mod $mod_name {
            use nc_common::{BitDepth, PlaneView};

            use crate::Filter;
            use crate::simd::SimdSample;

            use super::{
                HalfLumaTaps, HevcChromaTaps, HevcLumaTaps, VvcAffineLumaTaps, VvcChromaTaps,
                VvcLumaTaps, interpolate_lanes,
            };

            macro_rules! filter_fn {
                ($fn_name:ident, $taps:ty, $n:literal) => {
                    #[target_feature(enable = $feature)]
                    unsafe fn $fn_name<S: SimdSample>(
                        size: (usize, usize),
                        window: &PlaneView<'_, S>,
                        phases: (usize, usize),
                        out: &mut [i16],
                        depth: BitDepth,
                    ) {
                        interpolate_lanes::<$taps, $n, S, $w, $m, $v, $h>(
                            size, window, phases, out, depth,
                        );
                    }
                };
            }

            filter_fn!(hevc_luma, HevcLumaTaps, 8);
            filter_fn!(hevc_chroma, HevcChromaTaps, 4);
            filter_fn!(vvc_luma_half, HalfLumaTaps, 8);
            filter_fn!(vvc_luma, VvcLumaTaps, 8);
            filter_fn!(vvc_affine, VvcAffineLumaTaps, 8);
            filter_fn!(vvc_chroma, VvcChromaTaps, 4);

            #[target_feature(enable = $feature)]
            pub(crate) unsafe fn high_precision<S: SimdSample>(
                filter: Filter,
                size: (usize, usize),
                window: &PlaneView<'_, S>,
                phases: (usize, usize),
                alternate_half: bool,
                out: &mut [i16],
                depth: BitDepth,
            ) {
                // SAFETY: caller verified `$feature` and checked window/phases.
                unsafe {
                    match filter {
                        Filter::HevcLuma => hevc_luma(size, window, phases, out, depth),
                        Filter::HevcChroma => hevc_chroma(size, window, phases, out, depth),
                        Filter::VvcLuma if alternate_half => {
                            vvc_luma_half(size, window, phases, out, depth);
                        }
                        Filter::VvcLuma => vvc_luma(size, window, phases, out, depth),
                        Filter::VvcAffineLuma => vvc_affine(size, window, phases, out, depth),
                        Filter::VvcChroma => vvc_chroma(size, window, phases, out, depth),
                        _ => panic!("filter has no six-bit prediction precision"),
                    }
                }
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
isa_high_precision!(
    avx2,
    crate::simd::avx2::I16x16,
    crate::simd::avx2::I16x16,
    crate::simd::avx2::I16x8,
    crate::simd::avx2::I16x4,
    "avx2"
);

#[cfg(target_arch = "x86_64")]
isa_high_precision!(
    avx512,
    crate::simd::avx512_i16::I16x32,
    crate::simd::avx2::I16x16,
    crate::simd::avx2::I16x8,
    crate::simd::avx2::I16x4,
    "avx512f,avx512bw,avx512vl"
);

#[cfg(target_arch = "aarch64")]
isa_high_precision!(
    neon,
    crate::simd::neon::I16x8,
    crate::simd::neon::I16x8,
    crate::simd::neon::I16x8,
    crate::simd::neon::I16x4,
    "neon"
);

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
isa_high_precision!(
    simd128,
    crate::simd::simd128::I16x8,
    crate::simd::simd128::I16x8,
    crate::simd::simd128::I16x8,
    crate::simd::simd128::I16x4,
    "simd128"
);
