//! Multi-tier `u16` SIMD kernels for [`crate::Dsp::predict_high_precision`]
//! (H.266 §8.5.6.3.2, HEVC §8.5.3.3.3.1).
//!
//! At every bit depth `8..=12`, `shift1 = min(4, depth - 8)` scales horizontal
//! sums `(Σ cx[k] * s[x + k]) >> shift1` into `-6143..=22522`, which fits
//! signed `i16` lanes without saturation. In 2-D interpolation, vertical sums
//! `(Σ cy[k] * h[y + k]) >> 6` lie in `-12000..=33247`; subtracting `8192 << 6`
//! before narrowing keeps `-20192..=25055` strictly inside `i16`, and a
//! wrapping `+ 8192` reproduces `(sum >> 6) as i16` bit-for-bit.
//!
//! Single-vector widths (`16` on AVX2, `16`/`32` on AVX-512, `8` on NEON and
//! SIMD128) slide the active 4, 6, or 8 horizontal rows in SIMD registers with
//! zero stack scratch; `4`- and `8`-wide blocks pack 2 or 4 rows per vector;
//! wider blocks stream through an 8-row L1 ring buffer.

#![allow(unsafe_code, clippy::inline_always)]

use core::mem::MaybeUninit;

use nc_common::{BitDepth, PlaneView};

use crate::simd::{I16FilterLanes, I16Lanes};

use super::precision::HalfLumaTaps;
use super::scalar::{SECOND_PASS_SHIFT, interpolation_shifts};
use super::{
    HevcChromaTaps, HevcLumaTaps, SeparableTaps, VvcAffineLumaTaps, VvcChromaTaps, VvcLumaTaps,
};

const BIAS: i16 = 8192;
const BIAS_PRE_SHIFT: i32 = -(8192 << SECOND_PASS_SHIFT);

#[inline(always)]
unsafe fn load_2rows<L: I16FilterLanes>(p0: *const i16, p1: *const i16, x: usize) -> L {
    let half = L::LANES / 2;
    // SAFETY: caller validated `p0` and `p1` cover `half + N - 1` samples.
    unsafe {
        L::load_halves(
            core::slice::from_raw_parts(p0.add(x), half),
            core::slice::from_raw_parts(p1.add(x), half),
        )
    }
}

#[inline(always)]
unsafe fn load_4rows<L: I16FilterLanes>(p: [*const i16; 4], x: usize) -> L {
    let q = L::LANES / 4;
    // SAFETY: caller validated each `p[k]` covers `q + N - 1` samples.
    unsafe {
        L::load_quarters([
            core::slice::from_raw_parts(p[0].add(x), q),
            core::slice::from_raw_parts(p[1].add(x), q),
            core::slice::from_raw_parts(p[2].add(x), q),
            core::slice::from_raw_parts(p[3].add(x), q),
        ])
    }
}

#[inline(always)]
fn horiz_vec<const N: usize, L: I16Lanes>(src: &[u16], x: usize, cx: &[i16; N], shift1: u32) -> L {
    let p = src[x..x + L::LANES + N - 1].as_ptr();
    let mut acc = L::acc_zero();
    // SAFETY: `src[x..x + L::LANES + N - 1]` bounds-checked once above.
    unsafe {
        if N == 4 {
            acc = L::acc_mul_add2(
                acc,
                L::load_u16_at(p),
                cx[0],
                L::load_u16_at(p.add(1)),
                cx[1],
            );
            acc = L::acc_mul_add2(
                acc,
                L::load_u16_at(p.add(2)),
                cx[2],
                L::load_u16_at(p.add(3)),
                cx[3],
            );
        } else if cx[0] == 0 && cx[7] == 0 {
            acc = L::acc_mul_add2(
                acc,
                L::load_u16_at(p.add(1)),
                cx[1],
                L::load_u16_at(p.add(2)),
                cx[2],
            );
            acc = L::acc_mul_add2(
                acc,
                L::load_u16_at(p.add(3)),
                cx[3],
                L::load_u16_at(p.add(4)),
                cx[4],
            );
            acc = L::acc_mul_add2(
                acc,
                L::load_u16_at(p.add(5)),
                cx[5],
                L::load_u16_at(p.add(6)),
                cx[6],
            );
        } else {
            acc = L::acc_mul_add2(
                acc,
                L::load_u16_at(p),
                cx[0],
                L::load_u16_at(p.add(1)),
                cx[1],
            );
            acc = L::acc_mul_add2(
                acc,
                L::load_u16_at(p.add(2)),
                cx[2],
                L::load_u16_at(p.add(3)),
                cx[3],
            );
            acc = L::acc_mul_add2(
                acc,
                L::load_u16_at(p.add(4)),
                cx[4],
                L::load_u16_at(p.add(5)),
                cx[5],
            );
            acc = L::acc_mul_add2(
                acc,
                L::load_u16_at(p.add(6)),
                cx[6],
                L::load_u16_at(p.add(7)),
                cx[7],
            );
        }
    }
    L::acc_round_shift_narrow(acc, 0, shift1)
}

#[inline(always)]
fn horiz_2rows<const N: usize, L: I16FilterLanes>(
    r0: &[u16],
    r1: &[u16],
    cx: &[i16; N],
    shift1: u32,
) -> L {
    let span = L::LANES / 2 + N - 1;
    let (p0, p1) = (
        r0[..span].as_ptr().cast::<i16>(),
        r1[..span].as_ptr().cast::<i16>(),
    );
    let mut acc = L::acc_zero();
    // SAFETY: `r0[..span]` and `r1[..span]` bounds-checked once above.
    unsafe {
        if N == 4 {
            acc = L::acc_mul_add2(
                acc,
                load_2rows(p0, p1, 0),
                cx[0],
                load_2rows(p0, p1, 1),
                cx[1],
            );
            acc = L::acc_mul_add2(
                acc,
                load_2rows(p0, p1, 2),
                cx[2],
                load_2rows(p0, p1, 3),
                cx[3],
            );
        } else if cx[0] == 0 && cx[7] == 0 {
            acc = L::acc_mul_add2(
                acc,
                load_2rows(p0, p1, 1),
                cx[1],
                load_2rows(p0, p1, 2),
                cx[2],
            );
            acc = L::acc_mul_add2(
                acc,
                load_2rows(p0, p1, 3),
                cx[3],
                load_2rows(p0, p1, 4),
                cx[4],
            );
            acc = L::acc_mul_add2(
                acc,
                load_2rows(p0, p1, 5),
                cx[5],
                load_2rows(p0, p1, 6),
                cx[6],
            );
        } else {
            acc = L::acc_mul_add2(
                acc,
                load_2rows(p0, p1, 0),
                cx[0],
                load_2rows(p0, p1, 1),
                cx[1],
            );
            acc = L::acc_mul_add2(
                acc,
                load_2rows(p0, p1, 2),
                cx[2],
                load_2rows(p0, p1, 3),
                cx[3],
            );
            acc = L::acc_mul_add2(
                acc,
                load_2rows(p0, p1, 4),
                cx[4],
                load_2rows(p0, p1, 5),
                cx[5],
            );
            acc = L::acc_mul_add2(
                acc,
                load_2rows(p0, p1, 6),
                cx[6],
                load_2rows(p0, p1, 7),
                cx[7],
            );
        }
    }
    L::acc_round_shift_narrow(acc, 0, shift1)
}

#[inline(always)]
fn horiz_4rows<const N: usize, L: I16FilterLanes>(
    rows: [&[u16]; 4],
    cx: &[i16; N],
    shift1: u32,
) -> L {
    let span = L::LANES / 4 + N - 1;
    let r = [
        rows[0][..span].as_ptr().cast::<i16>(),
        rows[1][..span].as_ptr().cast::<i16>(),
        rows[2][..span].as_ptr().cast::<i16>(),
        rows[3][..span].as_ptr().cast::<i16>(),
    ];
    let mut acc = L::acc_zero();
    // SAFETY: each `rows[k][..span]` bounds-checked once above.
    unsafe {
        if N == 4 {
            acc = L::acc_mul_add2(acc, load_4rows(r, 0), cx[0], load_4rows(r, 1), cx[1]);
            acc = L::acc_mul_add2(acc, load_4rows(r, 2), cx[2], load_4rows(r, 3), cx[3]);
        } else if cx[0] == 0 && cx[7] == 0 {
            acc = L::acc_mul_add2(acc, load_4rows(r, 1), cx[1], load_4rows(r, 2), cx[2]);
            acc = L::acc_mul_add2(acc, load_4rows(r, 3), cx[3], load_4rows(r, 4), cx[4]);
            acc = L::acc_mul_add2(acc, load_4rows(r, 5), cx[5], load_4rows(r, 6), cx[6]);
        } else {
            acc = L::acc_mul_add2(acc, load_4rows(r, 0), cx[0], load_4rows(r, 1), cx[1]);
            acc = L::acc_mul_add2(acc, load_4rows(r, 2), cx[2], load_4rows(r, 3), cx[3]);
            acc = L::acc_mul_add2(acc, load_4rows(r, 4), cx[4], load_4rows(r, 5), cx[5]);
            acc = L::acc_mul_add2(acc, load_4rows(r, 6), cx[6], load_4rows(r, 7), cx[7]);
        }
    }
    L::acc_round_shift_narrow(acc, 0, shift1)
}

#[inline(always)]
fn vert_u16_vec<const N: usize, L: I16Lanes>(
    rows: &[&[u16]; N],
    x: usize,
    cy: &[i16; N],
    shift1: u32,
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
    L::acc_round_shift_narrow(acc, 0, shift1)
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
unsafe fn vert_ptr_vec<const N: usize, L: I16Lanes>(
    base: *const u16,
    w: usize,
    six_tap: bool,
    cy: &[i16; N],
) -> L {
    let mut acc = L::acc_zero();
    // SAFETY: caller validated `horiz` holds `rows * w` initialized `i16`s.
    unsafe {
        if N == 4 {
            acc = L::acc_mul_add2(
                acc,
                L::load_u16_at(base),
                cy[0],
                L::load_u16_at(base.add(w)),
                cy[1],
            );
            acc = L::acc_mul_add2(
                acc,
                L::load_u16_at(base.add(2 * w)),
                cy[2],
                L::load_u16_at(base.add(3 * w)),
                cy[3],
            );
        } else if six_tap {
            acc = L::acc_mul_add2(
                acc,
                L::load_u16_at(base),
                cy[1],
                L::load_u16_at(base.add(w)),
                cy[2],
            );
            acc = L::acc_mul_add2(
                acc,
                L::load_u16_at(base.add(2 * w)),
                cy[3],
                L::load_u16_at(base.add(3 * w)),
                cy[4],
            );
            acc = L::acc_mul_add2(
                acc,
                L::load_u16_at(base.add(4 * w)),
                cy[5],
                L::load_u16_at(base.add(5 * w)),
                cy[6],
            );
        } else {
            acc = L::acc_mul_add2(
                acc,
                L::load_u16_at(base),
                cy[0],
                L::load_u16_at(base.add(w)),
                cy[1],
            );
            acc = L::acc_mul_add2(
                acc,
                L::load_u16_at(base.add(2 * w)),
                cy[2],
                L::load_u16_at(base.add(3 * w)),
                cy[3],
            );
            acc = L::acc_mul_add2(
                acc,
                L::load_u16_at(base.add(4 * w)),
                cy[4],
                L::load_u16_at(base.add(5 * w)),
                cy[5],
            );
            acc = L::acc_mul_add2(
                acc,
                L::load_u16_at(base.add(6 * w)),
                cy[6],
                L::load_u16_at(base.add(7 * w)),
                cy[7],
            );
        }
    }
    L::acc_round_shift_narrow(acc, BIAS_PRE_SHIFT, SECOND_PASS_SHIFT).add(L::splat(BIAS))
}

#[inline(always)]
unsafe fn vert_ring_vec<const N: usize, L: I16Lanes>(
    r: &[*const u16; 8],
    x: usize,
    six_tap: bool,
    cy: &[i16; N],
) -> L {
    let mut acc = L::acc_zero();
    // SAFETY: caller initialized `active` ring rows across `0..w` (`x + LANES <= w`).
    unsafe {
        if N == 4 {
            acc = L::acc_mul_add2(
                acc,
                L::load_u16_at(r[0].add(x)),
                cy[0],
                L::load_u16_at(r[1].add(x)),
                cy[1],
            );
            acc = L::acc_mul_add2(
                acc,
                L::load_u16_at(r[2].add(x)),
                cy[2],
                L::load_u16_at(r[3].add(x)),
                cy[3],
            );
        } else if six_tap {
            acc = L::acc_mul_add2(
                acc,
                L::load_u16_at(r[0].add(x)),
                cy[1],
                L::load_u16_at(r[1].add(x)),
                cy[2],
            );
            acc = L::acc_mul_add2(
                acc,
                L::load_u16_at(r[2].add(x)),
                cy[3],
                L::load_u16_at(r[3].add(x)),
                cy[4],
            );
            acc = L::acc_mul_add2(
                acc,
                L::load_u16_at(r[4].add(x)),
                cy[5],
                L::load_u16_at(r[5].add(x)),
                cy[6],
            );
        } else {
            acc = L::acc_mul_add2(
                acc,
                L::load_u16_at(r[0].add(x)),
                cy[0],
                L::load_u16_at(r[1].add(x)),
                cy[1],
            );
            acc = L::acc_mul_add2(
                acc,
                L::load_u16_at(r[2].add(x)),
                cy[2],
                L::load_u16_at(r[3].add(x)),
                cy[3],
            );
            acc = L::acc_mul_add2(
                acc,
                L::load_u16_at(r[4].add(x)),
                cy[4],
                L::load_u16_at(r[5].add(x)),
                cy[5],
            );
            acc = L::acc_mul_add2(
                acc,
                L::load_u16_at(r[6].add(x)),
                cy[6],
                L::load_u16_at(r[7].add(x)),
                cy[7],
            );
        }
    }
    L::acc_round_shift_narrow(acc, BIAS_PRE_SHIFT, SECOND_PASS_SHIFT).add(L::splat(BIAS))
}

#[inline(always)]
fn vert_regs_4<const N: usize, L: I16Lanes>(r: [L; 4], cy: &[i16; N]) -> L {
    let mut acc = L::acc_zero();
    acc = L::acc_mul_add2(acc, r[0], cy[0], r[1], cy[1]);
    acc = L::acc_mul_add2(acc, r[2], cy[2], r[3], cy[3]);
    L::acc_round_shift_narrow(acc, BIAS_PRE_SHIFT, SECOND_PASS_SHIFT).add(L::splat(BIAS))
}

#[inline(always)]
fn vert_regs_6<const N: usize, L: I16Lanes>(r: [L; 6], cy: &[i16; N]) -> L {
    let mut acc = L::acc_zero();
    acc = L::acc_mul_add2(acc, r[0], cy[1], r[1], cy[2]);
    acc = L::acc_mul_add2(acc, r[2], cy[3], r[3], cy[4]);
    acc = L::acc_mul_add2(acc, r[4], cy[5], r[5], cy[6]);
    L::acc_round_shift_narrow(acc, BIAS_PRE_SHIFT, SECOND_PASS_SHIFT).add(L::splat(BIAS))
}

#[inline(always)]
fn vert_regs_8<const N: usize, L: I16Lanes>(r: [L; 8], cy: &[i16; N]) -> L {
    let mut acc = L::acc_zero();
    acc = L::acc_mul_add2(acc, r[0], cy[0], r[1], cy[1]);
    acc = L::acc_mul_add2(acc, r[2], cy[2], r[3], cy[3]);
    acc = L::acc_mul_add2(acc, r[4], cy[4], r[5], cy[5]);
    acc = L::acc_mul_add2(acc, r[6], cy[6], r[7], cy[7]);
    L::acc_round_shift_narrow(acc, BIAS_PRE_SHIFT, SECOND_PASS_SHIFT).add(L::splat(BIAS))
}

/// Zero-stack register-sliding 2-D interpolation when `w == L::LANES`.
#[inline(always)]
fn two_pass_1vec<const N: usize, L: I16Lanes>(
    h: usize,
    source: &PlaneView<'_, u16>,
    (cx, cy): (&[i16; N], &[i16; N]),
    shift1: u32,
    out: &mut [i16],
) {
    if N == 4 {
        let mut r0 = horiz_vec::<N, L>(source.row(0), 0, cx, shift1);
        let mut r1 = horiz_vec::<N, L>(source.row(1), 0, cx, shift1);
        let mut r2 = horiz_vec::<N, L>(source.row(2), 0, cx, shift1);
        for (y, dst) in out.chunks_exact_mut(L::LANES).take(h).enumerate() {
            let r3 = horiz_vec::<N, L>(source.row(y + 3), 0, cx, shift1);
            vert_regs_4([r0, r1, r2, r3], cy).store(dst);
            (r0, r1, r2) = (r1, r2, r3);
        }
    } else if cy[0] == 0 && cy[7] == 0 {
        let mut r1 = horiz_vec::<N, L>(source.row(1), 0, cx, shift1);
        let mut r2 = horiz_vec::<N, L>(source.row(2), 0, cx, shift1);
        let mut r3 = horiz_vec::<N, L>(source.row(3), 0, cx, shift1);
        let mut r4 = horiz_vec::<N, L>(source.row(4), 0, cx, shift1);
        let mut r5 = horiz_vec::<N, L>(source.row(5), 0, cx, shift1);
        for (y, dst) in out.chunks_exact_mut(L::LANES).take(h).enumerate() {
            let r6 = horiz_vec::<N, L>(source.row(y + 6), 0, cx, shift1);
            vert_regs_6([r1, r2, r3, r4, r5, r6], cy).store(dst);
            (r1, r2, r3, r4, r5) = (r2, r3, r4, r5, r6);
        }
    } else {
        let mut r0 = horiz_vec::<N, L>(source.row(0), 0, cx, shift1);
        let mut r1 = horiz_vec::<N, L>(source.row(1), 0, cx, shift1);
        let mut r2 = horiz_vec::<N, L>(source.row(2), 0, cx, shift1);
        let mut r3 = horiz_vec::<N, L>(source.row(3), 0, cx, shift1);
        let mut r4 = horiz_vec::<N, L>(source.row(4), 0, cx, shift1);
        let mut r5 = horiz_vec::<N, L>(source.row(5), 0, cx, shift1);
        let mut r6 = horiz_vec::<N, L>(source.row(6), 0, cx, shift1);
        for (y, dst) in out.chunks_exact_mut(L::LANES).take(h).enumerate() {
            let r7 = horiz_vec::<N, L>(source.row(y + 7), 0, cx, shift1);
            vert_regs_8([r0, r1, r2, r3, r4, r5, r6, r7], cy).store(dst);
            (r0, r1, r2, r3, r4, r5, r6) = (r1, r2, r3, r4, r5, r6, r7);
        }
    }
}

#[inline(always)]
fn horiz_row<const N: usize, W: I16Lanes, M: I16Lanes, V: I16Lanes, H: I16Lanes>(
    src: &[u16],
    cx: &[i16; N],
    shift1: u32,
    dst: &mut [i16],
) {
    let w = dst.len();
    let mut x = 0;
    while x + W::LANES <= w {
        horiz_vec::<N, W>(src, x, cx, shift1).store(&mut dst[x..]);
        x += W::LANES;
    }
    if M::LANES < W::LANES && x + M::LANES <= w {
        horiz_vec::<N, M>(src, x, cx, shift1).store(&mut dst[x..]);
        x += M::LANES;
    }
    if V::LANES < M::LANES && x + V::LANES <= w {
        horiz_vec::<N, V>(src, x, cx, shift1).store(&mut dst[x..]);
        x += V::LANES;
    }
    if x + H::LANES <= w {
        horiz_vec::<N, H>(src, x, cx, shift1).store(&mut dst[x..]);
        x += H::LANES;
    }
    if x < w && w >= H::LANES {
        let tail = w - H::LANES;
        horiz_vec::<N, H>(src, tail, cx, shift1).store(&mut dst[tail..]);
        return;
    }
    while x < w {
        let mut sum = 0i32;
        for k in 0..N {
            sum += i32::from(cx[k]) * i32::from(src[x + k]);
        }
        dst[x] = (sum >> shift1) as i16;
        x += 1;
    }
}

#[inline(always)]
fn two_pass_w4<const N: usize, M: I16FilterLanes, V: I16FilterLanes, H: I16Lanes>(
    h: usize,
    source: &PlaneView<'_, u16>,
    (cx, cy): (&[i16; N], &[i16; N]),
    shift1: u32,
    out: &mut [i16],
) {
    let six_tap = N == 8 && cy[0] == 0 && cy[7] == 0;
    let y_off = usize::from(six_tap);
    let rows = h + N - 1 - 2 * y_off;
    let mut scratch = [MaybeUninit::<i16>::uninit(); 39 * 4];
    // SAFETY: `h <= 32` so `rows <= 39`; every element in `0..rows * 4` is written
    // in the horizontal loop below before the vertical pass reads it.
    let horiz =
        unsafe { core::slice::from_raw_parts_mut(scratch.as_mut_ptr().cast::<i16>(), rows * 4) };
    let mut y = 0;
    while M::LANES == 16 && y + 4 <= rows {
        let r = [
            source.row(y + y_off),
            source.row(y + y_off + 1),
            source.row(y + y_off + 2),
            source.row(y + y_off + 3),
        ];
        horiz_4rows::<N, M>(r, cx, shift1).store(&mut horiz[y * 4..]);
        y += 4;
    }
    while V::LANES == 8 && y + 2 <= rows {
        horiz_2rows::<N, V>(source.row(y + y_off), source.row(y + y_off + 1), cx, shift1)
            .store(&mut horiz[y * 4..]);
        y += 2;
    }
    while y < rows {
        horiz_vec::<N, H>(source.row(y + y_off), 0, cx, shift1).store(&mut horiz[y * 4..]);
        y += 1;
    }
    let hp = horiz.as_ptr().cast::<u16>();
    let mut oy = 0;
    // SAFETY: `horiz` has `rows * 4` initialized `i16`s, covering all vertical taps.
    unsafe {
        while M::LANES == 16 && oy + 4 <= h {
            vert_ptr_vec::<N, M>(hp.add(oy * 4), 4, six_tap, cy).store(&mut out[oy * 4..]);
            oy += 4;
        }
        while V::LANES == 8 && oy + 2 <= h {
            vert_ptr_vec::<N, V>(hp.add(oy * 4), 4, six_tap, cy).store(&mut out[oy * 4..]);
            oy += 2;
        }
        while oy < h {
            vert_ptr_vec::<N, H>(hp.add(oy * 4), 4, six_tap, cy).store(&mut out[oy * 4..]);
            oy += 1;
        }
    }
}

#[inline(always)]
fn two_pass_w8<const N: usize, W: I16FilterLanes, M: I16FilterLanes, V: I16FilterLanes>(
    h: usize,
    source: &PlaneView<'_, u16>,
    (cx, cy): (&[i16; N], &[i16; N]),
    shift1: u32,
    out: &mut [i16],
) {
    let six_tap = N == 8 && cy[0] == 0 && cy[7] == 0;
    let y_off = usize::from(six_tap);
    let rows = h + N - 1 - 2 * y_off;
    let mut scratch = [MaybeUninit::<i16>::uninit(); 39 * 8];
    // SAFETY: `h <= 32` so `rows <= 39`; every element in `0..rows * 8` is written
    // in the horizontal loop below before the vertical pass reads it.
    let horiz =
        unsafe { core::slice::from_raw_parts_mut(scratch.as_mut_ptr().cast::<i16>(), rows * 8) };
    let mut y = 0;
    while W::LANES == 32 && y + 4 <= rows {
        let r = [
            source.row(y + y_off),
            source.row(y + y_off + 1),
            source.row(y + y_off + 2),
            source.row(y + y_off + 3),
        ];
        horiz_4rows::<N, W>(r, cx, shift1).store(&mut horiz[y * 8..]);
        y += 4;
    }
    while M::LANES == 16 && y + 2 <= rows {
        horiz_2rows::<N, M>(source.row(y + y_off), source.row(y + y_off + 1), cx, shift1)
            .store(&mut horiz[y * 8..]);
        y += 2;
    }
    while y < rows {
        horiz_vec::<N, V>(source.row(y + y_off), 0, cx, shift1).store(&mut horiz[y * 8..]);
        y += 1;
    }
    let hp = horiz.as_ptr().cast::<u16>();
    let mut oy = 0;
    // SAFETY: `horiz` has `rows * 8` initialized `i16`s, covering all vertical taps.
    unsafe {
        while W::LANES == 32 && oy + 4 <= h {
            vert_ptr_vec::<N, W>(hp.add(oy * 8), 8, six_tap, cy).store(&mut out[oy * 8..]);
            oy += 4;
        }
        while M::LANES == 16 && oy + 2 <= h {
            vert_ptr_vec::<N, M>(hp.add(oy * 8), 8, six_tap, cy).store(&mut out[oy * 8..]);
            oy += 2;
        }
        while oy < h {
            vert_ptr_vec::<N, V>(hp.add(oy * 8), 8, six_tap, cy).store(&mut out[oy * 8..]);
            oy += 1;
        }
    }
}

#[inline(always)]
fn two_pass_ring<
    const N: usize,
    W: I16FilterLanes,
    M: I16FilterLanes,
    V: I16FilterLanes,
    H: I16Lanes,
>(
    (w, h): (usize, usize),
    source: &PlaneView<'_, u16>,
    (cx, cy): (&[i16; N], &[i16; N]),
    shift1: u32,
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
    let mut ring = [[MaybeUninit::<i16>::uninit(); 64]; 8];
    for (k, slot_buf) in ring.iter_mut().enumerate().take(active - 1) {
        // SAFETY: `w <= 64`; `horiz_row` initializes all `0..w` elements of `slot`.
        let slot =
            unsafe { core::slice::from_raw_parts_mut(slot_buf.as_mut_ptr().cast::<i16>(), w) };
        horiz_row::<N, W, M, V, H>(source.row(y_off + k), cx, shift1, slot);
    }
    for (y, dst) in out.chunks_exact_mut(w).take(h).enumerate() {
        let new_idx = (y + active - 1) & 7;
        // SAFETY: `w <= 64`; `horiz_row` initializes `0..w` before `r_ptrs` reads it.
        let new_slot =
            unsafe { core::slice::from_raw_parts_mut(ring[new_idx].as_mut_ptr().cast::<i16>(), w) };
        horiz_row::<N, W, M, V, H>(source.row(y_off + y + active - 1), cx, shift1, new_slot);
        let mut r_ptrs = [core::ptr::null::<u16>(); 8];
        for (k, p) in r_ptrs.iter_mut().enumerate().take(active) {
            *p = ring[(y + k) & 7].as_ptr().cast::<u16>();
        }
        let mut x = 0;
        // SAFETY: `r_ptrs[0..active]` each hold `w` initialized `i16`s.
        unsafe {
            while x + W::LANES <= w {
                vert_ring_vec::<N, W>(&r_ptrs, x, six_tap, cy).store(&mut dst[x..]);
                x += W::LANES;
            }
            if M::LANES < W::LANES && x + M::LANES <= w {
                vert_ring_vec::<N, M>(&r_ptrs, x, six_tap, cy).store(&mut dst[x..]);
                x += M::LANES;
            }
            if V::LANES < M::LANES && x + V::LANES <= w {
                vert_ring_vec::<N, V>(&r_ptrs, x, six_tap, cy).store(&mut dst[x..]);
                x += V::LANES;
            }
            if x + H::LANES <= w {
                vert_ring_vec::<N, H>(&r_ptrs, x, six_tap, cy).store(&mut dst[x..]);
                x += H::LANES;
            }
            if x < w && w >= H::LANES {
                let tail = w - H::LANES;
                vert_ring_vec::<N, H>(&r_ptrs, tail, six_tap, cy).store(&mut dst[tail..]);
                continue;
            }
            while x < w {
                let mut sum = 0i32;
                for (k, &rp) in r_ptrs.iter().enumerate().take(active) {
                    sum += i32::from(cy[y_off + k]) * i32::from(*rp.add(x).cast::<i16>());
                }
                dst[x] = (sum >> SECOND_PASS_SHIFT) as i16;
                x += 1;
            }
        }
    }
}

#[inline(always)]
fn two_pass_lanes<
    T: SeparableTaps<N>,
    const N: usize,
    W: I16FilterLanes,
    M: I16FilterLanes,
    V: I16FilterLanes,
    H: I16Lanes,
>(
    (w, h): (usize, usize),
    source: &PlaneView<'_, u16>,
    (fx, fy): (usize, usize),
    shift1: u32,
    out: &mut [i16],
) {
    let cx = &T::TABLE[fx];
    let cy = &T::TABLE[fy];
    if w == 4 && h <= 32 {
        two_pass_w4::<N, M, V, H>(h, source, (cx, cy), shift1, out);
        return;
    }
    if w == 8 && V::LANES == 8 && W::LANES > 8 && h <= 32 {
        two_pass_w8::<N, W, M, V>(h, source, (cx, cy), shift1, out);
        return;
    }
    if w == W::LANES {
        two_pass_1vec::<N, W>(h, source, (cx, cy), shift1, out);
        return;
    }
    if M::LANES < W::LANES && w == M::LANES {
        two_pass_1vec::<N, M>(h, source, (cx, cy), shift1, out);
        return;
    }
    two_pass_ring::<N, W, M, V, H>((w, h), source, (cx, cy), shift1, out);
}

#[inline(always)]
fn vert_only_row<const N: usize, W: I16Lanes, M: I16Lanes, V: I16Lanes, H: I16Lanes>(
    rows: &[&[u16]; N],
    cy: &[i16; N],
    shift1: u32,
    dst: &mut [i16],
) {
    let w = dst.len();
    let mut x = 0;
    while x + W::LANES <= w {
        vert_u16_vec::<N, W>(rows, x, cy, shift1).store(&mut dst[x..]);
        x += W::LANES;
    }
    if M::LANES < W::LANES && x + M::LANES <= w {
        vert_u16_vec::<N, M>(rows, x, cy, shift1).store(&mut dst[x..]);
        x += M::LANES;
    }
    if V::LANES < M::LANES && x + V::LANES <= w {
        vert_u16_vec::<N, V>(rows, x, cy, shift1).store(&mut dst[x..]);
        x += V::LANES;
    }
    if x + H::LANES <= w {
        vert_u16_vec::<N, H>(rows, x, cy, shift1).store(&mut dst[x..]);
        x += H::LANES;
    }
    if x < w && w >= H::LANES {
        let tail = w - H::LANES;
        vert_u16_vec::<N, H>(rows, tail, cy, shift1).store(&mut dst[tail..]);
        return;
    }
    while x < w {
        let mut sum = 0i32;
        for k in 0..N {
            sum += i32::from(cy[k]) * i32::from(rows[k][x]);
        }
        dst[x] = (sum >> shift1) as i16;
        x += 1;
    }
}

#[inline(always)]
fn copy_scaled<W: I16Lanes, M: I16Lanes, V: I16Lanes, H: I16Lanes>(
    (w, _h): (usize, usize),
    source: &PlaneView<'_, u16>,
    before: usize,
    shift3: u32,
    out: &mut [i16],
) {
    let scale = 1i16 << shift3;
    for (y, dst) in out.chunks_exact_mut(w).enumerate() {
        let src = &source.row(y + before)[before..];
        let mut x = 0;
        while x + W::LANES <= w {
            W::load_u16(&src[x..])
                .mul(W::splat(scale))
                .store(&mut dst[x..]);
            x += W::LANES;
        }
        if M::LANES < W::LANES && x + M::LANES <= w {
            M::load_u16(&src[x..])
                .mul(M::splat(scale))
                .store(&mut dst[x..]);
            x += M::LANES;
        }
        if V::LANES < M::LANES && x + V::LANES <= w {
            V::load_u16(&src[x..])
                .mul(V::splat(scale))
                .store(&mut dst[x..]);
            x += V::LANES;
        }
        if x + H::LANES <= w {
            H::load_u16(&src[x..])
                .mul(H::splat(scale))
                .store(&mut dst[x..]);
            x += H::LANES;
        }
        if x < w && w >= H::LANES {
            let tail = w - H::LANES;
            H::load_u16(&src[tail..])
                .mul(H::splat(scale))
                .store(&mut dst[tail..]);
            continue;
        }
        while x < w {
            dst[x] = (i32::from(src[x]) << shift3) as i16;
            x += 1;
        }
    }
}

#[inline(always)]
fn interpolate_lanes<
    T: SeparableTaps<N>,
    const N: usize,
    W: I16FilterLanes,
    M: I16FilterLanes,
    V: I16FilterLanes,
    H: I16Lanes,
>(
    (w, h): (usize, usize),
    source: &PlaneView<'_, u16>,
    (fx, fy): (usize, usize),
    out: &mut [i16],
    depth: BitDepth,
) {
    let out = &mut out[..w * h];
    let (shift1, shift3) = interpolation_shifts(depth);
    debug_assert_eq!(shift1 + shift3, SECOND_PASS_SHIFT);
    let before = N / 2 - 1;
    if fx == 0 && fy == 0 {
        copy_scaled::<W, M, V, H>((w, h), source, before, shift3, out);
        return;
    }
    if fy == 0 {
        let cx = &T::TABLE[fx];
        for (y, dst) in out.chunks_exact_mut(w).enumerate() {
            horiz_row::<N, W, M, V, H>(source.row(y + before), cx, shift1, dst);
        }
        return;
    }
    if fx == 0 {
        let cy = &T::TABLE[fy];
        let empty: &[u16] = &[];
        for (y, dst) in out.chunks_exact_mut(w).enumerate() {
            let mut rows = [empty; N];
            for (k, slot) in rows.iter_mut().enumerate() {
                *slot = &source.row(y + k)[before..];
            }
            vert_only_row::<N, W, M, V, H>(&rows, cy, shift1, dst);
        }
        return;
    }
    two_pass_lanes::<T, N, W, M, V, H>((w, h), source, (fx, fy), shift1, out);
}

macro_rules! isa_high_precision {
    ($mod_name:ident, $w:ty, $m:ty, $v:ty, $h:ty, $feature:literal) => {
        pub(crate) mod $mod_name {
            use nc_common::{BitDepth, PlaneView};

            use crate::Filter;

            use super::{
                HalfLumaTaps, HevcChromaTaps, HevcLumaTaps, VvcAffineLumaTaps, VvcChromaTaps,
                VvcLumaTaps, interpolate_lanes,
            };

            macro_rules! filter_fn {
                ($fn_name:ident, $taps:ty, $n:literal) => {
                    #[target_feature(enable = $feature)]
                    unsafe fn $fn_name(
                        size: (usize, usize),
                        window: &PlaneView<'_, u16>,
                        phases: (usize, usize),
                        out: &mut [i16],
                        depth: BitDepth,
                    ) {
                        interpolate_lanes::<$taps, $n, $w, $m, $v, $h>(
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
            pub(crate) unsafe fn high_precision(
                filter: Filter,
                size: (usize, usize),
                window: &PlaneView<'_, u16>,
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
