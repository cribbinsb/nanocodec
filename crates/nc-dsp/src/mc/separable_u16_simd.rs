//! Shared 4-tier `u16` SIMD kernels for HEVC and VVC uni-prediction, compound
//! preparation (`predSamplesLX - 8192`), and default bi-prediction across all
//! five [`SeparableTaps`] families (`HevcLuma`, `HevcChroma`, `VvcLuma`,
//! `VvcAffineLuma`, `VvcChroma`).
//!
//! Reuses the 1-D and 2-D interpolation primitives of [`super::precision_simd`]
//! so every separable `u16` filter shares one implementation per pattern.

#![allow(unsafe_code, clippy::inline_always, clippy::too_many_arguments)]

use core::mem::MaybeUninit;

use nc_common::{BitDepth, PlaneView, copy_row};

use crate::simd::{I16FilterLanes, I16Lanes, I16Pairs};

use super::precision_simd::{
    PREP_SCALE, horiz_pair_vec_round, horiz_vec_round, two_pass_lanes, vert_only_lanes,
    vert_u16_vec_round,
};
use super::scalar::{SECOND_PASS_SHIFT, interpolation_shifts};
use super::{HEVC_PREP_OFFSET, SeparableTaps};

const PREDICTION_BITS: u32 = 14;
const ONE_D_ROUND: i32 = 1 << (SECOND_PASS_SHIFT - 1);
const MAX_BLOCK_SIDE: usize = 64;

#[inline(always)]
fn clamp_store_row<const WIDTH: usize, W: I16Lanes, M: I16Lanes, V: I16Lanes, H: I16Lanes>(
    src: &[i16],
    dst: &mut [u16],
    max_val: i16,
) {
    let mut x = 0;
    while x + W::LANES <= WIDTH {
        W::load(&src[x..])
            .max(W::splat(0))
            .min(W::splat(max_val))
            .store_u16(&mut dst[x..]);
        x += W::LANES;
    }
    if M::LANES < W::LANES && x + M::LANES <= WIDTH {
        M::load(&src[x..])
            .max(M::splat(0))
            .min(M::splat(max_val))
            .store_u16(&mut dst[x..]);
        x += M::LANES;
    }
    if V::LANES < M::LANES && x + V::LANES <= WIDTH {
        V::load(&src[x..])
            .max(V::splat(0))
            .min(V::splat(max_val))
            .store_u16(&mut dst[x..]);
        x += V::LANES;
    }
    if x + H::LANES <= WIDTH {
        H::load(&src[x..])
            .max(H::splat(0))
            .min(H::splat(max_val))
            .store_u16(&mut dst[x..]);
    }
}

#[inline(always)]
fn horiz_clamp_row<
    const N: usize,
    const WIDTH: usize,
    W: I16Pairs,
    M: I16Pairs,
    V: I16Pairs,
    H: I16Lanes,
>(
    src: &[u16],
    cx: &[i16; N],
    max_val: i16,
    dst: &mut [u16],
) {
    let mut x = 0;
    while x + W::LANES <= WIDTH {
        horiz_pair_vec_round::<N, W>(src, x, cx, ONE_D_ROUND, SECOND_PASS_SHIFT)
            .max(W::splat(0))
            .min(W::splat(max_val))
            .store_u16(&mut dst[x..]);
        x += W::LANES;
    }
    if M::LANES < W::LANES && x + M::LANES <= WIDTH {
        horiz_pair_vec_round::<N, M>(src, x, cx, ONE_D_ROUND, SECOND_PASS_SHIFT)
            .max(M::splat(0))
            .min(M::splat(max_val))
            .store_u16(&mut dst[x..]);
        x += M::LANES;
    }
    if V::LANES < M::LANES && x + V::LANES <= WIDTH {
        horiz_pair_vec_round::<N, V>(src, x, cx, ONE_D_ROUND, SECOND_PASS_SHIFT)
            .max(V::splat(0))
            .min(V::splat(max_val))
            .store_u16(&mut dst[x..]);
        x += V::LANES;
    }
    if x + H::LANES <= WIDTH {
        horiz_vec_round::<N, H>(src, x, cx, ONE_D_ROUND, SECOND_PASS_SHIFT)
            .max(H::splat(0))
            .min(H::splat(max_val))
            .store_u16(&mut dst[x..]);
    }
}

#[inline(always)]
fn vert_clamp_row<
    const N: usize,
    const WIDTH: usize,
    W: I16Lanes,
    M: I16Lanes,
    V: I16Lanes,
    H: I16Lanes,
>(
    rows: &[&[u16]; N],
    cy: &[i16; N],
    max_val: i16,
    dst: &mut [u16],
) {
    let mut x = 0;
    while x + W::LANES <= WIDTH {
        vert_u16_vec_round::<N, W>(rows, x, cy, ONE_D_ROUND, SECOND_PASS_SHIFT)
            .max(W::splat(0))
            .min(W::splat(max_val))
            .store_u16(&mut dst[x..]);
        x += W::LANES;
    }
    if M::LANES < W::LANES && x + M::LANES <= WIDTH {
        vert_u16_vec_round::<N, M>(rows, x, cy, ONE_D_ROUND, SECOND_PASS_SHIFT)
            .max(M::splat(0))
            .min(M::splat(max_val))
            .store_u16(&mut dst[x..]);
        x += M::LANES;
    }
    if V::LANES < M::LANES && x + V::LANES <= WIDTH {
        vert_u16_vec_round::<N, V>(rows, x, cy, ONE_D_ROUND, SECOND_PASS_SHIFT)
            .max(V::splat(0))
            .min(V::splat(max_val))
            .store_u16(&mut dst[x..]);
        x += V::LANES;
    }
    if x + H::LANES <= WIDTH {
        vert_u16_vec_round::<N, H>(rows, x, cy, ONE_D_ROUND, SECOND_PASS_SHIFT)
            .max(H::splat(0))
            .min(H::splat(max_val))
            .store_u16(&mut dst[x..]);
    }
}

#[inline(always)]
pub(crate) fn predict_lanes<
    T: SeparableTaps<N>,
    const N: usize,
    const WIDTH: usize,
    W: I16FilterLanes + I16Pairs,
    M: I16FilterLanes + I16Pairs,
    V: I16FilterLanes + I16Pairs,
    H: I16Lanes,
>(
    src: &[u16],
    ss: usize,
    dst: &mut [u16],
    ds: usize,
    h: usize,
    (fx, fy): (usize, usize),
    depth: BitDepth,
) {
    let before = N / 2 - 1;
    let max_val = depth.max() as i16;
    if fx == 0 && fy == 0 {
        for y in 0..h {
            let s = &src[(y + before) * ss + before..][..WIDTH];
            copy_row(&mut dst[y * ds..][..WIDTH], s);
        }
        return;
    }
    if fy == 0 {
        let cx = &T::TABLE[fx];
        for y in 0..h {
            horiz_clamp_row::<N, WIDTH, W, M, V, H>(
                &src[(y + before) * ss..],
                cx,
                max_val,
                &mut dst[y * ds..][..WIDTH],
            );
        }
        return;
    }
    if fx == 0 {
        if WIDTH >= V::LANES {
            let window = PlaneView::new(src, ss, WIDTH + N - 1, h + N - 1).expect("mc vert window");
            let mut scratch = [MaybeUninit::<i16>::uninit(); MAX_BLOCK_SIDE * MAX_BLOCK_SIDE];
            // SAFETY: `WIDTH <= 64`, `h <= 64`; `vert_only_lanes` writes `WIDTH * h` elements.
            let buf = unsafe {
                core::slice::from_raw_parts_mut(scratch.as_mut_ptr().cast::<i16>(), WIDTH * h)
            };
            vert_only_lanes::<N, W, M, V, H>(
                (WIDTH, h),
                &window,
                before,
                &T::TABLE[fy],
                (ONE_D_ROUND, SECOND_PASS_SHIFT, 0),
                buf,
            );
            for y in 0..h {
                clamp_store_row::<WIDTH, W, M, V, H>(
                    &buf[y * WIDTH..(y + 1) * WIDTH],
                    &mut dst[y * ds..][..WIDTH],
                    max_val,
                );
            }
            return;
        }
        let cy = &T::TABLE[fy];
        let empty: &[u16] = &[];
        for y in 0..h {
            let mut rows = [empty; N];
            for (k, slot) in rows.iter_mut().enumerate() {
                *slot = &src[(y + k) * ss + before..];
            }
            vert_clamp_row::<N, WIDTH, W, M, V, H>(&rows, cy, max_val, &mut dst[y * ds..][..WIDTH]);
        }
        return;
    }
    let (shift1, _) = interpolation_shifts(depth);
    let post_shift = SECOND_PASS_SHIFT + (PREDICTION_BITS - depth.bits());
    let scale = (1i32 << (post_shift - 1), post_shift, 0);
    let window = PlaneView::new(src, ss, WIDTH + N - 1, h + N - 1).expect("mc 2d window");
    let mut scratch = [MaybeUninit::<i16>::uninit(); MAX_BLOCK_SIDE * MAX_BLOCK_SIDE];
    // SAFETY: `WIDTH <= 64`, `h <= 64`; `two_pass_lanes` writes all `WIDTH * h` elements.
    let buf =
        unsafe { core::slice::from_raw_parts_mut(scratch.as_mut_ptr().cast::<i16>(), WIDTH * h) };
    two_pass_lanes::<T, N, W, M, V, H>((WIDTH, h), &window, (fx, fy), shift1, scale, buf);
    for y in 0..h {
        clamp_store_row::<WIDTH, W, M, V, H>(
            &buf[y * WIDTH..(y + 1) * WIDTH],
            &mut dst[y * ds..][..WIDTH],
            max_val,
        );
    }
}

#[inline(always)]
fn prep_horiz_row<
    const N: usize,
    const WIDTH: usize,
    W: I16Pairs,
    M: I16Pairs,
    V: I16Pairs,
    H: I16Lanes,
>(
    src: &[u16],
    cx: &[i16; N],
    bias_round: i32,
    shift1: u32,
    dst: &mut [i16],
) {
    let mut x = 0;
    while x + W::LANES <= WIDTH {
        horiz_pair_vec_round::<N, W>(src, x, cx, bias_round, shift1).store(&mut dst[x..]);
        x += W::LANES;
    }
    if M::LANES < W::LANES && x + M::LANES <= WIDTH {
        horiz_pair_vec_round::<N, M>(src, x, cx, bias_round, shift1).store(&mut dst[x..]);
        x += M::LANES;
    }
    if V::LANES < M::LANES && x + V::LANES <= WIDTH {
        horiz_pair_vec_round::<N, V>(src, x, cx, bias_round, shift1).store(&mut dst[x..]);
        x += V::LANES;
    }
    if x + H::LANES <= WIDTH {
        horiz_vec_round::<N, H>(src, x, cx, bias_round, shift1).store(&mut dst[x..]);
    }
}

#[inline(always)]
fn prep_vert_row<
    const N: usize,
    const WIDTH: usize,
    W: I16Lanes,
    M: I16Lanes,
    V: I16Lanes,
    H: I16Lanes,
>(
    rows: &[&[u16]; N],
    cy: &[i16; N],
    bias_round: i32,
    shift1: u32,
    dst: &mut [i16],
) {
    let mut x = 0;
    while x + W::LANES <= WIDTH {
        vert_u16_vec_round::<N, W>(rows, x, cy, bias_round, shift1).store(&mut dst[x..]);
        x += W::LANES;
    }
    if M::LANES < W::LANES && x + M::LANES <= WIDTH {
        vert_u16_vec_round::<N, M>(rows, x, cy, bias_round, shift1).store(&mut dst[x..]);
        x += M::LANES;
    }
    if V::LANES < M::LANES && x + V::LANES <= WIDTH {
        vert_u16_vec_round::<N, V>(rows, x, cy, bias_round, shift1).store(&mut dst[x..]);
        x += V::LANES;
    }
    if x + H::LANES <= WIDTH {
        vert_u16_vec_round::<N, H>(rows, x, cy, bias_round, shift1).store(&mut dst[x..]);
    }
}

#[inline(always)]
fn prep_copy_row<const WIDTH: usize, W: I16Lanes, M: I16Lanes, V: I16Lanes, H: I16Lanes>(
    src: &[u16],
    scale: i16,
    dst: &mut [i16],
) {
    let mut x = 0;
    while x + W::LANES <= WIDTH {
        W::load_u16(&src[x..])
            .mul(W::splat(scale))
            .sub(W::splat(HEVC_PREP_OFFSET))
            .store(&mut dst[x..]);
        x += W::LANES;
    }
    if M::LANES < W::LANES && x + M::LANES <= WIDTH {
        M::load_u16(&src[x..])
            .mul(M::splat(scale))
            .sub(M::splat(HEVC_PREP_OFFSET))
            .store(&mut dst[x..]);
        x += M::LANES;
    }
    if V::LANES < M::LANES && x + V::LANES <= WIDTH {
        V::load_u16(&src[x..])
            .mul(V::splat(scale))
            .sub(V::splat(HEVC_PREP_OFFSET))
            .store(&mut dst[x..]);
        x += V::LANES;
    }
    if x + H::LANES <= WIDTH {
        H::load_u16(&src[x..])
            .mul(H::splat(scale))
            .sub(H::splat(HEVC_PREP_OFFSET))
            .store(&mut dst[x..]);
    }
}

#[inline(always)]
pub(crate) fn prep_lanes<
    T: SeparableTaps<N>,
    const N: usize,
    const WIDTH: usize,
    W: I16FilterLanes + I16Pairs,
    M: I16FilterLanes + I16Pairs,
    V: I16FilterLanes + I16Pairs,
    H: I16Lanes,
>(
    src: &[u16],
    ss: usize,
    dst: &mut [i16],
    ds: usize,
    h: usize,
    (fx, fy): (usize, usize),
    depth: BitDepth,
) {
    let (shift1, shift3) = interpolation_shifts(depth);
    let before = N / 2 - 1;
    if fx == 0 && fy == 0 {
        let scale = 1i16 << shift3;
        for y in 0..h {
            prep_copy_row::<WIDTH, W, M, V, H>(
                &src[(y + before) * ss + before..],
                scale,
                &mut dst[y * ds..][..WIDTH],
            );
        }
        return;
    }
    let bias_round = -(i32::from(HEVC_PREP_OFFSET) << shift1);
    if fy == 0 {
        let cx = &T::TABLE[fx];
        for y in 0..h {
            prep_horiz_row::<N, WIDTH, W, M, V, H>(
                &src[(y + before) * ss..],
                cx,
                bias_round,
                shift1,
                &mut dst[y * ds..][..WIDTH],
            );
        }
        return;
    }
    if fx == 0 {
        if WIDTH >= V::LANES && ds == WIDTH {
            let window = PlaneView::new(src, ss, WIDTH + N - 1, h + N - 1).expect("mc prep vert");
            vert_only_lanes::<N, W, M, V, H>(
                (WIDTH, h),
                &window,
                before,
                &T::TABLE[fy],
                (bias_round, shift1, 0),
                &mut dst[..WIDTH * h],
            );
            return;
        }
        let cy = &T::TABLE[fy];
        let empty: &[u16] = &[];
        for y in 0..h {
            let mut rows = [empty; N];
            for (k, slot) in rows.iter_mut().enumerate() {
                *slot = &src[(y + k) * ss + before..];
            }
            prep_vert_row::<N, WIDTH, W, M, V, H>(
                &rows,
                cy,
                bias_round,
                shift1,
                &mut dst[y * ds..][..WIDTH],
            );
        }
        return;
    }
    let window = PlaneView::new(src, ss, WIDTH + N - 1, h + N - 1).expect("mc prep window");
    if ds == WIDTH {
        two_pass_lanes::<T, N, W, M, V, H>(
            (WIDTH, h),
            &window,
            (fx, fy),
            shift1,
            PREP_SCALE,
            &mut dst[..WIDTH * h],
        );
        return;
    }
    let mut scratch = [MaybeUninit::<i16>::uninit(); MAX_BLOCK_SIDE * MAX_BLOCK_SIDE];
    // SAFETY: `WIDTH <= 64`, `h <= 64`; `two_pass_lanes` writes all `WIDTH * h` elements.
    let buf =
        unsafe { core::slice::from_raw_parts_mut(scratch.as_mut_ptr().cast::<i16>(), WIDTH * h) };
    two_pass_lanes::<T, N, W, M, V, H>((WIDTH, h), &window, (fx, fy), shift1, PREP_SCALE, buf);
    for y in 0..h {
        dst[y * ds..][..WIDTH].copy_from_slice(&buf[y * WIDTH..(y + 1) * WIDTH]);
    }
}

#[inline(always)]
fn bi_avg_chunk<L: I16Lanes>(
    first: &[i16],
    second: &[i16],
    dst: &mut [u16],
    x: usize,
    (round, shift, max_val): (i32, u32, i16),
) {
    let acc = L::acc_mul_add2(
        L::acc_zero(),
        L::load(&first[x..]),
        1,
        L::load(&second[x..]),
        1,
    );
    L::acc_round_shift_narrow(acc, round, shift)
        .max(L::splat(0))
        .min(L::splat(max_val))
        .store_u16(&mut dst[x..]);
}

#[inline(always)]
pub(crate) fn bi_lanes<
    T: SeparableTaps<N>,
    const N: usize,
    const WIDTH: usize,
    W: I16FilterLanes + I16Pairs,
    M: I16FilterLanes + I16Pairs,
    V: I16FilterLanes + I16Pairs,
    H: I16Lanes,
>(
    src: &[u16],
    ss: usize,
    (first, fs): (&[i16], usize),
    dst: &mut [u16],
    ds: usize,
    h: usize,
    frac: (usize, usize),
    depth: BitDepth,
) {
    let mut scratch = [MaybeUninit::<i16>::uninit(); MAX_BLOCK_SIDE * MAX_BLOCK_SIDE];
    // SAFETY: `WIDTH <= 64`, `h <= 64`; `prep_lanes` initializes all `WIDTH * h` elements.
    let second =
        unsafe { core::slice::from_raw_parts_mut(scratch.as_mut_ptr().cast::<i16>(), WIDTH * h) };
    prep_lanes::<T, N, WIDTH, W, M, V, H>(src, ss, second, WIDTH, h, frac, depth);
    let shift = PREDICTION_BITS + 1 - depth.bits();
    let round = 2 * i32::from(HEVC_PREP_OFFSET) + (1 << (shift - 1));
    let params = (round, shift, depth.max() as i16);
    for y in 0..h {
        let f_row = &first[y * fs..][..WIDTH];
        let s_row = &second[y * WIDTH..(y + 1) * WIDTH];
        let d_row = &mut dst[y * ds..][..WIDTH];
        let mut x = 0;
        while x + W::LANES <= WIDTH {
            bi_avg_chunk::<W>(f_row, s_row, d_row, x, params);
            x += W::LANES;
        }
        if M::LANES < W::LANES && x + M::LANES <= WIDTH {
            bi_avg_chunk::<M>(f_row, s_row, d_row, x, params);
            x += M::LANES;
        }
        if V::LANES < M::LANES && x + V::LANES <= WIDTH {
            bi_avg_chunk::<V>(f_row, s_row, d_row, x, params);
            x += V::LANES;
        }
        if x + H::LANES <= WIDTH {
            bi_avg_chunk::<H>(f_row, s_row, d_row, x, params);
        }
    }
}
