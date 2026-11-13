//! VP9 intra predictors as borders and row slices (same results as
//! [`super::reference`]). With `A(i)` the above samples (`A(−1)` the
//! corner) and `L(i)` the left ones:
//!
//! - D45: `B[j] = avg3(A(j), A(j+1), A(j+2))`, the last entry `A(2s−1)`;
//!   row `r` is `B[r..r+s]`. Past the side the row above repeats its last
//!   sample (except for 4x4), so the reference's padding is this average.
//! - D63: `E[j] = avg2(A(j), A(j+1))`, `O[j] = avg3(...)`; row `r` is `E` or
//!   `O` (even or odd `r`) from `r/2`, the padding again an average.
//! - D117: even rows are a column of left averages (bottom-up) followed by
//!   the 2-tap row above, each row starting one earlier; odd rows likewise
//!   with the 3-tap row.
//! - D135: the reference's border, every row starting one earlier.
//! - D153: the left averages of columns 0 and 1, interleaved bottom-up,
//!   then row 0's 3-tap tail; each row starts two earlier.
//! - D207: columns 0 and 1 interleaved top-down, then the last left sample;
//!   each row starts two later.

#![allow(
    clippy::inline_always,
    reason = "the predictors must inline into each ISA entry point to vectorise"
)]

use super::{MAX_SIDE, Vp9IntraEdges, Vp9IntraMode, avg2, avg3, dc_value};

/// Longest border: D153's `2s` interleaved left averages plus `s − 2`.
const MAX_BORDER: usize = 3 * MAX_SIDE;

/// Predict into `out` (row stride `side`).
#[inline(always)]
pub(super) fn predict(mode: Vp9IntraMode, side: usize, n: &Vp9IntraEdges, out: &mut [u8]) {
    let out = &mut out[..side * side];
    match mode {
        Vp9IntraMode::Dc => out.fill(dc_value(side, n)),
        Vp9IntraMode::V => {
            for row in out.chunks_exact_mut(side) {
                row.copy_from_slice(&n.above[1..=side]);
            }
        }
        Vp9IntraMode::H => {
            for (row, &value) in out.chunks_exact_mut(side).zip(&n.left) {
                row.fill(value);
            }
        }
        Vp9IntraMode::Tm => tm(n, side, out),
        Vp9IntraMode::D45 => d45(n, side, out),
        Vp9IntraMode::D63 => d63(n, side, out),
        Vp9IntraMode::D117 => d117(n, side, out),
        Vp9IntraMode::D135 => d135(n, side, out),
        Vp9IntraMode::D153 => d153(n, side, out),
        Vp9IntraMode::D207 => d207(n, side, out),
    }
}

/// Row `r` is `border[start(r)..start(r) + side]`.
#[inline(always)]
fn rows_from(border: &[u8], side: usize, out: &mut [u8], start: impl Fn(usize) -> usize) {
    for (r, row) in out.chunks_exact_mut(side).enumerate() {
        row.copy_from_slice(&border[start(r)..start(r) + side]);
    }
}

#[inline(always)]
fn tm(n: &Vp9IntraEdges, side: usize, out: &mut [u8]) {
    let corner = i16::from(n.above[0]);
    let above = &n.above[1..=side];
    for (row, &left) in out.chunks_exact_mut(side).zip(&n.left) {
        let offset = i16::from(left) - corner;
        for (sample, &a) in row.iter_mut().zip(above) {
            *sample = (i16::from(a) + offset).clamp(0, 255) as u8;
        }
    }
}

#[inline(always)]
fn d45(n: &Vp9IntraEdges, side: usize, out: &mut [u8]) {
    let a = &n.above[1..=2 * side];
    let mut border = [0u8; MAX_BORDER];
    for (j, sample) in border[..2 * side - 2].iter_mut().enumerate() {
        *sample = avg3(a[j], a[j + 1], a[j + 2]);
    }
    border[2 * side - 2] = a[2 * side - 1];
    rows_from(&border, side, out, |r| r);
}

#[inline(always)]
fn d63(n: &Vp9IntraEdges, side: usize, out: &mut [u8]) {
    let a = &n.above[1..=2 * side];
    // Rows read up to index (s − 1) + (s − 1) / 2 of each border.
    let len = side + side / 2;
    let (mut even, mut odd) = ([0u8; MAX_BORDER], [0u8; MAX_BORDER]);
    for j in 0..len {
        even[j] = avg2(a[j], a[j + 1]);
        odd[j] = avg3(a[j], a[j + 1], a[j + 2]);
    }
    for (r, row) in out.chunks_exact_mut(side).enumerate() {
        let border = if r % 2 == 0 { &even } else { &odd };
        row.copy_from_slice(&border[r / 2..r / 2 + side]);
    }
}

#[inline(always)]
fn d117(n: &Vp9IntraEdges, side: usize, out: &mut [u8]) {
    let (a, l, corner) = (&n.above[1..=side], &n.left, n.above[0]);
    let half = side / 2;
    // Column 0 of rows 2.. : C[2] = avg3(X, L0, L1), C[r] = avg3(L(r−3..r)).
    let column = |r: usize| {
        if r == 2 {
            avg3(corner, l[0], l[1])
        } else {
            avg3(l[r - 3], l[r - 2], l[r - 1])
        }
    };
    let (mut even, mut odd) = ([0u8; MAX_BORDER], [0u8; MAX_BORDER]);
    // even = [C[2(h−1)], .., C[2], row 0]; odd = [C[2(h−1)+1], .., C[3], row 1].
    for k in 1..half {
        even[half - 1 - k] = column(2 * k);
        odd[half - 1 - k] = column(2 * k + 1);
    }
    let (row0, row1) = (
        &mut even[half - 1..half - 1 + side],
        &mut odd[half - 1..half - 1 + side],
    );
    row0[0] = avg2(corner, a[0]);
    row1[0] = avg3(l[0], corner, a[0]);
    row1[1] = avg3(corner, a[0], a[1]);
    for c in 1..side {
        row0[c] = avg2(a[c - 1], a[c]);
    }
    for c in 2..side {
        row1[c] = avg3(a[c - 2], a[c - 1], a[c]);
    }
    for (r, row) in out.chunks_exact_mut(side).enumerate() {
        let border = if r % 2 == 0 { &even } else { &odd };
        let start = half - 1 - r / 2;
        row.copy_from_slice(&border[start..start + side]);
    }
}

#[inline(always)]
fn d135(n: &Vp9IntraEdges, side: usize, out: &mut [u8]) {
    let (a, l, corner) = (&n.above[1..=side], &n.left, n.above[0]);
    let mut border = [0u8; MAX_BORDER];
    for i in 0..side - 2 {
        border[i] = avg3(l[side - 3 - i], l[side - 2 - i], l[side - 1 - i]);
    }
    border[side - 2] = avg3(corner, l[0], l[1]);
    border[side - 1] = avg3(l[0], corner, a[0]);
    border[side] = avg3(corner, a[0], a[1]);
    for i in 0..side - 2 {
        border[side + 1 + i] = avg3(a[i], a[i + 1], a[i + 2]);
    }
    rows_from(&border, side, out, |r| side - 1 - r);
}

#[inline(always)]
fn d153(n: &Vp9IntraEdges, side: usize, out: &mut [u8]) {
    let (a, l, corner) = (&n.above[1..=side], &n.left, n.above[0]);
    let mut border = [0u8; MAX_BORDER];
    // [C0[s−1], C1[s−1], .., C0[0], C1[0], row 0 from column 2].
    for r in 0..side {
        let (c0, c1) = match r {
            0 => (avg2(corner, l[0]), avg3(l[0], corner, a[0])),
            1 => (avg2(l[0], l[1]), avg3(corner, l[0], l[1])),
            _ => (avg2(l[r - 1], l[r]), avg3(l[r - 2], l[r - 1], l[r])),
        };
        border[2 * (side - 1 - r)] = c0;
        border[2 * (side - 1 - r) + 1] = c1;
    }
    let tail = &mut border[2 * side..3 * side - 2];
    tail[0] = avg3(corner, a[0], a[1]);
    for c in 1..side - 2 {
        tail[c] = avg3(a[c - 1], a[c], a[c + 1]);
    }
    rows_from(&border, side, out, |r| 2 * (side - 1 - r));
}

#[inline(always)]
fn d207(n: &Vp9IntraEdges, side: usize, out: &mut [u8]) {
    let l = &n.left[..side];
    let last = l[side - 1];
    let mut border = [last; MAX_BORDER];
    // [C0[0], C1[0], C0[1], C1[1], .., then `last`].
    for r in 0..side - 1 {
        border[2 * r] = avg2(l[r], l[r + 1]);
    }
    for r in 0..side - 2 {
        border[2 * r + 1] = avg3(l[r], l[r + 1], l[r + 2]);
    }
    border[2 * (side - 2) + 1] = avg3(l[side - 2], l[side - 1], l[side - 1]);
    rows_from(&border, side, out, |r| 2 * r);
}
