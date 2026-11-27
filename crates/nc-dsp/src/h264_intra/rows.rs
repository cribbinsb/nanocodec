//! H.264 `Intra_8x8` predictors as borders and row slices (same results as
//! [`super::reference`]). With `E` the border of the module docs, `A3[i] =
//! avg3(E[i−1], E[i], E[i+1])` and `A2[i] = avg2(E[i], E[i+1])`, every
//! diagonal mode's sample depends on one diagonal index only, so each row
//! is a window of one sequence:
//!
//! - diagonal down-left: row `y` is `A3[11 + y ..]`;
//! - diagonal down-right: row `y` is `A3[9 − y ..]`;
//! - vertical-left: even rows `A2[10 + y/2 ..]`, odd rows `A3[11 + y/2 ..]`;
//! - vertical-right: even rows `2k` are `[A3[4], A3[6], A3[8], A2[9..]]`
//!   from `3 − k`, odd rows `[A3[3], A3[5], A3[7], A3[9..]]` likewise;
//! - horizontal-down: `S[s]` for `zHD = 14 − s` (2-tap where `zHD` is even
//!   and non-negative, 3-tap otherwise), row `y` from `14 − 2y`;
//! - horizontal-up: `U[zHU]`, row `y` from `2y`.

#![allow(
    clippy::inline_always,
    reason = "the predictors must inline into each ISA entry point to vectorise"
)]

use super::{
    BORDER, H264_INTRA_8X8_SIDE, H264Intra8x8Refs, avg2, avg3, h264_intra_8x8_mode as mode,
};

const N: usize = H264_INTRA_8X8_SIDE;
/// Longest sequence a mode's rows read: horizontal-down and -up span 22.
const SEQUENCE: usize = 3 * N - 2;

/// Predict `mode` into `out` (raster, stride 8).
#[inline(always)]
pub(super) fn predict(refs: &H264Intra8x8Refs, m: u8, out: &mut [u8; 64]) {
    let e = &refs.border;
    match m {
        mode::VERTICAL => {
            for row in out.as_chunks_mut::<N>().0 {
                row.copy_from_slice(&e[super::TOP0..super::TOP0 + N]);
            }
        }
        mode::HORIZONTAL => {
            for (y, row) in out.as_chunks_mut::<N>().0.iter_mut().enumerate() {
                row.fill(refs.l(y));
            }
        }
        mode::DC => out.fill(refs.dc()),
        _ => diagonal(e, m, out),
    }
}

/// The 3-tap averages `A3` (index 0 and the last unused).
#[inline(always)]
fn three_tap(e: &[u8; BORDER]) -> [u8; BORDER] {
    let mut a3 = [0u8; BORDER];
    for i in 1..BORDER - 1 {
        a3[i] = avg3(e[i - 1], e[i], e[i + 1]);
    }
    a3
}

/// The 2-tap averages `A2` (the last unused).
#[inline(always)]
fn two_tap(e: &[u8; BORDER]) -> [u8; BORDER] {
    let mut a2 = [0u8; BORDER];
    for i in 0..BORDER - 1 {
        a2[i] = avg2(e[i], e[i + 1]);
    }
    a2
}

/// Row `y` is `sequence[start(y)..start(y) + 8]`.
#[inline(always)]
fn rows_from(sequence: &[u8], out: &mut [u8; 64], start: impl Fn(usize) -> usize) {
    for (y, row) in out.as_chunks_mut::<N>().0.iter_mut().enumerate() {
        row.copy_from_slice(&sequence[start(y)..start(y) + N]);
    }
}

#[inline(always)]
fn diagonal(e: &[u8; BORDER], m: u8, out: &mut [u8; 64]) {
    let a3 = three_tap(e);
    match m {
        mode::DIAGONAL_DOWN_LEFT => rows_from(&a3, out, |y| 11 + y),
        mode::DIAGONAL_DOWN_RIGHT => rows_from(&a3, out, |y| 9 - y),
        mode::VERTICAL_LEFT => {
            let a2 = two_tap(e);
            for (y, row) in out.as_chunks_mut::<N>().0.iter_mut().enumerate() {
                let (sequence, start) = if y % 2 == 0 {
                    (&a2, 10 + y / 2)
                } else {
                    (&a3, 11 + y / 2)
                };
                row.copy_from_slice(&sequence[start..start + N]);
            }
        }
        mode::VERTICAL_RIGHT => {
            let a2 = two_tap(e);
            // The leading three samples of each sequence come from the left
            // column, every second 3-tap average (zVR < −1).
            let mut even = [0u8; N + 3];
            let mut odd = [0u8; N + 3];
            even[..3].copy_from_slice(&[a3[4], a3[6], a3[8]]);
            odd[..3].copy_from_slice(&[a3[3], a3[5], a3[7]]);
            even[3..].copy_from_slice(&a2[9..9 + N]);
            odd[3..].copy_from_slice(&a3[9..9 + N]);
            for (y, row) in out.as_chunks_mut::<N>().0.iter_mut().enumerate() {
                let sequence = if y % 2 == 0 { &even } else { &odd };
                let start = 3 - y / 2;
                row.copy_from_slice(&sequence[start..start + N]);
            }
        }
        mode::HORIZONTAL_DOWN => {
            let a2 = two_tap(e);
            let mut s = [0u8; SEQUENCE];
            for (index, sample) in s.iter_mut().enumerate() {
                let z = 14 - index as isize;
                *sample = if z >= 0 && z % 2 == 0 {
                    a2[(8 - z / 2) as usize]
                } else if z >= -1 {
                    a3[(9 - (z + 1) / 2) as usize]
                } else {
                    a3[(8 - z) as usize]
                };
            }
            rows_from(&s, out, |y| 14 - 2 * y);
        }
        mode::HORIZONTAL_UP => {
            let a2 = two_tap(e);
            let mut u = [0u8; SEQUENCE];
            for (z, sample) in u.iter_mut().enumerate() {
                *sample = if z > 13 {
                    a2[0]
                } else if z % 2 == 0 {
                    a2[7 - z / 2]
                } else {
                    a3[7 - (z - 1) / 2]
                };
            }
            rows_from(&u, out, |y| 2 * y);
        }
        _ => unreachable!("diagonal Intra8x8PredMode is 3..=8"),
    }
}
