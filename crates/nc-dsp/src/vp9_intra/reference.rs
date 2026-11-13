//! Scalar reference VP9 intra predictors: libvpx's per-sample definitions
//! (`vpx_dsp/intrapred.c`), transcribed from `nc-vp9`'s original
//! predictors. The oracle of [`super::rows`].

use super::{MAX_SIDE, Vp9IntraEdges, Vp9IntraMode, avg2, avg3, dc_value};

/// Above sample `i` of `edges`, where -1 is the above-left corner.
fn a(edges: &Vp9IntraEdges, i: isize) -> u8 {
    edges.above[(i + 1) as usize]
}

/// Predict into `out` (row stride `side`).
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
        Vp9IntraMode::Tm => {
            for (r, row) in out.chunks_exact_mut(side).enumerate() {
                for (c, sample) in row.iter_mut().enumerate() {
                    let value =
                        i32::from(n.left[r]) + i32::from(a(n, c as isize)) - i32::from(a(n, -1));
                    *sample = value.clamp(0, 255) as u8;
                }
            }
        }
        Vp9IntraMode::D45 if side == 4 => d45_4x4(n, out),
        Vp9IntraMode::D45 => d45(n, side, out),
        Vp9IntraMode::D63 if side == 4 => d63_4x4(n, out),
        Vp9IntraMode::D63 => d63(n, side, out),
        Vp9IntraMode::D117 => d117(n, side, out),
        Vp9IntraMode::D135 => d135(n, side, out),
        Vp9IntraMode::D153 => d153(n, side, out),
        Vp9IntraMode::D207 => d207(n, side, out),
    }
}

/// `d45_predictor`: diagonals of the smoothed row, the last sample repeated.
fn d45(n: &Vp9IntraEdges, side: usize, out: &mut [u8]) {
    let above_right = n.above[side];
    let mut first = [0u8; MAX_SIDE];
    for (c, sample) in first[..side - 1].iter_mut().enumerate() {
        let c = c as isize;
        *sample = avg3(a(n, c), a(n, c + 1), a(n, c + 2));
    }
    first[side - 1] = above_right;
    for (r, row) in out.chunks_exact_mut(side).enumerate() {
        for (c, sample) in row.iter_mut().enumerate() {
            *sample = if r + c < side {
                first[r + c]
            } else {
                above_right
            };
        }
    }
}

/// `vpx_d45_predictor_4x4_c`: reads the full above-right.
fn d45_4x4(n: &Vp9IntraEdges, out: &mut [u8]) {
    for (r, row) in out.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        for (c, sample) in row.iter_mut().enumerate() {
            let i = (r + c) as isize;
            *sample = if r + c == 6 {
                n.above[8]
            } else {
                avg3(a(n, i), a(n, i + 1), a(n, i + 2))
            };
        }
    }
}

/// `d63_predictor`: alternating 2- and 3-tap rows shifted by one every two
/// rows, padded with the last above sample.
fn d63(n: &Vp9IntraEdges, side: usize, out: &mut [u8]) {
    let last = n.above[side];
    for (r, row) in out.chunks_exact_mut(side).enumerate() {
        let shift = r >> 1;
        let valid = side - 1 - shift;
        for (c, sample) in row.iter_mut().enumerate() {
            *sample = if r < 2 || c < valid {
                let i = (c + shift) as isize;
                if r % 2 == 0 {
                    avg2(a(n, i), a(n, i + 1))
                } else {
                    avg3(a(n, i), a(n, i + 1), a(n, i + 2))
                }
            } else {
                last
            };
        }
    }
}

/// `vpx_d63_predictor_4x4_c`: continues the averages into the above-right.
fn d63_4x4(n: &Vp9IntraEdges, out: &mut [u8]) {
    for (r, row) in out.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        for (c, sample) in row.iter_mut().enumerate() {
            let i = (c + (r >> 1)) as isize;
            *sample = if r % 2 == 0 {
                avg2(a(n, i), a(n, i + 1))
            } else {
                avg3(a(n, i), a(n, i + 1), a(n, i + 2))
            };
        }
    }
}

/// `d117_predictor`: rows 0/1 from the row above, column 0 from the left;
/// every other sample repeats the one two rows up and one column left.
fn d117(n: &Vp9IntraEdges, side: usize, out: &mut [u8]) {
    let l = |i: usize| n.left[i];
    for c in 0..side {
        let i = c as isize;
        out[c] = avg2(a(n, i - 1), a(n, i));
        out[side + c] = if c == 0 {
            avg3(l(0), a(n, -1), a(n, 0))
        } else {
            avg3(a(n, i - 2), a(n, i - 1), a(n, i))
        };
    }
    out[2 * side] = avg3(a(n, -1), l(0), l(1));
    for r in 3..side {
        out[r * side] = avg3(l(r - 3), l(r - 2), l(r - 1));
    }
    for r in 2..side {
        for c in 1..side {
            out[r * side + c] = out[(r - 2) * side + c - 1];
        }
    }
}

/// `d135_predictor`: every row is the outer border shifted by one.
fn d135(n: &Vp9IntraEdges, side: usize, out: &mut [u8]) {
    let l = |i: usize| n.left[i];
    let mut border = [0u8; 2 * MAX_SIDE - 1];
    for (i, sample) in border[..side - 2].iter_mut().enumerate() {
        *sample = avg3(l(side - 3 - i), l(side - 2 - i), l(side - 1 - i));
    }
    border[side - 2] = avg3(a(n, -1), l(0), l(1));
    border[side - 1] = avg3(l(0), a(n, -1), a(n, 0));
    border[side] = avg3(a(n, -1), a(n, 0), a(n, 1));
    for i in 0..side - 2 {
        let c = i as isize;
        border[side + 1 + i] = avg3(a(n, c), a(n, c + 1), a(n, c + 2));
    }
    for (r, row) in out.chunks_exact_mut(side).enumerate() {
        row.copy_from_slice(&border[side - 1 - r..2 * side - 1 - r]);
    }
}

/// `d153_predictor`: columns 0/1 from the left, row 0 from above; every
/// other sample repeats the one a row up and two columns left.
fn d153(n: &Vp9IntraEdges, side: usize, out: &mut [u8]) {
    let l = |i: usize| n.left[i];
    out[0] = avg2(a(n, -1), l(0));
    for r in 1..side {
        out[r * side] = avg2(l(r - 1), l(r));
    }
    out[1] = avg3(l(0), a(n, -1), a(n, 0));
    out[side + 1] = avg3(a(n, -1), l(0), l(1));
    for r in 2..side {
        out[r * side + 1] = avg3(l(r - 2), l(r - 1), l(r));
    }
    for c in 0..side - 2 {
        let i = c as isize;
        out[2 + c] = avg3(a(n, i - 1), a(n, i), a(n, i + 1));
    }
    for r in 1..side {
        for c in 2..side {
            out[r * side + c] = out[(r - 1) * side + c - 2];
        }
    }
}

/// `d207_predictor`: columns 0/1 from the left, the bottom row the last
/// left sample; every other sample repeats the one a row down and two
/// columns left.
fn d207(n: &Vp9IntraEdges, side: usize, out: &mut [u8]) {
    let l = |i: usize| n.left[i];
    let last = l(side - 1);
    for r in 0..side - 1 {
        out[r * side] = avg2(l(r), l(r + 1));
    }
    out[(side - 1) * side] = last;
    for r in 0..side - 2 {
        out[r * side + 1] = avg3(l(r), l(r + 1), l(r + 2));
    }
    out[(side - 2) * side + 1] = avg3(l(side - 2), l(side - 1), l(side - 1));
    out[(side - 1) * side + 1] = last;
    for c in 2..side {
        out[(side - 1) * side + c] = last;
    }
    for r in (0..side - 1).rev() {
        for c in 2..side {
            out[r * side + c] = out[(r + 1) * side + c - 2];
        }
    }
}
