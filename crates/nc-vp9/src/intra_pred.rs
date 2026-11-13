//! VP9 intra prediction of one transform block from the unfiltered
//! reconstruction (VP9 §8.5.1; libvpx `vp9_reconintra.c`
//! `build_intra_predictors`, BSD-3-Clause, see `THIRD_PARTY.md`): the edge
//! samples, gathered here, and the predictors of all ten modes in nc-dsp
//! (`Dsp::vp9_intra_predict`).
//!
//! Unavailable edges take VP9's fixed values: 127 above (and above-left),
//! 129 to the left. The above-right samples are real only for 4x4
//! transform blocks that are not in the rightmost column of their block;
//! otherwise the last above sample is repeated (libvpx's `have_right`
//! rule, which every VP9 decoder follows). Coded blocks always lie inside
//! the 8-aligned frame, so the frame-edge extension paths of libvpx never
//! apply.

use nc_common::PlaneView;
use nc_dsp::{Vp9IntraEdges, Vp9IntraMode};

use crate::types::IntraMode;

/// Fixed value of an unavailable above row.
const BASE_ABOVE: u8 = 127;
/// Fixed value of an unavailable left column (and of above-left when only
/// the row above is available).
const BASE_LEFT: u8 = 129;
/// Largest transform side.
const MAX_SIDE: usize = 32;

/// Which neighbours of the transform block are available.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Edges {
    pub top: bool,
    pub left: bool,
    /// The transform block is not in the rightmost column of its block.
    pub right: bool,
}

/// The edge samples of the `side`x`side` block at `(x, y)`: above-left,
/// the row above extended to twice the side, and the left column.
fn gather(recon: &PlaneView<'_>, x: usize, y: usize, side: usize, edges: Edges) -> Vp9IntraEdges {
    let mut above = [BASE_ABOVE; 2 * MAX_SIDE + 1];
    let mut left = [BASE_LEFT; MAX_SIDE];
    if edges.top {
        let row = recon.row(y - 1);
        above[1..=side].copy_from_slice(&row[x..x + side]);
        // Real above-right samples only for inner 4x4 transform blocks.
        if side == 4 && edges.right {
            above[side + 1..=2 * side].copy_from_slice(&row[x + side..x + 2 * side]);
        } else {
            let last = above[side];
            above[side + 1..=2 * side].fill(last);
        }
        above[0] = if edges.left { row[x - 1] } else { BASE_LEFT };
    }
    if edges.left {
        for (i, sample) in left[..side].iter_mut().enumerate() {
            *sample = recon.row(y + i)[x - 1];
        }
    }
    Vp9IntraEdges {
        above,
        left,
        has_above: edges.top,
        has_left: edges.left,
    }
}

/// Predict the `side`x`side` block at plane position `(x, y)` of `recon`
/// into `out` (row stride `side`).
pub(crate) fn predict(
    dsp: &nc_dsp::Dsp,
    recon: &PlaneView<'_>,
    (x, y): (usize, usize),
    side: usize,
    mode: IntraMode,
    edges: Edges,
    out: &mut [u8],
) {
    let neighbours = gather(recon, x, y, side, edges);
    dsp.vp9_intra_predict(
        Vp9IntraMode::ALL[mode as usize],
        side,
        &neighbours,
        &mut out[..side * side],
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn avg2(a: i32, b: i32) -> u8 {
        ((a + b + 1) >> 1) as u8
    }

    fn avg3(a: i32, b: i32, c: i32) -> u8 {
        ((a + 2 * b + c + 2) >> 2) as u8
    }

    fn edges(seed: u32) -> Vp9IntraEdges {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (state >> 16) as u8
        };
        let mut e = Vp9IntraEdges {
            above: [0; 2 * MAX_SIDE + 1],
            left: [0; MAX_SIDE],
            has_above: true,
            has_left: true,
        };
        e.above.iter_mut().for_each(|s| *s = next());
        e.left.iter_mut().for_each(|s| *s = next());
        e
    }

    fn predicted(e: &Vp9IntraEdges, mode: IntraMode) -> [u8; 16] {
        let mut out = [0; 16];
        nc_dsp::Dsp::new().vp9_intra_predict(Vp9IntraMode::ALL[mode as usize], 4, e, &mut out);
        out
    }

    /// libvpx's 4x4 predictors (`vpx_d*_predictor_4x4_c`) at a few closed
    /// forms, through the dispatched kernel.
    #[test]
    #[allow(clippy::many_single_char_names, reason = "libvpx's edge names")]
    fn directional_predictors_match_libvpx_4x4_closed_forms() {
        for seed in 0..50 {
            let e = edges(seed);
            let a = |i: isize| i32::from(e.above[(i + 1) as usize]);
            let l = |i: usize| i32::from(e.left[i]);
            let (i, j, k, ll) = (l(0), l(1), l(2), l(3));
            let (x, p0, p1, p2, p3) = (a(-1), a(0), a(1), a(2), a(3));
            let at = |out: &[u8; 16], col: usize, row: usize| out[row * 4 + col];

            let out = predicted(&e, IntraMode::D207);
            assert_eq!(at(&out, 0, 0), avg2(i, j));
            assert_eq!(at(&out, 3, 1), avg3(k, ll, ll));
            assert_eq!(at(&out, 3, 2), ll as u8);

            let out = predicted(&e, IntraMode::D117);
            assert_eq!(at(&out, 1, 2), avg2(x, p0));
            assert_eq!(at(&out, 3, 0), avg2(p2, p3));
            assert_eq!(at(&out, 0, 3), avg3(k, j, i));

            let out = predicted(&e, IntraMode::D135);
            assert_eq!(at(&out, 0, 3), avg3(j, k, ll));
            assert_eq!(at(&out, 3, 0), avg3(p3, p2, p1));

            let out = predicted(&e, IntraMode::D153);
            assert_eq!(at(&out, 2, 1), avg2(i, x));
            assert_eq!(at(&out, 3, 0), avg3(p0, p1, p2));

            let p: Vec<i32> = (0..8).map(a).collect();
            let out = predicted(&e, IntraMode::D63);
            assert_eq!(out[3], avg2(p[3], p[4]));
            assert_eq!(out[3 * 4 + 3], avg3(p[4], p[5], p[6]));
            let out = predicted(&e, IntraMode::D45);
            assert_eq!(out[0], avg3(p[0], p[1], p[2]));
            assert_eq!(out[15], p[7] as u8);
        }
    }
}
