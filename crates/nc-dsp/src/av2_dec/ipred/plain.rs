//! Vertical, horizontal and Paeth prediction. With multiple reference lines
//! vertical and horizontal copy the rounded mean of line `mrl_idx` and line
//! 0, which the edge buffer holds side by side (see [`super`]).
//!
//! Ported from dav2d `src/ipred_tmpl.c` (`ipred_v_c`, `ipred_h_c`,
//! `ipred_paeth_c`; commit 237229e, BSD-2-Clause, see `docs/licenses/dav2d`).
//! Paeth's choice is AV1's ([`crate::intra_prediction::paeth`]).

use super::{ANGLE_MULTI_MRL_FLAG, Block, SampleRef};
use crate::intra_prediction;

/// Offset of line 0's corner from line `mrl_idx`'s for vertical and
/// horizontal prediction, whose edges carry no reference-line margin.
fn line0_offset(width: usize, height: usize) -> isize {
    ((width + height) * 2 + 1) as isize
}

/// Rounded mean of two samples.
fn average(a: u8, b: u8) -> u8 {
    ((u16::from(a) + u16::from(b) + 1) >> 1) as u8
}

/// `VERT_PRED`: every row copies the top edge.
pub(super) fn vertical(
    dst: &mut Block<'_>,
    edge: SampleRef<'_>,
    width: usize,
    height: usize,
    angle: i32,
) {
    let mut top = [0; 64];
    let top = &mut top[..width];
    top.copy_from_slice(edge.slice(1, width));
    if angle & ANGLE_MULTI_MRL_FLAG != 0 {
        let line0 = edge.slice(line0_offset(width, height) + 1, width);
        for (t, &t0) in top.iter_mut().zip(line0) {
            *t = average(*t, t0);
        }
    }
    for y in 0..height {
        dst.row(y, width).copy_from_slice(top);
    }
}

/// `HOR_PRED`: every column copies the left edge.
pub(super) fn horizontal(
    dst: &mut Block<'_>,
    edge: SampleRef<'_>,
    width: usize,
    height: usize,
    angle: i32,
) {
    let multi_mrl = angle & ANGLE_MULTI_MRL_FLAG != 0;
    let line0 = edge.shifted(line0_offset(width, height));
    for y in 0..height {
        let below = -(y as isize + 1);
        let mut left = edge.at(below);
        if multi_mrl {
            left = average(left, line0.at(below));
        }
        dst.row(y, width).fill(left);
    }
}

/// `PAETH_PRED`: per sample, whichever of left, top and corner is nearest
/// to `left + top - corner`.
pub(super) fn paeth(dst: &mut Block<'_>, edge: SampleRef<'_>, width: usize, height: usize) {
    let corner = edge.at(0);
    let top = edge.slice(1, width);
    for y in 0..height {
        let left = edge.at(-(y as isize + 1));
        for (out, &t) in dst.row(y, width).iter_mut().zip(top) {
            *out = intra_prediction::paeth(left, t, corner);
        }
    }
}
