//! Intra prediction with explicit neighbour availability (slices and
//! `constrained_intra_pred_flag` make it more than "inside the picture"):
//! `Intra_4x4` (§8.3.1.2), `Intra_8x8` with reference sample filtering
//! (§8.3.2.2), `Intra_16x16` (§8.3.3) and 4:2:0 chroma (§8.3.4).
//!
//! The neighbouring samples are gathered into one line ([`Edge`]): the left
//! column bottom-up, the corner, then the row above. With that layout the
//! standard's `p[x, -1]` and `p[-1, y]` are plain indices for every
//! `x, y >= -1`, so each mode is a small loop writing samples straight into
//! the block in the picture, where the residual is then added.

use nc_common::PlaneViewMut;

use super::{DecodeError, Result};

/// Prediction value when no neighbour is available (`1 << (BitDepth - 1)`).
const NO_NEIGHBOUR: u8 = 128;
/// Largest block side with directional modes (`Intra_8x8`) and of any
/// prediction (`Intra_16x16`).
const MAX_DIRECTIONAL: usize = 8;
const MAX_SIDE: usize = 16;
/// Edge line length: `MAX_SIDE` left samples, the corner, `2 * MAX_SIDE`
/// above.
const LINE: usize = 3 * MAX_SIDE + 1;

/// `Intra4x4PredMode` / `Intra8x8PredMode` values (Tables 8-2, 8-3).
const VERTICAL: u8 = 0;
const HORIZONTAL: u8 = 1;
const DC: u8 = 2;
const DIAGONAL_DOWN_LEFT: u8 = 3;
const DIAGONAL_DOWN_RIGHT: u8 = 4;
const VERTICAL_RIGHT: u8 = 5;
const HORIZONTAL_DOWN: u8 = 6;
const VERTICAL_LEFT: u8 = 7;
const HORIZONTAL_UP: u8 = 8;
/// `Intra16x16PredMode` (Table 8-4) and `intra_chroma_pred_mode` (Table 8-5).
const I16_VERTICAL: u8 = 0;
const I16_HORIZONTAL: u8 = 1;
const I16_DC: u8 = 2;
const CHROMA_DC: u8 = 0;
const CHROMA_HORIZONTAL: u8 = 1;
const CHROMA_VERTICAL: u8 = 2;

/// Which neighbouring samples of a block may be used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "one availability per neighbouring sample group"
)]
pub(super) struct Avail {
    pub left: bool,
    pub top: bool,
    pub top_right: bool,
    pub top_left: bool,
}

/// The neighbouring samples of an `n` x `n` block in one line:
/// `line[n - 1 - y]` = p[-1, y] (`y` in `0..n`), `line[n]` = p[-1, -1] and
/// `line[n + 1 + x]` = p[x, -1] (`x` in `0..2n`, the top-right half
/// substituted when unavailable). Unavailable samples are 0 and unread.
#[derive(Clone, Copy)]
struct Edge {
    n: usize,
    line: [u8; LINE],
    avail: Avail,
}

impl Edge {
    fn gather(plane: &PlaneViewMut<'_>, x: usize, y: usize, n: usize, avail: Avail) -> Self {
        let view = plane.as_view();
        let (data, stride) = (view.data(), view.stride());
        // Sample (x, y) of the view; the neighbours are read relative to
        // it (one bounds check per group instead of one per row).
        let at = y * stride + x;
        let mut edge = Self {
            n,
            line: [0; LINE],
            avail,
        };
        if avail.top {
            let width = if avail.top_right { 2 * n } else { n };
            let above = &data[at - stride..at - stride + width];
            edge.line[n + 1..n + 1 + width].copy_from_slice(above);
            // §8.3.1.2 / §8.3.2.2: a missing top-right repeats p[n-1, -1].
            if !avail.top_right {
                let last = above[n - 1];
                edge.line[2 * n + 1..=3 * n].fill(last);
            }
        }
        if avail.left {
            let column = data[at - 1..].iter().step_by(stride).take(n);
            for (sample, &value) in edge.line[..n].iter_mut().rev().zip(column) {
                *sample = value;
            }
        }
        if avail.top_left {
            edge.line[n] = data[at - stride - 1];
        }
        edge
    }

    /// p[x, -1] for `x >= -1`.
    fn t(&self, x: isize) -> u16 {
        u16::from(self.line[(self.n as isize + 1 + x) as usize])
    }

    /// p[-1, y] for `y >= -1`.
    fn l(&self, y: isize) -> u16 {
        u16::from(self.line[(self.n as isize - 1 - y) as usize])
    }

    /// The `Intra_8x8` reference sample filter (§8.3.2.2.1).
    fn filter_8x8(&self) -> Self {
        let n = self.n as isize;
        let avail = self.avail;
        let mut out = *self;
        let set = |out: &mut Self, index: isize, value: u16| out.line[index as usize] = value as u8;
        let (top0, left0) = (self.n as isize + 1, self.n as isize - 1);
        if avail.top {
            let first = if avail.top_left {
                self.t(-1)
            } else {
                self.t(0)
            };
            set(&mut out, top0, (first + 2 * self.t(0) + self.t(1) + 2) >> 2);
            for i in 1..2 * n - 1 {
                let value = (self.t(i - 1) + 2 * self.t(i) + self.t(i + 1) + 2) >> 2;
                set(&mut out, top0 + i, value);
            }
            let last = 2 * n - 1;
            set(
                &mut out,
                top0 + last,
                (self.t(last - 1) + 3 * self.t(last) + 2) >> 2,
            );
        }
        if avail.top_left {
            let corner = self.t(-1);
            let value = match (avail.top, avail.left) {
                (true, true) => (self.t(0) + 2 * corner + self.l(0) + 2) >> 2,
                (true, false) => (3 * corner + self.t(0) + 2) >> 2,
                (false, true) => (3 * corner + self.l(0) + 2) >> 2,
                (false, false) => corner,
            };
            set(&mut out, self.n as isize, value);
        }
        if avail.left {
            let first = if avail.top_left {
                self.l(-1)
            } else {
                self.l(0)
            };
            set(
                &mut out,
                left0,
                (first + 2 * self.l(0) + self.l(1) + 2) >> 2,
            );
            for j in 1..n - 1 {
                let value = (self.l(j - 1) + 2 * self.l(j) + self.l(j + 1) + 2) >> 2;
                set(&mut out, left0 - j, value);
            }
            let last = n - 1;
            set(
                &mut out,
                left0 - last,
                (self.l(last - 1) + 3 * self.l(last) + 2) >> 2,
            );
        }
        out
    }

    /// The row above, p[0..n, -1].
    fn top(&self) -> &[u8] {
        &self.line[self.n + 1..=2 * self.n]
    }

    /// The sum of the row above and of the left column.
    fn sums(&self) -> (u32, u32) {
        let n = self.n;
        let sum = |s: &[u8]| s.iter().map(|&v| u32::from(v)).sum::<u32>();
        (sum(self.top()), sum(&self.line[..n]))
    }

    /// DC of the available row above and/or left column.
    fn dc(&self, top: bool, left: bool) -> u8 {
        let n = self.n as u32;
        let (sum_top, sum_left) = self.sums();
        let shift = n.trailing_zeros();
        (match (top, left) {
            (true, true) => (sum_top + sum_left + n) >> (shift + 1),
            (true, false) => (sum_top + n / 2) >> shift,
            (false, true) => (sum_left + n / 2) >> shift,
            (false, false) => u32::from(NO_NEIGHBOUR),
        }) as u8
    }
}

/// The `n` x `n` block at (`x0`, `y0`) of `plane`.
fn block<'p>(plane: &'p mut PlaneViewMut<'_>, x0: usize, y0: usize, n: usize) -> PlaneViewMut<'p> {
    plane
        .sub_view_mut(x0, y0, n, n)
        .expect("intra blocks lie inside the picture")
}

/// Write `pred(x, y)` over the `n` x `n` block at (`x0`, `y0`).
fn fill(
    plane: &mut PlaneViewMut<'_>,
    (x0, y0): (usize, usize),
    n: usize,
    pred: impl Fn(usize, usize) -> u8,
) {
    let mut out = block(plane, x0, y0, n);
    for y in 0..n {
        for (x, sample) in out.row_mut(y).iter_mut().enumerate() {
            *sample = pred(x, y);
        }
    }
}

/// Copy `row` into every row of the block (vertical prediction).
fn fill_rows(plane: &mut PlaneViewMut<'_>, (x0, y0): (usize, usize), row: &[u8]) {
    let n = row.len();
    let mut out = block(plane, x0, y0, n);
    for y in 0..n {
        out.row_mut(y).copy_from_slice(row);
    }
}

/// Fill row `y` of the block with `value(y)` (horizontal and DC prediction).
fn fill_flat(
    plane: &mut PlaneViewMut<'_>,
    (x0, y0): (usize, usize),
    n: usize,
    value: impl Fn(usize) -> u8,
) {
    let mut out = block(plane, x0, y0, n);
    for y in 0..n {
        out.row_mut(y).fill(value(y));
    }
}

/// Whether `mode` (`Intra_4x4` / `Intra_8x8`) has the samples it reads.
fn directional_allowed(mode: u8, avail: Avail) -> bool {
    match mode {
        VERTICAL | DIAGONAL_DOWN_LEFT | VERTICAL_LEFT => avail.top,
        HORIZONTAL | HORIZONTAL_UP => avail.left,
        DC => true,
        DIAGONAL_DOWN_RIGHT | VERTICAL_RIGHT | HORIZONTAL_DOWN => {
            avail.top && avail.left && avail.top_left
        }
        _ => false,
    }
}

/// `Intra_4x4` (`n` 4) or `Intra_8x8` (`n` 8) prediction of the block at
/// (`x0`, `y0`) in `mode` (§8.3.1.2.1..9, §8.3.2.2.2..10: the same
/// formulas at two sizes).
pub(super) fn predict_nxn(
    plane: &mut PlaneViewMut<'_>,
    x0: usize,
    y0: usize,
    n: usize,
    mode: u8,
    avail: Avail,
) -> Result<()> {
    debug_assert!(n == 4 || n == MAX_DIRECTIONAL);
    if !directional_allowed(mode, avail) {
        return Err(DecodeError::Malformed(
            "intra mode needs unavailable samples",
        ));
    }
    let raw = Edge::gather(plane, x0, y0, n, avail);
    let e = if n == MAX_DIRECTIONAL {
        raw.filter_8x8()
    } else {
        raw
    };
    let at = (x0, y0);
    let at_xy = |x: usize, y: usize| (x as isize, y as isize);
    match mode {
        VERTICAL => fill_rows(plane, at, e.top()),
        HORIZONTAL => fill_flat(plane, at, n, |y| e.line[n - 1 - y]),
        DC => {
            let dc = e.dc(avail.top, avail.left);
            fill_flat(plane, at, n, |_| dc);
        }
        DIAGONAL_DOWN_LEFT => fill(plane, at, n, |x, y| {
            let (x, y) = at_xy(x, y);
            diagonal_down_left(&e, x, y)
        }),
        DIAGONAL_DOWN_RIGHT => fill(plane, at, n, |x, y| {
            let (x, y) = at_xy(x, y);
            diagonal_down_right(&e, x, y)
        }),
        VERTICAL_RIGHT => fill(plane, at, n, |x, y| {
            let (x, y) = at_xy(x, y);
            vertical_right(&e, x, y)
        }),
        HORIZONTAL_DOWN => fill(plane, at, n, |x, y| {
            let (x, y) = at_xy(x, y);
            horizontal_down(&e, x, y)
        }),
        VERTICAL_LEFT => fill(plane, at, n, |x, y| {
            let (x, y) = at_xy(x, y);
            vertical_left(&e, x, y)
        }),
        _ => fill(plane, at, n, |x, y| {
            let (x, y) = at_xy(x, y);
            horizontal_up(&e, x, y)
        }),
    }
    Ok(())
}

fn avg2(a: u16, b: u16) -> u8 {
    ((a + b + 1) >> 1) as u8
}

fn avg3(a: u16, b: u16, c: u16) -> u8 {
    ((a + 2 * b + c + 2) >> 2) as u8
}

/// §8.3.1.2.4 / §8.3.2.2.5.
fn diagonal_down_left(e: &Edge, x: isize, y: isize) -> u8 {
    let last = e.n as isize - 1;
    if x == last && y == last {
        ((e.t(2 * last) + 3 * e.t(2 * last + 1) + 2) >> 2) as u8
    } else {
        avg3(e.t(x + y), e.t(x + y + 1), e.t(x + y + 2))
    }
}

/// §8.3.1.2.5 / §8.3.2.2.6: with the edge line, every case is the
/// [1 2 1] filter centred `x - y` samples from the corner.
fn diagonal_down_right(e: &Edge, x: isize, y: isize) -> u8 {
    let centre = (e.n as isize + x - y) as usize;
    let l = &e.line;
    avg3(
        u16::from(l[centre - 1]),
        u16::from(l[centre]),
        u16::from(l[centre + 1]),
    )
}

/// §8.3.1.2.6 / §8.3.2.2.7.
#[allow(
    clippy::many_single_char_names,
    reason = "x, y, z, k and the edge e follow the standard's formulas"
)]
fn vertical_right(e: &Edge, x: isize, y: isize) -> u8 {
    let z = 2 * x - y;
    let k = x - (y >> 1);
    if z >= 0 && z % 2 == 0 {
        avg2(e.t(k - 1), e.t(k))
    } else if z > 0 {
        avg3(e.t(k - 2), e.t(k - 1), e.t(k))
    } else if z == -1 {
        avg3(e.l(0), e.t(-1), e.t(0))
    } else {
        avg3(e.l(y - 2 * x - 1), e.l(y - 2 * x - 2), e.l(y - 2 * x - 3))
    }
}

/// §8.3.1.2.7 / §8.3.2.2.8.
#[allow(
    clippy::many_single_char_names,
    reason = "x, y, z, k and the edge e follow the standard's formulas"
)]
fn horizontal_down(e: &Edge, x: isize, y: isize) -> u8 {
    let z = 2 * y - x;
    let k = y - (x >> 1);
    if z >= 0 && z % 2 == 0 {
        avg2(e.l(k - 1), e.l(k))
    } else if z > 0 {
        avg3(e.l(k - 2), e.l(k - 1), e.l(k))
    } else if z == -1 {
        avg3(e.l(0), e.t(-1), e.t(0))
    } else {
        avg3(e.t(x - 2 * y - 1), e.t(x - 2 * y - 2), e.t(x - 2 * y - 3))
    }
}

/// §8.3.1.2.8 / §8.3.2.2.9.
fn vertical_left(e: &Edge, x: isize, y: isize) -> u8 {
    let k = x + (y >> 1);
    if y % 2 == 0 {
        avg2(e.t(k), e.t(k + 1))
    } else {
        avg3(e.t(k), e.t(k + 1), e.t(k + 2))
    }
}

/// §8.3.1.2.9 / §8.3.2.2.10: zHU = x + 2y against 2n - 3.
#[allow(
    clippy::many_single_char_names,
    reason = "x, y, z, k and the edge e follow the standard's formulas"
)]
fn horizontal_up(e: &Edge, x: isize, y: isize) -> u8 {
    let last = e.n as isize - 1;
    let z = x + 2 * y;
    let limit = 2 * last - 1;
    let k = y + (x >> 1);
    if z > limit {
        e.l(last) as u8
    } else if z == limit {
        ((e.l(last - 1) + 3 * e.l(last) + 2) >> 2) as u8
    } else if z % 2 == 0 {
        avg2(e.l(k), e.l(k + 1))
    } else {
        avg3(e.l(k), e.l(k + 1), e.l(k + 2))
    }
}

/// Plane prediction (§8.3.3.4 luma `n` 16, §8.3.4.4 4:2:0 chroma `n` 8).
#[allow(
    clippy::many_single_char_names,
    reason = "a, b, c, H and V are the standard's names"
)]
fn plane_prediction(plane: &mut PlaneViewMut<'_>, at: (usize, usize), e: &Edge) {
    let n = e.n as isize;
    let half = n / 2;
    let (mut h, mut v) = (0i32, 0i32);
    for k in 0..half {
        let weight = k as i32 + 1;
        h += weight * (i32::from(e.t(half + k)) - i32::from(e.t(half - 2 - k)));
        v += weight * (i32::from(e.l(half + k)) - i32::from(e.l(half - 2 - k)));
    }
    // Luma: b = (5H + 32) >> 6; 4:2:0 chroma: b = (34H + 32) >> 6.
    let scale = if n == 16 { 5 } else { 34 };
    let a = 16 * (i32::from(e.l(n - 1)) + i32::from(e.t(n - 1)));
    let b = (scale * h + 32) >> 6;
    let c = (scale * v + 32) >> 6;
    let centre = half as i32 - 1;
    fill(plane, at, e.n, |x, y| {
        ((a + b * (x as i32 - centre) + c * (y as i32 - centre) + 16) >> 5).clamp(0, 255) as u8
    });
}

/// `Intra_16x16` prediction of the macroblock at (`x0`, `y0`) (§8.3.3).
pub(super) fn predict_16x16(
    plane: &mut PlaneViewMut<'_>,
    x0: usize,
    y0: usize,
    mode: u8,
    avail: Avail,
) -> Result<()> {
    let needed = match mode {
        I16_VERTICAL => avail.top,
        I16_HORIZONTAL => avail.left,
        I16_DC => true,
        _ => avail.top && avail.left && avail.top_left,
    };
    if !needed {
        return Err(DecodeError::Malformed(
            "Intra_16x16 mode needs unavailable samples",
        ));
    }
    let avail = Avail {
        top_right: false,
        ..avail
    };
    let e = Edge::gather(plane, x0, y0, MAX_SIDE, avail);
    let at = (x0, y0);
    match mode {
        I16_VERTICAL => fill_rows(plane, at, e.top()),
        I16_HORIZONTAL => fill_flat(plane, at, MAX_SIDE, |y| e.line[MAX_SIDE - 1 - y]),
        I16_DC => {
            let dc = e.dc(avail.top, avail.left);
            fill_flat(plane, at, MAX_SIDE, |_| dc);
        }
        _ => plane_prediction(plane, at, &e),
    }
    Ok(())
}

/// 4:2:0 chroma prediction of the 8x8 block at (`x0`, `y0`) (§8.3.4).
pub(super) fn predict_chroma(
    plane: &mut PlaneViewMut<'_>,
    x0: usize,
    y0: usize,
    mode: u8,
    avail: Avail,
) -> Result<()> {
    const SIDE: usize = 8;
    const SUB: usize = 4;
    let needed = match mode {
        CHROMA_DC => true,
        CHROMA_HORIZONTAL => avail.left,
        CHROMA_VERTICAL => avail.top,
        _ => avail.top && avail.left && avail.top_left,
    };
    if !needed {
        return Err(DecodeError::Malformed(
            "chroma intra mode needs unavailable samples",
        ));
    }
    let avail = Avail {
        top_right: false,
        ..avail
    };
    let e = Edge::gather(plane, x0, y0, SIDE, avail);
    let at = (x0, y0);
    match mode {
        CHROMA_DC => {
            // §8.3.4.1..3: each 4x4 block averages the samples beside it;
            // the top-right block prefers those above, the bottom-left the
            // ones to its left.
            let mut dc = [NO_NEIGHBOUR; 4];
            for (blk, value) in dc.iter_mut().enumerate() {
                let (bx, by) = (blk % 2, blk / 2);
                let sum = |s: &[u8]| s.iter().map(|&v| u32::from(v)).sum::<u32>();
                let sum_top = sum(&e.top()[bx * SUB..bx * SUB + SUB]);
                // Left samples run bottom-up in the line.
                let sum_left = sum(&e.line[SIDE - SUB - by * SUB..SIDE - by * SUB]);
                let both = ((sum_top + sum_left + 4) >> 3) as u8;
                let top = ((sum_top + 2) >> 2) as u8;
                let left = ((sum_left + 2) >> 2) as u8;
                *value = match (blk, avail.top, avail.left) {
                    (0 | 3, true, true) => both,
                    (1, true, _) | (0 | 2 | 3, true, false) => top,
                    (2, _, true) | (0 | 1 | 3, false, true) => left,
                    _ => NO_NEIGHBOUR,
                };
            }
            let mut out = block(plane, x0, y0, SIDE);
            for y in 0..SIDE {
                let row = out.row_mut(y);
                let (left, right) = row.split_at_mut(SUB);
                left.fill(dc[(y / SUB) * 2]);
                right.fill(dc[(y / SUB) * 2 + 1]);
            }
        }
        CHROMA_HORIZONTAL => fill_flat(plane, at, SIDE, |y| e.line[SIDE - 1 - y]),
        CHROMA_VERTICAL => fill_rows(plane, at, e.top()),
        _ => plane_prediction(plane, at, &e),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_common::Plane;

    fn ramp_plane() -> Plane {
        let mut plane = Plane::new(32, 32, 0).unwrap();
        let mut view = plane.view_mut();
        for y in 0..32 {
            for (x, v) in view.row_mut(y).iter_mut().enumerate() {
                *v = (20 + 2 * x + y) as u8;
            }
        }
        plane
    }

    const ALL: Avail = Avail {
        left: true,
        top: true,
        top_right: true,
        top_left: true,
    };

    #[test]
    fn plane_mode_follows_a_ramp() {
        let mut plane = ramp_plane();
        predict_16x16(&mut plane.view_mut(), 16, 16, 3, ALL).unwrap();
        let view = plane.view();
        for y in 0..16 {
            for x in 0..16 {
                let want = 20 + 2 * (16 + x) + 16 + y;
                assert!((i32::from(view.row(16 + y)[16 + x]) - want as i32).abs() <= 1);
            }
        }
    }

    #[test]
    fn missing_samples_are_rejected_and_dc_defaults() {
        let mut plane = ramp_plane();
        let none = Avail {
            left: false,
            top: false,
            top_right: false,
            top_left: false,
        };
        assert!(predict_nxn(&mut plane.view_mut(), 0, 0, 4, VERTICAL, none).is_err());
        predict_nxn(&mut plane.view_mut(), 0, 0, 8, DC, none).unwrap();
        assert_eq!(plane.view().row(7)[7], 128);
    }

    /// The encoder predicts `Intra_8x8` with `nc_dsp`'s kernels, this
    /// decoder with its own transcription of §8.3.2.2: both must agree for
    /// every mode and availability.
    #[test]
    fn intra_8x8_matches_the_dsp_kernels() {
        use nc_dsp::{Dsp, H264Intra8x8Edges};
        let dsp = Dsp::new();
        let mut seed = 0x9E37_79B9_u32;
        let mut noise = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed >> 24) as u8
        };
        for flags in 0..16u8 {
            let avail = Avail {
                top: flags & 1 != 0,
                top_right: flags & 2 != 0,
                left: flags & 4 != 0,
                top_left: flags & 8 != 0,
            };
            for _ in 0..8 {
                let mut source = Plane::new(32, 32, 0).unwrap();
                for y in 0..32 {
                    for v in source.view_mut().row_mut(y) {
                        *v = noise();
                    }
                }
                let (x0, y0) = (8, 8);
                let view = source.view();
                let edges = H264Intra8x8Edges {
                    top: core::array::from_fn(|i| view.row(y0 - 1)[x0 + i]),
                    left: core::array::from_fn(|j| view.row(y0 + j)[x0 - 1]),
                    corner: view.row(y0 - 1)[x0 - 1],
                    has_top: avail.top,
                    has_top_right: avail.top_right,
                    has_left: avail.left,
                    has_corner: avail.top_left,
                };
                let refs = edges.filtered();
                for mode in (0..9).filter(|&m| directional_allowed(m, avail)) {
                    let mut decoded = source.clone();
                    predict_nxn(&mut decoded.view_mut(), x0, y0, 8, mode, avail).unwrap();
                    let mut kernel = [0u8; 64];
                    dsp.h264_intra_8x8(&refs, mode, &mut kernel);
                    let decoded = decoded.view();
                    for (i, &k) in kernel.iter().enumerate() {
                        assert_eq!(
                            decoded.row(y0 + i / 8)[x0 + i % 8],
                            k,
                            "mode {mode} flags {flags:#06b} sample {i}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn vertical_copies_the_filtered_row_for_8x8() {
        let mut plane = ramp_plane();
        predict_nxn(&mut plane.view_mut(), 8, 8, 8, VERTICAL, ALL).unwrap();
        // A linear row is unchanged by the [1 2 1] filter.
        assert_eq!(plane.view().row(12)[10], (20 + 2 * 10 + 7) as u8);
    }
}
