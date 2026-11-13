//! The self-guided filter of loop restoration (AV1 §7.17.3): up to two
//! box-filter passes (radius 2 on every other row, radius 1 on every row)
//! whose outputs are blended with the source by the unit's projection
//! weights.
//!
//! The arithmetic follows libaom's `av1_selfguided_restoration_c` and
//! `av1_apply_selfguided_restoration_c`, with `av1_sgr_params`,
//! `av1_x_by_xplus1` and `av1_one_by_x` (BSD-2-Clause, see
//! `THIRD_PARTY.md`). The schedule is dav1d 1.5.4's
//! (`src/looprestoration_tmpl.c`, BSD-2-Clause): per window row a
//! horizontal box sum into a ring, per box row the vertical sum and the
//! `A` / `B` statistics (§7.17.3 box filter process) into a second ring,
//! per output row the weighted neighbourhood of those, so nothing larger
//! than a few rows is live.

#![allow(
    clippy::many_single_char_names,
    reason = "the spec's and dav1d's sample, tap and statistic names"
)]
use nc_common::PlaneViewMut;

use super::kernels::Kernels;
use super::{AV1_LR_BORDER, AV1_LR_OVERREAD, Av1SgrProj};

/// `SGRPROJ_RST_BITS`: extra precision of the filtered samples.
const RST_BITS: u32 = 4;
/// `SGRPROJ_PRJ_BITS`: precision of the projection weights.
const PRJ_BITS: u32 = 7;
/// `SGRPROJ_SGR_BITS`: precision of the blend factor `A`.
pub(super) const SGR_BITS: u32 = 8;
/// `SGRPROJ_MTABLE_BITS`: precision of the scale `s`.
pub(super) const MTABLE_BITS: u32 = 20;
/// `SGRPROJ_RECIP_BITS`: precision of `1 / n`.
pub(super) const RECIP_BITS: u32 = 12;
/// Largest `z` the `A` lookup distinguishes (`av1_x_by_xplus1` has 256
/// entries).
pub(super) const Z_MAX: i32 = 255;
/// Shift of an output filtered from two box rows (weights summing to 32).
pub(super) const TWO_ROW_SHIFT: u32 = SGR_BITS + 5 - RST_BITS;
/// Shift of an output filtered from one box row (weights summing to 16).
pub(super) const ONE_ROW_SHIFT: u32 = SGR_BITS + 4 - RST_BITS;
/// Shift of the blended output.
pub(super) const BLEND_SHIFT: u32 = PRJ_BITS + RST_BITS;
/// The source sample's precision in the blend.
pub(super) const SOURCE_SHIFT: u32 = RST_BITS;
/// `1 << PRJ_BITS`: the projection weights' unit.
pub(super) const PRJ_ONE: i32 = 1 << PRJ_BITS;

/// `Sgr_Params` (AV1 §7.17.3): radius and scale of pass 0 and pass 1 by
/// set; radius 0 skips that pass.
const SGR_PARAMS: [[u32; 4]; 16] = [
    [2, 140, 1, 3236],
    [2, 112, 1, 2158],
    [2, 93, 1, 1618],
    [2, 80, 1, 1438],
    [2, 70, 1, 1295],
    [2, 58, 1, 1177],
    [2, 47, 1, 1079],
    [2, 37, 1, 996],
    [2, 30, 1, 925],
    [2, 25, 1, 863],
    [0, 0, 1, 2589],
    [0, 0, 1, 1618],
    [0, 0, 1, 1177],
    [0, 0, 1, 925],
    [2, 56, 0, 0],
    [2, 22, 0, 0],
];

/// Whether pass 0 (radius 2) and pass 1 (radius 1) of parameter set
/// `set` run: a set with a zero radius skips that pass.
pub(super) fn passes(set: u8) -> [bool; 2] {
    let [r0, _, r1, _] = SGR_PARAMS[usize::from(set)];
    [r0 > 0, r1 > 0]
}

/// `x / (x + 1)` in 1/256, with 0 mapped to 1 (libaom
/// `av1_x_by_xplus1`; equals the spec's `((z << 8) + z / 2) / (z + 1)`
/// formula for `0 < z < 255`).
pub(super) fn x_by_xplus1(z: u32) -> u32 {
    match z {
        0 => 1,
        255.. => 256,
        _ => ((z << SGR_BITS) + z / 2) / (z + 1),
    }
}

/// `round(2^12 / n)` (`av1_one_by_x`).
pub(super) fn one_by_x(n: u32) -> u32 {
    ((1 << RECIP_BITS) + n / 2) / n
}

// ------------------------------------------------- scalar references ----

/// Scalar reference of the horizontal box sums of radius `R`: `sum[j]`
/// and `squares[j]` over `src[j..j + 2R + 1]`, for `j < n`.
pub(super) fn box_sums<const R: usize>(src: &[u8], sum: &mut [i32], squares: &mut [i32], n: usize) {
    for j in 0..n {
        let samples = src[j..=j + 2 * R].iter().map(|&s| i32::from(s));
        sum[j] = samples.clone().sum();
        squares[j] = samples.map(|s| s * s).sum();
    }
}

/// Scalar reference of the box statistics from `N` (3 or 5) rows of
/// horizontal sums: per column, the box sum and sum of squares, then the
/// blend factor `A` into `a` and the scaled mean `B` into `b` (AV1
/// §7.17.3, `p`, `z`, `a2`, `b2`).
pub(super) fn box_ab<const N: usize>(
    sums: [&[i32]; N],
    squares: [&[i32]; N],
    scale: u32,
    (a, b): (&mut [i32], &mut [i32]),
    n: usize,
) {
    let count = (N * N) as u32;
    for j in 0..n {
        let sum = sums.iter().map(|row| row[j] as u32).sum::<u32>();
        let square = squares.iter().map(|row| row[j] as u32).sum::<u32>();
        let p = (square * count).saturating_sub(sum * sum);
        let z = (p * scale + (1 << (MTABLE_BITS - 1))) >> MTABLE_BITS;
        let factor = x_by_xplus1(z);
        a[j] = factor as i32;
        b[j] = ((((1 << SGR_BITS) - factor) * sum * one_by_x(count) + (1 << (RECIP_BITS - 1)))
            >> RECIP_BITS) as i32;
    }
}

/// The filtered sample from weighted sums of `A` and `B` (AV1 §7.17.3
/// `v = a * Src + b`).
fn filtered(weighted_a: i32, weighted_b: i32, source: u8, shift: u32) -> i32 {
    (weighted_a * i32::from(source) + weighted_b + (1 << (shift - 1))) >> shift
}

/// Scalar reference of a radius-2 output row between two box rows
/// (`above`, `below`): weights 6 straight across, 5 diagonally. Box
/// columns are offset by one: column `j` reads index `j + 1`.
pub(super) fn filter_between(
    a: [&[i32]; 2],
    b: [&[i32]; 2],
    src: &[u8],
    out: &mut [i32],
    n: usize,
) {
    let weigh = |v: [&[i32]; 2], j: usize| {
        (v[0][j + 1] + v[1][j + 1]) * 6 + (v[0][j] + v[0][j + 2] + v[1][j] + v[1][j + 2]) * 5
    };
    for j in 0..n {
        out[j] = filtered(weigh(a, j), weigh(b, j), src[j], TWO_ROW_SHIFT);
    }
}

/// Scalar reference of a radius-2 output row on a box row: weights 6 at
/// the centre, 5 left and right.
pub(super) fn filter_on(a: &[i32], b: &[i32], src: &[u8], out: &mut [i32], n: usize) {
    let weigh = |v: &[i32], j: usize| v[j + 1] * 6 + (v[j] + v[j + 2]) * 5;
    for j in 0..n {
        out[j] = filtered(weigh(a, j), weigh(b, j), src[j], ONE_ROW_SHIFT);
    }
}

/// Scalar reference of a radius-1 output row from the box rows above, on
/// and below: weights 4 on the cross, 3 on the diagonals.
pub(super) fn filter_around(a: [&[i32]; 3], b: [&[i32]; 3], src: &[u8], out: &mut [i32], n: usize) {
    let weigh = |v: [&[i32]; 3], j: usize| {
        (v[1][j] + v[1][j + 1] + v[1][j + 2] + v[0][j + 1] + v[2][j + 1]) * 4
            + (v[0][j] + v[0][j + 2] + v[2][j] + v[2][j + 2]) * 3
    };
    for j in 0..n {
        out[j] = filtered(weigh(a, j), weigh(b, j), src[j], TWO_ROW_SHIFT);
    }
}

/// Scalar reference of the projection (AV1 §7.17.3:
/// `v = w0 · (flt0 − u) + w1 · (flt1 − u) + u · 2^7`), `dst[j]` from
/// `src[j]` and the two filtered rows.
pub(super) fn blend(src: &[u8], filtered: [&[i32]; 2], weights: [i32; 2], dst: &mut [u8]) {
    for (j, dst) in dst.iter_mut().enumerate() {
        let u = i32::from(src[j]) << SOURCE_SHIFT;
        let v =
            (u << PRJ_BITS) + weights[0] * (filtered[0][j] - u) + weights[1] * (filtered[1][j] - u);
        *dst = ((v + (1 << (BLEND_SHIFT - 1))) >> BLEND_SHIFT).clamp(0, 255) as u8;
    }
}

// ----------------------------------------------------------- driver ----

/// Rows of `len` values in one buffer, addressed modulo their count.
#[derive(Debug, Default)]
struct Ring {
    data: Vec<i32>,
    len: usize,
    count: usize,
}

impl Ring {
    fn reset(&mut self, count: usize, len: usize) {
        super::grow(&mut self.data, count * len);
        (self.count, self.len) = (count, len);
    }

    fn row(&self, index: usize) -> &[i32] {
        let start = index % self.count * self.len;
        &self.data[start..start + self.len]
    }

    fn row_mut(&mut self, index: usize) -> &mut [i32] {
        let start = index % self.count * self.len;
        &mut self.data[start..start + self.len]
    }
}

/// The rings of one box-filter pass: horizontal sums and squares by
/// window row, `A` and `B` by box row.
#[derive(Debug, Default)]
struct PassRings {
    sums: Ring,
    squares: Ring,
    a: Ring,
    b: Ring,
    /// The next window row whose horizontal sums are due.
    next_window_row: usize,
}

impl PassRings {
    fn reset(&mut self, radius: usize, box_rows: usize, len: usize) {
        for ring in [&mut self.sums, &mut self.squares] {
            ring.reset(2 * radius + 1, len);
        }
        for ring in [&mut self.a, &mut self.b] {
            ring.reset(box_rows, len);
        }
        self.next_window_row = 0;
    }
}

/// Reusable buffers of the self-guided filter.
#[derive(Debug, Default)]
pub(super) struct SgrScratch {
    radius2: PassRings,
    radius1: PassRings,
    filtered: [Vec<i32>; 2],
}

/// The window rows of one region and what the kernels read of them.
struct Region<'a> {
    window: &'a [u8],
    stride: usize,
    w: usize,
}

impl Region<'_> {
    /// Window row `t` from column `x`, as far as the kernels may read.
    fn row(&self, t: usize, x: usize) -> &[u8] {
        let start = t * self.stride + x;
        &self.window[start..start + self.w + 2 * AV1_LR_BORDER - x + AV1_LR_OVERREAD]
    }

    /// The region's own samples on its row `i`.
    fn source(&self, i: usize) -> &[u8] {
        self.row(i + AV1_LR_BORDER, AV1_LR_BORDER)
    }

    /// Box statistics computed per row: the region's columns and one more
    /// on each side.
    fn box_columns(&self) -> usize {
        self.w + 2
    }
}

/// Compute the horizontal sums of radius `R` for window rows up to
/// `last`.
fn sums_through<const R: usize>(
    kernels: &Kernels,
    region: &Region<'_>,
    rings: &mut PassRings,
    last: usize,
) {
    // Box column -1 is centred on window column 2.
    let first_column = AV1_LR_BORDER - 1 - R;
    while rings.next_window_row <= last {
        let t = rings.next_window_row;
        kernels.sgr_box_sums(
            R,
            region.row(t, first_column),
            rings.sums.row_mut(t),
            rings.squares.row_mut(t),
            region.box_columns(),
        );
        rings.next_window_row += 1;
    }
}

/// Box row `i` (-1 ..= h) of the radius-2 pass, into `A` / `B` slot
/// `slot`: the window rows `i + 1 ..= i + 5` around its centre `i + 3`.
fn radius2_box_row(
    kernels: &Kernels,
    region: &Region<'_>,
    rings: &mut PassRings,
    scale: u32,
    (i, slot): (isize, usize),
) {
    let top = (i + 1) as usize;
    sums_through::<2>(kernels, region, rings, top + 4);
    let sums = core::array::from_fn(|k| rings.sums.row(top + k));
    let squares = core::array::from_fn(|k| rings.squares.row(top + k));
    let (a, b) = (rings.a.row_mut(slot), rings.b.row_mut(slot));
    kernels.sgr_box_ab5(sums, squares, scale, (a, b), region.box_columns());
}

/// Box row `i` (-1 ..= h) of the radius-1 pass, into slot `i + 1`.
fn radius1_box_row(
    kernels: &Kernels,
    region: &Region<'_>,
    rings: &mut PassRings,
    scale: u32,
    i: isize,
) {
    let top = (i + 2) as usize;
    sums_through::<1>(kernels, region, rings, top + 2);
    let sums = core::array::from_fn(|k| rings.sums.row(top + k));
    let squares = core::array::from_fn(|k| rings.squares.row(top + k));
    let slot = (i + 1) as usize;
    let (a, b) = (rings.a.row_mut(slot), rings.b.row_mut(slot));
    kernels.sgr_box_ab3(sums, squares, scale, (a, b), region.box_columns());
}

/// Self-guided-filter a `w`x`h` region (AV1 §7.17.3). `window` holds the
/// region with [`AV1_LR_BORDER`] source samples on every side, rows
/// `stride` apart, each readable [`AV1_LR_OVERREAD`] samples past its end;
/// the result goes to `dst` at `(x, y)`.
pub(super) fn self_guided(
    kernels: &Kernels,
    params: Av1SgrProj,
    (window, stride): (&[u8], usize),
    (w, h): (usize, usize),
    scratch: &mut SgrScratch,
    (dst, x, y): (&mut PlaneViewMut<'_>, usize, usize),
) {
    let [r0, s0, r1, s1] = SGR_PARAMS[usize::from(params.set)];
    let region = Region { window, stride, w };
    let len = region.box_columns() + AV1_LR_OVERREAD;
    // Radius 2 keeps the box rows either side of an even output row;
    // radius 1 the rows above, on and below.
    scratch.radius2.reset(2, 2, len);
    scratch.radius1.reset(1, 3, len);
    for filtered in &mut scratch.filtered {
        super::grow(filtered, w + AV1_LR_OVERREAD);
    }
    let SgrScratch {
        radius2,
        radius1,
        filtered: [flt0, flt1],
    } = scratch;
    if r0 > 0 {
        // Box rows -1, 1, 3, ... (every other row, §7.17.3), row `i` in
        // slot `(i + 1) / 2`.
        radius2_box_row(kernels, &region, radius2, s0, (-1, 0));
    }
    if r1 > 0 {
        radius1_box_row(kernels, &region, radius1, s1, -1);
        radius1_box_row(kernels, &region, radius1, s1, 0);
    }
    let weights = projection_weights(params, [r0 > 0, r1 > 0]);
    for i in 0..h {
        let source = region.source(i);
        if r0 > 0 {
            let up = i / 2;
            if i % 2 == 0 {
                radius2_box_row(kernels, &region, radius2, s0, (i as isize + 1, up + 1));
                let a = [radius2.a.row(up), radius2.a.row(up + 1)];
                let b = [radius2.b.row(up), radius2.b.row(up + 1)];
                kernels.sgr_filter_between(a, b, source, flt0, w);
            } else {
                let (a, b) = (radius2.a.row(up + 1), radius2.b.row(up + 1));
                kernels.sgr_filter_on(a, b, source, flt0, w);
            }
        }
        if r1 > 0 {
            radius1_box_row(kernels, &region, radius1, s1, i as isize + 1);
            let a = core::array::from_fn(|k| radius1.a.row(i + k));
            let b = core::array::from_fn(|k| radius1.b.row(i + k));
            kernels.sgr_filter_around(a, b, source, flt1, w);
        }
        kernels.sgr_blend(
            source,
            [flt0, flt1],
            weights,
            &mut dst.row_mut(y + i)[x..x + w],
        );
    }
}

/// The projection weights `w0`, `w1` of the two passes (libaom
/// `av1_decode_xq`); a pass that does not run gets weight 0.
fn projection_weights(params: Av1SgrProj, [pass0, pass1]: [bool; 2]) -> [i32; 2] {
    let [xqd0, xqd1] = params.xqd.map(i32::from);
    match (pass0, pass1) {
        (false, _) => [0, PRJ_ONE - xqd1],
        (_, false) => [xqd0, 0],
        _ => [xqd0, PRJ_ONE - xqd0 - xqd1],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The formula equals libaom's `av1_x_by_xplus1` table at its samples.
    #[test]
    fn x_by_xplus1_matches_libaom_samples() {
        for (z, want) in [
            (0, 1),
            (1, 128),
            (2, 171),
            (3, 192),
            (14, 239),
            (100, 253),
            (254, 255),
            (255, 256),
            (4000, 256),
        ] {
            assert_eq!(x_by_xplus1(z), want, "z = {z}");
        }
    }

    /// `one_by_x` equals libaom's `av1_one_by_x` for the two box sizes.
    #[test]
    fn one_by_x_matches_libaom() {
        assert_eq!(one_by_x(9), 455);
        assert_eq!(one_by_x(25), 164);
    }
}
