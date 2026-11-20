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
use nc_common::{BitDepth, PlaneViewMut, Sample};

use super::kernels::RowKernels;
use super::sgr_lanes::{FilteredScratch, LaneRow};
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
pub(super) const fn x_by_xplus1(z: u32) -> u32 {
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

/// The largest horizontal box sum, five samples of 255 (radius 2): the
/// `i16` storage of the sums is exact (dav1d 8bpc keeps them in `int16_t`).
const MAX_ROW_SUM: i32 = 5 * 255;
const _: () = assert!(MAX_ROW_SUM <= i16::MAX as i32);

/// The largest blend factor `A` (`x_by_xplus1(255)`); every `A` is in
/// `1..=256`, so its `i16` storage is exact.
const MAX_FACTOR: u32 = 1 << SGR_BITS;
const _: () = assert!(MAX_FACTOR <= i16::MAX as u32);

/// Scalar reference of the horizontal box sums of radius `R`: `sum[j]`
/// and `squares[j]` over `src[j..j + 2R + 1]`, for `j < n`.
pub(super) fn box_sums<T: Sample, const R: usize>(
    src: &[T],
    sum: &mut [i16],
    squares: &mut [i32],
    n: usize,
) {
    for j in 0..n {
        let samples = src[j..=j + 2 * R].iter().map(|&s| s.to_i32());
        // At most MAX_ROW_SUM (8 bits; 5 · 1023 at 10), so the narrowing
        // is exact.
        sum[j] = samples.clone().sum::<i32>() as i16;
        squares[j] = samples.map(|s| s * s).sum();
    }
}

/// Scalar reference of the box statistics from `N` (3 or 5) rows of
/// horizontal sums: per column, the box sum and sum of squares, then the
/// blend factor `A` into `a` and the scaled mean `B` into `b` (AV1
/// §7.17.3, `p`, `z`, `a2`, `b2`).
pub(super) fn box_ab<const N: usize>(
    sums: [&[i16]; N],
    squares: [&[i32]; N],
    scale: u32,
    ab: (&mut [i16], &mut [i32]),
    n: usize,
) {
    box_ab_at(sums, squares, scale, ab, (n, 0));
}

/// [`box_ab`] of samples `bits` bits deeper than 8: the variance is taken
/// of the sums rounded down to 8-bit precision (libaom `calc_ab` with
/// `bit_depth > 8`); `B` still scales the full-precision sum.
pub(super) fn box_ab_at<const N: usize>(
    sums: [&[i16]; N],
    squares: [&[i32]; N],
    scale: u32,
    (a, b): (&mut [i16], &mut [i32]),
    (n, bits): (usize, u32),
) {
    let count = (N * N) as u32;
    let round = |v: u32, shift: u32| {
        if shift == 0 {
            v
        } else {
            (v + (1 << (shift - 1))) >> shift
        }
    };
    for j in 0..n {
        let sum = sums.iter().map(|row| row[j] as u32).sum::<u32>();
        let square = squares.iter().map(|row| row[j] as u32).sum::<u32>();
        let (square_8, sum_8) = (round(square, 2 * bits), round(sum, bits));
        let p = (square_8 * count).saturating_sub(sum_8 * sum_8);
        let z = (p * scale + (1 << (MTABLE_BITS - 1))) >> MTABLE_BITS;
        let factor = x_by_xplus1(z);
        // At most MAX_FACTOR, so the narrowing is exact.
        a[j] = factor as i16;
        b[j] = ((((1 << SGR_BITS) - factor) * sum * one_by_x(count) + (1 << (RECIP_BITS - 1)))
            >> RECIP_BITS) as i32;
    }
}

/// The filtered sample from weighted sums of `A` and `B` (AV1 §7.17.3
/// `v = a * Src + b`).
///
/// It fits `i16`: per box `1 ≤ A ≤ 256` and `0 ≤ B ≤ ((256 − A) · 255 · n
/// · one_by_x(n) + 2^11) >> 12` (`n` = 9 or 25), so `A · Src + B ≤ 65344`
/// over every `A`; the weights (32 or 16) are removed by the shift (9 or
/// 8), so `0 ≤ v ≤ 4084 < 2^13`, as in dav1d's 8bpc `int16_t` rows.
fn filtered<T: Sample, F: LaneRow>(weighted_a: i32, weighted_b: i32, source: T, shift: u32) -> F {
    F::from_i32((weighted_a * source.to_i32() + weighted_b + (1 << (shift - 1))) >> shift)
}

/// `A` in 32 bits, for the weighted sums.
fn wide(v: &[i16], j: usize) -> i32 {
    i32::from(v[j])
}

/// Scalar reference of a radius-2 output row between two box rows
/// (`above`, `below`): weights 6 straight across, 5 diagonally. Box
/// columns are offset by one: column `j` reads index `j + 1`.
pub(super) fn filter_between<T: Sample, F: LaneRow>(
    a: [&[i16]; 2],
    b: [&[i32]; 2],
    src: &[T],
    out: &mut [F],
    n: usize,
) {
    let weigh_a = |v: [&[i16]; 2], j: usize| {
        (wide(v[0], j + 1) + wide(v[1], j + 1)) * 6
            + (wide(v[0], j) + wide(v[0], j + 2) + wide(v[1], j) + wide(v[1], j + 2)) * 5
    };
    let weigh_b = |v: [&[i32]; 2], j: usize| {
        (v[0][j + 1] + v[1][j + 1]) * 6 + (v[0][j] + v[0][j + 2] + v[1][j] + v[1][j + 2]) * 5
    };
    for j in 0..n {
        out[j] = filtered(weigh_a(a, j), weigh_b(b, j), src[j], TWO_ROW_SHIFT);
    }
}

/// Scalar reference of a radius-2 output row on a box row: weights 6 at
/// the centre, 5 left and right.
pub(super) fn filter_on<T: Sample, F: LaneRow>(
    a: &[i16],
    b: &[i32],
    src: &[T],
    out: &mut [F],
    n: usize,
) {
    let weigh_a = |v: &[i16], j: usize| wide(v, j + 1) * 6 + (wide(v, j) + wide(v, j + 2)) * 5;
    let weigh_b = |v: &[i32], j: usize| v[j + 1] * 6 + (v[j] + v[j + 2]) * 5;
    for j in 0..n {
        out[j] = filtered(weigh_a(a, j), weigh_b(b, j), src[j], ONE_ROW_SHIFT);
    }
}

/// Scalar reference of a radius-1 output row from the box rows above, on
/// and below: weights 4 on the cross, 3 on the diagonals.
pub(super) fn filter_around<T: Sample, F: LaneRow>(
    a: [&[i16]; 3],
    b: [&[i32]; 3],
    src: &[T],
    out: &mut [F],
    n: usize,
) {
    let weigh_a = |v: [&[i16]; 3], j: usize| {
        (wide(v[1], j)
            + wide(v[1], j + 1)
            + wide(v[1], j + 2)
            + wide(v[0], j + 1)
            + wide(v[2], j + 1))
            * 4
            + (wide(v[0], j) + wide(v[0], j + 2) + wide(v[2], j) + wide(v[2], j + 2)) * 3
    };
    let weigh_b = |v: [&[i32]; 3], j: usize| {
        (v[1][j] + v[1][j + 1] + v[1][j + 2] + v[0][j + 1] + v[2][j + 1]) * 4
            + (v[0][j] + v[0][j + 2] + v[2][j] + v[2][j + 2]) * 3
    };
    for j in 0..n {
        out[j] = filtered(weigh_a(a, j), weigh_b(b, j), src[j], TWO_ROW_SHIFT);
    }
}

/// Scalar reference of the projection (AV1 §7.17.3:
/// `v = w0 · (flt0 − u) + w1 · (flt1 − u) + u · 2^7`), `dst[j]` from
/// `src[j]` and the two filtered rows.
pub(super) fn blend(src: &[u8], filtered: [&[i16]; 2], weights: [i32; 2], dst: &mut [u8]) {
    blend_at(src, filtered, weights, dst, BitDepth::EIGHT);
}

/// [`blend`] at any depth.
pub(super) fn blend_at<T: Sample, F: LaneRow>(
    src: &[T],
    filtered: [&[F]; 2],
    weights: [i32; 2],
    dst: &mut [T],
    depth: BitDepth,
) {
    for (j, dst) in dst.iter_mut().enumerate() {
        let u = src[j].to_i32() << SOURCE_SHIFT;
        let [f0, f1] = filtered.map(|row| row[j].to_i32());
        let v = (u << PRJ_BITS) + weights[0] * (f0 - u) + weights[1] * (f1 - u);
        *dst = depth.clip_to((v + (1 << (BLEND_SHIFT - 1))) >> BLEND_SHIFT);
    }
}

// ----------------------------------------------------------- driver ----

/// Rows of `len` values in one buffer, addressed modulo their count.
#[derive(Debug, Default)]
struct Ring<T> {
    data: Vec<T>,
    len: usize,
    count: usize,
}

impl<T: Copy + Default> Ring<T> {
    fn reset(&mut self, count: usize, len: usize) {
        super::grow(&mut self.data, count * len);
        (self.count, self.len) = (count, len);
    }

    fn row(&self, index: usize) -> &[T] {
        let start = index % self.count * self.len;
        &self.data[start..start + self.len]
    }

    fn row_mut(&mut self, index: usize) -> &mut [T] {
        let start = index % self.count * self.len;
        &mut self.data[start..start + self.len]
    }
}

/// The rings of one box-filter pass: horizontal sums and squares by
/// window row, `A` and `B` by box row. The sums and `A` fit `i16`
/// ([`MAX_ROW_SUM`], [`MAX_FACTOR`]); the squares and `B` (up to about
/// 65280) do not.
#[derive(Debug, Default)]
struct PassRings {
    sums: Ring<i16>,
    squares: Ring<i32>,
    a: Ring<i16>,
    b: Ring<i32>,
    /// The next window row whose horizontal sums are due.
    next_window_row: usize,
}

impl PassRings {
    fn reset(&mut self, radius: usize, box_rows: usize, len: usize) {
        self.sums.reset(2 * radius + 1, len);
        self.squares.reset(2 * radius + 1, len);
        self.a.reset(box_rows, len);
        self.b.reset(box_rows, len);
        self.next_window_row = 0;
    }
}

/// Reusable buffers of the self-guided filter.
#[derive(Debug, Default)]
pub(super) struct SgrScratch {
    radius2: PassRings,
    radius1: PassRings,
    filtered: FilteredScratch,
    stripe: SgrStripeScratch,
}

/// Reusable rows of a complete-stripe kernel ([`RowKernels::sgr_stripe`]),
/// laid out by the kernel.
#[derive(Debug, Default)]
#[cfg_attr(
    not(any(target_arch = "x86_64", target_arch = "aarch64")),
    expect(
        dead_code,
        reason = "only the AVX2 and NEON tables have a stripe kernel"
    )
)]
pub(crate) struct SgrStripeScratch {
    pub(super) words: Vec<i16>,
    pub(super) dwords: Vec<i32>,
}

/// What a complete-stripe kernel needs of a unit's parameters.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(
    not(any(target_arch = "x86_64", target_arch = "aarch64")),
    expect(
        dead_code,
        reason = "only the AVX2 and NEON tables have a stripe kernel"
    )
)]
pub(crate) struct SgrStripe {
    /// The scale `s` of radius 2 and of radius 1; 0 when that pass does
    /// not run.
    pub(super) scales: [u32; 2],
    /// The projection weights `w0`, `w1` ([`projection_weights`]).
    pub(super) weights: [i32; 2],
}

/// The window rows of one region and what the kernels read of them.
struct Region<'a, T> {
    window: &'a [T],
    stride: usize,
    w: usize,
}

impl<T> Region<'_, T> {
    /// Window row `t` from column `x`, as far as the kernels may read.
    fn row(&self, t: usize, x: usize) -> &[T] {
        let start = t * self.stride + x;
        &self.window[start..start + self.w + 2 * AV1_LR_BORDER - x + AV1_LR_OVERREAD]
    }

    /// The region's own samples on its row `i`.
    fn source(&self, i: usize) -> &[T] {
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
fn sums_through<T: Sample, const R: usize>(
    kernels: &impl RowKernels<T>,
    region: &Region<'_, T>,
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
fn radius2_box_row<T: Sample>(
    kernels: &impl RowKernels<T>,
    region: &Region<'_, T>,
    rings: &mut PassRings,
    scale: u32,
    (i, slot): (isize, usize),
) {
    let top = (i + 1) as usize;
    sums_through::<T, 2>(kernels, region, rings, top + 4);
    let sums = core::array::from_fn(|k| rings.sums.row(top + k));
    let squares = core::array::from_fn(|k| rings.squares.row(top + k));
    let (a, b) = (rings.a.row_mut(slot), rings.b.row_mut(slot));
    kernels.sgr_box_ab5(sums, squares, scale, (a, b), region.box_columns());
}

/// Box row `i` (-1 ..= h) of the radius-1 pass, into slot `i + 1`.
fn radius1_box_row<T: Sample>(
    kernels: &impl RowKernels<T>,
    region: &Region<'_, T>,
    rings: &mut PassRings,
    scale: u32,
    i: isize,
) {
    let top = (i + 2) as usize;
    sums_through::<T, 1>(kernels, region, rings, top + 2);
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
pub(super) fn self_guided<T: Sample, K: RowKernels<T>>(
    kernels: &K,
    params: Av1SgrProj,
    (window, stride): (&[T], usize),
    (w, h): (usize, usize),
    scratch: &mut SgrScratch,
    (dst, x, y): (&mut PlaneViewMut<'_, T>, usize, usize),
) {
    let [r0, s0, r1, s1] = SGR_PARAMS[usize::from(params.set)];
    let weights = projection_weights(params, [r0 > 0, r1 > 0]);
    let stripe = SgrStripe {
        scales: [if r0 > 0 { s0 } else { 0 }, if r1 > 0 { s1 } else { 0 }],
        weights,
    };
    if kernels.sgr_stripe(
        stripe,
        (window, stride),
        (w, h),
        &mut scratch.stripe,
        (&mut *dst, x, y),
    ) {
        return;
    }
    let region = Region { window, stride, w };
    let len = region.box_columns() + AV1_LR_OVERREAD;
    // Radius 2 keeps the box rows either side of an even output row;
    // radius 1 the rows above, on and below.
    scratch.radius2.reset(2, 2, len);
    scratch.radius1.reset(1, 3, len);
    let SgrScratch {
        radius2,
        radius1,
        filtered,
        ..
    } = scratch;
    let [flt0, flt1] = <K::Filtered as LaneRow>::scratch_vecs(filtered);
    let flt0 = super::grow(flt0, w + AV1_LR_OVERREAD);
    let flt1 = super::grow(flt1, w + AV1_LR_OVERREAD);
    if r0 > 0 {
        // Box rows -1, 1, 3, ... (every other row, §7.17.3), row `i` in
        // slot `(i + 1) / 2`.
        radius2_box_row(kernels, &region, radius2, s0, (-1, 0));
    }
    if r1 > 0 {
        radius1_box_row(kernels, &region, radius1, s1, -1);
        radius1_box_row(kernels, &region, radius1, s1, 0);
    }
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
