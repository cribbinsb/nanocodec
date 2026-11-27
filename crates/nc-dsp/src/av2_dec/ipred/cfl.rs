//! Chroma from luma, explicit and implicit, for 4:2:0: both chroma planes
//! are their edges' DC plus alpha times the block's downsampled luma minus
//! its edges' DC. Explicit `CfL` codes the alphas; implicit `CfL` fits them to
//! up to eight edge sample pairs.
//!
//! Ported from dav2d `src/ipred_tmpl.c` (`cfl_pred`, `cfl_explicit_420_c`,
//! `cfl_implicit_420_c`) and `src/derivation.h` (`derive_alpha`; commit
//! 237229e, BSD-2-Clause, see `docs/licenses/dav2d`).

use nc_common::{BitDepth, Sample};

use super::dc::fast_div32_dc;
use super::tables::DIV_RECIP;
use super::{
    CFL_ALPHA_U_MASK, CFL_ALPHA_U_SHIFT, CFL_ALPHA_V_MASK, CFL_ALPHA_V_SHIFT, CFL_FLT_TYPE,
    CFL_FLT_TYPE_GAUSS, CFL_FLT_TYPE_VSTRIP, CFL_HAS_LEFT, CFL_HAS_TOP, CFL_IS_TOP_SB_EDGE,
    SampleRef, clip_pixel,
};

/// Luma values carry three fraction bits (the 4:2:0 filters' weights sum
/// to 8).
const LUMA_FRACTION_BITS: u32 = 3;
/// Bits of the alpha-scaled AC term (alphas are 1/32 steps scaled by 32).
const AC_SHIFT: u32 = 11;
/// Most edge sample pairs implicit `CfL` fits.
const MAX_PAIRS: usize = 8;
/// Sides of this length count only every other edge sample in the DC.
const SUBSAMPLED_SIDE: usize = 64;
/// Bits of the explicit alphas once extracted (1/32 units scaled by 32).
const ALPHA_EXTRACT_BITS: u32 = 5;
/// The GAUSS filter's top tap stays in the row at multiples of 32 rows
/// (AVM's luma line buffer boundary).
const GAUSS_TOP_ROW_PERIOD: usize = 32;
/// Luma columns per 64-sample unit, where the VSTRIP and GAUSS left tap
/// clamps.
const LUMA_UNIT: usize = 64;
/// Largest implicit alpha magnitude (`derive_alpha`'s `max`).
const MAX_IMPLICIT_ALPHA: i32 = (2 << 8) - 1;

/// One chroma plane's block for `CfL`: written in place, its left column read
/// at `offset - 1`. The row above comes from `top_sb_edge` (at the block's
/// first column) at a superblock's top, else from the row above in `buf`.
#[derive(Debug)]
pub struct CflChroma<'a, T = u8> {
    /// The plane.
    pub buf: &'a mut [T],
    /// Index of the block's first sample.
    pub offset: usize,
    /// The pre-filter row above, at a superblock's top edge.
    pub top_sb_edge: Option<SampleRef<'a, T>>,
}

impl<T: Sample> CflChroma<'_, T> {
    /// The sample `x` along the row above.
    fn top(&self, x: usize, stride: usize) -> T {
        match self.top_sb_edge {
            Some(row) => row.at(x as isize),
            None => self.buf[self.offset + x - stride],
        }
    }

    /// The sample `y` down the column to the left.
    fn left(&self, y: usize, stride: usize) -> T {
        self.buf[self.offset + y * stride - 1]
    }
}

/// The 4:2:0 luma filter `flags & CFL_FLT_TYPE` at luma column `xl` of
/// `row`, whose next row is `below` away and previous row `above` away;
/// `left` is the column of the left tap. The result has
/// [`LUMA_FRACTION_BITS`].
fn downsample<T: Sample>(
    flags: u32,
    row: SampleRef<'_, T>,
    xl: isize,
    left: isize,
    above: isize,
    below: isize,
) -> i32 {
    let p = |i: isize| row.wide(i);
    match flags & CFL_FLT_TYPE {
        CFL_FLT_TYPE_GAUSS => p(left) + 4 * p(xl) + p(xl + 1) + p(xl - above) + p(xl + below),
        CFL_FLT_TYPE_VSTRIP => {
            p(left)
                + 2 * p(xl)
                + p(xl + 1)
                + p(left + below)
                + 2 * p(xl + below)
                + p(xl + 1 + below)
        }
        _ => (p(xl) + p(xl + 1) + p(xl + below) + p(xl + 1 + below)) << 1,
    }
}

/// `n` edge sample pairs at even spacing: whether sample `i` of `len` is
/// one (the middle of each of `n` runs).
fn is_pair_sample(i: usize, len: usize, n: usize) -> bool {
    let step = len >> n.trailing_zeros();
    i & (step - 1) == step >> 1
}

/// DC sums and the implicit fit's sample pairs, gathered along the edges.
#[derive(Default)]
struct EdgeStats<T> {
    /// Sums of luma, U and V.
    sum: [i32; 3],
    /// The pairs: luma (integer) and the two chroma samples.
    pairs: [[T; 3]; MAX_PAIRS],
    n_pairs: usize,
}

impl<T: Sample> EdgeStats<T> {
    /// Adds edge position `i` of a side of `len` samples: luma `l`, chroma
    /// `u` and `v`; the DC skips odd positions of 64-sample sides and the
    /// fit takes `n_fit` evenly spaced pairs.
    fn add(&mut self, i: usize, len: usize, n_fit: usize, l: i32, u: T, v: T) {
        if len != SUBSAMPLED_SIDE || i & 1 == 0 {
            self.sum[0] += l;
            self.sum[1] += u.to_i32();
            self.sum[2] += v.to_i32();
        }
        if n_fit > 0 && is_pair_sample(i, len, n_fit) {
            self.pairs[self.n_pairs] = [T::from_i32(l >> LUMA_FRACTION_BITS), u, v];
            self.n_pairs += 1;
        }
    }
}

/// Implicit `CfL`'s pair counts from the top and left edges.
fn implicit_counts(has_t: bool, has_l: bool, w: usize, h: usize) -> (usize, usize) {
    match (has_t, has_l) {
        (true, true) if w > h * 2 => (MAX_PAIRS, 0),
        (true, true) if h > w * 2 => (0, MAX_PAIRS),
        (true, true) => (MAX_PAIRS / 2, MAX_PAIRS / 2),
        _ => (
            if has_t { w.min(MAX_PAIRS) } else { 0 },
            if has_l { h.min(MAX_PAIRS) } else { 0 },
        ),
    }
}

/// The slope `num / den` of a least-squares fit, as a fixed-point scale
/// with 8 fraction bits (1/32 steps scaled by 32 for chroma from luma),
/// saturated; `fallback` when the fit gives none or zero (dav2d
/// `derive_alpha`: 0 for chroma from luma, 256 for block-adaptive weights).
#[must_use]
pub fn derive_alpha(num: i32, den: i32, fallback: i32) -> i32 {
    const FRACTION_BITS: u32 = 7;
    if num == 0 || den == 0 {
        return fallback;
    }
    // Normalise both to 1.7 fixed point, divide by the reciprocal table.
    let normalise = |v: u32, log2: u32| -> i32 {
        if log2 > FRACTION_BITS {
            ((v + (1 << (log2 - FRACTION_BITS - 1))) >> (log2 - FRACTION_BITS)) as i32
        } else {
            (v << (FRACTION_BITS - log2)) as i32
        }
    };
    let num_abs = num.unsigned_abs();
    let shift_n = num_abs.ilog2();
    let den = den as u32;
    let shift_d = den.ilog2();
    let f_d = normalise(den - (1 << shift_d), shift_d);
    let f_n = normalise(num_abs, shift_n);
    let shift_add = shift_d as i32 - shift_n as i32 - 8;
    if shift_add > 1 {
        return fallback;
    }
    let shift = 9 + 7 + shift_add;
    let alpha = if shift < 0 {
        MAX_IMPLICIT_ALPHA
    } else {
        ((i32::from(DIV_RECIP[f_d as usize]) * f_n) >> shift).min(MAX_IMPLICIT_ALPHA)
    };
    match alpha {
        0 => fallback,
        _ if num < 0 => -alpha,
        _ => alpha,
    }
}

/// The least-squares slopes of U and V on luma over the pairs.
fn implicit_alphas<T: Sample>(pairs: &[[T; 3]]) -> [i32; 2] {
    let mut sum_x = 0;
    let mut sum_xx = 0;
    let mut sum_y = [0; 2];
    let mut sum_xy = [0; 2];
    for &[x, u, v] in pairs {
        let x = x.to_i32();
        sum_x += x;
        sum_xx += x * x;
        for (pl, c) in [u, v].into_iter().enumerate() {
            sum_y[pl] += c.to_i32();
            sum_xy[pl] += x * c.to_i32();
        }
    }
    let count_log2 = if pairs.is_empty() {
        0
    } else {
        pairs.len().trailing_zeros()
    };
    let den = sum_xx - ((i64::from(sum_x) * i64::from(sum_x)) >> count_log2) as i32;
    std::array::from_fn(|pl| {
        let num = sum_xy[pl] - ((i64::from(sum_x) * i64::from(sum_y[pl])) >> count_log2) as i32;
        derive_alpha(num, den, 0)
    })
}

/// The coded alphas in the explicit flags, sign-extended.
fn explicit_alphas(flags: u32) -> [i32; 2] {
    let u = i32::from((flags & CFL_ALPHA_U_MASK) as u16 as i16)
        >> (CFL_ALPHA_U_SHIFT - ALPHA_EXTRACT_BITS);
    let v = (flags & CFL_ALPHA_V_MASK) as i32 >> (CFL_ALPHA_V_SHIFT - ALPHA_EXTRACT_BITS);
    [u, v]
}

/// The rounded mean of `sum` over `count` samples.
fn mean(sum: i32, count: usize) -> i32 {
    if count.is_power_of_two() {
        (sum + (count >> 1) as i32) >> count.trailing_zeros()
    } else {
        fast_div32_dc(sum as u32, count as u32) as i32
    }
}

/// Chroma from luma for a 4:2:0 `w` x `h` chroma block (`cfl_pred` for
/// `cfl_explicit_420` / `cfl_implicit_420`).
///
/// `luma` is the co-located luma block (rows `luma_stride` apart) and
/// `luma_top` the luma row two above it, or the pre-filter row at a
/// superblock's top ([`CFL_IS_TOP_SB_EDGE`]), at the block's first column.
/// `u_plane` and `v_plane` share `chroma_stride`. `wpad`/`hpad` are the 4-sample units
/// of the block past the picture's right and bottom edge, which repeat the
/// last visible column and row. `flags` holds the downsampling filter,
/// [`CFL_HAS_TOP`]/[`CFL_HAS_LEFT`], [`CFL_IS_TOP_SB_EDGE`] and, for explicit
/// `CfL`, the coded alphas. `depth` is the samples' bit depth.
pub fn cfl_pred_420<T: Sample>(
    implicit: bool,
    luma_top: SampleRef<'_, T>,
    luma: SampleRef<'_, T>,
    luma_stride: usize,
    mut u_plane: CflChroma<'_, T>,
    mut v_plane: CflChroma<'_, T>,
    chroma_stride: usize,
    wpad: usize,
    hpad: usize,
    w: usize,
    h: usize,
    flags: u32,
    depth: BitDepth,
) {
    let depth = depth.of::<T>();
    debug_assert!(wpad * 4 < w / 2 && hpad * 4 < h / 2);
    let has_t = flags & CFL_HAS_TOP != 0;
    let has_l = flags & CFL_HAS_LEFT != 0;
    let xlim = w - 4 * wpad;
    let ylim = h - 4 * hpad;
    let ys = luma_stride as isize;
    let (n_top, n_left) = if implicit {
        implicit_counts(has_t, has_l, w, h)
    } else {
        (0, 0)
    };

    let mut stats = EdgeStats::default();
    if has_l {
        let mut l = 0;
        for y in 0..h {
            let yv = y.min(ylim - 1);
            if y < ylim {
                let row = luma.shifted(2 * y as isize * ys - 2);
                l = downsample(flags, row, 0, -1, if y > 0 { ys } else { 0 }, ys);
            }
            let (cu, cv) = (
                u_plane.left(yv, chroma_stride),
                v_plane.left(yv, chroma_stride),
            );
            stats.add(y, h, n_left, l, cu, cv);
        }
    }
    if has_t {
        let below = if flags & CFL_IS_TOP_SB_EDGE != 0 {
            0
        } else {
            ys
        };
        let mut l = 0;
        for x in 0..w {
            let xv = x.min(xlim - 1);
            if x < xlim {
                let xl = 2 * x as isize;
                l = downsample(flags, luma_top, xl, (xl - 1).max(0), below, below);
            }
            let (cu, cv) = (
                u_plane.top(xv, chroma_stride),
                v_plane.top(xv, chroma_stride),
            );
            stats.add(x, w, n_top, l, cu, cv);
        }
    }
    debug_assert_eq!(stats.n_pairs, n_top + n_left);

    let dc = if has_t || has_l {
        let count = if has_t {
            w >> usize::from(w == SUBSAMPLED_SIDE)
        } else {
            0
        } + if has_l {
            h >> usize::from(h == SUBSAMPLED_SIDE)
        } else {
            0
        };
        stats.sum.map(|s| mean(s, count))
    } else {
        // Mid-grey: luma `4 << bitdepth` (with its fraction bits), chroma
        // `(bitdepth_max + 1) >> 1`.
        let mid = depth.mid();
        [mid << LUMA_FRACTION_BITS, mid, mid]
    };
    let alphas = if implicit {
        implicit_alphas(&stats.pairs[..stats.n_pairs])
    } else {
        explicit_alphas(flags)
    };

    let ac = luma_ac(luma, ys, flags, xlim, ylim, dc[0]);
    write_plane(
        &mut u_plane,
        chroma_stride,
        &ac,
        alphas[0],
        dc[1],
        (xlim, ylim),
        (w, h),
        depth,
    );
    write_plane(
        &mut v_plane,
        chroma_stride,
        &ac,
        alphas[1],
        dc[2],
        (xlim, ylim),
        (w, h),
        depth,
    );
}

/// The block's downsampled luma minus its DC, `xlim` x `ylim` (32 wide).
fn luma_ac<T: Sample>(
    luma: SampleRef<'_, T>,
    ys: isize,
    flags: u32,
    xlim: usize,
    ylim: usize,
    dc: i32,
) -> [[i32; 32]; 32] {
    let mut ac = [[0; 32]; 32];
    for (y, ac_row) in ac.iter_mut().enumerate().take(ylim) {
        let row = luma.shifted(2 * y as isize * ys);
        let above = if y % GAUSS_TOP_ROW_PERIOD == 0 { 0 } else { ys };
        for (x, a) in ac_row.iter_mut().enumerate().take(xlim) {
            let xl = 2 * x as isize;
            let left = (xl & -(LUMA_UNIT as isize)).max(xl - 1);
            *a = downsample(flags, row, xl, left, above, ys) - dc;
        }
    }
    ac
}

/// Writes one chroma plane: flat `dc` when `alpha` is zero, else `dc` plus
/// the scaled AC, with the padded columns and rows repeating the last
/// visible ones; clipped to `depth`.
fn write_plane<T: Sample>(
    plane: &mut CflChroma<'_, T>,
    stride: usize,
    ac: &[[i32; 32]; 32],
    alpha: i32,
    dc: i32,
    (xlim, ylim): (usize, usize),
    (w, h): (usize, usize),
    depth: BitDepth,
) {
    let row_at = |y: usize| plane.offset + y * stride;
    if alpha == 0 {
        for y in 0..h {
            let start = row_at(y);
            plane.buf[start..start + w].fill(T::from_i32(dc));
        }
        return;
    }
    for (y, ac_row) in ac.iter().enumerate().take(ylim) {
        let start = row_at(y);
        let row = &mut plane.buf[start..start + w];
        for (out, &a) in row.iter_mut().zip(&ac_row[..xlim]) {
            let scaled = ((alpha * a).unsigned_abs() + (1 << (AC_SHIFT - 1))) >> AC_SHIFT;
            let scaled = if alpha * a < 0 {
                -(scaled as i32)
            } else {
                scaled as i32
            };
            *out = clip_pixel(dc + scaled, depth);
        }
        let last = row[xlim - 1];
        row[xlim..].fill(last);
    }
    let last_row = row_at(ylim - 1);
    for y in ylim..h {
        plane.buf.copy_within(last_row..last_row + w, row_at(y));
    }
}
