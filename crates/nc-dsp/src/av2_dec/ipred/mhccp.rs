//! Multi-hypothesis cross-component prediction (MHCCP), 4:2:0: chroma is a
//! three-term linear model of the downsampled luma, its rounded square
//! (taken from a neighbour in the chosen direction) and a constant, fitted
//! to the block's edges by a 3x3 least-squares solve in fixed point.
//!
//! The decoder runs four steps: [`cfl_mh_gen_y_420`] downsamples the luma
//! block and its edges into the scratch buffer, [`cfl_mh_gen_mat`] builds
//! the normal matrix from the edge luma, [`cfl_mh_calc_alphas`] solves it
//! against each chroma plane's edge, and [`cfl_mh_pred`] predicts.
//!
//! Ported from dav2d `src/ipred_tmpl.c` (`cfl_mh_gen_y_420_c`,
//! `cfl_mh_gen_mat_c`, `get_div_scale_sh`, `mul32`, `cfl_mh_calc_alphas_c`,
//! `cfl_mh_pred_c`; commit 237229e, BSD-2-Clause, see
//! `docs/licenses/dav2d`), bit-exact with AVM v1.0.0.

use super::tables::{DIV_SCALE_SH_BIAS, DIV_SCALE_SH_COEFW, DIV_SCALE_SH_OFFSET};
use super::{
    CFL_FLT_TYPE_GAUSS, CFL_FLT_TYPE_UNIFORM, CFL_FLT_TYPE_VSTRIP, CFL_HAS_LEFT, CFL_HAS_TOP,
    CflMhDir, SampleRef, clip_pixel,
};

/// Stride of the downsampled top rows in the luma scratch buffer.
pub const CFL_MH_TOP_STRIDE: usize = 64;
/// Luma scratch buffer size: two top rows, the 32x32 block, the left
/// columns and padding (`CFL_MH_MAX_LUMA_SIZE`).
pub const CFL_MH_MAX_LUMA_SIZE: usize = 1220;
/// Most edge samples of the fit (`CFL_MH_MAX_EDGE_SAMPLES`).
pub const CFL_MH_MAX_EDGE_SAMPLES: usize = 386;
/// Offset of the left columns from the block in the luma scratch buffer.
const LEFT_OFFSET: usize = 32 * 32;
/// Sample bit depth.
const BIT_DEPTH: u32 = 8;
/// Half the sample range, the constant term's unit.
const MID: i32 = 1 << (BIT_DEPTH - 1);
/// The fit's matrix and right-hand side are scaled to about this many bits.
const MATRIX_BITS: i32 = 22;
/// Regularisation added to the matrix diagonal.
const DIAGONAL_BIAS: i32 = 2 << (BIT_DEPTH - 8);
/// Fraction bits of the model coefficients.
const ALPHA_BITS: i32 = 16;
/// Fraction bits of a normalised divisor.
const DIVISOR_BITS: i32 = 14;
/// Bits of the divisor's polynomial region index.
const REGION_SHIFT: i32 = 11;
/// Bits a product may keep in [`mul32`].
const MUL_BITS: u32 = 29;

/// Downsampled squared luma: `(v * v + MID) >> BIT_DEPTH` (`SQRND`).
fn square(v: u8) -> i32 {
    (i32::from(v) * i32::from(v) + MID) >> BIT_DEPTH
}

/// The luma rows of a downsampling: the 2x2 window's top row, the row
/// below it `below` away, and the row the GAUSS filter's top tap reads.
#[derive(Clone, Copy)]
struct Window<'a> {
    row: SampleRef<'a>,
    below: isize,
    above_row: SampleRef<'a>,
}

impl Window<'_> {
    /// The downsampled luma at luma column `col` (left tap at `left`).
    fn sample(self, filter_type: u32, col: isize, left: isize) -> u8 {
        let px = |i: isize| self.row.wide(i);
        let below = self.below;
        let right = col + 1;
        let value = match filter_type {
            CFL_FLT_TYPE_UNIFORM => {
                (px(col) + px(right) + px(below + col) + px(below + right)) >> 2
            }
            CFL_FLT_TYPE_VSTRIP => {
                (px(left)
                    + 2 * px(col)
                    + px(right)
                    + px(below + left)
                    + 2 * px(below + col)
                    + px(below + right))
                    >> 3
            }
            _ => {
                (px(left) + 4 * px(col) + px(right) + self.above_row.wide(col) + px(below + col))
                    >> 3
            }
        };
        value as u8
    }
}

/// Downsamples the luma for MHCCP into `dst` (`cfl_mh_gen_y_420_*`): the
/// top rows (`needed_top_cols` each, one or two by the direction, 64 apart),
/// then the `w` x `h` block (rows `w` apart), then from `LEFT_OFFSET` past
/// the block the left columns (one or two per row, `needed_left_rows`).
///
/// `src` is the luma block (rows `src_stride` apart); `top_sb_edge` the
/// pre-filter luma row above at a superblock's top, at the block's first
/// column. `flags` holds [`CFL_HAS_TOP`], [`CFL_HAS_LEFT`] and the
/// direction; `filter_type` is the sequence's `CFL_FLT_TYPE_*`.
pub fn cfl_mh_gen_y_420(
    dst: &mut [u8],
    src: SampleRef<'_>,
    top_sb_edge: Option<SampleRef<'_>>,
    src_stride: usize,
    needed_top_cols: i32,
    needed_left_rows: i32,
    w: usize,
    h: usize,
    flags: u32,
    filter_type: u32,
) {
    debug_assert!(filter_type <= CFL_FLT_TYPE_GAUSS);
    let has_t = flags & CFL_HAS_TOP != 0;
    let has_l = flags & CFL_HAS_LEFT != 0;
    let dir = CflMhDir::from_flags(flags);
    let n_left_cols = if has_l {
        1 + usize::from(dir == CflMhDir::Left)
    } else {
        0
    };
    let n_top_rows = if has_t {
        1 + usize::from(dir == CflMhDir::Top)
    } else {
        0
    };
    let stride = src_stride as isize;
    // dav2d's counts go negative for an absent edge.
    let needed_top_cols = needed_top_cols.max(0) as usize;
    let needed_left_rows = needed_left_rows.max(0) as usize;
    let left_reach = 2 * n_left_cols as isize;
    // A single left column's left tap reaches one past it; two columns
    // clamp at the first.
    let edge_left_tap = |c: isize| {
        if n_left_cols & 1 == 1 {
            c - 1
        } else {
            (c - 1).max(0)
        }
    };

    if has_t {
        let (mut row, b) = match top_sb_edge {
            Some(edge) => (edge, 0),
            None => (src.shifted(-2 * n_top_rows as isize * stride), stride),
        };
        row = row.shifted(-left_reach);
        let mut above = if n_top_rows == 1 { -b } else { 0 };
        let start = usize::from(!has_l && dir == CflMhDir::Center);
        for y in 0..n_top_rows {
            let window = Window {
                row,
                below: b,
                above_row: row.shifted(above),
            };
            let out = &mut dst[y * CFL_MH_TOP_STRIDE..][..needed_top_cols];
            for (x, o) in (start..).zip(out.iter_mut()) {
                let c = 2 * x as isize;
                *o = window.sample(filter_type, c, edge_left_tap(c));
            }
            row = row.shifted(2 * b);
            above = -b;
        }
    }
    let block = n_top_rows * CFL_MH_TOP_STRIDE;

    let ystart = usize::from(!has_t && dir == CflMhDir::Center);
    let mut row = src.shifted(2 * ystart as isize * stride - left_reach);
    let mut above_row = match top_sb_edge {
        Some(edge) if has_t && ystart == 0 => edge.shifted(-left_reach),
        _ if has_t || ystart == 1 => row.shifted(-stride),
        _ => row,
    };
    let left = &mut dst[block + LEFT_OFFSET..];
    for out in left
        .chunks_exact_mut(n_left_cols.max(1))
        .take(needed_left_rows)
    {
        if n_left_cols == 0 {
            break;
        }
        let window = Window {
            row,
            below: stride,
            above_row,
        };
        for (x, o) in out.iter_mut().enumerate() {
            let c = 2 * x as isize;
            *o = window.sample(filter_type, c, edge_left_tap(c));
        }
        row = row.shifted(2 * stride);
        above_row = row.shifted(-stride);
    }

    let mut row = src;
    let mut above_row = match top_sb_edge {
        Some(edge) if has_t => edge,
        _ if has_t => src.shifted(-stride),
        _ => src,
    };
    for out in dst[block..].chunks_exact_mut(w).take(h) {
        let window = Window {
            row,
            below: stride,
            above_row,
        };
        for (x, o) in out.iter_mut().enumerate() {
            let c = 2 * x as isize;
            let l = if has_l { c - 1 } else { (c - 1).max(0) };
            *o = window.sample(filter_type, c, l);
        }
        row = row.shifted(2 * stride);
        above_row = row.shifted(-stride);
    }
}

/// The shift that scales sums over `n` samples to [`MATRIX_BITS`].
fn matrix_shift(n: usize) -> i32 {
    let n = n as u32;
    let log2 = n.ilog2() as i32;
    MATRIX_BITS - 2 * BIT_DEPTH as i32 - log2 - i32::from(!n.is_power_of_two())
}

/// `v << shift` for a nonnegative shift, else `v >> -shift`.
fn scale(v: i32, shift: i32) -> i32 {
    if shift >= 0 { v << shift } else { v >> -shift }
}

/// Builds the normal matrix of the fit from the downsampled luma `y` (laid
/// out by [`cfl_mh_gen_y_420`]) over `refw` top and `refh` left edge
/// samples (`cfl_mh_gen_mat`); `imat` keeps each sample's two luma terms
/// for [`cfl_mh_calc_alphas`].
pub fn cfl_mh_gen_mat(
    mat: &mut [[i32; 4]; 3],
    imat: &mut [[u8; CFL_MH_MAX_EDGE_SAMPLES]; 2],
    y: &[u8],
    refw: i32,
    refh: i32,
    flags: u32,
) {
    let has_t = flags & CFL_HAS_TOP != 0;
    let has_l = flags & CFL_HAS_LEFT != 0;
    let dir = CflMhDir::from_flags(flags);
    // dav2d passes -1 for an absent edge's count.
    let refw = refw.max(0) as usize;
    let refh = refh.max(0) as usize;
    let dir_t = usize::from(dir == CflMhDir::Top);
    let dir_l = usize::from(dir == CflMhDir::Left);
    let n_top = if has_t { 1 + dir_t } else { 0 };
    let n_left = if has_l { 1 + dir_l } else { 0 };
    let left = &y[n_top * CFL_MH_TOP_STRIDE + LEFT_OFFSET..];

    *mat = [[0; 4]; 3];
    let mut n = 0;
    let mut add = |v0: u8, v1: i32| {
        imat[0][n] = v0;
        imat[1][n] = v1 as u8;
        let v0 = i32::from(v0);
        mat[0][0] += v0 * v0;
        mat[0][1] += v0 * v1;
        mat[0][2] += v0 << (BIT_DEPTH - 1);
        mat[1][1] += v1 * v1;
        mat[1][2] += v1 << (BIT_DEPTH - 1);
        n += 1;
    };
    if has_t {
        let start = usize::from(!has_l) * dir_t;
        for i in start..start + refw {
            add(y[i], square(y[dir_t * CFL_MH_TOP_STRIDE + i + dir_l]));
        }
    }
    if has_l {
        // The top-left pair, when the top edge leads, pairs the second top
        // row's first sample with the first left sample.
        let extra_tl = usize::from(has_t) * dir_t;
        if extra_tl == 1 {
            add(y[CFL_MH_TOP_STRIDE], square(left[0]));
        }
        let start = dir_l * usize::from(!has_t);
        for i in start..start + refh - extra_tl {
            add(left[i * n_left], square(left[(i + dir_t) * n_left + dir_l]));
        }
    }
    mat[2][2] = (n as i32) << (2 * (BIT_DEPTH - 1));

    let shift = matrix_shift(n);
    for (i, row) in mat.iter_mut().enumerate() {
        for v in &mut row[i..3] {
            *v = scale(*v, shift);
        }
        row[i] += DIAGONAL_BIAS;
    }
    mat[1][0] = mat[0][1];
    mat[2][0] = mat[0][2];
    mat[2][1] = mat[1][2];
}

/// A reciprocal of `d` as `(scale, shift)`: `x / d ≈ x * scale >> shift`
/// (`get_div_scale_sh`), from a quadratic in each of eight regions of the
/// normalised divisor.
fn div_scale_shift(d: i32) -> (i32, i32) {
    let d = d.unsigned_abs().max(1) as i32;
    let sh = d.ilog2() as i32;
    let nsh = sh - DIVISOR_BITS;
    let mut d = if nsh >= 0 {
        let round = if nsh > 0 { 1 << (nsh - 1) } else { 0 };
        (d + round) >> nsh
    } else {
        d << -nsh
    };
    // The rounding may carry past [1, 2); clamp, then keep the fraction.
    d = d.clamp(1, 0x7fff) & ((1 << DIVISOR_BITS) - 1);
    let region = (d >> REGION_SHIFT) as usize;
    let d = d - i32::from(DIV_SCALE_SH_OFFSET[region]);
    let coef = i32::from(DIV_SCALE_SH_COEFW[region]);
    let scale = (((coef * ((d * d) >> DIVISOR_BITS)) >> 8) - (d >> 1)
        + i32::from(DIV_SCALE_SH_BIAS[region]))
        << 2;
    (scale, sh)
}

/// `(a * b) >> sh` rounded half away from zero, with both operands
/// pre-shifted so the product fits 32 bits (`mul32`), as AVM computes it.
fn mul32(a: i32, b: i32, sh: i32) -> i32 {
    let bits = |v: i32| (v.unsigned_abs() | 1).ilog2() + 1;
    let drop = (bits(a) + bits(b)).saturating_sub(MUL_BITS);
    let a_shift = drop >> 1;
    let b_shift = drop - a_shift;
    let adj = sh - drop as i32;
    let product = (a >> a_shift).wrapping_mul(b >> b_shift);
    if adj <= 0 {
        return product;
    }
    let bias = 1u32 << (adj - 1);
    let magnitude = (product.unsigned_abs().wrapping_add(bias) >> adj) as i32;
    if product >= 0 { magnitude } else { -magnitude }
}

/// Solves the fit for one chroma plane (`cfl_mh_calc_alphas`): the edge
/// chroma of the block at `c` in `chroma` (rows `stride` apart; the row
/// above from `top_sb_edge` at a superblock's top, at the block's first
/// column) against `mat` and `imat` from [`cfl_mh_gen_mat`]. `alpha[3]` is
/// unused.
pub fn cfl_mh_calc_alphas(
    alpha: &mut [i32; 4],
    chroma: SampleRef<'_>,
    top_sb_edge: Option<SampleRef<'_>>,
    stride: usize,
    refw: i32,
    refh: i32,
    mat: &[[i32; 4]; 3],
    imat: &[[u8; CFL_MH_MAX_EDGE_SAMPLES]; 2],
    flags: u32,
) {
    let has_t = flags & CFL_HAS_TOP != 0;
    let has_l = flags & CFL_HAS_LEFT != 0;
    let stride = stride as isize;
    let refw = refw.max(0) as usize;
    let refh = refh.max(0) as usize;
    let mut rhs = [0i32; 3];
    let mut n = 0;
    let mut add = |c: u8| {
        let c = i32::from(c);
        rhs[0] += i32::from(imat[0][n]) * c;
        rhs[1] += i32::from(imat[1][n]) * c;
        rhs[2] += c << (BIT_DEPTH - 1);
        n += 1;
    };
    if has_t {
        let top = top_sb_edge
            .unwrap_or(chroma.shifted(-stride))
            .shifted(-isize::from(has_l));
        let start = usize::from(!has_l);
        for i in start..start + refw {
            add(top.at(i as isize));
        }
    }
    if has_l {
        let start = usize::from(!has_t);
        for i in start..start + refh {
            add(chroma.at(i as isize * stride - 1));
        }
    }
    let shift = matrix_shift(n);
    let [mut a0, mut a1, mut a2] = rhs.map(|v| scale(v, shift));

    // Gaussian elimination with the reciprocal approximation.
    let (sc, sh) = div_scale_shift(mat[0][0]);
    let r01 = mul32(mat[0][1], sc, sh);
    let r02 = mul32(mat[0][2], sc, sh);
    a0 = mul32(a0, sc, sh);
    let r11 = mat[1][1] - mul32(mat[1][0], r01, ALPHA_BITS);
    let mut r12 = mat[1][2] - mul32(mat[1][0], r02, ALPHA_BITS);
    a1 -= mul32(mat[1][0], a0, ALPHA_BITS);
    let r21 = mat[2][1] - mul32(mat[2][0], r01, ALPHA_BITS);
    let mut r22 = mat[2][2] - mul32(mat[2][0], r02, ALPHA_BITS);
    a2 -= mul32(mat[2][0], a0, ALPHA_BITS);

    let (sc, sh) = div_scale_shift(r11);
    r12 = mul32(r12, sc, sh);
    a1 = mul32(a1, sc, sh);
    r22 -= mul32(r21, r12, ALPHA_BITS);
    a2 -= mul32(r21, a1, ALPHA_BITS);

    let (sc, sh) = div_scale_shift(r22);
    a2 = mul32(a2, sc, sh);
    a1 -= mul32(r12, a2, ALPHA_BITS);
    a0 -= mul32(r01, a1, ALPHA_BITS) + mul32(r02, a2, ALPHA_BITS);
    *alpha = [a0, a1, a2, 0];
}

/// Predicts the `w` x `h` chroma block at `offset` in `dst` (rows `stride`
/// apart) from the downsampled luma `src` (the block in the scratch buffer,
/// `n_top_rows * CFL_MH_TOP_STRIDE` into it) and the plane's `alpha`
/// (`cfl_mh_pred_{c,t,l}`; the direction is `dir`).
pub fn cfl_mh_pred(
    dir: CflMhDir,
    dst: &mut [u8],
    offset: usize,
    stride: usize,
    src: SampleRef<'_>,
    w: usize,
    h: usize,
    alpha: &[i32; 4],
    flags: u32,
) {
    let has_t = flags & CFL_HAS_TOP != 0;
    let has_l = flags & CFL_HAS_LEFT != 0;
    let constant = mul32(alpha[2], MID, ALPHA_BITS);
    let model = |v0: u8, v1: u8| {
        clip_pixel(
            mul32(alpha[0], i32::from(v0), ALPHA_BITS)
                + mul32(alpha[1], square(v1), ALPHA_BITS)
                + constant,
        )
    };
    let left = src.shifted(LEFT_OFFSET as isize);
    for y in 0..h {
        let row = src.shifted((y * w) as isize);
        let out = &mut dst[offset + y * stride..][..w];
        for (x, o) in out.iter_mut().enumerate() {
            let xi = x as isize;
            // The paired neighbour: above (from the top rows on the first
            // row), left (from the left columns in the first column), or
            // the sample itself.
            let v0 = match dir {
                CflMhDir::Top if y == 0 && has_t => {
                    row.at(xi - CFL_MH_TOP_STRIDE as isize + isize::from(has_l))
                }
                CflMhDir::Top if y == 0 => row.at(xi),
                CflMhDir::Top => row.at(xi - w as isize),
                CflMhDir::Left if x == 0 && has_l => left.at(y as isize * 2 + 1),
                CflMhDir::Left => row.at((xi - 1).max(0)),
                CflMhDir::Center => row.at(xi),
            };
            *o = model(v0, row.at(xi));
        }
    }
}
