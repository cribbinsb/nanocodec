//! The 2-D inverse driver (AV1 §7.13.3, libaom `inv_txfm2d_add_c`), generic
//! over [`I32Lanes`] and the transform's width and height.
//!
//! Like the square DCT drivers in `txfm::generic`: column-major
//! coefficients load lane-parallel for the row pass (`V::LANES` rows at a
//! time), which stores its output transposed into a row-major buffer; the
//! column pass then loads `V::LANES` columns at a time and adds whole output
//! rows. `V::LANES` must divide both sides (each ISA picks 4 lanes when a
//! side is 4).
//!
//! Row groups whose coefficients are all zero skip the row transform: every
//! 1-D inverse maps zeros to zeros, so their buffer rows are zero. Rows at
//! outside the EOB bound are cleared without loading their coefficients.
//! Rows beyond the coded height are never stored; the column pass feeds zeros.
//!
//! The functions are `unsafe` only because `V`'s instructions need its ISA
//! (see the `simd` module docs).

#![allow(clippy::inline_always)]

use core::mem::MaybeUninit;

use super::one_d::{iadst4, iadst8, iadst16, identity};
use super::{
    AV1_MAX_CODED_SIDE, Av1TxSize, COLUMN_SHIFT, COS_BIT, INV_SQRT2, Kernel1d, SQRT2_BITS,
};
use crate::simd::I32Lanes;
use crate::txfm::COSPI_12;
use crate::txfm::dct;
use crate::txfm::generic::store_pass;

/// Largest row-pass buffer: 32 coded rows of 64 outputs.
const MAX_BUFFER: usize = AV1_MAX_CODED_SIDE * 64;

/// `x` as a fixed-size array of `M` vectors; `x.len() == M` always holds at
/// the call sites (it is the const length being matched), so the check
/// folds away.
#[inline(always)]
fn fixed<V, const M: usize>(x: &[V]) -> &[V; M] {
    x.try_into().expect("1-D transform length")
}

#[inline(always)]
fn fixed_mut<V, const M: usize>(x: &mut [V]) -> &mut [V; M] {
    x.try_into().expect("1-D transform length")
}

/// The transform size with these sides.
const fn sides<const W: usize, const H: usize>() -> Av1TxSize {
    Av1TxSize {
        width_log2: W.trailing_zeros() as u8,
        height_log2: H.trailing_zeros() as u8,
    }
}

/// The `N`-point 1-D inverse `kind` of every lane of `x` into `y`; a flip
/// is applied by the caller. For `N == 64` the upper 32 inputs must be zero
/// (64-point zero-out), so the graph without them runs.
#[inline(always)]
fn transform_1d<V: I32Lanes, const N: usize>(kind: Kernel1d, x: &[V; N], y: &mut [V; N]) {
    match (kind, N) {
        (Kernel1d::Dct, 4) => dct::idct4(fixed(x), fixed_mut(y), &COSPI_12, COS_BIT),
        (Kernel1d::Dct, 8) => dct::idct8(fixed(x), fixed_mut(y), &COSPI_12, COS_BIT),
        (Kernel1d::Dct, 16) => dct::idct16(fixed(x), fixed_mut(y), &COSPI_12, COS_BIT),
        (Kernel1d::Dct, 32) => dct::idct32(fixed(x), fixed_mut(y), &COSPI_12, COS_BIT),
        (Kernel1d::Dct, 64) => dct::idct64_low32(fixed(x), fixed_mut(y), &COSPI_12, COS_BIT),
        (Kernel1d::Adst | Kernel1d::FlipAdst, 4) => iadst4(fixed(x), fixed_mut(y), COS_BIT),
        (Kernel1d::Adst | Kernel1d::FlipAdst, 8) => {
            iadst8(fixed(x), fixed_mut(y), &COSPI_12, COS_BIT);
        }
        (Kernel1d::Adst | Kernel1d::FlipAdst, 16) => {
            iadst16(fixed(x), fixed_mut(y), &COSPI_12, COS_BIT);
        }
        (Kernel1d::Identity, 4 | 8 | 16 | 32) => identity(x, y),
        _ => unreachable!("no {N}-point {kind:?} (checked by Av1TxType::fits)"),
    }
}

/// Inverse 2-D transform of a `W`x`H` block: `vertical` columns after
/// `horizontal` rows, column-major `coeffs` (`coded_width * coded_height`),
/// added to the prediction in `dst` (at `dst_stride`) with 8-bit clipping.
///
/// # Safety
///
/// `V`'s ISA must be available, and `V::LANES` must divide `W` and `H`.
#[inline(always)]
pub(super) unsafe fn inverse_add<V: I32Lanes, const W: usize, const H: usize>(
    vertical: Kernel1d,
    horizontal: Kernel1d,
    coeffs: &[i32],
    nonzero_rows: usize,
    dst: &mut [u8],
    dst_stride: usize,
) {
    let size = sides::<W, H>();
    let (coded_w, coded_h) = (size.coded_width(), size.coded_height());
    let active_h = nonzero_rows.div_ceil(V::LANES) * V::LANES;
    debug_assert!(active_h <= coded_h);
    let row_shift = const { sides::<W, H>().row_shift() };
    let scale_rows = size.is_two_to_one();
    let coeffs = &coeffs[..coded_w * coded_h];

    // Row pass: buffer row r holds row r's outputs, `buf[r * W + c]`.
    let mut buf = [MaybeUninit::<i32>::uninit(); MAX_BUFFER];
    let mut x = [V::splat(0); W];
    let mut y = [V::splat(0); W];
    for r0 in (0..coded_h).step_by(V::LANES) {
        // Retain the original constant trip count for dense transforms.
        // EOB skips coefficient loads and transforms of trailing row groups,
        // without duplicating the large DCT/ADST graphs. The first group is
        // always active (the dispatch already returned for EOB zero).
        if r0 != 0 && r0 >= active_h {
            y.fill(V::splat(0));
            store_pass::<V>(&y, r0, W, &mut buf);
            continue;
        }
        let mut any = V::splat(0);
        for (c, lane) in x[..coded_w].iter_mut().enumerate() {
            let v = V::load(&coeffs[c * coded_h + r0..]).clamp16();
            any = any.or(v);
            // The 16-bit clamp after the scaling (libaom `clamp_buf`) is
            // the identity here: |v| <= 2^15 shrinks.
            *lane = if scale_rows {
                v.mul_scalar(INV_SQRT2).round_shift(SQRT2_BITS)
            } else {
                v
            };
        }
        if any.count_nonzero() == 0 {
            y = [V::splat(0); W];
        } else {
            transform_1d::<V, W>(horizontal, &x, &mut y);
            for lane in &mut y {
                *lane = lane.round_shift(row_shift);
            }
            if horizontal == Kernel1d::FlipAdst {
                y.reverse();
            }
        }
        store_pass::<V>(&y, r0, W, &mut buf);
    }
    // SAFETY: the loop above wrote all coded rows (including zero rows),
    // all W columns; MaybeUninit<i32> has the layout of i32.
    let rows: &[i32] = unsafe { core::slice::from_raw_parts(buf.as_ptr().cast(), coded_h * W) };

    // Column pass.
    let mut x = [V::splat(0); H];
    let mut y = [V::splat(0); H];
    for c0 in (0..W).step_by(V::LANES) {
        for (r, lane) in x[..coded_h].iter_mut().enumerate() {
            *lane = V::load(&rows[r * W + c0..]).clamp16();
        }
        transform_1d::<V, H>(vertical, &x, &mut y);
        let flip = vertical == Kernel1d::FlipAdst;
        for (r, lane) in y.iter().enumerate() {
            let out_row = if flip { H - 1 - r } else { r };
            lane.round_shift(COLUMN_SHIFT)
                .add_to_u8(&mut dst[out_row * dst_stride + c0..]);
        }
    }
}

/// The residual of a `DCT_DCT` with only DC present, the same at every
/// position. The two cosine rotations and intervening row shift equal the
/// full graph for every 16-bit input. Keep our Q12 constants and rounding
/// (rather than dav1d's equivalent Q8 spelling). The design follows dav1d's
/// BSD-2 `itx_tmpl.c` DC shortcut; no code copied.
#[inline(always)]
pub(super) fn dc_residual(tx: Av1TxSize, coefficient: i32) -> i32 {
    let round = |v: i32, shift: u32| {
        if shift == 0 {
            v
        } else {
            (v + (1 << (shift - 1))) >> shift
        }
    };
    let mut dc = coefficient.clamp(i32::from(i16::MIN), i32::from(i16::MAX));
    if tx.is_two_to_one() {
        dc = round(dc * INV_SQRT2, SQRT2_BITS);
    }
    dc = round(round(dc * COSPI_12[32], COS_BIT), tx.row_shift());
    round(round(dc * COSPI_12[32], COS_BIT), COLUMN_SHIFT)
}
