//! The 2-D inverse driver (AV1 §7.13.3, libaom `inv_txfm2d_add_c`), generic
//! over [`I32Lanes`], bit-depth stage ranges, and the transform's width and
//! height — shared across `u8` (8-bit) and `u16` (10- and 12-bit).
//!
//! Column-major coefficients load lane-parallel for the row pass (`V::LANES`
//! rows at a time), which stores its output transposed into a row-major
//! buffer; the column pass then loads `V::LANES` columns at a time and adds
//! whole output rows. `V::LANES` must divide both sides (each ISA picks 4
//! lanes when a side is 4).
//!
//! EOB and sparsity shortcuts (following dav1d 1.5.4 `src/itx_tmpl.c`,
//! BSD-2-Clause; see `THIRD_PARTY.md`):
//! - Row groups at or beyond `active_h` (`nonzero_rows` rounded up to
//!   `V::LANES`) are never loaded, transformed, or written to `buf`; the
//!   column pass reads only `0..nonzero_rows`.
//! - Active row groups whose loaded vector is all-zero skip the 1-D row
//!   transform.
//! - 32- and 64-point `DCT` passes whose nonzero inputs fit in the low 8 or
//!   16 frequencies run the pruned `idct{32,64}_low{8,16}` graphs.

#![allow(clippy::inline_always)]

use core::mem::MaybeUninit;

use nc_common::{BitDepth, Sample};

use super::one_d::{iadst4, iadst8, iadst16, identity};
use super::{
    AV1_MAX_CODED_SIDE, Av1TxSize, COLUMN_SHIFT, COS_BIT, INV_SQRT2, Kernel1d, SQRT2, SQRT2_BITS,
};
use crate::simd::I32Lanes;
use crate::txfm::COSPI_12;
use crate::txfm::dct;
use crate::txfm::generic::store_pass;
use crate::txfm::inverse_lanes::InverseLanes;
use crate::txfm::narrow_inverse::NarrowCoefficient;

/// Largest row-pass buffer: 32 coded rows of 64 outputs.
const MAX_BUFFER: usize = AV1_MAX_CODED_SIDE * 64;

/// Largest `|x|` for which `x * INV_SQRT2 + 2048` fits signed `i32`, while
/// exceeding the 20-bit (12-bit video) stage clamp after `>> 12`:
/// `(741_533 * 2896 + 2048) >> 12 == 524_336 > 524_287`.
const PRESCALE_I32_CAP: i32 = 741_533;

#[inline(always)]
fn fixed<V, const M: usize>(x: &[V]) -> &[V; M] {
    x.try_into().expect("1-D transform length")
}

#[inline(always)]
fn fixed_mut<V, const M: usize>(x: &mut [V]) -> &mut [V; M] {
    x.try_into().expect("1-D transform length")
}

const fn sides<const W: usize, const H: usize>() -> Av1TxSize {
    Av1TxSize {
        width_log2: W.trailing_zeros() as u8,
        height_log2: H.trailing_zeros() as u8,
    }
}

/// Clamp every lane of `v` to signed `BITS` bits (`[-2^(BITS-1), 2^(BITS-1)-1]`).
#[inline(always)]
pub(super) fn clamp_bits<V: I32Lanes, const BITS: u32>(v: V) -> V {
    if BITS == 16 {
        v.clamp16()
    } else {
        let min = -(1i32 << (BITS - 1));
        let max = (1i32 << (BITS - 1)) - 1;
        v.max(V::splat(min)).min(V::splat(max))
    }
}

/// Transparent adapter giving an [`I32Lanes`] vector a `BITS`-bit stage clamp
/// (`16` at 8-bit and 10-bit column pass, `18` at 10-bit row / 12-bit column,
/// `20` at 12-bit row).
#[repr(transparent)]
#[derive(Clone, Copy)]
struct Clamped<V, const BITS: u32>(V);

impl<V: I32Lanes, const BITS: u32> InverseLanes for Clamped<V, BITS> {
    #[inline]
    fn splat(value: i32) -> Self {
        Self(V::splat(value))
    }
    #[inline]
    fn add(self, other: Self) -> Self {
        Self(self.0.add(other.0))
    }
    #[inline]
    fn sub(self, other: Self) -> Self {
        Self(self.0.sub(other.0))
    }
    #[inline]
    fn half_btf(w0: i32, a: Self, w1: i32, b: Self, bit: u32) -> Self {
        Self(V::half_btf(w0, a.0, w1, b.0, bit))
    }
    #[inline]
    fn mul_round(w: i32, a: Self, bit: u32) -> Self {
        Self(a.0.mul_scalar(w).round_shift(bit))
    }
    #[inline]
    fn clamp16(self) -> Self {
        Self(clamp_bits::<V, BITS>(self.0))
    }
}

#[inline(always)]
fn as_clamped<V, const BITS: u32, const N: usize>(x: &[V; N]) -> &[Clamped<V, BITS>; N] {
    // SAFETY: `Clamped<V, BITS>` is `#[repr(transparent)]` over `V`.
    unsafe { &*x.as_ptr().cast::<[Clamped<V, BITS>; N]>() }
}

#[inline(always)]
fn as_clamped_mut<V, const BITS: u32, const N: usize>(
    y: &mut [V; N],
) -> &mut [Clamped<V, BITS>; N] {
    // SAFETY: `Clamped<V, BITS>` is `#[repr(transparent)]` over `V`.
    unsafe { &mut *y.as_mut_ptr().cast::<[Clamped<V, BITS>; N]>() }
}

/// The `N`-point 1-D inverse `kind` of every lane of `x` into `y`, with
/// stage clamps of `BITS` signed bits, pruning 32- and 64-point `DCT` passes
/// when `active <= 8` or `active <= 16`.
#[inline(always)]
fn transform_1d_clamped<V: I32Lanes, const BITS: u32, const N: usize>(
    kind: Kernel1d,
    active: usize,
    x: &[V; N],
    y: &mut [V; N],
) {
    if matches!(kind, Kernel1d::Adst | Kernel1d::FlipAdst) && N == 4 {
        iadst4(fixed(x), fixed_mut(y), COS_BIT);
        return;
    }
    let xc = as_clamped::<V, BITS, N>(x);
    let yc = as_clamped_mut::<V, BITS, N>(y);
    match (kind, N) {
        (Kernel1d::Dct, 4) => dct::idct4(fixed(xc), fixed_mut(yc), &COSPI_12, COS_BIT),
        (Kernel1d::Dct, 8) => dct::idct8(fixed(xc), fixed_mut(yc), &COSPI_12, COS_BIT),
        (Kernel1d::Dct, 16) => dct::idct16(fixed(xc), fixed_mut(yc), &COSPI_12, COS_BIT),
        (Kernel1d::Dct, 32) => {
            if active <= 8 {
                dct::idct32_low8(fixed(xc), fixed_mut(yc), &COSPI_12, COS_BIT);
            } else if active <= 16 {
                dct::idct32_low16(fixed(xc), fixed_mut(yc), &COSPI_12, COS_BIT);
            } else {
                dct::idct32(fixed(xc), fixed_mut(yc), &COSPI_12, COS_BIT);
            }
        }
        (Kernel1d::Dct, 64) => {
            if active <= 8 {
                dct::idct64_low8(fixed(xc), fixed_mut(yc), &COSPI_12, COS_BIT);
            } else if active <= 16 {
                dct::idct64_low16(fixed(xc), fixed_mut(yc), &COSPI_12, COS_BIT);
            } else {
                dct::idct64_low32(fixed(xc), fixed_mut(yc), &COSPI_12, COS_BIT);
            }
        }
        (Kernel1d::Adst | Kernel1d::FlipAdst, 8) => {
            iadst8(fixed(xc), fixed_mut(yc), &COSPI_12, COS_BIT);
        }
        (Kernel1d::Adst | Kernel1d::FlipAdst, 16) => {
            iadst16(fixed(xc), fixed_mut(yc), &COSPI_12, COS_BIT);
        }
        (Kernel1d::Identity, 4 | 8 | 16 | 32) => identity(xc, yc),
        _ => unreachable!("no {N}-point {kind:?} (checked by Av1TxType::fits)"),
    }
}

/// Destination sample addition for [`inverse_add_lanes`].
pub(super) trait InverseSink: Sample {
    fn add_residual<V: I32Lanes>(residual: V, dst: &mut [Self], max_val: i32);
}

impl InverseSink for u8 {
    #[inline(always)]
    fn add_residual<V: I32Lanes>(residual: V, dst: &mut [u8], _max_val: i32) {
        residual.add_to_u8(dst);
    }
}

impl InverseSink for u16 {
    #[inline(always)]
    fn add_residual<V: I32Lanes>(residual: V, dst: &mut [u16], max_val: i32) {
        residual.add_to_u16(dst, max_val);
    }
}

/// Shared 2-D inverse driver for `u8` (`ROW=16, COL=16`) and `u16`
/// (`ROW=18, COL=16` at 10-bit; `ROW=20, COL=18` at 12-bit).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn inverse_add_lanes<
    V: I32Lanes,
    C: NarrowCoefficient,
    S: InverseSink,
    const ROW: u32,
    const COL: u32,
    const W: usize,
    const H: usize,
>(
    vertical: Kernel1d,
    horizontal: Kernel1d,
    coeffs: &[C],
    nonzero_rows: usize,
    dst: &mut [S],
    dst_stride: usize,
    max_val: i32,
) {
    let size = sides::<W, H>();
    let (coded_w, coded_h) = (size.coded_width(), size.coded_height());
    let rows_bound = nonzero_rows.clamp(1, coded_h);
    let active_h = (rows_bound.div_ceil(V::LANES) * V::LANES).min(coded_h);
    let row_shift = const { sides::<W, H>().row_shift() };
    let scale_rows = size.is_two_to_one();
    let coeffs = &coeffs[..coded_w * coded_h];

    let mut buf = [MaybeUninit::<i32>::uninit(); MAX_BUFFER];
    let mut x = [V::splat(0); W];
    let mut y = [V::splat(0); W];
    for r0 in (0..active_h).step_by(V::LANES) {
        let mut any_lo = V::splat(0);
        let mut any_mid = V::splat(0);
        let mut any_hi = V::splat(0);
        for (c, lane) in x[..coded_w].iter_mut().enumerate() {
            let raw = C::wide::<V>(&coeffs[c * coded_h + r0..]);
            let v = if ROW == 16 {
                let clamped = raw.clamp16();
                if scale_rows {
                    clamped.mul_scalar(INV_SQRT2).round_shift(SQRT2_BITS)
                } else {
                    clamped
                }
            } else if scale_rows {
                let capped = raw
                    .max(V::splat(-PRESCALE_I32_CAP))
                    .min(V::splat(PRESCALE_I32_CAP));
                clamp_bits::<V, ROW>(capped.mul_scalar(INV_SQRT2).round_shift(SQRT2_BITS))
            } else {
                clamp_bits::<V, ROW>(raw)
            };
            if c < 8 {
                any_lo = any_lo.or(v);
            } else if c < 16 {
                any_mid = any_mid.or(v);
            } else {
                any_hi = any_hi.or(v);
            }
            *lane = v;
        }
        let any = any_lo.or(any_mid).or(any_hi);
        if any.count_nonzero() == 0 {
            y = [V::splat(0); W];
        } else {
            let active_w = if W >= 32 && horizontal == Kernel1d::Dct {
                if any_hi.count_nonzero() != 0 {
                    32
                } else if any_mid.count_nonzero() != 0 {
                    16
                } else {
                    8
                }
            } else {
                coded_w
            };
            transform_1d_clamped::<V, ROW, W>(horizontal, active_w, &x, &mut y);
            for lane in &mut y {
                *lane = lane.round_shift(row_shift);
            }
            if horizontal == Kernel1d::FlipAdst {
                y.reverse();
            }
        }
        store_pass::<V>(&y, r0, W, &mut buf);
    }
    // SAFETY: the loop above initialized `active_h` complete rows of `W` `i32`s.
    let rows: &[i32] = unsafe { core::slice::from_raw_parts(buf.as_ptr().cast(), active_h * W) };

    let mut x = [V::splat(0); H];
    let mut y = [V::splat(0); H];
    for c0 in (0..W).step_by(V::LANES) {
        for (r, lane) in x[..rows_bound].iter_mut().enumerate() {
            *lane = clamp_bits::<V, COL>(V::load(&rows[r * W + c0..]));
        }
        transform_1d_clamped::<V, COL, H>(vertical, rows_bound, &x, &mut y);
        let flip = vertical == Kernel1d::FlipAdst;
        for (r, lane) in y.iter().enumerate() {
            let out_row = if flip { H - 1 - r } else { r };
            S::add_residual(
                lane.round_shift(COLUMN_SHIFT),
                &mut dst[out_row * dst_stride + c0..],
                max_val,
            );
        }
    }
}

/// 8-bit 2-D inverse transform entry point (`ROW = 16, COL = 16`).
///
/// # Safety
///
/// `V`'s ISA must be available, and `V::LANES` must divide `W` and `H`.
#[inline(always)]
pub(super) unsafe fn inverse_add<
    V: I32Lanes,
    C: NarrowCoefficient,
    const W: usize,
    const H: usize,
>(
    vertical: Kernel1d,
    horizontal: Kernel1d,
    coeffs: &[C],
    nonzero_rows: usize,
    dst: &mut [u8],
    dst_stride: usize,
) {
    // SAFETY: forwarded contract.
    unsafe {
        inverse_add_lanes::<V, C, u8, 16, 16, W, H>(
            vertical,
            horizontal,
            coeffs,
            nonzero_rows,
            dst,
            dst_stride,
            255,
        );
    }
}

#[inline(always)]
fn round_i64(value: i64, shift: u32) -> i32 {
    if shift == 0 {
        value as i32
    } else {
        ((value + (1 << (shift - 1))) >> shift) as i32
    }
}

#[inline(always)]
fn clamp_signed_bits(value: i32, bits: u32) -> i32 {
    let min = -(1i32 << (bits - 1));
    let max = (1i32 << (bits - 1)) - 1;
    value.clamp(min, max)
}

/// The residual of an 8-bit `DCT_DCT` with only DC present.
#[inline(always)]
pub(super) fn dc_residual(tx: Av1TxSize, coefficient: i32) -> i32 {
    let mut dc = coefficient.clamp(i32::from(i16::MIN), i32::from(i16::MAX));
    if tx.is_two_to_one() {
        dc = round_i64(i64::from(dc) * i64::from(INV_SQRT2), SQRT2_BITS);
    }
    dc = round_i64(
        i64::from(round_i64(i64::from(dc) * i64::from(COSPI_12[32]), COS_BIT)),
        tx.row_shift(),
    );
    round_i64(
        i64::from(round_i64(i64::from(dc) * i64::from(COSPI_12[32]), COS_BIT)),
        COLUMN_SHIFT,
    )
}

/// The residual of a `DCT_DCT` with only DC present at high bit depth
/// (`10` or `12` bits), matching `high::inverse_add_at` bit-for-bit.
#[inline(always)]
pub(crate) fn dc_residual_high(tx: Av1TxSize, coefficient: i32, depth: BitDepth) -> i32 {
    let row_bits = depth.bits() + 8;
    let col_bits = (depth.bits() + 6).max(16);
    let prescaled = if tx.is_two_to_one() {
        round_i64(i64::from(coefficient) * i64::from(INV_SQRT2), SQRT2_BITS)
    } else {
        coefficient
    };
    let dc_row = clamp_signed_bits(prescaled, row_bits);
    let after_row = round_i64(
        i64::from(round_i64(
            i64::from(dc_row) * i64::from(COSPI_12[32]),
            COS_BIT,
        )),
        tx.row_shift(),
    );
    let dc_col = clamp_signed_bits(after_row, col_bits);
    round_i64(
        i64::from(round_i64(
            i64::from(dc_col) * i64::from(COSPI_12[32]),
            COS_BIT,
        )),
        COLUMN_SHIFT,
    )
}

#[inline(always)]
fn identity_1d_scalar(side: usize, x: i32) -> i32 {
    match side {
        4 => round_i64(i64::from(x) * i64::from(SQRT2), SQRT2_BITS),
        8 => x.wrapping_add(x),
        16 => round_i64(i64::from(x) * i64::from(2 * SQRT2), SQRT2_BITS),
        _ => x.wrapping_mul(4),
    }
}

/// Single-sample `(0, 0)` residual of an `IDTX` block with `eob == 1` (all
/// other positions in `W x H` receive zero residual).
#[inline(always)]
pub(crate) fn idtx_dc_residual(tx: Av1TxSize, coefficient: i32, depth: BitDepth) -> i32 {
    let row_bits = depth.bits() + 8;
    let col_bits = (depth.bits() + 6).max(16);
    let dc_row = if depth == BitDepth::EIGHT {
        let clamped = coefficient.clamp(i32::from(i16::MIN), i32::from(i16::MAX));
        if tx.is_two_to_one() {
            round_i64(i64::from(clamped) * i64::from(INV_SQRT2), SQRT2_BITS)
        } else {
            clamped
        }
    } else {
        let prescaled = if tx.is_two_to_one() {
            round_i64(i64::from(coefficient) * i64::from(INV_SQRT2), SQRT2_BITS)
        } else {
            coefficient
        };
        clamp_signed_bits(prescaled, row_bits)
    };
    let after_row = round_i64(
        i64::from(identity_1d_scalar(tx.width(), dc_row)),
        tx.row_shift(),
    );
    let dc_col = clamp_signed_bits(after_row, col_bits);
    round_i64(
        i64::from(identity_1d_scalar(tx.height(), dc_col)),
        COLUMN_SHIFT,
    )
}
