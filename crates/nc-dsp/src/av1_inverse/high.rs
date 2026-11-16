//! AV1 inverse transforms at any bit depth (AV1 §7.13.3): the scalar 2-D
//! driver for samples deeper than 8 bits, on the same 1-D graphs as the
//! 8-bit kernels ([`super::one_d`], `txfm::dct`).
//!
//! The arithmetic is libaom's `inv_txfm2d_add_c` with its high-bit-depth
//! stage ranges (`av1_gen_inv_stage_range`, BSD-2-Clause, Copyright (c)
//! 2016, Alliance for Open Media; see `THIRD_PARTY.md`), as dav1d's 16 bpc
//! `itx` does (`row_clip` / `col_clip`):
//!
//! - **Row pass:** coefficients (dequantised to `BitDepth + 8` bits,
//!   §7.12.3) are scaled by `1/sqrt(2)` for 2:1 rectangles, clamped to
//!   `BitDepth + 8` bits, transformed with every add/sub stage clamped to
//!   `BitDepth + 8` bits, and rounded by the size's row shift.
//! - **Column pass:** the row outputs are clamped to `Max(BitDepth + 6,
//!   16)` bits, transformed with the stages clamped to the same range, and
//!   rounded by [`COLUMN_SHIFT`]; the result is added to the prediction
//!   with `Clip1` at the depth.
//!
//! At 8 bits both ranges are 16 bits and this equals the 8-bit kernels on
//! every coefficient the dequantiser can produce (`high_depth` tests).

use nc_common::{BitDepth, Sample};

use super::one_d::{iadst4, iadst8, iadst16, identity};
use super::{
    AV1_MAX_CODED_SIDE, Av1TxSize, Av1TxType, COLUMN_SHIFT, COS_BIT, INV_SQRT2, Kernel1d,
    SQRT2_BITS,
};
use crate::txfm::COSPI_12;
use crate::txfm::dct;
use crate::txfm::inverse_lanes::InverseLanes;

/// One 32-bit value whose add/sub stages saturate to `BITS` signed bits
/// (libaom `clamp_value(x, stage_range)`).
#[derive(Debug, Clone, Copy, Default)]
struct Lane<const BITS: u32>(i32);

impl<const BITS: u32> Lane<BITS> {
    const MIN: i32 = -(1 << (BITS - 1));
    const MAX: i32 = (1 << (BITS - 1)) - 1;

    fn clamped(value: i32) -> Self {
        Self(value.clamp(Self::MIN, Self::MAX))
    }
}

impl<const BITS: u32> InverseLanes for Lane<BITS> {
    fn splat(value: i32) -> Self {
        Self(value)
    }

    fn add(self, other: Self) -> Self {
        Self(self.0.wrapping_add(other.0))
    }

    fn sub(self, other: Self) -> Self {
        Self(self.0.wrapping_sub(other.0))
    }

    fn half_btf(w0: i32, a: Self, w1: i32, b: Self, bit: u32) -> Self {
        Self(<i32 as crate::simd::I32Lanes>::half_btf(
            w0, a.0, w1, b.0, bit,
        ))
    }

    fn mul_round(w: i32, a: Self, bit: u32) -> Self {
        Self(round_shift(i64::from(w) * i64::from(a.0), bit))
    }

    /// The stage clamp, at this lane's range rather than 16 bits.
    fn clamp16(self) -> Self {
        Self::clamped(self.0)
    }
}

/// libaom `round_shift`: `(x + (1 << (bits − 1))) >> bits`, the identity
/// for `bits == 0`.
fn round_shift(value: i64, bits: u32) -> i32 {
    if bits == 0 {
        value as i32
    } else {
        ((value + (1 << (bits - 1))) >> bits) as i32
    }
}

fn fixed<V, const M: usize>(x: &[V]) -> &[V; M] {
    x.try_into().expect("1-D transform length")
}

fn fixed_mut<V, const M: usize>(x: &mut [V]) -> &mut [V; M] {
    x.try_into().expect("1-D transform length")
}

/// The `n`-point 1-D inverse `kind` of `x` into `y` (no flip). For `n ==
/// 64` the upper 32 inputs are zero (§7.13.3 zero-out).
fn transform_1d<const BITS: u32>(kind: Kernel1d, x: &[Lane<BITS>], y: &mut [Lane<BITS>]) {
    let n = x.len();
    match (kind, n) {
        (Kernel1d::Dct, 4) => dct::idct4(fixed(x), fixed_mut(y), &COSPI_12, COS_BIT),
        (Kernel1d::Dct, 8) => dct::idct8(fixed(x), fixed_mut(y), &COSPI_12, COS_BIT),
        (Kernel1d::Dct, 16) => dct::idct16(fixed(x), fixed_mut(y), &COSPI_12, COS_BIT),
        (Kernel1d::Dct, 32) => dct::idct32(fixed(x), fixed_mut(y), &COSPI_12, COS_BIT),
        (Kernel1d::Dct, 64) => dct::idct64_low32(fixed(x), fixed_mut(y), &COSPI_12, COS_BIT),
        (Kernel1d::Adst | Kernel1d::FlipAdst, 4) => {
            // ADST4 has no stage clamps (libaom `av1_iadst4`).
            let input: [i32; 4] = core::array::from_fn(|i| x[i].0);
            let mut output = [0; 4];
            iadst4(&input, &mut output, COS_BIT);
            for (lane, value) in y.iter_mut().zip(output) {
                *lane = Lane(value);
            }
        }
        (Kernel1d::Adst | Kernel1d::FlipAdst, 8) => {
            iadst8(fixed(x), fixed_mut(y), &COSPI_12, COS_BIT);
        }
        (Kernel1d::Adst | Kernel1d::FlipAdst, 16) => {
            iadst16(fixed(x), fixed_mut(y), &COSPI_12, COS_BIT);
        }
        (Kernel1d::Identity, 4) => identity::<_, 4>(fixed(x), fixed_mut(y)),
        (Kernel1d::Identity, 8) => identity::<_, 8>(fixed(x), fixed_mut(y)),
        (Kernel1d::Identity, 16) => identity::<_, 16>(fixed(x), fixed_mut(y)),
        (Kernel1d::Identity, 32) => identity::<_, 32>(fixed(x), fixed_mut(y)),
        _ => unreachable!("no {n}-point {kind:?} (checked by Av1TxType::fits)"),
    }
}

/// Inverse transform of column-major `coeffs` (coefficient (row `r`,
/// column `c`) at `c * tx.coded_height() + r`), added to the `tx`-sized
/// block of `dst` at `stride` with `Clip1` at `depth`.
///
/// # Panics
///
/// If `kind` does not fit `tx`, a slice is too short, or `depth` is not 8,
/// 10 or 12 bits.
pub(crate) fn inverse_add_at<T: Sample>(
    tx: Av1TxSize,
    kind: Av1TxType,
    coeffs: &[i32],
    dst: &mut [T],
    stride: usize,
    depth: BitDepth,
) {
    match depth.bits() {
        8 => by_size::<T, 16, 16>(tx, kind, coeffs, (dst, stride), depth),
        10 => by_size::<T, 18, 16>(tx, kind, coeffs, (dst, stride), depth),
        12 => by_size::<T, 20, 18>(tx, kind, coeffs, (dst, stride), depth),
        bits => panic!("no AV1 inverse transform at {bits} bits"),
    }
}

/// [`inverse_add`] instantiated for `tx`'s sides, so that its buffers are
/// exactly the block's size (a 64x64 buffer zeroed for every 4x4 block was
/// a third of a 10-bit decode's instructions).
fn by_size<T: Sample, const ROW: u32, const COL: u32>(
    tx: Av1TxSize,
    kind: Av1TxType,
    coeffs: &[i32],
    dst: (&mut [T], usize),
    depth: BitDepth,
) {
    macro_rules! sides {
        ($(($w:literal, $h:literal)),*) => {
            match (tx.width(), tx.height()) {
                $(($w, $h) => inverse_add::<T, ROW, COL, $w, $h>(tx, kind, coeffs, dst, depth),)*
                _ => panic!("not an AV1 transform size: {tx:?}"),
            }
        };
    }
    sides!(
        (4, 4),
        (8, 8),
        (16, 16),
        (32, 32),
        (64, 64),
        (4, 8),
        (8, 4),
        (8, 16),
        (16, 8),
        (16, 32),
        (32, 16),
        (32, 64),
        (64, 32),
        (4, 16),
        (16, 4),
        (8, 32),
        (32, 8),
        (16, 64),
        (64, 16)
    );
}

/// [`inverse_add_at`] of a `W`x`H` transform with the row range `ROW`
/// (`BitDepth + 8`) and the column range `COL` (`Max(BitDepth + 6, 16)`).
fn inverse_add<T: Sample, const ROW: u32, const COL: u32, const W: usize, const H: usize>(
    tx: Av1TxSize,
    kind: Av1TxType,
    coeffs: &[i32],
    (dst, stride): (&mut [T], usize),
    depth: BitDepth,
) {
    assert!(kind.fits(tx), "{kind:?} is not defined for {tx:?}");
    let (coded_w, coded_h) = (tx.coded_width(), tx.coded_height());
    let coeffs = &coeffs[..coded_w * coded_h];
    let (vertical, horizontal) = kind.kernels();
    let row_shift = tx.row_shift();

    // Row pass: row r's outputs at `rows[r]`; rows beyond the coded height
    // (64-sample columns) are zero and never stored.
    let mut rows = [[0i32; W]; H];
    let mut x = [Lane::<ROW>::default(); W];
    let mut y = [Lane::<ROW>::default(); W];
    debug_assert!(coded_h <= AV1_MAX_CODED_SIDE);
    for (r, out) in rows.iter_mut().take(coded_h).enumerate() {
        for (c, lane) in x.iter_mut().enumerate() {
            let value = if c < coded_w {
                coeffs[c * coded_h + r]
            } else {
                0
            };
            let value = if tx.is_two_to_one() {
                round_shift(i64::from(value) * i64::from(INV_SQRT2), SQRT2_BITS)
            } else {
                value
            };
            *lane = Lane::clamped(value);
        }
        transform_1d(horizontal, &x, &mut y);
        for (c, value) in out.iter_mut().enumerate() {
            let source = if horizontal == Kernel1d::FlipAdst {
                W - 1 - c
            } else {
                c
            };
            *value = round_shift(i64::from(y[source].0), row_shift);
        }
    }

    // Column pass.
    let mut x = [Lane::<COL>::default(); H];
    let mut y = [Lane::<COL>::default(); H];
    for c in 0..W {
        for (lane, row) in x.iter_mut().zip(&rows) {
            *lane = Lane::clamped(row[c]);
        }
        transform_1d(vertical, &x, &mut y);
        for r in 0..H {
            let source = if vertical == Kernel1d::FlipAdst {
                H - 1 - r
            } else {
                r
            };
            let residual = round_shift(i64::from(y[source].0), COLUMN_SHIFT);
            let sample = &mut dst[r * stride + c];
            *sample = depth.clip_to(sample.to_i32() + residual);
        }
    }
}
