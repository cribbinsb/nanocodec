//! WebAssembly SIMD128 staged HEVC forward rows on eight 16-bit lanes, as
//! the NEON `neon_forward::hevc_rows`: the staged rows' graph
//! (`codec_txfm::generic::staged_hevc_pass`) keeps every stage of 8-bit
//! residuals scaled by 4 within 16 bits, so eight lines go through it per
//! vector instead of four in 32-bit lanes. Rotations multiply interleaved
//! pairs with `i32x4.dot_i16x8_s` (exact in 32 bits) and round back to 16;
//! equal-weight ones add or subtract first and take the rounding Q15
//! multiply. The results equal the generic staged rows (checkasm).
//!
//! Kernel contract: see the `codec_txfm` module docs. This module only
//! exists in builds with SIMD128 enabled, so its kernels need no run-time
//! check.

#![allow(clippy::inline_always)]

use core::arch::wasm32::{
    i16x8_add, i16x8_narrow_i32x4, i16x8_shl, i16x8_shuffle, i16x8_splat, i16x8_sub, i32x4_add,
    i32x4_dot_i16x8, i32x4_extend_high_i16x8, i32x4_extend_low_i16x8, i32x4_mul, i32x4_shr,
    i32x4_splat, v128, v128_load, v128_store,
};
use core::mem::MaybeUninit;

use super::forward_lanes::ForwardLanes;
use super::simd128_inverse::transpose8;
use super::{cospi, dct};
use crate::simd::simd128::relaxed::q15_mul_round;

/// Lines per vector.
const LANES: usize = 8;
/// Cosine bits of the staged graph.
const STAGED_COS_BIT: u32 = 12;
/// Input scale of the staged rows.
const ROW_INPUT_SHIFT: u32 = crate::codec_txfm::generic::ROW_INPUT_SHIFT as u32;
/// The staged rows' output scale, `181 / 2^(8 + 1/2)`: about `1/sqrt(2)`
/// relative to the graph's orthonormal scaling.
const OUTPUT_SCALE: i32 = 181;

/// Eight lines' samples or butterfly values in 16-bit lanes.
#[derive(Clone, Copy)]
struct Column(v128);

impl ForwardLanes for Column {
    #[inline(always)]
    fn splat(value: i32) -> Self {
        Self(i16x8_splat(value as i16))
    }
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // Staged-row stages fit i16 (module docs).
        Self(i16x8_add(self.0, other.0))
    }
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        Self(i16x8_sub(self.0, other.0))
    }
    #[inline(always)]
    fn half_btf(w0: i32, a: Self, w1: i32, b: Self, bit: u32) -> Self {
        // Equal weights: the sum or difference fits i16 (as NEON), and the
        // rounding Q15 multiply by `w << (15 - bit)` is libaom's
        // `round_shift(x * w, bit)`.
        if (w0 == w1 || w0 == -w1) && w0.abs() < (1 << bit) {
            let sum = if w0 == w1 {
                i16x8_add(a.0, b.0)
            } else {
                i16x8_sub(a.0, b.0)
            };
            return Self(q15_mul_round(sum, i16x8_splat((w0 << (15 - bit)) as i16)));
        }
        let weights = i32x4_splat((w0 & 0xFFFF) | (w1 << 16));
        let low = i32x4_dot_i16x8(i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(a.0, b.0), weights);
        let high = i32x4_dot_i16x8(
            i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(a.0, b.0),
            weights,
        );
        let round = i32x4_splat(1 << (bit - 1));
        // The graph keeps every rounded result in 16 bits: the saturating
        // narrow never clamps.
        Self(i16x8_narrow_i32x4(
            i32x4_shr(i32x4_add(low, round), bit),
            i32x4_shr(i32x4_add(high, round), bit),
        ))
    }
}

/// `OUTPUT_SCALE · x` rounded by `shift`, for four 32-bit lanes.
#[inline(always)]
fn scale_output(x: v128, shift: u32) -> v128 {
    let product = i32x4_mul(x, i32x4_splat(OUTPUT_SCALE));
    i32x4_shr(i32x4_add(product, i32x4_splat(1 << (shift - 1))), shift)
}

/// Staged HEVC rows of an `N`-point block (`N` 16 or 32) into the row-major
/// intermediate `out` (`N * N`), as `generic::staged_hevc_pass` with the
/// first-pass shift `log2(N) - 1` and an exact DC.
///
/// # Safety
///
/// `residual` covers `N` rows of `N` 8-bit differences at `stride`; `out`
/// holds `N * N` values.
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn hevc_rows<const N: usize>(
    residual: &[i16],
    stride: usize,
    out: &mut [MaybeUninit<i32>],
) {
    assert!(matches!(N, 16 | 32));
    // The first pass shifts by log2(N) - 1, plus 1 for the output scale and
    // the input scale.
    let output_shift = N.ilog2() - 1 + 1 + ROW_INPUT_SHIFT;
    let mut input = [Column::splat(0); N];
    let mut output = input;
    for r0 in (0..N).step_by(LANES) {
        for c0 in (0..N).step_by(LANES) {
            let rows = core::array::from_fn(|i| {
                let row = &residual[(r0 + i) * stride + c0..][..LANES];
                // SAFETY: `row` holds eight samples (16 bytes).
                unsafe { v128_load(row.as_ptr().cast()) }
            });
            for (i, column) in transpose8(rows).into_iter().enumerate() {
                input[c0 + i] = Column(i16x8_shl(column, ROW_INPUT_SHIFT));
            }
        }
        // The scaled inputs sum to at most 32 · 4 · 255 = 32640: i16.
        let dc = input.iter().fold(Column::splat(0), |sum, &v| sum.add(v));
        match N {
            16 => dct::fdct16::<Column>(
                input.as_slice().try_into().unwrap(),
                output.as_mut_slice().try_into().unwrap(),
                cospi(STAGED_COS_BIT),
                STAGED_COS_BIT,
            ),
            32 => dct::fdct32::<Column>(
                input.as_slice().try_into().unwrap(),
                output.as_mut_slice().try_into().unwrap(),
                cospi(STAGED_COS_BIT),
                STAGED_COS_BIT,
            ),
            _ => unreachable!(),
        }
        for k in (0..N).step_by(LANES) {
            let tile = transpose8(core::array::from_fn(|i| output[k + i].0));
            for (i, row) in tile.into_iter().enumerate() {
                let dest = &mut out[(r0 + i) * N + k..][..LANES];
                let low = scale_output(i32x4_extend_low_i16x8(row), output_shift);
                let high = scale_output(i32x4_extend_high_i16x8(row), output_shift);
                // SAFETY: `dest` holds eight `i32` slots (32 bytes).
                unsafe {
                    let ptr = dest.as_mut_ptr().cast::<i32>();
                    v128_store(ptr.cast(), low);
                    v128_store(ptr.add(4).cast(), high);
                }
            }
        }
        // The exact DC: 64 · sum >> (log2(N) - 1) of the unscaled inputs,
        // i.e. the scaled sum << (5 - log2(N)).
        let mut dc_lanes = [0_i16; LANES];
        // SAFETY: `dc_lanes` holds eight `i16`s (16 bytes).
        unsafe { v128_store(dc_lanes.as_mut_ptr().cast(), dc.0) };
        for (i, value) in dc_lanes.into_iter().enumerate() {
            out[(r0 + i) * N].write(i32::from(value) << (5 - N.ilog2()));
        }
    }
}
