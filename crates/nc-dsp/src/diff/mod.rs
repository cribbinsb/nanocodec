//! Residual and distortion kernels between a source block and its
//! prediction.
//!
//! - **Residual:** `src - pred` as `i16`, row-major at a given stride.
//! - **Add residual:** `dst = clamp(dst + residual, 0, 255)`, the
//!   reconstruction step of codecs whose inverse transform produces a
//!   residual (AV1's inverse DCT adds directly: `Dsp::av1_inverse_dct_add`).
//! - **Variance:** the sum and sum of squares of `src - pred`
//!   ([`Variance`]); `n · variance` is `sse - sum^2 / n` (libaom
//!   `aom_variance`).
//! - **Variance with energy:** the [`Variance`] of `src - pred` and, in
//!   the same pass, the prediction's own sums ([`VarianceEnergy`]), whose
//!   `n · variance` is the prediction's AC energy (the engine's psychovisual
//!   term, `nc_encoder::psy`).
//! - **Block sums:** the sum and sum of squares of one block's own samples
//!   (a [`Variance`] against zero): `n · variance` is its AC energy, the
//!   psychovisual term's measure of a source block or a whole-sample
//!   prediction.
//! - **SSE of 8x8 cells:** the sum of squared differences of each 8x8
//!   block along a row of `src - pred` (the encoder's reconstruction error
//!   per analysis cell, `nc_encoder::reuse`). Exact integers on every ISA.
//! - **Sub-pixel variance:** the variance of `src` against a bilinear
//!   prediction at 1/8-sample offsets, exactly libaom's
//!   `aom_sub_pixel_variance` (two passes, each `(a * f0 + b * f1 + 64) >> 7`
//!   with `bilinear_filters_2t`), but with libaom's difference reversed
//!   (`src - pred`, like [`Variance`] everywhere), which flips the sign of
//!   `sum` only. SVT-AV1's fixed-stage sub-pel search ranks candidates with
//!   it.
//!
//! # Kernel contract
//!
//! Every kernel is an `unsafe fn`, called only from [`crate::Dsp`], which
//! guarantees the CPU supports the kernel's ISA and that every slice covers
//! its block at its stride: `h` rows of `w` samples (`w` fixed by the
//! kernel), and for the sub-pixel reference `h + 1` rows of `w + 1`. Loads
//! and stores index their slices, so a broken contract panics rather than
//! reads out of bounds.

#[allow(unsafe_code)]
pub(crate) mod generic;
pub(crate) mod scalar;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub(crate) mod avx2;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub(crate) mod avx512;

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
pub(crate) mod neon;

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon_dotprod;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
pub(crate) mod simd128;

/// Sum and sum of squares of the differences `src - pred` over a block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Variance {
    /// Sum of squared differences (at most `4096 * 255^2` for 64x64).
    pub sse: u32,
    /// Sum of differences.
    pub sum: i32,
}

impl Variance {
    /// `n · variance` over a block of `area` samples, libaom's definition:
    /// `sse - sum^2 / area`, the division rounding down.
    #[must_use]
    pub fn n_variance(self, area: u32) -> u32 {
        let sum = i64::from(self.sum);
        self.sse - ((sum * sum) / i64::from(area)) as u32
    }
}

/// The [`Variance`] of `src - pred` and the prediction's own sums, from one
/// pass over the block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VarianceEnergy {
    /// Sums of `src - pred`.
    pub residual: Variance,
    /// Sums of `pred` itself (`sse` its sum of squares, at most
    /// `4096 * 255^2`): `prediction.n_variance(area)` is its AC energy.
    pub prediction: Variance,
}

/// Positions of the bilinear sub-pixel filter (1/8 sample).
pub const SUBPEL_VARIANCE_POSITIONS: usize = 8;

/// libaom `bilinear_filters_2t`: taps `(128 - 16 k, 16 k)` at offset `k/8`.
pub(crate) fn bilinear_taps(offset: usize) -> (i16, i16) {
    let second = 16 * offset as i16;
    (128 - second, second)
}

/// `src - pred` into `out` (`w` wide), optionally returning its absolute sum.
pub(crate) type ResidualFn = unsafe fn(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    out: &mut [i16],
    out_stride: usize,
    h: usize,
) -> u32;
/// `dst = clamp(dst + residual)` (`w` wide).
pub(crate) type AddResidualFn = unsafe fn(
    dst: &mut [u8],
    dst_stride: usize,
    residual: &[i16],
    residual_stride: usize,
    h: usize,
);
/// Variance of `src - pred` (`w` wide).
pub(crate) type VarianceFn =
    unsafe fn(src: &[u8], src_stride: usize, pred: &[u8], pred_stride: usize, h: usize) -> Variance;
/// [`VarianceEnergy`] of `src` and `pred` (`w` wide).
pub(crate) type VarianceEnergyFn = unsafe fn(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    h: usize,
) -> VarianceEnergy;
/// Sums of the samples of `src` (`w` wide) as a [`Variance`] against zero.
pub(crate) type BlockSumsFn = unsafe fn(src: &[u8], src_stride: usize, h: usize) -> Variance;
/// Variance of `src` against a bilinear sub-pixel prediction from `rf`.
pub(crate) type SubpelVarianceFn = unsafe fn(
    rf: &[u8],
    ref_stride: usize,
    x_offset: usize,
    y_offset: usize,
    src: &[u8],
    src_stride: usize,
    h: usize,
) -> Variance;

/// Sum of squared differences of each 8x8 block along a row of `src` and
/// `pred` (block i covers columns 8i..8i+8 of 8 rows) into `out`.
pub(crate) type Sse8x8RowFn =
    unsafe fn(src: &[u8], src_stride: usize, pred: &[u8], pred_stride: usize, out: &mut [u32]);

/// One ISA's residual kernels, indexed by width class (4..64).
#[derive(Debug, Clone, Copy)]
pub(crate) struct DiffFns {
    pub residual: [ResidualFn; 5],
    pub residual_sad: [ResidualFn; 5],
    pub add_residual: [AddResidualFn; 5],
    pub variance: [VarianceFn; 5],
    pub variance_energy: [VarianceEnergyFn; 5],
    pub block_sums: [BlockSumsFn; 5],
    pub sub_pixel_variance: [SubpelVarianceFn; 5],
    pub sse_8x8_row: Sse8x8RowFn,
}
