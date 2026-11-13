//! Single-plane sample kernels: 8x8 block moments and 2x downscaling
//! (pre-analysis and the motion-search pyramid).
//!
//! - **Moments:** the sum and sum of squares of each 8x8 block along a row
//!   of blocks, from which callers derive mean and variance. Exact
//!   integers, so every ISA agrees. A whole row per call keeps the per-block
//!   cost to the arithmetic.
//! - **Moments 8x4:** the same over 4-row blocks, for statistics taken on
//!   every other row (the caller passes a view at twice the stride).
//! - **Downscale 2x:** each output sample is the rounded mean of a 2x2 input
//!   box, `(a + b + c + d + 2) >> 2`.
//! - **Pairs 2x:** each output sample comes from a horizontal pair of the
//!   same input row: their rounded mean `(a + b + 1) >> 1`, or the left one.
//!   With a view of every other row this decimates a plane 2x vertically
//!   and averages (or decimates) 2x horizontally.
//!
//! # Kernel contract
//!
//! As for the pixel kernels (`pixel` module docs): every kernel is an
//! `unsafe fn`, called only from [`crate::Dsp`], which guarantees
//!
//! 1. `moments_8x8_row` (`moments_8x4_row`): `src` covers 8 (4) rows of
//!    `8 * out.len()` samples at `src_stride`;
//! 2. `downscale_2x`: `src` covers `2 * h` rows of `2 * w` samples and `dst`
//!    covers `h` rows of `w` samples, at their strides, with `w, h >= 1`;
//!    `pair_mean_2x` / `pair_pick_2x`: as `downscale_2x` with `h` source
//!    rows;
//! 3. the CPU supports the kernel's ISA.

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

/// Sum and sum of squares of a block's samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Moments {
    /// Sum of samples (at most 64 * 255 for 8x8).
    pub sum: u32,
    /// Sum of squared samples (at most 64 * 255^2 for 8x8).
    pub sum_sq: u32,
}

impl Moments {
    /// Samples in the 8x8 block these moments describe.
    pub const SAMPLES_8X8: u32 = 64;

    /// Mean sample value of an 8x8 block, rounded to nearest.
    #[must_use]
    pub fn mean_8x8(self) -> u8 {
        ((self.sum + Self::SAMPLES_8X8 / 2) / Self::SAMPLES_8X8) as u8
    }

    /// Sum of squared deviations from the mean of an 8x8 block,
    /// `(64 * sum_sq - sum^2) / 64` rounded down. Divide by 64 for the variance.
    #[must_use]
    pub fn sum_sq_dev_8x8(self) -> u32 {
        // `64 * sum_sq` is a multiple of 64, so the floor of the quotient
        // is `sum_sq - ceil(sum^2 / 64)`; `sum^2` (at most 16320^2) fits u32.
        self.sum_sq - (self.sum * self.sum).div_ceil(Self::SAMPLES_8X8)
    }
}

/// Moments of each 8x8 block along a row: block i covers columns 8i..8i+8.
pub(crate) type Moments8x8RowFn = unsafe fn(src: &[u8], src_stride: usize, out: &mut [Moments]);

/// Horizontal pairs of a `2w` x `h` source into a `w` x `h` destination.
pub(crate) type Pairs2xFn =
    unsafe fn(src: &[u8], src_stride: usize, dst: &mut [u8], dst_stride: usize, w: usize, h: usize);

/// 2x downscale of a `2w` x `2h` source into a `w` x `h` destination.
pub(crate) type Downscale2xFn =
    unsafe fn(src: &[u8], src_stride: usize, dst: &mut [u8], dst_stride: usize, w: usize, h: usize);

/// One ISA's implementations of every sample kernel.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SampleFns {
    pub moments_8x8_row: Moments8x8RowFn,
    pub moments_8x4_row: Moments8x8RowFn,
    pub downscale_2x: Downscale2xFn,
    pub pair_mean_2x: Pairs2xFn,
    pub pair_pick_2x: Pairs2xFn,
}

/// The rounded mean of a horizontal pair: the reference definition every
/// ISA matches.
#[inline]
pub(crate) const fn pair_mean(a: u8, b: u8) -> u8 {
    ((a as u16 + b as u16 + 1) >> 1) as u8
}

/// The rounded mean of a 2x2 box: the reference definition every ISA matches.
#[inline]
pub(crate) const fn box_mean(a: u8, b: u8, c: u8, d: u8) -> u8 {
    ((a as u16 + b as u16 + c as u16 + d as u16 + 2) >> 2) as u8
}
