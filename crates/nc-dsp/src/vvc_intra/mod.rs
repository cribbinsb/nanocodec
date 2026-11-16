//! H.266 planar/DC/angular pixel prediction and PDPC, after reference preparation.
//! Shared lane kernels instantiate AVX2, NEON and SIMD128; reference availability
//! and smoothing remain in the codec. Equations follow `VVenC` (`THIRD_PARTY.md`).
//!
//! The scalar references are generic over the sample type with the depth
//! as a runtime [`BitDepth`] (`docs/handoff/high-bit-depth.md`): the 8-bit
//! [`crate::Dsp`] methods call them at `u8` and [`BitDepth::EIGHT`], the
//! decoders' `u16` instance at the stream's depth. Depth enters only as the
//! `Clip1` bound of the predicted samples (H.266 §8.4.5.2.11–15) and, in
//! MIP, the mid value `1 << (BitDepth − 1)` (§8.4.5.2.2).
use nc_common::{BitDepth, Sample};

use crate::Isa;

mod angular;
mod angular_generic;
mod cclm;
pub use cclm::{vvc_cclm_downsample, vvc_cclm_downsample_collocated, vvc_cclm_predict};
mod generic;
mod isp;
mod mip;
pub(crate) use mip::{bdpcm_predict, mip_predict};
pub(crate) mod rectangular;

/// Supported VVC intra pixel predictors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VvcIntraMode {
    /// Planar interpolation with PDPC.
    Planar,
    /// DC average with PDPC.
    Dc,
    /// Horizontal axis with PDPC.
    Horizontal,
    /// Vertical axis with PDPC.
    Vertical,
    /// Square-block direction 2–66. Luma uses cubic/Gaussian interpolation;
    /// chroma uses linear interpolation. References contain `2*side` samples.
    Angular {
        /// H.266 intra direction index (2–66).
        direction: u8,
        /// True for luma, false for chroma.
        luma: bool,
    },
}

/// `(side, mode, top, left, corner, transposed_ok, out) -> transposed`: with
/// `transposed_ok`, horizontal angular directions may be written transposed
/// (their natural computation order), which the result reports.
pub(crate) type PredictFn =
    unsafe fn(usize, VvcIntraMode, &[i32], &[i32], i32, bool, &mut [u8]) -> bool;

pub(crate) fn kernel(isa: Isa) -> PredictFn {
    match isa {
        Isa::Scalar => scalar,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => avx2::predict,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::predict,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => simd128::predict,
        #[allow(unreachable_patterns)]
        _ => unreachable!("unsupported ISA"),
    }
}

/// The 8-bit scalar [`PredictFn`]: [`scalar_at`] at 8 bits.
fn scalar(
    n: usize,
    mode: VvcIntraMode,
    top: &[i32],
    left: &[i32],
    corner: i32,
    transposed_ok: bool,
    out: &mut [u8],
) -> bool {
    scalar_at(
        n,
        mode,
        top,
        left,
        corner,
        transposed_ok,
        out,
        BitDepth::EIGHT,
    )
}

/// The square predictor's scalar reference at `depth` (the [`PredictFn`]
/// contract, samples of type `T`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn scalar_at<T: Sample>(
    n: usize,
    mode: VvcIntraMode,
    top: &[i32],
    left: &[i32],
    corner: i32,
    transposed_ok: bool,
    out: &mut [T],
    depth: BitDepth,
) -> bool {
    // Folds to the constant 8 in the 8-bit instance.
    let depth = depth.of::<T>();
    let mode = match mode {
        VvcIntraMode::Angular { direction: 18, .. } => VvcIntraMode::Horizontal,
        VvcIntraMode::Angular { direction: 50, .. } => VvcIntraMode::Vertical,
        VvcIntraMode::Angular { direction, luma } => {
            return angular::predict(
                n,
                direction,
                luma,
                top,
                left,
                corner,
                transposed_ok,
                out,
                depth,
            );
        }
        mode => mode,
    };
    let dc = (top[..n].iter().chain(&left[..n]).sum::<i32>() + n as i32) / (2 * n as i32);
    let scale = (2 * n.ilog2() - 2) >> 2;
    for dy in 0..n {
        for dx in 0..n {
            let wl = 32 >> ((2 * dx) >> scale).min(31);
            let wt = 32 >> ((2 * dy) >> scale).min(31);
            let value = match mode {
                VvcIntraMode::Angular { .. } => unreachable!("handled above"),
                VvcIntraMode::Planar => {
                    let value = (((n - 1 - dx) as i32 * left[dy] + (dx + 1) as i32 * top[n])
                        + ((n - 1 - dy) as i32 * top[dx] + (dy + 1) as i32 * left[n])
                        + n as i32)
                        >> (n.ilog2() + 1);
                    value + ((wl * (left[dy] - value) + wt * (top[dx] - value) + 32) >> 6)
                }
                VvcIntraMode::Dc => dc + ((wl * (left[dy] - dc) + wt * (top[dx] - dc) + 32) >> 6),
                VvcIntraMode::Vertical => top[dx] + ((wl * (left[dy] - corner) + 32) >> 6),
                VvcIntraMode::Horizontal => left[dy] + ((wt * (top[dx] - corner) + 32) >> 6),
            };
            out[dy * n + dx] = depth.clip_to(value);
        }
    }
    false
}

/// Check the inputs of a square prediction ([`crate::Dsp::vvc_intra_predict`]):
/// a side of 4 to 32, a direction of 2 to 66, edges of `side + 1`
/// samples (`2 * side` for angular directions) and an output of the block,
/// every reference sample inside the depth.
///
/// # Panics
/// If any of these fails.
pub(crate) fn check_square(
    side: usize,
    mode: VvcIntraMode,
    (top, left, corner): (&[i32], &[i32], i32),
    out_len: usize,
    depth: BitDepth,
) {
    assert!([4, 8, 16, 32].contains(&side));
    let required = match mode {
        VvcIntraMode::Angular { direction, .. } => {
            assert!((2..=66).contains(&direction));
            2 * side
        }
        _ => side + 1,
    };
    assert!(out_len >= side * side && top.len() >= required && left.len() >= required);
    check_references(&top[..required], &left[..required], corner, depth);
}

/// Check the inputs of a rectangular prediction
/// ([`crate::Dsp::vvc_intra_predict_rect`]) and return the edges cut to
/// twice their block dimension.
///
/// # Panics
/// If a dimension is not a power of two in 2..=64, the direction is not
/// 2..=66, a buffer is short or a reference sample is outside the depth.
pub(crate) fn check_rect<'a>(
    (width, height): (usize, usize),
    mode: VvcIntraMode,
    (top, left, corner): (&'a [i32], &'a [i32], i32),
    out_len: usize,
    depth: BitDepth,
) -> (&'a [i32], &'a [i32]) {
    let supported = |n: usize| n.is_power_of_two() && (2..=64).contains(&n);
    assert!(supported(width) && supported(height));
    assert!(top.len() >= 2 * width && left.len() >= 2 * height);
    assert!(out_len >= width * height);
    if let VvcIntraMode::Angular { direction, .. } = mode {
        assert!((2..=66).contains(&direction));
    }
    let (top, left) = (&top[..2 * width], &left[..2 * height]);
    check_references(top, left, corner, depth);
    (top, left)
}

/// Whether a rectangle is one the square kernels ([`PredictFn`]) take.
pub(crate) fn square_kernel_fits((width, height): (usize, usize)) -> bool {
    width == height && (4..=32).contains(&width)
}

/// Every reference sample lies in `0..=(1 << BitDepth) − 1`: the kernels'
/// intermediate widths (and the SIMD tiers' 16-bit lanes at 8 bits) rely
/// on it.
fn check_references(top: &[i32], left: &[i32], corner: i32, depth: BitDepth) {
    assert!(
        top.iter()
            .chain(left)
            .chain(std::iter::once(&corner))
            .all(|&v| (0..=depth.max()).contains(&v))
    );
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod avx2 {
    use crate::simd::avx2::{I16x4, I16x8, I16x16};
    super::generic::kernels!(#[target_feature(enable = "avx2")]; crate::intra_runs::avx2::Avx2Runs; I16x4, I16x8, I16x16);
}
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon {
    use crate::simd::neon::{I16x4, I16x8};
    super::generic::kernels!(#[target_feature(enable = "neon")]; crate::intra_runs::neon::NeonRuns; I16x4, I16x8, I16x8);
}
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
mod simd128 {
    use crate::simd::simd128::{I16x4, I16x8};
    super::generic::kernels!(#[target_feature(enable = "simd128")]; crate::intra_runs::simd128::Simd128Runs; I16x4, I16x8, I16x8);
}

#[cfg(test)]
mod general_tests;
