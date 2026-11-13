//! H.266 planar/DC/angular pixel prediction and PDPC, after reference preparation.
//! Shared lane kernels instantiate AVX2, NEON and SIMD128; reference availability
//! and smoothing remain in the codec. Equations follow `VVenC` (`THIRD_PARTY.md`).
use crate::Isa;

mod angular;
mod angular_generic;
mod cclm;
pub use cclm::{vvc_cclm_downsample, vvc_cclm_downsample_collocated, vvc_cclm_predict};
mod generic;
mod isp;
mod mip;
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

fn scalar(
    n: usize,
    mode: VvcIntraMode,
    top: &[i32],
    left: &[i32],
    corner: i32,
    transposed_ok: bool,
    out: &mut [u8],
) -> bool {
    let mode = match mode {
        VvcIntraMode::Angular { direction: 18, .. } => VvcIntraMode::Horizontal,
        VvcIntraMode::Angular { direction: 50, .. } => VvcIntraMode::Vertical,
        VvcIntraMode::Angular { direction, luma } => {
            return angular::predict(n, direction, luma, top, left, corner, transposed_ok, out);
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
            out[dy * n + dx] = value.clamp(0, 255) as u8;
        }
    }
    false
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod avx2 {
    use crate::simd::avx2::{I16x4, I16x8, I16x16, I32x4, I32x8};
    super::generic::kernels!(#[target_feature(enable = "avx2")]; I32x4, I32x8; I16x4, I16x8, I16x16);
}
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon {
    use crate::simd::neon::{I16x4, I16x8, I32x4};
    super::generic::kernels!(#[target_feature(enable = "neon")]; I32x4, I32x4; I16x4, I16x8, I16x8);
}
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
mod simd128 {
    use crate::simd::simd128::{I16x4, I16x8, I32x4};
    super::generic::kernels!(#[target_feature(enable = "simd128")]; I32x4, I32x4; I16x4, I16x8, I16x8);
}

#[cfg(test)]
mod general_tests;
