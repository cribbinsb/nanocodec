//! HEVC intra prediction kernels (HEVC §8.4.4.2.4, §8.4.4.2.6): planar, and
//! the line projection of the angular modes.
//!
//! The codec gathers and filters the references and builds the angular
//! modes' main reference array (extended with projected side samples for
//! negative angles); the kernels compute the samples. An angular line `j`
//! (a row for vertical modes, a column written as a row for horizontal
//! ones) is the 2-tap blend
//! `((32 − f) · ref[i + 1] + f · ref[i + 2] + 16) >> 5` of the main
//! reference at a fixed offset, so each line is one vector pass. [`reference`] is the scalar oracle.

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[allow(unsafe_code)]
mod lanes;
pub(crate) mod reference;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod avx2;
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
mod simd128;

use crate::Isa;

/// Largest HEVC transform (and intra prediction) side.
pub(crate) const MAX_SIDE: usize = 32;
/// Fractional bits of an angular position (1/32 sample).
pub(crate) const ANGLE_FRACTION_BITS: u32 = 5;
/// Largest angle magnitude (HEVC Table 8-5: intraPredAngle in −32..=32).
pub(crate) const MAX_ANGLE: i32 = 32;

/// Angular lines of an `n`x`n` block: `refs[n + i]` is the main reference
/// `ref[i]` for `i` in `−n..=2n` (`refs.len() == 3n + 1`); `out` gets `n`
/// lines of `n` samples (`angle` in −32..=32).
pub(crate) type AngularFn = unsafe fn(refs: &[u8], n: usize, angle: i32, out: &mut [u8]);

/// Planar prediction of an `n`x`n` block from the `n` samples above, the `n`
/// to the left, and the top-right and bottom-left samples.
pub(crate) type PlanarFn = unsafe fn(edges: &PlanarEdges<'_>, n: usize, out: &mut [u8]);

/// The references planar prediction reads (§8.4.4.2.4).
#[derive(Debug, Clone, Copy)]
pub struct PlanarEdges<'a> {
    /// `p[x][−1]` for `x` in `0..n`.
    pub above: &'a [u8],
    /// `p[−1][y]` for `y` in `0..n`.
    pub left: &'a [u8],
    /// `p[n][−1]`.
    pub top_right: u8,
    /// `p[−1][n]`.
    pub bottom_left: u8,
}

/// One ISA's HEVC intra kernels.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HevcIntraFns {
    pub angular: AngularFn,
    pub planar: PlanarFn,
}

/// The kernels for `isa` (the scalar reference where it has none).
pub(crate) fn kernels(isa: Isa) -> HevcIntraFns {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => avx2::KERNELS,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::KERNELS,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => simd128::KERNELS,
        _ => reference::KERNELS,
    }
}
