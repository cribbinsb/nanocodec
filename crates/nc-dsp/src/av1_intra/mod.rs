//! AV1 intra prediction kernels with SIMD versions: the patches of filter
//! intra (AV1 §7.11.2.3), directional prediction (§7.11.2.4: its
//! interpolation runs and column-tile transposes), and smooth and Paeth
//! prediction (§7.11.2.6, §7.11.2.2).
//!
//! The codec-side logic (edges, grid, zones) and the scalar references live
//! in [`crate::intra_prediction::filter`] and
//! [`crate::intra_prediction::directional`];
//! [`crate::Dsp::av1_intra_predict_filter`] and
//! [`crate::Dsp::av1_intra_predict_angle`] call this ISA's kernels through
//! them.
//!
//! # Kernel contract
//!
//! Every SIMD kernel is an `unsafe fn` that executes its ISA's
//! instructions, so it may only be called through a table built after
//! [`crate::Isa::is_supported`]. Grid accesses index arrays, so a wrong
//! side panics rather than reading out of bounds.

use nc_common::PlaneViewMut;

use crate::cpu::Isa;
use crate::intra_prediction::filter::{self, FilterGrid, FilterIntraMode};
use crate::intra_prediction::{IntraEdges, IntraMode};

pub(crate) mod generic;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub(crate) mod avx2;

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
pub(crate) mod neon;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
pub(crate) mod simd128;

/// The patches of a `width`x`height` filter intra block
/// (`filter::predict_patches`).
pub(crate) type FilterPatchesFn =
    unsafe fn(grid: &mut FilterGrid, width: usize, height: usize, mode: FilterIntraMode);

/// The safe scalar reference as a table entry.
#[allow(unsafe_code)]
unsafe fn scalar_patches(
    grid: &mut FilterGrid,
    width: usize,
    height: usize,
    mode: FilterIntraMode,
) {
    filter::predict_patches(grid, width, height, mode);
}

/// The kernel of `isa`. Calling it needs `isa`'s support, which
/// `Dsp::with_isa` checks before building a table.
pub(crate) fn filter_patches(isa: Isa) -> FilterPatchesFn {
    match isa {
        Isa::Scalar => scalar_patches,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => avx2::filter_patches,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::filter_patches,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => simd128::filter_patches,
        #[allow(unreachable_patterns)]
        _ => unreachable!("is_supported() is false for ISAs not compiled in"),
    }
}

/// Directional prediction of a square
/// ([`IntraEdges::predict_angle`]'s contract).
pub(crate) type DirectionalFn =
    unsafe fn(edges: &IntraEdges, angle: u16, dst: &mut PlaneViewMut<'_>);

/// The safe scalar reference as a table entry.
#[allow(unsafe_code)]
unsafe fn scalar_directional(edges: &IntraEdges, angle: u16, dst: &mut PlaneViewMut<'_>) {
    edges.predict_angle(angle, dst);
}

/// The directional kernel of `isa`. Calling it needs `isa`'s support,
/// which `Dsp::with_isa` checks before building a table.
pub(crate) fn directional(isa: Isa) -> DirectionalFn {
    match isa {
        Isa::Scalar => scalar_directional,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => avx2::predict_angle,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::predict_angle,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => simd128::predict_angle,
        #[allow(unreachable_patterns)]
        _ => unreachable!("is_supported() is false for ISAs not compiled in"),
    }
}

/// Smooth (any of the three) or Paeth prediction of a square
/// ([`IntraEdges::predict`]'s contract for those modes).
pub(crate) type SmoothPaethFn =
    unsafe fn(edges: &IntraEdges, mode: IntraMode, dst: &mut PlaneViewMut<'_>);

/// The safe scalar reference as a table entry.
#[allow(unsafe_code)]
unsafe fn scalar_smooth_paeth(edges: &IntraEdges, mode: IntraMode, dst: &mut PlaneViewMut<'_>) {
    edges.predict(mode, dst);
}

/// The smooth and Paeth kernel of `isa`. Calling it needs `isa`'s support,
/// which `Dsp::with_isa` checks before building a table.
pub(crate) fn smooth_paeth(isa: Isa) -> SmoothPaethFn {
    match isa {
        Isa::Scalar => scalar_smooth_paeth,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => avx2::smooth_paeth,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::smooth_paeth,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => simd128::smooth_paeth,
        #[allow(unreachable_patterns)]
        _ => unreachable!("is_supported() is false for ISAs not compiled in"),
    }
}
