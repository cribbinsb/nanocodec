//! Packed-byte VP9 edge filtering, using libvpx's byte masks/narrow arithmetic
//! and rolling widened flat sums (BSD-3-Clause; attribution in `THIRD_PARTY.md`).
//! Each vector holds the eight independent lines of a VP9 edge. Vertical edges
//! transpose 8x8 byte tiles in registers. Outer taps and wide sums are skipped
//! unless a flat lane needs them. The unchanged `vp9_deblock` is the oracle.

use crate::{Isa, Vp9FilterLimits, Vp9FilterWidth};

mod line;
pub(crate) use line::{LineFn, kernel as line_kernel};

/// Filter settings for one eight-line VP9 edge segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vp9EdgeFilter {
    /// Maximum smoothing width permitted by the transform layout.
    pub width: Vp9FilterWidth,
    /// Thresholds derived by the codec from level and sharpness.
    pub limits: Vp9FilterLimits,
}

/// Number of lines described by one VP9 edge descriptor.
pub const VP9_EDGE_LINES: usize = 8;

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
mod packed;

pub(crate) type FilterFn =
    unsafe fn(&mut [u8], usize, usize, usize, usize, Vp9FilterWidth, Vp9FilterLimits);

pub(crate) fn kernel(isa: Isa) -> FilterFn {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => packed::avx2::filter,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => packed::neon::filter,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => packed::simd128::filter,
        _ => crate::vp9_deblock::vp9_filter_edge,
    }
}
