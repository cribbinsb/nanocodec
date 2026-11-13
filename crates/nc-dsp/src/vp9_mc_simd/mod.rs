//! VP9 byte-clipped separable interpolation, with ISA-specific row kernels.
//! The unchanged `vp9_mc::vp9_convolve` is the numerical oracle. Unlike AV1,
//! every pass rounds by seven bits and clips to bytes, including intermediates.

#![allow(
    clippy::inline_always,
    reason = "row operations must inline into ISA entry points"
)]

use crate::Isa;
use crate::vp9_mc::Vp9Filter;

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod simd128;

pub(crate) const MAX_SIDE: usize = 64;

pub(crate) type ConvolveFn =
    unsafe fn(Vp9Filter, &[u8], usize, (usize, usize), (usize, usize), &mut [u8], usize, bool);

pub(crate) fn kernel(isa: Isa) -> ConvolveFn {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 | Isa::Avx512 => avx2::convolve,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::convolve,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => simd128::convolve,
        _ => crate::vp9_mc::vp9_convolve_into,
    }
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
mod driver;
