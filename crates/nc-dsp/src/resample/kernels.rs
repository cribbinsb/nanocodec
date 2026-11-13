//! Per-ISA SIMD dispatch for Lanczos-3 polyphase plane resampling.

#![allow(unsafe_code)]

use nc_common::{PlaneView, PlaneViewMut};

use super::{PlaneResampler, resample_scalar_with};
use crate::Isa;

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
mod generic;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm;
#[cfg(target_arch = "x86_64")]
mod x86;

/// Kernel contract: `src` covers `plan.in_size()`, `dst` covers at least
/// `plan.out_size()`, and `isa` is supported on the current CPU.
pub(crate) type ResampleFn =
    unsafe fn(plan: &mut PlaneResampler, src: &PlaneView<'_>, dst: &mut PlaneViewMut<'_>);

pub(crate) fn kernel(isa: Isa) -> ResampleFn {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => x86::resample,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::resample,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => wasm::resample,
        _ => scalar,
    }
}

fn scalar(plan: &mut PlaneResampler, src: &PlaneView<'_>, dst: &mut PlaneViewMut<'_>) {
    resample_scalar_with(plan, src, dst);
}
