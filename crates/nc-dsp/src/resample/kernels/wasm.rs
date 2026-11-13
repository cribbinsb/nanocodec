//! WebAssembly SIMD128 instantiation of Lanczos-3 polyphase plane resampling.

#![allow(unsafe_code)]

use nc_common::{PlaneView, PlaneViewMut};

use super::generic;
use crate::resample::PlaneResampler;
use crate::simd::simd128::{I16x4, I16x8};

pub(super) unsafe fn resample(
    plan: &mut PlaneResampler,
    src: &PlaneView<'_>,
    dst: &mut PlaneViewMut<'_>,
) {
    // SAFETY: SIMD128 is enabled for the whole module when `wasm` is compiled.
    unsafe { generic::resample::<I16x4, I16x8, I16x8>(plan, src, dst) }
}
