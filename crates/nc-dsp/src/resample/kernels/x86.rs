//! AVX2 instantiation of Lanczos-3 polyphase plane resampling.

#![allow(unsafe_code)]

use nc_common::{PlaneView, PlaneViewMut};

use super::generic;
use crate::resample::PlaneResampler;
use crate::simd::avx2::{I16x4, I16x8, I16x16};

#[target_feature(enable = "avx2")]
pub(super) unsafe fn resample(
    plan: &mut PlaneResampler,
    src: &PlaneView<'_>,
    dst: &mut PlaneViewMut<'_>,
) {
    // SAFETY: AVX2 is enabled by this wrapper.
    unsafe { generic::resample::<I16x4, I16x8, I16x16>(plan, src, dst) }
}
