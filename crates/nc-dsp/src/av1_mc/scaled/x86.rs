//! AVX2 instantiations of scaled AV1 inter prediction.

#![allow(unsafe_code)]

use nc_common::{PlaneView, PlaneViewMut};

use super::{Kernels, generic};
use crate::av1_mc::convolve::Av1McAxis;
use crate::av1_mc::filter::Av1Filters;
use crate::simd::avx2::{I16x4, I16x8, I16x16};

pub(super) const KERNELS: Kernels = Kernels {
    predict,
    predict_compound,
};

#[target_feature(enable = "avx2")]
unsafe fn predict(
    filters: Av1Filters,
    window: &PlaneView<'_>,
    size: (usize, usize),
    axes: (Av1McAxis, Av1McAxis),
    dst: &mut PlaneViewMut<'_>,
) {
    // SAFETY: AVX2 is enabled by this wrapper.
    unsafe { generic::predict::<I16x4, I16x8, I16x16>(filters, window, size, axes, dst) }
}

#[target_feature(enable = "avx2")]
unsafe fn predict_compound(
    filters: Av1Filters,
    window: &PlaneView<'_>,
    size: (usize, usize),
    axes: (Av1McAxis, Av1McAxis),
    dst: &mut [i16],
    dst_stride: usize,
) {
    // SAFETY: AVX2 is enabled by this wrapper.
    unsafe {
        generic::predict_compound::<I16x4, I16x8, I16x16>(
            filters, window, size, axes, dst, dst_stride,
        );
    }
}
