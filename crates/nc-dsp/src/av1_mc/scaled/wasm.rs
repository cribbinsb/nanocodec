//! WebAssembly SIMD128 instantiations of scaled AV1 inter prediction.

#![allow(unsafe_code)]

use nc_common::{PlaneView, PlaneViewMut};

use super::{Kernels, generic};
use crate::av1_mc::convolve::Av1McAxis;
use crate::av1_mc::filter::Av1Filters;
use crate::simd::simd128::{I16x4, I16x8};

pub(super) const KERNELS: Kernels = Kernels {
    predict,
    predict_compound,
};

unsafe fn predict(
    filters: Av1Filters,
    window: &PlaneView<'_>,
    size: (usize, usize),
    axes: (Av1McAxis, Av1McAxis),
    dst: &mut PlaneViewMut<'_>,
) {
    // SAFETY: SIMD128 is enabled for the whole module when `wasm` is compiled.
    unsafe { generic::predict::<I16x4, I16x8, I16x8>(filters, window, size, axes, dst) }
}

unsafe fn predict_compound(
    filters: Av1Filters,
    window: &PlaneView<'_>,
    size: (usize, usize),
    axes: (Av1McAxis, Av1McAxis),
    dst: &mut [i16],
    dst_stride: usize,
) {
    // SAFETY: SIMD128 is enabled for the whole module when `wasm` is compiled.
    unsafe {
        generic::predict_compound::<I16x4, I16x8, I16x8>(
            filters, window, size, axes, dst, dst_stride,
        );
    }
}
