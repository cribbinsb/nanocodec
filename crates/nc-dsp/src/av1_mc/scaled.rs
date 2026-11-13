//! SIMD-dispatched AV1 scaled inter prediction (AV1 §7.11.3.4): arbitrary
//! `xStep` / `yStep` in 1/1024 samples, dual or single filters across all
//! four interpolation families (including 4-tap short edges), at both final
//! 8-bit (`InterRound1 = 11`) and compound (`InterRound1 = 7`) precision.

#![allow(unsafe_code)]

use nc_common::{PlaneView, PlaneViewMut};

use super::convolve::{Av1McAxis, check_prediction, predict_compound_scalar, predict_scalar};
use super::filter::Av1Filters;
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

/// Final 8-bit scaled block inter prediction kernel.
pub(crate) type PredictFn = unsafe fn(
    filters: Av1Filters,
    window: &PlaneView<'_>,
    size: (usize, usize),
    axes: (Av1McAxis, Av1McAxis),
    dst: &mut PlaneViewMut<'_>,
);

/// Compound-precision (`i16`) scaled block inter prediction kernel.
pub(crate) type PredictCompoundFn = unsafe fn(
    filters: Av1Filters,
    window: &PlaneView<'_>,
    size: (usize, usize),
    axes: (Av1McAxis, Av1McAxis),
    dst: &mut [i16],
    dst_stride: usize,
);

/// Per-ISA scaled AV1 inter prediction kernels.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Kernels {
    pub predict: PredictFn,
    pub predict_compound: PredictCompoundFn,
}

/// Select the scaled AV1 prediction kernels for `isa`.
pub(crate) fn kernels(isa: Isa) -> Kernels {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => x86::KERNELS,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::KERNELS,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => wasm::KERNELS,
        _ => Kernels {
            predict: scalar_predict,
            predict_compound: scalar_predict_compound,
        },
    }
}

fn scalar_predict(
    filters: Av1Filters,
    window: &PlaneView<'_>,
    size: (usize, usize),
    axes: (Av1McAxis, Av1McAxis),
    dst: &mut PlaneViewMut<'_>,
) {
    predict_scalar(filters, window, size, axes, dst);
}

fn scalar_predict_compound(
    filters: Av1Filters,
    window: &PlaneView<'_>,
    size: (usize, usize),
    axes: (Av1McAxis, Av1McAxis),
    dst: &mut [i16],
    dst_stride: usize,
) {
    check_prediction(window, size, axes);
    predict_compound_scalar(filters, window, size, axes, dst, dst_stride);
}
