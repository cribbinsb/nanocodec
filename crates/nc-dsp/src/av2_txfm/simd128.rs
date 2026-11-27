//! WebAssembly SIMD128 kernels on four lanes. The small forward transforms
//! use the generic butterfly. The other forward transforms and the inverse
//! transforms share the paired butterflies of [`super::paired`]
//! (`i32x4.dot_i16x8_s`). This
//! module only exists in builds with SIMD128 enabled.

use super::generic as g;
use super::{Av2TxSize, Av2TxfmFns, paired, size_of_side};
use crate::simd::simd128::I32x4;

super::typed_lanes::typed_fns!(av2_typed_fns, ; I32x4, I32x4);

/// The AV2 transform table of SIMD128.
pub(crate) fn av2_txfm_fns() -> Av2TxfmFns {
    Av2TxfmFns {
        forward: [
            forward::<4, 4>,
            forward::<8, 8>,
            paired_forward::<16, 16>,
            paired_forward::<32, 32>,
            paired_forward::<64, 32>,
        ],
        inverse: [
            inverse::<4>,
            inverse::<8>,
            inverse::<16>,
            inverse::<32>,
            inverse64,
        ],
    }
}

unsafe fn forward<const N: usize, const C: usize>(
    residual: &[i16],
    stride: usize,
    out: &mut [i32],
) {
    let shifts = const { size_of_side(N) }.forward_shifts();
    // SAFETY: SIMD128 is enabled for this whole module.
    unsafe { g::forward::<I32x4, N, C>(residual, stride, shifts, out) }
}

unsafe fn paired_forward<const N: usize, const C: usize>(
    residual: &[i16],
    stride: usize,
    out: &mut [i32],
) {
    let shifts = const { size_of_side(N) }.forward_shifts();
    // SAFETY: SIMD128 is enabled for this whole module, and the coded side
    // is at least four coefficients wide for every entry above.
    unsafe { paired::forward::<I32x4, N, C>(residual, stride, shifts, out) }
}

unsafe fn inverse<const C: usize>(coeffs: &[i16], nonzero: usize, out: &mut [i16]) {
    let shifts = const { size_of_side(C) }.inverse_shifts();
    // SAFETY: SIMD128 is enabled for this whole module.
    unsafe { paired::inverse::<I32x4, C>(coeffs, nonzero, shifts, out) }
}

unsafe fn inverse64(coeffs: &[i16], nonzero: usize, out: &mut [i16]) {
    // SAFETY: SIMD128 is enabled for this whole module.
    unsafe { paired::inverse64::<I32x4>(coeffs, nonzero, Av2TxSize::Tx64.inverse_shifts(), out) }
}
