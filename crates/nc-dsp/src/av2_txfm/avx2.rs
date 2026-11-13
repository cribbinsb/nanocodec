//! AVX2 kernels: the generic code on 4 lanes for 4x4 and for the 64x64
//! inverse (the 32x32 inverse repeated, rarely used); every other size runs
//! the 16-bit-pair butterflies of [`super::avx2_paired`] on 8 lanes.

use super::generic as g;
use super::{Av2TxSize, Av2TxfmFns, avx2_paired as paired};
use crate::simd::avx2::{I32x4, I32x8};

super::typed_lanes::typed_fns!(av2_typed_fns, #[target_feature(enable = "avx2")]; I32x4, I32x8);

/// The AV2 transform table of AVX2.
pub(crate) fn av2_txfm_fns() -> Av2TxfmFns {
    Av2TxfmFns {
        forward: [
            forward4,
            forward::<8, 8>,
            forward::<16, 16>,
            forward::<32, 32>,
            forward::<64, 32>,
        ],
        inverse: [
            inverse4,
            inverse::<8>,
            inverse::<16>,
            inverse::<32>,
            inverse64,
        ],
    }
}

/// The size whose forward transform has side `N`.
const fn size_of_side(n: usize) -> Av2TxSize {
    match Av2TxSize::from_side(n) {
        Some(size) => size,
        None => panic!("not an AV2 transform side"),
    }
}

#[target_feature(enable = "avx2")]
unsafe fn forward4(residual: &[i16], stride: usize, out: &mut [i32]) {
    // SAFETY: AVX2 is enabled for this function.
    unsafe { g::forward::<I32x4, 4, 4>(residual, stride, Av2TxSize::Tx4.forward_shifts(), out) }
}

#[target_feature(enable = "avx2")]
unsafe fn forward<const N: usize, const C: usize>(
    residual: &[i16],
    stride: usize,
    out: &mut [i32],
) {
    let shifts = const { size_of_side(N) }.forward_shifts();
    // SAFETY: AVX2 is enabled for this function.
    unsafe { paired::forward::<N, C>(residual, stride, shifts, out) }
}

#[target_feature(enable = "avx2")]
unsafe fn inverse4(coeffs: &[i32], nonzero: usize, out: &mut [i32]) {
    // SAFETY: AVX2 is enabled for this function.
    unsafe { g::inverse::<I32x4, 4>(coeffs, nonzero, Av2TxSize::Tx4.inverse_shifts(), out) }
}

#[target_feature(enable = "avx2")]
unsafe fn inverse<const C: usize>(coeffs: &[i32], nonzero: usize, out: &mut [i32]) {
    let shifts = const { size_of_side(C) }.inverse_shifts();
    // SAFETY: AVX2 is enabled for this function.
    unsafe { paired::inverse::<C>(coeffs, nonzero, shifts, out) }
}

#[target_feature(enable = "avx2")]
unsafe fn inverse64(coeffs: &[i32], nonzero: usize, out: &mut [i32]) {
    // SAFETY: AVX2 is enabled for this function.
    unsafe { g::inverse64::<I32x8>(coeffs, nonzero, Av2TxSize::Tx64.inverse_shifts(), out) }
}
