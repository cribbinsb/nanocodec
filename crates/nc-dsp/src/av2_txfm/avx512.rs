//! AVX-512 AV2 forward transforms at 16, 32 and 64 points: the paired
//! 16-bit butterflies of [`super::paired`] (as AVX2) on 16 lanes, so each
//! pass covers sixteen lines per vector instead of eight. The arithmetic is
//! the AVX2 kernel's (`vpmaddwd` on word pairs), so results are identical.

use super::typed::Av2TxType;
use super::typed_lanes::{self, Av2TypedFns};
use super::{Av2TxfmFns, paired, size_of_side};
use crate::simd::avx512::I32x16;

/// Overrides the forward transforms of 16 points and more.
pub(crate) fn install(table: &mut Av2TxfmFns) {
    table.forward[2..].copy_from_slice(&[forward::<16, 16>, forward::<32, 32>, forward::<64, 32>]);
}

pub(crate) fn install_typed(table: &mut Av2TypedFns) {
    table.forward[2..].copy_from_slice(&[typed_forward::<16>, typed_forward::<32>]);
    table.inverse[2..].copy_from_slice(&[typed_inverse::<16>, typed_inverse::<32>]);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn typed_forward<const N: usize>(
    transform: Av2TxType,
    residual: &[i16],
    stride: usize,
    output: &mut [i32],
) {
    // SAFETY: AVX-512 is enabled, and both installed sizes have sixteen lanes per tile.
    unsafe { typed_lanes::forward::<I32x16, N>(transform, residual, stride, output) }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn typed_inverse<const N: usize>(
    transform: Av2TxType,
    coefficients: &[i16],
    output: &mut [i16],
) {
    // SAFETY: same ISA and tile-size contract as the forward entry.
    unsafe { typed_lanes::inverse::<I32x16, N>(transform, coefficients, output) }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn forward<const N: usize, const C: usize>(
    residual: &[i16],
    stride: usize,
    out: &mut [i32],
) {
    let shifts = const { size_of_side(N) }.forward_shifts();
    // SAFETY: AVX-512F/BW/VL are enabled for this function; 16 lanes <= C.
    unsafe { paired::forward::<I32x16, N, C>(residual, stride, shifts, out) }
}
