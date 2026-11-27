//! AVX2 kernels: the generic code on four lanes for 4x4 and the
//! 16-bit-pair butterflies of [`super::paired`] on eight lanes otherwise.

use core::arch::x86_64::{
    _mm256_add_epi16, _mm256_loadu_si256, _mm256_or_si256, _mm256_packus_epi16,
    _mm256_setzero_si256, _mm256_storeu_si256, _mm256_testz_si256, _mm256_unpackhi_epi8,
    _mm256_unpackhi_epi16, _mm256_unpacklo_epi8, _mm256_unpacklo_epi16,
};
use nc_common::PlaneViewMut;

use super::generic as g;
use super::{Av2TxSize, Av2TxfmFns, paired, size_of_side};
use crate::simd::avx2::{I32x4, I32x8};
use crate::{BlockSize, Dsp, Isa};

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
    unsafe { paired::forward::<I32x8, N, C>(residual, stride, shifts, out) }
}

#[target_feature(enable = "avx2")]
unsafe fn inverse4(coeffs: &[i16], nonzero: usize, out: &mut [i16]) {
    // SAFETY: AVX2 is enabled for this function.
    unsafe { g::inverse::<I32x4, 4>(coeffs, nonzero, Av2TxSize::Tx4.inverse_shifts(), out) }
}

#[target_feature(enable = "avx2")]
unsafe fn inverse<const C: usize>(coeffs: &[i16], nonzero: usize, out: &mut [i16]) {
    let shifts = const { size_of_side(C) }.inverse_shifts();
    // SAFETY: AVX2 is enabled for this function.
    unsafe { paired::inverse::<I32x8, C>(coeffs, nonzero, shifts, out) }
}

#[target_feature(enable = "avx2")]
unsafe fn inverse64(coeffs: &[i16], nonzero: usize, out: &mut [i16]) {
    // SAFETY: AVX2 is enabled for this function.
    unsafe { paired::inverse64::<I32x8>(coeffs, nonzero, Av2TxSize::Tx64.inverse_shifts(), out) }
}

/// The large decoder DCTs consume column-major coefficients, and the
/// 64x64 prediction add duplicates the residual directly in registers.
pub(crate) fn inverse_add_large(
    dsp: &Dsp,
    size: Av2TxSize,
    coeffs: &mut [i16],
    rows: usize,
    dst: &mut PlaneViewMut<'_>,
) {
    assert!(matches!(dsp.isa(), Isa::Avx2 | Isa::Avx512));
    assert!(matches!(size, Av2TxSize::Tx32 | Av2TxSize::Tx64));
    let width = size.side();
    let shape = BlockSize::new(width, width).expect("AV2 square transform");
    let coeffs = &mut coeffs[..1024];
    let mut residual = [0; 1024];
    // SAFETY: both tiers support AVX2, the coefficient slice is the full
    // 32x32 coded area, and the view covers the checked output block.
    unsafe {
        let shifts = size.inverse_shifts();
        let narrow = rows <= 16 && only_first_eight_columns(coeffs);
        if dsp.isa() == Isa::Avx512 {
            super::avx512::inverse_large(coeffs, rows, narrow, shifts, &mut residual);
        } else {
            inverse_large(coeffs, rows, narrow, shifts, &mut residual);
        }
        if size == Av2TxSize::Tx64 {
            if dsp.isa() == Isa::Avx512 {
                super::avx512::add_repeated_residual(dst, &residual);
            } else {
                add_repeated_residual(dst, &residual);
            }
        } else {
            dsp.add_residual(shape, dst, &residual, 32);
        }
    }
    coeffs.fill(0);
}

/// Check the live row band of the columns the sparse path would skip.
/// Reading up to sixteen rows can only make the decision more conservative
/// at a shorter end of block.
#[target_feature(enable = "avx2")]
unsafe fn only_first_eight_columns(coeffs: &[i16]) -> bool {
    let mut found = _mm256_setzero_si256();
    for column in coeffs[8 * 32..].as_chunks::<32>().0 {
        // SAFETY: each input column contains 32 coefficients.
        found = _mm256_or_si256(found, unsafe { _mm256_loadu_si256(column.as_ptr().cast()) });
    }
    _mm256_testz_si256(found, found) != 0
}

#[target_feature(enable = "avx2")]
unsafe fn inverse_large(
    coefficients: &[i16],
    rows: usize,
    narrow: bool,
    shifts: (u32, u32),
    residual: &mut [i16; 1024],
) {
    // SAFETY: the caller checked the coded area and the row bound.
    unsafe {
        paired::inverse_from_columns::<I32x8, 32>(coefficients, rows, narrow, shifts, residual);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn add_repeated_residual(dst: &mut PlaneViewMut<'_>, residual: &[i16; 1024]) {
    let stride = dst.stride();
    let data = &mut dst.data_mut()[..63 * stride + 64];
    let zero = _mm256_setzero_si256();
    // SAFETY: every load and store is one checked 32-byte slice. The signed
    // sums fit i16 because the shared inverse clamps to -256..=255.
    unsafe {
        for (y, row) in residual.as_chunks::<32>().0.iter().enumerate() {
            let dst = data.as_mut_ptr().add(2 * y * stride);
            for x in [0, 16] {
                let r = _mm256_loadu_si256(row.as_ptr().add(x).cast());
                let lo = _mm256_unpacklo_epi16(r, r);
                let hi = _mm256_unpackhi_epi16(r, r);
                for offset in [2 * x, 2 * x + stride] {
                    let ptr = dst.add(offset);
                    let pred = _mm256_loadu_si256(ptr.cast());
                    let a = _mm256_add_epi16(_mm256_unpacklo_epi8(pred, zero), lo);
                    let b = _mm256_add_epi16(_mm256_unpackhi_epi8(pred, zero), hi);
                    _mm256_storeu_si256(ptr.cast(), _mm256_packus_epi16(a, b));
                }
            }
        }
    }
}
