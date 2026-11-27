//! The sample and coefficient types of the general AV2 decoder's residual
//! kernels ([`super::itx`], [`super::stx`]): [`Av2Pixel`] ties each sample
//! type to its coefficient storage, as dav2d's bitdepth template does
//! (`coef` is `int16_t` at 8 bpc and `int32_t` at 16 bpc,
//! `include/common/bitdepth.h`, commit 237229e), and both traits route the
//! 8-bit instance to its SIMD kernels while deeper samples take the scalar
//! reference (`docs/handoff/high-bit-depth.md`).

use nc_common::{BitDepth, Sample};

use super::itx::{RectTxSize, TxType};
use super::stx;
use crate::{Av1Coefficient, Dsp, Isa};

/// Bits above the sample depth that dav2d's intermediate clamps keep: the
/// row-pass output (`row_clip_min = ~bitdepth_max << 7`), the secondary
/// transform (`±128 · (1 + BITDEPTH_MAX)`) and the cross-chroma rotation
/// (`±1 << (bd + 7)`) all clamp to `bd + 8` signed bits (AVM `clamp_buf`).
const INTERMEDIATE_EXTRA_BITS: u32 = 7;

/// The inclusive range of dav2d's `bd + 8`-bit intermediate clamps at
/// `depth` (see [`INTERMEDIATE_EXTRA_BITS`]): the `i16` range at 8 bits.
#[inline]
pub(crate) fn intermediate_bounds(depth: BitDepth) -> (i32, i32) {
    let half = 1 << (depth.bits() + INTERMEDIATE_EXTRA_BITS);
    (-half, half - 1)
}

/// A dequantised AV2 coefficient: `i16` at 8 bits, `i32` deeper. The
/// storage is AV1's ([`Av1Coefficient`]: both codecs clamp coefficients
/// to `BitDepth + 8` bits); this adds the SIMD routing of the secondary
/// transform.
pub trait Av2Coefficient: Av1Coefficient + sealed::Sealed {
    /// The secondary transform's matrix product on the vector tier of
    /// `isa`, or `None` where this type has none (the caller then takes the
    /// scalar product).
    #[doc(hidden)]
    fn secondary_matrix_simd<const N: usize, const GROUPS: usize>(
        isa: Isa,
        cf: &[Self],
        kernel: &[i8],
    ) -> Option<[Self; N]>;

    /// The whole 8x8 secondary transform (product and write-out) on the
    /// vector tier of `isa`; `false` where this type has none.
    #[doc(hidden)]
    fn secondary8_simd(
        isa: Isa,
        cf: &mut [Self],
        stride: usize,
        orientation: (bool, usize),
        kernel: &[i8],
        inputs: usize,
    ) -> bool;
}

/// A sample type of the general AV2 decoder, with its coefficient storage.
pub trait Av2Pixel: Sample {
    /// dav2d's `coef` at this sample type.
    type Coefficient: Av2Coefficient;

    /// The inverse transform and add on the vector tier of `dsp`, or
    /// `false` (nothing read or written) where this type has no kernel for
    /// `size` and `tx_type`; the caller then takes the scalar path.
    #[doc(hidden)]
    fn inverse_transform_add_simd(
        dsp: &Dsp,
        dst: &mut [Self],
        position: (usize, usize),
        coeffs: &mut [Self::Coefficient],
        transform: (RectTxSize, TxType),
        eob: u16,
    ) -> bool;
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for i16 {}
    impl Sealed for i32 {}
}

impl Av2Coefficient for i16 {
    #[inline]
    fn secondary_matrix_simd<const N: usize, const GROUPS: usize>(
        isa: Isa,
        cf: &[Self],
        kernel: &[i8],
    ) -> Option<[Self; N]> {
        stx::matrix_simd::<N, GROUPS>(isa, cf, kernel)
    }

    #[inline]
    fn secondary8_simd(
        isa: Isa,
        cf: &mut [Self],
        stride: usize,
        orientation: (bool, usize),
        kernel: &[i8],
        inputs: usize,
    ) -> bool {
        stx::secondary8_simd(isa, cf, stride, orientation, kernel, inputs)
    }
}

impl Av2Coefficient for i32 {
    #[inline]
    fn secondary_matrix_simd<const N: usize, const GROUPS: usize>(
        _: Isa,
        _: &[Self],
        _: &[i8],
    ) -> Option<[Self; N]> {
        None
    }

    #[inline]
    fn secondary8_simd(
        _: Isa,
        _: &mut [Self],
        _: usize,
        _: (bool, usize),
        _: &[i8],
        _: usize,
    ) -> bool {
        false
    }
}

impl Av2Pixel for u8 {
    type Coefficient = i16;

    #[inline]
    fn inverse_transform_add_simd(
        dsp: &Dsp,
        dst: &mut [Self],
        position: (usize, usize),
        coeffs: &mut [i16],
        transform: (RectTxSize, TxType),
        eob: u16,
    ) -> bool {
        super::itx::inverse_transform_add_simd(dsp, dst, position, coeffs, transform, eob)
    }
}

impl Av2Pixel for u16 {
    type Coefficient = i32;

    #[inline]
    fn inverse_transform_add_simd(
        _: &Dsp,
        _: &mut [Self],
        _: (usize, usize),
        _: &mut [i32],
        _: (RectTxSize, TxType),
        _: u16,
    ) -> bool {
        false
    }
}
