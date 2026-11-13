//! Each ISA's table of 2-D inverses, inside functions compiled for the ISA:
//! sizes with both sides at least 8 run that ISA's guarded 16-bit drivers
//! (`txfm::narrow_inverse`); sizes with a 4-sample side run the generic
//! driver on 32-bit lanes (4 lanes where a side is 4). The scalar reference
//! runs the generic driver everywhere.

use super::Av1InverseFns;

/// Defines `inverse_fns()`, the table of one ISA: `$large` serves sizes with
/// both sides at least 8; `$small` lanes serve the sizes with a 4-sample side
/// and `$wide` lanes those `$large` leaves to the generic driver; `$constant`
/// adds DC-only residuals.
macro_rules! inverse_fns {
    ($(#[$isa:meta])*; $small:ty, $wide:ty, $constant:path, $large:ident) => {
        $(#[$isa])*
        unsafe fn add<V: crate::simd::I32Lanes, const W: usize, const H: usize>(
            vertical: super::super::Kernel1d,
            horizontal: super::super::Kernel1d,
            coeffs: &[i32],
            nonzero_rows: usize,
            dst: &mut [u8],
            dst_stride: usize,
        ) {
            // SAFETY: the table holding this function is only installed
            // when its ISA is supported (`Dsp::with_isa`), and the lane
            // counts chosen below divide both sides.
            unsafe {
                if W >= 8 && H >= 8 {
                    $large::<W, H>((vertical, horizontal), coeffs, nonzero_rows, dst, dst_stride);
                } else {
                    super::super::generic::inverse_add::<V, W, H>(
                        vertical, horizontal, coeffs, nonzero_rows, dst, dst_stride,
                    );
                }
            }
        }

        $(#[$isa])*
        unsafe fn dc(tx: super::super::Av1TxSize, coefficient: i32, dst: &mut [u8], stride: usize) {
            let residual = super::super::generic::dc_residual(tx, coefficient);
            // SAFETY: the table is installed only for a supported ISA, and
            // Dsp validated the destination.
            unsafe { $constant(residual, tx.width(), tx.height(), dst, stride) }
        }

        /// This ISA's inverses, indexed like `Av1TxSize::ALL`.
        pub(crate) fn inverse_fns() -> super::Av1InverseFns {
            super::Av1InverseFns {
                dc,
                add: [
                    add::<$small, 4, 4>,
                    add::<$wide, 8, 8>,
                    add::<$wide, 16, 16>,
                    add::<$wide, 32, 32>,
                    add::<$wide, 64, 64>,
                    add::<$small, 4, 8>,
                    add::<$small, 8, 4>,
                    add::<$wide, 8, 16>,
                    add::<$wide, 16, 8>,
                    add::<$wide, 16, 32>,
                    add::<$wide, 32, 16>,
                    add::<$wide, 32, 64>,
                    add::<$wide, 64, 32>,
                    add::<$small, 4, 16>,
                    add::<$small, 16, 4>,
                    add::<$wide, 8, 32>,
                    add::<$wide, 32, 8>,
                    add::<$wide, 16, 64>,
                    add::<$wide, 64, 16>,
                ],
            }
        }
    };
}

/// The scalar reference: the same driver on one `i32` lane, whose
/// `half_btf` and rounding compute in 64 bits as libaom does.
pub(crate) mod scalar {
    inverse_fns!(; i32, i32, constant_add, large);

    /// The generic driver on one `i32` lane, with the SIMD entry points'
    /// signature.
    unsafe fn large<const W: usize, const H: usize>(
        (vertical, horizontal): (super::super::Kernel1d, super::super::Kernel1d),
        coeffs: &[i32],
        nonzero_rows: usize,
        dst: &mut [u8],
        stride: usize,
    ) {
        // SAFETY: scalar lanes need no ISA; one lane divides every side.
        unsafe {
            super::super::generic::inverse_add::<i32, W, H>(
                vertical,
                horizontal,
                coeffs,
                nonzero_rows,
                dst,
                stride,
            );
        }
    }

    /// The scalar constant add, with the SIMD entry points' signature.
    unsafe fn constant_add(
        residual: i32,
        width: usize,
        height: usize,
        dst: &mut [u8],
        stride: usize,
    ) {
        crate::constant_add::scalar(residual, width, height, dst, stride);
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2 {
    use crate::simd::avx2::{I32x4, I32x8};
    use crate::txfm::avx2_inverse::av1_inverse_add as large;
    inverse_fns!(#[target_feature(enable = "avx2")]; I32x4, I32x8, crate::constant_add::avx2::add, large);
}

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon {
    use crate::simd::neon::I32x4;
    use crate::txfm::neon_inverse::av1_inverse_add as large;
    inverse_fns!(; I32x4, I32x4, crate::constant_add::neon::add, large);
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) mod simd128 {
    use crate::simd::simd128::I32x4;
    use crate::txfm::simd128_inverse::av1_inverse_add as large;
    inverse_fns!(; I32x4, I32x4, crate::constant_add::simd128::add, large);
}

/// The table of `isa`, which must be compiled in.
pub(crate) fn for_isa(isa: crate::Isa) -> Av1InverseFns {
    match isa {
        crate::Isa::Scalar => scalar::inverse_fns(),
        #[cfg(target_arch = "x86_64")]
        crate::Isa::Avx2 => avx2::inverse_fns(),
        #[cfg(target_arch = "aarch64")]
        crate::Isa::Neon => neon::inverse_fns(),
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        crate::Isa::Simd128 => simd128::inverse_fns(),
        #[allow(unreachable_patterns)]
        _ => unreachable!("is_supported() is false for ISAs not compiled in"),
    }
}
