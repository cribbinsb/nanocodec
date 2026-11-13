//! VP9 inverse-transform dispatch with the unchanged scalar fallback.

use crate::{Isa, Vp9Pass};

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
mod packed;

pub(crate) type Kernel = unsafe fn(u8, Vp9Pass, Vp9Pass, &[i32], &mut [i32]);

/// `(log2, columns, rows, coefficients, (nonzero rows, nonzero columns),
/// dst, dst_stride)`: the inverse added into `dst`
/// ([`crate::Dsp::vp9_inverse_add`]).
pub(crate) type AddKernel =
    unsafe fn(u8, Vp9Pass, Vp9Pass, &[i32], (usize, usize), &mut [u8], usize);

pub(crate) fn add_kernel(isa: Isa) -> AddKernel {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 | Isa::Avx512 => packed::avx2::inverse_add,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => packed::neon::inverse_add,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => packed::simd128::inverse_add,
        _ => scalar_add,
    }
}

/// The scalar oracle; the extent is only a hint for the SIMD kernels.
fn scalar_add(
    log2: u8,
    columns: Vp9Pass,
    rows: Vp9Pass,
    input: &[i32],
    _nonzero: (usize, usize),
    dst: &mut [u8],
    dst_stride: usize,
) {
    crate::vp9_txfm::vp9_inverse_add(log2, columns, rows, input, dst, dst_stride);
}

pub(crate) fn kernel(isa: Isa) -> Kernel {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 | Isa::Avx512 => packed::avx2::inverse,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => packed::neon::inverse,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => packed::simd128::inverse,
        _ => crate::vp9_txfm::vp9_inverse_transform,
    }
}
