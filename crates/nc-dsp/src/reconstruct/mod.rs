//! Fused prediction + inverse-transform residual + clipped plane store.
//! The residual is a packed i32 block or a constant DC term. Saturating it
//! to i16, saturating the prediction addition, then narrowing to u8 is exact
//! even for extreme i32 residuals. No intermediate reconstructed block is needed.
use crate::Isa;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod avx2;
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
mod simd128;

/// Packed inverse-transform output, or a spatially constant DC residual.
#[derive(Debug, Clone, Copy)]
pub enum Residual32<'a> {
    /// One signed residual per pixel, in packed row order.
    Samples(&'a [i32]),
    /// The same signed residual at every pixel.
    Constant(i32),
}
impl Residual32<'_> {
    pub(crate) fn value(self, index: usize) -> i32 {
        match self {
            Self::Samples(samples) => samples[index],
            Self::Constant(dc) => dc,
        }
    }
}

/// Contract: prediction/destination cover height rows of width samples at
/// their strides; Samples covers width*height values; the ISA is supported.
/// Width is four or a multiple of eight (the shared `BlockSize` contract).
pub(crate) type Kernel = unsafe fn(&[u8], usize, Residual32<'_>, &mut [u8], usize, usize, usize);

pub(crate) fn kernel(isa: Isa) -> Kernel {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => avx2::reconstruct,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::reconstruct,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => simd128::reconstruct,
        _ => scalar,
    }
}

fn scalar(
    pred: &[u8],
    ps: usize,
    residual: Residual32<'_>,
    dst: &mut [u8],
    ds: usize,
    width: usize,
    height: usize,
) {
    for y in 0..height {
        for x in 0..width {
            dst[y * ds + x] = sample(pred[y * ps + x], residual.value(y * width + x));
        }
    }
}
pub(super) fn sample(pred: u8, residual: i32) -> u8 {
    (i64::from(pred) + i64::from(residual)).clamp(0, 255) as u8
}
