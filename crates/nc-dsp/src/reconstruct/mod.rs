//! Fused prediction + inverse-transform residual + clipped plane store.
//! The residual is a packed 16-bit block or a constant DC term (AV2's
//! inverse output, `-256..=255` for conforming streams). A saturating 16-bit
//! addition of the prediction, then narrowing to u8, equals clamping the
//! exact sum for every `i16` residual. No intermediate reconstructed block
//! is needed.
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
pub enum Residual16<'a> {
    /// One signed residual per pixel, in packed row order.
    Samples(&'a [i16]),
    /// The same signed residual at every pixel.
    Constant(i16),
}
impl Residual16<'_> {
    pub(crate) fn value(self, index: usize) -> i16 {
        match self {
            Self::Samples(samples) => samples[index],
            Self::Constant(dc) => dc,
        }
    }

    /// The constant residual, or zero for `Samples`: the value the SIMD
    /// kernels splat once per block.
    pub(crate) fn constant_or_zero(self) -> i16 {
        match self {
            Self::Constant(dc) => dc,
            Self::Samples(_) => 0,
        }
    }
}

/// Contract: prediction/destination cover height rows of width samples at
/// their strides; Samples covers width*height values; the ISA is supported.
/// Width is four or a multiple of eight (the shared `BlockSize` contract).
pub(crate) type Kernel = unsafe fn(&[u8], usize, Residual16<'_>, &mut [u8], usize, usize, usize);

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
    residual: Residual16<'_>,
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
fn sample(pred: u8, residual: i16) -> u8 {
    (i32::from(pred) + i32::from(residual)).clamp(0, 255) as u8
}
