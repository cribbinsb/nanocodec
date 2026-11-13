//! HEVC weighted sample prediction (HEVC §8.5.3.3.4): combining one or two
//! 14-bit prediction blocks into 8-bit samples.
//!
//! One formula covers every case: `clamp((a·w0 + b·w1 + round) >> shift)`.
//!
//! - default bi-prediction: `w0 = w1 = 1`, `round = 64`, `shift = 7`;
//! - explicit uni-prediction (eq. 8-252): `b = a`, `w1 = 0`,
//!   `round = 2^(log2WD−1) + (o << log2WD)`, `shift = log2WD` (adding the
//!   offset before the shift is exact: it is a multiple of `2^log2WD`);
//! - explicit bi-prediction (eq. 8-253):
//!   `round = (o0 + o1 + 1) << log2WD`, `shift = log2WD + 1`.
//!
//! Products and sums stay well inside 32 bits (`|a|, |b| < 2^15`, weights
//! in −128..=255). [`reference`] is the scalar oracle; the vector kernels
//! widen into 32-bit accumulators.

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[allow(unsafe_code)]
mod lanes;
pub(crate) mod reference;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod avx2;
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
mod simd128;

use crate::Isa;

/// Largest right shift: `log2WD + 1` with `log2WD = denominator + 6 <= 13`.
pub(crate) const MAX_SHIFT: u32 = 14;

/// The weights of one weighted prediction (module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HevcWeights {
    /// Weight of the first block.
    pub w0: i16,
    /// Weight of the second block (0 for uni-prediction).
    pub w1: i16,
    /// Added before the shift.
    pub round: i32,
    /// Right shift, 1..=14.
    pub shift: u32,
}

/// Weighted prediction of a `w`x`h` block: `a` and `b` are row-major at
/// stride `w`, `dst` at `dst_stride`.
pub(crate) type WeightFn = unsafe fn(
    weights: &HevcWeights,
    a: &[i16],
    b: &[i16],
    w: usize,
    h: usize,
    dst: &mut [u8],
    dst_stride: usize,
);

/// The kernel for `isa` (the scalar reference where it has none).
pub(crate) fn kernel(isa: Isa) -> WeightFn {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 | Isa::Avx512 => avx2::WEIGHT,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::WEIGHT,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => simd128::WEIGHT,
        _ => reference::weight as WeightFn,
    }
}
