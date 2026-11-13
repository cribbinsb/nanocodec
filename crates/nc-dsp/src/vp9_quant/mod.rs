//! VP9 deadzone quantization: `level = min((|c| << shift + offset) / step,
//! cap)` with the sign of `c`, DC (index 0) and AC at their own steps. The
//! division is an exact reciprocal multiplication, as in `nc-vp9`'s
//! quantizer, whose numbers this reproduces bit for bit.
//!
//! The existing quantize kernels cannot serve: H.264/HEVC add their offset
//! after a 32-bit multiply and shift, and AV1's adds `step · rounding / 128`,
//! neither of which equals VP9's `floor(step / divisor)` added before the
//! division for every step; VP9 also caps DC and AC levels separately.
//!
//! # Exactness and ranges
//!
//! Contract: `|c| < 2^16` (VP9 coefficients of 8-bit video; 32x32 ones are
//! at half scale and shifted by one), so numerators stay below `2^18`. With
//! `reciprocal = ceil(2^32 / step)` and steps in `MIN_STEP..=MAX_STEP`, the
//! reciprocal's error is under `step / 2^32` per unit, so
//! `numerator · reciprocal >> 32` is exactly `floor(numerator / step)`
//! (the error stays below `2^18 · 2^11 / 2^32 < 1 / step`). Products fit
//! 64 bits; AVX2 and SIMD128 compute them with a widening 32x32 multiply.
//! NEON instead uses `high_multiplier = ceil(2^31 / step)` with a doubling
//! high multiply (`(2 · n · m) >> 32 = (n · m) >> 31`), exact by the same
//! argument while `n · step < 2^31` (here below `2^29`); both reciprocals
//! give `floor(n / step)`, so the results are identical.

#![allow(
    clippy::inline_always,
    reason = "the shared frame must inline into each ISA entry point"
)]

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod simd128;

use crate::Isa;

/// Smallest supported step (VP9's smallest quantizer step is 4; the
/// reciprocal of 1 would not fit 32 bits).
const MIN_STEP: i32 = 2;
/// Largest supported step (VP9's largest is 1828).
const MAX_STEP: i32 = 2048;
/// Largest dequantized magnitude: the reference decoder stores 16 bits.
const MAX_DEQUANTIZED: i32 = i16::MAX as i32;
/// Fixed-point bits of the reciprocal.
const RECIPROCAL_BITS: u32 = 32;
/// Fixed-point bits of the high multiplier (a doubling high multiply's).
const HIGH_MULTIPLIER_BITS: u32 = 31;

/// One step's quantizer parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vp9QuantStep {
    pub(crate) reciprocal: u32,
    #[cfg_attr(not(any(test, target_arch = "aarch64")), allow(dead_code))]
    pub(crate) high_multiplier: i32,
    pub(crate) offset: u32,
    pub(crate) shift: u32,
    pub(crate) cap: u32,
}

impl Vp9QuantStep {
    /// Quantizer for `step`, rounding by `step / rounding_divisor` (the
    /// deadzone), with coefficients shifted left by `shift` first (1 for
    /// 32x32, else 0). Levels are capped so that `level · step >> shift`
    /// fits the 16-bit dequantized range.
    ///
    /// # Panics
    ///
    /// If `step` is outside `2..=2048`, `rounding_divisor < 1` or `shift > 1`.
    #[must_use]
    pub fn new(step: i32, rounding_divisor: i32, shift: u32) -> Self {
        assert!(
            (MIN_STEP..=MAX_STEP).contains(&step),
            "VP9 step {step} out of range"
        );
        assert!(rounding_divisor >= 1 && shift <= 1);
        let reciprocal = (1u64 << RECIPROCAL_BITS).div_ceil(step as u64);
        Self {
            reciprocal: u32::try_from(reciprocal).expect("step >= 2"),
            high_multiplier: i32::try_from((1u64 << HIGH_MULTIPLIER_BITS).div_ceil(step as u64))
                .expect("step >= 2"),
            offset: (step / rounding_divisor) as u32,
            shift,
            cap: (((MAX_DEQUANTIZED + 1) << shift) / step - 1) as u32,
        }
    }

    /// The level of coefficient `c` (scalar reference).
    #[inline]
    pub(crate) fn level(&self, c: i32) -> i32 {
        let numerator = u64::from((c.unsigned_abs() << self.shift) + self.offset);
        let magnitude = ((numerator * u64::from(self.reciprocal)) >> RECIPROCAL_BITS) as u32;
        let capped = magnitude.min(self.cap) as i32;
        if c < 0 { -capped } else { capped }
    }
}

/// Quantize `coefficients` into `levels` (same length): index 0 with `dc`,
/// the rest with `ac`. Returns whether any level is nonzero.
pub(crate) type QuantizeFn = unsafe fn(&[i32], &Vp9QuantStep, &Vp9QuantStep, &mut [i32]) -> bool;

/// The scalar reference.
pub(crate) unsafe fn quantize(
    coefficients: &[i32],
    dc: &Vp9QuantStep,
    ac: &Vp9QuantStep,
    levels: &mut [i32],
) -> bool {
    let mut nonzero = false;
    for (i, (&c, level)) in coefficients.iter().zip(levels.iter_mut()).enumerate() {
        *level = if i == 0 { dc.level(c) } else { ac.level(c) };
        nonzero |= *level != 0;
    }
    nonzero
}

/// The largest coefficient magnitude `max |c|` (0 for none): the encoder's
/// all-zero test before quantizing.
pub(crate) type MagnitudeFn = unsafe fn(&[i32]) -> u32;

/// The scalar reference of [`MagnitudeFn`].
pub(crate) unsafe fn largest_magnitude(coefficients: &[i32]) -> u32 {
    coefficients
        .iter()
        .map(|c| c.unsigned_abs())
        .max()
        .unwrap_or(0)
}

/// The largest-magnitude kernel of `isa`.
pub(crate) fn magnitude_kernel(isa: Isa) -> MagnitudeFn {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => avx2::largest_magnitude,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::largest_magnitude,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => simd128::largest_magnitude,
        _ => largest_magnitude,
    }
}

/// The kernel of `isa`.
pub(crate) fn kernel(isa: Isa) -> QuantizeFn {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => avx2::quantize,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::quantize,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => simd128::quantize,
        _ => quantize,
    }
}

/// Vector kernels' shared frame: whole vectors of `LANES` with `ac`, then
/// the scalar reference on the tail and the DC fix-up.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
fn split<const LANES: usize>(
    coefficients: &[i32],
    dc: &Vp9QuantStep,
    ac: &Vp9QuantStep,
    levels: &mut [i32],
    vectors: impl FnOnce(&[[i32; LANES]], &mut [[i32; LANES]]) -> bool,
) -> bool {
    let n = coefficients.len().min(levels.len());
    let (coefficients, levels) = (&coefficients[..n], &mut levels[..n]);
    let (whole, tail) = coefficients.as_chunks::<LANES>();
    let (whole_out, tail_out) = levels.as_chunks_mut::<LANES>();
    let mut nonzero = vectors(whole, whole_out);
    for (&c, level) in tail.iter().zip(tail_out) {
        *level = ac.level(c);
        nonzero |= *level != 0;
    }
    if let Some(&c) = coefficients.first() {
        let at_ac_step = levels[0];
        levels[0] = dc.level(c);
        // The vector pass counted index 0 at the AC step. Only when that was
        // its only nonzero contribution and the DC level is zero (rare: the
        // DC step is usually the smaller) does the flag need a recount.
        nonzero = if at_ac_step != 0 && levels[0] == 0 {
            levels.iter().any(|&l| l != 0)
        } else {
            nonzero || levels[0] != 0
        };
    }
    nonzero
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Largest numerator under the coefficient contract (`|c| < 2^16`,
    /// shifted by one, plus an offset below `MAX_STEP`).
    const MAX_NUMERATOR: u32 = (1 << 17) + MAX_STEP as u32;

    #[test]
    fn reciprocal_division_is_exact() {
        // Every numerator for the extreme and some awkward steps.
        for step in [MIN_STEP, 3, 4, 7, 1000, 1827, 1828, MAX_STEP - 1, MAX_STEP] {
            let q = Vp9QuantStep::new(step, 1, 0);
            for numerator in 0..=MAX_NUMERATOR {
                let want = u64::from(numerator / step as u32);
                let quotient = (u64::from(numerator) * u64::from(q.reciprocal)) >> RECIPROCAL_BITS;
                assert_eq!(quotient, want, "step {step} n {numerator}");
                let high =
                    (u64::from(numerator) * q.high_multiplier as u64) >> HIGH_MULTIPLIER_BITS;
                assert_eq!(high, want, "high multiplier: step {step} n {numerator}");
            }
        }
    }
}
