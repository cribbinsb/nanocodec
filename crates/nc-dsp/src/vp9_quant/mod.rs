//! VP9 deadzone quantization: `level = min((|c| << shift + offset) / step,
//! cap)` with the sign of `c`, DC (index 0) and AC at their own steps,
//! `offset = floor(step / rounding_divisor)`, and the encoder's all-zero test
//! (the largest coefficient magnitude).
//!
//! The quantization runs on the shared reciprocal dead-zone quantiser of the
//! AV1 family (`txfm` module docs, `docs/analysis/encode-kernel-sharing.md`
//! section 3): [`terms`] states VP9's quantiser in its terms. Only the
//! parameters are VP9's: AV1 adds `step * rounding / 128` where VP9 adds
//! `floor(step / divisor)` for divisors (3, 6) that do not divide 128, and
//! VP9 caps DC and AC levels separately. Scaling numerator and divisor by
//! 128 gives the shared kernel's form with every quotient unchanged
//! (`floor(128 n / (128 s)) = floor(n / s)`) and divisors of at least 512,
//! which its exact reciprocal needs.
//!
//! Coefficients are 16-bit (VP9 coefficients of 8-bit video, as the shared
//! forward stores them; 32x32 ones are at half scale and shifted by one), so
//! scaled numerators stay below `2^25`, inside the shared kernel's `2^31`
//! contract. Levels are 32-bit.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod simd128;

use crate::Isa;
use crate::txfm::{QuantTerms, StepTerms};

/// Smallest supported step (VP9's smallest quantizer step is 4).
const MIN_STEP: i32 = 2;
/// Largest supported step (VP9's largest is 1828).
const MAX_STEP: i32 = 2048;
/// Largest dequantized magnitude: the reference decoder stores 16 bits.
const MAX_DEQUANTIZED: i32 = i16::MAX as i32;
/// Bits by which [`terms`] scales numerators and divisors (the shared
/// quantiser's reciprocal needs divisors of at least 512, and `MIN_STEP`
/// times this is 256 short of that only for steps below 4, which VP9 never
/// uses; smaller steps fall back to the exact reference division).
const SCALE_BITS: u32 = 7;

/// One step's quantizer parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vp9QuantStep {
    step: u32,
    offset: u32,
    shift: u32,
    cap: u32,
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
        Self {
            step: step as u32,
            offset: (step / rounding_divisor) as u32,
            shift,
            cap: Self::max_level(step, shift),
        }
    }

    /// The largest level whose dequantized magnitude `level · step >> shift`
    /// the 16-bit range holds with a margin (VP9 §8.6.2 requires
    /// `Dequant` to fit 8 + `BitDepth` bits): every quantizer, the trellis
    /// included, keeps its levels at most this.
    #[must_use]
    pub fn max_level(step: i32, shift: u32) -> u32 {
        (((MAX_DEQUANTIZED + 1) << shift) / step - 1) as u32
    }

    /// This step in the shared quantiser's terms, scaled by `SCALE_BITS`.
    fn scaled(&self) -> StepTerms {
        StepTerms {
            offset: self.offset << SCALE_BITS,
            divisor: self.step << SCALE_BITS,
            cap: self.cap as i32,
        }
    }
}

/// VP9's quantizer of a block (`dc` for index 0, `ac` for the rest) in the
/// shared quantiser's terms (module docs).
///
/// # Panics
///
/// If the two steps' coefficient shifts differ (one transform size has one).
pub(crate) fn terms(dc: &Vp9QuantStep, ac: &Vp9QuantStep) -> QuantTerms {
    assert_eq!(
        dc.shift, ac.shift,
        "VP9 DC and AC coefficient shifts differ"
    );
    QuantTerms {
        input_shift: ac.shift + SCALE_BITS,
        dc: dc.scaled(),
        ac: ac.scaled(),
    }
}

/// The largest coefficient magnitude `max |c|` (0 for none): the encoder's
/// all-zero test before quantizing.
pub(crate) type MagnitudeFn = unsafe fn(&[i16]) -> u32;

/// The scalar reference of [`MagnitudeFn`].
pub(crate) unsafe fn largest_magnitude(coefficients: &[i16]) -> u32 {
    coefficients
        .iter()
        .map(|c| u32::from(c.unsigned_abs()))
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The direct VP9 definition: `min((|c| << shift) + floor(step /
    /// divisor)) / step, cap)`, signed.
    fn vp9_level(c: i32, step: i32, divisor: i32, shift: u32) -> i32 {
        let magnitude = ((c.unsigned_abs() << shift) + (step / divisor) as u32) / step as u32;
        let level = magnitude.min(Vp9QuantStep::max_level(step, shift)) as i32;
        if c < 0 { -level } else { level }
    }

    #[test]
    fn scaled_terms_give_vp9_levels() {
        for step in [MIN_STEP, 4, 7, 64, 1000, 1827, 1828, MAX_STEP] {
            for divisor in [2, 3, 6, 9] {
                for shift in [0, 1] {
                    let q = Vp9QuantStep::new(step, divisor, shift);
                    let terms = terms(&q, &q);
                    for c in (i32::from(i16::MIN)..=i32::from(i16::MAX)).step_by(7) {
                        assert_eq!(
                            terms.level(c, 1),
                            vp9_level(c, step, divisor, shift),
                            "step {step} divisor {divisor} shift {shift} c {c}"
                        );
                    }
                }
            }
        }
    }
}
