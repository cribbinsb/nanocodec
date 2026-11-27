//! A heuristic's fixed threshold made a linear function of local features:
//! `threshold(features) = fixed + bias + weights · features`.
//!
//! Why: a decision that compares a computed quantity against a tuned
//! constant can instead compare it against a function of what is known at
//! the decision point. With the bias and the weights all zero the function
//! is the constant, so today's rule is one setting of the parameters and a
//! tuner warm-started there can only move away where encodes measure a
//! gain (the owner's "today's heuristic is a degenerate case of the
//! network"). The parameters are registered and tuned on real encodes
//! (`nceval.tune`), not fitted offline. Users: the RD partition search's
//! context gates and split selector (nc-encoder `rd_split.rs`,
//! `split_selector.rs`), the sub-pel exits (nc-encoder `decision.rs`) and
//! the motion search's zero-vector gates and window exit (`motion.rs`). It
//! lives in nc-analysis, the lowest engine crate, so both crates share it.

/// `bias + weights · features`, the shift added to a fixed threshold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearThreshold<const N: usize> {
    /// The shift where every feature is 0.
    pub bias: f32,
    /// The shift per unit of each feature.
    pub weights: [f32; N],
}

impl<const N: usize> LinearThreshold<N> {
    /// The shift, or `None` when every term is zero (the fixed threshold,
    /// with no features to compute).
    #[must_use]
    pub fn new(bias: f32, weights: [f32; N]) -> Option<Self> {
        (bias != 0.0 || weights.iter().any(|&w| w != 0.0)).then_some(Self { bias, weights })
    }

    /// The shift at `features`.
    #[must_use]
    pub fn at(&self, features: &[f32; N]) -> f32 {
        self.weights
            .iter()
            .zip(features)
            .fold(self.bias, |sum, (w, f)| sum + w * f)
    }
}

/// The floor on λ before its log, so a zero λ gives a finite feature.
const LAMBDA_FLOOR: f32 = 1e-3;
/// The ln λ the λ feature is centred on (λ ≈ 33, a mid-rate frame).
const LOG_LAMBDA_CENTRE: f32 = 3.5;
/// Scales ln λ to about unit range over the coded rates (ln λ spans ~4).
const LOG_LAMBDA_SCALE: f32 = 0.25;
/// Scales the luma noise sigma to about unit range (sigma rarely exceeds 5).
const NOISE_SCALE: f32 = 0.2;
/// The log2 block area the size feature is centred on (16x16).
const LOG2_AREA_CENTRE: f32 = 8.0;
/// Scales log2 area to about unit range (4x4 .. 64x64 spans 8).
const LOG2_AREA_SCALE: f32 = 0.25;

/// The frame λ as a feature: ln λ, centred and scaled.
#[must_use]
pub fn log_lambda_feature(lambda: f32) -> f32 {
    (lambda.max(LAMBDA_FLOOR).ln() - LOG_LAMBDA_CENTRE) * LOG_LAMBDA_SCALE
}

/// The source's luma noise sigma (`noise.rs`) as a feature.
#[must_use]
pub fn noise_feature(noise_sigma: f32) -> f32 {
    noise_sigma * NOISE_SCALE
}

/// A block's area (luma samples) as a feature: log2 area, centred and scaled.
#[must_use]
pub fn block_size_feature(area: f32) -> f32 {
    (area.log2() - LOG2_AREA_CENTRE) * LOG2_AREA_SCALE
}

/// A 0/1 flag as a feature.
#[must_use]
pub fn flag_feature(flag: bool) -> f32 {
    if flag { 1.0 } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_zero_is_the_fixed_threshold() {
        assert_eq!(LinearThreshold::<3>::new(0.0, [0.0; 3]), None);
    }

    #[test]
    fn shift_is_bias_plus_weighted_features() {
        let shift = LinearThreshold::new(0.5, [1.0, -2.0]).unwrap();
        assert_eq!(shift.at(&[3.0, 1.0]), 1.5);
    }
}
