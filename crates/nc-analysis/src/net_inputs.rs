//! Integer inputs of the encoder's small networks: values formed with
//! exact IEEE operations and an integer logarithm, clamped to the
//! activation range, so every platform computes the same bytes.
//!
//! Shared by every network's input builder (the analysis' zero-gate network,
//! `gate_net.rs`; the encoder's trial and stop networks, `trial_inputs.rs`,
//! `stop_inputs.rs`): the label dump records the bytes those builders make,
//! and training reads them.

/// The largest input value: the network's activation range (clipped `ReLU`
/// in seven bits, so a u8 x i8 pair sum stays inside sixteen bits).
pub const INPUT_MAX: i32 = 127;

/// Steps per octave of [`log2_steps`].
const STEPS_PER_OCTAVE: i32 = 16;
/// Fraction bits of the fixed-point value [`log2_steps`] takes the log of.
const FIXED_BITS: u32 = 16;
/// Mantissa bits [`log2_steps`] resolves below the leading one.
const MANTISSA_BITS: u32 = 4;
/// `round(16 * log2(1 + m / 16))` for the four bits `m` below the leading one.
const MANTISSA_STEPS: [u8; 1 << MANTISSA_BITS] =
    [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15];

/// `16 * log2(x)` to the nearest step or so, for `x` in `[2^-16, 2^47]`
/// (clamped outside; 0 and NaN read as the smallest value). Integer after
/// one exact scaling, so it is the same on every platform.
#[must_use]
pub fn log2_steps(x: f32) -> i32 {
    let fixed = ((f64::from(x) * f64::from(1u32 << FIXED_BITS)) as u64).max(1);
    let exponent = fixed.ilog2();
    let mantissa = if exponent >= MANTISSA_BITS {
        fixed >> (exponent - MANTISSA_BITS)
    } else {
        fixed << (MANTISSA_BITS - exponent)
    } & ((1 << MANTISSA_BITS) - 1);
    STEPS_PER_OCTAVE * (exponent as i32 - FIXED_BITS as i32)
        + i32::from(MANTISSA_STEPS[mantissa as usize])
}

/// `value` clamped to the input range, as a byte.
#[must_use]
pub fn input(value: i32) -> u8 {
    value.clamp(0, INPUT_MAX) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log2_steps_matches_the_logarithm() {
        for &x in &[1.0f32, 1.5, 2.0, 3.0, 10.0, 0.25, 0.001, 12345.0, 1e6] {
            let exact = 16.0 * x.log2();
            assert!(
                (log2_steps(x) as f32 - exact).abs() <= 1.5,
                "{x}: {} vs {exact}",
                log2_steps(x)
            );
        }
        assert_eq!(log2_steps(1.0), 0);
        assert_eq!(log2_steps(2.0), STEPS_PER_OCTAVE);
        assert_eq!(log2_steps(0.0), log2_steps(f32::NAN));
    }

    #[test]
    fn log2_steps_never_decreases() {
        let mut last = i32::MIN;
        for i in 1..4000u32 {
            let steps = log2_steps(i as f32 / 64.0);
            assert!(steps >= last);
            last = steps;
        }
    }
}
