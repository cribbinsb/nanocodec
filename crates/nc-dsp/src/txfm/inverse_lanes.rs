//! Minimal inverse-butterfly operations: what the generated inverse DCT
//! graphs (`dct.rs`) need from a lane type. The graph topology is generated
//! once; the lane type supplies the arithmetic policy:
//!
//! - every [`I32Lanes`] type: libaom's exact 32-bit AV1 arithmetic, whose
//!   stage clamps saturate to 16 bits;
//! - the guarded NEON and SIMD128 inverses (`narrow_inverse.rs`): the same
//!   results on 16-bit lanes, for blocks whose input L1 norm proves no stage
//!   leaves the 16-bit range;
//! - VP9's scalar adapter (`vp9_txfm.rs`): libvpx's normative arithmetic,
//!   which wraps intermediates to 16 bits instead of saturating.

use crate::simd::I32Lanes;

/// Largest input L1 norm (sum of absolute values of one 1-D transform's
/// inputs) for which every node of the 32- and 64-point inverse graphs, and
/// every equal-weight pre-add of the 16-bit `half_btf`, fits `i16`. Then
/// every 8-bit stage clamp is the identity and wrapping 16-bit arithmetic is
/// exact. An integer-overflow envelope, not an RD control: the
/// outward-rounded interval test in `range_probe` checks every stage.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
    test
))]
pub(super) const NARROW_INVERSE_L1_LIMIT: u16 = 23_000;

pub(crate) trait InverseLanes: Copy {
    fn splat(value: i32) -> Self;
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    /// libaom `half_btf`: `round_shift(w0 * a + w1 * b, bit)`.
    fn half_btf(w0: i32, a: Self, w1: i32, b: Self, bit: u32) -> Self;
    /// `half_btf` whose second input is a structural zero:
    /// `round_shift(w * a, bit)`.
    fn mul_round(w: i32, a: Self, bit: u32) -> Self;
    /// Clamp to the 8-bit stage range (16 bits).
    fn clamp16(self) -> Self;
}

// Plain `#[inline]`: in unoptimised builds every `inline(always)` wrapper
// level gets its own stack slots at each of the thousands of butterfly sites
// (the scalar 64-point inverse frame grew from 12 KiB to 45 KiB). Optimised
// builds inline these one-instruction forwards anyway.
impl<V: I32Lanes> InverseLanes for V {
    #[inline]
    fn splat(value: i32) -> Self {
        V::splat(value)
    }
    #[inline]
    fn add(self, other: Self) -> Self {
        I32Lanes::add(self, other)
    }
    #[inline]
    fn sub(self, other: Self) -> Self {
        I32Lanes::sub(self, other)
    }
    #[inline]
    fn half_btf(w0: i32, a: Self, w1: i32, b: Self, bit: u32) -> Self {
        V::half_btf(w0, a, w1, b, bit)
    }
    #[inline]
    fn mul_round(w: i32, a: Self, bit: u32) -> Self {
        a.mul_scalar(w).round_shift(bit)
    }
    #[inline]
    fn clamp16(self) -> Self {
        I32Lanes::clamp16(self)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn av1_policy_matches_original_libaom_inverse8() {
        // Original libaom v3.15.3 av1_idct8, cos_bit=12, stage_range=16.
        // Full i16 inputs exercise clamps that must not become VP9 wrapping.
        let mut state = 17u32;
        let mut hash = 14_695_981_039_346_656_037u64;
        for _ in 0..200 {
            let input = std::array::from_fn(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state & 65535) as i32 - 32768
            });
            let mut output = [0; 8];
            crate::txfm::dct::idct8(&input, &mut output, &crate::txfm::COSPI_12, 12);
            for value in output {
                for byte in value.to_le_bytes() {
                    hash = (hash ^ u64::from(byte)).wrapping_mul(1_099_511_628_211);
                }
            }
        }
        assert_eq!(hash, 0x4d22_0565_b03a_1ed4);
    }
}
