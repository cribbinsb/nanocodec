//! AV2 deblocking ramps ([`Av2Ramps`]) generic over [`I32Lanes`]: the
//! decisions stay scalar (`deblock::av2`), the sample updates run in
//! vectors.
//!
//! - Horizontal edge: the four lines are four consecutive bytes of each
//!   row, so one 4-lane vector per moved row applies every line at once.
//! - Vertical edge: each line's moved samples are consecutive; one window
//!   of [`AV2_MAX_RAMP`] samples on each side of the edge is updated with
//!   per-position factors that are zero beyond the ramp width (a zero
//!   offset rounds to zero, so those samples are rewritten unchanged).
//!   Lines whose window would leave the plane use the scalar line.
//!
//! A whole chunk of segments is filtered by `av2_decide` instead.
//!
//! The functions are `unsafe` only because `V`'s instructions need its ISA
//! (see the `simd` module docs).

#![allow(clippy::inline_always)]

use super::av2::{AV2_EDGE_LINES, AV2_MAX_RAMP, AV2_RAMP_SHIFT, Av2Ramps, ramp_line};
use crate::simd::I32Lanes;

/// Ramps of a horizontal edge (`along == 1`); `V` has one lane per line.
///
/// # Safety
///
/// `V`'s ISA must be available.
#[inline(always)]
pub(crate) unsafe fn horizontal<V: I32Lanes>(plane: &mut [u8], ramps: &Av2Ramps) {
    debug_assert!(V::LANES == AV2_EDGE_LINES && ramps.along == 1);
    let (width_neg, width_pos) = ramps.widths;
    let scaled_neg = V::load(&ramps.scaled.map(|(neg, _)| neg));
    let scaled_pos = V::load(&ramps.scaled.map(|(_, pos)| pos));
    for i in 0..width_neg {
        let offset = scaled_neg
            .mul_scalar((width_neg - i) as i32)
            .round_shift(AV2_RAMP_SHIFT);
        offset.add_to_u8(&mut plane[ramps.q0 - (i + 1) * ramps.across..]);
    }
    for i in 0..width_pos {
        let offset = scaled_pos
            .mul_scalar((width_pos - i) as i32)
            .round_shift(AV2_RAMP_SHIFT);
        V::splat(0)
            .sub(offset)
            .add_to_u8(&mut plane[ramps.q0 + i * ramps.across..]);
    }
}

/// Ramps of a vertical edge (`across == 1`), `V::LANES` positions of a
/// line per step (a divisor of [`AV2_MAX_RAMP`]).
///
/// # Safety
///
/// `V`'s ISA must be available.
#[inline(always)]
pub(crate) unsafe fn vertical<V: I32Lanes>(plane: &mut [u8], ramps: &Av2Ramps) {
    debug_assert!(ramps.across == 1 && AV2_MAX_RAMP.is_multiple_of(V::LANES));
    let (width_neg, width_pos) = ramps.widths;
    // Window position k on the p side is sample p(AV2_MAX_RAMP − 1 − k); on
    // the q side, sample q(k). The factor is the ramp's `width − i`, or 0.
    let factors_neg: [i32; AV2_MAX_RAMP] =
        core::array::from_fn(|k| (width_neg + k + 1).saturating_sub(AV2_MAX_RAMP) as i32);
    let factors_pos: [i32; AV2_MAX_RAMP] =
        core::array::from_fn(|k| width_pos.saturating_sub(k) as i32);
    for (line, &(scaled_neg, scaled_pos)) in ramps.scaled.iter().enumerate() {
        let q0 = ramps.q0 + line * ramps.along;
        if q0 < AV2_MAX_RAMP || q0 + AV2_MAX_RAMP > plane.len() {
            ramp_line(
                plane,
                q0,
                ramps.across,
                ramps.widths,
                (scaled_neg, scaled_pos),
            );
            continue;
        }
        let window = &mut plane[q0 - AV2_MAX_RAMP..q0 + AV2_MAX_RAMP];
        let (neg, pos) = window.split_at_mut(AV2_MAX_RAMP);
        for k in (0..AV2_MAX_RAMP).step_by(V::LANES) {
            let offset = V::load(&factors_neg[k..])
                .mul(V::splat(scaled_neg))
                .round_shift(AV2_RAMP_SHIFT);
            offset.add_to_u8(&mut neg[k..]);
            let offset = V::load(&factors_pos[k..])
                .mul(V::splat(scaled_pos))
                .round_shift(AV2_RAMP_SHIFT);
            V::splat(0).sub(offset).add_to_u8(&mut pos[k..]);
        }
    }
}
