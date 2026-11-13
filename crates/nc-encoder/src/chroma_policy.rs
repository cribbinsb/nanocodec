//! The engine's chroma policy: how finely chroma is quantised relative to
//! luma, and how chroma error is weighed against bits, for every codec.
//!
//! One policy, decided across codecs (docs/ARCHITECTURE.md, "Chroma
//! allocation"). It is expressed in a codec-independent unit, the H.26x QP
//! of a sample-domain quantiser step ("QP-equivalent": 6 per doubling of
//! the step), so each backend maps the same target to its own syntax:
//! H.264 `chroma_qp_index_offset`, HEVC/VVC chroma offsets and the VVC SPS
//! chroma QP table, AV1/AV2 `delta_q_u/v`, VP9 `delta_q_uv`.
//!
//! - **Step target** ([`chroma_q`]): chroma follows luma's step up to QP-eq
//!   [`FOLLOW_UNTIL_Q`], then gets finer by half a QP per luma QP, to
//!   [`MAX_FINER_Q`] finer from QP-eq 42 on. This is the JVET SDR common test
//!   conditions' chroma QP map (`QpInValCb 17 22 34 42`, `QpOutValCb 17 23 35
//!   39`, also `VVenC`'s and VTM's default), without its one-QP coarser bump
//!   below QP 36. It sits in the middle of the reference encoders at their
//!   real-time defaults: SVT-AV1, libaom, libvpx and AVM code chroma at the
//!   luma qindex (0 finer); `VVenC` with its default QPA about 0-2 finer;
//!   x265 (HEVC Table 8-10, offset 0) 3-6 finer and x264 (Table 8-15,
//!   offset 0) 4-8 finer at QP 37-45. The registered `chroma.offset`
//!   shifts the curve; its default -1 (1 QP finer at every rate, so 1-4
//!   finer from QP 36 on) measured better than 0 in every codec screened.
//! - **Distortion weight** ([`distortion_weight`]): chroma squared error
//!   counts (`step_Y` / `step_C`)² times luma's against bits, so a finer
//!   chroma step is not traded away again by chroma's own decisions (HM,
//!   VTM and `VVenC` weigh chroma by `2^((QP_Y − QP_C)/3)`; libaom, SVT-AV1 and
//!   AVM by 1 at their equal steps).

use crate::params::{ChromaParams, ParamOverrides};

/// The policy as a backend applies it: `None` from [`ChromaPolicy::from_params`]
/// while `chroma.policy` is 0 (the codec keeps its own chroma settings).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChromaPolicy {
    /// `chroma.offset`, QP-equivalent units (positive is coarser chroma).
    pub offset: f64,
}

impl ChromaPolicy {
    /// The policy of an encoder's parameters, if switched on.
    #[must_use]
    pub fn from_params(params: &ParamOverrides) -> Option<Self> {
        let chroma = ChromaParams::from_overrides(params);
        (chroma.policy >= 0.5).then_some(Self {
            offset: f64::from(chroma.offset),
        })
    }

    /// [`chroma_q`] at this policy's offset.
    #[must_use]
    pub fn chroma_q(&self, luma_q: f64) -> f64 {
        chroma_q(luma_q, self.offset)
    }

    /// [`chroma_step`] at this policy's offset.
    #[must_use]
    pub fn chroma_step(&self, luma_step: f64) -> f64 {
        chroma_step(luma_step, self.offset)
    }
}

/// Luma QP-equivalent up to which chroma uses luma's step.
pub const FOLLOW_UNTIL_Q: f64 = 36.0;
/// Chroma QP-equivalent drop per luma QP above [`FOLLOW_UNTIL_Q`].
const FINER_SLOPE: f64 = 0.5;
/// Largest drop of chroma below luma, in QP-equivalent units.
pub const MAX_FINER_Q: f64 = 3.0;
/// H.264/HEVC quantiser step at QP 0 (`Qstep` doubles every 6 QP).
const STEP_AT_Q0: f64 = 0.625;
/// QP units per doubling of the step.
const Q_PER_OCTAVE: f64 = 6.0;

/// The policy's chroma QP-equivalent for luma QP-equivalent `luma_q`, plus
/// `offset` (the registered `chroma.offset`: positive is coarser chroma).
#[must_use]
pub fn chroma_q(luma_q: f64, offset: f64) -> f64 {
    let finer = ((luma_q - FOLLOW_UNTIL_Q) * FINER_SLOPE).clamp(0.0, MAX_FINER_Q);
    luma_q - finer + offset
}

/// QP-equivalent of a quantiser step in H.264/HEVC sample-domain units
/// (`Qstep`; AV1/VP9 8-bit `ac_q` / 8, AV2 likewise).
#[must_use]
pub fn q_of_step(step: f64) -> f64 {
    Q_PER_OCTAVE * (step / STEP_AT_Q0).log2()
}

/// The policy's chroma step for luma step `luma_step` (same units).
#[must_use]
pub fn chroma_step(luma_step: f64, offset: f64) -> f64 {
    let luma_q = q_of_step(luma_step);
    luma_step * 2f64.powf((chroma_q(luma_q, offset) - luma_q) / Q_PER_OCTAVE)
}

/// Offset into a monotone step table whose step is closest (in log domain)
/// to `target`: how a backend with a qindex table (AV1, VP9, AV2) realises
/// [`chroma_step`]. `step` gives the table's step at each index of `range`.
#[must_use]
pub fn closest_index(
    range: std::ops::RangeInclusive<i32>,
    step: impl Fn(i32) -> f64,
    target: f64,
) -> i32 {
    // Ties (a clamped table repeats its end steps) go to the index nearest
    // zero: the smallest offset that reaches the step.
    let distance = |i: i32| (step(i) / target).ln().abs();
    range
        .min_by(|&a, &b| {
            distance(a)
                .total_cmp(&distance(b))
                .then(a.abs().cmp(&b.abs()))
        })
        .expect("non-empty qindex range")
}

/// Weight of chroma squared error against luma's in decisions that trade
/// distortion for bits: (`luma_step` / `chroma_step`)².
#[must_use]
pub fn distortion_weight(luma_step: f64, chroma_step: f64) -> f64 {
    (luma_step / chroma_step).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chroma_follows_luma_then_gets_up_to_three_finer() {
        for (luma, chroma) in [
            (20.0, 20.0),
            (36.0, 36.0),
            (38.0, 37.0),
            (42.0, 39.0),
            (51.0, 48.0),
        ] {
            assert!((chroma_q(luma, 0.0) - chroma).abs() < 1e-9, "QP {luma}");
        }
        assert!((chroma_q(42.0, 2.0) - 41.0).abs() < 1e-9);
    }

    #[test]
    fn steps_round_trip_through_qp_equivalents() {
        assert!((q_of_step(STEP_AT_Q0 * 2.0) - 6.0).abs() < 1e-9);
        let luma = STEP_AT_Q0 * 2f64.powf(42.0 / 6.0);
        let chroma = chroma_step(luma, 0.0);
        assert!((q_of_step(chroma) - 39.0).abs() < 1e-9);
        assert!((distortion_weight(luma, chroma) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn closest_index_picks_the_nearest_step_in_log_domain() {
        let table = |i: i32| f64::from(i + 1);
        assert_eq!(closest_index(0..=9, table, 4.2), 3);
        assert_eq!(closest_index(0..=9, table, 100.0), 9);
    }
}
