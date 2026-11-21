//! The quarter-sample step predicted from the half-sample stage's costs
//! where the fixed-stage search does not run its quarter stage
//! (`decision.subpel_quarter_predict`, off by default).
//!
//! The sub-sample search in mode decision (`decision.rs`) probes the four
//! half-sample neighbours of its start, then the four quarter-sample
//! neighbours of the winner. The fast rungs skip the quarter stage when no
//! half-sample move won (`decision.subpel_quarter_needs_half_move`) or
//! always (`decision.subpel_half_only`), for speed. Where the half stage did
//! not move, its four probes and the centre give, per axis, three costs a
//! half sample apart; the parabola through them predicts the cost at each
//! quarter-sample neighbour (Hill, Chiew, Bull and Canagarajah,
//! "Interpolation free subpixel accuracy motion estimation", IEEE TCSVT
//! 16(12), 2006). The neighbour predicted best is then either probed once
//! and taken only if it measures cheaper ([`QuarterPredict::Probe`]: it
//! can never end worse than the half-stage vector), or taken unmeasured
//! where the predicted gain is a large enough share of the centre's
//! distortion ([`QuarterPredict::Model`]).
//!
//! Measured in `docs/analysis/literature-ideas-2026-11-21.md` §4.

/// What the search does where it skips the quarter stage after a half stage
/// that did not move.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum QuarterPredict {
    /// Nothing: the half-stage vector stands (the default).
    Off,
    /// Probe the predicted best quarter neighbour once; keep it if cheaper.
    Probe,
    /// Take the predicted best quarter neighbour unmeasured where its
    /// predicted gain is at least `strength` times the centre's distortion.
    Model { strength: f32 },
}

impl QuarterPredict {
    /// The mode of `decision.subpel_quarter_predict` (0 off, 1 probe,
    /// 2 model) with `decision.subpel_quarter_predict_strength`.
    pub(crate) fn of(mode: f32, strength: f32) -> Self {
        if mode >= 1.5 {
            Self::Model { strength }
        } else if mode >= 0.5 {
            Self::Probe
        } else {
            Self::Off
        }
    }
}

/// The quarter neighbour the half stage's distortions predict best: its
/// index in the half stage's direction order (`half` holds the distortions
/// at the negative and positive x, then the negative and positive y
/// half-sample neighbours) and its predicted distortion, where that is
/// below the centre's (`centre`).
pub(crate) fn predicted_step(centre: f32, half: [f32; 4]) -> Option<(usize, f32)> {
    let mut best: Option<(usize, f32)> = None;
    for (axis, (minus, plus)) in [(half[0], half[1]), (half[2], half[3])]
        .into_iter()
        .enumerate()
    {
        // The parabola through (-1, minus), (0, centre), (1, plus) at -1/2
        // and +1/2 of the half-sample distance.
        let curvature = minus - 2.0 * centre + plus;
        let slope = (plus - minus) / 4.0;
        let bend = curvature / 8.0;
        for (side, predicted) in [(0, centre - slope + bend), (1, centre + slope + bend)] {
            if predicted < best.map_or(centre, |(_, d)| d) {
                best = Some((2 * axis + side, predicted));
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_symmetric_bowl_predicts_no_step() {
        assert_eq!(predicted_step(10.0, [20.0, 20.0, 30.0, 30.0]), None);
    }

    #[test]
    fn the_lower_side_of_a_tilted_bowl_is_predicted() {
        // x: minus 12, centre 10, plus 30: at -1/2: 10 - 4.5 + 2.75 = 8.25.
        let (index, predicted) = predicted_step(10.0, [12.0, 30.0, 40.0, 40.0]).unwrap();
        assert_eq!(index, 0);
        assert!((predicted - 8.25).abs() < 1e-6);
    }

    #[test]
    fn the_axis_with_the_larger_predicted_gain_wins() {
        // y: minus 40, plus 11: at +1/2: 10 - 7.25 + 3.875 = 6.625, below x's 8.25.
        let (index, _) = predicted_step(10.0, [12.0, 30.0, 40.0, 11.0]).unwrap();
        assert_eq!(index, 3);
    }

    #[test]
    fn modes_follow_the_parameter() {
        assert_eq!(QuarterPredict::of(0.0, 0.1), QuarterPredict::Off);
        assert_eq!(QuarterPredict::of(1.0, 0.1), QuarterPredict::Probe);
        assert_eq!(
            QuarterPredict::of(2.0, 0.1),
            QuarterPredict::Model { strength: 0.1 }
        );
    }
}
