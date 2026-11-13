//! SVT-AV1's light-PD1 pre-transform skip score (`lpd1_should_perform_tx`,
//! `docs/reference/svt-av1-m12-rtc.md` §3), shared by the AV1-family
//! backends: an inter block whose luma prediction error is clearly below
//! what the quantizer would keep skips its luma transform trials.
//!
//! The score adds a low-energy signal (with neighbour bonuses) and a
//! rate–distortion signal, minus a protection that keeps fine quantizers
//! (where small residuals are still coded) out of the skip. Rate units are
//! bits, not SVT's 1/512 bits. Ported from SVT-AV1 (BSD-3-Clause-Clear,
//! Alliance for Open Media; see `THIRD_PARTY.md`).

/// AV1 8-bit `ac_qlookup` steps at qindex 32, 64 and 128, SVT's protection
/// band edges, so codecs on another quantizer scale find their band by step.
const AV1_STEP_AT_QINDEX_32: f32 = 39.0;
const AV1_STEP_AT_QINDEX_64: f32 = 71.0;
const AV1_STEP_AT_QINDEX_128: f32 = 176.0;

/// SVT's quantizer bands for the low-quantizer protection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizerBand {
    /// AV1 qindex below 32.
    Fine,
    /// AV1 qindex 32..64.
    Low,
    /// AV1 qindex 64..128.
    Mid,
    /// AV1 qindex 128 and above.
    Coarse,
}

impl QuantizerBand {
    /// The band of AV1 (8-bit) qindex `qindex`.
    #[must_use]
    pub fn from_av1_qindex(qindex: u32) -> Self {
        match qindex {
            0..32 => Self::Fine,
            32..64 => Self::Low,
            64..128 => Self::Mid,
            _ => Self::Coarse,
        }
    }

    /// The band of AC step `step` on AV1's 8-bit `ac_qlookup` scale: the
    /// band whose AV1 qindex range has that step.
    #[must_use]
    pub fn from_av1_step(step: f32) -> Self {
        if step < AV1_STEP_AT_QINDEX_32 {
            Self::Fine
        } else if step < AV1_STEP_AT_QINDEX_64 {
            Self::Low
        } else if step < AV1_STEP_AT_QINDEX_128 {
            Self::Mid
        } else {
            Self::Coarse
        }
    }
}

/// Pinned SVT light-PD1 pre-transform score. Neighbour bonuses (`(both
/// skipped, both NEARESTMV)`) apply only with low prediction energy and both
/// neighbours available. Low-quantizer penalties protect detail, especially
/// in luma-dominant input.
#[must_use]
pub fn pre_tx_skip_score(
    low_energy: bool,
    neighbours: Option<(bool, bool)>,
    rd_skip: bool,
    band: QuantizerBand,
    luma_dominant: bool,
) -> i32 {
    let mut score = 0;
    if low_energy {
        score += 50;
        if let Some((both_skip, both_nearest)) = neighbours {
            score += if both_skip { 20 } else { 0 };
            score += if both_nearest { 15 } else { 0 };
            score += if both_skip && both_nearest { 15 } else { 0 };
        }
    }
    if rd_skip {
        score += RD_SKIP_SCORE;
    }
    score
        - match (band, luma_dominant) {
            (QuantizerBand::Fine, true) => 200,
            (QuantizerBand::Fine, false) => 20,
            (QuantizerBand::Low, true) => 100,
            (QuantizerBand::Mid, true) => 60,
            _ => 0,
        }
}

/// What the rate–distortion signal adds to the score.
const RD_SKIP_SCORE: i32 = 150;

/// Whether an inter block skips its luma transforms: its
/// [`pre_tx_skip_score`] reaches `threshold`.
///
/// Speed (exact): the rate–distortion signal `rd_skip`, the costly one (a
/// zero test of the whole block's residual, or the coded alternative's
/// fixed-guess pricing), is evaluated only where it decides: not when the
/// other signals reach `threshold` without it, nor when even its bonus
/// cannot. On AV2 Fast that settles low-energy blocks between skipped
/// neighbours at coarse quantizers and every block at fine ones.
#[must_use]
pub fn pre_tx_skips(
    low_energy: bool,
    neighbours: Option<(bool, bool)>,
    rd_skip: impl FnOnce() -> bool,
    band: QuantizerBand,
    luma_dominant: bool,
    threshold: f32,
) -> bool {
    let without = pre_tx_skip_score(low_energy, neighbours, false, band, luma_dominant);
    if without as f32 >= threshold {
        true
    } else if ((without + RD_SKIP_SCORE) as f32) < threshold {
        false
    } else {
        rd_skip()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deciding_lazily_matches_scoring_with_the_rd_signal() {
        let bands = [
            QuantizerBand::Fine,
            QuantizerBand::Low,
            QuantizerBand::Mid,
            QuantizerBand::Coarse,
        ];
        let neighbours = [
            None,
            Some((false, false)),
            Some((true, false)),
            Some((true, true)),
        ];
        for band in bands {
            for luma_dominant in [false, true] {
                for low_energy in [false, true] {
                    for n in neighbours {
                        for rd in [false, true] {
                            for threshold in [0.0, 50.0, 70.0, 70.5, 120.0, 250.0] {
                                let scored =
                                    pre_tx_skip_score(low_energy, n, rd, band, luma_dominant)
                                        as f32
                                        >= threshold;
                                let lazy = pre_tx_skips(
                                    low_energy,
                                    n,
                                    || rd,
                                    band,
                                    luma_dominant,
                                    threshold,
                                );
                                assert_eq!(lazy, scored);
                            }
                        }
                    }
                }
            }
        }
    }

    fn score(low_energy: bool, neighbours: Option<(bool, bool)>, rd: bool) -> i32 {
        pre_tx_skip_score(low_energy, neighbours, rd, QuantizerBand::Coarse, true)
    }

    #[test]
    fn pre_transform_score_keeps_energy_neighbour_and_rd_signals_separate() {
        assert_eq!(score(false, Some((true, true)), false), 0);
        assert_eq!(score(true, None, false), 50);
        assert_eq!(score(true, Some((true, false)), false), 70);
        assert_eq!(score(true, Some((false, true)), false), 65);
        assert_eq!(score(true, Some((true, true)), false), 100);
        assert_eq!(score(false, None, true), 150);
        assert_eq!(score(true, Some((true, true)), true), 250);
    }

    #[test]
    fn pre_transform_quantizer_protection_matches_source_boundaries() {
        for (q, penalty) in [
            (31, 200),
            (32, 100),
            (63, 100),
            (64, 60),
            (127, 60),
            (128, 0),
        ] {
            let band = QuantizerBand::from_av1_qindex(q);
            assert_eq!(
                pre_tx_skip_score(false, None, true, band, true),
                150 - penalty
            );
            assert_eq!(
                pre_tx_skip_score(false, None, true, band, false),
                if q < 32 { 130 } else { 150 }
            );
        }
    }

    #[test]
    fn step_bands_follow_the_av1_table_edges() {
        assert_eq!(QuantizerBand::from_av1_step(38.0), QuantizerBand::Fine);
        assert_eq!(QuantizerBand::from_av1_step(39.0), QuantizerBand::Low);
        assert_eq!(QuantizerBand::from_av1_step(71.0), QuantizerBand::Mid);
        assert_eq!(QuantizerBand::from_av1_step(175.0), QuantizerBand::Mid);
        assert_eq!(QuantizerBand::from_av1_step(176.0), QuantizerBand::Coarse);
    }
}
