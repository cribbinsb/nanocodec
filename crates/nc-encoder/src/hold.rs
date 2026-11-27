//! Frame hold: an input frame that would not change the picture is not
//! coded at all; the decoder keeps showing the last frame.
//!
//! Cameras and pipelines repeat frames (a 15 fps source sent at 30 fps, a
//! stalled capture, a frozen screen). Coding a repeat costs a frame header
//! and, worse, counts as a frame everywhere: rate control budgets it,
//! the temporal-layer pattern and key period advance on it, and in a layered
//! structure it can take a base-layer place. A held frame takes no place:
//! no coded unit, no frame index, and no state change except time. Rate
//! control drains the buffers for its interval and spreads the inter budget
//! it saved over the next coded frames (`RateController::hold_frame`), so
//! the frame after a hold is budgeted like any other. At fixed QP the
//! stream is the one the input without the repeats would give; under rate
//! control it differs only in that spreading (the input without repeats
//! hands a long interval's whole budget to one frame). All codecs signal it
//! the same way, by sending nothing: a decoder shows a picture until the
//! next one arrives (H.264/HEVC/VVC carry no timing in the elementary
//! stream, containers carry timestamps; AV1/VP9/AV2 `show_existing_frame`
//! would need the last frame in a reference slot, which a top-layer frame
//! never is, and would still cost a unit).
//!
//! **Decision.** Holding a frame shows the last coded frame instead, so the
//! input is compared with the last coded frame's source (not the previous
//! input, so slow change cannot creep through a run of holds), per 8x8 luma
//! block, from the frame's pre-analysis, which its analysis reuses: the
//! test adds no pass over the picture.
//!
//! - When the frame's reference is the last coded frame (a flat chain), each
//!   block's SAD against it must be at most `hold.change` × the current
//!   inter step: the change's energy is at most its SAD, so no transform
//!   coefficient of it reaches half a step (0.5) and none would be coded.
//!   It must also be at most `hold.max_mean_abs` per sample (1): that bound
//!   holds however coarse the step, since motion compensation fixes a small
//!   displacement cheaply (a sample-energy bound of half a step held
//!   `static_london_bus` frames with small moving details at H.264 QP 51:
//!   `J_real` +4.7%). So exact and near-exact repeats are held.
//! - Otherwise (a layered frame predicted from further back) only an exact
//!   repeat is held: every block's sum and spread equal the last coded
//!   frame's.
//!
//! A frame is held whole or not at all.
//!
//! **Recurrence.** Only a repeat source is held: a repeat counts when another
//! came at most `hold.recur_frames` input frames before it, so the first
//! repeat of a run is coded and an isolated one always is. Coding a repeat
//! costs a near-empty frame; holding one shifts rate control (its interval's
//! budget moves to the frames after it). Clips often duplicate their first
//! frame: holding that repeat right after the first key lost 0.6-6.3%
//! `J_real` on five of six such held-out clips (AV1, `broad_independent_v1`:
//! `cevo_indoor_night` -6.3%, `nyc_crowd` -3.8%), while the gains on repeat
//! sources (every 2nd or 4th frame) need every repeat after the first.
//!
//! **Refinement.** When the last key or high-quality frame (base layer,
//! cadence base) was coded much coarser than the current inter level (a
//! starved key, a rate dip), a frame is coded even if nothing moved: the
//! reuse term (`reuse.rs`) refines still content only in coded frames, and
//! holding would freeze the poor picture. Upper-layer and cadence frames are
//! coarser by design and do not count.
//!
//! **Spare-budget refinement** (`hold.refine_fullness`, rate control only).
//! A scene that becomes exactly still after motion (a webcam whose subject
//! stops or leaves) is otherwise held at the quality it had while it moved,
//! with the budget of every held frame unspent: the inter level does not
//! move while nothing is coded, so the coarser-key test above never fires
//! (scene-sanity 2026-11-22, the demo's waving-hand scene, 854x480 CBR
//! 400 kbps: the still room stayed at 35-37 dB luma for its 2.5 s, against
//! 42-43 dB with every frame coded, at 348-354 against 400 kbps). With it, a
//! repeat that would be held, and that repeats a repeat (the source has
//! stopped; a repeat source alternates new and repeated frames and keeps
//! its holds), is coded instead, as an ordinary inter frame
//! that rate control budgets (the held time's budget included) and the reuse
//! term refines, while the receiver's buffer is below that share of its
//! capacity, so the bits stay under the CBR ceiling. The first refinement
//! frames cost little: rate control lowers the inter level only as cheap
//! frames leave budget unspent, and over several frames. So refinement ends
//! after [`REFINE_PATIENCE`] refinement frames in a row each take under
//! [`REFINE_DONE_SHARE`] of an inter budget without the inter level getting
//! finer (the level floor or the quality cap reached): the repeats after
//! them are held until a frame that is not a repeat is coded.
//!
//! **Repeat frames** (`hold.code_skip`). Encoders that code every input
//! frame (the reference encoders of the benchmark procedure) are compared
//! with streams that carry one picture per input frame. With it on, a held
//! frame is coded instead as a repeat frame: an ordinary inter frame in the
//! structure (it takes a frame index, a layer place and a key-period place,
//! refreshes its reference slots and rate control charges its bits), every
//! block of which the backend codes as its skip at the zero vector on the
//! primary reference, with no residual (H.264 `P_Skip`, HEVC/VVC skip CUs,
//! AV1/AV2/VP9 skip blocks), at the largest leaf size, with the in-loop
//! filters off, so its picture is a copy of the reference and it costs the
//! codec's frame header plus a few bits a block. A frame is coded as a
//! repeat only where its reference is the frame just before it (a flat
//! chain, or a layered frame one frame on from its reference) and intra
//! refresh is off; elsewhere the held frame is coded as usual, so the
//! decoder never shows an older picture. A repeat frame is not a
//! refinement frame: it leaves spare-budget refinement where it was.

use nc_analysis::preanalysis::PreAnalysis;

use crate::config::{EncoderConfig, TimingSignal};
use crate::params::HoldParams;

/// Samples of a pre-analysis block (8x8).
const BLOCK_SAMPLES: f64 = 64.;
/// Share of an inter budget under which a refinement frame counts as having
/// nothing left to refine (module doc, "Spare-budget refinement"): a still
/// frame coded at the level it was already refined to is little more than
/// its skip flags.
const REFINE_DONE_SHARE: f64 = 0.1;
/// Cheap refinement frames in a row at an unchanged level that end
/// refinement (module doc): rate control's level follows unspent budget over
/// a few frames, and a cheap frame costs little more than its skip flags.
const REFINE_PATIENCE: u32 = 8;

/// Whether an encoder for `config` holds frames: `hold.enable`, unless the
/// stream signals a constant frame rate ([`TimingSignal::Constant`]), which
/// a frame left out would contradict. Backends whose headers otherwise
/// claim a fixed rate (H.264 `fixed_frame_rate_flag`) read this.
#[must_use]
pub fn holds_frames(config: &EncoderConfig) -> bool {
    HoldParams::from_overrides(&config.params).enable >= 0.5
        && config.signal.timing != TimingSignal::Constant
}

/// When frames are held (`hold.*` parameters), and the input frames since
/// the last repeat.
#[derive(Debug, Clone)]
pub(crate) struct FrameHold {
    /// Largest block SAD, as a fraction of the quantizer step.
    change: f64,
    /// Largest block SAD, whatever the step.
    max_block_sad: f64,
    /// Largest ratio of the last key or high-quality frame's step to the
    /// current one.
    coarser_ratio: f64,
    /// Most input frames from one repeat to the next for the second to be
    /// held.
    recur_frames: u32,
    /// Input frames since the last repeat (`None` before the first).
    since_repeat: Option<u32>,
    /// Buffer fullness below which a held repeat is coded as a refinement
    /// (`hold.refine_fullness`; 0: never).
    refine_fullness: f64,
    /// Whether the last refinement frame still gained (or none was coded
    /// since the last frame that was not a repeat).
    refine_gaining: bool,
    /// Whether the frame being coded is a held repeat coded as a refinement.
    refining: bool,
    /// Whether the last input repeated the one before it too (a still
    /// source, not a repeat source alternating new and repeated frames).
    still_run: bool,
    /// The inter level after the last refinement frame (`None`: none yet).
    refine_level: Option<u32>,
    /// Cheap refinement frames in a row at an unchanged level.
    refine_idle: u32,
    /// What a held frame becomes (`hold.code_skip`).
    held_as: HeldAs,
}

/// What a held frame becomes in the stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HeldAs {
    /// Nothing is sent.
    Nothing,
    /// A repeat frame (module doc, "Repeat frames").
    Repeat,
}

/// What a frame is compared with: the last coded frame's pre-analysis, and
/// whether the frame's own pre-analysis was measured against that frame.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LastCoded<'a> {
    pub pre: &'a PreAnalysis,
    pub referenced: bool,
}

impl FrameHold {
    /// The hold policy of `config`, or `None` when it holds no frames.
    pub fn new(config: &EncoderConfig) -> Option<Self> {
        let params = HoldParams::from_overrides(&config.params);
        holds_frames(config).then(|| Self {
            change: f64::from(params.change),
            max_block_sad: f64::from(params.max_mean_abs) * BLOCK_SAMPLES,
            coarser_ratio: f64::from(params.refine_octaves).exp2(),
            // The range [0, 64] makes the cast exact.
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            recur_frames: params.recur_frames.round() as u32,
            since_repeat: None,
            refine_fullness: f64::from(params.refine_fullness),
            refine_gaining: true,
            refining: false,
            still_run: false,
            refine_level: None,
            refine_idle: 0,
            held_as: if params.code_skip >= 0.5 {
                HeldAs::Repeat
            } else {
                HeldAs::Nothing
            },
        })
    }

    /// Whether a held frame is coded as a repeat frame instead of sending
    /// nothing (module doc, "Repeat frames").
    pub fn emits_repeats(&self) -> bool {
        self.held_as == HeldAs::Repeat
    }

    /// Whether a repeat that [`Self::holds`] is coded instead as a
    /// refinement frame, at the receiver's buffer `fullness` (share of its
    /// capacity; `None` without rate control): module doc, "Spare-budget
    /// refinement". Call once per held repeat, before it is coded or held.
    pub fn refine_instead(&mut self, fullness: Option<f64>) -> bool {
        self.refining = self.still_run
            && self.refine_gaining
            && fullness.is_some_and(|f| f < self.refine_fullness);
        self.refining
    }

    /// Note a coded frame: its bits as a share of an inter budget and rate
    /// control's inter level for it. A frame that is not a refinement
    /// re-arms refinement; [`REFINE_PATIENCE`] cheap refinements in a row at
    /// an unchanged level end it.
    pub fn coded(&mut self, budget_share: f64, level: Option<u32>) {
        if self.refining {
            let finer = match (level, self.refine_level) {
                (Some(now), Some(before)) => now < before,
                _ => true,
            };
            let idle = budget_share < REFINE_DONE_SHARE && !finer;
            self.refine_idle = if idle { self.refine_idle + 1 } else { 0 };
            self.refine_gaining = self.refine_idle < REFINE_PATIENCE;
            self.refine_level = level;
        } else {
            self.refine_gaining = true;
            self.refine_level = None;
            self.refine_idle = 0;
        }
        self.refining = false;
    }

    /// Whether to hold an input frame that is (`repeat`) or is not a repeat
    /// of the last coded picture ([`Self::repeats`]): only a repeat that
    /// recurs (module doc, "Recurrence"). Call once per input frame.
    pub fn holds(&mut self, repeat: bool) -> bool {
        let since = self.since_repeat.map(|n| n.saturating_add(1));
        self.since_repeat = if repeat { Some(0) } else { since };
        // A repeat right after a repeat: the source has stopped (a repeat
        // source alternates new and repeated frames).
        self.still_run = repeat && since == Some(1);
        repeat && since.is_some_and(|n| n <= self.recur_frames)
    }

    /// Whether the frame with pre-analysis `next` repeats the last coded
    /// frame: showing that frame instead would not change the picture.
    /// `step` is the current inter level's quantizer step, `last_step` the
    /// last key or high-quality frame's (sample units).
    pub fn repeats(
        &self,
        next: &PreAnalysis,
        last: LastCoded<'_>,
        (step, last_step): (f64, f64),
    ) -> bool {
        if last_step > step * self.coarser_ratio {
            return false;
        }
        let (blocks, before) = (next.blocks.cells(), last.pre.blocks.cells());
        if last.referenced {
            let limit = (self.change * step).min(self.max_block_sad);
            blocks
                .iter()
                .all(|b| b.sad_prev.is_some_and(|sad| f64::from(sad) <= limit))
        } else {
            blocks.len() == before.len()
                && blocks
                    .iter()
                    .zip(before)
                    .all(|(b, a)| (b.sum, b.sum_sq_dev) == (a.sum, a.sum_sq_dev))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Codec, FrameRate, StreamSignal};
    use crate::params::ParamOverrides;
    use nc_analysis::grid::Grid;
    use nc_analysis::preanalysis::BlockStats;

    fn config(timing: TimingSignal) -> EncoderConfig {
        EncoderConfig {
            codec: Codec::H264,
            width: 64,
            height: 64,
            frame_rate: FrameRate { num: 30, den: 1 },
            quantizer: 30,
            intra_period: 30,
            temporal_layers: 1,
            block_log2: 4,
            key_qp_offset: None,
            signal: StreamSignal {
                timing,
                ..StreamSignal::default()
            },
            params: ParamOverrides::default(),
        }
    }

    fn policy() -> FrameHold {
        FrameHold::new(&config(TimingSignal::Nominal)).unwrap()
    }

    /// A one-row pre-analysis of blocks with these SADs and sums.
    fn pre(blocks: &[(u16, u16)]) -> PreAnalysis {
        let cells = blocks
            .iter()
            .map(|&(sad, sum)| BlockStats {
                sad_prev: Some(sad),
                sum,
                ..BlockStats::default()
            })
            .collect();
        PreAnalysis {
            blocks: Grid::from_cells(blocks.len(), 1, cells),
            static_fraction: 0.,
            mean_luma: 0.,
        }
    }

    #[test]
    fn a_constant_rate_stream_holds_nothing() {
        assert!(holds_frames(&config(TimingSignal::Variable)));
        assert!(!holds_frames(&config(TimingSignal::Constant)));
        let mut off = config(TimingSignal::Nominal);
        off.params = ParamOverrides::default().coding_every_frame();
        assert!(!holds_frames(&off));
    }

    #[test]
    fn a_change_below_half_a_step_and_one_per_sample_holds() {
        let last = pre(&[(0, 100), (0, 200)]);
        let against = |next: &PreAnalysis, step: f64| {
            let last = LastCoded {
                pre: &last,
                referenced: true,
            };
            policy().repeats(next, last, (step, step))
        };
        assert!(against(&pre(&[(0, 100), (0, 200)]), 16.));
        assert!(against(&pre(&[(0, 100), (8, 200)]), 16.));
        assert!(!against(&pre(&[(0, 100), (9, 200)]), 16.));
        // However coarse the step, no more than 1 per sample.
        assert!(against(&pre(&[(64, 100), (0, 200)]), 1000.));
        assert!(!against(&pre(&[(65, 100), (0, 200)]), 1000.));
    }

    #[test]
    fn against_another_reference_only_exact_repeats_hold() {
        let last = pre(&[(0, 100), (0, 200)]);
        let other = |next: &PreAnalysis| {
            let last = LastCoded {
                pre: &last,
                referenced: false,
            };
            policy().repeats(next, last, (16., 16.))
        };
        assert!(other(&pre(&[(500, 100), (500, 200)])));
        assert!(!other(&pre(&[(0, 100), (0, 201)])));
    }

    #[test]
    fn only_recurring_repeats_hold() {
        let mut hold = policy();
        // An isolated repeat (a duplicated first frame) is coded.
        assert!(!hold.holds(true));
        for _ in 0..8 {
            assert!(!hold.holds(false));
        }
        assert!(!hold.holds(true));
        // A repeat within `recur_frames` inputs of the last one is held.
        for _ in 0..7 {
            assert!(!hold.holds(false));
        }
        assert!(hold.holds(true));
        assert!(!hold.holds(false));
        assert!(hold.holds(true));
    }

    #[test]
    fn spare_budget_refinement_codes_repeats_until_they_stop_gaining() {
        assert!(!policy().refine_instead(Some(0.)), "off by default");
        let mut on = config(TimingSignal::Nominal);
        on.params.set("hold.refine_fullness", 0.5);
        let mut hold = FrameHold::new(&on).unwrap();
        // A repeat source (every other input) keeps its holds.
        hold.holds(true);
        hold.holds(false);
        assert!(hold.holds(true));
        assert!(!hold.refine_instead(Some(0.2)), "a repeat source is held");
        // A still source: a repeat of a repeat.
        assert!(hold.holds(true));
        assert!(!hold.refine_instead(Some(0.5)), "the buffer has no room");
        assert!(
            !hold.refine_instead(None),
            "fixed QP has no budget to spend"
        );
        assert!(hold.refine_instead(Some(0.2)));
        hold.coded(0.05, Some(40));
        assert!(
            hold.refine_instead(Some(0.2)),
            "the first refinement starts a run"
        );
        hold.coded(0.05, Some(38));
        assert!(
            hold.refine_instead(Some(0.2)),
            "cheap, but the level got finer"
        );
        hold.coded(0.4, Some(38));
        assert!(hold.refine_instead(Some(0.2)), "still spending");
        for _ in 1..REFINE_PATIENCE {
            hold.coded(0.05, Some(38));
            assert!(
                hold.refine_instead(Some(0.2)),
                "patient while cheap at one level"
            );
        }
        hold.coded(0.05, Some(38));
        assert!(
            !hold.refine_instead(Some(0.2)),
            "cheap at an unchanged level: done"
        );
        hold.coded(0.05, Some(38));
        assert!(hold.refine_instead(Some(0.2)), "a coded change re-arms it");
    }

    #[test]
    fn a_much_coarser_last_frame_is_coded_to_be_refined() {
        let same = pre(&[(0, 100)]);
        let last = LastCoded {
            pre: &same,
            referenced: true,
        };
        assert!(policy().repeats(&same, last, (10., 20.)));
        assert!(!policy().repeats(&same, last, (10., 20.5)));
    }
}
