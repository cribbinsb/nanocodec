//! Long-term reference with receiver feedback (`docs/ARCHITECTURE.md` §6.1).
//!
//! A flat chain predicts every frame from the previous one, so one lost
//! frame corrupts everything after it until a key frame. With a long-term
//! reference (LTR) the encoder keeps one older frame the receiver is known
//! to hold, and after a loss report codes a *recovery frame* that predicts
//! from it instead, which costs far less than a key frame:
//!
//! - every `ltr.interval` frames the frame being coded is also stored in the
//!   LTR slot, unless the previous LTR is still waiting for its
//!   acknowledgement (one slot: a new mark would discard the only frame the
//!   receiver might hold); key frames are always marked;
//! - [`LongTermReference::acknowledge`] records that the receiver decoded a
//!   frame; when it is the marked one, it becomes the acknowledged LTR;
//! - after [`LongTermReference::report_loss`] the next frame predicts from
//!   the acknowledged LTR (if it is older than the loss) and refreshes the
//!   short-term slot; without one the encoder codes a key frame.
//!
//! With temporal layers only base-layer (T0) frames are marked, so every
//! sub-stream a receiver may keep holds the LTR, and a recovery frame
//! restarts the layer pattern as a T0 frame refreshing every short-term
//! slot.
//!
//! Frame indices are the coded-frame indices of `FrameMetadata::index`.

use nc_analysis::link::{ReferenceLink, SlotMask};

use crate::config::EncoderConfig;
use crate::params::LtrParams;
use crate::structure::{FrameReferences, longest_reference_distance, reference_slots};

/// The reference slot holding the long-term reference in a structure of
/// `temporal_layers` layers: the first one its pattern does not use.
#[must_use]
pub fn long_term_slot(temporal_layers: u8) -> usize {
    reference_slots(temporal_layers)
}

/// Oldest frame any reference may name, in frames. Codecs size their
/// reference syntax to it: AV1 `delta_frame_id_minus_1` (§5.9.2; nc-av1
/// codes frame ids so its decoder can tell a lost reference, §7.20) and
/// HEVC POC deltas (at most 2^15, §7.4.8). A long-term reference this old
/// is replaced, and no longer recovers a loss. 256 frames is 8.5 s at
/// 30 fps; the LTR sweep (`docs/handoff/loss-resilience.md`) wants marks
/// every 1-2 s.
pub const MAX_REFERENCE_AGE: u64 = 1 << 8;

/// Age, in frames, at which the long-term reference of a stream keyed
/// every `intra_period` frames is too old: [`MAX_REFERENCE_AGE`], or
/// `intra_period + 1` when keys come sooner. A periodic key refreshes the
/// long-term slot, so the LTR is at most `intra_period - 1` frames old and
/// this limit never fires before the key: it only bounds the age for codecs
/// that size their reference syntax to it ([`reference_reach`]).
fn long_term_age_limit(intra_period: u32) -> u64 {
    (u64::from(intra_period) + 1).min(MAX_REFERENCE_AGE)
}

/// Whether `config` asks for a long-term reference (`ltr.interval` > 0).
#[must_use]
pub fn long_term_requested(config: &EncoderConfig) -> bool {
    LtrParams::from_overrides(&config.params).interval.round() as u32 > 0
}

/// Farthest back, in coded frames, the reference of any frame
/// (`FrameReferences::reference`) of a stream configured by `config`
/// reaches: the structure's longest distance and, with a long-term
/// reference, the oldest it may get. Codecs that code reference distances
/// size that syntax to it (AV1 `delta_frame_id_length`, nc-av1 `obu.rs`).
#[must_use]
pub fn reference_reach(config: &EncoderConfig) -> u64 {
    let short_term = longest_reference_distance(config.temporal_layers);
    if long_term_requested(config) {
        short_term.max(long_term_age_limit(config.intra_period) - 1)
    } else {
        short_term
    }
}

/// What the LTR policy does to the next frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LtrAction {
    /// Code it as planned.
    None,
    /// Also store it in the LTR slot.
    Mark,
    /// Predict it from the acknowledged LTR (a loss was reported).
    Recover,
    /// Code a key frame (a loss was reported and no LTR can recover it).
    Key,
}

/// The LTR state between frames.
#[derive(Debug, Clone)]
pub(crate) struct LongTermReference {
    /// The slot holding the LTR; the slots below it are the structure's.
    slot: usize,
    /// Frames between LTR marks.
    interval: u64,
    /// Age at which the LTR is replaced and no longer recovers a loss
    /// ([`long_term_age_limit`]).
    age_limit: u64,
    /// Index of the frame in the LTR slot, if any.
    held: Option<u64>,
    /// Whether the receiver acknowledged `held`.
    acknowledged: bool,
    /// Index of the frame most recently marked (the mark cadence's origin).
    last_mark: u64,
    /// The earliest lost frame reported and not yet recovered.
    lost: Option<u64>,
}

impl LongTermReference {
    /// LTR marks every `interval` frames (at least 1) in slot `slot`, in a
    /// stream keyed every `intra_period` frames.
    pub fn new(interval: u32, slot: usize, intra_period: u32) -> Self {
        Self {
            slot,
            interval: u64::from(interval.max(1)),
            age_limit: long_term_age_limit(intra_period),
            held: None,
            acknowledged: false,
            last_mark: 0,
            lost: None,
        }
    }

    /// The receiver decoded frame `index`.
    pub fn acknowledge(&mut self, index: u64) {
        if self.held == Some(index) {
            self.acknowledged = true;
        }
    }

    /// The receiver is missing frame `index`.
    pub fn report_loss(&mut self, index: u64) {
        self.lost = Some(self.lost.map_or(index, |l| l.min(index)));
    }

    /// Whether a reported loss is not yet recovered: the receiver's picture
    /// is broken until the next coded frame repairs it.
    pub fn recovery_pending(&self) -> bool {
        self.lost.is_some()
    }

    /// What to do with inter frame `index`, a base-layer frame if `base`
    /// (only those are marked); `Mark` and `Recover` must be followed by
    /// [`LongTermReference::coded`] once the frame is coded.
    pub fn action(&self, index: u64, base: bool) -> LtrAction {
        match (self.lost, self.held) {
            (Some(lost), Some(held))
                if self.acknowledged && held < lost && index - held < self.age_limit =>
            {
                LtrAction::Recover
            }
            (Some(_), _) => LtrAction::Key,
            (None, _) if !base => LtrAction::None,
            (None, _) => {
                let due = index >= self.last_mark + self.interval;
                // Codecs bound how far back a reference may be; a mark that
                // old is replaced, acknowledged or not.
                let too_old = self
                    .held
                    .is_some_and(|held| index - held + 1 >= self.age_limit);
                if (due && (self.held.is_none() || self.acknowledged)) || too_old {
                    LtrAction::Mark
                } else {
                    LtrAction::None
                }
            }
        }
    }

    /// `references` of inter frame `index` changed by `action`. A
    /// recovery frame's `references` must be those of a restarted pattern
    /// (position 0, T0).
    pub fn apply(
        &self,
        action: LtrAction,
        index: u64,
        references: FrameReferences,
    ) -> FrameReferences {
        match action {
            LtrAction::None | LtrAction::Key => references,
            LtrAction::Mark => FrameReferences {
                refresh: SlotMask(references.refresh.0 | SlotMask::only(self.slot).0),
                ..references
            },
            LtrAction::Recover => FrameReferences {
                reference: ReferenceLink {
                    slot: self.slot,
                    distance: u32::try_from(index - self.held.unwrap_or(index))
                        .unwrap_or(u32::MAX)
                        .max(1),
                },
                // Every short-term slot: the frames they held may be lost.
                refresh: SlotMask((1 << self.slot) - 1),
                ..references
            },
        }
    }

    /// Frame `index` was coded: as a key frame (`key`, which every slot
    /// holds now), or as an inter frame after `action`.
    pub fn coded(&mut self, index: u64, key: bool, action: LtrAction) {
        if key || action == LtrAction::Mark {
            self.held = Some(index);
            self.acknowledged = false;
            self.last_mark = index;
        }
        if key || action == LtrAction::Recover {
            self.lost = None;
        }
    }

    /// Whether the next frame must not be dropped: it is due to be marked
    /// (only base-layer frames are, and only top-layer frames of a layered
    /// structure are ever dropped).
    pub fn protects(&self, index: u64) -> bool {
        self.action(index, true) != LtrAction::None
    }
}

#[cfg(test)]
mod tests {
    use super::{LongTermReference, LtrAction, MAX_REFERENCE_AGE, long_term_slot, reference_reach};
    use crate::config::{Codec, EncoderConfig, FrameRate, StreamSignal};
    use crate::params::ParamOverrides;
    use crate::structure::FrameReferences;

    const FLAT_SLOT: usize = 1;
    /// Key frames rarely enough not to matter to the policy tests.
    const INTRA_PERIOD: u32 = 300;

    #[test]
    fn marks_waits_for_acknowledgement_and_recovers() {
        let mut ltr = LongTermReference::new(4, FLAT_SLOT, INTRA_PERIOD);
        ltr.coded(0, true, LtrAction::None); // key frame: the first LTR
        assert_eq!(
            ltr.action(4, true),
            LtrAction::None,
            "LTR 0 not acknowledged yet"
        );
        ltr.acknowledge(0);
        assert_eq!(ltr.action(4, true), LtrAction::Mark);
        ltr.coded(4, false, LtrAction::Mark);
        ltr.acknowledge(4);
        ltr.report_loss(6);
        assert_eq!(ltr.action(7, true), LtrAction::Recover);
        let refs = ltr.apply(LtrAction::Recover, 7, FrameReferences::FLAT);
        assert_eq!(
            (refs.reference.slot, refs.reference.distance),
            (FLAT_SLOT, 3)
        );
        assert!(!refs.refresh.contains(FLAT_SLOT));
        ltr.coded(7, false, LtrAction::Recover);
        assert_eq!(ltr.action(8, true), LtrAction::Mark);
    }

    #[test]
    fn a_loss_without_an_acknowledged_older_ltr_needs_a_key() {
        let mut ltr = LongTermReference::new(4, FLAT_SLOT, INTRA_PERIOD);
        ltr.coded(0, true, LtrAction::None);
        ltr.report_loss(2);
        assert_eq!(ltr.action(3, true), LtrAction::Key);
        ltr.acknowledge(0);
        assert_eq!(ltr.action(3, true), LtrAction::Recover);
        // The LTR itself lost: no help.
        ltr.report_loss(0);
        assert_eq!(ltr.action(3, true), LtrAction::Key);
    }

    #[test]
    fn layered_streams_mark_only_base_frames() {
        assert_eq!(long_term_slot(1), FLAT_SLOT);
        assert_eq!(long_term_slot(2), FLAT_SLOT);
        let mut ltr = LongTermReference::new(2, FLAT_SLOT, INTRA_PERIOD);
        ltr.coded(0, true, LtrAction::None);
        ltr.acknowledge(0);
        assert_eq!(
            ltr.action(3, false),
            LtrAction::None,
            "a T1 frame is not marked"
        );
        assert_eq!(ltr.action(4, true), LtrAction::Mark);
        assert!(ltr.protects(3));
    }

    #[test]
    fn an_unacknowledged_mark_is_replaced_at_the_age_limit() {
        let mut ltr = LongTermReference::new(4, FLAT_SLOT, INTRA_PERIOD);
        ltr.coded(0, true, LtrAction::None);
        // The key's acknowledgement was lost: no new mark until it is old.
        assert_eq!(ltr.action(MAX_REFERENCE_AGE - 2, true), LtrAction::None);
        assert_eq!(ltr.action(MAX_REFERENCE_AGE - 1, true), LtrAction::Mark);
        ltr.acknowledge(0);
        ltr.report_loss(MAX_REFERENCE_AGE - 1);
        assert_eq!(
            ltr.action(MAX_REFERENCE_AGE, true),
            LtrAction::Key,
            "the LTR is too old to recover from"
        );
    }

    #[test]
    fn reach_covers_the_structure_and_the_ltr_within_the_key_period() {
        let mut config = EncoderConfig {
            codec: Codec::Av1,
            width: 64,
            height: 64,
            frame_rate: FrameRate { num: 30, den: 1 },
            quantizer: 180,
            intra_period: 60,
            temporal_layers: 3,
            block_log2: 4,
            key_qp_offset: None,
            signal: StreamSignal::default(),
            params: ParamOverrides::default(),
        };
        assert_eq!(reference_reach(&config), 4);
        config.params.set("ltr.interval", 15.0);
        assert_eq!(reference_reach(&config), 60);
        config.intra_period = 3000;
        assert_eq!(reference_reach(&config), MAX_REFERENCE_AGE - 1);
    }
}
