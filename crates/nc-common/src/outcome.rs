//! What a receiver does with each frame: show it, or freeze on the last
//! good picture and say why. One type for every codec and every front end
//! (the Rust decoders' callers, the C API, the browser decoder), so a
//! player tells a lost frame from a corrupt one the same way everywhere and
//! knows which frame to name in its loss report (`docs/handoff/loss-resilience.md`).
//!
//! The decoders find a lost reference from the stream itself where its
//! syntax allows (H.264 `frame_num`, HEVC POC, AV1 frame ids; not VP9 or
//! AV2) and return [`DecodeError::MissingReference`] or
//! [`DecodeError::MissingKey`]. A receiver that also has the transport's
//! dependency signal (the frames each frame predicts from, as the AV1 RTP
//! dependency descriptor carries) checks it first with [`ReceivedFrames`],
//! which also covers VP9 and AV2 and names the missing frame.

use std::collections::VecDeque;

use crate::decode::DecodeError;

/// Why a frame was not shown. The picture on screen stays the last good one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FreezeReason {
    /// The frame predicts from a picture the receiver does not hold: an
    /// earlier frame was lost (or refused). Decoding resumes at a key frame
    /// or at a recovery frame from a picture the receiver holds.
    MissingReference,
    /// Nothing can be decoded yet: the key frame (and its parameter sets)
    /// that starts the stream was lost.
    MissingKey,
    /// The frame's data is malformed, truncated or outside the supported
    /// subset. Frames that predict from it are refused in turn.
    Corrupt,
}

impl FreezeReason {
    /// The reason a decoder's error stands for.
    #[must_use]
    pub fn of(error: &DecodeError) -> Self {
        match error {
            DecodeError::MissingReference => Self::MissingReference,
            DecodeError::MissingKey => Self::MissingKey,
            DecodeError::Truncated
            | DecodeError::Malformed(_)
            | DecodeError::Unsupported(_)
            | DecodeError::InvalidData(_) => Self::Corrupt,
        }
    }

    /// A short stable name (`missing-reference`, `missing-key`, `corrupt`).
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::MissingReference => "missing-reference",
            Self::MissingKey => "missing-key",
            Self::Corrupt => "corrupt",
        }
    }
}

/// The outcome of one frame at the receiver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameOutcome {
    /// Decoded: the new picture is shown.
    Decoded,
    /// Not decoded: the last good picture stays on screen.
    Frozen {
        /// Why.
        reason: FreezeReason,
        /// The coded frame (`FrameMetadata::index`) to name in the loss
        /// report: the missing reference when the transport's dependency
        /// signal named it, otherwise the refused frame itself (the encoder
        /// repairs everything a reported frame depends on). `None` when the
        /// caller gave no frame index.
        report: Option<u64>,
    },
}

impl FrameOutcome {
    /// The outcome of decoding frame `index` (if known) with `result`.
    #[must_use]
    pub fn of<T>(result: &Result<T, DecodeError>, index: Option<u64>) -> Self {
        match result {
            Ok(_) => Self::Decoded,
            Err(error) => Self::Frozen {
                reason: FreezeReason::of(error),
                report: index,
            },
        }
    }
}

/// Frames remembered as decoded; references further back are treated as
/// missing (codecs bound reference age to 256 frames,
/// `nc_encoder::MAX_REFERENCE_AGE`).
const REMEMBERED_FRAMES: usize = 256;

/// The receiver side of the transport's dependency signal: which coded
/// frames were decoded, so a frame whose references are not all among them
/// is refused before it reaches the decoder.
#[derive(Debug, Clone, Default)]
pub struct ReceivedFrames {
    decoded: VecDeque<u64>,
}

impl ReceivedFrames {
    /// The first of `references` that was not decoded, if any.
    #[must_use]
    pub fn missing(&self, references: &[u64]) -> Option<u64> {
        references
            .iter()
            .copied()
            .find(|reference| !self.decoded.contains(reference))
    }

    /// Frame `index` was decoded.
    pub fn decoded(&mut self, index: u64) {
        if self.decoded.len() == REMEMBERED_FRAMES {
            self.decoded.pop_front();
        }
        self.decoded.push_back(index);
    }

    /// Frame `index` turned out not to be a usable reference (it decoded,
    /// but not to the sender's picture).
    pub fn forget(&mut self, index: u64) {
        self.decoded.retain(|&decoded| decoded != index);
    }

    /// Run `decode` for frame `index` unless one of `references` is missing,
    /// and record the outcome.
    pub fn receive<T>(
        &mut self,
        index: u64,
        references: &[u64],
        decode: impl FnOnce() -> Result<T, DecodeError>,
    ) -> (FrameOutcome, Option<T>) {
        if let Some(needed) = self.missing(references) {
            let outcome = FrameOutcome::Frozen {
                reason: if self.decoded.is_empty() {
                    FreezeReason::MissingKey
                } else {
                    FreezeReason::MissingReference
                },
                report: Some(needed),
            };
            return (outcome, None);
        }
        let result = decode();
        let outcome = FrameOutcome::of(&result, Some(index));
        if outcome == FrameOutcome::Decoded {
            self.decoded(index);
        }
        (outcome, result.ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_map_to_reasons() {
        assert_eq!(
            FreezeReason::of(&DecodeError::MissingReference),
            FreezeReason::MissingReference
        );
        assert_eq!(
            FreezeReason::of(&DecodeError::MissingKey),
            FreezeReason::MissingKey
        );
        for corrupt in [
            DecodeError::Truncated,
            DecodeError::Malformed("x"),
            DecodeError::Unsupported("x"),
            DecodeError::InvalidData("x".into()),
        ] {
            assert_eq!(FreezeReason::of(&corrupt), FreezeReason::Corrupt);
        }
    }

    #[test]
    fn dependency_signal_names_the_missing_frame() {
        let mut frames = ReceivedFrames::default();
        let (outcome, _) = frames.receive(1, &[0], || Ok::<_, DecodeError>(()));
        assert_eq!(
            outcome,
            FrameOutcome::Frozen {
                reason: FreezeReason::MissingKey,
                report: Some(0)
            }
        );
        assert_eq!(
            frames.receive(2, &[], || Ok::<_, DecodeError>(())).0,
            FrameOutcome::Decoded
        );
        assert_eq!(
            frames.receive(4, &[3], || Ok::<_, DecodeError>(())).0,
            FrameOutcome::Frozen {
                reason: FreezeReason::MissingReference,
                report: Some(3)
            }
        );
        let (outcome, _) = frames.receive(5, &[2], || Err::<(), _>(DecodeError::Truncated));
        assert_eq!(
            outcome,
            FrameOutcome::Frozen {
                reason: FreezeReason::Corrupt,
                report: Some(5)
            }
        );
        assert_eq!(frames.missing(&[2, 5]), Some(5));
    }
}
