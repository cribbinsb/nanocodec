//! Every decoder's per-frame outcome (`nc_common::outcome`) under loss and
//! damage: a dropped frame, a dropped key frame, a truncated frame and
//! random garbage. No case may panic or show a picture decoded from the
//! wrong reference; malformed data is `Corrupt`, kept apart from a missing
//! reference or key.

use nc_common::decode::DecodeError;
use nc_common::outcome::{FrameOutcome, FreezeReason, ReceivedFrames};
use nc_common::{Picture, PlaneIndex};
use nc_encoder::{
    Codec, CodecBackend, EncodedFrame, Encoder, EncoderConfig, FrameRate, ParamOverrides,
    StreamSignal,
};

use crate::decode::{Decoder, Shown};

const WIDTH: usize = 64;
const HEIGHT: usize = 64;
const FRAMES: usize = 8;
/// Quantizer of the QP codecs, and qindex of the others.
const QP: u32 = 30;
const QINDEX: u32 = 120;

/// A moving pattern, so inter frames carry real vectors and residuals.
fn source(frame: usize) -> Picture {
    let mut picture = Picture::new(WIDTH, HEIGHT, 64).unwrap();
    for plane in PlaneIndex::ALL {
        let mut view = picture.plane_mut(plane).view_mut();
        for y in 0..view.height() {
            for (x, sample) in view.row_mut(y).iter_mut().enumerate() {
                *sample = (((x + 2 * frame) * 7 + (y + frame) * 13 + (x ^ y) * 3) % 251) as u8;
            }
        }
    }
    picture.extend_borders();
    picture
}

/// `FRAMES` coded frames (one key frame, then inter frames) and the MD5 of
/// each reconstruction.
fn stream<B: CodecBackend>(codec: Codec) -> Vec<(EncodedFrame, String)> {
    let mut params = ParamOverrides::default();
    if codec == Codec::Av1 {
        // In-band loss detection needs frame ids, off by default.
        params.set(nc_av1::FRAME_IDS_PARAM, 1.0);
    }
    let config = EncoderConfig {
        codec,
        width: WIDTH,
        height: HEIGHT,
        frame_rate: FrameRate { num: 30, den: 1 },
        quantizer: match codec {
            Codec::H264 | Codec::Hevc | Codec::Vvc => QP,
            Codec::Av1 | Codec::Vp9 | Codec::Av2 => QINDEX,
        },
        intra_period: 1000,
        block_log2: if codec == Codec::Av2 { 3 } else { 4 },
        key_qp_offset: None,
        temporal_layers: 1,
        signal: StreamSignal::default(),
        params,
    };
    let mut encoder = Encoder::<B>::new(config).unwrap();
    (0..FRAMES)
        .map(|index| {
            let frame = encoder.encode(&source(index));
            let md5 = Shown::whole(encoder.reconstruction().clone()).md5();
            (frame, md5)
        })
        .collect()
}

/// Deterministic pseudo-random bytes.
fn garbage(len: usize, seed: u32) -> Vec<u8> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        })
        .collect()
}

/// A receiver: our decoder behind the shared outcome, optionally with the
/// transport's dependency signal.
struct Receiver {
    decoder: Decoder,
    frames: ReceivedFrames,
    dependency_signal: bool,
}

impl Receiver {
    fn new(codec: Codec, dependency_signal: bool) -> Self {
        Self {
            decoder: Decoder::new(codec),
            frames: ReceivedFrames::default(),
            dependency_signal,
        }
    }

    /// The outcome of `data` as frame `frame`; a decoded picture must be
    /// the encoder's reconstruction.
    fn receive(&mut self, frame: &(EncodedFrame, String), data: &[u8]) -> FrameOutcome {
        let meta = &frame.0.metadata;
        let references: Vec<u64> = if self.dependency_signal {
            meta.references.iter().collect()
        } else {
            Vec::new()
        };
        let decoder = &mut self.decoder;
        let (outcome, decoded) = self.frames.receive(meta.index, &references, || {
            decoder
                .decode(data)?
                .ok_or(DecodeError::Malformed("packet shows no picture"))
        });
        if let Some(decoded) = decoded {
            assert_eq!(
                decoded.md5(),
                frame.1,
                "frame {} decoded wrongly",
                meta.index
            );
        }
        outcome
    }
}

fn frozen(reason: FreezeReason, report: u64) -> FrameOutcome {
    FrameOutcome::Frozen {
        reason,
        report: Some(report),
    }
}

/// Every case for one codec; `in_band` says whether its stream reveals a
/// lost reference by itself (H.264, HEVC, AV1; not VP9, AV2).
fn check_codec<B: CodecBackend>(codec: Codec, in_band: bool) {
    let frames = stream::<B>(codec);

    // A dropped key frame: nothing decodes, in band, for every codec.
    let mut receiver = Receiver::new(codec, false);
    for frame in &frames[1..3] {
        let index = frame.0.metadata.index;
        assert_eq!(
            receiver.receive(frame, &frame.0.data),
            frozen(FreezeReason::MissingKey, index),
            "{codec:?} frame {index} without its key frame"
        );
    }

    // A dropped inter frame (3): the frames after it are refused, by the
    // stream itself where it can tell, else by the dependency signal,
    // which names the missing frame.
    for dependency_signal in [false, true] {
        if !in_band && !dependency_signal {
            continue;
        }
        let mut receiver = Receiver::new(codec, dependency_signal);
        for frame in &frames {
            let meta = &frame.0.metadata;
            let index = meta.index;
            // Frame 3 is a reference here (frame 4 predicts from it).
            assert!(frames[4].0.metadata.references.iter().any(|r| r == 3));
            let outcome = match index {
                3 => continue,
                0..=2 => FrameOutcome::Decoded,
                // The signal names the frame's own reference: 3, or a
                // frame refused after it.
                _ if dependency_signal => frozen(
                    FreezeReason::MissingReference,
                    meta.references.iter().next().unwrap(),
                ),
                _ => frozen(FreezeReason::MissingReference, index),
            };
            assert_eq!(
                receiver.receive(frame, &frame.0.data),
                outcome,
                "{codec:?} frame {index}, dependency signal {dependency_signal}"
            );
        }
    }

    // A truncated frame is corrupt; nothing after it panics or shows a
    // wrong picture (checked in `receive`).
    let mut receiver = Receiver::new(codec, !in_band);
    for frame in &frames {
        let data = &frame.0.data;
        let index = frame.0.metadata.index;
        if index == 2 {
            let outcome = receiver.receive(frame, &data[..data.len() / 2]);
            assert_eq!(outcome, frozen(FreezeReason::Corrupt, 2), "{codec:?}");
        } else {
            receiver.receive(frame, data);
        }
    }

    // Garbage before the first key frame and between frames is refused
    // and leaves the decoder as it was: the next frame still decodes.
    // Garbage is usually `Corrupt`; bytes that happen to parse as a header
    // naming a reference the receiver lacks are refused as that.
    let mut receiver = Receiver::new(codec, false);
    for (seed, frame) in frames.iter().enumerate() {
        let noise = garbage(300, seed as u32 + 1);
        let outcome = receiver.receive(frame, &noise);
        assert_ne!(outcome, FrameOutcome::Decoded, "{codec:?} garbage {seed}");
        assert_eq!(
            receiver.receive(frame, &frame.0.data),
            FrameOutcome::Decoded,
            "{codec:?} frame {seed} after garbage"
        );
    }
}

#[test]
fn h264_outcomes() {
    check_codec::<nc_h264::H264Backend>(Codec::H264, true);
}

#[test]
fn hevc_outcomes() {
    check_codec::<nc_hevc::HevcBackend>(Codec::Hevc, true);
}

#[test]
fn av1_outcomes() {
    check_codec::<nc_av1::Av1Backend>(Codec::Av1, true);
}

#[test]
fn vp9_outcomes() {
    check_codec::<nc_vp9::Vp9Backend>(Codec::Vp9, false);
}

#[test]
fn av2_outcomes() {
    check_codec::<nc_av2::Av2Backend>(Codec::Av2, false);
}
