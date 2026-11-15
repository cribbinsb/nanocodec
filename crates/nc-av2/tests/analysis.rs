//! Analysis records (feature `analysis`): collecting them leaves decoding
//! bit-exact, and each frame's coding blocks tile the picture.
#![cfg(feature = "analysis")]

use nc_av2::{Av2Backend, Decoder};
use nc_common::stream_stats::{Pred, RecordKind, StreamStats, keys};
use nc_common::{Picture, PlaneIndex};
use nc_encoder::{Codec, Encoder, EncoderConfig, FrameRate, ParamOverrides, StreamSignal};

const SIDE: usize = 64;
const QUANTIZER: u32 = 120;
const FRAMES: usize = 4;

fn picture(phase: usize) -> Picture {
    let mut picture = Picture::new(SIDE, SIDE, 0).unwrap();
    for plane in PlaneIndex::ALL {
        let mut view = picture.plane_mut(plane).view_mut();
        for y in 0..view.height() {
            for (x, sample) in view.row_mut(y).iter_mut().enumerate() {
                *sample = ((x * 5 + y * 3 + phase * 2) % 256) as u8;
            }
        }
    }
    picture
}

#[test]
fn analysis_leaves_decoding_unchanged_and_tiles_each_frame() {
    let config = EncoderConfig {
        codec: Codec::Av2,
        width: SIDE,
        height: SIDE,
        frame_rate: FrameRate { num: 30, den: 1 },
        quantizer: QUANTIZER,
        intra_period: 30,
        temporal_layers: 1,
        block_log2: 3,
        key_qp_offset: None,
        signal: StreamSignal::default(),
        params: ParamOverrides::default().coding_every_frame(),
    };
    let mut encoder = Encoder::<Av2Backend>::new(config).unwrap();
    let mut decoder = Decoder::default();
    decoder.collect_stats(StreamStats::default());
    for phase in 0..FRAMES {
        let packet = encoder.encode(&picture(phase)).data;
        let decoded = decoder.decode_ref(&packet).unwrap();
        for plane in PlaneIndex::ALL {
            let (actual, expected) = (
                decoded.plane(plane).view(),
                encoder.reconstruction().plane(plane).view(),
            );
            for row in 0..actual.height() {
                assert_eq!(actual.row(row), expected.row(row), "frame {phase}");
            }
        }
        let records = decoder.stats().unwrap().records();
        assert!(records.iter().any(|r| r.kind == RecordKind::Prediction));
    }
    let stats = decoder.take_stats().unwrap();
    assert_eq!(stats.frames.len(), FRAMES);
    for (index, frame) in stats.frames.iter().enumerate() {
        let covered: u64 = frame.tallies[keys::CB_SIZE]
            .values()
            .map(|t| t.weight)
            .sum();
        assert_eq!(
            covered,
            (SIDE * SIDE) as u64,
            "frame {index}: blocks tile the frame"
        );
        // The key is all intra; the shared engine's inter frames may mix in
        // intra blocks.
        if index == 0 {
            let intra = frame.get(keys::CB_PRED, Pred::Intra as i64).weight;
            assert_eq!(intra, covered, "frame {index}");
        }
    }
}
