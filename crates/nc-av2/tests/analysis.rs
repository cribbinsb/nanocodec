//! Analysis records (feature `analysis`): collecting them leaves decoding
//! bit-exact, and each frame's coding blocks tile the picture.
#![cfg(feature = "analysis")]

use nc_av2::stream;
use nc_common::stream_stats::{Pred, RecordKind, StreamStats, keys};
use nc_common::{Picture, PlaneIndex};

const SIDE: usize = 64;
const QUANTIZER: u8 = 120;
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
    let mut encoder = stream::Encoder::default();
    let mut decoder = stream::Decoder::default();
    decoder.collect_stats(StreamStats::default());
    for phase in 0..FRAMES {
        let encoded = encoder.encode(&picture(phase), QUANTIZER, false).unwrap();
        let decoded = decoder.decode_ref(&encoded.packet).unwrap();
        for plane in PlaneIndex::ALL {
            let (actual, expected) = (
                decoded.plane(plane).view(),
                encoded.reconstruction.plane(plane).view(),
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
        let intra = frame.get(keys::CB_PRED, Pred::Intra as i64).weight;
        assert_eq!(intra == covered, index == 0, "frame {index}");
    }
}
