//! Long-term reference recovery (`nc_encoder` `ltr.rs`): the long-term
//! slot's picture is a long-term entry of the reference picture lists
//! (`references.rs`, H.266 §7.3.10, §8.3.2), and after a reported loss a
//! recovery picture predicting from it decodes exactly in our decoder,
//! through the own-stream path (flat streams) and the general path
//! (layered streams), with the frames after the loss withheld from it.
use crate::{Decoder, VvcBackend};
use nc_common::{Picture, PlaneIndex};
use nc_encoder::{
    Codec, CodecBackend, Encoder, EncoderConfig, FrameRate, FrameType, ParamOverrides, StreamSignal,
};

const WIDTH: usize = 128;
const HEIGHT: usize = 64;

/// A texture panning by 3 samples per frame: temporal candidates and
/// vectors that the recovery picture's long reference distance changes.
fn panning(frame: usize) -> Picture {
    let mut picture = Picture::new(WIDTH, HEIGHT, 0).unwrap();
    for plane in PlaneIndex::ALL {
        let scale = if plane == PlaneIndex::Y { 1 } else { 2 };
        let mut view = picture.plane_mut(plane).view_mut();
        for y in 0..view.height() {
            for (x, v) in view.row_mut(y).iter_mut().enumerate() {
                let gx = x + 3 * frame / scale;
                *v = ((gx * 7 + y * 5 + (gx * y) % 23) % 200 + plane as usize * 20) as u8;
            }
        }
    }
    picture
}

fn exact(a: &Picture, b: &Picture) -> bool {
    PlaneIndex::ALL.iter().all(|&p| {
        let (a, b) = (a.plane(p).view(), b.plane(p).view());
        (0..a.height()).all(|y| a.row(y) == b.row(y))
    })
}

/// Code `frames` frames with temporal candidates (the Default's tools) and
/// an LTR marked every `interval` frames; the frames in `lost` never reach
/// the decoder and are reported at once. Every frame decoded matches the
/// encoder's reconstruction.
fn long_term_recovery(temporal_layers: u8, interval: f32, lost: &[u64], frames: u64) {
    let mut params = ParamOverrides::default();
    for &(name, value) in <VvcBackend as CodecBackend>::DEFAULT_PARAMS {
        params.set(name, value);
    }
    params.set("ltr.interval", interval);
    params.set("layers.adaptive", 0.0);
    let config = EncoderConfig {
        codec: Codec::Vvc,
        width: WIDTH,
        height: HEIGHT,
        frame_rate: FrameRate { num: 30, den: 1 },
        quantizer: 32,
        intra_period: 1000,
        block_log2: 6,
        key_qp_offset: None,
        temporal_layers,
        signal: StreamSignal::default(),
        params,
    };
    let mut encoder = Encoder::<VvcBackend>::new(config).unwrap();
    let mut decoder = Decoder::new();
    let mut recoveries = 0;
    for index in 0..frames {
        let packet = encoder.encode(&panning(index as usize));
        assert_eq!(packet.metadata.frame_type == FrameType::Key, index == 0);
        if lost.contains(&index) {
            encoder.report_loss(index);
            continue;
        }
        // A recovery frame predicts from before a loss it follows.
        let recovery = lost
            .iter()
            .any(|&l| l < index && packet.metadata.references.iter().any(|r| r < l));
        recoveries += usize::from(recovery);
        let picture = decoder
            .decode(&packet.data)
            .unwrap_or_else(|e| panic!("frame {index}: {e}"));
        assert!(exact(&picture, encoder.reconstruction()), "frame {index}");
        encoder.acknowledge(index);
    }
    assert!(recoveries > 0, "no frame after a loss");
}

/// Flat: the own-stream path.
#[test]
fn long_term_reference_recovers_a_lost_frame() {
    long_term_recovery(1, 4.0, &[9], 16);
}

/// The long-term picture 140 frames back: beyond the short-term distances
/// the 8-bit POC LSBs allow (`references::max_listed_distance`).
#[test]
fn long_term_reference_recovers_from_far_back() {
    long_term_recovery(1, 200.0, &[140], 146);
}

/// Layered: the general path.
#[test]
fn long_term_reference_recovers_with_two_layers() {
    long_term_recovery(2, 4.0, &[10, 14], 20);
}

#[test]
fn long_term_reference_recovers_with_three_layers() {
    long_term_recovery(3, 4.0, &[10, 14], 20);
}
