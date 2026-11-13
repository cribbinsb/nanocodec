//! Frame hold (`nc-encoder` `hold.rs`): repeated input frames are not coded
//! and take no place in the stream, so at fixed QP an input with every frame
//! repeated codes byte for byte as the input without the repeats, given the
//! same capture times. The first repeat is coded (only a recurring repeat is
//! held), so both inputs give the first frame twice. Under CBR rate control
//! spreads each held interval's bits over the next frames, where the input
//! without repeats gives the whole interval to one frame, so the streams
//! differ: there every later repeat is held and both spend the same within
//! `CBR_RATE_SPREAD`. Every codec, with the codec defaults (H.264: adaptive
//! layers).
use nc_common::{Picture, PlaneIndex};
use nc_encoder::{
    Codec, CodecBackend, Encoder, EncoderConfig, FrameRate, ParamOverrides, RateControlConfig,
    StreamSignal, TimingSignal,
};

const WIDTH: usize = 128;
const HEIGHT: usize = 64;
/// Distinct source frames.
const FRAMES: usize = 48;
/// Capture interval of the repeated input (a 15 fps source sent at 30 fps):
/// source frame `i`'s copies are captured from `2 i` intervals on.
const FRAME_US: i64 = 33_333;
const TARGET_BPS: u64 = 60_000;
/// Largest relative difference of the CBR streams' sizes.
const CBR_RATE_SPREAD: f64 = 0.05;

/// Source frame `index`: a fixed texture on the left, a texture moving 2
/// samples per frame on the right.
fn frame(index: usize) -> Picture {
    let mut picture = Picture::new(WIDTH, HEIGHT, 0).unwrap();
    for plane in PlaneIndex::ALL {
        let mut view = picture.plane_mut(plane).view_mut();
        let half = view.width() / 2;
        for y in 0..view.height() {
            for (x, v) in view.row_mut(y).iter_mut().enumerate() {
                let moved = if x < half { x } else { x + 2 * index };
                *v = (100.
                    + 60. * (moved as f64 * 0.3 + y as f64 * 0.2).sin()
                    + ((moved * 37 + y * 91) % 23) as f64) as u8;
            }
        }
    }
    picture
}

/// The coded units of the source frames, each given twice (`repeated`, and
/// always the first) or once at [`FRAME_US`] intervals, and how many inputs
/// were held.
fn encode<B: CodecBackend>(codec: Codec, cbr: bool, repeated: bool) -> (Vec<Vec<u8>>, usize) {
    let quantizer = match codec {
        Codec::Av1 | Codec::Vp9 | Codec::Av2 => 140,
        Codec::Vvc => 34,
        _ => 30,
    };
    let mut encoder = Encoder::<B>::new(EncoderConfig {
        codec,
        width: WIDTH,
        height: HEIGHT,
        frame_rate: FrameRate { num: 30, den: 1 },
        quantizer,
        intra_period: 1000,
        block_log2: match codec {
            Codec::Av2 => 3,
            Codec::Vvc => 5,
            _ => 4,
        },
        key_qp_offset: None,
        temporal_layers: 1,
        // VVC has no timing signalling (the stream is untimed).
        signal: StreamSignal {
            timing: if codec == Codec::Vvc {
                TimingSignal::Nominal
            } else {
                TimingSignal::Variable
            },
            ..StreamSignal::default()
        },
        params: ParamOverrides::default(),
    })
    .unwrap();
    if cbr {
        encoder
            .set_rate_control(RateControlConfig::new(TARGET_BPS))
            .unwrap();
    }
    let (mut units, mut held) = (Vec::new(), 0);
    for index in 0..FRAMES {
        let source = frame(index);
        let copies = if repeated || index == 0 { 2 } else { 1 };
        for copy in 0..copies {
            let time = (2 * index + copy) as i64 * FRAME_US;
            let coded = encoder.encode_at(&source, time);
            if coded.metadata.held {
                assert_eq!(coded.data, Vec::<u8>::new());
                held += 1;
            } else {
                units.push(coded.data);
            }
        }
    }
    (units, held)
}

fn check<B: CodecBackend>(codec: Codec) {
    for cbr in [false, true] {
        let (once, none) = encode::<B>(codec, cbr, false);
        let (twice, held) = encode::<B>(codec, cbr, true);
        assert_eq!((none, held), (0, FRAMES - 1), "{codec:?} cbr {cbr}");
        if cbr {
            let bytes = |units: &[Vec<u8>]| units.iter().map(Vec::len).sum::<usize>() as f64;
            let spread = (bytes(&twice) / bytes(&once) - 1.).abs();
            assert!(
                spread < CBR_RATE_SPREAD,
                "{codec:?}: sizes differ by {spread:.3} (once {}, twice {})",
                bytes(&once),
                bytes(&twice)
            );
        } else {
            assert!(once == twice, "{codec:?}: streams differ");
        }
    }
}

#[test]
fn h264_repeats_code_as_their_absence() {
    check::<nc_h264::H264Backend>(Codec::H264);
}

#[test]
fn hevc_repeats_code_as_their_absence() {
    check::<nc_hevc::HevcBackend>(Codec::Hevc);
}

#[test]
fn vvc_repeats_code_as_their_absence() {
    check::<nc_vvc::VvcBackend>(Codec::Vvc);
}

#[test]
fn av1_repeats_code_as_their_absence() {
    check::<nc_av1::Av1Backend>(Codec::Av1);
}

#[test]
fn vp9_repeats_code_as_their_absence() {
    check::<nc_vp9::Vp9Backend>(Codec::Vp9);
}

#[test]
fn av2_repeats_code_as_their_absence() {
    check::<nc_av2::Av2Backend>(Codec::Av2);
}
