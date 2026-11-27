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
//!
//! Spare-budget refinement (`hold.refine_fullness`): a scene that stops
//! moving is refined by coding repeats while the budget allows, then held;
//! without it every later repeat is held at the quality the motion left.
//!
//! Repeat frames (`hold.code_skip`): a held frame is coded instead as a
//! copy of the frame before it, so every input gives a unit, and each such
//! unit reconstructs exactly the picture before it.
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
        signal: StreamSignal {
            timing: TimingSignal::Variable,
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

/// Luma PSNR of `recon` against `source`.
fn luma_psnr(source: &Picture, recon: &Picture) -> f64 {
    let (a, b) = (
        source.plane(PlaneIndex::Y).view(),
        recon.plane(PlaneIndex::Y).view(),
    );
    let mut sse = 0.0;
    for y in 0..a.height() {
        for (p, q) in a.row(y).iter().zip(b.row(y)) {
            sse += (f64::from(*p) - f64::from(*q)).powi(2);
        }
    }
    let mse = (sse / (a.width() * a.height()) as f64).max(1e-10);
    10.0 * (255.0 * 255.0 / mse).log10()
}

/// Motion for [`MOVING_FRAMES`] frames, then the last frame repeated for
/// [`STILL_FRAMES`]: the coded repeats, the luma PSNR when the motion stops
/// and at the end, and the bits.
fn still_after_motion<B: CodecBackend>(codec: Codec, refine: bool) -> (usize, f64, f64, u64) {
    const MOVING_FRAMES: usize = 12;
    const STILL_FRAMES: usize = 90;
    let mut params = ParamOverrides::default();
    params.set("hold.refine_fullness", if refine { 0.5 } else { 0.0 });
    let mut encoder = Encoder::<B>::new(EncoderConfig {
        codec,
        width: WIDTH,
        height: HEIGHT,
        frame_rate: FrameRate { num: 30, den: 1 },
        quantizer: 30,
        intra_period: 1000,
        block_log2: match codec {
            Codec::Av2 => 3,
            Codec::Vvc => 5,
            _ => 4,
        },
        key_qp_offset: None,
        temporal_layers: 1,
        signal: StreamSignal {
            timing: TimingSignal::Variable,
            ..StreamSignal::default()
        },
        params,
    })
    .unwrap();
    encoder
        .set_rate_control(RateControlConfig::new(TARGET_BPS))
        .unwrap();
    let (mut coded_repeats, mut bits, mut stopped) = (0, 0u64, 0.0);
    let last = frame(MOVING_FRAMES - 1);
    for index in 0..MOVING_FRAMES + STILL_FRAMES {
        let source = if index < MOVING_FRAMES {
            frame(index)
        } else {
            last.clone()
        };
        let coded = encoder.encode_at(&source, index as i64 * FRAME_US);
        bits += 8 * coded.data.len() as u64;
        if index == MOVING_FRAMES - 1 {
            stopped = luma_psnr(&last, encoder.reconstruction());
        }
        // The first repeat is always coded (a repeat counts when another recurs).
        if index > MOVING_FRAMES && !coded.metadata.held {
            coded_repeats += 1;
        }
    }
    (
        coded_repeats,
        stopped,
        luma_psnr(&last, encoder.reconstruction()),
        bits,
    )
}

fn check_refinement<B: CodecBackend>(codec: Codec) {
    let (held_repeats, _, held_end, _) = still_after_motion::<B>(codec, false);
    let (coded, stopped, refined_end, bits) = still_after_motion::<B>(codec, true);
    // Without refinement a repeat is coded only to refine a much coarser
    // last frame (`hold.refine_octaves`).
    assert!(
        coded > held_repeats,
        "{codec:?}: refinement coded {coded} repeats, without it {held_repeats}"
    );
    assert!(
        refined_end > held_end + 1.0 && refined_end > stopped + 1.0,
        "{codec:?}: refined to {refined_end:.2} dB, held {held_end:.2}, at the stop {stopped:.2}"
    );
    // Under the CBR ceiling over the whole run (a key's loan is repaid within it).
    let seconds = 102.0 * FRAME_US as f64 / 1e6;
    assert!(
        bits as f64 <= TARGET_BPS as f64 * seconds * (1.0 + CBR_RATE_SPREAD),
        "{codec:?}: {bits} bits over {seconds:.2} s"
    );
}

#[test]
fn h264_refines_a_still_scene_while_its_repeats_would_be_held() {
    check_refinement::<nc_h264::H264Backend>(Codec::H264);
}

#[test]
fn hevc_refines_a_still_scene_while_its_repeats_would_be_held() {
    check_refinement::<nc_hevc::HevcBackend>(Codec::Hevc);
}

#[test]
fn av1_refines_a_still_scene_while_its_repeats_would_be_held() {
    check_refinement::<nc_av1::Av1Backend>(Codec::Av1);
}

/// Largest repeat frame of the test picture (a few bits a block plus the
/// frame header).
const MAX_REPEAT_BYTES: usize = 64;

/// Every source frame given twice with `hold.code_skip`: each input codes a
/// unit, and every unit hold would have left out reconstructs the picture
/// before it exactly, at fixed QP and under CBR.
fn check_repeat_frames<B: CodecBackend>(codec: Codec) {
    for cbr in [false, true] {
        let mut params = ParamOverrides::default();
        params.set("hold.code_skip", 1.0);
        let mut encoder = Encoder::<B>::new(EncoderConfig {
            codec,
            width: WIDTH,
            height: HEIGHT,
            frame_rate: FrameRate { num: 30, den: 1 },
            quantizer: match codec {
                Codec::Av1 | Codec::Vp9 | Codec::Av2 => 140,
                Codec::Vvc => 34,
                _ => 30,
            },
            intra_period: 1000,
            block_log2: 4,
            key_qp_offset: None,
            temporal_layers: 1,
            signal: StreamSignal {
                timing: TimingSignal::Variable,
                ..StreamSignal::default()
            },
            params,
        })
        .unwrap();
        if cbr {
            encoder
                .set_rate_control(RateControlConfig::new(TARGET_BPS))
                .unwrap();
        }
        for index in 0..FRAMES {
            let source = frame(index);
            for copy in 0..2 {
                let before = encoder.reconstruction().clone();
                let time = (2 * index + copy) as i64 * FRAME_US;
                let coded = encoder.encode_at(&source, time);
                assert!(!coded.metadata.held && !coded.data.is_empty());
                // The first repeat is coded as usual (an isolated repeat is
                // not held); every later one is a repeat frame.
                if copy == 1 && index > 0 {
                    let bytes = coded.data.len();
                    assert!(
                        bytes <= MAX_REPEAT_BYTES,
                        "{codec:?} cbr {cbr}: repeat of frame {index} takes {bytes} bytes"
                    );
                    assert!(
                        planes_equal(&before, encoder.reconstruction()),
                        "{codec:?} cbr {cbr}: repeat of frame {index} is not a copy"
                    );
                }
            }
        }
    }
}

/// Whether two pictures' visible samples are equal in every plane.
fn planes_equal(a: &Picture, b: &Picture) -> bool {
    PlaneIndex::ALL.into_iter().all(|plane| {
        let (a, b) = (a.plane(plane).view(), b.plane(plane).view());
        (0..a.height()).all(|y| a.row(y) == b.row(y))
    })
}

#[test]
fn h264_repeat_frames_copy_the_picture_before() {
    check_repeat_frames::<nc_h264::H264Backend>(Codec::H264);
}

#[test]
fn hevc_repeat_frames_copy_the_picture_before() {
    check_repeat_frames::<nc_hevc::HevcBackend>(Codec::Hevc);
}

#[test]
fn vvc_repeat_frames_copy_the_picture_before() {
    check_repeat_frames::<nc_vvc::VvcBackend>(Codec::Vvc);
}

#[test]
fn av1_repeat_frames_copy_the_picture_before() {
    check_repeat_frames::<nc_av1::Av1Backend>(Codec::Av1);
}

#[test]
fn vp9_repeat_frames_copy_the_picture_before() {
    check_repeat_frames::<nc_vp9::Vp9Backend>(Codec::Vp9);
}

#[test]
fn av2_repeat_frames_copy_the_picture_before() {
    check_repeat_frames::<nc_av2::Av2Backend>(Codec::Av2);
}
