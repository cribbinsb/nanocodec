//! The recode contract in every codec: coding every frame twice
//! (`debug.recode_identity`: a throw-away first coding at the other end of
//! the quantizer range, then the real one) leaves the stream and the
//! reconstruction unchanged. A frame re-coded for a buffer overflow
//! (`rate.recode_overflow`) goes through the same path, so any state the
//! discarded coding leaks changes real streams. (2026-11-20: the reuse
//! term kept the discarded coding's blocks; H.264 and AV1, whose backends
//! report copied blocks, changed from frame 58 / 21 of a 720p crowd clip at
//! fixed QP.) A coding stopped part-way (`debug.stop_identity`: halfway,
//! as an early re-code, `rate.recode_early`, stops one) skips the rest of
//! the frame, its filters and `end_frame`; the next coding must not see it.
use nc_common::{Picture, PlaneIndex};
use nc_encoder::{
    Codec, CodecBackend, Encoder, EncoderConfig, FrameRate, ParamOverrides, RateControlConfig,
    StreamSignal,
};

const WIDTH: usize = 128;
const HEIGHT: usize = 64;
const FRAMES: usize = 24;
/// The CBR case's target: low enough that the static half is refined by
/// the reuse term while the moving half takes the bits.
const TARGET_BPS: u64 = 40_000;

/// Frame `index`: a fixed texture with a little temporal noise on the
/// left (copied, refined and re-measured by the reuse term), a texture
/// moving 3 samples per frame on the right.
fn frame(index: usize) -> Picture {
    let mut picture = Picture::new(WIDTH, HEIGHT, 0).unwrap();
    for plane in PlaneIndex::ALL {
        let mut view = picture.plane_mut(plane).view_mut();
        for y in 0..view.height() {
            for (x, v) in view.row_mut(y).iter_mut().enumerate() {
                let noise = ((x * 31 + y * 17 + index * 101) % 7) as f64 / 3.;
                *v = if plane != PlaneIndex::Y {
                    128
                } else if x < WIDTH / 2 {
                    let (xf, yf) = (x as f64, y as f64);
                    (120. + 40. * (xf * 0.7 + yf * 0.3).sin() + 30. * (yf * 1.1).sin() + noise)
                        as u8
                } else {
                    let moved = x + 3 * index;
                    (128. + 50. * (moved as f64 * 0.5).sin() + ((moved * 53 + y * 17) % 29) as f64)
                        as u8
                };
            }
        }
    }
    picture
}

/// How a test stream is coded.
#[derive(Debug, Clone, Copy)]
enum Mode {
    /// A fixed mid-range quantizer: the throw-away coding is the coarsest.
    Fixed,
    /// The coarsest quantizer: the throw-away coding is the finest, as the
    /// discarded coding of an overflow re-code is finer than the kept one.
    Coarsest,
    /// Rate control.
    Cbr,
}

/// The two ways of coding every frame twice that must change nothing.
const IDENTITY_MODES: [&str; 2] = ["debug.recode_identity", "debug.stop_identity"];

/// The stream and reconstructions of `codec`'s Default coded in `mode`,
/// every frame once or, under an `identity` mode, twice.
fn coded<B: CodecBackend>(
    codec: Codec,
    mode: Mode,
    identity: Option<&str>,
) -> (Vec<u8>, Vec<Picture>) {
    let quantizer = match (mode, codec) {
        (Mode::Coarsest, Codec::Av1 | Codec::Vp9 | Codec::Av2) => 255,
        (Mode::Coarsest, Codec::Vvc) => 63,
        (Mode::Coarsest, _) => 51,
        (_, Codec::Av1 | Codec::Vp9 | Codec::Av2) => 140,
        _ => 32,
    };
    let block_log2 = match codec {
        Codec::Av2 => 3,
        Codec::Vvc => 5,
        _ => 4,
    };
    let mut params = ParamOverrides::default();
    if let Some(name) = identity {
        params.set(name, 1.);
    }
    let mut encoder = Encoder::<B>::new(EncoderConfig {
        codec,
        width: WIDTH,
        height: HEIGHT,
        frame_rate: FrameRate { num: 15, den: 1 },
        quantizer,
        intra_period: 1000,
        block_log2,
        key_qp_offset: None,
        temporal_layers: 1,
        signal: StreamSignal::default(),
        params,
    })
    .unwrap();
    if matches!(mode, Mode::Cbr) {
        encoder
            .set_rate_control(RateControlConfig::new(TARGET_BPS))
            .unwrap();
    }
    let mut stream = Vec::new();
    let mut recons = Vec::new();
    for index in 0..FRAMES {
        stream.extend(encoder.encode(&frame(index)).data);
        recons.push(encoder.reconstruction().clone());
    }
    (stream, recons)
}

/// Whether two reconstructions are equal in every visible sample.
fn same(a: &Picture, b: &Picture) -> bool {
    PlaneIndex::ALL.into_iter().all(|plane| {
        let (a, b) = (a.plane(plane).view(), b.plane(plane).view());
        (0..a.height()).all(|y| a.row(y) == b.row(y))
    })
}

fn check<B: CodecBackend>(codec: Codec) {
    for (mode, identity) in [Mode::Fixed, Mode::Coarsest, Mode::Cbr]
        .into_iter()
        .flat_map(|mode| IDENTITY_MODES.map(|identity| (mode, identity)))
    {
        let (once, once_recons) = coded::<B>(codec, mode, None);
        let (twice, twice_recons) = coded::<B>(codec, mode, Some(identity));
        let first_difference = once_recons
            .iter()
            .zip(&twice_recons)
            .position(|(a, b)| !same(a, b));
        assert_eq!(
            first_difference, None,
            "{codec:?} {mode:?} {identity}: reconstruction differs from frame"
        );
        assert!(
            once == twice,
            "{codec:?} {mode:?} {identity}: streams differ"
        );
    }
}

#[test]
fn recoding_changes_nothing_in_h264() {
    check::<nc_h264::H264Backend>(Codec::H264);
}

#[test]
fn recoding_changes_nothing_in_hevc() {
    check::<nc_hevc::HevcBackend>(Codec::Hevc);
}

#[test]
fn recoding_changes_nothing_in_vvc() {
    check::<nc_vvc::VvcBackend>(Codec::Vvc);
}

#[test]
fn recoding_changes_nothing_in_av1() {
    check::<nc_av1::Av1Backend>(Codec::Av1);
}

#[test]
fn recoding_changes_nothing_in_vp9() {
    check::<nc_vp9::Vp9Backend>(Codec::Vp9);
}

#[test]
fn recoding_changes_nothing_in_av2() {
    check::<nc_av2::Av2Backend>(Codec::Av2);
}
