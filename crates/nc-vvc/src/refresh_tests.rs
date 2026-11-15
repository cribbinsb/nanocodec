//! Intra refresh recovery (`nc_encoder` `refresh.rs`): a decoder whose
//! references went wrong (as after a lost picture) decodes the stream
//! exactly again within two sweeps, with no key picture. Our decoder's
//! test hook overwrites its references; everything it parses stays right,
//! so only sample dependencies across the clean/dirty boundary can keep a
//! picture wrong: in-loop filtering past `loop_filter_reach`, and intra
//! prediction from dirty samples (H.266 has no gradual-refresh constraint
//! of its own outside GDR pictures, so the encoder keeps to these).
use crate::{Decoder, VvcBackend};
use nc_common::{Picture, PlaneIndex};
use nc_encoder::{
    Codec, CodecBackend, Encoder, EncoderConfig, FrameRate, FrameType, IntraRefresh,
    ParamOverrides, StreamSignal,
};

const WIDTH: usize = 256;
const HEIGHT: usize = 128;
const FRAMES: u64 = 20;
const CORRUPT_AFTER: u64 = 5;

/// A diagonal texture panning left by 12 samples per frame and down by 2:
/// motion vectors point right, towards the dirty region, and diagonal
/// intra directions (reading above-right samples) predict it well.
fn textured(frame: usize) -> Picture {
    picture(|plane, x, y| {
        let scale = if plane == PlaneIndex::Y { 1 } else { 2 };
        let gx = x + 12 * frame / scale;
        let gy = y + 2 * frame / scale;
        let diagonal = (gx + gy) * 9 / scale;
        (diagonal % 160) + (gx * gx * 3 + gy * 7) % 61 + plane as usize * 20
    })
}

/// Smooth ramps panning left by 6 samples per frame: large CUs on both
/// sides of the intra range's edges, so VVC's long deblocking filter
/// (7 samples, §8.8.3.6.6) and ALF reach across them.
fn smooth(frame: usize) -> Picture {
    picture(|plane, x, y| {
        let scale = if plane == PlaneIndex::Y { 1 } else { 2 };
        let gx = (x * scale + 6 * frame) % 512;
        let ramp = if gx < 256 { gx } else { 511 - gx };
        ramp * 3 / 4 + y * scale / 4 + plane as usize * 10
    })
}

fn picture(sample: impl Fn(PlaneIndex, usize, usize) -> usize) -> Picture {
    let mut picture = Picture::new(WIDTH, HEIGHT, 0).unwrap();
    for plane in PlaneIndex::ALL {
        let mut view = picture.plane_mut(plane).view_mut();
        for y in 0..view.height() {
            for (x, v) in view.row_mut(y).iter_mut().enumerate() {
                *v = (sample(plane, x, y) % 256) as u8;
            }
        }
    }
    picture
}

/// The Default's parameters (ALF, SAO search and the other in-loop tools
/// it enables over the engine defaults).
fn default_params() -> ParamOverrides {
    let mut params = ParamOverrides::default();
    for &(name, value) in <VvcBackend as CodecBackend>::DEFAULT_PARAMS {
        params.set(name, value);
    }
    params
}

fn exact(a: &Picture, b: &Picture) -> bool {
    PlaneIndex::ALL.iter().all(|&p| {
        let (a, b) = (a.plane(p).view(), b.plane(p).view());
        (0..a.height()).all(|y| a.row(y) == b.row(y))
    })
}

/// Code `source` with intra refresh over `period` frames at `quantizer`,
/// corrupting the decoder's references after frame [`CORRUPT_AFTER`].
fn check_recovery(name: &str, source: fn(usize) -> Picture, period: u32, quantizer: u32) {
    let config = EncoderConfig {
        codec: Codec::Vvc,
        width: WIDTH,
        height: HEIGHT,
        frame_rate: FrameRate { num: 30, den: 1 },
        quantizer,
        intra_period: 4,
        block_log2: 6,
        key_qp_offset: None,
        temporal_layers: 1,
        signal: StreamSignal::default(),
        params: default_params(),
    };
    let mut encoder = Encoder::<VvcBackend>::new(config).unwrap();
    encoder
        .set_intra_refresh(Some(IntraRefresh { period }))
        .unwrap();
    let mut decoder = Decoder::new();
    let mut mismatched = false;
    for index in 0..FRAMES {
        let frame = encoder.encode(&source(index as usize));
        assert_eq!(frame.metadata.frame_type == FrameType::Key, index == 0);
        let picture = decoder.decode(&frame.data).unwrap();
        if index == CORRUPT_AFTER {
            decoder.corrupt_references(0);
        }
        let exact = exact(&picture, encoder.reconstruction());
        // Recovery takes at most two sweeps: one to start a new one, one
        // to complete it.
        if index > CORRUPT_AFTER + 2 * u64::from(period) {
            assert!(exact, "{name}: frame {index} not recovered");
        } else if index > CORRUPT_AFTER {
            mismatched |= !exact;
        } else {
            assert!(exact, "{name}: frame {index}");
        }
    }
    assert!(mismatched, "{name}: the corruption had no effect");
}

/// Intra CUs on the dirty edge must not read above-right samples.
#[test]
fn intra_refresh_recovers_textured_content() {
    check_recovery("textured", textured, 3, 27);
}

/// Range edges on 64-sample boundaries between large CUs: the filters'
/// reach (`loop_filter_reach`) must cover deblocking, SAO and ALF.
#[test]
fn intra_refresh_recovers_across_long_filters() {
    for quantizer in [27, 42] {
        check_recovery("smooth", smooth, 4, quantizer);
    }
}
