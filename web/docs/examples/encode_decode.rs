//! Minimal use of the Rust API: encode a moving synthetic picture with
//! `nc_encoder::Encoder` driving the H.264 backend under 300 kbps rate
//! control, decode every access unit with this crate's decoder, and print
//! the mean luma PSNR. The project site's Rust API page shows this file.
//!
//!     cargo run --release -p nc-h264 --example encode_decode

use nc_common::{PlaneIndex, RawFrame};
use nc_encoder::{
    Codec, Encoder, EncoderConfig, FrameRate, ParamOverrides, RateControlConfig, StreamSignal,
};
use nc_h264::{Decoder, H264Backend};

const WIDTH: usize = 320;
const HEIGHT: usize = 180;
const FRAMES: usize = 30;
const FPS: u32 = 30;
const TARGET_BPS: u64 = 300_000;
/// 16x16 leaf blocks: the CLI's default for H.264 (`--block 16`).
const BLOCK_LOG2: u8 = 4;

/// A packed I420 frame: a diagonal gradient that moves two pixels a frame.
fn draw(index: usize) -> Vec<u8> {
    let mut frame = vec![128u8; WIDTH * HEIGHT * 3 / 2];
    for (i, sample) in frame[..WIDTH * HEIGHT].iter_mut().enumerate() {
        let (row, col) = (i / WIDTH, i % WIDTH);
        *sample = (16 + (col + row + 2 * index) % 220) as u8;
    }
    frame
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = EncoderConfig {
        codec: Codec::H264,
        width: WIDTH,
        height: HEIGHT,
        frame_rate: FrameRate { num: FPS, den: 1 },
        quantizer: 32, // the starting QP; rate control chooses from there
        intra_period: 10 * FPS,
        block_log2: BLOCK_LOG2,
        key_qp_offset: None,
        temporal_layers: 1,
        signal: StreamSignal::default(),
        params: ParamOverrides::default(), // the codec's Default
    };
    let mut encoder = Encoder::<H264Backend>::new(config)?;
    encoder.set_rate_control(RateControlConfig::new(TARGET_BPS))?;
    let mut decoder = Decoder::new();

    let (mut bytes, mut decoded, mut psnr_sum) = (0, 0, 0.0);
    for index in 0..FRAMES {
        let source = draw(index);
        let frame = encoder.encode_raw(&RawFrame::packed_i420(&source, WIDTH, HEIGHT)?)?;
        if frame.data.is_empty() {
            continue; // dropped or held: nothing to send
        }
        bytes += frame.data.len();
        let picture = decoder.decode(&frame.data)?;
        let luma = picture.plane(PlaneIndex::Y).view();
        let sse: f64 = (0..HEIGHT)
            .flat_map(|row| {
                let decoded_row = &luma.row(row)[..WIDTH];
                let source_row = &source[row * WIDTH..(row + 1) * WIDTH];
                decoded_row
                    .iter()
                    .zip(source_row)
                    .map(|(&a, &b)| f64::from(a.abs_diff(b)).powi(2))
            })
            .sum();
        psnr_sum += 10.0 * (255.0f64.powi(2) * (WIDTH * HEIGHT) as f64 / sse.max(1.0)).log10();
        decoded += 1;
    }
    println!(
        "{decoded} of {FRAMES} frames decoded, {bytes} bytes, mean luma PSNR {:.1} dB",
        psnr_sum / f64::from(decoded.max(1))
    );
    Ok(())
}
