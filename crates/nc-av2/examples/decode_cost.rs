//! Bounded AV2 decoder probe: IVF on stdin, no decoded video output.
//! Uses the same borrowed-output decoder entry points as the browser.
use nc_common::PlaneIndex;
use std::{hint::black_box, io::Read, time::Instant};

const IVF_HEADER_BYTES: usize = 32;
const FRAME_HEADER_BYTES: usize = 12;
const MAX_INPUT_BYTES: u64 = 64 * 1024 * 1024;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rounds: usize = std::env::args().nth(1).map_or(Ok(50), |s| s.parse())?;
    let mut input = Vec::new();
    std::io::stdin()
        .take(MAX_INPUT_BYTES)
        .read_to_end(&mut input)?;
    if input.len() < IVF_HEADER_BYTES || &input[..4] != b"DKIF" {
        return Err("expected an IVF stream".into());
    }
    let mut remaining = &input[IVF_HEADER_BYTES..];
    let mut packets = Vec::new();
    while !remaining.is_empty() {
        let header = remaining
            .get(..FRAME_HEADER_BYTES)
            .ok_or("truncated frame header")?;
        let size = u32::from_le_bytes(header[..4].try_into()?) as usize;
        remaining = &remaining[FRAME_HEADER_BYTES..];
        packets.push(remaining.get(..size).ok_or("truncated frame")?);
        remaining = &remaining[size..];
    }
    let start = Instant::now();
    let mut checksum = 0u64;
    for _ in 0..rounds {
        let mut decoder = nc_av2::Decoder::default();
        for packet in &packets {
            decoder.decode_shown(black_box(packet))?;
            let shown = decoder.shown().ok_or("missing picture")?;
            checksum += u64::from(black_box(shown.plane(PlaneIndex::Y).row(0)[0]));
        }
    }
    println!(
        "{} frames {:.3} ms checksum {checksum}",
        rounds * packets.len(),
        start.elapsed().as_secs_f64() * 1000.
    );
    Ok(())
}
