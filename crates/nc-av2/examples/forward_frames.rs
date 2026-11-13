//! Forward AV2 conformance probe. Raw pixels are streamed over stdin/stdout.
use nc_common::{Picture, PlaneIndex};
use std::io::{Read, Write};
fn write_picture(picture: &Picture, out: &mut impl Write) -> std::io::Result<()> {
    for p in PlaneIndex::ALL {
        let view = picture.plane(p).view();
        for row in 0..view.height() {
            out.write_all(view.row(row))?;
        }
    }
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let mut input = Vec::new();
    std::io::stdin()
        .take(64 * 1024 * 1024)
        .read_to_end(&mut input)?;
    let mut out = std::io::stdout().lock();
    if args.get(1).is_some_and(|s| s == "--decode") {
        let mut decoder = nc_av2::stream::Decoder::default();
        let mut pending = Vec::new();
        let mut remaining = input.as_slice();
        while !remaining.is_empty() {
            let (unit, rest) = nc_av2::obu::parse(remaining)?;
            let consumed = remaining.len() - rest.len();
            pending.extend_from_slice(&remaining[..consumed]);
            remaining = rest;
            if matches!(unit.header.obu_type, 4 | 7) {
                write_picture(decoder.decode_ref(&pending)?, &mut out)?;
                pending.clear();
            }
        }
        if !pending.is_empty() {
            return Err("trailing data after last picture".into());
        }
    } else {
        if !(4..=7).contains(&args.len()) {
            return Err(
                "usage: forward_frames WIDTH HEIGHT QINDEX [--motion] [--adaptive] [--tx8] (or --decode)"
                    .into(),
            );
        }
        let mut source = Picture::new(args[1].parse()?, args[2].parse()?, 0)?;
        let qindex = args[3].parse()?;
        let frame_len = source.packed_frame_len();
        if !input.len().is_multiple_of(frame_len) {
            return Err("truncated input frame".into());
        }
        let mut encoder = if args[4..].iter().any(|s| s == "--motion") {
            nc_av2::stream::Encoder::with_motion()
        } else {
            nc_av2::stream::Encoder::default()
        };
        encoder.set_cdf_adaptation(args[4..].iter().any(|s| s == "--adaptive"));
        encoder.set_tx8(args[4..].iter().any(|s| s == "--tx8"));
        for frame in input.chunks_exact(frame_len) {
            let mut offset = 0;
            for p in PlaneIndex::ALL {
                let mut view = source.plane_mut(p).view_mut();
                for row in 0..view.height() {
                    let line = view.row_mut(row);
                    let n = line.len();
                    line.copy_from_slice(&frame[offset..offset + n]);
                    offset += n;
                }
            }
            let encoded = encoder.encode(&source, qindex, false)?;
            out.write_all(&encoded.packet)?;
        }
    }
    Ok(())
}
