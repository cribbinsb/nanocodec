//! Native AV2 lossless picture probe; stream planar 8-bit 4:2:0 through stdin.
use std::io::{Read, Write};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let mut input = Vec::new();
    std::io::stdin()
        .take(64 * 1024 * 1024)
        .read_to_end(&mut input)?;
    let output = if args.get(1).is_some_and(|s| s == "--decode") {
        {
            let picture = nc_av2::intra::decode_keyframe(&input)?;
            let mut bytes = Vec::with_capacity(picture.packed_frame_len());
            for p in nc_common::PlaneIndex::ALL {
                let v = picture.plane(p).view();
                for row in 0..v.height() {
                    bytes.extend_from_slice(v.row(row));
                }
            }
            bytes
        }
    } else {
        if !(3..=5).contains(&args.len()) {
            return Err(
                "usage: intra_frame WIDTH HEIGHT [QINDEX] [--tx8] < input.yuv (or --decode < input.obu)"
                    .into(),
            );
        }
        let mut picture = nc_common::Picture::new(args[1].parse()?, args[2].parse()?, 0)?;
        if input.len() != picture.packed_frame_len() {
            return Err("input pixel length mismatch".into());
        }
        let mut offset = 0;
        for p in nc_common::PlaneIndex::ALL {
            let mut v = picture.plane_mut(p).view_mut();
            for row in 0..v.height() {
                let line = v.row_mut(row);
                let n = line.len();
                line.copy_from_slice(&input[offset..offset + n]);
                offset += n;
            }
        }
        nc_av2::intra::encode_keyframe_with_options(
            &picture,
            args.get(3).map_or(Ok(0), |s| s.parse())?,
            true,
            args.get(4).is_some_and(|s| s == "--tx8"),
        )?
        .packet
    };
    std::io::stdout().lock().write_all(&output)?;
    Ok(())
}
