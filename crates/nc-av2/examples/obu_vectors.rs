//! Write framing-only vectors and print decoded headers for the AVM oracle check.
//! Payload bytes are opaque test data, not complete AV2 pictures.
use nc_av2::obu::{Header, parse, write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: obu_vectors OUTPUT.obu")?;
    let mut stream = Vec::new();
    for (obu_type, temporal_id, embedded_id, extended_id) in [
        (2, 0, 0, 31),
        (1, 0, 0, 0),
        (7, 3, 2, 5),
        (25, 0, 0, 31),
        (31, 2, 1, 4),
    ] {
        for size in [0, 1, 126, 127, 128, 16383, 16384] {
            let header = Header {
                obu_type,
                temporal_id,
                embedded_id,
                extended_id,
            };
            write(header, &vec![0x55; size], &mut stream)?;
        }
    }
    std::fs::write(path, &stream)?;
    let mut remaining = stream.as_slice();
    while !remaining.is_empty() {
        let (obu, rest) = parse(remaining)?;
        let h = obu.header;
        println!(
            "{} {} {} {} {}",
            h.obu_type,
            h.temporal_id,
            h.embedded_id,
            h.extended_id,
            obu.payload.len()
        );
        remaining = rest;
    }
    Ok(())
}
