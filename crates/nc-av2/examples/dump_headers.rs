//! Print the headers of an AV2 stream (IVF or raw OBU) as the general
//! decoder parses them, one OBU a line: a diagnostic to hold against
//! dav2d's `DEBUG_SEQ_HDR` / `DEBUG_FRAME_HDR` output.
//!
//! `cargo run -p nc-av2 --example dump_headers -- stream.obu`
use nc_av2::general::sequence;
use nc_av2::obu;

/// IVF file and frame header sizes.
const IVF_FILE_HEADER: usize = 32;
const IVF_FRAME_HEADER: usize = 12;
/// OBU type of the sequence header.
const SEQUENCE_HEADER: u8 = 1;

/// The OBU bytes of `file`: an IVF file's packets joined, or the file.
fn obu_bytes(file: &[u8]) -> Vec<u8> {
    if !file.starts_with(b"DKIF") {
        return file.to_vec();
    }
    let mut bytes = Vec::new();
    let mut position = IVF_FILE_HEADER;
    while let Some(header) = file.get(position..position + IVF_FRAME_HEADER) {
        let size = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
        let start = position + IVF_FRAME_HEADER;
        let Some(packet) = file.get(start..start + size) else {
            break;
        };
        bytes.extend_from_slice(packet);
        position = start + size;
    }
    bytes
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("usage: dump_headers FILE")?;
    let bytes = obu_bytes(&std::fs::read(path)?);
    let mut remaining = bytes.as_slice();
    while !remaining.is_empty() {
        let (unit, rest) = obu::parse(remaining)?;
        remaining = rest;
        let header = unit.header;
        print!(
            "OBU type={} size={} temporal={}",
            header.obu_type,
            unit.payload.len(),
            header.temporal_id
        );
        if header.obu_type == SEQUENCE_HEADER {
            match sequence::parse(unit.payload) {
                Ok(parsed) => print!(" {parsed:?}"),
                Err(error) => print!(" ERROR {error}"),
            }
        }
        println!();
    }
    Ok(())
}
