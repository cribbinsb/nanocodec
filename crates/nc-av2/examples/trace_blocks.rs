//! Decode an AV2 stream (IVF or raw OBU) and print the general block
//! layer's trace in dav2d's `DEBUG_BLOCK_INFO` format, then the first
//! error: what `tools/dev/av2_foreign/block_trace.py` holds against dav2d.
//!
//! `cargo run -p nanocodec-av2 --example trace_blocks -- FILE [OUT.yuv
//! [--filtered] [--invisible]]`
//!
//! With a second path, it also writes every output picture (I420): before
//! the loop filters, to hold against dav2d `--inloopfilters none`, or
//! after them with `--filtered`; `--invisible` adds the frames not shown
//! when decoded (dav2d `--outputinvisible 1`).
use std::io::Write;

/// IVF file and frame header sizes.
const IVF_FILE_HEADER: usize = 32;
const IVF_FRAME_HEADER: usize = 12;

/// The temporal units of `file`: an IVF file's packets, or a raw OBU
/// stream split as avmdec splits it.
fn units(file: &[u8]) -> Result<Vec<&[u8]>, Box<dyn std::error::Error>> {
    if !file.starts_with(b"DKIF") {
        return Ok(nc_av2::obu::temporal_units(file)?);
    }
    let mut units = Vec::new();
    let mut position = IVF_FILE_HEADER;
    while let Some(header) = file.get(position..position + IVF_FRAME_HEADER) {
        let size = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
        let start = position + IVF_FRAME_HEADER;
        let Some(packet) = file.get(start..start + size) else {
            break;
        };
        units.push(packet);
        position = start + size;
    }
    Ok(units)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("usage: trace_blocks FILE")?;
    let file = std::fs::read(path)?;
    let mut decoder = nc_av2::Decoder::default();
    let flags: Vec<String> = std::env::args().skip(3).collect();
    let mut unfiltered = match std::env::args().nth(2) {
        Some(path) => {
            if !flags.iter().any(|flag| flag == "--filtered") {
                decoder.skip_loop_filters();
            }
            if flags.iter().any(|flag| flag == "--invisible") {
                decoder.output_invisible_frames();
            }
            Some(std::fs::File::create(path)?)
        }
        None => None,
    };
    decoder.trace_blocks(Box::new(|line| {
        let _ = writeln!(std::io::stdout().lock(), "{line}");
    }));
    for unit in units(&file)? {
        if let Err(error) = decoder.decode_unit(unit) {
            println!("ERROR {error}");
            break;
        }
        while decoder.next_output() {
            if let (Some(out), Some(shown)) = (unfiltered.as_mut(), decoder.shown()) {
                for plane in nc_common::PlaneIndex::ALL {
                    let view = shown.plane(plane);
                    for y in 0..view.height() {
                        out.write_all(&view.row(y)[..view.width()])?;
                    }
                }
            }
        }
    }
    Ok(())
}
