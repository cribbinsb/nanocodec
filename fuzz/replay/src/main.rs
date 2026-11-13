//! Replay fuzz inputs through the fuzz targets' drivers
//! (`fuzz_targets/common/drive.rs`): `replay CODEC < archive`, where the
//! archive (`pack.py`) is a sequence of little-endian u32 lengths, each
//! followed by one input. Built for wasm32-wasip1 it runs the decoders as
//! the browser does, under node (`tools/ci/wasi-run.mjs`); a decoder panic
//! traps there, so each input's index is printed before it runs.
#![forbid(unsafe_code)]

#[path = "../../fuzz_targets/common/mod.rs"]
mod common;

use std::io::Read;
use std::process::ExitCode;

const LENGTH_BYTES: usize = 4;

fn main() -> ExitCode {
    let codec = std::env::args().nth(1).unwrap_or_default();
    let drive: fn(&[u8]) = match codec.as_str() {
        "h264" => common::drive::h264,
        "hevc" => common::drive::hevc,
        "vvc" => common::drive::vvc,
        "av1" => common::drive::av1,
        "vp9" => common::drive::vp9,
        "av2" => common::drive::av2,
        _ => {
            eprintln!("usage: replay h264|hevc|vvc|av1|vp9|av2 < archive");
            return ExitCode::FAILURE;
        }
    };
    let mut archive = Vec::new();
    if let Err(error) = std::io::stdin().read_to_end(&mut archive) {
        eprintln!("reading the archive: {error}");
        return ExitCode::FAILURE;
    }
    let mut rest = archive.as_slice();
    let mut count = 0;
    while let Some((length, tail)) = rest.split_first_chunk::<LENGTH_BYTES>() {
        let length = u32::from_le_bytes(*length) as usize;
        let Some((input, tail)) = tail.split_at_checked(length) else {
            eprintln!("truncated archive after input {count}");
            return ExitCode::FAILURE;
        };
        eprintln!("input {count}");
        drive(input);
        count += 1;
        rest = tail;
    }
    println!("{codec}: {count} inputs replayed without a panic");
    ExitCode::SUCCESS
}
