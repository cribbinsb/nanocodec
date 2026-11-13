//! Shared decoder-fuzz harness: one input drives one of the public decode
//! paths a visitor's bytes can reach.
//!
//! The first input byte selects the path ([`Mode`]) and its options; the
//! rest is either a sequence of packets (browser and C API paths, which
//! receive one coded frame at a time) or a whole container file (the
//! command-line decoder: Annex-B for H.264/HEVC/VVC, IVF for AV1/VP9/AV2,
//! read by the CLI's own framing code). Every shown picture is read row by
//! row exactly as the browser and CLI output stages read it, so a decoder
//! that reports a window outside its picture fails here too.

#![allow(
    dead_code,
    reason = "each target uses the subset of helpers its codec's API needs"
)]

use std::hint::black_box;

use nc_common::{Picture, PlaneIndex, ShownPicture};

/// The command-line decoder's container readers, compiled from the CLI's own
/// source so the fuzzer exercises the exact framing code it ships.
#[allow(
    dead_code,
    unused_imports,
    reason = "the encoder-side writers in this module are not used here"
)]
#[path = "../../../crates/nc-cli/src/framing.rs"]
pub mod framing;

/// Bytes that separate packets in the packet paths (absent from the seed
/// streams' payload framing by construction of the seed builder, and given
/// to libFuzzer as a dictionary entry).
pub const PACKET_SEPARATOR: &[u8] = b"NCPK";

/// Bit layout of the first input byte.
const MODE_MASK: u8 = 0b11;
const OPERATING_POINT_SHIFT: u32 = 2;
const OPERATING_POINT_MASK: u8 = 0b1_1111;
const SKIP_FILM_GRAIN_BIT: u8 = 0x80;

/// Which public entry point the input drives.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// `BrowserDecoder` (nc-web): `decode_shown` per packet, then `shown()`.
    Browser,
    /// C API receiver (nc-capi): `decode` (AV1 `decode_optional`) per packet.
    CApi,
    /// Command-line decoder: the input is a container file.
    Cli,
    /// Command-line decoder with `--all-layers 1` (AV1, VP9).
    CliAllLayers,
}

/// The decoding path and options of one input.
#[derive(Clone, Copy)]
pub struct Config {
    pub mode: Mode,
    /// AV1 operating point (`0..=31`); on VP9 `> 0` selects base-only.
    pub operating_point: u8,
    /// AV1: show frames without film grain (`--film-grain skip`).
    pub skip_film_grain: bool,
}

/// Split the configuration byte from the payload; `None` for empty input.
pub fn config(data: &[u8]) -> Option<(Config, &[u8])> {
    let (&first, rest) = data.split_first()?;
    let mode = match first & MODE_MASK {
        0 => Mode::Browser,
        1 => Mode::CApi,
        2 => Mode::Cli,
        _ => Mode::CliAllLayers,
    };
    let config = Config {
        mode,
        operating_point: (first >> OPERATING_POINT_SHIFT) & OPERATING_POINT_MASK,
        skip_film_grain: first & SKIP_FILM_GRAIN_BIT != 0,
    };
    Some((config, rest))
}

/// The packets of `data`, separated by [`PACKET_SEPARATOR`].
pub fn packets(data: &[u8]) -> impl Iterator<Item = &[u8]> {
    let mut rest = Some(data);
    std::iter::from_fn(move || {
        let current = rest?;
        match find(current, PACKET_SEPARATOR) {
            Some(at) => {
                rest = Some(&current[at + PACKET_SEPARATOR.len()..]);
                Some(&current[..at])
            }
            None => {
                rest = None;
                Some(current)
            }
        }
    })
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Read every visible sample of a shown picture (browser display path).
pub fn read_shown(shown: Option<ShownPicture<'_>>) {
    let Some(shown) = shown else { return };
    let mut sum = 0u64;
    for plane in PlaneIndex::ALL {
        for row in shown.plane(plane).rows() {
            sum = sum.wrapping_add(row.iter().map(|&s| u64::from(s)).sum::<u64>());
        }
    }
    black_box(sum);
}

/// Read the `size` luma window at `origin` of `picture` (and its chroma),
/// exactly as the CLI writes a decoder's shared or borrowed picture.
pub fn read_window(picture: &Picture, (x0, y0): (usize, usize), (width, height): (usize, usize)) {
    let mut sum = 0u64;
    for plane in PlaneIndex::ALL {
        let ss = usize::from(plane != PlaneIndex::Y);
        let (w, h) = ((width + ss) >> ss, (height + ss) >> ss);
        let view = picture.plane(plane).view();
        for y in 0..h {
            let row = view.row(y + (y0 >> ss));
            let row = &row[x0 >> ss..(x0 >> ss) + w];
            sum = sum.wrapping_add(row.iter().map(|&s| u64::from(s)).sum::<u64>());
        }
    }
    black_box(sum);
}

/// Read a whole picture.
pub fn read_picture(picture: &Picture) {
    read_window(picture, (0, 0), (picture.width(), picture.height()));
}
