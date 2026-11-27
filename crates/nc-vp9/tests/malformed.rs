//! Corrupt streams found by fuzzing the VP9 decoder: each must decode to
//! frames or errors, never panic. Fixtures and provenance: `tests/data`.

use nc_vp9::stream::Decoder;

/// IVF file header and per-frame header sizes.
const IVF_HEADER: usize = 32;
const IVF_FRAME_HEADER: usize = 12;

/// Feed every IVF frame of `data` to one decoder, ignoring results.
fn decode_all(data: &[u8]) {
    let mut decoder = Decoder::new();
    let mut at = IVF_HEADER;
    while let Some(frame_header) = data.get(at..at + IVF_FRAME_HEADER) {
        let size = u32::from_le_bytes(frame_header[..4].try_into().expect("four bytes")) as usize;
        at += IVF_FRAME_HEADER;
        let Some(packet) = data.get(at..at + size) else {
            break;
        };
        let _ = decoder.decode(packet);
        at += size;
    }
}

/// Coefficients far outside the valid 16-bit range overflowed the 4x4
/// inverse DCT (`hybrid::inverse_1d`).
#[test]
fn corrupt_coefficients_do_not_overflow_the_4x4_inverse_dct() {
    decode_all(include_bytes!("data/fuzz-4x4-idct-overflow.ivf"));
}

/// The packets of a fuzz input (`fuzz/README.md`): after the config byte,
/// `NCPK`-separated packets (browser layout, `& 3` 0) or an IVF file
/// (command-line layout, 2).
fn fuzz_packets(input: &[u8]) -> Vec<&[u8]> {
    const SEPARATOR: &[u8] = b"NCPK";
    const BROWSER: u8 = 0;
    let (config, mut rest) = input.split_first().expect("a config byte");
    let mut packets = Vec::new();
    if config & 3 == BROWSER {
        while let Some(at) = rest.windows(SEPARATOR.len()).position(|w| w == SEPARATOR) {
            packets.push(&rest[..at]);
            rest = &rest[at + SEPARATOR.len()..];
        }
        packets.push(rest);
    } else {
        let mut at = IVF_HEADER;
        while let Some(frame_header) = rest.get(at..at + IVF_FRAME_HEADER) {
            let size =
                u32::from_le_bytes(frame_header[..4].try_into().expect("four bytes")) as usize;
            at += IVF_FRAME_HEADER;
            let Some(packet) = rest.get(at..at + size) else {
                break;
            };
            packets.push(packet);
            at += size;
        }
    }
    packets
}

/// Fuzz round 3 (2026-11-23): key frames declaring pictures such as
/// 2896x65536 took 6-26 s each under the sanitizer. Pictures beyond the
/// highest level (6.2) are now rejected at the header, so the inputs finish
/// at once. `decode_kept` takes every depth (one input is profile 2).
#[test]
fn pictures_beyond_level_6_2_are_rejected_quickly() {
    /// Generous for a debug build: the decodes take milliseconds.
    const TIME_LIMIT: std::time::Duration = std::time::Duration::from_secs(10);
    let inputs: [&[u8]; 5] = [
        include_bytes!("data/fuzz-large-picture-18213922.fuzz"),
        include_bytes!("data/fuzz-large-picture-220c4518.fuzz"),
        include_bytes!("data/fuzz-large-picture-8899d99d.fuzz"),
        include_bytes!("data/fuzz-large-picture-95d3aa02.fuzz"),
        include_bytes!("data/fuzz-large-picture-df521afc.fuzz"),
    ];
    let start = std::time::Instant::now();
    for input in inputs {
        let mut decoder = Decoder::new();
        let errors: Vec<_> = fuzz_packets(input)
            .into_iter()
            .filter_map(|packet| decoder.decode_kept(packet).err())
            .collect();
        assert!(
            errors.contains(&nc_common::decode::DecodeError::Unsupported(
                "VP9 frame size beyond level 6.2"
            )),
            "{errors:?}"
        );
    }
    assert!(start.elapsed() < TIME_LIMIT, "took {:?}", start.elapsed());
}
