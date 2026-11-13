//! Arbitrary input to the public VP9 Profile 0 decoder: the bytes are cut
//! into up to four packets (a little-endian u16 length before each) so
//! that inter frames, references and probability contexts carry over.
#![no_main]
#![forbid(unsafe_code)]

use libfuzzer_sys::fuzz_target;

/// Packets per input.
const MAX_PACKETS: usize = 4;

fuzz_target!(|data: &[u8]| {
    let mut decoder = nc_vp9::stream::Decoder::new();
    let mut rest = data;
    for _ in 0..MAX_PACKETS {
        let Some((size, tail)) = rest.split_first_chunk::<2>() else {
            break;
        };
        let size = usize::from(u16::from_le_bytes(*size)).min(tail.len());
        let (packet, tail) = tail.split_at(size);
        let _ = decoder.decode_packet(packet);
        rest = tail;
    }
});
