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
