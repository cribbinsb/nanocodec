//! Annex-B NAL unit framing shared by the H.264 and HEVC backends.
//!
//! A NAL unit on the wire is a start code, the NAL unit header, and the
//! payload (the RBSP) with *emulation prevention* applied: whenever two zero
//! bytes would be followed by a byte `<= 0x03`, an `0x03` byte is inserted so
//! the payload can never imitate a start code (H.264 §7.4.1, HEVC §7.4.2).
//!
//! The header is copied verbatim: its first byte always has the forbidden
//! zero bit clear and a non-zero type or layer field, so it never starts an
//! emulated pattern itself, but it does count towards the zero run that the
//! payload continues (both specifications apply the rule to the whole NAL
//! unit after the start code).

/// The 4-byte start code used before every NAL unit.
///
/// Annex B allows a 3-byte start code for most NAL units, but the 4-byte
/// form (a leading `zero_byte`) is required before parameter sets and the
/// first NAL unit of an access unit; using it everywhere keeps the writer
/// simple and is always conformant.
pub const START_CODE: [u8; 4] = [0, 0, 0, 1];

/// The emulation prevention byte (`emulation_prevention_three_byte`).
const EMULATION_PREVENTION_BYTE: u8 = 0x03;

/// Append one Annex-B NAL unit (start code, `header`, escaped `rbsp`) to `out`.
///
/// `rbsp` must already end in `rbsp_trailing_bits()`, so its last byte is
/// non-zero and no trailing `cabac_zero_word` handling is needed.
pub fn write_nal_unit(header: &[u8], rbsp: &[u8], out: &mut Vec<u8>) {
    out.reserve(START_CODE.len() + header.len() + rbsp.len() + rbsp.len() / 64 + 1);
    out.extend_from_slice(&START_CODE);
    let mut zeros = 0usize;
    for &byte in header.iter().chain(rbsp) {
        if zeros >= 2 && byte <= EMULATION_PREVENTION_BYTE {
            out.push(EMULATION_PREVENTION_BYTE);
            zeros = 0;
        }
        out.push(byte);
        zeros = if byte == 0 { zeros + 1 } else { 0 };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nal(rbsp: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        write_nal_unit(&[0x40, 0x01], rbsp, &mut out);
        out
    }

    #[test]
    fn plain_payload_is_copied() {
        assert_eq!(nal(&[0x12, 0x80]), [0, 0, 0, 1, 0x40, 0x01, 0x12, 0x80]);
    }

    #[test]
    fn every_low_byte_after_two_zeros_is_escaped() {
        for b in 0..=3u8 {
            assert_eq!(nal(&[0, 0, b, 0x80])[6..], [0, 0, 3, b, 0x80]);
        }
        // 0x04 cannot complete a start code and is left alone.
        assert_eq!(nal(&[0, 0, 4])[6..], [0, 0, 4]);
    }

    #[test]
    fn long_zero_runs_are_broken_every_two_bytes() {
        assert_eq!(
            nal(&[0, 0, 0, 0, 0, 0x80])[6..],
            [0, 0, 3, 0, 0, 3, 0, 0x80]
        );
    }

    #[test]
    fn zero_run_is_reset_by_the_inserted_byte() {
        // After "00 00 03 00" only one zero is pending, so "00 01" is fine.
        assert_eq!(nal(&[0, 0, 0, 1])[6..], [0, 0, 3, 0, 1]);
    }

    #[test]
    fn zero_run_spans_the_header_boundary() {
        let mut out = Vec::new();
        write_nal_unit(&[0x00], &[0x00, 0x01], &mut out);
        assert_eq!(out, [0, 0, 0, 1, 0, 0, 3, 1]);
    }
}
