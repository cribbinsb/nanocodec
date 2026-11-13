//! Final AV2 v1.0.0 OBU framing (§5 `open_bitstream_unit` and `obu_header`).
//!
//! Unlike AV1, AV2 prefixes the header with a LEB128 size INCLUDING the
//! header; the header contains a five-bit type and two-bit temporal layer.
//! Checked against AVM v1.0.0 `av2/common/obu_util.c`. Payload syntax and
//! temporal-unit ordering are deliberately left to the future picture decoder.

use nc_common::decode::{BitReader, DecodeError, Result};

const TEMPORAL_DELIMITER: u8 = 2;
const MULTI_STREAM_DECODER_OPERATION: u8 = 20;
/// Extended-layer ID used for OBUs that apply globally.
pub const GLOBAL_XLAYER_ID: u8 = 31;

/// A parsed final-AV2 OBU header. Reserved type values are preserved for skipping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    /// Five-bit OBU type, including reserved values.
    pub obu_type: u8,
    /// Two-bit temporal layer.
    pub temporal_id: u8,
    /// Three-bit embedded layer (zero without an extension).
    pub embedded_id: u8,
    /// Five-bit extended layer; inferred as global for TD/MSDO without extension.
    pub extended_id: u8,
}

impl Header {
    fn global_type(self) -> bool {
        matches!(
            self.obu_type,
            TEMPORAL_DELIMITER | MULTI_STREAM_DECODER_OPERATION
        )
    }

    fn validate(self) -> Result<()> {
        if self.obu_type > 31
            || self.temporal_id > 3
            || self.embedded_id > 7
            || self.extended_id > GLOBAL_XLAYER_ID
        {
            return Err(DecodeError::Malformed(
                "AV2 OBU field exceeds its bit width",
            ));
        }
        if self.global_type() && self.extended_id != GLOBAL_XLAYER_ID {
            return Err(DecodeError::Malformed(
                "AV2 global OBU requires global extended layer",
            ));
        }
        // Header-only constraints from §6.2.2; sequence-dependent maxima
        // and OBU ordering are checked by the future picture decoder.
        if self.extended_id == GLOBAL_XLAYER_ID {
            if self.temporal_id != 0 || self.embedded_id != 0 {
                return Err(DecodeError::Malformed(
                    "AV2 global OBU must use base layers",
                ));
            }
            if !matches!(
                self.obu_type,
                2 | 8 | 9 | 15 | 16 | 17 | 18 | 20 | 25 | 0 | 26..=31
            ) {
                return Err(DecodeError::Malformed("AV2 OBU type cannot be global"));
            }
        }
        if matches!(self.obu_type, 1 | 2 | 16 | 17 | 18)
            && (self.temporal_id != 0 || self.embedded_id != 0)
        {
            return Err(DecodeError::Malformed(
                "AV2 structural OBU must use base layers",
            ));
        }
        if matches!(self.obu_type, 4 | 5 | 10 | 21) && self.temporal_id != 0 {
            return Err(DecodeError::Malformed(
                "AV2 random-access OBU must use temporal base layer",
            ));
        }
        Ok(())
    }
}

/// One complete OBU, borrowing its payload from the input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Obu<'a> {
    /// Parsed header.
    pub header: Header,
    /// Payload bytes, excluding length and header.
    pub payload: &'a [u8],
}

/// Parse one length-prefixed OBU and return the unconsumed suffix.
/// # Errors
/// Rejects truncated headers/payloads, sizes beyond 32 bits and invalid headers.
/// Accepts legal non-minimal LEB128 encodings up to eight bytes (§4.11.6).
pub fn parse(data: &[u8]) -> Result<(Obu<'_>, &[u8])> {
    let mut length = BitReader::new(data);
    let size = length.read_leb128()?;
    let size =
        u32::try_from(size).map_err(|_| DecodeError::Malformed("AV2 OBU size exceeds 32 bits"))?;
    let prefix = length.position() / 8;
    let end = prefix
        .checked_add(size as usize)
        .ok_or(DecodeError::Truncated)?;
    let unit = data.get(prefix..end).ok_or(DecodeError::Truncated)?;
    let first = *unit
        .first()
        .ok_or(DecodeError::Malformed("AV2 OBU has no header"))?;
    let has_extension = first & 0x80 != 0;
    let mut header = Header {
        obu_type: (first >> 2) & 31,
        temporal_id: first & 3,
        embedded_id: 0,
        extended_id: 0,
    };
    let header_size = if has_extension {
        let extension = *unit.get(1).ok_or(DecodeError::Truncated)?;
        header.embedded_id = extension >> 5;
        header.extended_id = extension & 31;
        2
    } else {
        if header.global_type() {
            header.extended_id = GLOBAL_XLAYER_ID;
        }
        1
    };
    header.validate()?;
    Ok((
        Obu {
            header,
            payload: &unit[header_size..],
        },
        &data[end..],
    ))
}

/// Append a minimally length-prefixed OBU. Invalid arguments leave output intact.
/// # Errors
/// Rejects out-of-range header fields and payloads too large for a 32-bit size.
pub fn write(header: Header, payload: &[u8], output: &mut Vec<u8>) -> Result<()> {
    header.validate()?;
    let inferred_extended = if header.global_type() {
        GLOBAL_XLAYER_ID
    } else {
        0
    };
    let extended = header.embedded_id != 0 || header.extended_id != inferred_extended;
    let header_size = if extended { 2 } else { 1 };
    let size = payload
        .len()
        .checked_add(header_size)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or(DecodeError::Malformed("AV2 OBU size exceeds 32 bits"))?;
    let mut size_left = size;
    loop {
        let byte = (size_left & 127) as u8;
        size_left >>= 7;
        output.push(byte | if size_left == 0 { 0 } else { 128 });
        if size_left == 0 {
            break;
        }
    }
    output.push((u8::from(extended) << 7) | (header.obu_type << 2) | header.temporal_id);
    if extended {
        output.push((header.embedded_id << 5) | header.extended_id);
    }
    output.extend_from_slice(payload);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_spec_header_layout_and_concatenation() {
        // Size includes the header; TD's extended layer is inferred as global.
        let bytes = [1, 8, 4, 0x9f, 0x45, 0xaa, 0xbb];
        let (td, rest) = parse(&bytes).unwrap();
        assert_eq!(td.header.extended_id, GLOBAL_XLAYER_ID);
        assert_eq!(td.payload, []);
        let (sequence, rest) = parse(rest).unwrap();
        assert_eq!(rest, []);
        assert_eq!(
            sequence.header,
            Header {
                obu_type: 7,
                temporal_id: 3,
                embedded_id: 2,
                extended_id: 5
            }
        );
        assert_eq!(sequence.payload, [0xaa, 0xbb]);
        let mut written = Vec::new();
        write(td.header, td.payload, &mut written).unwrap();
        write(sequence.header, sequence.payload, &mut written).unwrap();
        assert_eq!(written, bytes);
    }

    #[test]
    fn bounds_and_nonminimal_lengths() {
        assert!(parse(&[0]).is_err());
        assert!(parse(&[1, 0x84, 0]).is_err()); // extension lies outside declared size
        assert!(parse(&[2, 4]).is_err()); // truncated payload
        assert!(parse(&[0x80; 8]).is_err());
        assert!(parse(&[0x81, 0x80, 0x80, 0x80, 0x10, 8]).is_err());
        let (obu, rest) = parse(&[0x81, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0, 8]).unwrap();
        assert_eq!(obu.header.obu_type, TEMPORAL_DELIMITER);
        assert_eq!(rest, []);
    }

    #[test]
    fn rejects_invalid_fields_without_modifying_output() {
        let header = Header {
            obu_type: 32,
            temporal_id: 0,
            embedded_id: 0,
            extended_id: 0,
        };
        let mut output = vec![9];
        assert!(write(header, &[], &mut output).is_err());
        assert_eq!(output, [9]);
    }

    #[test]
    fn payload_length_boundaries() {
        for len in [0, 1, 126, 127, 128, 16383, 16384] {
            let header = Header {
                obu_type: 25,
                temporal_id: 0,
                embedded_id: 0,
                extended_id: 0,
            };
            let payload = vec![42; len];
            let mut bytes = Vec::new();
            write(header, &payload, &mut bytes).unwrap();
            let (obu, rest) = parse(&bytes).unwrap();
            assert_eq!(obu.payload, payload);
            assert_eq!(rest, []);
        }
    }
}
