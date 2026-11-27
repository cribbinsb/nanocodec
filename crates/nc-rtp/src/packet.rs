//! The RTP fixed header (RFC 3550 §5.1) and RTP header extensions
//! (RFC 8285), written and parsed.
//!
//! Every payload format in this crate sits on top of this module: a
//! packetizer hands it a header, its extension elements and the payload
//! bytes; a depacketizer gets the same three back from [`RtpPacket::parse`].
//! CSRC lists and padding are parsed and skipped (a mixer's business, not a
//! video receiver's). Extensions are written in the one-byte form when every
//! element fits it (id 1-14, 1-16 bytes), otherwise in the two-byte form.

use crate::error::{Result, RtpError};

/// The RTP version this crate speaks (RFC 3550 §5.1 `V`).
const RTP_VERSION: u8 = 2;
/// Size of the fixed header without CSRCs.
pub const FIXED_HEADER_BYTES: usize = 12;
/// Size of one CSRC identifier.
const CSRC_BYTES: usize = 4;
/// Size of the extension block header: profile and length in words.
const EXTENSION_HEADER_BYTES: usize = 4;
/// RFC 8285 §4.2: the one-byte header form's "defined by profile" value.
const ONE_BYTE_PROFILE: u16 = 0xBEDE;
/// RFC 8285 §4.3: the two-byte form's profile, with 4 application bits.
const TWO_BYTE_PROFILE: u16 = 0x1000;
const TWO_BYTE_PROFILE_MASK: u16 = 0xFFF0;
/// One-byte form: ids 1-14 carry data, 15 stops parsing (RFC 8285 §4.2).
const ONE_BYTE_MAX_ID: u8 = 14;
const ONE_BYTE_STOP_ID: u8 = 15;
/// One-byte form: 1-16 data bytes (`L` is the length minus one).
const ONE_BYTE_MAX_LEN: usize = 16;
/// Two-byte form: 0-255 data bytes.
const TWO_BYTE_MAX_LEN: usize = 255;
/// Extension blocks are padded to whole 32-bit words.
const WORD_BYTES: usize = 4;
/// The 7-bit payload type field (RFC 3550 §5.1 `PT`).
pub const MAX_PAYLOAD_TYPE: u8 = 127;

/// The fields of the RTP fixed header a video sender sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RtpHeader {
    /// `M`: the last packet of a frame (each payload format defines which).
    pub marker: bool,
    /// `PT`: the dynamic payload type the session negotiated.
    pub payload_type: u8,
    /// Sequence number, +1 per packet.
    pub sequence: u16,
    /// Media timestamp (90 kHz for video).
    pub timestamp: u32,
    /// Synchronisation source.
    pub ssrc: u32,
}

/// One RFC 8285 header extension element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Extension<'a> {
    /// Local identifier the session mapped to the extension URI.
    pub id: u8,
    /// The element's data.
    pub data: &'a [u8],
}

/// A parsed RTP packet, borrowing the datagram.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RtpPacket<'a> {
    /// The fixed header.
    pub header: RtpHeader,
    /// Header extension elements in order (empty without `X`).
    pub extensions: Vec<Extension<'a>>,
    /// The payload, padding removed.
    pub payload: &'a [u8],
}

impl<'a> RtpPacket<'a> {
    /// Parse one datagram.
    ///
    /// # Errors
    /// A wrong version, a header or extension block longer than the
    /// datagram, or padding longer than the payload.
    pub fn parse(data: &'a [u8]) -> Result<Self> {
        if data.len() < FIXED_HEADER_BYTES {
            return Err(RtpError::Truncated("RTP fixed header"));
        }
        if data[0] >> 6 != RTP_VERSION {
            return Err(RtpError::Malformed("RTP version is not 2"));
        }
        let padding = data[0] & 0x20 != 0;
        let has_extension = data[0] & 0x10 != 0;
        let csrc_count = usize::from(data[0] & 0x0F);
        let header = RtpHeader {
            marker: data[1] & 0x80 != 0,
            payload_type: data[1] & MAX_PAYLOAD_TYPE,
            sequence: u16::from_be_bytes([data[2], data[3]]),
            timestamp: u32::from_be_bytes([data[4], data[5], data[6], data[7]]),
            ssrc: u32::from_be_bytes([data[8], data[9], data[10], data[11]]),
        };
        let mut offset = FIXED_HEADER_BYTES + csrc_count * CSRC_BYTES;
        if data.len() < offset {
            return Err(RtpError::Truncated("CSRC list"));
        }
        let mut extensions = Vec::new();
        if has_extension {
            let block = data
                .get(offset..offset + EXTENSION_HEADER_BYTES)
                .ok_or(RtpError::Truncated("extension header"))?;
            let profile = u16::from_be_bytes([block[0], block[1]]);
            let words = usize::from(u16::from_be_bytes([block[2], block[3]]));
            let start = offset + EXTENSION_HEADER_BYTES;
            let end = start + words * WORD_BYTES;
            let elements = data
                .get(start..end)
                .ok_or(RtpError::Truncated("extension block"))?;
            if profile == ONE_BYTE_PROFILE {
                parse_one_byte(elements, &mut extensions)?;
            } else if profile & TWO_BYTE_PROFILE_MASK == TWO_BYTE_PROFILE {
                parse_two_byte(elements, &mut extensions)?;
            }
            offset = end;
        }
        let mut end = data.len();
        if padding {
            let count = usize::from(*data.last().expect("length checked above"));
            if count == 0 || count > end - offset {
                return Err(RtpError::Malformed("padding longer than the payload"));
            }
            end -= count;
        }
        Ok(Self {
            header,
            extensions,
            payload: &data[offset..end],
        })
    }

    /// The data of the extension element with `id`, if present.
    #[must_use]
    pub fn extension(&self, id: u8) -> Option<&'a [u8]> {
        self.extensions
            .iter()
            .find(|element| element.id == id)
            .map(|element| element.data)
    }
}

/// RFC 8285 §4.2: `ID(4) L(4)` then `L + 1` bytes; id 0 is padding.
fn parse_one_byte<'a>(mut data: &'a [u8], out: &mut Vec<Extension<'a>>) -> Result<()> {
    while let Some(&first) = data.first() {
        let id = first >> 4;
        if id == 0 {
            data = &data[1..];
            continue;
        }
        if id == ONE_BYTE_STOP_ID {
            break;
        }
        let len = usize::from(first & 0x0F) + 1;
        let element = data
            .get(1..=len)
            .ok_or(RtpError::Truncated("one-byte extension element"))?;
        out.push(Extension { id, data: element });
        data = &data[1 + len..];
    }
    Ok(())
}

/// RFC 8285 §4.3: `ID(8) L(8)` then `L` bytes; id 0 is padding.
fn parse_two_byte<'a>(mut data: &'a [u8], out: &mut Vec<Extension<'a>>) -> Result<()> {
    while let Some(&id) = data.first() {
        if id == 0 {
            data = &data[1..];
            continue;
        }
        let len = usize::from(
            *data
                .get(1)
                .ok_or(RtpError::Truncated("two-byte extension header"))?,
        );
        let element = data
            .get(2..2 + len)
            .ok_or(RtpError::Truncated("two-byte extension element"))?;
        out.push(Extension { id, data: element });
        data = &data[2 + len..];
    }
    Ok(())
}

/// True when every element fits the one-byte form.
fn one_byte_form(extensions: &[Extension<'_>]) -> bool {
    extensions.iter().all(|e| {
        (1..=ONE_BYTE_MAX_ID).contains(&e.id) && (1..=ONE_BYTE_MAX_LEN).contains(&e.data.len())
    })
}

/// Bytes the extension block of `extensions` adds to a packet (0 for none).
#[must_use]
pub fn extension_overhead(extensions: &[Extension<'_>]) -> usize {
    if extensions.is_empty() {
        return 0;
    }
    let element_header = if one_byte_form(extensions) { 1 } else { 2 };
    let elements: usize = extensions
        .iter()
        .map(|e| element_header + e.data.len())
        .sum();
    EXTENSION_HEADER_BYTES + elements.div_ceil(WORD_BYTES) * WORD_BYTES
}

/// Write one packet: the fixed header, the extension block and `payload`.
///
/// # Errors
/// An extension element with id 0 or longer than 255 bytes.
pub fn write_packet(
    header: &RtpHeader,
    extensions: &[Extension<'_>],
    payload: &[u8],
) -> Result<Vec<u8>> {
    let overhead = extension_overhead(extensions);
    let mut out = Vec::with_capacity(FIXED_HEADER_BYTES + overhead + payload.len());
    let extension_bit = if extensions.is_empty() { 0 } else { 0x10 };
    out.push((RTP_VERSION << 6) | extension_bit);
    out.push((u8::from(header.marker) << 7) | (header.payload_type & MAX_PAYLOAD_TYPE));
    out.extend_from_slice(&header.sequence.to_be_bytes());
    out.extend_from_slice(&header.timestamp.to_be_bytes());
    out.extend_from_slice(&header.ssrc.to_be_bytes());
    if !extensions.is_empty() {
        let one_byte = one_byte_form(extensions);
        let profile = if one_byte {
            ONE_BYTE_PROFILE
        } else {
            TWO_BYTE_PROFILE
        };
        let words = (overhead - EXTENSION_HEADER_BYTES) / WORD_BYTES;
        out.extend_from_slice(&profile.to_be_bytes());
        out.extend_from_slice(&(words as u16).to_be_bytes());
        let block_start = out.len();
        for element in extensions {
            if element.id == 0 || element.data.len() > TWO_BYTE_MAX_LEN {
                return Err(RtpError::Config(format!(
                    "header extension id {} with {} bytes",
                    element.id,
                    element.data.len()
                )));
            }
            if one_byte {
                out.push((element.id << 4) | (element.data.len() - 1) as u8);
            } else {
                out.push(element.id);
                out.push(element.data.len() as u8);
            }
            out.extend_from_slice(element.data);
        }
        out.resize(block_start + words * WORD_BYTES, 0);
    }
    out.extend_from_slice(payload);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADER: RtpHeader = RtpHeader {
        marker: true,
        payload_type: 96,
        sequence: 0xFFFE,
        timestamp: 0x1234_5678,
        ssrc: 0xDEAD_BEEF,
    };

    #[test]
    fn plain_packet_round_trips() {
        let bytes = write_packet(&HEADER, &[], &[1, 2, 3]).unwrap();
        assert_eq!(bytes.len(), FIXED_HEADER_BYTES + 3);
        let packet = RtpPacket::parse(&bytes).unwrap();
        assert_eq!(packet.header, HEADER);
        assert_eq!(packet.payload, &[1, 2, 3]);
        assert_eq!(packet.extensions, vec![]);
    }

    #[test]
    fn one_byte_extensions_round_trip_padded_to_words() {
        let ext = [
            Extension {
                id: 3,
                data: &[9, 8, 7],
            },
            Extension { id: 14, data: &[1] },
        ];
        assert_eq!(extension_overhead(&ext), 4 + 8);
        let bytes = write_packet(&HEADER, &ext, &[5]).unwrap();
        assert_eq!(&bytes[12..14], &[0xBE, 0xDE]);
        let packet = RtpPacket::parse(&bytes).unwrap();
        assert_eq!(packet.extensions, ext);
        assert_eq!(packet.payload, &[5]);
    }

    #[test]
    fn long_extension_uses_the_two_byte_form() {
        let long = [7u8; 20];
        let ext = [Extension { id: 1, data: &long }];
        let bytes = write_packet(&HEADER, &ext, &[]).unwrap();
        assert_eq!(&bytes[12..14], &[0x10, 0x00]);
        assert_eq!(
            RtpPacket::parse(&bytes).unwrap().extension(1),
            Some(&long[..])
        );
    }

    #[test]
    fn padding_and_csrcs_are_skipped() {
        let mut bytes = vec![0xA1, 96, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3];
        bytes.extend_from_slice(&[0, 0, 0, 9]); // one CSRC
        bytes.extend_from_slice(&[0xAA, 0xBB, 0, 0, 3]); // payload, 3 padding bytes
        let packet = RtpPacket::parse(&bytes).unwrap();
        assert_eq!(packet.payload, &[0xAA, 0xBB]);
    }

    #[test]
    fn bad_packets_are_rejected() {
        assert!(RtpPacket::parse(&[0x80; 5]).is_err());
        assert!(RtpPacket::parse(&[0x40; 12]).is_err());
        let mut bytes = write_packet(&HEADER, &[], &[1]).unwrap();
        bytes[0] |= 0x20;
        *bytes.last_mut().unwrap() = 9; // 9 padding bytes in a 1-byte payload
        assert!(RtpPacket::parse(&bytes).is_err());
    }
}
