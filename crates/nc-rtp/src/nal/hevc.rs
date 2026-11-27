//! HEVC over RTP (RFC 7798) without DON fields (`sprop-max-don-diff = 0`):
//! single NAL unit packets, APs and FUs.
//!
//! The two-byte NAL header is `F(1) Type(6) LayerId(6) TID(3)` (HEVC
//! §7.3.1.2). An AP header takes the OR of `F` and the lowest `LayerId` and
//! `TID` of its units (RFC 7798 §4.4.2); an FU payload header copies the
//! unit's header with type 49, and the FU header is `S E FuType(6)`
//! (§4.4.3). PACI (type 50) is not carried.

use super::NalFormat;

const FORBIDDEN_BIT: u8 = 0x80;
const TYPE_SHIFT: u8 = 1;
const TYPE_MASK: u8 = 0x3F;
/// `nuh_layer_id` straddles the two bytes: 1 bit, then 5.
const LAYER_HIGH_BIT: u8 = 0x01;
const LAYER_LOW_MASK: u8 = 0xF8;
const TID_MASK: u8 = 0x07;
/// IRAP NAL types 16-23 (BLA, IDR, CRA and reserved IRAP; HEVC Table 7-1).
const FIRST_IRAP: u8 = 16;
const LAST_IRAP: u8 = 23;
/// VCL NAL types 0-31.
const LAST_VCL: u8 = 31;

/// The HEVC payload format.
#[derive(Debug)]
pub(crate) struct Hevc;

/// `nuh_layer_id` of a header.
fn layer_id(header: &[u8]) -> u8 {
    ((header[0] & LAYER_HIGH_BIT) << 5) | (header[1] >> 3)
}

/// A header with `kind` in place of the type field.
fn with_type(header: &[u8], kind: u8) -> [u8; 2] {
    [
        (header[0] & !(TYPE_MASK << TYPE_SHIFT)) | (kind << TYPE_SHIFT),
        header[1],
    ]
}

impl NalFormat for Hevc {
    const HEADER_BYTES: usize = 2;
    const AGGREGATION_TYPE: u8 = 48;
    const FRAGMENTATION_TYPE: u8 = 49;
    const FU_TYPE_MASK: u8 = TYPE_MASK;

    fn nal_type(header: &[u8]) -> u8 {
        (header[0] >> TYPE_SHIFT) & TYPE_MASK
    }

    fn aggregation_header(headers: &[&[u8]]) -> [u8; 2] {
        let forbidden = headers
            .iter()
            .fold(0, |acc, h| acc | (h[0] & FORBIDDEN_BIT));
        let layer = headers.iter().map(|h| layer_id(h)).min().unwrap_or(0);
        let tid = headers.iter().map(|h| h[1] & TID_MASK).min().unwrap_or(1);
        [
            forbidden | (Self::AGGREGATION_TYPE << TYPE_SHIFT) | (layer >> 5),
            ((layer << 3) & LAYER_LOW_MASK) | tid,
        ]
    }

    fn fragmentation_header(header: &[u8]) -> [u8; 2] {
        with_type(header, Self::FRAGMENTATION_TYPE)
    }

    fn rebuilt_header(payload_header: &[u8], fu_type: u8) -> [u8; 2] {
        with_type(payload_header, fu_type)
    }

    fn is_key(nal_type: u8) -> bool {
        (FIRST_IRAP..=LAST_IRAP).contains(&nal_type)
    }

    fn is_vcl(nal_type: u8) -> bool {
        nal_type <= LAST_VCL
    }
}
