//! VVC over RTP (RFC 9328) without DON fields (`sprop-max-don-diff = 0`):
//! single NAL unit packets, APs and FUs.
//!
//! The two-byte NAL header is `F(1) Z(1) LayerId(6) Type(5) TID(3)` (VVC
//! §7.3.1.2). An AP header takes the OR of `F` and the lowest `LayerId` and
//! `TID` of its units (RFC 9328 §4.3.2); an FU payload header copies the
//! unit's header with type 29, and the FU header is `S E P FuType(5)`,
//! `P` marking the last fragment of the picture's last VCL NAL unit
//! (§4.3.3).

use super::NalFormat;

const FORBIDDEN_BIT: u8 = 0x80;
const LAYER_MASK: u8 = 0x3F;
const TYPE_SHIFT: u8 = 3;
const TYPE_MASK: u8 = 0x1F;
const TID_MASK: u8 = 0x07;
/// FU header `P`: the picture's last VCL fragment.
const PICTURE_END: u8 = 0x20;
/// `IDR_W_RADL`, `IDR_N_LP` and CRA (VVC Table 5); GDR (10) recovers over
/// several pictures and is not a key.
const FIRST_KEY: u8 = 7;
const LAST_KEY: u8 = 9;
/// VCL NAL types 0-11.
const LAST_VCL: u8 = 11;

/// The VVC payload format.
#[derive(Debug)]
pub(crate) struct Vvc;

/// A header with `kind` in place of the type field.
fn with_type(header: &[u8], kind: u8) -> [u8; 2] {
    [header[0], (kind << TYPE_SHIFT) | (header[1] & TID_MASK)]
}

impl NalFormat for Vvc {
    const HEADER_BYTES: usize = 2;
    const AGGREGATION_TYPE: u8 = 28;
    const FRAGMENTATION_TYPE: u8 = 29;
    const FU_TYPE_MASK: u8 = TYPE_MASK;

    fn nal_type(header: &[u8]) -> u8 {
        (header[1] >> TYPE_SHIFT) & TYPE_MASK
    }

    fn aggregation_header(headers: &[&[u8]]) -> [u8; 2] {
        let forbidden = headers
            .iter()
            .fold(0, |acc, h| acc | (h[0] & FORBIDDEN_BIT));
        let layer = headers.iter().map(|h| h[0] & LAYER_MASK).min().unwrap_or(0);
        let tid = headers.iter().map(|h| h[1] & TID_MASK).min().unwrap_or(1);
        [
            forbidden | layer,
            (Self::AGGREGATION_TYPE << TYPE_SHIFT) | tid,
        ]
    }

    fn fragmentation_header(header: &[u8]) -> [u8; 2] {
        with_type(header, Self::FRAGMENTATION_TYPE)
    }

    fn rebuilt_header(payload_header: &[u8], fu_type: u8) -> [u8; 2] {
        with_type(payload_header, fu_type)
    }

    fn picture_end_bits() -> u8 {
        PICTURE_END
    }

    fn is_key(nal_type: u8) -> bool {
        (FIRST_KEY..=LAST_KEY).contains(&nal_type)
    }

    fn is_vcl(nal_type: u8) -> bool {
        nal_type <= LAST_VCL
    }
}
