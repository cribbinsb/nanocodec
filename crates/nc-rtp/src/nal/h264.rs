//! H.264 over RTP (RFC 6184), `packetization-mode=1`: single NAL unit
//! packets, STAP-A and FU-A.
//!
//! The one-byte NAL header is `F(1) NRI(2) Type(5)` (H.264 §7.3.1). A STAP-A
//! takes the OR of its units' `F` and the highest `NRI` (RFC 6184 §5.7.1);
//! an FU indicator carries the fragmented unit's `F` and `NRI` with type
//! 28, and the FU header `S E R Type` (§5.8).

use super::NalFormat;

const FORBIDDEN_BIT: u8 = 0x80;
const NRI_MASK: u8 = 0x60;
const TYPE_MASK: u8 = 0x1F;
/// `nal_unit_type` of an IDR slice (H.264 Table 7-1).
const IDR_SLICE: u8 = 5;
/// VCL NAL types 1-5 (H.264 Table 7-1).
const FIRST_VCL: u8 = 1;
const LAST_VCL: u8 = 5;

/// The H.264 payload format.
#[derive(Debug)]
pub(crate) struct H264;

impl NalFormat for H264 {
    const HEADER_BYTES: usize = 1;
    const AGGREGATION_TYPE: u8 = 24; // STAP-A
    const FRAGMENTATION_TYPE: u8 = 28; // FU-A
    const FU_TYPE_MASK: u8 = TYPE_MASK;

    fn nal_type(header: &[u8]) -> u8 {
        header[0] & TYPE_MASK
    }

    fn aggregation_header(headers: &[&[u8]]) -> [u8; 2] {
        let forbidden = headers
            .iter()
            .fold(0, |acc, h| acc | (h[0] & FORBIDDEN_BIT));
        let nri = headers.iter().map(|h| h[0] & NRI_MASK).max().unwrap_or(0);
        [forbidden | nri | Self::AGGREGATION_TYPE, 0]
    }

    fn fragmentation_header(header: &[u8]) -> [u8; 2] {
        [(header[0] & !TYPE_MASK) | Self::FRAGMENTATION_TYPE, 0]
    }

    fn rebuilt_header(payload_header: &[u8], fu_type: u8) -> [u8; 2] {
        [(payload_header[0] & !TYPE_MASK) | fu_type, 0]
    }

    fn is_key(nal_type: u8) -> bool {
        nal_type == IDR_SLICE
    }

    fn is_vcl(nal_type: u8) -> bool {
        (FIRST_VCL..=LAST_VCL).contains(&nal_type)
    }

    /// `nal_ref_idc == 0`: no picture references this one (H.264 §7.4.1).
    fn may_be_reference(header: &[u8]) -> bool {
        header[0] & NRI_MASK != 0
    }
}
