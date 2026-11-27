//! The AV2 multi-frame header OBU (`OBU_MULTI_FRAME_HEADER`): defaults a
//! frame header may name by id instead of coding them itself: the
//! sequence header id, the frame size, the deblocking switches and the
//! segment features frames reuse (`avmenc --enable-mfh-obu-signaling=1`
//! sends one with every key frame).
//!
//! Syntax from AVM v1.0.0 `av2/decoder/decodeframe.c`
//! `av2_read_multi_frame_header` and `av2/decoder/obu.c`
//! `read_multi_frame_header_obu`; dav2d 237229e (`src/obu.c`
//! `parse_multi_frame_hdr`, BSD-2-Clause, see `THIRD_PARTY.md`) reads the
//! same fields but no segment features, and uses only the sequence id.
//! AVM drops headers sent with a leading frame when the next regular frame
//! does not repeat them; this decoder keeps them (a conforming stream does
//! not name them then).
use nc_common::decode::{BitReader, DecodeError, Result};

use super::sequence::{SegmentFeatures, Segmentation};

/// Multi-frame header ids (`MAX_MFH_NUM`); id 0 is the frame header's own
/// fields, so an OBU sets 1 to 15.
pub const MAX_HEADERS: usize = 16;

/// The headers received so far, by id.
pub type MultiFrameHeaders = [Option<MultiFrameHeader>; MAX_HEADERS];

/// Bits of `mfh_frame_width_bits_minus1` and its height twin.
const SIZE_BITS_BITS: u32 = 4;

/// One multi-frame header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultiFrameHeader {
    /// `mfh_seq_header_id`.
    pub sequence_id: u32,
    /// The frame size of frames that code none (`mfh_frame_width`,
    /// `mfh_frame_height`); `None`: the sequence's largest.
    pub size: Option<(u32, u32)>,
    /// `mfh_apply_deblocking_filter` (luma vertical and horizontal edges,
    /// U, V), which frames then do not code; `None`: frames code them.
    pub deblock: Option<[bool; 4]>,
    /// Segment features frames may reuse in place of the sequence's
    /// (`mfh_seg_info_present_flag`); always with features.
    pub segmentation: Option<Segmentation>,
}

/// Parse a multi-frame header OBU's payload into `headers`.
/// # Errors
/// Truncation, and ids out of range.
pub fn parse(payload: &[u8], headers: &mut MultiFrameHeaders) -> Result<()> {
    let mut bits = BitReader::new(payload);
    let sequence_id = bits.read_uvlc()?;
    let id = bits.read_uvlc()? as usize + 1;
    if id >= MAX_HEADERS {
        return Err(DecodeError::Malformed("AV2 multi-frame header id"));
    }
    let size = if bits.read_bit()? {
        let width_bits = bits.read_bits(SIZE_BITS_BITS)? + 1;
        let height_bits = bits.read_bits(SIZE_BITS_BITS)? + 1;
        Some((
            bits.read_bits(width_bits)? + 1,
            bits.read_bits(height_bits)? + 1,
        ))
    } else {
        None
    };
    let deblock = if bits.read_bit()? {
        Some([
            bits.read_bit()?,
            bits.read_bit()?,
            bits.read_bit()?,
            bits.read_bit()?,
        ])
    } else {
        None
    };
    let segmentation = if bits.read_bit()? {
        let extended = bits.read_bit()?;
        let adaptive = bits.read_bit()?;
        let mut segmentation = Segmentation {
            extended,
            adaptive,
            features: None,
        };
        segmentation.features = Some(SegmentFeatures::parse(&mut bits, segmentation.count())?);
        Some(segmentation)
    } else {
        None
    };
    // `mfh_extension_present_flag`: extension data, then trailing bits,
    // neither of which this version reads.
    bits.read_bit()?;
    headers[id] = Some(MultiFrameHeader {
        sequence_id,
        size,
        deblock,
        segmentation,
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// avmenc's header for a key frame at the sequence's size: sequence 0,
    /// id 1, nothing else, no extension, trailing one bit.
    #[test]
    fn parses_avmenc_key_frame_header() {
        // uvlc 0 = `1`, uvlc 0 = `1`, then `0 0 0 0` and the trailing `1`.
        let payload = [0b1100_0010];
        let mut headers = MultiFrameHeaders::default();
        parse(&payload, &mut headers).unwrap();
        assert_eq!(
            headers[1],
            Some(MultiFrameHeader {
                sequence_id: 0,
                size: None,
                deblock: None,
                segmentation: None,
            })
        );
    }

    #[test]
    fn rejects_id_out_of_range() {
        // uvlc 0, then uvlc 15 (`000010000`): id 16.
        let payload = [0b1000_0100, 0b0010_0000];
        let mut headers = MultiFrameHeaders::default();
        assert!(parse(&payload, &mut headers).is_err());
    }
}
