//! The NAL-unit payload formats (H.264 RFC 6184, HEVC RFC 7798, VVC
//! RFC 9328), written once.
//!
//! The three formats share one design: a packet carries a single NAL unit,
//! an aggregation packet (STAP-A / AP) of several small ones, each preceded
//! by a 16-bit size, or a fragmentation unit (FU-A / FU) of one large one,
//! whose payload header and FU header rebuild the NAL unit header. They
//! differ only in the NAL header size (1 or 2 bytes), the aggregation and
//! fragmentation type numbers and how those headers' fields are formed,
//! which [`NalFormat`] supplies (`h264`, `hevc`, `vvc`). Only the
//! non-interleaved modes are carried: no DON fields (`sprop-max-don-diff =
//! 0`), no STAP-B, MTAP or FU-B.

pub(crate) mod h264;
pub(crate) mod hevc;
pub(crate) mod vvc;

use nc_common::decode::annex_b_units;
use nc_common::nal::START_CODE;

use crate::error::{Result, RtpError};
use crate::payload::{OutPayload, PacketInfo, ParsedPayload, Piece};

/// The 16-bit NAL unit size before each aggregation unit.
const AGGREGATION_SIZE_BYTES: usize = 2;
/// The FU header after the payload header.
const FU_HEADER_BYTES: usize = 1;
/// FU header `S` and `E` bits (all three formats).
const FU_START: u8 = 0x80;
const FU_END: u8 = 0x40;
/// Largest NAL unit an aggregation unit's size field can describe.
const MAX_AGGREGATED_NAL: usize = u16::MAX as usize;

/// What distinguishes one NAL-unit payload format from another.
pub(crate) trait NalFormat {
    /// NAL unit header size in bytes.
    const HEADER_BYTES: usize;
    /// Payload header type of an aggregation packet.
    const AGGREGATION_TYPE: u8;
    /// Payload header type of a fragmentation unit.
    const FRAGMENTATION_TYPE: u8;
    /// Mask of the FU header's type field.
    const FU_TYPE_MASK: u8;

    /// `nal_unit_type` of a NAL unit header.
    fn nal_type(header: &[u8]) -> u8;
    /// The payload header of an aggregation packet of NAL units with these
    /// headers.
    fn aggregation_header(headers: &[&[u8]]) -> [u8; 2];
    /// The payload header of a fragmentation unit of a NAL unit with this
    /// header.
    fn fragmentation_header(header: &[u8]) -> [u8; 2];
    /// The NAL unit header a fragmentation unit's payload header and FU
    /// type rebuild.
    fn rebuilt_header(payload_header: &[u8], fu_type: u8) -> [u8; 2];
    /// Extra FU header bits of the last fragment of the picture's last
    /// VCL NAL unit (VVC's `P`; none elsewhere).
    fn picture_end_bits() -> u8 {
        0
    }
    /// True for NAL types that start a key picture (IDR / IRAP).
    fn is_key(nal_type: u8) -> bool;
    /// True for VCL NAL types.
    fn is_vcl(nal_type: u8) -> bool;
    /// False when the header says no later picture references this one.
    fn may_be_reference(_header: &[u8]) -> bool {
        true
    }
}

/// Cut one Annex-B access unit into payloads of at most `budget` bytes.
pub(crate) fn packetize<F: NalFormat>(unit: &[u8], budget: usize) -> Result<Vec<OutPayload>> {
    let minimum = F::HEADER_BYTES + FU_HEADER_BYTES + 1;
    if budget < minimum {
        return Err(RtpError::Config(format!(
            "payload budget {budget} below the {minimum} bytes of one fragment"
        )));
    }
    let nals =
        annex_b_units(unit).map_err(|_| RtpError::Malformed("access unit is not Annex B"))?;
    if nals.iter().any(|nal| nal.len() <= F::HEADER_BYTES) {
        return Err(RtpError::Malformed("NAL unit without payload"));
    }
    let last_vcl = nals.iter().rposition(|nal| F::is_vcl(F::nal_type(nal)));
    let mut payloads = Vec::new();
    let mut group: Vec<&[u8]> = Vec::new();
    for (index, &nal) in nals.iter().enumerate() {
        if nal.len() > budget {
            flush_group::<F>(&mut group, &mut payloads);
            fragment::<F>(nal, budget, Some(index) == last_vcl, &mut payloads);
            continue;
        }
        if !group.is_empty() && aggregated_size::<F>(&group, nal) > budget {
            flush_group::<F>(&mut group, &mut payloads);
        }
        group.push(nal);
    }
    flush_group::<F>(&mut group, &mut payloads);
    if let Some(first) = payloads.first_mut() {
        first.start_of_frame = true;
    }
    if let Some(last) = payloads.last_mut() {
        last.end_of_frame = true;
    }
    Ok(payloads)
}

/// Size of an aggregation packet of `group` plus `next`.
fn aggregated_size<F: NalFormat>(group: &[&[u8]], next: &[u8]) -> usize {
    let units: usize = group
        .iter()
        .chain(std::iter::once(&next))
        .map(|nal| AGGREGATION_SIZE_BYTES + nal.len())
        .sum();
    F::HEADER_BYTES + units
}

/// Emit `group` as one single-NAL packet or one aggregation packet.
fn flush_group<F: NalFormat>(group: &mut Vec<&[u8]>, out: &mut Vec<OutPayload>) {
    let bytes = match group.as_slice() {
        [] => return,
        [single] => single.to_vec(),
        many => {
            let header = F::aggregation_header(many);
            let mut bytes = header[..F::HEADER_BYTES].to_vec();
            for nal in many {
                bytes.extend_from_slice(&(nal.len() as u16).to_be_bytes());
                bytes.extend_from_slice(nal);
            }
            bytes
        }
    };
    group.clear();
    out.push(OutPayload {
        bytes,
        frame: 0,
        start_of_frame: false,
        end_of_frame: false,
    });
}

/// Emit `nal` as fragmentation units of near-equal size.
fn fragment<F: NalFormat>(nal: &[u8], budget: usize, picture_end: bool, out: &mut Vec<OutPayload>) {
    let header = &nal[..F::HEADER_BYTES];
    let body = &nal[F::HEADER_BYTES..];
    let capacity = budget - F::HEADER_BYTES - FU_HEADER_BYTES;
    let count = body.len().div_ceil(capacity);
    let size = body.len().div_ceil(count);
    let payload_header = F::fragmentation_header(header);
    let fu_type = F::nal_type(header) & F::FU_TYPE_MASK;
    for (index, chunk) in body.chunks(size).enumerate() {
        let last = index == count - 1;
        let mut fu_header = fu_type;
        if index == 0 {
            fu_header |= FU_START;
        }
        if last {
            fu_header |= FU_END;
            if picture_end {
                fu_header |= F::picture_end_bits();
            }
        }
        let mut bytes = Vec::with_capacity(F::HEADER_BYTES + FU_HEADER_BYTES + chunk.len());
        bytes.extend_from_slice(&payload_header[..F::HEADER_BYTES]);
        bytes.push(fu_header);
        bytes.extend_from_slice(chunk);
        out.push(OutPayload {
            bytes,
            frame: 0,
            start_of_frame: false,
            end_of_frame: false,
        });
    }
}

/// Parse one payload into NAL unit pieces.
pub(crate) fn depacketize<F: NalFormat>(payload: &[u8]) -> Result<ParsedPayload> {
    if payload.len() <= F::HEADER_BYTES {
        return Err(RtpError::Truncated("NAL unit payload"));
    }
    let header = &payload[..F::HEADER_BYTES];
    let kind = F::nal_type(header);
    let pieces = if kind == F::AGGREGATION_TYPE {
        aggregation_units::<F>(&payload[F::HEADER_BYTES..])?
    } else if kind == F::FRAGMENTATION_TYPE {
        let fu = *payload
            .get(F::HEADER_BYTES)
            .ok_or(RtpError::Truncated("FU header"))?;
        let start = fu & FU_START != 0;
        let end = fu & FU_END != 0;
        let body = &payload[F::HEADER_BYTES + FU_HEADER_BYTES..];
        let mut data = Vec::with_capacity(F::HEADER_BYTES + body.len());
        if start {
            data.extend_from_slice(
                &F::rebuilt_header(header, fu & F::FU_TYPE_MASK)[..F::HEADER_BYTES],
            );
        }
        data.extend_from_slice(body);
        vec![Piece {
            data,
            continues_previous: !start,
            continues_next: !end,
        }]
    } else {
        vec![Piece::whole(payload.to_vec())]
    };
    let headers = pieces
        .iter()
        .filter(|piece| !piece.continues_previous)
        .map(|piece| &piece.data[..F::HEADER_BYTES]);
    let mut info = PacketInfo::default();
    for header in headers {
        let kind = F::nal_type(header);
        info.key |= F::is_key(kind);
        if F::is_vcl(kind) {
            info.reference = Some(info.reference == Some(true) || F::may_be_reference(header));
        }
    }
    Ok(ParsedPayload { pieces, info })
}

/// The NAL units of an aggregation packet body.
fn aggregation_units<F: NalFormat>(mut body: &[u8]) -> Result<Vec<Piece>> {
    let mut pieces = Vec::new();
    while !body.is_empty() {
        let size_field = body
            .get(..AGGREGATION_SIZE_BYTES)
            .ok_or(RtpError::Truncated("aggregation unit size"))?;
        let size = usize::from(u16::from_be_bytes([size_field[0], size_field[1]]));
        if size <= F::HEADER_BYTES || size > MAX_AGGREGATED_NAL {
            return Err(RtpError::Malformed("aggregation unit size"));
        }
        let nal = body
            .get(AGGREGATION_SIZE_BYTES..AGGREGATION_SIZE_BYTES + size)
            .ok_or(RtpError::Truncated("aggregation unit"))?;
        pieces.push(Piece::whole(nal.to_vec()));
        body = &body[AGGREGATION_SIZE_BYTES + size..];
    }
    if pieces.is_empty() {
        return Err(RtpError::Malformed("empty aggregation packet"));
    }
    Ok(pieces)
}

/// Join NAL units into an Annex-B access unit.
pub(crate) fn join(nals: &[Vec<u8>]) -> Vec<u8> {
    let bytes: usize = nals.iter().map(|nal| START_CODE.len() + nal.len()).sum();
    let mut unit = Vec::with_capacity(bytes);
    for nal in nals {
        unit.extend_from_slice(&START_CODE);
        unit.extend_from_slice(nal);
    }
    unit
}

#[cfg(test)]
mod tests {
    use super::h264::H264;
    use super::hevc::Hevc;
    use super::vvc::Vvc;
    use super::*;

    /// An access unit of NAL units with the given headers and body sizes.
    fn access_unit(nals: &[(&[u8], usize)]) -> (Vec<u8>, Vec<Vec<u8>>) {
        let mut list = Vec::new();
        for (index, &(header, size)) in nals.iter().enumerate() {
            let mut nal = header.to_vec();
            // Bodies free of start-code patterns: 0x00 never appears.
            nal.extend((0..size).map(|i| (1 + (i * 7 + index) % 250) as u8));
            list.push(nal);
        }
        (join(&list), list)
    }

    /// Depacketize `payloads` and concatenate their pieces into NAL units.
    fn reassemble<F: NalFormat>(payloads: &[OutPayload]) -> Vec<Vec<u8>> {
        let mut nals: Vec<Vec<u8>> = Vec::new();
        let mut open = false;
        for payload in payloads {
            for piece in depacketize::<F>(&payload.bytes).unwrap().pieces {
                assert_eq!(piece.continues_previous, open);
                if piece.continues_previous {
                    nals.last_mut().unwrap().extend_from_slice(&piece.data);
                } else {
                    nals.push(piece.data);
                }
                open = piece.continues_next;
            }
        }
        assert!(!open);
        nals
    }

    fn round_trip<F: NalFormat>(nals: &[(&[u8], usize)], budget: usize) -> Vec<OutPayload> {
        let (unit, list) = access_unit(nals);
        let payloads = packetize::<F>(&unit, budget).unwrap();
        assert!(payloads.iter().all(|p| p.bytes.len() <= budget));
        assert!(payloads[0].start_of_frame && payloads.last().unwrap().end_of_frame);
        assert_eq!(reassemble::<F>(&payloads), list);
        payloads
    }

    #[test]
    fn h264_aggregates_small_and_fragments_large_units() {
        // SPS, PPS, an IDR slice of 3000 bytes.
        let payloads = round_trip::<H264>(&[(&[0x67], 20), (&[0x68], 4), (&[0x65], 3000)], 1200);
        assert_eq!(payloads[0].bytes[0] & 0x1F, 24, "STAP-A first");
        assert_eq!(payloads[1].bytes[0] & 0x1F, 28, "then FU-A");
        assert_eq!(payloads.len(), 4);
        let info = depacketize::<H264>(&payloads[0].bytes).unwrap().info;
        assert!(!info.key && info.reference.is_none(), "an SPS/PPS packet");
        let info = depacketize::<H264>(&payloads[1].bytes).unwrap().info;
        assert!(info.key && info.reference == Some(true));
    }

    #[test]
    fn h264_single_nal_and_non_reference_slices() {
        let payloads = round_trip::<H264>(&[(&[0x01], 100)], 1200);
        assert_eq!(payloads.len(), 1);
        assert_eq!(payloads[0].bytes[0], 0x01, "single NAL unit packet");
        let info = depacketize::<H264>(&payloads[0].bytes).unwrap().info;
        assert!(!info.key && info.reference == Some(false));
    }

    #[test]
    fn hevc_round_trips() {
        // VPS, SPS, PPS, IDR_W_RADL slice.
        let payloads = round_trip::<Hevc>(
            &[
                (&[0x40, 0x01], 22),
                (&[0x42, 0x01], 40),
                (&[0x44, 0x01], 6),
                (&[0x26, 0x01], 2500),
            ],
            1000,
        );
        assert_eq!((payloads[0].bytes[0] >> 1) & 0x3F, 48);
        assert_eq!((payloads[1].bytes[0] >> 1) & 0x3F, 49);
        assert!(depacketize::<Hevc>(&payloads[1].bytes).unwrap().info.key);
    }

    #[test]
    fn vvc_marks_the_picture_end_fragment() {
        // SPS (type 15), PPS (16), IDR_N_LP slice (8), TID 0.
        let payloads = round_trip::<Vvc>(
            &[
                (&[0x00, 0x79], 30),
                (&[0x00, 0x81], 8),
                (&[0x00, 0x41], 2500),
            ],
            1000,
        );
        assert_eq!(payloads[0].bytes[1] >> 3, 28);
        let last = payloads.last().unwrap();
        assert_eq!(last.bytes[1] >> 3, 29);
        assert_eq!(
            last.bytes[2] & 0xE0,
            0x40 | 0x20,
            "E and P on the last fragment"
        );
        assert!(depacketize::<Vvc>(&payloads[1].bytes).unwrap().info.key);
    }

    #[test]
    fn malformed_payloads_are_rejected() {
        assert!(depacketize::<H264>(&[24, 0, 9, 0x65]).is_err());
        assert!(depacketize::<H264>(&[24]).is_err());
        assert!(depacketize::<Hevc>(&[0x60, 0x01, 0x00, 0x01]).is_err());
        assert!(packetize::<H264>(&[1, 2, 3], 1200).is_err());
    }
}
