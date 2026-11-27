//! AV1 over RTP: the AOM "RTP Payload Format For AV1" (v1.0), §4-§5.
//!
//! A payload is a one-byte aggregation header `Z Y W(2) N` and a series of
//! OBU elements, each a whole OBU or a fragment, each preceded by its leb128
//! length except the last when `W` counts the elements (§4.4). The sender
//! clears `obu_has_size_field` (the length is in the element; §4.5), drops
//! temporal delimiters and tile lists, never puts two layers in one packet,
//! puts layer-less OBUs (sequence header, metadata) first with the frame
//! that follows them, and sets `N` on the first packet of a key temporal
//! unit (§5). The receiver rebuilds low-overhead OBUs (size fields set)
//! behind a temporal delimiter, the form every AV1 decoder reads.
//!
//! The few OBU header fields read here are the AV1 specification's
//! (§5.3.2-§5.3.3); this crate does not depend on the codec crate.

use nc_common::leb128;

use crate::error::{Result, RtpError};
use crate::payload::{OutPayload, PacketInfo, ParsedPayload, Piece};

/// Aggregation header bits (§4.4).
const Z_CONTINUES_PREVIOUS: u8 = 0x80;
const Y_CONTINUES_NEXT: u8 = 0x40;
const W_SHIFT: u8 = 4;
const W_MASK: u8 = 0x03;
const N_NEW_SEQUENCE: u8 = 0x08;
/// `W` can count at most three elements.
const MAX_COUNTED_ELEMENTS: usize = 3;
const AGGREGATION_HEADER_BYTES: usize = 1;

/// OBU header fields (AV1 §5.3.2).
const OBU_TYPE_SHIFT: u8 = 3;
const OBU_TYPE_MASK: u8 = 0x0F;
const OBU_EXTENSION_FLAG: u8 = 0x04;
const OBU_HAS_SIZE_FIELD: u8 = 0x02;
/// `obu_extension_header()` (AV1 §5.3.3): `temporal_id(3) spatial_id(2)`.
const TEMPORAL_ID_SHIFT: u8 = 5;
const SPATIAL_ID_SHIFT: u8 = 3;
const SPATIAL_ID_MASK: u8 = 0x03;
/// OBU types (AV1 §6.2.2).
const OBU_SEQUENCE_HEADER: u8 = 1;
const OBU_TEMPORAL_DELIMITER: u8 = 2;
const OBU_TILE_LIST: u8 = 8;
/// A temporal delimiter OBU with an empty size field.
const TEMPORAL_DELIMITER: [u8; 2] = [
    OBU_TEMPORAL_DELIMITER << OBU_TYPE_SHIFT | OBU_HAS_SIZE_FIELD,
    0,
];
/// Largest leb128 this module reads (8 bytes, AV1 §4.10.5).
const MAX_LEB128_BYTES: usize = 8;

/// One OBU of a temporal unit, its size field removed.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Obu {
    /// Header (and extension) with `obu_has_size_field` cleared, then the
    /// payload.
    element: Vec<u8>,
    /// `(temporal_id, spatial_id)` from the extension header.
    layer: Option<(u8, u8)>,
    kind: u8,
}

/// Read a leb128 value from the start of `data`: value and byte count.
fn read_leb128(data: &[u8]) -> Result<(usize, usize)> {
    let mut value = 0u64;
    for (index, &byte) in data.iter().take(MAX_LEB128_BYTES).enumerate() {
        value |= u64::from(byte & 0x7F) << (7 * index);
        if byte & 0x80 == 0 {
            let value = usize::try_from(value).map_err(|_| RtpError::Malformed("leb128 size"))?;
            return Ok((value, index + 1));
        }
    }
    Err(RtpError::Truncated("leb128 size"))
}

/// Split a low-overhead temporal unit into OBUs.
fn parse_obus(mut data: &[u8]) -> Result<Vec<Obu>> {
    let mut obus = Vec::new();
    while !data.is_empty() {
        let header = data[0];
        let has_extension = header & OBU_EXTENSION_FLAG != 0;
        let header_bytes = 1 + usize::from(has_extension);
        let extension = *data
            .get(header_bytes - 1)
            .ok_or(RtpError::Truncated("OBU extension"))?;
        let (payload_bytes, size_bytes) = if header & OBU_HAS_SIZE_FIELD != 0 {
            read_leb128(&data[header_bytes..])?
        } else {
            (data.len() - header_bytes, 0)
        };
        let payload_start = header_bytes + size_bytes;
        let payload = data
            .get(payload_start..payload_start + payload_bytes)
            .ok_or(RtpError::Truncated("OBU payload"))?;
        let mut element = Vec::with_capacity(header_bytes + payload.len());
        element.push(header & !OBU_HAS_SIZE_FIELD);
        if has_extension {
            element.push(extension);
        }
        element.extend_from_slice(payload);
        obus.push(Obu {
            element,
            layer: has_extension.then_some((
                extension >> TEMPORAL_ID_SHIFT,
                (extension >> SPATIAL_ID_SHIFT) & SPATIAL_ID_MASK,
            )),
            kind: (header >> OBU_TYPE_SHIFT) & OBU_TYPE_MASK,
        });
        data = &data[payload_start + payload_bytes..];
    }
    Ok(obus)
}

/// The OBUs of one layer frame: the layer-less OBUs before it, then its own.
#[derive(Debug, Default)]
struct LayerGroup {
    spatial_id: Option<u8>,
    elements: Vec<Vec<u8>>,
}

/// Group the OBUs to send: one group per layer frame, temporal delimiters
/// and tile lists dropped (§5).
fn group_by_layer(obus: Vec<Obu>) -> Vec<LayerGroup> {
    let mut groups: Vec<LayerGroup> = Vec::new();
    let mut pending: Vec<Vec<u8>> = Vec::new();
    for obu in obus {
        if obu.kind == OBU_TEMPORAL_DELIMITER || obu.kind == OBU_TILE_LIST {
            continue;
        }
        match obu.layer {
            None => pending.push(obu.element),
            Some((_, spatial)) => {
                let same = groups
                    .last()
                    .is_some_and(|g| g.spatial_id == Some(spatial) && pending.is_empty());
                if !same {
                    groups.push(LayerGroup {
                        spatial_id: Some(spatial),
                        elements: std::mem::take(&mut pending),
                    });
                }
                groups
                    .last_mut()
                    .expect("pushed above")
                    .elements
                    .push(obu.element);
            }
        }
    }
    if !pending.is_empty() {
        match groups.last_mut() {
            // Trailing layer-less OBUs (padding, metadata) ride with the last frame.
            Some(last) => last.elements.extend(pending),
            None => groups.push(LayerGroup {
                spatial_id: None,
                elements: pending,
            }),
        }
    }
    groups
}

/// Packets under construction for one layer frame.
struct PacketBuilder {
    budget: usize,
    frame: usize,
    packets: Vec<OutPayload>,
    /// Elements of the open packet and whether its first continues.
    elements: Vec<Vec<u8>>,
    continues_previous: bool,
}

impl PacketBuilder {
    /// Room left in the open packet (counting a length field on every
    /// element, the worst case).
    fn room(&self) -> usize {
        let used: usize = AGGREGATION_HEADER_BYTES
            + self
                .elements
                .iter()
                .map(|e| leb128(e.len() as u64).len() + e.len())
                .sum::<usize>();
        self.budget.saturating_sub(used)
    }

    /// Close the open packet; `continues_next` when its last element is cut.
    fn close(&mut self, continues_next: bool) {
        if self.elements.is_empty() {
            return;
        }
        let count = self.elements.len();
        let counted = count <= MAX_COUNTED_ELEMENTS;
        let mut header = 0u8;
        if self.continues_previous {
            header |= Z_CONTINUES_PREVIOUS;
        }
        if continues_next {
            header |= Y_CONTINUES_NEXT;
        }
        if counted {
            header |= (count as u8) << W_SHIFT;
        }
        let mut bytes = vec![header];
        for (index, element) in self.elements.iter().enumerate() {
            if !(counted && index == count - 1) {
                bytes.extend_from_slice(&leb128(element.len() as u64));
            }
            bytes.extend_from_slice(element);
        }
        self.packets.push(OutPayload {
            bytes,
            frame: self.frame,
            start_of_frame: self.packets.is_empty(),
            end_of_frame: false,
        });
        self.elements.clear();
        self.continues_previous = continues_next;
    }

    /// Add one OBU element, fragmenting it across packets as needed.
    fn push(&mut self, mut element: &[u8]) {
        loop {
            let room = self.room();
            if leb128(element.len() as u64).len() + element.len() <= room {
                self.elements.push(element.to_vec());
                return;
            }
            // Largest fragment that fits with its own length field.
            let mut take = room.saturating_sub(1);
            while take > 0 && leb128(take as u64).len() + take > room {
                take -= 1;
            }
            if take == 0 {
                self.close(false);
                continue;
            }
            self.elements.push(element[..take].to_vec());
            element = &element[take..];
            self.close(true);
        }
    }
}

/// Cut one temporal unit into payloads of at most `budget` bytes.
///
/// `frame_of_spatial` maps a group's spatial id to the layer frame index
/// the payloads are tagged with (`None`: groups are numbered in order).
/// `key` sets `N` on the first packet.
pub(crate) fn packetize(
    unit: &[u8],
    budget: usize,
    key: bool,
    frame_of_spatial: impl Fn(u8) -> Option<usize>,
) -> Result<Vec<OutPayload>> {
    if budget <= AGGREGATION_HEADER_BYTES + 2 {
        return Err(RtpError::Config(format!(
            "payload budget {budget} too small for AV1"
        )));
    }
    let groups = group_by_layer(parse_obus(unit)?);
    let mut payloads = Vec::new();
    for (order, group) in groups.into_iter().enumerate() {
        let frame = group
            .spatial_id
            .and_then(&frame_of_spatial)
            .unwrap_or(order);
        let mut builder = PacketBuilder {
            budget,
            frame,
            packets: Vec::new(),
            elements: Vec::new(),
            continues_previous: false,
        };
        for element in &group.elements {
            builder.push(element);
        }
        builder.close(false);
        if let Some(last) = builder.packets.last_mut() {
            last.end_of_frame = true;
        }
        payloads.extend(builder.packets);
    }
    if key && let Some(first) = payloads.first_mut() {
        first.bytes[0] |= N_NEW_SEQUENCE;
    }
    Ok(payloads)
}

/// True when the unit holds a sequence header (a key unit of our encoder,
/// which writes one only before a key frame).
pub(crate) fn has_sequence_header(unit: &[u8]) -> bool {
    parse_obus(unit).is_ok_and(|obus| obus.iter().any(|obu| obu.kind == OBU_SEQUENCE_HEADER))
}

/// Parse one payload into OBU pieces.
pub(crate) fn depacketize(payload: &[u8]) -> Result<ParsedPayload> {
    let (&header, mut body) = payload
        .split_first()
        .ok_or(RtpError::Truncated("AV1 aggregation header"))?;
    let counted = usize::from((header >> W_SHIFT) & W_MASK);
    let mut pieces = Vec::new();
    while !body.is_empty() {
        let last_counted = counted != 0 && pieces.len() + 1 == counted;
        let (len, field) = if last_counted {
            (body.len(), 0)
        } else {
            read_leb128(body)?
        };
        let element = body
            .get(field..field + len)
            .ok_or(RtpError::Truncated("OBU element"))?;
        pieces.push(Piece::whole(element.to_vec()));
        body = &body[field + len..];
        if last_counted {
            break;
        }
    }
    if pieces.is_empty() || (counted != 0 && pieces.len() != counted) {
        return Err(RtpError::Malformed("AV1 element count"));
    }
    if header & Z_CONTINUES_PREVIOUS != 0 {
        pieces[0].continues_previous = true;
    }
    if header & Y_CONTINUES_NEXT != 0 {
        pieces.last_mut().expect("non-empty").continues_next = true;
    }
    Ok(ParsedPayload {
        pieces,
        info: PacketInfo {
            key: header & N_NEW_SEQUENCE != 0,
            ..PacketInfo::default()
        },
    })
}

/// Join received OBUs into a decoder-ready temporal unit: a temporal
/// delimiter, then every OBU with its size field (temporal delimiters and
/// tile lists received are dropped, §5).
pub(crate) fn join(obus: &[Vec<u8>]) -> Vec<u8> {
    let mut unit = TEMPORAL_DELIMITER.to_vec();
    for obu in obus {
        let Some(&header) = obu.first() else { continue };
        let kind = (header >> OBU_TYPE_SHIFT) & OBU_TYPE_MASK;
        if kind == OBU_TEMPORAL_DELIMITER || kind == OBU_TILE_LIST {
            continue;
        }
        if header & OBU_HAS_SIZE_FIELD != 0 {
            unit.extend_from_slice(obu); // a sender kept the size field
            continue;
        }
        let header_bytes = (1 + usize::from(header & OBU_EXTENSION_FLAG != 0)).min(obu.len());
        unit.push(header | OBU_HAS_SIZE_FIELD);
        unit.extend_from_slice(&obu[1..header_bytes]);
        unit.extend_from_slice(&leb128((obu.len() - header_bytes) as u64));
        unit.extend_from_slice(&obu[header_bytes..]);
    }
    unit
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A low-overhead OBU with size field.
    fn obu(kind: u8, layer: Option<(u8, u8)>, payload_len: usize) -> Vec<u8> {
        let mut out = vec![(kind << OBU_TYPE_SHIFT) | OBU_HAS_SIZE_FIELD];
        if let Some((t, s)) = layer {
            out[0] |= OBU_EXTENSION_FLAG;
            out.push((t << TEMPORAL_ID_SHIFT) | (s << SPATIAL_ID_SHIFT));
        }
        out.extend_from_slice(&leb128(payload_len as u64));
        out.extend((0..payload_len).map(|i| (i * 13 % 251) as u8));
        out
    }

    /// Reassemble payloads into OBUs, checking the Z/Y chaining.
    fn reassemble(payloads: &[OutPayload]) -> Vec<Vec<u8>> {
        let mut obus: Vec<Vec<u8>> = Vec::new();
        let mut open = false;
        for p in payloads {
            for piece in depacketize(&p.bytes).unwrap().pieces {
                assert_eq!(piece.continues_previous, open);
                if piece.continues_previous {
                    obus.last_mut().unwrap().extend_from_slice(&piece.data);
                } else {
                    obus.push(piece.data);
                }
                open = piece.continues_next;
            }
        }
        assert!(!open);
        obus
    }

    #[test]
    fn single_layer_unit_round_trips_without_its_delimiter() {
        let unit = [
            obu(OBU_TEMPORAL_DELIMITER, None, 0),
            obu(OBU_SEQUENCE_HEADER, None, 12),
            obu(6, None, 5000),
        ]
        .concat();
        let payloads = packetize(&unit, 1200, true, |_| None).unwrap();
        assert!(payloads.iter().all(|p| p.bytes.len() <= 1200));
        assert_eq!(payloads[0].bytes[0] & N_NEW_SEQUENCE, N_NEW_SEQUENCE);
        assert!(
            payloads[1..]
                .iter()
                .all(|p| p.bytes[0] & N_NEW_SEQUENCE == 0)
        );
        let joined = join(&reassemble(&payloads));
        assert_eq!(joined, unit, "TD restored, size fields restored");
    }

    #[test]
    fn layers_never_share_a_packet() {
        let unit = [
            obu(OBU_TEMPORAL_DELIMITER, None, 0),
            obu(OBU_SEQUENCE_HEADER, None, 10),
            obu(6, Some((0, 0)), 30),
            obu(6, Some((0, 1)), 40),
        ]
        .concat();
        let payloads = packetize(&unit, 1200, false, |s| Some(usize::from(s))).unwrap();
        assert_eq!(payloads.len(), 2);
        assert_eq!((payloads[0].frame, payloads[1].frame), (0, 1));
        assert_eq!(payloads[0].bytes[0] >> W_SHIFT, 2, "SH and the base frame");
        assert!(payloads.iter().all(|p| p.start_of_frame && p.end_of_frame));
        assert_eq!(join(&reassemble(&payloads)), unit);
    }

    #[test]
    fn many_small_obus_use_w_zero() {
        let mut unit = obu(OBU_TEMPORAL_DELIMITER, None, 0);
        for _ in 0..5 {
            unit.extend(obu(5, None, 3));
        }
        let payloads = packetize(&unit, 1200, false, |_| None).unwrap();
        assert_eq!(payloads.len(), 1);
        assert_eq!(payloads[0].bytes[0] >> W_SHIFT & W_MASK, 0);
        assert_eq!(join(&reassemble(&payloads)), unit);
    }

    #[test]
    fn foreign_payloads_with_size_fields_join_unchanged() {
        let whole = obu(6, None, 4);
        let payload = [&[0x10u8][..], &whole].concat(); // W = 1
        let parsed = depacketize(&payload).unwrap();
        let unit = join(&[parsed.pieces[0].data.clone()]);
        assert_eq!(&unit[2..], &whole[..]);
    }

    #[test]
    fn malformed_payloads_are_rejected() {
        assert!(depacketize(&[]).is_err());
        assert!(depacketize(&[0x00, 0x05, 1]).is_err());
        assert!(
            depacketize(&[0x30, 0x01, 1]).is_err(),
            "W = 3 but one element"
        );
    }
}
