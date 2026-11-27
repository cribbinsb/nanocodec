//! VP9 over RTP (RFC 9628): the payload descriptor in flexible and
//! non-flexible mode, the scalability structure (SS), and fragmentation of
//! layer frames.
//!
//! Each VP9 frame (one per spatial layer of a picture) is sent on its own,
//! fragmented in order between packets with `B` (start) and `E` (end) set
//! (§4.3); the descriptor carries a 15-bit picture ID, the layer indices,
//! and either the references as picture-ID diffs (flexible mode, `F = 1`)
//! or a TL0PICIDX (non-flexible mode). The SS goes on the first packet of a
//! key picture (§4.2.1). A superframe is never sent whole: the caller
//! gives each frame's byte range.

use crate::error::{Result, RtpError};
use crate::payload::{FrameTag, OutPayload, PacketInfo, ParsedPayload, Piece};

/// First octet bits (§4.2).
const I_PICTURE_ID: u8 = 0x80;
const P_INTER_PICTURE: u8 = 0x40;
const L_LAYER_INDICES: u8 = 0x20;
const F_FLEXIBLE: u8 = 0x10;
const B_START: u8 = 0x08;
const E_END: u8 = 0x04;
const V_SCALABILITY: u8 = 0x02;
const Z_NOT_UPPER_REFERENCE: u8 = 0x01;
/// Picture ID: `M` selects the 15-bit form.
const M_EXTENDED: u8 = 0x80;
const PICTURE_ID_MASK: u16 = 0x7FFF;
const SHORT_PICTURE_ID_MASK: u8 = 0x7F;
/// Layer indices `TID(3) U(1) SID(3) D(1)`.
const TID_SHIFT: u8 = 5;
const U_SWITCH_UP: u8 = 0x10;
const SID_SHIFT: u8 = 1;
const SID_MASK: u8 = 0x07;
const D_INTER_LAYER: u8 = 0x01;
/// Reference indices `P_DIFF(7) N(1)`.
const N_MORE_DIFFS: u8 = 0x01;
/// At most three `P_DIFF`s, each 1-127.
const MAX_P_DIFFS: usize = 3;
const MAX_P_DIFF: u64 = 127;
/// SS header `N_S(3) Y(1) G(1)`.
const SS_COUNT_SHIFT: u8 = 5;
const SS_Y_SIZES: u8 = 0x10;
const SS_G_GROUP: u8 = 0x08;
/// SS picture group entries `TID(3) U(1) R(2)`.
const PG_R_SHIFT: u8 = 2;
const PG_R_MASK: u8 = 0x03;
/// VP9 uncompressed header (VP9 §6.2): `frame_marker(2) profile_low(1)
/// profile_high(1)`, a reserved bit for profile 3, then
/// `show_existing_frame(1) frame_type(1)` (0 = `KEY_FRAME`).
const FRAME_MARKER: u8 = 0b10;
const PROFILE_3: u8 = 3;
/// Frame identities: picture ID times this plus the spatial id.
const SPATIAL_SLOTS: u64 = 8;

/// The descriptor mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Vp9Mode {
    /// References as picture-ID diffs in every packet (`F = 1`).
    #[default]
    Flexible,
    /// TL0PICIDX only (`F = 0`).
    NonFlexible,
}

/// The descriptor of one packet.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "the descriptor's one-bit fields (RFC 9628 §4.2)"
)]
pub(crate) struct Descriptor {
    pub picture_id: Option<u16>,
    pub inter_picture: bool,
    /// `(tid, switch_up, sid, inter_layer)`.
    pub layer: Option<(u8, bool, u8, bool)>,
    pub flexible: bool,
    pub start: bool,
    pub end: bool,
    pub not_upper_reference: bool,
    pub tl0_picture_index: Option<u8>,
    pub p_diffs: Vec<u8>,
    /// Picture size per spatial layer, when the SS is sent.
    pub scalability: Option<Vec<(u16, u16)>>,
}

impl Descriptor {
    pub fn write(&self, out: &mut Vec<u8>) {
        let mut first = 0;
        for (on, bit) in [
            (self.picture_id.is_some(), I_PICTURE_ID),
            (self.inter_picture, P_INTER_PICTURE),
            (self.layer.is_some(), L_LAYER_INDICES),
            (self.flexible, F_FLEXIBLE),
            (self.start, B_START),
            (self.end, E_END),
            (self.scalability.is_some(), V_SCALABILITY),
            (self.not_upper_reference, Z_NOT_UPPER_REFERENCE),
        ] {
            if on {
                first |= bit;
            }
        }
        out.push(first);
        if let Some(id) = self.picture_id {
            out.push(M_EXTENDED | (id >> 8) as u8);
            out.push(id as u8);
        }
        if let Some((tid, switch_up, sid, inter_layer)) = self.layer {
            out.push(
                (tid << TID_SHIFT)
                    | if switch_up { U_SWITCH_UP } else { 0 }
                    | ((sid & SID_MASK) << SID_SHIFT)
                    | u8::from(inter_layer),
            );
            if !self.flexible {
                out.push(self.tl0_picture_index.unwrap_or(0));
            }
        }
        if self.flexible && self.inter_picture {
            for (index, &diff) in self.p_diffs.iter().enumerate() {
                let more = index + 1 < self.p_diffs.len();
                out.push((diff << 1) | u8::from(more));
            }
        }
        if let Some(sizes) = &self.scalability {
            out.push((((sizes.len() - 1) as u8) << SS_COUNT_SHIFT) | SS_Y_SIZES);
            for &(width, height) in sizes {
                out.extend_from_slice(&width.to_be_bytes());
                out.extend_from_slice(&height.to_be_bytes());
            }
        }
    }

    /// Parse a descriptor; returns it and its length.
    pub fn parse(data: &[u8]) -> Result<(Self, usize)> {
        let truncated = RtpError::Truncated("VP9 payload descriptor");
        let mut at = 0;
        let mut next = || -> Result<u8> {
            let byte = *data.get(at).ok_or(truncated.clone())?;
            at += 1;
            Ok(byte)
        };
        let first = next()?;
        let flexible = first & F_FLEXIBLE != 0;
        let mut d = Self {
            inter_picture: first & P_INTER_PICTURE != 0,
            flexible,
            start: first & B_START != 0,
            end: first & E_END != 0,
            not_upper_reference: first & Z_NOT_UPPER_REFERENCE != 0,
            ..Self::default()
        };
        if first & I_PICTURE_ID != 0 {
            let high = next()?;
            d.picture_id = Some(if high & M_EXTENDED != 0 {
                (u16::from(high & !M_EXTENDED) << 8) | u16::from(next()?)
            } else {
                u16::from(high & SHORT_PICTURE_ID_MASK)
            });
        }
        if first & L_LAYER_INDICES != 0 {
            let byte = next()?;
            d.layer = Some((
                byte >> TID_SHIFT,
                byte & U_SWITCH_UP != 0,
                (byte >> SID_SHIFT) & SID_MASK,
                byte & D_INTER_LAYER != 0,
            ));
            if !flexible {
                d.tl0_picture_index = Some(next()?);
            }
        }
        if flexible && d.inter_picture {
            loop {
                let byte = next()?;
                d.p_diffs.push(byte >> 1);
                if byte & N_MORE_DIFFS == 0 {
                    break;
                }
                if d.p_diffs.len() == MAX_P_DIFFS {
                    return Err(RtpError::Malformed("more than three VP9 P_DIFFs"));
                }
            }
        }
        if first & V_SCALABILITY != 0 {
            let header = next()?;
            let layers = usize::from(header >> SS_COUNT_SHIFT) + 1;
            let mut sizes = Vec::new();
            if header & SS_Y_SIZES != 0 {
                for _ in 0..layers {
                    let width = u16::from_be_bytes([next()?, next()?]);
                    let height = u16::from_be_bytes([next()?, next()?]);
                    sizes.push((width, height));
                }
            }
            if header & SS_G_GROUP != 0 {
                for _ in 0..next()? {
                    let entry = next()?;
                    for _ in 0..(entry >> PG_R_SHIFT) & PG_R_MASK {
                        next()?;
                    }
                }
            }
            d.scalability = Some(sizes);
        }
        Ok((d, at))
    }
}

/// True for a VP9 frame whose uncompressed header says `KEY_FRAME`.
pub(crate) fn is_key_frame(frame: &[u8]) -> bool {
    let Some(&byte) = frame.first() else {
        return false;
    };
    if byte >> 6 != FRAME_MARKER {
        return false;
    }
    let profile = ((byte >> 5) & 1) | (((byte >> 4) & 1) << 1);
    // Bits after the profile: [reserved_zero], show_existing_frame, frame_type.
    let shift = if profile == PROFILE_3 { 2 } else { 3 };
    let show_existing = (byte >> shift) & 1 != 0;
    let frame_type = (byte >> (shift - 1)) & 1;
    !show_existing && frame_type == 0
}

/// One layer frame to send, and its descriptor fields.
#[derive(Debug, Clone)]
pub(crate) struct FrameToSend<'a> {
    pub data: &'a [u8],
    pub index: usize,
    pub descriptor: Descriptor,
}

/// Fragment each frame into payloads of at most `budget` bytes; the SS
/// (when present) goes on the frame's first packet only.
pub(crate) fn packetize(frames: &[FrameToSend<'_>], budget: usize) -> Result<Vec<OutPayload>> {
    let mut payloads = Vec::new();
    for frame in frames {
        let mut probe = Vec::new();
        frame.descriptor.write(&mut probe);
        let capacity = budget
            .checked_sub(probe.len())
            .filter(|&c| c > 0)
            .ok_or_else(|| {
                RtpError::Config(format!("payload budget {budget} too small for VP9"))
            })?;
        if frame.data.is_empty() {
            return Err(RtpError::Malformed("empty VP9 frame"));
        }
        let count = frame.data.len().div_ceil(capacity);
        let size = frame.data.len().div_ceil(count);
        for (index, chunk) in frame.data.chunks(size).enumerate() {
            let mut descriptor = frame.descriptor.clone();
            descriptor.start = index == 0;
            descriptor.end = index == count - 1;
            if index > 0 {
                descriptor.scalability = None;
            }
            let mut bytes = Vec::with_capacity(probe.len() + chunk.len());
            descriptor.write(&mut bytes);
            bytes.extend_from_slice(chunk);
            payloads.push(OutPayload {
                bytes,
                frame: frame.index,
                start_of_frame: descriptor.start,
                end_of_frame: descriptor.end,
            });
        }
    }
    Ok(payloads)
}

/// Parse one payload: one piece of a VP9 frame. `unwrap_picture_id` turns
/// a picture ID into a stream-wide picture number.
pub(crate) fn depacketize(
    payload: &[u8],
    unwrap_picture_id: impl FnOnce(u16) -> u64,
) -> Result<ParsedPayload> {
    let (d, length) = Descriptor::parse(payload)?;
    let body = &payload[length..];
    if body.is_empty() {
        return Err(RtpError::Truncated("VP9 payload"));
    }
    let (tid, sid, inter_layer) = d.layer.map_or((0, 0, false), |(t, _, s, dl)| (t, s, dl));
    // The packet starting a base-layer key frame marks the picture as a key.
    let key = d.start && !d.inter_picture && sid == 0 && is_key_frame(body);
    let frame = d.picture_id.map(|id| {
        let picture = unwrap_picture_id(id);
        let mut references = None;
        if d.flexible {
            let mut refs: Vec<u64> = d
                .p_diffs
                .iter()
                .filter_map(|&diff| picture.checked_sub(u64::from(diff)))
                .map(|p| p * SPATIAL_SLOTS + u64::from(sid))
                .collect();
            if inter_layer && sid > 0 {
                refs.push(picture * SPATIAL_SLOTS + u64::from(sid - 1));
            }
            references = Some(refs);
        }
        FrameTag {
            id: picture * SPATIAL_SLOTS + u64::from(sid),
            start: d.start,
            end: d.end,
            references,
            spatial_id: sid,
            temporal_id: tid,
        }
    });
    Ok(ParsedPayload {
        pieces: vec![Piece {
            data: body.to_vec(),
            continues_previous: !d.start,
            continues_next: !d.end,
        }],
        info: PacketInfo {
            key,
            frame,
            ..PacketInfo::default()
        },
    })
}

/// `P_DIFF`s from a picture to the pictures it references (1-127 each, at
/// most three; farther ones cannot be expressed and are left out).
pub(crate) fn picture_diffs(picture: u64, referenced: &[u64]) -> Vec<u8> {
    let mut diffs: Vec<u8> = referenced
        .iter()
        .filter_map(|&r| picture.checked_sub(r))
        .filter(|&d| (1..=MAX_P_DIFF).contains(&d))
        .map(|d| d as u8)
        .collect();
    diffs.sort_unstable();
    diffs.dedup();
    diffs.truncate(MAX_P_DIFFS);
    diffs
}

/// The 15-bit picture ID of a picture number.
pub(crate) fn picture_id(picture: u64) -> u16 {
    (picture & u64::from(PICTURE_ID_MASK)) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptors_round_trip() {
        let flexible = Descriptor {
            picture_id: Some(0x1234),
            inter_picture: true,
            layer: Some((2, false, 1, true)),
            flexible: true,
            start: true,
            end: false,
            not_upper_reference: false,
            tl0_picture_index: None,
            p_diffs: vec![1, 4],
            scalability: None,
        };
        let key = Descriptor {
            picture_id: Some(7),
            inter_picture: false,
            layer: Some((0, true, 0, false)),
            flexible: false,
            start: true,
            end: true,
            tl0_picture_index: Some(200),
            scalability: Some(vec![(320, 180), (640, 360)]),
            ..Descriptor::default()
        };
        for d in [flexible, key] {
            let mut bytes = Vec::new();
            d.write(&mut bytes);
            let (parsed, length) = Descriptor::parse(&bytes).unwrap();
            assert_eq!(parsed, d);
            assert_eq!(length, bytes.len());
        }
    }

    #[test]
    fn frames_fragment_with_b_and_e() {
        let data: Vec<u8> = (0..3000).map(|i| (i % 255) as u8).collect();
        let frame = FrameToSend {
            data: &data,
            index: 0,
            descriptor: Descriptor {
                picture_id: Some(1),
                flexible: true,
                layer: Some((0, false, 0, false)),
                scalability: Some(vec![(640, 360)]),
                ..Descriptor::default()
            },
        };
        let payloads = packetize(&[frame], 1200).unwrap();
        assert_eq!(payloads.len(), 3);
        let mut joined = Vec::new();
        for (index, p) in payloads.iter().enumerate() {
            assert!(p.bytes.len() <= 1200);
            let parsed = depacketize(&p.bytes, u64::from).unwrap();
            let piece = &parsed.pieces[0];
            assert_eq!(piece.continues_previous, index > 0);
            assert_eq!(piece.continues_next, index < 2);
            joined.extend_from_slice(&piece.data);
        }
        assert_eq!(joined, data);
    }

    #[test]
    fn key_frames_are_recognised() {
        // Profile 0: 10 0 0 | show_existing 0 | frame_type 0.
        assert!(is_key_frame(&[0b1000_0000]));
        assert!(!is_key_frame(&[0b1000_0100]), "inter frame");
        assert!(!is_key_frame(&[0b1000_1000]), "show_existing_frame");
    }

    #[test]
    fn picture_diffs_drop_unreachable_references() {
        assert_eq!(picture_diffs(500, &[499, 496, 200]), vec![1, 4]);
    }
}
