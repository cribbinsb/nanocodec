//! Layer dropping as a selective forwarding server (SFU) does it: forward
//! the packets of the frames one receiver's layers need, drop the rest,
//! renumber the sequence so the receiver sees no gap, and set the marker on
//! the last forwarded packet of each picture.
//!
//! Two sources say which frames those are:
//! - the Dependency Descriptor alone (AV1 RTP Appendix A.1, any format):
//!   the frames present in the decode target;
//! - the VP9 payload descriptor (RFC 9628 §4.2): the frames whose layer ids
//!   are at most the wanted ones, less the lower spatial layers the wanted
//!   layer does not predict from. Those the descriptor does not say ahead
//!   of the wanted layer's frame, so the SFU knows them from the stream's
//!   inter-layer mode, as from the scalability mode WebRTC negotiates
//!   (K-SVC: lower layers on key pictures only).
//!
//! The marker goes on the end of the wanted layer's frame (or stays where
//! the sender put it, when the picture has no frame of that layer).
//!
//! This is the transport-side check of spatial and temporal scalability:
//! what it forwards must decode on its own. It expects packets in sending
//! order (an SFU's input after its own jitter handling).

use crate::dependency_descriptor::{DependencyDescriptor, Dti, TemplateStructure};
use crate::error::{Result, RtpError};
use crate::layering::InterLayer;
use crate::packet::RtpPacket;
use crate::vp9;

/// Byte offset of the sequence number and the marker byte in the RTP
/// fixed header (RFC 3550 §5.1).
const SEQUENCE_OFFSET: usize = 2;
const MARKER_BYTE: usize = 1;
const MARKER_BIT: u8 = 0x80;

/// Where the filter reads each packet's layers from.
#[derive(Debug, Clone)]
enum Selector {
    /// The Dependency Descriptor at this extension id, and the last
    /// template structure it sent.
    Descriptor {
        id: u8,
        structure: Option<TemplateStructure>,
    },
    /// The VP9 payload descriptor; `key_picture` is the picture ID of the
    /// last key picture seen (its base frame has `P = 0`).
    Vp9 {
        inter_layer: InterLayer,
        key_picture: Option<u16>,
    },
}

/// Forwards the packets of one receiver's layers.
#[derive(Debug, Clone)]
pub struct LayerFilter {
    selector: Selector,
    /// Highest spatial and temporal layer to forward.
    target: (u8, u8),
    /// Packets dropped so far: forwarded sequence numbers move down by it.
    dropped: u16,
}

impl LayerFilter {
    /// Forward up to spatial layer `spatial` and temporal layer `temporal`,
    /// reading the Dependency Descriptor at extension id `descriptor_id`.
    #[must_use]
    pub fn new(descriptor_id: u8, spatial: u8, temporal: u8) -> Self {
        Self {
            selector: Selector::Descriptor {
                id: descriptor_id,
                structure: None,
            },
            target: (spatial, temporal),
            dropped: 0,
        }
    }

    /// Forward up to spatial layer `spatial` and temporal layer `temporal`
    /// of a VP9 stream whose layers predict from the layer below as
    /// `inter_layer` says, reading the VP9 payload descriptor.
    #[must_use]
    pub fn vp9(inter_layer: InterLayer, spatial: u8, temporal: u8) -> Self {
        Self {
            selector: Selector::Vp9 {
                inter_layer,
                key_picture: None,
            },
            target: (spatial, temporal),
            dropped: 0,
        }
    }

    /// The forwarded packet, rewritten, or `None` to drop it.
    ///
    /// # Errors
    /// A packet that does not parse, or a descriptor before any structure.
    pub fn filter(&mut self, datagram: &[u8]) -> Result<Option<Vec<u8>>> {
        let packet = RtpPacket::parse(datagram)?;
        let decision = match &mut self.selector {
            Selector::Descriptor { id, structure } => {
                descriptor_decision(&packet, *id, structure, self.target)?
            }
            Selector::Vp9 {
                inter_layer,
                key_picture,
            } => vp9_decision(&packet, *inter_layer, key_picture, self.target)?,
        };
        if let Some(end_of_target) = decision {
            return Ok(Some(
                self.renumber(datagram, packet.header.marker || end_of_target),
            ));
        }
        self.dropped = self.dropped.wrapping_add(1);
        Ok(None)
    }

    fn renumber(&self, datagram: &[u8], marker: bool) -> Vec<u8> {
        let mut out = datagram.to_vec();
        let sequence = u16::from_be_bytes([out[SEQUENCE_OFFSET], out[SEQUENCE_OFFSET + 1]])
            .wrapping_sub(self.dropped);
        out[SEQUENCE_OFFSET..SEQUENCE_OFFSET + 2].copy_from_slice(&sequence.to_be_bytes());
        if marker {
            out[MARKER_BYTE] |= MARKER_BIT;
        } else {
            out[MARKER_BYTE] &= !MARKER_BIT;
        }
        out
    }
}

/// Whether to forward a packet by its Dependency Descriptor: `None` to drop
/// it, else whether it ends the target layer's frame (the forwarded
/// picture's last packet). A packet without the descriptor is forwarded.
fn descriptor_decision(
    packet: &RtpPacket<'_>,
    id: u8,
    structure: &mut Option<TemplateStructure>,
    (spatial, temporal): (u8, u8),
) -> Result<Option<bool>> {
    let Some(data) = packet.extension(id) else {
        return Ok(Some(false));
    };
    let descriptor = DependencyDescriptor::parse(data, structure.as_ref())?;
    if let Some(sent) = &descriptor.structure {
        *structure = Some(sent.clone());
    }
    let structure = structure
        .as_ref()
        .ok_or(RtpError::Malformed("descriptor before any structure"))?;
    let frame = descriptor.resolve(structure)?;
    // The decode target that best matches the wanted layers: the highest
    // one not above them.
    let target = structure
        .decode_target_layers()
        .iter()
        .enumerate()
        .filter(|(_, (s, t))| *s <= spatial && *t <= temporal)
        .max_by_key(|&(_, layers)| *layers)
        .map(|(index, _)| index)
        .ok_or(RtpError::Config(
            "no decode target within the wanted layers".into(),
        ))?;
    // The structure must reach every receiver, whatever its layer.
    if frame.dtis[target] == Dti::NotPresent && descriptor.structure.is_none() {
        return Ok(None);
    }
    let (target_spatial, _) = structure.decode_target_layers()[target];
    Ok(Some(
        descriptor.end_of_frame && frame.spatial_id == target_spatial,
    ))
}

/// Whether to forward a packet by its VP9 payload descriptor (RFC 9628
/// §4.2), as [`descriptor_decision`]. A lower spatial layer's frame goes
/// to the wanted layer's receiver when the wanted layer may predict from
/// it (always with `Full` inter-layer prediction) and on key pictures (the
/// base key frame starts every layer's stream).
fn vp9_decision(
    packet: &RtpPacket<'_>,
    inter_layer: InterLayer,
    key_picture: &mut Option<u16>,
    (spatial, temporal): (u8, u8),
) -> Result<Option<bool>> {
    let (descriptor, _) = vp9::Descriptor::parse(packet.payload)?;
    let (tid, _, sid, _) = descriptor.layer.unwrap_or_default();
    if sid == 0 && descriptor.start && !descriptor.inter_picture {
        *key_picture = descriptor.picture_id;
    }
    let key = descriptor.picture_id.is_some() && *key_picture == descriptor.picture_id;
    let lower_needed = inter_layer == InterLayer::Full || key;
    let forward = tid <= temporal && (sid == spatial || (sid < spatial && lower_needed));
    Ok(forward.then_some(descriptor.end && sid == spatial))
}
