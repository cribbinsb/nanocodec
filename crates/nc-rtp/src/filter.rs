//! Layer dropping as a selective forwarding server (SFU) does it, driven by
//! the Dependency Descriptor alone (AV1 RTP Appendix A.1): forward the
//! packets of frames that belong to one decode target, drop the rest,
//! renumber the sequence so the receiver sees no gap, and set the marker on
//! the last packet of the target layer's frame.
//!
//! This is the transport-side check of spatial and temporal scalability:
//! what it forwards must decode on its own. It expects packets in sending
//! order (an SFU's input after its own jitter handling).

use crate::dependency_descriptor::{DependencyDescriptor, Dti, TemplateStructure};
use crate::error::{Result, RtpError};
use crate::packet::RtpPacket;

/// Byte offset of the sequence number and the marker byte in the RTP
/// fixed header (RFC 3550 §5.1).
const SEQUENCE_OFFSET: usize = 2;
const MARKER_BYTE: usize = 1;
const MARKER_BIT: u8 = 0x80;

/// Forwards the packets of one decode target.
#[derive(Debug, Clone)]
pub struct LayerFilter {
    descriptor_id: u8,
    /// Highest spatial and temporal layer to forward.
    target: (u8, u8),
    structure: Option<TemplateStructure>,
    /// Packets dropped so far: forwarded sequence numbers move down by it.
    dropped: u16,
}

impl LayerFilter {
    /// Forward up to spatial layer `spatial` and temporal layer `temporal`,
    /// reading the descriptor at extension id `descriptor_id`.
    #[must_use]
    pub fn new(descriptor_id: u8, spatial: u8, temporal: u8) -> Self {
        Self {
            descriptor_id,
            target: (spatial, temporal),
            structure: None,
            dropped: 0,
        }
    }

    /// The decode target that best matches the wanted layers: the highest
    /// one not above them.
    fn decode_target(&self, structure: &TemplateStructure) -> Option<usize> {
        structure
            .decode_target_layers()
            .iter()
            .enumerate()
            .filter(|(_, (s, t))| *s <= self.target.0 && *t <= self.target.1)
            .max_by_key(|&(_, layers)| *layers)
            .map(|(index, _)| index)
    }

    /// The forwarded packet, rewritten, or `None` to drop it.
    ///
    /// # Errors
    /// A packet that does not parse, or a descriptor before any structure.
    pub fn filter(&mut self, datagram: &[u8]) -> Result<Option<Vec<u8>>> {
        let packet = RtpPacket::parse(datagram)?;
        let Some(data) = packet.extension(self.descriptor_id) else {
            return Ok(Some(self.renumber(datagram, packet.header.marker)));
        };
        let descriptor = DependencyDescriptor::parse(data, self.structure.as_ref())?;
        if let Some(structure) = &descriptor.structure {
            self.structure = Some(structure.clone());
        }
        let structure = self
            .structure
            .as_ref()
            .ok_or(RtpError::Malformed("descriptor before any structure"))?;
        let frame = descriptor.resolve(structure)?;
        let target = self.decode_target(structure).ok_or(RtpError::Config(
            "no decode target within the wanted layers".into(),
        ))?;
        // The structure must reach every receiver, whatever its layer.
        let forward = frame.dtis[target] != Dti::NotPresent || descriptor.structure.is_some();
        if !forward {
            self.dropped = self.dropped.wrapping_add(1);
            return Ok(None);
        }
        let (target_spatial, _) = structure.decode_target_layers()[target];
        let marker =
            packet.header.marker || (descriptor.end_of_frame && frame.spatial_id == target_spatial);
        Ok(Some(self.renumber(datagram, marker)))
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
