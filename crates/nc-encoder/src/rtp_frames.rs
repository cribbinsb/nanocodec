//! The encoder's description of its layers for RTP transport
//! (`nanocodec-rtp`): the stream's layer structure and, per coded frame,
//! each layer picture's ids, frame number and references, from which the
//! packetizer writes the AV1 Dependency Descriptor and the VP9 payload
//! descriptor (docs/handoff/rtp.md). The one mapping every front end (C
//! API, CLI, browser module) uses.

use nc_rtp::{InterLayer, LayerFrame, LayerStructure};

use crate::backend::{CodecBackend, FrameType};
use crate::encoder::{EncodedFrame, Encoder};
use crate::spatial::InterLayerMode;

/// Picture sizes on the wire are 16-bit (Dependency Descriptor render
/// sizes, VP9 SS); our encoders' sizes are far below this.
fn size16(width: usize, height: usize) -> (u16, u16) {
    (
        u16::try_from(width).unwrap_or(u16::MAX),
        u16::try_from(height).unwrap_or(u16::MAX),
    )
}

/// The RTP layer structure of `encoder`'s stream as it is now.
#[must_use]
pub fn rtp_layer_structure<B: CodecBackend>(encoder: &Encoder<B>) -> LayerStructure {
    if let Some(spatial) = encoder.spatial_config() {
        LayerStructure {
            spatial_sizes: spatial
                .layer_sizes
                .iter()
                .map(|&(w, h)| size16(w, h))
                .collect(),
            temporal_layers: spatial.temporal_layers,
            inter_layer: match spatial.inter_layer {
                InterLayerMode::On => InterLayer::Full,
                InterLayerMode::KeyOnly => InterLayer::KeyOnly,
                InterLayerMode::Off => InterLayer::Off,
            },
        }
    } else {
        let config = encoder.config();
        let (width, height) = size16(config.width, config.height);
        LayerStructure::single(width, height, config.temporal_layers)
    }
}

/// The layer pictures of one coded frame, in decode order (empty when it
/// was dropped or held: nothing is sent).
#[must_use]
pub fn rtp_layer_frames(frame: &EncodedFrame) -> Vec<LayerFrame> {
    if frame.data.is_empty() {
        return Vec::new();
    }
    if frame.layers.is_empty() {
        let metadata = &frame.metadata;
        return vec![LayerFrame {
            spatial_id: 0,
            temporal_id: metadata.temporal_layer,
            frame_number: metadata.index,
            references: metadata.references.iter().collect(),
            key: metadata.frame_type == FrameType::Key,
            inter_layer: false,
            range: None,
        }];
    }
    frame
        .layers
        .iter()
        .map(|layer| LayerFrame {
            spatial_id: layer.spatial_id,
            temporal_id: layer.temporal_layer,
            frame_number: layer.picture_index,
            references: layer.references.iter().collect(),
            // A key picture, or an enhancement INTRA_ONLY picture: no
            // inter-picture prediction either way.
            key: layer.frame_type == FrameType::Key,
            inter_layer: layer.inter_layer_ref,
            range: Some(layer.byte_offset..layer.byte_offset + layer.byte_len),
        })
        .collect()
}
