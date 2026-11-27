//! What a scalable stream's layer frames mean to a forwarding server:
//! decode targets, DTIs, frame diffs and chains for the Dependency
//! Descriptor, derived from the encoder's layer structure and per-frame
//! layer information.
//!
//! The encoder describes each coded layer picture with a [`LayerFrame`]
//! (its layer ids, frame number, the frames it predicts from) and the
//! stream with a [`LayerStructure`]. From those, [`DescriptorWriter`] makes
//! one decode target per (spatial, temporal) pair (`dt = s * T + t`), one
//! chain per spatial layer, one template per layer, and per frame the exact
//! DTIs, frame diffs and chain diffs, overriding the templates where the
//! frame differs (key frames, long-term references, decimated or repaired
//! enhancement layers). The template structure is resent, with a new id
//! offset, on every key unit or layer size change.

use crate::dependency_descriptor::{
    DependencyDescriptor, Dti, FrameTemplate, MAX_DECODE_TARGETS, MAX_FRAME_FDIFF,
    TemplateStructure,
};
use crate::error::{Result, RtpError};

/// Frame numbers are 16-bit on the wire.
const FRAME_NUMBER_MASK: u64 = 0xFFFF;
/// Custom chain diffs are 8-bit.
const MAX_CHAIN_FDIFF: u64 = 255;
/// Template ids are 6-bit.
const TEMPLATE_ID_RANGE: usize = 64;
/// Most spatial and temporal layers a structure describes.
pub const MAX_SPATIAL_LAYERS: usize = 4;
/// Most temporal layers (`temporal_id` is 3 bits, our encoder uses 1-3).
pub const MAX_TEMPORAL_LAYERS: usize = 8;

/// How enhancement layers predict from the layer below.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InterLayer {
    /// On every temporal unit (L modes).
    #[default]
    Full,
    /// On key units only (K-SVC, `L*T*_KEY`).
    KeyOnly,
    /// Never: independent layers in one stream (S modes).
    Off,
}

/// The layer structure of a stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerStructure {
    /// Picture size per spatial layer, lowest first (one entry for a
    /// single-layer stream).
    pub spatial_sizes: Vec<(u16, u16)>,
    /// Number of temporal layers (1 for none).
    pub temporal_layers: u8,
    /// Inter-layer prediction mode.
    pub inter_layer: InterLayer,
}

impl LayerStructure {
    /// A single-layer stream of `width` x `height` with `temporal_layers`.
    #[must_use]
    pub fn single(width: u16, height: u16, temporal_layers: u8) -> Self {
        Self {
            spatial_sizes: vec![(width, height)],
            temporal_layers,
            inter_layer: InterLayer::Full,
        }
    }

    fn spatial_layers(&self) -> usize {
        self.spatial_sizes.len()
    }

    fn temporal(&self) -> usize {
        usize::from(self.temporal_layers.max(1))
    }

    /// Number of decode targets.
    #[must_use]
    pub fn decode_targets(&self) -> usize {
        self.spatial_layers() * self.temporal()
    }

    /// The decode target of spatial layer `s` up to temporal layer `t`.
    #[must_use]
    pub fn decode_target(&self, spatial: u8, temporal: u8) -> usize {
        usize::from(spatial) * self.temporal() + usize::from(temporal)
    }

    /// Check the structure fits the descriptor.
    ///
    /// # Errors
    /// No layers, too many, or a zero size.
    pub fn validate(&self) -> Result<()> {
        if self.spatial_sizes.is_empty()
            || self.spatial_layers() > MAX_SPATIAL_LAYERS
            || self.temporal() > MAX_TEMPORAL_LAYERS
            || self.decode_targets() > MAX_DECODE_TARGETS
            || self.spatial_sizes.iter().any(|&(w, h)| w == 0 || h == 0)
        {
            return Err(RtpError::Config(format!("layer structure {self:?}")));
        }
        Ok(())
    }
}

/// One coded layer picture of a temporal unit, as the encoder describes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerFrame {
    /// Spatial layer (0 = base).
    pub spatial_id: u8,
    /// Temporal layer (0 = base).
    pub temporal_id: u8,
    /// The encoder's index of this picture, +1 per coded layer picture in
    /// decode order.
    pub frame_number: u64,
    /// Frame numbers of the earlier pictures it predicts from (the
    /// same-unit lower layer included or not: see `inter_layer`).
    pub references: Vec<u64>,
    /// No inter-picture prediction (a key or intra-only picture).
    pub key: bool,
    /// Predicts from the next lower spatial layer of the same unit.
    pub inter_layer: bool,
    /// Byte range of this picture in the unit, for formats that send layer
    /// frames separately (VP9); `None` for the whole unit.
    pub range: Option<std::ops::Range<usize>>,
}

/// True when frame `frame` of this unit belongs to decode targets of
/// spatial layer `target` (`target >= frame.spatial_id`).
fn needed_by_layer(
    structure: &LayerStructure,
    unit: &[LayerFrame],
    frame: &LayerFrame,
    target: u8,
) -> bool {
    if target == frame.spatial_id {
        return true;
    }
    match structure.inter_layer {
        InterLayer::Full => true,
        InterLayer::Off => false,
        // Each layer up to the target must predict from the one below in
        // this unit, or this frame is not on its path.
        InterLayer::KeyOnly => (frame.spatial_id + 1..=target)
            .all(|layer| unit.iter().any(|f| f.spatial_id == layer && f.inter_layer)),
    }
}

/// The DTI of `frame` for every decode target.
fn frame_dtis(structure: &LayerStructure, unit: &[LayerFrame], frame: &LayerFrame) -> Vec<Dti> {
    let top_temporal =
        structure.temporal() > 1 && usize::from(frame.temporal_id) == structure.temporal() - 1;
    let mut dtis = Vec::with_capacity(structure.decode_targets());
    for s in 0..structure.spatial_layers() as u8 {
        for t in 0..structure.temporal() as u8 {
            let dti = if frame.spatial_id > s
                || frame.temporal_id > t
                || !needed_by_layer(structure, unit, frame, s)
            {
                Dti::NotPresent
            } else if frame.key {
                Dti::Switch
            } else if top_temporal && frame.spatial_id == s {
                // Our encoder never references top-temporal-layer frames.
                Dti::Discardable
            } else {
                Dti::Required
            };
            dtis.push(dti);
        }
    }
    dtis
}

/// The template structure of `structure` with id offset `offset`.
fn template_structure(structure: &LayerStructure, offset: u8) -> TemplateStructure {
    let spatial = structure.spatial_layers();
    let temporal = structure.temporal();
    // Template values for the commonest frame of each layer: a non-key
    // frame, inter-layer per the mode. A single-layer, single-temporal
    // stream's frames reference the previous frame, which is also the
    // previous frame of its chain: 3-byte descriptors.
    let simple = spatial == 1 && temporal == 1;
    let mut templates = Vec::with_capacity(spatial * temporal);
    for s in 0..spatial as u8 {
        for t in 0..temporal as u8 {
            let typical = LayerFrame {
                spatial_id: s,
                temporal_id: t,
                frame_number: 0,
                references: Vec::new(),
                key: false,
                inter_layer: s > 0 && structure.inter_layer == InterLayer::Full,
                range: None,
            };
            let unit: Vec<LayerFrame> = (s..spatial as u8)
                .map(|layer| LayerFrame {
                    spatial_id: layer,
                    ..typical.clone()
                })
                .collect();
            templates.push(FrameTemplate {
                spatial_id: s,
                temporal_id: t,
                dtis: frame_dtis(structure, &unit, &typical),
                fdiffs: if simple { vec![1] } else { Vec::new() },
                chain_fdiffs: vec![u8::from(simple); spatial],
            });
        }
    }
    TemplateStructure {
        template_id_offset: offset,
        decode_targets: structure.decode_targets(),
        templates,
        chains: spatial,
        decode_target_protected_by: (0..spatial as u8)
            .flat_map(|s| std::iter::repeat_n(s, temporal))
            .collect(),
        resolutions: structure.spatial_sizes.clone(),
    }
}

/// Writes the Dependency Descriptor of every layer frame of a stream.
#[derive(Debug, Clone, Default)]
pub struct DescriptorWriter {
    layers: Option<LayerStructure>,
    structure: Option<TemplateStructure>,
    /// Frame number of the last frame of each chain.
    chain_last: Vec<Option<u64>>,
}

/// The descriptors of one layer frame: one for its first packet (which may
/// carry the template structure) and one for the rest; the caller sets the
/// start and end flags per packet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameDescriptors {
    /// For the frame's first packet.
    pub first: DependencyDescriptor,
    /// For its other packets.
    pub rest: DependencyDescriptor,
}

impl DescriptorWriter {
    /// A writer that has sent nothing yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The structure in effect (after the last [`Self::describe_unit`]).
    #[must_use]
    pub fn structure(&self) -> Option<&TemplateStructure> {
        self.structure.as_ref()
    }

    /// Describe every layer frame of one temporal unit, in order.
    ///
    /// A key unit, or a change of `layers`, starts a new structure (sent on
    /// the unit's first packet) and restarts the chains.
    ///
    /// # Errors
    /// An invalid structure, a frame outside it, or a first unit that is
    /// not a key unit.
    pub fn describe_unit(
        &mut self,
        layers: &LayerStructure,
        unit: &[LayerFrame],
    ) -> Result<Vec<FrameDescriptors>> {
        layers.validate()?;
        let key_unit = unit.first().is_some_and(|f| f.key && f.spatial_id == 0);
        let changed = self.layers.as_ref() != Some(layers);
        let mut new_structure = None;
        if key_unit || changed {
            let offset = self.structure.as_ref().map_or(0, |s| {
                (usize::from(s.template_id_offset) + s.templates.len()) % TEMPLATE_ID_RANGE
            });
            let structure = template_structure(layers, offset as u8);
            structure.validate()?;
            new_structure = Some(structure.clone());
            self.structure = Some(structure);
            self.layers = Some(layers.clone());
            self.chain_last = vec![None; layers.spatial_layers()];
        }
        let structure = self.structure.clone().expect("set above or earlier");
        let mut out = Vec::with_capacity(unit.len());
        for (index, frame) in unit.iter().enumerate() {
            if usize::from(frame.spatial_id) >= layers.spatial_layers()
                || usize::from(frame.temporal_id) >= layers.temporal()
            {
                return Err(RtpError::Config(format!(
                    "layer frame S{}T{} outside the structure",
                    frame.spatial_id, frame.temporal_id
                )));
            }
            let template_index = layers.decode_target(frame.spatial_id, frame.temporal_id);
            let template = &structure.templates[template_index];
            let dtis = frame_dtis(layers, unit, frame);
            let fdiffs = Self::fdiffs(unit, frame);
            let chains = self.chain_fdiffs(layers, unit, frame);
            let descriptor = DependencyDescriptor {
                start_of_frame: false,
                end_of_frame: false,
                template_id: structure.template_id(template_index),
                frame_number: (frame.frame_number & FRAME_NUMBER_MASK) as u16,
                structure: None,
                active_decode_targets: None,
                custom_dtis: (dtis != template.dtis).then_some(dtis),
                custom_fdiffs: (fdiffs != template.fdiffs).then_some(fdiffs),
                custom_chains: (chains != template.chain_fdiffs).then_some(chains),
            };
            let mut first = descriptor.clone();
            if index == 0 {
                first.structure.clone_from(&new_structure);
            }
            out.push(FrameDescriptors {
                first,
                rest: descriptor,
            });
        }
        Ok(out)
    }

    /// Frame diffs to every expressible reference (inter-layer included).
    fn fdiffs(unit: &[LayerFrame], frame: &LayerFrame) -> Vec<u16> {
        let lower = frame
            .inter_layer
            .then(|| {
                unit.iter()
                    .find(|f| f.spatial_id + 1 == frame.spatial_id)
                    .map(|f| f.frame_number)
            })
            .flatten();
        let mut references: Vec<u64> = frame.references.iter().copied().chain(lower).collect();
        references.sort_unstable_by(|a, b| b.cmp(a));
        references.dedup();
        references
            .into_iter()
            .filter_map(|reference| {
                let diff = frame.frame_number.checked_sub(reference)?;
                // Older references (a long-term picture over 4096 frames
                // back) cannot be expressed; they are left out.
                (1..=u64::from(MAX_FRAME_FDIFF))
                    .contains(&diff)
                    .then_some(diff as u16)
            })
            .collect()
    }

    /// Chain diffs, and the chains' last frames advanced past `frame`.
    fn chain_fdiffs(
        &mut self,
        layers: &LayerStructure,
        unit: &[LayerFrame],
        frame: &LayerFrame,
    ) -> Vec<u8> {
        let mut diffs = Vec::with_capacity(self.chain_last.len());
        for (chain, last) in self.chain_last.iter_mut().enumerate() {
            let diff = last
                .and_then(|last| frame.frame_number.checked_sub(last))
                .filter(|&d| d <= MAX_CHAIN_FDIFF)
                .unwrap_or(0);
            diffs.push(diff as u8);
            let in_chain = frame.temporal_id == 0
                && usize::from(frame.spatial_id) <= chain
                && needed_by_layer(layers, unit, frame, chain as u8);
            if in_chain {
                *last = Some(frame.frame_number);
            }
        }
        diffs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dependency_descriptor::DependencyDescriptor;

    fn frame(s: u8, t: u8, number: u64, refs: &[u64], key: bool, inter: bool) -> LayerFrame {
        LayerFrame {
            spatial_id: s,
            temporal_id: t,
            frame_number: number,
            references: refs.to_vec(),
            key,
            inter_layer: inter,
            range: None,
        }
    }

    #[test]
    fn single_layer_p_frames_need_only_the_mandatory_bytes() {
        let layers = LayerStructure::single(640, 360, 1);
        let mut writer = DescriptorWriter::new();
        let key = writer
            .describe_unit(&layers, &[frame(0, 0, 0, &[], true, false)])
            .unwrap();
        let structure = key[0].first.structure.clone().unwrap();
        let p = writer
            .describe_unit(&layers, &[frame(0, 0, 1, &[0], false, false)])
            .unwrap();
        let bytes = p[0].first.write(&structure).unwrap();
        assert_eq!(bytes.len(), 3);
        let resolved = DependencyDescriptor::parse(&bytes, Some(&structure))
            .unwrap()
            .resolve(&structure)
            .unwrap();
        assert_eq!((resolved.fdiffs, resolved.chain_fdiffs), (vec![1], vec![1]));
    }

    #[test]
    fn l1t3_top_layer_is_discardable_and_off_the_chain() {
        let layers = LayerStructure::single(640, 360, 3);
        let mut w = DescriptorWriter::new();
        w.describe_unit(&layers, &[frame(0, 0, 0, &[], true, false)])
            .unwrap();
        let t2 = w
            .describe_unit(&layers, &[frame(0, 2, 1, &[0], false, false)])
            .unwrap();
        let s = w.structure().unwrap().clone();
        let d = t2[0].rest.resolve(&s).unwrap();
        assert_eq!(
            d.dtis,
            vec![Dti::NotPresent, Dti::NotPresent, Dti::Discardable]
        );
        assert_eq!(d.fdiffs, vec![1]);
        let t1 = w
            .describe_unit(&layers, &[frame(0, 1, 2, &[0], false, false)])
            .unwrap();
        let d = t1[0].rest.resolve(&s).unwrap();
        assert_eq!(d.chain_fdiffs, vec![2], "chain is the T0 frames");
        assert_eq!(d.dtis, vec![Dti::NotPresent, Dti::Required, Dti::Required]);
    }

    #[test]
    fn l2t1_enhancement_references_the_base_and_k_svc_drops_it() {
        for (mode, base_in_top) in [(InterLayer::Full, true), (InterLayer::KeyOnly, false)] {
            let layers = LayerStructure {
                spatial_sizes: vec![(320, 180), (640, 360)],
                temporal_layers: 1,
                inter_layer: mode,
            };
            let mut w = DescriptorWriter::new();
            let key = w
                .describe_unit(
                    &layers,
                    &[
                        frame(0, 0, 0, &[], true, false),
                        frame(1, 0, 1, &[], true, true),
                    ],
                )
                .unwrap();
            let s = w.structure().unwrap().clone();
            assert!(key[0].first.structure.is_some() && key[1].first.structure.is_none());
            let enh = key[1].rest.resolve(&s).unwrap();
            assert_eq!(enh.fdiffs, vec![1], "inter-layer reference");
            assert_eq!(enh.chain_fdiffs, vec![1, 1]);
            let inter = mode == InterLayer::Full;
            let p = w
                .describe_unit(
                    &layers,
                    &[
                        frame(0, 0, 2, &[0], false, false),
                        frame(1, 0, 3, &[1], false, inter),
                    ],
                )
                .unwrap();
            let base = p[0].rest.resolve(&s).unwrap();
            let present = if base_in_top {
                Dti::Required
            } else {
                Dti::NotPresent
            };
            assert_eq!(base.dtis, vec![Dti::Required, present]);
            let enh = p[1].rest.resolve(&s).unwrap();
            let expected = if inter { vec![1, 2] } else { vec![2] };
            assert_eq!(enh.fdiffs, expected);
            // Chain 1 last held frame 1 (K-SVC) or frame 2 (full SVC).
            assert_eq!(enh.chain_fdiffs[1], if inter { 1 } else { 2 });
        }
    }

    #[test]
    fn structures_get_fresh_offsets() {
        let layers = LayerStructure::single(64, 64, 2);
        let mut w = DescriptorWriter::new();
        w.describe_unit(&layers, &[frame(0, 0, 0, &[], true, false)])
            .unwrap();
        let first = w.structure().unwrap().template_id_offset;
        w.describe_unit(&layers, &[frame(0, 0, 9, &[], true, false)])
            .unwrap();
        assert_eq!(w.structure().unwrap().template_id_offset, first + 2);
    }
}
