//! The Dependency Descriptor RTP header extension (AV1 RTP payload format,
//! Appendix A): written and parsed bit for bit as §A.8.2 specifies.
//!
//! The descriptor tells a forwarding server (SFU) and a receiver, per
//! packet, which layer frame the packet belongs to, which frames that frame
//! needs (frame diffs), which decode targets (operating points) it belongs
//! to (DTIs) and the last frame of each chain, without parsing the media.
//! A *template structure*, sent on the first packet of each coded video
//! sequence, predefines per-layer values; a packet names a template and may
//! override its DTIs, frame diffs and chain diffs ("custom" fields).
//!
//! This module is the syntax only. What the fields should say for our
//! encoder's layer structures is `layering`'s job.

use nc_common::BitWriter;
use nc_common::decode::BitReader;

use crate::error::{Result, RtpError};

/// The extension's URI in SDP `a=extmap` (§A.9).
pub const URI: &str =
    "https://aomediacodec.github.io/av1-rtp-spec/#dependency-descriptor-rtp-header-extension";
/// Template ids and offsets are 6-bit (§A.8.2).
const TEMPLATE_ID_RANGE: u8 = 64;
/// Bytes of the mandatory fields; a longer descriptor has extended fields.
const MANDATORY_BYTES: usize = 3;
/// `dt_cnt_minus_one` is 5 bits.
pub const MAX_DECODE_TARGETS: usize = 32;
/// `template_fdiff`s are 4-bit `fdiff_minus_one`.
const MAX_TEMPLATE_FDIFF: u16 = 16;
/// `frame_fdiff`s are up to three nibbles of `fdiff_minus_one`.
pub const MAX_FRAME_FDIFF: u16 = 1 << 12;
/// Template chain fdiffs are 4 bits, frame chain fdiffs 8 bits.
const MAX_TEMPLATE_CHAIN_FDIFF: u8 = 15;
/// `next_layer_idc` values (Table A.2).
const NEXT_SAME_LAYER: u32 = 0;
const NEXT_TEMPORAL: u32 = 1;
const NEXT_SPATIAL: u32 = 2;
const NO_MORE_TEMPLATES: u32 = 3;

/// A Decode Target Indication (Table A.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Dti {
    /// No payload for this decode target.
    NotPresent = 0,
    /// Present; no later frame of the target references it.
    Discardable = 1,
    /// Present; the target can be switched to here.
    Switch = 2,
    /// Present and needed.
    Required = 3,
}

impl Dti {
    const fn from_bits(bits: u32) -> Self {
        match bits {
            0 => Self::NotPresent,
            1 => Self::Discardable,
            2 => Self::Switch,
            _ => Self::Required,
        }
    }
}

/// One frame dependency template.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameTemplate {
    /// Spatial layer.
    pub spatial_id: u8,
    /// Temporal layer.
    pub temporal_id: u8,
    /// One DTI per decode target.
    pub dtis: Vec<Dti>,
    /// Frame diffs to the referenced frames (1-16).
    pub fdiffs: Vec<u16>,
    /// Frame diff to the previous frame of each chain (0-15; 0: none).
    pub chain_fdiffs: Vec<u8>,
}

/// The template dependency structure (§A.8.2 `template_dependency_structure`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateStructure {
    /// `template_id_offset`: the template id of `templates[0]`.
    pub template_id_offset: u8,
    /// Number of decode targets.
    pub decode_targets: usize,
    /// Templates in layer order: spatial ids non-decreasing, temporal ids
    /// rising by at most one within a spatial layer and restarting at 0.
    pub templates: Vec<FrameTemplate>,
    /// Number of chains (0: no chains).
    pub chains: usize,
    /// The chain protecting each decode target (empty without chains).
    pub decode_target_protected_by: Vec<u8>,
    /// Maximum render size per spatial layer (empty: not sent).
    pub resolutions: Vec<(u16, u16)>,
}

impl TemplateStructure {
    /// The template index of a template id, if the id is in range.
    #[must_use]
    pub fn template_index(&self, template_id: u8) -> Option<usize> {
        let index = (usize::from(template_id) + usize::from(TEMPLATE_ID_RANGE)
            - usize::from(self.template_id_offset))
            % usize::from(TEMPLATE_ID_RANGE);
        (index < self.templates.len()).then_some(index)
    }

    /// The template id of template `index`.
    #[must_use]
    pub fn template_id(&self, index: usize) -> u8 {
        ((usize::from(self.template_id_offset) + index) % usize::from(TEMPLATE_ID_RANGE)) as u8
    }

    /// Spatial and temporal layer of each decode target
    /// (§A.8.2 `decode_target_layers`).
    #[must_use]
    pub fn decode_target_layers(&self) -> Vec<(u8, u8)> {
        (0..self.decode_targets)
            .map(|dt| {
                self.templates
                    .iter()
                    .filter(|t| t.dtis[dt] != Dti::NotPresent)
                    .fold((0, 0), |(s, t), template| {
                        (s.max(template.spatial_id), t.max(template.temporal_id))
                    })
            })
            .collect()
    }

    /// Check the structure can be written.
    ///
    /// # Errors
    /// Counts or values outside the syntax's ranges, or templates out of
    /// layer order.
    pub fn validate(&self) -> Result<()> {
        let bad = |what: &str| Err(RtpError::Config(format!("dependency structure: {what}")));
        if self.decode_targets == 0 || self.decode_targets > MAX_DECODE_TARGETS {
            return bad("decode target count");
        }
        if self.templates.is_empty() || self.templates.len() > usize::from(TEMPLATE_ID_RANGE) {
            return bad("template count");
        }
        if self.template_id_offset >= TEMPLATE_ID_RANGE || self.chains > self.decode_targets {
            return bad("offset or chain count");
        }
        let first = &self.templates[0];
        if (first.spatial_id, first.temporal_id) != (0, 0) {
            return bad("first template is not layer (0, 0)");
        }
        for pair in self.templates.windows(2) {
            if next_layer_idc(&pair[0], &pair[1]).is_none() {
                return bad("templates out of layer order");
            }
        }
        for template in &self.templates {
            if template.dtis.len() != self.decode_targets
                || template.chain_fdiffs.len() != self.chains
                || template
                    .fdiffs
                    .iter()
                    .any(|&d| d == 0 || d > MAX_TEMPLATE_FDIFF)
                || template
                    .chain_fdiffs
                    .iter()
                    .any(|&d| d > MAX_TEMPLATE_CHAIN_FDIFF)
            {
                return bad("template fields");
            }
        }
        if self.chains > 0 && self.decode_target_protected_by.len() != self.decode_targets {
            return bad("decode target protection");
        }
        let max_spatial = self
            .templates
            .last()
            .map_or(0, |t| usize::from(t.spatial_id));
        if !self.resolutions.is_empty() && self.resolutions.len() != max_spatial + 1 {
            return bad("one resolution per spatial layer");
        }
        Ok(())
    }
}

/// `next_layer_idc` from template `a` to template `b` (Table A.2).
fn next_layer_idc(a: &FrameTemplate, b: &FrameTemplate) -> Option<u32> {
    if (b.spatial_id, b.temporal_id) == (a.spatial_id, a.temporal_id) {
        Some(NEXT_SAME_LAYER)
    } else if b.spatial_id == a.spatial_id && b.temporal_id == a.temporal_id + 1 {
        Some(NEXT_TEMPORAL)
    } else if b.spatial_id == a.spatial_id + 1 && b.temporal_id == 0 {
        Some(NEXT_SPATIAL)
    } else {
        None
    }
}

/// One descriptor, as written in a packet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DependencyDescriptor {
    /// The packet's first byte begins a frame.
    pub start_of_frame: bool,
    /// The packet's last byte ends a frame.
    pub end_of_frame: bool,
    /// `frame_dependency_template_id`.
    pub template_id: u8,
    /// 16-bit frame number, +1 per frame in decode order.
    pub frame_number: u16,
    /// The template structure, on the first packet of a coded video sequence.
    pub structure: Option<TemplateStructure>,
    /// `active_decode_targets_bitmask`, when sent.
    pub active_decode_targets: Option<u32>,
    /// Custom DTIs overriding the template's.
    pub custom_dtis: Option<Vec<Dti>>,
    /// Custom frame diffs (1-4096) overriding the template's.
    pub custom_fdiffs: Option<Vec<u16>>,
    /// Custom chain diffs (0-255) overriding the template's.
    pub custom_chains: Option<Vec<u8>>,
}

/// A descriptor resolved against its template: the frame's full description.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameDependencies {
    /// Spatial layer.
    pub spatial_id: u8,
    /// Temporal layer.
    pub temporal_id: u8,
    /// One DTI per decode target.
    pub dtis: Vec<Dti>,
    /// Frame diffs to the referenced frames.
    pub fdiffs: Vec<u16>,
    /// Frame diff to the previous frame of each chain.
    pub chain_fdiffs: Vec<u8>,
}

/// `write_ns(n, v)` (§A.8.2): a value in `0..n` in one bit fewer for the
/// low part of the range.
fn write_ns(writer: &mut BitWriter, n: u32, value: u32) {
    if n <= 1 {
        return;
    }
    let w = u32::BITS - n.leading_zeros();
    let m = (1 << w) - n;
    if value < m {
        writer.put_bits(value, w - 1);
    } else {
        writer.put_bits(value + m, w);
    }
}

/// `ns(n)` (§A.8.2).
fn read_ns(reader: &mut BitReader<'_>, n: u32) -> Result<u32> {
    if n <= 1 {
        return Ok(0);
    }
    let w = u32::BITS - n.leading_zeros();
    let m = (1 << w) - n;
    let v = read(reader, w - 1)?;
    if v < m {
        return Ok(v);
    }
    let extra = read(reader, 1)?;
    Ok((v << 1) - m + extra)
}

fn read(reader: &mut BitReader<'_>, bits: u32) -> Result<u32> {
    reader
        .read_bits(bits)
        .map_err(|_| RtpError::Truncated("dependency descriptor"))
}

impl DependencyDescriptor {
    /// True when the descriptor needs the extended fields.
    fn extended(&self) -> bool {
        self.structure.is_some()
            || self.active_decode_targets.is_some()
            || self.custom_dtis.is_some()
            || self.custom_fdiffs.is_some()
            || self.custom_chains.is_some()
    }

    /// Serialize; `current` is the structure in effect (this descriptor's
    /// own when it carries one), which fixes the decode target and chain
    /// counts.
    ///
    /// # Errors
    /// Field counts that do not match the structure, or out-of-range values.
    pub fn write(&self, current: &TemplateStructure) -> Result<Vec<u8>> {
        let structure = self.structure.as_ref().unwrap_or(current);
        structure.validate()?;
        if structure.template_index(self.template_id).is_none() {
            return Err(RtpError::Config("template id outside the structure".into()));
        }
        let mut w = BitWriter::new();
        w.put_bit(self.start_of_frame);
        w.put_bit(self.end_of_frame);
        w.put_bits(u32::from(self.template_id), 6);
        w.put_bits(u32::from(self.frame_number), 16);
        if self.extended() {
            w.put_bit(self.structure.is_some());
            w.put_bit(self.active_decode_targets.is_some());
            w.put_bit(self.custom_dtis.is_some());
            w.put_bit(self.custom_fdiffs.is_some());
            w.put_bit(self.custom_chains.is_some());
            if let Some(structure) = &self.structure {
                write_structure(&mut w, structure);
            }
            if let Some(mask) = self.active_decode_targets {
                w.put_bits(mask, structure.decode_targets as u32);
            }
        }
        if let Some(dtis) = &self.custom_dtis {
            if dtis.len() != structure.decode_targets {
                return Err(RtpError::Config("custom DTI count".into()));
            }
            for &dti in dtis {
                w.put_bits(dti as u32, 2);
            }
        }
        if let Some(fdiffs) = &self.custom_fdiffs {
            for &fdiff in fdiffs {
                if fdiff == 0 || fdiff > MAX_FRAME_FDIFF {
                    return Err(RtpError::Config(format!("frame diff {fdiff}")));
                }
                let minus_one = u32::from(fdiff - 1);
                let nibbles = (u32::BITS - minus_one.leading_zeros()).div_ceil(4).max(1);
                w.put_bits(nibbles, 2);
                w.put_bits(minus_one, 4 * nibbles);
            }
            w.put_bits(0, 2);
        }
        if let Some(chains) = &self.custom_chains {
            if chains.len() != structure.chains {
                return Err(RtpError::Config("custom chain count".into()));
            }
            for &chain in chains {
                w.put_bits(u32::from(chain), 8);
            }
        }
        // The extended flags take bits 24-28, so an extended descriptor is
        // always longer than the mandatory three bytes, as the reader needs.
        w.byte_align_zero();
        Ok(w.into_bytes())
    }

    /// Parse one descriptor. `current` is the structure in effect (from an
    /// earlier packet); a descriptor carrying its own replaces it.
    ///
    /// # Errors
    /// Truncated data, a template id outside the structure, or no structure.
    pub fn parse(data: &[u8], current: Option<&TemplateStructure>) -> Result<Self> {
        if data.len() < MANDATORY_BYTES {
            return Err(RtpError::Truncated("dependency descriptor"));
        }
        let mut r = BitReader::new(data);
        let start_of_frame = read(&mut r, 1)? != 0;
        let end_of_frame = read(&mut r, 1)? != 0;
        let template_id = read(&mut r, 6)? as u8;
        let frame_number = read(&mut r, 16)? as u16;
        let mut descriptor = Self {
            start_of_frame,
            end_of_frame,
            template_id,
            frame_number,
            structure: None,
            active_decode_targets: None,
            custom_dtis: None,
            custom_fdiffs: None,
            custom_chains: None,
        };
        let (mut dtis_flag, mut fdiffs_flag, mut chains_flag) = (false, false, false);
        let mut active_flag = false;
        if data.len() > MANDATORY_BYTES {
            let structure_flag = read(&mut r, 1)? != 0;
            active_flag = read(&mut r, 1)? != 0;
            dtis_flag = read(&mut r, 1)? != 0;
            fdiffs_flag = read(&mut r, 1)? != 0;
            chains_flag = read(&mut r, 1)? != 0;
            if structure_flag {
                descriptor.structure = Some(read_structure(&mut r)?);
            }
        }
        let structure = descriptor
            .structure
            .as_ref()
            .or(current)
            .ok_or(RtpError::Malformed(
                "dependency descriptor before any structure",
            ))?;
        if active_flag {
            descriptor.active_decode_targets = Some(read(&mut r, structure.decode_targets as u32)?);
        }
        if structure.template_index(template_id).is_none() {
            return Err(RtpError::Malformed("template id outside the structure"));
        }
        if dtis_flag {
            descriptor.custom_dtis = Some(
                (0..structure.decode_targets)
                    .map(|_| read(&mut r, 2).map(Dti::from_bits))
                    .collect::<Result<_>>()?,
            );
        }
        if fdiffs_flag {
            let mut fdiffs = Vec::new();
            loop {
                let nibbles = read(&mut r, 2)?;
                if nibbles == 0 {
                    break;
                }
                fdiffs.push(read(&mut r, 4 * nibbles)? as u16 + 1);
            }
            descriptor.custom_fdiffs = Some(fdiffs);
        }
        if chains_flag {
            descriptor.custom_chains = Some(
                (0..structure.chains)
                    .map(|_| read(&mut r, 8).map(|v| v as u8))
                    .collect::<Result<_>>()?,
            );
        }
        Ok(descriptor)
    }

    /// The frame's description: the template's values with any custom
    /// fields applied (§A.8.2 `frame_dependency_definition`).
    ///
    /// # Errors
    /// The template id is outside `structure`.
    pub fn resolve(&self, structure: &TemplateStructure) -> Result<FrameDependencies> {
        let index = structure
            .template_index(self.template_id)
            .ok_or(RtpError::Malformed("template id outside the structure"))?;
        let template = &structure.templates[index];
        Ok(FrameDependencies {
            spatial_id: template.spatial_id,
            temporal_id: template.temporal_id,
            dtis: self
                .custom_dtis
                .clone()
                .unwrap_or_else(|| template.dtis.clone()),
            fdiffs: self
                .custom_fdiffs
                .clone()
                .unwrap_or_else(|| template.fdiffs.clone()),
            chain_fdiffs: self
                .custom_chains
                .clone()
                .unwrap_or_else(|| template.chain_fdiffs.clone()),
        })
    }
}

/// `template_dependency_structure()`.
fn write_structure(w: &mut BitWriter, s: &TemplateStructure) {
    w.put_bits(u32::from(s.template_id_offset), 6);
    w.put_bits(s.decode_targets as u32 - 1, 5);
    for pair in s.templates.windows(2) {
        w.put_bits(next_layer_idc(&pair[0], &pair[1]).expect("validated"), 2);
    }
    w.put_bits(NO_MORE_TEMPLATES, 2);
    for template in &s.templates {
        for &dti in &template.dtis {
            w.put_bits(dti as u32, 2);
        }
    }
    for template in &s.templates {
        for &fdiff in &template.fdiffs {
            w.put_bit(true);
            w.put_bits(u32::from(fdiff - 1), 4);
        }
        w.put_bit(false);
    }
    write_ns(w, s.decode_targets as u32 + 1, s.chains as u32);
    if s.chains > 0 {
        for &chain in &s.decode_target_protected_by {
            write_ns(w, s.chains as u32, u32::from(chain));
        }
        for template in &s.templates {
            for &fdiff in &template.chain_fdiffs {
                w.put_bits(u32::from(fdiff), 4);
            }
        }
    }
    w.put_bit(!s.resolutions.is_empty());
    for &(width, height) in &s.resolutions {
        w.put_bits(u32::from(width) - 1, 16);
        w.put_bits(u32::from(height) - 1, 16);
    }
}

/// `template_dependency_structure()`, read.
fn read_structure(r: &mut BitReader<'_>) -> Result<TemplateStructure> {
    let template_id_offset = read(r, 6)? as u8;
    let decode_targets = read(r, 5)? as usize + 1;
    let mut layers = vec![(0u8, 0u8)];
    loop {
        let &(s, t) = layers.last().expect("non-empty");
        match read(r, 2)? {
            NEXT_SAME_LAYER => layers.push((s, t)),
            NEXT_TEMPORAL => layers.push((s, t + 1)),
            NEXT_SPATIAL => layers.push((s + 1, 0)),
            _ => break,
        }
        if layers.len() > usize::from(TEMPLATE_ID_RANGE) {
            return Err(RtpError::Malformed("more than 64 templates"));
        }
    }
    let mut templates: Vec<FrameTemplate> = layers
        .iter()
        .map(|&(spatial_id, temporal_id)| FrameTemplate {
            spatial_id,
            temporal_id,
            dtis: Vec::new(),
            fdiffs: Vec::new(),
            chain_fdiffs: Vec::new(),
        })
        .collect();
    for template in &mut templates {
        template.dtis = (0..decode_targets)
            .map(|_| read(r, 2).map(Dti::from_bits))
            .collect::<Result<_>>()?;
    }
    for template in &mut templates {
        while read(r, 1)? != 0 {
            template.fdiffs.push(read(r, 4)? as u16 + 1);
        }
    }
    let chains = read_ns(r, decode_targets as u32 + 1)? as usize;
    let mut decode_target_protected_by = Vec::new();
    if chains > 0 {
        for _ in 0..decode_targets {
            decode_target_protected_by.push(read_ns(r, chains as u32)? as u8);
        }
        for template in &mut templates {
            template.chain_fdiffs = (0..chains)
                .map(|_| read(r, 4).map(|v| v as u8))
                .collect::<Result<_>>()?;
        }
    }
    let mut resolutions = Vec::new();
    if read(r, 1)? != 0 {
        let spatial_layers = usize::from(templates.last().expect("non-empty").spatial_id) + 1;
        for _ in 0..spatial_layers {
            let width = read(r, 16)? + 1;
            let height = read(r, 16)? + 1;
            resolutions.push((width as u16, height as u16));
        }
    }
    Ok(TemplateStructure {
        template_id_offset,
        decode_targets,
        templates,
        chains,
        decode_target_protected_by,
        resolutions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// L1T2: two templates, decode targets "T0 only" and "T0+T1", one chain.
    fn l1t2() -> TemplateStructure {
        TemplateStructure {
            template_id_offset: 5,
            decode_targets: 2,
            templates: vec![
                FrameTemplate {
                    spatial_id: 0,
                    temporal_id: 0,
                    dtis: vec![Dti::Switch, Dti::Switch],
                    fdiffs: vec![2],
                    chain_fdiffs: vec![2],
                },
                FrameTemplate {
                    spatial_id: 0,
                    temporal_id: 1,
                    dtis: vec![Dti::NotPresent, Dti::Discardable],
                    fdiffs: vec![1],
                    chain_fdiffs: vec![1],
                },
            ],
            chains: 1,
            decode_target_protected_by: vec![0, 0],
            resolutions: vec![(1280, 720)],
        }
    }

    #[test]
    fn structure_and_custom_fields_round_trip() {
        let structure = l1t2();
        let descriptor = DependencyDescriptor {
            start_of_frame: true,
            end_of_frame: false,
            template_id: 6,
            frame_number: 0xBEEF,
            structure: Some(structure.clone()),
            active_decode_targets: Some(0b01),
            custom_dtis: Some(vec![Dti::Required, Dti::Discardable]),
            custom_fdiffs: Some(vec![1, 17, 300, 4096]),
            custom_chains: Some(vec![200]),
        };
        let bytes = descriptor.write(&structure).unwrap();
        let parsed = DependencyDescriptor::parse(&bytes, None).unwrap();
        assert_eq!(parsed, descriptor);
        assert_eq!(
            parsed.structure.unwrap().decode_target_layers(),
            vec![(0, 0), (0, 1)]
        );
    }

    #[test]
    fn a_template_only_descriptor_is_three_bytes() {
        let structure = l1t2();
        let descriptor = DependencyDescriptor {
            start_of_frame: true,
            end_of_frame: true,
            template_id: 5,
            frame_number: 7,
            structure: None,
            active_decode_targets: None,
            custom_dtis: None,
            custom_fdiffs: None,
            custom_chains: None,
        };
        let bytes = descriptor.write(&structure).unwrap();
        assert_eq!(bytes, vec![0xC5, 0x00, 0x07]);
        let parsed = DependencyDescriptor::parse(&bytes, Some(&structure)).unwrap();
        let frame = parsed.resolve(&structure).unwrap();
        assert_eq!(
            (frame.spatial_id, frame.temporal_id, frame.fdiffs),
            (0, 0, vec![2])
        );
        assert!(DependencyDescriptor::parse(&bytes, None).is_err());
    }

    #[test]
    fn template_ids_wrap_modulo_64() {
        let mut structure = l1t2();
        structure.template_id_offset = 63;
        assert_eq!(structure.template_index(63), Some(0));
        assert_eq!(structure.template_index(0), Some(1));
        assert_eq!(structure.template_index(1), None);
        assert_eq!(structure.template_id(1), 0);
    }

    #[test]
    fn ns_coding_matches_the_spec_table() {
        // ns(5): 0..2 in two bits, 3 -> 110, 4 -> 111 (§A.8.2).
        for (value, bits, code) in [(0, 2, 0b00), (2, 2, 0b10), (3, 3, 0b110), (4, 3, 0b111)] {
            let mut w = BitWriter::new();
            write_ns(&mut w, 5, value);
            assert_eq!(w.bit_len(), bits);
            w.put_bits(0, 8 - bits as u32);
            assert_eq!(w.into_bytes()[0] >> (8 - bits), code);
            let mut w = BitWriter::new();
            write_ns(&mut w, 5, value);
            w.byte_align_zero();
            let bytes = w.into_bytes();
            assert_eq!(read_ns(&mut BitReader::new(&bytes), 5).unwrap(), value);
        }
    }

    #[test]
    fn invalid_structures_are_refused() {
        let mut structure = l1t2();
        structure.templates[1].temporal_id = 2;
        assert!(structure.validate().is_err());
        let mut structure = l1t2();
        structure.templates[0].fdiffs = vec![17];
        assert!(structure.validate().is_err());
    }
}
