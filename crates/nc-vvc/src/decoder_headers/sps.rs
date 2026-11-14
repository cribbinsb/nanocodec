//! H.266 §7.3.2.4 single-layer, 8-bit Main sequence parameter sets.
//! Geometry, DPB and coding tools are parsed before any picture allocation.
use super::{
    DecodeError, Reader, Result,
    ptl::Profile,
    tools::{Tools, Transforms},
    vui,
};
use nc_common::ColourInfo;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Partition {
    pub min_qt_log2: u8,
    pub mtt_depth: u8,
    pub max_bt_log2: u8,
    pub max_tt_log2: u8,
}
impl Partition {
    pub(super) fn parse(r: &mut Reader<'_>, min_cb: u32, ctu: u32, chroma: bool) -> Result<Self> {
        let qt = min_cb + r.ue(ctu.min(6) - min_cb, "VVC minimum quadtree size")?;
        let mtt_depth = r.ue(2 * (ctu - min_cb), "VVC MTT depth")? as u8;
        let max_bt_log2 = qt
            + if mtt_depth > 0 {
                r.ue(
                    if chroma { ctu.min(6) } else { ctu } - qt,
                    "VVC maximum binary tree size",
                )?
            } else {
                0
            };
        let max_tt_log2 = qt
            + if mtt_depth > 0 {
                r.ue(ctu.min(6) - qt, "VVC maximum ternary tree size")?
            } else {
                0
            };
        Ok(Self {
            min_qt_log2: qt as u8,
            mtt_depth,
            max_bt_log2: max_bt_log2 as u8,
            max_tt_log2: max_tt_log2 as u8,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)] // Independent flags mandated by SPS syntax.
pub(crate) struct Sequence {
    pub id: u8,
    pub profile: Profile,
    pub chroma: bool,
    pub ctu_log2: u8,
    pub dimensions: [u32; 2],
    pub crop: [u32; 4],
    pub gdr: bool,
    pub resampling: bool,
    pub resolution_change: bool,
    pub entropy_sync: bool,
    pub entry_offsets: bool,
    pub poc_bits: u8,
    pub poc_msb_bits: Option<u8>,
    pub extra_ph: Vec<bool>,
    pub extra_sh: Vec<bool>,
    pub dpb: Vec<[u32; 3]>,
    pub min_cb_log2: u8,
    pub partition_override: bool,
    pub partitions: [Partition; 3],
    pub dual_tree: bool,
    pub max_transform_log2: u8,
    pub transforms: Transforms,
    pub tools: Tools,
    pub field_pictures: bool,
    pub colour: ColourInfo,
}
impl Sequence {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let mut r = Reader::new(bytes);
        let id = r.bits(4)? as u8;
        if r.bits(4)? != 0 {
            return Err(DecodeError::Unsupported("VVC VPS-dependent SPS"));
        }
        let max_layer = r.bits(3)? as usize;
        if max_layer > 6 {
            return Err(DecodeError::Malformed("VVC temporal sublayers"));
        }
        let chroma = match r.bits(2)? {
            0 => false,
            1 => true,
            _ => return Err(DecodeError::Unsupported("VVC non-Main chroma format")),
        };
        let ctu_log2 = r.bits(2)? + 5;
        if ctu_log2 > 7 {
            return Err(DecodeError::Malformed("VVC CTU size"));
        }
        if !r.flag()? {
            return Err(DecodeError::Malformed("VVC SPS without PTL or VPS"));
        }
        let profile = Profile::parse(&mut r, max_layer)?;
        let gdr = r.flag()?;
        let resampling = r.flag()?;
        let resolution_change = resampling && r.flag()?;
        let (dimensions, crop) = geometry(&mut r, chroma)?;
        let entropy_sync = r.flag()?;
        let entry_offsets = r.flag()?;
        let poc_bits = r.bits(4)? + 4;
        if poc_bits > 16 {
            return Err(DecodeError::Malformed("VVC POC width"));
        }
        let poc_msb_bits = if r.flag()? {
            Some(r.ue(31 - poc_bits, "VVC POC MSB cycle width")? as u8 + 1)
        } else {
            None
        };
        let extra_ph = extra_bits(&mut r)?;
        let extra_sh = extra_bits(&mut r)?;
        let dpb = dpb(&mut r, max_layer)?;
        let min_cb = r.ue(4.min(ctu_log2 - 2), "VVC minimum coding block")? + 2;
        if dimensions
            .iter()
            .any(|&size| !size.is_multiple_of(1 << min_cb.max(3)))
        {
            return Err(DecodeError::Malformed("VVC coding-block alignment"));
        }
        let partition_override = r.flag()?;
        let intra_y = Partition::parse(&mut r, min_cb, ctu_log2, false)?;
        let dual_tree = chroma && r.flag()?;
        let intra_c = if dual_tree {
            Partition::parse(&mut r, min_cb, ctu_log2, true)?
        } else {
            intra_y.clone()
        };
        let inter = Partition::parse(&mut r, min_cb, ctu_log2, false)?;
        let max_transform_log2 = 5 + u8::from(ctu_log2 > 5 && r.flag()?);
        let transforms = Transforms::parse(&mut r, chroma)?;
        let tools = Tools::parse(&mut r, chroma, ctu_log2, poc_bits, dimensions, &transforms)?;
        if r.flag()? {
            vui::hrd(&mut r, max_layer)?;
        }
        let field_pictures = r.flag()?;
        if field_pictures && profile.frame_only {
            return Err(DecodeError::Malformed("VVC field/frame constraint"));
        }
        let colour = if r.flag()? {
            vui::parse(&mut r)?
        } else {
            ColourInfo::default()
        };
        let extension = r.flag()?;
        r.finish(extension)?;
        Ok(Self {
            id,
            profile,
            chroma,
            ctu_log2: ctu_log2 as u8,
            dimensions,
            crop,
            gdr,
            resampling,
            resolution_change,
            entropy_sync,
            entry_offsets,
            poc_bits: poc_bits as u8,
            poc_msb_bits,
            extra_ph,
            extra_sh,
            dpb,
            min_cb_log2: min_cb as u8,
            partition_override,
            partitions: [intra_y, inter, intra_c],
            dual_tree,
            max_transform_log2,
            transforms,
            tools,
            field_pictures,
            colour,
        })
    }
}

fn extra_bits(r: &mut Reader<'_>) -> Result<Vec<bool>> {
    let bytes = r.bits(2)?;
    if bytes > 2 {
        return Err(DecodeError::Malformed("VVC extra header bytes"));
    }
    r.flags(bytes as usize * 8)
}
fn dpb(r: &mut Reader<'_>, max_layer: usize) -> Result<Vec<[u32; 3]>> {
    let every_layer = max_layer > 0 && r.flag()?;
    let mut layers = vec![[0; 3]; max_layer + 1];
    for layer in &mut layers[if every_layer { 0 } else { max_layer }..=max_layer] {
        let buffering = r.ue(15, "VVC DPB size")?;
        let reorder = r.ue(buffering, "VVC DPB reorder count")?;
        let latency = r.ue(u32::MAX - 1, "VVC DPB latency")?;
        *layer = [buffering + 1, reorder, latency];
    }
    if !every_layer {
        let highest = layers[max_layer];
        layers.fill(highest);
    }
    Ok(layers)
}

fn geometry(r: &mut Reader<'_>, chroma: bool) -> Result<([u32; 2], [u32; 4])> {
    let dimensions = [
        r.ue(4096, "VVC maximum width")?,
        r.ue(4096, "VVC maximum height")?,
    ];
    if dimensions.contains(&0) {
        return Err(DecodeError::Malformed("VVC empty dimensions"));
    }
    let mut crop = [0; 4];
    if r.flag()? {
        for value in &mut crop {
            *value = r.ue(4096, "VVC conformance crop")?;
        }
        let scale = if chroma { 2 } else { 1 };
        if scale * (crop[0] + crop[1]) >= dimensions[0]
            || scale * (crop[2] + crop[3]) >= dimensions[1]
        {
            return Err(DecodeError::Malformed(
                "VVC conformance crop exceeds picture",
            ));
        }
    }
    if r.flag()? {
        return Err(DecodeError::Unsupported("VVC SPS subpicture geometry"));
    }
    if r.ue(8, "VVC bit depth")? != 0 {
        return Err(DecodeError::Unsupported(
            "VVC decoder requires 8-bit samples",
        ));
    }
    Ok((dimensions, crop))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{chroma_qp::ChromaQpTable, headers};

    #[test]
    fn encoder_sps_has_equivalent_semantics() {
        let sizes = [(32, 32), (34, 66), (640, 360)];
        for ((width, height), ctu_log2) in sizes
            .into_iter()
            .flat_map(|size| headers::CTU_LOG2_RANGE.map(move |ctu| (size, ctu)))
        {
            let map = ChromaQpTable::hevc(3);
            let geometry = headers::Geometry::new(width, height, ctu_log2);
            let bytes = headers::sps(
                &geometry,
                ColourInfo::default(),
                &map,
                headers::Tools::default(),
                None,
            );
            let s = Sequence::parse(&bytes).unwrap();
            assert_eq!(
                s.dimensions,
                [geometry.coded_width as u32, geometry.coded_height as u32]
            );
            assert_eq!(s.ctu_log2, ctu_log2);
            assert_eq!(s.max_transform_log2, crate::transform_tree::MAX_TB_LOG2);
            assert!(!s.dual_tree);
            assert_eq!(s.partitions[0].mtt_depth, 0);
            for qp in 0..64 {
                assert_eq!(u32::from(s.transforms.qp_tables[0][qp]), map.map(qp as u32));
            }
            for end in 0..bytes.len() {
                assert!(Sequence::parse(&bytes[..end]).is_err());
            }
        }
    }

    #[test]
    fn timed_encoder_sps_parses() {
        let geometry = headers::Geometry::new(640, 360, headers::MIN_CTU_LOG2);
        for fixed_rate in [false, true] {
            let timing = headers::Timing {
                num_units_in_tick: 1001,
                time_scale: 30000,
                fixed_rate,
            };
            for layers in [1, 3] {
                let bytes = headers::sps_layered(
                    &geometry,
                    ColourInfo::default(),
                    &ChromaQpTable::hevc(0),
                    headers::Tools::default(),
                    Some(timing),
                    layers,
                );
                let s = Sequence::parse(&bytes).unwrap();
                assert_eq!(s.ctu_log2, headers::MIN_CTU_LOG2);
            }
        }
    }

    #[test]
    fn rejects_ten_bit_before_parsing_coding_tools() {
        let mut w = nc_common::BitWriter::new();
        w.put_bits(0, 8); // SPS/VPS ids.
        w.put_bits(0, 3);
        w.put_bits(1, 2);
        w.put_bits(0, 2);
        w.put_bit(true);
        w.put_bits(1, 7);
        w.put_bit(false);
        w.put_bits(120, 8);
        w.put_bit(true);
        w.put_bit(false);
        w.put_bit(false);
        w.byte_align_zero();
        w.put_bits(0, 8);
        w.put_bit(false);
        w.put_bit(false);
        w.put_ue(32);
        w.put_ue(32);
        w.put_bit(false);
        w.put_bit(false);
        w.put_ue(2);
        w.trailing_bits();
        assert!(matches!(
            Sequence::parse(&w.into_bytes()),
            Err(DecodeError::Unsupported(
                "VVC decoder requires 8-bit samples"
            ))
        ));
    }

    #[test]
    #[ignore = "requires downloaded official VVC 8-bit corpus"]
    fn official_headers_parse() {
        let root = std::env::var_os("NC_VVC_CONFORMANCE").map_or_else(
            || {
                std::path::PathBuf::from(std::env::var_os("HOME").unwrap())
                    .join("nanocodec-data/conformance/vvc/streams")
            },
            std::path::PathBuf::from,
        );
        for name in [
            "8b400_A_2",
            "8b400_B_2",
            "8b420_A_2",
            "8b420_B_2",
            "CodingToolsSets_A_2",
            "CodingToolsSets_B_2",
            "LOSSLESS_A_3",
            "QUANT_C_2",
            "QUANT_D_4",
        ] {
            let bytes = std::fs::read(root.join(name).join(format!("{name}.bit"))).unwrap();
            let mut count = 0;
            let mut sequence = None;
            let mut picture = None;
            let mut pictures = 0;
            let mut picture_header = None;
            for unit in nc_common::decode::annex_b_units(&bytes).unwrap() {
                if matches!(unit[1] >> 3, 0..=3 | 7..=10 | 19) {
                    let kind = unit[1] >> 3;
                    let rbsp = nc_common::decode::unescape_rbsp(&unit[2..]).unwrap();
                    let mut r = Reader::new(&rbsp);
                    if kind == 19 {
                        picture_header = Some(
                            super::super::PictureHeader::parse(
                                &mut r,
                                sequence.as_ref().unwrap(),
                                picture.as_ref().unwrap(),
                            )
                            .unwrap_or_else(|error| panic!("{name} PH {pictures}: {error}")),
                        );
                        r.finish(false).unwrap();
                    } else {
                        let header = super::super::SliceHeader::parse(
                            &rbsp,
                            kind,
                            sequence.as_ref().unwrap(),
                            picture.as_ref().unwrap(),
                            picture_header.as_ref(),
                        )
                        .unwrap_or_else(|error| panic!("{name} SH {pictures}: {error}"));
                        assert!(header.payload_offset < rbsp.len());
                        pictures += 1;
                    }
                    continue;
                }
                if matches!(unit[1] >> 3, 17 | 18) {
                    let rbsp = nc_common::decode::unescape_rbsp(&unit[2..]).unwrap();
                    super::super::Adaptation::parse(&rbsp)
                        .unwrap_or_else(|error| panic!("{name} APS: {error}"));
                    continue;
                }
                if unit[1] >> 3 == 16 {
                    let rbsp = nc_common::decode::unescape_rbsp(&unit[2..]).unwrap();
                    picture = Some(
                        super::super::pps::PictureParameters::parse(
                            &rbsp,
                            sequence.as_ref().unwrap(),
                        )
                        .unwrap_or_else(|error| panic!("{name} PPS: {error}")),
                    );
                    continue;
                }
                if unit[1] >> 3 != 15 {
                    continue;
                }
                let rbsp = nc_common::decode::unescape_rbsp(&unit[2..]).unwrap();
                let s = Sequence::parse(&rbsp).unwrap_or_else(|error| panic!("{name}: {error}"));
                assert_eq!(s.chroma, !name.starts_with("8b400"));
                sequence = Some(s);
                count += 1;
            }
            assert!(count > 0, "{name}");
            let expected = match name {
                "CodingToolsSets_A_2" => 2,
                "CodingToolsSets_B_2" => 9,
                "LOSSLESS_A_3" => 100,
                "QUANT_C_2" | "QUANT_D_4" => 3,
                _ => 49,
            };
            assert_eq!(pictures, expected, "{name}");
        }
    }
}
