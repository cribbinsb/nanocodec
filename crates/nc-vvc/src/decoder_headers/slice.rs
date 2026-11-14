//! H.266 §7.3.7 slice headers and byte-aligned CABAC entry locations.
use super::{
    DecodeError, PictureHeader, PictureParameters, Reader, Result, Sequence,
    picture::{AlfSelection, Deblocking, extension},
    rpl,
    weights::Weights,
};

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)] // Independent SH syntax flags.
pub(crate) struct SliceHeader {
    pub picture: PictureHeader,
    pub kind: u8,
    pub no_prior_output: bool,
    pub alf: AlfSelection,
    pub lmcs: bool,
    pub scaling: bool,
    pub references: [rpl::Selected; 2],
    pub active: [u8; 2],
    pub cabac_init: bool,
    pub collocated: [u8; 2],
    pub weights: Weights,
    pub qp: i32,
    pub chroma_offsets: [i8; 3],
    pub cu_chroma_offsets: bool,
    pub sao: [bool; 2],
    pub deblock: Deblocking,
    pub dependent_quant: bool,
    pub sign_hiding: bool,
    pub ts_residual_disabled: bool,
    pub entry_offsets: Vec<u64>,
    pub payload_offset: usize,
}
impl SliceHeader {
    pub fn parse(
        bytes: &[u8],
        nal_kind: u8,
        s: &Sequence,
        p: &PictureParameters,
        previous: Option<&PictureHeader>,
    ) -> Result<Self> {
        let mut r = Reader::new(bytes);
        let (embedded, picture) = picture_header(&mut r, s, p, previous)?;
        slice_address(&mut r, s, p)?;
        let kind = if picture.inter_allowed {
            r.ue(2, "VVC slice type")? as u8
        } else {
            2
        };
        if kind == 2 && !picture.intra_allowed {
            return Err(DecodeError::Malformed("VVC intra slice disallowed"));
        }
        let no_prior_output = matches!(nal_kind, 7..=10) && r.flag()?;
        let alf = if s.tools.alf && !p.location.alf_in_ph {
            AlfSelection::parse(&mut r, s)?
        } else {
            picture.alf.clone()
        };
        let lmcs = picture.lmcs.is_some() && (embedded || r.flag()?);
        let scaling = picture.scaling.is_some() && (embedded || r.flag()?);
        let references = if p.location.rpl_in_ph {
            picture.references.clone()
        } else if matches!(nal_kind, 7 | 8) && !s.tools.idr_rpl {
            Default::default()
        } else {
            rpl::select(&mut r, s, p)?
        };
        let active = active_references(&mut r, p, kind, &references)?;
        let cabac_init = kind != 2 && p.cabac_init && r.flag()?;
        let collocated = collocated(&mut r, p, &picture, kind, active)?;
        let weights =
            if kind != 2 && !p.location.weights_in_ph && p.weighted[usize::from(kind == 0)] {
                Weights::parse(&mut r, s, p, active)?
            } else {
                picture.weights.clone()
            };
        let qp = if let Some(qp) = picture.qp {
            qp
        } else {
            p.initial_qp + r.se(-p.initial_qp, 63 - p.initial_qp, "VVC slice QP")?
        };
        let chroma_offsets = chroma_offsets(&mut r, s, p)?;
        let cu_chroma_offsets = !p.cu_chroma_offsets.is_empty() && r.flag()?;
        let sao = if s.tools.sao && !p.location.sao_in_ph {
            [r.flag()?, s.chroma && r.flag()?]
        } else {
            picture.sao
        };
        let deblock = if p.deblock.override_enabled && !p.deblock.in_ph {
            Deblocking::parse(&mut r, p, true)?
        } else {
            picture.deblock.clone()
        };
        let dependent_quant = s.tools.dependent_quant && r.flag()?;
        let sign_hiding = s.tools.sign_hiding && !dependent_quant && r.flag()?;
        let ts_residual_disabled =
            s.transforms.skip_log2.is_some() && !dependent_quant && !sign_hiding && r.flag()?;
        extension(&mut r, p.sh_extension)?;
        let entry_offsets = entry_offsets(&mut r, s, p)?;
        r.0.trailing_bits()?; // byte_alignment(): one followed by zero padding.
        let payload_offset = bytes.len() - r.0.remaining_bytes()?.len();
        if payload_offset == bytes.len() {
            return Err(DecodeError::Truncated);
        }
        Ok(Self {
            picture,
            kind,
            no_prior_output,
            alf,
            lmcs,
            scaling,
            references,
            active,
            cabac_init,
            collocated,
            weights,
            qp,
            chroma_offsets,
            cu_chroma_offsets,
            sao,
            deblock,
            dependent_quant,
            sign_hiding,
            ts_residual_disabled,
            entry_offsets,
            payload_offset,
        })
    }
}
fn slice_address(r: &mut Reader<'_>, s: &Sequence, p: &PictureParameters) -> Result<()> {
    let tile_count = p.partitioning.tiles.iter().map(Vec::len).product::<usize>();
    let address = if !p.partitioning.rectangular_slices && tile_count > 1 {
        let address = r.bits(usize::BITS - (tile_count - 1).leading_zeros())? as usize;
        if address >= tile_count {
            return Err(DecodeError::Malformed("VVC slice tile address"));
        }
        address
    } else {
        0
    };
    for &present in &s.extra_sh {
        if present {
            r.flag()?;
        }
    }
    let count = if !p.partitioning.rectangular_slices && tile_count - address > 1 {
        r.ue((tile_count - address - 1) as u32, "VVC slice tile count")? as usize + 1
    } else {
        tile_count
    };
    if address != 0 || count != tile_count {
        return Err(DecodeError::Unsupported("VVC partial-picture raster slice"));
    }
    Ok(())
}
fn active_references(
    r: &mut Reader<'_>,
    p: &PictureParameters,
    kind: u8,
    lists: &[rpl::Selected; 2],
) -> Result<[u8; 2]> {
    let counts = lists.each_ref().map(|list| list.entries.len() as u8);
    let override_active =
        ((kind != 2 && counts[0] > 1) || (kind == 0 && counts[1] > 1)) && r.flag()?;
    let mut active = [0; 2];
    for (index, count) in active.iter_mut().enumerate().take(match kind {
        2 => 0,
        1 => 1,
        _ => 2,
    }) {
        *count = if override_active && counts[index] > 1 {
            r.ue(14, "VVC active references")? as u8 + 1
        } else {
            p.default_active[index].min(counts[index])
        };
        if *count == 0 || *count > counts[index] {
            return Err(DecodeError::Malformed("VVC active reference count"));
        }
    }
    Ok(active)
}
fn collocated(
    r: &mut Reader<'_>,
    p: &PictureParameters,
    ph: &PictureHeader,
    kind: u8,
    active: [u8; 2],
) -> Result<[u8; 2]> {
    let mut selection = ph.motion.collocated;
    if kind != 2 && ph.motion.temporal && !p.location.rpl_in_ph {
        selection[0] = if kind == 0 { u8::from(!r.flag()?) } else { 0 };
        let count = active[usize::from(selection[0])];
        selection[1] = if count > 1 {
            r.ue(u32::from(count - 1), "VVC collocated index")? as u8
        } else {
            0
        };
    }
    Ok(selection)
}
fn chroma_offsets(r: &mut Reader<'_>, s: &Sequence, p: &PictureParameters) -> Result<[i8; 3]> {
    let mut offsets = p.chroma_offsets;
    if p.slice_chroma_offsets {
        for value in offsets
            .iter_mut()
            .take(if s.transforms.joint_cbcr { 3 } else { 2 })
        {
            let delta = r.se(-12, 12, "VVC slice chroma offset")?;
            let total = i32::from(*value) + delta;
            if !(-12..=12).contains(&total) {
                return Err(DecodeError::Malformed("VVC chroma offset sum"));
            }
            *value = total as i8;
        }
    }
    Ok(offsets)
}
fn entry_offsets(r: &mut Reader<'_>, s: &Sequence, p: &PictureParameters) -> Result<Vec<u64>> {
    if !s.entry_offsets {
        return Ok(Vec::new());
    }
    let columns = p.partitioning.tiles[0].len();
    let rows = if s.entropy_sync {
        p.dimensions[1].div_ceil(1 << s.ctu_log2) as usize
    } else {
        p.partitioning.tiles[1].len()
    };
    let count = columns * rows - 1;
    let mut offsets = Vec::with_capacity(count);
    if count > 0 {
        let bits = r.ue(31, "VVC entry offset width")? + 1;
        for _ in 0..count {
            offsets.push(u64::from(r.bits(bits)?) + 1);
        }
    }
    Ok(offsets)
}

fn picture_header(
    r: &mut Reader<'_>,
    s: &Sequence,
    p: &PictureParameters,
    previous: Option<&PictureHeader>,
) -> Result<(bool, PictureHeader)> {
    let embedded = r.flag()?;
    let picture = if embedded {
        if p.location.rpl_in_ph
            || p.deblock.in_ph
            || p.location.sao_in_ph
            || p.location.alf_in_ph
            || p.location.weights_in_ph
            || p.location.qp_in_ph
        {
            return Err(DecodeError::Malformed("VVC embedded PH with PH-only tools"));
        }
        PictureHeader::parse(r, s, p)?
    } else {
        previous
            .ok_or(DecodeError::Malformed("VVC missing PH"))?
            .clone()
    };
    Ok((embedded, picture))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{chroma_qp::ChromaQpTable, headers};
    use nc_common::ColourInfo;
    #[test]
    fn encoder_headers_end_at_exact_cabac_boundary() {
        let geometry = headers::Geometry::new(64, 64, headers::MIN_CTU_LOG2);
        let s = Sequence::parse(&headers::sps(
            &geometry,
            ColourInfo::default(),
            &ChromaQpTable::hevc(0),
            headers::Tools::default(),
            None,
        ))
        .unwrap();
        let p = PictureParameters::parse(&headers::pps(&geometry, false), &s).unwrap();
        for inter in [false, true] {
            for qp in [0, 22, 48, 63] {
                let poc = if inter { 255 } else { 0 };
                let mut bytes = headers::slice(qp, inter, poc, None, false, None);
                let expected = bytes.len();
                bytes.extend_from_slice(&[0x89, 0xab, 0xcd]);
                let kind = if inter {
                    headers::TRAIL_NUT
                } else {
                    headers::IDR_N_LP
                };
                let header = SliceHeader::parse(&bytes, kind, &s, &p, None).unwrap();
                assert_eq!(header.payload_offset, expected);
                assert_eq!(header.qp, qp as i32);
                assert_eq!(header.picture.order.lsb, u32::from(poc));
                assert_eq!(header.kind, if inter { 1 } else { 2 });
                assert_eq!(header.active, [u8::from(inter), 0]);
                assert!(header.sign_hiding);
                for end in 0..expected {
                    assert!(SliceHeader::parse(&bytes[..end], kind, &s, &p, None).is_err());
                }
            }
        }
    }
}
