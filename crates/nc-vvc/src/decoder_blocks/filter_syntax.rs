//! H.266 CTU filter syntax, sharing the codec's SAO/ALF context state.
//! General alternatives and control bins follow BSD-Clear `VVdeC`.
use super::{Context, Contexts, DecodeError, Reader, Result, SliceHeader, initialization::bank};
use crate::{
    decoder_headers::{Adaptation, AdaptationData},
    sao::CtuSao,
};
use nc_common::BitDepth;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CtuFilters {
    pub edge: CtuSao,
    pub bands: [Option<u8>; 3],
    /// Luma: fixed sets 0..15 or APS-list index + 16; chroma: alternative.
    pub alf: [Option<u8>; 3],
    pub cross: [u8; 2],
    /// The ALF APS IDs the CTU's filters come from, from its slice header
    /// (slices may name different ones).
    pub banks: Banks,
}
/// ALF APS IDs of one CTU.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Banks {
    /// The luma filter set, when the CTU uses one of an APS (`alf[0]` >= 16).
    pub luma: Option<u8>,
    /// The chroma filters, when either chroma plane is filtered.
    pub chroma: Option<u8>,
    /// The cross-component filters of Cb and Cr, when used.
    pub cross: [Option<u8>; 2],
}
#[derive(Clone, Copy, Default)]
pub(crate) struct Counts {
    chroma: u8,
    cross: [u8; 2],
}
pub(crate) fn counts(banks: &[Option<Box<Adaptation>>; 8], header: &SliceHeader) -> Result<Counts> {
    let alf = |id: u8| match banks
        .get(usize::from(id))
        .and_then(Option::as_deref)
        .map(|a| &a.data)
    {
        Some(AdaptationData::Alf(alf)) => Ok(alf),
        _ => Err(DecodeError::Malformed("VVC missing ALF APS")),
    };
    for &id in &header.alf.luma {
        if alf(id)?.luma.is_empty() {
            return Err(DecodeError::Malformed("VVC empty luma ALF bank"));
        }
    }
    let mut counts = Counts::default();
    if let Some(id) = header.alf.chroma {
        counts.chroma = alf(id)?.chroma.len() as u8;
        if counts.chroma == 0 {
            return Err(DecodeError::Malformed("VVC empty chroma ALF bank"));
        }
    }
    for (plane, count) in counts.cross.iter_mut().enumerate() {
        if let Some(id) = header.alf.cross[plane] {
            *count = alf(id)?.cross[plane].len() as u8;
            if *count == 0 {
                return Err(DecodeError::Malformed("VVC empty cross-component ALF bank"));
            }
        }
    }
    Ok(counts)
}
#[derive(Clone)]
pub(super) struct Models {
    chroma: [[Context; 3]; 2],
    alternative: [Context; 2],
    cross: [Context; 6],
}
impl Models {
    pub fn new(qp: u32, kind: u8) -> Self {
        Self {
            chroma: [
                bank(
                    [[25, 61, 54], [4, 61, 54], [54, 39, 39]],
                    [4, 0, 0],
                    qp,
                    kind,
                ),
                bank(
                    [[25, 61, 54], [19, 46, 54], [31, 39, 39]],
                    [1, 0, 0],
                    qp,
                    kind,
                ),
            ],
            alternative: bank([[11, 26], [20, 12], [11, 11]], [0, 0], qp, kind),
            cross: bank(
                [
                    [25, 35, 38, 25, 28, 38],
                    [18, 21, 38, 18, 21, 38],
                    [18, 30, 31, 18, 30, 31],
                ],
                [4, 1, 4, 4, 1, 4],
                qp,
                kind,
            ),
        }
    }
    pub fn read(
        &mut self,
        reader: &mut Reader<'_>,
        contexts: &mut Contexts,
        header: &SliceHeader,
        neighbours: [Option<CtuFilters>; 2],
        (counts, depth): (Counts, BitDepth),
    ) -> Result<CtuFilters> {
        let mut filters = CtuFilters::default();
        if header.sao.contains(&true) {
            let largest = sao_largest_offset(depth);
            read_sao(
                reader,
                contexts,
                (header.sao, largest),
                neighbours,
                &mut filters,
            )?;
        }
        for plane in 0..3 {
            if !header.alf.enabled[plane] {
                continue;
            }
            let context = neighbours
                .iter()
                .flatten()
                .filter(|n| n.alf[plane].is_some())
                .count();
            let model = if plane == 0 {
                &mut contexts.alf_ctb[context]
            } else {
                &mut self.chroma[plane - 1][context]
            };
            if !reader.bin(model)? {
                continue;
            }
            let index = if plane == 0 {
                if !header.alf.luma.is_empty() && reader.bin(&mut contexts.alf_use_aps)? {
                    16 + truncated(reader, header.alf.luma.len() as u32)? as u8
                } else {
                    reader.bypass_bits(4)? as u8
                }
            } else {
                let mut index = 0;
                while index + 1 < counts.chroma && reader.bin(&mut self.alternative[plane - 1])? {
                    index += 1;
                }
                index
            };
            filters.alf[plane] = Some(index);
            if plane == 0 {
                filters.banks.luma = index
                    .checked_sub(16)
                    .map(|selection| header.alf.luma[usize::from(selection)]);
            } else {
                filters.banks.chroma = header.alf.chroma;
            }
        }
        for plane in 0..2 {
            if header.alf.cross[plane].is_none() {
                continue;
            }
            let context = neighbours
                .iter()
                .flatten()
                .filter(|n| n.cross[plane] != 0)
                .count();
            if reader.bin(&mut self.cross[3 * plane + context])? {
                let mut index = 1;
                while index < counts.cross[plane] && reader.bypass()? {
                    index += 1;
                }
                filters.cross[plane] = index;
                filters.banks.cross[plane] = header.alf.cross[plane];
            }
        }
        Ok(filters)
    }
}
fn truncated(reader: &mut Reader<'_>, count: u32) -> Result<u32> {
    let bits = count.ilog2();
    let threshold = (1 << (bits + 1)) - count;
    let value = reader.bypass_bits(bits)?;
    Ok(if value < threshold {
        value
    } else {
        2 * value + u32::from(reader.bypass()?) - threshold
    })
}
/// The bits of `sao_offset_abs`'s cMax above 5 (H.266 §7.4.9.3.2: cMax is
/// `(1 << (Min(BitDepth, 10) − 5)) − 1`).
const SAO_OFFSET_BITS_BASE: u32 = 5;
/// The depth beyond which SAO offsets are scaled instead of growing.
const SAO_OFFSET_MAX_DEPTH: u32 = 10;

/// cMax of `sao_offset_abs` at `depth` (7 at 8 bits, 31 at 10). Up to 10
/// bits `SaoOffsetVal` is the offset itself (its shift `BitDepth −
/// Min(BitDepth, 10)` is zero).
fn sao_largest_offset(depth: BitDepth) -> i8 {
    ((1 << (depth.bits().min(SAO_OFFSET_MAX_DEPTH) - SAO_OFFSET_BITS_BASE)) - 1) as i8
}

fn read_sao(
    reader: &mut Reader<'_>,
    contexts: &mut Contexts,
    (enabled, largest): ([bool; 2], i8),
    neighbours: [Option<CtuFilters>; 2],
    filters: &mut CtuFilters,
) -> Result<()> {
    for neighbour in neighbours.into_iter().flatten() {
        if reader.bin(&mut contexts.sao_merge)? {
            filters.edge = neighbour.edge;
            filters.bands = neighbour.bands;
            return Ok(());
        }
    }
    let mut types = [0; 2];
    for plane in 0..3 {
        let channel = usize::from(plane != 0);
        if !enabled[channel] {
            continue;
        }
        if plane != 2 && reader.bin(&mut contexts.sao_type)? {
            types[channel] = if reader.bypass()? { 2 } else { 1 };
        }
        if types[channel] == 0 {
            continue;
        }
        let mut offsets = [0i8; 4];
        for offset in &mut offsets {
            while *offset < largest && reader.bypass()? {
                *offset += 1;
            }
        }
        if types[channel] == 1 {
            for offset in &mut offsets {
                if *offset != 0 && reader.bypass()? {
                    *offset = -*offset;
                }
            }
            filters.bands[plane] = Some(reader.bypass_bits(5)? as u8);
        } else {
            offsets[2] = -offsets[2];
            offsets[3] = -offsets[3];
            if plane != 2 {
                filters.edge.classes[channel] = Some(reader.bypass_bits(2)? as u8);
            }
        }
        filters.edge.offsets[plane] = offsets;
    }
    Ok(())
}
