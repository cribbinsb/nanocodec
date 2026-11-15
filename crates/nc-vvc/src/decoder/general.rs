//! General streams' pictures, assembled slice by slice (H.266 §7.4.2.4.4:
//! the slices of a picture cover its CTUs once each).
//!
//! Each slice is parsed, its motion derived and its CUs reconstructed with
//! its own header, as soon as it arrives: neighbours in other slices are
//! unavailable (§6.4.4), so a slice needs nothing from the others. The
//! picture-level stages (inverse luma mapping, deblocking, SAO and ALF) run
//! once the last CTU is covered, looking up each CU's or CTU's slice.
use super::Decoder;
use crate::{
    decoder_blocks::{self as blocks, CtuFilters, Unit},
    decoder_headers::SliceHeader,
};
use nc_common::{
    Picture, PlaneIndex,
    decode::{DecodeError, Result},
};

/// A picture whose slices are still arriving.
#[derive(Debug)]
pub(super) struct Assembly {
    poc: i32,
    /// The reconstruction so far, before the loop filters.
    picture: Picture,
    /// Every decoded slice's CUs, slice after slice.
    units: Vec<Unit>,
    /// The SAO/ALF syntax of each CTU, by CTU address.
    filters: Vec<CtuFilters>,
    /// The slice headers, by slice index (decoding order).
    headers: Vec<SliceHeader>,
    /// Each slice's active reference POCs.
    lists: Vec<[Vec<i32>; 2]>,
    /// The slice index of each CTU decoded so far, by CTU address.
    owners: Vec<Option<u16>>,
    /// CTUs not yet covered by a slice.
    remaining: usize,
}

impl Decoder {
    /// Decode one slice NAL of a general stream; returns its picture
    /// (before cropping) when this slice completed it.
    pub(super) fn decode_general(
        &mut self,
        rbsp: &[u8],
        kind: u8,
        temporal_id: u8,
    ) -> Result<Option<Picture>> {
        // A failed slice abandons its picture: the next slice starts anew.
        let assembly = self.assembly.take();
        let sequence = self.general_sequence.as_deref().unwrap();
        let parameters = self.general_picture.as_deref().unwrap();
        if usize::from(temporal_id - 1) >= sequence.dpb.len() {
            return Err(DecodeError::Malformed("VVC temporal layer exceeds SPS"));
        }
        let header = SliceHeader::parse(
            rbsp,
            kind,
            sequence,
            parameters,
            self.picture_header.as_deref(),
        )?;
        let mut assembly = match assembly {
            Some(assembly) => {
                if assembly.headers[0].picture != header.picture {
                    return Err(DecodeError::Malformed("VVC picture with missing slices"));
                }
                *assembly
            }
            None => self.begin_general(&header, kind, temporal_id)?,
        };
        self.decode_general_slice(&mut assembly, rbsp, header)?;
        if assembly.remaining > 0 {
            self.assembly = Some(Box::new(assembly));
            return Ok(None);
        }
        self.finish_general(assembly).map(Some)
    }

    /// Start a picture at its first slice `header`: the DPB update, POC and
    /// an empty reconstruction.
    fn begin_general(
        &mut self,
        header: &SliceHeader,
        kind: u8,
        temporal_id: u8,
    ) -> Result<Assembly> {
        let sequence = self.general_sequence.as_deref().unwrap();
        let parameters = self.general_picture.as_deref().unwrap();
        // A GDR picture after the first picture of the stream decodes as any
        // other (H.266 §8.1.2: NoOutputBeforeRecoveryFlag is 0). One that
        // starts the stream needs generated references and withheld output.
        if header.picture.gdr && self.general_references.is_none() {
            return Err(DecodeError::Unsupported("VVC GDR stream start"));
        }
        let references = self.general_references.get_or_insert_with(Default::default);
        let (poc, _) =
            references.prepare(self.reference.take(), sequence, header, kind, temporal_id)?;
        let [width, height] = parameters.dimensions.map(|v| v as usize);
        let picture = Picture::new(width, height, 0)
            .map_err(|_| DecodeError::Malformed("VVC picture dimensions"))?;
        let ctus = parameters.partitioning.tiles.ctus(0) * parameters.partitioning.tiles.ctus(1);
        Ok(Assembly {
            poc,
            picture,
            units: Vec::new(),
            filters: vec![CtuFilters::default(); ctus as usize],
            headers: Vec::new(),
            lists: Vec::new(),
            owners: vec![None; ctus as usize],
            remaining: ctus as usize,
        })
    }

    /// Parse, predict and reconstruct slice `header` into `assembly`.
    fn decode_general_slice(
        &mut self,
        assembly: &mut Assembly,
        rbsp: &[u8],
        header: SliceHeader,
    ) -> Result<()> {
        let sequence = self.general_sequence.as_deref().unwrap();
        let parameters = self.general_picture.as_deref().unwrap();
        let references = self.general_references.as_deref().unwrap();
        let slice = u16::try_from(assembly.headers.len())
            .map_err(|_| DecodeError::Malformed("VVC slice count"))?;
        for &address in &header.ctus {
            let owner = &mut assembly.owners[address as usize];
            if owner.replace(slice).is_some() {
                return Err(DecodeError::Malformed("VVC slice repeats CTUs"));
            }
        }
        let poc = assembly.poc;
        let pocs = references.slice_lists(poc, &header)?;
        let dimensions = parameters.dimensions.map(|v| v as usize);
        let edges = reference_edges(sequence, parameters, header.subpicture);
        let reshape = blocks::reshape_model(sequence, &header, &self.adaptation[1])?;
        let filter_counts = blocks::filter_counts(&self.adaptation[0], &header)?;
        let mut units = blocks::parse_filtered(
            rbsp,
            sequence,
            parameters,
            &header,
            filter_counts,
            slice,
            &mut assembly.filters,
        )?;
        blocks::derive_motion(
            &mut units,
            dimensions,
            sequence,
            &header,
            [&pocs[0], &pocs[1]],
            references
                .temporal(poc, [&pocs[0], &pocs[1]], &header, sequence.ctu_log2)
                .map(|temporal| temporal.within(edges.area))
                .as_ref(),
        )?;
        let pictures = pocs.each_ref().map(|list| references.pictures(list));
        blocks::derive_refinement(
            &mut units,
            &header,
            poc,
            [&pocs[0], &pocs[1]],
            ([&pictures[0], &pictures[1]], edges),
            &self.dsp,
        );
        #[cfg(feature = "analysis")]
        if let Some(s) = &mut self.stats {
            blocks::record_analysis(s, &units, &header, rbsp.len());
        }
        let lists = blocks::scaling_lists(&header, &self.adaptation[2])?;
        blocks::inverse_residuals(&mut units, sequence, &header, lists, &self.dsp);
        blocks::reconstruct(
            &mut assembly.picture,
            &units,
            sequence,
            dimensions,
            header.picture.joint_sign,
            blocks::InterSources::new([&pictures[0], &pictures[1]], &header, edges),
            &self.dsp,
            reshape.as_ref().map(|m| (m, header.picture.chroma_scale)),
        );
        assembly.remaining -= header.ctus.len();
        assembly.units.append(&mut units);
        assembly.headers.push(header);
        assembly.lists.push(pocs);
        Ok(())
    }

    /// The picture-level stages of a complete picture: its motion kept for
    /// later pictures, inverse luma mapping, then the loop filters.
    fn finish_general(&mut self, assembly: Assembly) -> Result<Picture> {
        let Assembly {
            poc,
            mut picture,
            units,
            filters,
            headers,
            lists,
            owners,
            ..
        } = assembly;
        let sequence = self.general_sequence.as_deref().unwrap();
        let parameters = self.general_picture.as_deref().unwrap();
        let dimensions = parameters.dimensions.map(|v| v as usize);
        #[cfg(test)]
        blocks::compare_trace(&picture, "NC_RECO ", poc);
        self.inverse_mapping(&mut picture, &headers, &owners)?;
        self.general_references
            .as_deref_mut()
            .unwrap()
            .record_motion(blocks::MotionField::new(&units, dimensions, poc, lists));
        blocks::deblock(
            &mut picture,
            &units,
            sequence,
            parameters,
            &headers,
            &self.dsp,
        );
        #[cfg(test)]
        blocks::compare_trace(&picture, "NC_DBF ", poc);
        blocks::apply_filters(
            &mut picture,
            &filters,
            sequence,
            &self.adaptation[0],
            &self.dsp,
        )?;
        #[cfg(test)]
        blocks::compare_trace(&picture, "NC_FINAL ", poc);
        Ok(picture)
    }

    /// Map the luma of the CTUs of slices using LMCS back from the mapped
    /// domain (§8.8.2.2; every slice shares the picture header's model).
    fn inverse_mapping(
        &self,
        picture: &mut Picture,
        headers: &[SliceHeader],
        owners: &[Option<u16>],
    ) -> Result<()> {
        let Some(first) = headers.iter().find(|h| h.lmcs) else {
            return Ok(());
        };
        let sequence = self.general_sequence.as_deref().unwrap();
        let model = blocks::reshape_model(sequence, first, &self.adaptation[1])?
            .ok_or(DecodeError::Malformed("VVC missing LMCS model"))?;
        let plane = picture.plane_mut(PlaneIndex::Y);
        if headers.iter().all(|h| h.lmcs) {
            self.dsp.vvc_inverse_lmcs(&model, &mut plane.view_mut());
            return Ok(());
        }
        let side = 1usize << sequence.ctu_log2;
        let columns = plane.width().div_ceil(side);
        for (address, owner) in owners.iter().enumerate() {
            if !owner.is_some_and(|slice| headers[usize::from(slice)].lmcs) {
                continue;
            }
            let (x, y) = (address % columns * side, address / columns * side);
            let (width, height) = (side.min(plane.width() - x), side.min(plane.height() - y));
            if let Some(mut region) = plane.region_mut(x as isize, y as isize, width, height) {
                self.dsp.vvc_inverse_lmcs(&model, &mut region);
            }
        }
        Ok(())
    }
}

/// Where the luma reference samples of a slice of subpicture `subpicture`
/// may be read from: its subpicture, when that is treated as a picture
/// and is not the whole picture; else the picture, wrapped around when the
/// PPS says so (H.266 §8.5.6.3.2).
fn reference_edges(
    sequence: &crate::decoder_headers::Sequence,
    parameters: &crate::decoder_headers::PictureParameters,
    subpicture: usize,
) -> blocks::Edges {
    let layout = &sequence.subpictures.layout;
    let own = &layout[subpicture];
    if layout.len() > 1 && own.treated_as_picture {
        let side = 1i32 << sequence.ctu_log2;
        let [x, y, width, height] = own.ctus.map(|v| v as i32);
        let [picture_width, picture_height] = parameters.dimensions.map(|v| v as i32);
        return blocks::Edges {
            wrap: None,
            area: Some([
                x * side,
                y * side,
                ((x + width) * side).min(picture_width) - 1,
                ((y + height) * side).min(picture_height) - 1,
            ]),
        };
    }
    blocks::Edges {
        wrap: parameters.wraparound.map(|offset| offset as i32),
        area: None,
    }
}
