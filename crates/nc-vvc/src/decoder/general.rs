//! General streams' pictures, assembled slice by slice (H.266 §7.4.2.4.4:
//! the slices of a picture cover its CTUs once each), at the sample type
//! of the coded video sequence: `u8` for 8-bit streams, `u16` for deeper
//! ones (`docs/handoff/high-bit-depth.md`).
//!
//! Each slice is parsed, its motion derived and its CUs reconstructed with
//! its own header, as soon as it arrives: neighbours in other slices are
//! unavailable (§6.4.4), so a slice needs nothing from the others. The
//! picture-level stages (inverse luma mapping, deblocking, SAO and ALF) run
//! once the last CTU is covered, looking up each CU's or CTU's slice.
use crate::{
    decoder_blocks::{self as blocks, CtuFilters, Geometry, Scaling, Unit},
    decoder_dpb::References,
    decoder_headers::{Adaptation, PictureHeader, PictureParameters, Sequence, SliceHeader},
};
use nc_common::{
    BitDepth, BitDepths, DecodedPicture, OutputPicture, Picture, PictureRecycler, PlaneIndex,
    ShownPicture,
    decode::{DecodeError, Result},
};
use nc_dsp::{DepthKernels, Dsp, Pixel};
use std::sync::Arc;

/// `nal_unit_type` of trailing pictures (H.266 Table 5).
const TRAIL_NUT: u8 = 0;
/// `nal_unit_type` of RASL pictures.
const RASL_NUT: u8 = 3;
/// `nal_unit_type` of CRA pictures.
const CRA_NUT: u8 = 9;
/// `sh_slice_type` of I slices (H.266 Table 9).
const I_SLICE: u8 = 2;
/// `nal_unit_type` of IRAP pictures (`IDR_W_RADL`, `IDR_N_LP`, `CRA`).
const IRAP_NUTS: std::ops::RangeInclusive<u8> = 7..=9;

/// The active parameters and shared state a general slice decodes with.
pub(super) struct Active<'a> {
    pub sequence: &'a Sequence,
    pub parameters: &'a PictureParameters,
    /// The picture header NAL before the slice, if any.
    pub picture_header: Option<&'a PictureHeader>,
    /// The ALF, LMCS and scaling-list APSs, by kind and id.
    pub adaptation: &'a [[Option<Box<Adaptation>>; 8]; 3],
    pub dsp: &'a Dsp,
    /// Where analysis records go, when they are collected.
    #[cfg(feature = "analysis")]
    pub stats: Option<&'a mut nc_common::stream_stats::StreamStats>,
}

impl Active<'_> {
    /// The kernels at the sequence's depth as code for samples `T` sees it.
    fn kernels<T: Pixel>(&self) -> DepthKernels<'_> {
        self.dsp.at(self.sequence.depth.of::<T>())
    }
}

/// A picture whose slices are still arriving.
#[derive(Debug)]
struct Assembly<T: Pixel> {
    poc: i32,
    /// The reconstruction so far, before the loop filters.
    picture: Picture<T>,
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

/// The general decoding process of one coded video sequence at samples
/// `T`: its DPB, the picture being assembled and the latest picture.
#[derive(Debug)]
pub(super) struct General<T: Pixel> {
    /// The DPB; `None` until the sequence's first picture starts.
    references: Option<Box<References<T>>>,
    /// A picture whose slices are still arriving.
    assembly: Option<Box<Assembly<T>>>,
    /// The latest decoded picture (coded size) and its geometry, until the
    /// next picture moves them into the DPB.
    latest: Option<(Picture<T>, Geometry)>,
    /// The visible window of `latest`: its origin and size.
    window: ((usize, usize), (usize, usize)),
    /// The depths of the sequence's pictures.
    depths: BitDepths,
    /// The next IRAP picture starts the coded video sequence: the first
    /// picture, or the first after an end of sequence NAL unit
    /// (`NoOutputBeforeRecoveryFlag`, H.266 §8.1.1).
    sequence_start: bool,
    /// The RASL pictures of the current CRA picture are not decoded (it
    /// started the sequence, so their references are missing; §8.1.1).
    skip_rasl: bool,
    /// While recovering from a GDR picture that started the sequence: the
    /// recovery point's POC (`RpPicOrderCntVal`, §8.3.1). Pictures before
    /// it are not output and their missing references are generated.
    recovery: Option<i32>,
    /// Output copies' buffers.
    pictures: PictureRecycler<T>,
}

impl<T: Pixel> Default for General<T> {
    fn default() -> Self {
        Self {
            references: None,
            assembly: None,
            latest: None,
            window: ((0, 0), (0, 0)),
            depths: BitDepths::EIGHT,
            sequence_start: true,
            skip_rasl: false,
            recovery: None,
            pictures: PictureRecycler::default(),
        }
    }
}

impl<T: Pixel> General<T> {
    /// Whether a picture's slices are still arriving.
    pub fn assembling(&self) -> bool {
        self.assembly.is_some()
    }

    /// An end of sequence NAL unit: the next IRAP picture starts a new
    /// coded video sequence.
    pub fn end_sequence(&mut self) {
        self.sequence_start = true;
    }

    /// Abandon the picture being assembled.
    pub fn abandon(&mut self) {
        self.assembly = None;
    }

    /// Decode one slice NAL; `true` when this slice completed its picture,
    /// which [`Self::shown`] then shows.
    pub fn decode_slice(
        &mut self,
        active: &mut Active<'_>,
        rbsp: &[u8],
        kind: u8,
        temporal_id: u8,
    ) -> Result<bool> {
        if kind == RASL_NUT && self.skip_rasl {
            return Ok(false);
        }
        // A failed slice abandons its picture: the next slice starts anew.
        let assembly = self.assembly.take();
        // A TemporalId beyond sps_max_sublayers_minus1 is decoded as the
        // reference decoder does (`RPL_A_2` codes such pictures); nothing
        // here depends on the SPS's sublayer count.
        let header = SliceHeader::parse(
            rbsp,
            kind,
            active.sequence,
            active.parameters,
            active.picture_header,
        )?;
        let mut assembly = match assembly {
            Some(assembly) => {
                if assembly.headers[0].picture != header.picture {
                    return Err(DecodeError::Malformed("VVC picture with missing slices"));
                }
                *assembly
            }
            None => self.begin(active, &header, kind, temporal_id)?,
        };
        self.decode_into(active, &mut assembly, rbsp, header)?;
        if assembly.remaining > 0 {
            self.assembly = Some(Box::new(assembly));
            return Ok(false);
        }
        let picture = self.finish(active, assembly)?;
        self.show(active, picture);
        Ok(true)
    }

    /// Start a picture at its first slice `header`: the DPB update, POC and
    /// an empty reconstruction.
    fn begin(
        &mut self,
        active: &Active<'_>,
        header: &SliceHeader,
        kind: u8,
        temporal_id: u8,
    ) -> Result<Assembly<T>> {
        // A GDR picture after the first picture of a sequence decodes as any
        // other (H.266 §8.1.1: NoOutputBeforeRecoveryFlag is 0). One that
        // starts it withholds the output of the pictures before its recovery
        // point, whose missing references are generated.
        // A picture whose slices mix IRAP and other NAL unit types is not an
        // IRAP picture (H.266 §7.4.3.4 `pps_mixed_nalu_types_in_pic_flag`):
        // the DPB and the POC continue as for a trailing picture.
        let kind = if active.parameters.mixed_nal_types {
            TRAIL_NUT
        } else {
            kind
        };
        let irap = IRAP_NUTS.contains(&kind);
        let starting_gdr = header.picture.gdr && self.sequence_start;
        let starts_sequence = (irap || starting_gdr) && self.sequence_start;
        if irap || starting_gdr {
            self.skip_rasl = starts_sequence && kind == CRA_NUT;
            self.sequence_start = false;
            self.recovery = None;
        }
        let [width, height] = active.parameters.dimensions.map(|v| v as usize);
        let geometry = Geometry::of(active.sequence, active.parameters);
        let filler = if starting_gdr || self.recovery.is_some() {
            Some(unavailable_picture(width, height, active.sequence.depth)?)
        } else {
            None
        };
        let references = self.references.get_or_insert_with(Default::default);
        let (poc, _) = references.prepare(
            self.latest.take(),
            active.sequence,
            header,
            (kind, starts_sequence),
            temporal_id,
            filler.as_ref().map(|filler| (filler, geometry)),
        )?;
        if starting_gdr {
            let distance = header.picture.order.recovery.unwrap_or(0);
            self.recovery = Some(poc.saturating_add_unsigned(distance));
        }
        match self.recovery {
            Some(point) if poc < point => references.withhold_output(),
            _ => self.recovery = None,
        }
        let picture = Picture::new(width, height, 0)
            .map_err(|_| DecodeError::Malformed("VVC picture dimensions"))?;
        let tiles = &active.parameters.partitioning.tiles;
        let ctus = (tiles.ctus(0) * tiles.ctus(1)) as usize;
        Ok(Assembly {
            poc,
            picture,
            units: Vec::new(),
            filters: vec![CtuFilters::default(); ctus],
            headers: Vec::new(),
            lists: Vec::new(),
            owners: vec![None; ctus],
            remaining: ctus,
        })
    }

    /// Parse, predict and reconstruct slice `header` into `assembly`.
    fn decode_into(
        &self,
        active: &mut Active<'_>,
        assembly: &mut Assembly<T>,
        rbsp: &[u8],
        header: SliceHeader,
    ) -> Result<()> {
        let (sequence, parameters) = (active.sequence, active.parameters);
        let references = self.references.as_deref().unwrap();
        let slice = u16::try_from(assembly.headers.len())
            .map_err(|_| DecodeError::Malformed("VVC slice count"))?;
        for &address in &header.ctus {
            let owner = &mut assembly.owners[address as usize];
            if owner.replace(slice).is_some() {
                return Err(DecodeError::Malformed("VVC slice repeats CTUs"));
            }
        }
        let poc = assembly.poc;
        let pocs = references.slice_lists(poc, &header, sequence.poc_bits)?;
        let geometry = Geometry::of(sequence, parameters);
        let scalings = [
            references.scalings(&pocs[0], geometry, sequence)?,
            references.scalings(&pocs[1], geometry, sequence)?,
        ];
        if collocated_scaled(&header, &scalings) {
            return Err(DecodeError::Malformed("VVC scaled collocated picture"));
        }
        let dimensions = parameters.dimensions.map(|v| v as usize);
        let edges = reference_edges(sequence, parameters, header.subpicture);
        let reshape = blocks::reshape_model::<T>(sequence, &header, &active.adaptation[1])?;
        let filter_counts = blocks::filter_counts(&active.adaptation[0], &header)?;
        let mut units = blocks::parse_filtered(
            rbsp,
            sequence,
            parameters,
            &header,
            filter_counts,
            (
                slice,
                pocs.iter().flatten().all(|&reference| reference <= poc),
            ),
            &mut assembly.filters,
        )?;
        blocks::derive_motion(
            &mut units,
            dimensions,
            sequence,
            &header,
            ([&pocs[0], &pocs[1]], poc),
            references
                .temporal(poc, [&pocs[0], &pocs[1]], &header, sequence.ctu_log2)
                .map(|temporal| temporal.within(edges.area))
                .as_ref(),
        )?;
        let pictures = pocs.each_ref().map(|list| references.pictures(list));
        // Borrows only the DSP and the SPS, so the analysis build can still
        // record into `active.stats` below.
        let kernels = active.dsp.at(active.sequence.depth.of::<T>());
        let sources = blocks::InterSources::new(
            [&pictures[0], &pictures[1]],
            &header,
            edges,
            [&scalings[0], &scalings[1]],
        );
        blocks::derive_refinement(
            &mut units,
            &header,
            poc,
            [&pocs[0], &pocs[1]],
            sources,
            kernels,
        );
        #[cfg(feature = "analysis")]
        if let Some(s) = active.stats.as_deref_mut() {
            blocks::record_analysis(s, &units, &header, rbsp.len());
        }
        let lists = blocks::scaling_lists(&header, &active.adaptation[2])?;
        blocks::inverse_residuals(&mut units, sequence, &header, lists, kernels);
        blocks::reconstruct(
            &mut assembly.picture,
            &units,
            sequence,
            dimensions,
            header.picture.joint_sign,
            sources,
            kernels,
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
    fn finish(&mut self, active: &Active<'_>, assembly: Assembly<T>) -> Result<Picture<T>> {
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
        let (sequence, parameters) = (active.sequence, active.parameters);
        let dimensions = parameters.dimensions.map(|v| v as usize);
        let kernels = active.kernels::<T>();
        #[cfg(test)]
        blocks::compare_trace(&picture, "NC_RECO ", poc);
        inverse_mapping(active, &mut picture, &headers, &owners)?;
        self.references
            .as_deref_mut()
            .unwrap()
            .record_motion(blocks::MotionField::new(
                &units,
                dimensions,
                poc,
                (lists, &headers),
            ));
        blocks::deblock(
            &mut picture,
            &units,
            sequence,
            parameters,
            &headers,
            kernels,
        );
        #[cfg(test)]
        blocks::compare_trace(&picture, "NC_DBF ", poc);
        let closures = blocks::Closures::new(
            (sequence, parameters),
            &headers,
            &owners,
            (picture.width(), picture.height()),
        );
        blocks::apply_filters(
            &mut picture,
            &filters,
            (sequence, &closures),
            &active.adaptation[0],
            (active.dsp, sequence.depth),
        )?;
        #[cfg(test)]
        blocks::compare_trace(&picture, "NC_FINAL ", poc);
        Ok(picture)
    }

    /// Keep the decoded `picture` as the latest, with its visible window
    /// (the PPS conformance window, §7.4.3.5) and depths.
    fn show(&mut self, active: &Active<'_>, picture: Picture<T>) {
        let crop_unit = if active.sequence.chroma { 2 } else { 1 };
        let [left, right, top, bottom] = active.parameters.crop.map(|v| crop_unit * v as usize);
        let [width, height] = active.parameters.dimensions.map(|v| v as usize);
        self.window = ((left, top), (width - left - right, height - top - bottom));
        let depth = active.sequence.depth;
        self.depths = BitDepths {
            luma: depth,
            chroma: depth,
        };
        self.latest = Some((picture, Geometry::of(active.sequence, active.parameters)));
    }

    /// Depths of the latest decoded picture, including monochrome output.
    pub fn depths(&self) -> BitDepths {
        self.depths
    }

    /// The visible samples of the latest decoded picture.
    pub fn shown(&self) -> Option<ShownPicture<'_, T>> {
        let (origin, size) = self.window;
        ShownPicture::window(&self.latest.as_ref()?.0, origin, size)
    }

    /// Queue a copy of the visible samples of the latest picture for output
    /// in display order (the DPB's reorder rules, §C.5.2).
    pub fn queue_output(&mut self) -> Result<()> {
        let (origin, size) = self.window;
        let shown = self
            .latest
            .as_ref()
            .and_then(|(latest, _)| ShownPicture::window(latest, origin, size))
            .ok_or(DecodeError::Malformed("missing VVC picture"))?;
        let mut output = self.pictures.take(shown.width(), shown.height(), 0)?;
        shown.copy_to(&mut output);
        if let Some(references) = &mut self.references {
            references.queue_output(output);
        }
        Ok(())
    }

    /// Release every picture still waiting for output.
    pub fn flush_output(&mut self) {
        if let Some(references) = &mut self.references {
            references.flush_output();
        }
    }

    /// The next picture in display order, if one is ready.
    pub fn next_output(&mut self) -> Option<OutputPicture<T>> {
        let picture = self.references.as_mut()?.next_output()?;
        Some(OutputPicture {
            origin: (0, 0),
            size: (picture.width(), picture.height()),
            picture: Arc::new(picture),
            depths: self.depths,
        })
    }
}

/// The general decoding process at the current sequence's sample type.
#[derive(Debug)]
pub(super) enum Instance {
    /// 8-bit samples.
    Eight(Box<General<u8>>),
    /// 9- and 10-bit samples, in `u16`.
    High(Box<General<u16>>),
}

impl Default for Instance {
    fn default() -> Self {
        Self::Eight(Box::default())
    }
}

/// Run `$body` on the current instance's general process, bound to `$general`.
macro_rules! with_general {
    ($instance:expr, $general:ident => $body:expr) => {
        match $instance {
            Instance::Eight($general) => $body,
            Instance::High($general) => $body,
        }
    };
}
pub(super) use with_general;

impl Instance {
    /// A fresh process for a sequence of samples of `depth`.
    pub fn for_depth(depth: BitDepth) -> Self {
        if depth == BitDepth::EIGHT {
            Self::Eight(Box::default())
        } else {
            Self::High(Box::default())
        }
    }

    /// The next picture in display order, of either sample type.
    pub fn next_output(&mut self) -> Option<DecodedPicture> {
        with_general!(self, general => general.next_output().map(DecodedPicture::from))
    }
}

/// A generated reference picture (H.266 §8.3.4.2): every sample the
/// mid-grey of the depth, borders extended for prediction.
fn unavailable_picture<T: Pixel>(
    width: usize,
    height: usize,
    depth: BitDepth,
) -> Result<Picture<T>> {
    let mut picture = Picture::new(width, height, 0)
        .map_err(|_| DecodeError::Malformed("VVC picture dimensions"))?;
    let mid = T::from_i32(depth.of::<T>().mid());
    for plane in PlaneIndex::ALL {
        picture.plane_mut(plane).view_mut().fill(mid);
    }
    picture.extend_borders();
    Ok(picture)
}

/// Map the luma of the CTUs of slices using LMCS back from the mapped
/// domain (§8.8.2.2; every slice shares the picture header's model).
fn inverse_mapping<T: Pixel>(
    active: &Active<'_>,
    picture: &mut Picture<T>,
    headers: &[SliceHeader],
    owners: &[Option<u16>],
) -> Result<()> {
    let Some(first) = headers.iter().find(|h| h.lmcs) else {
        return Ok(());
    };
    let sequence = active.sequence;
    let model = blocks::reshape_model::<T>(sequence, first, &active.adaptation[1])?
        .ok_or(DecodeError::Malformed("VVC missing LMCS model"))?;
    let kernels = active.kernels::<T>();
    let plane = picture.plane_mut(PlaneIndex::Y);
    if headers.iter().all(|h| h.lmcs) {
        kernels.vvc_inverse_lmcs(&model, &mut plane.view_mut());
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
            kernels.vvc_inverse_lmcs(&model, &mut region);
        }
    }
    Ok(())
}

/// Whether the collocated picture of slice `header` is a scaled reference,
/// which H.266 §7.4.8 forbids (`sh_collocated_ref_idx`: its
/// `RprConstraintsActiveFlag` shall be 0).
fn collocated_scaled(header: &SliceHeader, scalings: &[Vec<Option<Scaling>>; 2]) -> bool {
    let [list, index] = header.collocated.map(usize::from);
    header.picture.motion.temporal
        && header.kind != I_SLICE
        && scalings[list].get(index).is_some_and(Option::is_some)
}

/// Where the luma reference samples of a slice of subpicture `subpicture`
/// may be read from: its subpicture, when that is treated as a picture
/// and is not the whole picture; else the picture, wrapped around when the
/// PPS says so (H.266 §8.5.6.3.2).
fn reference_edges(
    sequence: &Sequence,
    parameters: &PictureParameters,
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
