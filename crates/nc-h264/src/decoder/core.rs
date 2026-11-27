//! The decoding process of one sample type (`docs/handoff/high-bit-depth.md`):
//! [`Core`] holds a coded video sequence's references, pictures and
//! scratch in `T` samples (`u8` for 8-bit streams, `u16` deeper ones) and
//! decodes its slices; [`super::Decoder`] holds the parameter sets and one
//! core for the sequence's depth.
//!
//! Field pictures are decoded into the rows of their frame's buffer
//! (`structure.rs`). A first field waits for the second field of its frame
//! (§7.4.1.2.4: the next picture, opposite parity, the same `frame_num`,
//! both reference or both not); the frame is output once both are decoded.
//! A field left unpaired is output as JM writes one: the other field's rows
//! mid-grey.
//!
//! Separately coded colour planes (`separate_colour_plane_flag`, §7.4.2.1.1)
//! are three monochrome pictures in one: each plane's slices decode into
//! its own plane with its own macroblock states (`mbs` holds one picture's
//! worth per plane), are deblocked as luma, and predict from the same plane
//! of the references.

use std::sync::Arc;

#[cfg(feature = "analysis")]
use nc_common::stream_stats::StreamStats;
use nc_common::{BitDepths, Picture, PlaneIndex};
use nc_dsp::Dsp;

use super::cabac::Cabac;
use super::cabac_syntax::CabacSyntax;
use super::cavlc::{CavlcSyntax, CavlcTables};
use super::chroma::ChromaArray;
use super::deblock::{DeblockContext, RowDeblocker, SliceFilter, filter_mbaff};
use super::direct::{Current, DirectContext, dist_scale_factor};
use super::dpb::{Dpb, ListPicture, NewReference, RefPicture, field_list};
use super::inter::{IMPLICIT_LOG2_DENOM, InterContext, InterKernels, Weighting};
use super::mb::{MbExtra, SliceParams};
use super::mbstate::MbState;
use super::output::{DecodedFrame, OutputQueue};
use super::partition::{self, PartitionedCavlc};
use super::poc::{FramePoc, PocState};
use super::pool::PicturePool;
use super::pps::{BipredWeights, Pps};
use super::reader::Reader;
use super::recon::ReconContext;
use super::sample::H264Sample;
use super::scaling::{self, LevelScale};
use super::slice::{self, SliceHeader, SliceType, WeightTable};
use super::slice_data::{FieldMbContexts, SliceDecoder};
use super::slice_groups::MapUnits;
use super::sps::Sps;
#[cfg(feature = "analysis")]
use super::stats;
use super::structure::{MAX_FRAME_ID, Parity, Structure, picture_planes};
use super::transform::{Dequantiser, Residual};
use super::{DecodeError, Planes, Result};

/// Picture border kept for motion compensation (whole-window reads).
const REFERENCE_BORDER: usize = 64;

/// The views of `picture`'s planes.
/// The views of `picture`'s planes, colour plane `colour_plane` first (0
/// unless the planes are coded separately: the plane a monochrome slice
/// decodes into).
fn planes<T: H264Sample>(picture: &mut Picture<T>, colour_plane: usize) -> Planes<'_, T> {
    let [y, u, v] = picture.planes_mut();
    let mut views = [y.view_mut(), u.view_mut(), v.view_mut()];
    views.rotate_left(colour_plane);
    views
}

/// The output of a decoded first field, kept until its frame is complete.
#[derive(Debug, Clone, Copy)]
struct FieldOutput {
    /// `PicOrderCnt()` of the field.
    poc: i64,
    /// An IDR field, or one with memory management operation 5.
    new_sequence: bool,
}

/// A decoded first field waiting for the second field of its frame.
#[derive(Debug)]
struct FirstField<T: H264Sample> {
    sps: Arc<Sps>,
    /// Identity of its frame.
    id: u32,
    parity: Parity,
    frame_num: u32,
    reference: bool,
    /// The frame buffer, shared with the DPB when the field is a reference.
    picture: Arc<Picture<T>>,
    output: FieldOutput,
}

impl<T: H264Sample> FirstField<T> {
    /// Whether the picture `header` starts is this field's second field
    /// (§7.4.1.2.4: opposite parity, the same `frame_num`, not an IDR
    /// picture; JM pairs reference fields with reference fields only).
    fn pairs_with(&self, header: &SliceHeader) -> bool {
        header.structure == Structure::Field(self.parity.opposite())
            && header.frame_num == self.frame_num
            && !header.idr
            && header.reference == self.reference
    }
}

/// The picture being decoded.
#[derive(Debug)]
struct CurrentPicture<T: H264Sample> {
    sps: Arc<Sps>,
    /// The first slice's header (`frame_num`, structure, marking and order
    /// count fields are the same in every slice of a picture).
    header: SliceHeader,
    poc: FramePoc,
    /// Identity of its frame (shared by the two fields of a pair).
    id: u32,
    /// The frame buffer.
    picture: Picture<T>,
    /// `MbToSliceGroupMap` (§8.2.2) of a picture with slice groups.
    slice_groups: Option<Box<[u8]>>,
    /// The first field of the frame, when this picture is its second field.
    first_field: Option<FieldOutput>,
    /// Serial number (`MbState::slice`) of the picture's first slice.
    first_slice: u32,
    /// Deblocking control of each slice, by `MbState::slice - first_slice`.
    slices: Vec<SliceFilter>,
    /// The macroblock rows filtered so far, per coded colour plane.
    deblockers: [RowDeblocker; MAX_PLANES],
    /// Macroblocks decoded so far, per coded colour plane.
    decoded: [usize; MAX_PLANES],
}

/// Colour planes a picture may code separately.
const MAX_PLANES: usize = 3;

impl<T: H264Sample> CurrentPicture<T> {
    /// Whether every macroblock of every coded plane is decoded.
    fn complete(&self) -> bool {
        let mbs = self.sps.picture_mbs(self.header.structure);
        self.decoded[..self.sps.coded_planes()]
            .iter()
            .all(|&n| n == mbs)
    }
}

/// The visible window of a decoded picture.
#[derive(Debug)]
pub(super) struct Visible<T: H264Sample> {
    /// The deblocked picture, shared with the DPB when it is a reference.
    pub picture: Arc<Picture<T>>,
    /// Top-left luma sample of the cropping window.
    pub origin: (usize, usize),
    /// Luma size of the cropping window.
    pub size: (usize, usize),
}

/// A finished frame and what its output order depends on.
#[derive(Debug)]
pub(super) struct Finished<T: H264Sample> {
    pub visible: Visible<T>,
    pub poc: i64,
    pub new_sequence: bool,
    pub reorder_depth: usize,
    pub depths: BitDepths,
}

impl<T: H264Sample> Finished<T> {
    /// The frame `picture` of a sequence with `sps` as an output frame.
    fn new(sps: &Sps, picture: Arc<Picture<T>>, output: FieldOutput) -> Self {
        Self {
            visible: Visible {
                picture,
                origin: (sps.crop.left, sps.crop.top),
                size: sps.visible_size(),
            },
            poc: output.poc,
            new_sequence: output.new_sequence,
            reorder_depth: sps.reorder_frames(),
            depths: sps.depths,
        }
    }

    /// The picture as an output frame.
    pub fn frame(self) -> DecodedFrame<T> {
        let Visible {
            picture,
            origin,
            size,
        } = self.visible;
        DecodedFrame {
            picture,
            origin,
            size,
            poc: self.poc,
            new_sequence: self.new_sequence,
            reorder_depth: self.reorder_depth,
            depths: self.depths,
        }
    }
}

/// The decoding state of one coded video sequence in `T` samples.
#[derive(Debug)]
pub(super) struct Core<T: H264Sample> {
    /// The SPS activated by the latest IDR picture.
    pub active_sps: Option<Arc<Sps>>,
    dpb: Dpb<T>,
    pub pool: PicturePool<T>,
    poc: PocState,
    current: Option<CurrentPicture<T>>,
    /// The last picture, when it is a first field.
    first_field: Option<FirstField<T>>,
    /// Frames completed since the caller last took them, in decoding order.
    finished: Vec<Finished<T>>,
    /// Per-macroblock state of the current picture (reused, not reset:
    /// see `MbState::slice`). A reference picture keeps its states for
    /// direct prediction (`RefPicture::motion`); the next picture then takes
    /// a released one from `motion_pool`.
    mbs: Vec<MbState>,
    /// Every macroblock state vector handed to a reference picture (one
    /// whose only holder is this pool is free again).
    motion_pool: Vec<Arc<Vec<MbState>>>,
    /// Serial number of the last slice started.
    slice_serial: u32,
    residual: Box<Residual<T::Coeff>>,
    extra: Box<MbExtra>,
    dsp: Dsp,
    kernels: InterKernels,
    cavlc: CavlcTables,
    /// Identity of the next decoded frame.
    next_id: u32,
    /// Decoded pictures waiting for output order.
    pub queue: OutputQueue<T>,
    /// Coding-tool analysis, collected only when enabled.
    #[cfg(feature = "analysis")]
    pub stats: Option<StreamStats>,
}

impl<T: H264Sample> Core<T> {
    /// An empty core; its sequence starts with an IDR picture.
    pub fn new() -> Self {
        let dsp = Dsp::new();
        let kernels = InterKernels::new(&dsp);
        Self {
            active_sps: None,
            dpb: Dpb::default(),
            pool: PicturePool::default(),
            poc: PocState::default(),
            current: None,
            first_field: None,
            finished: Vec::new(),
            mbs: Vec::new(),
            motion_pool: Vec::new(),
            slice_serial: 0,
            residual: Box::new(Residual::new()),
            extra: Box::new(MbExtra::new()),
            dsp,
            kernels,
            cavlc: CavlcTables::new(),
            next_id: 1,
            queue: OutputQueue::default(),
            #[cfg(feature = "analysis")]
            stats: None,
        }
    }

    /// Test hook: overwrite every reference picture with `value`.
    #[cfg(test)]
    pub fn corrupt_references(&mut self, value: T) {
        self.dpb.corrupt_pictures(value);
    }

    /// Drop the picture in progress (its slice failed).
    pub fn drop_current(&mut self) {
        self.current = None;
    }

    /// The frames completed since the last call, in decoding order.
    pub fn take_finished(&mut self) -> Vec<Finished<T>> {
        core::mem::take(&mut self.finished)
    }

    /// End of stream: output a first field still waiting for its pair.
    pub fn finish_stream(&mut self) -> Result<()> {
        if let Some(first) = self.first_field.take() {
            self.output_unpaired(first)?;
        }
        Ok(())
    }

    /// One slice, from its NAL unit's RBSP or its partitions; returns whether it completed a picture
    /// (a frame or a field; frames then wait in [`Self::take_finished`]).
    pub fn slice(
        &mut self,
        data: SliceData<'_>,
        (idr, nal_ref_idc): (bool, u32),
        (sps, pps): (&Arc<Sps>, &Arc<Pps>),
    ) -> Result<bool> {
        let (sps, pps) = (sps.clone(), pps.clone());
        let rbsp = match data {
            SliceData::Whole(rbsp) | SliceData::Partitioned { a: rbsp, .. } => rbsp,
        };
        let mut reader = Reader::new(rbsp);
        let header = slice::parse(&mut reader, &sps, &pps, idr, nal_ref_idc)?;
        let source = match data {
            SliceData::Whole(_) => SliceSource::Whole(reader),
            SliceData::Partitioned { residuals, .. } => {
                partition::parse_slice_id(&mut reader)?;
                let [b, c] = residuals.map(|rbsp| residual_reader(rbsp, &pps));
                SliceSource::Partitioned([reader, b?, c?])
            }
        };
        if header.redundant_pic_cnt > 0 {
            // A slice of a redundant coded picture: a decoder may discard
            // it, the primary coded picture being decoded (§7.4.3).
            return Ok(false);
        }
        match &self.current {
            // Any slice may come first (arbitrary slice order, Annex A).
            None => {
                #[cfg(feature = "analysis")]
                if let Some(s) = &mut self.stats {
                    stats::picture(s, &sps, &pps);
                }
                self.start_picture(&sps, &pps, &header)?;
            }
            Some(current) if !header.same_picture(&current.header) => {
                self.current = None;
                return Err(DecodeError::Malformed("picture with missing slices"));
            }
            Some(_) => {}
        }
        #[cfg(feature = "analysis")]
        if let Some(s) = &mut self.stats {
            stats::slice(s, &header, rbsp.len());
        }
        let current = self.current.as_ref().expect("picture in progress");
        if !Arc::ptr_eq(&current.sps, &sps) {
            return Err(DecodeError::Malformed("slice of another picture"));
        }
        self.decode_slice_data(&pps, &header, source)?;
        let current = self.current.as_ref().expect("picture in progress");
        if !current.complete() {
            return Ok(false);
        }
        self.finish_picture()?;
        Ok(true)
    }

    /// Activate parameter sets and references for a new picture (§8.2.1,
    /// §8.2.5.2) and allocate it, or take its first field's frame.
    fn start_picture(&mut self, sps: &Arc<Sps>, pps: &Pps, header: &SliceHeader) -> Result<()> {
        let first = match self.first_field.take() {
            // Parameter sets may be repeated before each field.
            Some(first) if first.pairs_with(header) && *first.sps == **sps => Some(first),
            Some(unpaired) => {
                self.output_unpaired(unpaired)?;
                None
            }
            None => None,
        };
        if header.idr {
            self.dpb
                .start_idr(sps.max_num_ref_frames, sps.max_frame_num());
            self.active_sps = Some(sps.clone());
        } else {
            // Before the first IDR picture: it was lost.
            let active = self.active_sps.as_ref().ok_or(DecodeError::MissingKey)?;
            if **active != **sps {
                return Err(DecodeError::Malformed("SPS changed outside an IDR picture"));
            }
            // A second field repeats its first field's frame_num.
            let missing = if first.is_some() {
                Vec::new()
            } else {
                self.dpb.gap(header.frame_num)
            };
            for &frame_num in &missing {
                self.poc.gap_frame(sps, frame_num);
                self.dpb.insert_gap_frame(frame_num)?;
            }
            if missing.contains(&0) {
                self.dpb.lost_idr();
            }
        }
        let poc = self.poc.derive(sps, header);
        let slice_groups = pps
            .slice_groups
            .as_ref()
            .map(|groups| {
                groups.mb_map(
                    MapUnits::of_frame(sps),
                    header.slice_group_change_cycle,
                    header.structure.is_field(),
                )
            })
            .transpose()?;
        let (id, picture, first_field) = if let Some(first) = first {
            let picture = self.second_field_buffer(first.picture, first.parity)?;
            (first.id, picture, Some(first.output))
        } else {
            let id = self.next_id;
            self.next_id = (self.next_id % MAX_FRAME_ID) + 1;
            (id, self.new_frame(sps)?, None)
        };
        // A picture has at most one slice per macroblock: reset the states
        // (and the serials) only for a new size or before serials could
        // wrap. Otherwise a 2 MB rewrite per 1080p picture, which also
        // evicted the reference blocks from the caches.
        let picture_mbs = sps.picture_mbs(header.structure) * sps.coded_planes();
        let picture_slices = u32::try_from(picture_mbs).unwrap_or(u32::MAX);
        if self.slice_serial > u32::MAX - picture_slices {
            // Serials restart: states of earlier pictures must not look
            // decoded, so pooled ones are not reused.
            self.motion_pool.clear();
            self.mbs.clear();
            self.slice_serial = 0;
        }
        if self.mbs.len() != picture_mbs {
            let states = self.free_motion(picture_mbs, sps.mbs() * sps.coded_planes());
            let previous = core::mem::replace(&mut self.mbs, states);
            if !previous.is_empty() {
                self.motion_pool.push(Arc::new(previous));
            }
        }
        self.current = Some(CurrentPicture {
            sps: sps.clone(),
            header: header.clone(),
            poc,
            id,
            picture,
            first_field,
            slice_groups,
            first_slice: self.slice_serial + 1,
            slices: Vec::new(),
            deblockers: Default::default(),
            decoded: [0; MAX_PLANES],
        });
        Ok(())
    }

    /// A frame buffer for a new frame or first field.
    fn new_frame(&mut self, sps: &Sps) -> Result<Picture<T>> {
        let mut picture = self.pool.take_with_format(
            (sps.width_mbs * 16, sps.height_mbs * 16),
            sps.format,
            REFERENCE_BORDER,
        )?;
        if sps.monochrome {
            // 4:0:0 is output with mid-grey chroma (`1 << (BitDepthC − 1)`).
            let grey = T::from_i32(sps.depths.chroma.mid());
            for plane in [PlaneIndex::U, PlaneIndex::V] {
                picture.plane_mut(plane).view_mut().fill(grey);
            }
        }
        Ok(picture)
    }

    /// The frame buffer of a second field: its first field's, taken back,
    /// or a copy of the first field's rows when the DPB still holds it (as
    /// a reference the second field may predict from).
    fn second_field_buffer(
        &mut self,
        first: Arc<Picture<T>>,
        parity: Parity,
    ) -> Result<Picture<T>> {
        let shared = match self.pool.reclaim(first) {
            Ok(picture) => return Ok(picture),
            Err(shared) => shared,
        };
        let mut picture = self.pool.take_with_format(
            (shared.width(), shared.height()),
            shared.chroma_format(),
            REFERENCE_BORDER,
        )?;
        for plane in PlaneIndex::ALL {
            let source = shared.plane(plane).view().every_other_row(parity.index());
            picture
                .plane_mut(plane)
                .view_mut()
                .every_other_row(parity.index())
                .copy_from(&source);
        }
        Ok(picture)
    }

    /// Output a field whose frame has no second field, with mid-grey rows
    /// in place of the missing field (as JM).
    fn output_unpaired(&mut self, first: FirstField<T>) -> Result<()> {
        let mut picture = match self.pool.reclaim(first.picture) {
            Ok(picture) => picture,
            Err(shared) => {
                let mut copy = self.pool.take_with_format(
                    (shared.width(), shared.height()),
                    shared.chroma_format(),
                    REFERENCE_BORDER,
                )?;
                for plane in PlaneIndex::ALL {
                    copy.plane_mut(plane)
                        .view_mut()
                        .copy_from(&shared.plane(plane).view());
                }
                copy
            }
        };
        let missing = first.parity.opposite().index();
        let depths = [
            first.sps.depths.luma,
            first.sps.depths.chroma,
            first.sps.depths.chroma,
        ];
        for (plane, depth) in PlaneIndex::ALL.into_iter().zip(depths) {
            picture
                .plane_mut(plane)
                .view_mut()
                .every_other_row(missing)
                .fill(T::from_i32(depth.mid()));
        }
        let picture = self.pool.share(picture);
        self.finished
            .push(Finished::new(&first.sps, picture, first.output));
        Ok(())
    }

    /// Record a new slice of the current picture (its deblocking control)
    /// and return its serial number.
    fn begin_slice(&mut self, pps: &Pps, header: &SliceHeader) -> Result<u32> {
        let current = self.current.as_mut().expect("picture in progress");
        current.slices.push(SliceFilter::new(
            header.deblock,
            pps.chroma_qp_offset,
            header.slice_type == SliceType::B,
            header.switching.is_some(),
            current.sps.depths,
        ));
        // Within the picture's budget (`start_picture`) unless the stream
        // has more slices than macroblocks.
        let coded_planes = current.sps.coded_planes();
        if current.slices.len() > current.sps.picture_mbs(current.header.structure) * coded_planes {
            return Err(DecodeError::Malformed("more slices than macroblocks"));
        }
        self.slice_serial += 1;
        Ok(self.slice_serial)
    }

    /// Decode the macroblocks of one slice into the current picture.
    fn decode_slice_data(
        &mut self,
        pps: &Pps,
        header: &SliceHeader,
        source: SliceSource<'_>,
    ) -> Result<()> {
        let slice_no = self.begin_slice(pps, header)?;
        let current = self.current.as_mut().expect("picture in progress");
        let sps = current.sps.clone();
        let structure = current.header.structure;
        let picture_mbs = sps.picture_mbs(structure);
        let poc = current.poc.value();
        let coded_size = (current.picture.width(), current.picture.height());
        let list_picture = ListPicture {
            frame_num: header.frame_num,
            structure,
            poc,
        };
        let lists = ref_lists(&self.dpb, header, &list_picture, coded_size)?;
        let ref_ids = lists.each_ref().map(|list| ids_of(list));
        let kind = Current::of(structure, header.mbaff);
        let size = (sps.width_mbs, picture_mbs);
        let colour_plane = header.colour_plane;
        let plane = (colour_plane, sps.coded_planes());
        let direct = direct_context(header, &sps, (&lists, &ref_ids), (kind, poc), (size, plane))?;
        let frame_weighting = weighting(
            header.weights.as_ref(),
            (header.slice_type, pps),
            &lists,
            poc,
        );
        let dequantiser = dequantiser(&sps, pps, colour_plane);
        // MBAFF: the field macroblocks' references by parity.
        let field_mbs = header
            .mbaff
            .then(|| FieldMbLists::new(&lists, header, [current.poc.top, current.poc.bottom]));
        let field_direct = field_mbs
            .as_ref()
            .map(|fields| fields.direct(header, &sps, size))
            .transpose()?;
        let params = SliceParams {
            slice_type: header.slice_type,
            num_ref_idx_active: header.num_ref_idx_active,
            ref_ids: [&ref_ids[0], &ref_ids[1]],
            direct: direct.as_ref(),
            transform_8x8_mode: pps.transform_8x8_mode,
            constrained_intra_pred: pps.constrained_intra_pred,
            chroma: sps.chroma,
            switching: header.switching.is_some(),
            dequantiser: &dequantiser,
            qp_bd_offset: 6 * sps.depths.luma.above_eight() as i32,
            transform_bypass: sps.transform_bypass,
            pcm_bits: [sps.depths.luma.bits(), sps.depths.chroma.bits()],
        };
        let recon = ReconContext {
            dsp: &self.dsp,
            inter: InterContext::new(
                (&self.kernels, &self.dsp),
                ([&lists[0], &lists[1]], structure),
                frame_weighting,
                (sps.chroma, colour_plane),
                sps.depths,
            ),
            constrained_intra_pred: pps.constrained_intra_pred,
            chroma: sps.chroma,
            switching: header.switching,
            depths: sps.depths,
        };
        let plane_mbs = picture_mbs * colour_plane..picture_mbs * (colour_plane + 1);
        let pair_fields = field_mbs
            .as_ref()
            .zip(field_direct.as_ref())
            .map(|(fields, directs)| {
                fields.contexts(
                    &params,
                    directs,
                    (&self.kernels, &self.dsp),
                    (header, pps, &sps),
                )
            });
        let mut decoder = SliceDecoder {
            params,
            recon,
            pair_fields,
            planes: picture_planes(planes(&mut current.picture, colour_plane), structure),
            slices: &current.slices,
            first_slice: current.first_slice,
            deblocker: &mut current.deblockers[colour_plane],
            mbs: &mut self.mbs[plane_mbs],
            slice_groups: current.slice_groups.as_deref(),
            width_mbs: sps.width_mbs,
            field: structure.is_field(),
            slice_no,
            residual: &mut self.residual,
            extra: &mut self.extra,
            qp: header.qp,
            decoded: 0,
            #[cfg(feature = "analysis")]
            stats: self.stats.as_mut(),
            #[cfg(feature = "analysis")]
            cabac: pps.cabac,
        };
        decode_macroblocks(&mut decoder, source, (pps, header, sps.chroma), &self.cavlc)?;
        current.decoded[colour_plane] += decoder.decoded;
        Ok(())
    }

    /// Deblock and mark the completed current picture; a frame (or the
    /// second field completing one) becomes due for output, a first field
    /// waits for its pair.
    fn finish_picture(&mut self) -> Result<()> {
        let CurrentPicture {
            sps,
            header,
            poc,
            id,
            mut picture,
            first_field,
            first_slice,
            slices,
            mut deblockers,
            ..
        } = self.current.take().expect("picture in progress");
        #[cfg(feature = "analysis")]
        if let Some(s) = &mut self.stats {
            // Its records become the completed picture's.
            s.finish();
        }
        let structure = header.structure;
        let last_row = sps.picture_height_mbs(structure) - 1;
        let picture_mbs = sps.picture_mbs(structure);
        for (colour_plane, deblocker) in deblockers.iter_mut().enumerate().take(sps.coded_planes())
        {
            let ctx = DeblockContext {
                dsp: &self.dsp,
                mbs: &self.mbs[picture_mbs * colour_plane..picture_mbs * (colour_plane + 1)],
                width_mbs: sps.width_mbs,
                first_slice,
                slices: &slices,
                chroma: sps.chroma,
                depths: sps.depths,
                field: structure.is_field(),
            };
            if header.mbaff {
                // Filtered whole, pair by pair (`deblock::filter_mbaff`).
                filter_mbaff(&mut planes(&mut picture, colour_plane), &ctx);
            } else {
                let planes = &mut picture_planes(planes(&mut picture, colour_plane), structure);
                deblocker.filter_through(last_row, planes, &ctx);
            }
        }
        let reset = header.marking.resets();
        if header.reference {
            picture.extend_borders();
        }
        // Shared even when not a reference: the pool takes the buffer back
        // once the caller is done with it.
        let picture = self.pool.share(picture);
        let poc = if reset { poc.rebased() } else { poc };
        if header.reference {
            let motion = Arc::new(core::mem::take(&mut self.mbs));
            self.motion_pool.push(motion.clone());
            let reference = NewReference {
                frame_id: id,
                structure,
                picture: picture.clone(),
                poc: [poc.top, poc.bottom],
                motion,
                mbaff: header.mbaff,
            };
            self.dpb
                .mark(header.frame_num, reference, &header.marking)?;
        }
        self.poc.picture_done(&sps, &header, poc, reset);
        let output = FieldOutput {
            poc: poc.value(),
            new_sequence: header.idr || reset,
        };
        match (structure, first_field) {
            (Structure::Frame, _) => self.finished.push(Finished::new(&sps, picture, output)),
            (Structure::Field(_), Some(first)) => {
                let frame = FieldOutput {
                    poc: first.poc.min(output.poc),
                    new_sequence: first.new_sequence || output.new_sequence,
                };
                self.finished.push(Finished::new(&sps, picture, frame));
            }
            (Structure::Field(parity), None) => {
                self.first_field = Some(FirstField {
                    sps,
                    id,
                    parity,
                    frame_num: header.frame_num,
                    reference: header.reference,
                    picture,
                    output,
                });
            }
        }
        Ok(())
    }

    /// A macroblock state vector of `mbs` entries: one no reference
    /// picture holds any more, else a new one. Released vectors of other
    /// sizes than a frame's or a field's of `frame_mbs` are dropped.
    fn free_motion(&mut self, mbs: usize, frame_mbs: usize) -> Vec<MbState> {
        self.motion_pool.retain(|m| {
            Arc::strong_count(m) > 1 || m.len() == frame_mbs || m.len() == frame_mbs / 2
        });
        if let Some(index) = self
            .motion_pool
            .iter()
            .position(|m| Arc::strong_count(m) == 1 && m.len() == mbs)
        {
            let free = self.motion_pool.swap_remove(index);
            if let Ok(states) = Arc::try_unwrap(free) {
                return states;
            }
        }
        vec![MbState::EMPTY; mbs]
    }
}

/// The macroblock loop of a slice over its data, by entropy coder.
fn decode_macroblocks<T: H264Sample>(
    decoder: &mut SliceDecoder<'_, T>,
    source: SliceSource<'_>,
    (pps, header, chroma): (&Pps, &SliceHeader, ChromaArray),
    cavlc: &CavlcTables,
) -> Result<()> {
    let first = header.first_mb;
    match source {
        SliceSource::Whole(reader) if pps.cabac => {
            let data = reader.remaining_bytes();
            let init = (header.slice_type, header.cabac_init_idc);
            let engine = Cabac::new(data, init, header.qp, chroma.is_444())?;
            let mut syntax = CabacSyntax::new(engine, chroma);
            if header.mbaff {
                decoder.cabac_pairs(&mut syntax, first)?;
            } else {
                decoder.cabac(&mut syntax, first)?;
            }
        }
        SliceSource::Whole(reader) => {
            let mut syntax = CavlcSyntax::<T>::new(reader, cavlc, chroma);
            if header.mbaff {
                decoder.cavlc_pairs(&mut syntax, first)?;
            } else {
                decoder.cavlc(&mut syntax, first)?;
            }
        }
        SliceSource::Partitioned(_) if header.mbaff => {
            return Err(DecodeError::Unsupported(
                "H264 MBAFF frames with data partitioning",
            ));
        }
        SliceSource::Partitioned(readers) => {
            // Data partitioning is the Extended profile's: 4:2:0 (A.2.3).
            let cip = pps.constrained_intra_pred;
            let mut syntax = PartitionedCavlc::<T>::new(readers, cavlc, cip);
            decoder.cavlc(&mut syntax, header.first_mb)?;
        }
    }
    Ok(())
}

/// The coded data of one slice: a slice NAL unit's RBSP, or partition A's
/// and those of the residual partitions B and C that are present
/// (§7.3.2.9, CAVLC only: the Extended profile has no CABAC).
#[derive(Debug, Clone, Copy)]
pub(super) enum SliceData<'a> {
    Whole(&'a [u8]),
    Partitioned {
        a: &'a [u8],
        residuals: [Option<&'a [u8]>; 2],
    },
}

/// Where a slice's data is read, after its header.
enum SliceSource<'a> {
    Whole(Reader<'a>),
    /// Partitions A, B and C, each at its slice data.
    Partitioned([Reader<'a>; 3]),
}

/// A reader at the slice data of residual partition `rbsp`, or over no
/// data for an absent partition.
fn residual_reader<'a>(rbsp: Option<&'a [u8]>, pps: &Pps) -> Result<Reader<'a>> {
    let Some(rbsp) = rbsp else {
        return Ok(Reader::new(&[]));
    };
    let mut reader = Reader::new(rbsp);
    partition::parse_residual_header(&mut reader, pps)?;
    Ok(reader)
}

/// The weighting of a slice's predictions (§8.4.2.3).
fn weighting<'a, T: H264Sample>(
    table: Option<&'a WeightTable>,
    (slice_type, pps): (SliceType, &Pps),
    lists: &[Vec<Option<RefPicture<T>>>; 2],
    poc: i64,
) -> Weighting<'a> {
    match (table, slice_type) {
        (Some(table), _) => Weighting::Explicit(table),
        (None, SliceType::B) if pps.weighted_bipred == BipredWeights::Implicit => {
            Weighting::Implicit(implicit_weights(lists, poc))
        }
        _ => Weighting::Default,
    }
}

/// A slice's two reference lists.
type Lists<T> = [Vec<Option<RefPicture<T>>>; 2];
/// The identities of their pictures.
type ListIds = [Vec<Option<u32>>; 2];

/// The references of the field macroblocks of an MBAFF slice, by parity
/// (§8.4.2.1): their lists, the lists' identities, the current frame's
/// field order counts and the explicit weights by field reference index
/// (`refIdx >> 1`, §8.4.2.3).
struct FieldMbLists<T: H264Sample> {
    lists: [Lists<T>; 2],
    ids: [ListIds; 2],
    pocs: [i64; 2],
    weights: Option<WeightTable>,
}

impl<T: H264Sample> FieldMbLists<T> {
    fn new(frame_lists: &Lists<T>, header: &SliceHeader, pocs: [i64; 2]) -> Self {
        let lists = Parity::BOTH.map(|p| frame_lists.each_ref().map(|list| field_list(list, p)));
        let ids = lists
            .each_ref()
            .map(|by_list| by_list.each_ref().map(|list| ids_of(list)));
        Self {
            lists,
            ids,
            pocs,
            weights: header.weights.as_ref().map(WeightTable::by_field),
        }
    }

    /// Each parity's direct prediction.
    fn direct(
        &self,
        header: &SliceHeader,
        sps: &Sps,
        size: (usize, usize),
    ) -> Result<[Option<DirectContext<'_>>; 2]> {
        let [top, bottom] = [0, 1].map(|p| {
            direct_context(
                header,
                sps,
                (&self.lists[p], &self.ids[p]),
                (Current::PairField(Parity::BOTH[p]), self.pocs[p]),
                // One coded plane (`slice::check_mbaff`).
                (size, (0, 1)),
            )
        });
        Ok([top?, bottom?])
    }

    /// The field macroblocks' slice parameters and reconstruction, from
    /// the frame macroblocks' `frame` ones.
    fn contexts<'a>(
        &'a self,
        frame: &SliceParams<'a>,
        directs: &'a [Option<DirectContext<'a>>; 2],
        (kernels, dsp): (&'a InterKernels, &'a Dsp),
        (header, pps, sps): (&SliceHeader, &Pps, &Sps),
    ) -> FieldMbContexts<'a, T> {
        let params = |p: usize| SliceParams {
            num_ref_idx_active: header.num_ref_idx_active.map(|n| 2 * n),
            ref_ids: [&self.ids[p][0], &self.ids[p][1]],
            direct: directs[p].as_ref(),
            ..*frame
        };
        let recon = |p: usize| ReconContext {
            dsp,
            inter: InterContext::new(
                (kernels, dsp),
                (
                    [&self.lists[p][0], &self.lists[p][1]],
                    Structure::Field(Parity::BOTH[p]),
                ),
                weighting(
                    self.weights.as_ref(),
                    (header.slice_type, pps),
                    &self.lists[p],
                    self.pocs[p],
                ),
                // MBAFF frames are 4:2:0 or 4:0:0 here (`slice::parse`).
                (sps.chroma, 0),
                sps.depths,
            ),
            constrained_intra_pred: pps.constrained_intra_pred,
            chroma: sps.chroma,
            depths: sps.depths,
            switching: header.switching,
        };
        FieldMbContexts {
            params: [params(0), params(1)],
            recon: [recon(0), recon(1)],
        }
    }
}

/// The identities of a list's pictures (`None`: no picture).
fn ids_of<T: H264Sample>(list: &[Option<RefPicture<T>>]) -> Vec<Option<u32>> {
    list.iter().map(|r| r.as_ref().map(|r| r.id)).collect()
}

/// The direct prediction of a B slice's `kind` of picture or macroblock
/// with order count `poc` from `lists` (identities `ids`); `None` for other
/// slices.
fn direct_context<'a, T: H264Sample>(
    header: &SliceHeader,
    sps: &Sps,
    (lists, ids): (&'a Lists<T>, &'a ListIds),
    (kind, poc): (Current, i64),
    (size, plane): ((usize, usize), (usize, usize)),
) -> Result<Option<DirectContext<'a>>> {
    (header.slice_type == SliceType::B)
        .then(|| {
            DirectContext::new(
                (header.direct_spatial, sps.direct_8x8_inference),
                [&lists[0], &lists[1]],
                &ids[0],
                (kind, poc),
                size,
                plane,
            )
        })
        .transpose()
}

/// The scaling of the active parameter sets (§7.4.2.1.1.1, §8.5.9) for
/// colour plane `colour_plane` (0 unless coded separately).
fn dequantiser(sps: &Sps, pps: &Pps, colour_plane: usize) -> Dequantiser {
    let scaling = match &pps.scaling {
        Some(parsed) => scaling::resolve(parsed, sps.scaling_present.then_some(&sps.scaling)),
        None => sps.scaling,
    };
    let depths = [sps.depths.luma, sps.depths.chroma];
    // The 8x8 lists of each component that codes with them.
    let lists_8x8 = scaling::LISTS_8X8_PER_PLANE
        * if sps.chroma.is_444() || sps.separate_planes {
            MAX_PLANES
        } else {
            1
        };
    Dequantiser::new(
        LevelScale::new(&scaling, lists_8x8),
        pps.chroma_qp_offset,
        depths,
        (sps.chroma, colour_plane),
    )
}

/// The reference lists of a slice of picture `current` (§8.2.4), whose
/// pictures must have its `coded_size`.
fn ref_lists<T: H264Sample>(
    dpb: &Dpb<T>,
    header: &SliceHeader,
    current: &ListPicture,
    coded_size: (usize, usize),
) -> Result<[Vec<Option<RefPicture<T>>>; 2]> {
    let lists = match header.slice_type {
        SliceType::P => [
            dpb.ref_list(
                current,
                header.num_ref_idx_active[0],
                &header.modifications[0],
            )?,
            Vec::new(),
        ],
        SliceType::B => {
            dpb.b_ref_lists(current, header.num_ref_idx_active, &header.modifications)?
        }
        SliceType::I | SliceType::Si => [Vec::new(), Vec::new()],
    };
    if lists
        .iter()
        .flatten()
        .flatten()
        .any(|r| (r.picture.width(), r.picture.height()) != coded_size)
    {
        return Err(DecodeError::Malformed("reference of another size"));
    }
    Ok(lists)
}

/// Implicit bi-prediction weights `(w0, w1)` of every pair of list
/// entries (§8.4.2.3.1, eqs. 8-302..8-304): from the order count distances,
/// equal where either entry is long-term, the distances coincide or the
/// weights would leave their range.
fn implicit_weights<T: H264Sample>(
    lists: &[Vec<Option<RefPicture<T>>>; 2],
    poc: i64,
) -> Vec<Vec<[i32; 2]>> {
    const EQUAL: [i32; 2] = [1 << IMPLICIT_LOG2_DENOM; 2];
    const MIN_W1: i32 = -64;
    const MAX_W1: i32 = 128;
    let total = 1 << (IMPLICIT_LOG2_DENOM + 1);
    lists[0]
        .iter()
        .map(|pic0| {
            lists[1]
                .iter()
                .map(|pic1| {
                    let (Some(pic0), Some(pic1)) = (pic0, pic1) else {
                        return EQUAL;
                    };
                    if pic0.long_term || pic1.long_term {
                        return EQUAL;
                    }
                    match dist_scale_factor(poc, pic0.poc, pic1.poc).map(|f| f >> 2) {
                        Some(w1) if (MIN_W1..=MAX_W1).contains(&w1) => [total - w1, w1],
                        _ => EQUAL,
                    }
                })
                .collect()
        })
        .collect()
}
