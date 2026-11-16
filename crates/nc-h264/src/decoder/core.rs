//! The decoding process of one sample type (`docs/handoff/high-bit-depth.md`):
//! [`Core`] holds a coded video sequence's references, pictures and
//! scratch in `T` samples (`u8` for 8-bit streams, `u16` deeper ones) and
//! decodes its slices; [`super::Decoder`] holds the parameter sets and one
//! core for the sequence's depth.

use std::sync::Arc;

#[cfg(feature = "analysis")]
use nc_common::stream_stats::StreamStats;
use nc_common::{BitDepths, Picture, PlaneIndex, ShownPicture};
use nc_dsp::Dsp;

use super::cabac::Cabac;
use super::cabac_syntax::CabacSyntax;
use super::cavlc::{CavlcSyntax, CavlcTables};
use super::deblock::{DeblockContext, RowDeblocker, SliceFilter};
use super::direct::{DirectContext, dist_scale_factor};
use super::dpb::{Dpb, RefPicture};
use super::inter::{IMPLICIT_LOG2_DENOM, InterContext, InterKernels, Weighting};
use super::mb::{MbExtra, SliceParams};
use super::mbstate::MbState;
use super::output::{DecodedFrame, OutputQueue};
use super::poc::{FramePoc, PocState};
use super::pool::PicturePool;
use super::pps::{BipredWeights, Pps};
use super::reader::Reader;
use super::recon::ReconContext;
use super::sample::H264Sample;
use super::scaling::{self, LevelScale};
use super::slice::{self, SliceHeader, SliceType};
use super::slice_data::SliceDecoder;
use super::sps::Sps;
#[cfg(feature = "analysis")]
use super::stats;
use super::transform::{Dequantiser, Residual};
use super::{DecodeError, Planes, Result};

/// Picture border kept for motion compensation (whole-window reads).
const REFERENCE_BORDER: usize = 64;

/// The views of `picture`'s planes.
fn planes<T: H264Sample>(picture: &mut Picture<T>) -> Planes<'_, T> {
    let [y, u, v] = picture.planes_mut();
    [y.view_mut(), u.view_mut(), v.view_mut()]
}

/// The picture being decoded.
#[derive(Debug)]
struct CurrentPicture<T: H264Sample> {
    sps: Arc<Sps>,
    /// The first slice's header (`frame_num`, marking and order count fields
    /// are the same in every slice of a picture).
    header: SliceHeader,
    poc: FramePoc,
    id: u32,
    picture: Picture<T>,
    /// Serial number (`MbState::slice`) of the picture's first slice.
    first_slice: u32,
    /// Deblocking control of each slice, by `MbState::slice - first_slice`.
    slices: Vec<SliceFilter>,
    /// The macroblock rows filtered so far.
    deblocker: RowDeblocker,
    decoded: usize,
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

impl<T: H264Sample> Visible<T> {
    pub fn view(&self) -> Option<ShownPicture<'_, T>> {
        ShownPicture::window(&self.picture, self.origin, self.size)
    }

    /// The window copied into a picture from `pool`.
    pub fn copy(&self, pool: &mut PicturePool<T>) -> Result<Picture<T>> {
        let view = self
            .view()
            .ok_or(DecodeError::Malformed("frame cropping window"))?;
        let mut picture = pool.take(view.width(), view.height(), 0)?;
        view.copy_to(&mut picture);
        Ok(picture)
    }
}

/// A finished picture and what its output order depends on.
#[derive(Debug)]
pub(super) struct Finished<T: H264Sample> {
    pub visible: Visible<T>,
    pub poc: i64,
    pub new_sequence: bool,
    pub reorder_depth: usize,
    pub depths: BitDepths,
}

impl<T: H264Sample> Finished<T> {
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
    /// Identity of the next decoded picture.
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

    /// One slice NAL unit's RBSP.
    pub fn slice(
        &mut self,
        rbsp: &[u8],
        (idr, nal_ref_idc): (bool, u32),
        (sps, pps): (&Arc<Sps>, &Arc<Pps>),
    ) -> Result<Option<Finished<T>>> {
        let (sps, pps) = (sps.clone(), pps.clone());
        let mut reader = Reader::new(rbsp);
        let header = slice::parse(&mut reader, &sps, &pps, idr, nal_ref_idc)?;
        if header.first_mb == 0 {
            if self.current.take().is_some() {
                return Err(DecodeError::Malformed("picture with missing slices"));
            }
            #[cfg(feature = "analysis")]
            if let Some(s) = &mut self.stats {
                stats::picture(s, &sps, &pps);
            }
            self.start_picture(&sps, &header)?;
        }
        #[cfg(feature = "analysis")]
        if let Some(s) = &mut self.stats {
            stats::slice(s, &header, rbsp.len());
        }
        let current = self.current.as_mut().ok_or(DecodeError::Malformed(
            "slice of a picture without its first slice",
        ))?;
        if !Arc::ptr_eq(&current.sps, &sps)
            || current.header.frame_num != header.frame_num
            || current.header.idr != header.idr
        {
            return Err(DecodeError::Malformed("slice of another picture"));
        }
        self.decode_slice_data(&pps, &header, reader)?;
        let current = self.current.as_ref().expect("picture in progress");
        if current.decoded < current.sps.mbs() {
            return Ok(None);
        }
        self.finish_picture().map(Some)
    }

    /// Activate parameter sets and references for a new picture (§8.2.1,
    /// §8.2.5.2) and allocate it.
    fn start_picture(&mut self, sps: &Arc<Sps>, header: &SliceHeader) -> Result<()> {
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
            let missing = self.dpb.gap(header.frame_num);
            for &frame_num in &missing {
                self.poc.gap_frame(sps, frame_num);
                self.dpb.insert_gap_frame(frame_num)?;
            }
            if missing.contains(&0) {
                self.dpb.lost_idr();
            }
        }
        let poc = self.poc.derive(sps, header);
        let mut picture =
            self.pool
                .take(sps.width_mbs * 16, sps.height_mbs * 16, REFERENCE_BORDER)?;
        if sps.monochrome {
            // 4:0:0 is output with mid-grey chroma (`1 << (BitDepthC − 1)`).
            let grey = T::from_i32(sps.depths.chroma.mid());
            for plane in [PlaneIndex::U, PlaneIndex::V] {
                picture.plane_mut(plane).view_mut().fill(grey);
            }
        }
        // A picture has at most one slice per macroblock: reset the states
        // (and the serials) only for a new size or before serials could
        // wrap. Otherwise a 2 MB rewrite per 1080p picture, which also
        // evicted the reference blocks from the caches.
        let picture_slices = u32::try_from(sps.mbs()).unwrap_or(u32::MAX);
        if self.slice_serial > u32::MAX - picture_slices {
            // Serials restart: states of earlier pictures must not look
            // decoded, so pooled ones are not reused.
            self.motion_pool.clear();
            self.mbs.clear();
            self.slice_serial = 0;
        }
        if self.mbs.len() != sps.mbs() {
            self.mbs = self.free_motion(sps.mbs());
        }
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1).max(1);
        self.current = Some(CurrentPicture {
            sps: sps.clone(),
            header: header.clone(),
            poc,
            id,
            picture,
            first_slice: self.slice_serial + 1,
            slices: Vec::new(),
            deblocker: RowDeblocker::default(),
            decoded: 0,
        });
        Ok(())
    }

    /// Decode the macroblocks of one slice into the current picture.
    fn decode_slice_data(
        &mut self,
        pps: &Pps,
        header: &SliceHeader,
        reader: Reader<'_>,
    ) -> Result<()> {
        let current = self.current.as_mut().expect("picture in progress");
        let sps = current.sps.clone();
        current.slices.push(SliceFilter::new(
            header.deblock,
            pps.chroma_qp_offset,
            header.slice_type == SliceType::B,
            sps.depths,
        ));
        // Within the picture's budget (`start_picture`) unless the stream
        // has more slices than macroblocks.
        if current.slices.len() > current.sps.mbs() {
            return Err(DecodeError::Malformed("more slices than macroblocks"));
        }
        self.slice_serial += 1;
        let slice_no = self.slice_serial;
        let poc = current.poc.value();
        let coded_size = (current.picture.width(), current.picture.height());
        let lists = ref_lists(&self.dpb, header, poc, coded_size)?;
        let ref_ids: [Vec<Option<u32>>; 2] = lists
            .each_ref()
            .map(|list| list.iter().map(|r| r.as_ref().map(|r| r.id)).collect());
        let direct = (header.slice_type == SliceType::B)
            .then(|| {
                DirectContext::new(
                    header.direct_spatial,
                    sps.direct_8x8_inference,
                    [&lists[0], &lists[1]],
                    &ref_ids[0],
                    poc,
                    sps.mbs(),
                )
            })
            .transpose()?;
        let weighting = weighting(header, pps, &lists, poc);
        let dequantiser = dequantiser(&sps, pps);
        let params = SliceParams {
            slice_type: header.slice_type,
            num_ref_idx_active: header.num_ref_idx_active,
            ref_ids: [&ref_ids[0], &ref_ids[1]],
            direct: direct.as_ref(),
            transform_8x8_mode: pps.transform_8x8_mode,
            constrained_intra_pred: pps.constrained_intra_pred,
            monochrome: sps.monochrome,
            dequantiser: &dequantiser,
            qp_bd_offset: 6 * sps.depths.luma.above_eight() as i32,
            pcm_bits: [sps.depths.luma.bits(), sps.depths.chroma.bits()],
        };
        let recon = ReconContext {
            dsp: &self.dsp,
            inter: InterContext::new(
                (&self.kernels, &self.dsp),
                [&lists[0], &lists[1]],
                weighting,
                sps.monochrome,
                sps.depths,
            ),
            constrained_intra_pred: pps.constrained_intra_pred,
            monochrome: sps.monochrome,
            depths: sps.depths,
        };
        let mut decoder = SliceDecoder {
            params,
            recon,
            planes: planes(&mut current.picture),
            slices: &current.slices,
            first_slice: current.first_slice,
            deblocker: &mut current.deblocker,
            mbs: &mut self.mbs,
            width_mbs: sps.width_mbs,
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
        if pps.cabac {
            let engine = Cabac::new(
                reader.remaining_bytes(),
                header.slice_type,
                header.cabac_init_idc,
                header.qp,
            )?;
            decoder.cabac(&mut CabacSyntax::new(engine), header.first_mb)?;
        } else {
            decoder.cavlc(&mut CavlcSyntax::new(reader, &self.cavlc), header.first_mb)?;
        }
        current.decoded += decoder.decoded;
        Ok(())
    }

    /// Deblock, mark and output the completed current picture.
    fn finish_picture(&mut self) -> Result<Finished<T>> {
        let CurrentPicture {
            sps,
            header,
            poc,
            id,
            mut picture,
            first_slice,
            slices,
            mut deblocker,
            ..
        } = self.current.take().expect("picture in progress");
        #[cfg(feature = "analysis")]
        if let Some(s) = &mut self.stats {
            // Its records become the completed picture's.
            s.finish();
        }
        let ctx = DeblockContext {
            dsp: &self.dsp,
            mbs: &self.mbs,
            width_mbs: sps.width_mbs,
            first_slice,
            slices: &slices,
            monochrome: sps.monochrome,
            depths: sps.depths,
        };
        deblocker.filter_through(sps.height_mbs - 1, &mut planes(&mut picture), &ctx);
        let reset = header.marking.resets();
        if header.reference {
            picture.extend_borders();
        }
        // Shared even when not a reference: the pool takes the buffer back
        // once the caller is done with it.
        let picture = self.pool.share(picture);
        let output_poc = if reset { 0 } else { poc.value() };
        if header.reference {
            let motion = Arc::new(core::mem::take(&mut self.mbs));
            self.motion_pool.push(motion.clone());
            let reference = RefPicture {
                id,
                picture: picture.clone(),
                poc: output_poc,
                long_term: false,
                motion,
            };
            self.dpb
                .mark(header.frame_num, reference, &header.marking)?;
        }
        self.poc.picture_done(&sps, &header, poc, reset);
        Ok(Finished {
            visible: Visible {
                picture,
                origin: (sps.crop.left, sps.crop.top),
                size: sps.visible_size(),
            },
            poc: output_poc,
            new_sequence: header.idr || reset,
            reorder_depth: sps
                .max_num_reorder_frames
                .map_or_else(|| sps.dpb_frames(), |frames| frames as usize),
            depths: sps.depths,
        })
    }

    /// A macroblock state vector of `mbs` entries: one no reference
    /// picture holds any more, else a new one.
    fn free_motion(&mut self, mbs: usize) -> Vec<MbState> {
        // Released vectors of another size are dropped.
        self.motion_pool
            .retain(|m| Arc::strong_count(m) > 1 || m.len() == mbs);
        if let Some(index) = self
            .motion_pool
            .iter()
            .position(|m| Arc::strong_count(m) == 1)
        {
            let free = self.motion_pool.swap_remove(index);
            if let Ok(states) = Arc::try_unwrap(free) {
                return states;
            }
        }
        vec![MbState::EMPTY; mbs]
    }
}

/// The weighting of a slice's predictions (§8.4.2.3).
fn weighting<'a, T: H264Sample>(
    header: &'a SliceHeader,
    pps: &Pps,
    lists: &[Vec<Option<RefPicture<T>>>; 2],
    poc: i64,
) -> Weighting<'a> {
    match (&header.weights, header.slice_type) {
        (Some(table), _) => Weighting::Explicit(table),
        (None, SliceType::B) if pps.weighted_bipred == BipredWeights::Implicit => {
            Weighting::Implicit(implicit_weights(lists, poc))
        }
        _ => Weighting::Default,
    }
}

/// The scaling of the active parameter sets (§7.4.2.1.1.1, §8.5.9).
fn dequantiser(sps: &Sps, pps: &Pps) -> Dequantiser {
    let scaling = match &pps.scaling {
        Some(parsed) => scaling::resolve(parsed, sps.scaling_present.then_some(&sps.scaling)),
        None => sps.scaling,
    };
    let depths = [sps.depths.luma, sps.depths.chroma];
    Dequantiser::new(LevelScale::new(&scaling), pps.chroma_qp_offset, depths)
}

/// The reference lists of a slice of the picture with order count `poc`
/// (§8.2.4), whose pictures must have its `coded_size`.
fn ref_lists<T: H264Sample>(
    dpb: &Dpb<T>,
    header: &SliceHeader,
    poc: i64,
    coded_size: (usize, usize),
) -> Result<[Vec<Option<RefPicture<T>>>; 2]> {
    let lists = match header.slice_type {
        SliceType::P => [
            dpb.ref_list(
                header.frame_num,
                header.num_ref_idx_active[0],
                &header.modifications[0],
            )?,
            Vec::new(),
        ],
        SliceType::B => dpb.b_ref_lists(
            header.frame_num,
            poc,
            header.num_ref_idx_active,
            &header.modifications,
        )?,
        SliceType::I => [Vec::new(), Vec::new()],
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
