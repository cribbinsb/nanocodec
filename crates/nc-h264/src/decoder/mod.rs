//! H.264 decoder for the Constrained High profile (Annex A.2.4.2), 8-bit:
//! progressive I and P slices in 4:2:0 or 4:0:0, CAVLC and CABAC, 8x8
//! transforms and intra prediction, scaling matrices, explicit weighted
//! prediction, any number of slices per picture (in raster order),
//! long-term references, memory management operations and `frame_num`
//! gaps. That covers the Constrained Baseline and Main-without-B streams a
//! Constrained High decoder must also accept. Everything else (B, SP/SI
//! slices, interlace, FMO/ASO, redundant pictures, data partitioning,
//! other chroma formats and bit depths) is rejected as unsupported.
//!
//! Module map:
//! - parameter sets and headers: `sps`, `pps`, `scaling`, `slice`, `poc`;
//! - reference handling: `dpb` (marking, lists) and `output` (output order);
//! - slice data: `slice_data` (macroblock loop), `mb` (macroblock layer),
//!   `syntax` (the entropy interface) with `cavlc` / `vlc` and
//!   `cabac_syntax` / `cabac`, `mbstate` (neighbour state), `mvpred`;
//! - reconstruction: `recon`, `intra`, `inter`, `transform`, `deblock`;
//! - `reader`: the RBSP bit reader.
//!
//! [`Decoder::decode_nal`] takes NAL units one at a time and returns each
//! picture when its last macroblock is decoded; [`OutputQueue`] puts them
//! in output order. [`Decoder::decode`] takes one access unit per call and
//! returns its picture directly (decoding order: low delay);
//! [`Decoder::decode_shown`] does the same without the output copy, showing
//! the picture through [`Decoder::shown`] until the next call.

mod cabac;
mod cabac_syntax;
mod cavlc;
mod deblock;
mod dpb;
mod inter;
mod intra;
mod mb;
mod mbstate;
mod mvpred;
mod output;
mod poc;
mod pool;
mod pps;
mod reader;
mod recon;
mod scaling;
mod slice;
mod slice_data;
mod sps;
mod syntax;
#[cfg(test)]
mod tests;
mod transform;
mod vlc;

use std::sync::Arc;

pub use nc_common::decode::DecodeError;
use nc_common::decode::{annex_b_units, unescape_rbsp};
use nc_common::{ColourInfo, Picture, PlaneIndex, PlaneViewMut, ShownPicture};
use nc_dsp::Dsp;

use cabac::Cabac;
use cabac_syntax::CabacSyntax;
#[cfg(test)]
pub(crate) use cavlc::block_for_tests as cavlc_block_for_tests;
use cavlc::{CavlcSyntax, CavlcTables};
use deblock::{DeblockContext, RowDeblocker, SliceFilter};
use dpb::{Dpb, RefPicture};
use inter::{InterContext, InterKernels};
use mb::{MbExtra, SliceParams};
use mbstate::MbState;
pub use output::{DecodedFrame, OutputQueue};
use poc::{FramePoc, PocState};
use pool::PicturePool;
use pps::Pps;
use reader::Reader;
use recon::ReconContext;
use scaling::LevelScale;
use slice::{SliceHeader, SliceType};
use slice_data::SliceDecoder;
use sps::Sps;
use transform::{Dequantiser, Residual};

type Result<T> = std::result::Result<T, DecodeError>;

/// Largest packet [`Decoder::decode`] accepts.
const MAX_PACKET_BYTES: usize = 64 * 1024 * 1024;
/// NAL unit header fields (§7.3.1).
const FORBIDDEN_ZERO_BIT: u8 = 0x80;
const NAL_REF_IDC_SHIFT: u8 = 5;
const NAL_REF_IDC_MASK: u8 = 0x03;
const NAL_TYPE_MASK: u8 = 0x1f;
/// `nal_unit_type` values (Table 7-1).
const NAL_SLICE: u8 = 1;
const NAL_PARTITION_A: u8 = 2;
const NAL_PARTITION_C: u8 = 4;
const NAL_IDR_SLICE: u8 = 5;
const NAL_SPS: u8 = 7;
const NAL_PPS: u8 = 8;
/// Number of SPS and PPS ids (§7.4.2.1.1, §7.4.2.2).
const SPS_IDS: usize = sps::MAX_SPS_ID as usize + 1;
const PPS_IDS: usize = pps::MAX_PPS_ID as usize + 1;
/// Writable views of a picture's visible Y, Cb and Cr samples, made once per
/// slice (or deblocking pass) rather than per macroblock.
type Planes<'a> = [PlaneViewMut<'a>; 3];

/// The views of `picture`'s planes.
fn planes(picture: &mut Picture) -> Planes<'_> {
    let [y, u, v] = picture.planes_mut();
    [y.view_mut(), u.view_mut(), v.view_mut()]
}

/// Picture border kept for motion compensation (whole-window reads).
const REFERENCE_BORDER: usize = 64;
/// Chroma value of 4:0:0 output (`1 << (BitDepthC - 1)`).
const MONOCHROME_CHROMA: u8 = 128;

/// The picture being decoded.
#[derive(Debug)]
struct CurrentPicture {
    sps: Arc<Sps>,
    /// The first slice's header (`frame_num`, marking and order count fields
    /// are the same in every slice of a picture).
    header: SliceHeader,
    poc: FramePoc,
    id: u32,
    picture: Picture,
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
struct Visible {
    /// The deblocked picture, shared with the DPB when it is a reference.
    picture: Arc<Picture>,
    /// Top-left luma sample of the cropping window.
    origin: (usize, usize),
    /// Luma size of the cropping window.
    size: (usize, usize),
}

impl Visible {
    fn view(&self) -> Option<ShownPicture<'_>> {
        ShownPicture::window(&self.picture, self.origin, self.size)
    }

    /// The window copied into a picture from `pool`.
    fn copy(&self, pool: &mut PicturePool) -> Result<Picture> {
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
struct Finished {
    visible: Visible,
    poc: i64,
    new_sequence: bool,
    reorder_depth: usize,
}

/// A Constrained High decoder.
#[derive(Debug)]
pub struct Decoder {
    sps: Vec<Option<Arc<Sps>>>,
    pps: Vec<Option<Arc<Pps>>>,
    /// The SPS activated by the latest IDR picture.
    active_sps: Option<Arc<Sps>>,
    dpb: Dpb,
    pool: PicturePool,
    poc: PocState,
    current: Option<CurrentPicture>,
    /// Per-macroblock state of the current picture (reused, not reset:
    /// see `MbState::slice`).
    mbs: Vec<MbState>,
    /// Serial number of the last slice started.
    slice_serial: u32,
    residual: Box<Residual>,
    extra: Box<MbExtra>,
    dsp: Dsp,
    kernels: InterKernels,
    cavlc: CavlcTables,
    /// Identity of the next decoded picture.
    next_id: u32,
    /// The picture of the last successful [`Decoder::decode_shown`]; its
    /// buffer goes back to `pool` once the next call drops this.
    shown: Option<Visible>,
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder {
    /// An empty decoder; the stream must start with parameter sets and an
    /// IDR picture.
    #[must_use]
    pub fn new() -> Self {
        let dsp = Dsp::new();
        let kernels = InterKernels::new(&dsp);
        Self {
            sps: vec![None; SPS_IDS],
            pps: vec![None; PPS_IDS],
            active_sps: None,
            dpb: Dpb::default(),
            pool: PicturePool::default(),
            poc: PocState::default(),
            current: None,
            mbs: Vec::new(),
            slice_serial: 0,
            residual: Box::new(Residual::new()),
            extra: Box::new(MbExtra::new()),
            dsp,
            kernels,
            cavlc: CavlcTables::new(),
            next_id: 1,
            shown: None,
        }
    }

    /// Test hook: overwrite every reference picture with `value`, as a
    /// decoder that lost a frame would hold wrong references.
    #[cfg(test)]
    pub(crate) fn corrupt_references(&mut self, value: u8) {
        self.dpb.corrupt_pictures(value);
    }

    /// Hand back a picture this decoder returned, once the caller is done
    /// with it: a later picture reuses its buffer.
    pub fn recycle(&mut self, picture: Picture) {
        self.pool.give_back(picture);
    }

    /// Colour signalling of the active SPS (`None` before the first one).
    #[must_use]
    pub fn colour(&self) -> Option<ColourInfo> {
        self.active_sps.as_ref().map(|sps| sps.colour)
    }

    /// Decode one access unit (Annex-B, one picture) and return its visible
    /// samples, in decoding order.
    ///
    /// # Errors
    ///
    /// Rejects truncated, malformed, unsupported or excessively large input,
    /// and packets that do not complete exactly one picture.
    pub fn decode(&mut self, packet: &[u8]) -> Result<Picture> {
        self.shown = None;
        self.decode_packet(packet)?.copy(&mut self.pool)
    }

    /// [`Decoder::decode`] without the output copy: the picture is shown
    /// by [`Decoder::shown`] until the next call.
    ///
    /// # Errors
    ///
    /// As [`Decoder::decode`].
    pub fn decode_shown(&mut self, packet: &[u8]) -> Result<()> {
        // Release the previous picture first, so this one can reuse it.
        self.shown = None;
        self.shown = Some(self.decode_packet(packet)?);
        Ok(())
    }

    /// The picture of one access unit.
    fn decode_packet(&mut self, packet: &[u8]) -> Result<Visible> {
        if packet.len() > MAX_PACKET_BYTES {
            return Err(DecodeError::Unsupported("H264 packet allocation limit"));
        }
        let mut frame = None;
        for unit in annex_b_units(packet)? {
            if let Some(decoded) = self.decode_unit(unit)? {
                if frame.is_some() {
                    return Err(DecodeError::Unsupported("several pictures in one packet"));
                }
                frame = Some(decoded);
            }
        }
        frame
            .map(|f| f.visible)
            .ok_or(DecodeError::Malformed("packet does not complete a picture"))
    }

    /// The visible samples of the picture the last successful
    /// [`Decoder::decode_shown`] decoded.
    #[must_use]
    pub fn shown(&self) -> Option<ShownPicture<'_>> {
        self.shown.as_ref()?.view()
    }

    /// Decode one NAL unit (header and payload, without start code).
    /// Returns the picture its last slice completes.
    ///
    /// # Errors
    ///
    /// Malformed or unsupported syntax. A picture that fails is dropped.
    pub fn decode_nal(&mut self, nal: &[u8]) -> Result<Option<DecodedFrame>> {
        let Some(finished) = self.decode_unit(nal)? else {
            return Ok(None);
        };
        let Visible {
            picture,
            origin,
            size,
        } = finished.visible;
        Ok(Some(DecodedFrame {
            picture,
            origin,
            size,
            poc: finished.poc,
            new_sequence: finished.new_sequence,
            reorder_depth: finished.reorder_depth,
        }))
    }

    /// [`Decoder::decode_nal`] returning the picture uncopied.
    fn decode_unit(&mut self, nal: &[u8]) -> Result<Option<Finished>> {
        let (&header, payload) = nal.split_first().ok_or(DecodeError::Truncated)?;
        if header & FORBIDDEN_ZERO_BIT != 0 {
            return Err(DecodeError::Malformed("forbidden_zero_bit"));
        }
        let nal_ref_idc = u32::from((header >> NAL_REF_IDC_SHIFT) & NAL_REF_IDC_MASK);
        match header & NAL_TYPE_MASK {
            NAL_SLICE | NAL_IDR_SLICE => {
                let idr = header & NAL_TYPE_MASK == NAL_IDR_SLICE;
                let rbsp = unescape_rbsp(payload)?;
                let result = self.slice(&rbsp, idr, nal_ref_idc);
                if result.is_err() {
                    self.current = None;
                }
                result
            }
            NAL_SPS => {
                let sps = sps::parse(&unescape_rbsp(payload)?)?;
                let id = sps.id as usize;
                self.sps[id] = Some(Arc::new(sps));
                Ok(None)
            }
            NAL_PPS => {
                let pps = pps::parse(&unescape_rbsp(payload)?)?;
                let id = pps.id as usize;
                self.pps[id] = Some(Arc::new(pps));
                Ok(None)
            }
            NAL_PARTITION_A..=NAL_PARTITION_C => {
                Err(DecodeError::Unsupported("H264 data partitioning"))
            }
            // SEI, delimiters, end of sequence/stream, filler, SPS
            // extensions, auxiliary pictures and the SVC/MVC extension NAL
            // units do not affect the primary pictures of this profile.
            _ => Ok(None),
        }
    }

    /// One slice NAL unit's RBSP.
    fn slice(&mut self, rbsp: &[u8], idr: bool, nal_ref_idc: u32) -> Result<Option<Finished>> {
        if idr && nal_ref_idc == 0 {
            return Err(DecodeError::Malformed("IDR picture with nal_ref_idc 0"));
        }
        let pps_id = slice::parse_pps_id(rbsp)? as usize;
        // Missing parameter sets: the IDR access unit carrying them was lost.
        let pps = self.pps[pps_id].clone().ok_or(DecodeError::MissingKey)?;
        let sps = self.sps[pps.sps_id as usize]
            .clone()
            .ok_or(DecodeError::MissingKey)?;
        let mut reader = Reader::new(rbsp);
        let header = slice::parse(&mut reader, &sps, &pps, idr, nal_ref_idc)?;
        if header.first_mb == 0 {
            if self.current.take().is_some() {
                return Err(DecodeError::Malformed("picture with missing slices"));
            }
            self.start_picture(&sps, &header)?;
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
            for plane in [PlaneIndex::U, PlaneIndex::V] {
                picture.plane_mut(plane).view_mut().fill(MONOCHROME_CHROMA);
            }
        }
        // A picture has at most one slice per macroblock: reset the states
        // (and the serials) only for a new size or before serials could
        // wrap. Otherwise a 2 MB rewrite per 1080p picture, which also
        // evicted the reference blocks from the caches.
        let picture_slices = u32::try_from(sps.mbs()).unwrap_or(u32::MAX);
        if self.mbs.len() != sps.mbs() || self.slice_serial > u32::MAX - picture_slices {
            self.mbs.clear();
            self.mbs.resize(sps.mbs(), MbState::EMPTY);
            self.slice_serial = 0;
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
        current
            .slices
            .push(SliceFilter::new(header.deblock, pps.chroma_qp_offset));
        // Within the picture's budget (`start_picture`) unless the stream
        // has more slices than macroblocks.
        if current.slices.len() > current.sps.mbs() {
            return Err(DecodeError::Malformed("more slices than macroblocks"));
        }
        self.slice_serial += 1;
        let slice_no = self.slice_serial;
        let refs = match header.slice_type {
            SliceType::P => self.dpb.ref_list(
                header.frame_num,
                header.num_ref_idx_active,
                &header.modifications,
            )?,
            SliceType::I => Vec::new(),
        };
        let coded_size = (current.picture.width(), current.picture.height());
        if refs
            .iter()
            .flatten()
            .any(|r| (r.picture.width(), r.picture.height()) != coded_size)
        {
            return Err(DecodeError::Malformed("reference of another size"));
        }
        let ref_ids: Vec<Option<u32>> = refs.iter().map(|r| r.as_ref().map(|r| r.id)).collect();
        let scaling = match &pps.scaling {
            Some(parsed) => scaling::resolve(parsed, sps.scaling_present.then_some(&sps.scaling)),
            None => sps.scaling,
        };
        let dequantiser = Dequantiser::new(LevelScale::new(&scaling), pps.chroma_qp_offset);
        let params = SliceParams {
            slice_type: header.slice_type,
            num_ref_idx_active: header.num_ref_idx_active,
            ref_ids: &ref_ids,
            transform_8x8_mode: pps.transform_8x8_mode,
            constrained_intra_pred: pps.constrained_intra_pred,
            monochrome: sps.monochrome,
            dequantiser: &dequantiser,
        };
        let recon = ReconContext {
            dsp: &self.dsp,
            inter: InterContext::new(
                &self.kernels,
                &refs,
                header.weights.as_ref(),
                sps.monochrome,
            ),
            constrained_intra_pred: pps.constrained_intra_pred,
            monochrome: sps.monochrome,
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
    fn finish_picture(&mut self) -> Result<Finished> {
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
        let ctx = DeblockContext {
            dsp: &self.dsp,
            mbs: &self.mbs,
            width_mbs: sps.width_mbs,
            first_slice,
            slices: &slices,
            monochrome: sps.monochrome,
        };
        deblocker.filter_through(sps.height_mbs - 1, &mut planes(&mut picture), &ctx);
        let reset = header.marking.resets();
        if header.reference {
            picture.extend_borders();
        }
        // Shared even when not a reference: the pool takes the buffer back
        // once the caller is done with it.
        let picture = self.pool.share(picture);
        if header.reference {
            let reference = RefPicture {
                id,
                picture: picture.clone(),
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
            poc: if reset { 0 } else { poc.value() },
            new_sequence: header.idr || reset,
            reorder_depth: sps
                .max_num_reorder_frames
                .map_or_else(|| sps.dpb_frames(), |frames| frames as usize),
        })
    }
}
