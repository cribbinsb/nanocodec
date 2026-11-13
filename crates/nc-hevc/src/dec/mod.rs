//! HEVC Main / Main Still Picture profile decoder (H.265 Annex A.3.2,
//! A.3.5): 8-bit 4:2:0 streams of any structure the profiles allow — I, P
//! and B slices, multiple and dependent slice segments, tiles, WPP, AMP,
//! PCM, lossless, transform skip, scaling lists, weighted prediction, long-
//! term references, temporal sub-layers, CRA/BLA/RASL handling and output
//! in output order through the bumping process of Annex C.5.2.
//!
//! Module map:
//! - `params`, `slice`: parameter sets and slice segment headers;
//! - `cabac`, `ctx`: the arithmetic decoder and context variables;
//! - `ctu`, `residual`: slice data syntax and block reconstruction;
//! - `mvpred`, `mc`: motion vector prediction and inter prediction;
//! - `deblock`, `sao`: the in-loop filters;
//! - `picture`: per-picture block state, CTB layout and stored motion;
//! - `stats`: coding-tool analysis facts (feature `analysis`;
//!   `Decoder::collect_stats`, block records in `ctu`);
//! - this module: NAL units, POC (§8.3.1), reference picture sets (§8.3.2,
//!   §8.3.3), reference picture lists (§8.3.4) and the DPB (C.5.2).

mod cabac;
mod ctu;
mod ctx;
mod deblock;
mod loop_filter;
mod mc;
mod mvpred;
mod params;
mod picture;
mod residual;
mod sao;
mod slice;
#[cfg(feature = "analysis")]
mod stats;
#[cfg(test)]
mod tests;

use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::Arc;

#[cfg(test)]
use nc_common::PlaneIndex;
use nc_common::decode::{DecodeError, annex_b_units, unescape_rbsp};
#[cfg(feature = "analysis")]
use nc_common::stream_stats::StreamStats;
use nc_common::{ColourInfo, Picture, PictureRecycler, ShownPicture};
use nc_dsp::Dsp;

use crate::scan::Scans;
use ctu::{Carry, Scratch, SliceContext};
use loop_filter::LoopFilter;
use mvpred::Collocated;
use params::{MAX_PPS, MAX_SPS, Pps, Sps};
use picture::{Layout, Motion, PictureState, Pool, RefPic, SliceInfo};
use slice::{SliceHeader, SliceType, nal};

/// How a DPB picture is used for reference (§8.3.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Marking {
    Unused,
    ShortTerm,
    LongTerm,
}

/// A conformance window: top-left luma sample and luma size.
type Window = ((usize, usize), (usize, usize));

/// A decoded picture in the DPB.
#[derive(Debug)]
struct Frame {
    id: u64,
    poc: i32,
    /// Shared with the output queue until the caller is done with it.
    samples: Arc<Picture>,
    motion: Motion,
    marking: Marking,
    needed_for_output: bool,
    /// Conformance window (left, right, top, bottom).
    crop: [usize; 4],
}

impl Frame {
    fn reference(&self) -> bool {
        self.marking != Marking::Unused
    }

    fn info(&self) -> RefPic {
        RefPic {
            id: self.id,
            poc: self.poc,
            long_term: self.marking == Marking::LongTerm,
        }
    }

    /// The conformance window: top-left luma sample and size.
    fn window(&self) -> Result<Window, DecodeError> {
        let [left, right, top, bottom] = self.crop;
        let (w, h) = (self.samples.width(), self.samples.height());
        let size = (w.checked_sub(left + right), h.checked_sub(top + bottom));
        let (Some(width), Some(height)) = size else {
            return Err(DecodeError::Malformed("conformance window"));
        };
        Ok(((left, top), (width, height)))
    }

    /// The samples inside the conformance window.
    fn visible(&self) -> Result<ShownPicture<'_>, DecodeError> {
        let (origin, size) = self.window()?;
        ShownPicture::window(&self.samples, origin, size)
            .ok_or(DecodeError::Malformed("conformance window"))
    }

    /// The picture for output order, shared with the DPB (no copy).
    fn shared(&self) -> Result<OutputPicture, DecodeError> {
        self.visible()?;
        let (origin, size) = self.window()?;
        Ok(OutputPicture {
            picture: Arc::clone(&self.samples),
            origin,
            size,
        })
    }

    /// The cropped output picture, in a buffer from `recycler` (every
    /// sample is written).
    fn output(&self, recycler: &mut PictureRecycler) -> Result<Picture, DecodeError> {
        let visible = self.visible()?;
        let mut out = recycler.take(visible.width(), visible.height(), 0)?;
        visible.copy_to(&mut out);
        Ok(out)
    }
}

/// A picture in output order: the decoded picture, shared with the
/// decoder's reference buffers (no copy), and its conformance window. Its
/// buffer returns to the decoder once both have dropped it.
#[derive(Debug, Clone)]
pub struct OutputPicture {
    /// The decoded picture, with its padding and border.
    pub picture: Arc<Picture>,
    /// Top-left luma sample of the conformance window.
    pub origin: (usize, usize),
    /// Luma size of the conformance window.
    pub size: (usize, usize),
}

impl OutputPicture {
    /// The visible samples (inside the conformance window).
    #[must_use]
    pub fn view(&self) -> ShownPicture<'_> {
        ShownPicture::window(&self.picture, self.origin, self.size)
            .expect("window checked when the picture was output")
    }
}

/// The reference picture set of the current picture, as DPB indices
/// (§8.3.2): `RefPicSetStCurrBefore`, `StCurrAfter`, `LtCurr`.
#[derive(Debug, Default)]
struct ReferenceSet {
    before: Vec<usize>,
    after: Vec<usize>,
    long_term: Vec<usize>,
}

/// The picture being decoded.
struct Current {
    sps: Rc<Sps>,
    pps: Rc<Pps>,
    state: PictureState,
    carry: Carry,
    id: u64,
    poc: i32,
    output: bool,
    set: ReferenceSet,
    /// The header of the last independent slice segment, for dependent ones.
    header: Option<Rc<SliceHeader>>,
}

/// POC and random access state carried between pictures.
#[derive(Debug, Default)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "independent random access flags"
)]
struct Sequence {
    /// `PicOrderCntVal` of the previous `TemporalId` 0 anchor (§8.3.1).
    prev_tid0_poc: i32,
    /// The next IRAP picture starts a new coded video sequence with
    /// `NoRaslOutputFlag` 1: the first picture, or the first after EOS.
    new_cvs: bool,
    /// `NoRaslOutputFlag` of the last IRAP picture: its RASL pictures are
    /// skipped.
    skip_rasl: bool,
    /// Pictures are skipped until the first IRAP picture.
    seen_irap: bool,
    /// A picture has been decoded (the first one never discards others).
    decoded_any: bool,
}

/// HEVC Main profile decoder. Feed NAL units (or Annex-B data) in decoding
/// order; decoded pictures come out in output order.
pub struct Decoder {
    sps: Vec<Option<Rc<Sps>>>,
    pps: Vec<Option<Rc<Pps>>>,
    dpb: Vec<Frame>,
    current: Option<Current>,
    /// Pictures bumped out of the DPB, in output order.
    output: VecDeque<OutputPicture>,
    /// The id of the last picture finished in decoding-order use, which
    /// [`Decoder::shown`] shows: every picture stays in the DPB at least
    /// until the next picture's reference picture set (§8.3.2), so it
    /// outlives the decode that produced it.
    shown: Option<u64>,
    /// Set by [`Decoder::decode_shown`]: pictures are shown in decoding
    /// order and none are queued for output order.
    decoding_order: bool,
    sequence: Sequence,
    next_id: u64,
    /// Buffers of pictures that left the DPB, for new ones.
    pool: Pool,
    /// The CTB layout of the last SPS/PPS pair.
    layout: Option<(Rc<Sps>, Rc<Pps>, Rc<Layout>)>,
    /// Output pictures handed back by the caller ([`Decoder::recycle`]).
    outputs: PictureRecycler,
    /// Working memory of the in-loop filters, kept between pictures.
    filter: LoopFilter,
    /// Working memory of block decoding.
    scratch: Scratch,
    colour: Option<ColourInfo>,
    dsp: Dsp,
    scans: Scans,
}

impl std::fmt::Debug for Decoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Decoder")
            .field("dpb", &self.dpb.len())
            .field("output", &self.output.len())
            .finish_non_exhaustive()
    }
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder {
    /// An empty decoder; the stream must start with parameter sets and an
    /// IRAP picture (pictures before the first IRAP are skipped).
    #[must_use]
    pub fn new() -> Self {
        Self {
            sps: vec![None; MAX_SPS],
            pps: vec![None; MAX_PPS],
            dpb: Vec::new(),
            current: None,
            output: VecDeque::new(),
            shown: None,
            decoding_order: false,
            sequence: Sequence {
                new_cvs: true,
                ..Sequence::default()
            },
            next_id: 0,
            pool: Pool::default(),
            layout: None,
            outputs: PictureRecycler::default(),
            filter: LoopFilter::default(),
            scratch: Scratch::default(),
            colour: None,
            dsp: Dsp::new(),
            scans: Scans::new(),
        }
    }

    /// Start collecting coding-tool analysis into `stats` (block records,
    /// and per-picture tallies unless [`StreamStats::records_only`]) from
    /// the next picture on. Decoding is unchanged.
    #[cfg(feature = "analysis")]
    pub fn collect_stats(&mut self, stats: StreamStats) {
        self.scratch.stats = Some(stats);
    }

    /// The analysis collected so far, if enabled: the block records of
    /// the last completed picture and the tallies of every completed one.
    #[cfg(feature = "analysis")]
    #[must_use]
    pub fn stats(&self) -> Option<&StreamStats> {
        self.scratch.stats.as_ref()
    }

    /// Complete the analysis of the last picture and hand it over (after
    /// [`Decoder::flush`]); collection stops.
    #[cfg(feature = "analysis")]
    pub fn take_stats(&mut self) -> Option<StreamStats> {
        let mut stats = self.scratch.stats.take()?;
        stats.finish();
        Some(stats)
    }

    /// Colour signalling of the last activated SPS (`None` before the first).
    #[must_use]
    pub fn colour(&self) -> Option<ColourInfo> {
        self.colour
    }

    /// Test hook: overwrite every DPB picture with `value`, as a decoder
    /// that lost a frame would hold wrong references.
    #[cfg(test)]
    pub(crate) fn corrupt_references(&mut self, value: u8) {
        for frame in &mut self.dpb {
            let samples = Arc::get_mut(&mut frame.samples).expect("no output queued in tests");
            for plane in PlaneIndex::ALL {
                samples.plane_mut(plane).view_mut().fill(value);
            }
            samples.extend_borders();
        }
    }

    /// Decode one access unit of Annex-B data and return its picture in
    /// decoding order (the low-delay use: each packet is one picture).
    /// A decoder used this way queues nothing for [`Decoder::next_output`].
    /// # Errors
    /// Rejects malformed or truncated data and syntax outside the Main profile.
    pub fn decode(&mut self, packet: &[u8]) -> Result<Picture, DecodeError> {
        self.decode_shown(packet)?;
        shown_frame(&self.dpb, self.shown)
            .ok_or(DecodeError::Malformed("packet has no picture"))?
            .output(&mut self.outputs)
    }

    /// [`Decoder::decode`] without the output copy: the picture is shown
    /// by [`Decoder::shown`] until the next call.
    /// # Errors
    /// As [`Decoder::decode`].
    pub fn decode_shown(&mut self, packet: &[u8]) -> Result<(), DecodeError> {
        self.decoding_order = true;
        self.shown = None;
        self.push(packet)?;
        self.finish_picture()?;
        if self.shown.is_none() {
            // Pictures before the first IRAP are not decodable (§8.1.3):
            // the key frame that starts the stream was lost.
            return Err(if self.sequence.seen_irap {
                DecodeError::Malformed("packet has no picture")
            } else {
                DecodeError::MissingKey
            });
        }
        Ok(())
    }

    /// The visible samples of the picture the last successful
    /// [`Decoder::decode_shown`] decoded.
    #[must_use]
    pub fn shown(&self) -> Option<ShownPicture<'_>> {
        shown_frame(&self.dpb, self.shown)?.visible().ok()
    }

    /// Feed Annex-B data holding whole NAL units.
    /// # Errors
    /// As [`Decoder::push_nal`].
    pub fn push(&mut self, data: &[u8]) -> Result<(), DecodeError> {
        for nal in annex_b_units(data)? {
            self.push_nal(nal)?;
        }
        Ok(())
    }

    /// Signal the end of the stream: the last picture is finished and every
    /// picture still waiting is queued for output.
    /// # Errors
    /// As [`Decoder::push_nal`] for the last picture.
    pub fn flush(&mut self) -> Result<(), DecodeError> {
        self.finish_picture()?;
        self.bump_all();
        Ok(())
    }

    /// Hand back a picture this decoder returned, once the caller is done
    /// with it: later output pictures of its size reuse the buffer instead
    /// of allocating and clearing one.
    pub fn recycle(&mut self, picture: Picture) {
        self.outputs.give(picture);
    }

    /// The next picture in output order, if one is ready.
    pub fn next_output(&mut self) -> Option<OutputPicture> {
        self.output.pop_front()
    }

    /// Decode one NAL unit (without start code).
    /// # Errors
    /// Rejects malformed or truncated data and syntax outside the Main
    /// profile; the decoder should then be reset before further use.
    pub fn push_nal(&mut self, nal: &[u8]) -> Result<(), DecodeError> {
        if nal.len() < 2 {
            return Err(DecodeError::Truncated);
        }
        if nal[0] & 0x80 != 0 {
            return Err(DecodeError::Malformed("forbidden_zero_bit"));
        }
        let nal_type = (nal[0] >> 1) & 63;
        let layer = ((nal[0] & 1) << 5) | (nal[1] >> 3);
        let temporal_id_plus1 = nal[1] & 7;
        if temporal_id_plus1 == 0 {
            return Err(DecodeError::Malformed("nuh_temporal_id_plus1"));
        }
        if layer != 0 {
            // Other layers are not part of the Main profile base layer.
            return Ok(());
        }
        let rbsp = unescape_rbsp(&nal[2..])?;
        match nal_type {
            0..=nal::RSV_IRAP_23 => {
                if nal_type <= nal::RSV_VCL_N14 && nal_type > nal::RASL_R || nal_type > nal::CRA {
                    // Reserved VCL types: ignored (§7.4.2.2).
                    return Ok(());
                }
                let result = self.slice_segment(&rbsp, nal_type, temporal_id_plus1 - 1);
                if result.is_err() {
                    // A picture with a bad slice is never finished: it
                    // would otherwise enter the DPB with its POC, and the
                    // pictures predicting from it would decode from garbage.
                    self.current = None;
                }
                result
            }
            nal::SPS => {
                self.finish_picture()?;
                let (id, sps) = Sps::parse(&rbsp)?;
                self.sps[id] = Some(Rc::new(sps));
                Ok(())
            }
            nal::PPS => {
                self.finish_picture()?;
                let (id, pps) = Pps::parse(&rbsp)?;
                self.pps[id] = Some(Rc::new(pps));
                Ok(())
            }
            nal::EOS | nal::EOB => {
                // The next picture starts a new coded video sequence; what
                // happens to pictures still waiting is decided by its
                // NoOutputOfPriorPicsFlag (C.5.2.2).
                self.finish_picture()?;
                self.sequence.new_cvs = true;
                Ok(())
            }
            // VPS, AUD, SEI, filler and reserved / unspecified units.
            _ => Ok(()),
        }
    }

    /// Decode one slice segment NAL unit.
    fn slice_segment(
        &mut self,
        rbsp: &[u8],
        nal_type: u8,
        temporal_id: u8,
    ) -> Result<(), DecodeError> {
        let (first, pps_id) = slice::first_and_pps(rbsp, nal_type)?;
        if first {
            self.finish_picture()?;
        }
        let irap = (nal::BLA_W_LP..=nal::RSV_IRAP_23).contains(&nal_type);
        if first {
            if irap {
                self.sequence.seen_irap = true;
                self.sequence.skip_rasl = self.sequence.new_cvs || nal_type != nal::CRA;
            }
            let rasl = matches!(nal_type, nal::RASL_N | nal::RASL_R);
            if !self.sequence.seen_irap || (rasl && self.sequence.skip_rasl) {
                // Not decodable: before the first IRAP, or a RASL picture
                // of an IRAP that starts a sequence (§8.1.3, C.5.2.2).
                return Ok(());
            }
        } else if self.current.is_none() {
            // A later segment of a skipped picture.
            return Ok(());
        }
        let pps = self.pps[pps_id]
            .clone()
            .ok_or(DecodeError::Malformed("missing PPS"))?;
        let sps = self.sps[pps.sps_id]
            .clone()
            .ok_or(DecodeError::Malformed("missing SPS"))?;
        let segment = slice::parse(rbsp, nal_type, &sps, &pps)?;
        if first {
            let header = segment
                .header
                .as_ref()
                .ok_or(DecodeError::Malformed("dependent first slice segment"))?;
            self.start_picture(sps, pps, header, nal_type, temporal_id)?;
        }
        let current = self
            .current
            .as_mut()
            .ok_or(DecodeError::Malformed("slice outside a picture"))?;
        let header = match segment.header.clone() {
            Some(h) => {
                let h = Rc::new(h);
                current.header = Some(h.clone());
                let lists = reference_lists(&current.set, &h)?;
                current.state.slices.push(SliceInfo {
                    address: segment.address,
                    deblock_disabled: h.deblock_disabled,
                    beta_offset_div2: h.beta_offset_div2,
                    tc_offset_div2: h.tc_offset_div2,
                    loop_filter_across_slices: h.loop_filter_across_slices,
                    lists: lists.map(|l| l.iter().map(|&i| self.dpb[i].info()).collect()),
                });
                h
            }
            None => current.header.clone().ok_or(DecodeError::Malformed(
                "dependent slice segment without a slice",
            ))?,
        };
        #[cfg(feature = "analysis")]
        if let Some(s) = &mut self.scratch.stats {
            stats::segment(s, first.then_some(&*current), &segment, rbsp.len());
        }
        let slice_index = current.state.slices.len();
        let lists = reference_lists(&current.set, &header)?;
        let refs = lists
            .clone()
            .map(|l| l.iter().map(|&i| &*self.dpb[i].samples).collect());
        let col = (header.temporal_mvp && header.slice_type != SliceType::I)
            .then(|| {
                let list = usize::from(!header.collocated_from_l0);
                lists[list]
                    .get(header.collocated_ref_idx)
                    .map(|&i| Collocated {
                        motion: &self.dpb[i].motion,
                        poc: self.dpb[i].poc,
                    })
            })
            .flatten();
        let no_backward_pred = lists
            .iter()
            .flatten()
            .all(|&i| self.dpb[i].poc <= current.poc);
        let cx = SliceContext {
            sps: &current.sps,
            pps: &current.pps,
            header: &header,
            slice: slice_index as u16,
            refs,
            col,
            poc: current.poc,
            no_backward_pred,
            dsp: &self.dsp,
            scans: &self.scans,
        };
        ctu::decode_segment(
            &cx,
            &mut current.state,
            &mut current.carry,
            &mut self.scratch,
            &mut self.filter,
            rbsp,
            &segment,
        )
    }

    /// Begin a picture: POC, reference picture set, DPB output and removal
    /// (§8.1.3, C.5.2.2).
    fn start_picture(
        &mut self,
        sps: Rc<Sps>,
        pps: Rc<Pps>,
        header: &SliceHeader,
        nal_type: u8,
        temporal_id: u8,
    ) -> Result<(), DecodeError> {
        let irap = (nal::BLA_W_LP..=nal::RSV_IRAP_23).contains(&nal_type);
        let no_rasl_output = irap && self.sequence.skip_rasl;
        let poc = self.picture_order_count(&sps, header, nal_type, no_rasl_output);
        // §8.3.1: the anchor for the next POC is the previous TemporalId 0
        // picture that is not RASL, RADL or a sub-layer non-reference one.
        let sub_layer_non_reference = nal_type <= nal::RSV_VCL_N14 && nal_type.is_multiple_of(2);
        let leading = (nal::RADL_N..=nal::RASL_R).contains(&nal_type);
        if temporal_id == 0 && !leading && !sub_layer_non_reference {
            self.sequence.prev_tid0_poc = poc;
        }
        if irap && no_rasl_output {
            // NoOutputOfPriorPicsFlag: always 1 for CRA (C.5.2.2).
            let discard = self.sequence.decoded_any
                && (nal_type == nal::CRA || header.no_output_of_prior_pics);
            if discard {
                self.release_where(|_| true);
            } else {
                self.bump_all();
            }
            for f in &mut self.dpb {
                f.marking = Marking::Unused;
            }
        }
        self.sequence.new_cvs = false;
        self.sequence.decoded_any = true;
        let ids = if matches!(nal_type, nal::IDR_W_RADL | nal::IDR_N_LP) {
            for f in &mut self.dpb {
                f.marking = Marking::Unused;
            }
            IdSet::default()
        } else {
            self.reference_set(&sps, header, poc)?
        };
        // Remove pictures neither waiting for output nor referenced, then
        // bump while the DPB is over its reorder or size limits (C.5.2.2).
        // The latency limit is not applied: it only makes output earlier,
        // never changes its order.
        self.release_where(|f| !f.needed_for_output && !f.reference());
        let (buffering, reorder, _) = sps.dpb[sps.max_sub_layers - 1];
        while self.waiting() > reorder || self.dpb.len() >= buffering {
            if !self.bump() {
                break;
            }
        }
        let set = ids.indices(&self.dpb)?;
        self.colour = Some(sps.colour);
        let layout = match &self.layout {
            Some((s, p, layout)) if Rc::ptr_eq(s, &sps) && Rc::ptr_eq(p, &pps) => layout.clone(),
            _ => {
                let layout = Rc::new(Layout::new(&sps, &pps)?);
                self.layout = Some((sps.clone(), pps.clone(), layout.clone()));
                layout
            }
        };
        let state = PictureState::new(&sps, layout, &mut self.pool)?;
        let id = self.next_id;
        self.next_id += 1;
        self.current = Some(Current {
            sps,
            pps,
            state,
            carry: Carry::default(),
            id,
            poc,
            output: header.pic_output,
            set,
            header: None,
        });
        Ok(())
    }

    /// `PicOrderCntVal` (§8.3.1).
    fn picture_order_count(
        &self,
        sps: &Sps,
        header: &SliceHeader,
        nal_type: u8,
        no_rasl_output: bool,
    ) -> i32 {
        let max_lsb = 1i32 << sps.log2_max_poc_lsb;
        let lsb = header.poc_lsb as i32;
        let msb = if no_rasl_output && (nal::BLA_W_LP..=nal::RSV_IRAP_23).contains(&nal_type) {
            0
        } else {
            let prev = self.sequence.prev_tid0_poc;
            let prev_lsb = prev & (max_lsb - 1);
            let prev_msb = prev - prev_lsb;
            if lsb < prev_lsb && prev_lsb - lsb >= max_lsb / 2 {
                prev_msb + max_lsb
            } else if lsb > prev_lsb && lsb - prev_lsb > max_lsb / 2 {
                prev_msb - max_lsb
            } else {
                prev_msb
            }
        };
        msb + lsb
    }

    /// Mark the DPB from the slice's reference picture set (§8.3.2).
    /// Returns the sets by picture id.
    ///
    /// # Errors
    ///
    /// [`DecodeError::MissingReference`] when a reference the picture uses
    /// is not in the DPB: "no reference picture", an unintentional picture
    /// loss (§8.3.2 NOTE). The marking still applies, so the DPB matches
    /// the encoder's for the pictures that follow; stand-ins (§8.3.3) are
    /// only for the skipped leading pictures of a CRA or BLA, which this
    /// decoder never decodes.
    fn reference_set(
        &mut self,
        sps: &Sps,
        header: &SliceHeader,
        poc: i32,
    ) -> Result<IdSet, DecodeError> {
        let max_lsb = 1i32 << sps.log2_max_poc_lsb;
        let mut set = IdSet::default();
        let mut keep: Vec<u64> = Vec::new();
        let mut missing = false;
        // Long-term entries first: any reference picture may become one.
        let mut long_term = Vec::new();
        for lt in &header.long_term {
            let full = if lt.msb_present {
                Some(poc - (poc & (max_lsb - 1)) + lt.poc)
            } else {
                None
            };
            let found = self.dpb.iter().position(|f| {
                f.reference()
                    && match full {
                        Some(p) => f.poc == p,
                        None => f.poc & (max_lsb - 1) == lt.poc,
                    }
            });
            let Some(index) = found else {
                missing |= lt.used;
                continue;
            };
            long_term.push(self.dpb[index].id);
            keep.push(self.dpb[index].id);
            if lt.used {
                set.long_term.push(self.dpb[index].id);
            }
        }
        for f in &mut self.dpb {
            if long_term.contains(&f.id) {
                f.marking = Marking::LongTerm;
            }
        }
        for (entries, target) in [
            (&header.short_term.negative, &mut set.before),
            (&header.short_term.positive, &mut set.after),
        ] {
            for &(delta, used) in entries {
                let p = poc + delta;
                let found = self
                    .dpb
                    .iter()
                    .position(|f| f.marking == Marking::ShortTerm && f.poc == p);
                let Some(index) = found else {
                    missing |= used;
                    continue;
                };
                keep.push(self.dpb[index].id);
                if used {
                    target.push(self.dpb[index].id);
                }
            }
        }
        for f in &mut self.dpb {
            if !keep.contains(&f.id) {
                f.marking = Marking::Unused;
            }
        }
        if missing {
            return Err(DecodeError::MissingReference);
        }
        Ok(set)
    }

    /// Pictures waiting for output.
    fn waiting(&self) -> usize {
        self.dpb.iter().filter(|f| f.needed_for_output).count()
    }

    /// The bumping process (C.5.2.4): output the waiting picture with the
    /// smallest POC and drop it if no longer referenced. Returns whether a
    /// picture was output.
    fn bump(&mut self) -> bool {
        let Some(index) = self
            .dpb
            .iter()
            .enumerate()
            .filter(|(_, f)| f.needed_for_output)
            .min_by_key(|(_, f)| f.poc)
            .map(|(i, _)| i)
        else {
            return false;
        };
        if !self.decoding_order
            && let Ok(picture) = self.dpb[index].shared()
        {
            self.output.push_back(picture);
        }
        self.dpb[index].needed_for_output = false;
        if !self.dpb[index].reference() {
            let frame = self.dpb.remove(index);
            self.pool.recycle(frame.samples, frame.motion);
        }
        true
    }

    fn bump_all(&mut self) {
        while self.bump() {}
        self.release_where(|f| !f.reference());
    }

    /// Remove the DPB pictures matching `remove`, keeping their buffers.
    fn release_where(&mut self, remove: impl Fn(&Frame) -> bool) {
        let mut i = 0;
        while i < self.dpb.len() {
            if remove(&self.dpb[i]) {
                let frame = self.dpb.swap_remove(i);
                self.pool.recycle(frame.samples, frame.motion);
            } else {
                i += 1;
            }
        }
    }

    /// Finish the current picture: in-loop filters, DPB storage and output
    /// (C.5.2.3).
    fn finish_picture(&mut self) -> Result<(), DecodeError> {
        let Some(current) = self.current.take() else {
            return Ok(());
        };
        #[cfg(feature = "analysis")]
        if let Some(s) = &mut self.scratch.stats {
            // Its records become the completed picture's.
            s.finish();
        }
        let Current {
            sps,
            pps,
            mut state,
            id,
            poc,
            output,
            ..
        } = current;
        if !state.progress.complete() {
            return Err(DecodeError::Malformed("picture not completely decoded"));
        }
        self.filter.advance(&self.dsp, &pps, &mut state, true);
        let (samples, motion) = state.into_motion(&mut self.pool);
        let frame = Frame {
            id,
            poc,
            samples: Arc::new(samples),
            motion,
            marking: Marking::ShortTerm,
            needed_for_output: output,
            crop: sps.crop,
        };
        if self.decoding_order {
            frame.visible()?;
            self.shown = Some(id);
        }
        self.dpb.push(frame);
        let (_, reorder, _) = sps.dpb[sps.max_sub_layers - 1];
        while self.waiting() > reorder {
            self.bump();
        }
        Ok(())
    }
}

/// The DPB picture with id `shown`.
fn shown_frame(dpb: &[Frame], shown: Option<u64>) -> Option<&Frame> {
    let id = shown?;
    dpb.iter().find(|f| f.id == id)
}

/// A reference picture set by picture id (ids survive DPB removals).
#[derive(Debug, Default)]
struct IdSet {
    before: Vec<u64>,
    after: Vec<u64>,
    long_term: Vec<u64>,
}

impl IdSet {
    /// The set as DPB indices.
    fn indices(&self, dpb: &[Frame]) -> Result<ReferenceSet, DecodeError> {
        let index = |ids: &[u64]| -> Result<Vec<usize>, DecodeError> {
            ids.iter()
                .map(|id| {
                    dpb.iter()
                        .position(|f| f.id == *id)
                        .ok_or(DecodeError::Malformed("reference picture removed"))
                })
                .collect()
        };
        Ok(ReferenceSet {
            before: index(&self.before)?,
            after: index(&self.after)?,
            long_term: index(&self.long_term)?,
        })
    }
}

/// `RefPicList0` / `RefPicList1` as DPB indices (§8.3.4).
fn reference_lists(
    set: &ReferenceSet,
    header: &SliceHeader,
) -> Result<[Vec<usize>; 2], DecodeError> {
    let mut lists = [Vec::new(), Vec::new()];
    if header.slice_type == SliceType::I {
        return Ok(lists);
    }
    let total = set.before.len() + set.after.len() + set.long_term.len();
    if total == 0 {
        return Err(DecodeError::Malformed(
            "inter slice without current references",
        ));
    }
    let count = if header.slice_type == SliceType::B {
        2
    } else {
        1
    };
    for (l, list) in lists.iter_mut().enumerate().take(count) {
        let order: [&[usize]; 3] = if l == 0 {
            [&set.before, &set.after, &set.long_term]
        } else {
            [&set.after, &set.before, &set.long_term]
        };
        let n = header.num_ref_idx[l];
        let mut temp = Vec::with_capacity(n.max(total));
        while temp.len() < n.max(total) {
            for part in order {
                for &i in part {
                    if temp.len() < n.max(total) {
                        temp.push(i);
                    }
                }
            }
        }
        *list = (0..n)
            .map(|i| match &header.list_entries[l] {
                Some(entries) => temp[entries[i]],
                None => temp[i],
            })
            .collect();
    }
    Ok(lists)
}
